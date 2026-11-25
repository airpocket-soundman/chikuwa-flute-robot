"""PC: a whistled-language or bird recording from the archives, played back by the flute's body (simulated rigs).

    python scripts/archive_demo.py runs/archive/silbo_gomero/sentences/<file>.wav --kind silbo
    python scripts/archive_demo.py runs/archive/xeno_canto/parus_minor/XC285813.mp3 --kind bird --start 3 --seconds 6
    python scripts/archive_demo.py <file> --kind silbo --slow 2 --steps classic step7c   # speak half as fast
    python scripts/archive_demo.py --list silbo                                          # what there is

Fetch the archives first (scripts/archive_fetch.py).  The steps:
  1. listen   the pitch every 10 ms (YIN, the training-side ear; a whistled language is a pitch line, and a bird
              note is close to one), cleaned like a whistle (flute_rl/yamabiko/whistle.py)
  2. fit      into the flute's range (the profile's low..high cents) with the shape kept: ``--fit compress``
              scales the intervals down when the phrase is wider than the flute (whistled speech and birds
              usually are), ``--fit clip`` keeps them and cuts the ends; ``--slow`` stretches time
  3. play     every ``--steps`` profile plays it on ``--rigs`` rigs drawn around the twin (scripts/yamabiko_compare_sim.py,
              NumPy only), and the first rig's take is drawn and sounded
Writes runs/archive_demo/<stamp>_<name>/: source.wav (the clip), target.npz (cents, voice: the score the real rig
can play with ``scripts/yamabiko_whistle_copy.py --target``), <step>.wav (the simulated flute: its pitch, sounding
only when the flute does), demo.png and report.json.

Per step, over the rigs (the pitch line is what a whistled language carries, so this is the score that matters):
  shape  mean |played - target| in cent after removing the key, where both sound
  cover  share of the target's voiced time the flute sounds
  corr   correlation of the played and target pitch lines
and for the phrase: ``fast`` share of voiced steps that ask for more than the actuator's top speed (about
1100 cent/s at fan 80 %, full stroke 1.39 s), the reason to try ``--slow``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flute_rl.yamabiko.profiles import load_profiles  # noqa: E402
from flute_rl.yamabiko.sim_numpy import play, rig_arrays, rigs_around  # noqa: E402
from flute_rl.yamabiko.whistle import clean_track, to_target, whistle_track  # noqa: E402
from yamabiko_compare_sim import runtime_maker  # noqa: E402

DEFAULT_TWIN = ROOT / "models/step7c/twin_y1f80_05_runtime.npz"
TOP_SPEED = 1100.0           # cent/s the plunger can follow (fan 80 %, docs/handoff.md 0 章)
KINDS = {   # YIN band, frame and cleaning per kind of source
    "silbo": dict(fmin=700.0, fmax=4000.0, frame=640, median=5, min_note_steps=4, max_gap_steps=3),
    "whistle": dict(fmin=500.0, fmax=4000.0, frame=640, median=5, min_note_steps=6, max_gap_steps=5),
    "bird": dict(fmin=1800.0, fmax=9000.0, frame=400, median=3, min_note_steps=3, max_gap_steps=2),
}
SR_OUT = 16000


def read_audio(path: pathlib.Path):
    """(mono float -1..1, sample rate).  WAV with the standard library; MP3/FLAC/OGG need ``pip install soundfile``."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path)) as w:
                sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(n)
            if width == 2:
                return np.frombuffer(raw, "<i2").reshape(-1, ch).mean(1) / 32768.0, sr
        except wave.Error:
            pass
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return x.mean(1), sr


def write_wav(path, x, sr):
    pcm = np.clip(np.asarray(x) * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def resample(x, sr, to=SR_OUT):
    if sr == to:
        return x
    t = np.arange(int(len(x) * to / sr)) * sr / to
    return np.interp(t, np.arange(len(x)), x)


def listen(x, sr, kind):
    """Cleaned pitch line (cent re A4, NaN = silent) every 10 ms; the gate follows the clip's own loudness."""
    k = KINDS[kind]
    hop = int(round(sr * .01))
    frames = np.lib.stride_tricks.sliding_window_view(x, k["frame"])[::hop] if len(x) > k["frame"] else x[None]
    loud = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    gate = max(np.percentile(loud, 99) - 30.0, -70.0)            # 30 dB under the loudest notes
    raw = whistle_track(x, sr, fmin=k["fmin"], fmax=k["fmax"], frame=k["frame"], gate_db=gate)
    return raw, clean_track(raw, median=k["median"], min_note_steps=k["min_note_steps"],
                            max_gap_steps=k["max_gap_steps"])


def fit(cents, low, high, how, slow):
    """(cents for to_target, scale applied): stretched in time by ``slow``, intervals scaled to fit ``how``."""
    c = np.asarray(cents, np.float64)
    if slow != 1.0:
        idx = np.arange(int(len(c) * slow)) / slow
        voiced = np.isfinite(c)
        filled = np.interp(np.arange(len(c)), np.flatnonzero(voiced), c[voiced]) if voiced.any() else c
        c = np.where(np.interp(idx, np.arange(len(c)), voiced.astype(float)) > .5,
                     np.interp(idx, np.arange(len(c)), filled), np.nan)
    scale = 1.0
    v = np.isfinite(c)
    if how == "compress" and v.any():
        span = np.nanpercentile(c, 98) - np.nanpercentile(c, 2)
        room = (high - low) * .9
        if span > room:
            scale = room / span
            mid = np.nanmedian(c)
            c = mid + (c - mid) * scale
    return c, scale


def synth(cents, sounding, sr=SR_OUT, hz=None):
    """A plain flute-like tone (fundamental + soft 2nd/3rd) on the pitch line, faded where it does not sound."""
    step = int(sr * .01)
    f = 440.0 * 2 ** (np.repeat(np.nan_to_num(cents, nan=0.0), step) / 1200) if hz is None else np.repeat(hz, step)
    amp = np.repeat(np.asarray(sounding, float), step)
    amp = np.convolve(amp, np.ones(160) / 160, mode="same")          # 10 ms fades
    phase = 2 * np.pi * np.cumsum(f) / sr
    return .25 * amp * (np.sin(phase) + .3 * np.sin(2 * phase) + .1 * np.sin(3 * phase))


def transcription(src: pathlib.Path) -> str:
    """The sentence a Silbo Gomero clip says (transc.csv beside its clips/ folder), or ''."""
    import csv
    table = src.parent.parent / "transc.csv"
    if not table.exists():
        return ""
    with open(table, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if pathlib.PurePosixPath(row.get("path", "")).name == src.name:
                return row.get("transcription", "")
    return ""


def scores(out, target, voice):
    pitch, sounding = out["emitted"], out["sounding"]
    both = sounding & voice
    if both.sum() < 5:
        return {"shape": float("nan"), "cover": float(both.sum() / max(voice.sum(), 1)), "corr": float("nan")}
    d = pitch[both] - target[both]
    key = np.median(d)
    corr = np.corrcoef(pitch[both], target[both])[0, 1] if np.std(target[both]) > 1 else float("nan")
    return {"shape": float(np.abs(d - key).mean()), "cover": float(both.sum() / max(voice.sum(), 1)),
            "corr": float(corr)}


def list_sources(what):
    if what == "silbo":
        for part in ("sentences", "fragments", "words"):
            files = sorted((ROOT / "runs/archive/silbo_gomero" / part).rglob("*.wav"))
            print(f"{part}: {len(files)} files" + (f", e.g. {files[0].relative_to(ROOT)}" if files else ""))
        for t in sorted((ROOT / "runs/archive/silbo_gomero").rglob("*")):
            if t.is_file() and t.suffix.lower() not in (".wav", ".zip") and t.name != "README.txt":
                print(f"transcription: {t.relative_to(ROOT)}")
    else:
        manifest = ROOT / "runs/archive/xeno_canto/parus_minor/manifest.json"
        for r in json.loads(manifest.read_text("utf-8")) if manifest.exists() else []:
            print(f"XC{r['id']:<8d} {r['type'][:28]:28s} {r['length']:>6s}  {r['recordist']}, {r['place'][:30]}, "
                  f"{r['licence'].split('/licenses/')[-1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="a .wav or .mp3 from runs/archive/")
    ap.add_argument("--list", choices=["silbo", "bird"], help="list the fetched sources and stop")
    ap.add_argument("--kind", choices=list(KINDS), default="silbo")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the source")
    ap.add_argument("--seconds", type=float, default=8.0, help="length of the clip (at most)")
    ap.add_argument("--fit", choices=["compress", "clip"], default="compress")
    ap.add_argument("--slow", type=float, default=1.0, help="stretch time by this factor (2 = half as fast)")
    ap.add_argument("--steps", nargs="+", default=["classic", "step7c"], help="profiles in models/ to play it")
    ap.add_argument("--twin", default=str(DEFAULT_TWIN))
    ap.add_argument("--rigs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--out", default=None, help="folder (default runs/archive_demo/<stamp>_<name>)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):                 # Spanish ¿ ñ on a cp932 console
        sys.stdout.reconfigure(errors="replace")
    if args.list:
        return list_sources(args.list)
    if not args.source:
        ap.error("give a source file, or --list")
    src = pathlib.Path(args.source)
    x, sr = read_audio(src)
    x = x[int(args.start * sr):int((args.start + args.seconds) * sr)]
    if len(x) < sr * .2:
        raise SystemExit("the clip is shorter than 0.2 s")
    x = x / max(np.abs(x).max(), 1e-9) * .9
    out_dir = pathlib.Path(args.out or ROOT / "runs/archive_demo" / f"{time.strftime('%Y%m%d_%H%M%S')}_{src.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / "source.wav", resample(x, sr), SR_OUT)

    raw, clean = listen(x, sr, args.kind)
    profiles = load_profiles()
    first = profiles[args.steps[-1]]
    low, high = first["low_cents"], first["high_cents"]
    fitted, scale = fit(clean, low, high, args.fit, args.slow)
    try:
        target, voice, report = to_target(fitted, low, high)
    except ValueError:
        raise SystemExit("nothing pitched was heard in the clip; try another --start or --kind")
    slope = np.abs(np.diff(target)) / .01
    fast = float(((slope > TOP_SPEED) & voice[1:] & voice[:-1]).sum() / max(voice.sum(), 1))
    said = transcription(src)
    report.update(source=str(src), says=said, kind=args.kind, start=args.start, seconds=len(x) / sr, fit=args.fit,
                  interval_scale=scale, slow=args.slow, fast=fast,
                  source_span_cents=float(np.nanpercentile(clean, 98) - np.nanpercentile(clean, 2))
                  if np.isfinite(clean).any() else 0.0)
    np.savez(out_dir / "target.npz", cents=target, voice=voice, source_cents=raw)

    constants, twin = rig_arrays(args.twin)
    results, takes = {}, {}
    for name in args.steps:
        make = runtime_maker(profiles[name])
        rows = []
        for i, rig in enumerate(rigs_around(constants, twin, args.rigs, .12, args.seed)):
            take = play(make(), rig, target.astype(np.float64), voice)
            rows.append(scores(take, target, voice))
            if i == 0:
                takes[name] = take
        results[name] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
        write_wav(out_dir / f"{name}.wav", synth(takes[name]["emitted"], takes[name]["sounding"]), SR_OUT)
    write_wav(out_dir / "target.wav", synth(target, voice), SR_OUT)
    report["steps"] = results
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if said:
        print(f'says: "{said}"')
    print(f"{src.name}: {report['seconds']:.1f} s, span {report['source_span_cents']:.0f} cent -> "
          f"x{scale:.2f} into {low:.0f}..{high:.0f}, slow x{args.slow}, too fast for the plunger {fast:.0%}")
    print(f"{'step':10s} {'shape':>7s} {'cover':>7s} {'corr':>6s}   ({args.rigs} simulated rigs)")
    for name, m in results.items():
        print(f"{name:10s} {m['shape']:6.0f}c {m['cover']:7.0%} {m['corr']:6.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = 2 + len(takes)
    fig, axes = plt.subplots(rows, 1, figsize=(13, 2.5 * rows), squeeze=False)
    ax = axes[0, 0]
    y = resample(x, sr)
    ax.specgram(y, NFFT=512, Fs=SR_OUT, noverlap=384, cmap="magma")
    t = np.arange(len(raw)) * .01
    ax.plot(t, 440 * 2 ** (clean / 1200), "c", lw=1.2)
    ax.set_ylim(0, min(SR_OUT / 2, KINDS[args.kind]["fmax"] * 1.1)); ax.set_ylabel("Hz")
    ax.set_title(f"source: {src.name}" + (f' "{said}"' if said else "") + " (the line is what the ear heard)",
                 fontsize=9)
    tt = np.arange(len(target)) * .01
    axes[1, 0].plot(tt, np.where(voice, target, np.nan), "k", lw=2)
    axes[1, 0].set_title(f"score for the flute: intervals x{scale:.2f}, time x{args.slow}, "
                         f"{fast:.0%} of it faster than the plunger", fontsize=9)
    for ax, (name, take) in zip(axes[2:, 0], takes.items()):
        ax.plot(tt, np.where(voice, target, np.nan), "k", lw=2, label="score")
        ax.plot(tt, np.where(take["sounding"], take["emitted"], np.nan), lw=1.2, label="flute (sim)")
        m = results[name]
        ax.set_title(f"{name}: shape {m['shape']:.0f} cent, cover {m['cover']:.0%}, corr {m['corr']:.2f}", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    for ax in axes[1:, 0]:
        ax.set_ylabel("cent"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out_dir / "demo.png", dpi=70); plt.close(fig)
    print(f"saved {out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir}")


if __name__ == "__main__":
    main()
