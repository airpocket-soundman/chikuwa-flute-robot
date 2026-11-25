"""PC: human whistles from the archives for the ear and the memory, in the form of the rig's recordings.

    python scripts/archive_fetch.py silbo; python scripts/archive_fetch.py mlend      # once
    python scripts/yamabiko_archive_whistles.py --corpus silbo   # -> runs/archive/silbo_whistles/*.npz
    python scripts/yamabiko_archive_whistles.py --corpus mlend   # -> runs/archive/mlend_whistles/*.npz

silbo  Silbo Gomero (OpenSLR SLR137, CC BY-NC-SA 4.0): about 55 minutes of expert whistled speech by 4 people,
       the pitch lines with the wobble, glides and dips of speech.  The "fragments" part (the three parts are
       the same recordings cut differently, so one is used).  Speaker ``--holdout`` (04) is the test.
mlend  MLEnd Hums and Whistles (Queen Mary University of London, no licence stated: here only): whistled tunes
       (8 songs) by 129 everyday whistlers on their own phones and laptops.  Every 8th person is the test.
Together ten times and more the 3.5 minutes of whistles recorded on the rig so far.  Every clip is turned into
what the rig's record-only page saves (<stem>_whistle.npz: pcm int16 at 16 kHz, sr, kind):
  - resampled 44.1 -> 16 kHz (polyphase, anti-aliased)
  - brought to the loudness of the rig's recorded whistles (their median RMS over the voiced 20 ms frames,
    times a random 0.5..2)
  - with the rig's room under it: a random piece of a recorded "quiet" take, at a random 0.5..3 times its own
    level (the microphone, the room and the board's noise of the real thing)
Speaker ``--holdout`` (default 04) is written with "_test" in the stem, so the trainers can keep it out.
The clips are long phrases with pauses; the ear's and the memory's labellers find the whistled parts.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import wave

import numpy as np
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
SR = 16000
CORPORA = {
    "silbo": (ROOT / "runs/archive/silbo_gomero/fragments/fragments/clips", ROOT / "runs/archive/silbo_whistles"),
    "mlend": (ROOT / "runs/archive/mlend_hw/audio", ROOT / "runs/archive/mlend_whistles"),
}


def held_out(corpus, path, holdout, people):
    """True for the test takes: Silbo's speaker ``holdout``; every 8th MLEnd whistler."""
    if corpus == "silbo":
        return path.name[:2] == holdout
    return int(people.get(path.name, 0)) % 8 == 0


def read_wav(path):
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, ch).mean(1)
    return x / 32768.0, sr


def voiced_rms(x, frame=320):
    """Median RMS of the loudest half of the 20 ms frames (the whistled part of a take)."""
    n = len(x) // frame
    if n == 0:
        return 0.0
    r = np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(1))
    return float(np.median(np.sort(r)[n // 2:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", nargs="*", default=[str(ROOT / "runs/real/whistle_data")],
                    help="the rig's record-only takes (their whistles set the loudness, their quiet takes the room)")
    ap.add_argument("--corpus", choices=list(CORPORA), default="silbo")
    ap.add_argument("--holdout", default="04", help="Silbo speaker kept for the test")
    ap.add_argument("--minutes", type=float, default=0, help="at most this much (0: all), spread over the speakers")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    whistles, rooms = [], []
    for folder in args.rig:
        for p in sorted(pathlib.Path(folder).glob("*.npz")):
            d = np.load(p)
            kind = str(d["kind"]) if "kind" in d.files else "whistle"
            x = d["pcm"].astype(np.float64) / 32768.0
            (whistles if kind == "whistle" else rooms if kind == "quiet" else []).append(x)
    if not whistles or not rooms:
        raise SystemExit("need the rig's whistle and quiet takes (runs/real/whistle_data)")
    level = float(np.median([voiced_rms(x) for x in whistles]))
    room = np.concatenate(rooms)
    room_rms = float(np.sqrt(np.mean(room ** 2)))
    print(f"rig whistles: median level {20 * np.log10(level):.1f} dBFS; room {20 * np.log10(room_rms):.1f} dBFS "
          f"({len(room) / SR:.0f} s of quiet takes)")
    src, OUT = CORPORA[args.corpus]
    clips = sorted(src.glob("*.wav"))
    if not clips:
        raise SystemExit(f"no clips in {src} (python scripts/archive_fetch.py {args.corpus})")
    people = {}
    manifest = src.parent / "manifest.json"
    if args.corpus == "mlend" and manifest.exists():
        import json
        people = {m["file"]: m["interpreter"] for m in json.loads(manifest.read_text("utf-8"))}
    if args.minutes:
        rng.shuffle(clips)
    OUT.mkdir(parents=True, exist_ok=True)
    total = {}
    for path in clips:
        x, sr = read_wav(path)
        y = resample_poly(x, 160, 441) if sr == 44100 else resample_poly(x, SR, sr)
        spk = path.name[:2] if args.corpus == "silbo" else ("test" if held_out(args.corpus, path, args.holdout, people)
                                                                 else "train")
        if args.minutes and sum(total.values()) / 60 > args.minutes:
            break
        y = y / max(voiced_rms(y), 1e-6) * level * rng.uniform(.5, 2.0)
        start = rng.integers(0, max(1, len(room) - len(y)))
        bg = np.resize(room[start:start + len(y)], len(y))
        y = np.clip(y + bg * rng.uniform(.5, 3.0), -1, 1)
        test = held_out(args.corpus, path, args.holdout, people)
        stem = f"{args.corpus}_{path.stem}" + ("_test" if test else "")
        np.savez_compressed(OUT / f"{stem}_whistle.npz", pcm=(y * 32767).astype(np.int16), sr=SR, kind="whistle")
        total[spk] = total.get(spk, 0) + len(y) / SR
    print(f"{len(total) and sum(1 for _ in OUT.glob('*.npz'))} takes in {OUT.relative_to(ROOT)}: "
          + ", ".join(f"{k} {v / 60:.1f} min" for k, v in sorted(total.items())))


if __name__ == "__main__":
    sys.exit(main())
