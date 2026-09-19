"""How well the real rig hears a flute-like tone, measured with the microphone alone (PC side).

The PC plays a known sequence of tones through its speaker while the UNO Q runs the real control loop
with the plunger idle (yamabiko_collect.py --songs 0), which logs the pitch heard at every 10 ms step
exactly as the controller gets it (20 ms YIN window, flute_rl/yamabiko/hw.py). The log is then compared
with what was played:

* error and spread of the heard pitch on each steady tone [cents]  -> pitch_noise
* steady frames with no pitch                                      -> dropout
* frames an octave or a twelfth off                                -> octave_err
* steps from the sound reaching the MCU to the first pitch heard   -> part of obs_delay

The tones cover the flute's range (650-1350 Hz), as sines and as closed-tube tones (odd harmonics:
1, 3, 5 at -10, -20 dB), whose third harmonic is where the estimator could slip a twelfth up.

    python scripts/yamabiko_mic_check.py run --out runs/real/mic_check.npz     # needs adb and the UNO Q
    python scripts/yamabiko_mic_check.py analyze runs/real/mic_check.npz
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FS = 48000
TONE_S, GAP_S, LEAD_S = 0.8, 0.4, 2.0
FREQS = (659.3, 740.0, 830.6, 880.0, 987.8, 1108.7, 1244.5, 1318.5)
ADB = os.environ.get("ADB", r"C:\Users\yamas\platform-tools\adb.exe")
REMOTE = "/home/arduino/yamabiko"


def sequence() -> list[tuple[str, float]]:
    return [(kind, f) for kind in ("sine", "closed") for f in FREQS]


def render(seq, amp: float = 0.3) -> bytes:
    """The whole sequence as a 16-bit mono WAV: LEAD_S of silence, then TONE_S tones with GAP_S gaps."""
    n_tone, n_gap = int(TONE_S * FS), int(GAP_S * FS)
    parts = [np.zeros(int(LEAD_S * FS))]
    t = np.arange(n_tone) / FS
    ramp = np.minimum(1.0, np.minimum(t, TONE_S - t) / 0.005)        # 5 ms fades: no clicks
    for kind, f in seq:
        if kind == "sine":
            x = np.sin(2 * np.pi * f * t)
        else:
            x = np.sin(2 * np.pi * f * t) + 0.316 * np.sin(2 * np.pi * 3 * f * t) + 0.1 * np.sin(2 * np.pi * 5 * f * t)
            x /= 1.416
        parts += [amp * ramp * x, np.zeros(n_gap)]
    y = np.concatenate(parts + [np.zeros(FS)])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def cmd_run(out: pathlib.Path) -> None:
    import winsound
    seq = sequence()
    wav = render(seq)
    seconds = LEAD_S + len(seq) * (TONE_S + GAP_S) + 3.0
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    remote = f"runs/real/{out.name}"
    proc = subprocess.Popen([ADB, "shell", f"cd {REMOTE} && python3 scripts/yamabiko_collect.py --songs 0 --fan 0 "
                                           f"--settle {seconds + 2.0:.1f} --out {remote}"],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    reader = threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True)
    reader.start()
    time.sleep(4.0)                                   # python and numpy start on the UNO Q, the stream settles
    winsound.PlaySound(wav, winsound.SND_MEMORY)      # blocks while playing
    proc.wait(timeout=seconds + 60)
    reader.join(timeout=5)
    print("".join(lines).strip())
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ADB, "pull", f"{REMOTE}/{remote}", str(out)], env=env, check=True, capture_output=True)
    print("pulled", out)


def segments(level_db: np.ndarray, floor_db: float) -> list[tuple[int, int]]:
    """Runs of steps clearly over the floor: (first, one past the last)."""
    loud = level_db > floor_db + 15.0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], loud.astype(int), [0]])))
    runs = list(zip(edges[::2], edges[1::2]))
    return [(a, b) for a, b in runs if b - a >= int(0.5 * TONE_S / 0.01)]


def cmd_analyze(path: pathlib.Path) -> dict:
    d = np.load(path)
    heard, level = d["heard"].astype(float), d["level_db"].astype(float)
    t_step, sample = d["t_step"].astype(float), d["sample"].astype(float)
    floor = float(np.percentile(level, 20))
    segs = segments(level, floor)
    seq = sequence()
    print(f"{len(heard)} steps, floor {floor:.1f} dBFS, {len(segs)} tones found (played {len(seq)})")
    if len(segs) != len(seq):
        raise SystemExit("the tones found do not match the sequence played: check the speaker volume")
    rows, steady_err, n_steady, n_missing, n_slip = [], [], 0, 0, 0
    delays = []
    audio, a0 = d["audio"].astype(float) / 32768.0, int(d["audio_start"])
    for (a, b), (kind, f) in zip(segs, seq):
        want = 1200.0 * np.log2(f / 440.0)
        # steady part: skip the first and last 60 ms (window filling, fade)
        s = slice(a + 6, b - 6)
        h = heard[s]
        ok = np.isfinite(h)
        dev = h[ok] - want
        slip = np.abs(dev) > 600.0                     # an octave (1200) or a twelfth (1902) off
        good = dev[~slip]
        n_steady += len(h)
        n_missing += int((~ok).sum())
        n_slip += int(slip.sum())
        steady_err.append(good)
        # onset: the sample where the envelope rises, then the first step with a pitch
        lo = max(0, int(sample[max(a - 10, 0)]) - a0)
        x = np.abs(audio[lo:lo + int(0.3 * 16000)])
        env = np.convolve(x, np.ones(32) / 32, mode="same")
        on = lo + int(np.argmax(env > 0.25 * env.max()))
        # the step whose newest sample first reaches past the onset, and the first step with a pitch
        t_on = float(np.interp(on + a0, sample, t_step))
        a0s = max(a - 10, 0)                           # a pitch can come before the level passes the gate
        first = a0s + int(np.argmax(np.isfinite(heard[a0s:b])))
        delays.append(1000.0 * (t_step[first] - t_on))
        rows.append((kind, f, float(np.median(good)) if good.size else float("nan"),
                     float(np.std(good)) if good.size > 1 else float("nan"), int((~ok).sum()), int(slip.sum()), len(h)))
    print(f"\n{'tone':7s} {'Hz':>7s} {'median':>8s} {'std':>6s} {'missing':>8s} {'slips':>6s}   [cents, steps]")
    for kind, f, med, sd, miss, sl, n in rows:
        print(f"{kind:7s} {f:7.1f} {med:8.1f} {sd:6.1f} {miss:5d}/{n:<3d} {sl:5d}")
    allg = np.concatenate(steady_err)
    offset = float(np.median(allg))
    noise = float(1.4826 * np.median(np.abs(allg - offset)))
    res = {"pitch_noise": noise, "offset_cents": offset, "std_cents": float(np.std(allg)),
           "dropout": n_missing / n_steady, "octave_err": n_slip / n_steady,
           "onset_ms_median": float(np.median(delays)), "onset_ms": [round(v, 1) for v in delays]}
    print(f"\nheard pitch on steady tones: offset {offset:+.2f} cents, noise {noise:.2f} cents (robust), "
          f"std {res['std_cents']:.2f}")
    print(f"missing {n_missing}/{n_steady} ({100 * res['dropout']:.2f} %), octave / twelfth slips {n_slip} "
          f"({100 * res['octave_err']:.2f} %)")
    print(f"sound reaching the MCU -> first step with a pitch: median {res['onset_ms_median']:.1f} ms "
          f"(min {min(delays):.1f}, max {max(delays):.1f}); add ~2.8 ms for the link")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/real/mic_check.npz"))
    a = sub.add_parser("analyze")
    a.add_argument("log", type=pathlib.Path)
    a.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args.out)
        res = cmd_analyze(args.out)
        args.out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    else:
        res = cmd_analyze(args.log)
        if args.json:
            args.json.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
