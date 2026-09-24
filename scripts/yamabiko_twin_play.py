"""UNO Q: play a target track on the real rig with the NumPy runtime.

    python3 scripts/yamabiko_twin_play.py --runtime runs/real/twin_01_runtime.npz --target runs/real/target.npz \
        --repeats 3 --out runs/real/play_01.npz

The target file holds ``cents`` (float, per 10 ms) and ``voice`` (bool),
written by scripts/yamabiko_twin_target.py (a reference phrase) or by the
listening pipeline.  The song memory is kept across the repeats, so later
plays should be better; every play is recorded for scripts/yamabiko_twin_fit.py.
Only NumPy is needed on the UNO Q.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.hw import Link, RealRig  # noqa: E402
from flute_rl.yamabiko.residual_numpy import Runtime  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dev", default="/dev/ttyHS1")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--fan", type=float, default=.7)
    ap.add_argument("--max-pwm", type=float, default=1.0)
    ap.add_argument("--homing-seconds", type=float, default=1.3)
    args = ap.parse_args()
    runtime = Runtime(dict(np.load(args.runtime)))
    target = np.load(args.target); cents, voice = target["cents"].astype(np.float64), target["voice"].astype(bool)
    link = Link(args.dev); rig = RealRig(link, flip=args.flip, max_pwm=args.max_pwm)
    rig.start(); link.fan(True, args.fan)
    records = {}
    try:
        for play in range(1, args.repeats + 1):
            rig.resync()
            for _ in range(int(round(args.homing_seconds / .01))):
                rig.step(-1.0, False)
            runtime.prepare(cents, voice, keep_song_memory=play > 1)
            pwm_log, heard_log, valid_log = [], [], []
            for t in range(len(cents)):
                pwm, valve = runtime.command(t)
                out = rig.step(pwm, valve)
                heard = float(out["heard"][0]); valid = bool(np.isfinite(heard))
                runtime.observe(heard if valid else 0.0, valid)
                pwm_log.append(pwm); heard_log.append(heard if valid else 0.0); valid_log.append(valid)
            heard_arr, valid_arr = np.array(heard_log), np.array(valid_log)
            mask = valid_arr & voice
            error = np.abs(heard_arr[mask] - cents[mask]).mean() if mask.any() else float("nan")
            print(f"play {play}: heard error {error:.1f} cent over {int(mask.sum())} frames, late steps {rig.late_steps}",
                  flush=True)
            index = play - 1
            records[f"{index}/pwm"] = np.array(pwm_log, np.float32)[None]
            records[f"{index}/heard"] = heard_arr.astype(np.float32)[None]
            records[f"{index}/valid"] = valid_arr[None]
            records[f"{index}/valve"] = voice.astype(np.float32)[None]
            records[f"{index}/target_cents"] = cents.astype(np.float32)[None]
            records[f"{index}/target_voice"] = voice[None]
    finally:
        rig.safe(); link.fan(False, 0.0); link.close()
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, count=args.repeats, **records); print(f"saved {args.out}")


if __name__ == "__main__":
    main()
