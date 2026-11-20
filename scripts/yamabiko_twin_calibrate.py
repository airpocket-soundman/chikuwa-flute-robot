"""UNO Q: run the calibration on the real rig and save its record.

    python3 scripts/yamabiko_twin_calibrate.py --out runs/real/calibration_01.npz [--max-pwm 0.6 --sweep 0.5]

Homes the plunger, then plays the fixed PWM sequence with the valve open and
records commands and heard pitch (nothing else).  Use ``--max-pwm`` and
``--sweep`` on a new mechanism until end-stop hits and overblowing are
understood.  Copy the file to the PC and run scripts/yamabiko_twin_fit.py.
Only NumPy is needed on the UNO Q.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.calibration import calibrate_real  # noqa: E402
from flute_rl.yamabiko.hw import Link, RealRig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default="/dev/ttyHS1")
    ap.add_argument("--flip", action="store_true", help="pwm > 0 must shorten the tube (raise the pitch)")
    ap.add_argument("--fan", type=float, default=.7)
    ap.add_argument("--max-pwm", type=float, default=1.0)
    ap.add_argument("--sweep", type=float, default=1.0)
    ap.add_argument("--homing-seconds", type=float, default=1.3)
    args = ap.parse_args()
    link = Link(args.dev); rig = RealRig(link, flip=args.flip, max_pwm=args.max_pwm)
    rig.start(); link.fan(True, args.fan)
    try:
        record = calibrate_real(rig, max_pwm=args.max_pwm, sweep=args.sweep, homing_seconds=args.homing_seconds)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **record)
        valid = record["0/valid"][0]
        heard = record["0/heard"][0][valid]
        pitch = f"pitch {heard.min():.0f}..{heard.max():.0f} cent" if valid.any() else "no pitch heard"
        print(f"saved {args.out}: {valid.size} steps, {int(valid.sum())} voiced, {pitch}, late steps {rig.late_steps}")
    finally:
        rig.safe(); link.fan(False, 0.0); link.close()


if __name__ == "__main__":
    main()
