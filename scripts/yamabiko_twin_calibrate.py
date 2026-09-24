"""UNO Q: run the calibration on the real rig and save its record.

    python3 scripts/yamabiko_twin_calibrate.py --out runs/real/calibration_01.npz [--max-pwm 0.6 --sweep 0.5]

Homes the plunger, then plays the fixed PWM sequence with the valve open and
records commands and heard pitch (nothing else).  Use ``--max-pwm`` and
``--sweep`` on a new mechanism until end-stop hits and overblowing are
understood.  Copy the file to the PC and run scripts/yamabiko_twin_fit.py.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.device import RealDevice, save_logs  # noqa: E402
from flute_rl.yamabiko.fitting import run_calibration  # noqa: E402
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
    device = RealDevice(rig, homing_seconds=args.homing_seconds)
    try:
        log = run_calibration(device, max_pwm=args.max_pwm, sweep=args.sweep)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_logs(args.out, device.logs)
        heard = log.heard[log.valid]
        print(f"saved {args.out}: {log.pwm.shape[1]} steps, {int(log.valid.sum())} voiced, "
              f"pitch {float(heard.min()):.0f}..{float(heard.max()):.0f} cent, late steps {rig.late_steps}")
    finally:
        rig.safe(); link.fan(False, 0.0); link.close()


if __name__ == "__main__":
    main()
