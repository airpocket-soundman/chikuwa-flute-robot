"""PC: jog the actuator with the EXEC button through the UNO Q's command server.

    python scripts/yamabiko_button_jog.py [--pwm 1.0 --seconds 180]

While the button is held the actuator runs; each new press reverses the direction, starting with
pwm > 0 (extends on the first rig).  Released, or when the script ends, the actuator stops.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.remote import Remote  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pwm", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    with Remote(port=args.port) as rig:
        cap = rig.status()["max_pwm"]
        rig.limit(max(cap, abs(args.pwm)))
        direction, held, presses = 1.0, False, 0
        print(f"ready for {args.seconds:.0f} s: hold EXEC to extend, the next press retracts", flush=True)
        until = time.monotonic() + args.seconds
        try:
            while time.monotonic() < until:
                now_held = bool(rig.status().get("button_held"))
                if now_held and not held:
                    presses += 1
                    rig.actuator(direction * abs(args.pwm))
                    print(f"press {presses}: {'extend' if direction > 0 else 'retract'}", flush=True)
                elif held and not now_held:
                    rig.actuator(0.0)
                    direction = -direction
                    print("  released", flush=True)
                held = now_held
                time.sleep(.02)
        finally:
            rig.actuator(0.0)
            rig.limit(cap)
            print("done", flush=True)


if __name__ == "__main__":
    main()
