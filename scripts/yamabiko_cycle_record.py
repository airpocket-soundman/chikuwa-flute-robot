"""PC: run the plunger end to end a few times while recording the microphone, for analysis afterwards.

    python scripts/yamabiko_cycle_record.py [--cycles 5 --fan 0.5 --out runs/real/cycle_rec_01.npz]

Two connections to the UNO Q's command server (through adb forward): one drives (valve open, fan on,
the actuator at full PWM out for --stroke-s, then in), the other records the microphone over the whole
run.  The status (applied PWM, motor current, pitch) is logged every ~40 ms on the PC clock; the audio
starts at `rec_t0` on the same clock (when the record request was sent).  The bench web page must not be
driving the outputs meanwhile.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.remote import Remote  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--fan", type=float, default=.5)
    ap.add_argument("--pwm", type=float, default=1.0)
    ap.add_argument("--stroke-s", type=float, default=2.0, help="drive time per direction (an end takes ~1.4 s)")
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--valve-us", type=int, nargs=2, default=(1560, 1820), metavar=("OPEN", "CLOSED"),
                    help="SG90 pulses set from the bench page on the current rig")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default="runs/real/cycle_rec_01.npz")
    args = ap.parse_args()

    plan = [("home", -args.pwm)] + [(f"{d}{c}", s * args.pwm) for c in range(args.cycles)
                                    for d, s in (("out", 1), ("in", -1))]
    seconds = args.warmup_s + len(plan) * args.stroke_s + 1.0
    if seconds > 30:
        raise SystemExit(f"{seconds:.0f} s is more than one 30 s recording: use fewer cycles")

    drive = Remote(port=args.port)
    ear = Remote(port=args.port, forward=False)
    rec = {}

    def record():
        rec["t0"] = time.monotonic()
        rec["pcm"], rec["sr"] = ear.record(seconds)

    t0 = time.monotonic()
    log, phases = [], []
    th = threading.Thread(target=record)
    try:
        th.start()
        time.sleep(.05)
        drive.valve_positions(*args.valve_us, move_ms=20)
        drive.valve(True)
        drive.fan(True, args.fan)

        def watch(until, tag):
            while time.monotonic() < until:
                s = drive.status()
                log.append((time.monotonic() - t0, s.get("applied_pwm", np.nan), s.get("isense_mv", -1),
                            s["pitch_hz"] or np.nan, s["mic_db"]))
                time.sleep(.04)

        watch(time.monotonic() + args.warmup_s, "warmup")
        for tag, pwm in plan:
            start = time.monotonic()
            phases.append((tag, start - t0, pwm))
            drive.actuator(pwm)
            watch(start + args.stroke_s, tag)
            drive.actuator(0.0)
        th.join()
    finally:
        drive.stop()
        drive.close(); ear.close()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pcm=rec["pcm"], sr=rec["sr"], rec_t0=rec["t0"] - t0,
                        log=np.array(log, float), log_cols=np.array(["t", "applied_pwm", "isense_mv", "pitch_hz", "mic_db"]),
                        phase_tag=np.array([p[0] for p in phases]), phase_t=np.array([p[1] for p in phases]),
                        phase_pwm=np.array([p[2] for p in phases]), fan=args.fan)
    print(f"saved {out}: {rec['pcm'].size / rec['sr']:.1f} s of audio, {len(log)} status rows, {len(phases)} strokes")


if __name__ == "__main__":
    main()
