"""UNO Q: step the plunger across its stroke and log the heard pitch, through the bench web page's API.

    python3 scripts/yamabiko_pitch_sweep.py [--fan 0.5 --step-ms 40 --out runs/real/pitch_sweep.csv]

Needs scripts/yamabiko_web.py running (it owns the outputs).  Opens the valve, runs the fan, drives the
plunger to the retracted end, then steps it out to the extended end and back in, measuring the pitch
after every step.  The position is open loop: it is the summed drive time at full PWM from the
retracted end.  Leave the bench page alone while this runs.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
import urllib.request


class Bench:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def _req(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"{self.url}/api/{path}", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def status(self):
        return self._req("status")

    def actuator(self, pwm):
        return self._req("actuator", {"pwm": pwm})

    def fan(self, on, duty):
        return self._req("fan", {"on": on, "duty": duty})

    def valve(self, open_):
        return self._req("valve", {"open": open_})

    def stop(self):
        return self._req("stop", {})

    def drive(self, pwm, seconds):
        """Full-PWM run for `seconds`, repeating the command so the page's 0.4 s watchdog stays quiet."""
        end = time.monotonic() + seconds
        self.actuator(pwm)
        while (left := end - time.monotonic()) > 0:
            time.sleep(min(left, .1))
            if end - time.monotonic() > 0:
                self.actuator(pwm)
        self.actuator(0.0)

    def listen(self, n=6, gap=.06):
        """Median pitch (Hz, None when unvoiced) and level over n status reads."""
        hz, db, conf = [], [], []
        for _ in range(n):
            s = self.status()
            db.append(s["mic_db"]); conf.append(s["pitch_conf"])
            if s["pitch_hz"] is not None:
                hz.append(s["pitch_hz"])
            time.sleep(gap)
        voiced = len(hz) * 2 > n
        return (statistics.median(hz) if voiced else None), statistics.median(db), statistics.median(conf)


def cents(f, ref):
    return 1200 * math.log2(f / ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--fan", type=float, default=.5)
    ap.add_argument("--pwm", type=float, default=1.0)
    ap.add_argument("--step-ms", type=float, default=40)
    ap.add_argument("--home-s", type=float, default=2.5, help="drive time that surely reaches the retracted end")
    ap.add_argument("--max-out-ms", type=float, default=1280, help="stop stepping out after this much drive")
    ap.add_argument("--max-in-ms", type=float, default=1800, help="stop stepping in after this much drive")
    ap.add_argument("--warmup-s", type=float, default=4.0, help="fan run before the first reading")
    ap.add_argument("--settle-s", type=float, default=.35)
    ap.add_argument("--out", default="runs/real/pitch_sweep.csv")
    args = ap.parse_args()

    b = Bench(args.url)
    rows = []
    try:
        b.valve(True); b.fan(True, args.fan); time.sleep(args.warmup_s)
        b.drive(-args.pwm, args.home_s); time.sleep(.5)
        for direction, name in ((1, "out"), (-1, "in")):
            pos, still = 0.0 if direction > 0 else rows[-1][1], 0
            prev = None
            limit = args.max_out_ms if direction > 0 else args.max_in_ms
            for step in range(int(limit // args.step_ms) + 1):
                if step:
                    b.drive(direction * args.pwm, args.step_ms / 1000)
                    pos += direction * args.step_ms
                time.sleep(args.settle_s)
                hz, db, conf = b.listen()
                peak = b.status()["isense_peak_mv"]
                rows.append((name, pos, hz, db, conf, peak))
                print(f"{name:3s} {pos:6.0f} ms  {'-' if hz is None else f'{hz:7.1f} Hz'}  {db:6.1f} dB  conf {conf:.2f}",
                      flush=True)
                # the end stop: three steps in a row that move the pitch by under 3 cents
                if step and hz is not None and prev is not None and abs(cents(hz, prev)) < 3:
                    still += 1
                else:
                    still = 0
                if still >= 3:
                    break
                prev = hz
    finally:
        b.stop()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("sweep,drive_ms,pitch_hz,mic_db,pitch_conf,isense_peak_mv\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
