"""PC: drive the plunger through a test program while recording the microphone, for analysis afterwards.

    python scripts/yamabiko_motion_test.py sweep --pwm 0.6 --cycles 3 --out runs/real/motion/sweep_060.npz
    python scripts/yamabiko_motion_test.py steps --pulse-ms 40 --out-steps 18 --repeats 3 --out ...
    python scripts/yamabiko_motion_test.py return --times 0.2 0.4 0.6 --repeats 5 --out ...

Programs (all start by homing to the retracted end; "home" = full PWM in until the current stops):
  sweep   full strokes at --pwm: out until the end switch cuts the current (or --stroke-s), rest, back in
  steps   --pulse-ms pulses at full PWM with --rest-s between: --out-steps out, then in until home
  return  from home, one full-PWM run of each --times, rest, then home again; --repeats times
  fan     from home, one full-PWM run of each --times, then the fan at each --fan-levels for --fan-hold-s
          (the only program that changes the fan; the others keep --fan)

Three connections to the UNO Q's command server (adb forward): one sends only the drive commands (so
the pulse timing is not held up), one polls the status every ~25 ms, one records the microphone in
back-to-back 5 s chunks.  Everything is stamped on the PC's monotonic clock.  The npz holds the audio
(`pcm`, with `chunk_t0`/`chunk_n`), the status log and every actuator command actually sent.  The bench
web page must not be driving the outputs meanwhile.
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

CHUNK_S = 5.0
END_MV = 3          # motor current at or below this (mV on the 0.1 ohm shunt) = the end switch has cut it


class Rig:
    def __init__(self, host, port, t0):
        self.t0 = t0
        self.drive = Remote(host, port=port, forward=host == "127.0.0.1")
        self.watch = Remote(host, port=port, forward=False)
        self.ear = Remote(host, port=port, forward=False)
        self.cmds = []                  # (t, pwm, label)
        self.log = []                   # (t, applied_pwm, isense_mv, pitch_hz, mic_db, isense_peak_mv)
        self.pcm, self.chunk_t0, self.chunk_n = [], [], []
        self.done = threading.Event()
        self.isense = 0
        self.threads = [threading.Thread(target=self._poll, daemon=True),
                        threading.Thread(target=self._listen, daemon=True)]
        for th in self.threads:
            th.start()

    def now(self):
        return time.monotonic() - self.t0

    def _poll(self):
        while not self.done.is_set():
            s = self.watch.status()
            self.isense = s.get("isense_mv", 0)
            self.log.append((self.now(), s.get("applied_pwm", np.nan), self.isense,
                             s["pitch_hz"] or np.nan, s["mic_db"], s.get("isense_peak_hold_mv", s.get("isense_peak_mv", 0))))
            time.sleep(.025)

    def _listen(self):
        while not self.done.is_set():
            t = self.now()
            try:
                pcm, sr = self.ear.record(CHUNK_S)
            except Exception as exc:     # the server restarted a stalled stream: lose this chunk, go on
                print(f"recording chunk at {t:.1f} s lost: {exc}", flush=True)
                continue
            self.pcm.append(pcm); self.chunk_t0.append(t); self.chunk_n.append(pcm.size)
            self.sr = sr

    def act(self, pwm, label=""):
        self.drive.actuator(pwm)
        self.cmds.append((self.now(), pwm, label))

    def hold(self, seconds):
        time.sleep(max(0.0, seconds))

    def pulse(self, pwm, seconds, label):
        start = time.monotonic()
        self.act(pwm, label)
        while (left := seconds - (time.monotonic() - start)) > 0:
            time.sleep(min(left, .002) if left < .01 else left - .008)
        self.act(0.0, label)

    def run_to_end(self, pwm, label, limit_s=4.0):
        """Drive until the current drops (the end switch) or limit_s; returns the run time."""
        start = time.monotonic()
        self.act(pwm, label)
        time.sleep(.25)                 # past the inrush, and long enough for the poll to see the motor
        while time.monotonic() - start < limit_s:
            if self.isense <= END_MV:
                break
            time.sleep(.01)
        took = time.monotonic() - start
        self.act(0.0, label)
        return took

    def home(self):
        return self.run_to_end(-1.0, "home", 3.0)

    def finish(self):
        self.drive.stop()
        time.sleep(.3)
        self.done.set()
        for th in self.threads:
            th.join(timeout=CHUNK_S + 15)
        for r in (self.drive, self.watch, self.ear):
            r.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program", choices=["sweep", "steps", "return", "fan"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fan", type=float, default=.5)
    ap.add_argument("--valve-us", type=int, nargs=2, default=(1560, 1820), metavar=("OPEN", "CLOSED"))
    ap.add_argument("--host", default="127.0.0.1", help="the UNO Q's address when its server listens on Wi-Fi")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--rest-s", type=float, default=.5)
    ap.add_argument("--pwm", type=float, default=1.0, help="sweep: drive level")
    ap.add_argument("--cycles", type=int, default=3, help="sweep: strokes out and in")
    ap.add_argument("--stroke-s", type=float, default=5.0, help="sweep: give up a stroke after this long")
    ap.add_argument("--pulse-ms", type=float, default=40, help="steps: pulse length")
    ap.add_argument("--out-steps", type=int, default=18, help="steps: pulses out before coming back")
    ap.add_argument("--max-in-steps", type=int, default=60, help="steps: give up coming back after this many")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--times", type=float, nargs="+", default=[.2, .4, .6], help="return, fan: run times out")
    ap.add_argument("--fan-levels", type=float, nargs="+", default=[.5, .6, .7, .8, .9, 1.0])
    ap.add_argument("--fan-hold-s", type=float, default=3.0)
    args = ap.parse_args()

    t0 = time.monotonic()
    rig = Rig(args.host, args.port, t0)
    marks = []                          # (t, label) for the analysis

    def mark(label):
        marks.append((rig.now(), label))

    took = []
    try:
        time.sleep(.2)
        rig.drive.valve_positions(*args.valve_us, move_ms=20)
        rig.drive.valve(True)
        rig.drive.fan(True, args.fan)
        rig.hold(args.warmup_s)
        mark("home"); rig.home(); rig.hold(1.0)
        if args.program == "sweep":
            for c in range(args.cycles):
                for d in (1, -1):
                    tag = f"{'out' if d > 0 else 'in'}{c}"
                    mark(tag)
                    took.append((tag, rig.run_to_end(d * args.pwm, tag, args.stroke_s)))
                    mark(f"rest_{tag}"); rig.hold(1.0)
        elif args.program == "steps":
            pulse = args.pulse_ms / 1000
            def dead():
                """The last pulse drew no current (an end switch had cut the motor)."""
                recent = [row[5] for row in rig.log if row[0] > rig.cmds[-2][0] + .01]
                return bool(recent) and max(recent) <= END_MV

            for r in range(args.repeats):
                # out until --out-steps, or until two pulses in a row found the extended end
                quiet = 0
                for i in range(args.out_steps):
                    rig.pulse(1.0, pulse, f"r{r}_out{i}"); mark(f"rest_r{r}_out{i}"); rig.hold(args.rest_s)
                    quiet = quiet + 1 if dead() and i > 2 else 0
                    if quiet >= 2:
                        break
                quiet = 0
                for i in range(args.max_in_steps):
                    rig.pulse(-1.0, pulse, f"r{r}_in{i}"); mark(f"rest_r{r}_in{i}"); rig.hold(args.rest_s)
                    quiet = quiet + 1 if dead() and i > 2 else 0
                    if quiet >= 2:
                        break
                mark(f"home_r{r}"); rig.home(); rig.hold(.6)
        elif args.program == "fan":
            for t_out in args.times:
                tag = f"t{int(round(t_out * 1000))}"
                if t_out > 0:
                    rig.pulse(1.0, t_out, tag)
                for level in args.fan_levels:
                    rig.drive.fan(True, level)
                    mark(f"rest_{tag}_f{int(round(level * 100))}"); rig.hold(args.fan_hold_s)
                rig.drive.fan(True, args.fan)
                mark(f"home_{tag}"); took.append((tag, rig.home())); rig.hold(1.0)
        else:
            for r in range(args.repeats):
                for t_out in args.times:
                    tag = f"r{r}_t{int(t_out * 1000)}"
                    rig.pulse(1.0, t_out, tag); mark(f"rest_{tag}"); rig.hold(1.0)
                    mark(f"home_{tag}"); took.append((tag, rig.home())); rig.hold(.6)
    finally:
        rig.finish()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, program=args.program, args=str(vars(args)), sr=rig.sr,
        pcm=np.concatenate(rig.pcm), chunk_t0=np.array(rig.chunk_t0), chunk_n=np.array(rig.chunk_n),
        log=np.array(rig.log, float), log_cols=np.array(["t", "applied_pwm", "isense_mv", "pitch_hz", "mic_db", "isense_peak_mv"]),
        cmd_t=np.array([c[0] for c in rig.cmds]), cmd_pwm=np.array([c[1] for c in rig.cmds]),
        cmd_label=np.array([c[2] for c in rig.cmds]),
        mark_t=np.array([m[0] for m in marks]), mark_label=np.array([m[1] for m in marks]),
        took_label=np.array([t[0] for t in took]), took_s=np.array([t[1] for t in took]))
    print(f"saved {out}: {sum(rig.chunk_n) / rig.sr:.1f} s audio, {len(rig.log)} status rows, {len(rig.cmds)} commands")
    for tag, s in took:
        print(f"  {tag}: {s:.2f} s")


if __name__ == "__main__":
    main()
