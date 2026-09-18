"""Collect training data on the real Yamabiko No.1 (runs on the UNO Q's Linux side).

The rig plays random songs from the same generator as the simulator's training, with a controller of
control.py and, by default, some noise added to its commands so the log covers more than one way of
moving. Before each song the plunger is homed (1 s pulling out against the end stop), as in the
simulator. Every 10 ms step is logged: the command sent, the pitch heard, the motor current, the fan
speed, and the whole audio. scripts/yamabiko_fit.py fits the simulator's rig to these logs (on the PC).

    sudo systemctl stop arduino-router arduino-router-serial     # the firmware owns /dev/ttyHS1
    python3 scripts/yamabiko_collect.py --songs 20 --out runs/real/collect_01.npz

The log is saved after every song, so Ctrl-C keeps what was played. The fan is switched off at the end.
Songs of one level only (--level 0 = one long note, 2 = glides) exercise one thing at a time.
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def add_rig_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dev", default="/dev/ttyHS1")
    ap.add_argument("--rig", default=None, help="nominal rig from yamabiko_fit.py (sets YAMABIKO_RIG)")
    ap.add_argument("--fan", type=float, default=0.7, help="fan PWM duty 0..1")
    ap.add_argument("--settle", type=float, default=3.0, help="seconds for the fan to reach speed")
    ap.add_argument("--valve-open", type=int, default=590, help="SCS0009 position of the open valve (0..1023)")
    ap.add_argument("--valve-closed", type=int, default=512)
    ap.add_argument("--valve-ms", type=int, default=50, help="time the servo takes to open or close")
    ap.add_argument("--flip", action="store_true", help="pwm > 0 must shorten the tube: flip if it does not")
    ap.add_argument("--max-pwm", type=float, default=1.0, help="limit of |pwm| in the firmware")


def open_rig(args):
    """Link and RealRig started, fan on and settled. Import flute_rl only after YAMABIKO_RIG is set."""
    from flute_rl.yamabiko.hw import Link, RealRig
    link = Link(args.dev)
    link.valve_positions(args.valve_open, args.valve_closed, args.valve_ms)
    rig = RealRig(link, flip=args.flip, max_pwm=args.max_pwm)
    rig.start()
    link.fan(args.fan > 0.0, args.fan)
    return link, rig


def close_rig(link, rig) -> None:
    rig.safe()
    try:
        link.fan(False, 0.0)
        link.stop()
    finally:
        link.close()


class OUNoise:
    """Slowly varying noise on the PWM (Ornstein-Uhlenbeck), so the plunger also moves in ways the
    controller would not choose."""

    def __init__(self, rng, sigma: float, tau: float, dt: float):
        self.rng, self.sigma, self.a, self.e = rng, sigma, dt / tau, 0.0

    def __call__(self, pwm):
        self.e += -self.e * self.a + self.sigma * (2.0 * self.a) ** 0.5 * float(self.rng.standard_normal())
        return pwm + self.e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_rig_args(ap)
    ap.add_argument("--songs", type=int, default=10)
    ap.add_argument("--level", type=int, default=-1, help="song level 0..4, -1 = mixed as in training")
    ap.add_argument("--controller", choices=("observer", "open_loop", "gru"), default="observer")
    ap.add_argument("--policy", default="runs/yamabiko_gru.npz", help="for --controller gru")
    ap.add_argument("--noise", type=float, default=0.15, help="std of the noise added to the PWM (0 = none)")
    ap.add_argument("--noise-tau", type=float, default=0.15, help="its time constant [s]")
    ap.add_argument("--rest", type=float, default=0.5, help="seconds between songs, plunger stopped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.rig:
        os.environ["YAMABIKO_RIG"] = str(pathlib.Path(args.rig).resolve())

    import numpy as np

    from flute_rl.sim import DT
    from flute_rl.targets import make_target, sample_level
    from flute_rl.yamabiko import GRUPolicy, Observer, OpenLoop, make_schedule
    from flute_rl.yamabiko.hw import SessionLog, hold, play_song
    from flute_rl.yamabiko.rig import NOMINAL

    rng = np.random.default_rng(args.seed)
    if args.controller == "gru":
        d = np.load(args.policy)
        ctrl = GRUPolicy(d["theta"][None, :], int(d["hidden"]))
    else:
        ctrl = Observer() if args.controller == "observer" else OpenLoop()
    explore = OUNoise(rng, args.noise, args.noise_tau, DT) if args.noise > 0 else None
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(args.out or f"runs/real/collect_{stamp}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = SessionLog({"kind": "collect", "date": stamp, "args": vars(args), "controller": args.controller,
                      "nominal": NOMINAL})

    link, rig = open_rig(args)
    try:
        hold(rig, log, args.settle, song=-1)                       # fan reaching speed, valve shut
        rpm = link.status.fan_rpm if link.status else -1
        print(f"fan {args.fan:.2f}: {rpm} rpm, room + fan level {np.median(rig.log['level_db'][-100:]):.1f} dBFS")
        for k in range(args.songs):
            level = args.level if args.level >= 0 else sample_level(rng, 0.8)
            target = make_target(rng, level)
            sched = make_schedule([[target]])
            play_song(rig, ctrl, sched, k, log, first=(k == 0), explore=explore)
            hold(rig, log, args.rest, song=k)
            heard = np.asarray(rig.log["heard"][-sched.T:])
            note = np.isfinite(sched.target[0])
            both = note & np.isfinite(heard)
            err = np.abs(heard[both] - sched.target[0][both])
            print(f"song {k}: level {level}, {sched.T * DT:.1f} s, heard on {both.sum()}/{note.sum()} note steps, "
                  f"median |error| {np.median(err) if err.size else float('nan'):.0f} cents, "
                  f"current peak {max(rig.log['isense_peak_mv'][-sched.T:]) / 100.0:.2f} A, late steps {rig.late_steps}")
            log.save(out, rig)
    finally:
        close_rig(link, rig)
        log.save(out, rig)
        print("saved", out)


if __name__ == "__main__":
    main()
