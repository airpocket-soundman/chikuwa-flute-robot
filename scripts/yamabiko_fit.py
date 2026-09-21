"""Fit the simulator's rig to logs of the real Yamabiko No.1 (on the PC).

The logs of scripts/yamabiko_collect.py (and yamabiko_play.py) hold, for every 10 ms step, the PWM and
valve command that was sent and the pitch that was heard. There is no position sensor, so the rig is
fitted the only way the network itself could: replay the same commands through the simulator (Rig, noise
switched off) and pick the rig properties whose heard pitch matches the real one best. The homing before
every song (1 s pulling out against the end stop) gives the known starting point; steps before the first
homing are not scored.

The search is a cross-entropy method over the properties below, all candidates replayed at once. What
is left over (the spread of the pitch around the fitted rig, frames missed, octave slips) becomes the
listening noise of the rig. The result is a nominal rig to train on:

    python scripts/yamabiko_fit.py runs/real/collect_*.npz --out runs/real/rig_fit.json
    YAMABIKO_RIG=runs/real/rig_fit.json python scripts/yamabiko_train.py --device cuda ... --out runs/yamabiko_gru_real.npz
    python3 scripts/yamabiko_play.py --rig runs/real/rig_fit.json --policy runs/yamabiko_gru_real.npz   # UNO Q

--selftest makes logs with the simulator from a random rig first and fits those, to see which properties
the data pins down.

Temperature and the tube length trade off against each other (only c / 4L is heard): the temperature is
fixed (--temp) and the length absorbs the rest.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.sim import DT  # noqa: E402
from flute_rl.targets import make_target, sample_level  # noqa: E402
from flute_rl.yamabiko import Observer, Rig, RigParams, make_schedule  # noqa: E402
from flute_rl.yamabiko.rig import INTS, NOMINAL, QUEUE  # noqa: E402

# (property, low, high): the search box
FIT = [
    ("tube_len", 0.110, 0.190), ("press_cents", -60.0, 60.0), ("overblow_len", 0.020, 0.080),
    ("v_in", 0.040, 0.300), ("v_out", 0.040, 0.300), ("deadband", 0.0, 0.5), ("tau_v", 0.005, 0.100),
    ("backlash", 0.0, 0.002), ("onset_s", 0.0, 0.080), ("onset_cents", -60.0, 30.0), ("onset_tau", 0.005, 0.080),
    ("surge_cents", -10.0, 40.0), ("surge_tau", 0.020, 0.300),
    ("cmd_delay", 0, 4), ("obs_delay", 0, 6), ("valve_delay", 0, 4),
]
QUIET = dict(pitch_noise=0.0, dropout=0.0, octave_err=0.0, pitch_jitter=0.0, delay_jitter=0.0, motion_noise=0.0,
             speed_drift=0.0, stiction=0.0, load_slope=0.0, pwm_curve=1.0)
CAP = 300.0       # cents: a larger miss counts as this much
MISMATCH = 1.0    # cost of a step heard in one and silent in the other, in units of 100 cents


@dataclasses.dataclass
class Log:
    name: str
    pwm: np.ndarray
    valve: np.ndarray
    heard: np.ndarray
    scored: np.ndarray    # steps after the first homing
    isense: np.ndarray | None = None   # motor current sense [mV], mean over each step


def load_log(path) -> Log:
    d = np.load(path)
    homing = d["homing"].astype(bool)
    done = np.flatnonzero(homing[:-1] & ~homing[1:])
    scored = np.zeros(len(homing), bool)
    if done.size:
        scored[done[0] + 1:] = True
    scored &= ~homing
    isense = d["isense_mv"].astype(float) if "isense_mv" in d.files else None
    return Log(str(path), d["pwm"].astype(float), d["valve"].astype(bool), d["heard"].astype(float), scored, isense)


def validation_split(logs: list[Log], fraction: float) -> tuple[list[Log], list[Log]]:
    """Split score masks chronologically while replaying every command in both sets.

    The simulator state still evolves through the complete recording; only
    which frames contribute to calibration versus validation changes.  A
    held-out tail catches wrong velocity, drift and transient assumptions that
    an in-sample fit can otherwise hide by adjusting tube length or pressure.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if fraction == 0.0:
        return logs, []
    train, validation = [], []
    for log in logs:
        idx = np.flatnonzero(log.scored)
        count = min(max(1, int(np.ceil(len(idx) * fraction))), max(0, len(idx) - 1))
        val_mask = np.zeros_like(log.scored)
        if count:
            val_mask[idx[-count:]] = True
        train_mask = log.scored & ~val_mask
        train.append(dataclasses.replace(log, scored=train_mask))
        validation.append(dataclasses.replace(log, scored=val_mask))
    return train, validation


def current_delay(logs: list[Log], max_lag: int = 4) -> tuple[int, float] | None:
    """Steps from a PWM command to the motor current, from the current sense: the one delay that the pitch
    cannot tell apart from the listening delay (only their sum is heard)."""
    a, b = [], []
    for log in logs:
        if log.isense is None or np.all(log.isense <= 0):
            continue
        a.append(np.abs(log.pwm))
        b.append(log.isense)
    if not a:
        return None
    u, i = np.concatenate(a), np.concatenate(b)
    if i.std() < 1.0:          # no current seen (nothing connected)
        return None
    cors = [np.corrcoef(u[:len(u) - k], i[k:])[0, 1] for k in range(max_lag + 1)]
    k = int(np.nanargmax(cors))
    return k, float(cors[k])


def params_from(x: np.ndarray, temp_c: float, stroke: float) -> RigParams:
    """(P, len(FIT)) values -> P quiet rigs."""
    p = RigParams.nominal(len(x), temp_c=temp_c, stroke=stroke, **QUIET)
    for j, (name, _, _) in enumerate(FIT):
        v = x[:, j]
        setattr(p, name, np.clip(np.round(v), 0, QUEUE - 1).astype(int) if name in INTS else v.astype(float))
    return p


def replay(p: RigParams, log: Log) -> np.ndarray:
    """(P, T) pitch the rigs would have been heard at, for the logged commands."""
    rig = Rig(p, np.random.default_rng(0))
    P, T = p.n, len(log.pwm)
    out = np.empty((P, T))
    for t in range(T):
        o = rig.step(np.full(P, log.pwm[t]), np.full(P, log.valve[t]))
        out[:, t] = o["heard"]
    return out


def score(sim: np.ndarray, log: Log) -> tuple[np.ndarray, int]:
    real = log.heard[None, log.scored]
    s = sim[:, log.scored]
    a, b = np.isfinite(real), np.isfinite(s)
    miss = np.where(a & b, np.minimum(np.abs(np.nan_to_num(real - s)), CAP) / 100.0, 0.0)
    return (miss.sum(axis=1) + MISMATCH * (a ^ b).sum(axis=1)), int(log.scored.sum())


def loss(x: np.ndarray, logs: list[Log], temp_c: float, stroke: float) -> np.ndarray:
    total, n = np.zeros(len(x)), 0
    p = params_from(x, temp_c, stroke)
    for log in logs:
        s, m = score(replay(p, log), log)
        total += s
        n += m
    return total / max(n, 1)


def fix(name: str, value: float) -> None:
    """Hold one property at a known value (its search box shrinks to that value)."""
    i = [f[0] for f in FIT].index(name)
    FIT[i] = (name, value, value)


def cem(logs, temp_c, stroke, x0, sd0, gens, pop, elite, rng, best, label):
    lo = np.array([f[1] for f in FIT], float)
    hi = np.array([f[2] for f in FIT], float)
    span = np.where(hi > lo, hi - lo, 1.0)
    mean, sd = (x0 - lo) / span, np.full(len(FIT), sd0)
    for g in range(gens):
        t0 = time.time()
        z = np.clip(mean + sd * rng.standard_normal((pop, len(FIT))), 0.0, 1.0)
        z[0] = mean
        x = lo + z * span * (hi > lo)
        L = loss(x, logs, temp_c, stroke)
        order = np.argsort(L)
        if L[order[0]] < best[0]:
            best = (float(L[order[0]]), x[order[0]].copy())
        e = z[order[:elite]]
        mean = 0.3 * mean + 0.7 * e.mean(axis=0)
        sd = np.maximum(0.3 * sd + 0.7 * e.std(axis=0), 0.02)
        print(f"{label} {g:3d}  best {L[order[0]]:.3f}  mean of elite {L[order[:elite]].mean():.3f}  "
              f"({time.time() - t0:.1f} s)", flush=True)
    return best


DELAYS = ("cmd_delay", "obs_delay", "valve_delay")


def delay_grid(logs, temp_c, stroke, x):
    """Every combination of the three delays, the rest held at x: (losses, candidates) best first."""
    idx = [[f[0] for f in FIT].index(n) for n in DELAYS]
    rng_of = [range(int(FIT[j][1]), int(FIT[j][2]) + 1) for j in idx]
    combos = np.array([(c, o, v) for c in rng_of[0] for o in rng_of[1] for v in rng_of[2]], float)
    X = np.tile(x, (len(combos), 1))
    X[:, idx] = combos
    L = loss(X, logs, temp_c, stroke)
    order = np.argsort(L)
    return L[order], X[order]


def fit(logs: list[Log], temp_c: float, stroke: float, gens: int, pop: int, elite: int, seed: int, top: int = 3):
    """Wide search from the nominal rig; then the delays on a grid, and a narrow search of the rest from
    each of the `top` best delay combinations with the delays held (the search alone trades a step of
    delay against tau_v / onset_s and stays there).

    Even so, the listening delay and the valve delay can come out a step off each other: the sound starts
    valve_delay + onset_s + obs_delay after the valve command, and a step of one is nearly the same as a
    step of the other plus a change of onset_s and tau_v. Train with the default spread (+-2 steps)."""
    rng = np.random.default_rng(seed)
    x0 = np.array([np.clip(NOMINAL[f[0]], f[1], f[2]) for f in FIT], float)
    nominal_loss = float(loss(x0[None, :], logs, temp_c, stroke)[0])
    print(f"nominal rig: loss {nominal_loss:.3f}")
    best = cem(logs, temp_c, stroke, x0, 0.3, gens, pop, elite, rng, (nominal_loss, x0), "wide")
    L, X = delay_grid(logs, temp_c, stroke, best[1])
    saved = list(FIT)
    idx = [[f[0] for f in FIT].index(n) for n in DELAYS]
    for i in range(min(top, len(L))):
        for j in idx:
            fix(FIT[j][0], X[i, j])
        tag = "delays " + " ".join(f"{X[i, j]:.0f}" for j in idx)
        cand = cem(logs, temp_c, stroke, X[i], 0.05, max(10, gens // 3), pop, elite, rng, (float(L[i]), X[i].copy()),
                   f"narrow ({tag})")
        FIT[:] = saved
        print(f"{tag}: loss {L[i]:.3f} -> {cand[0]:.3f}")
        if cand[0] < best[0]:
            best = cand
    return best, nominal_loss


def leftover_noise(x: np.ndarray, logs: list[Log], temp_c: float, stroke: float) -> dict:
    """What the fitted rig does not explain, as the listening noise of the simulator."""
    p = params_from(x[None, :], temp_c, stroke)
    d_all, sim_on, real_off = [], 0, 0
    for log in logs:
        s = replay(p, log)[0, log.scored]
        r = log.heard[log.scored]
        both = np.isfinite(s) & np.isfinite(r)
        d_all.append(r[both] - s[both])
        sim_on += int(np.isfinite(s).sum())
        real_off += int((np.isfinite(s) & ~np.isfinite(r)).sum())
    d = np.concatenate(d_all) if d_all else np.zeros(0)
    near = d[np.abs(d) < 100.0]
    mad = float(np.median(np.abs(near - np.median(near)))) if near.size else 0.0
    return {"pitch_noise": 1.4826 * mad,
            "dropout": min(0.2, real_off / max(sim_on, 1)),
            "octave_err": float(np.mean(np.abs(np.abs(d) - 1200.0) < 100.0)) if d.size else 0.0,
            "_median_miss_cents": float(np.median(np.abs(d))) if d.size else float("nan"),
            "_frames_compared": int(d.size)}


def synthetic_logs(seed: int, songs: int, noise: float) -> tuple[list[Log], RigParams]:
    """Logs made the way yamabiko_collect.py makes them, from a random simulated rig."""
    rng = np.random.default_rng(seed)
    true = RigParams.sample(rng, 1)
    rig = Rig(true, rng)
    ctrl = Observer()
    pwm_l, valve_l, heard_l, home_l = [], [], [], []

    def step(pwm, valve, home):
        out = rig.step(np.full(1, pwm), np.full(1, valve))
        pwm_l.append(pwm); valve_l.append(valve); heard_l.append(out["heard"][0]); home_l.append(home)
        return out

    for _ in range(100):                 # fan settling, plunger somewhere
        step(0.0, False, False)
    e, a = 0.0, DT / 0.15
    for k in range(songs):
        sched = make_schedule([[make_target(rng, sample_level(rng, 0.8))]])
        if k == 0:
            ctrl.begin(sched, rig)
        else:
            ctrl.sched = sched
        ctrl.song_start(k)
        for t in range(sched.T):
            home = bool(sched.homing[t])
            if home:
                pwm, valve = -1.0, False
            else:
                e += -e * a + noise * (2.0 * a) ** 0.5 * rng.standard_normal()
                pwm, valve = float(np.clip(ctrl.act(t)[0] + e, -1.0, 1.0)), bool(sched.valve[0, t])
            out = step(pwm, valve, home)
            ctrl.update(t, np.full(1, pwm), out)
            if home and not sched.homing[min(t + 1, sched.T - 1)]:
                ctrl.homed()
        for _ in range(50):
            step(0.0, False, False)
    homing = np.array(home_l)
    done = np.flatnonzero(homing[:-1] & ~homing[1:])
    scored = np.zeros(len(homing), bool)
    scored[done[0] + 1:] = True
    scored &= ~homing
    return [Log("synthetic", np.array(pwm_l), np.array(valve_l), np.array(heard_l), scored)], true


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--out", default="runs/real/rig_fit.json")
    ap.add_argument("--temp", type=float, default=NOMINAL["temp_c"], help="room temperature during the logs [C]")
    ap.add_argument("--stroke", type=float, default=NOMINAL["stroke"], help="plunger travel in the flute [m]")
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elite", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE",
                    help="hold a property, e.g. --fix cmd_delay=1 (repeatable)")
    ap.add_argument("--selftest", action="store_true", help="fit logs made with the simulator instead")
    ap.add_argument("--selftest-songs", type=int, default=10)
    ap.add_argument("--validation-fraction", type=float, default=0.2,
                    help="chronological tail excluded from fitting and used for predictive validation")
    args = ap.parse_args()

    if args.selftest:
        logs, true = synthetic_logs(args.seed + 1, args.selftest_songs, 0.15)
        temp_c, stroke = float(true.temp_c[0]), float(true.stroke[0])
    else:
        if not args.logs:
            ap.error("give the logs of yamabiko_collect.py / yamabiko_play.py, or --selftest")
        logs, true = [load_log(p) for p in args.logs], None
        temp_c, stroke = args.temp, args.stroke
    try:
        train_logs, validation_logs = validation_split(logs, args.validation_fraction)
    except ValueError as exc:
        ap.error(str(exc))
    fixed = dict(f.split("=", 1) for f in args.fix)
    if args.selftest and "cmd_delay" not in fixed:     # stands in for the current sense of the real rig
        fixed["cmd_delay"] = str(int(true.cmd_delay[0]))
    if not args.selftest and "cmd_delay" not in fixed:
        est = current_delay(logs)
        if est is not None and est[1] > 0.3:
            fixed["cmd_delay"] = str(est[0])
            print(f"command delay from the motor current: {est[0]} steps (correlation {est[1]:.2f})")
        else:
            print("no motor current in the logs: the command delay is searched too, and only its sum with "
                  "the listening delay is pinned down")
    for name, v in fixed.items():
        fix(name, float(v))
        print(f"fixed {name} = {v}")
    n = sum(int(l.scored.sum()) for l in train_logs)
    n_validation = sum(int(l.scored.sum()) for l in validation_logs)
    heard = sum(int(np.isfinite(l.heard[l.scored]).sum()) for l in train_logs)
    print(f"{len(logs)} log(s), {n} fit steps ({n * DT:.0f} s), {n_validation} held-out steps, "
          f"pitch heard on {heard} fit steps")

    (best_loss, x), nominal_loss = fit(train_logs, temp_c, stroke, args.gens, args.pop, args.elite, args.seed)
    noise = leftover_noise(x, train_logs, temp_c, stroke)
    validation_loss = (float(loss(x[None, :], validation_logs, temp_c, stroke)[0])
                       if n_validation else float("nan"))
    nominal_validation_loss = (float(loss(np.array([[NOMINAL[f[0]] for f in FIT]]), validation_logs,
                                                   temp_c, stroke)[0])
                               if n_validation else float("nan"))
    fitted = {name: (int(round(v)) if name in INTS else float(v)) for (name, _, _), v in zip(FIT, x)}
    print(f"\nloss: nominal {nominal_loss:.3f} -> fitted {best_loss:.3f}   "
          f"(median miss {noise['_median_miss_cents']:.1f} cents over {noise['_frames_compared']} frames)")
    if n_validation:
        print(f"held-out loss: nominal {nominal_validation_loss:.3f} -> fitted {validation_loss:.3f}   "
              f"(generalization gap {validation_loss - best_loss:+.3f})")
    print(f"{'property':14s} {'nominal':>10s} {'fitted':>10s}" + (f" {'true':>10s}" if true is not None else ""))
    for name, v in fitted.items():
        line = f"{name:14s} {NOMINAL[name]:10.4g} {v:10.4g}"
        if true is not None:
            line += f" {float(getattr(true, name)[0]):10.4g}"
        print(line)
    for name in ("pitch_noise", "dropout", "octave_err"):
        line = f"{name:14s} {NOMINAL[name]:10.4g} {noise[name]:10.4g}"
        if true is not None:
            line += f" {float(getattr(true, name)[0]):10.4g}"
        print(line)

    if args.selftest:
        return
    nominal = {**fitted, "temp_c": temp_c, "stroke": stroke,
               **{k: v for k, v in noise.items() if not k.startswith("_")}}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"nominal": nominal, "loss": best_loss, "loss_nominal": nominal_loss,
                               "validation_loss": validation_loss,
                               "validation_loss_nominal": nominal_validation_loss,
                               "validation_fraction": args.validation_fraction,
                               "median_miss_cents": noise["_median_miss_cents"], "steps": n,
                               "logs": [l.name for l in logs]}, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
