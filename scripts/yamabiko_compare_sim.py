"""PC or UNO Q: play fixed phrases with any steps (models/<step>/, the classic PID one included) on simulated
rigs drawn around a twin, and compare them.  NumPy only (no PyTorch, no rig).

    python scripts/yamabiko_compare_sim.py --steps classic step1 step2 step3 step4
    python scripts/yamabiko_compare_sim.py --steps classic step4 --targets f80_doremi --rigs 16 --plot runs/compare

Each step runs exactly as the server runs it (the actuator runtime, the valve network if the profile has one,
otherwise the runtime's own valve).  The ear is not simulated: every step hears the pitch the flute emits, a
few steps late, with noise, drop-outs and motor masking (flute_rl/yamabiko/sim_numpy.py).  So this compares the
controllers, not the ears; the ears are compared on the rig (scripts/yamabiko_step_eval.py).

Per step and phrase, the mean over ``--rigs`` rigs:
  hit    share of note time within 25 cent of the score, the first 0.3 s of every note left out
  mae    mean |pitch - score| over the notes (pitch the flute emits, where the score has a note)
  glide  share of the time the flute sounds that is more than 50 cent off the score
  cover  share of the score's note time the flute sounds
  flips  PWM sign changes (|PWM| > 0.05) per play
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flute_rl.yamabiko.profiles import load_profiles  # noqa: E402
from flute_rl.yamabiko.residual_numpy import WithValve, runtime_for  # noqa: E402
from flute_rl.yamabiko.sim_numpy import play, rig_arrays, rigs_around  # noqa: E402

ONSET = 30
DEFAULT_TWIN = ROOT / "models/step4/twin_y1f80_04_runtime.npz"


def target_file(name):
    for path in sorted(ROOT.glob(f"models/*/target_{name}.npz")) + [ROOT / f"runs/real/target_{name}.npz"]:
        if path.exists():
            return path
    raise FileNotFoundError(name)


def load_target(name):
    d = np.load(target_file(name))
    return d["cents"].astype(np.float64), d["voice"].astype(bool)


def settled(cents, voice):
    keep = np.zeros_like(voice)
    since = 0
    for t in range(len(cents)):
        new = voice[t] and (t == 0 or not voice[t - 1] or abs(cents[t] - cents[t - 1]) > 1)
        since = 0 if new else since + 1
        keep[t] = voice[t] and since >= ONSET
    return keep


def take_metrics(out, cents, voice):
    pitch, sounding, pwm = out["emitted"], out["sounding"], out["pwm"]
    err = np.abs(pitch - cents)
    keep = settled(cents, voice)
    s = np.sign(pwm[np.abs(pwm) > .05])
    return {"hit": float(((err <= 25) & sounding)[keep].mean()),
            "mae": float(err[voice].mean()),
            "glide": float((err[sounding] > 50).mean()) if sounding.any() else 0.0,
            "cover": float((sounding & voice).sum() / max(voice.sum(), 1)),
            "flips": int((s[1:] != s[:-1]).sum())}


def runtime_maker(profile):
    actuator = dict(np.load(profile["runtime_path"]))
    valve = dict(np.load(profile["valve_runtime_path"])) if profile.get("valve_runtime_path") else None

    def make():
        runtime = runtime_for(actuator)
        return WithValve(runtime, valve) if valve is not None else runtime
    return make


def evaluate(make, rigs, cents, voice, repeats=1):
    """Mean metrics over the rigs; with repeats > 1, every rig plays the phrase again (song memory kept)
    and the metrics are of the last play."""
    rows, takes = [], []
    for rig in rigs:
        runtime = make()
        for k in range(repeats):
            out = play(runtime, rig, cents, voice, keep_song_memory=k > 0)
        rows.append(take_metrics(out, cents, voice)); takes.append(out)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, takes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=None, help="profile names (default: all in models/)")
    ap.add_argument("--targets", nargs="+", default=["f80_doremi", "f80_scale", "f80_leaps"])
    ap.add_argument("--twin", default=str(DEFAULT_TWIN), help="runtime .npz whose twin the rigs are drawn around")
    ap.add_argument("--rigs", type=int, default=32)
    ap.add_argument("--spread", type=float, default=.12)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--repeats", type=int, default=1, help="plays per rig; the last one is scored")
    ap.add_argument("--plot", help="folder for one figure per phrase (the first rig of every step)")
    ap.add_argument("--json", help="write the table here")
    args = ap.parse_args()
    profiles = load_profiles()
    steps = args.steps or list(profiles)
    constants, twin = rig_arrays(args.twin)
    print(f"twin {pathlib.Path(args.twin).name}, {args.rigs} rigs, spread {args.spread}")
    print(f"{'step':8s} {'target':14s} {'hit':>6s} {'mae':>6s} {'glide':>6s} {'cover':>6s} {'flips':>6s}")
    table, examples = {}, {}
    for name in steps:
        make = runtime_maker(profiles[name])
        for target in args.targets:
            cents, voice = load_target(target)
            rigs = rigs_around(constants, twin, args.rigs, args.spread, args.seed)
            m, takes = evaluate(make, rigs, cents, voice, args.repeats)
            table.setdefault(name, {})[target] = m
            examples.setdefault(target, {})[name] = takes[0]
            print(f"{name:8s} {target:14s} {m['hit']:6.1%} {m['mae']:6.1f} {m['glide']:6.1%} {m['cover']:6.1%} {m['flips']:6.1f}",
                  flush=True)
    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(table, indent=2), encoding="utf-8")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        folder = pathlib.Path(args.plot); folder.mkdir(parents=True, exist_ok=True)
        for target, per in examples.items():
            cents, voice = load_target(target)
            fig, axes = plt.subplots(len(per), 1, figsize=(13, 2.6 * len(per)), squeeze=False, sharex=True)
            tt = np.arange(len(cents)) * .01
            for ax, (name, out) in zip(axes[:, 0], per.items()):
                ax.plot(tt, np.where(voice, cents, np.nan), "k", lw=2, label="score")
                ax.plot(tt, np.where(out["sounding"], out["emitted"], np.nan), lw=1.2, label="flute")
                ax.fill_between(tt, -150, -150 + 80 * out["valve"], color="tab:green", alpha=.25, step="mid")
                a2 = ax.twinx(); a2.plot(tt, out["pwm"], "r", lw=.5, alpha=.5); a2.set_ylim(-1.1, 1.1)
                m = table[name][target]
                ax.set_title(f"{name}  hit {m['hit']:.0%}  mae {m['mae']:.0f} cent  flips {m['flips']:.0f}", fontsize=9)
                ax.set_ylim(-160, 1350); ax.grid(alpha=.3)
            axes[-1, 0].set_xlabel("s")
            plt.tight_layout(); plt.savefig(folder / f"{target}.png", dpi=80); plt.close()
        print("figures:", folder)


if __name__ == "__main__":
    main()
