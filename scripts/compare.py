"""Compare controllers take by take on the same randomised rigs and targets.

    python scripts/compare.py --takes 5 --episodes 100 --residual runs/residual.npz
    python scripts/compare.py --takes 3 --episodes 100 --feedback --residual runs/feedback.npz

Prints the mean |cents| per take with a 95% bootstrap interval, and the
paired difference of every controller against the first one. With --plot, a
per-take curve is written to docs/ (needs matplotlib).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl import MLP, AdaptivePolicy, FluteEnv, ILCPolicy, ModelResidualPolicy, PhysicsPriorPolicy, rollout  # noqa: E402
from flute_rl.feedback import FeedbackPolicy, FeedbackResidualPolicy  # noqa: E402
from flute_rl.policy import load_net  # noqa: E402
from flute_rl.targets import BIRDS, bird_song, make_target, scale_notes  # noqa: E402

_A: dict = {}


def controllers(residuals: list[str], feedback: bool) -> dict:
    out = {
        "physics prior": lambda: PhysicsPriorPolicy(),
        "classic ILC (gain 0.7)": lambda: ILCPolicy(gain=0.7),
        "adaptive (rig ID + ILC)": lambda: AdaptivePolicy(),
    }
    if feedback:
        out["adaptive + feedback (classic)"] = lambda: FeedbackPolicy()
    for path in residuals:
        net, meta = load_net(path)
        kw = dict(scale=float(meta["scale"]), horizon=int(meta["horizon"]))
        if bool(meta.get("feedback", False)):
            fb_gain = float(meta["fb_gain"])
            history = int(meta.get("history", 0))
            label = "feedback + RL " + (f"{net.arch.upper()}" if net.arch == "gru" else f"MLP hist {history}")
            out[label] = lambda net=net, kw=kw, g=fb_gain, ilc=float(meta["ilc_gain"]), hs=history: (
                FeedbackResidualPolicy(FeedbackPolicy(fb_gain=g, ilc_gain=ilc), net, history=hs, **kw))
        else:
            out["adaptive + RL residual"] = lambda net=net, kw=kw, ilc=float(meta["ilc_gain"]): (
                ModelResidualPolicy(AdaptivePolicy(ilc_gain=ilc), net, **kw))
    return out


def _init(args):
    _A["args"] = args
    _A["ctrl"] = controllers(args.residual, args.feedback)


def long_piece(rng):
    """Outside the training range: several pieces in a row (10 s or more) with 2-4 s held notes mixed in."""
    parts = []
    while sum(len(p) for p in parts) * 0.01 < 10.0:
        if rng.random() < 0.4:
            parts.append(np.concatenate([np.full(int(rng.integers(20, 40)), np.nan),
                                         np.full(int(rng.integers(200, 400)), float(rng.choice(scale_notes())))]))
        else:
            parts.append(make_target(rng, int(rng.integers(1, 5))))
    return np.concatenate(parts + [np.full(20, np.nan)])


def bird_piece(rng):
    return bird_song(rng, str(rng.choice(BIRDS)))


TARGET_SETS = {"curriculum": None, "long": long_piece, "birds": bird_piece}


def _job(task):
    name, seed = task
    a = _A["args"]
    env = FluteEnv(progress=a.progress, takes=a.takes, feedback=a.feedback, target_fn=TARGET_SETS[a.targets])
    r = rollout(env, _A["ctrl"][name](), seed=seed)
    return [[t["mean_abs_cents"], t["mean_reward"], t["sounding_rate"], t["gap_leak"]] for t in r["per_take"]]


def boot_ci(x: np.ndarray, rng, n: int = 2000) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--progress", type=float, default=0.8)
    ap.add_argument("--residual", action="append", default=[], help="saved residual network (repeatable)")
    ap.add_argument("--feedback", action="store_true", help="env delivers the delayed live pitch; adds the feedback controllers")
    ap.add_argument("--targets", choices=sorted(TARGET_SETS), default="curriculum",
                    help="curriculum: like training; long: 10 s+ pieces with 2-4 s notes; birds: uguisu / cuckoo / great tit")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=2_000_000, help="first evaluation seed (keep away from training seeds)")
    ap.add_argument("--plot", type=str, default=None, help="write a per-take plot to this PNG path")
    args = ap.parse_args()

    names = list(controllers(args.residual, args.feedback))
    seeds = [args.seed0 + i for i in range(args.episodes)]
    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        res = pool.map(_job, [(n, s) for n in names for s in seeds])
    data = {n: np.array(res[i * len(seeds):(i + 1) * len(seeds)]) for i, n in enumerate(names)}  # (episodes, takes, 3)

    rng = np.random.default_rng(0)
    print(f"{args.episodes} randomised rigs x {args.takes} takes, targets: {args.targets}, curriculum progress {args.progress}")
    print("mean |cents| per take [95% CI]")
    for n in names:
        cells = []
        for k in range(args.takes):
            x = data[n][:, k, 0]
            lo, hi = boot_ci(x, rng)
            cells.append(f"{np.nanmean(x):6.1f} [{lo:5.1f},{hi:5.1f}]")
        print(f"  {n:26s}" + "  ".join(cells))
    ref = names[1] if len(names) > 1 else names[0]
    print(f"paired difference vs '{ref}' (negative = better)")
    for n in names:
        if n == ref:
            continue
        cells = []
        for k in range(args.takes):
            d = data[n][:, k, 0] - data[ref][:, k, 0]
            lo, hi = boot_ci(d, rng)
            cells.append(f"{np.nanmean(d):+6.1f} [{lo:+5.1f},{hi:+5.1f}]")
        print(f"  {n:26s}" + "  ".join(cells))
    print("reward/step per take")
    for n in names:
        print(f"  {n:26s}" + "  ".join(f"{np.nanmean(data[n][:, k, 1]):+.3f}" for k in range(args.takes)))
    print("sounding rate during notes per take (lower = it went silent to avoid pitch errors)")
    for n in names:
        print(f"  {n:26s}" + "  ".join(f"{np.nanmean(data[n][:, k, 2]):.3f}" for k in range(args.takes)))
    print("sounding inside the short gaps between detached notes (should be 0)")
    for n in names:
        print(f"  {n:26s}" + "  ".join(f"{np.nanmean(data[n][:, k, 3]):.3f}" for k in range(args.takes)))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        takes = np.arange(1, args.takes + 1)
        for n in names:
            ax.plot(takes, np.nanmean(data[n][:, :, 0], axis=0), marker="o", label=n)
        ax.set_xlabel("take")
        ax.set_ylabel("mean |pitch error| [cents]")
        ax.set_xticks(takes)
        ax.set_yscale("log")
        ticks = [10, 20, 30, 50, 100, 200]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.grid(True, which="major", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
