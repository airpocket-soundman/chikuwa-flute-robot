"""Train a residual policy on top of the adaptive controller (rig identification + ILC) with ES.

Each episode plays the same target `--takes` times. Take 1 is played with the
nominal model; from take 2 on the controller uses the rig it identified from
the previous take, so the residual network learns in a nearly known rig
instead of fighting the unobservable actuator-speed spread. The fitness is the
mean reward of takes 2..N (take 1 is also played with the residual, since its
PWM feeds the identification).

Candidates are evaluated in parallel (multiprocessing), numpy only:

    python scripts/train_residual_es.py --gens 150 --pop 96 --episodes 8 --workers 20 --out runs/residual.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl import MLP, FluteEnv, ModelResidualPolicy, rollout  # noqa: E402
from flute_rl.adapt import AdaptivePolicy  # noqa: E402

KEYS = ("mean_reward", "mean_abs_cents", "sounding_rate", "rest_leak", "overblow_rate")
_W: dict = {}


def build(args):
    env = FluteEnv(progress=args.progress, spread=args.spread, takes=args.takes)
    net = MLP(ModelResidualPolicy.feature_dim(args.horizon), 2, hidden=args.hidden, rng=np.random.default_rng(args.seed))
    policy = ModelResidualPolicy(AdaptivePolicy(ilc_gain=args.ilc_gain), net, scale=args.scale, horizon=args.horizon)
    return env, net, policy


def _init(args):
    _W["env"], _W["net"], _W["policy"] = build(args)


def _fitness(task):
    theta, seed = task
    _W["net"].set_flat(theta)
    r = rollout(_W["env"], _W["policy"], seed=int(seed))
    return float(np.mean([t["mean_reward"] for t in r["per_take"][1:]]))


def _metrics(task):
    theta, seed = task
    _W["net"].set_flat(theta)
    r = rollout(_W["env"], _W["policy"], seed=int(seed))
    return [[t[k] for k in KEYS] for t in r["per_take"]]


def evaluate(pool, theta, seeds) -> np.ndarray:
    """(takes, len(KEYS)) metrics averaged over seeds."""
    return np.nanmean(np.array(pool.map(_metrics, [(theta, s) for s in seeds])), axis=0)


def report(label: str, m: np.ndarray) -> None:
    for i, row in enumerate(m):
        d = dict(zip(KEYS, row))
        print(f"{label} take {i + 1}: reward/step {d['mean_reward']:+.3f}  |cents| {d['mean_abs_cents']:6.1f}  "
              f"sounding {d['sounding_rate']:.3f}  rest-leak {d['rest_leak']:.3f}  overblow {d['overblow_rate']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--pop", type=int, default=96, help="population size (even)")
    ap.add_argument("--episodes", type=int, default=8, help="episodes per candidate (common random numbers)")
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--scale", type=float, default=0.3, help="residual action scale")
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--ilc-gain", type=float, default=0.5)
    ap.add_argument("--progress", type=float, default=0.8)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default=None, help="start from a saved network")
    ap.add_argument("--out", type=str, default="runs/residual.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    _, net, _ = build(args)
    if args.init:
        net.set_flat(MLP.load(args.init)[0].get_flat())
    theta = net.get_flat()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    eval_seeds = np.arange(1_000_000, 1_000_000 + args.eval_episodes)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"params: {net.n_params}  features: {net.in_dim}  takes: {args.takes}  workers: {args.workers}", flush=True)

    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        base = evaluate(pool, theta, eval_seeds)
        report("start  ", base)
        best = (base[1:, 0].mean(), theta.copy())
        half = args.pop // 2
        for g in range(1, args.gens + 1):
            t0 = time.time()
            seeds = rng.integers(0, 2**31 - 1, size=args.episodes)
            eps = rng.standard_normal((half, theta.size))
            cands = [theta + s * args.sigma * e for e in eps for s in (1.0, -1.0)]
            f = np.array(pool.map(_fitness, [(c, sd) for c in cands for sd in seeds]))
            scores = f.reshape(half, 2, args.episodes).mean(axis=2)
            ranks = scores.ravel().argsort().argsort().reshape(scores.shape) / (scores.size - 1) - 0.5
            grad = ((ranks[:, 0] - ranks[:, 1])[:, None] * eps).sum(axis=0) / (half * args.sigma)
            grad -= 0.005 * theta / args.sigma  # weight decay
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            theta = theta + args.lr * (m / (1 - 0.9**g)) / (np.sqrt(v / (1 - 0.999**g)) + 1e-8)
            print(f"gen {g:3d}  pop mean {scores.mean():+.4f}  best {scores.max():+.4f}  ({time.time() - t0:.1f}s)", flush=True)
            if g % args.eval_every == 0 or g == args.gens:
                cur = evaluate(pool, theta, eval_seeds)
                report(f"gen {g:3d}", cur)
                if cur[1:, 0].mean() > best[0]:
                    best = (cur[1:, 0].mean(), theta.copy())
                    net.set_flat(theta)
                    net.save(out, scale=args.scale, horizon=args.horizon, ilc_gain=args.ilc_gain)
                    print(f"saved {out} (best so far)", flush=True)
    print(f"best eval reward/step (takes 2..): {best[0]:+.4f}")


if __name__ == "__main__":
    main()
