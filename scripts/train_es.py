"""Train a residual policy on top of the physics prior with Evolution Strategies.

numpy only, so the same script runs on a PC or on the Linux side of the UNO Q.

    python scripts/train_es.py --takes 3 --gens 30 --pop 16 --episodes 4 --out runs/es_residual.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl import MLP, FluteEnv, ILCPolicy, PhysicsPriorPolicy, ResidualPolicy, rollout  # noqa: E402

PRIORS = {"physics": PhysicsPriorPolicy, "ilc": ILCPolicy}

KEYS = ("mean_reward", "mean_abs_cents", "sounding_rate", "rest_leak", "overblow_rate")


def evaluate(env, policy, seeds) -> list[dict]:
    """Per-take metrics averaged over `seeds`."""
    runs = [rollout(env, policy, seed=int(s)) for s in seeds]
    return [{k: float(np.nanmean([r["per_take"][i][k] for r in runs])) for k in KEYS} for i in range(env.takes)]


def report(label: str, per_take: list[dict]) -> None:
    for i, m in enumerate(per_take):
        print(f"{label} take {i + 1}: reward/step {m['mean_reward']:+.3f}  |cents| {m['mean_abs_cents']:6.1f}  "
              f"sounding {m['sounding_rate']:.2f}  rest-leak {m['rest_leak']:.2f}  overblow {m['overblow_rate']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--pop", type=int, default=16, help="population size (even)")
    ap.add_argument("--episodes", type=int, default=4, help="episodes per candidate")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--scale", type=float, default=0.3, help="residual action scale")
    ap.add_argument("--takes", type=int, default=3, help="takes of the same target per episode")
    ap.add_argument("--prior", choices=sorted(PRIORS), default="physics", help="hand-written policy the residual is added to")
    ap.add_argument("--feedback", action="store_true", help="add delayed live pitch to the observation")
    ap.add_argument("--progress", type=float, default=0.8, help="curriculum progress (0..1)")
    ap.add_argument("--spread", type=float, default=1.0, help="domain-randomisation spread")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/es_residual.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = FluteEnv(progress=args.progress, spread=args.spread, takes=args.takes, feedback=args.feedback)
    net = MLP(env.obs_dim + 2, 2, hidden=args.hidden, rng=rng)
    policy = ResidualPolicy(PRIORS[args.prior](), net, scale=args.scale)
    eval_seeds = np.arange(1_000_000, 1_000_000 + args.eval_episodes)

    theta = net.get_flat()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    print(f"params: {net.n_params}  obs_dim: {env.obs_dim}  takes: {env.takes}  feedback: {env.feedback}  prior: {args.prior}")
    report(f"{args.prior} prior".ljust(13), evaluate(env, policy, eval_seeds))

    half = args.pop // 2
    for g in range(1, args.gens + 1):
        t0 = time.time()
        seeds = rng.integers(0, 2**31 - 1, size=args.episodes)  # common random numbers
        eps = rng.standard_normal((half, theta.size))
        scores = np.zeros((half, 2))
        for i in range(half):
            for j, sign in enumerate((1.0, -1.0)):
                net.set_flat(theta + sign * args.sigma * eps[i])
                scores[i, j] = np.mean([rollout(env, policy, seed=int(s))["mean_reward"] for s in seeds])
        ranks = scores.ravel().argsort().argsort().reshape(scores.shape) / (scores.size - 1) - 0.5
        grad = ((ranks[:, 0] - ranks[:, 1])[:, None] * eps).sum(axis=0) / (half * args.sigma)
        # Adam (gradient ascent)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        theta = theta + args.lr * (m / (1 - 0.9**g)) / (np.sqrt(v / (1 - 0.999**g)) + 1e-8)
        print(f"gen {g:3d}  pop mean {scores.mean():+.3f}  best {scores.max():+.3f}  ({time.time() - t0:.1f}s)", flush=True)

    net.set_flat(theta)
    report("trained      ", evaluate(env, policy, eval_seeds))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(out, scale=args.scale, takes=args.takes, feedback=args.feedback)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
