"""ES for the live-feedback policy on the GPU (batched environment).

Same method as `train_residual_es.py --feedback`, but a whole generation
(candidates x episodes) is simulated at once with flute_rl.torch_env. Rigs,
targets and the sounding-compensation angle are shared by all candidates of a
generation; the measurement noise is drawn separately per rig. Networks are
saved in the same format as the CPU script, so compare.py and the audio
export read them unchanged.

    python scripts/train_es_gpu.py --arch gru --hidden 16 --gens 150 --out runs/gpu_gru.npz --keep-snapshots
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.feedback import FeedbackResidualPolicy  # noqa: E402
from flute_rl.policy import GRU, MLP, load_net  # noqa: E402
from flute_rl.targets import load_bank, make_bank, make_target, sample_level, save_bank  # noqa: E402
from flute_rl.torch_env import BatchAgent, BatchFluteEnv, BatchGRU, BatchMLP, rig_from_seed, run_episodes  # noqa: E402

KEYS = ("|cents|", "reward/step", "sounding", "gap leak")


def play(thetas: torch.Tensor, rigs, targets, args, gen: torch.Generator) -> torch.Tensor:
    """thetas (B, P), one row per episode -> metrics (B, takes, 4)."""
    env = BatchFluteEnv(rigs, targets, takes=args.takes, device=args.device, generator=gen)
    in_dim = FeedbackResidualPolicy.feature_dim(args.horizon, args.history)
    net = (BatchGRU if args.arch == "gru" else BatchMLP)(thetas, in_dim, 3, args.hidden)
    agent = BatchAgent(env, fb_gain=args.fb_gain, ilc_gain=args.ilc_gain, net=net, scale=args.scale,
                       horizon=args.horizon, history=args.history)
    return run_episodes(env, agent)


def report(label: str, m: np.ndarray) -> None:
    for k, row in enumerate(m):
        print(f"{label} take {k + 1}: " + "  ".join(f"{n} {v:.3f}" if n != "|cents|" else f"{n} {v:6.1f}" for n, v in zip(KEYS, row)),
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("mlp", "gru"), default="mlp")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--history", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--fb-gain", type=float, default=0.05)
    ap.add_argument("--ilc-gain", type=float, default=0.5)
    ap.add_argument("--progress", type=float, default=0.8)
    ap.add_argument("--harsh", type=float, default=0.0,
                    help="0..1: rig effects the controllers do not model (training and evaluation rigs)")
    ap.add_argument("--bank", type=int, default=1000)
    ap.add_argument("--bank-file", default="runs/target_bank.npz")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--keep-snapshots", action="store_true")
    ap.add_argument("--init", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/gpu_es.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    in_dim = FeedbackResidualPolicy.feature_dim(args.horizon, args.history)
    net = (GRU if args.arch == "gru" else MLP)(in_dim, 3, hidden=args.hidden, rng=np.random.default_rng(args.seed))
    if args.init:
        net.set_flat(load_net(args.init)[0].get_flat())
    theta = net.get_flat()
    m, v = np.zeros_like(theta), np.zeros_like(theta)

    bank_path = pathlib.Path(args.bank_file)
    if not bank_path.exists() or len(load_bank(bank_path)) != args.bank:
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        save_bank(bank_path, make_bank(args.bank, progress=args.progress))
    bank = load_bank(bank_path)

    # evaluation: fixed rigs and targets generated like FluteEnv(seed=s) would, kept for the whole run
    eval_seeds = range(1_000_000, 1_000_000 + args.eval_episodes)
    eval_rigs = [rig_from_seed(s, harsh=args.harsh) for s in eval_seeds]
    eval_targets = []
    for s in eval_seeds:
        r = np.random.default_rng(s)
        from flute_rl.sim import FluteParams, FluteSim  # noqa: E402  (replay the env's rng use)
        FluteSim(FluteParams.sample(r, harsh=args.harsh), r)
        eval_targets.append(make_target(r, sample_level(r, args.progress)))

    def evaluate(th: np.ndarray) -> np.ndarray:
        rows = torch.tensor(np.tile(th, (args.eval_episodes, 1)), device=args.device, dtype=torch.float32)
        met = play(rows, eval_rigs, eval_targets, args, gen).cpu().numpy()
        return np.nanmean(met, axis=0)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(scale=args.scale, horizon=args.horizon, ilc_gain=args.ilc_gain, feedback=True,
                fb_gain=args.fb_gain, history=args.history)
    print(f"params: {net.n_params}  arch: {args.arch}  pop {args.pop} x episodes {args.episodes} on {args.device}  "
          f"harsh {args.harsh}", flush=True)
    t0 = time.time()
    cur = evaluate(theta)
    report("start  ", cur)
    print(f"(evaluation {time.time() - t0:.1f}s)", flush=True)
    best = (cur[:, 1].mean(), theta.copy())
    half = args.pop // 2
    for g in range(1, args.gens + 1):
        t0 = time.time()
        seeds = rng.integers(0, 2**31 - 1, size=args.episodes)
        idxs = rng.integers(0, args.bank, size=args.episodes)
        rigs = [rig_from_seed(int(s), harsh=args.harsh) for s in seeds]
        targets = [bank[int(i)] for i in idxs]
        eps = rng.standard_normal((half, theta.size))
        cands = np.stack([theta + s * args.sigma * e for e in eps for s in (1.0, -1.0)])  # (pop, P)
        rows = torch.tensor(np.repeat(cands, args.episodes, axis=0), device=args.device, dtype=torch.float32)
        met = play(rows, rigs * args.pop, targets * args.pop, args, gen)
        f = met[:, :, 1].mean(1).reshape(args.pop, args.episodes).mean(1).cpu().numpy()
        scores = f.reshape(half, 2)
        ranks = scores.ravel().argsort().argsort().reshape(scores.shape) / (scores.size - 1) - 0.5
        grad = ((ranks[:, 0] - ranks[:, 1])[:, None] * eps).sum(axis=0) / (half * args.sigma)
        grad -= 0.005 * theta / args.sigma
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        theta = theta + args.lr * (m / (1 - 0.9**g)) / (np.sqrt(v / (1 - 0.999**g)) + 1e-8)
        print(f"gen {g:3d}  pop mean {scores.mean():+.4f}  best {scores.max():+.4f}  ({time.time() - t0:.1f}s)", flush=True)
        if g % args.eval_every == 0 or g == args.gens:
            cur = evaluate(theta)
            report(f"gen {g:3d}", cur)
            net.set_flat(theta)
            if args.keep_snapshots:
                net.save(out.with_name(f"{out.stem}_gen{g:03d}.npz"), **meta)
            if cur[:, 1].mean() > best[0]:
                best = (cur[:, 1].mean(), theta.copy())
                net.save(out, **meta)
                print(f"saved {out} (best so far)", flush=True)
    print(f"best eval reward/step: {best[0]:+.4f}")


if __name__ == "__main__":
    main()
