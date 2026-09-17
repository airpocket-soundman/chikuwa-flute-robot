"""Read the rig out of the recurrent policy's hidden state (see docs/yamabiko.md).

After each song, a ridge regression from the hidden state to a true rig
property is fitted on half of the rigs and scored (R^2) on the other half.
If the policy learns the rig while playing, R^2 should start near 0 at power
on and rise over the first songs.

    python scripts/yamabiko_probe.py --policy runs/yamabiko_gru.npz --rigs 400 --songs 5
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.yamabiko import GRUPolicy, Rig, RigParams, draw_pieces, make_schedule, run_session  # noqa: E402

TARGETS = ("s_in", "s_out", "dL_mm", "obs_delay", "cmd_delay", "deadband", "temp_c", "press_cents")


def ridge_r2(X: np.ndarray, y: np.ndarray, train: np.ndarray, alpha: float) -> float:
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-6
    Z = (X - mu) / sd
    Zt, yt = Z[train], y[train] - y[train].mean()
    w = np.linalg.solve(Zt.T @ Zt + alpha * np.eye(Z.shape[1]), Zt.T @ yt)
    test = ~train
    pred = Z[test] @ w + y[train].mean()
    ss = ((y[test] - y[test].mean()) ** 2).sum()
    return float(1.0 - ((y[test] - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, required=True)
    ap.add_argument("--rigs", type=int, default=400)
    ap.add_argument("--songs", type=int, default=5)
    ap.add_argument("--harsh", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    d = np.load(args.policy)
    rng = np.random.default_rng(args.seed)
    params = RigParams.sample(rng, args.rigs, harsh=args.harsh)
    sched = make_schedule(draw_pieces(rng, args.rigs, args.songs))
    ctrl = GRUPolicy(d["theta"], int(d["hidden"]), carry=True, record=True)
    res = run_session(Rig(params, np.random.default_rng(args.seed + 1)), sched, ctrl)
    train = np.random.default_rng(0).random(args.rigs) < 0.5

    print("R^2 of a linear read-out of the hidden state (held-out rigs); column k = after k songs")
    print(f"{'':12s}" + "".join(f"{k:>8d}" for k in range(args.songs + 1)))
    for key in TARGETS:
        row = [ridge_r2(s["h"], s[key], train, args.alpha) for s in res["probe"]]
        print(f"{key:12s}" + "".join(f"{r:8.2f}" for r in row))


if __name__ == "__main__":
    main()
