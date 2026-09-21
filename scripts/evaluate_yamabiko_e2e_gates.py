"""Evaluate actuator-facing E2E gates with trajectory metrics, not MAE alone."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from train_yamabiko_e2e_rl import rollout  # noqa: E402


def metrics(take):
    correlations, directions, lags = [], [], []
    for i in range(len(take.mae)):
        length = int(take.lengths[i]); target = take.target[i, :length]
        actual = take.true_cents[i, :length]; sounding = take.true_sounding[i, :length]
        valid = np.isfinite(target) & sounding
        if valid.sum() > 4 and np.std(target[valid]) > 10 and np.std(actual[valid]) > 10:
            correlations.append(float(np.corrcoef(target[valid], actual[valid])[0, 1]))
        changes = np.flatnonzero(np.isfinite(target[1:]) & np.isfinite(target[:-1])
                                 & (np.abs(target[1:] - target[:-1]) >= 25)) + 1
        for t in changes:
            before = actual[max(0, t - 5):t]
            desired = np.sign(target[t] - target[t - 1])
            found = False
            for delay in range(1, min(31, length - t)):
                after = actual[t + delay:min(length, t + delay + 5)]
                if len(before) and len(after) and np.sign(np.nanmean(after) - np.nanmean(before)) == desired:
                    directions.append(1.0); lags.append(delay * 10.0); found = True; break
            if not found: directions.append(0.0)
    finite = np.isfinite(take.mae)
    return {"mae_cents": float(np.mean(take.mae[finite])),
            "trajectory_correlation": float(np.mean(correlations)) if correlations else float("nan"),
            "change_direction_accuracy": float(np.mean(directions)) if directions else float("nan"),
            "transition_lag_ms": float(np.mean(lags)) if lags else float("nan"),
            "sounding_rate": float(np.mean(take.sounding))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True); ap.add_argument("--report", default=None)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--progress", type=float, default=.4); ap.add_argument("--harsh", type=float, default=0.0)
    ap.add_argument("--audio-domain", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); model = E2EImitator.from_checkpoint(torch.load(args.model, map_location=args.device), args.device).eval()
    ff = rollout(model, np.random.default_rng(args.seed + 100_000), args.batch, 1, args.progress, args.harsh,
                 .15, .98, args.device, deterministic=True, mute_self=True, audio_domain=args.audio_domain)[0]
    result = metrics(ff)
    result["gate3_pass"] = (result["mae_cents"] < 150 and result["trajectory_correlation"] >= .9
                            and result["change_direction_accuracy"] >= .95)
    print(json.dumps(result, indent=2))
    if args.report:
        path = pathlib.Path(args.report); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
