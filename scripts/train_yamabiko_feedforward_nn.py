"""Continue only the reference-only feed-forward controller."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.staged_nn import FeedForwardPolicy, ReferenceMemory, StagedConfig  # noqa: E402
from train_yamabiko_staged_nn import collate, ff_action_metrics, make_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_staged_nn.pt"); ap.add_argument("--out", default=None)
    ap.add_argument("--report", default="runs/yamabiko_feedforward_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=10291); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = args.out or args.init
    ck = torch.load(args.init, map_location=args.device); cfg = StagedConfig(**ck["config"])
    memory, policy = ReferenceMemory(cfg).to(args.device), FeedForwardPolicy(cfg).to(args.device)
    memory.load_state_dict(ck["memory"]); policy.load_state_dict(ck["feedforward"]); memory.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    train = make_samples(ear, rng, args.songs, args.device, .8, random_rig=False)
    valid = make_samples(ear, np.random.default_rng(args.seed + 100_000), args.valid_songs, args.device, .8, random_rig=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=4e-4, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(train), size=args.batch); ear_in, _, actions, lengths, mask = collate(train, ids, args.device)
        with torch.no_grad(): _, stored = memory(ear_in, lengths); decoded = memory.decode(stored)
        # Half AR and half teacher-forced batches; select by fixed held-out set,
        # never by the lowest random training batch.
        pred = policy(decoded, lengths, teacher_actions=actions if step % 2 else None)
        pwm = F.mse_loss(pred[..., 0][mask], actions[..., 0][mask])
        valve = F.binary_cross_entropy(pred[..., 1][mask].clamp(1e-5, 1 - 1e-5), actions[..., 1][mask])
        # Transitions need decisive PWM; otherwise MSE is dominated by long holds.
        delta = actions[:, 1:, 0] - actions[:, :-1, 0]
        transition = (delta.abs() > .25) & mask[:, 1:]
        transition_loss = F.mse_loss(pred[:, 1:, 0][transition], actions[:, 1:, 0][transition]) if transition.any() else 0.0
        loss = pwm + .5 * transition_loss + .2 * valve
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"step {step:4d} loss {float(loss.detach()):.5f} pwm {float(pwm.detach()):.5f}", flush=True)
    metrics = ff_action_metrics(memory, policy.eval(), valid, args.device)
    ck["feedforward"] = policy.state_dict(); ck.setdefault("metrics", {})["feedforward_continued"] = metrics
    torch.save(ck, out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {out}")


if __name__ == "__main__": main()
