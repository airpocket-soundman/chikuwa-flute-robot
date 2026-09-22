"""Train a neural motor state estimator and a separate position controller."""
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
from flute_rl.yamabiko.staged_nn import (MotorStateEstimator, PositionActionController,
                                         ReferenceMemory, StagedConfig,
                                         TargetPositionPlanner)  # noqa: E402
from train_yamabiko_staged_nn import collate, make_samples  # noqa: E402


def position_batch(samples, ids, steps, device):
    out = torch.zeros(len(ids), steps, device=device)
    for row, index in enumerate(ids):
        value = samples[int(index)].position.to(device); out[row, :len(value)] = value
    current = torch.cat([torch.zeros(len(ids), 1, device=device), out[:, :-1]], 1)
    velocity = torch.cat([torch.zeros(len(ids), 1, device=device), current[:, 1:] - current[:, :-1]], 1)
    return current, velocity


def controller_rollout(controller, desired, target, estimate, actions=None):
    batch, steps = target.shape[:2]; previous = torch.zeros(batch, 2, device=target.device); out = []
    for t in range(steps):
        action = controller(desired, target, estimate[:, t], previous, t); out.append(action)
        previous = actions[:, t] if actions is not None else action
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_staged_nn_position_motor_v2.pt")
    ap.add_argument("--out", default="runs/yamabiko_staged_nn_state_controller.pt")
    ap.add_argument("--report", default="runs/yamabiko_state_controller_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=64)
    ap.add_argument("--state-steps", type=int, default=700); ap.add_argument("--controller-steps", type=int, default=700)
    ap.add_argument("--batch", type=int, default=12); ap.add_argument("--seed", type=int, default=13331)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); cfg = StagedConfig(**ck["config"])
    memory = ReferenceMemory(cfg).to(args.device); memory.load_state_dict(ck["memory"]); memory.eval()
    planner = TargetPositionPlanner(cfg).to(args.device); planner.load_state_dict(ck["position_planner"]); planner.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    train = make_samples(ear, rng, args.songs, args.device, .8, random_rig=False)
    valid = make_samples(ear, np.random.default_rng(args.seed + 100_000), args.valid_songs, args.device, .8, random_rig=False)
    state_net, controller = MotorStateEstimator(cfg).to(args.device), PositionActionController(cfg).to(args.device)
    optimizer = torch.optim.AdamW(state_net.parameters(), lr=8e-4, weight_decay=1e-6)
    for step in range(1, args.state_steps + 1):
        ids = rng.integers(len(train), size=args.batch); _, _, actions, lengths, mask = collate(train, ids, args.device)
        pos, vel = position_batch(train, ids, actions.shape[1], args.device); pred = state_net(actions)
        loss_pos = F.mse_loss(pred[..., 0][mask], pos[mask]); loss_vel = F.mse_loss(pred[..., 1][mask], vel[mask])
        loss = loss_pos + 5 * loss_vel
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(state_net.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"state {step:4d} pos {float(loss_pos.detach()):.6f} vel {float(loss_vel.detach()):.6f}", flush=True)
    state_net.eval(); [p.requires_grad_(False) for p in state_net.parameters()]
    optimizer = torch.optim.AdamW(controller.parameters(), lr=8e-4, weight_decay=1e-6)
    for step in range(1, args.controller_steps + 1):
        ids = rng.integers(len(train), size=args.batch); ear_in, _, actions, lengths, mask = collate(train, ids, args.device)
        with torch.no_grad():
            _, stored = memory(ear_in, lengths); target = memory.decode(stored); desired = planner(target); estimate = state_net(actions)
        teacher = actions if step <= int(.7 * args.controller_steps) or step % 3 else None
        pred = controller_rollout(controller, desired, target, estimate, teacher)
        pwm = F.mse_loss(pred[..., 0][mask], actions[..., 0][mask])
        valve = F.binary_cross_entropy(pred[..., 1][mask].clamp(1e-5, 1 - 1e-5), actions[..., 1][mask])
        loss = pwm + .2 * valve
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(controller.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"controller {step:4d} pwm {float(pwm.detach()):.5f} valve {float(valve.detach()):.5f}", flush=True)
    # Held-out tests for the two newly separated NNs.
    pos_errors, vel_errors, pwm_errors, valve_ok = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(valid), 16):
            ids = list(range(start, min(start + 16, len(valid)))); ear_in, _, actions, lengths, mask = collate(valid, ids, args.device)
            pos, vel = position_batch(valid, ids, actions.shape[1], args.device); estimate = state_net(actions)
            _, stored = memory(ear_in, lengths); target = memory.decode(stored); desired = planner(target)
            pred = controller_rollout(controller, desired, target, estimate)
            pos_errors.append((estimate[..., 0][mask] - pos[mask]).abs().cpu())
            vel_errors.append((estimate[..., 1][mask] - vel[mask]).abs().cpu())
            pwm_errors.append((pred[..., 0][mask] - actions[..., 0][mask]).abs().cpu())
            valve_ok.append(((pred[..., 1][mask] >= .5) == (actions[..., 1][mask] >= .5)).cpu())
    metrics = {"state_position_mae_percent_stroke": float(torch.cat(pos_errors).mean() * 100),
               "state_velocity_mae_percent_stroke_per_step": float(torch.cat(vel_errors).mean() * 100),
               "controller_pwm_mae": float(torch.cat(pwm_errors).mean()),
               "controller_valve_accuracy": float(torch.cat(valve_ok).float().mean())}
    ck["motor_state"] = state_net.state_dict(); ck["position_controller"] = controller.eval().state_dict()
    ck["state_controller_training"] = metrics; torch.save(ck, args.out)
    pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
