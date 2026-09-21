"""Train Gate 3 as two explicit NNs: acoustic aim and inverse motor."""
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

from flute_rl.yamabiko import RigParams  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.staged_nn import (MotorInversePolicy, ReferenceMemory, StagedConfig,
                                         TargetPositionPlanner)  # noqa: E402
from train_yamabiko_staged_nn import PITCH_CENTER, PITCH_SCALE, collate, make_samples  # noqa: E402


def true_position(target):
    cents = target[..., 0].detach().cpu().numpy() * PITCH_SCALE + PITCH_CENTER
    nominal = RigParams.nominal(1)
    position = nominal.x_for_cents(cents) / nominal.stroke[0]
    return torch.from_numpy(np.clip(position, 0, 1).astype(np.float32)).to(target.device)[..., None]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_staged_nn_comparator.pt")
    ap.add_argument("--out", default="runs/yamabiko_staged_nn_position_motor.pt")
    ap.add_argument("--report", default="runs/yamabiko_position_motor_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=64)
    ap.add_argument("--planner-steps", type=int, default=600); ap.add_argument("--motor-steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seed", type=int, default=12263)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); cfg = StagedConfig(**ck["config"])
    memory = ReferenceMemory(cfg).to(args.device); memory.load_state_dict(ck["memory"]); memory.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    train = make_samples(ear, rng, args.songs, args.device, .8, random_rig=False)
    valid = make_samples(ear, np.random.default_rng(args.seed + 100_000), args.valid_songs, args.device, .8, random_rig=False)
    planner, motor = TargetPositionPlanner(cfg).to(args.device), MotorInversePolicy(cfg).to(args.device)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=1e-3)
    for step in range(1, args.planner_steps + 1):
        ids = rng.integers(len(train), size=args.batch); ear_in, target, _, lengths, mask = collate(train, ids, args.device)
        with torch.no_grad(): _, stored = memory(ear_in, lengths); decoded = memory.decode(stored)
        label = true_position(target); voiced = target[..., 1].bool() & mask
        pred = planner(decoded); loss = F.mse_loss(pred[..., 0][voiced], label[..., 0][voiced])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"planner {step:4d} loss {float(loss.detach()):.6f}", flush=True)
    planner.eval(); [p.requires_grad_(False) for p in planner.parameters()]
    optimizer = torch.optim.AdamW(motor.parameters(), lr=5e-4, weight_decay=1e-6)
    for step in range(1, args.motor_steps + 1):
        ids = rng.integers(len(train), size=args.batch); ear_in, _, actions, lengths, mask = collate(train, ids, args.device)
        actual_position = torch.zeros_like(mask, dtype=torch.float32)
        for row, index in enumerate(ids):
            value = train[int(index)].position.to(args.device); actual_position[row, :len(value)] = value
        with torch.no_grad(): _, stored = memory(ear_in, lengths); decoded = memory.decode(stored); position = planner(decoded)
        # Gradually expose the motor to its own previous commands.
        teacher_probability = max(.1, 1.0 - step / max(args.motor_steps * .8, 1))
        teacher = actions if rng.random() < teacher_probability else None
        pred, estimated_position = motor(position, decoded, lengths, teacher, return_position=True)
        pwm = F.mse_loss(pred[..., 0][mask], actions[..., 0][mask])
        valve = F.binary_cross_entropy(pred[..., 1][mask].clamp(1e-5, 1 - 1e-5), actions[..., 1][mask])
        delta = actions[:, 1:, 0] - actions[:, :-1, 0]; transitions = (delta.abs() > .25) & mask[:, 1:]
        edge = F.mse_loss(pred[:, 1:, 0][transitions], actions[:, 1:, 0][transitions]) if transitions.any() else 0.0
        state_loss = F.mse_loss(estimated_position[mask], actual_position[mask])
        loss = pwm + .5 * edge + .2 * valve + .5 * state_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(motor.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"motor {step:4d} loss {float(loss.detach()):.5f} pwm {float(pwm.detach()):.5f} state {float(state_loss.detach()):.5f}", flush=True)
    # Fixed held-out action diagnostic; physical sound is evaluated by the
    # separate listening script and is the actual Gate 3 decision.
    abs_pwm, valve_ok, pos_error = [], [], []
    with torch.inference_mode():
        for start in range(0, len(valid), 16):
            ear_in, target, actions, lengths, mask = collate(valid, range(start, min(start + 16, len(valid))), args.device)
            _, stored = memory(ear_in, lengths); decoded = memory.decode(stored); pos = planner(decoded)
            pred = motor(pos, decoded, lengths)
            abs_pwm.append((pred[..., 0][mask] - actions[..., 0][mask]).abs().cpu())
            valve_ok.append(((pred[..., 1][mask] >= .5) == (actions[..., 1][mask] >= .5)).cpu())
            voiced = target[..., 1].bool() & mask; pos_error.append((pos[..., 0][voiced] - true_position(target)[..., 0][voiced]).abs().cpu())
    metrics = {"position_mae_percent_stroke": float(torch.cat(pos_error).mean() * 100),
               "pwm_mae": float(torch.cat(abs_pwm).mean()),
               "valve_accuracy": float(torch.cat(valve_ok).float().mean())}
    ck["position_planner"] = planner.state_dict(); ck["motor_inverse"] = motor.eval().state_dict()
    ck["position_motor_training"] = metrics; torch.save(ck, args.out)
    pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
