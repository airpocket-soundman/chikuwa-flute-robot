"""Calibrate the neural Position Planner to the adopted linear flute model."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.staged_nn import StagedConfig, TargetPositionPlanner  # noqa: E402


PITCH_SPAN_CENTS = 1200.0


@torch.inference_mode()
def evaluate(planner: TargetPositionPlanner, device: str) -> dict:
    pitch = torch.linspace(-1.0, 1.0, 2001, device=device)
    logits = torch.tensor([-12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0], device=device)
    p = pitch[:, None].expand(-1, len(logits))
    v = logits[None, :].expand(len(pitch), -1)
    inputs = torch.stack([p, v], -1)
    prediction = planner(inputs)[..., 0]
    truth = (p + 1.0) * 0.5
    error_cents = (prediction - truth).abs() * PITCH_SPAN_CENTS
    voice_spread_cents = (prediction.max(1).values - prediction.min(1).values) * PITCH_SPAN_CENTS
    center = prediction[:, len(logits) // 2]
    return {
        "position_mae_percent_stroke": float((prediction - truth).abs().mean() * 100),
        "position_p95_percent_stroke": float(torch.quantile((prediction - truth).abs(), .95) * 100),
        "model_added_pitch_mae_cents": float(error_cents.mean()),
        "model_added_pitch_p95_cents": float(torch.quantile(error_cents, .95)),
        "model_added_pitch_max_cents": float(error_cents.max()),
        "voice_invariance_max_cents": float(voice_spread_cents.max()),
        "monotonic_violations": int((center[1:] < center[:-1]).sum()),
        "pass": bool(error_cents.mean() <= 2.0 and torch.quantile(error_cents, .95) <= 5.0
                     and error_cents.max() <= 20.0 and voice_spread_cents.max() <= 2.0
                     and (center[1:] < center[:-1]).sum() == 0),
        "simulator_contract": "linear flute: position=(normalized_pitch+1)/2",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="runs/yamabiko_timeline_position_v4.pt")
    parser.add_argument("--out", default="runs/yamabiko_timeline_position_v5.pt")
    parser.add_argument("--report", default="runs/yamabiko_timeline_position_v5_report.json")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=923611)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.init, map_location=args.device, weights_only=False)
    planner = TargetPositionPlanner(StagedConfig()).to(args.device)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=2e-3, weight_decay=0.0)
    voice_levels = torch.tensor([-12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0], device=args.device)

    for step in range(1, args.steps + 1):
        pitch = torch.empty(args.batch, device=args.device).uniform_(-1.0, 1.0)
        voice_a = voice_levels[torch.randint(len(voice_levels), (args.batch,), device=args.device)]
        voice_b = voice_levels[torch.randint(len(voice_levels), (args.batch,), device=args.device)]
        truth = (pitch + 1.0) * 0.5
        pred_a = planner(torch.stack([pitch, voice_a], -1))[:, 0]
        pred_b = planner(torch.stack([pitch, voice_b], -1))[:, 0]
        # The linear flute simulator defines the position target.  The second
        # term prevents voicing confidence from moving the plunger target.
        loss = F.smooth_l1_loss(pred_a, truth, beta=.01) + F.smooth_l1_loss(pred_b, truth, beta=.01)
        loss = loss + 2.0 * F.mse_loss(pred_a, pred_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0:
            print(f"step {step:4d} loss {float(loss.detach()):.8f}", flush=True)

    metrics = evaluate(planner.eval(), args.device)
    metrics.update({"seed": args.seed, "split": "dense-linear-flute-contract-v1",
                    "input_contract": "normalized-pitch-plus-voice-logit-v2",
                    "training_steps": args.steps})
    checkpoint["timeline_position_planner"] = planner.state_dict()
    checkpoint["timeline_position_metrics"] = metrics
    checkpoint["timeline_position_simulator_contract"] = metrics["simulator_contract"]
    torch.save(checkpoint, args.out)
    pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
