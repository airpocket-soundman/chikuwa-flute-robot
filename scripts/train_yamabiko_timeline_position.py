"""Train the separated Timeline Memory -> physical plunger-position NN."""
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

from flute_rl.yamabiko.beat_grid import BeatGridConfig, BeatTimelineRecallNet, TempoBeatNet  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.staged_nn import StagedConfig, TargetPositionPlanner  # noqa: E402
from train_yamabiko_tempo_beat import collate, dataset  # noqa: E402
from train_yamabiko_temporal_aligner import frame_targets  # noqa: E402

CENTER, SCALE = 1300.0, 600.0


def labels(items, steps, device):
    target, mask, _, _ = frame_targets(items, steps, device)
    # The adopted physical simulator uses a linear 700..1900 cent flute.
    # Normalized pitch -1..+1 therefore maps exactly to stroke 0..1.
    return target, mask, ((target[..., 0] + 1.0) * 0.5).clamp(0, 1)


def oracle_planner_input(target):
    """Match the connected contract: normalized pitch plus voicing logit."""
    voice_logit = torch.where(target[..., 1:2] >= .5, 6.0, -6.0)
    return torch.cat([target[..., :1], voice_logit], -1)


def ideal_position_for_pitch(pitch, device):
    """Supervise the Planner's transform without asking it to fix upstream pitch."""
    return ((pitch.detach() + 1.0) * 0.5).clamp(0, 1).to(device)


@torch.inference_mode()
def evaluate(tempo, timeline, planner, samples, device):
    errors = []; pitch_errors = []; connected_errors = []; connected_pitch_errors = []
    inherited_pitch_errors = []; model_added_pitch_errors = []; model_added_position_errors = []
    inherited_signed = []; model_added_signed = []; total_signed = []
    for start in range(0, len(samples), 12):
        ids = list(range(start, min(start + 12, len(samples))))
        features, lengths, _, _, _ = collate(samples, ids, device)
        _, beat_out, encoded, _ = tempo(features, lengths); decoded, stored = timeline(features, encoded, beat_out, lengths)
        decoded = timeline.decode(stored); target, mask, label = labels([samples[i].beat for i in ids], decoded.shape[1], device)
        voice = mask & target[..., 1].bool(); position = planner(oracle_planner_input(target))[..., 0]
        connected = planner(decoded[..., :2])[..., 0]
        errors.append((position[voice] - label[voice]).abs().cpu()); connected_errors.append((connected[voice] - label[voice]).abs().cpu())
        cents = 700.0 + 1200.0 * position[voice].cpu().numpy()
        connected_cents = 700.0 + 1200.0 * connected[voice].cpu().numpy()
        truth = target[..., 0][voice].cpu().numpy() * SCALE + CENTER
        requested = decoded[..., 0][voice].cpu().numpy() * SCALE + CENTER
        ideal_connected_position = np.clip((requested - 700.0) / 1200.0, 0, 1)
        ideal_connected_cents = 700.0 + 1200.0 * ideal_connected_position
        inherited = ideal_connected_cents - truth
        model_added = connected_cents - ideal_connected_cents
        total = connected_cents - truth
        pitch_errors.append(torch.from_numpy(np.abs(cents - truth).astype(np.float32)))
        connected_pitch_errors.append(torch.from_numpy(np.abs(connected_cents - truth).astype(np.float32)))
        inherited_pitch_errors.append(torch.from_numpy(np.abs(inherited).astype(np.float32)))
        model_added_pitch_errors.append(torch.from_numpy(np.abs(model_added).astype(np.float32)))
        model_added_position_errors.append(torch.from_numpy(
            np.abs(connected[voice].cpu().numpy() - ideal_connected_position).astype(np.float32)))
        inherited_signed.append(torch.from_numpy(inherited.astype(np.float32)))
        model_added_signed.append(torch.from_numpy(model_added.astype(np.float32)))
        total_signed.append(torch.from_numpy(total.astype(np.float32)))
    inherited_signed_all = torch.cat(inherited_signed)
    model_added_signed_all = torch.cat(model_added_signed)
    total_signed_all = torch.cat(total_signed)
    result = {"position_mae_percent_stroke": float(torch.cat(errors).mean() * 100),
              "position_p95_percent_stroke": float(torch.quantile(torch.cat(errors), .95) * 100),
              "steady_pitch_mae_cents": float(torch.cat(pitch_errors).mean()),
              "connected_position_mae_percent_stroke": float(torch.cat(connected_errors).mean() * 100),
              "connected_steady_pitch_mae_cents": float(torch.cat(connected_pitch_errors).mean()),
              "inherited_pitch_mae_cents": float(torch.cat(inherited_pitch_errors).mean()),
              "model_added_position_mae_percent_stroke": float(torch.cat(model_added_position_errors).mean() * 100),
              "model_added_pitch_mae_cents": float(torch.cat(model_added_pitch_errors).mean()),
              "model_added_pitch_p95_cents": float(torch.quantile(torch.cat(model_added_pitch_errors), .95)),
              "output_total_pitch_mae_cents": float(total_signed_all.abs().mean()),
              "output_total_pitch_p90_cents": float(torch.quantile(total_signed_all.abs(), .90)),
              "inherited_pitch_bias_cents": float(inherited_signed_all.mean()),
              "model_added_pitch_bias_cents": float(model_added_signed_all.mean()),
              "output_total_pitch_bias_cents": float(total_signed_all.mean()),
              "net_absolute_change_cents": float(total_signed_all.abs().mean() - inherited_signed_all.abs().mean()),
              "attribution_residual_max_cents": float(
                  (inherited_signed_all + model_added_signed_all - total_signed_all).abs().max())}
    result["oracle_input_pass"] = (result["position_mae_percent_stroke"] <= 1.0 and
                                   result["steady_pitch_mae_cents"] <= 30)
    result["real_input_transform_pass"] = (result["model_added_position_mae_percent_stroke"] <= 1.0 and
                                            result["model_added_pitch_mae_cents"] <= 30)
    # The preceding Timeline gate allows up to 60 cent MAE.  The connected
    # output budget therefore includes that carry-in plus a small transform
    # allowance, while the Planner's own residual remains capped at 30 cent.
    result["connected_output_pass"] = (result["output_total_pitch_mae_cents"] <= 80 and
                                       result["output_total_pitch_mae_cents"] <=
                                       result["inherited_pitch_mae_cents"] + 20)
    result["pass"] = result["oracle_input_pass"] and result["real_input_transform_pass"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_timeline_timbre_v2_final.pt")
    ap.add_argument("--out", default="runs/yamabiko_timeline_position.pt"); ap.add_argument("--report", default="runs/yamabiko_timeline_position_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=700); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=23611); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); cfg = BeatGridConfig(**ck["config"])
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    tempo = TempoBeatNet(cfg).to(args.device); tempo.load_state_dict(ck["tempo_beat"]); tempo.eval()
    timeline = BeatTimelineRecallNet(cfg, direct_pitch=ck.get("timeline_memory_kind") == "direct-pitch-v1").to(args.device)
    timeline.load_state_dict(ck["timeline_memory"]); timeline.eval()
    for module in (ear, tempo, timeline):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    train = dataset(ear, rng, args.songs, args.device); bpms = [73, 97, 113, 137, 157, 173]
    valid = dataset(ear, np.random.default_rng(args.seed + 100000), args.valid_songs, args.device, bpms)
    planner = TargetPositionPlanner(StagedConfig()).to(args.device); optimizer = torch.optim.AdamW(planner.parameters(), lr=8e-4)
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(train), size=args.batch).tolist(); features, lengths, _, _, _ = collate(train, ids, args.device)
        with torch.no_grad():
            _, beat_out, encoded, _ = tempo(features, lengths); _, stored = timeline(features, encoded, beat_out, lengths); decoded = timeline.decode(stored)
            target, mask, label = labels([train[i].beat for i in ids], decoded.shape[1], args.device)
        voice = mask & target[..., 1].bool()
        oracle_pred = planner(oracle_planner_input(target))[..., 0]
        connected_pred = planner(decoded[..., :2])[..., 0]
        connected_label = ideal_position_for_pitch(decoded[..., 0], args.device)
        # Mix clean and predicted-upstream inputs.  The connected target is the
        # ideal position for the pitch actually received, so this block learns
        # only its own transform and does not hide Timeline error.
        loss_oracle = F.mse_loss(oracle_pred[voice], label[voice])
        loss_connected = F.mse_loss(connected_pred[voice], connected_label[voice])
        loss = .35 * loss_oracle + .65 * loss_connected
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"step {step:4d} loss {float(loss.detach()):.7f}", flush=True)
    metrics = evaluate(tempo, timeline, planner.eval(), valid, args.device)
    metrics.update({"seed": args.seed + 100000, "split": "frozen-unknown-bpm", "official_path": "stored_timeline",
                    "input_contract": "normalized-pitch-plus-voice-logit-v2",
                    "oracle_loss_weight": .35, "predicted_upstream_loss_weight": .65})
    ck["timeline_position_input_contract"] = "normalized-pitch-plus-voice-logit-v2"
    ck["timeline_position_planner"] = planner.state_dict(); ck["timeline_position_metrics"] = metrics
    torch.save(ck, args.out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
