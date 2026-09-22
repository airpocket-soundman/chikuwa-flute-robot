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

from flute_rl.yamabiko import RigParams  # noqa: E402
from flute_rl.yamabiko.beat_grid import BeatGridConfig, BeatTimelineRecallNet, TempoBeatNet  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.staged_nn import StagedConfig, TargetPositionPlanner  # noqa: E402
from train_yamabiko_tempo_beat import collate, dataset  # noqa: E402
from train_yamabiko_temporal_aligner import frame_targets  # noqa: E402

CENTER, SCALE = 1300.0, 600.0


def labels(items, steps, device):
    target, mask, _, _ = frame_targets(items, steps, device)
    cents = target[..., 0].cpu().numpy() * SCALE + CENTER
    rig = RigParams.nominal(1); x = rig.x_for_cents(cents) / rig.stroke[0]
    return target, mask, torch.from_numpy(np.clip(x, 0, 1).astype(np.float32)).to(device)


@torch.inference_mode()
def evaluate(tempo, timeline, planner, samples, device):
    errors = []; pitch_errors = []; connected_errors = []; connected_pitch_errors = []
    rig = RigParams.nominal(1)
    for start in range(0, len(samples), 12):
        ids = list(range(start, min(start + 12, len(samples))))
        features, lengths, _, _, _ = collate(samples, ids, device)
        _, beat_out, encoded, _ = tempo(features, lengths); decoded, stored = timeline(features, encoded, beat_out, lengths)
        decoded = timeline.decode(stored); target, mask, label = labels([samples[i].beat for i in ids], decoded.shape[1], device)
        voice = mask & target[..., 1].bool(); position = planner(target[..., :2])[..., 0]
        connected = planner(decoded[..., :2])[..., 0]
        errors.append((position[voice] - label[voice]).abs().cpu()); connected_errors.append((connected[voice] - label[voice]).abs().cpu())
        cents = rig.cents_at((position[voice].cpu().numpy() * rig.stroke[0]))
        connected_cents = rig.cents_at((connected[voice].cpu().numpy() * rig.stroke[0]))
        truth = target[..., 0][voice].cpu().numpy() * SCALE + CENTER
        pitch_errors.append(torch.from_numpy(np.abs(cents - truth).astype(np.float32)))
        connected_pitch_errors.append(torch.from_numpy(np.abs(connected_cents - truth).astype(np.float32)))
    result = {"position_mae_percent_stroke": float(torch.cat(errors).mean() * 100),
              "position_p95_percent_stroke": float(torch.quantile(torch.cat(errors), .95) * 100),
              "steady_pitch_mae_cents": float(torch.cat(pitch_errors).mean()),
              "connected_position_mae_percent_stroke": float(torch.cat(connected_errors).mean() * 100),
              "connected_steady_pitch_mae_cents": float(torch.cat(connected_pitch_errors).mean())}
    result["pass"] = result["position_mae_percent_stroke"] <= 1.0 and result["steady_pitch_mae_cents"] <= 30
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
        voice = mask & target[..., 1].bool(); pred = planner(target[..., :2])[..., 0]
        loss = F.mse_loss(pred[voice], label[voice])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"step {step:4d} loss {float(loss.detach()):.7f}", flush=True)
    metrics = evaluate(tempo, timeline, planner.eval(), valid, args.device)
    metrics.update({"seed": args.seed + 100000, "split": "frozen-unknown-bpm", "official_path": "stored_timeline"})
    ck["timeline_position_planner"] = planner.state_dict(); ck["timeline_position_metrics"] = metrics
    torch.save(ck, args.out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
