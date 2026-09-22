"""Train the separated offline Tempo/Beat -> stored timing-profile NN."""
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

from flute_rl.yamabiko.beat_grid import BeatGridConfig, ReferenceTimingProfileNet, TempoBeatNet  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from train_yamabiko_tempo_beat import collate, dataset  # noqa: E402


def true_pointer(samples, ids, steps, device, subdivision):
    out = torch.zeros(len(ids), steps, device=device); valid = torch.zeros(len(ids), steps, dtype=torch.bool, device=device)
    for row, idx in enumerate(ids):
        beat = samples[int(idx)].beat; n = len(beat.target); valid[row, :n] = True
        time = np.arange(n) / 100.0
        out[row, :n] = torch.from_numpy(((time - beat.beat0_s) * beat.bpm / 60 * subdivision).astype(np.float32)).to(device)
    return out, valid


@torch.inference_mode()
def evaluate(tempo, profile, samples, device):
    errors = []; backwards = 0
    for start in range(0, len(samples), 12):
        ids = list(range(start, min(start + 12, len(samples))))
        features, lengths, _, _, _ = collate(samples, ids, device)
        _, beat_out, encoded, mask = tempo(features, lengths); pred = profile(encoded, beat_out, mask)
        truth, valid = true_pointer(samples, ids, pred.shape[1], device, tempo.config.subdivision)
        errors.append((pred[valid] - truth[valid]).abs().cpu()); backwards += int((pred[:, 1:] < pred[:, :-1]).sum())
    error = torch.cat(errors)
    return {"pointer_mae_cells": float(error.mean()), "pointer_p95_cells": float(torch.quantile(error, .95)),
            "backward_jumps": backwards}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_temporal_connected_tuned_v2.pt")
    ap.add_argument("--out", default="runs/yamabiko_timing_profile.pt")
    ap.add_argument("--report", default="runs/yamabiko_timing_profile_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1400); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=27491); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); cfg = BeatGridConfig(**ck["config"])
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    tempo = TempoBeatNet(cfg).to(args.device); tempo.load_state_dict(ck["tempo_beat"]); tempo.eval()
    for module in (ear, tempo):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    train = dataset(ear, rng, args.songs, args.device)
    bpms = [69, 87, 103, 129, 153, 177]
    valid = dataset(ear, np.random.default_rng(args.seed + 100000), args.valid_songs, args.device, bpms)
    profile = ReferenceTimingProfileNet(cfg).to(args.device)
    if ck.get("timing_profile_trained"): profile.load_state_dict(ck["timing_profile"])
    optimizer = torch.optim.AdamW(profile.parameters(), lr=args.lr, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(train), size=args.batch).tolist(); features, lengths, _, _, _ = collate(train, ids, args.device)
        with torch.no_grad(): _, beat_out, encoded, mask = tempo(features, lengths)
        pred = profile(encoded, beat_out, mask); truth, valid_mask = true_pointer(train, ids, pred.shape[1], args.device, cfg.subdivision)
        loss_position = F.smooth_l1_loss(pred[valid_mask], truth[valid_mask])
        pair = valid_mask[:, 1:] & valid_mask[:, :-1]
        loss_delta = F.smooth_l1_loss((pred[:, 1:] - pred[:, :-1])[pair],
                                      (truth[:, 1:] - truth[:, :-1])[pair])
        loss_start = F.smooth_l1_loss(pred[:, 0], truth[:, 0])
        loss = loss_position + 20 * loss_delta + 2 * loss_start
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(profile.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"step {step:4d} loss {float(loss.detach()):.6f} position {float(loss_position.detach()):.6f}", flush=True)
    metrics = evaluate(tempo, profile.eval(), valid, args.device)
    metrics["pass"] = metrics["pointer_mae_cells"] <= .10 and metrics["pointer_p95_cells"] <= .25 and metrics["backward_jumps"] == 0
    metrics.update({"seed": args.seed + 100000, "split": "development-unknown-bpm", "official_path": "predicted_tempo_features"})
    ck["timing_profile"] = profile.state_dict(); ck["timing_profile_trained"] = True; ck["timing_profile_metrics"] = metrics
    torch.save(ck, args.out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
