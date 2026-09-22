"""Train Musical Memory strictly from predicted Tempo/Beat upstream features."""
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

from flute_rl.yamabiko.beat_grid import BeatGridConfig, MusicalMemoryNet, TempoBeatNet  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from train_yamabiko_tempo_beat import collate, dataset  # noqa: E402

CENTER, SCALE = 1300.0, 600.0


def cell_targets(samples, ids, device):
    chosen = [samples[int(i)] for i in ids]; cells = len(chosen[0].beat.cell_pitch)
    out = torch.zeros(len(chosen), cells, 5, device=device)
    for row, item in enumerate(chosen):
        beat = item.beat
        out[row, :, 0] = torch.from_numpy((beat.cell_pitch - CENTER) / SCALE).to(device)
        out[row, :, 1] = torch.from_numpy(beat.cell_voice.astype(np.float32)).to(device)
        out[row, :, 2] = torch.from_numpy(beat.cell_onset.astype(np.float32)).to(device)
        out[row, :, 3] = torch.from_numpy(beat.cell_offset.astype(np.float32)).to(device)
        out[row, :, 4] = torch.from_numpy(beat.cell_slope / SCALE).to(device)
    return out


@torch.inference_mode()
def evaluate(tempo, memory, samples, device):
    pitch_errors, voice_truth, voice_pred, onset_truth, onset_pred, offset_truth, offset_pred = [], [], [], [], [], [], []
    slope_errors, pitch_truth, pitch_pred = [], [], []
    for start in range(0, len(samples), 16):
        ids = list(range(start, min(start + 16, len(samples)))); features, lengths, _, _, _ = collate(samples, ids, device)
        target = cell_targets(samples, ids, device); _, beat_out, encoded, mask = tempo(features, lengths)
        pred, _ = memory(features, encoded, beat_out, mask, target.shape[1]); voiced = target[..., 1].bool()
        pitch_errors.append((pred[..., 0][voiced] - target[..., 0][voiced]).abs().cpu() * SCALE)
        slope_errors.append((pred[..., 4][voiced] - target[..., 4][voiced]).abs().cpu() * SCALE)
        pitch_truth.append(target[..., 0][voiced].cpu()); pitch_pred.append(pred[..., 0][voiced].cpu())
        voice_truth.append(target[..., 1].cpu()); voice_pred.append((pred[..., 1] >= 0).cpu())
        onset_truth.append(target[..., 2].cpu()); onset_pred.append((pred[..., 2] >= 0).cpu())
        offset_truth.append(target[..., 3].cpu()); offset_pred.append((pred[..., 3] >= 0).cpu())
    def f1(truths, preds):
        truth = torch.cat(truths).bool(); pred = torch.cat(preds).bool()
        tp = int((truth & pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1)
    truth, pred = torch.cat(pitch_truth).numpy(), torch.cat(pitch_pred).numpy()
    return {"pitch_mae_cents": float(torch.cat(pitch_errors).mean()),
            "pitch_p90_cents": float(torch.quantile(torch.cat(pitch_errors), .9)),
            "trajectory_correlation": float(np.corrcoef(truth, pred)[0, 1]),
            "voice_f1": f1(voice_truth, voice_pred), "onset_f1": f1(onset_truth, onset_pred),
            "offset_f1": f1(offset_truth, offset_pred), "slope_mae_cents_per_cell": float(torch.cat(slope_errors).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_tempo_beat_v2.pt")
    ap.add_argument("--out", default="runs/yamabiko_musical_memory.pt"); ap.add_argument("--report", default="runs/yamabiko_musical_memory_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=16673); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); config = BeatGridConfig(**ck["config"])
    tempo, memory = TempoBeatNet(config).to(args.device), MusicalMemoryNet(config).to(args.device)
    tempo.load_state_dict(ck["tempo_beat"]); memory.load_state_dict(ck["musical_memory"]); tempo.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    for module in (ear, tempo):
        for p in module.parameters(): p.requires_grad_(False)
    train = dataset(ear, rng, args.songs, args.device)
    valid = dataset(ear, np.random.default_rng(args.seed + 100_000), args.valid_songs, args.device,
                    [71, 89, 107, 131, 149, 179])
    optimizer = torch.optim.AdamW(memory.parameters(), lr=6e-4, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(train), size=args.batch); features, lengths, _, _, _ = collate(train, ids, args.device)
        target = cell_targets(train, ids, args.device)
        with torch.no_grad(): _, beat_out, encoded, mask = tempo(features, lengths)
        pred, _ = memory(features, encoded, beat_out, mask, target.shape[1]); voiced = target[..., 1].bool()
        loss_pitch = F.smooth_l1_loss(pred[..., 0][voiced], target[..., 0][voiced])
        loss_voice = F.binary_cross_entropy_with_logits(pred[..., 1], target[..., 1])
        loss_onset = F.binary_cross_entropy_with_logits(pred[..., 2], target[..., 2], pos_weight=torch.tensor(3.0, device=args.device))
        loss_offset = F.binary_cross_entropy_with_logits(pred[..., 3], target[..., 3], pos_weight=torch.tensor(3.0, device=args.device))
        loss_slope = F.smooth_l1_loss(pred[..., 4][voiced], target[..., 4][voiced])
        loss = loss_pitch + .3 * loss_voice + .2 * (loss_onset + loss_offset) + .2 * loss_slope
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(memory.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"step {step:4d} loss {float(loss.detach()):.5f} pitch {float(loss_pitch.detach()):.5f} voice {float(loss_voice.detach()):.5f}", flush=True)
    metrics = evaluate(tempo, memory.eval(), valid, args.device)
    metrics["pass"] = (metrics["pitch_mae_cents"] <= 50 and metrics["trajectory_correlation"] >= .95 and
                       metrics["voice_f1"] >= .95 and metrics["onset_f1"] >= .9 and metrics["offset_f1"] >= .9)
    metrics.update({"seed": args.seed, "split": "frozen-unknown-bpm", "official_path": "predicted_upstream"})
    ck["musical_memory"] = memory.state_dict(); ck["musical_memory_trained"] = True; ck["musical_memory_metrics"] = metrics
    torch.save(ck, args.out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
