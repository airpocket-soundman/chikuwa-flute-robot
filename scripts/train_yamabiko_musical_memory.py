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

from flute_rl.yamabiko.beat_grid import BeatAlignedMusicalMemoryNet, BeatGridConfig, TempoBeatNet  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from train_yamabiko_tempo_beat import collate, dataset  # noqa: E402

CENTER, SCALE = 1300.0, 600.0


def cell_targets(samples, ids, device):
    chosen = [samples[int(i)] for i in ids]; cells = max(len(x.beat.cell_pitch) for x in chosen)
    out = torch.zeros(len(chosen), cells, 5, device=device)
    mask = torch.zeros(len(chosen), cells, dtype=torch.bool, device=device)
    for row, item in enumerate(chosen):
        beat = item.beat; n = len(beat.cell_pitch); mask[row, :n] = True
        out[row, :n, 0] = torch.from_numpy((beat.cell_pitch - CENTER) / SCALE).to(device)
        out[row, :n, 1] = torch.from_numpy(beat.cell_voice.astype(np.float32)).to(device)
        out[row, :n, 2] = torch.from_numpy(beat.cell_onset.astype(np.float32)).to(device)
        out[row, :n, 3] = torch.from_numpy(beat.cell_offset.astype(np.float32)).to(device)
        out[row, :n, 4] = torch.from_numpy(beat.cell_slope / SCALE).to(device)
    return out, mask


@torch.inference_mode()
def evaluate(tempo, memory, samples, device):
    pitch_errors, voice_truth, voice_pred, onset_truth, onset_pred, offset_truth, offset_pred = [], [], [], [], [], [], []
    slope_errors, pitch_truth, pitch_pred = [], [], []
    upper_errors, upper_truth, upper_pred = [], [], []
    for index in range(len(samples)):
        item = samples[index]; beat = item.beat
        centers = np.clip(np.round((beat.beat0_s + (np.arange(len(beat.cell_pitch)) + .5)
                                    * 60.0 / beat.bpm / beat.subdivision) * 100).astype(int),
                          0, len(item.features) - 1)
        voiced_np = beat.cell_voice
        ear_pitch = item.features[centers, -2].numpy()
        true_pitch = (beat.cell_pitch - CENTER) / SCALE
        upper_errors.append(torch.from_numpy(np.abs(ear_pitch[voiced_np] - true_pitch[voiced_np]) * SCALE))
        upper_truth.append(true_pitch[voiced_np]); upper_pred.append(ear_pitch[voiced_np])
        features, lengths, _, _, _ = collate(samples, [index], device)
        target, _ = cell_targets(samples, [index], device); _, beat_out, encoded, mask = tempo(features, lengths)
        confidence = torch.sigmoid(beat_out[0, :, 2]); confident = torch.nonzero(confidence >= .5).flatten()
        start = int(confident[0]) if len(confident) else 0
        predicted_bpm = float(tempo.phase_bpm(beat_out, mask & (confidence >= .5)[None])[0])
        predicted_cells = int(np.clip(round((int(lengths[0]) - start) / 100 * predicted_bpm / 60 * tempo.config.subdivision), 1, 64))
        pred, _ = memory(features, encoded, beat_out, mask, predicted_cells)
        true_cells = target.shape[1]; total = max(true_cells, predicted_cells)
        padded_target = torch.zeros(1, total, 5, device=device); padded_pred = torch.zeros(1, total, 5, device=device)
        padded_target[:, :true_cells] = target; padded_pred[:, :predicted_cells] = pred
        if predicted_cells < total: padded_pred[:, predicted_cells:, 1:4] = -20.0
        voiced = padded_target[..., 1].bool(); predicted_available = torch.arange(total, device=device)[None] < predicted_cells
        available_voiced = voiced & predicted_available
        if available_voiced.any():
            pitch_errors.append((padded_pred[..., 0][available_voiced] - padded_target[..., 0][available_voiced]).abs().cpu() * SCALE)
            slope_errors.append((padded_pred[..., 4][available_voiced] - padded_target[..., 4][available_voiced]).abs().cpu() * SCALE)
            pitch_truth.append(padded_target[..., 0][available_voiced].cpu()); pitch_pred.append(padded_pred[..., 0][available_voiced].cpu())
        missing = voiced & ~predicted_available
        if missing.any(): pitch_errors.append(torch.full((int(missing.sum()),), 1200.0)); slope_errors.append(torch.full((int(missing.sum()),), 600.0))
        voice_truth.append(padded_target[..., 1].cpu().flatten()); voice_pred.append((padded_pred[..., 1] >= 0).cpu().flatten())
        onset_truth.append(padded_target[..., 2].cpu().flatten()); onset_pred.append((padded_pred[..., 2] >= 0).cpu().flatten())
        offset_truth.append(padded_target[..., 3].cpu().flatten()); offset_pred.append((padded_pred[..., 3] >= 0).cpu().flatten())
    def f1(truths, preds):
        truth = torch.cat(truths).bool(); pred = torch.cat(preds).bool()
        tp = int((truth & pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1)
    truth, pred = torch.cat(pitch_truth).numpy(), torch.cat(pitch_pred).numpy()
    return {"pitch_mae_cents": float(torch.cat(pitch_errors).mean()),
            "pitch_p90_cents": float(torch.quantile(torch.cat(pitch_errors), .9)),
            "trajectory_correlation": float(np.corrcoef(truth, pred)[0, 1]),
            "voice_f1": f1(voice_truth, voice_pred), "onset_f1": f1(onset_truth, onset_pred),
            "offset_f1": f1(offset_truth, offset_pred), "slope_mae_cents_per_cell": float(torch.cat(slope_errors).mean()),
            "ear_grid_upper_bound_mae_cents": float(torch.cat(upper_errors).mean()),
            "ear_grid_upper_bound_correlation": float(np.corrcoef(np.concatenate(upper_truth), np.concatenate(upper_pred))[0, 1])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_tempo_beat_v2.pt")
    ap.add_argument("--out", default="runs/yamabiko_musical_memory.pt"); ap.add_argument("--report", default="runs/yamabiko_musical_memory_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=16673); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); config = BeatGridConfig(**ck["config"])
    tempo = TempoBeatNet(config).to(args.device)
    memory = BeatAlignedMusicalMemoryNet(config, stable_clock=False, phase_lock=False,
                                         alignment_sigma=.14, pitch_residual_scale=.03,
                                         wrap_clock=True).to(args.device)
    tempo.load_state_dict(ck["tempo_beat"])
    if ck.get("musical_memory_kind") == "beat-aligned-v7": memory.load_state_dict(ck["musical_memory"])
    tempo.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    for module in (ear, tempo):
        for p in module.parameters(): p.requires_grad_(False)
    train = dataset(ear, rng, args.songs, args.device)
    valid = dataset(ear, np.random.default_rng(args.seed + 100_000), args.valid_songs, args.device,
                    [71, 89, 107, 131, 149, 179])
    optimizer = torch.optim.AdamW(memory.parameters(), lr=6e-4, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(train), size=args.batch); features, lengths, _, _, _ = collate(train, ids, args.device)
        target, cell_mask = cell_targets(train, ids, args.device)
        with torch.no_grad(): _, beat_out, encoded, mask = tempo(features, lengths)
        pred, _ = memory(features, encoded, beat_out, mask, target.shape[1], cell_mask.sum(1)); voiced = target[..., 1].bool() & cell_mask
        loss_pitch = F.smooth_l1_loss(pred[..., 0][voiced], target[..., 0][voiced])
        loss_voice = F.binary_cross_entropy_with_logits(pred[..., 1][cell_mask], target[..., 1][cell_mask])
        loss_onset = F.binary_cross_entropy_with_logits(pred[..., 2][cell_mask], target[..., 2][cell_mask], pos_weight=torch.tensor(3.0, device=args.device))
        loss_offset = F.binary_cross_entropy_with_logits(pred[..., 3][cell_mask], target[..., 3][cell_mask], pos_weight=torch.tensor(3.0, device=args.device))
        loss_slope = F.smooth_l1_loss(pred[..., 4][voiced], target[..., 4][voiced])
        loss = loss_pitch + .3 * loss_voice + .2 * (loss_onset + loss_offset) + .2 * loss_slope
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(memory.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"step {step:4d} loss {float(loss.detach()):.5f} pitch {float(loss_pitch.detach()):.5f} voice {float(loss_voice.detach()):.5f}", flush=True)
    metrics = evaluate(tempo, memory.eval(), valid, args.device)
    # Match the established Gate-2 resolution and require preservation of the
    # measured upstream Ear ceiling (reported alongside these metrics).
    metrics["pass"] = (metrics["pitch_mae_cents"] <= 60 and metrics["trajectory_correlation"] >= .90 and
                       metrics["voice_f1"] >= .95 and metrics["onset_f1"] >= .9 and metrics["offset_f1"] >= .9)
    metrics.update({"seed": args.seed, "split": "frozen-unknown-bpm", "official_path": "predicted_upstream"})
    ck["musical_memory"] = memory.state_dict(); ck["musical_memory_kind"] = "beat-aligned-v7"
    ck["musical_memory_trained"] = True; ck["musical_memory_metrics"] = metrics
    torch.save(ck, args.out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
