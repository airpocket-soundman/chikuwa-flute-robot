"""Train and independently evaluate the neural 100 Hz performance clock/aligner."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.targets import BeatTarget, make_beat_target  # noqa: E402
from flute_rl.yamabiko.beat_grid import (BeatGridConfig, NeuralPerformanceClock,
                                         NeuralTemporalAligner, TempoBeatNet)  # noqa: E402

CENTER, SCALE, RATE = 1300.0, 600.0, 100.0


def grid_tensor(items: list[BeatTarget], device):
    lengths = torch.tensor([len(x.cell_pitch) for x in items], device=device)
    grid = torch.zeros(len(items), int(lengths.max()), 5, device=device)
    for row, x in enumerate(items):
        n = len(x.cell_pitch); voice = torch.from_numpy(x.cell_voice).to(device)
        grid[row, :n, 0] = torch.from_numpy((x.cell_pitch - CENTER) / SCALE).to(device)
        grid[row, :n, 1] = torch.where(voice, 6.0, -6.0)
        grid[row, :n, 2] = torch.where(torch.from_numpy(x.cell_onset).to(device), 6.0, -6.0)
        grid[row, :n, 3] = torch.where(torch.from_numpy(x.cell_offset).to(device), 6.0, -6.0)
        grid[row, :n, 4] = torch.from_numpy(x.cell_slope / SCALE).to(device)
    return grid, lengths


def frame_targets(items: list[BeatTarget], steps: int, device):
    out = torch.zeros(len(items), steps, 4, device=device); mask = torch.zeros(len(items), steps, dtype=torch.bool, device=device)
    pointer = torch.zeros(len(items), steps, device=device); done = torch.zeros(len(items), steps, device=device)
    for row, x in enumerate(items):
        n = min(len(x.target), steps); mask[row, :n] = True
        time = np.arange(steps) / RATE
        p = (time - x.beat0_s) * x.bpm / 60.0 * x.subdivision; pointer[row] = torch.from_numpy(p.astype(np.float32)).to(device)
        done[row] = torch.from_numpy((p >= len(x.cell_pitch)).astype(np.float32)).to(device)
        voice = np.isfinite(x.target[:n]); pitch = np.nan_to_num((x.target[:n] - CENTER) / SCALE)
        onset = voice & np.r_[True, ~voice[:-1]]; offset = voice & np.r_[~voice[1:], True]
        # Pitch changes without a rest are attacks too.
        onset[1:] |= voice[1:] & voice[:-1] & (np.abs(x.target[1:n] - x.target[:n - 1]) > 20)
        out[row, :n, 0] = torch.from_numpy(pitch.astype(np.float32)).to(device)
        out[row, :n, 1] = torch.from_numpy(voice.astype(np.float32)).to(device)
        out[row, :n, 2] = torch.from_numpy(onset.astype(np.float32)).to(device)
        out[row, :n, 3] = torch.from_numpy(offset.astype(np.float32)).to(device)
    return out, mask, pointer, done


def f1(truth, pred):
    truth, pred = truth.bool(), pred.bool(); tp = (truth & pred).sum(); fp = (~truth & pred).sum(); fn = (truth & ~pred).sum()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


def tolerant_event_f1(truth: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor, radius: int = 3):
    """One-to-one event matching within +/-30 ms after local-peak decoding."""
    tp = fp = fn = 0
    score = torch.sigmoid(logits)
    for row in range(len(truth)):
        n = int(mask[row].sum()); y = score[row, :n]
        peaks = torch.nonzero((y >= .5) & (y >= torch.roll(y, 1)) & (y > torch.roll(y, -1))).flatten().tolist()
        actual = torch.nonzero(truth[row, :n] >= .5).flatten().tolist(); used = set()
        for p in peaks:
            choices = [(abs(p - a), j) for j, a in enumerate(actual) if j not in used and abs(p - a) <= radius]
            if choices: used.add(min(choices)[1]); tp += 1
            else: fp += 1
        fn += len(actual) - len(used)
    return 2 * tp / max(2 * tp + fp + fn, 1)


@torch.inference_mode()
def evaluate(config, clock, aligner, items, device):
    grid, lengths = grid_tensor(items, device); steps = max(len(x.target) for x in items) + 100
    target, mask, true_pointer, done = frame_targets(items, steps, device)
    bpm = torch.tensor([x.bpm for x in items], device=device)
    normalized = torch.log(bpm / config.bpm_center) / config.bpm_log_scale
    pointer, done_logit = clock(normalized, lengths, steps); pred, _ = aligner(grid, lengths, pointer)
    voice = mask & target[..., 1].bool(); estimated_voice = pred[..., 1] >= 0
    pointer_error = (pointer[mask] - true_pointer[mask]).abs()
    eos_errors = []
    for row, x in enumerate(items):
        ids = torch.nonzero(done_logit[row] >= 0).flatten(); predicted = int(ids[0]) if len(ids) else steps
        eos_errors.append(abs(predicted - len(x.target)) / RATE)
    corr = np.corrcoef(target[..., 0][voice].cpu(), pred[..., 0][voice].cpu())[0, 1]
    result = {
        "clock_pointer_mae_cells": float(pointer_error.mean()),
        "clock_pointer_p95_cells": float(torch.quantile(pointer_error, .95)),
        "clock_backward_jumps": int((pointer[:, 1:] < pointer[:, :-1]).sum()),
        "clock_eos_mae_ms": float(np.mean(eos_errors) * 1000),
        "pitch_mae_cents": float((pred[..., 0][voice] - target[..., 0][voice]).abs().mean() * SCALE),
        "trajectory_correlation": float(corr),
        "voice_f1": f1(target[..., 1][mask], estimated_voice[mask]),
        "rest_false_positive_rate": float(estimated_voice[mask & ~target[..., 1].bool()].float().mean()),
        "onset_f1_exact_frame": f1(target[..., 2][mask], pred[..., 2][mask] >= 0),
        "offset_f1_exact_frame": f1(target[..., 3][mask], pred[..., 3][mask] >= 0),
        "onset_f1_30ms": tolerant_event_f1(target[..., 2], pred[..., 2], mask),
        "offset_f1_30ms": tolerant_event_f1(target[..., 3], pred[..., 3], mask),
    }
    result["pass"] = (result["clock_pointer_mae_cells"] <= .10 and result["clock_backward_jumps"] == 0 and
                      result["clock_eos_mae_ms"] <= 60 and result["pitch_mae_cents"] <= 35 and
                      result["trajectory_correlation"] >= .97 and result["voice_f1"] >= .95 and
                      result["rest_false_positive_rate"] <= .03 and result["onset_f1_30ms"] >= .85)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_musical_memory_aligned_v7_pass.pt")
    ap.add_argument("--out", default="runs/yamabiko_temporal_aligner.pt")
    ap.add_argument("--report", default="runs/yamabiko_temporal_aligner_report.json")
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=18431)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); config = BeatGridConfig(**ck["config"])
    clock = NeuralPerformanceClock(config).to(args.device); aligner = NeuralTemporalAligner(config).to(args.device)
    if ck.get("temporal_aligner_trained"):
        clock.load_state_dict(ck["performance_clock"]); aligner.load_state_dict(ck["temporal_aligner"])
    optimizer = torch.optim.AdamW(list(clock.parameters()) + list(aligner.parameters()), lr=8e-4, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        items = [make_beat_target(rng, beats=int(rng.integers(6, 13))) for _ in range(args.batch)]
        grid, lengths = grid_tensor(items, args.device); frames = max(len(x.target) for x in items) + 100
        target, mask, true_pointer, done = frame_targets(items, frames, args.device)
        bpm = torch.tensor([x.bpm for x in items], device=args.device)
        normalized = torch.log(bpm / config.bpm_center) / config.bpm_log_scale
        pointer, done_logit = clock(normalized, lengths, frames); pred, _ = aligner(grid, lengths, pointer)
        voiced = mask & target[..., 1].bool()
        loss_clock = F.smooth_l1_loss(pointer[mask], true_pointer[mask])
        positives = done.sum(); negatives = done.numel() - positives
        loss_done = F.binary_cross_entropy_with_logits(done_logit, done, pos_weight=(negatives / positives.clamp_min(1)).detach())
        loss_pitch = F.smooth_l1_loss(pred[..., 0][voiced], target[..., 0][voiced])
        loss_voice = F.binary_cross_entropy_with_logits(pred[..., 1][mask], target[..., 1][mask])
        loss_on = F.binary_cross_entropy_with_logits(pred[..., 2][mask], target[..., 2][mask], pos_weight=torch.tensor(15., device=args.device))
        loss_off = F.binary_cross_entropy_with_logits(pred[..., 3][mask], target[..., 3][mask], pos_weight=torch.tensor(15., device=args.device))
        loss = 2 * loss_clock + .6 * loss_done + loss_pitch + .5 * loss_voice + .2 * (loss_on + loss_off)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(clock.parameters()) + list(aligner.parameters()), 1); optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:4d} loss {float(loss):.5f} clock {float(loss_clock):.5f} pitch {float(loss_pitch):.5f}", flush=True)
    valid_rng = np.random.default_rng(args.seed + 100_000)
    valid_bpms = [71, 89, 107, 131, 149, 179]
    valid = [make_beat_target(valid_rng, bpm=float(valid_bpms[i % len(valid_bpms)]), beats=int(valid_rng.integers(6, 13))) for i in range(96)]
    metrics = evaluate(config, clock.eval(), aligner.eval(), valid, args.device)
    metrics.update({"seed": args.seed, "split": "frozen-oracle-grid-unknown-bpm", "official_path": "oracle_memory"})
    ck.update({"performance_clock": clock.state_dict(), "temporal_aligner": aligner.state_dict(),
               "temporal_aligner_trained": True, "temporal_aligner_metrics": metrics})
    torch.save(ck, args.out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
