"""Fine-tune Clock/Aligner on frozen, predicted upstream memory tensors."""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flute_rl.yamabiko.beat_grid import (BeatAlignedMusicalMemoryNet, BeatGridConfig,
                                         DurationConditionedPerformanceClock,
                                         NeuralTemporalAligner, TempoBeatNet)  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from train_yamabiko_tempo_beat import collate, dataset  # noqa: E402
from train_yamabiko_temporal_aligner import frame_targets  # noqa: E402


@torch.inference_mode()
def predicted_records(ck, cfg, count, seed, device):
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=device), device).eval()
    tempo = TempoBeatNet(cfg).to(device); tempo.load_state_dict(ck["tempo_beat"]); tempo.eval()
    memory = BeatAlignedMusicalMemoryNet(cfg, stable_clock=False, phase_lock=False,
                                         alignment_sigma=.14, pitch_residual_scale=.03,
                                         wrap_clock=True).to(device)
    memory.load_state_dict(ck["musical_memory"]); memory.eval()
    samples = dataset(ear, np.random.default_rng(seed), count, device)
    records = []
    for index, sample in enumerate(samples):
        features, lengths, _, _, _ = collate(samples, [index], device)
        _, beat_out, encoded, mask = tempo(features, lengths)
        confidence = torch.sigmoid(beat_out[..., 2]); valid = mask & (confidence >= .5)
        ids = torch.nonzero(valid[0]).flatten(); start = int(ids[0]) if len(ids) else 0
        bpm = tempo.phase_bpm(beat_out, valid)
        cells = int(np.clip(round((int(lengths[0]) - start) / 100 * float(bpm[0]) / 60 * cfg.subdivision), 1, 64))
        grid, _ = memory(features, encoded, beat_out, mask, cells)
        records.append((grid[0].cpu(), float(tempo.normalized_bpm(bpm)[0]), sample.beat))
    return records


def batch(records, ids, device):
    chosen = [records[int(i)] for i in ids]; lengths = torch.tensor([len(x[0]) for x in chosen], device=device)
    grid = torch.zeros(len(chosen), int(lengths.max()), 5, device=device)
    for row, item in enumerate(chosen): grid[row, :len(item[0])] = item[0].to(device)
    normalized = torch.tensor([x[1] for x in chosen], device=device)
    return grid, lengths, normalized, [x[2] for x in chosen]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_temporal_duration_final.pt")
    ap.add_argument("--out", default="runs/yamabiko_temporal_connected_tuned.pt")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--batch", type=int, default=12); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20473)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    ck = torch.load(args.init, map_location=args.device); cfg = BeatGridConfig(**ck["config"])
    records = predicted_records(ck, cfg, args.songs, args.seed, args.device)
    clock = DurationConditionedPerformanceClock(cfg).to(args.device); clock.load_state_dict(ck["performance_clock"])
    aligner = NeuralTemporalAligner(cfg).to(args.device); aligner.load_state_dict(ck["temporal_aligner"])
    parameters = list(clock.parameters()) + list(aligner.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        grid, lengths, normalized, items = batch(records, rng.integers(len(records), size=args.batch), args.device)
        frames = max(len(x.target) for x in items) + 100
        target, mask, true_pointer, done = frame_targets(items, frames, args.device)
        reference_steps = torch.tensor([len(x.target) for x in items], device=args.device)
        pointer, done_logit = clock(normalized, lengths, reference_steps, frames)
        pred, _ = aligner(grid, lengths, pointer); voiced = mask & target[..., 1].bool(); rests = mask & ~target[..., 1].bool()
        loss_clock = F.smooth_l1_loss(pointer[mask], true_pointer[mask])
        positives = done.sum(); negatives = done.numel() - positives
        loss_done = F.binary_cross_entropy_with_logits(done_logit, done, pos_weight=(negatives / positives.clamp_min(1)).detach())
        loss_pitch = F.smooth_l1_loss(pred[..., 0][voiced], target[..., 0][voiced])
        loss_voice = F.binary_cross_entropy_with_logits(pred[..., 1][mask], target[..., 1][mask])
        loss_rest = F.binary_cross_entropy_with_logits(pred[..., 1][rests], target[..., 1][rests])
        kernel = torch.tensor([.12, .45, 1., .45, .12], device=args.device).view(1, 1, 5).repeat(2, 1, 1)
        events = F.conv1d(target[..., 2:4].transpose(1, 2), kernel, padding=2, groups=2).clamp_max(1).transpose(1, 2)
        loss_on = F.binary_cross_entropy_with_logits(pred[..., 2][mask], events[..., 0][mask], pos_weight=torch.tensor(5., device=args.device))
        loss_off = F.binary_cross_entropy_with_logits(pred[..., 3][mask], events[..., 1][mask], pos_weight=torch.tensor(5., device=args.device))
        loss = 1.5 * loss_clock + .6 * loss_done + loss_pitch + 1.2 * loss_voice + .8 * loss_rest + .6 * (loss_on + loss_off)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1); optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:4d} loss {float(loss.detach()):.5f} clock {float(loss_clock.detach()):.5f} pitch {float(loss_pitch.detach()):.5f}", flush=True)
    ck["performance_clock"] = clock.state_dict(); ck["temporal_aligner"] = aligner.state_dict()
    ck["temporal_connected_training"] = {"seed": args.seed, "songs": args.songs, "steps": args.steps,
                                          "upstream": "frozen-predicted"}
    ck.pop("temporal_connected_metrics", None); torch.save(ck, args.out); print(f"saved {args.out}")


if __name__ == "__main__": main()
