"""Frozen connected Gate: raw reference audio through memory and 100 Hz aligner."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flute_rl.yamabiko.beat_grid import (BeatAlignedMusicalMemoryNet, BeatGridConfig,
                                         NeuralPerformanceClock, NeuralTemporalAligner,
                                         TempoBeatNet)  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from train_yamabiko_tempo_beat import collate, dataset  # noqa: E402
from train_yamabiko_temporal_aligner import frame_targets, f1, tolerant_event_f1  # noqa: E402

SCALE = 600.0


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/yamabiko_temporal_aligner_final.pt")
    ap.add_argument("--out", default="runs/yamabiko_temporal_connected_report.json")
    ap.add_argument("--checkpoint-out", default="runs/yamabiko_temporal_connected.pt")
    ap.add_argument("--seed", type=int, default=938431); ap.add_argument("--songs", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); ck = torch.load(args.model, map_location=args.device); cfg = BeatGridConfig(**ck["config"])
    tempo = TempoBeatNet(cfg).to(args.device); tempo.load_state_dict(ck["tempo_beat"]); tempo.eval()
    memory = BeatAlignedMusicalMemoryNet(cfg, stable_clock=False, phase_lock=False,
                                         alignment_sigma=.14, pitch_residual_scale=.03,
                                         wrap_clock=True).to(args.device)
    memory.load_state_dict(ck["musical_memory"]); memory.eval()
    clock = NeuralPerformanceClock(cfg).to(args.device); clock.load_state_dict(ck["performance_clock"]); clock.eval()
    aligner = NeuralTemporalAligner(cfg).to(args.device); aligner.load_state_dict(ck["temporal_aligner"]); aligner.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    bpms = [67, 83, 101, 127, 151, 181]
    samples = dataset(ear, np.random.default_rng(args.seed), args.songs, args.device, bpms)
    pitch_errors = []; truth_pitch = []; pred_pitch = []; voice_t = []; voice_p = []; rest_p = []
    pointer_errors = []; eos_errors = []; onset_scores = []; offset_scores = []; backwards = 0
    for index, sample in enumerate(samples):
        features, lengths, _, _, _ = collate(samples, [index], args.device)
        _, beat_out, encoded, mask = tempo(features, lengths)
        confidence = torch.sigmoid(beat_out[..., 2]); valid = mask & (confidence >= .5)
        confident = torch.nonzero(valid[0]).flatten(); start = int(confident[0]) if len(confident) else 0
        bpm = tempo.phase_bpm(beat_out, valid)
        cells = int(np.clip(round((int(lengths[0]) - start) / 100 * float(bpm[0]) / 60 * cfg.subdivision), 1, 64))
        grid, _ = memory(features, encoded, beat_out, mask, cells)
        cell_lengths = torch.tensor([cells], device=args.device); steps = len(sample.beat.target) + 100
        pointer, done_logit = clock(tempo.normalized_bpm(bpm), cell_lengths, steps)
        pred, _ = aligner(grid, cell_lengths, pointer)
        target, frame_mask, true_pointer, _ = frame_targets([sample.beat], steps, args.device)
        voiced = frame_mask & target[..., 1].bool(); rests = frame_mask & ~target[..., 1].bool()
        pitch_errors.append((pred[..., 0][voiced] - target[..., 0][voiced]).abs().cpu() * SCALE)
        truth_pitch.append(target[..., 0][voiced].cpu()); pred_pitch.append(pred[..., 0][voiced].cpu())
        voice_t.append(target[..., 1][frame_mask].cpu()); voice_p.append((pred[..., 1][frame_mask] >= 0).cpu())
        rest_p.append((pred[..., 1][rests] >= 0).cpu())
        pointer_errors.append((pointer[frame_mask] - true_pointer[frame_mask]).abs().cpu())
        backwards += int((pointer[:, 1:] < pointer[:, :-1]).sum())
        ids = torch.nonzero(done_logit[0] >= 0).flatten(); end = int(ids[0]) if len(ids) else steps
        eos_errors.append(abs(end - len(sample.beat.target)) * 10)
        onset_scores.append(tolerant_event_f1(target[..., 2], pred[..., 2], frame_mask))
        offset_scores.append(tolerant_event_f1(target[..., 3], pred[..., 3], frame_mask))
    truth = torch.cat(truth_pitch).numpy(); predicted = torch.cat(pred_pitch).numpy()
    metrics = {
        "clock_pointer_mae_cells": float(torch.cat(pointer_errors).mean()),
        "clock_backward_jumps": backwards, "clock_eos_mae_ms": float(np.mean(eos_errors)),
        "pitch_mae_cents": float(torch.cat(pitch_errors).mean()),
        "trajectory_correlation": float(np.corrcoef(truth, predicted)[0, 1]),
        "voice_f1": f1(torch.cat(voice_t), torch.cat(voice_p)),
        "rest_false_positive_rate": float(torch.cat(rest_p).float().mean()),
        "onset_f1_30ms": float(np.mean(onset_scores)), "offset_f1_30ms": float(np.mean(offset_scores)),
    }
    metrics["pass"] = (metrics["clock_backward_jumps"] == 0 and metrics["clock_eos_mae_ms"] <= 200 and
                       metrics["pitch_mae_cents"] <= 75 and metrics["trajectory_correlation"] >= .88 and
                       metrics["voice_f1"] >= .93 and metrics["rest_false_positive_rate"] <= .08 and
                       metrics["onset_f1_30ms"] >= .80)
    metrics.update({"seed": args.seed, "split": "frozen-final-predicted-upstream", "official_path": "raw_audio_to_100hz_target"})
    ck["temporal_connected_metrics"] = metrics
    torch.save(ck, args.checkpoint_out); pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.checkpoint_out}")


if __name__ == "__main__": main()
