"""Train the separated raw-reference -> BPM/beat-phase neural block."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.audio import SOURCE_KINDS, room, synth_source  # noqa: E402
from flute_rl.targets import BeatTarget, make_beat_target  # noqa: E402
from flute_rl.yamabiko.beat_grid import BeatGridConfig, MusicalMemoryNet, TempoBeatNet, checkpoint  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_io import SAMPLE_RATE, frame_audio_numpy  # noqa: E402


@dataclass
class Sample:
    features: torch.Tensor
    beat: BeatTarget


@torch.inference_mode()
def dataset(ear, rng, count, device, bpms=None):
    result = []
    for i in range(count):
        beat = make_beat_target(rng, bpm=None if bpms is None else float(bpms[i % len(bpms)]))
        kind = str(rng.choice(SOURCE_KINDS)); wave, _ = synth_source(beat.target, kind, rng, sr=SAMPLE_RATE)
        frames = frame_audio_numpy(room(wave, SAMPLE_RATE, rng), len(beat.target))
        features = ear.audio_features(torch.from_numpy(frames).to(device)).cpu()
        result.append(Sample(features, beat))
    return result


def collate(samples, ids, device):
    chosen = [samples[int(i)] for i in ids]; lengths = torch.tensor([len(x.beat.target) for x in chosen], device=device)
    steps, dim = int(lengths.max()), chosen[0].features.shape[-1]
    features = torch.zeros(len(chosen), steps, dim, device=device)
    phase = torch.zeros(len(chosen), steps, 2, device=device); valid = torch.zeros(len(chosen), steps, dtype=torch.bool, device=device)
    bpm = torch.tensor([x.beat.bpm for x in chosen], device=device)
    for row, item in enumerate(chosen):
        n = len(item.beat.target); features[row, :n] = item.features.to(device)
        angle = torch.from_numpy((2 * np.pi * item.beat.frame_phase).astype(np.float32)).to(device)
        phase[row, :n] = torch.stack([torch.sin(angle), torch.cos(angle)], 1)
        valid[row, :n] = torch.from_numpy(item.beat.phase_valid).to(device)
    return features, lengths, bpm, phase, valid


def circular_error(pred_xy, true_xy):
    pred = torch.atan2(pred_xy[..., 0], pred_xy[..., 1]) / (2 * torch.pi)
    truth = torch.atan2(true_xy[..., 0], true_xy[..., 1]) / (2 * torch.pi)
    return torch.remainder(pred - truth + .5, 1.0) - .5


@torch.inference_mode()
def evaluate(model, samples, device):
    bpm_errors, phase_errors, confidences = [], [], []
    for start in range(0, len(samples), 16):
        features, lengths, bpm, phase, valid = collate(samples, range(start, min(start + 16, len(samples))), device)
        bpm_log, out, _, _ = model(features, lengths); predicted_bpm = model.bpm(bpm_log)
        bpm_errors.append(((predicted_bpm - bpm).abs() / bpm).cpu())
        phase_errors.append(circular_error(out[..., :2][valid], phase[valid]).abs().cpu())
        confidences.append(torch.sigmoid(out[..., 2][valid]).cpu())
    bpm_error, phase_error = torch.cat(bpm_errors), torch.cat(phase_errors)
    return {"tempo_relative_error_median": float(torch.median(bpm_error)),
            "tempo_relative_error_p90": float(torch.quantile(bpm_error, .9)),
            "phase_circular_mae_cycle": float(torch.cat(phase_errors).mean()),
            "phase_confidence_mean": float(torch.cat(confidences).mean()),
            "half_double_confusion_rate": float(((bpm_error > .35) & (bpm_error < 1.1)).float().mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ear", default="runs/yamabiko_e2e_pc_ear_gate1_balanced.pt")
    ap.add_argument("--out", default="runs/yamabiko_tempo_beat.pt"); ap.add_argument("--report", default="runs/yamabiko_tempo_beat_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=14449); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ear = E2EImitator.from_checkpoint(torch.load(args.ear, map_location=args.device), args.device).eval()
    for p in ear.parameters(): p.requires_grad_(False)
    # Final-test BPMs are deliberately absent as exact values from the public
    # demo set; the seed and split are persisted in the report.
    train = dataset(ear, rng, args.songs, args.device)
    valid_bpms = [73, 97, 113, 137, 157, 173]
    valid = dataset(ear, np.random.default_rng(args.seed + 100_000), args.valid_songs, args.device, valid_bpms)
    config = BeatGridConfig(input_dim=ear.config.audio_dim + 2)
    tempo, memory = TempoBeatNet(config).to(args.device), MusicalMemoryNet(config).to(args.device)
    optimizer = torch.optim.AdamW(tempo.parameters(), lr=7e-4, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(train), size=args.batch); features, lengths, bpm, phase, valid_phase = collate(train, ids, args.device)
        bpm_log, out, _, mask = tempo(features, lengths)
        target_bpm = tempo.normalized_bpm(bpm)
        loss_bpm = F.smooth_l1_loss(bpm_log, target_bpm)
        loss_phase = F.mse_loss(out[..., :2][valid_phase], phase[valid_phase])
        loss_conf = F.binary_cross_entropy_with_logits(out[..., 2][mask], valid_phase[mask].float())
        loss = loss_bpm + loss_phase + .1 * loss_conf
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(tempo.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:4d} loss {float(loss.detach()):.5f} bpm {float(loss_bpm.detach()):.5f} phase {float(loss_phase.detach()):.5f}", flush=True)
    metrics = evaluate(tempo.eval(), valid, args.device)
    metrics["pass"] = (metrics["tempo_relative_error_median"] <= .03 and
                       metrics["phase_circular_mae_cycle"] <= .08 and
                       metrics["half_double_confusion_rate"] <= .05)
    metrics.update({"seed": args.seed, "split": "frozen-heldout-bpm", "heldout_bpms": valid_bpms,
                    "official_path": "predicted_upstream"})
    torch.save(checkpoint(config, tempo, memory, ear_checkpoint=args.ear, tempo_metrics=metrics,
                          musical_memory_trained=False), args.out)
    pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
