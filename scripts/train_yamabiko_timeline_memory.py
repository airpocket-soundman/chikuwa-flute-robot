"""Train beat-conditioned 100 Hz Timeline Memory for drift-free repeat plays."""
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
from train_yamabiko_tempo_beat import collate, dataset  # noqa: E402
from train_yamabiko_temporal_aligner import f1, frame_targets, tolerant_event_f1  # noqa: E402

SCALE = 600.0


@torch.inference_mode()
def evaluate(tempo, timeline, samples, device):
    pitch_e = []; truth_p = []; pred_p = []; voice_t = []; voice_p = []; rest_p = []; onset = []; offset = []
    source_pitch = {}
    for start in range(0, len(samples), 12):
        ids = list(range(start, min(start + 12, len(samples))))
        features, lengths, _, _, _ = collate(samples, ids, device)
        _, beat_out, encoded, mask = tempo(features, lengths)
        _, stored = timeline(features, encoded, beat_out, lengths)
        pred = timeline.decode(stored)  # raw audio/features are no longer used
        target, frame_mask, _, _ = frame_targets([samples[i].beat for i in ids], pred.shape[1], device)
        voiced = frame_mask & target[..., 1].bool(); rests = frame_mask & ~target[..., 1].bool()
        pitch_e.append((pred[..., 0][voiced] - target[..., 0][voiced]).abs().cpu() * SCALE)
        truth_p.append(target[..., 0][voiced].cpu()); pred_p.append(pred[..., 0][voiced].cpu())
        voice_t.append(target[..., 1][frame_mask].cpu()); voice_p.append((pred[..., 1][frame_mask] >= 0).cpu())
        rest_p.append((pred[..., 1][rests] >= 0).cpu())
        onset.append(tolerant_event_f1(target[..., 2], pred[..., 2], frame_mask))
        offset.append(tolerant_event_f1(target[..., 3], pred[..., 3], frame_mask))
        for row, sample_id in enumerate(ids):
            v = voiced[row]; kind = samples[sample_id].source_kind
            bucket = source_pitch.setdefault(kind, [[], [], []])
            bucket[0].append((pred[row, :, 0][v] - target[row, :, 0][v]).abs().cpu() * SCALE)
            bucket[1].append(target[row, :, 0][v].cpu()); bucket[2].append(pred[row, :, 0][v].cpu())
    truth, pred = torch.cat(truth_p).numpy(), torch.cat(pred_p).numpy()
    result = {"pitch_mae_cents": float(torch.cat(pitch_e).mean()),
              "trajectory_correlation": float(np.corrcoef(truth, pred)[0, 1]),
              "voice_f1": f1(torch.cat(voice_t), torch.cat(voice_p)),
              "rest_false_positive_rate": float(torch.cat(rest_p).float().mean()),
              "onset_f1_30ms": float(np.mean(onset)), "offset_f1_30ms": float(np.mean(offset))}
    result["by_source"] = {kind: {"pitch_mae_cents": float(torch.cat(values[0]).mean()),
                                  "trajectory_correlation": float(np.corrcoef(torch.cat(values[1]), torch.cat(values[2]))[0, 1])}
                           for kind, values in source_pitch.items()}
    result["pass"] = (result["pitch_mae_cents"] <= 60 and result["trajectory_correlation"] >= .90 and
                      result["voice_f1"] >= .95 and result["rest_false_positive_rate"] <= .03 and
                      result["onset_f1_30ms"] >= .85)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_timing_profile_direct_v1.pt")
    ap.add_argument("--out", default="runs/yamabiko_timeline_memory.pt"); ap.add_argument("--report", default="runs/yamabiko_timeline_memory_report.json")
    ap.add_argument("--songs", type=int, default=384); ap.add_argument("--valid-songs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--direct-pitch", action="store_true")
    ap.add_argument("--seed", type=int, default=31817); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-seed", type=int, default=131817); ap.add_argument("--split", default="development-unknown-bpm")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); cfg = BeatGridConfig(**ck["config"])
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    tempo = TempoBeatNet(cfg).to(args.device); tempo.load_state_dict(ck["tempo_beat"]); tempo.eval()
    for module in (ear, tempo):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    train = dataset(ear, rng, args.songs, args.device) if args.steps else []
    bpms = [71, 91, 109, 133, 157, 179]
    valid = dataset(ear, np.random.default_rng(args.eval_seed), args.valid_songs, args.device, bpms)
    timeline = BeatTimelineRecallNet(cfg, direct_pitch=args.direct_pitch).to(args.device)
    desired_kind = "direct-pitch-v1" if args.direct_pitch else "residual-pitch-v1"
    if ck.get("timeline_memory_trained") and ck.get("timeline_memory_kind", "residual-pitch-v1") == desired_kind:
        timeline.load_state_dict(ck["timeline_memory"])
    optimizer = torch.optim.AdamW(timeline.parameters(), lr=args.lr, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(train), size=args.batch).tolist(); features, lengths, _, _, _ = collate(train, ids, args.device)
        with torch.no_grad(): _, beat_out, encoded, mask = tempo(features, lengths)
        pred, _ = timeline(features, encoded, beat_out, lengths)
        target, frame_mask, _, _ = frame_targets([train[i].beat for i in ids], pred.shape[1], args.device)
        voiced = frame_mask & target[..., 1].bool(); rests = frame_mask & ~target[..., 1].bool()
        loss_pitch = F.smooth_l1_loss(pred[..., 0][voiced], target[..., 0][voiced])
        pair = voiced[:, 1:] & voiced[:, :-1]
        loss_trajectory = F.smooth_l1_loss((pred[:, 1:, 0] - pred[:, :-1, 0])[pair],
                                           (target[:, 1:, 0] - target[:, :-1, 0])[pair])
        px, tx = pred[..., 0][voiced], target[..., 0][voiced]
        loss_correlation = 1 - F.cosine_similarity(px - px.mean(), tx - tx.mean(), dim=0)
        loss_voice = F.binary_cross_entropy_with_logits(pred[..., 1][frame_mask], target[..., 1][frame_mask])
        loss_rest = F.binary_cross_entropy_with_logits(pred[..., 1][rests], target[..., 1][rests])
        kernel = torch.tensor([.12, .45, 1., .45, .12], device=args.device).view(1, 1, 5).repeat(2, 1, 1)
        events = F.conv1d(target[..., 2:4].transpose(1, 2), kernel, padding=2, groups=2).clamp_max(1).transpose(1, 2)
        loss_on = F.binary_cross_entropy_with_logits(pred[..., 2][frame_mask], events[..., 0][frame_mask], pos_weight=torch.tensor(5., device=args.device))
        loss_off = F.binary_cross_entropy_with_logits(pred[..., 3][frame_mask], events[..., 1][frame_mask], pos_weight=torch.tensor(5., device=args.device))
        loss = 2 * loss_pitch + .5 * loss_trajectory + .5 * loss_correlation + .7 * loss_voice + .4 * loss_rest + .3 * (loss_on + loss_off)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(timeline.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"step {step:4d} loss {float(loss.detach()):.5f} pitch {float(loss_pitch.detach()):.5f}", flush=True)
    metrics = evaluate(tempo, timeline.eval(), valid, args.device)
    metrics.update({"seed": args.eval_seed, "split": args.split, "official_path": "stored_timeline_only"})
    ck["timeline_memory"] = timeline.state_dict(); ck["timeline_memory_trained"] = True
    ck["timeline_memory_kind"] = desired_kind; ck["timeline_memory_metrics"] = metrics
    torch.save(ck, args.out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
