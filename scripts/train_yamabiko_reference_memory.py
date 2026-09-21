"""Train Gate 2 reference memory on the causal-aligned Gate 1 neural ear."""
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

from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_stages import E2EStageProbes  # noqa: E402
from train_yamabiko_e2e_stages import PITCH_CENTER, PITCH_SCALE, examples  # noqa: E402


@torch.inference_mode()
def encode_audio(model, raw, device):
    result = []
    for reference, own, pitch, voiced, error_valid, error, song_id in raw:
        ref = model.audio_features(torch.from_numpy(reference).to(device)).cpu()
        own_feature = model.audio_features(torch.from_numpy(own).to(device)).cpu()
        result.append((ref, own_feature, torch.from_numpy((pitch - PITCH_CENTER) / PITCH_SCALE),
                       torch.from_numpy(voiced), torch.from_numpy(error_valid),
                       torch.from_numpy(error / PITCH_SCALE), song_id))
    return result


def batch_sequences(data, indices, device):
    chosen = [data[int(i)] for i in indices]
    lengths = torch.tensor([len(x[2]) for x in chosen], device=device)
    steps, feature = int(lengths.max()), chosen[0][0].shape[-1]
    ref = torch.zeros(len(chosen), steps, feature, device=device)
    own = torch.zeros_like(ref)
    pitch = torch.zeros(len(chosen), steps, device=device)
    voiced = torch.zeros(len(chosen), steps, dtype=torch.bool, device=device)
    error_valid = torch.zeros_like(voiced)
    error = torch.zeros_like(pitch)
    for row, item in enumerate(chosen):
        n = len(item[2])
        ref[row, :n], own[row, :n] = item[0].to(device), item[1].to(device)
        pitch[row, :n], voiced[row, :n] = item[2].to(device), item[3].to(device)
        error_valid[row, :n], error[row, :n] = item[4].to(device), item[5].to(device)
    mask = torch.arange(steps, device=device)[None] < lengths[:, None]
    return ref, own, pitch, voiced, error_valid, error, lengths, mask


def memory(model, ref, lengths):
    packed = torch.nn.utils.rnn.pack_padded_sequence(ref, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_out, _ = model.reference(packed)
    recurrent, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True,
                                                           total_length=ref.shape[1])
    return torch.cat([recurrent, ref], dim=-1)


@torch.inference_mode()
def evaluate(model, probes, data, device, batch_size=16, direct=False):
    predicted, own_all, pitch_all, voiced_all, valid_all, error_all, seq_all = [], [], [], [], [], [], []
    for start in range(0, len(data), batch_size):
        ids = np.arange(start, min(len(data), start + batch_size))
        ref, own, pitch, voiced, valid, error, lengths, mask = batch_sequences(data, ids, device)
        pred = ref[..., -2:] if direct else probes.reference(memory(model, ref, lengths))
        predicted.append(pred[mask].cpu()); own_all.append(own[..., -2][mask].cpu())
        pitch_all.append(pitch[mask].cpu()); voiced_all.append(voiced[mask].cpu())
        valid_all.append(valid[mask].cpu()); error_all.append(error[mask].cpu())
        seq = torch.arange(start, start + len(ids), device=device)[:, None].expand_as(mask)
        seq_all.append(seq[mask].cpu())
    pred, own = torch.cat(predicted), torch.cat(own_all)
    pitch, voiced = torch.cat(pitch_all), torch.cat(voiced_all)
    valid, true_error, seq = torch.cat(valid_all), torch.cat(error_all), torch.cat(seq_all)
    ref_mae = float((pred[:, 0][voiced] - pitch[voiced]).abs().mean() * PITCH_SCALE)
    pv = pred[:, 1] >= 0; tp = int((pv & voiced).sum()); fp = int((pv & ~voiced).sum()); fn = int((~pv & voiced).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    # A 20 ms acoustic window cannot localise a discontinuity to one exact
    # 10 ms frame.  Judge direction across a 30 ms neighbourhood; Gate 3
    # measures the remaining transition latency separately.
    same = (seq[3:] == seq[:-3])
    dt = (pitch[2:-1] - pitch[1:-2]) * PITCH_SCALE
    dp = (pred[3:, 0] - pred[:-3, 0]) * PITCH_SCALE
    changed = same & voiced[2:-1] & voiced[1:-2] & (dt.abs() >= 25)
    direction = float((torch.sign(dt[changed]) == torch.sign(dp[changed])).float().mean())
    estimated_error = pred[:, 0] - own
    error_mae = float((estimated_error[valid] - true_error[valid]).abs().mean() * PITCH_SCALE)
    sign = valid & (true_error.abs() * PITCH_SCALE >= 25)
    sign_acc = float((torch.sign(estimated_error[sign]) == torch.sign(true_error[sign])).float().mean())
    own_mae = float((own[valid] - (pitch - true_error)[valid]).abs().mean() * PITCH_SCALE)
    return {"reference_mae_cents": ref_mae, "voice_f1": f1, "change_direction_accuracy": direction,
            "self_pitch_mae_cents": own_mae, "error_mae_cents": error_mae,
            "error_sign_accuracy": sign_acc,
            "gate2_pass": ref_mae < 60 and f1 >= .95 and direction + 1e-6 >= .90,
            "gate4_pass": error_mae < 60 and sign_acc >= .90}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ear", default="runs/yamabiko_e2e_pc_ear_aligned.pt")
    ap.add_argument("--out", default="runs/yamabiko_e2e_pc_memory.pt")
    ap.add_argument("--probes", default="runs/yamabiko_e2e_memory_probes.pt")
    ap.add_argument("--probes-init", default=None, help="load trained probes for evaluation/continuation")
    ap.add_argument("--report", default="runs/yamabiko_e2e_memory_report.json")
    ap.add_argument("--songs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=6060)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    model = E2EImitator.from_checkpoint(torch.load(args.ear, map_location=args.device), args.device)
    raw = examples(rng, args.songs, .8, 1.0); split = int(.8 * len(raw))
    train = encode_audio(model.eval(), raw[:split], args.device); validation = encode_audio(model.eval(), raw[split:], args.device)
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.reference.parameters(): p.requires_grad_(True)
    probes = E2EStageProbes(model.config).to(args.device)
    if args.probes_init:
        probes.load_state_dict(torch.load(args.probes_init, map_location=args.device)["state"])
    for p in probes.parameters(): p.requires_grad_(False)
    for p in probes.reference.parameters(): p.requires_grad_(True)
    model.reference.train(); probes.reference.train()
    optimizer = torch.optim.AdamW(list(model.reference.parameters()) + list(probes.reference.parameters()), lr=1e-3)
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(train), size=args.batch)
        ref, _, pitch, voiced, _, _, lengths, mask = batch_sequences(train, ids, args.device)
        pred = probes.reference(memory(model, ref, lengths))
        delta_valid = ((voiced[:, 1:] & voiced[:, :-1] & mask[:, 1:] & mask[:, :-1])
                       & ((pitch[:, 1:] - pitch[:, :-1]).abs() * PITCH_SCALE >= 25.0))
        predicted_delta = pred[:, 1:, 0] - pred[:, :-1, 0]
        target_delta = pitch[:, 1:] - pitch[:, :-1]
        loss = (F.smooth_l1_loss(pred[..., 0][voiced & mask], pitch[voiced & mask])
                + .25 * F.binary_cross_entropy_with_logits(pred[..., 1][mask], voiced[mask].float())
                + F.smooth_l1_loss(predicted_delta[delta_valid], target_delta[delta_valid]))
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(
            list(model.reference.parameters()) + list(probes.reference.parameters()), 1.0); optimizer.step()
        if step == 1 or step % 100 == 0: print(f"step {step:4d} loss {float(loss.detach()):.4f}", flush=True)
    metrics = evaluate(model.eval(), probes.eval(), validation, args.device,
                       direct=args.steps == 0 and not args.probes_init)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); torch.save(model.checkpoint(), out)
    torch.save({"format": "yamabiko-e2e-memory-probes-v1", "state": probes.state_dict(),
                "config": vars(model.config), "metrics": metrics}, args.probes)
    pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {out}")


if __name__ == "__main__": main()
