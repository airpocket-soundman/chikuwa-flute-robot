"""Train and evaluate Gate 2 (reference memory) and Gate 4 (error comparison).

The base E2E model is frozen.  Diagnostic probes learn from its latent
features, so passing means the information exists in the representation;
failing prevents a controller from hiding the defect behind aggregate MAE.
Pitch labels are training/evaluation-only and never become runtime inputs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.audio import SOURCE_KINDS, room, synth_source  # noqa: E402
from flute_rl.targets import make_target, sample_level  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_audio import RawSelfAudio  # noqa: E402
from flute_rl.yamabiko.e2e_io import FRAME, SAMPLE_RATE, frame_audio_numpy  # noqa: E402
from flute_rl.yamabiko.e2e_stages import E2EStageProbes  # noqa: E402

PITCH_CENTER = 1300.0
PITCH_SCALE = 600.0


def examples(rng: np.random.Generator, songs: int, progress: float, audio_domain: float):
    out = []
    offsets = np.array([-300.0, -100.0, -50.0, 0.0, 50.0, 100.0, 300.0])
    for song_id in range(songs):
        target = make_target(rng, sample_level(rng, progress))
        voiced = np.isfinite(target)
        kind = str(rng.choice(SOURCE_KINDS))
        reference, _ = synth_source(target, kind, rng, sr=SAMPLE_RATE)
        reference = frame_audio_numpy(room(reference, SAMPLE_RATE, rng), len(target))
        offset = np.empty(len(target))
        at = 0
        while at < len(target):
            stop = min(len(target), at + int(rng.integers(12, 61)))
            offset[at:stop] = float(rng.choice(offsets))
            at = stop
        actual = np.where(voiced, np.nan_to_num(target) + offset, 0.0)
        own = RawSelfAudio(1, rng, domain=audio_domain).render(actual[None], voiced[None])[0]
        observed = np.concatenate([[0.0], actual[:-1]])
        observed_voiced = np.concatenate([[False], voiced[:-1]])
        error = np.where(voiced & observed_voiced, np.nan_to_num(target) - observed, 0.0)
        out.append((reference, own, np.where(voiced, np.nan_to_num(target), 0.0).astype(np.float32),
                    voiced, (voiced & observed_voiced), error.astype(np.float32), song_id))
    return out


@torch.inference_mode()
def encode(model, data, device, batch_size: int):
    collected = {k: [] for k in ("memory", "own", "pitch", "voiced", "error_valid", "error", "sequence")}
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        lengths = np.array([len(x[2]) for x in batch], int)
        steps = int(lengths.max())
        ref = np.zeros((len(batch), steps, FRAME), np.float32)
        own = np.zeros_like(ref)
        pitch = np.zeros((len(batch), steps), np.float32)
        voiced = np.zeros((len(batch), steps), bool)
        error_valid = np.zeros_like(voiced)
        error = np.zeros_like(pitch)
        sequence = np.zeros((len(batch), steps), np.int64)
        for i, (r, o, p, v, ev, e, sid) in enumerate(batch):
            n = len(p)
            ref[i, :n], own[i, :n], pitch[i, :n] = r, o, p
            voiced[i, :n], error_valid[i, :n], error[i, :n], sequence[i, :n] = v, ev, e, sid
        lengths_t = torch.from_numpy(lengths).to(device)
        memory, _, _ = model.encode_reference(torch.from_numpy(ref).to(device), lengths_t)
        own_feature = model.audio_features(torch.from_numpy(own).to(device))
        mask = np.arange(steps)[None] < lengths[:, None]
        collected["memory"].append(memory.cpu()[mask])
        collected["own"].append(own_feature.cpu()[mask])
        collected["pitch"].append(torch.from_numpy((pitch[mask] - PITCH_CENTER) / PITCH_SCALE))
        collected["voiced"].append(torch.from_numpy(voiced[mask]))
        collected["error_valid"].append(torch.from_numpy(error_valid[mask]))
        collected["error"].append(torch.from_numpy(error[mask] / PITCH_SCALE))
        collected["sequence"].append(torch.from_numpy(sequence[mask]))
    return {key: torch.cat(value) for key, value in collected.items()}


@torch.inference_mode()
def evaluate(probes, data, device):
    ref, own_pitch, error = probes(data["memory"].to(device), data["own"].to(device))
    ref, own_pitch, error = ref.cpu(), own_pitch.cpu(), error.cpu()
    voiced = data["voiced"]
    reference_mae = float((ref[:, 0][voiced] - data["pitch"][voiced]).abs().mean() * PITCH_SCALE)
    prediction = ref[:, 1] >= 0
    tp = int((prediction & voiced).sum()); fp = int((prediction & ~voiced).sum()); fn = int((~prediction & voiced).sum())
    voice_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    same = data["sequence"][3:] == data["sequence"][:-3]
    true_delta = (data["pitch"][2:-1] - data["pitch"][1:-2]) * PITCH_SCALE
    pred_delta = (ref[3:, 0] - ref[:-3, 0]) * PITCH_SCALE
    changed = same & data["voiced"][2:-1] & data["voiced"][1:-2] & (true_delta.abs() >= 25.0)
    direction = float((torch.sign(pred_delta[changed]) == torch.sign(true_delta[changed])).float().mean())
    valid = data["error_valid"]
    error_mae = float((error[valid] - data["error"][valid]).abs().mean() * PITCH_SCALE)
    sign_mask = valid & (data["error"].abs() * PITCH_SCALE >= 25.0)
    error_sign = float((torch.sign(error[sign_mask]) == torch.sign(data["error"][sign_mask])).float().mean())
    return {"reference_mae_cents": reference_mae, "voice_f1": voice_f1,
            "change_direction_accuracy": direction, "error_mae_cents": error_mae,
            "error_sign_accuracy": error_sign,
            "self_pitch_mae_cents": float((own_pitch[valid] - (data["pitch"][valid] - data["error"][valid])).abs().mean()
                                          * PITCH_SCALE),
            "gate2_pass": reference_mae < 60.0 and voice_f1 >= 0.95 and direction + 1e-6 >= 0.90,
            "gate4_pass": error_mae < 60.0 and error_sign >= 0.90}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="runs/yamabiko_e2e_pc_domain_rl.pt")
    ap.add_argument("--out", default="runs/yamabiko_e2e_stage_probes.pt")
    ap.add_argument("--report", default="runs/yamabiko_e2e_stage_report.json")
    ap.add_argument("--songs", type=int, default=192)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--encode-batch", type=int, default=12)
    ap.add_argument("--audio-domain", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=4040)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    base = E2EImitator.from_checkpoint(torch.load(args.base, map_location=args.device), args.device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    raw = examples(rng, args.songs, 0.8, args.audio_domain)
    split = int(0.8 * len(raw))
    train = encode(base, raw[:split], args.device, args.encode_batch)
    validation = encode(base, raw[split:], args.device, args.encode_batch)
    probes = E2EStageProbes(base.config).to(args.device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=1e-3, weight_decay=1e-5)
    n = len(train["pitch"])
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(rng.integers(n, size=args.batch)).long()
        memory, own = train["memory"][idx].to(args.device), train["own"][idx].to(args.device)
        ref, own_pitch, error = probes(memory, own)
        voiced = train["voiced"][idx].to(args.device)
        error_valid = train["error_valid"][idx].to(args.device)
        pitch_target = train["pitch"][idx].to(args.device)
        error_target = train["error"][idx].to(args.device)
        loss = (F.smooth_l1_loss(ref[:, 0][voiced], pitch_target[voiced])
                + 0.25 * F.binary_cross_entropy_with_logits(ref[:, 1], voiced.float())
                + F.smooth_l1_loss(own_pitch[error_valid],
                                   (pitch_target - error_target)[error_valid])
                + F.smooth_l1_loss(error[error_valid], error_target[error_valid]))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:4d} loss {float(loss.detach()):.4f}", flush=True)
    metrics = evaluate(probes.eval(), validation, args.device)
    checkpoint = {"format": "yamabiko-e2e-stage-probes-v1", "config": vars(base.config),
                  "state": probes.state_dict(), "base": args.base, "metrics": metrics}
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, out)
    report = pathlib.Path(args.report); report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {out} and {report}")


if __name__ == "__main__":
    main()
