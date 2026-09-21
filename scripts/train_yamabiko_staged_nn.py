"""Train independently testable neural stages for Yamabiko.

The training order is strict: neural ear (pretrained and frozen), reference
memory, reference-only feed-forward controller, then signed-error comparator.
The script writes one checkpoint and one JSON file with held-out metrics.  No
stage can inherit a PASS from a bypass around its own network.
"""
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
from flute_rl.targets import make_target, sample_level  # noqa: E402
from flute_rl.yamabiko import Oracle, Rig, RigParams, make_schedule  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_audio import RawSelfAudio  # noqa: E402
from flute_rl.yamabiko.e2e_io import SAMPLE_RATE, frame_audio_numpy  # noqa: E402
from flute_rl.yamabiko.staged_nn import (ErrorComparator, FeedForwardPolicy,
                                         FeedbackResidualPolicy, ReferenceMemory,
                                         StagedConfig, checkpoint)  # noqa: E402

PITCH_CENTER = 1300.0
PITCH_SCALE = 600.0


@dataclass
class Sample:
    ear: torch.Tensor
    target: torch.Tensor
    actions: torch.Tensor


def oracle_actions(target: np.ndarray, rng: np.random.Generator, *, random_rig: bool) -> np.ndarray:
    params = RigParams.sample(rng, 1, harsh=0.0) if random_rig else RigParams.nominal(1)
    rig = Rig(params, np.random.default_rng(int(rng.integers(2**31))))
    schedule = make_schedule([[target]])
    control = Oracle(); control.begin(schedule, rig)
    result = []
    for t in range(schedule.T):
        home = bool(schedule.homing[t])
        if home:
            pwm, valve = np.array([-1.0]), np.array([False])
        else:
            pwm = np.asarray(control.act(t)); valve = schedule.valve[:, t]
        out = rig.step(pwm, valve); control.update(t, pwm, out)
        if home and (t + 1 == schedule.T or not schedule.homing[t + 1]):
            control.homed()
        if not home:
            result.append((float(pwm[0]), float(valve[0])))
    return np.asarray(result, np.float32)


@torch.inference_mode()
def make_samples(ear_model, rng, count: int, device, progress: float, random_rig: bool):
    samples = []
    for _ in range(count):
        target = make_target(rng, sample_level(rng, progress))
        kind = str(rng.choice(SOURCE_KINDS))
        wave, _ = synth_source(target, kind, rng, sr=SAMPLE_RATE)
        frames = frame_audio_numpy(room(wave, SAMPLE_RATE, rng), len(target))
        learned = ear_model.audio_features(torch.from_numpy(frames).to(device))[:, -2:].cpu()
        pitch = np.where(np.isfinite(target), (np.nan_to_num(target) - PITCH_CENTER) / PITCH_SCALE, 0.0)
        voice = np.isfinite(target).astype(np.float32)
        samples.append(Sample(learned, torch.from_numpy(np.column_stack([pitch, voice]).astype(np.float32)),
                              torch.from_numpy(oracle_actions(target, rng, random_rig=random_rig))))
    return samples


def collate(samples, ids, device):
    chosen = [samples[int(i)] for i in ids]
    lengths = torch.tensor([len(x.target) for x in chosen], device=device)
    steps = int(lengths.max())
    ear = torch.zeros(len(chosen), steps, 2, device=device)
    target = torch.zeros_like(ear); actions = torch.zeros_like(ear)
    for row, item in enumerate(chosen):
        n = len(item.target)
        ear[row, :n] = item.ear.to(device); target[row, :n] = item.target.to(device)
        actions[row, :n] = item.actions.to(device)
    mask = torch.arange(steps, device=device)[None] < lengths[:, None]
    return ear, target, actions, lengths, mask


@torch.inference_mode()
def memory_metrics(memory, samples, device):
    errors, truths, predictions, voice_truth, voice_pred = [], [], [], [], []
    for start in range(0, len(samples), 16):
        ear, target, _, lengths, mask = collate(samples, range(start, min(start + 16, len(samples))), device)
        pred, stored = memory(ear, lengths)
        # Decode only the stored tensor.  This deliberately discards ``ear``.
        pred = memory.decode(stored.detach())
        voiced = target[..., 1].bool() & mask
        errors.append((pred[..., 0][voiced] - target[..., 0][voiced]).abs().cpu())
        truths.append(target[..., 0][voiced].cpu()); predictions.append(pred[..., 0][voiced].cpu())
        voice_truth.append(target[..., 1][mask].cpu()); voice_pred.append(pred[..., 1][mask].cpu())
    err = torch.cat(errors) * PITCH_SCALE
    truth, pred = torch.cat(truths).numpy(), torch.cat(predictions).numpy()
    vt, vp = torch.cat(voice_truth).bool(), torch.cat(voice_pred) >= 0
    tp = int((vt & vp).sum()); fp = int((~vt & vp).sum()); fn = int((vt & ~vp).sum())
    return {"mae_cents": float(err.mean()), "p90_cents": float(torch.quantile(err, .9)),
            "trajectory_correlation": float(np.corrcoef(truth, pred)[0, 1]),
            "voice_f1": 2 * tp / max(2 * tp + fp + fn, 1)}


@torch.inference_mode()
def ff_action_metrics(memory, policy, samples, device):
    abs_pwm, valve_ok = [], []
    for start in range(0, len(samples), 16):
        ear, _, action, lengths, mask = collate(samples, range(start, min(start + 16, len(samples))), device)
        decoded, stored = memory(ear, lengths); decoded = memory.decode(stored)
        pred = policy(decoded, lengths)
        abs_pwm.append((pred[..., 0][mask] - action[..., 0][mask]).abs().cpu())
        valve_ok.append(((pred[..., 1][mask] >= .5) == (action[..., 1][mask] >= .5)).cpu())
    e = torch.cat(abs_pwm)
    return {"pwm_mae": float(e.mean()), "pwm_p90": float(torch.quantile(e, .9)),
            "valve_accuracy": float(torch.cat(valve_ok).float().mean())}


def train_comparator(comparator, ear_model, memory, samples, rng, device, steps):
    optimizer = torch.optim.AdamW(comparator.parameters(), lr=1e-3)
    offsets = np.asarray([-400, -200, -100, -50, 50, 100, 200, 400], np.float32)
    for step in range(steps):
        item = samples[int(rng.integers(len(samples)))]
        n = len(item.target); cents = item.target[:, 0].numpy() * PITCH_SCALE + PITCH_CENTER
        voiced = item.target[:, 1].numpy() > .5
        offset = np.empty(n, np.float32); at = 0
        while at < n:
            stop = min(n, at + int(rng.integers(10, 50))); offset[at:stop] = rng.choice(offsets); at = stop
        actual = np.where(voiced, cents + offset, 0.0)
        own_frames = RawSelfAudio(1, rng, domain=1.0).render(actual[None], voiced[None])[0]
        with torch.no_grad():
            own = ear_model.audio_features(torch.from_numpy(own_frames).to(device))[:, -2:]
            decoded, stored = memory(item.ear[None].to(device), torch.tensor([n], device=device))
            target = memory.decode(stored)[0]
        observed = np.r_[0.0, actual[:-1]]
        valid = torch.from_numpy(voiced & np.r_[False, voiced[:-1]]).to(device)
        truth = torch.from_numpy(((cents - observed) / PITCH_SCALE).astype(np.float32)).to(device)
        out = comparator(target, own)
        loss = F.smooth_l1_loss(out[:, 0][valid], truth[valid]) + .1 * F.binary_cross_entropy_with_logits(
            out[:, 1], valid.float())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()


@torch.inference_mode()
def comparator_metrics(comparator, ear_model, memory, samples, rng, device):
    errors, signs = [], []
    offsets = np.asarray([-300, -150, -75, 75, 150, 300], np.float32)
    for item in samples:
        n = len(item.target); cents = item.target[:, 0].numpy() * PITCH_SCALE + PITCH_CENTER
        voiced = item.target[:, 1].numpy() > .5
        off = rng.choice(offsets, n); actual = np.where(voiced, cents + off, 0.0)
        frames = RawSelfAudio(1, rng, domain=1.0).render(actual[None], voiced[None])[0]
        own = ear_model.audio_features(torch.from_numpy(frames).to(device))[:, -2:]
        _, stored = memory(item.ear[None].to(device), torch.tensor([n], device=device))
        target = memory.decode(stored)[0]; pred = comparator(target, own)[:, 0]
        observed = np.r_[0.0, actual[:-1]]; valid_np = voiced & np.r_[False, voiced[:-1]]
        valid = torch.from_numpy(valid_np).to(device)
        truth = torch.from_numpy(((cents - observed) / PITCH_SCALE).astype(np.float32)).to(device)
        errors.append((pred[valid] - truth[valid]).abs().cpu() * PITCH_SCALE)
        strong = valid & (truth.abs() * PITCH_SCALE >= 50)
        signs.append((torch.sign(pred[strong]) == torch.sign(truth[strong])).cpu())
    err = torch.cat(errors)
    return {"error_mae_cents": float(err.mean()), "error_p90_cents": float(torch.quantile(err, .9)),
            "error_sign_accuracy": float(torch.cat(signs).float().mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ear", default="runs/yamabiko_e2e_pc_ear_gate1_balanced.pt")
    ap.add_argument("--out", default="runs/yamabiko_staged_nn.pt")
    ap.add_argument("--report", default="runs/yamabiko_staged_nn_report.json")
    ap.add_argument("--train-songs", type=int, default=256); ap.add_argument("--valid-songs", type=int, default=64)
    ap.add_argument("--memory-steps", type=int, default=500); ap.add_argument("--ff-steps", type=int, default=700)
    ap.add_argument("--comparator-steps", type=int, default=400); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7319); ap.add_argument("--progress", type=float, default=.8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ear_model = E2EImitator.from_checkpoint(torch.load(args.ear, map_location=args.device), args.device).eval()
    for p in ear_model.parameters(): p.requires_grad_(False)
    print("generating fixed nominal-rig train/held-out sets", flush=True)
    train = make_samples(ear_model, rng, args.train_songs, args.device, args.progress, random_rig=False)
    valid = make_samples(ear_model, np.random.default_rng(args.seed + 100_000), args.valid_songs,
                         args.device, args.progress, random_rig=False)
    config = StagedConfig(); memory = ReferenceMemory(config).to(args.device)
    feedforward = FeedForwardPolicy(config).to(args.device); comparator = ErrorComparator(config).to(args.device)
    feedback = FeedbackResidualPolicy(config).to(args.device)
    opt = torch.optim.AdamW(memory.parameters(), lr=1e-3)
    for step in range(1, args.memory_steps + 1):
        ids = rng.integers(len(train), size=args.batch); ear, target, _, lengths, mask = collate(train, ids, args.device)
        pred, _ = memory(ear, lengths); voiced = target[..., 1].bool() & mask
        loss = F.smooth_l1_loss(pred[..., 0][voiced], target[..., 0][voiced]) + .2 * F.binary_cross_entropy_with_logits(
            pred[..., 1][mask], target[..., 1][mask])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(memory.parameters(), 1); opt.step()
        if step == 1 or step % 100 == 0: print(f"memory {step:4d} loss {float(loss):.4f}", flush=True)
    memory.eval(); [p.requires_grad_(False) for p in memory.parameters()]
    opt = torch.optim.AdamW(feedforward.parameters(), lr=1e-3)
    for step in range(1, args.ff_steps + 1):
        ids = rng.integers(len(train), size=args.batch); ear, _, actions, lengths, mask = collate(train, ids, args.device)
        with torch.no_grad(): decoded, stored = memory(ear, lengths); decoded = memory.decode(stored)
        # Scheduled sampling closes the previous-action gap over training.
        teacher = actions if rng.random() > min(.9, step / max(args.ff_steps, 1)) else None
        pred = feedforward(decoded, lengths, teacher_actions=teacher)
        loss = F.smooth_l1_loss(pred[..., 0][mask], actions[..., 0][mask]) + .3 * F.binary_cross_entropy(
            pred[..., 1][mask].clamp(1e-5, 1 - 1e-5), actions[..., 1][mask])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(feedforward.parameters(), 1); opt.step()
        if step == 1 or step % 100 == 0: print(f"feedforward {step:4d} loss {float(loss):.4f}", flush=True)
    feedforward.eval(); train_comparator(comparator, ear_model, memory, train, rng, args.device, args.comparator_steps)
    comparator.eval()
    mem = memory_metrics(memory, valid, args.device); ff = ff_action_metrics(memory, feedforward, valid, args.device)
    comp = comparator_metrics(comparator, ear_model, memory, valid, np.random.default_rng(args.seed + 200_000), args.device)
    result = {"seed": args.seed, "held_out_songs": len(valid), "memory": mem, "feedforward_action": ff,
              "comparator": comp, "notes": ["feedforward action metrics are diagnostic only; physical audio Gate 3 is evaluated separately",
                                               "feedback network is initialized but is not marked trained"]}
    result["memory"]["pass"] = mem["mae_cents"] < 60 and mem["voice_f1"] >= .95 and mem["trajectory_correlation"] >= .95
    result["comparator"]["pass"] = comp["error_mae_cents"] < 60 and comp["error_sign_accuracy"] >= .9
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(config, memory, feedforward, comparator, feedback, ear_checkpoint=args.ear,
                          training_seed=args.seed, metrics=result), args.out)
    pathlib.Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
