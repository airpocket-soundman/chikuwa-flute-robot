"""Causal DAgger for the reference-only feed-forward controller."""
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

from flute_rl.audio import room, synth_source  # noqa: E402
from flute_rl.targets import make_target, sample_level  # noqa: E402
from flute_rl.yamabiko import Oracle, Rig, RigParams, make_schedule  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_io import SAMPLE_RATE, frame_audio_numpy  # noqa: E402
from flute_rl.yamabiko.staged_nn import FeedForwardPolicy, ReferenceMemory, StagedConfig  # noqa: E402


@torch.inference_mode()
def decoded_reference(target, rng, ear, memory, batch, device):
    wave, _ = synth_source(target, "recorder", rng, sr=SAMPLE_RATE)
    frames = frame_audio_numpy(room(wave, SAMPLE_RATE, rng), len(target))
    learned = ear.audio_features(torch.from_numpy(frames).to(device))[:, -2:]
    length = torch.tensor([len(target)], device=device)
    _, stored = memory(learned[None], length)
    return memory.decode(stored).expand(batch, -1, -1).contiguous()


@torch.inference_mode()
def dagger_labels(policy, decoded, target, rng):
    batch, steps = decoded.shape[:2]; device = decoded.device
    lengths = torch.full((batch,), steps, device=device)
    action = policy(decoded, lengths).cpu().numpy()
    rig = Rig(RigParams.nominal(batch), np.random.default_rng(int(rng.integers(2**31))))
    schedule = make_schedule([[target for _ in range(batch)]])
    oracle = Oracle(); oracle.begin(schedule, rig)
    for t in range(100):
        pwm = -np.ones(batch); out = rig.step(pwm, np.zeros(batch, bool)); oracle.update(t, pwm, out)
    oracle.homed(); labels = []
    for t in range(steps):
        labels.append(np.asarray(oracle.act(100 + t), np.float32))
        out = rig.step(action[:, t, 0], action[:, t, 1] >= .5)
        oracle.update(100 + t, action[:, t, 0], out)
    return torch.from_numpy(np.stack(labels, 1)).to(device), torch.from_numpy(schedule.valve[:, 100:].astype(np.float32)).to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_staged_nn_ff2.pt"); ap.add_argument("--out", default="runs/yamabiko_staged_nn_ff_dagger.pt")
    ap.add_argument("--report", default="runs/yamabiko_feedforward_dagger_report.json")
    ap.add_argument("--iterations", type=int, default=400); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=11239)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    ck = torch.load(args.init, map_location=args.device); cfg = StagedConfig(**ck["config"])
    memory, policy = ReferenceMemory(cfg).to(args.device), FeedForwardPolicy(cfg).to(args.device)
    memory.load_state_dict(ck["memory"]); policy.load_state_dict(ck["feedforward"]); memory.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    for module in (ear, memory):
        for p in module.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6); losses = []
    for iteration in range(1, args.iterations + 1):
        target = make_target(rng, sample_level(rng, .8)); decoded = decoded_reference(target, rng, ear, memory, args.batch, args.device)
        pwm_label, valve_label = dagger_labels(policy.eval(), decoded, target, rng)
        decoded, pwm_label, valve_label = decoded.clone(), pwm_label.clone(), valve_label.clone()
        lengths = torch.full((args.batch,), len(target), device=args.device)
        pred = policy.train()(decoded, lengths)
        pwm_loss = F.mse_loss(pred[..., 0], pwm_label); valve_loss = F.binary_cross_entropy(
            pred[..., 1].clamp(1e-5, 1 - 1e-5), valve_label)
        loss = pwm_loss + .2 * valve_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1); optimizer.step()
        losses.append(float(loss.detach()))
        if iteration == 1 or iteration % 20 == 0: print(f"iteration {iteration:4d} loss {losses[-1]:.5f} pwm {float(pwm_loss.detach()):.5f}", flush=True)
    ck["feedforward"] = policy.eval().state_dict(); ck["feedforward_dagger"] = {
        "iterations": args.iterations, "batch": args.batch, "seed": args.seed,
        "final_loss": losses[-1], "mean_last_20": float(np.mean(losses[-20:]))}
    torch.save(ck, args.out); pathlib.Path(args.report).write_text(json.dumps(ck["feedforward_dagger"], indent=2), encoding="utf-8")
    print(json.dumps(ck["feedforward_dagger"], indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
