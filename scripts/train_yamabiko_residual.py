"""Pre-train the residual network on top of the deterministic twin skeleton.

Randomized rigs over the realistic range; each rig's "twin" is its true
values disturbed like a real fit (torque and friction +-20 %, the hearing
delay off by one 30 % of the time), because the skeleton will run on fitted
twins.  Each song is played twice so the song memory learns to help the
repeat.  Gradients flow through the simulator (pre-training only).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.residual_performer import ResidualConfig, ResidualTwinPerformer  # noqa: E402


def disturbed(true, generator, amount=.2, delay_flip=.3):
    fields = dict(vars(true))
    for name in ("torque_gain", "coulomb_friction", "viscous_friction", "torque_tau_s"):
        noise = torch.rand(true.torque_gain.shape[0], device=true.torque_gain.device, generator=generator) * 2 - 1
        fields[name] = true.__dict__[name] * (1 + amount * noise)
    flip = torch.rand(true.torque_gain.shape[0], device=true.torque_gain.device, generator=generator) < delay_flip
    step = torch.where(torch.rand(true.torque_gain.shape[0], device=true.torque_gain.device, generator=generator) < .5,
                       -1, 1)
    fields["hearing_delay_steps"] = torch.where(flip, (true.hearing_delay_steps + step).clamp_min(0),
                                                true.hearing_delay_steps)
    return type(true)(**fields)


def songs(rng, batch, steps, device):
    heavy = int(round(batch * .3))
    a = random_melodies(rng, batch - heavy, steps, device, varied=True)
    b = random_melodies(rng, heavy, steps, device, varied=True, rest_probability=.45, repeats=10)
    return torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]])


def episode(model, plant, params, twin, cents, voice, generator):
    memory, loss, maes = None, 0.0, []
    for weight in (.3, 1.0):
        result, memory = model.perform(plant, cents, voice, params, memory, generator, twin=twin)
        error = result["pitch_cents"] - cents
        loss = loss + weight * F.smooth_l1_loss(error[voice] / 100, torch.zeros_like(error[voice]), beta=.2)
        maes.append(float(error[voice].abs().mean()))
    return loss / 1.3, maes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--song-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out", default="runs/yamabiko_residual_v1.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    model = ResidualTwinPerformer(plant, ResidualConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .1)
    generator = torch.Generator(device).manual_seed(args.seed)
    eval_true = plant.parameters(48, device, spread=1.0, generator=torch.Generator(device).manual_seed(7777))
    eval_twin = disturbed(eval_true, torch.Generator(device).manual_seed(7778))
    eval_songs = songs(np.random.default_rng(900), 48, 500, device)
    best, history, started = None, [], time.time()
    for step in range(1, args.steps + 1):
        params = plant.parameters(args.batch, device, spread=1.0, generator=generator)
        twin = disturbed(params, generator)
        cents, voice = songs(rng, args.batch, args.song_steps, device)
        loss, maes = episode(model, plant, params, twin, cents, voice, generator)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); schedule.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:4d} loss {float(loss.detach()):.4f} MAE {maes[0]:6.1f} {maes[1]:6.1f}"
                  f"  {time.time() - started:6.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                model.eval(); _, eval_maes = episode(model, plant, eval_true, eval_twin, *eval_songs,
                                                     torch.Generator(device).manual_seed(0)); model.train()
            history.append({"step": step, "maes": eval_maes})
            print(f"eval {step}: play1 {eval_maes[0]:.1f} play2 {eval_maes[1]:.1f}", flush=True)
            if best is None or eval_maes[1] < best["play2"]:
                best = {"step": step, "play1": eval_maes[0], "play2": eval_maes[1]}
                torch.save(model.checkpoint(best=best, seed=args.seed), args.out)
    if step == args.steps:
        pathlib.Path(args.out).with_suffix(".json").write_text(json.dumps({"best": best, "history": history,
                                                                          **vars(args)}, indent=2))


if __name__ == "__main__":
    main()
