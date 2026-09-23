"""Train the neural Planner/Controller/Feedback with in-network rig memory.

Each episode is one randomized rig (closed-tube flute, deadband, hearing
delay/noise/drop-outs) playing several *different* random songs in a row.
The slow memory is carried from song to song, so the only thing worth storing
in it is what stays true for the rig.  The loss is the heard-independent
emitted pitch error on sounding frames; no network sees simulator state or
rig parameters.  Gradients flow through the differentiable simulator.
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
from flute_rl.yamabiko.rig_adaptive import RigAdaptiveConfig, RigAdaptivePerformer  # noqa: E402


def episode_loss(model, plant, songs, params, generator, carry=True, calibration_songs=None):
    """``calibration_songs=k``: only the first k songs write the rig memory."""
    memory, losses, maes = None, [], []
    for index, (cents, voice) in enumerate(songs):
        write = calibration_songs is None or index < calibration_songs
        result, new_memory = model.perform(plant, cents, voice, params, memory, generator,
                                           write_memory=write)
        memory = new_memory if carry else None
        error = result["pitch_cents"] - cents
        pitch = F.smooth_l1_loss(error[voice], torch.zeros_like(error[voice]), beta=20.0) / 100.0
        pwm = result["pwm"]
        effort = 1e-3 * pwm.square().mean() + 2e-3 * (pwm[:, 1:] - pwm[:, :-1]).square().mean()
        losses.append(pitch + effort); maes.append(error[voice].abs().mean())
    return torch.stack(losses).mean(), torch.stack(maes)


@torch.no_grad()
def evaluate(model, plant, songs, params, calibration_songs=None):
    model.eval()
    out = {}
    for label, carry in (("carry", True), ("reset", False)):
        _, maes = episode_loss(model, plant, songs, params, torch.Generator(params.torque_gain.device).manual_seed(0),
                               carry, calibration_songs)
        out[label] = [float(x) for x in maes]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--songs", type=int, default=3)
    ap.add_argument("--song-steps", type=int, default=450)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup-steps", type=int, default=300,
                    help="first steps use 2 short songs so the tracker learns before the memory")
    ap.add_argument("--calibration-songs", type=int, default=None,
                    help="learning-mode design: only the first N songs write the rig memory")
    ap.add_argument("--init", default=None, help="start from this rig-adaptive checkpoint")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/yamabiko_rig_adaptive_v1.pt")
    ap.add_argument("--report", default="runs/yamabiko_rig_adaptive_v1_report.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = args.device
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    model = (RigAdaptivePerformer.from_checkpoint(torch.load(args.init, map_location=device, weights_only=False), device)
             if args.init else RigAdaptivePerformer(RigAdaptiveConfig()).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .1)
    eval_params = plant.parameters(96, device, spread=1.0, generator=torch.Generator(device).manual_seed(7777))
    eval_songs = [random_melodies(np.random.default_rng(900 + k), 96, 600, device) for k in range(4)]
    history, best = [], None
    generator = torch.Generator(device).manual_seed(args.seed)
    started = time.time()
    for step in range(1, args.steps + 1):
        params = plant.parameters(args.batch, device, spread=1.0, generator=generator)
        warm = step <= args.warmup_steps
        songs = [random_melodies(rng, args.batch, 250 if warm else args.song_steps, device)
                 for _ in range(2 if warm else args.songs)]
        loss, maes = episode_loss(model, plant, songs, params, generator,
                                  calibration_songs=None if warm else args.calibration_songs)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); schedule.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:4d} loss {float(loss.detach()):.4f} song MAE " +
                  " ".join(f"{float(m):6.1f}" for m in maes) + f"  {time.time() - started:6.0f}s", flush=True)
        if (step % args.eval_every == 0 and step > args.warmup_steps) or step == args.steps:
            metrics = evaluate(model, plant, eval_songs, eval_params, args.calibration_songs)
            score = float(np.mean(metrics["carry"]))
            history.append({"step": step, **metrics})
            print(f"eval {step}: carry {metrics['carry']} reset {metrics['reset']}", flush=True)
            if best is None or score < best["score"]:
                best = {"step": step, "score": score, **metrics}
                torch.save(model.checkpoint(plant_config=vars(plant.config), best=best, seed=args.seed), args.out)
    report = {"best": best, "warmup_steps": args.warmup_steps, "calibration_songs": args.calibration_songs,
              "init": args.init, "lr": args.lr, "history": history, "seed": args.seed, "steps": args.steps,
              "batch": args.batch, "songs": args.songs, "song_steps": args.song_steps,
              "simulator": "PhysicalPlantConfig.realistic()", "training": "bptt-through-differentiable-simulator",
              "network_received_simulator_internal_state": False, "real_rig_validated": False}
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
