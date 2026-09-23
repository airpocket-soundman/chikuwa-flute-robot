"""Train the three-memory performer on sessions with part swaps.

One episode is a session on one rig: a few different songs, each repeated
one to three times.  Between songs the motor or the flute of some rigs is
swapped and only that part's memory is wiped, and the song memory is wiped
on every new song.  Memories are randomly frozen for whole plays so that the
evaluation can freeze all but one kind of adaptation.  The loss is the pitch
error of every play (per-rig normalized); no network sees simulator state.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.adaptive_memory import (PARTS, AdaptiveMemoryConfig,  # noqa: E402
                                               AdaptiveMemoryPerformer)
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402

MOTOR_FIELDS = ("torque_gain", "torque_tau_s", "inertia", "coulomb_friction", "viscous_friction",
                "max_velocity_strokes_s", "deadband")
FLUTE_FIELDS = ("tube_offset_m", "temp_offset_c", "flute_offset_cents")


def swap(params, other, fields, rows):
    result = type(params)(**vars(params))
    for field in fields:
        setattr(result, field, torch.where(rows, getattr(other, field), getattr(params, field)))
    return result


def play_loss(model, plant, cents, voice, params, memory, generator, write, normalize=True):
    result, memory = model.perform(plant, cents, voice, params, memory, generator, write=write)
    error = result["pitch_cents"] - cents
    per_frame = F.smooth_l1_loss(error, torch.zeros_like(error), beta=20.0, reduction="none") / 100.0
    per_rig = (per_frame * voice).sum(1) / voice.sum(1).clamp_min(1)
    if normalize:
        per_rig = per_rig / (per_rig.detach() + .2)
    pwm = result["pwm"]
    effort = 1e-3 * pwm.square().mean() + 2e-3 * (pwm[:, 1:] - pwm[:, :-1]).square().mean()
    return per_rig.mean() + effort, error[voice].abs().mean(), memory


def transfer_from_rig_adaptive(model, path, device):
    """Start from a trained single-memory model: copy control weights, zero the new memory inputs."""
    source = torch.load(path, map_location=device, weights_only=False)["state_dict"]
    own = model.state_dict()
    window = 20  # 2 * len(WINDOW)
    copied = 0
    for key, value in source.items():
        target = key.replace("memory.", "unused.")
        if target not in own:
            continue
        if own[target].shape == value.shape:
            own[target] = value.clone(); copied += 1
        elif key in ("planner.net.0.weight",):
            own[target].zero_(); own[target][:, :window] = value[:, :window]; copied += 1
        elif key in ("core.cell.weight_ih",):
            own[target].zero_(); own[target][:, :9] = value[:, :9]; copied += 1
    model.load_state_dict(own)
    return copied


@torch.no_grad()
def evaluate(model, plant, songs, params):
    """Fresh rig, one song three times with every memory writing: the learning curve."""
    model.eval(); plays = []
    memory = None
    for r in range(3):
        _, mae, memory = play_loss(model, plant, songs[0][0], songs[0][1], params, memory,
                                   torch.Generator(params.torque_gain.device).manual_seed(r), True)
        plays.append(float(mae))
    # A new song on the now familiar rig (song memory wiped), then its repeat.
    memory = model.forget_song(memory)
    for r in range(2):
        _, mae, memory = play_loss(model, plant, songs[1][0], songs[1][1], params, memory,
                                   torch.Generator(params.torque_gain.device).manual_seed(10 + r), True)
        plays.append(float(mae))
    model.train()
    return plays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--song-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--motor-curriculum", type=int, default=400)
    ap.add_argument("--swap-probability", type=float, default=.35)
    ap.add_argument("--freeze-probability", type=float, default=.2)
    ap.add_argument("--init-from-rig-adaptive", default="runs/yamabiko_rig_adaptive_v3.pt")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/yamabiko_adaptive_memory_v1.pt")
    ap.add_argument("--report", default="runs/yamabiko_adaptive_memory_v1_report.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); device = args.device
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    model = AdaptiveMemoryPerformer(AdaptiveMemoryConfig()).to(device)
    if args.init_from_rig_adaptive and pathlib.Path(args.init_from_rig_adaptive).exists():
        print("transferred", transfer_from_rig_adaptive(model, args.init_from_rig_adaptive, device), "tensors")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .1)
    eval_params = plant.parameters(64, device, spread=1.0, generator=torch.Generator(device).manual_seed(7777))
    eval_songs = [random_melodies(np.random.default_rng(900 + k), 64, 600, device, varied=True) for k in range(2)]
    generator = torch.Generator(device).manual_seed(args.seed)
    history, best, started = [], None, time.time()
    for step in range(1, args.steps + 1):
        k = min(1.0, step / max(1, args.motor_curriculum))
        low, high = plant.config.motor_scale_range
        span = (math.exp(math.log(.8) + k * (math.log(low) - math.log(.8))),
                math.exp(math.log(1.25) + k * (math.log(high) - math.log(1.25))))
        train_plant = DifferentiableMotorFlute(dataclasses.replace(plant.config, motor_scale_range=span))
        params = train_plant.parameters(args.batch, device, spread=1.0, generator=generator)
        repeats = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
        memory, losses, maes = None, [], []
        for segment, reps in enumerate(repeats):
            cents, voice = random_melodies(rng, args.batch, args.song_steps, device, varied=True)
            if segment:
                memory = model.forget_song(memory)
                for fields, part in ((MOTOR_FIELDS, "motor"), (FLUTE_FIELDS, "flute")):
                    rows = torch.rand(args.batch, device=device, generator=generator) < args.swap_probability
                    other = train_plant.parameters(args.batch, device, spread=1.0, generator=generator)
                    params = swap(params, other, fields, rows)
                    memory = model.forget(memory, part, rows=rows)
            for _ in range(reps):
                write = {part: torch.rand(args.batch, device=device, generator=generator) >= args.freeze_probability
                         for part in PARTS}
                loss, mae, memory = play_loss(model, train_plant, cents, voice, params, memory, generator, write)
                losses.append(loss); maes.append(float(mae))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); schedule.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:4d} loss {float(loss.detach()):.4f} plays {repeats} MAE "
                  + " ".join(f"{m:6.1f}" for m in maes) + f"  {time.time() - started:6.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            plays = evaluate(model, plant, eval_songs, eval_params)
            score = float(np.mean(plays[1:]))
            history.append({"step": step, "plays": plays})
            print(f"eval {step}: song A x3 {[round(x, 1) for x in plays[:3]]} "
                  f"song B x2 {[round(x, 1) for x in plays[3:]]}", flush=True)
            if best is None or score < best["score"]:
                best = {"step": step, "score": score, "plays": plays}
                torch.save(model.checkpoint(best=best, seed=args.seed, protocol="session-swaps"), args.out)
    pathlib.Path(args.report).write_text(json.dumps(
        {"best": best, "history": history, **{k: v for k, v in vars(args).items()}}, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
