"""Pre-train the rig world model on black-box logs of many simulated rigs.

Every sample is what a real rig could record: the performer plays two songs
and a random PWM excitation through a :class:`BlackBoxDevice`.  The context
encoder reads the first song's log; the world model must then predict the
heard pitch of the other song and of the excitation from their commands.
No simulator state or parameter is used as input or label.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.adaptive_memory import AdaptiveMemoryPerformer  # noqa: E402
from flute_rl.yamabiko.device import BlackBoxDevice  # noqa: E402
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.world_model import DeviceWorldModel, WorldModelConfig  # noqa: E402


def excitation(rng, batch, steps, device):
    """Random held PWM levels with the valve open: broad coverage of the motor."""
    pwm = np.zeros((batch, steps), np.float32)
    for row in range(batch):
        t = 0
        while t < steps:
            stop = min(steps, t + int(rng.integers(5, 40)))
            pwm[row, t:stop] = rng.uniform(-1, 1); t = stop
    return torch.from_numpy(pwm).to(device)


def play_excitation(device_env, pwm):
    with torch.no_grad():
        state = device_env.reset(pwm.device)
        valve = torch.ones_like(pwm)
        heard, valid, emitted = [], [], []
        for t in range(pwm.shape[1]):
            h, v, state, e = device_env.step(state, pwm[:, t], valve[:, t])
            heard.append(h); valid.append(v); emitted.append(e)
        heard, valid, emitted = (torch.stack(x, 1) for x in (heard, valid, emitted))
        device_env.record(torch.zeros_like(pwm), torch.zeros_like(pwm, dtype=torch.bool), pwm, valve,
                          heard, valid, emitted)


def collect(policy, plant, rng, batch, steps, device, generator):
    params = plant.parameters(batch, device, spread=1.0, generator=generator)
    rig = BlackBoxDevice(plant, params, generator)
    with torch.no_grad():
        for _ in range(2):
            cents, voice = random_melodies(rng, batch, steps, device, varied=True)
            policy.play(rig, cents, voice)
    play_excitation(rig, excitation(rng, batch, steps, device))
    return rig.logs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="runs/yamabiko_adaptive_memory_v1.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--song-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/yamabiko_world_model_v1.pt")
    ap.add_argument("--report", default="runs/yamabiko_world_model_v1_report.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); device = args.device
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    policy = AdaptiveMemoryPerformer.from_checkpoint(
        torch.load(args.policy, map_location=device, weights_only=False), device).eval()
    model = DeviceWorldModel(WorldModelConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .05)
    generator = torch.Generator(device).manual_seed(args.seed)
    eval_logs = collect(policy, plant, np.random.default_rng(1), 64, args.song_steps, device,
                        torch.Generator(device).manual_seed(4242))
    history, best, started = [], None, time.time()
    for step in range(1, args.steps + 1):
        logs = collect(policy, plant, rng, args.batch, args.song_steps, device, generator)
        context = model.encoder(logs[:1])
        loss, mae = model.loss(logs[1:], context)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); schedule.step()
        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                _, eval_mae = model.loss(eval_logs[1:], model.encoder(eval_logs[:1]))
            history.append({"step": step, "train_mae": float(mae), "eval_mae": float(eval_mae)})
            print(f"step {step:4d} loss {float(loss):.4f} heard MAE {float(mae):6.1f} eval {float(eval_mae):6.1f}"
                  f"  {time.time() - started:6.0f}s", flush=True)
            if best is None or float(eval_mae) < best["eval_mae"]:
                best = {"step": step, "eval_mae": float(eval_mae)}
                torch.save(model.checkpoint(best=best, seed=args.seed, policy=args.policy), args.out)
    pathlib.Path(args.report).write_text(json.dumps({"best": best, "history": history, **vars(args)}, indent=2),
                                         encoding="utf-8")
    print(json.dumps(best))


if __name__ == "__main__":
    main()
