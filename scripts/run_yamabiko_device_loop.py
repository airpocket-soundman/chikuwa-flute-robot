"""Learn on a rig the way the real one allows: play, log, model, fine-tune, repeat.

For each black-box rig (a simulated rig that only exposes commands in and
heard pitch out):

1. play the evaluation songs and some practice songs, keeping the logs;
2. fit the world model to this rig from its logs only (context vector, then
   optionally the whole model);
3. fine-tune the performer inside that world model (no simulator gradient);
4. go back to the rig and play again.

The rig's hidden emitted pitch is used only to score the evaluation songs.
A "memory only" arm keeps the pre-trained weights and adapts through its
memories alone, so the gain from on-rig weight learning is visible.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.adaptive_memory import AdaptiveMemoryPerformer  # noqa: E402
from flute_rl.yamabiko.device import BlackBoxDevice  # noqa: E402
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.world_model import DeviceWorldModel  # noqa: E402


def replicate(params, row, copies):
    """One rig, ``copies`` times (parallel plays stand in for successive plays)."""
    return type(params)(*(None if v is None else v[row:row + 1].expand(copies, *v.shape[1:]).clone()
                          for v in vars(params).values()))


def score(played, cents, voice):
    return float((played - cents).abs()[voice].mean())


def fit_world_model(world, logs, steps, lr_context, lr_model, device):
    """Adapt the world model to this rig's logs: context first, then the whole model gently."""
    world = copy.deepcopy(world)
    with torch.no_grad():
        context = world.encoder(logs[:1])[:1].clone()
    context.requires_grad_(True)
    optimizer = torch.optim.Adam([{"params": [context], "lr": lr_context},
                                  {"params": [p for n, p in world.named_parameters() if not n.startswith("encoder")],
                                   "lr": lr_model}])
    for _ in range(steps):
        batch = [logs[i] for i in np.random.default_rng(len(logs) + _).choice(len(logs), min(3, len(logs)), False)]
        loss, mae = world.loss(batch, context.expand(batch[0].pwm.shape[0], -1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return world, context.detach(), float(mae)


def fine_tune_policy(policy, world, context, songs, steps, lr):
    policy = copy.deepcopy(policy); policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(steps):
        cents, voice = songs[step % len(songs)]
        world.bind(context.expand(cents.shape[0], -1))
        result, _ = policy.play(world, cents, voice)
        error = result["pitch_cents"] - cents
        loss = F.smooth_l1_loss(error[voice] / 100, torch.zeros_like(error[voice]), beta=.2)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
    policy.eval()
    return policy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="runs/yamabiko_adaptive_memory_v1.pt")
    ap.add_argument("--world-model", default="runs/yamabiko_world_model_v1.pt")
    ap.add_argument("--rigs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--songs", type=int, default=4, help="evaluation songs, played in parallel")
    ap.add_argument("--song-steps", type=int, default=400)
    ap.add_argument("--world-steps", type=int, default=60)
    ap.add_argument("--policy-steps", type=int, default=40)
    ap.add_argument("--policy-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="docs/e2e-rig-adaptive-results/device_loop.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    base_policy = AdaptiveMemoryPerformer.from_checkpoint(
        torch.load(args.policy, map_location=device, weights_only=False), device).eval()
    base_world = DeviceWorldModel.from_checkpoint(
        torch.load(args.world_model, map_location=device, weights_only=False), device)
    all_params = plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(args.seed))
    speed = all_params.max_velocity_strokes_s / plant.config.max_velocity_strokes_s
    rng = np.random.default_rng(args.seed)
    eval_songs = random_melodies(np.random.default_rng(args.seed + 1), args.songs, args.song_steps, device, varied=True)
    results, started = [], time.time()
    for row in range(args.rigs):
        params = replicate(all_params, row, args.songs)
        # The same rig for both arms, each with its own logs.
        arms = {name: {"policy": base_policy, "memory": None, "curve": [],
                       "rig": BlackBoxDevice(plant, params, torch.Generator(device).manual_seed(row))}
                for name in ("memory_only", "on_rig_learning")}
        for round_index in range(args.rounds + 1):
            practice = random_melodies(rng, args.songs, args.song_steps, device, varied=True)
            for name, arm in arms.items():
                with torch.no_grad():
                    policy, rig = arm["policy"], arm["rig"]
                    memory = policy.forget_song(arm["memory"]) if arm["memory"] is not None else None
                    _, memory = policy.play(rig, *practice, memory)            # practice song (logged)
                    memory = policy.forget_song(memory)
                    result, memory = policy.play(rig, *eval_songs, memory)     # evaluation song (logged)
                    arm["memory"] = memory
                arm["curve"].append(score(result["pitch_cents"], *eval_songs))
            if round_index == args.rounds:
                break
            # On-rig learning between rounds, from the logs only.
            logs = arms["on_rig_learning"]["rig"].logs
            world, context, world_mae = fit_world_model(base_world, logs, args.world_steps, 5e-2, 2e-4, device)
            songs = [practice, eval_songs] + [random_melodies(rng, args.songs, args.song_steps, device, varied=True)
                                              for _ in range(2)]
            arms["on_rig_learning"]["policy"] = fine_tune_policy(arms["on_rig_learning"]["policy"], world, context,
                                                                 songs, args.policy_steps, args.policy_lr)
            print(f"rig {row} (motor x{float(speed[row]):.2f}) round {round_index + 1}: world model MAE {world_mae:.1f}"
                  f"  memory-only {arms['memory_only']['curve'][-1]:.1f}  learning {arms['on_rig_learning']['curve'][-1]:.1f}"
                  f"  {time.time() - started:.0f}s", flush=True)
        results.append({"rig": row, "motor_factor": float(speed[row]),
                        **{name: arm["curve"] for name, arm in arms.items()}})
        print(f"rig {row}: memory-only {[round(x, 1) for x in arms['memory_only']['curve']]}  "
              f"on-rig learning {[round(x, 1) for x in arms['on_rig_learning']['curve']]}", flush=True)
    summary = {name: [float(np.mean([r[name][k] for r in results])) for k in range(args.rounds + 1)]
               for name in ("memory_only", "on_rig_learning")}
    pathlib.Path(args.out).write_text(json.dumps({"summary": summary, "rigs": results, **vars(args)}, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
