"""Learn on a rig the way the real one allows: play, log, model, fine-tune, repeat.

For each black-box rig (a simulated rig that only exposes commands in and
heard pitch out):

1. play the evaluation songs and some practice songs, keeping the logs;
2. fit a world model to this rig from its play logs only, warm-starting
   from the previous round's fit;
3. fine-tune the performer inside that world model (no simulator gradient);
4. go back to the rig and play again.

The rig's hidden emitted pitch is used only to score the evaluation songs.
A "memory only" arm keeps the pre-trained weights and adapts through its
memories alone, so the gain from on-rig weight learning is visible.  The
weights that sounded best so far are kept; an update that makes what the rig
hears worse is dropped and the next round fine-tunes from the best weights
again.  That judgement uses heard pitch only, as the real rig would.
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
from flute_rl.yamabiko.world_model import DeviceWorldModel, WorldModelConfig  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_yamabiko_world_model import truncate  # noqa: E402


def replicate(params, row, copies):
    """One rig, ``copies`` times (parallel plays stand in for successive plays)."""
    return type(params)(*(None if v is None else v[row:row + 1].expand(copies, *v.shape[1:]).clone()
                          for v in vars(params).values()))


def score(played, cents, voice):
    """Evaluator only: the simulator's true emitted pitch against the target."""
    return float((played - cents).abs()[voice].mean())


def heard_error(result, cents, voice, delay=2):
    """What the rig itself can judge: heard pitch against the target it was playing."""
    heard, valid = result["heard"][:, delay:], result["valid"][:, delay:]
    target, sounding = cents[:, :-delay], voice[:, :-delay]
    mask = valid & sounding
    return float((heard - target).abs()[mask].mean())


def fit_world_model(logs, steps, lr, device, init=None):
    """Fit a world model to this rig from its logs alone.

    Every log starts at the home stop, so the fitted prefix grows from 30
    steps to the whole log (a velocity error integrated over a whole song is
    too rough a start).  ``init`` optionally warm-starts from an earlier fit.
    """
    world = copy.deepcopy(init) if init is not None else DeviceWorldModel(WorldModelConfig()).to(device)
    optimizer = torch.optim.Adam(world.parameters(), lr=lr)
    full = max(log.pwm.shape[1] for log in logs)
    for step in range(steps):
        horizon = min(full, 30 + (full - 30) * step // max(1, int(steps * .6)))
        batch = [truncate(log, horizon) for log in logs[-6:]]
        loss, mae = world.loss(batch, world.encoder(logs[-6:]))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(world.parameters(), 1.0); optimizer.step()
    with torch.no_grad():
        context = world.encoder(logs[-6:])
        _, mae = world.loss(logs[-6:], context)
    return world, context.detach(), float(mae)


def fine_tune_policy(policy, world, context, songs, steps, lr, anchor=None, anchor_weight=0.0):
    """Fine-tune inside the world model; ``anchor`` (the pre-trained weights) keeps
    the update close to what is known to work, so an imperfect model of this
    rig cannot pull the policy far away."""
    policy = copy.deepcopy(policy); policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    anchored = list(anchor.parameters()) if anchor is not None else []
    for step in range(steps):
        cents, voice = songs[step % len(songs)]
        world.bind(context.expand(cents.shape[0], -1))
        result, _ = policy.play(world, cents, voice)
        error = result["pitch_cents"] - cents
        loss = F.smooth_l1_loss(error[voice] / 100, torch.zeros_like(error[voice]), beta=.2)
        if anchored and anchor_weight > 0:
            loss = loss + anchor_weight * sum((p - q.detach()).square().sum()
                                              for p, q in zip(policy.parameters(), anchored))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
    policy.eval()
    return policy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="runs/yamabiko_adaptive_memory_v1.pt")
    ap.add_argument("--rigs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--songs", type=int, default=4, help="evaluation songs, played in parallel")
    ap.add_argument("--song-steps", type=int, default=400)
    ap.add_argument("--world-steps", type=int, default=500)
    ap.add_argument("--world-lr", type=float, default=5e-3)
    ap.add_argument("--policy-steps", type=int, default=40)
    ap.add_argument("--policy-lr", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=1e-3,
                    help="L2 pull toward the pre-trained weights during on-rig fine-tuning")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="docs/e2e-rig-adaptive-results/device_loop.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    base_policy = AdaptiveMemoryPerformer.from_checkpoint(
        torch.load(args.policy, map_location=device, weights_only=False), device).eval()
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
                arm.setdefault("heard_curve", []).append(heard_error(result, *eval_songs))
                # Safety on the rig, judged from heard pitch only: keep the weights
                # that sounded best so far; a worse update is dropped and the next
                # fine-tuning starts again from the best weights.
                if name == "on_rig_learning":
                    heard_now = arm["heard_curve"][-1]
                    if arm.get("best_heard") is None or heard_now <= arm["best_heard"]:
                        arm["best_heard"], arm["best_policy"] = heard_now, arm["policy"]
                    else:
                        arm["policy"] = arm["best_policy"]
                        arm.setdefault("rollbacks", []).append(round_index)
            if round_index == args.rounds:
                break
            # On-rig learning between rounds, from the play logs only.  (A random
            # PWM excitation was tried: it drives the plunger over the whole
            # stroke, up to ~3500 cent, far outside what the songs use, and its
            # error swamped the fit of the songs.)
            rig = arms["on_rig_learning"]["rig"]
            world, context, world_mae = fit_world_model(rig.logs, args.world_steps, args.world_lr, device,
                                                        init=arms["on_rig_learning"].get("world"))
            arms["on_rig_learning"]["world"] = world
            songs = [practice, eval_songs] + [random_melodies(rng, args.songs, args.song_steps, device, varied=True)
                                              for _ in range(2)]
            arms["on_rig_learning"]["policy"] = fine_tune_policy(arms["on_rig_learning"]["policy"], world, context,
                                                                 songs, args.policy_steps, args.policy_lr,
                                                                 anchor=base_policy, anchor_weight=args.anchor_weight)
            print(f"rig {row} (motor x{float(speed[row]):.2f}) round {round_index + 1}: world model MAE {world_mae:.1f}"
                  f"  memory-only {arms['memory_only']['curve'][-1]:.1f}  learning {arms['on_rig_learning']['curve'][-1]:.1f}"
                  f"  {time.time() - started:.0f}s", flush=True)
        results.append({"rig": row, "motor_factor": float(speed[row]),
                        **{name: arm["curve"] for name, arm in arms.items()},
                        "rollbacks": arms["on_rig_learning"].get("rollbacks", [])})
        print(f"rig {row}: memory-only {[round(x, 1) for x in arms['memory_only']['curve']]}  "
              f"on-rig learning {[round(x, 1) for x in arms['on_rig_learning']['curve']]}", flush=True)
    summary = {name: [float(np.mean([r[name][k] for r in results])) for k in range(args.rounds + 1)]
               for name in ("memory_only", "on_rig_learning")}
    pathlib.Path(args.out).write_text(json.dumps({"summary": summary, "rigs": results, **vars(args)}, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
