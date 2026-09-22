"""Black-box policy training against the deterministic motor/flute simulator.

Position Planner is the musical feedforward path. The motor controller sees
only its target trajectory and past PWM. Acoustic feedback additionally sees
the simulator's public sound output. Neither network receives position,
velocity, torque, physical parameters, nor state labels. Hidden plant values
are read only after training for diagnostic evaluation.
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

from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.staged_nn import (AcousticFeedbackResidual, MotorAudioWorldModel,
                                         MotorTrajectoryController)  # noqa: E402


def make_plans(rng: np.random.Generator, batch: int, steps: int, device):
    plan = np.zeros((batch, steps), np.float32)
    voice = np.zeros((batch, steps), np.float32)
    for row in range(batch):
        t, current, first = 0, float(rng.uniform(.12, .88)), True
        while t < steps:
            stop = min(steps, t + int(rng.integers(24, 65)))
            sounding = False if first else bool(rng.random() > .22)
            first = False
            if sounding:
                current = float(np.clip(current + rng.choice([-1, 1]) * rng.uniform(.08, .32), .06, .94))
            plan[row, t:stop], voice[row, t:stop] = current, float(sounding)
            t = stop
    return torch.from_numpy(plan).to(device), torch.from_numpy(voice).to(device)


def discounted_advantage(reward: torch.Tensor, gamma: float = .985) -> torch.Tensor:
    """Reward-to-go with ensemble baseline; it uses no hidden plant state."""
    returns = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[0], device=reward.device, dtype=reward.dtype)
    for t in range(reward.shape[1] - 1, -1, -1):
        running = reward[:, t] + gamma * running
        returns[:, t] = running
    advantage = returns - returns.mean(0, keepdim=True)
    return advantage / advantage.std().clamp_min(1e-4)


def make_excitation(rng, batch, steps, device):
    pwm = np.zeros((batch, steps), np.float32)
    for row in range(batch):
        t = 0
        while t < steps:
            stop = min(steps, t + int(rng.integers(4, 31)))
            pwm[row, t:stop] = rng.uniform(-1, 1)
            t = stop
    return torch.from_numpy(pwm).to(device)


def train_world_model(world, plant, rng, args):
    """System identification from public PWM/audio pairs only."""
    world.train(); optimizer = torch.optim.AdamW(world.parameters(), lr=args.lr)
    for step in range(1, args.world_steps + 1):
        commands = make_excitation(rng, args.batch, args.sequence_steps, args.device)
        params = plant.parameters(args.batch, args.device, spread=0.0)
        physical = plant.initial_state(args.batch, args.device)
        hidden = world.initial_state(args.batch, args.device)
        predictions, targets = [], []
        for t in range(args.sequence_steps):
            prediction, hidden = world.step(commands[:, t], hidden)
            with torch.no_grad():
                physical = plant.step(physical, commands[:, t], params)
                cents, _ = plant.flute(physical, torch.ones(args.batch, device=args.device), params)
                target = (cents - plant.config.low_cents) / plant.config.pitch_span_cents
            predictions.append(prediction); targets.append(target)
        predictions, targets = torch.stack(predictions, 1), torch.stack(targets, 1)
        loss = F.smooth_l1_loss(predictions, targets, beta=.02)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(world.parameters(), 2.0); optimizer.step()
        if step == 1 or step % 50 == 0:
            mae = (predictions - targets).abs().mean() * plant.config.pitch_span_cents
            print(f"world      {step:4d} audio MAE {float(mae.detach()):.2f} cent", flush=True)


def train_controller_in_world(controller, world, rng, args):
    """Optimize plan tracking through the learned I/O relation, not the plant."""
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    controller.train(); optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr)
    for step in range(1, args.surrogate_steps + 1):
        plan, voice = make_plans(rng, args.batch, args.sequence_steps, args.device)
        control_state = controller.initial_state(args.batch, args.device)
        world_state = world.initial_state(args.batch, args.device)
        previous = torch.zeros(args.batch, device=args.device)
        pitches, actions, valves = [], [], []
        for t in range(args.sequence_steps):
            action, valve, control_state = controller.step(plan, voice, previous, control_state, t)
            pitch, world_state = world.step(action, world_state)
            pitches.append(pitch); actions.append(action); valves.append(valve)
            previous = action
        pitches, actions = torch.stack(pitches, 1), torch.stack(actions, 1)
        valves = torch.stack(valves, 1)
        tracking = F.smooth_l1_loss(pitches[:, 10:], plan[:, 10:], beta=.02)
        valve_loss = F.binary_cross_entropy_with_logits(valves, voice)
        smooth = (actions[:, 1:] - actions[:, :-1]).square().mean()
        loss = tracking + .08 * valve_loss + .002 * smooth
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 2.0); optimizer.step()
        if step == 1 or step % 50 == 0:
            mae = (pitches[:, 10:] - plan[:, 10:]).abs().mean() * 1200
            print(f"surrogate  {step:4d} pitch MAE {float(mae.detach()):.2f} cent", flush=True)


def rollout(controller, feedback, plant, plan, voice, params, feedback_enabled: bool,
            controller_std: float = 0.0, feedback_std: float = 0.0,
            initial_feedback_state=None):
    batch, steps = plan.shape
    physical = plant.initial_state(batch, plan.device, plan.dtype)
    control_state = controller.initial_state(batch, plan.device, plan.dtype)
    feedback_state = (initial_feedback_state if initial_feedback_state is not None else
                      (feedback.initial_state(batch, plan.device, plan.dtype) if feedback is not None else None))
    previous_pwm = torch.zeros(batch, device=plan.device, dtype=plan.dtype)
    previous_error = torch.zeros_like(previous_pwm)
    previous_valve = torch.zeros_like(previous_pwm)
    previous_heard = torch.zeros_like(previous_pwm)
    positions, velocities, torques, pitches, soundings = [], [], [], [], []
    pwms, bases, residuals, valve_logits, rewards = [], [], [], [], []
    base_log_probs, residual_log_probs = [], []
    for t in range(steps):
        base_mean, valve_logit, control_state = controller.step(
            plan, voice, previous_pwm, control_state, t)
        if controller_std > 0:
            distribution = torch.distributions.Normal(base_mean, controller_std)
            sample = distribution.sample()
            base_log_probs.append(distribution.log_prob(sample))
            base_pwm = sample.clamp(-1.0, 1.0)
        else:
            base_pwm = base_mean
        heard_cents, heard_sounding = plant.flute(physical, previous_valve, params)
        heard_pitch = (heard_cents - plant.config.low_cents) / plant.config.pitch_span_cents
        observable_heard = torch.where(heard_sounding, heard_pitch, previous_heard)
        if feedback is not None and feedback_enabled:
            residual_mean, previous_error, feedback_state = feedback.step(
                plan[:, t], observable_heard, heard_sounding, voice[:, t], base_pwm,
                previous_pwm, previous_error, feedback_state,
                target_future=plan[:, min(t + 20, steps - 1)], previous_heard=previous_heard)
            if feedback_std > 0:
                distribution = torch.distributions.Normal(residual_mean, feedback_std)
                sample = distribution.sample()
                residual_log_probs.append(distribution.log_prob(sample))
                residual = sample.clamp(-feedback.limit, feedback.limit)
            else:
                residual = residual_mean
        else:
            residual = torch.zeros_like(base_pwm)
            previous_error = torch.where(heard_sounding, plan[:, t] - heard_pitch,
                                         torch.zeros_like(base_pwm))
        pwm = (base_pwm + residual).clamp(-1.0, 1.0)
        valve_probability = torch.sigmoid(valve_logit)
        # Detaches make the deterministic simulator a black-box environment.
        physical = plant.step(physical, pwm.detach(), params)
        cents, sounding = plant.flute(physical, valve_probability.detach(), params)
        normalized_pitch = (cents - plant.config.low_cents) / plant.config.pitch_span_cents
        desired_voice = voice[:, t].bool()
        valid = desired_voice & sounding
        pitch_cost = torch.where(valid, (normalized_pitch - plan[:, t]).abs(),
                                 desired_voice.to(plan.dtype))
        false_sound_cost = ((~desired_voice) & sounding).to(plan.dtype) * .25
        reward = -pitch_cost - false_sound_cost - .003 * pwm.detach().square()
        positions.append(physical.position); velocities.append(physical.velocity); torques.append(physical.torque)
        pitches.append(cents); soundings.append(sounding); pwms.append(pwm); bases.append(base_pwm)
        residuals.append(residual); valve_logits.append(valve_logit); rewards.append(reward.detach())
        previous_pwm, previous_valve = pwm.detach(), valve_probability.detach()
        previous_heard = torch.where(heard_sounding, heard_pitch, previous_heard).detach()
    stack = lambda values: torch.stack(values, 1)
    result = {"position": stack(positions), "velocity": stack(velocities), "torque": stack(torques),
              "pitch_cents": stack(pitches), "sounding": stack(soundings), "pwm": stack(pwms),
              "base_pwm": stack(bases), "residual_pwm": stack(residuals),
              "valve_logits": stack(valve_logits), "reward": stack(rewards),
              "feedback_state": feedback_state}
    if base_log_probs:
        result["base_log_prob"] = stack(base_log_probs)
    if residual_log_probs:
        result["residual_log_prob"] = stack(residual_log_probs)
    return result


def pretrain_from_acoustic_teacher(controller, plant, rng, args):
    """Distil a probing controller that observes only the public flute pitch.

    The teacher deliberately opens the simulated valve while collecting its
    demonstrations. It estimates motion from successive audible pitches; it
    never reads position, velocity, torque, parameters, or simulator equations.
    The deployed student receives none of those pitches in its feedforward path.
    """
    controller.train()
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=1e-6)
    for step in range(1, args.pretrain_steps + 1):
        plan, voice = make_plans(rng, args.batch, args.sequence_steps, args.device)
        # The feedforward path learns the nominal instrument. Instrument and
        # motor deviations are intentionally left for the adaptive feedback.
        params = plant.parameters(args.batch, args.device, spread=0.0)
        physical = plant.initial_state(args.batch, args.device)
        previous_heard = torch.zeros(args.batch, device=args.device)
        teacher_actions = []
        with torch.no_grad():
            for t in range(args.sequence_steps):
                cents, _ = plant.flute(physical, torch.ones(args.batch, device=args.device), params)
                heard = (cents - plant.config.low_cents) / plant.config.pitch_span_cents
                audible_velocity = (heard - previous_heard) / plant.config.dt if t else torch.zeros_like(heard)
                desired = plan[:, min(t + 30, args.sequence_steps - 1)]
                action = (6.0 * (desired - heard) - 1.0 * audible_velocity).clamp(-1.0, 1.0)
                teacher_actions.append(action)
                physical = plant.step(physical, action, params)
                previous_heard = heard
        teacher_actions = torch.stack(teacher_actions, 1)
        hidden = controller.initial_state(args.batch, args.device)
        previous = torch.zeros(args.batch, device=args.device)
        predictions, valves = [], []
        for t in range(args.sequence_steps):
            action, valve_logit, hidden = controller.step(plan, voice, previous, hidden, t)
            predictions.append(action); valves.append(valve_logit)
            previous = teacher_actions[:, t]
        predictions, valves = torch.stack(predictions, 1), torch.stack(valves, 1)
        action_loss = F.smooth_l1_loss(predictions, teacher_actions, beta=.08)
        valve_loss = F.binary_cross_entropy_with_logits(valves, voice)
        loss = action_loss + .12 * valve_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 2.0); optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"acoustic   {step:4d} action {float(action_loss.detach()):.5f}", flush=True)


def train_blackbox_dagger(controller, plant, rng, args):
    """Relabel policy-visited trajectories using only audible plant output."""
    controller.train()
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=1e-6)
    for step in range(1, args.dagger_steps + 1):
        plan, voice = make_plans(rng, args.batch, args.sequence_steps, args.device)
        params = plant.parameters(args.batch, args.device, spread=0.0)
        physical = plant.initial_state(args.batch, args.device)
        hidden = controller.initial_state(args.batch, args.device)
        previous_pwm = torch.zeros(args.batch, device=args.device)
        previous_heard = torch.zeros_like(previous_pwm)
        predicted, teachers, valves = [], [], []
        for t in range(args.sequence_steps):
            action, valve_logit, hidden = controller.step(plan, voice, previous_pwm, hidden, t)
            # Identification probe: public flute output with valve open. The
            # teacher has no access to any PhysicalPlantState member.
            with torch.no_grad():
                cents, _ = plant.flute(physical, torch.ones_like(action), params)
                heard = (cents - plant.config.low_cents) / plant.config.pitch_span_cents
                audible_velocity = ((heard - previous_heard) / plant.config.dt
                                     if t else torch.zeros_like(heard))
                desired = plan[:, min(t + 30, args.sequence_steps - 1)]
                teacher = (6.0 * (desired - heard) - 1.0 * audible_velocity).clamp(-1.0, 1.0)
                physical = plant.step(physical, action.detach(), params)
            predicted.append(action); teachers.append(teacher); valves.append(valve_logit)
            previous_pwm, previous_heard = action.detach(), heard
        predicted = torch.stack(predicted, 1); teachers = torch.stack(teachers, 1)
        valves = torch.stack(valves, 1)
        action_loss = F.smooth_l1_loss(predicted, teachers, beta=.08)
        valve_loss = F.binary_cross_entropy_with_logits(valves, voice)
        loss = action_loss + .1 * valve_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 2.0); optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"dagger     {step:4d} action {float(action_loss.detach()):.5f}", flush=True)


def train_controller(controller, plant, rng, args):
    """Policy gradient on public audio reward; no state labels or gradients."""
    controller.train()
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=1e-6)
    for step in range(1, args.controller_steps + 1):
        plan, voice = make_plans(rng, args.batch, args.sequence_steps, args.device)
        params = plant.parameters(args.batch, args.device, spread=0.0)
        out = rollout(controller, None, plant, plan, voice, params, False, args.controller_std)
        advantage = discounted_advantage(out["reward"])
        policy_loss = -(out["base_log_prob"] * advantage).mean()
        valve_loss = F.binary_cross_entropy_with_logits(out["valve_logits"], voice)
        smooth = (out["base_pwm"][:, 1:] - out["base_pwm"][:, :-1]).square().mean()
        loss = policy_loss + .12 * valve_loss + .003 * smooth
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 2.0); optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"controller {step:4d} reward {float(out['reward'].mean()):.4f} policy {float(policy_loss.detach()):.5f}", flush=True)


def train_feedback(controller, feedback, plant, rng, args):
    """Residual policy gets audible pitch only while the base policy is frozen."""
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    feedback.train()
    optimizer = torch.optim.AdamW(feedback.parameters(), lr=args.lr, weight_decay=1e-6)
    for step in range(1, args.feedback_steps + 1):
        plan, voice = make_plans(rng, args.batch, args.sequence_steps, args.device)
        params = plant.parameters(args.batch, args.device, spread=1.0)
        state = None; repetitions = []
        for _ in range(args.practice_repetitions):
            out = rollout(controller, feedback, plant, plan, voice, params, True,
                          feedback_std=args.feedback_std, initial_feedback_state=state)
            state = out["feedback_state"]
            repetitions.append(out)
            # A new performance starts with no fast motor-motion memory, while
            # the slow instrument context survives as learned practice memory.
            state = torch.cat([torch.zeros_like(state[:, :feedback.hidden]),
                               state[:, feedback.hidden:]], 1)
        rewards = torch.cat([x["reward"] for x in repetitions], 1)
        log_probs = torch.cat([x["residual_log_prob"] for x in repetitions], 1)
        advantage = discounted_advantage(rewards)
        # Meta-RL term: reward a later performance specifically for improving
        # over the first performance on the same deterministic instrument.
        improvement = (repetitions[-1]["reward"].mean(1) -
                       repetitions[0]["reward"].mean(1))
        improvement = ((improvement - improvement.mean()) /
                       improvement.std().clamp_min(1e-4))
        final_start = (args.practice_repetitions - 1) * args.sequence_steps
        advantage[:, final_start:] += .75 * improvement[:, None]
        policy_loss = -(log_probs * advantage).mean()
        residual = torch.cat([x["residual_pwm"] for x in repetitions], 1)
        effort = residual.square().mean()
        smooth = (residual[:, 1:] - residual[:, :-1]).square().mean()
        loss = policy_loss + .005 * (effort + smooth)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(feedback.parameters(), 2.0); optimizer.step()
        if step == 1 or step % 50 == 0:
            first = float(repetitions[0]["reward"].mean())
            last = float(repetitions[-1]["reward"].mean())
            print(f"feedback   {step:4d} reward {first:.4f}->{last:.4f} policy {float(policy_loss.detach()):.5f}", flush=True)


@torch.inference_mode()
def evaluate(controller, feedback, plant, seed: int, args):
    rng, batches = np.random.default_rng(seed), []
    for index in range(4):
        plan, voice = make_plans(rng, args.eval_batch, args.eval_steps, args.device)
        generator = torch.Generator(device=args.device).manual_seed(seed + index)
        params = plant.parameters(args.eval_batch, args.device, spread=1.0, generator=generator)
        base = rollout(controller, feedback, plant, plan, voice, params, False)
        closed_first = rollout(controller, feedback, plant, plan, voice, params, True)
        closed = closed_first
        for _ in range(1, args.practice_repetitions):
            carry = closed["feedback_state"]
            carry = torch.cat([torch.zeros_like(carry[:, :feedback.hidden]),
                               carry[:, feedback.hidden:]], 1)
            closed = rollout(controller, feedback, plant, plan, voice, params, True,
                             initial_feedback_state=carry)
        target = plant.config.low_cents + plant.config.pitch_span_cents * plan
        base_abs = torch.where(voice.bool() & base["sounding"], (base["pitch_cents"] - target).abs(), torch.nan)
        first_abs = torch.where(voice.bool() & closed_first["sounding"],
                                (closed_first["pitch_cents"] - target).abs(), torch.nan)
        closed_abs = torch.where(voice.bool() & closed["sounding"], (closed["pitch_cents"] - target).abs(), torch.nan)
        batches.append((base, closed, base_abs, closed_abs, torch.nanmean(base_abs, 1),
                        torch.nanmean(closed_abs, 1), first_abs))
    base_abs = torch.cat([x[2].flatten() for x in batches]); base_abs = base_abs[torch.isfinite(base_abs)]
    closed_abs = torch.cat([x[3].flatten() for x in batches]); closed_abs = closed_abs[torch.isfinite(closed_abs)]
    base_song, closed_song = torch.cat([x[4] for x in batches]), torch.cat([x[5] for x in batches])
    first_abs = torch.cat([x[6].flatten() for x in batches]); first_abs = first_abs[torch.isfinite(first_abs)]
    plan, voice = make_plans(np.random.default_rng(seed + 999), args.eval_batch, args.eval_steps, args.device)
    nominal = rollout(controller, feedback, plant, plan, voice, plant.parameters(args.eval_batch, args.device), False)
    position_error = (nominal["position"] - plan).abs()
    mask = (torch.arange(args.eval_steps, device=args.device)[None] >= 20).expand_as(plan)
    result = {
        "motor_position_mae_percent_stroke": float(position_error[mask].mean() * 100),
        "motor_position_p95_percent_stroke": float(torch.quantile(position_error[mask], .95) * 100),
        "feedforward_pitch_mae_cents": float((position_error[voice.bool()] * plant.config.pitch_span_cents).mean()),
        "randomized_base_pitch_mae_cents": float(base_abs.mean()),
        "feedback_pitch_mae_cents": float(closed_abs.mean()),
        "feedback_first_play_pitch_mae_cents": float(first_abs.mean()),
        "practice_adaptation_improvement_fraction": 1.0 - float(closed_abs.mean() / first_abs.mean()),
        "feedback_improvement_fraction": 1.0 - float(closed_abs.mean() / base_abs.mean()),
        "feedback_nonworse_rig_fraction": float((closed_song <= base_song * 1.05).float().mean()),
        "feedback_residual_rms": float(torch.cat([x[1]["residual_pwm"].flatten() for x in batches]).square().mean().sqrt()),
    }
    result["motor_controller_pass"] = result["motor_position_mae_percent_stroke"] <= 8 and result["feedforward_pitch_mae_cents"] <= 100
    result["feedback_pass"] = result["feedback_pitch_mae_cents"] <= 80 and result["feedback_improvement_fraction"] >= .20 and result["feedback_nonworse_rig_fraction"] >= .75
    result["pass"] = result["motor_controller_pass"] and result["feedback_pass"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init")
    ap.add_argument("--out", default="runs/yamabiko_physical_blackbox_v1.pt")
    ap.add_argument("--report", default="runs/yamabiko_physical_blackbox_v1_report.json")
    ap.add_argument("--world-steps", type=int, default=600)
    ap.add_argument("--surrogate-steps", type=int, default=600)
    ap.add_argument("--pretrain-steps", type=int, default=0)
    ap.add_argument("--dagger-steps", type=int, default=0)
    ap.add_argument("--controller-steps", type=int, default=0)
    ap.add_argument("--feedback-steps", type=int, default=900)
    ap.add_argument("--practice-repetitions", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sequence-steps", type=int, default=180)
    ap.add_argument("--eval-batch", type=int, default=48)
    ap.add_argument("--eval-steps", type=int, default=260)
    ap.add_argument("--controller-std", type=float, default=.16)
    ap.add_argument("--feedback-std", type=float, default=.07)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=29431)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    cfg, plant = PhysicalPlantConfig(), DifferentiableMotorFlute()
    world = MotorAudioWorldModel().to(args.device)
    controller = MotorTrajectoryController().to(args.device)
    feedback = AcousticFeedbackResidual(limit=cfg.feedback_limit).to(args.device)
    if args.init:
        initial = torch.load(args.init, map_location=args.device)
        controller.load_state_dict(initial["controller"]); feedback.load_state_dict(initial["feedback"])
        if "world_model" in initial:
            world.load_state_dict(initial["world_model"])
    train_world_model(world, plant, rng, args)
    train_controller_in_world(controller, world, rng, args)
    pretrain_from_acoustic_teacher(controller, plant, rng, args)
    train_blackbox_dagger(controller, plant, rng, args)
    train_controller(controller, plant, rng, args)
    base_path = pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + "_base.pt")
    torch.save({"format": "yamabiko-physical-base-v2", "plant_config": vars(cfg),
                "world_model": world.state_dict(), "controller": controller.state_dict(),
                "feedback": feedback.state_dict()}, base_path)
    print(f"saved base {base_path}", flush=True)
    train_feedback(controller, feedback, plant, rng, args)
    metrics = evaluate(controller.eval(), feedback.eval(), plant, args.seed + 100_000, args)
    metrics.update({"seed": args.seed + 100_000, "split": "frozen-randomized-deterministic-physical-v1",
                    "training": "learned-pwm-audio-world-model-plus-adaptive-policy-gradient",
                    "position_planner_is_feedforward": True,
                    "simulator_deterministic_with_fixed_parameters": True,
                    "network_received_simulator_internal_state": False, "real_rig_validated": False})
    checkpoint = {"format": "yamabiko-physical-blackbox-v2", "plant_config": vars(cfg),
                  "world_model": world.state_dict(), "controller": controller.state_dict(),
                  "feedback": feedback.state_dict(), "metrics": metrics}
    torch.save(checkpoint, args.out)
    pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__":
    main()
