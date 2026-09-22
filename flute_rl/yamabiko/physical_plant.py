"""Minimal differentiable motor + linear-flute plant for physical-control training.

This module is an environment, not part of the deployed inference graph.  It
intentionally models only the effects needed by the current controller gate:
torque rise, inertia, Coulomb/viscous friction, stroke end stops, and a linear
position-to-musical-pitch flute gated by an on/off valve.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhysicalPlantConfig:
    dt: float = 0.01
    stroke_m: float = 0.105
    low_cents: float = 700.0
    high_cents: float = 1900.0
    torque_gain: float = 5.0
    torque_tau_s: float = 0.055
    inertia: float = 1.0
    coulomb_friction: float = 0.38
    viscous_friction: float = 1.25
    max_velocity_strokes_s: float = 1.35
    feedback_limit: float = 0.45

    @property
    def pitch_span_cents(self) -> float:
        return self.high_cents - self.low_cents

    def position_for_cents(self, cents: torch.Tensor) -> torch.Tensor:
        return ((cents - self.low_cents) / self.pitch_span_cents).clamp(0.0, 1.0)

    def cents_for_position(self, position: torch.Tensor) -> torch.Tensor:
        return self.low_cents + self.pitch_span_cents * position


@dataclass
class PhysicalPlantState:
    position: torch.Tensor
    velocity: torch.Tensor
    torque: torch.Tensor


@dataclass
class PhysicalPlantParameters:
    torque_gain: torch.Tensor
    torque_tau_s: torch.Tensor
    inertia: torch.Tensor
    coulomb_friction: torch.Tensor
    viscous_friction: torch.Tensor
    max_velocity_strokes_s: torch.Tensor
    flute_offset_cents: torch.Tensor
    flute_scale: torch.Tensor


class DifferentiableMotorFlute:
    """Deterministic batched 100 Hz plant used only for training/evaluation.

    ``step`` contains no noise or learned component: initial state, PWM and a
    parameter set uniquely determine the complete trajectory.  Domain
    randomization happens only when selecting a fixed parameter set before a
    rollout; it does not make the motor dynamics stochastic within a rollout.
    """

    def __init__(self, config: PhysicalPlantConfig = PhysicalPlantConfig()):
        self.config = config

    def initial_state(self, batch: int, device, dtype=torch.float32) -> PhysicalPlantState:
        z = torch.zeros(batch, device=device, dtype=dtype)
        return PhysicalPlantState(z, z.clone(), z.clone())

    def parameters(self, batch: int, device, dtype=torch.float32, spread: float = 0.0,
                   generator: torch.Generator | None = None) -> PhysicalPlantParameters:
        cfg = self.config

        def vary(base: float, amount: float) -> torch.Tensor:
            if spread <= 0:
                return torch.full((batch,), base, device=device, dtype=dtype)
            r = torch.rand(batch, device=device, dtype=dtype, generator=generator) * 2.0 - 1.0
            return base * (1.0 + amount * spread * r)

        def offset(amount: float) -> torch.Tensor:
            if spread <= 0:
                return torch.zeros(batch, device=device, dtype=dtype)
            r = torch.rand(batch, device=device, dtype=dtype, generator=generator) * 2.0 - 1.0
            return amount * spread * r

        return PhysicalPlantParameters(
            torque_gain=vary(cfg.torque_gain, .35),
            torque_tau_s=vary(cfg.torque_tau_s, .55).clamp_min(cfg.dt * 1.1),
            inertia=vary(cfg.inertia, .35).clamp_min(.2),
            coulomb_friction=vary(cfg.coulomb_friction, .65).clamp_min(.02),
            viscous_friction=vary(cfg.viscous_friction, .55).clamp_min(.05),
            max_velocity_strokes_s=vary(cfg.max_velocity_strokes_s, .35).clamp_min(.2),
            flute_offset_cents=offset(90.0),
            flute_scale=vary(1.0, .08),
        )

    def step(self, state: PhysicalPlantState, pwm: torch.Tensor,
             params: PhysicalPlantParameters) -> PhysicalPlantState:
        cfg = self.config
        pwm = pwm.clamp(-1.0, 1.0)
        alpha = (cfg.dt / params.torque_tau_s).clamp(max=1.0)
        torque = state.torque + alpha * (params.torque_gain * pwm - state.torque)
        # Smooth Coulomb friction keeps the plant differentiable around zero.
        friction = (params.coulomb_friction * torch.tanh(state.velocity / .015) +
                    params.viscous_friction * state.velocity)
        acceleration = (torque - friction) / params.inertia
        velocity = (state.velocity + cfg.dt * acceleration).clamp(
            -params.max_velocity_strokes_s, params.max_velocity_strokes_s)
        raw_position = state.position + cfg.dt * velocity
        position = raw_position.clamp(0.0, 1.0)
        hit_low = (raw_position <= 0.0) & (velocity < 0.0)
        hit_high = (raw_position >= 1.0) & (velocity > 0.0)
        velocity = torch.where(hit_low | hit_high, torch.zeros_like(velocity), velocity)
        return PhysicalPlantState(position, velocity, torque)

    def flute(self, state: PhysicalPlantState, valve: torch.Tensor,
              params: PhysicalPlantParameters) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        cents = (cfg.low_cents + cfg.pitch_span_cents * state.position * params.flute_scale +
                 params.flute_offset_cents)
        sounding = valve >= .5
        return cents, sounding
