"""Minimal differentiable motor + flute plant for physical-control training.

This module is an environment, not part of the deployed inference graph.  It
intentionally models only the effects needed by the current controller gate:
torque rise, inertia, Coulomb/viscous friction, stroke end stops, and a flute
gated by an on/off valve.

Two flute models are available.  ``linear`` (the historical default) maps
position linearly to cents.  ``closed_tube`` is the physical plunger flute,
f = c / 4(L - x): its *period* 1/f is exactly linear in plunger displacement,
so a rig is described by an intercept (effective length L) and a slope
(speed of sound, i.e. temperature), while cents are not linear in x.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

A4_HZ = 440.0
SPEED_OF_SOUND_25C = 331.3 + 0.606 * 25.0


def speed_of_sound(temp_c):
    return 331.3 + 0.606 * temp_c


@dataclass(frozen=True)
class PhysicalPlantConfig:
    dt: float = 0.01
    stroke_m: float = 0.105
    low_cents: float = 700.0
    high_cents: float = 1900.0
    torque_gain: float = 6.0
    # Provisional fast-rise actuator contract: the real mechanism is rated at
    # 150 mm/s.  These values reach 90% of that speed in about 300 ms instead
    # of the previous ~370 ms, while retaining visible acceleration dynamics.
    torque_tau_s: float = 0.040
    inertia: float = 1.0
    coulomb_friction: float = 0.38
    viscous_friction: float = 1.25
    max_velocity_strokes_s: float = 1.4285714285714286  # 150 mm/s over 105 mm
    feedback_limit: float = 0.45
    flute_model: str = "linear"  # "linear" or "closed_tube"
    temp_c: float = 25.0
    # Closed tube only.  Chosen so the nominal home position (x = 0) sounds
    # low_cents; the effective length includes the open-end correction.
    effective_length_m: float = SPEED_OF_SOUND_25C / (4.0 * A4_HZ * 2.0 ** (700.0 / 1200.0))
    min_length_m: float = 0.02
    # Realism options, all off by default so older checkpoints reproduce.
    # deadband: |PWM| below this fraction produces no torque (rig.py: 0.20).
    deadband: float = 0.0
    # Audible pitch reaches the controller this many extra 10 ms steps late
    # (the measured MCU path plus a 20 ms pitch window is about 1-2 steps),
    # with per-rig jitter, Gaussian pitch noise and random drop-outs.
    hearing_delay_steps: int = 0
    hearing_delay_jitter: int = 0
    pitch_noise_cents: float = 0.0
    dropout: float = 0.0

    @classmethod
    def realistic(cls, **overrides) -> "PhysicalPlantConfig":
        """Closed tube with the rig.py deadband and hearing imperfections."""
        values = dict(flute_model="closed_tube", deadband=0.20, hearing_delay_steps=1,
                      hearing_delay_jitter=1, pitch_noise_cents=3.0, dropout=0.02)
        values.update(overrides)
        return cls(**values)

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
    # Closed tube only: effective-length error [m] and temperature error [C].
    tube_offset_m: torch.Tensor | None = None
    temp_offset_c: torch.Tensor | None = None
    deadband: torch.Tensor | None = None
    hearing_delay_steps: torch.Tensor | None = None  # int64, extra steps late


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

        closed = cfg.flute_model == "closed_tube"
        return PhysicalPlantParameters(
            torque_gain=vary(cfg.torque_gain, .35),
            torque_tau_s=vary(cfg.torque_tau_s, .55).clamp_min(cfg.dt * 1.1),
            inertia=vary(cfg.inertia, .35).clamp_min(.2),
            coulomb_friction=vary(cfg.coulomb_friction, .65).clamp_min(.02),
            viscous_friction=vary(cfg.viscous_friction, .55).clamp_min(.05),
            max_velocity_strokes_s=vary(cfg.max_velocity_strokes_s, .35).clamp_min(.2),
            # Closed tube: blowing-pressure offset, as in rig.py (+-10 cents).
            flute_offset_cents=offset(10.0 if closed else 90.0),
            flute_scale=vary(1.0, 0.0 if closed else .08),
            # rig.py spreads: tube length +-8 mm, end correction +-2 mm, +-8 C.
            tube_offset_m=offset(0.010),
            temp_offset_c=offset(8.0),
            deadband=vary(cfg.deadband, .5).clamp(0.0, .6),
            hearing_delay_steps=self._delays(batch, device, spread, generator),
        )

    def _delays(self, batch, device, spread, generator):
        cfg = self.config
        delay = torch.full((batch,), cfg.hearing_delay_steps, device=device, dtype=torch.long)
        if spread > 0 and cfg.hearing_delay_jitter > 0:
            jitter = torch.randint(-cfg.hearing_delay_jitter, cfg.hearing_delay_jitter + 1, (batch,),
                                   generator=generator, device="cpu" if generator is None else generator.device)
            delay = (delay + jitter.to(device)).clamp_min(0)
        return delay

    def speed_of_sound(self, params: PhysicalPlantParameters) -> torch.Tensor:
        temp = self.config.temp_c + (0.0 if params.temp_offset_c is None else params.temp_offset_c)
        return speed_of_sound(temp) * torch.ones_like(params.flute_offset_cents)

    def acoustic_length(self, position: torch.Tensor, params: PhysicalPlantParameters) -> torch.Tensor:
        cfg = self.config
        offset = 0.0 if params.tube_offset_m is None else params.tube_offset_m
        return (cfg.effective_length_m + offset - position * cfg.stroke_m).clamp_min(cfg.min_length_m)

    def period_s(self, position: torch.Tensor, params: PhysicalPlantParameters) -> torch.Tensor:
        """Closed-tube fundamental period (without pressure offset): linear in position."""
        return 4.0 * self.acoustic_length(position, params) / self.speed_of_sound(params)

    def cents_at(self, position: torch.Tensor, params: PhysicalPlantParameters) -> torch.Tensor:
        cfg = self.config
        if cfg.flute_model == "linear":
            return (cfg.low_cents + cfg.pitch_span_cents * position * params.flute_scale +
                    params.flute_offset_cents)
        if cfg.flute_model != "closed_tube":
            raise ValueError(f"unknown flute_model {cfg.flute_model!r}")
        frequency = 1.0 / self.period_s(position, params)
        return 1200.0 * torch.log2(frequency / A4_HZ) + params.flute_offset_cents

    def position_for_cents(self, cents: torch.Tensor, params: PhysicalPlantParameters) -> torch.Tensor:
        """Exact inverse of :meth:`cents_at` for the given rig, clamped to the stroke."""
        cfg = self.config
        if cfg.flute_model == "linear":
            return ((cents - params.flute_offset_cents - cfg.low_cents) /
                    (cfg.pitch_span_cents * params.flute_scale)).clamp(0.0, 1.0)
        frequency = A4_HZ * torch.pow(2.0, (cents - params.flute_offset_cents) / 1200.0)
        length = self.speed_of_sound(params) / (4.0 * frequency)
        offset = 0.0 if params.tube_offset_m is None else params.tube_offset_m
        return ((cfg.effective_length_m + offset - length) / cfg.stroke_m).clamp(0.0, 1.0)

    def step(self, state: PhysicalPlantState, pwm: torch.Tensor,
             params: PhysicalPlantParameters) -> PhysicalPlantState:
        cfg = self.config
        pwm = pwm.clamp(-1.0, 1.0)
        if params.deadband is not None and cfg.deadband > 0:
            band = params.deadband
            pwm = torch.sign(pwm) * torch.relu(pwm.abs() - band) / (1.0 - band)
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
        cents = self.cents_at(state.position, params)
        sounding = valve >= .5
        return cents, sounding


@dataclass
class ListenerState:
    history_cents: torch.Tensor     # (B, Q), newest last
    history_sounding: torch.Tensor  # (B, Q) bool
    last_cents: torch.Tensor


class AudibleListener:
    """What the controller can hear: delayed, noisy pitch with drop-outs.

    ``step`` receives the cents/sounding the flute emits *now* and returns the
    pitch heard now, which left the flute ``hearing_delay_steps`` ago.  The
    returned ``valid`` is false while silent or on a drop-out.
    """

    def __init__(self, config: PhysicalPlantConfig):
        self.config = config
        self.queue = max(1, config.hearing_delay_steps + config.hearing_delay_jitter + 1)

    def initial_state(self, batch, device, dtype=torch.float32) -> ListenerState:
        return ListenerState(torch.zeros(batch, self.queue, device=device, dtype=dtype),
                             torch.zeros(batch, self.queue, device=device, dtype=torch.bool),
                             torch.zeros(batch, device=device, dtype=dtype))

    def step(self, cents, sounding, params: PhysicalPlantParameters, state: ListenerState,
             generator: torch.Generator | None = None):
        cfg = self.config
        history_cents = torch.cat([state.history_cents[:, 1:], cents[:, None]], 1)
        history_sounding = torch.cat([state.history_sounding[:, 1:], sounding[:, None]], 1)
        delay = (torch.zeros_like(cents, dtype=torch.long) if params.hearing_delay_steps is None
                 else params.hearing_delay_steps.clamp(0, self.queue - 1))
        index = (self.queue - 1 - delay)[:, None]
        heard = history_cents.gather(1, index)[:, 0]
        valid = history_sounding.gather(1, index)[:, 0]
        if cfg.pitch_noise_cents > 0:
            noise = torch.randn(heard.shape, generator=generator, device=heard.device, dtype=heard.dtype)
            heard = heard + cfg.pitch_noise_cents * noise
        if cfg.dropout > 0:
            keep = torch.rand(heard.shape, generator=generator, device=heard.device) >= cfg.dropout
            valid = valid & keep
        heard = torch.where(valid, heard, state.last_cents)
        return heard, valid, ListenerState(history_cents, history_sounding, heard)
