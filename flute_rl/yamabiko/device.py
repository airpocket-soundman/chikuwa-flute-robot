"""Environments a performer can play on, and the black-box device rule.

Every performer talks to its world through the same four calls, so a model
trained in simulation plays the real rig unchanged:

* ``reset(batch)``: home the plunger against the end stop (valve closed);
* ``step(state, pwm, valve)``: apply one 10 ms command;
* the step returns what the robot can *hear*: the heard pitch (cents) and
  whether it was valid, plus the next environment state.

Three environments implement this:

* :class:`SimulatorEnv` wraps the differentiable plant and listener.  It is
  for pre-training and for scoring (it also returns the emitted pitch).
* :class:`BlackBoxDevice` wraps the same simulator but behaves like the real
  rig: no gradient, hidden parameters, and a log of observable signals only.
  Anything that learns "on the device" may use nothing but these logs.
* world models (see :mod:`world_model`) predict heard pitch from commands and
  are differentiable; a policy can be fine-tuned inside them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .physical_plant import AudibleListener, DifferentiableMotorFlute, PhysicalPlantParameters

HOMING_STEPS = 130


@dataclass
class SimState:
    plant: object
    listener: object


class SimulatorEnv:
    """Differentiable simulator environment (pre-training and scoring)."""

    observable_only = False

    def __init__(self, plant: DifferentiableMotorFlute, parameters: PhysicalPlantParameters,
                 generator: torch.Generator | None = None):
        self.plant, self.parameters, self.generator = plant, parameters, generator
        self.listener = AudibleListener(plant.config)

    @property
    def batch(self):
        return self.parameters.torque_gain.shape[0]

    def reset(self, device, dtype=torch.float32, homing_steps=HOMING_STEPS):
        batch = self.batch
        state = self.plant.initial_state(batch, device, dtype)
        with torch.no_grad():
            state.position = torch.rand(batch, generator=self.generator, device=device, dtype=dtype) * .8
        for _ in range(homing_steps):
            state = self.plant.step(state, -torch.ones(batch, device=device, dtype=dtype), self.parameters)
        return SimState(state, self.listener.initial_state(batch, device, dtype))

    def step(self, state: SimState, pwm, valve):
        plant_state = self.plant.step(state.plant, pwm, self.parameters)
        emitted, sounding = self.plant.flute(plant_state, valve, self.parameters)
        heard, valid, listener = self.listener.step(emitted, sounding, self.parameters, state.listener,
                                                    self.generator)
        return heard, valid, SimState(plant_state, listener), emitted


@dataclass
class PlayLog:
    """Everything the real rig could record about one play."""
    target_cents: torch.Tensor
    target_voice: torch.Tensor
    pwm: torch.Tensor
    valve: torch.Tensor
    heard: torch.Tensor
    valid: torch.Tensor


@dataclass
class BlackBoxDevice:
    """A simulated rig that only exposes what the real one does.

    Commands go in, heard pitch comes out.  No gradient flows through it and
    its parameters are private; ``logs`` keeps the observable record of
    every play.  ``_score_emitted`` is for the evaluator only (the sim knows
    what the flute truly played); learners must not read it.
    """
    plant: DifferentiableMotorFlute
    _parameters: PhysicalPlantParameters
    generator: torch.Generator | None = None
    logs: list = field(default_factory=list)
    _score_emitted: list = field(default_factory=list)

    observable_only = True

    def __post_init__(self):
        self._env = SimulatorEnv(self.plant, self._parameters, self.generator)

    @property
    def batch(self):
        return self._env.batch

    def reset(self, device, dtype=torch.float32, homing_steps=HOMING_STEPS):
        with torch.no_grad():
            return self._env.reset(device, dtype, homing_steps)

    def step(self, state, pwm, valve):
        with torch.no_grad():
            heard, valid, state, emitted = self._env.step(state, pwm.detach(), valve.detach())
        return heard, valid, state, emitted

    def record(self, target_cents, target_voice, pwm, valve, heard, valid, emitted):
        self.logs.append(PlayLog(target_cents.detach(), target_voice.detach(), pwm.detach(),
                                 valve.detach(), heard.detach(), valid.detach()))
        self._score_emitted.append(emitted.detach())
