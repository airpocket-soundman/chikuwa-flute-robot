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


def save_logs(path, logs):
    """Write play logs as one .npz (what the real rig can record, nothing else)."""
    import numpy as np
    arrays = {}
    for index, log in enumerate(logs):
        for name, value in vars(log).items():
            arrays[f"{index}/{name}"] = value.detach().cpu().numpy()
    np.savez_compressed(path, count=len(logs), **arrays)


def load_logs(path, device="cpu"):
    import numpy as np
    data = np.load(path)
    logs = []
    for index in range(int(data["count"])):
        fields = {name: torch.as_tensor(data[f"{index}/{name}"], device=device)
                  for name in ("target_cents", "target_voice", "pwm", "valve", "heard", "valid")}
        logs.append(PlayLog(**fields))
    return logs


class RealDevice:
    """The real rig behind the same reset/step/record interface as the black box.

    Wraps :class:`flute_rl.yamabiko.hw.RealRig` (one rig, batch of 1, real
    time at 100 Hz).  Heard pitch comes from the 20 ms YIN window of the
    microphone; ``valid`` is whether a pitch was found.  There is no true
    emitted pitch on a real rig, so the 4th value returned by ``step`` is the
    heard pitch again.  ``reset`` homes the plunger (pull against the end
    stop with the valve shut), as scripts/yamabiko_collect.py does.
    """

    observable_only = True
    batch = 1
    device = "cpu"

    def __init__(self, rig, homing_seconds: float = 1.3, homing_pwm: float = -1.0):
        self.rig, self.homing_seconds, self.homing_pwm = rig, homing_seconds, homing_pwm
        self.logs, self._score_emitted = [], []

    def reset(self, device="cpu", dtype=torch.float32, homing_steps=None):
        self.rig.resync()
        steps = int(round(self.homing_seconds / 0.01)) if homing_steps is None else int(homing_steps)
        for _ in range(steps):
            self.rig.step(self.homing_pwm, False)
        return None

    def step(self, state, pwm, valve):
        out = self.rig.step(float(pwm.reshape(-1)[0]), bool(float(valve.reshape(-1)[0]) >= .5))
        heard = torch.tensor([float(out["heard"][0])], dtype=torch.float32)
        valid = torch.isfinite(heard)
        heard = torch.where(valid, heard, torch.zeros_like(heard))
        return heard, valid, None, heard

    def record(self, target_cents, target_voice, pwm, valve, heard, valid, emitted):
        self.logs.append(PlayLog(target_cents.detach(), target_voice.detach(), pwm.detach(),
                                 valve.detach(), heard.detach(), valid.detach()))
        self._score_emitted.append(emitted.detach())
