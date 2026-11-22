"""A performer whose network does the feedback too: no PD loop, no pitch observer.

:class:`~.residual_performer.ResidualTwinPerformer` keeps a hand-written skeleton (a PD loop on a position
estimate that a pitch observer corrects) and lets a network nudge it.  On the first real rig the skeleton
dithers: every small correction has to jump the static-friction deadband, so the plunger shakes around the
note, and the motor's noise then hides the flute.  Here one recurrent network maps, every 10 ms,

* the target track ahead (WINDOW), normalised by the rig's feature range,
* what it heard on the previous step (pitch, whether it heard anything, target minus heard),
* its own previous command,
* the twin's open-loop guess of position and velocity (the motor model driven by the commands only; it
  is never corrected from pitch: that correction is the network's job),

to the PWM directly.  It starts by imitating a residual performer (``imitate``) and is then trained
through the simulator on pitch error plus a smoothness cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .deterministic_pipeline import EncoderlessDeterministicPerformer
from .features import WINDOW
from .physical_plant import AudibleListener, DifferentiableMotorFlute

FEATURES = 2 * len(WINDOW) + 7     # window, heard (3), previous pwm, twin position / velocity, voice now


@dataclass(frozen=True)
class NNConfig:
    hidden: int = 64
    # also the twin's stroke position of the target now and of what was heard (closed-tube inverse), so the
    # feedback "heard here, aiming there" is a difference the network does not have to learn to compute
    positions: bool = False

    @property
    def features(self) -> int:
        return FEATURES + (2 if self.positions else 0)


class NNTwinPerformer(nn.Module):
    def __init__(self, plant: DifferentiableMotorFlute, config: NNConfig = NNConfig()):
        super().__init__()
        self.config, self.plant = config, plant
        self.motor = EncoderlessDeterministicPerformer(plant).model     # the twin's motor model, open loop
        self.inp = nn.Linear(config.features, config.hidden)
        self.cell = nn.GRUCell(config.hidden, config.hidden)
        self.head = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.SiLU(), nn.Linear(config.hidden, 1))

    # ----- one step --------------------------------------------------------------------------------
    def features(self, cents, voice, t, heard, valid, prev_pwm, estimate, twin=None):
        normalize = self.plant.config.normalize
        steps = cents.shape[1]
        ids = [min(t + k, steps - 1) for k in WINDOW]
        dtype = cents.dtype
        heard_n = torch.where(valid, normalize(heard), torch.zeros_like(heard))
        gap = torch.where(valid, normalize(cents[:, t]) - heard_n, torch.zeros_like(heard))
        parts = [normalize(cents[:, ids]), voice[:, ids].to(dtype),
                 torch.stack([heard_n, valid.to(dtype), gap, prev_pwm, estimate.position,
                              estimate.velocity, voice[:, t].to(dtype)], 1)]
        if self.config.positions:
            aim = self.plant.position_for_cents(cents[:, t], twin)
            at = torch.where(valid, self.plant.position_for_cents(heard, twin), torch.zeros_like(heard))
            parts.append(torch.stack([aim, at], 1))
        return torch.cat(parts, 1)

    def act(self, x, fast):
        fast = self.cell(torch.tanh(self.inp(x)), fast)
        return torch.tanh(self.head(fast)[:, 0]), fast

    def start(self, batch, device, dtype, twin):
        estimate = self.motor.initial_state(batch, device, dtype)
        estimate.torque = -twin.torque_gain.to(dtype)                     # homing leaves it pressed home
        return estimate, torch.zeros(batch, self.config.hidden, device=device, dtype=dtype)

    # ----- closed loop on the simulator ------------------------------------------------------------
    def perform(self, plant, cents, voice, parameters, twin, generator=None, start_position=None):
        """Play ``cents``/``voice`` (B, T) on the simulated rigs ``parameters``; ``twin`` is the fitted rig
        whose motor model gives the open-loop guess.  Returns pitch_cents (emitted) and pwm, (B, T)."""
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        listener = AudibleListener(plant.config)
        state = plant.initial_state(batch, device, dtype)
        if start_position is not None:
            state.position = start_position
        for _ in range(130):                                              # homing, as on the rig
            state = plant.step(state, -torch.ones(batch, device=device, dtype=dtype), parameters)
        heard_state = listener.initial_state(batch, device, dtype)
        estimate, fast = self.start(batch, device, dtype, twin)
        heard = torch.zeros(batch, device=device, dtype=dtype)
        valid = torch.zeros(batch, device=device, dtype=torch.bool)
        pwm = torch.zeros(batch, device=device, dtype=dtype)
        band = (twin.deadband if twin.deadband is not None else torch.zeros(batch, device=device)).to(dtype)
        played, pwms = [], []
        for t in range(steps):
            x = self.features(cents, voice, t, heard, valid, pwm, estimate, twin)
            pwm, fast = self.act(x, fast)
            state = plant.step(state, pwm, parameters)
            estimate = self.motor.step(estimate, pwm, twin)
            emitted, sounding = plant.flute(state, voice[:, t].to(dtype), parameters)
            heard, valid, heard_state = listener.step(emitted, sounding, parameters, heard_state, generator)
            if plant.config.motor_masking > 0:
                lost = torch.rand(valid.shape, generator=generator, device=device) < plant.config.motor_masking
                valid = valid & ~((pwm.abs() > band) & lost)
            played.append(emitted); pwms.append(pwm)
        return {"pitch_cents": torch.stack(played, 1), "pwm": torch.stack(pwms, 1)}

    # ----- imitation of a recorded play ------------------------------------------------------------
    def imitate(self, cents, voice, heard, valid, pwm, twin):
        """The network's commands along a recorded closed-loop play (teacher forcing): it sees what the
        teacher saw and did, step by step.  heard/valid[t] is what was heard after the command of step t.
        Returns the predicted PWM, (B, T)."""
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        estimate, fast = self.start(batch, device, dtype, twin)
        h = torch.zeros(batch, device=device, dtype=dtype)
        v = torch.zeros(batch, device=device, dtype=torch.bool)
        prev = torch.zeros(batch, device=device, dtype=dtype)
        out = []
        for t in range(steps):
            x = self.features(cents, voice, t, h, v, prev, estimate, twin)
            u, fast = self.act(x, fast)
            out.append(u)
            prev = pwm[:, t]
            estimate = self.motor.step(estimate, prev, twin)
            h, v = heard[:, t], valid[:, t]
        return torch.stack(out, 1)

    def checkpoint(self, **metadata):
        return {"format": "yamabiko-nn-twin-v1", "config": asdict(self.config), "state_dict": self.state_dict(),
                **metadata}

    @classmethod
    def from_checkpoint(cls, checkpoint, plant, device="cpu"):
        model = cls(plant, NNConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
