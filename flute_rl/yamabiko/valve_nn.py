"""A separate network for the valve (tonguing), run beside the actuator network.

The actuator network (nn_performer.NNTwinPerformer, step1) keeps driving the plunger; this small recurrent
network decides every 10 ms whether the air flows.  It learns to open the valve only when the flute will
sound the note (the plunger has arrived), so the glide between notes is not heard, and to shut it for rests
and for repeated notes (tonguing), ahead of the servo's delay.  The two networks are trained apart and run
side by side; they can be merged later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .features import WINDOW
from .physical_plant import AudibleListener

VALVE_FEATURES = 2 * len(WINDOW) + 7    # window, heard (3), actuator pwm, twin position / velocity, valve before


@dataclass(frozen=True)
class ValveConfig:
    hidden: int = 32
    twin_free: bool = False     # step7: no twin position / velocity among the inputs

    @property
    def features(self) -> int:
        return VALVE_FEATURES - (2 if self.twin_free else 0)


class ValveNN(nn.Module):
    def __init__(self, normalize, config: ValveConfig = ValveConfig()):
        super().__init__()
        self.config, self.normalize = config, normalize
        self.inp = nn.Linear(config.features, config.hidden)
        self.cell = nn.GRUCell(config.hidden, config.hidden)
        self.out = nn.Linear(config.hidden, 1)

    def features(self, cents, voice, t, heard, valid, pwm, estimate, prev_open):
        n = self.normalize
        ids = [min(t + k, cents.shape[1] - 1) for k in WINDOW]
        dtype = cents.dtype
        heard_n = torch.where(valid, n(heard), torch.zeros_like(heard))
        gap = torch.where(valid, n(cents[:, t]) - heard_n, torch.zeros_like(heard))
        now = [heard_n, valid.to(dtype), gap, pwm]
        if not self.config.twin_free:
            now += [estimate.position, estimate.velocity]
        return torch.cat([n(cents[:, ids]), voice[:, ids].to(dtype), torch.stack(now + [prev_open], 1)], 1)

    def step(self, x, h):
        h = self.cell(torch.tanh(self.inp(x)), h)
        return torch.sigmoid(self.out(h)[:, 0]), h

    def checkpoint(self, **metadata):
        return {"format": "yamabiko-valve-v1", "config": asdict(self.config), "state_dict": self.state_dict(),
                **metadata}

    @classmethod
    def from_checkpoint(cls, checkpoint, normalize, device="cpu"):
        model = cls(normalize, ValveConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model


def play_with_valve(actuator, valve, plant, cents, voice, parameters, twin, generator=None):
    """Closed loop on the simulator: ``actuator`` (an NNTwinPerformer without its own valve, frozen) drives the
    plunger, ``valve`` opens and shuts the air (it follows ``plant.config.valve_delay_steps`` late).  Only the
    valve network's outputs carry gradients.  Returns emitted pitch, the open probability the air follows,
    whether the flute sounded, and the commanded open probability, all (B, T)."""
    batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
    listener = AudibleListener(plant.config)
    with torch.no_grad():
        state = plant.initial_state(batch, device, dtype)
        state.position = torch.rand(batch, generator=generator, device=device, dtype=dtype) * .8
        for _ in range(130):
            state = plant.step(state, -torch.ones(batch, device=device, dtype=dtype), parameters)
        heard_state = listener.initial_state(batch, device, dtype)
        estimate, fast = actuator.start(batch, device, dtype, twin)
    heard = torch.zeros(batch, device=device, dtype=dtype)
    valid = torch.zeros(batch, device=device, dtype=torch.bool)
    pwm = torch.zeros(batch, device=device, dtype=dtype)
    h = torch.zeros(batch, valve.config.hidden, device=device, dtype=dtype)
    delay = plant.config.valve_delay_steps
    queue = [torch.zeros(batch, device=device, dtype=dtype) for _ in range(delay)]
    prev = torch.zeros(batch, device=device, dtype=dtype)
    band = twin.deadband.to(dtype) if twin.deadband is not None else torch.zeros(batch, device=device, dtype=dtype)
    emitted_all, air_all, sounding_all, cmd_all = [], [], [], []
    for t in range(steps):
        with torch.no_grad():
            pwm, fast = actuator.act(actuator.features(cents, voice, t, heard, valid, pwm, estimate, twin), fast)
        p, h = valve.step(valve.features(cents, voice, t, heard, valid, pwm, estimate, prev.detach()), h)
        queue.append(p); air = queue.pop(0); prev = p
        with torch.no_grad():
            state = plant.step(state, pwm, parameters)
            estimate = actuator.motor.step(estimate, pwm, twin)
            emitted, sounding = plant.flute(state, (air >= .5).to(dtype), parameters)
            heard, valid, heard_state = listener.step(emitted, sounding, parameters, heard_state, generator)
            if plant.config.motor_masking > 0:
                lost = torch.rand(valid.shape, generator=generator, device=device) < plant.config.motor_masking
                valid = valid & ~((pwm.abs() > band) & lost)
        emitted_all.append(emitted); air_all.append(air); sounding_all.append(sounding); cmd_all.append(p)
    return (torch.stack(emitted_all, 1), torch.stack(air_all, 1), torch.stack(sounding_all, 1),
            torch.stack(cmd_all, 1))
