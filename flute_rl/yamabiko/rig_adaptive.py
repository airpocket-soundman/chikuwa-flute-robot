"""Neural Planner / Controller / Feedback with in-network rig memory.

Neural counterpart of :class:`EncoderlessDeterministicPerformer`, with the
role split agreed for the closed-tube flute:

* ``RigAwarePlanner``: remembered target (cents + voicing) over a future
  window, plus the slow rig context ``z`` -> plunger aim.  It decides *where*
  and *when* (pre-positioning before a note).
* ``MotorFeedbackCore``: fast recurrent tracker.  Sees the aim, the heard
  pitch (delayed, noisy, with drop-outs), its validity and past PWM, and
  outputs PWM.  It plays the observer + PD + pitch-trim part.
* ``SlowRigMemory``: slow recurrent context kept across songs on the same
  rig.  It is never supervised with rig parameters; whatever it stores must
  pay off in later songs.

No network receives the plant's position, velocity or parameters.  Training
back-propagates through the differentiable simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .physical_plant import AudibleListener, DifferentiableMotorFlute

PITCH_CENTER, PITCH_SCALE = 1300.0, 600.0
WINDOW = (0, 3, 6, 10, 15, 20, 30, 40, 55, 70)


@dataclass(frozen=True)
class RigAdaptiveConfig:
    fast_hidden: int = 96
    memory_hidden: int = 16
    planner_hidden: int = 96
    memory_rate: float = .02       # max fraction of the slow state renewed per step
    homing_steps: int = 130
    # Smooth deadband compensation, the same prior as the deterministic
    # controller: without it a small initial PWM sits in the deadband where
    # the simulator gradient is exactly zero and nothing is learned.
    deadband_compensation: float = .25


def normalize(cents):
    return (cents - PITCH_CENTER) / PITCH_SCALE


def future_window(cents, voice, t):
    steps = cents.shape[1]
    ids = [min(t + k, steps - 1) for k in WINDOW]
    return torch.cat([normalize(cents[:, ids]), voice[:, ids].to(cents.dtype)], 1)


class RigAwarePlanner(nn.Module):
    def __init__(self, config: RigAdaptiveConfig):
        super().__init__()
        inputs = 2 * len(WINDOW) + config.memory_hidden
        self.net = nn.Sequential(nn.Linear(inputs, config.planner_hidden), nn.SiLU(),
                                 nn.Linear(config.planner_hidden, config.planner_hidden), nn.SiLU(),
                                 nn.Linear(config.planner_hidden, 1))

    def forward(self, window, memory):
        return torch.sigmoid(self.net(torch.cat([window, memory], 1)))[:, 0]


class MotorFeedbackCore(nn.Module):
    FEATURES = 9

    def __init__(self, config: RigAdaptiveConfig):
        super().__init__()
        self.deadband_compensation = config.deadband_compensation
        self.cell = nn.GRUCell(self.FEATURES + config.memory_hidden, config.fast_hidden)
        self.head = nn.Sequential(nn.Linear(config.fast_hidden + 3, config.fast_hidden), nn.SiLU(),
                                  nn.Linear(config.fast_hidden, 1))

    def forward(self, aim, aim_future, previous_aim, heard, valid, target_now, voice_now,
                previous_pwm, memory, state):
        heard_n = torch.where(valid, normalize(heard), torch.zeros_like(heard))
        features = torch.stack([aim, aim_future, aim - previous_aim, heard_n, valid.to(aim.dtype),
                                torch.where(valid, normalize(target_now) - heard_n, torch.zeros_like(heard)),
                                voice_now.to(aim.dtype), previous_pwm, aim_future - aim], 1)
        state = self.cell(torch.cat([features, memory], 1), state)
        u = torch.tanh(self.head(torch.cat([state, aim[:, None], aim_future[:, None],
                                            previous_pwm[:, None]], 1)))[:, 0]
        band = self.deadband_compensation
        return band * torch.tanh(u / .02) + (1 - band) * u, state


class SlowRigMemory(nn.Module):
    FEATURES = 6

    def __init__(self, config: RigAdaptiveConfig):
        super().__init__()
        self.rate = config.memory_rate
        self.candidate = nn.GRUCell(self.FEATURES + config.fast_hidden, config.memory_hidden)
        self.gate = nn.Linear(self.FEATURES + config.fast_hidden, 1)

    def forward(self, heard, valid, target_then, voice_then, aim, previous_pwm, fast, memory):
        heard_n = torch.where(valid, normalize(heard), torch.zeros_like(heard))
        features = torch.stack([heard_n, valid.to(heard.dtype), normalize(target_then),
                                voice_then.to(heard.dtype), aim, previous_pwm], 1)
        x = torch.cat([features, fast], 1)
        proposal = self.candidate(x, memory)
        # Bounded, learned write rate: the memory can only drift slowly.
        rate = self.rate * torch.sigmoid(self.gate(x)) * valid.to(heard.dtype)[:, None]
        return memory + rate * (proposal - memory)


class RigAdaptivePerformer(nn.Module):
    def __init__(self, config: RigAdaptiveConfig = RigAdaptiveConfig()):
        super().__init__()
        self.config = config
        self.planner = RigAwarePlanner(config)
        self.core = MotorFeedbackCore(config)
        self.memory = SlowRigMemory(config)

    def initial_memory(self, batch, device, dtype=torch.float32):
        return torch.zeros(batch, self.config.memory_hidden, device=device, dtype=dtype)

    def perform(self, plant: DifferentiableMotorFlute, cents, voice, parameters, memory=None,
                generator: torch.Generator | None = None, target_delay: int = 2,
                write_memory: bool = True):
        """One song on one rig: home, then play.  Returns logs and the kept memory.

        ``write_memory=False`` is the performance mode of the learning-mode
        design: the rig memory is only read, so what a calibration run wrote
        stays fixed until the next calibration.
        """
        cfg = self.config
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        memory = self.initial_memory(batch, device, dtype) if memory is None else memory
        listener = AudibleListener(plant.config)
        state = plant.initial_state(batch, device, dtype)
        with torch.no_grad():
            start = torch.rand(batch, generator=generator, device=device, dtype=dtype) * .8
        state.position = start
        for _ in range(cfg.homing_steps):
            state = plant.step(state, -torch.ones(batch, device=device, dtype=dtype), parameters)
        heard_state = listener.initial_state(batch, device, dtype)
        fast = torch.zeros(batch, cfg.fast_hidden, device=device, dtype=dtype)
        heard = torch.zeros(batch, device=device, dtype=dtype)
        valid = torch.zeros(batch, device=device, dtype=torch.bool)
        previous_pwm = torch.zeros(batch, device=device, dtype=dtype)
        previous_aim = torch.zeros(batch, device=device, dtype=dtype)
        played, pwms, aims = [], [], []
        for t in range(steps):
            aim = self.planner(future_window(cents, voice, t), memory)
            aim_future = self.planner(future_window(cents, voice, min(t + 10, steps - 1)), memory)
            pwm, fast = self.core(aim, aim_future, previous_aim, heard, valid, cents[:, t], voice[:, t],
                                  previous_pwm, memory, fast)
            state = plant.step(state, pwm, parameters)
            emitted, sounding = plant.flute(state, voice[:, t].to(dtype), parameters)
            heard, valid, heard_state = listener.step(emitted, sounding, parameters, heard_state, generator)
            then = max(0, t - target_delay)
            if write_memory:
                memory = self.memory(heard, valid, cents[:, then], voice[:, then], aim, pwm, fast, memory)
            played.append(emitted); pwms.append(pwm); aims.append(aim)
            previous_pwm, previous_aim = pwm, aim
        return ({"pitch_cents": torch.stack(played, 1), "pwm": torch.stack(pwms, 1),
                 "aim": torch.stack(aims, 1)}, memory)

    def checkpoint(self, **metadata):
        return {"format": "yamabiko-rig-adaptive-v1", "config": asdict(self.config),
                "state_dict": self.state_dict(), **metadata}

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        model = cls(RigAdaptiveConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
