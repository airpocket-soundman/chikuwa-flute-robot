"""A world model of one rig, learned from what the rig can record.

It predicts the heard pitch from the commands alone (PWM and valve), so it
can be fitted to the logs of the real rig and a policy can be fine-tuned
inside it.  No simulator state or parameter is ever a label.

The structure borrows only the shape of the physics, never its values:

* a latent stroke position integrates a latent velocity and stops at the
  two ends (the plunger has end stops; the home run starts it at 0);
* the velocity approaches a learned steady speed with a learned lag; the
  steady speed is a learned increasing function of PWM, so deadband,
  friction and speed limit are all learned shapes of that curve;
* pitch is a learned monotone function of the latent position (the flute);
* what is heard is that pitch after a learned mixture of 0-3 steps' delay.

``context`` is a small per-rig vector inferred by :class:`ContextEncoder`
from the rig's earlier logs (in-context system identification).  On a new
rig it can be refined by gradient on that rig's logs, or the whole model can
be fine-tuned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .rig_adaptive import PITCH_CENTER, PITCH_SCALE

MAX_DELAY = 4


@dataclass(frozen=True)
class WorldModelConfig:
    hidden: int = 64
    context: int = 8
    dt: float = .01
    max_speed: float = 6.0  # strokes/s, an upper bound for the learned velocity


@dataclass
class WorldState:
    hidden: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    pitch_history: torch.Tensor   # (B, MAX_DELAY) newest last
    valve_history: torch.Tensor


class MonotonePitch(nn.Module):
    """Latent position in [0, 1] -> normalized pitch, increasing by construction."""

    def __init__(self, knots: int = 24, context: int = 8):
        super().__init__()
        self.knots = knots
        self.base = nn.Linear(context, 1)
        self.increments = nn.Linear(context, knots)

    def forward(self, position, context):
        steps = F.softplus(self.increments(context)) / self.knots * 3.0
        levels = torch.cat([torch.zeros_like(steps[:, :1]), steps.cumsum(1)], 1) + self.base(context) - 1.5
        scaled = position.clamp(0, 1) * self.knots
        index = scaled.floor().clamp(max=self.knots - 1).long()
        frac = scaled - index
        left = levels.gather(1, index[:, None])[:, 0]; right = levels.gather(1, index[:, None] + 1)[:, 0]
        return left + frac * (right - left)


class ContextEncoder(nn.Module):
    """Earlier logs of a rig -> its context vector (observable signals only)."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.gru = nn.GRU(4, config.hidden, batch_first=True)
        self.out = nn.Linear(config.hidden, config.context)

    def forward(self, logs):
        contexts = []
        for log in logs:
            heard_n = torch.where(log.valid, (log.heard - PITCH_CENTER) / PITCH_SCALE, torch.zeros_like(log.heard))
            x = torch.stack([log.pwm, log.valve, heard_n, log.valid.to(log.pwm.dtype)], -1)
            _, h = self.gru(x)
            contexts.append(torch.tanh(self.out(h[-1])))
        return torch.stack(contexts).mean(0)


class MonotoneCurve(nn.Module):
    """x in [-1, 1] -> increasing piecewise-linear value, shaped by the rig context."""

    def __init__(self, knots: int, context: int, span: float):
        super().__init__()
        self.knots, self.span = knots, span
        self.increments = nn.Linear(context, knots)
        self.offset = nn.Linear(context, 1)

    def forward(self, x, context):
        steps = F.softplus(self.increments(context)) / self.knots * self.span
        levels = torch.cat([torch.zeros_like(steps[:, :1]), steps.cumsum(1)], 1)
        levels = levels - levels[:, self.knots // 2:self.knots // 2 + 1] + self.offset(context)
        scaled = (x.clamp(-1, 1) + 1) * .5 * self.knots
        index = scaled.floor().clamp(max=self.knots - 1).long()
        frac = scaled - index
        left = levels.gather(1, index[:, None])[:, 0]; right = levels.gather(1, index[:, None] + 1)[:, 0]
        return left + frac * (right - left)


class DeviceWorldModel(nn.Module):
    observable_only = False

    def __init__(self, config: WorldModelConfig = WorldModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = ContextEncoder(config)
        self._context = None
        self.speed = MonotoneCurve(16, config.context, span=2.0)     # PWM -> steady speed (strokes/s)
        self.lag = nn.Linear(config.context, 1)                      # approach rate per step
        self.pitch = MonotonePitch(context=config.context)
        self.delay_logits = nn.Parameter(torch.tensor([0., 2., 0., -2.]))
        with torch.no_grad():
            self.lag.bias.fill_(-1.5)

    # environment interface -------------------------------------------------
    def reset(self, device, dtype=torch.float32, homing_steps=0, batch=None):
        batch = batch or self._batch
        z = torch.zeros(batch, device=device, dtype=dtype)
        return WorldState(torch.zeros(batch, 1, device=device, dtype=dtype), z, z.clone(),
                          torch.zeros(batch, MAX_DELAY, device=device, dtype=dtype),
                          torch.zeros(batch, MAX_DELAY, device=device, dtype=dtype))

    def step(self, state: WorldState, pwm, valve):
        cfg = self.config
        context = self._context.expand(pwm.shape[0], -1)
        steady = self.speed(pwm, context)
        alpha = torch.sigmoid(self.lag(context))[:, 0]
        velocity = state.velocity + alpha * (steady - state.velocity)
        raw = state.position + cfg.dt * velocity
        # End stops hold the value, but let the gradient through (straight-through):
        # a hard clamp would leave a model that starts pushing into the home stop
        # with no gradient to learn that it should move out.
        position = raw + (raw.clamp(0, 1) - raw).detach()
        blocked = ((raw <= 0) & (velocity < 0) | (raw >= 1) & (velocity > 0)).to(velocity.dtype)
        velocity = velocity - (velocity * blocked).detach()
        pitch = self.pitch(position, context)
        pitch_history = torch.cat([state.pitch_history[:, 1:], pitch[:, None]], 1)
        valve_history = torch.cat([state.valve_history[:, 1:], valve[:, None]], 1)
        weights = torch.softmax(self.delay_logits, 0).flip(0)  # weight for delay 0 applies to the newest
        heard_n = (pitch_history * weights).sum(1)
        valid = (valve_history * weights).sum(1) >= .5
        heard = PITCH_CENTER + PITCH_SCALE * heard_n
        next_state = WorldState(state.hidden, position, velocity, pitch_history, valve_history)
        # 4th value: the model's estimate of the pitch sounding now (before the hearing delay).
        return heard, valid, next_state, PITCH_CENTER + PITCH_SCALE * pitch

    def bind(self, context):
        """Fix the rig context, (B, C) or (C,), for playing inside the model."""
        self._context = context
        self._batch = context.shape[0] if context.dim() == 2 else 1
        return self

    # fitting ----------------------------------------------------------------
    def rollout(self, pwm, valve, context):
        self._context = context
        state = self.reset(pwm.device, pwm.dtype, batch=pwm.shape[0])
        heard = []
        for t in range(pwm.shape[1]):
            h, _, state, _ = self.step(state, pwm[:, t], valve[:, t])
            heard.append(h)
        return torch.stack(heard, 1)

    def loss(self, logs, context):
        """Heard-pitch loss and MAE (cents) over valid frames of ``logs`` given the rig context."""
        pwm = torch.cat([log.pwm for log in logs]); valve = torch.cat([log.valve for log in logs])
        heard = torch.cat([log.heard for log in logs]); valid = torch.cat([log.valid for log in logs])
        predicted = self.rollout(pwm, valve, context.repeat(len(logs), 1) if context.dim() == 2 else context)
        error = (predicted - heard)[valid]
        return F.smooth_l1_loss(error / 100, torch.zeros_like(error), beta=.1) , error.abs().mean()

    def checkpoint(self, **metadata):
        return {"format": "yamabiko-world-model-v1", "config": asdict(self.config),
                "state_dict": self.state_dict(), **metadata}

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        model = cls(WorldModelConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
