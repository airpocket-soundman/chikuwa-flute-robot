"""Independently trainable neural stages for the Yamabiko imitation pipeline.

The classes in this file intentionally expose the boundaries that the final
end-to-end policy will later fuse.  No pitch estimator is used at runtime: the
``NeuralEar`` is the raw-waveform encoder from :mod:`e2e`, and every downstream
stage consumes only its learned outputs or another neural stage's outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StagedConfig:
    memory_hidden: int = 96
    control_hidden: int = 128
    comparator_hidden: int = 64


class ReferenceMemory(nn.Module):
    """Turn a complete neural-ear sequence into stored, contextual memory.

    The returned tensor is the robot's in-memory demonstration.  The raw audio
    and the input ear sequence may be discarded before ``decode`` is called.
    """

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__()
        self.config = config
        self.encoder = nn.GRU(2, config.memory_hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.Linear(2 * config.memory_hidden, config.memory_hidden),
            nn.SiLU(),
            nn.Linear(config.memory_hidden, 2),
        )

    def remember(self, ear_sequence: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            return self.encoder(ear_sequence)[0]
        packed = nn.utils.rnn.pack_padded_sequence(
            ear_sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        memory, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=ear_sequence.shape[1]
        )
        return memory

    def decode(self, memory: torch.Tensor) -> torch.Tensor:
        """Return normalized pitch and a voicing logit from stored memory."""
        return self.decoder(memory)

    def forward(self, ear_sequence: torch.Tensor, lengths: torch.Tensor | None = None):
        memory = self.remember(ear_sequence, lengths)
        return self.decode(memory), memory


class FeedForwardPolicy(nn.Module):
    """Reference-only recurrent motor program (no self-audio input)."""

    FEATURES = 8

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__()
        self.config = config
        self.cell = nn.GRUCell(self.FEATURES, config.control_hidden)
        self.head = nn.Sequential(
            nn.Linear(config.control_hidden + self.FEATURES, config.control_hidden),
            nn.SiLU(),
            nn.Linear(config.control_hidden, 2),
        )

    @staticmethod
    def features(target: torch.Tensor, previous_action: torch.Tensor, t: int) -> torch.Tensor:
        """Build neural-target features; ``target`` is ``(B,T,2)``."""
        steps = target.shape[1]
        future = target[:, min(t + 5, steps - 1)]
        now = target[:, t]
        return torch.cat([now, future, future[:, :1] - now[:, :1], previous_action,
                          torch.full_like(now[:, :1], float(t) / max(steps - 1, 1))], 1)

    def forward(self, target: torch.Tensor, lengths: torch.Tensor | None = None,
                teacher_actions: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps = target.shape[:2]
        state = torch.zeros(batch, self.config.control_hidden, device=target.device, dtype=target.dtype)
        previous = torch.zeros(batch, 2, device=target.device, dtype=target.dtype)
        outputs = []
        for t in range(steps):
            x = self.features(target, previous, t)
            nxt = self.cell(x, state)
            raw = self.head(torch.cat([nxt, x], 1))
            action = torch.stack([torch.tanh(raw[:, 0]), torch.sigmoid(raw[:, 1])], 1)
            if lengths is not None:
                active = (t < lengths)[:, None]
                state = torch.where(active, nxt, state)
                action = torch.where(active, action, torch.zeros_like(action))
            else:
                state = nxt
            outputs.append(action)
            previous = teacher_actions[:, t] if teacher_actions is not None else action
        return torch.stack(outputs, 1)


class TargetPositionPlanner(nn.Module):
    """Neural map from remembered acoustic target to normalized actuator aim."""

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(target))


class MotorInversePolicy(nn.Module):
    """Stateful neural inverse motor driven by a separately learned aim."""

    FEATURES = 12

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__(); self.config = config
        self.cell = nn.GRUCell(self.FEATURES, config.control_hidden)
        self.head = nn.Sequential(nn.Linear(config.control_hidden + self.FEATURES, config.control_hidden),
                                  nn.SiLU(), nn.Linear(config.control_hidden, 3))

    @staticmethod
    def features(position: torch.Tensor, target: torch.Tensor, previous: torch.Tensor, t: int):
        steps = position.shape[1]
        ids = [t, min(t + 5, steps - 1), min(t + 20, steps - 1), min(t + 50, steps - 1)]
        p = [position[:, i] for i in ids]
        voice = [target[:, i, 1:2] for i in ids]
        return torch.cat([p[0], p[1], p[2], p[3], p[1] - p[0], p[2] - p[0],
                          voice[0], voice[1], voice[2], previous,
                          torch.full_like(p[0], float(t) / max(steps - 1, 1))], 1)

    def forward(self, position: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor | None = None,
                teacher_actions: torch.Tensor | None = None, return_position: bool = False):
        batch, steps = target.shape[:2]
        state = torch.zeros(batch, self.config.control_hidden, device=target.device, dtype=target.dtype)
        previous = torch.zeros(batch, 2, device=target.device, dtype=target.dtype); outputs = []; positions = []
        for t in range(steps):
            x = self.features(position, target, previous, t); nxt = self.cell(x, state)
            raw = self.head(torch.cat([nxt, x], 1))
            action = torch.stack([torch.tanh(raw[:, 0]), torch.sigmoid(raw[:, 1])], 1)
            estimated_position = torch.sigmoid(raw[:, 2])
            if lengths is not None:
                active = (t < lengths)[:, None]; state = torch.where(active, nxt, state)
                action = torch.where(active, action, torch.zeros_like(action))
            else: state = nxt
            outputs.append(action); positions.append(estimated_position)
            previous = teacher_actions[:, t] if teacher_actions is not None else action
        actions = torch.stack(outputs, 1)
        return (actions, torch.stack(positions, 1)) if return_position else actions


class ErrorComparator(nn.Module):
    """Estimate signed target-minus-self pitch error from two neural ears."""

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__()
        h = config.comparator_hidden
        self.net = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, target: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
        # Include difference as an inductive bias, but the output is still the
        # learned NN result and is evaluated independently.  The explicit
        # signed skip makes the physically correct comparison the default;
        # the bounded residual only calibrates representation bias.
        learned = self.net(torch.cat([target, own, target - own], -1))
        error = target[..., :1] - own[..., :1] + 0.1 * learned[..., :1]
        return torch.cat([error, learned[..., 1:2]], -1)


class FeedbackResidualPolicy(nn.Module):
    """Causal neural residual that corrects a frozen feed-forward motor plan."""

    FEATURES = 8

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__()
        self.config = config
        self.cell = nn.GRUCell(self.FEATURES, config.control_hidden)
        self.head = nn.Sequential(nn.Linear(config.control_hidden, 64), nn.SiLU(), nn.Linear(64, 1))

    def step(self, target: torch.Tensor, own: torch.Tensor, estimated_error: torch.Tensor,
             base_action: torch.Tensor, previous_action: torch.Tensor, state: torch.Tensor):
        x = torch.cat([target, own, estimated_error[:, :1], base_action, previous_action[:, :1]], 1)
        state = self.cell(x, state)
        correction = 0.5 * torch.tanh(self.head(state))
        pwm = (base_action[:, :1] + correction).clamp(-1.0, 1.0)
        return torch.cat([pwm, base_action[:, 1:2]], 1), state


def checkpoint(config: StagedConfig, memory: ReferenceMemory, feedforward: FeedForwardPolicy,
               comparator: ErrorComparator, feedback: FeedbackResidualPolicy, **metadata) -> dict:
    return {"format": "yamabiko-staged-nn-v1", "config": asdict(config),
            "memory": memory.state_dict(), "feedforward": feedforward.state_dict(),
            "comparator": comparator.state_dict(), "feedback": feedback.state_dict(), **metadata}
