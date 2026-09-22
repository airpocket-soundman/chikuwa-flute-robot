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
    """Neural map from remembered acoustic target to normalized actuator aim.

    The second input channel is a voicing *logit* throughout the connected
    pipeline.  Convert it to a probability inside the module so upstream
    confidence magnitude cannot shift the requested physical position.
    """

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        normalized = torch.cat([target[..., :1], torch.sigmoid(target[..., 1:2])], -1)
        return torch.sigmoid(self.net(normalized))


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


class MotorStateEstimator(nn.Module):
    """Neural dead reckoner: previous commands -> current position/velocity."""

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__(); self.config = config
        self.cell = nn.GRUCell(3, config.control_hidden)
        self.head = nn.Sequential(nn.Linear(config.control_hidden, 64), nn.SiLU(), nn.Linear(64, 2))

    def initial_state(self, batch, device, dtype=torch.float32):
        return torch.zeros(batch, self.config.control_hidden, device=device, dtype=dtype)

    def step(self, previous_action: torch.Tensor, state: torch.Tensor, reset=None):
        if reset is None: reset = torch.zeros(len(previous_action), 1, device=previous_action.device)
        state = torch.where(reset.bool(), torch.zeros_like(state), state)
        state = self.cell(torch.cat([previous_action, reset], 1), state)
        raw = self.head(state)
        estimate = torch.cat([torch.sigmoid(raw[:, :1]), .2 * torch.tanh(raw[:, 1:2])], 1)
        return estimate, state

    def forward(self, actions: torch.Tensor):
        batch, steps = actions.shape[:2]; state = self.initial_state(batch, actions.device, actions.dtype)
        previous = torch.zeros(batch, 2, device=actions.device, dtype=actions.dtype); outputs = []
        for t in range(steps):
            estimate, state = self.step(previous, state, torch.ones(batch, 1, device=actions.device) if t == 0 else None)
            outputs.append(estimate); previous = actions[:, t]
        return torch.stack(outputs, 1)


class PositionActionController(nn.Module):
    """Memoryless neural control law from desired and estimated motor state."""

    def __init__(self, config: StagedConfig = StagedConfig()):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(14, config.control_hidden), nn.SiLU(),
                                 nn.Linear(config.control_hidden, config.control_hidden), nn.SiLU(),
                                 nn.Linear(config.control_hidden, 2))

    def forward(self, desired: torch.Tensor, target: torch.Tensor, motor_state: torch.Tensor,
                previous_action: torch.Tensor, t: int):
        steps = desired.shape[1]; ids = [t, min(t + 5, steps - 1), min(t + 20, steps - 1), min(t + 50, steps - 1)]
        p = [desired[:, i] for i in ids]; voice = [target[:, i, 1:2] for i in ids[:3]]
        pos, vel = motor_state[:, :1], motor_state[:, 1:2]
        x = torch.cat([*p, p[0] - pos, p[1] - pos, p[2] - pos, pos, vel,
                       *voice, previous_action], 1)
        raw = self.net(x)
        return torch.stack([torch.tanh(raw[:, 0]), torch.sigmoid(raw[:, 1])], 1)


class MotorTrajectoryController(nn.Module):
    """Execute a precomputed Position Planner trajectory without self audio.

    This is the low-level motor tracker, not a second musical feedforward path.
    Its recurrent state may infer motor motion from the planned trajectory and
    previously applied PWM, but it never receives simulator true state.
    """

    FEATURES = 8

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(self.FEATURES, hidden)
        self.latent_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.head = nn.Sequential(nn.Linear(hidden + self.FEATURES + 5, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2))

    def initial_state(self, batch: int, device, dtype=torch.float32):
        return torch.zeros(batch, self.hidden, device=device, dtype=dtype)

    @staticmethod
    def features(plan: torch.Tensor, voice: torch.Tensor, previous_pwm: torch.Tensor, t: int):
        steps = plan.shape[1]
        ids = [t, min(t + 5, steps - 1), min(t + 20, steps - 1), min(t + 50, steps - 1)]
        p = [plan[:, i] for i in ids]
        return torch.stack([p[0], p[1], p[2], p[3], p[1] - p[0], p[2] - p[0],
                            voice[:, t], previous_pwm], 1)

    def step(self, plan: torch.Tensor, voice: torch.Tensor, previous_pwm: torch.Tensor,
             state: torch.Tensor, t: int):
        features = self.features(plan, voice, previous_pwm, t)
        state = self.cell(features, state)
        # These are learned latent features, not supervised simulator state.
        # The physical plant exposes no position/velocity/torque to this NN.
        estimated = self.latent_motor_features(state)
        desired_now, desired_future = features[:, 0], features[:, 2]
        control = torch.cat([estimated, (desired_now[:, None] - estimated[:, 0:1]),
                             (desired_future[:, None] - estimated[:, 0:1])], 1)
        raw = self.head(torch.cat([state, features, control], 1))
        return torch.tanh(raw[:, 0]), raw[:, 1], state

    def latent_motor_features(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.latent_head(state)
        return torch.stack([torch.sigmoid(raw[:, 0]), torch.tanh(raw[:, 1]), raw[:, 2]], 1)


class MotorAudioWorldModel(nn.Module):
    """Learn the black-box PWM-to-audible-pitch transition relation.

    It predicts normalized audible pitch directly and has no heads or labels
    for position, velocity, torque, friction, slope, intercept, or time constant.
    """

    def __init__(self, hidden: int = 64):
        super().__init__(); self.hidden = hidden
        self.cell = nn.GRUCell(1, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def initial_state(self, batch: int, device, dtype=torch.float32):
        return torch.zeros(batch, self.hidden, device=device, dtype=dtype)

    def step(self, pwm: torch.Tensor, state: torch.Tensor):
        state = self.cell(pwm[:, None], state)
        pitch = torch.sigmoid(self.head(state)[:, 0])
        return pitch, state


class AcousticFeedbackResidual(nn.Module):
    """Adaptive causal tracker with an implicit instrument/motor context.

    The recurrent state is never supervised as slope, intercept, friction or
    time constant.  It learns whatever latent statistic is useful from audible
    pitch transitions and past PWM, and can be kept across repeated plays.
    """

    FEATURES = 11

    def __init__(self, hidden: int = 48, limit: float = .45):
        super().__init__()
        self.hidden = hidden
        self.limit = limit
        self.fast_cell = nn.GRUCell(self.FEATURES, hidden)
        self.context_cell = nn.GRUCell(self.FEATURES, hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden + self.FEATURES, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1))

    def initial_state(self, batch: int, device, dtype=torch.float32):
        return torch.zeros(batch, 2 * self.hidden, device=device, dtype=dtype)

    def step(self, target_pitch: torch.Tensor, heard_pitch: torch.Tensor,
             heard_valid: torch.Tensor, target_voice: torch.Tensor, base_pwm: torch.Tensor,
             previous_pwm: torch.Tensor, previous_error: torch.Tensor, state: torch.Tensor,
             target_future: torch.Tensor | None = None,
             previous_heard: torch.Tensor | None = None,
             external_error: torch.Tensor | None = None):
        target_future = target_pitch if target_future is None else target_future
        previous_heard = heard_pitch if previous_heard is None else previous_heard
        measured_error = target_pitch - heard_pitch if external_error is None else external_error
        error = torch.where(heard_valid, measured_error, torch.zeros_like(target_pitch))
        pitch_velocity = torch.where(heard_valid, heard_pitch - previous_heard,
                                     torch.zeros_like(heard_pitch))
        features = torch.stack([target_pitch, target_future, target_future - target_pitch,
                                heard_pitch, error, pitch_velocity,
                                heard_valid.to(target_pitch.dtype), target_voice,
                                base_pwm, previous_pwm, previous_error], 1)
        fast, context = state.split(self.hidden, dim=1)
        fast = self.fast_cell(features, fast)
        proposed_context = self.context_cell(features, context)
        # Slow context survives a whole performance and repeated practice.
        context = context + .04 * (proposed_context - context)
        state = torch.cat([fast, context], 1)
        residual = self.limit * torch.tanh(self.head(torch.cat([state, features], 1))[:, 0])
        return residual, error, state


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
