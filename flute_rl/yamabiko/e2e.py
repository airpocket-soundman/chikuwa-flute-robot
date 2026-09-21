"""One-model, raw-audio-to-actuation policy for Yamabiko No.1.

The inference path deliberately contains no pitch estimator, score, tube model,
dead reckoning, or hand-written valve schedule.  A single :class:`E2EImitator`
first encodes the complete demonstration recording and then, at 100 Hz, maps
that memory plus the latest microphone waveform directly to

    (plunger PWM, valve probability, finished probability).

The same audio encoder is used for the demonstration and the robot's own sound.
The controller hidden state is carried across demonstrations, allowing the one
model to learn both imitation and adaptation to a particular physical rig.

Framing and amplitude normalisation are representation plumbing, not semantic
audio processing: no pitch, onset, note, octave, or silence decision is made
outside the network.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .e2e_io import CONTROL_HZ, FRAME, HOP, SAMPLE_RATE, frame_audio_numpy


def frame_audio(audio, steps: int | None = None, *, causal: bool = False) -> torch.Tensor:
    """Return raw waveform windows as ``(steps, FRAME)`` float tensors.

    Reference audio is framed around each 10 ms instant because the complete
    demonstration is available before playback.  Self audio is causal: frame
    ``t`` ends where control step ``t`` starts, so it cannot leak the result of
    the action being selected.
    """
    return torch.from_numpy(frame_audio_numpy(audio, steps, causal=causal))


@dataclass(frozen=True)
class E2EConfig:
    # Sized for 100 Hz NumPy inference on the UNO Q Linux Cortex-A53.  The
    # larger architecture remains available by passing an explicit config.
    audio_width: int = 8
    audio_dim: int = 24
    reference_hidden: int = 24
    controller_hidden: int = 48
    adaptation_dim: int = 16

    @classmethod
    def pc(cls) -> "E2EConfig":
        """Capacity-first teacher model; compress only after control works."""
        return cls(audio_width=24, audio_dim=64, reference_hidden=96,
                   controller_hidden=192, adaptation_dim=64)


class AudioEncoder(nn.Module):
    """Shared learned representation for both kinds of raw microphone audio."""

    def __init__(self, width: int, out_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            # 63 samples span 2.6 cycles even at the 650 Hz lower limit.
            # The old 15-sample kernel never saw one complete low note cycle.
            nn.Conv1d(1, width, 63, stride=2, padding=31), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv1d(width, 2 * width, 9, stride=2, padding=4), nn.GroupNorm(4, 2 * width), nn.SiLU(),
            nn.Conv1d(2 * width, 2 * width, 7, stride=2, padding=3), nn.GroupNorm(4, 2 * width), nn.SiLU(),
            nn.Conv1d(2 * width, 3 * width, 5, stride=2, padding=2), nn.GroupNorm(4, 3 * width), nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Linear(12 * width, out_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        shape = frames.shape[:-1]
        x = frames.reshape(-1, frames.shape[-1])
        # The RMS itself is retained as a learned feature; only the waveform
        # scale is bounded to make real microphone gain changes manageable.
        rms = x.square().mean(1, keepdim=True).sqrt()
        x = x / (rms + 1e-3)
        z = self.body(x[:, None]).flatten(1)
        z = self.proj(z) + torch.log1p(100.0 * rms)
        return z.reshape(*shape, -1)


class E2EImitator(nn.Module):
    """A single recurrent model from raw demonstration/self audio to actuators.

    ``adaptation`` is the controller state retained between songs.  It is never
    populated from explicit rig parameters; anything the model learns about the
    rig must be inferred from waveform/action history.
    """

    def __init__(self, config: E2EConfig = E2EConfig()):
        super().__init__()
        self.config = config
        c = config
        self.audio = AudioEncoder(c.audio_width, c.audio_dim)
        # Neural ear: two learned latent measurements (octave-folded pitch and
        # voicing).  They are inferred inside the model from raw audio and are
        # concatenated to the unconstrained embedding; no external pitch track
        # enters the inference pipeline.
        self.ear = nn.Linear(c.audio_dim, 2)
        feature_dim = c.audio_dim + 2
        self.reference = nn.GRU(feature_dim, c.reference_hidden, batch_first=True, bidirectional=True)
        # Preserve the instantaneous learned ear feature alongside the
        # contextual bidirectional GRU memory.  The skip prevents exact pitch
        # evidence from being diluted while the GRU models phrase context.
        memory_dim = 2 * c.reference_hidden + feature_dim
        self.memory_key = nn.Linear(memory_dim, c.controller_hidden, bias=False)
        self.query = nn.Linear(c.controller_hidden, c.controller_hidden, bias=False)
        self.controller = nn.GRUCell(feature_dim + memory_dim + 11, c.controller_hidden)
        self.action = nn.Sequential(
            nn.Linear(c.controller_hidden + memory_dim + feature_dim + 8, c.controller_hidden),
            nn.SiLU(),
            nn.Linear(c.controller_hidden, 3),
        )
        # A feed-forward motor plan is decoded from the time-aligned reference
        # memory.  The recurrent action head then learns only the closed-loop
        # residual needed for the particular physical rig and recent sound.
        # Both paths are neural and are trained jointly as one policy.
        self.planner = nn.Sequential(
            nn.Linear(memory_dim, c.controller_hidden), nn.SiLU(), nn.Linear(c.controller_hidden, 2)
        )
        self.profile = nn.Sequential(
            nn.Linear(memory_dim, c.controller_hidden), nn.SiLU(), nn.Linear(c.controller_hidden, 1)
        )
        self.self_position = nn.Sequential(
            nn.Linear(feature_dim, c.controller_hidden), nn.SiLU(), nn.Linear(c.controller_hidden, 1)
        )
        self.rig_estimator = nn.Sequential(
            nn.Linear(c.adaptation_dim, 32), nn.SiLU(), nn.Linear(32, 4)
        )
        # Articulation is feedforward: self sound must never create the circular
        # rule "open only after I already hear myself".  Pitch PWM and rig
        # adaptation still use the full self-audio recurrent path above.
        self.valve = nn.Sequential(
            nn.Linear(memory_dim, c.controller_hidden), nn.SiLU(), nn.Linear(c.controller_hidden, 1)
        )

    def encode_reference(self, frames: torch.Tensor, lengths: torch.Tensor | None = None):
        """Encode complete raw demonstrations; returns memory, keys and mask."""
        z = self.audio_features(frames)
        if lengths is None:
            recurrent, _ = self.reference(z)
            memory = torch.cat([recurrent, z], dim=-1)
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_memory, _ = self.reference(packed)
            recurrent, _ = nn.utils.rnn.pad_packed_sequence(
                packed_memory, batch_first=True, total_length=z.shape[1]
            )
            memory = torch.cat([recurrent, z], dim=-1)
            mask = torch.arange(z.shape[1], device=z.device)[None, :] < lengths[:, None]
        return memory, self.memory_key(memory), mask

    def audio_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Learned raw-audio embedding augmented by the internal neural ear."""
        z = self.audio(frames)
        return torch.cat([z, self.ear(z)], dim=-1)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.config.controller_hidden, device=device or next(self.parameters()).device)

    def step(self, self_frame: torch.Tensor, memory: torch.Tensor, keys: torch.Tensor,
             memory_mask: torch.Tensor, state: torch.Tensor, previous_action: torch.Tensor,
             new_song: torch.Tensor | None = None, reference_step=None):
        """One 10 ms control step.  Returns raw outputs, next state and attention."""
        if self_frame.ndim == 2:
            self_frame = self_frame[:, None, :]
        own = self.audio_features(self_frame)[:, 0]
        return self.step_encoded(own, memory, keys, memory_mask, state, previous_action, new_song, reference_step)

    def step_encoded(self, own: torch.Tensor, memory: torch.Tensor, keys: torch.Tensor,
                     memory_mask: torch.Tensor, state: torch.Tensor, previous_action: torch.Tensor,
                     new_song: torch.Tensor | None = None, reference_step=None):
        """One step from an already encoded self-audio frame (training fast path)."""
        if new_song is None:
            new_song = torch.zeros(len(own), 1, device=own.device, dtype=own.dtype)
        elif new_song.ndim == 1:
            new_song = new_song[:, None]
        # Preserve only the slow rig-adaptation subspace between performances;
        # clear transient tracking dynamics so they cannot accumulate drift.
        keep = torch.zeros_like(state)
        keep[:, :self.config.adaptation_dim] = state[:, :self.config.adaptation_dim]
        state = torch.where(new_song.bool(), keep, state)
        score = (keys * self.query(state)[:, None, :]).sum(-1) / math.sqrt(keys.shape[-1])
        score = score.masked_fill(~memory_mask, torch.finfo(score.dtype).min)
        attention = torch.softmax(score, 1)
        attention_context = (attention[:, :, None] * memory).sum(1)
        # Playback is clocked from the same 100 Hz timeline as the recorded
        # demonstration.  Give the policy the learned feature at that instant
        # as well as content-based attention.  This is still raw-audio E2E
        # (there is no pitch/note conversion), but avoids asking attention to
        # rediscover elapsed time in every repeated or silent passage.
        if reference_step is None:
            aligned = attention_context
            future_aligned = aligned
            context = attention_context
        else:
            index = torch.as_tensor(reference_step, device=memory.device, dtype=torch.long).flatten()
            if index.numel() == 1:
                index = index.expand(len(memory))
            last = memory_mask.sum(1) - 1
            index = torch.minimum(index, last).clamp_min(0)
            aligned = memory[torch.arange(len(memory), device=memory.device), index]
            future_index = torch.minimum(index + 10, last)
            future_aligned = memory[torch.arange(len(memory), device=memory.device), future_index]
            context = 0.5 * (attention_context + aligned)
        profile = torch.sigmoid(self.profile(aligned))
        future_profile = torch.sigmoid(self.profile(future_aligned))
        self_position = torch.sigmoid(self.self_position(own))
        target_pitch = aligned[:, -2:-1]
        future_pitch = future_aligned[:, -2:-1]
        motor_features = torch.cat([profile, self_position, profile - self_position,
                                    future_profile, future_profile - profile,
                                    target_pitch, future_pitch, future_pitch - target_pitch], 1)
        state = self.controller(torch.cat([own, context, previous_action, motor_features, new_song], 1), state)
        residual = self.action(torch.cat([state, context, own, motor_features], 1))
        plan = self.planner(aligned)
        raw = torch.stack([plan[:, 0] + 0.5 * residual[:, 0],
                           self.valve(aligned)[:, 0],
                           plan[:, 1] + residual[:, 2],
                           profile[:, 0], self_position[:, 0]], 1)
        return raw, state, attention

    def forward(self, reference_frames: torch.Tensor, self_frames: torch.Tensor,
                lengths: torch.Tensor | None = None, teacher_actions: torch.Tensor | None = None,
                state: torch.Tensor | None = None):
        """Run padded training sequences.

        ``teacher_actions`` supplies the previous PWM/valve during behaviour
        cloning.  Omitting it feeds the model's own previous action back.
        """
        memory, keys, mask = self.encode_reference(reference_frames, lengths)
        batch, steps = self_frames.shape[:2]
        own_sequence = self.audio_features(self_frames)
        state = self.initial_state(batch, self_frames.device) if state is None else state
        previous = torch.zeros(batch, 2, device=self_frames.device, dtype=self_frames.dtype)
        raws, attentions = [], []
        for t in range(steps):
            pulse = torch.ones(batch, device=self_frames.device) if t == 0 else None
            old_state = state
            raw, next_state, attn = self.step_encoded(own_sequence[:, t], memory, keys, mask, state, previous,
                                                       pulse, t)
            active = None if lengths is None else (t < lengths)[:, None]
            state = next_state if active is None else torch.where(active, next_state, old_state)
            raws.append(raw)
            attentions.append(attn)
            if teacher_actions is None:
                next_previous = torch.stack([torch.tanh(raw[:, 0]), torch.sigmoid(raw[:, 1])], 1)
            else:
                next_previous = teacher_actions[:, t]
            previous = next_previous if active is None else torch.where(active, next_previous, previous)
        return torch.stack(raws, 1), state, torch.stack(attentions, 1)

    def checkpoint(self) -> dict:
        return {"format": "yamabiko-e2e-v1", "config": asdict(self.config), "state": self.state_dict(),
                "sample_rate": SAMPLE_RATE, "frame": FRAME, "hop": HOP}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device="cpu") -> "E2EImitator":
        if checkpoint.get("format") != "yamabiko-e2e-v1":
            raise ValueError("not a Yamabiko E2E checkpoint")
        model = cls(E2EConfig(**checkpoint["config"])).to(device)
        incoming = checkpoint["state"]
        # v1 checkpoints made before the long-period audio front end used a
        # 15-sample first kernel.  Reuse all compatible learned control weights
        # while freshly initialising only that representation layer.
        mismatched = {key for key, value in incoming.items()
                      if key in model.state_dict() and value.shape != model.state_dict()[key].shape}
        incoming = {key: value for key, value in incoming.items() if key not in mismatched}
        missing, unexpected = model.load_state_dict(incoming, strict=False)
        allowed = {"valve.0.weight", "valve.0.bias", "valve.2.weight", "valve.2.bias",
                   "planner.0.weight", "planner.0.bias", "planner.2.weight", "planner.2.bias",
                   "audio.body.0.weight", "ear.weight", "ear.bias", "reference.weight_ih_l0",
                   "reference.weight_ih_l0_reverse", "controller.weight_ih", "action.0.weight",
                   "memory_key.weight", "query.weight", "planner.0.weight", "valve.0.weight",
                   "profile.0.weight", "profile.0.bias", "profile.2.weight", "profile.2.bias",
                   "self_position.0.weight", "self_position.0.bias",
                   "self_position.2.weight", "self_position.2.bias",
                   "rig_estimator.0.weight", "rig_estimator.0.bias",
                   "rig_estimator.2.weight", "rig_estimator.2.bias"}
        if unexpected or set(missing) - allowed:
            raise ValueError(f"incompatible E2E checkpoint: missing={missing}, unexpected={unexpected}")
        return model


class E2ERuntime:
    """Stateful inference wrapper.  Its recurrent state survives new songs."""

    def __init__(self, model: E2EImitator, device="cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.state = model.initial_state(1, self.device)
        self.previous = torch.zeros(1, 2, device=self.device)
        self.memory = self.keys = self.mask = None
        self.first = False
        self.position = 0

    @classmethod
    def load(cls, path, device="cpu") -> "E2ERuntime":
        checkpoint = torch.load(path, map_location=device)
        return cls(E2EImitator.from_checkpoint(checkpoint, device), device)

    @torch.inference_mode()
    def start_reference(self, audio, *, keep_adaptation: bool = True) -> int:
        frames = frame_audio(audio).to(self.device)[None]
        self.memory, self.keys, self.mask = self.model.encode_reference(frames)
        if not keep_adaptation:
            self.state.zero_()
        self.previous.zero_()
        self.first = True
        self.position = 0
        return frames.shape[1]

    @torch.inference_mode()
    def act(self, latest_self_audio) -> tuple[float, bool, float]:
        if self.memory is None:
            raise RuntimeError("start_reference must be called before act")
        x = torch.as_tensor(latest_self_audio, dtype=torch.float32, device=self.device).flatten()
        if x.numel() and x.abs().max() > 2.0:
            x = x / 32768.0
        if x.numel() < FRAME:
            x = torch.cat([torch.zeros(FRAME - x.numel(), device=self.device), x])
        x = x[-FRAME:][None]
        pulse = torch.ones(1, device=self.device) if self.first else None
        raw, self.state, _ = self.model.step(
            x, self.memory, self.keys, self.mask, self.state, self.previous, pulse, self.position
        )
        pwm = torch.tanh(raw[0, 0])
        valve_p = torch.sigmoid(raw[0, 1])
        done_p = torch.sigmoid(raw[0, 2])
        self.previous = torch.stack([pwm, valve_p])[None]
        self.first = False
        self.position += 1
        return float(pwm), bool(valve_p >= 0.5), float(done_p)
