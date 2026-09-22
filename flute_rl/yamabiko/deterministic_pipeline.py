"""Deterministic reference modules for every Yamabiko inference boundary.

These modules are deliberately non-learned.  They share semantic contracts
with the neural stages and provide debuggable baselines, oracle tests, and a
way to replace one gate at a time without changing the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig


@dataclass
class DeterministicProfile:
    pitch: torch.Tensor       # normalized pitch, (B,T)
    voice: torch.Tensor       # boolean, (B,T)
    onset: torch.Tensor       # boolean, (B,T)
    bpm: torch.Tensor         # (B,)


class DeterministicEar(nn.Module):
    """Windowed-FFT pitch/voice baseline for 16 kHz, 320-sample frames."""

    def __init__(self, sample_rate=16_000, pitch_center_cents=1300.0, pitch_scale_cents=600.0,
                 rms_threshold=.008):
        super().__init__(); self.sample_rate = sample_rate
        self.pitch_center_cents, self.pitch_scale_cents = pitch_center_cents, pitch_scale_cents
        self.rms_threshold = rms_threshold

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(frames.shape[-1], device=frames.device, dtype=frames.dtype)
        spectrum = torch.fft.rfft(frames * window).abs()
        frequencies = torch.fft.rfftfreq(frames.shape[-1], 1 / self.sample_rate).to(frames.device)
        useful = (frequencies >= 500) & (frequencies <= 1600)
        local = spectrum[..., useful]
        frequency = frequencies[useful][local.argmax(-1)]
        cents = 1200.0 * torch.log2(frequency.clamp_min(1.0) / 440.0)
        pitch = ((cents - self.pitch_center_cents) / self.pitch_scale_cents).clamp(-1, 1)
        rms = frames.square().mean(-1).sqrt()
        voice = rms >= self.rms_threshold
        return torch.stack([pitch, voice.to(frames.dtype)], -1)


class DeterministicTempoBeat(nn.Module):
    """Autocorrelation tempo baseline over pitch/voice event strength."""

    def __init__(self, frame_rate=100.0, minimum_bpm=40, maximum_bpm=240):
        super().__init__(); self.frame_rate = frame_rate
        self.minimum_bpm, self.maximum_bpm = minimum_bpm, maximum_bpm

    def forward(self, ear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        voice = ear[..., 1]
        pitch_change = torch.nn.functional.pad((ear[:, 1:, 0] - ear[:, :-1, 0]).abs(), (1, 0))
        onset = torch.nn.functional.pad((voice[:, 1:] - voice[:, :-1]).clamp_min(0), (1, 0))
        strength = onset + pitch_change
        min_lag = max(1, round(self.frame_rate * 60 / self.maximum_bpm))
        max_lag = min(strength.shape[1] - 1, round(self.frame_rate * 60 / self.minimum_bpm))
        scores = []
        for lag in range(min_lag, max_lag + 1):
            scores.append((strength[:, lag:] * strength[:, :-lag]).mean(1))
        score = torch.stack(scores, 1)
        lag = score.argmax(1) + min_lag
        bpm = self.frame_rate * 60.0 / lag.to(ear.dtype)
        time = torch.arange(ear.shape[1], device=ear.device, dtype=ear.dtype)[None]
        phase = torch.remainder(time * bpm[:, None] / (60 * self.frame_rate), 1.0)
        return bpm, phase


class DeterministicTimelineMemory(nn.Module):
    """Exact semantic memory with deterministic tempo scaling/articulation."""

    def remember(self, ear: torch.Tensor, bpm: torch.Tensor) -> DeterministicProfile:
        voice = ear[..., 1] >= .5
        changed = torch.nn.functional.pad((ear[:, 1:, 0] - ear[:, :-1, 0]).abs() > .02, (1, 0))
        onset = torch.nn.functional.pad((voice[:, 1:] & ~voice[:, :-1]), (1, 0)) | (changed & voice)
        return DeterministicProfile(ear[..., 0].clone(), voice.clone(), onset, bpm.clone())

    def recall(self, profile: DeterministicProfile, tempo_scale=2, articulation_frames=4) -> DeterministicProfile:
        if int(tempo_scale) != tempo_scale or tempo_scale < 1:
            raise ValueError("deterministic baseline currently requires an integer tempo_scale >= 1")
        scale = int(tempo_scale)
        pitch = profile.pitch.repeat_interleave(scale, 1)
        voice = profile.voice.repeat_interleave(scale, 1)
        onset = profile.onset.repeat_interleave(scale, 1)
        for row in range(len(voice)):
            starts = torch.nonzero(onset[row], as_tuple=False)[:, 0]
            for start in starts.tolist():
                voice[row, start:min(len(voice[row]), start + articulation_frames)] = False
        return DeterministicProfile(pitch, voice, onset, profile.bpm / scale)


class DeterministicPositionPlanner(nn.Module):
    """Linear flute inverse: normalized musical pitch equals stroke fraction."""

    def forward(self, profile: DeterministicProfile) -> torch.Tensor:
        return ((profile.pitch + 1.0) * .5).clamp(0, 1)


class DeterministicMotorController(nn.Module):
    """Explicit PD reference controller for simulator/oracle diagnostics."""

    def __init__(self, kp=6.0, kd=1.0):
        super().__init__(); self.kp, self.kd = kp, kd

    def forward(self, target_position, actual_position, actual_velocity):
        return (self.kp * (target_position - actual_position) - self.kd * actual_velocity).clamp(-1, 1)


class DeterministicComparator(nn.Module):
    def forward(self, target_pitch, heard_pitch, valid=None):
        error = target_pitch - heard_pitch
        return error if valid is None else torch.where(valid, error, torch.zeros_like(error))


@dataclass
class PIDState:
    integral: torch.Tensor
    previous_error: torch.Tensor


class DeterministicFeedback(nn.Module):
    """Bounded PID residual baseline using only audible signed pitch error."""

    def __init__(self, kp=2.0, ki=.08, kd=.25, limit=.45):
        super().__init__(); self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit

    def initial_state(self, batch, device, dtype=torch.float32):
        zero = torch.zeros(batch, device=device, dtype=dtype)
        return PIDState(zero, zero.clone())

    def step(self, error, valid, state: PIDState):
        error = torch.where(valid, error, torch.zeros_like(error))
        integral = (state.integral + error).clamp(-2, 2)
        derivative = error - state.previous_error
        residual = (self.kp * error + self.ki * integral + self.kd * derivative).clamp(-self.limit, self.limit)
        return residual, PIDState(integral, error)


class DeterministicYamabikoPipeline(nn.Module):
    """Fully connected, non-learned oracle/reference implementation.

    The PD controller intentionally reads simulator position/velocity, so this
    class is a diagnostic upper baseline rather than the encoder-less deployed
    path.  Every other stage uses observable audio/semantic values.
    """

    def __init__(self, config: PhysicalPlantConfig = PhysicalPlantConfig(), tempo_scale=2,
                 articulation_frames=4):
        super().__init__(); self.ear = DeterministicEar()
        self.tempo = DeterministicTempoBeat(); self.memory = DeterministicTimelineMemory()
        self.planner = DeterministicPositionPlanner(); self.controller = DeterministicMotorController()
        self.comparator = DeterministicComparator(); self.feedback = DeterministicFeedback(limit=config.feedback_limit)
        self.plant = DifferentiableMotorFlute(config)
        self.tempo_scale, self.articulation_frames = tempo_scale, articulation_frames

    def listen(self, reference_frames: torch.Tensor):
        ear = self.ear(reference_frames); bpm, _ = self.tempo(ear)
        stored = self.memory.remember(ear, bpm)
        profile = self.memory.recall(stored, self.tempo_scale, self.articulation_frames)
        return profile, self.planner(profile)

    def perform(self, profile: DeterministicProfile, position_plan: torch.Tensor, parameters=None):
        batch, steps = position_plan.shape; device, dtype = position_plan.device, position_plan.dtype
        parameters = parameters or self.plant.parameters(batch, device, dtype)
        physical = self.plant.initial_state(batch, device, dtype)
        feedback_state = self.feedback.initial_state(batch, device, dtype)
        outputs = {key: [] for key in ("pwm", "position", "pitch_cents", "error")}
        for t in range(steps):
            cents, sounding = self.plant.flute(physical, profile.voice[:, t].to(dtype), parameters)
            heard = ((cents - self.plant.config.low_cents) /
                     self.plant.config.pitch_span_cents).clamp(0, 1)
            target = ((profile.pitch[:, t] + 1.0) * .5).clamp(0, 1)
            error = self.comparator(target, heard, sounding)
            residual, feedback_state = self.feedback.step(error, sounding, feedback_state)
            base = self.controller(position_plan[:, t], physical.position, physical.velocity)
            pwm = (base + residual).clamp(-1, 1)
            physical = self.plant.step(physical, pwm, parameters)
            emitted, _ = self.plant.flute(physical, profile.voice[:, t].to(dtype), parameters)
            for key, value in (("pwm", pwm), ("position", physical.position),
                               ("pitch_cents", emitted), ("error", error)):
                outputs[key].append(value)
        return {key: torch.stack(value, 1) for key, value in outputs.items()}

    def forward(self, reference_frames: torch.Tensor, parameters=None):
        profile, position = self.listen(reference_frames)
        return profile, position, self.perform(profile, position, parameters)
