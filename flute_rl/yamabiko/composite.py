"""Connected neural/physical Yamabiko inference graph.

The class in this module is the deployable composition missing from the
individual Gate experiments.  It listens once, stores a neural timeline, then
can perform it repeatedly while retaining only the slow feedback context.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import pathlib

import torch
from torch import nn

from .beat_grid import BeatGridConfig, BeatTimelineRecallNet, TempoBeatNet
from .e2e import E2EImitator
from .physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig
from .staged_nn import (AcousticFeedbackResidual, ErrorComparator,
                        MotorTrajectoryController, StagedConfig,
                        TargetPositionPlanner)


@dataclass
class NeuralPerformanceProfile:
    target: torch.Tensor          # normalized pitch + voice logit, (B,T,2)
    position: torch.Tensor        # planned stroke, (B,T)
    bpm: torch.Tensor             # original detected BPM, (B,)
    playback_bpm: torch.Tensor    # mechanically feasible replay BPM, (B,)
    lengths: torch.Tensor         # replay lengths at 100 Hz, (B,)
    stored_timeline: torch.Tensor # sufficient reference memory; raw audio may be discarded


@dataclass
class RendererState:
    previous_samples: torch.Tensor
    phase: torch.Tensor


class DeterministicFluteRenderer(nn.Module):
    """Deterministically turn simulated pitch into causal 16 kHz ear frames."""

    FRAME = 320
    HOP = 160

    def __init__(self, sample_rate=16_000):
        super().__init__(); self.sample_rate = sample_rate

    def initial_state(self, batch, device, dtype=torch.float32):
        return RendererState(torch.zeros(batch, self.HOP, device=device, dtype=dtype),
                             torch.zeros(batch, device=device, dtype=dtype))

    def step(self, cents: torch.Tensor, sounding: torch.Tensor, state: RendererState):
        frequency = 440.0 * torch.pow(2.0, cents / 1200.0)
        index = torch.arange(1, self.HOP + 1, device=cents.device, dtype=cents.dtype)[None]
        angle = state.phase[:, None] + 2 * torch.pi * frequency[:, None] * index / self.sample_rate
        fresh = (.24 * torch.sin(angle) + .05 * torch.sin(2 * angle)) * sounding[:, None].to(cents.dtype)
        frame = torch.cat([state.previous_samples, fresh], 1)
        phase = torch.remainder(angle[:, -1], 2 * torch.pi)
        return frame, RendererState(fresh, phase)


class YamabikoComposite(nn.Module):
    """Raw reference audio to adaptive physical performance in one graph."""

    def __init__(self, ear: E2EImitator, tempo: TempoBeatNet,
                 timeline: BeatTimelineRecallNet, planner: TargetPositionPlanner,
                 controller: MotorTrajectoryController, comparator: ErrorComparator,
                 feedback: AcousticFeedbackResidual,
                 plant_config: PhysicalPlantConfig = PhysicalPlantConfig(),
                 tempo_scale: int = 2, articulation_frames: int = 4):
        super().__init__()
        self.ear, self.tempo, self.timeline, self.planner = ear, tempo, timeline, planner
        self.controller, self.comparator, self.feedback = controller, comparator, feedback
        self.plant = DifferentiableMotorFlute(plant_config)
        self.renderer = DeterministicFluteRenderer()
        self.tempo_scale, self.articulation_frames = tempo_scale, articulation_frames

    @classmethod
    def from_checkpoints(cls, ear_path, timeline_position_path, comparator_path,
                         physical_path, device="cpu"):
        load = lambda path: torch.load(pathlib.Path(path), map_location=device, weights_only=False)
        ear_ck, timeline_ck, comparator_ck, physical_ck = map(
            load, (ear_path, timeline_position_path, comparator_path, physical_path))
        ear = E2EImitator.from_checkpoint(ear_ck, device)
        beat_config = BeatGridConfig(**timeline_ck["config"])
        tempo = TempoBeatNet(beat_config).to(device); tempo.load_state_dict(timeline_ck["tempo_beat"])
        timeline = BeatTimelineRecallNet(
            beat_config, direct_pitch=timeline_ck.get("timeline_memory_kind") == "direct-pitch-v1").to(device)
        timeline.load_state_dict(timeline_ck["timeline_memory"])
        planner = TargetPositionPlanner().to(device); planner.load_state_dict(timeline_ck["timeline_position_planner"])
        staged_config = StagedConfig(**comparator_ck["config"])
        comparator = ErrorComparator(staged_config).to(device); comparator.load_state_dict(comparator_ck["comparator"])
        controller = MotorTrajectoryController().to(device); controller.load_state_dict(physical_ck["controller"])
        plant_config = PhysicalPlantConfig.from_dict(physical_ck["plant_config"])
        feedback = AcousticFeedbackResidual(limit=plant_config.feedback_limit).to(device)
        feedback.load_state_dict(physical_ck["feedback"])
        return cls(ear, tempo, timeline, planner, controller, comparator, feedback, plant_config).to(device)

    def checkpoint(self):
        """Return one self-contained checkpoint for the connected graph."""
        return {
            "format": "yamabiko-connected-composite-v1",
            "ear_config": asdict(self.ear.config), "ear": self.ear.state_dict(),
            "beat_config": asdict(self.tempo.config), "tempo": self.tempo.state_dict(),
            "timeline": self.timeline.state_dict(), "timeline_direct_pitch": self.timeline.direct_pitch,
            "timeline_pitch_residual": self.timeline.pitch_residual,
            "planner": self.planner.state_dict(), "controller": self.controller.state_dict(),
            "controller_hidden": self.controller.hidden,
            "comparator_hidden": self.comparator.net[0].out_features,
            "comparator": self.comparator.state_dict(), "feedback": self.feedback.state_dict(),
            "feedback_hidden": self.feedback.hidden,
            "plant_config": asdict(self.plant.config), "tempo_scale": self.tempo_scale,
            "articulation_frames": self.articulation_frames,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device="cpu"):
        from .e2e import E2EConfig
        ear = E2EImitator(E2EConfig(**checkpoint["ear_config"])).to(device)
        ear.load_state_dict(checkpoint["ear"])
        beat_config = BeatGridConfig(**checkpoint["beat_config"])
        tempo = TempoBeatNet(beat_config).to(device); tempo.load_state_dict(checkpoint["tempo"])
        timeline = BeatTimelineRecallNet(beat_config, direct_pitch=checkpoint["timeline_direct_pitch"],
                                         pitch_residual=checkpoint.get("timeline_pitch_residual", .10)).to(device)
        timeline.load_state_dict(checkpoint["timeline"])
        planner = TargetPositionPlanner().to(device); planner.load_state_dict(checkpoint["planner"])
        controller = MotorTrajectoryController(hidden=checkpoint["controller_hidden"]).to(device)
        controller.load_state_dict(checkpoint["controller"])
        comparator = ErrorComparator(StagedConfig(comparator_hidden=checkpoint["comparator_hidden"])).to(device)
        comparator.load_state_dict(checkpoint["comparator"])
        plant_config = PhysicalPlantConfig.from_dict(checkpoint["plant_config"])
        feedback = AcousticFeedbackResidual(hidden=checkpoint["feedback_hidden"],
                                            limit=plant_config.feedback_limit).to(device)
        feedback.load_state_dict(checkpoint["feedback"])
        return cls(ear, tempo, timeline, planner, controller, comparator, feedback, plant_config,
                   checkpoint["tempo_scale"], checkpoint["articulation_frames"]).to(device)

    @staticmethod
    def _articulate(decoded: torch.Tensor, gap: int):
        result = decoded.clone()
        voice = torch.sigmoid(result[..., 1]) >= .5
        pitch_change = torch.nn.functional.pad((result[:, 1:, 0] - result[:, :-1, 0]).abs() > .08, (1, 0))
        voice_edge = torch.nn.functional.pad(voice[:, 1:] & ~voice[:, :-1], (1, 0))
        learned_onset = torch.sigmoid(result[..., 2]) >= .5 if result.shape[-1] > 2 else voice_edge
        learned_edge = learned_onset & ~torch.nn.functional.pad(learned_onset[:, :-1], (1, 0))
        onset = voice_edge | learned_edge | (pitch_change & voice)
        for row in range(len(result)):
            for start in torch.nonzero(onset[row], as_tuple=False)[:, 0].tolist():
                result[row, start:min(result.shape[1], start + gap), 1] = -12.0
        return result

    def listen(self, reference_frames: torch.Tensor, lengths: torch.Tensor) -> NeuralPerformanceProfile:
        features = self.ear.audio_features(reference_frames)
        bpm_log, beat, encoded, _ = self.tempo(features, lengths)
        decoded, stored = self.timeline(features, encoded, beat, lengths)
        expanded = decoded.repeat_interleave(self.tempo_scale, 1)
        expanded = self._articulate(expanded, self.articulation_frames)
        target = expanded[..., :2]
        position = self.planner(target)[..., 0]
        bpm = self.tempo.bpm(bpm_log)
        return NeuralPerformanceProfile(target, position, bpm, bpm / self.tempo_scale,
                                        lengths * self.tempo_scale, stored)

    def initial_feedback_context(self, batch, device, dtype=torch.float32):
        return torch.zeros(batch, self.feedback.hidden, device=device, dtype=dtype)

    def perform(self, profile: NeuralPerformanceProfile, parameters=None,
                slow_context: torch.Tensor | None = None):
        plan, target = profile.position, profile.target
        batch, steps = plan.shape; device, dtype = plan.device, plan.dtype
        parameters = parameters or self.plant.parameters(batch, device, dtype)
        physical = self.plant.initial_state(batch, device, dtype)
        control_state = self.controller.initial_state(batch, device, dtype)
        fast = torch.zeros(batch, self.feedback.hidden, device=device, dtype=dtype)
        slow = (self.initial_feedback_context(batch, device, dtype) if slow_context is None else slow_context)
        feedback_state = torch.cat([fast, slow], 1)
        render_state = self.renderer.initial_state(batch, device, dtype)
        previous_pwm = torch.zeros(batch, device=device, dtype=dtype)
        previous_error = torch.zeros_like(previous_pwm); previous_heard = torch.zeros_like(previous_pwm)
        previous_valve = torch.zeros_like(previous_pwm)
        outputs = {key: [] for key in ("pwm", "base_pwm", "residual_pwm", "valve", "position",
                                        "pitch_cents", "heard_pitch", "comparator_error")}
        for t in range(steps):
            base, valve_logit, control_state = self.controller.step(plan, torch.sigmoid(target[..., 1]),
                                                                    previous_pwm, control_state, t)
            cents, sounding = self.plant.flute(physical, previous_valve, parameters)
            self_frame, render_state = self.renderer.step(cents, sounding, render_state)
            own_features = self.ear.audio_features(self_frame[:, None])[:, 0, -2:]
            heard = own_features[:, 0]
            valid = sounding & (torch.sigmoid(own_features[:, 1]) >= .5)
            heard_for_feedback = torch.where(valid, heard, previous_heard)
            comparison = self.comparator(target[:, t], own_features)
            # Neural Ear/Timeline pitch uses (cents-1300)/600 in [-1,1], while
            # the physical feedback policy was trained on stroke pitch [0,1].
            target_physical = ((target[:, t, 0] + 1.0) * .5).clamp(0, 1)
            heard_physical = ((heard_for_feedback + 1.0) * .5).clamp(0, 1)
            residual, previous_error, feedback_state = self.feedback.step(
                target_physical, heard_physical, valid, torch.sigmoid(target[:, t, 1]), base,
                previous_pwm, previous_error, feedback_state,
                target_future=((target[:, min(t + 20, steps - 1), 0] + 1.0) * .5).clamp(0, 1),
                previous_heard=((previous_heard + 1.0) * .5).clamp(0, 1),
                external_error=.5 * comparison[:, 0])
            pwm = (base + residual).clamp(-1, 1)
            valve = torch.sigmoid(valve_logit) * (torch.sigmoid(target[:, t, 1]) >= .5).to(dtype)
            physical = self.plant.step(physical, pwm, parameters)
            emitted_cents, _ = self.plant.flute(physical, valve, parameters)
            for key, value in (("pwm", pwm), ("base_pwm", base), ("residual_pwm", residual),
                               ("valve", valve), ("position", physical.position),
                               ("pitch_cents", emitted_cents), ("heard_pitch", heard),
                               ("comparator_error", comparison[:, 0])):
                outputs[key].append(value)
            previous_pwm, previous_valve = pwm, valve
            previous_heard = torch.where(valid, heard, previous_heard)
        result = {key: torch.stack(value, 1) for key, value in outputs.items()}
        result["slow_context"] = feedback_state[:, self.feedback.hidden:]
        return result

    def forward(self, reference_frames: torch.Tensor, lengths: torch.Tensor,
                repetitions: int = 1, parameters=None):
        profile = self.listen(reference_frames, lengths)
        performances, context = [], None
        for _ in range(repetitions):
            result = self.perform(profile, parameters=parameters, slow_context=context)
            context = result["slow_context"]
            performances.append(result)
        return profile, performances
