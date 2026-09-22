"""Export the connected Yamabiko model as UNO Q ONNX inference graphs."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.audio import room, synth_source  # noqa: E402
from flute_rl.yamabiko.composite import YamabikoComposite  # noqa: E402
from flute_rl.yamabiko.e2e_io import frame_audio_numpy  # noqa: E402


def onnx_audio_features(ear, frames):
    """Equivalent AudioEncoder path with a fixed-window pool ONNX can shape dynamically."""
    shape = frames.shape[:-1]; x = frames.reshape(-1, frames.shape[-1])
    rms = x.square().mean(1, keepdim=True).sqrt(); x = x / (rms + 1e-3)
    # A 320-sample input is 20 samples wide after the four strided convolutions.
    # AdaptiveAvgPool1d(4) is therefore exactly AvgPool1d(5, 5), but the latter
    # remains exportable when the number of reference frames is dynamic.
    z = ear.audio.body[:-1](x[:, None])
    z = torch.nn.functional.avg_pool1d(z, 5, 5).flatten(1)
    z = ear.audio.proj(z) + torch.log1p(100.0 * rms)
    z = z.reshape(*shape, -1)
    return torch.cat([z, ear.ear(z)], -1)


class ListenGraph(nn.Module):
    """One-shot raw-reference graph; all frames are valid for UNO Q recording."""

    def __init__(self, model: YamabikoComposite):
        super().__init__(); self.model = model

    def forward(self, frames):
        features = onnx_audio_features(self.model.ear, frames)
        encoded, _ = self.model.tempo.encoder(features)
        pooled = encoded.mean(1)
        bpm_log = self.model.tempo.tempo(pooled)[:, 0]
        phase = self.model.tempo.phase(encoded)
        phase_xy = phase[..., :2] / phase[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        beat = torch.cat([phase_xy, phase[..., 2:3]], -1)
        source = torch.cat([features, encoded, beat], -1)
        memory, _ = self.model.timeline.encoder(source)
        stored = torch.cat([memory, features[..., -2:]], -1)
        raw = self.model.timeline.decoder(stored)
        pitch = raw[..., :1] if self.model.timeline.direct_pitch else stored[..., -2:-1] + .1 * raw[..., :1]
        decoded = torch.cat([pitch, raw[..., 1:]], -1).repeat_interleave(self.model.tempo_scale, 1)

        voice = torch.sigmoid(decoded[..., 1]) >= .5
        pitch_change = torch.nn.functional.pad((decoded[:, 1:, 0] - decoded[:, :-1, 0]).abs() > .08, (1, 0))
        voice_edge = torch.nn.functional.pad(voice[:, 1:] & ~voice[:, :-1], (1, 0))
        learned_onset = torch.sigmoid(decoded[..., 2]) >= .5
        learned_edge = learned_onset & ~torch.nn.functional.pad(learned_onset[:, :-1], (1, 0))
        onset = voice_edge | learned_edge | (pitch_change & voice)
        closed = onset
        for delay in range(1, self.model.articulation_frames):
            closed = closed | torch.nn.functional.pad(onset[:, :-delay], (delay, 0))
        voice_logit = torch.where(closed, torch.full_like(decoded[..., 1], -12.0), decoded[..., 1])
        target = torch.stack([decoded[..., 0], voice_logit], -1)
        position = self.model.planner(target)[..., 0]
        bpm = self.model.tempo.bpm(bpm_log)
        return target, position, bpm, stored


class ControlStepGraph(nn.Module):
    """One 100 Hz neural step; the host supplies four planner look-aheads."""

    def __init__(self, model: YamabikoComposite):
        super().__init__(); self.ear = model.ear; self.controller = model.controller
        self.comparator = model.comparator; self.feedback = model.feedback

    def forward(self, plan_taps, target, target_future_pitch, self_frame, sounding,
                previous_pwm, previous_error, previous_heard, controller_state, feedback_state):
        voice = torch.sigmoid(target[:, 1])
        features = torch.stack([plan_taps[:, 0], plan_taps[:, 1], plan_taps[:, 2], plan_taps[:, 3],
                                plan_taps[:, 1] - plan_taps[:, 0],
                                plan_taps[:, 2] - plan_taps[:, 0], voice, previous_pwm], 1)
        controller_state = self.controller.cell(features, controller_state)
        estimated = self.controller.latent_motor_features(controller_state)
        control = torch.cat([estimated, plan_taps[:, :1] - estimated[:, :1],
                             plan_taps[:, 2:3] - estimated[:, :1]], 1)
        raw = self.controller.head(torch.cat([controller_state, features, control], 1))
        base_pwm, valve_logit = torch.tanh(raw[:, 0]), raw[:, 1]

        own = onnx_audio_features(self.ear, self_frame[:, None])[:, 0, -2:]
        heard = own[:, 0]
        valid = (sounding >= .5) & (torch.sigmoid(own[:, 1]) >= .5)
        heard_for_feedback = torch.where(valid, heard, previous_heard)
        comparison = self.comparator(target, own)
        target_physical = ((target[:, 0] + 1.0) * .5).clamp(0, 1)
        heard_physical = ((heard_for_feedback + 1.0) * .5).clamp(0, 1)
        residual, error, feedback_state = self.feedback.step(
            target_physical, heard_physical, valid, voice, base_pwm, previous_pwm,
            previous_error, feedback_state,
            target_future=((target_future_pitch + 1.0) * .5).clamp(0, 1),
            previous_heard=((previous_heard + 1.0) * .5).clamp(0, 1),
            external_error=.5 * comparison[:, 0])
        pwm = (base_pwm + residual).clamp(-1, 1)
        valve = torch.sigmoid(valve_logit) * (voice >= .5).to(target.dtype)
        next_heard = torch.where(valid, heard, previous_heard)
        return (pwm, valve, base_pwm, residual, heard, valid.to(target.dtype), comparison[:, 0],
                error, next_heard, controller_state, feedback_state)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/yamabiko_connected_composite_v1.pt")
    ap.add_argument("--out", default="runs/yamabiko_composite_onnx")
    ap.add_argument("--reference-steps", type=int, default=260)
    args = ap.parse_args(); out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = YamabikoComposite.from_checkpoint(checkpoint).eval()
    frames = torch.randn(1, args.reference_steps, 320)
    listen_path, step_path = out / "listen.onnx", out / "control_step.onnx"
    with torch.inference_mode():
        torch.onnx.export(ListenGraph(model), (frames,), listen_path, opset_version=18,
                          input_names=["reference_frames"],
                          output_names=["target", "position", "bpm", "stored_timeline"],
                          dynamic_axes={"reference_frames": {1: "reference_steps"},
                                        "target": {1: "playback_steps"},
                                        "position": {1: "playback_steps"},
                                        "stored_timeline": {1: "reference_steps"}},
                          dynamo=False)
        b, ch, fh = 1, model.controller.hidden, model.feedback.hidden
        inputs = (torch.rand(b, 4), torch.rand(b, 2), torch.rand(b), torch.rand(b, 320),
                  torch.zeros(b), torch.zeros(b), torch.zeros(b), torch.zeros(b),
                  torch.zeros(b, ch), torch.zeros(b, 2 * fh))
        input_names = ["plan_taps", "target", "target_future_pitch", "self_frame", "sounding",
                       "previous_pwm", "previous_error", "previous_heard",
                       "controller_state", "feedback_state"]
        output_names = ["pwm", "valve", "base_pwm", "residual_pwm", "heard_pitch", "heard_valid",
                        "comparator_error", "error", "next_heard", "next_controller_state",
                        "next_feedback_state"]
        torch.onnx.export(ControlStepGraph(model), inputs, step_path, opset_version=18,
                          input_names=input_names, output_names=output_names, dynamo=False)

    rng = np.random.default_rng(88031)
    target = np.r_[np.full(40, np.nan), np.full(55, 1000.), np.full(55, 1450.),
                   np.full(55, 1200.), np.full(55, 1650.)]
    waveform, _ = synth_source(target, "recorder", rng, sr=16_000, shift=0)
    demo = frame_audio_numpy(room(waveform, 16_000, rng), len(target))[None].astype(np.float32)
    np.save(out / "demo_reference_frames.npy", demo)
    manifest = {"format": "yamabiko-composite-onnx-v1", "opset": 18,
                "reference_steps": args.reference_steps, "tempo_scale": model.tempo_scale,
                "controller_hidden": ch, "feedback_hidden": fh,
                "checkpoint": args.checkpoint, "graphs": {"listen": listen_path.name, "control": step_path.name},
                "physical_plant": "deterministic NumPy host simulator; replace with MCU I/O on the real rig"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
