"""Precompute every NN/deterministic stage choice for the report's hybrid lab."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.composite import NeuralPerformanceProfile, YamabikoComposite  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import (DeterministicEar, DeterministicProfile,
                                                       DeterministicYamabikoPipeline)  # noqa: E402


def compact(values):
    return np.round(np.asarray(values), 3).tolist()


def articulate(target, gap=4):
    padded = torch.cat([target, torch.zeros(*target.shape[:-1], 2, device=target.device)], -1)
    return YamabikoComposite._articulate(padded, gap)[..., :2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/yamabiko_connected_composite_v1.pt")
    ap.add_argument("--frames", default="runs/yamabiko_composite_onnx/demo_reference_frames.npy")
    ap.add_argument("--out", default="docs/e2e-composite-results/hybrid-lab.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    model = YamabikoComposite.from_checkpoint(
        torch.load(args.checkpoint, map_location=device, weights_only=False), device).eval()
    frames = torch.from_numpy(np.load(args.frames)).to(device)
    lengths = torch.tensor([frames.shape[1]], device=device)
    with torch.inference_mode():
        nn_features = model.ear.audio_features(frames)
        nn_profile = model.listen(frames, lengths)
        det_semantic = DeterministicEar().to(device)(frames)
    nn_semantic = nn_features[..., -2:]
    det_voice_logit = torch.where(det_semantic[..., 1] >= .5,
                                  torch.full_like(det_semantic[..., 1], 12.0),
                                  torch.full_like(det_semantic[..., 1], -12.0))
    det_semantic = torch.stack([det_semantic[..., 0], det_voice_logit], -1)
    semantic = {"nn": nn_semantic, "det": det_semantic}
    nn_ear_recalled = articulate(nn_semantic.repeat_interleave(model.tempo_scale, 1), model.articulation_frames)
    learned_memory_residual = nn_profile.target - nn_ear_recalled
    deterministic_control = DeterministicYamabikoPipeline(model.plant.config).to(device)
    results = {}

    for ear_kind in ("nn", "det"):
        selected_ear = semantic[ear_kind]
        for memory_kind in ("nn", "det"):
            upstream = articulate(selected_ear.repeat_interleave(model.tempo_scale, 1), model.articulation_frames)
            if memory_kind == "nn":
                target = upstream + learned_memory_residual
                target = torch.stack([target[..., 0].clamp(-1, 1), target[..., 1].clamp(-12, 12)], -1)
            else:
                target = upstream
            for planner_kind in ("nn", "det"):
                with torch.inference_mode():
                    position = (model.planner(target)[..., 0] if planner_kind == "nn"
                                else ((target[..., 0] + 1) * .5).clamp(0, 1))
                for control_kind in ("nn", "det"):
                    if control_kind == "nn":
                        profile = NeuralPerformanceProfile(target, position, torch.tensor([120.], device=device),
                                                           torch.tensor([60.], device=device),
                                                           torch.tensor([target.shape[1]], device=device), target)
                        slow = None
                        with torch.inference_mode():
                            for _ in range(3):
                                output = model.perform(profile, slow_context=slow)
                                slow = output["slow_context"]
                        emitted = output["pitch_cents"][0]
                        emitted_voice = output["valve"][0] >= .5
                    else:
                        voice = torch.sigmoid(target[..., 1]) >= .5
                        onset = torch.nn.functional.pad(voice[:, 1:] & ~voice[:, :-1], (1, 0))
                        profile = DeterministicProfile(target[..., 0], voice, onset,
                                                       torch.tensor([60.], device=device))
                        with torch.inference_mode():
                            output = deterministic_control.perform(profile, position)
                        emitted = output["pitch_cents"][0]; emitted_voice = voice[0]
                    wanted = 1300 + 600 * target[0, :, 0]
                    valid = (torch.sigmoid(target[0, :, 1]) >= .5) & emitted_voice
                    mae = (emitted[valid] - wanted[valid]).abs().mean().item() if valid.any() else None
                    key = "-".join((ear_kind, memory_kind, planner_kind, control_kind))
                    results[key] = {
                        "choices": {"ear": ear_kind, "memory": memory_kind,
                                    "planner": planner_kind, "control": control_kind},
                        "metrics": {"final_pitch_mae_cents": mae,
                                    "voiced_fraction": emitted_voice.float().mean().item()},
                        "ear": {"pitch_cents": compact(1300 + 600 * selected_ear[0, :, 0].cpu()),
                                "voice": selected_ear[0, :, 1].ge(0).to(torch.uint8).cpu().tolist()},
                        "memory": {"pitch_cents": compact(wanted.cpu()),
                                   "voice": target[0, :, 1].ge(0).to(torch.uint8).cpu().tolist()},
                        "planner": {"position": compact(position[0].cpu()),
                                    "nominal_cents": compact((700 + 1200 * position[0]).cpu()),
                                    "voice": target[0, :, 1].ge(0).to(torch.uint8).cpu().tolist()},
                        "performance": {"pitch_cents": compact(emitted.cpu()),
                                        "voice": emitted_voice.to(torch.uint8).cpu().tolist()},
                    }
    document = {
        "format": "yamabiko-hybrid-lab-v1", "sample": "4音ステップ（2.6秒のお手本 / 0.5倍速再生）",
        "frame_rate": 100, "reference_frame_rate": 100,
        "note": ("全16組合せを実モデルと決定論モジュールで事前評価。NN MemoryへDet Earを渡す場合は、"
                 "共通pitch/voice契約上で学習済みMemory residualを加えるrepresentation adapterを使用。"
                 "NN Controlは遅い適応状態を保持した3回目。"),
        "combinations": results,
    }
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({len(results)} combinations, {out.stat().st_size} bytes)")


if __name__ == "__main__": main()
