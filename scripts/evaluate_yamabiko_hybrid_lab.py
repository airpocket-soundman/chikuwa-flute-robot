"""Precompute ten reference songs across every NN/deterministic stage choice."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.audio import room, synth_source  # noqa: E402
from flute_rl.yamabiko.composite import NeuralPerformanceProfile, YamabikoComposite  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import (DeterministicEar, DeterministicProfile,
                                                       DeterministicYamabikoPipeline)  # noqa: E402
from flute_rl.yamabiko.e2e_io import frame_audio_numpy  # noqa: E402


def compact(values):
    return np.round(np.asarray(values), 2).tolist()


def track(pitch, voice):
    return {"pitch_cents": compact(pitch),
            "voice": np.asarray(voice, dtype=np.uint8).tolist()}


def articulate(target, gap=4):
    padded = torch.cat([target, torch.zeros(*target.shape[:-1], 2, device=target.device)], -1)
    return YamabikoComposite._articulate(padded, gap)[..., :2]


def note_sequence(notes, durations, leading=24, trailing=16):
    """Build a 100 Hz pitch track; ``None`` denotes a rest."""
    parts = [np.full(leading, np.nan)]
    parts.extend(np.full(duration, np.nan if note is None else float(note))
                 for note, duration in zip(notes, durations))
    parts.append(np.full(trailing, np.nan))
    return np.concatenate(parts)


def sample_specs():
    """Ten short patterns that isolate contour, rests, repetition and jumps."""
    return [
        ("four-step", "4音ステップ", "上下する4音。従来サンプルとの比較用。",
         [1000, 1450, 1200, 1650], [55] * 4),
        ("ascending-scale", "上昇音階", "順次上がる音程を追えるかを見る。",
         [900, 1050, 1200, 1350, 1500, 1650], [34] * 6),
        ("descending-scale", "下降音階", "順次下がる音程を追えるかを見る。",
         [1700, 1540, 1380, 1220, 1060, 900], [34] * 6),
        ("zigzag", "ジグザグ旋律", "上昇と下降が交互に現れる。",
         [1050, 1550, 1150, 1650, 1250, 1450], [34] * 6),
        ("articulation", "同音連打と休符", "同じ音を休符で区切って再発音できるかを見る。",
         [1250, None, 1250, None, 1250, None, 1250], [38, 14, 38, 14, 38, 14, 38]),
        ("large-leaps", "大跳躍", "モーターに厳しい大きな音程跳躍。",
         [900, 1750, 980, 1680, 1080], [41] * 5),
        ("rhythm", "長短リズム", "音価の長短を記憶できるかを見る。",
         [1100, 1350, 1500, 1250, 1600], [62, 24, 46, 24, 50]),
        ("arpeggio", "分散和音", "跳躍を含む規則的な上昇・下降。",
         [950, 1300, 1600, 1300, 950, 1300], [34] * 6),
        ("chromatic", "半音進行", "小さな音程差を識別・再現できるかを見る。",
         [1050, 1150, 1250, 1350, 1450, 1550], [34] * 6),
        ("rests-and-phrase", "休符を含むフレーズ", "途中の休符とフレーズ再開を同時に評価する。",
         [1050, 1300, None, 1500, 1250, None, 1650], [38, 38, 24, 38, 38, 24, 38]),
    ]


def write_wav(path: pathlib.Path, samples: np.ndarray, sample_rate=16_000):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def evaluate_sample(model, deterministic_control, target_np, sample_index, device):
    rng = np.random.default_rng(88031 + sample_index)
    waveform, _ = synth_source(target_np, "recorder", rng, sr=16_000, shift=0)
    heard = room(waveform, 16_000, rng)
    frames = torch.from_numpy(frame_audio_numpy(heard, len(target_np))).to(device)[None]
    lengths = torch.tensor([frames.shape[1]], device=device)
    with torch.inference_mode():
        nn_features = model.ear.audio_features(frames)
        nn_profile = model.listen(frames, lengths)
        det_features = DeterministicEar().to(device)(frames)
    nn_semantic = nn_features[..., -2:]
    det_voice_logit = torch.where(det_features[..., 1] >= .5,
                                  torch.full_like(det_features[..., 1], 12.0),
                                  torch.full_like(det_features[..., 1], -12.0))
    det_semantic = torch.stack([det_features[..., 0], det_voice_logit], -1)
    semantic = {"nn": nn_semantic, "det": det_semantic}
    nn_ear_recalled = articulate(nn_semantic.repeat_interleave(model.tempo_scale, 1),
                                 model.articulation_frames)
    learned_memory_residual = nn_profile.target - nn_ear_recalled

    outputs = {"ear": {}, "memory": {}, "planner": {}, "performance": {}}
    for ear_kind, selected_ear in semantic.items():
        outputs["ear"][ear_kind] = track(1300 + 600 * selected_ear[0, :, 0].cpu(),
                                               selected_ear[0, :, 1].ge(0).cpu())
        for memory_kind in ("nn", "det"):
            upstream = articulate(selected_ear.repeat_interleave(model.tempo_scale, 1),
                                  model.articulation_frames)
            if memory_kind == "nn":
                target = upstream + learned_memory_residual
                target = torch.stack([target[..., 0].clamp(-1, 1),
                                      target[..., 1].clamp(-12, 12)], -1)
            else:
                target = upstream
            memory_key = f"{ear_kind}-{memory_kind}"
            wanted = 1300 + 600 * target[0, :, 0]
            target_voice = target[0, :, 1].ge(0)
            outputs["memory"][memory_key] = track(wanted.cpu(), target_voice.cpu())
            for planner_kind in ("nn", "det"):
                with torch.inference_mode():
                    position = (model.planner(target)[..., 0] if planner_kind == "nn"
                                else ((target[..., 0] + 1) * .5).clamp(0, 1))
                planner_key = f"{memory_key}-{planner_kind}"
                outputs["planner"][planner_key] = {
                    "position": compact(position[0].cpu()),
                    **track((700 + 1200 * position[0]).cpu(), target_voice.cpu()),
                }
                for control_kind in ("nn", "det"):
                    if control_kind == "nn":
                        profile = NeuralPerformanceProfile(
                            target, position, torch.tensor([120.], device=device),
                            torch.tensor([60.], device=device),
                            torch.tensor([target.shape[1]], device=device), target)
                        slow = None
                        with torch.inference_mode():
                            for _ in range(3):
                                result = model.perform(profile, slow_context=slow)
                                slow = result["slow_context"]
                        emitted = result["pitch_cents"][0]
                        emitted_voice = result["valve"][0] >= .5
                    else:
                        onset = torch.nn.functional.pad(
                            target_voice[None, 1:] & ~target_voice[None, :-1], (1, 0))
                        profile = DeterministicProfile(target[..., 0], target_voice[None], onset,
                                                       torch.tensor([60.], device=device))
                        with torch.inference_mode():
                            result = deterministic_control.perform(profile, position)
                        emitted = result["pitch_cents"][0]
                        emitted_voice = target_voice
                    valid = target_voice & emitted_voice
                    mae = (emitted[valid] - wanted[valid]).abs().mean().item() if valid.any() else None
                    route_key = f"{planner_key}-{control_kind}"
                    outputs["performance"][route_key] = {
                        **track(emitted.cpu(), emitted_voice.cpu()),
                        "metrics": {"final_pitch_mae_cents": mae,
                                    "voiced_fraction": emitted_voice.float().mean().item()},
                    }
    reference_voice = np.isfinite(target_np)
    return waveform, {
        "reference": track(np.nan_to_num(target_np, nan=0.0), reference_voice),
        "reference_steps": int(len(target_np)),
        "outputs": outputs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/yamabiko_connected_composite_v1.pt")
    ap.add_argument("--out", default="docs/e2e-composite-results/hybrid-lab.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    model = YamabikoComposite.from_checkpoint(
        torch.load(args.checkpoint, map_location=device, weights_only=False), device).eval()
    deterministic_control = DeterministicYamabikoPipeline(model.plant.config).to(device)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = out.parent / "hybrid-samples"
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, (sample_id, title, description, notes, durations) in enumerate(sample_specs(), 1):
        target_np = note_sequence(notes, durations)
        waveform, evaluated = evaluate_sample(model, deterministic_control, target_np, index, device)
        wav_name = f"{index:02d}-{sample_id}.wav"
        write_wav(audio_dir / wav_name, waveform)
        samples.append({"id": sample_id, "title": title, "description": description,
                        "reference_wav": f"e2e-composite-results/hybrid-samples/{wav_name}",
                        **evaluated})
        print(f"[{index:02d}/10] {title}: {len(target_np)} reference frames", flush=True)
    document = {
        "format": "yamabiko-hybrid-lab-v2", "frame_rate": 100,
        "reference_frame_rate": 100, "sample_count": len(samples),
        "route_count_per_sample": 16,
        "note": ("10サンプル×全16組合せを実モデルと決定論モジュールで事前評価。"
                 "NN MemoryへDet Earを渡す場合は、共通pitch/voice契約上で学習済みMemory residualを"
                 "加えるrepresentation adapterを使用。NN Controlは遅い適応状態を保持した3回目。"),
        "samples": samples,
    }
    out.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({len(samples)} samples x 16 routes, {out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
