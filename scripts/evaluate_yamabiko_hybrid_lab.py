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
from flute_rl.yamabiko.deterministic_pipeline import EncoderlessDeterministicPerformer  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.rig_adaptive import RigAdaptivePerformer  # noqa: E402

# Closed-tube controls take the remembered target directly and plan inside.
CLOSED_TUBE_CONTROLS = ("ra", "enc")
CLOSED_TUBE_RIGS = 16


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


def add_closed_tube_controls(document, checkpoint_path, device):
    """Add the rig-adaptive NN ("ra") and encoder-less deterministic ("enc") controls.

    Both play every remembered target on the realistic closed-tube simulator
    (deadband, hearing delay/noise/drop-outs) for the same randomized rigs.
    The stored track is rig 0; the metrics average all rigs.  They plan
    internally, so the entry is the same for either Planner choice.
    """
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    params = plant.parameters(CLOSED_TUBE_RIGS, device, spread=1.0,
                              generator=torch.Generator(device).manual_seed(2718))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    neural = RigAdaptivePerformer.from_checkpoint(checkpoint, device).eval()
    learning_mode = bool(checkpoint.get("calibration_songs"))
    rig_memory = (neural.calibrate(plant, params, torch.Generator(device).manual_seed(999))
                  if learning_mode else None)
    deterministic = EncoderlessDeterministicPerformer(plant)
    for sample_index, sample in enumerate(document["samples"]):
        performance = sample["outputs"]["performance"]
        for memory_key, memory in sample["outputs"]["memory"].items():
            wanted = torch.tensor(memory["pitch_cents"], device=device, dtype=torch.float32)
            voice = torch.tensor(memory["voice"], device=device).bool()
            cents = wanted[None].expand(CLOSED_TUBE_RIGS, -1).contiguous()
            voices = voice[None].expand(CLOSED_TUBE_RIGS, -1).contiguous()
            for control in CLOSED_TUBE_CONTROLS:
                generator = torch.Generator(device).manual_seed(sample_index)
                with torch.inference_mode():
                    if control == "ra":
                        result, _ = neural.perform(plant, cents, voices, params, rig_memory, generator,
                                                   write_memory=not learning_mode)
                    else:
                        result, _ = deterministic.perform(cents, voices, params, None, generator)
                played = result["pitch_cents"]
                mae = (played - wanted)[:, voice].abs().mean().item() if voice.any() else None
                entry = {**track(played[0].cpu(), voice.cpu()),
                         "metrics": {"final_pitch_mae_cents": mae,
                                     "voiced_fraction": voice.float().mean().item(),
                                     "rigs": CLOSED_TUBE_RIGS}}
                for planner in ("nn", "det"):
                    performance[f"{memory_key}-{planner}-{control}"] = entry
    document["route_count_per_sample"] = 16 + 4 * len(CLOSED_TUBE_CONTROLS)
    document["closed_tube_controls"] = {
        "checkpoint": str(checkpoint_path), "rigs": CLOSED_TUBE_RIGS, "learning_mode": learning_mode,
        "simulator": "PhysicalPlantConfig.realistic()",
        "note": ("閉管笛・不感帯・聴こえの遅れ/ノイズ/欠落を含む現実的simulatorで、乱数化機体16台を演奏。"
                 "音と線は1台目、MAEは16台平均。計画は制御の内部で行うため、3計画の選択は影響しない。"),
    }
    return document


def summarize_planner(samples):
    """Attribute the selected NN Planner's error on the ten public samples."""
    inherited_all, added_all, total_all, position_added_all = [], [], [], []
    for sample in samples:
        expected = np.repeat(np.asarray(sample["reference"]["pitch_cents"], float), 2)
        expected_voice = np.repeat(np.asarray(sample["reference"]["voice"], bool), 2)
        memory = sample["outputs"]["memory"]["nn-nn"]
        planner = sample["outputs"]["planner"]["nn-nn-nn"]
        memory_pitch = np.asarray(memory["pitch_cents"], float)
        planner_pitch = np.asarray(planner["pitch_cents"], float)
        valid = expected_voice & np.asarray(memory["voice"], bool)
        inherited = memory_pitch[valid] - expected[valid]
        added = planner_pitch[valid] - memory_pitch[valid]
        inherited_all.append(inherited); added_all.append(added); total_all.append(planner_pitch[valid] - expected[valid])
        ideal_position = np.clip((memory_pitch[valid] - 700.0) / 1200.0, 0, 1)
        # Derive from the emitted-cent track rather than the compact display
        # position (rounded for JSON size), so attribution keeps sub-cent precision.
        planned_position = np.clip((planner_pitch[valid] - 700.0) / 1200.0, 0, 1)
        position_added_all.append(planned_position - ideal_position)
    inherited = np.concatenate(inherited_all); added = np.concatenate(added_all)
    total = np.concatenate(total_all); position_added = np.concatenate(position_added_all)
    result = {
        "position_mae_percent_stroke": float(np.mean(np.abs(position_added)) * 100),
        "position_p95_percent_stroke": float(np.quantile(np.abs(position_added), .95) * 100),
        "steady_pitch_mae_cents": float(np.mean(np.abs(added))),
        "inherited_pitch_mae_cents": float(np.mean(np.abs(inherited))),
        "model_added_position_mae_percent_stroke": float(np.mean(np.abs(position_added)) * 100),
        "model_added_pitch_mae_cents": float(np.mean(np.abs(added))),
        "model_added_pitch_p95_cents": float(np.quantile(np.abs(added), .95)),
        "output_total_pitch_mae_cents": float(np.mean(np.abs(total))),
        "output_total_pitch_p90_cents": float(np.quantile(np.abs(total), .90)),
        "inherited_pitch_bias_cents": float(np.mean(inherited)),
        "model_added_pitch_bias_cents": float(np.mean(added)),
        "output_total_pitch_bias_cents": float(np.mean(total)),
        "net_absolute_change_cents": float(np.mean(np.abs(total)) - np.mean(np.abs(inherited))),
        "attribution_residual_max_cents": float(np.max(np.abs(inherited + added - total))),
        "oracle_input_pass": bool(np.mean(np.abs(added)) <= 5 and np.quantile(np.abs(added), .95) <= 10),
        "real_input_transform_pass": bool(np.mean(np.abs(added)) <= 5 and np.quantile(np.abs(added), .95) <= 10),
        "connected_output_pass": bool(np.mean(np.abs(total)) <= 80 and
                                      np.mean(np.abs(total)) <= np.mean(np.abs(inherited)) + 20),
        "split": "ten-public-hybrid-samples-v1",
        "simulator_contract": "linear flute 700..1900 cent",
    }
    result["pass"] = result["oracle_input_pass"] and result["real_input_transform_pass"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/yamabiko_connected_composite_v2.pt")
    ap.add_argument("--out", default="docs/e2e-composite-results/hybrid-lab.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--closed-tube-downstream", default="runs/yamabiko_rig_adaptive_v2.pt",
                    help="rig-adaptive NN checkpoint for the closed-tube controls ('' to skip)")
    ap.add_argument("--extend-existing", action="store_true",
                    help="Only (re)compute the closed-tube controls on the existing JSON")
    ap.add_argument("--summarize-existing", action="store_true",
                    help="Recompute aggregate Planner attribution without rerunning models")
    args = ap.parse_args()
    device = args.device
    out = pathlib.Path(args.out)
    if args.summarize_existing:
        document = json.loads(out.read_text(encoding="utf-8"))
        document["planner_summary"] = summarize_planner(document["samples"])
        out.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(json.dumps(document["planner_summary"], indent=2))
        return
    if args.extend_existing:
        document = json.loads(out.read_text(encoding="utf-8"))
        document = add_closed_tube_controls(document, args.closed_tube_downstream, device)
        out.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"extended {out} with closed-tube controls ({out.stat().st_size} bytes)")
        return
    model = YamabikoComposite.from_checkpoint(
        torch.load(args.checkpoint, map_location=device, weights_only=False), device).eval()
    deterministic_control = DeterministicYamabikoPipeline(model.plant.config).to(device)
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
        "planner_summary": summarize_planner(samples), "samples": samples,
    }
    if args.closed_tube_downstream:
        document = add_closed_tube_controls(document, args.closed_tube_downstream, device)
    out.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({len(samples)} samples x 16 routes, {out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
