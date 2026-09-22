"""Render fixed held-out before/after WAVs for Tempo/Beat and Musical Memory."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from export_targets import write_wav  # noqa: E402
from flute_rl.audio import room, synth_self, synth_source  # noqa: E402
from flute_rl.targets import make_beat_target  # noqa: E402
from flute_rl.yamabiko.beat_grid import (BeatAlignedMusicalMemoryNet, BeatGridConfig,
                                         MusicalMemoryNet, TempoBeatNet)  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_io import HOP, SAMPLE_RATE, frame_audio_numpy  # noqa: E402


def click_track(phase_xy, confidence, steps):
    phase = np.mod(np.arctan2(phase_xy[:, 0], phase_xy[:, 1]) / (2 * np.pi), 1.0)
    wraps = np.flatnonzero((phase[1:] < phase[:-1] - .5) & (confidence[1:] >= .5)) + 1
    y = np.zeros(steps * HOP, np.float32); length = int(.025 * SAMPLE_RATE)
    envelope = np.exp(-np.arange(length) / (SAMPLE_RATE * .006))
    tone = .7 * envelope * np.sin(2 * np.pi * 1300 * np.arange(length) / SAMPLE_RATE)
    for frame in wraps:
        at = frame * HOP; stop = min(len(y), at + length); y[at:stop] += tone[:stop - at]
    return y, wraps


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/yamabiko_tempo_beat.pt")
    ap.add_argument("--out", default="docs/e2e-beat-results"); ap.add_argument("--seed", type=int, default=15551)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); ck = torch.load(args.model, map_location=args.device)
    cfg = BeatGridConfig(**ck["config"]); tempo = TempoBeatNet(cfg).to(args.device); tempo.load_state_dict(ck["tempo_beat"]); tempo.eval()
    kind = ck.get("musical_memory_kind")
    memory = (BeatAlignedMusicalMemoryNet(cfg, pitch_skip=kind in ("beat-aligned-v2", "beat-aligned-v3", "beat-aligned-v4"),
                                          sharp_alignment=kind in ("beat-aligned-v3", "beat-aligned-v4"),
                                          stable_clock=kind == "beat-aligned-v4")
              if kind in ("beat-aligned-v1", "beat-aligned-v2", "beat-aligned-v3", "beat-aligned-v4")
              else MusicalMemoryNet(cfg)).to(args.device)
    memory.load_state_dict(ck["musical_memory"]); memory.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    out = pathlib.Path(args.out); audio = out / "audio"; audio.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed); cases = []; tempo_errors = []; phase_errors = []
    for index, bpm in enumerate((73.0, 97.0, 137.0, 173.0)):
        example = make_beat_target(rng, bpm=bpm); wave, _ = synth_source(example.target, "recorder", rng, sr=SAMPLE_RATE)
        wave = room(wave, SAMPLE_RATE, rng); frames = frame_audio_numpy(wave, len(example.target))
        features = ear.audio_features(torch.from_numpy(frames).to(args.device))[None]
        lengths = torch.tensor([len(example.target)], device=args.device)
        bpm_log, beat_out, encoded, mask = tempo(features, lengths)
        predicted_bpm = float(tempo.phase_bpm(beat_out, mask)[0])
        phase_xy = beat_out[0, :, :2].cpu().numpy(); confidence = torch.sigmoid(beat_out[0, :, 2]).cpu().numpy()
        clicks, wraps = click_track(phase_xy, confidence, len(example.target))
        name = f"bpm_{int(bpm)}"; before = audio / f"{name}_tempo_before_reference.wav"; after = audio / f"{name}_tempo_after_clicks.wav"
        write_wav(before, wave, SAMPLE_RATE); write_wav(after, clicks, SAMPLE_RATE)
        pred_phase = np.mod(np.arctan2(phase_xy[:, 0], phase_xy[:, 1]) / (2 * np.pi), 1.0)
        valid = example.phase_valid; d = np.remainder(pred_phase[valid] - example.frame_phase[valid] + .5, 1) - .5
        tempo_errors.append(abs(predicted_bpm - bpm) / bpm); phase_errors.append(float(np.mean(abs(d))))
        item = {"case": name, "true_bpm": bpm, "predicted_bpm": predicted_bpm,
                "tempo_relative_error": tempo_errors[-1], "phase_circular_mae_cycle": phase_errors[-1],
                "detected_beats": int(len(wraps)), "tempo_before": str(before.relative_to(out)).replace("\\", "/"),
                "tempo_after": str(after.relative_to(out)).replace("\\", "/")}
        if ck.get("musical_memory_trained"):
            confident = np.flatnonzero(confidence >= .5); start_frame = int(confident[0]) if len(confident) else 0
            cells = int(np.clip(round((len(example.target) - start_frame) / 100 * predicted_bpm / 60 * cfg.subdivision), 1, 64))
            decoded, _ = memory(features, encoded, beat_out, mask, cells)
            pitch = decoded[0, :, 0].cpu().numpy() * 600 + 1300; voice = decoded[0, :, 1].cpu().numpy() >= 0
            seconds_cell = 60.0 / predicted_bpm / cfg.subdivision
            frames_cell = max(1, int(round(seconds_cell * 100)))
            track_pitch = np.r_[np.zeros(start_frame), np.repeat(pitch, frames_cell)]
            track_voice = np.r_[np.zeros(start_frame, bool), np.repeat(voice, frames_cell)]
            rendered = synth_self(track_pitch, track_voice, np.random.default_rng(16000 + index), sr=SAMPLE_RATE)
            mem_after = audio / f"{name}_memory_after_recall.wav"; write_wav(mem_after, rendered, SAMPLE_RATE)
            item["memory_before"] = item["tempo_before"]
            item["memory_after"] = str(mem_after.relative_to(out)).replace("\\", "/")
        cases.append(item)
    demo_metrics = {"tempo_relative_error_median": float(np.median(tempo_errors)),
               "tempo_relative_error_p90": float(np.quantile(tempo_errors, .9)),
               "phase_circular_mae_cycle": float(np.mean(phase_errors))}
    demo_metrics["pass"] = demo_metrics["tempo_relative_error_median"] <= .03 and demo_metrics["phase_circular_mae_cycle"] <= .08
    # The four audible cases are a stable demo set, not the official Gate.
    # Never let repeated tuning against public WAVs replace the frozen test.
    metrics = ck.get("tempo_metrics", demo_metrics)
    manifest = {"format": "yamabiko-beat-audio-v1", "checkpoint": args.model, "seed": args.seed,
                "official_path": "predicted_upstream", "audio_kind": "tempo_after is diagnostic click track, not physical performance",
                "tempo_beat": metrics, "demo_metrics": demo_metrics,
                "musical_memory_trained": bool(ck.get("musical_memory_trained")),
                "musical_memory": ck.get("musical_memory_metrics"), "cases": cases}
    out.mkdir(parents=True, exist_ok=True); (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"wrote {out / 'manifest.json'}")


if __name__ == "__main__": main()
