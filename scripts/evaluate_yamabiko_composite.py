"""Evaluate the connected neural graph against the deterministic reference graph."""
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
from flute_rl.yamabiko.composite import YamabikoComposite  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import DeterministicYamabikoPipeline  # noqa: E402
from flute_rl.yamabiko.e2e_io import frame_audio_numpy  # noqa: E402


def write_pitch_plot(path, target, neural, neural_voice, deterministic, deterministic_voice, frame_rate=100):
    """Write a dependency-free target/connected/oracle pitch comparison."""
    width, height, left, top, right, bottom = 960, 380, 72, 28, 24, 52
    low, high = 850.0, 1750.0

    def xy(index, cents):
        x = left + index * (width - left - right) / max(1, len(target) - 1)
        y = top + (high - cents) * (height - top - bottom) / (high - low)
        return f"{x:.1f},{y:.1f}"

    def segments(values, valid):
        result, current = [], []
        for index, (value, keep) in enumerate(zip(values, valid)):
            if keep and np.isfinite(value):
                current.append(xy(index, float(value)))
            elif current:
                if len(current) > 1: result.append(" ".join(current))
                current = []
        if len(current) > 1: result.append(" ".join(current))
        return result

    series = ((target, np.isfinite(target), "#e5e7eb", 4),
              (neural, neural_voice, "#34d399", 2.5),
              (deterministic, deterministic_voice, "#f59e0b", 2.5))
    lines = []
    for values, valid, color, stroke in series:
        lines += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke}"/>'
                  for points in segments(values, valid)]
    ticks = []
    for cents in (900, 1100, 1300, 1500, 1700):
        y = xy(0, cents).split(",")[1]
        ticks.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#334155"/>')
        ticks.append(f'<text x="{left-10}" y="{float(y)+5:.1f}" text-anchor="end">{cents}</text>')
    duration = (len(target) - 1) / frame_rate
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#071521"/><g font-family="sans-serif" font-size="13" fill="#cbd5e1">
{''.join(ticks)}<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#94a3b8"/>
<text x="{width/2}" y="{height-12}" text-anchor="middle">time [s]  0 — {duration:.1f}</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">pitch [cent]</text>
<text x="{left}" y="18" fill="#e5e7eb">— target</text><text x="{left+110}" y="18" fill="#34d399">— connected NN</text><text x="{left+275}" y="18" fill="#f59e0b">— deterministic oracle</text>
{''.join(lines)}</g></svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/yamabiko_connected_composite_v1.pt")
    ap.add_argument("--out", default="docs/e2e-composite-results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(88031)
    target = np.r_[np.full(40, np.nan), np.full(55, 1000.), np.full(55, 1450.),
                   np.full(55, 1200.), np.full(55, 1650.)]
    waveform, _ = synth_source(target, "recorder", rng, sr=16_000, shift=0)
    frames = torch.from_numpy(frame_audio_numpy(room(waveform, 16_000, rng), len(target))).to(args.device)[None]
    expected = np.repeat(target, 2); expected_voice = np.isfinite(expected)
    for boundary in (80, 190, 300, 410): expected_voice[boundary:boundary + 4] = False
    expected_audio_cents = np.where(expected_voice, expected, 0.0)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = YamabikoComposite.from_checkpoint(checkpoint, args.device).eval()
    with torch.inference_mode():
        profile, performances = model(frames, torch.tensor([len(target)], device=args.device), repetitions=3)
        deterministic = DeterministicYamabikoPipeline().to(args.device)
        d_profile, _, d_result = deterministic(frames)
    neural = performances[-1]
    predicted_target = (1300 + 600 * profile.target[0, :, 0]).cpu().numpy()
    memory_voice = (torch.sigmoid(profile.target[0, :, 1]) >= .5).cpu().numpy()
    neural_cents = neural["pitch_cents"][0].cpu().numpy(); neural_voice = (neural["valve"][0] >= .5).cpu().numpy()
    deterministic_cents = d_result["pitch_cents"][0].cpu().numpy(); deterministic_voice = d_profile.voice[0].cpu().numpy()
    valid_memory = expected_voice & memory_voice
    valid_neural = expected_voice & neural_voice
    valid_deterministic = expected_voice & deterministic_voice
    metrics = {
        "format": "yamabiko-connected-composite-eval-v1",
        "reference_steps": len(target), "playback_steps": len(expected), "tempo_scale": 2,
        "timeline_pitch_mae_cents": float(np.mean(np.abs(predicted_target[valid_memory] - expected[valid_memory]))),
        "timeline_missing_voice_fraction": float(np.mean(expected_voice & ~memory_voice)),
        "neural_e2e_pitch_mae_cents": float(np.mean(np.abs(neural_cents[valid_neural] - expected[valid_neural]))),
        "neural_e2e_missing_voice_fraction": float(np.mean(expected_voice & ~neural_voice)),
        "deterministic_e2e_pitch_mae_cents": float(np.mean(np.abs(deterministic_cents[valid_deterministic] - expected[valid_deterministic]))),
        "deterministic_e2e_missing_voice_fraction": float(np.mean(expected_voice & ~deterministic_voice)),
        "pass": False, "real_rig_validated": False,
        "note": "Frozen cross-gate connection test; no joint fine-tuning has been performed.",
    }
    write_wav(out / "reference.wav", waveform, 16_000)
    write_wav(out / "target_slow.wav", synth_self(expected_audio_cents, expected_voice, np.random.default_rng(1), sr=16_000), 16_000)
    write_wav(out / "neural_composite.wav", synth_self(neural_cents, neural_voice, np.random.default_rng(2), sr=16_000), 16_000)
    write_wav(out / "deterministic_composite.wav", synth_self(deterministic_cents, deterministic_voice, np.random.default_rng(3), sr=16_000), 16_000)
    write_pitch_plot(out / "pitch.svg", expected, neural_cents, neural_voice,
                     deterministic_cents, deterministic_voice)
    manifest = {"summary": metrics, "audio": {"reference": "reference.wav", "target": "target_slow.wav",
                "neural": "neural_composite.wav", "deterministic": "deterministic_composite.wav"},
                "plot": "pitch.svg"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
