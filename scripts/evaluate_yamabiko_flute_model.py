"""Audit the flute simulator and the connected graph's role split.

Compares the historical linear flute with the closed-tube flute
(f = c / 4(L - x)), and re-runs the frozen connected composite on both,
on nominal and randomized rigs.  Also checks whether the slow feedback
context actually helps across repeated plays, and what the feedback residual
contributes.  Writes a manifest (the report's source of truth) and a plot.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.audio import room, synth_source  # noqa: E402
from flute_rl.yamabiko.composite import YamabikoComposite  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import DeterministicYamabikoPipeline  # noqa: E402
from flute_rl.yamabiko.e2e_io import frame_audio_numpy  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402


def linearity(cfg: PhysicalPlantConfig) -> dict:
    """Shape of the closed-tube pitch curve against a straight line."""
    plant = DifferentiableMotorFlute(dataclasses.replace(cfg, flute_model="closed_tube"))
    params = plant.parameters(1, "cpu", torch.float64)
    full = torch.linspace(0, 1, 1001, dtype=torch.float64)[:, None]
    full_cents = plant.cents_at(full, params)[:, 0].numpy()
    octave_end = float(plant.position_for_cents(torch.tensor([cfg.high_cents], dtype=torch.float64), params))
    used = torch.linspace(0, octave_end, 1001, dtype=torch.float64)[:, None]
    used_cents = plant.cents_at(used, params)[:, 0].numpy()

    def minimax_line_error(x, y):
        chord = np.polyval(np.polyfit(x[[0, -1]], y[[0, -1]], 1), x)
        deviation = y - chord
        return float((deviation.max() - deviation.min()) / 2)

    mm = cfg.stroke_m * 1000
    used_slope = np.diff(used_cents) / (np.diff(used[:, 0].numpy()) * mm)
    full_slope = np.diff(full_cents) / (np.diff(full[:, 0].numpy()) * mm)
    period = plant.period_s(used, params)[:, 0].numpy()
    period_residual = period - np.polyval(np.polyfit(used[:, 0].numpy(), period, 1), used[:, 0].numpy())
    return {
        "octave_stroke_fraction": octave_end,
        "octave_travel_mm": octave_end * mm,
        "octave_slope_cents_per_mm": [float(used_slope[0]), float(used_slope[-1])],
        "octave_best_line_error_cents": minimax_line_error(used[:, 0].numpy(), used_cents),
        "full_stroke_cents": [float(full_cents[0]), float(full_cents[-1])],
        "full_stroke_slope_cents_per_mm": [float(full_slope[0]), float(full_slope[-1])],
        "full_stroke_best_line_error_cents": minimax_line_error(full[:, 0].numpy(), full_cents),
        "period_line_residual_s": float(np.abs(period_residual).max()),
        "linear_model_slope_cents_per_mm": cfg.pitch_span_cents / mm,
    }


def rig_spread(cfg: PhysicalPlantConfig, count=512, seed=7) -> dict:
    """Pitch error left when a planner ignores the rig (nominal inverse)."""
    plant = DifferentiableMotorFlute(dataclasses.replace(cfg, flute_model="closed_tube"))
    params = plant.parameters(count, "cpu", torch.float64, spread=1.0,
                              generator=torch.Generator().manual_seed(seed))
    nominal = plant.parameters(count, "cpu", torch.float64)
    cents = torch.linspace(cfg.low_cents, cfg.high_cents, 13, dtype=torch.float64)[:, None].expand(-1, count)
    played = plant.cents_at(plant.position_for_cents(cents, nominal), params)
    error = (played - cents).abs()
    return {"rigs": count, "nominal_inverse_mae_cents": float(error.mean()),
            "nominal_inverse_p95_cents": float(torch.quantile(error.flatten(), .95)),
            "low_note_mae_cents": float(error[0].mean()), "high_note_mae_cents": float(error[-1].mean())}


def curve_plot(path: pathlib.Path, cfg: PhysicalPlantConfig):
    width, height, left, top, right, bottom = 720, 360, 64, 24, 20, 48
    low, high = 500.0, 2500.0
    x = torch.linspace(0, 1, 201, dtype=torch.float64)[:, None]

    def xy(position, cents):
        px = left + position * (width - left - right)
        py = top + (high - cents) * (height - top - bottom) / (high - low)
        return f"{px:.1f},{py:.1f}"

    def line(cents, color, stroke, dash=""):
        keep = [(float(p), float(c)) for p, c in zip(x[:, 0], cents) if low <= c <= high]
        points = " ".join(xy(p, c) for p, c in keep)
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke}"{dash}/>'

    linear = DifferentiableMotorFlute(cfg)
    tube = DifferentiableMotorFlute(dataclasses.replace(cfg, flute_model="closed_tube"))
    lines = [line(linear.cents_at(x, linear.parameters(1, "cpu", torch.float64))[:, 0].numpy(), "#94a3b8", 3)]
    rigs = tube.parameters(12, "cpu", torch.float64, spread=1.0, generator=torch.Generator().manual_seed(3))
    rig_cents = tube.cents_at(x, rigs).numpy()
    lines += [line(rig_cents[:, i], "#f59e0b", 1, ' stroke-opacity=".45"') for i in range(rig_cents.shape[1])]
    lines.append(line(tube.cents_at(x, tube.parameters(1, "cpu", torch.float64))[:, 0].numpy(), "#34d399", 3))
    ticks = []
    for cents in (700, 1100, 1500, 1900, 2300):
        y = xy(0, cents).split(",")[1]
        ticks.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#334155"/>'
                     f'<text x="{left-8}" y="{float(y)+4:.1f}" text-anchor="end">{cents}</text>')
    for position in (0, .25, .5, .75, 1):
        px = xy(position, low).split(",")[0]
        ticks.append(f'<text x="{px}" y="{height-bottom+18}" text-anchor="middle">{position * cfg.stroke_m * 1000:.0f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#071521"/><g font-family="sans-serif" font-size="12" fill="#cbd5e1">
{''.join(ticks)}{''.join(lines)}
<text x="{width/2}" y="{height-8}" text-anchor="middle">plunger displacement [mm]</text>
<text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle">pitch [cent]</text>
<text x="{left+8}" y="{top+14}" fill="#94a3b8">— linear (current training)</text>
<text x="{left+8}" y="{top+30}" fill="#34d399">— closed tube, nominal</text>
<text x="{left+8}" y="{top+46}" fill="#f59e0b">— closed tube, 12 randomized rigs</text></g></svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/yamabiko_connected_composite_v1.pt")
    ap.add_argument("--out", default="docs/e2e-flute-model-results")
    ap.add_argument("--rigs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args(); out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    # Same four-note reference as evaluate_yamabiko_composite.py.
    rng = np.random.default_rng(88031)
    target = np.r_[np.full(40, np.nan), np.full(55, 1000.), np.full(55, 1450.),
                   np.full(55, 1200.), np.full(55, 1650.)]
    waveform, _ = synth_source(target, "recorder", rng, sr=16_000, shift=0)
    frames = torch.from_numpy(frame_audio_numpy(room(waveform, 16_000, rng), len(target)))[None]
    expected = np.repeat(target, 2); expected_voice = np.isfinite(expected)
    for boundary in (80, 190, 300, 410): expected_voice[boundary:boundary + 4] = False
    n = args.rigs

    def mae(cents, voice):
        valid = voice & expected_voice[None]
        return float(np.mean([np.abs(cents[i][valid[i]] - expected[valid[i]]).mean() for i in range(len(cents))]))

    model = YamabikoComposite.from_checkpoint(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)).eval()
    base_config = model.plant.config
    profile = model.listen(frames, torch.tensor([len(target)]))
    batch = type(profile)(profile.target.expand(n, -1, -1), profile.position.expand(n, -1),
                          profile.bpm.expand(n), profile.playback_bpm.expand(n),
                          profile.lengths.expand(n), profile.stored_timeline)

    def neural(result):
        return mae(result["pitch_cents"].numpy(), result["valve"].numpy() >= .5)

    matrix = []
    for flute in ("linear", "closed_tube"):
        cfg = dataclasses.replace(base_config, flute_model=flute)
        model.plant = DifferentiableMotorFlute(cfg)
        deterministic = DeterministicYamabikoPipeline(cfg)
        d_profile, d_plan = deterministic.listen(frames)
        d_profile = type(d_profile)(d_profile.pitch.expand(n, -1), d_profile.voice.expand(n, -1),
                                    d_profile.onset.expand(n, -1), d_profile.bpm.expand(n))
        for rigs, spread in (("nominal", 0.0), ("randomized", 1.0)):
            params = model.plant.parameters(n, "cpu", spread=spread,
                                            generator=torch.Generator().manual_seed(args.seed))
            d_result = deterministic.perform(d_profile, d_plan.expand(n, -1), parameters=params)
            matrix.append({"flute": flute, "rigs": rigs,
                           "neural_mae_cents": neural(model.perform(batch, parameters=params)),
                           "deterministic_oracle_mae_cents": mae(d_result["pitch_cents"].numpy(),
                                                                 d_profile.voice.numpy())})

    model.plant = DifferentiableMotorFlute(base_config)
    params = model.plant.parameters(n, "cpu", spread=1.0, generator=torch.Generator().manual_seed(args.seed))
    practice = {}
    for label, carry in (("carry_context", True), ("reset_context", False)):
        context, rows = None, []
        for _ in range(4):
            result = model.perform(batch, parameters=params, slow_context=context if carry else None)
            context = result["slow_context"]; rows.append(neural(result))
        practice[label] = rows
    feedback_step = model.feedback.step

    def no_residual(*a, **k):
        residual, error, state = feedback_step(*a, **k)
        return torch.zeros_like(residual), error, state

    model.feedback.step = no_residual
    without_feedback = neural(model.perform(batch, parameters=params))
    model.feedback.step = feedback_step

    first_note = float((1000.0 - base_config.low_cents) / base_config.pitch_span_cents)
    summary = {
        "format": "yamabiko-flute-model-audit-v1",
        "checkpoint": args.checkpoint, "rigs": n, "seed": args.seed,
        "reference": "four-note step phrase of evaluate_yamabiko_composite.py, 0.5x replay",
        "linearity": linearity(base_config),
        "rig_spread": rig_spread(base_config),
        "matrix": matrix,
        "practice_linear_randomized": practice,
        "feedback_linear_randomized": {"with_residual_mae_cents": practice["reset_context"][0],
                                       "without_residual_mae_cents": without_feedback},
        "rest_plan": {"neural_mean_position": float(profile.position[0, :80].mean()),
                      "first_note_position": first_note},
        "deterministic_is_oracle": True,
        "real_rig_validated": False,
        "note": "Existing checkpoints were trained on the linear flute; closed-tube rows measure transfer, not retrained models.",
    }
    curve_plot(out / "flute_curve.svg", base_config)
    manifest = {"summary": summary, "plot": "flute_curve.svg"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
