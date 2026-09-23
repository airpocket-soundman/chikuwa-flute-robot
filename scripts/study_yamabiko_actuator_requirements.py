"""How fast must the plunger actuator be for a given tempo?

Sweeps the actuator speed (torque scales with it) and the note length, plays
random melodies over the full 700-1900 cent range with the practical
encoder-less deterministic controller (motor calibrated) and with an encoder
oracle (upper bound), and reports musical scores with 95 % intervals:
the hit rate (notes in tune within 0.3 s), the in-tune fraction of sounding
time and the median time to reach a note.  Other rig values stay randomized.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_yamabiko_rig_adaptive import encoder_oracle, oracle_coefficients  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import EncoderlessDeterministicPerformer  # noqa: E402
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.musical_metrics import merge, summarize, unit_scores  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402

NOMINAL_MM_S = 150.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speeds", type=float, nargs="+", default=[60, 90, 120, 150, 200, 300, 450])
    ap.add_argument("--length-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--rigs", type=int, default=48)
    ap.add_argument("--songs", type=int, default=3)
    ap.add_argument("--out", default="docs/e2e-rig-adaptive-results/actuator_requirements.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    torch.set_grad_enabled(False)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic(motor_scale_range=None)); cfg = plant.config
    performer = EncoderlessDeterministicPerformer(plant)
    octave_mm = float(plant.position_for_cents(torch.tensor([cfg.high_cents]), plant.parameters(1, "cpu"))
                      - plant.position_for_cents(torch.tensor([cfg.low_cents]), plant.parameters(1, "cpu"))) * cfg.stroke_m * 1000
    rows = []
    for scale in args.length_scales:
        songs = [random_melodies(np.random.default_rng(3000 + k), args.rigs, int(600 * max(scale, .5)), device,
                                 varied=True, length_scale=scale) for k in range(args.songs)]
        for speed in args.speeds:
            factor = speed / NOMINAL_MM_S
            params = plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(11))
            params.max_velocity_strokes_s = torch.full_like(params.max_velocity_strokes_s, cfg.max_velocity_strokes_s * factor)
            params.torque_gain = torch.full_like(params.torque_gain, cfg.torque_gain * factor)
            ratio = performer.measure_motor(params, torch.Generator(device).manual_seed(5))
            oracle_memory = oracle_coefficients(plant, params)
            units = {"deterministic": [], "encoder_oracle": []}
            for k, (cents, voice) in enumerate(songs):
                played = performer.perform(cents, voice, params, None, torch.Generator(device).manual_seed(k),
                                           motor_ratio=ratio)[0]["pitch_cents"]
                units["deterministic"].append(unit_scores(played, cents, voice))
                units["encoder_oracle"].append(unit_scores(
                    encoder_oracle(plant, performer, cents, voice, params, oracle_memory), cents, voice))
            row = {"speed_mm_s": speed, "length_scale": scale,
                   "note_seconds": [.48 * scale, 1.31 * scale],
                   **{name: summarize(merge(u)) for name, u in units.items()}}
            rows.append(row)
            d = row["deterministic"]
            print(f"notes {row['note_seconds'][0]:.2f}-{row['note_seconds'][1]:.2f}s  {speed:5.0f} mm/s  "
                  f"hit {d['hit_rate']['mean']:.2f} [{d['hit_rate']['low']:.2f},{d['hit_rate']['high']:.2f}]  "
                  f"in-tune {d['in_tune']['mean']:.2f}  reach {d['reach_ms']['mean']:.0f} ms  |  "
                  f"oracle hit {row['encoder_oracle']['hit_rate']['mean']:.2f}", flush=True)
    result = {"octave_travel_mm": octave_mm, "nominal_speed_mm_s": NOMINAL_MM_S, "rigs": args.rigs,
              "songs": args.songs, "tolerance_cents": 25, "hit_within_ms": 300, "rows": rows,
              "note": "Replay tempo 0.5x of the reference is length_scale 1.0; 0.5 is the original tempo."}
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
