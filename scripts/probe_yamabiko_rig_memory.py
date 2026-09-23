"""What does the rig-adaptive NN's rig memory know about the rig?

After the learning-mode calibration piece, fit a linear read-out (ridge
regression) from the memory vector to each hidden rig value on one set of
randomized rigs and report R^2 on another.  Nothing here feeds back into
training; the rig values are only used as probe labels.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.rig_adaptive import RigAdaptivePerformer  # noqa: E402

FIELDS = ("tube_offset_m", "temp_offset_c", "flute_offset_cents", "torque_gain", "coulomb_friction",
          "max_velocity_strokes_s", "deadband", "hearing_delay_steps")


def memories(model, plant, params, device, learning_mode):
    generator = torch.Generator(device).manual_seed(0)
    if learning_mode:
        return model.calibrate(plant, params, generator)
    cents, voice = random_melodies(np.random.default_rng(1), params.torque_gain.shape[0], 600, device)
    return model.perform(plant, cents, voice, params, None, generator)[1]


def ridge_r2(x_train, y_train, x_test, y_test, alpha=1e-2):
    mean_x, std_x = x_train.mean(0), x_train.std(0) + 1e-6
    a = np.c_[(x_train - mean_x) / std_x, np.ones(len(x_train))]
    b = np.c_[(x_test - mean_x) / std_x, np.ones(len(x_test))]
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ y_train)
    residual = ((b @ w - y_test) ** 2).sum()
    return float(1 - residual / ((y_test - y_test.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", nargs="+",
                    default=["runs/yamabiko_rig_adaptive_v1.pt", "runs/yamabiko_rig_adaptive_v2.pt"])
    ap.add_argument("--rigs", type=int, default=512)
    ap.add_argument("--out", default="docs/e2e-rig-adaptive-results/memory_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    torch.set_grad_enabled(False)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    sets = [plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(seed))
            for seed in (101, 202)]
    result = {"rigs_per_split": args.rigs, "fields": list(FIELDS), "models": {}}
    for path in map(pathlib.Path, args.checkpoint):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = RigAdaptivePerformer.from_checkpoint(checkpoint, device).eval()
        learning_mode = bool(checkpoint.get("calibration_songs"))
        x = [memories(model, plant, p, device, learning_mode).cpu().numpy() for p in sets]
        scores = {}
        for field in FIELDS:
            y = [getattr(p, field).float().cpu().numpy() for p in sets]
            scores[field] = ridge_r2(x[0], y[0], x[1], y[1])
        result["models"][path.stem] = {"learning_mode": learning_mode, "r2": scores}
        print(path.stem, {k: round(v, 2) for k, v in scores.items()})
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
