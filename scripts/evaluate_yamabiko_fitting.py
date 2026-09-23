"""Fixed benchmark of the fit -> twin -> rig-specific control plan.

32 unknown rigs (black boxes over the realistic range, fixed seed) each run
the calibration, get a fitted twin, and then play the ten reference phrases
(true score at 0.5x) and random songs.  Controllers:

* ``det_nominal``: deterministic, motor speed calibrated only (the old way);
* ``det_twin``: deterministic using the fitted twin;
* ``det_true``: deterministic given the rig's true values (a perfect fit);
* ``nn``: the pre-trained neural performer, same song played three times;
* ``sensor``: position-sensor reference.

Musical scores with 95 % intervals go to ``manifest.json``; tracks of two
representative rigs go to ``lab.json`` for the interactive page.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_yamabiko_hybrid_lab import note_sequence, sample_specs  # noqa: E402
from evaluate_yamabiko_integrated import pad, score_track  # noqa: E402
from evaluate_yamabiko_rig_adaptive import encoder_oracle, oracle_coefficients  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import EncoderlessDeterministicPerformer  # noqa: E402
from flute_rl.yamabiko.device import BlackBoxDevice  # noqa: E402
from flute_rl.yamabiko.fitting import TwinFitter, run_calibration  # noqa: E402
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.musical_metrics import merge, summarize, unit_scores  # noqa: E402
from flute_rl.yamabiko.performers import load_performer  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402

TAIL = 60


def phrases(device, rigs):
    result = []
    for sample_id, title, description, notes, durations in sample_specs():
        truth, truth_voice = score_track(note_sequence(notes, durations))
        truth, truth_voice = pad(truth, truth_voice, len(truth) + TAIL)
        cents = torch.tensor(truth, device=device, dtype=torch.float32)[None].expand(rigs, -1).contiguous()
        voice = torch.tensor(truth_voice, device=device)[None].expand(rigs, -1).contiguous()
        result.append({"id": sample_id, "title": title, "description": description, "cents": cents, "voice": voice})
    return result


def rounded(x):
    return np.round(x.detach().cpu().numpy().astype(float), 1).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rigs", type=int, default=32)
    ap.add_argument("--random-songs", type=int, default=3)
    ap.add_argument("--fit-steps", type=int, default=200)
    ap.add_argument("--nn", default="runs/yamabiko_adaptive_memory_v1.pt")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="docs/plan-results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device; out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    true = plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(args.seed))
    g = lambda k: torch.Generator(device).manual_seed(args.seed + k)

    # F1 + F2: calibration run and twin fit (commands and heard pitch only).
    rig = BlackBoxDevice(plant, true, g(1))
    calibration = run_calibration(rig)
    fit = TwinFitter(plant).fit(calibration, steps=args.fit_steps)
    twin = fit.parameters
    fit_seconds = time.time() - started
    print(f"twin fitted in {fit_seconds:.0f}s, calibration heard MAE median {fit.heard_mae.median():.1f}", flush=True)

    det = EncoderlessDeterministicPerformer(plant)
    nn, _ = load_performer(args.nn, device)
    ratio = det.measure_motor(true, g(2))
    oracle_memory = oracle_coefficients(plant, true)

    # G1 on a held-out song: the twin predicts what the rig will be heard to play.
    held = random_melodies(np.random.default_rng(args.seed), args.rigs, 500, device)
    with torch.no_grad():
        played, _ = det.perform(*held, true, None, g(3), motor_ratio=ratio)
        fitter = TwinFitter(plant)
        per_rig = []
        for row in range(args.rigs):
            one = type(twin)(*(v[row:row + 1] if torch.is_tensor(v) else v for v in vars(twin).values()))
            predicted = fitter.predict(one, played["pwm"][row:row + 1], int(fit.delay[row]))[0]
            heard, valid = played["heard"][row], held[1][row]
            valid = valid & torch.isfinite(heard)
            per_rig.append(float((predicted - heard).abs()[valid].mean()))
    recovery = {
        "speed_ratio_error": float(((twin.max_velocity_strokes_s / true.max_velocity_strokes_s) - 1).abs().median()),
        "torque_ratio_error": float(((twin.torque_gain / true.torque_gain) - 1).abs().median()),
        "deadband_error": float((twin.deadband - true.deadband).abs().median()),
        "tube_offset_error_mm": float((twin.tube_offset_m - true.tube_offset_m).abs().median() * 1000),
        "delay_correct": float((fit.delay == true.hearing_delay_steps).float().mean()),
    }

    controllers = ("det_nominal", "det_twin", "det_true", "nn_play1", "nn_play2", "nn_play3", "sensor")

    def play_all(cents, voice, k):
        tracks = {}
        with torch.no_grad():
            tracks["det_nominal"] = det.perform(cents, voice, true, None, g(10 + k), motor_ratio=ratio)[0]["pitch_cents"]
            tracks["det_twin"] = det.perform(cents, voice, true, None, g(10 + k), twin=twin)[0]["pitch_cents"]
            tracks["det_true"] = det.perform(cents, voice, true, None, g(10 + k), twin=true)[0]["pitch_cents"]
            memory = None
            for play in (1, 2, 3):
                result, memory = nn.perform(plant, cents, voice, true, memory, g(100 * play + k))
                tracks[f"nn_play{play}"] = result["pitch_cents"]
            tracks["sensor"] = encoder_oracle(plant, det, cents, voice, true, oracle_memory)
        return tracks

    units = {"phrases": {c: [] for c in controllers}, "random": {c: [] for c in controllers}}
    per_phrase, lab = [], []
    lab_rows = {"typical": int((true.max_velocity_strokes_s - plant.config.max_velocity_strokes_s).abs().argmin()),
                "slow": int(true.max_velocity_strokes_s.argmin())}
    for k, phrase in enumerate(phrases(device, args.rigs)):
        tracks = play_all(phrase["cents"], phrase["voice"], k)
        scores = {}
        for name, track in tracks.items():
            unit = unit_scores(track, phrase["cents"], phrase["voice"])
            units["phrases"][name].append(unit); scores[name] = summarize(unit)
        per_phrase.append({"id": phrase["id"], "title": phrase["title"], "scores": scores})
        lab.append({"id": phrase["id"], "title": phrase["title"], "description": phrase["description"],
                    "target": rounded(phrase["cents"][0]), "voice": phrase["voice"][0].int().tolist(),
                    "rigs": {label: {name: rounded(track[row]) for name, track in tracks.items()}
                             for label, row in lab_rows.items()}})
        print(f"{phrase['title']}: " + "  ".join(f"{n} {s['hit_rate']['mean'] * 100:.0f}%" for n, s in scores.items()),
              flush=True)
    for k in range(args.random_songs):
        cents, voice = random_melodies(np.random.default_rng(args.seed + 50 + k), args.rigs, 600, device, varied=True)
        for name, track in play_all(cents, voice, 50 + k).items():
            units["random"][name].append(unit_scores(track, cents, voice))

    summary = {
        "format": "yamabiko-plan-benchmark-v1", "rigs": args.rigs, "seed": args.seed,
        "simulator": "PhysicalPlantConfig.realistic(): closed tube with 200 cent home margin, deadband, "
                     "hearing delay/noise/drop-outs, motor speed and torque x0.4-2.5",
        "fit": {"seconds": fit_seconds, "calibration_steps": int(calibration.pwm.shape[1]),
                "calibration_heard_mae_median": float(fit.heard_mae.median()),
                "held_out_heard_mae_median": float(np.median(per_rig)), "recovery": recovery},
        "phrases": {name: summarize(merge(u)) for name, u in units["phrases"].items()},
        "random": {name: summarize(merge(u)) for name, u in units["random"].items()},
        "per_phrase": per_phrase,
        "lab_rigs": {label: {"row": row, "motor_factor": float(true.max_velocity_strokes_s[row] /
                                                                plant.config.max_velocity_strokes_s)}
                     for label, row in lab_rows.items()},
        "real_rig_validated": False,
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "lab.json").write_text(json.dumps({"samples": lab, "rigs": summary["lab_rigs"]}, ensure_ascii=False,
                                             separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("fit",)}, indent=2))
    print(json.dumps({k: {n: round(v["hit_rate"]["mean"] * 100, 1) for n, v in summary[k].items()}
                      for k in ("phrases", "random")}, indent=2))


if __name__ == "__main__":
    main()
