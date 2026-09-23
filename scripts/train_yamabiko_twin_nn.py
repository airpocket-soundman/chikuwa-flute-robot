"""F3: finish the neural performer for each rig inside its digital twin.

For every benchmark rig: run the calibration on the black box, fit the twin
(same as evaluate_yamabiko_fitting), then fine-tune a copy of the
pre-trained performer on that twin only (twin values +-10 %, random songs
played twice so the song memory keeps working, pulled toward the
pre-trained weights).  The fine-tuned model then plays the black box, three
plays per song.  Only the twin (built from commands and heard pitch) is used
for learning; the rig's true values only score.  Results are merged into
docs/plan-results/manifest.json and lab.json as ``nn_twin_play1..3``.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_yamabiko_fitting import phrases, rounded  # noqa: E402
from flute_rl.yamabiko.device import BlackBoxDevice  # noqa: E402
from flute_rl.yamabiko.fitting import TwinFitter, run_calibration  # noqa: E402
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.musical_metrics import merge, summarize, unit_scores  # noqa: E402
from flute_rl.yamabiko.performers import load_performer  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402

JITTER = ("torque_gain", "torque_tau_s", "coulomb_friction", "viscous_friction", "max_velocity_strokes_s")


def row_of(params, row, copies, generator=None, jitter=0.0):
    """One rig's parameters, ``copies`` times, optionally jittered +-``jitter``."""
    values = {}
    for name, value in vars(params).items():
        if value is None:
            values[name] = None; continue
        one = value[row:row + 1].expand(copies, *value.shape[1:]).clone()
        if jitter and name in JITTER:
            noise = torch.rand(copies, device=one.device, generator=generator) * 2 - 1
            one = one * (1 + jitter * noise)
        values[name] = one
    return type(params)(**values)


def practice_songs(rng, batch, song_steps, device, rest_heavy_fraction):
    """Random songs; a fraction of them full of rests and repeated notes."""
    heavy = int(round(batch * rest_heavy_fraction))
    parts = [random_melodies(rng, batch - heavy, song_steps, device, varied=True)] if batch - heavy else []
    if heavy:
        parts.append(random_melodies(rng, heavy, song_steps, device, varied=True, rest_probability=.45, repeats=10))
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def fine_tune(base, plant, twin_rows, steps, lr, anchor_weight, batch, song_steps, seed, device,
              rest_heavy_fraction=0.0):
    model = copy.deepcopy(base).train()
    anchor = [p.detach().clone() for p in base.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed); generator = torch.Generator(device).manual_seed(seed)
    for _ in range(steps):
        params = row_of(twin_rows, 0, batch, generator, jitter=.10)
        cents, voice = practice_songs(rng, batch, song_steps, device, rest_heavy_fraction)
        memory, loss = None, 0.0
        for weight in (.3, 1.0):  # the same song twice: keep the song memory useful
            result, memory = model.perform(plant, cents, voice, params, memory, generator)
            error = result["pitch_cents"] - cents
            loss = loss + weight * F.smooth_l1_loss(error[voice] / 100, torch.zeros_like(error[voice]), beta=.2)
        loss = loss / 1.3 + anchor_weight * sum((p - q).square().sum() for p, q in zip(model.parameters(), anchor))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rigs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fit-steps", type=int, default=200)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--song-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=1e-3)
    ap.add_argument("--random-songs", type=int, default=3)
    ap.add_argument("--rest-heavy-fraction", type=float, default=.5,
                    help="share of practice songs full of rests and repeated notes")
    ap.add_argument("--nn", default="runs/yamabiko_adaptive_memory_v1.pt")
    ap.add_argument("--results", default="docs/plan-results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device; results = pathlib.Path(args.results)
    started = time.time()
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    g = lambda k: torch.Generator(device).manual_seed(args.seed + k)
    # The same 32 rigs, calibration and twin fit as the benchmark (same seeds).
    true = plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(args.seed))
    rig = BlackBoxDevice(plant, true, g(1))
    fit = TwinFitter(plant).fit(run_calibration(rig), steps=args.fit_steps)
    twin = fit.parameters
    print(f"twin fitted {time.time() - started:.0f}s", flush=True)
    base, _ = load_performer(args.nn, device)
    song_list = phrases(device, 1)
    random_songs = [random_melodies(np.random.default_rng(args.seed + 50 + k), 1, 600, device, varied=True)
                    for k in range(args.random_songs)]
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((results / "lab.json").read_text(encoding="utf-8"))
    lab_rows = {label: info["row"] for label, info in manifest["lab_rigs"].items()}
    units = {"phrases": {p: [] for p in (1, 2, 3)}, "random": {p: [] for p in (1, 2, 3)}}
    per_phrase = {p["id"]: {play: [] for play in (1, 2, 3)} for p in song_list}
    lab_tracks = {label: {} for label in lab_rows}
    for row in range(args.rigs):
        model = fine_tune(base, plant, row_of(twin, row, 1), args.steps, args.lr, args.anchor_weight,
                          args.batch, args.song_steps, args.seed + row, device, args.rest_heavy_fraction)
        rig_params = row_of(true, row, 1)
        with torch.no_grad():
            for phrase in song_list:
                memory = None
                for play in (1, 2, 3):
                    result, memory = model.perform(plant, phrase["cents"], phrase["voice"], rig_params, memory,
                                                   g(100 * play + row))
                    unit = unit_scores(result["pitch_cents"], phrase["cents"], phrase["voice"])
                    units["phrases"][play].append(unit); per_phrase[phrase["id"]][play].append(unit)
                    for label, lab_row in lab_rows.items():
                        if lab_row == row:
                            lab_tracks[label].setdefault(phrase["id"], {})[f"nn_twin_play{play}"] = rounded(
                                result["pitch_cents"][0])
            for k, (cents, voice) in enumerate(random_songs):
                memory = None
                for play in (1, 2, 3):
                    result, memory = model.perform(plant, cents, voice, rig_params, memory, g(300 + 10 * play + k))
                    units["random"][play].append(unit_scores(result["pitch_cents"], cents, voice))
        print(f"rig {row + 1}/{args.rigs} (motor x{float(true.max_velocity_strokes_s[row] / plant.config.max_velocity_strokes_s):.2f})"
              f"  {time.time() - started:.0f}s", flush=True)
    for play in (1, 2, 3):
        name = f"nn_twin_play{play}"
        manifest["phrases"][name] = summarize(merge(units["phrases"][play]))
        manifest["random"][name] = summarize(merge(units["random"][play]))
        for entry in manifest["per_phrase"]:
            entry["scores"][name] = summarize(merge(per_phrase[entry["id"]][play]))
    manifest["nn_twin"] = {"rest_heavy_fraction": args.rest_heavy_fraction, "fit_steps": args.fit_steps,
                           "steps": args.steps, "batch": args.batch, "lr": args.lr,
                           "anchor_weight": args.anchor_weight, "twin_jitter": .10,
                           "seconds": time.time() - started}
    for sample in lab["samples"]:
        for label in lab_rows:
            sample["rigs"][label].update(lab_tracks[label].get(sample["id"], {}))
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (results / "lab.json").write_text(json.dumps(lab, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: {p: round(v[f'nn_twin_play{p}']['hit_rate']['mean'] * 100, 1) for p in (1, 2, 3)}
                      for k, v in (("phrases", manifest["phrases"]), ("random", manifest["random"]))}))


if __name__ == "__main__":
    main()
