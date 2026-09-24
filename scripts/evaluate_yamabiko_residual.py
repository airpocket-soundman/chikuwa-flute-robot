"""Score the residual performer (deterministic twin skeleton + NN) on the plan benchmark.

Same 32 rigs, calibration and fitted twins as evaluate_yamabiko_fitting
(cached), same phrases and random songs; three plays per song with the song
memory carried.  Results are merged into docs/plan-results as
``residual_play1..3``.
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
from evaluate_yamabiko_fitting import phrases, rounded  # noqa: E402
from flute_rl.yamabiko.fitting import cached_twin  # noqa: E402
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.musical_metrics import merge, summarize, unit_scores  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.residual_performer import ResidualTwinPerformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/yamabiko_residual_v1.pt")
    ap.add_argument("--rigs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fit-steps", type=int, default=300)
    ap.add_argument("--random-songs", type=int, default=3)
    ap.add_argument("--results", default="docs/plan-results")
    ap.add_argument("--fit-only", action="store_true", help="only fit and cache the twins")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device; results = pathlib.Path(args.results)
    torch.set_grad_enabled(False)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    true = plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(args.seed))
    with torch.enable_grad():
        twin = cached_twin(plant, true, args.seed, args.fit_steps, device)
    if args.fit_only:
        print("twins cached"); return
    model = ResidualTwinPerformer.from_checkpoint(torch.load(args.checkpoint, map_location=device, weights_only=False),
                                                  plant, device).eval()
    g = lambda k: torch.Generator(device).manual_seed(args.seed + k)
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((results / "lab.json").read_text(encoding="utf-8"))
    lab_rows = {label: info["row"] for label, info in manifest["lab_rigs"].items()}
    units = {"phrases": {p: [] for p in (1, 2, 3)}, "random": {p: [] for p in (1, 2, 3)}}
    for k, phrase in enumerate(phrases(device, args.rigs)):
        memory = None
        entry = next(p for p in manifest["per_phrase"] if p["id"] == phrase["id"])
        sample = next(s for s in lab["samples"] if s["id"] == phrase["id"])
        for play in (1, 2, 3):
            result, memory = model.perform(plant, phrase["cents"], phrase["voice"], true, memory,
                                           g(100 * play + k), twin=twin)
            unit = unit_scores(result["pitch_cents"], phrase["cents"], phrase["voice"])
            units["phrases"][play].append(unit)
            entry["scores"][f"residual_play{play}"] = summarize(unit)
            for label, row in lab_rows.items():
                sample["rigs"][label][f"residual_play{play}"] = rounded(result["pitch_cents"][row])
        print(phrase["title"], [round(entry["scores"][f"residual_play{p}"]["hit_rate"]["mean"] * 100)
                                for p in (1, 2, 3)], flush=True)
    for k in range(args.random_songs):
        cents, voice = random_melodies(np.random.default_rng(args.seed + 50 + k), args.rigs, 600, device, varied=True)
        memory = None
        for play in (1, 2, 3):
            result, memory = model.perform(plant, cents, voice, true, memory, g(300 + 10 * play + k), twin=twin)
            units["random"][play].append(unit_scores(result["pitch_cents"], cents, voice))
    for play in (1, 2, 3):
        manifest["phrases"][f"residual_play{play}"] = summarize(merge(units["phrases"][play]))
        manifest["random"][f"residual_play{play}"] = summarize(merge(units["random"][play]))
    manifest["residual"] = {"checkpoint": args.checkpoint, "fit_steps": args.fit_steps}
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (results / "lab.json").write_text(json.dumps(lab, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({s: {p: round(manifest[s][f"residual_play{p}"]["hit_rate"]["mean"] * 100, 1) for p in (1, 2, 3)}
                      for s in ("phrases", "random")}))


if __name__ == "__main__":
    main()
