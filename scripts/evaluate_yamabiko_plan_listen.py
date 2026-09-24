"""The whole route on the plan benchmark: listen to the demonstration, then play.

The reference phrases are synthesized as audio (recorder + room), the neural
Ear / Tempo / Timeline of the connected composite listens once and stores
the target, and the downstream plays that stored target on the 32 unknown
rigs with their fitted twins.  Scores are against the *true* score, so the
listening error counts.  Results are merged into docs/plan-results as
``listen_det_twin`` and ``listen_residual_play1..3``.
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
from evaluate_yamabiko_fitting import TAIL, rounded  # noqa: E402
from evaluate_yamabiko_hybrid_lab import note_sequence, sample_specs  # noqa: E402
from evaluate_yamabiko_integrated import pad, score_track  # noqa: E402
from flute_rl.audio import room, synth_source  # noqa: E402
from flute_rl.yamabiko.composite import YamabikoComposite  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import EncoderlessDeterministicPerformer  # noqa: E402
from flute_rl.yamabiko.e2e_io import frame_audio_numpy  # noqa: E402
from flute_rl.yamabiko.fitting import cached_twin  # noqa: E402
from flute_rl.yamabiko.musical_metrics import merge, summarize, unit_scores  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.residual_performer import ResidualTwinPerformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--composite", default="runs/yamabiko_connected_composite_v2.pt")
    ap.add_argument("--checkpoint", default="runs/yamabiko_residual_v1.pt")
    ap.add_argument("--rigs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fit-steps", type=int, default=300)
    ap.add_argument("--results", default="docs/plan-results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device; results = pathlib.Path(args.results)
    torch.set_grad_enabled(False)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    true = plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(args.seed))
    with torch.enable_grad():
        twin = cached_twin(plant, true, args.seed, args.fit_steps, device)
    composite = YamabikoComposite.from_checkpoint(
        torch.load(args.composite, map_location=device, weights_only=False), device).eval()
    model = ResidualTwinPerformer.from_checkpoint(torch.load(args.checkpoint, map_location=device, weights_only=False),
                                                  plant, device).eval()
    det = EncoderlessDeterministicPerformer(plant)
    g = lambda k: torch.Generator(device).manual_seed(args.seed + k)
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((results / "lab.json").read_text(encoding="utf-8"))
    lab_rows = {label: info["row"] for label, info in manifest["lab_rigs"].items()}
    units = {"listen_det_twin": [], **{f"listen_residual_play{p}": [] for p in (1, 2, 3)}}
    memory_units = []
    for k, (sample_id, title, description, notes, durations) in enumerate(sample_specs()):
        reference = note_sequence(notes, durations)
        rng = np.random.default_rng(88031 + k)
        waveform, _ = synth_source(reference, "recorder", rng, sr=16_000, shift=0)
        frames = torch.from_numpy(frame_audio_numpy(room(waveform, 16_000, rng), len(reference))).to(device)[None]
        profile = composite.listen(frames, torch.tensor([len(reference)], device=device))
        heard_cents = (1300 + 600 * profile.target[0, :, 0].clamp(-1, 1)).cpu().numpy()
        heard_voice = (torch.sigmoid(profile.target[0, :, 1]) >= .5).cpu().numpy()
        truth, truth_voice = score_track(reference)
        steps = len(truth) + TAIL
        truth, truth_voice = pad(truth, truth_voice, steps)
        heard_cents, heard_voice = pad(heard_cents[:steps - TAIL], heard_voice[:steps - TAIL], steps)
        truth_t = torch.tensor(truth, device=device, dtype=torch.float32)[None].expand(args.rigs, -1).contiguous()
        truth_voice_t = torch.tensor(truth_voice, device=device)[None].expand(args.rigs, -1).contiguous()
        cents = torch.tensor(heard_cents, device=device, dtype=torch.float32)[None].expand(args.rigs, -1).contiguous()
        voice = torch.tensor(heard_voice, device=device)[None].expand(args.rigs, -1).contiguous()
        both = truth_voice_t & voice
        memory_units.append(unit_scores(cents[:1], truth_t[:1], both[:1]))
        entry = next(p for p in manifest["per_phrase"] if p["id"] == sample_id)
        sample = next(s for s in lab["samples"] if s["id"] == sample_id)
        played, _ = det.perform(cents, voice, true, None, g(10 + k), twin=twin)
        unit = unit_scores(played["pitch_cents"], truth_t, both)
        units["listen_det_twin"].append(unit); entry["scores"]["listen_det_twin"] = summarize(unit)
        for label, row in lab_rows.items():
            sample["rigs"][label]["listen_det_twin"] = rounded(played["pitch_cents"][row])
        memory = None
        for play in (1, 2, 3):
            result, memory = model.perform(plant, cents, voice, true, memory, g(100 * play + k), twin=twin)
            unit = unit_scores(result["pitch_cents"], truth_t, both)
            units[f"listen_residual_play{play}"].append(unit)
            entry["scores"][f"listen_residual_play{play}"] = summarize(unit)
            for label, row in lab_rows.items():
                sample["rigs"][label][f"listen_residual_play{play}"] = rounded(result["pitch_cents"][row])
        sample["heard_target"] = rounded(cents[0]); sample["heard_voice"] = voice[0].int().tolist()
        print(title, "det", round(entry["scores"]["listen_det_twin"]["hit_rate"]["mean"] * 100),
              "residual", [round(entry["scores"][f"listen_residual_play{p}"]["hit_rate"]["mean"] * 100) for p in (1, 2, 3)],
              flush=True)
    for name, rows in units.items():
        manifest["phrases"][name] = summarize(merge(rows))
    manifest["listening"] = {"composite": args.composite, "memory_in_tune": summarize(merge(memory_units))}
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (results / "lab.json").write_text(json.dumps(lab, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({n: round(manifest["phrases"][n]["hit_rate"]["mean"] * 100, 1) for n in units}))


if __name__ == "__main__":
    main()
