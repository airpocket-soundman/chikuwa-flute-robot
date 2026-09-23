"""Evaluate the connected pipeline end to end on the ten reference phrases.

Upstream (listen + remember) is either the neural Ear/Tempo/Timeline of the
connected composite or the true score.  Downstream (plan + motor + feedback)
is either the rig-adaptive NN or the encoder-less deterministic performer.
Every route plays on the same randomized realistic rigs (closed-tube flute,
deadband, hearing delay/noise/drop-outs), and every error is measured against
the *true* score, so upstream mistakes count.
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
from evaluate_yamabiko_hybrid_lab import note_sequence, sample_specs, write_wav  # noqa: E402
from flute_rl.audio import room, synth_self, synth_source  # noqa: E402
from flute_rl.yamabiko.composite import YamabikoComposite  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import EncoderlessDeterministicPerformer  # noqa: E402
from flute_rl.yamabiko.e2e_io import frame_audio_numpy  # noqa: E402
from flute_rl.yamabiko.melodies import ARTICULATION_FRAMES, note_age  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.rig_adaptive import RigAdaptivePerformer  # noqa: E402

TEMPO_SCALE = 2
TAIL_STEPS = 60  # let the last note finish before the song ends


def score_track(reference: np.ndarray):
    """True replay target at 0.5x with a valve-off gap at every note start."""
    cents = np.repeat(reference, TEMPO_SCALE)
    voice = np.isfinite(cents)
    onset = voice & ~np.r_[False, voice[:-1]]
    onset |= voice & np.r_[False, np.abs(np.diff(np.nan_to_num(cents))) > 1]
    for start in np.nonzero(onset)[0]:
        voice[start:start + ARTICULATION_FRAMES] = False
    # Hold the last pitch through rests so the downstream always has a target.
    filled = cents.copy()
    last = np.nanmean(reference)
    for i, value in enumerate(filled):
        if np.isfinite(value): last = value
        else: filled[i] = last
    return filled, voice


def pad(values, voice, steps):
    extra = steps - len(values)
    return np.r_[values, np.full(extra, values[-1])], np.r_[voice, np.zeros(extra, bool)]


def errors(played, target, target_voice, played_voice):
    both = target_voice & played_voice
    age = note_age(torch.from_numpy(target_voice)[None])[0].numpy()
    e = np.abs(played - target)
    return {"all": float(e[both].mean()) if both.any() else None,
            "settled": float(e[both & (age > 30)].mean()) if (both & (age > 30)).any() else None,
            "missing_voice": float((target_voice & ~played_voice).sum() / max(1, target_voice.sum())),
            "extra_voice": float((~target_voice & played_voice).sum() / max(1, (~target_voice).sum()))}


def average(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--composite", default="runs/yamabiko_connected_composite_v2.pt")
    ap.add_argument("--downstream", default="runs/yamabiko_rig_adaptive_v2.pt")
    ap.add_argument("--out", default="docs/e2e-integrated-results")
    ap.add_argument("--rigs", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False); device = args.device
    composite = YamabikoComposite.from_checkpoint(
        torch.load(args.composite, map_location=device, weights_only=False), device).eval()
    downstream_checkpoint = torch.load(args.downstream, map_location=device, weights_only=False)
    downstream = RigAdaptivePerformer.from_checkpoint(downstream_checkpoint, device).eval()
    learning_mode = bool(downstream_checkpoint.get("calibration_songs"))
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    deterministic = EncoderlessDeterministicPerformer(plant)
    params = plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(4711))
    n = args.rigs
    # Learning mode: each rig plays the fixed calibration piece once; every
    # evaluated phrase then only reads that rig memory.
    rig_memory = downstream.calibrate(plant, params, torch.Generator(device).manual_seed(999)) if learning_mode else None

    routes = ("nn_listen+nn_play", "nn_listen+det_play", "score+nn_play", "score+det_play")
    samples, per_route = [], {route: [] for route in routes}
    for index, (sample_id, title, description, notes, durations) in enumerate(sample_specs()):
        reference = note_sequence(notes, durations)
        rng = np.random.default_rng(88031 + index)
        waveform, _ = synth_source(reference, "recorder", rng, sr=16_000, shift=0)
        frames = torch.from_numpy(frame_audio_numpy(room(waveform, 16_000, rng), len(reference))).to(device)[None]
        profile = composite.listen(frames, torch.tensor([len(reference)], device=device))
        heard_cents = (1300 + 600 * profile.target[0, :, 0].clamp(-1, 1)).cpu().numpy()
        heard_voice = (torch.sigmoid(profile.target[0, :, 1]) >= .5).cpu().numpy()
        truth, truth_voice = score_track(reference)
        steps = len(truth) + TAIL_STEPS
        truth, truth_voice = pad(truth, truth_voice, steps)
        heard_cents, heard_voice = pad(heard_cents[:len(truth) - TAIL_STEPS], heard_voice[:len(truth) - TAIL_STEPS], steps)
        memory_error = errors(heard_cents, truth, truth_voice, heard_voice)
        row = {"id": sample_id, "title": title, "description": description,
               "memory": memory_error, "routes": {}}
        tracks = {}
        for route in routes:
            source = route.split("+")[0]
            cents_np, voice_np = (heard_cents, heard_voice) if source == "nn_listen" else (truth, truth_voice)
            cents = torch.tensor(cents_np, device=device, dtype=torch.float32)[None].expand(n, -1).contiguous()
            voice = torch.tensor(voice_np, device=device)[None].expand(n, -1).contiguous()
            generator = torch.Generator(device).manual_seed(index)
            if route.endswith("nn_play"):
                result, _ = downstream.perform(plant, cents, voice, params, rig_memory, generator,
                                               write_memory=not learning_mode)
            else:
                result, _ = deterministic.perform(cents, voice, params, None, generator)
            played = result["pitch_cents"].cpu().numpy()
            rig_rows = [errors(played[r], truth, truth_voice, voice_np) for r in range(n)]
            row["routes"][route] = average(rig_rows)
            per_route[route].append(row["routes"][route])
            tracks[route] = played[0]
        samples.append(row)
        if index in (0, 5, 9):
            for route, name in (("nn_listen+nn_play", "nn"), ("nn_listen+det_play", "det")):
                write_wav(out / f"{sample_id}-{name}.wav",
                          synth_self(tracks[route], heard_voice, np.random.default_rng(index), sr=16_000))
            write_wav(out / f"{sample_id}-target.wav",
                      synth_self(truth, truth_voice, np.random.default_rng(99), sr=16_000))
            row["audio"] = {"target": f"{sample_id}-target.wav", "nn": f"{sample_id}-nn.wav",
                            "det": f"{sample_id}-det.wav"}
        print(sample_id, {r: round(row["routes"][r]["all"], 1) for r in routes}, flush=True)

    summary = {
        "format": "yamabiko-integrated-eval-v1", "rigs": n, "samples": len(samples),
        "simulator": "PhysicalPlantConfig.realistic()",
        "upstream_checkpoint": args.composite, "downstream_checkpoint": args.downstream,
        "downstream_learning_mode": learning_mode,
        "routes": {route: average(rows) for route, rows in per_route.items()},
        "memory": average([s["memory"] for s in samples]),
        "error_reference": "true score (not the Ear's estimate)",
        "real_rig_validated": False,
    }
    (out / "manifest.json").write_text(json.dumps({"summary": summary, "samples": samples}, indent=2),
                                       encoding="utf-8")
    print(json.dumps(summary["routes"], indent=2)); print(json.dumps(summary["memory"], indent=2))


if __name__ == "__main__":
    main()
