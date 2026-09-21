"""Evaluate every separated NN and export before/after listening WAV files."""
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
from flute_rl.yamabiko import Rig, RigParams  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_audio import RawSelfAudio  # noqa: E402
from flute_rl.yamabiko.e2e_io import SAMPLE_RATE, frame_audio_numpy  # noqa: E402
from flute_rl.yamabiko.session import HOME_STEPS  # noqa: E402
from flute_rl.yamabiko.staged_nn import (ErrorComparator, FeedForwardPolicy,
                                         FeedbackResidualPolicy, MotorInversePolicy,
                                         ReferenceMemory, StagedConfig,
                                         TargetPositionPlanner)  # noqa: E402

CENTER, SCALE = 1300.0, 600.0


def challenges():
    rest = lambda seconds: np.full(round(seconds * 100), np.nan)
    hold = lambda cents, seconds: np.full(round(seconds * 100), float(cents))
    lead, tail = rest(.9), rest(.2)
    return {
        "sustain": np.r_[lead, hold(1300, 1.0), tail],
        "step_up_down": np.r_[lead, hold(1100, .5), hold(1500, .5), hold(1200, .5), tail],
        "glide": np.r_[lead, np.linspace(1000, 1600, 100), tail],
        "rests": np.r_[lead, hold(1200, .4), rest(.15), hold(1500, .4), rest(.15), hold(1100, .4), tail],
        "melody": np.r_[lead, hold(1100, .3), hold(1300, .3), hold(1500, .3), hold(1200, .3), tail],
    }


def sonify(normalized_pitch, voice, seed):
    cents = np.asarray(normalized_pitch) * SCALE + CENTER
    return synth_self(cents, np.asarray(voice, bool), np.random.default_rng(seed), sr=SAMPLE_RATE)


def load_stages(path, device):
    ck = torch.load(path, map_location=device)
    if ck.get("format") != "yamabiko-staged-nn-v1": raise ValueError("not a staged checkpoint")
    config = StagedConfig(**ck["config"])
    modules = [ReferenceMemory(config), FeedForwardPolicy(config), ErrorComparator(config),
               FeedbackResidualPolicy(config)]
    for module, key in zip(modules, ("memory", "feedforward", "comparator", "feedback")):
        module.load_state_dict(ck[key]); module.to(device).eval()
    planner = motor = None
    if "position_planner" in ck and "motor_inverse" in ck:
        planner, motor = TargetPositionPlanner(config).to(device), MotorInversePolicy(config).to(device)
        planner.load_state_dict(ck["position_planner"]); motor.load_state_dict(ck["motor_inverse"])
        planner.eval(); motor.eval()
    return ck, modules, planner, motor


def corr(a, b):
    if len(a) < 4: return 0.0
    if np.std(a) < 1e-6: return 1.0
    if np.std(b) < 1e-6: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def stage_pitch_metrics(pred_pitch, pred_voice, target):
    note = np.isfinite(target); cents = np.asarray(pred_pitch) * SCALE + CENTER
    voice = np.asarray(pred_voice, bool); valid = note & voice
    penalty = np.where(valid, np.abs(cents - np.nan_to_num(target)), 1200.0)[note]
    rest = ~note
    return {"mae_cents_with_silence_penalty": float(np.mean(penalty)),
            "p90_cents_with_silence_penalty": float(np.quantile(penalty, .9)),
            "trajectory_correlation": corr(target[valid], cents[valid]),
            "voiced_recall": float(np.mean(voice[note])) if note.any() else 1.0,
            "rest_false_positive_rate": float(np.mean(voice[rest])) if rest.any() else 0.0}


def physical_metrics(target, cents, sounding):
    result = stage_pitch_metrics((cents - CENTER) / SCALE, sounding, target)
    changes = np.flatnonzero(np.isfinite(target[1:]) & np.isfinite(target[:-1]) &
                             (np.abs(target[1:] - target[:-1]) >= 50)) + 1
    direction, gains, settled = [], [], []
    for t in changes:
        before = slice(max(0, t - 10), t); after = slice(t + 15, min(len(target), t + 25))
        if after.start >= after.stop: continue
        desired = target[t] - target[t - 1]
        actual = np.nanmean(cents[after]) - np.nanmean(cents[before])
        direction.append(float(np.sign(actual) == np.sign(desired)))
        gains.append(float(abs(actual) / max(abs(desired), 1)))
        settled.append(float(np.mean(np.abs(cents[after] - target[after]))))
    result.update({"transition_direction_accuracy": float(np.mean(direction)) if direction else 1.0,
                   "transition_gain_median": float(np.median(gains)) if gains else 1.0,
                   "settled_transition_mae_cents": float(np.mean(settled)) if settled else 0.0})
    return result


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/yamabiko_staged_nn.pt")
    ap.add_argument("--out", default="docs/e2e-results")
    ap.add_argument("--seed", type=int, default=9471)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = pathlib.Path(args.out); audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ck, (memory, ff, comparator, feedback), position_planner, motor_inverse = load_stages(args.model, args.device)
    ear_path = ck["ear_checkpoint"]
    ear_model = E2EImitator.from_checkpoint(torch.load(ear_path, map_location=args.device), args.device).eval()
    rng = np.random.default_rng(args.seed)
    reports, files = {key: [] for key in ("ear", "memory", "feedforward", "comparator")}, []
    for case_index, (name, target) in enumerate(challenges().items()):
        wave, _ = synth_source(target, "recorder", rng, sr=SAMPLE_RATE); wave = room(wave, SAMPLE_RATE, rng)
        frames = frame_audio_numpy(wave, len(target)); ear = ear_model.audio_features(
            torch.from_numpy(frames).to(args.device))[:, -2:]
        ear_pitch = ear[:, 0].cpu().numpy(); ear_voice = (ear[:, 1] >= 0).cpu().numpy()
        length = torch.tensor([len(target)], device=args.device)
        decoded, stored = memory(ear[None], length); decoded = memory.decode(stored)[0]
        mem_pitch = decoded[:, 0].cpu().numpy(); mem_voice = (decoded[:, 1] >= 0).cpu().numpy()
        if position_planner is not None:
            planned_position = position_planner(decoded[None])
            actions = motor_inverse(planned_position, decoded[None], length)[0].cpu().numpy()
        else:
            actions = ff(decoded[None], length)[0].cpu().numpy()
        rig = Rig(RigParams.nominal(1), np.random.default_rng(args.seed + 100 + case_index))
        for _ in range(HOME_STEPS): rig.step(np.array([-1.0]), np.array([False]))
        cents, sounding = [], []
        for action in actions:
            state = rig.step(np.array([action[0]]), np.array([action[1] >= .5]))
            cents.append(state["cents"][0]); sounding.append(state["sounding"][0])
        cents, sounding = np.asarray(cents), np.asarray(sounding, bool)

        case = audio_dir / name; case.mkdir(parents=True, exist_ok=True)
        paths = {"ear_before": case / "01_ear_before_reference.wav",
                 "ear_after": case / "01_ear_after_prediction.wav",
                 "memory_before": case / "02_memory_before_ear.wav",
                 "memory_after": case / "02_memory_after_recall.wav",
                 "ff_before": case / "03_feedforward_before_target.wav",
                 "ff_after": case / "03_feedforward_after_physical.wav"}
        write_wav(paths["ear_before"], wave, SAMPLE_RATE)
        write_wav(paths["ear_after"], sonify(ear_pitch, ear_voice, 1000 + case_index), SAMPLE_RATE)
        write_wav(paths["memory_before"], sonify(ear_pitch, ear_voice, 1100 + case_index), SAMPLE_RATE)
        write_wav(paths["memory_after"], sonify(mem_pitch, mem_voice, 1200 + case_index), SAMPLE_RATE)
        write_wav(paths["ff_before"], sonify(mem_pitch, mem_voice, 1300 + case_index), SAMPLE_RATE)
        write_wav(paths["ff_after"], synth_self(cents, sounding, np.random.default_rng(1400 + case_index),
                                                sr=SAMPLE_RATE), SAMPLE_RATE)

        reports["ear"].append({"case": name, **stage_pitch_metrics(ear_pitch, ear_voice, target)})
        reports["memory"].append({"case": name, **stage_pitch_metrics(mem_pitch, mem_voice, target)})
        reports["feedforward"].append({"case": name, **physical_metrics(target, cents, sounding)})

        # Comparator listening diagnostic: it must reconstruct target pitch
        # from a deliberately detuned self sound.  This is not actuator audio.
        offset = np.where(np.arange(len(target)) < len(target) // 2, -150.0, 150.0)
        own_cents = np.where(np.isfinite(target), np.nan_to_num(target) + offset, 0.0)
        own_frames = RawSelfAudio(1, np.random.default_rng(2000 + case_index), domain=1.0).render(
            own_cents[None], np.isfinite(target)[None])[0]
        own_ear = ear_model.audio_features(torch.from_numpy(own_frames).to(args.device))[:, -2:]
        error = comparator(decoded, own_ear)[:, 0]
        corrected = own_ear[:, 0] + error
        comp_before = case / "04_comparator_before_detuned_self.wav"
        comp_after = case / "04_comparator_after_reconstructed_target.wav"
        write_wav(comp_before, synth_self(own_cents, np.isfinite(target), np.random.default_rng(2100 + case_index),
                                          sr=SAMPLE_RATE), SAMPLE_RATE)
        write_wav(comp_after, sonify(corrected.cpu().numpy(), mem_voice, 2200 + case_index), SAMPLE_RATE)
        # Score against the physical cents trace, not the convenient neural
        # target-minus-neural-self identity used inside the model.
        prior_note = np.r_[False, np.isfinite(target[:-1])]
        valid = torch.from_numpy(np.isfinite(target) & prior_note).to(args.device)
        observed_cents = np.r_[0.0, own_cents[:-1]]
        true_error = torch.from_numpy(((np.nan_to_num(target) - observed_cents) / SCALE).astype(np.float32)).to(args.device)
        e = (error[valid] - true_error[valid]).abs().cpu().numpy() * SCALE
        reports["comparator"].append({"case": name, "error_mae_cents": float(np.mean(e)),
                                      "error_p90_cents": float(np.quantile(e, .9)),
                                      "error_sign_accuracy": float((torch.sign(error[valid]) ==
                                                                    torch.sign(true_error[valid])).float().mean())})
        files.append({"case": name, **{k: str(v.relative_to(out)).replace("\\", "/") for k, v in paths.items()},
                      "comparator_before": str(comp_before.relative_to(out)).replace("\\", "/"),
                      "comparator_after": str(comp_after.relative_to(out)).replace("\\", "/")})

    def average(stage, key): return float(np.mean([x[key] for x in reports[stage]]))
    summary = {
        "ear": {"mae": average("ear", "mae_cents_with_silence_penalty"),
                "correlation": average("ear", "trajectory_correlation")},
        "memory": {"mae": average("memory", "mae_cents_with_silence_penalty"),
                   "correlation": average("memory", "trajectory_correlation")},
        "feedforward": {"mae": average("feedforward", "mae_cents_with_silence_penalty"),
                        "correlation": average("feedforward", "trajectory_correlation"),
                        "voiced_recall": average("feedforward", "voiced_recall"),
                        "direction": average("feedforward", "transition_direction_accuracy"),
                        "gain": average("feedforward", "transition_gain_median")},
        "comparator": {"error_mae": average("comparator", "error_mae_cents"),
                       "sign_accuracy": average("comparator", "error_sign_accuracy")},
    }
    summary["ear"]["pass"] = summary["ear"]["mae"] < 60 and summary["ear"]["correlation"] >= .9
    summary["memory"]["pass"] = summary["memory"]["mae"] < 60 and summary["memory"]["correlation"] >= .9
    f = summary["feedforward"]; f["pass"] = (f["mae"] < 100 and f["correlation"] >= .9 and
                                               f["voiced_recall"] >= .95 and f["direction"] >= .9 and
                                               .8 <= f["gain"] <= 1.2)
    c = summary["comparator"]; c["pass"] = c["error_mae"] < 60 and c["sign_accuracy"] >= .9
    manifest = {"format": "yamabiko-staged-audio-v1", "checkpoint": args.model,
                "ear_checkpoint": ear_path, "seed": args.seed,
                "sonification_note": "Ear/memory/comparator after-WAVs sonify predicted pitch for diagnosis; only feedforward after-WAV is simulated physical performance.",
                "summary": summary, "details": reports, "audio": files,
                "feedback": {"trained": False, "pass": False, "reason": "feedback residual training is the next gated stage"}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); print(f"wrote {out / 'manifest.json'}")


if __name__ == "__main__": main()
