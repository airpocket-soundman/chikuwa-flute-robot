"""Compare encoder-less deterministic control and the rig-adaptive NN.

Both play the same random songs on the same randomized realistic rigs
(closed-tube flute, deadband, hearing delay/noise/drop-outs), several songs
in a row per rig.  Neither reads the plant's state or parameters; the
encoder oracle row does, as an upper bound.  Writes the report manifest, a
pitch plot and WAVs of one example.
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_targets import write_wav  # noqa: E402
from flute_rl.audio import synth_self  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import (EncoderlessConfig,  # noqa: E402
                                                       EncoderlessDeterministicPerformer, RigMemory,
                                                       period_ms)
from flute_rl.yamabiko.error_regions import error_regions, region_errors  # noqa: E402
from flute_rl.yamabiko.melodies import next_voiced, note_age, random_melodies  # noqa: E402
from flute_rl.yamabiko.musical_metrics import merge, summarize, unit_scores  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.performers import load_performer  # noqa: E402


REGIONS = {}


def errors(played, cents, voice, settle=30):
    age = note_age(voice); e = (played - cents).abs()
    result = {"all": float(e[voice].mean()), "onset": float(e[voice & (age <= settle)].mean()),
              "settled": float(e[voice & (age > settle)].mean())}
    masks = REGIONS.get(id(cents))
    if masks is not None:
        regions = region_errors(played, cents, voice, masks)
        result.update({key: regions[key] for key in ("transit", "departing", "core", "core_frames",
                                                      "transit_frames", "departing_frames")})
    return result


def mean_rows(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def oracle_coefficients(plant, params):
    zero = torch.zeros_like(params.torque_gain)
    a = plant.period_s(zero, params) * 1000.0
    return RigMemory(torch.stack([a, a - plant.period_s(zero + 1, params) * 1000.0], 1),
                     torch.full((len(a), 2), 1e-9, device=a.device))


def encoder_oracle(plant, performer, cents, voice, params, memory):
    """PD on the plant's true position, true flute coefficients and true motor
    speed for the anticipation: a position-sensor reference (same PD gains)."""
    cfg = performer.config
    a, b = memory.theta[:, 0], memory.theta[:, 1]
    true_ratio = params.max_velocity_strokes_s / plant.config.max_velocity_strokes_s
    aim = performer._anticipate(next_voiced(cents, voice), voice, a, b, true_ratio)
    state = plant.initial_state(*cents.shape[:1], cents.device)
    band = cfg.deadband_compensation
    played = []
    for t in range(cents.shape[1]):
        x_cmd = ((a - period_ms(aim[:, t])) / b).clamp(0, 1)
        u = (cfg.kp * (x_cmd - state.position) - cfg.kd * state.velocity).clamp(-1, 1)
        pwm = torch.where(u.abs() > .01, torch.sign(u) * (band + (1 - band) * u.abs()), torch.zeros_like(u))
        state = plant.step(state, pwm, params)
        played.append(plant.flute(state, voice[:, t].float(), params)[0])
    return torch.stack(played, 1)


def pitch_plot(path, target, voice, series, frame_rate=100):
    width, height, left, top, right, bottom = 960, 360, 64, 28, 20, 48
    low, high = 650.0, 1950.0

    def xy(i, c):
        return f"{left + i * (width - left - right) / (len(target) - 1):.1f},{top + (high - c) * (height - top - bottom) / (high - low):.1f}"

    def lines(values, mask, color, stroke):
        out, run = [], []
        for i, (value, keep) in enumerate(zip(values, mask)):
            if keep: run.append(xy(i, float(np.clip(value, low, high))))
            elif run:
                if len(run) > 1: out.append(run)
                run = []
        if len(run) > 1: out.append(run)
        return "".join(f'<polyline points="{" ".join(r)}" fill="none" stroke="{color}" stroke-width="{stroke}"/>' for r in out)

    body = lines(target, voice, "#e5e7eb", 4)
    legend = ['<text x="70" y="18" fill="#e5e7eb">— target</text>']
    for k, (label, values, color) in enumerate(series):
        body += lines(values, voice, color, 2)
        legend.append(f'<text x="{160 + 230 * k}" y="18" fill="{color}">— {label}</text>')
    ticks = "".join(f'<line x1="{left}" y1="{xy(0, c).split(",")[1]}" x2="{width - right}" y2="{xy(0, c).split(",")[1]}" stroke="#334155"/>'
                    f'<text x="{left - 8}" y="{float(xy(0, c).split(",")[1]) + 4:.1f}" text-anchor="end">{c}</text>'
                    for c in (700, 1000, 1300, 1600, 1900))
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#071521"/><g font-family="sans-serif" font-size="12" fill="#cbd5e1">{ticks}{body}{''.join(legend)}
<text x="{width / 2}" y="{height - 10}" text-anchor="middle">time [s] 0 — {(len(target) - 1) / frame_rate:.1f}</text></g></svg>''', encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", nargs="+",
                    default=["runs/yamabiko_rig_adaptive_v3.pt", "runs/yamabiko_adaptive_memory_v1.pt"],
                    help="rig-adaptive checkpoints; the last one is reported as the current model")
    ap.add_argument("--out", default="docs/e2e-rig-adaptive-results")
    ap.add_argument("--rigs", type=int, default=128)
    ap.add_argument("--songs", type=int, default=4)
    ap.add_argument("--song-steps", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False); device = args.device
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    params = plant.parameters(args.rigs, device, spread=1.0, generator=torch.Generator(device).manual_seed(31337))
    songs = [random_melodies(np.random.default_rng(5000 + k), args.rigs, args.song_steps, device)
             for k in range(args.songs)]
    for cents, voice in songs:  # mechanism-limited regions, fixed by score and rig only
        REGIONS[id(cents)] = error_regions(plant, params, cents, voice)

    def generator(k):
        return torch.Generator(device).manual_seed(k)

    def deterministic(name, config, carry=False, memory=None, learn=None, calibrate=False):
        performer = EncoderlessDeterministicPerformer(plant, config)
        if learn is not None: performer.learn = learn
        ratio = performer.measure_motor(params, generator(998)) if calibrate else None
        rows, results = [], []
        for k, (cents, voice) in enumerate(songs):
            result, new = performer.perform(cents, voice, params, memory, generator(k), motor_ratio=ratio)
            rows.append(errors(result["pitch_cents"], cents, voice)); results.append(result)
            if carry: memory = new
        musical[name] = summarize(merge(
            [unit_scores(r["pitch_cents"], c, v) for r, (c, v) in zip(results, songs)]))
        return rows, results

    base = EncoderlessConfig()
    variants, musical = {}, {}
    variants["dead_reckoning_only"] = deterministic("dead_reckoning_only", dataclasses.replace(
        base, observer_gain=0.0, observer_velocity_gain=0.0, anticipation_speed=None))[0]
    variants["pitch_observer"] = deterministic("pitch_observer", dataclasses.replace(base, anticipation_speed=None))[0]
    variants["observer_anticipation"] = deterministic("observer_anticipation", base)[0]
    variants["observer_anticipation_calibrated"], det_results = deterministic(
        "observer_anticipation_calibrated", base, calibrate=True)
    variants["plus_coefficient_learning"] = deterministic("plus_coefficient_learning", base, carry=True, learn=True)[0]
    oracle_memory = oracle_coefficients(plant, params)
    variants["plus_true_coefficients"] = deterministic("plus_true_coefficients", base, memory=oracle_memory)[0]
    performer = EncoderlessDeterministicPerformer(plant, base)
    oracle_played = [encoder_oracle(plant, performer, c, v, params, oracle_memory) for c, v in songs]
    variants["encoder_oracle"] = [errors(p, c, v) for p, (c, v) in zip(oracle_played, songs)]
    musical_by_variant = musical
    musical_by_variant["encoder_oracle"] = summarize(merge([unit_scores(p, c, v)
                                                           for p, (c, v) in zip(oracle_played, songs)]))

    def neural_repeat(model, learning_mode, p, song_list):
        """Play each song twice on the same rigs; memory from play 1 carries into play 2."""
        firsts, seconds, results = [], [], []
        for k, (cents, voice) in enumerate(song_list):
            first, memory = model.perform(plant, cents, voice, p, None, generator(k))
            second, _ = model.perform(plant, cents, voice, p, memory, generator(k + 100),
                                      write_memory=not learning_mode)
            firsts.append(errors(first["pitch_cents"], cents, voice))
            seconds.append(errors(second["pitch_cents"], cents, voice)); results.append((first, second))
        return firsts, seconds, results

    models, neural_results, training = {}, None, {}
    for path in map(pathlib.Path, args.checkpoint):
        if not path.exists():
            continue
        model, checkpoint = load_performer(path, device)
        learning_mode = bool(checkpoint.get("calibration_songs"))
        firsts, seconds, results = neural_repeat(model, learning_mode, params, songs)
        plays = {label: summarize(merge([unit_scores(pair[i]["pitch_cents"], c, v)
                                         for pair, (c, v) in zip(results, songs)]))
                 for i, label in ((0, "first_play"), (1, "second_play"))}
        models[path.stem] = {"musical": plays,"checkpoint": str(path), "learning_mode": learning_mode,
                             "protocol": checkpoint.get("protocol", "songs"),
                             "best_step": checkpoint.get("best", {}).get("step"), "seed": checkpoint.get("seed"),
                             "first_play": {"per_song": firsts, "mean": mean_rows(firsts)},
                             "second_play": {"per_song": seconds, "mean": mean_rows(seconds)}}
        neural_results = [pair[1] for pair in results]
    neural = {key: value for key, value in list(models.values())[-1].items()
              if key in ("first_play", "second_play")} if models else {}
    if models:
        training = {key: value for key, value in list(models.values())[-1].items()
                    if key not in ("first_play", "second_play")}

    # Motor robustness: the unmeasured actuator could be much slower or faster.
    robustness = {"factors": [0.4, 0.6, 1.0, 1.6, 2.5], "rows": {}}
    last_model = None
    if models:
        last_path = pathlib.Path(list(models.values())[-1]["checkpoint"])
        last_model, last_checkpoint = load_performer(last_path, device)
        last_learning = bool(last_checkpoint.get("calibration_songs"))
    cfg = plant.config
    for factor in robustness["factors"]:
        p = plant.parameters(96, device, spread=1.0, generator=torch.Generator(device).manual_seed(55))
        p.max_velocity_strokes_s = torch.full_like(p.max_velocity_strokes_s, cfg.max_velocity_strokes_s * factor)
        p.torque_gain = torch.full_like(p.torque_gain, cfg.torque_gain * factor)
        trial = [random_melodies(np.random.default_rng(7000 + k), 96, args.song_steps, device) for k in range(2)]
        performer = EncoderlessDeterministicPerformer(plant, base)
        ratio = performer.measure_motor(p, generator(998))
        cells = {"det_fixed": [], "det_calibrated": [], "nn_first": [], "nn_second": []}
        for k, (c, v) in enumerate(trial):
            masks = error_regions(plant, p, c, v)
            cells["det_fixed"].append(region_errors(performer.perform(c, v, p, None, generator(k))[0]["pitch_cents"], c, v, masks))
            cells["det_calibrated"].append(region_errors(
                performer.perform(c, v, p, None, generator(k), motor_ratio=ratio)[0]["pitch_cents"], c, v, masks))
            if last_model is not None:
                first, memory = last_model.perform(plant, c, v, p, None, generator(k))
                second, _ = last_model.perform(plant, c, v, p, memory, generator(k + 100),
                                               write_memory=not last_learning)
                cells["nn_first"].append(region_errors(first["pitch_cents"], c, v, masks))
                cells["nn_second"].append(region_errors(second["pitch_cents"], c, v, masks))
        robustness["rows"][str(factor)] = {
            key: {"all": float(np.mean([r["all"] for r in rows])), "core": float(np.mean([r["core"] for r in rows]))}
            for key, rows in cells.items() if rows}
        robustness["rows"][str(factor)]["measured_ratio"] = float(ratio.median())

    cents, voice = songs[0][0][0].cpu().numpy(), songs[0][1][0].cpu().numpy()
    series = [("deterministic encoder-less", det_results[0]["pitch_cents"][0].cpu().numpy(), "#f59e0b")]
    write_wav(out / "target.wav", synth_self(cents, voice, np.random.default_rng(1), sr=16_000), 16_000)
    write_wav(out / "deterministic.wav", synth_self(series[0][1], voice, np.random.default_rng(2), sr=16_000), 16_000)
    audio = {"target": "target.wav", "deterministic": "deterministic.wav"}
    if neural_results is not None:
        nn_first = neural_results[0]["pitch_cents"][0].cpu().numpy()
        nn_last = neural_results[-1]["pitch_cents"][0].cpu().numpy()
        series.append(("rig-adaptive NN", nn_first, "#34d399"))
        write_wav(out / "neural.wav", synth_self(nn_first, voice, np.random.default_rng(3), sr=16_000), 16_000)
        audio["neural"] = "neural.wav"
    pitch_plot(out / "pitch.svg", cents, voice, series)

    summary = {
        "format": "yamabiko-rig-adaptive-eval-v1",
        "rigs": args.rigs, "songs": args.songs, "song_steps": args.song_steps,
        "simulator": "PhysicalPlantConfig.realistic(): closed tube, deadband 0.20, hearing delay 1+-1, noise 3 cent, dropout 2%",
        "settle_steps": 30,
        "deterministic": {key: {"per_song": rows, "mean": mean_rows(rows), "musical": musical_by_variant.get(key)}
                          for key, rows in variants.items()},
        "neural": neural,
        "neural_models": models,
        "motor_robustness": robustness,
        "neural_training": training,
        "network_received_simulator_internal_state": False,
        "real_rig_validated": False,
    }
    manifest = {"summary": summary, "audio": audio, "plot": "pitch.svg"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in summary["deterministic"].items()}, indent=2))
    for name, entry in models.items():
        print(name, {k: [round(r["all"], 1) for r in entry[k]["per_song"]] for k in ("first_play", "second_play")})


if __name__ == "__main__":
    main()
