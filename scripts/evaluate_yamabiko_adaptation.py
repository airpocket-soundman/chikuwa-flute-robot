"""Measure the three kinds of adaptation separately.

* Song: on a familiar rig, the same song is played several times; the error
  per repetition is the learning curve.
* Motor: the motor is swapped (speed and torque x factor) on a familiar
  flute and *different* songs follow; the error per song shows how the rig
  memory transfers across songs.
* Flute: the flute is swapped (tube length, temperature, blowing pressure)
  on a familiar motor, same protocol.

Only the tested kind of memory may write (the others are frozen), and songs
are never repeated in the motor/flute tests, so "getting used to a song"
cannot leak into them.  Single-memory models cannot freeze the others; they
are run with their one memory writing and reported as mixed.  The
deterministic baseline re-runs its motor measurement after a motor swap.  A model's memory is either one tensor (rig and
song mixed) or a (rig, song) pair; ``perform`` receives whatever the model
returned last, and ``new_song`` / ``new_rig`` tell the model which part to
forget (models with one memory only forget on a new rig).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.adaptive_memory import AdaptiveMemoryPerformer  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import EncoderlessDeterministicPerformer  # noqa: E402
from flute_rl.yamabiko.error_regions import error_regions, region_errors  # noqa: E402
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.rig_adaptive import RigAdaptivePerformer  # noqa: E402

MOTOR_FIELDS = ("torque_gain", "torque_tau_s", "inertia", "coulomb_friction", "viscous_friction",
                "max_velocity_strokes_s", "deadband")
FLUTE_FIELDS = ("tube_offset_m", "temp_offset_c", "flute_offset_cents")


class NeuralPlayer:
    """Wraps either memory model.  ``separable`` models take per-part write flags."""

    def __init__(self, model, plant, learning_mode=False):
        self.model, self.plant, self.learning_mode = model, plant, learning_mode
        self.separable = hasattr(model, "writers")
        self.memory = None

    def new_rig(self):
        self.memory = None

    def new_song(self):
        if self.separable:
            self.memory = self.model.forget_song(self.memory)

    def swap(self, part):
        if self.separable:
            self.memory = self.model.forget(self.memory, part)
        # a single mixed memory has no part to forget: it must re-adapt as it is

    def play(self, cents, voice, params, generator, only=None):
        """``only``: the one memory allowed to write (None: all)."""
        if self.separable:
            write = True if only is None else {part: part == only for part in ("motor", "flute", "song")}
            result, self.memory = self.model.perform(self.plant, cents, voice, params, self.memory, generator,
                                                     write=write)
        else:
            write = not self.learning_mode or self.memory is None
            result, self.memory = self.model.perform(self.plant, cents, voice, params, self.memory, generator,
                                                     write_memory=write)
        return result["pitch_cents"]


class DeterministicPlayer:
    """Encoder-less baseline: re-runs its motor measurement on a new rig or a motor swap."""

    separable = True

    def __init__(self, plant):
        self.performer, self.plant = EncoderlessDeterministicPerformer(plant), plant
        self.ratio = None

    def new_rig(self):
        self.ratio = None

    def new_song(self):
        pass

    def swap(self, part):
        if part == "motor":
            self.ratio = None

    def play(self, cents, voice, params, generator, only=None):
        if self.ratio is None:
            self.ratio = self.performer.measure_motor(params, generator)
        return self.performer.perform(cents, voice, params, None, generator, motor_ratio=self.ratio)[0]["pitch_cents"]


def score(plant, params, cents, voice, played):
    regions = region_errors(played, cents, voice, error_regions(plant, params, cents, voice))
    return {"all": regions["all"], "core": regions["core"]}


def swapped(base, other, fields):
    result = type(base)(**vars(base))
    for field in fields:
        setattr(result, field, getattr(other, field).clone())
    return result


def mean_curve(rows_by_song, count):
    return [{key: float(np.mean([rows[i][key] for rows in rows_by_song])) for key in ("all", "core")}
            for i in range(count)]


def run(player, plant, device, rigs, songs_per_test, repeats, song_steps, seed):
    g = lambda k: torch.Generator(device).manual_seed(seed + k)
    base = plant.parameters(rigs, device, spread=1.0, generator=g(1))
    other = plant.parameters(rigs, device, spread=1.0, generator=g(2))
    song = lambda k: random_melodies(np.random.default_rng(seed * 10 + k), rigs, song_steps, device, varied=True)
    familiar = [song(90 + k) for k in range(2)]

    def warm_up(params):  # every memory writes while the rig becomes familiar
        player.new_rig()
        for k, (c, v) in enumerate(familiar):
            player.new_song(); player.play(c, v, params, g(300 + k))

    # Song: familiar rig, only the song memory adapts.
    curves = []
    for k in range(songs_per_test):
        warm_up(base)
        c, v = song(k); player.new_song()
        curves.append([score(plant, base, c, v, player.play(c, v, base, g(400 + 10 * k + r), only="song"))
                       for r in range(repeats)])
    song_curve = mean_curve(curves, repeats)

    def swap_test(fields, part):
        new_params = swapped(base, other, fields)
        warm_up(base)
        player.swap(part)
        rows = []
        for k in range(repeats):  # different songs, only this part's memory adapts
            c, v = song(20 + k); player.new_song()
            rows.append(score(plant, new_params, c, v, player.play(c, v, new_params, g(500 + k), only=part)))
        return rows

    # Overall: a new rig, every memory adapting, each song three times.
    player.new_rig(); overall = []
    for k in range(songs_per_test):
        c, v = song(60 + k); player.new_song()
        overall.append([score(plant, base, c, v, player.play(c, v, base, g(700 + 10 * k + r))) for r in range(3)])
    return {"song_curve": song_curve,
            "motor_swap": swap_test(MOTOR_FIELDS, "motor"),
            "flute_swap": swap_test(FLUTE_FIELDS, "flute"),
            "overall": mean_curve(overall, 3),
            "separable": bool(player.separable)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", nargs="+",
                    default=["runs/yamabiko_rig_adaptive_v2.pt", "runs/yamabiko_rig_adaptive_v3.pt",
                             "runs/yamabiko_adaptive_memory_v1.pt"])
    ap.add_argument("--rigs", type=int, default=64)
    ap.add_argument("--songs", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--song-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default="docs/e2e-rig-adaptive-results/adaptation.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    torch.set_grad_enabled(False)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    players = {"deterministic": DeterministicPlayer(plant)}
    for path in map(pathlib.Path, args.checkpoint):
        if path.exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if checkpoint.get("format") == "yamabiko-adaptive-memory-v1":
                model = AdaptiveMemoryPerformer.from_checkpoint(checkpoint, device).eval()
                name = "v4 (3 memories)"
            else:
                model = RigAdaptivePerformer.from_checkpoint(checkpoint, device).eval()
                name = path.stem.replace("yamabiko_rig_adaptive_", "") + " (mixed memory)"
            players[name] = NeuralPlayer(model, plant, bool(checkpoint.get("calibration_songs")))
    result = {"rigs": args.rigs, "songs": args.songs, "repeats": args.repeats, "players": {}}
    for name, player in players.items():
        result["players"][name] = outcome = run(player, plant, device, args.rigs, args.songs,
                                                args.repeats, args.song_steps, args.seed)
        fmt = lambda rows: " ".join(f"{r['all']:5.1f}" for r in rows)
        print(f"{name:18s} song {fmt(outcome['song_curve'])} | motor swap {fmt(outcome['motor_swap'])} | "
              f"flute swap {fmt(outcome['flute_swap'])} | overall {fmt(outcome['overall'])}", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
