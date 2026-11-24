"""A NumPy stand-in for the rig, to compare performers side by side without PyTorch or hardware.

The rig is the twin's own equations (``Runtime.model_step`` and ``Runtime.cents_at``, the same as the PyTorch
simulator ``physical_plant.DifferentiableMotorFlute``) with its parameters drawn around a fitted twin
(as ``scripts/yamabiko_twin_finetune.rigs_around``), and the imperfections of ``PhysicalPlantConfig.yamabiko1_fan80``:
the heard pitch arrives ``hearing_delay_steps`` late with noise and drop-outs, the motor hides the flute in
some of the steps it drives, the valve follows its command ``valve_delay_steps`` late, and above
``overblow_cents`` the flute does not sound.

``play(runtime, rig, cents, voice)`` runs one take as the server does (homing, then command / step / observe
every 10 ms) and returns what the flute emitted, what was heard, the PWM and the valve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .residual_numpy import CONSTANTS, TWIN_FIELDS, Runtime

SCALED = ("torque_gain", "torque_tau_s", "inertia", "coulomb_friction", "viscous_friction",
          "max_velocity_strokes_s", "deadband")
SHIFTED = {"flute_offset_cents": 5.0, "tube_offset_m": .002, "temp_offset_c": 1.0}


@dataclass
class RigImperfections:
    """Defaults: PhysicalPlantConfig.yamabiko1_fan80 (realistic(): noise 3 cent, 2 % drop-outs)."""
    pitch_noise_cents: float = 3.0
    dropout: float = .02
    motor_masking: float = .2
    valve_delay_steps: int = 3
    overblow_cents: float = 1250.0
    homing_steps: int = 130


class NumpyRig:
    """One simulated rig: ``step(pwm, valve)`` -> (emitted cents, sounding, heard cents, heard valid)."""

    def __init__(self, constants: dict, twin: dict, imperfections: RigImperfections = RigImperfections(),
                 rng: np.random.Generator | None = None):
        self.c, self.t, self.imp = dict(constants), dict(twin), imperfections
        self.rng = rng or np.random.default_rng(0)
        self.delay = int(max(self.t["hearing_delay_steps"], 0))
        self.reset(0.0)

    def reset(self, position: float):
        self.position, self.velocity, self.torque = float(position), 0.0, 0.0
        self.emitted = [(0.0, False)] * (self.delay + 1)
        self.valve_queue = [False] * self.imp.valve_delay_steps
        self.last_heard = 0.0

    cents_at = Runtime.cents_at
    model_step = Runtime.model_step

    def step(self, pwm: float, valve: bool):
        self.model_step(float(pwm))
        self.valve_queue.append(bool(valve))
        air = self.valve_queue.pop(0)
        cents = float(self.cents_at(self.position))
        sounding = air and cents <= self.imp.overblow_cents
        self.emitted = self.emitted[1:] + [(cents, sounding)]
        heard, valid = self.emitted[0]
        valid = valid and self.rng.random() >= self.imp.dropout
        band = self.t["deadband"] if self.c["deadband"] > 0 else 0.0
        if valid and abs(pwm) > band and self.rng.random() < self.imp.motor_masking:
            valid = False
        if valid:
            heard = heard + self.imp.pitch_noise_cents * self.rng.normal()
            self.last_heard = heard
        return cents, sounding, (heard if valid else self.last_heard), valid


def rig_arrays(path_or_arrays) -> tuple[dict, dict]:
    """(constants, twin) dicts of a runtime ``.npz`` (a twin runtime, a step's actuator runtime, ...)."""
    a = np.load(path_or_arrays) if isinstance(path_or_arrays, (str, bytes)) or hasattr(path_or_arrays, "__fspath__") \
        else path_or_arrays
    return dict(zip(CONSTANTS, np.asarray(a["constants"]).tolist())), dict(zip(TWIN_FIELDS, np.asarray(a["twin"]).tolist()))


def rigs_around(constants: dict, twin: dict, count: int, spread: float = .12, seed: int = 7777,
                imperfections: RigImperfections = RigImperfections(), delay_flip: float = .3) -> list[NumpyRig]:
    """Rigs the real one could be, given the fitted twin (as yamabiko_twin_finetune.rigs_around)."""
    rng = np.random.default_rng(seed)
    rigs = []
    for k in range(count):
        t = dict(twin)
        for name in SCALED:
            t[name] = t[name] * (1 + spread * rng.uniform(-1, 1))
        for name, amount in SHIFTED.items():
            t[name] = t[name] + amount * rng.uniform(-1, 1)
        t["torque_tau_s"] = max(t["torque_tau_s"], .011)
        t["deadband"] = min(max(t["deadband"], 0.0), .6)
        if rng.random() < delay_flip:
            t["hearing_delay_steps"] = max(t["hearing_delay_steps"] + (1 if rng.random() < .5 else -1), 0)
        rigs.append(NumpyRig(constants, t, imperfections, np.random.default_rng(seed * 1000 + k)))
    return rigs


def play(runtime, rig: NumpyRig, cents, voice, start_position: float | None = None, keep_song_memory=False,
         max_cents: float | None = None) -> dict:
    """One take as the server plays it: homing at full reverse, then command / step / observe per 10 ms."""
    cents = np.asarray(cents, np.float64); voice = np.asarray(voice, bool)
    rig.reset(rig.rng.uniform(0, .8) if start_position is None else start_position)
    if not getattr(runtime, "self_home", False):
        for _ in range(rig.imp.homing_steps):
            rig.step(-1.0, False)
    if max_cents is None:
        max_cents = cents[voice].max() + 250 if voice.any() else np.inf
    runtime.prepare(cents, voice, keep_song_memory=keep_song_memory)
    out = {k: [] for k in ("emitted", "sounding", "heard", "valid", "pwm", "valve")}
    for t in range(len(cents)):
        pwm, valve = runtime.command(t)
        emitted, sounding, heard, valid = rig.step(pwm, valve)
        ok = valid and heard <= max_cents
        runtime.observe(heard if ok else 0.0, ok)
        for k, v in zip(out, (emitted, sounding, heard if ok else 0.0, ok, pwm, bool(valve))):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}
