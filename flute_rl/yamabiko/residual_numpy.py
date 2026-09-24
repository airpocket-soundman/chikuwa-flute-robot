"""NumPy runtime of the twin skeleton + neural residual performer (UNO Q).

The UNO Q has no PyTorch.  This module replays :class:`ResidualTwinPerformer`
with plain NumPy: the closed-tube motor model of the twin, the pitch
observer and PD skeleton, the anticipation planner, the residual network and
the time-indexed song memory.  ``export`` writes everything a rig needs into
one ``.npz`` (network weights, the fitted twin, the controller constants);
``Runtime`` plays a target track one 10 ms step at a time:

    runtime.prepare(cents, voice)          # once per song (keeps the song memory on a repeat)
    pwm, valve = runtime.command(t)        # before the step
    runtime.observe(heard, valid)          # after the step, with what was heard

It matches the PyTorch performer to floating-point precision (see the test).
"""
from __future__ import annotations

import math

import numpy as np

from .adaptive_memory import SONG_LOOKAHEAD
from .rig_adaptive import PITCH_CENTER, PITCH_SCALE, WINDOW

A4 = 440.0
SPAN = 6


def _linear(name, module):
    return {f"{name}.w": module.weight.detach().cpu().numpy().astype(np.float32),
            f"{name}.b": module.bias.detach().cpu().numpy().astype(np.float32)}


def export(model, twin, row: int = 0):
    """Weights + one rig's twin + constants of ``model`` as a dict of arrays."""
    plant, det = model.plant, model.skeleton
    cfg, dcfg, pcfg = model.config, det.config, plant.config
    arrays = {}
    for index, layer in enumerate(model.planner):
        if hasattr(layer, "weight"):
            arrays.update(_linear(f"planner.{index}", layer))
    arrays.update({"cell.w_ih": model.cell.weight_ih.detach().cpu().numpy().astype(np.float32),
                   "cell.w_hh": model.cell.weight_hh.detach().cpu().numpy().astype(np.float32),
                   "cell.b_ih": model.cell.bias_ih.detach().cpu().numpy().astype(np.float32),
                   "cell.b_hh": model.cell.bias_hh.detach().cpu().numpy().astype(np.float32)})
    for index, layer in enumerate(model.head):
        if hasattr(layer, "weight"):
            arrays.update(_linear(f"head.{index}", layer))
    for index, layer in enumerate(model.writer.candidate):
        if hasattr(layer, "weight"):
            arrays.update(_linear(f"writer.candidate.{index}", layer))
    arrays.update(_linear("writer.gate", model.writer.gate))
    twin_values = {}
    for name, value in vars(twin).items():
        if value is not None:
            twin_values[name] = float(value[row])
    arrays["twin"] = np.array([twin_values[name] for name in TWIN_FIELDS], np.float64)
    arrays["constants"] = np.array([
        cfg.hidden, cfg.song_slot, cfg.song_rate, cfg.max_aim_offset, cfg.max_pwm_offset,
        dcfg.kp, dcfg.kd, dcfg.observer_gain, dcfg.observer_velocity_gain, dcfg.anticipation_speed,
        dcfg.anticipation_lag_steps, dcfg.hold_fraction, dcfg.note_threshold_cents, dcfg.homing_steps,
        pcfg.dt, pcfg.stroke_m, pcfg.temp_c, pcfg.effective_length_m, pcfg.min_length_m, pcfg.deadband,
        pcfg.max_velocity_strokes_s, 1.0 if pcfg.flute_model == "closed_tube" else 0.0,
        pcfg.low_cents, pcfg.pitch_span_cents], np.float64)
    return arrays


TWIN_FIELDS = ("torque_gain", "torque_tau_s", "inertia", "coulomb_friction", "viscous_friction",
               "max_velocity_strokes_s", "flute_offset_cents", "flute_scale", "tube_offset_m", "temp_offset_c",
               "deadband", "hearing_delay_steps")
CONSTANTS = ("hidden", "song_slot", "song_rate", "max_aim_offset", "max_pwm_offset", "kp", "kd", "observer_gain",
             "observer_velocity_gain", "anticipation_speed", "anticipation_lag_steps", "hold_fraction",
             "note_threshold_cents", "homing_steps", "dt", "stroke_m", "temp_c", "effective_length_m",
             "min_length_m", "deadband", "max_velocity_strokes_s", "closed_tube", "low_cents", "pitch_span_cents")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def period_ms(cents):
    return 1000.0 / (A4 * 2.0 ** (cents / 1200.0))


class Runtime:
    def __init__(self, arrays):
        self.a = {k: np.asarray(v) for k, v in arrays.items()}
        self.c = dict(zip(CONSTANTS, self.a["constants"].tolist()))
        self.t = dict(zip(TWIN_FIELDS, self.a["twin"].tolist()))
        c, t = self.c, self.t
        self.hidden, self.slot = int(c["hidden"]), int(c["song_slot"])
        self.delay = int(min(max(t["hearing_delay_steps"], 0), SPAN - 1))
        self.heard_index = SPAN - 1 - self.delay
        self.band = min(t["deadband"] + .03, .6)
        # Flute coefficients of the twin: T(x) = a - b x in ms.
        a0 = period_ms(self.cents_at(0.0)); self.coef_a = a0; self.coef_b = max(a0 - period_ms(self.cents_at(1.0)), .2)
        self.ratio = t["max_velocity_strokes_s"] / c["max_velocity_strokes_s"]
        self.song = None

    # ----- twin physics ------------------------------------------------------
    def cents_at(self, position):
        c, t = self.c, self.t
        if c["closed_tube"] < .5:
            return c["low_cents"] + c["pitch_span_cents"] * position * t["flute_scale"] + t["flute_offset_cents"]
        speed = 331.3 + .606 * (c["temp_c"] + t["temp_offset_c"])
        length = max(c["effective_length_m"] + t["tube_offset_m"] - position * c["stroke_m"], c["min_length_m"])
        return 1200.0 * math.log2(speed / (4.0 * length) / A4) + t["flute_offset_cents"]

    def model_step(self, pwm):
        """One step of the twin's motor model (the same equations as the simulator)."""
        c, t = self.c, self.t
        pwm = min(max(pwm, -1.0), 1.0)
        if c["deadband"] > 0:
            band = t["deadband"]
            pwm = math.copysign(max(abs(pwm) - band, 0.0) / (1.0 - band), pwm)
        alpha = min(c["dt"] / t["torque_tau_s"], 1.0)
        self.torque += alpha * (t["torque_gain"] * pwm - self.torque)
        friction = t["coulomb_friction"] * math.tanh(self.velocity / .015) + t["viscous_friction"] * self.velocity
        acceleration = (self.torque - friction) / t["inertia"]
        vmax = t["max_velocity_strokes_s"]
        self.velocity = min(max(self.velocity + c["dt"] * acceleration, -vmax), vmax)
        raw = self.position + c["dt"] * self.velocity
        self.position = min(max(raw, 0.0), 1.0)
        if (raw <= 0.0 and self.velocity < 0.0) or (raw >= 1.0 and self.velocity > 0.0):
            self.velocity = 0.0

    # ----- planning ----------------------------------------------------------
    def anticipate(self, aim, voice):
        c = self.c
        x = np.clip((self.coef_a - period_ms(aim)) / self.coef_b, 0.0, 1.0)
        result = aim.copy()
        starts, anchor = [0], float(aim[0])
        for i in range(1, len(aim)):
            if abs(float(aim[i]) - anchor) > c["note_threshold_cents"]:
                starts.append(i); anchor = float(aim[i])
        bounds = starts + [len(aim)]
        medians = [float(np.median(aim[bounds[k]:bounds[k + 1]])) for k in range(len(starts))]
        positions = [min(max((self.coef_a - period_ms(m)) / self.coef_b, 0.0), 1.0) for m in medians]
        steps_per_stroke = 1.0 / (c["anticipation_speed"] * self.ratio * c["dt"])
        for k in range(1, len(starts)):
            onset = starts[k]
            lead = int(round(c["anticipation_lag_steps"] + abs(positions[k] - positions[k - 1]) * steps_per_stroke))
            earliest = starts[k - 1] + int(c["hold_fraction"] * (onset - starts[k - 1]))
            result[max(earliest, onset - lead):onset] = medians[k]
        return result

    def prepare(self, cents, voice, keep_song_memory=True):
        """Set the target track (cents (T,), voice bool (T,)).  Call before each play."""
        self.cents = np.asarray(cents, np.float64); self.voice = np.asarray(voice, bool)
        steps = len(self.cents)
        upcoming, last = self.cents.copy(), float(self.cents[-1])
        for i in range(steps - 1, -1, -1):
            last = float(self.cents[i]) if self.voice[i] else last
            upcoming[i] = last
        self.aim = self.anticipate(upcoming, self.voice)
        if not (keep_song_memory and self.song is not None and self.song.shape[0] == steps):
            self.song = np.zeros((steps, self.slot), np.float32)
        # homing leaves the model pressed against the home stop
        self.position, self.velocity, self.torque = 0.0, 0.0, -self.t["torque_gain"]
        self.fast = np.zeros(self.hidden, np.float32)
        self.heard, self.valid, self.pwm = 0.0, False, 0.0
        self.x_history = np.zeros(SPAN); self.voice_history = np.zeros(SPAN, bool)
        self.step_index = 0

    # ----- one control step --------------------------------------------------
    def _window(self, t):
        steps = len(self.cents)
        ids = [min(t + k, steps - 1) for k in WINDOW]
        return np.concatenate([(self.cents[ids] - PITCH_CENTER) / PITCH_SCALE, self.voice[ids].astype(np.float64)])

    def _read(self, t):
        steps = len(self.cents)
        return np.concatenate([self.song[min(t + k, steps - 1)] for k in SONG_LOOKAHEAD]).astype(np.float64)

    def _mlp(self, name, x, layers):
        for index in layers[:-1]:
            x = silu(self.a[f"{name}.{index}.w"] @ x + self.a[f"{name}.{index}.b"])
        index = layers[-1]
        return self.a[f"{name}.{index}.w"] @ x + self.a[f"{name}.{index}.b"]

    def command(self, t):
        c = self.c
        read = self._read(t)
        base_aim = min(max((self.coef_a - period_ms(self.aim[t])) / self.coef_b, 0.0), 1.0)
        planner_in = np.concatenate([self._window(t), read, [base_aim, self.position]])
        x_cmd = min(max(base_aim + c["max_aim_offset"] * math.tanh(self._mlp("planner", planner_in, (0, 2, 4))[0]), 0.0), 1.0)
        u = min(max(c["kp"] * (x_cmd - self.position) - c["kd"] * self.velocity, -1.0), 1.0)
        pwm_det = math.copysign(self.band + (1 - self.band) * abs(u), u) if abs(u) > .01 else 0.0
        heard_n = (self.heard - PITCH_CENTER) / PITCH_SCALE if self.valid else 0.0
        target_n = (self.cents[t] - PITCH_CENTER) / PITCH_SCALE
        features = np.array([x_cmd, self.position, self.velocity, heard_n, float(self.valid),
                             (target_n - heard_n) if self.valid else 0.0, float(self.voice[t]), pwm_det, base_aim])
        x = np.concatenate([features, read]).astype(np.float32)
        gi = self.a["cell.w_ih"] @ x + self.a["cell.b_ih"]; gh = self.a["cell.w_hh"] @ self.fast + self.a["cell.b_hh"]
        h = self.hidden
        r = sigmoid(gi[:h] + gh[:h]); z = sigmoid(gi[h:2 * h] + gh[h:2 * h])
        n = np.tanh(gi[2 * h:] + r * gh[2 * h:])
        self.fast = ((1 - z) * n + z * self.fast).astype(np.float32)
        pwm = min(max(pwm_det + c["max_pwm_offset"] * math.tanh(self._mlp("head", self.fast, (0, 2))[0]), -1.0), 1.0)
        self._x_cmd, self.pwm, self._t = x_cmd, pwm, t
        self.model_step(pwm)
        return pwm, bool(self.voice[t])

    def observe(self, heard, valid):
        """What the rig heard after the command of the same step."""
        c, t = self.c, self._t
        self.heard, self.valid = float(heard) if valid else 0.0, bool(valid)
        self.x_history = np.concatenate([self.x_history[1:], [self.position]])
        self.voice_history = np.concatenate([self.voice_history[1:], [self.voice[t]]])
        usable = self.valid and bool(self.voice_history[self.heard_index])
        innovation = ((self.coef_a - period_ms(self.heard)) / self.coef_b - self.x_history[self.heard_index]) if usable else 0.0
        shift = c["observer_gain"] * innovation
        self.position = min(max(self.position + shift, 0.0), 1.0)
        self.velocity += c["observer_velocity_gain"] * innovation / c["dt"]
        self.x_history = np.clip(self.x_history + shift, 0.0, 1.0)
        then = max(0, t - 2)
        heard_n = (self.heard - PITCH_CENTER) / PITCH_SCALE if self.valid else 0.0
        features = np.array([heard_n, float(self.valid), (self.cents[then] - PITCH_CENTER) / PITCH_SCALE,
                             float(self.voice[then]), self._x_cmd, self.pwm], np.float32)
        x = np.concatenate([features, self.fast])
        proposal = np.tanh(self._mlp("writer.candidate", np.concatenate([x, self.song[then]]), (0, 2)))
        rate = c["song_rate"] * sigmoid(self.a["writer.gate.w"] @ x + self.a["writer.gate.b"])[0]
        self.song[then] = self.song[then] + rate * (proposal - self.song[then])
