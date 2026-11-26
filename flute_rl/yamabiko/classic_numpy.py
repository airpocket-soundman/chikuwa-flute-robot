"""A classic deterministic performer (no network anywhere), to compare the learned steps against.

Same calls as the other runtimes (residual_numpy.Runtime / NNRuntime), so the server, the step evaluation and
the NumPy simulator run it unchanged:

    runtime.prepare(cents, voice)          # once per song
    pwm, valve = runtime.command(t)        # before the step
    runtime.observe(heard, valid)          # after the step, with what was heard

Every 10 ms:

1. **Plan** (once per song): the score with rests filled by the next note, and each note moved earlier by
   the time the plunger needs to get there (``Runtime.anticipate``: lag + distance / speed).
2. **Where to be**: the planned pitch turned into a stroke position by the twin's closed-tube formula, plus
   a slow *pitch trim* (an outer integral loop on the heard pitch error that absorbs the twin's offset).
3. **Where it is**: no encoder.  The twin's motor model is driven by the commands (dead reckoning) and a
   pitch observer moves the estimate toward the position the heard pitch implies (the heard pitch is a
   few steps late, so it is compared with the estimate of that many steps ago).
4. **PID** on position: ``u = kp e + ki ∫e - kd v``, the integral only near the target and clamped
   (anti-windup); the output jumps the static-friction deadband (``band + (1 - band) |u|``) and a hold zone
   with hysteresis stops the plunger dithering around a note.
5. **Valve**: open ``valve_lead`` steps before a note (the servo's delay), shut for rests; at a note's start
   it waits (at most ``valve_wait_steps``) until the plunger is within ``valve_gate`` of the note, so the
   glide between notes is not heard.
6. **Practice (iterative learning control)**, with ``ilc_gain`` > 0: every play records the heard pitch
   error of each instant (as position, put back to the instant it was emitted: the hearing delay); the next
   play of the same song (``prepare(keep_song_memory=True)``) adds ``ilc_gain`` x that error, smoothed and
   taken ``ilc_shift`` steps early (the plunger's response), to the position target of each instant.  The
   corrections add up play after play (clamped to ``ilc_max``), as a player who practises a passage.

``export(twin_runtime_arrays, **gains)`` makes the ``.npz``: the twin and controller constants of a fitted twin
runtime (``twin_*_runtime.npz``) plus the gains.  ``residual_numpy.runtime_for`` picks this class for it.
"""
from __future__ import annotations

import math

import numpy as np

from .residual_numpy import A4, SPAN, Runtime, period_ms

CLASSIC_FIELDS = ("kp", "ki", "kd", "i_zone", "i_max", "hold_tol", "hold_release", "hold_speed",
                  "pitch_ki", "trim_max", "trim_gate_cents", "settle_steps",
                  "valve_lead", "valve_gate", "valve_wait_steps", "ilc_gain", "ilc_shift", "ilc_smooth", "ilc_max", "obs_gate_cents")

DEFAULT_GAINS = dict(       # tuned on the NumPy simulator around twin_y1f80_04 (scripts/yamabiko_classic_export.py)
    kp=20.0,                # PWM per stroke of position error
    ki=2.6,                 # PWM per stroke-second
    kd=.72,                 # PWM per stroke/s of estimated velocity
    i_zone=.06,             # integrate only within this many strokes of the target
    i_max=.25,              # |ki ∫e| at most this (PWM)
    hold_tol=.004,          # enter the hold zone within this many strokes (about 5 cent) ...
    hold_release=.04,       # ... and leave it beyond this many
    hold_speed=.05,         # and only when the estimate moves slower than this (strokes/s)
    pitch_ki=.0016,         # share of the heard pitch error (as position) added to the trim per step
    trim_max=.10,           # |trim| at most this many strokes
    trim_gate_cents=150.0,  # ignore heard errors larger than this (a wrong octave, a glide)
    settle_steps=25,        # trim only after the note has sounded this many steps
    valve_lead=3,           # valve commands go this many steps ahead (servo + duct)
    valve_gate=.098,        # open at a note's start only within this many strokes of it ...
    valve_wait_steps=25,    # ... or after this many steps at the latest
    ilc_gain=0.0,           # practice: share of the last play's error added to the target (0: no practice)
    ilc_shift=8.0,          # ... taken this many steps early
    ilc_smooth=9.0,         # ... smoothed over this many steps
    ilc_max=.08,            # ... the correction at most this many strokes
    obs_gate_cents=1e4,     # the observer ignores a heard pitch this far from the estimate's (a wrong octave)
)


def export(twin_arrays, **gains):
    """The ``.npz`` arrays of a classic runtime: twin and constants of a twin runtime, plus the gains."""
    unknown = set(gains) - set(CLASSIC_FIELDS)
    if unknown:
        raise ValueError(f"unknown gains {sorted(unknown)}")
    values = {**DEFAULT_GAINS, **gains}
    return {"twin": np.asarray(twin_arrays["twin"]), "constants": np.asarray(twin_arrays["constants"]),
            "classic.gains": np.array([values[k] for k in CLASSIC_FIELDS], np.float64)}


class ClassicRuntime(Runtime):
    """Feedforward plan + pitch observer + PID on position + rule valve, on the twin's equations only."""

    def __init__(self, arrays):
        super().__init__(arrays)
        values = self.a["classic.gains"].tolist()          # files from before the practice fields: no practice
        self.g = {**DEFAULT_GAINS, **dict(zip(CLASSIC_FIELDS, values)), **({} if len(values) >= 19 else {"ilc_gain": 0.0})}
        self.ilc, self.rec = None, None
        self.self_home = False

    def position_for(self, cents):
        """Inverse of cents_at for the twin, clamped to the stroke."""
        c, t = self.c, self.t
        if c["closed_tube"] < .5:
            return min(max((cents - t["flute_offset_cents"] - c["low_cents"]) / (c["pitch_span_cents"] * t["flute_scale"]),
                           0.0), 1.0)
        speed = 331.3 + .606 * (c["temp_c"] + t["temp_offset_c"])
        length = speed / (4.0 * A4 * 2.0 ** ((cents - t["flute_offset_cents"]) / 1200.0))
        return min(max((c["effective_length_m"] + t["tube_offset_m"] - length) / c["stroke_m"], 0.0), 1.0)

    def prepare(self, cents, voice, keep_song_memory=True):
        self.cents = np.asarray(cents, np.float64); self.voice = np.asarray(voice, bool)
        steps = len(self.cents)
        upcoming, last = self.cents.copy(), float(self.cents[-1])
        for i in range(steps - 1, -1, -1):              # rests take the next note: move there early
            last = float(self.cents[i]) if self.voice[i] else last
            upcoming[i] = last
        self.aim = self.anticipate(upcoming, self.voice)
        self.aim_x = np.array([self.position_for(a) for a in self.aim])
        # note starts and how long each step's note has sounded
        onset = np.zeros(steps, bool)
        for i in range(steps):
            onset[i] = self.voice[i] and (i == 0 or not self.voice[i - 1] or abs(self.cents[i] - self.cents[i - 1]) > 1)
        self.onset = onset
        self.position, self.velocity, self.torque = 0.0, 0.0, -self.t["torque_gain"]    # homing: pressed home
        self.heard, self.valid, self.pwm = 0.0, False, 0.0
        self.x_history = np.zeros(SPAN); self.voice_history = np.zeros(SPAN, bool)
        self.integral, self.holding = 0.0, False
        if not keep_song_memory:
            self.trim = 0.0
        else:
            self.trim = getattr(self, "trim", 0.0)
        self.sounded = 0                                 # steps the current note has been heard
        self.valve_open, self.waiting = False, 0
        # practice: the last play of this song corrects this one
        g = self.g
        same = keep_song_memory and self.rec is not None and len(self.rec[0]) == steps
        if not same or self.ilc is None or len(self.ilc) != steps:
            self.ilc = np.zeros(steps)
        if same and g["ilc_gain"] > 0:
            err, ok = self.rec
            k = max(1, int(g["ilc_smooth"]))
            num = np.convolve(np.where(ok, err, 0.0), np.ones(k) / k, "same")
            den = np.convolve(ok.astype(float), np.ones(k) / k, "same")
            smooth = np.where(den > .2, num / np.maximum(den, 1e-6), 0.0)
            shift = int(g["ilc_shift"])
            early = np.r_[smooth[shift:], np.zeros(shift)] if shift > 0 else smooth
            self.ilc = np.clip(self.ilc + g["ilc_gain"] * early, -g["ilc_max"], g["ilc_max"])
        self.rec = (np.zeros(steps), np.zeros(steps, bool))

    def _band(self):
        band = self.band
        ratio = self.c["kinetic_deadband_ratio"]
        if ratio != 1.0:
            band *= ratio + (1.0 - ratio) * math.exp(-(self.velocity / .02) ** 2)
        return band

    def command(self, t):
        g, dt = self.g, self.c["dt"]
        x_ref = min(max(self.aim_x[t] + self.trim + self.ilc[t], 0.0), 1.0)
        e = x_ref - self.position
        if self.holding and abs(e) > g["hold_release"]:
            self.holding = False
        elif not self.holding and abs(e) < g["hold_tol"] and abs(self.velocity) < g["hold_speed"]:
            self.holding, self.integral = True, 0.0
        if self.holding:
            pwm = 0.0
        else:
            if abs(e) < g["i_zone"]:
                self.integral += e * dt
            else:
                self.integral = 0.0
            if g["ki"] > 0:
                limit = g["i_max"] / g["ki"]
                self.integral = min(max(self.integral, -limit), limit)
            u = g["kp"] * e + g["ki"] * self.integral - g["kd"] * self.velocity
            u = min(max(u, -1.0), 1.0)
            band = self._band()
            pwm = math.copysign(band + (1.0 - band) * abs(u), u) if abs(u) > 1e-3 else 0.0
        self.pwm, self._t, self._x_ref = pwm, t, x_ref
        self.model_step(pwm)
        return pwm, self._valve(t)

    def _valve(self, t):
        g, steps = self.g, len(self.cents)
        ahead = min(t + int(g["valve_lead"]), steps - 1)
        if not self.voice[ahead]:
            self.valve_open, self.waiting = False, 0
            return False
        if self.onset[ahead] or not self.valve_open:     # a note is about to start: wait for the plunger
            near = abs(self.aim_x[ahead] + self.trim - self.position) <= g["valve_gate"]
            if near or self.waiting >= g["valve_wait_steps"]:
                self.valve_open, self.waiting = True, 0
            else:
                self.valve_open = False
                self.waiting += 1
        return self.valve_open

    def observe(self, heard, valid, *rest):
        c, g, t = self.c, self.g, self._t
        self.heard, self.valid = float(heard) if valid else 0.0, bool(valid)
        self.x_history = np.concatenate([self.x_history[1:], [self.position]])
        self.voice_history = np.concatenate([self.voice_history[1:], [self.voice[t]]])
        usable = self.valid and bool(self.voice_history[self.heard_index])
        if usable and g["obs_gate_cents"] < 1e3:          # a heard pitch far from where it should be: not used
            usable = abs(self.heard - self.cents_at(self.x_history[self.heard_index])) < g["obs_gate_cents"]
        if usable:                                        # pitch observer: correct the dead reckoning
            innovation = (self.coef_a - period_ms(self.heard)) / self.coef_b - self.x_history[self.heard_index]
            shift = c["observer_gain"] * innovation
            self.position = min(max(self.position + shift, 0.0), 1.0)
            self.velocity += c["observer_velocity_gain"] * innovation / c["dt"]
            self.x_history = np.clip(self.x_history + shift, 0.0, 1.0)
        # outer loop: trim the position target by the heard pitch error once a note has settled
        if self.onset[t]:
            self.sounded = 0
        if self.voice[t] and self.valid:
            self.sounded += 1
        error = self.cents[t] - self.heard
        then = t - (SPAN - 1 - self.heard_index)          # the instant the heard pitch was emitted
        if self.valid and 0 <= then < len(self.cents) and self.voice[then] and abs(self.cents[then] - self.heard) < 300:
            slope = self.position_for(self.cents[then] + 10.0) - self.position_for(self.cents[then] - 10.0)
            self.rec[0][then] = (self.cents[then] - self.heard) * slope / 20.0
            self.rec[1][then] = True
        if (self.valid and self.voice[t] and self.sounded >= g["settle_steps"]
                and abs(error) < g["trim_gate_cents"]):
            slope = self.position_for(self.cents[t] + 10.0) - self.position_for(self.cents[t] - 10.0)
            self.trim += g["pitch_ki"] * error * slope / 20.0
            self.trim = min(max(self.trim, -g["trim_max"]), g["trim_max"])
