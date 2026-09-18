"""Batched simulator of the Yamabiko No.1 rig (see docs/yamabiko.md).

Many rigs are stepped at once (arrays of shape (n,)), numpy only.

* Flute: slide-whistle type. A fipple mouthpiece on a tube closed by the
  plunger, so f = c / (4 L) with L = tube_len + end_corr - x. A closed tube
  has odd harmonics only: when it overblows it jumps to the 3rd mode, a
  twelfth up (+1902 cents), not an octave.
* Air: a fan runs at a constant speed for the whole session and a 2-way
  valve shuts the air off. The flute draws a few percent of the fan's free
  flow, so the fan works close to shut-off either way and the blowing
  pressure barely changes. After the valve opens, the air arrives
  `valve_delay` steps later, the tone needs `onset_s` to build up and starts
  `onset_cents` off pitch, settling with `onset_tau`. Closing the valve stops
  the tone after the same valve delay. While the valve is shut the chamber
  before it charges up to the shut-off pressure (time constant `surge_tau`),
  and after it opens the excess drains out through the flute with the same
  time constant: the pitch is `surge_cents` sharp times the charge, so a
  note after a long rest starts sharper than one after a short rest.
* Plunger: DC linear actuator driven by PWM with no position sensor (same
  model as flute_rl.sim: command delay, dead band, velocity lag, end stops,
  backlash, and the optional "harsh" effects).
* Listening: pitch estimate noise, drop-outs, octave mistakes of the
  estimator (+-1200, a different thing from overblowing), heard
  `obs_delay` steps late.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..sim import DT, hz_to_cents

OVERBLOW_CENTS = 1200.0 * np.log2(3.0)  # third mode of a closed tube: a twelfth
QUEUE = 8  # longest delay line [steps]


def speed_of_sound(temp_c):
    return 331.3 + 0.606 * np.asarray(temp_c, dtype=float)


NOMINAL = dict(
    tube_len=0.150, end_corr=0.005, temp_c=25.0, press_cents=0.0, overblow_len=0.050, pitch_jitter=2.0,
    valve_delay=1, onset_s=0.03, onset_cents=-20.0, onset_tau=0.03, surge_cents=10.0, surge_tau=0.05,
    stroke=0.105, v_in=0.150, v_out=0.150, deadband=0.20, tau_v=0.030, cmd_delay=1, backlash=0.0003,
    stiction=0.0, pwm_curve=1.0, load_slope=0.0, speed_drift=0.0, motion_noise=0.0, delay_jitter=0.0,
    pitch_noise=3.0, dropout=0.02, octave_err=0.0, obs_delay=3,
)
INTS = ("valve_delay", "cmd_delay", "obs_delay")


@dataclass
class RigParams:
    """One field per rig property; every field is an array of shape (n,)."""

    # flute
    tube_len: np.ndarray      # acoustic length with the plunger at home [m]
    end_corr: np.ndarray      # open-end correction [m]
    temp_c: np.ndarray
    press_cents: np.ndarray   # pitch offset from this rig's blowing pressure [cents]
    overblow_len: np.ndarray  # effective lengths shorter than this overblow at this pressure [m]
    pitch_jitter: np.ndarray  # slow pitch wobble [cents std]
    # air / valve
    valve_delay: np.ndarray   # steps from the valve command to the air arriving (int)
    onset_s: np.ndarray       # the tone needs this long to build up [s]
    onset_cents: np.ndarray   # the tone starts this far off pitch ...
    onset_tau: np.ndarray     # ... and settles with this time constant [s]
    surge_cents: np.ndarray   # pitch offset from the pressure stored while the valve was shut, fully charged [cents]
    surge_tau: np.ndarray     # the chamber before the valve charges and drains with this time constant [s]
    # plunger
    stroke: np.ndarray
    v_in: np.ndarray          # speed at pwm = +1 (tube shorter, pitch up) [m/s]
    v_out: np.ndarray
    deadband: np.ndarray
    tau_v: np.ndarray
    cmd_delay: np.ndarray     # int
    backlash: np.ndarray
    stiction: np.ndarray
    pwm_curve: np.ndarray
    load_slope: np.ndarray
    speed_drift: np.ndarray
    motion_noise: np.ndarray
    delay_jitter: np.ndarray
    # listening
    pitch_noise: np.ndarray
    dropout: np.ndarray
    octave_err: np.ndarray
    obs_delay: np.ndarray     # int

    @property
    def n(self) -> int:
        return len(self.tube_len)

    @classmethod
    def nominal(cls, n: int = 1, **override) -> "RigParams":
        vals = {**NOMINAL, **override}
        return cls(**{k: np.full(n, v, dtype=int if k in INTS else float) for k, v in vals.items()})

    @classmethod
    def sample(cls, rng: np.random.Generator, n: int, spread: float = 1.0, harsh: float = 0.0) -> "RigParams":
        """Rig-to-rig variation around the nominal values (all ranges are guesses until measured).

        `harsh` in [0, 1] scales effects the hand-written controllers do not model. They are
        drawn after the basic ones, so a seed gives the same basic rigs at any harshness."""
        N = NOMINAL

        def u(key, half):
            return N[key] + spread * rng.uniform(-half, half, n)

        def ui(key, half, lo=0):
            return np.clip(np.round(u(key, half)), lo, QUEUE - 1).astype(int)

        p = cls(
            tube_len=u("tube_len", 0.008), end_corr=np.maximum(0.0, u("end_corr", 0.002)), temp_c=u("temp_c", 8.0),
            press_cents=u("press_cents", 10.0), overblow_len=u("overblow_len", 0.004),
            pitch_jitter=np.maximum(0.0, u("pitch_jitter", 1.0)),
            valve_delay=ui("valve_delay", 1.0), onset_s=np.maximum(DT, u("onset_s", 0.015)),
            onset_cents=u("onset_cents", 15.0), onset_tau=np.maximum(0.005, u("onset_tau", 0.015)),
            surge_cents=np.zeros(n), surge_tau=np.full(n, N["surge_tau"]),
            stroke=np.full(n, N["stroke"]), v_in=u("v_in", 0.030), v_out=u("v_out", 0.030),
            deadband=u("deadband", 0.08), tau_v=np.maximum(0.005, u("tau_v", 0.015)), cmd_delay=ui("cmd_delay", 1.0),
            backlash=np.maximum(0.0, u("backlash", 0.0002)),
            stiction=np.zeros(n), pwm_curve=np.ones(n), load_slope=np.zeros(n), speed_drift=np.zeros(n),
            motion_noise=np.zeros(n), delay_jitter=np.zeros(n),
            pitch_noise=np.maximum(0.0, u("pitch_noise", 2.0)), dropout=np.maximum(0.0, u("dropout", 0.02)),
            octave_err=np.full(n, 0.005 * spread), obs_delay=ui("obs_delay", 2.0, lo=1),
        )
        if harsh > 0.0:
            h = float(harsh)

            def g(lo, hi):
                return h * rng.uniform(lo, hi, n)

            p.backlash = p.backlash + g(0.0, 0.0007)
            p.stiction = g(0.0, 0.15)
            p.pwm_curve = 1.0 + g(-0.3, 0.5)
            p.load_slope = g(0.0, 0.3)
            p.speed_drift = g(0.0, 0.05)
            p.motion_noise = g(0.0, 0.1)
            p.delay_jitter = g(0.0, 0.15)
            p.octave_err = p.octave_err + g(0.0, 0.03)
        # drawn last, so the other values of a seed are the same as before the 2-way valve
        p.surge_cents = u("surge_cents", 8.0)
        p.surge_tau = np.maximum(2.0 * DT, u("surge_tau", 0.03))
        return p

    def tile(self, reps: int) -> "RigParams":
        """The same rigs repeated `reps` times (population x rigs, for common random numbers)."""
        return RigParams(**{f.name: np.tile(getattr(self, f.name), reps) for f in dataclasses.fields(self)})

    def replace_where(self, mask: np.ndarray, other: "RigParams") -> "RigParams":
        return RigParams(**{f.name: np.where(mask, getattr(other, f.name), getattr(self, f.name))
                            for f in dataclasses.fields(self)})

    def cents_at(self, x) -> np.ndarray:
        """Steady pitch at plunger position x (fundamental, no transient) [cents]."""
        length = np.maximum(self.tube_len + self.end_corr - x, 0.02)
        return hz_to_cents(speed_of_sound(self.temp_c) / (4.0 * length)) + self.press_cents

    def x_for_cents(self, cents) -> np.ndarray:
        f = 440.0 * 2.0 ** ((np.asarray(cents, dtype=float) - self.press_cents) / 1200.0)
        return self.tube_len + self.end_corr - speed_of_sound(self.temp_c) / (4.0 * f)


class Rig:
    """`n` rigs stepped together. step(pwm, valve) takes arrays of shape (n,)."""

    def __init__(self, params: RigParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        self.reset()

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Power on (all rigs, or only where `mask`): plunger at home, valve shut, chamber not charged, no sound."""
        n = self.p.n
        if mask is None:
            self.t = 0
            self.x_motor, self.x, self.v = np.zeros(n), np.zeros(n), np.zeros(n)
            self.pwm_q = np.zeros((n, QUEUE))
            self.valve_q = np.zeros((n, QUEUE), bool)
            self.meas_q = np.full((n, QUEUE), np.nan)
            self.u_last, self.speed, self.jit = np.zeros(n), np.ones(n), np.zeros(n)
            self.air_steps, self.over = np.zeros(n, int), np.zeros(n, bool)
            self.charge = np.zeros(n)
            return
        m = np.asarray(mask, bool)
        for a in ("x_motor", "x", "v", "u_last", "jit", "pwm_q", "charge"):
            getattr(self, a)[m] = 0.0
        self.speed[m] = 1.0
        self.air_steps[m] = 0
        self.over[m] = False
        self.valve_q[m] = False
        self.meas_q[m] = np.nan

    def swap(self, mask: np.ndarray, new: RigParams) -> None:
        """Put a different flute/actuator in the rigs where `mask` (power cycle of those rigs)."""
        self.p = self.p.replace_where(mask, new)
        self.reset(mask)

    def step(self, pwm: np.ndarray, valve: np.ndarray) -> dict:
        p, r, n = self.p, self.rng, self.p.n
        rows = np.arange(n)

        # plunger: delayed PWM -> dead band / stiction -> nonlinear speed -> velocity lag -> end stops -> backlash
        self.pwm_q[:, 1:] = self.pwm_q[:, :-1].copy()
        self.pwm_q[:, 0] = np.clip(pwm, -1.0, 1.0)
        u = self.pwm_q[rows, p.cmd_delay]
        late = r.random(n) < p.delay_jitter
        u = np.where(late, self.u_last, u)
        self.u_last = u
        mag = np.abs(u)
        threshold = p.deadband + np.where(np.abs(self.v) < 0.005, p.stiction, 0.0)
        drive = np.where(mag < threshold, 0.0, np.sign(u) * (mag - p.deadband) / (1.0 - p.deadband))
        drive = np.sign(drive) * np.abs(drive) ** p.pwm_curve
        v_max = np.where(drive > 0, p.v_in * (1.0 - p.load_slope * np.clip(self.x_motor / p.stroke, 0.0, 1.0)), p.v_out)
        self.speed = np.clip(self.speed + r.standard_normal(n) * p.speed_drift * np.sqrt(DT), 0.8, 1.2)
        v_cmd = drive * v_max * self.speed
        self.v = self.v + (v_cmd - self.v) * np.minimum(1.0, DT / p.tau_v)
        moved = self.v * (1.0 + r.standard_normal(n) * p.motion_noise)
        self.x_motor = self.x_motor + moved * DT
        lo, hi = self.x_motor <= 0.0, self.x_motor >= p.stroke
        self.x_motor = np.clip(self.x_motor, 0.0, p.stroke)
        self.v = np.where(lo, np.maximum(self.v, 0.0), np.where(hi, np.minimum(self.v, 0.0), self.v))
        half = p.backlash / 2.0
        self.x = np.where(self.x_motor - self.x > half, self.x_motor - half,
                          np.where(self.x - self.x_motor > half, self.x_motor + half, self.x))

        # air: the valve command reaches the flute valve_delay steps later
        self.valve_q[:, 1:] = self.valve_q[:, :-1].copy()
        self.valve_q[:, 0] = valve
        flow = self.valve_q[rows, p.valve_delay]
        self.air_steps = np.where(flow, self.air_steps + 1, 0)
        t_air = self.air_steps * DT
        sounding = flow & (t_air >= p.onset_s - 1e-9)
        # the chamber before the valve charges while it is shut and drains through the flute while it is open
        k = np.minimum(1.0, DT / p.surge_tau)
        self.charge = np.where(flow, self.charge * (1.0 - k), self.charge + (1.0 - self.charge) * k)

        length = np.maximum(p.tube_len + p.end_corr - self.x, 0.02)
        # overblowing with a little hysteresis: it starts below overblow_len and stops 2 mm above it
        self.over = sounding & np.where(self.over, length < p.overblow_len + 0.002, length < p.overblow_len)
        self.jit = self.jit - self.jit * DT / 0.05 + p.pitch_jitter * np.sqrt(2.0 * DT / 0.05) * r.standard_normal(n)
        transient = np.where(sounding, p.onset_cents * np.exp(-np.maximum(t_air - p.onset_s, 0.0) / p.onset_tau), 0.0)
        cents = (hz_to_cents(speed_of_sound(p.temp_c) / (4.0 * length)) + p.press_cents
                 + OVERBLOW_CENTS * self.over + self.jit + transient + np.where(sounding, p.surge_cents * self.charge, 0.0))

        # listening
        detected = sounding & (r.random(n) >= p.dropout)
        measured = np.where(detected, cents + p.pitch_noise * r.standard_normal(n), np.nan)
        octave = detected & (r.random(n) < p.octave_err)
        measured = measured + np.where(octave, np.where(r.random(n) < 0.5, 1200.0, -1200.0), 0.0)
        self.meas_q[:, 1:] = self.meas_q[:, :-1].copy()
        self.meas_q[:, 0] = measured
        heard = self.meas_q[rows, p.obs_delay]

        self.t += 1
        return {"x": self.x.copy(), "cents": cents, "sounding": sounding, "overblown": self.over.copy(),
                "measured": measured, "heard": heard}
