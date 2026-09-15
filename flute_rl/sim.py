"""Physics simulator for the slide-flute rig (chikuwa / 3D-printed flute).

Units are SI (m, s) and degrees for angles. Pitch is expressed in cents
relative to A4 = 440 Hz so that errors are musically meaningful.

Model summary
-------------
* Plunger: DC linear actuator on a lead screw, driven by PWM without a
  position sensor. Command delay, dead band, first-order velocity lag
  (inertia / coasting), end stops and backlash are modelled.
* Blowing angle: Feetech SCS0009 serial-bus servo. Commands are quantised to
  the 10-bit step (300 deg / 1024), rate limited and lagged; the read-back
  angle is quantised as well.
* Acoustics: tube closed by the plunger, f = c / (4 L). The blowing angle
  bends the pitch slightly and must stay inside a "sounding window" whose
  width shrinks as the tube gets shorter. Pushing the angle to the upper side
  of the window on a short tube overblows one octave.
* Measurement: pitch estimate noise and random drop-outs.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass

import numpy as np

DT = 0.01  # control period [s]
A4_HZ = 440.0


def speed_of_sound(temp_c: float) -> float:
    return 331.3 + 0.606 * temp_c


def hz_to_cents(f):
    return 1200.0 * np.log2(np.asarray(f, dtype=float) / A4_HZ)


def cents_to_hz(c):
    return A4_HZ * 2.0 ** (np.asarray(c, dtype=float) / 1200.0)


@dataclass
class FluteParams:
    # tube / environment
    tube_len: float = 0.150        # acoustic length with the plunger at home (x = 0) [m]
    end_corr: float = 0.005        # open-end correction [m]
    stroke: float = 0.120          # usable plunger travel [m]
    temp_c: float = 25.0
    # plunger (DC linear actuator, 24 V, 150 mm/s class)
    v_max_in: float = 0.150        # speed at pwm = +1 (tube shorter, pitch up) [m/s]
    v_max_out: float = 0.150       # speed at pwm = -1 [m/s]
    deadband: float = 0.20         # |pwm| below this does not move the screw
    tau_v: float = 0.030           # velocity time constant (inertia, coasting) [s]
    cmd_delay: int = 1             # PWM command delay [steps]
    backlash: float = 0.0003       # [m]
    # blowing-angle servo (SCS0009)
    angle_range_deg: float = 15.0  # action +-1 maps to +-15 deg around the mount centre
    servo_rate_dps: float = 600.0  # 0.1 s / 60 deg
    servo_tau: float = 0.020       # [s]
    servo_step_deg: float = 300.0 / 1024.0
    # edge-tone behaviour (angles relative to the servo centre)
    theta_opt_deg: float = 0.0     # best-sounding angle
    win_lo_deg: float = 5.0        # sounding window below the optimum
    win_hi_deg: float = 4.0        # sounding window above the optimum
    win_narrowing: float = 0.30    # window shrinks by this fraction at full stroke
    k_theta: float = 10.0          # pitch bend per degree away from the optimum [cents/deg]
    overblow_x: float = 0.085      # beyond this plunger position the tube can overblow [m]
    # measurement
    pitch_noise: float = 3.0       # [cents]
    dropout: float = 0.02          # probability that a frame has no pitch estimate
    obs_delay: int = 3             # latency until a feedback controller hears the pitch [steps]
    # "harsh" effects (all off by default; FluteParams.sample(harsh=...) turns them on).
    # The controllers do not model any of these: they make the rig differ from the
    # controller's model in form, not only in parameter values. Ranges are guesses
    # until the real actuator, servo and flute are measured.
    stiction: float = 0.0          # extra PWM needed to break away from standstill (stick-slip)
    pwm_curve: float = 1.0         # speed ~ drive ** pwm_curve (nonlinear PWM-to-speed)
    load_slope: float = 0.0        # pushing in gets slower by this fraction at full stroke
    speed_drift: float = 0.0       # slow random walk of the speed during a take [fraction / sqrt(s)]
    motion_noise: float = 0.0      # per-step multiplicative noise on the motion
    delay_jitter: float = 0.0      # probability that a PWM command is applied one step late
    onset_s: float = 0.0           # the edge tone needs this long inside the start window before it sounds [s]
    hysteresis_deg: float = 0.0    # the start window is this much narrower on each side than the sustain window
    edge_soft_deg: float = 0.0     # near the window edges the tone breaks off at random
    opt_slope_deg: float = 0.0     # the best angle moves by this much across the stroke
    pitch_jitter: float = 0.0      # breath: slow random pitch wobble [cents std]
    overblow_margin: float = 0.0   # overblowing starts this fraction earlier in the window
    octave_err: float = 0.0        # probability that the pitch estimate is off by an octave

    @classmethod
    def sample(cls, rng: np.random.Generator, spread: float = 1.0, harsh: float = 0.0) -> "FluteParams":
        """Domain randomisation around the nominal values (spread = 0 gives nominal).

        `harsh` in [0, 1] scales the effects the controllers do not model. They are
        drawn after the usual parameters, so a seed gives the same basic rig at any
        harshness (paired comparisons)."""
        base = cls._sample_basic(rng, spread)
        if harsh <= 0.0:
            return base
        h = float(harsh)

        def g(lo: float, hi: float) -> float:
            return float(h * rng.uniform(lo, hi))

        return dataclasses.replace(
            base,
            backlash=base.backlash + g(0.0, 0.0007),
            stiction=g(0.0, 0.15),
            pwm_curve=1.0 + g(-0.3, 0.5),
            load_slope=g(0.0, 0.3),
            speed_drift=g(0.0, 0.05),
            motion_noise=g(0.0, 0.1),
            delay_jitter=g(0.0, 0.15),
            onset_s=g(0.02, 0.08),
            hysteresis_deg=g(0.0, 1.5),
            edge_soft_deg=g(0.0, 1.0),
            opt_slope_deg=g(-3.0, 3.0),
            pitch_jitter=g(0.0, 8.0),
            overblow_margin=g(0.0, 0.3),
            octave_err=g(0.0, 0.05),
        )

    @classmethod
    def _sample_basic(cls, rng: np.random.Generator, spread: float) -> "FluteParams":
        n = cls()

        def u(center: float, half: float) -> float:
            return float(center + spread * rng.uniform(-half, half))

        return cls(
            tube_len=u(n.tube_len, 0.010),
            end_corr=max(0.0, u(n.end_corr, 0.002)),
            stroke=n.stroke,
            temp_c=u(n.temp_c, 10.0),
            v_max_in=u(n.v_max_in, 0.030),
            v_max_out=u(n.v_max_out, 0.030),
            deadband=u(n.deadband, 0.08),
            tau_v=u(n.tau_v, 0.015),
            cmd_delay=max(0, int(round(u(n.cmd_delay, 1.0)))),
            backlash=max(0.0, u(n.backlash, 0.0002)),
            angle_range_deg=n.angle_range_deg,
            servo_rate_dps=u(n.servo_rate_dps, 100.0),
            servo_tau=u(n.servo_tau, 0.010),
            servo_step_deg=n.servo_step_deg,
            theta_opt_deg=u(n.theta_opt_deg, 4.0),
            win_lo_deg=u(n.win_lo_deg, 1.5),
            win_hi_deg=u(n.win_hi_deg, 1.5),
            win_narrowing=u(n.win_narrowing, 0.15),
            k_theta=u(n.k_theta, 5.0),
            overblow_x=u(n.overblow_x, 0.010),
            pitch_noise=max(0.0, u(n.pitch_noise, 2.0)),
            dropout=max(0.0, u(n.dropout, 0.02)),
            obs_delay=max(0, int(round(u(n.obs_delay, 2.0)))),
        )


def x_for_cents(cents: float, p: FluteParams) -> float:
    """Plunger position that gives `cents` at the optimum angle (inverse of the tube model)."""
    f = float(cents_to_hz(cents))
    length = speed_of_sound(p.temp_c) / (4.0 * f)
    return p.tube_len + p.end_corr - length


@dataclass
class SimState:
    step: int
    x: float               # plunger rod position [m]
    v: float               # screw velocity [m/s]
    theta: float           # true blowing angle [deg]
    theta_readback: float  # servo read-back (quantised) [deg]
    cents: float           # true pitch [cents]
    sounding: bool
    overblown: bool
    measured: float        # pitch estimate [cents], NaN if nothing was detected


class FluteSim:
    def __init__(self, params: FluteParams, rng: np.random.Generator | None = None):
        self.p = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self, x0: float = 0.0) -> SimState:
        self.x_motor = x0
        self.x = x0
        self.v = 0.0
        self.theta = 0.0
        self.t = 0
        self._pwm_q = deque([0.0] * self.p.cmd_delay)
        self._u_last = 0.0
        self._speed = 1.0
        self._jit = 0.0
        self._snd = self.pitch_at(self.x, self.theta)[1]  # tone state (onset / hysteresis)
        self._cnt = 1 if self._snd else 0
        return self.observe()

    def _window(self, x: float, theta: float) -> tuple[float, float, float]:
        """(angle from the best angle, lower and upper half-widths of the sounding window) [deg]."""
        p = self.p
        frac = float(np.clip(x / p.stroke, 0.0, 1.0))
        d = theta - (p.theta_opt_deg + p.opt_slope_deg * (frac - 0.5))
        shrink = 1.0 - p.win_narrowing * frac
        return d, p.win_lo_deg * shrink, p.win_hi_deg * shrink

    def pitch_at(self, x: float, theta: float) -> tuple[float, bool, bool]:
        """(cents, sounding, overblown) for a plunger position and a steady angle."""
        p = self.p
        length = max(p.tube_len - x + p.end_corr, 0.02)
        f = speed_of_sound(p.temp_c) / (4.0 * length)
        d, lo, hi = self._window(x, theta)
        sounding = -lo < d < hi
        cents = float(hz_to_cents(f)) + p.k_theta * d
        overblown = sounding and x > p.overblow_x and d > (0.5 - p.overblow_margin) * hi
        if overblown:
            cents += 1200.0
        return cents, sounding, overblown

    def _update_tone(self) -> None:
        """Edge-tone state: it starts only inside the (narrower) start window after onset_s,
        then keeps sounding anywhere in the sustain window; near the edges it may break off."""
        p = self.p
        d, lo, hi = self._window(self.x, self.theta)
        in_win = -lo < d < hi
        if self._snd:
            self._snd = in_win
        else:
            h = p.hysteresis_deg
            self._cnt = self._cnt + 1 if (-lo + h < d < hi - h) else 0
            self._snd = self._cnt > 0 and self._cnt * DT >= p.onset_s - 1e-12
        if self._snd and p.edge_soft_deg > 0.0:
            dist = min(d + lo, hi - d)
            if dist < p.edge_soft_deg and self.rng.random() < 0.5 * (1.0 - dist / p.edge_soft_deg):
                self._snd, self._cnt = False, 0

    def step(self, pwm: float, angle_cmd: float) -> SimState:
        p = self.p

        # plunger: delayed PWM -> dead band / stiction -> nonlinear speed -> velocity lag -> end stops -> backlash
        self._pwm_q.append(float(np.clip(pwm, -1.0, 1.0)))
        u = self._pwm_q.popleft()
        if p.delay_jitter > 0.0 and self.rng.random() < p.delay_jitter:
            u = self._u_last  # this command arrives a step late
        self._u_last = u
        mag = abs(u)
        threshold = p.deadband + (p.stiction if abs(self.v) < 0.005 else 0.0)
        drive = 0.0 if mag < threshold else float(np.sign(u)) * (mag - p.deadband) / (1.0 - p.deadband)
        if p.pwm_curve != 1.0:
            drive = float(np.sign(drive)) * abs(drive) ** p.pwm_curve
        v_max = p.v_max_in * (1.0 - p.load_slope * float(np.clip(self.x_motor / p.stroke, 0.0, 1.0))) if drive > 0 else p.v_max_out
        if p.speed_drift > 0.0:
            self._speed = float(np.clip(self._speed + self.rng.normal(0.0, p.speed_drift * np.sqrt(DT)), 0.8, 1.2))
        v_cmd = drive * v_max * self._speed
        self.v += (v_cmd - self.v) * min(1.0, DT / p.tau_v)
        moved = self.v * (1.0 + self.rng.normal(0.0, p.motion_noise)) if p.motion_noise > 0.0 else self.v
        self.x_motor += moved * DT
        if self.x_motor <= 0.0:
            self.x_motor, self.v = 0.0, max(self.v, 0.0)
        elif self.x_motor >= p.stroke:
            self.x_motor, self.v = p.stroke, min(self.v, 0.0)
        half = p.backlash / 2.0
        if self.x_motor - self.x > half:
            self.x = self.x_motor - half
        elif self.x - self.x_motor > half:
            self.x = self.x_motor + half

        # servo: quantised target, first-order lag, rate limit
        target = float(np.clip(angle_cmd, -1.0, 1.0)) * p.angle_range_deg
        target = round(target / p.servo_step_deg) * p.servo_step_deg
        d_theta = (target - self.theta) * min(1.0, DT / p.servo_tau)
        lim = p.servo_rate_dps * DT
        self.theta += float(np.clip(d_theta, -lim, lim))

        self._update_tone()
        if p.pitch_jitter > 0.0:  # breath: Ornstein-Uhlenbeck wobble with a 50 ms time constant
            self._jit += -self._jit * DT / 0.05 + p.pitch_jitter * np.sqrt(2.0 * DT / 0.05) * self.rng.normal()
        self.t += 1
        return self.observe()

    def observe(self) -> SimState:
        p = self.p
        cents, _, overblown = self.pitch_at(self.x, self.theta)
        sounding = self._snd
        overblown = overblown and sounding
        cents += self._jit
        readback = round(self.theta / p.servo_step_deg) * p.servo_step_deg
        measured = float("nan")
        if sounding and self.rng.random() >= p.dropout:
            measured = cents + float(self.rng.normal(0.0, p.pitch_noise))
            if p.octave_err > 0.0 and self.rng.random() < p.octave_err:
                measured += 1200.0 if self.rng.random() < 0.5 else -1200.0
        return SimState(self.t, self.x, self.v, self.theta, readback, cents, sounding, overblown, measured)

    def find_sounding_angle(self, x_ref: float, step_deg: float = 0.5, repeats: int = 3) -> float:
        """'Sounding compensation': sweep the angle at one plunger position and
        return the centre of the widest sounding run [deg]. Uses the (noisy)
        detector only, as the real rig would."""
        p = self.p
        angles = np.arange(-p.angle_range_deg, p.angle_range_deg + 1e-9, step_deg)
        heard = []
        for a in angles:
            _, sounding, _ = self.pitch_at(x_ref, float(a))
            hits = sum(sounding and self.rng.random() >= p.dropout for _ in range(repeats))
            heard.append(hits > 0)
        best_len, best_mid, run_start = 0, 0.0, None
        for i, h in enumerate(heard + [False]):
            if h and run_start is None:
                run_start = i
            elif not h and run_start is not None:
                if i - run_start > best_len:
                    best_len = i - run_start
                    best_mid = float(angles[run_start:i].mean())
                run_start = None
        return best_mid
