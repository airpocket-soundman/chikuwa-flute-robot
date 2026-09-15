"""Take-to-take adaptation: identify the rig from the previous take's recording.

During a take the policy plays open loop and cannot know how fast *this*
actuator is or how long *this* tube is; that is where most of the first-take
error comes from. Between takes, though, the rig has the whole recording of
the take it just played (the measured pitch track) and the PWM profile it
sent. `AdaptivePolicy` fits a small world model to that data and plays the
next take with it:

    measured cents  ~=  tube(x(t)) + c0,
    x(t) = dead-reckoned plunger position for the PWM it sent, with the
           actuator speeds scaled by (s_in, s_out),

fitting (s_in, s_out, tube length offset dL, cents offset c0) with a few
Gauss-Newton steps. The fitted parameters replace the nominal ones inside the
physics-prior controller. On top of that an ILC offset corrects what the
model cannot explain, frame by frame.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .baseline import ILCPolicy
from .env import CENTS_SCALE
from .sim import DT, FluteParams, hz_to_cents, speed_of_sound

# prior std of the fitted quantities (matches the domain-randomisation spreads)
PRIOR_STD = np.array([0.12, 0.12, 0.006, 0.15])  # log s_in, log s_out, dL [m], c0 [100 cents]
OUTLIER_CENTS = 400.0  # frames further than this from the model are ignored (overblow, glitches)


def plunger_trace(pwm: np.ndarray, p: FluteParams) -> np.ndarray:
    """Plunger position after each step for a PWM sequence, from home (same model as the sim, no backlash)."""
    q = [0.0] * p.cmd_delay
    x = v = 0.0
    k = min(1.0, DT / p.tau_v)
    db = p.deadband
    out = np.empty(len(pwm))
    for t, u_new in enumerate(pwm):
        q.append(float(u_new))
        u = q.pop(0)
        mag = abs(u)
        drive = 0.0 if mag < db else (mag - db) / (1.0 - db) * (1.0 if u > 0 else -1.0)
        v += (drive * (p.v_max_in if drive > 0 else p.v_max_out) - v) * k
        x += v * DT
        if x <= 0.0:
            x, v = 0.0, max(v, 0.0)
        elif x >= p.stroke:
            x, v = p.stroke, min(v, 0.0)
        out[t] = x
    return out


def apply_z(z: np.ndarray, base: FluteParams) -> FluteParams:
    return dataclasses.replace(
        base,
        v_max_in=base.v_max_in * float(np.exp(z[0])),
        v_max_out=base.v_max_out * float(np.exp(z[1])),
        tube_len=base.tube_len + float(z[2]),
    )


def predict_cents(z: np.ndarray, pwm: np.ndarray, base: FluteParams) -> np.ndarray:
    p = apply_z(z, base)
    x = plunger_trace(pwm, p)
    length = np.maximum(p.tube_len - x + p.end_corr, 0.02)
    return hz_to_cents(speed_of_sound(p.temp_c) / (4.0 * length)) + 100.0 * z[3]


def fit_rig(takes: list[tuple[np.ndarray, np.ndarray]], base: FluteParams, z0: np.ndarray | None = None,
            iters: int = 6, reg: float = 1.0) -> tuple[np.ndarray, float]:
    """Gauss-Newton fit of z = (log s_in, log s_out, dL, c0/100) to [(pwm, measured cents)] of past takes.

    Returns (z, rms residual in cents over the inlier frames)."""
    z = np.zeros(4) if z0 is None else np.array(z0, dtype=float)
    data = [(np.asarray(pwm, float), np.asarray(meas, float)) for pwm, meas in takes]
    eps = np.array([1e-3, 1e-3, 1e-5, 1e-3])

    def residuals(zz):
        rs = []
        for pwm, meas in data:
            ok = np.isfinite(meas)
            rs.append((predict_cents(zz, pwm, base)[ok] - meas[ok]) / 100.0)
        return np.concatenate(rs) if rs else np.zeros(0)

    if z0 is None or not np.any(z0):
        # coarse grid over the two speeds first: a badly-off first take can be several hundred cents
        # away everywhere, which a local fit with a tight outlier gate would reject as all outliers
        best = (np.inf, z)
        for a in np.log(np.linspace(0.7, 1.3, 13)):
            for b in np.log(np.linspace(0.7, 1.3, 13)):
                zz = np.array([a, b, 0.0, 0.0])
                r = residuals(zz)
                if r.size:
                    loss = float(np.mean(np.minimum(np.abs(r), 6.0) ** 2))
                    if loss < best[0]:
                        best = (loss, zz)
        z = best[1]

    rms = float("nan")
    gates = np.maximum(np.geomspace(1500.0, OUTLIER_CENTS, max(iters, 2)), OUTLIER_CENTS)
    for gate in gates:
        r = residuals(z)
        inl = np.abs(r) < gate / 100.0
        if inl.sum() < 10:
            break
        J = np.stack([(residuals(z + e) - r) / e[i] for i, e in enumerate(np.diag(eps))], axis=1)
        J, ri = J[inl], r[inl]
        # Gauss-Newton with a Gaussian prior on z (keeps unidentifiable directions near nominal)
        w = reg * len(ri) / 200.0
        H = J.T @ J + w * np.diag(1.0 / PRIOR_STD**2)
        g = J.T @ ri + w * z / PRIOR_STD**2
        step = np.linalg.solve(H, g)
        z = z - step
        rms = float(np.sqrt(np.mean(ri**2)) * 100.0)
        if gate <= OUTLIER_CENTS and np.all(np.abs(step) < eps):
            break
    return z, rms


class AdaptivePolicy(ILCPolicy):
    """Physics prior + rig identification between takes + ILC on what the model cannot explain.

    Needs an env with takes > 1. Between takes it reads the previous take's
    recording (`env.prev_err` + `env.target`, i.e. the measured pitch track the
    rig has after each take) and the PWM profile it sent itself.
    """

    def __init__(self, ilc_gain: float = 0.5, reg: float = 1.0, fit_iters: int = 6, refit: bool = False, **kw):
        super().__init__(gain=ilc_gain, **kw)
        self.reg = reg
        self.refit = refit
        self.fit_iters = fit_iters
        self.nominal = self.p

    def reset(self, env) -> None:
        self.env = env
        self.p = self.nominal
        self.z = np.zeros(4)
        self.history_takes: list[tuple[np.ndarray, np.ndarray]] = []
        self.pwm_log: list[float] = []
        self.fit_rms = float("nan")
        super().reset(env)

    def _end_of_take(self) -> None:
        measured = self.env.target + self.env.prev_err
        self.history_takes.append((np.array(self.pwm_log), measured))
        self.pwm_log = []
        if len(self.history_takes) == 1 or self.refit:
            self.z, self.fit_rms = fit_rig(self.history_takes, self.nominal, z0=self.z,
                                           iters=self.fit_iters, reg=self.reg)
            self.p = apply_z(self.z, self.nominal)

    def model_cents(self, x: float) -> float:
        return super().model_cents(x) + 100.0 * float(self.z[3])

    def commit(self, pwm: float) -> None:
        super().commit(pwm)
        self.pwm_log[-1] = float(pwm)  # identification must see the PWM that was really sent

    def act(self, obs: np.ndarray) -> np.ndarray:
        lay = self.layout
        if obs[lay["time"]][0] == 0.0 and self.pwm_log:
            self._end_of_take()
        shifted = obs.copy()
        mask = shifted[lay["target_mask"]] > 0.5
        shifted[lay["target"]][mask] -= self.z[3] * 100.0 / CENTS_SCALE
        if len(self.history_takes) <= 1 and np.isfinite(self.fit_rms) and "prev_err" in lay:
            # take 2: the take-1 errors are what the fitted model already explains; do not feed them to the ILC
            shifted[lay["prev_mask"]] = 0.0
        a = super().act(shifted)
        self.pwm_log.append(float(a[0]))
        return a
