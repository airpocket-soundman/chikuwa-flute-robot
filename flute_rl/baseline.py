"""Physics-prior controller: the hand-written starting point before learning.

It only knows the *nominal* model (not the randomised episode parameters)
and uses only the observation, like a learned policy would:

* plunger: dead-reckons its own position from the commands it sent (nominal
  delay, dead band and velocity lag) and steers toward the position that the
  nominal tube model gives for the target a few steps ahead;
* angle: holds the sounding-compensation angle during notes and swings to
  the lower side (out of the sounding window) during rests.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .env import CENTER_CENTS, CENTS_SCALE
from .sim import DT, FluteParams, hz_to_cents, speed_of_sound, x_for_cents


class PhysicsPriorPolicy:
    def __init__(self, track_time: float = 0.04, angle_lead: int = 3, rest_angle: float = -1.0,
                 params: FluteParams | None = None):
        self.track_time = track_time
        self.angle_lead = angle_lead
        self.rest_angle = rest_angle
        self.p = params or FluteParams()

    def reset(self, env) -> None:
        self.layout = env.obs_layout
        self.lookahead = env.lookahead
        p = self.p
        self.lead = min(p.cmd_delay + int(round(p.tau_v / DT)) + 1, self.lookahead - 1)
        self.x_hat = 0.0
        self.v_hat = 0.0
        self.q = deque([0.0] * p.cmd_delay)

    def _v_in(self, x: float) -> float:
        p = self.p
        return p.v_max_in * (1.0 - p.load_slope * float(np.clip(x / p.stroke, 0.0, 1.0)))

    def _pwm_for(self, x_des: float) -> float:
        """PWM that the model says gives the wanted speed (dead band, PWM curve, load and stiction included;
        with the default FluteParams these reduce to the plain linear dead-band model)."""
        p = self.p
        v_in = self._v_in(self.x_hat)
        v_des = np.clip((x_des - self.x_hat) / self.track_time, -p.v_max_out, v_in)
        drive = v_des / (v_in if v_des > 0 else p.v_max_out)
        if abs(drive) < 0.02:
            return 0.0
        mag = abs(drive) ** (1.0 / p.pwm_curve) if p.pwm_curve != 1.0 else abs(drive)
        pwm = p.deadband + (1.0 - p.deadband) * mag
        if p.stiction > 0.0 and abs(self.v_hat) < 0.005:
            pwm = max(pwm, p.deadband + p.stiction + 0.01)  # enough to break away from standstill
        return float(np.sign(drive) * min(pwm, 1.0))

    def _dead_reckon(self, pwm: float) -> None:
        p = self.p
        self.q.append(pwm)
        u = self.q.popleft()
        mag = abs(u)
        threshold = p.deadband + (p.stiction if abs(self.v_hat) < 0.005 else 0.0)
        drive = 0.0 if mag < threshold else np.sign(u) * (mag - p.deadband) / (1.0 - p.deadband)
        if p.pwm_curve != 1.0:
            drive = np.sign(drive) * abs(drive) ** p.pwm_curve
        v_cmd = drive * (self._v_in(self.x_hat) if drive > 0 else p.v_max_out)
        self.v_hat += (v_cmd - self.v_hat) * min(1.0, DT / p.tau_v)
        self.x_hat = float(np.clip(self.x_hat + self.v_hat * DT, 0.0, p.stroke))

    def model_cents(self, x: float) -> float:
        """Pitch the controller's model expects at plunger position x (optimum angle)."""
        p = self.p
        length = max(p.tube_len - x + p.end_corr, 0.02)
        return float(hz_to_cents(speed_of_sound(p.temp_c) / (4.0 * length)))

    def rewind(self) -> None:
        """Undo the dead reckoning of the last act() (before committing a modified PWM)."""
        self.x_hat, self.v_hat, self.q = self._pre

    def commit(self, pwm: float) -> None:
        """Dead-reckon the PWM that was actually sent (after rewind())."""
        self._dead_reckon(float(pwm))

    def act(self, obs: np.ndarray) -> np.ndarray:
        lay = self.layout
        if obs[lay["time"]][0] == 0.0:  # start of a take: the rig has been homed
            self.x_hat, self.v_hat = 0.0, 0.0
            self.q = deque([0.0] * self.p.cmd_delay)
        vals = obs[lay["target"]]
        mask = obs[lay["target_mask"]] > 0.5

        # aim at the first note at/after the lead time (pre-positions during rests)
        ahead = np.flatnonzero(mask[self.lead:])
        if ahead.size:
            idx = self.lead + int(ahead[0])
        else:
            anywhere = np.flatnonzero(mask)
            idx = int(anywhere[-1]) if anywhere.size else None
        if idx is None:
            x_des = self.x_hat
        else:
            x_des = x_for_cents(float(vals[idx]) * CENTS_SCALE + CENTER_CENTS, self.p)
        pwm = self._pwm_for(float(np.clip(x_des, 0.0, self.p.stroke)))
        self._pre = (self.x_hat, self.v_hat, deque(self.q))
        self._dead_reckon(pwm)

        note_soon = bool(mask[min(self.angle_lead, self.lookahead - 1)] or mask[0])
        angle = float(obs[lay["angle_comp"]][0]) if note_soon else self.rest_angle
        return np.array([pwm, angle], dtype=np.float32)


class ILCPolicy(PhysicsPriorPolicy):
    """Physics prior + classic iterative learning control across takes.

    Keeps a per-time-step pitch offset that is updated from the previous
    take's error (obs block `prev_err`): aim = target - offset, with
    offset_k+1[t] = offset_k[t] + gain * err_k[t]. Needs an env with takes > 1.
    """

    def __init__(self, gain: float = 0.7, **kw):
        super().__init__(**kw)
        self.gain = gain

    def reset(self, env) -> None:
        super().reset(env)
        n = len(env.target)
        self.offset = np.zeros(n + self.lookahead)
        self.k = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        lay = self.layout
        if obs[lay["time"]][0] == 0.0:
            self.k = 0
        if "prev_err" in lay:
            i = min(self.lead, self.lookahead - 1)
            if obs[lay["prev_mask"]][i] > 0.5:
                # err_k at time k + lead is known from the previous take: update before aiming there
                self.offset[self.k + i] += self.gain * float(obs[lay["prev_err"]][i]) * CENTS_SCALE
        shifted = obs.copy()
        vals = shifted[lay["target"]]
        mask = shifted[lay["target_mask"]] > 0.5
        corr = self.offset[self.k:self.k + self.lookahead] / CENTS_SCALE
        vals[mask] -= corr[mask]
        self.k += 1
        return super().act(shifted)


class ZeroPolicy:
    """Does nothing: useful as a sanity floor."""

    def reset(self, env) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)
