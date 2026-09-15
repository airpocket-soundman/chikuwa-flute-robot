"""Listening while playing: correct the plunger belief from the delayed pitch.

Needs an env with feedback=True. The measured pitch error arrives
`obs_delay` steps late. If the flute sounds e cents sharp, the plunger is
further in than the controller believes, so its dead-reckoned position is
moved by the length change that e cents corresponds to (a delayed
observer, not a pitch integrator: the correction lives in the position
belief, which is what drifts when the actuator speed is off).
"""
from __future__ import annotations

import numpy as np

from .adapt import AdaptivePolicy
from .env import CENTS_SCALE


class FeedbackPolicy(AdaptivePolicy):
    def __init__(self, fb_gain: float = 0.05, max_err: float = 300.0, **kw):
        super().__init__(**kw)
        self.fb_gain = fb_gain
        self.max_err = max_err
        self.gain_scale = 1.0  # set per step by a learned policy (FeedbackResidualPolicy)

    def act(self, obs: np.ndarray) -> np.ndarray:
        lay = self.layout
        if "fb_err" in lay and obs[lay["fb_valid"]][0] > 0.5 and obs[lay["time"]][0] > 0.0:
            err = float(np.clip(obs[lay["fb_err"]][0] * CENTS_SCALE, -self.max_err, self.max_err))
            if abs(err) < self.max_err:  # skip octave jumps and glitches
                # d(cents)/dx = 1200 / ln2 / L  ->  dx = err * L * ln2 / 1200
                length = max(self.p.tube_len - self.x_hat + self.p.end_corr, 0.02)
                dx = err * length * np.log(2.0) / 1200.0
                self.x_hat = float(np.clip(self.x_hat + self.fb_gain * self.gain_scale * dx, 0.0, self.p.stroke))
        return super().act(obs)


class FeedbackResidualPolicy:
    """FeedbackPolicy + a small network that (a) scales the feedback gain step by step
    and (b) adds a residual to the action.

    The network decides when the delayed pitch is worth trusting (e.g. steady
    notes vs. note changes, first take vs. later takes). Output 3 sets the gain
    scale to 1 + tanh(.) in [0, 2]; outputs 1-2 are the action residual.

    Features (2 * horizon + 8): expected error of the next `horizon` target
    frames under the current plunger belief [100 cents], their note mask, the
    believed velocity, servo offset from the sounding angle, the delayed pitch
    error [100 cents] and its validity, the take index, and the last action.
    With `history` > 0, the last `history` steps of (delayed pitch error,
    PWM, angle) are appended, so a memoryless network can still relate what
    it did to what it hears. A GRU net keeps its own memory instead (its state
    is reset once per episode, so it carries over between takes of one rig).
    """

    def __init__(self, base: FeedbackPolicy, net, scale: float = 0.3, horizon: int = 30, history: int = 0):
        self.base, self.net, self.scale, self.horizon, self.history = base, net, scale, horizon, history

    @staticmethod
    def feature_dim(horizon: int = 30, history: int = 0) -> int:
        return 2 * horizon + 8 + 3 * history

    def reset(self, env) -> None:
        self.base.reset(env)
        self.lay = env.obs_layout
        self.takes = env.takes
        self.last = np.zeros(2)
        self.hist = np.zeros((self.history, 3))
        if hasattr(self.net, "reset_state"):
            self.net.reset_state()

    def act(self, obs: np.ndarray) -> np.ndarray:
        from .env import CENTER_CENTS

        lay, base, h = self.lay, self.base, self.horizon
        if obs[lay["time"]][0] == 0.0:
            base.x_hat, base.v_hat = 0.0, 0.0  # homed (the base does the same inside act)
        tgt = obs[lay["target"]][:h] * CENTS_SCALE + CENTER_CENTS
        mask = obs[lay["target_mask"]][:h] > 0.5
        err = np.where(mask, np.clip((tgt - base.model_cents(base.x_hat)) / 100.0, -3.0, 3.0), 0.0)
        fb_valid = obs[lay["fb_valid"]][0]
        fb_err = np.clip(obs[lay["fb_err"]][0] * CENTS_SCALE / 100.0, -3.0, 3.0) * fb_valid
        take = obs[lay["take"]][0] if "take" in lay else 0.0
        angle_off = obs[lay["angle_readback"]][0] - obs[lay["angle_comp"]][0]
        if self.history:
            self.hist = np.roll(self.hist, -1, axis=0)
            self.hist[-1] = (fb_err, self.last[0], self.last[1])
        feats = np.concatenate([err, mask.astype(float), [base.v_hat / 0.15, angle_off, fb_err, fb_valid, take],
                                self.last, [1.0], self.hist.ravel()])
        out = self.net(feats)
        base.gain_scale = 1.0 + float(out[2])
        b = np.asarray(base.act(obs), dtype=float)
        base.rewind()
        a = np.clip(b + self.scale * out[:2], -1.0, 1.0)
        base.commit(a[0])
        self.last = a
        return a.astype(np.float32)
