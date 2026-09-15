"""Reinforcement-learning environment for the slide-flute rig.

Gymnasium-compatible API (reset / step / observation_space / action_space).
If gymnasium is installed the class subclasses gymnasium.Env; otherwise a
tiny stand-in is used so the environment runs with numpy only (e.g. on the
Linux side of an Arduino UNO Q).

Modes
-----
* Open loop (default): the profile is played without listening, as in the
  "play the whole phrase, then compare" scheme. The observation contains NO
  live pitch: only the target (with look-ahead), the policy's own action
  history, the servo read-back and the sounding-compensation angle.
* takes > 1: one episode plays the same target `takes` times on the same
  rig, homing the plunger between takes. From the second take on, the
  observation also carries the previous take's pitch error (with the same
  look-ahead) and action, so the policy can learn to improve take by take
  (a learned iterative-learning-control update).
* feedback=True: adds the measured pitch error delayed by `obs_delay` steps,
  for training a feedback (residual) policy that must cope with latency.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .sim import DT, FluteParams, FluteSim, hz_to_cents, x_for_cents
from .targets import make_target, sample_level

try:
    import gymnasium as gym
    from gymnasium import spaces

    _EnvBase = gym.Env
    HAS_GYMNASIUM = True
except ImportError:  # numpy-only fallback
    HAS_GYMNASIUM = False

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.low = np.full(self.shape, low, dtype=dtype)
            self.high = np.full(self.shape, high, dtype=dtype)

        def sample(self):
            lo = np.where(np.isfinite(self.low), self.low, -1.0)
            hi = np.where(np.isfinite(self.high), self.high, 1.0)
            return np.random.uniform(lo, hi).astype(self.dtype)

        def contains(self, x) -> bool:
            x = np.asarray(x)
            return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    class spaces:  # noqa: N801 - mimics the gymnasium module name
        Box = _Box

    class _EnvBase:
        metadata: dict = {}
        np_random = None

        def reset(self, *, seed=None, options=None):
            if seed is not None or self.np_random is None:
                self.np_random = np.random.default_rng(seed)


CENTER_CENTS = float(hz_to_cents(1000.0))  # observation normalisation centre
CENTS_SCALE = 600.0
PITCH_CLIP = 300.0  # cents, per-frame error cap in the reward
SILENCE_PENALTY = 2.0
REST_PENALTY = 1.0
OCTAVE_PENALTY = 1.0
SMOOTH_WEIGHT = 0.01
X_REF = x_for_cents(CENTER_CENTS, FluteParams())  # where sounding compensation is done
V_NOMINAL = FluteParams().v_max_in


def _window(arr: np.ndarray, start: int, n: int) -> np.ndarray:
    w = arr[start:start + n]
    if len(w) < n:
        w = np.concatenate([w, np.full(n - len(w), np.nan)])
    return w


class FluteEnv(_EnvBase):
    metadata = {"render_modes": []}

    def __init__(
        self,
        level: int | None = None,
        progress: float = 1.0,
        target_fn=None,
        lookahead: int = 50,
        history: int = 10,
        takes: int = 1,
        feedback: bool = False,
        randomize: bool = True,
        spread: float = 1.0,
        params: FluteParams | None = None,
        harsh: float = 0.0,
        reward_on: str = "measured",
    ):
        """reward_on: "measured" scores the pitch estimate (what a real rig can do); "true" scores the
        pitch actually played (simulation only; not disturbed by the estimator's octave errors)."""
        self.harsh = harsh
        self.reward_on = reward_on
        self.level = level
        self.progress = progress
        self.target_fn = target_fn
        self.lookahead = lookahead
        self.history = history
        self.takes = max(1, int(takes))
        self.feedback = feedback
        self.randomize = randomize
        self.spread = spread
        self.fixed_params = params

        blocks = [
            ("target", lookahead),
            ("target_mask", lookahead),
            ("act_hist", history * 2),
            ("travel", 1),
            ("angle_readback", 1),
            ("angle_comp", 1),
            ("time", 1),
        ]
        if self.takes > 1:
            blocks += [("prev_err", lookahead), ("prev_mask", lookahead), ("prev_act", 2), ("take", 1)]
        if feedback:
            blocks += [("fb_err", 1), ("fb_heard", 1), ("fb_valid", 1)]
        self.obs_layout: dict[str, slice] = {}
        size = 0
        for name, n in blocks:
            self.obs_layout[name] = slice(size, size + n)
            size += n
        self.obs_dim = size
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)  # [plunger pwm, blowing angle]

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        if self.randomize:
            self.params = FluteParams.sample(rng, self.spread, self.harsh)
        else:
            self.params = self.fixed_params or FluteParams()
        self.sim = FluteSim(self.params, rng)

        if options and "target" in options:
            target = options["target"]
        elif self.target_fn is not None:
            target = self.target_fn(rng)
        else:
            lvl = self.level if self.level is not None else sample_level(rng, self.progress)
            target = make_target(rng, lvl)
        self.target = np.asarray(target, dtype=float)

        # sounding compensation: the only calibration done on the real rig
        self.angle_comp = self.sim.find_sounding_angle(X_REF)

        self.take = 0
        self.prev_err = np.full(len(self.target), np.nan)
        self.prev_act = np.zeros((len(self.target), 2))
        self._start_take()
        return self._obs(), {"params": self.params, "angle_comp_deg": self.angle_comp}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        s = self.sim.step(a[0], a[1])
        tgt = self.target[self.t]
        reward, terms = self._reward(tgt, s, a)

        heard = bool(np.isfinite(s.measured))
        err = s.measured - tgt if (heard and np.isfinite(tgt)) else np.nan
        self._fb_q.append((err, heard))
        self._fb_seen = self._fb_q.popleft()
        self.cur_err[self.t] = err
        self.cur_act[self.t] = a

        self.acts = np.roll(self.acts, -1, axis=0)
        self.acts[-1] = a
        self.travel = float(np.clip(self.travel + a[0] * V_NOMINAL * DT / self.params.stroke, -0.5, 1.5))
        self.state = s
        self.t += 1
        info = {"state": s, "target": tgt, "reward_terms": terms, "take": self.take}

        terminated = False
        if self.t >= len(self.target):
            if self.take < self.takes - 1:
                self.prev_err, self.prev_act = self.cur_err, self.cur_act
                self.take += 1
                self._start_take()
            else:
                terminated = True
        return self._obs(), float(reward), terminated, False, info

    # ---------------------------------------------------------------- internals
    def _start_take(self) -> None:
        """Home the plunger and clear per-take state (same rig, same target)."""
        self.state = self.sim.reset()
        self.t = 0
        self.acts = np.zeros((self.history, 2))
        self.travel = 0.0
        self._fb_q = deque([(np.nan, False)] * self.params.obs_delay)
        self._fb_seen = (np.nan, False)
        self.cur_err = np.full(len(self.target), np.nan)
        self.cur_act = np.zeros((len(self.target), 2))

    def _reward(self, tgt: float, s, a: np.ndarray) -> tuple[float, dict]:
        # Silence is judged from the true state for simplicity; on the rig the
        # detector (with its drop-outs) plays this role.
        terms = {"pitch": 0.0, "silence": 0.0, "rest": 0.0, "octave": 0.0, "smooth": 0.0}
        pitch = s.cents if self.reward_on == "true" else s.measured
        if np.isfinite(tgt):
            if not s.sounding:
                terms["silence"] = -SILENCE_PENALTY
            elif np.isfinite(pitch):
                terms["pitch"] = -min(abs(pitch - tgt), PITCH_CLIP) / 100.0
                if s.overblown:
                    terms["octave"] = -OCTAVE_PENALTY
        elif s.sounding:
            terms["rest"] = -REST_PENALTY
        terms["smooth"] = -SMOOTH_WEIGHT * float(np.sum((a - self.acts[-1]) ** 2))
        return sum(terms.values()), terms

    def _obs(self) -> np.ndarray:
        n = self.lookahead
        w = _window(self.target, self.t, n)
        mask = np.isfinite(w)
        parts = [
            np.where(mask, (w - CENTER_CENTS) / CENTS_SCALE, 0.0),
            mask.astype(float),
            self.acts.ravel(),
            [self.travel],
            [self.state.theta_readback / self.params.angle_range_deg],
            [self.angle_comp / self.params.angle_range_deg],
            [self.t / max(1, len(self.target))],
        ]
        if self.takes > 1:
            pe = _window(self.prev_err, self.t, n)
            pm = np.isfinite(pe)
            prev_a = self.prev_act[self.t] if self.t < len(self.target) else np.zeros(2)
            parts += [
                np.where(pm, np.clip(pe / CENTS_SCALE, -3.0, 3.0), 0.0),
                pm.astype(float),
                prev_a,
                [self.take / (self.takes - 1)],
            ]
        if self.feedback:
            err, heard = self._fb_seen
            valid = bool(np.isfinite(err))
            parts += [[float(np.clip(err / CENTS_SCALE, -3.0, 3.0)) if valid else 0.0], [float(heard)], [float(valid)]]
        return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------- tools
def rollout(env: FluteEnv, policy, seed=None, options=None) -> dict:
    """Run one episode with `policy` (an object with reset(env) and act(obs))."""
    obs, _ = env.reset(seed=seed, options=options)
    if hasattr(policy, "reset"):
        policy.reset(env)
    keys = ("x", "theta", "cents", "measured", "sounding", "overblown")
    log = {k: [] for k in keys}
    log.update(actions=[], rewards=[], take=[])
    done = False
    while not done:
        a = policy.act(obs)
        obs, r, terminated, truncated, info = env.step(a)
        s = info["state"]
        for k in keys:
            log[k].append(getattr(s, k))
        log["actions"].append(np.clip(np.asarray(a, dtype=float), -1, 1))
        log["rewards"].append(r)
        log["take"].append(info["take"])
        done = terminated or truncated
    out = {k: np.asarray(v) for k, v in log.items()}
    out["target"] = np.tile(env.target, env.takes)
    out.update(episode_metrics(out))
    out["per_take"] = [
        episode_metrics({k: v[out["take"] == i] for k, v in out.items() if isinstance(v, np.ndarray)})
        for i in range(env.takes)
    ]
    return out


def episode_metrics(log: dict) -> dict:
    target, sounding, measured = log["target"], log["sounding"].astype(bool), log["measured"]
    active = np.isfinite(target)
    valid = active & sounding & np.isfinite(measured)
    err = np.abs(measured[valid] - target[valid])
    # the pitch the flute really played (the simulator knows it); the measured one above also
    # contains the pitch estimator's own errors, e.g. octave mistakes
    played = active & sounding
    true_err = np.abs(log["cents"][played] - target[played]) if "cents" in log else np.zeros(0)
    # short silences between notes (detached playing), excluding the lead-in and the tail
    inner = np.zeros(len(target), bool)
    idx = np.flatnonzero(active)
    if idx.size:
        inner[idx[0]:idx[-1]] = True
    inner &= ~active
    return {
        "gap_leak": float(np.mean(sounding[inner])) if inner.any() else float("nan"),
        "return": float(np.sum(log["rewards"])),
        "mean_reward": float(np.mean(log["rewards"])),
        "mean_abs_cents": float(np.mean(err)) if err.size else float("nan"),
        "true_abs_cents": float(np.mean(true_err)) if true_err.size else float("nan"),
        "sounding_rate": float(np.mean(sounding[active])) if active.any() else float("nan"),
        "rest_leak": float(np.mean(sounding[~active])) if (~active).any() else 0.0,
        "overblow_rate": float(np.mean(log["overblown"][active])) if active.any() else 0.0,
    }
