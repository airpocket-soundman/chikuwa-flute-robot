"""Batched environment and controllers in PyTorch, for running a whole ES generation on the GPU.

This is the numpy pipeline FluteEnv + FeedbackResidualPolicy(FeedbackPolicy(...), net)
(physics prior, rig identification between takes, ILC, live feedback and an
optional residual network) rewritten over a batch of rigs. Every rig can
use a different network (one ES candidate per group of rigs).

Episodes in one batch may have different target lengths: the batch runs to
the longest one and the extra steps of shorter pieces are ignored (they are
rests, excluded from reward and metrics, and every take starts from home).

With noise-free rigs the actions match the numpy pipeline step by step
(tests/test_torch_env.py), so results transfer between the two.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .adapt import OUTLIER_CENTS, PRIOR_STD
from .env import OCTAVE_PENALTY, PITCH_CLIP, REST_PENALTY, SILENCE_PENALTY, SMOOTH_WEIGHT, X_REF
from .sim import DT, FluteParams, FluteSim
from .torch_sim import BatchFluteSim, BatchParams, hz_to_cents, speed_of_sound

NOM = FluteParams()
LOOKAHEAD = 50
MAX_OBS_DELAY = 8
NAN = float("nan")


def rig_from_seed(seed: int, spread: float = 1.0, params: FluteParams | None = None,
                  harsh: float = 0.0) -> tuple[FluteParams, float]:
    """(params, sounding-compensation angle) exactly as FluteEnv(harsh=...).reset(seed=seed) builds them."""
    rng = np.random.default_rng(int(seed))
    p = FluteParams.sample(rng, spread, harsh) if params is None else params
    return p, FluteSim(p, rng).find_sounding_angle(X_REF)


class BatchFluteEnv:
    def __init__(self, rigs: list[tuple[FluteParams, float]], targets: list[np.ndarray], takes: int = 2,
                 device=None, dtype=torch.float32, generator: torch.Generator | None = None):
        self.B, self.takes, self.dtype = len(rigs), takes, dtype
        self.p = BatchParams([r[0] for r in rigs], device, dtype)
        self.dev = self.p.tube_len.device
        if int(self.p.obs_delay.max()) >= MAX_OBS_DELAY:
            raise ValueError("obs_delay too large")
        self.sim = BatchFluteSim(self.p, generator)
        self.angle_comp = torch.tensor([r[1] for r in rigs], device=self.dev, dtype=dtype)
        lengths = [len(t) for t in targets]
        self.T = max(lengths)
        tg = np.full((self.B, self.T + LOOKAHEAD), np.nan)
        for i, t in enumerate(targets):
            tg[i, :len(t)] = t
        self.target_pad = torch.tensor(tg, device=self.dev, dtype=dtype)
        self.target = self.target_pad[:, :self.T]
        self.length = torch.tensor(lengths, device=self.dev)
        self.inside = torch.arange(self.T, device=self.dev)[None, :] < self.length[:, None]
        active = torch.isfinite(self.target)
        idx = torch.arange(self.T, device=self.dev)[None, :]
        first = torch.where(active, idx, self.T).min(1).values
        last = torch.where(active, idx, -1).max(1).values
        self.inner_rest = (~active) & (idx >= first[:, None]) & (idx < last[:, None])
        self.take = 0
        self.prev_err = torch.full((self.B, self.T + LOOKAHEAD), NAN, device=self.dev, dtype=dtype)
        self.logs: list[dict] = []
        self._start_take()

    def _start_take(self) -> None:
        self.state = self.sim.reset()
        self.t = 0
        self.last_a = torch.zeros(self.B, 2, device=self.dev, dtype=self.dtype)
        self.fb_hist = torch.full((self.B, MAX_OBS_DELAY), NAN, device=self.dev, dtype=self.dtype)
        self.fb_seen = torch.full((self.B,), NAN, device=self.dev, dtype=self.dtype)
        self.cur_err = torch.full((self.B, self.T + LOOKAHEAD), NAN, device=self.dev, dtype=self.dtype)
        z = lambda dt=self.dtype: torch.zeros(self.B, self.T, device=self.dev, dtype=dt)  # noqa: E731
        self.log = {"reward": z(), "measured": z(), "sounding": z(torch.bool), "overblown": z(torch.bool), "action": torch.zeros(self.B, self.T, 2, device=self.dev, dtype=self.dtype)}

    def window(self) -> torch.Tensor:
        return self.target_pad[:, self.t:self.t + LOOKAHEAD]

    def step(self, a: torch.Tensor) -> bool:
        a = torch.clamp(a.to(self.dtype), -1.0, 1.0)
        s = self.sim.step(a[:, 0], a[:, 1])
        tgt = self.target[:, self.t]
        active = torch.isfinite(tgt)
        meas = s["measured"]
        heard = torch.isfinite(meas)
        snd = s["sounding"]
        zero = torch.zeros_like(tgt)
        pitch = torch.where(active & snd & heard, -torch.clamp((meas - tgt).abs(), max=PITCH_CLIP) / 100.0, zero)
        octave = torch.where(active & snd & heard & s["overblown"], zero - OCTAVE_PENALTY, zero)
        silence = torch.where(active & ~snd, zero - SILENCE_PENALTY, zero)
        rest = torch.where(~active & snd, zero - REST_PENALTY, zero)
        smooth = -SMOOTH_WEIGHT * ((a - self.last_a) ** 2).sum(1)
        err = torch.where(heard & active, meas - tgt, torch.full_like(tgt, NAN))
        buf = torch.cat([err[:, None], self.fb_hist], dim=1)
        self.fb_seen = buf.gather(1, self.p.obs_delay[:, None]).squeeze(1)
        self.fb_hist = buf[:, :MAX_OBS_DELAY]
        self.cur_err[:, self.t] = err
        self.last_a = a
        L = self.log
        L["reward"][:, self.t] = pitch + octave + silence + rest + smooth
        L["measured"][:, self.t] = meas
        L["sounding"][:, self.t] = snd
        L["overblown"][:, self.t] = s["overblown"]
        L["action"][:, self.t] = a
        self.state = s
        self.t += 1
        if self.t < self.T:
            return False
        self.logs.append(self.log)
        if self.take < self.takes - 1:
            self.prev_err = self.cur_err
            self.take += 1
            self._start_take()
            return False
        return True

    def metrics(self) -> torch.Tensor:
        """(B, takes, 4): mean |cents|, mean reward per step, sounding rate in notes, sounding inside short gaps."""
        out = []
        active = torch.isfinite(self.target)
        n_in = self.inside.sum(1).clamp_min(1)
        for L in self.logs:
            valid = active & L["sounding"] & torch.isfinite(L["measured"]) & self.inside
            dev = torch.where(valid, (L["measured"] - self.target).abs(), torch.zeros_like(self.target))
            cents = dev.sum(1) / valid.sum(1).clamp_min(1)
            cents = torch.where(valid.any(1), cents, torch.full_like(cents, NAN))
            rew = (L["reward"] * self.inside).sum(1) / n_in
            snd = (L["sounding"] & active).sum(1) / active.sum(1).clamp_min(1)
            gaps = self.inner_rest.sum(1)
            leak = (L["sounding"] & self.inner_rest).sum(1) / gaps.clamp_min(1)
            leak = torch.where(gaps > 0, leak.to(self.dtype), torch.full_like(cents, NAN))
            out.append(torch.stack([cents, rew, snd.to(self.dtype), leak], 1))
        return torch.stack(out, 1)


# ---------------------------------------------------------------- rig identification
def _trace(pwm: torch.Tensor, v_in: torch.Tensor, v_out: torch.Tensor) -> torch.Tensor:
    """Dead-reckoned plunger position after each step (nominal delay, dead band and lag; fitted speeds)."""
    N, T = pwm.shape
    x = torch.zeros(N, dtype=pwm.dtype, device=pwm.device)
    v = torch.zeros_like(x)
    q = [torch.zeros_like(x) for _ in range(NOM.cmd_delay)]
    k, db = min(1.0, DT / NOM.tau_v), NOM.deadband
    out = torch.empty_like(pwm)
    for t in range(T):
        q.append(pwm[:, t])
        u = q.pop(0)
        mag = u.abs()
        drive = torch.where(mag < db, torch.zeros_like(u), torch.sign(u) * (mag - db) / (1.0 - db))
        v = v + (drive * torch.where(drive > 0, v_in, v_out) - v) * k
        x = x + v * DT
        low, high = x <= 0.0, x >= NOM.stroke
        v = torch.where(low, torch.clamp(v, min=0.0), torch.where(high, torch.clamp(v, max=0.0), v))
        x = torch.clamp(x, 0.0, NOM.stroke)
        out[:, t] = x
    return out


def _predict(z: torch.Tensor, pwm: torch.Tensor) -> torch.Tensor:
    x = _trace(pwm, NOM.v_max_in * torch.exp(z[:, 0]), NOM.v_max_out * torch.exp(z[:, 1]))
    length = torch.clamp(NOM.tube_len + z[:, 2:3] - x + NOM.end_corr, min=0.02)
    c = speed_of_sound(torch.tensor(NOM.temp_c, dtype=pwm.dtype))
    return hz_to_cents(c / (4.0 * length)) + 100.0 * z[:, 3:4]


def batch_fit(pwm: torch.Tensor, meas: torch.Tensor, iters: int = 6, reg: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """flute_rl.adapt.fit_rig for a batch (one take per rig, starting from nominal). Returns (z (B,4), rms (B,), NaN = failed)."""
    pwm, meas = pwm.double(), meas.double()
    B, T = pwm.shape
    ok = torch.isfinite(meas)
    m0 = torch.where(ok, meas, torch.zeros_like(meas))

    def res(zz, p):
        return (_predict(zz, p) - m0.repeat(len(zz) // B, 1)) / 100.0

    # coarse grid over the two speeds (same order and tie-breaking as the numpy version)
    g = torch.log(torch.linspace(0.7, 1.3, 13, dtype=torch.float64, device=pwm.device))
    grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)  # a outer, b inner
    G = len(grid)
    zg = torch.zeros(G * B, 4, dtype=torch.float64, device=pwm.device)
    zg[:, :2] = grid.repeat_interleave(B, 0)
    r = res(zg, pwm.repeat(G, 1)).reshape(G, B, T)
    okg = ok[None].expand(G, B, T)
    loss = torch.where(okg, torch.clamp(r.abs(), max=6.0) ** 2, torch.zeros_like(r)).sum(2) / ok.sum(1).clamp_min(1)
    best = loss.argmin(0)
    z = torch.zeros(B, 4, dtype=torch.float64, device=pwm.device)
    has = ok.any(1)
    z[:, :2] = torch.where(has[:, None], grid[best], z[:, :2])

    eps = torch.tensor([1e-3, 1e-3, 1e-5, 1e-3], dtype=torch.float64, device=pwm.device)
    prior = torch.tensor(PRIOR_STD, dtype=torch.float64, device=pwm.device) ** 2
    rms = torch.full((B,), NAN, dtype=torch.float64, device=pwm.device)
    live = torch.ones(B, dtype=torch.bool, device=pwm.device)
    gates = np.maximum(np.geomspace(1500.0, OUTLIER_CENTS, max(iters, 2)), OUTLIER_CENTS)
    for gate in gates:
        r = res(z, pwm)
        inl = ok & (r.abs() < gate / 100.0)
        n = inl.sum(1)
        live = live & (n >= 10)
        if not live.any():
            break
        zs = torch.cat([z + eps[i] * torch.eye(4, dtype=torch.float64, device=pwm.device)[i] for i in range(4)])
        J = ((res(zs, pwm.repeat(4, 1)).reshape(4, B, T) - r[None]) / eps[:, None, None]).permute(1, 2, 0)  # (B,T,4)
        w8 = inl.to(torch.float64)
        Jm = J * w8[..., None]
        rm = torch.where(inl, r, torch.zeros_like(r))
        wreg = reg * n.to(torch.float64) / 200.0
        H = Jm.transpose(1, 2) @ Jm + wreg[:, None, None] * torch.diag(1.0 / prior)[None]
        gvec = (Jm.transpose(1, 2) @ rm[..., None]).squeeze(-1) + wreg[:, None] * z / prior
        step = torch.linalg.solve(H, gvec[..., None]).squeeze(-1)
        z = torch.where(live[:, None], z - step, z)
        rms = torch.where(live, torch.sqrt((rm ** 2).sum(1) / n.clamp_min(1)) * 100.0, rms)
        if gate <= OUTLIER_CENTS:
            live = live & ~(step.abs() < eps).all(1)
    return z, rms


# ---------------------------------------------------------------- networks, one per rig
def _split(theta: torch.Tensor, shapes: list[tuple]) -> list[torch.Tensor]:
    parts, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        parts.append(theta[:, i:i + n].reshape(theta.shape[0], *s))
        i += n
    return parts


class BatchMLP:
    """MLP with per-row weights (flat layout of flute_rl.policy.MLP)."""

    def __init__(self, theta: torch.Tensor, in_dim: int, out_dim: int, hidden: int):
        self.w1, self.b1, self.w2, self.b2 = _split(theta, [(in_dim, hidden), (hidden,), (hidden, out_dim), (out_dim,)])

    def reset_state(self) -> None:
        pass

    def __call__(self, x: torch.Tensor, live: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.tanh(torch.bmm(x[:, None], self.w1)[:, 0] + self.b1)
        return torch.tanh(torch.bmm(h[:, None], self.w2)[:, 0] + self.b2)


class BatchGRU:
    """GRU with per-row weights (flat layout of flute_rl.policy.GRU)."""

    def __init__(self, theta: torch.Tensor, in_dim: int, out_dim: int, hidden: int):
        H = hidden
        self.H = H
        self.W, self.U, self.b, self.Wo, self.bo = _split(theta, [(in_dim, 3 * H), (H, 3 * H), (3 * H,), (H + in_dim, out_dim), (out_dim,)])
        self.reset_state()

    def reset_state(self) -> None:
        self.h = torch.zeros(self.W.shape[0], self.H, dtype=self.W.dtype, device=self.W.device)

    def __call__(self, x: torch.Tensor, live: torch.Tensor | None = None) -> torch.Tensor:
        """`live` (B,) bool: rows whose piece is over keep their memory unchanged (padding steps)."""
        H, h = self.H, self.h
        gx = torch.bmm(x[:, None], self.W)[:, 0] + self.b
        gh = torch.bmm(h[:, None], self.U)[:, 0]
        z = torch.sigmoid(gx[:, :H] + gh[:, :H])
        r = torch.sigmoid(gx[:, H:2 * H] + gh[:, H:2 * H])
        n = torch.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
        new = z * h + (1.0 - z) * n
        self.h = new if live is None else torch.where(live[:, None], new, h)
        return torch.tanh(torch.bmm(torch.cat([new, x], 1)[:, None], self.Wo)[:, 0] + self.bo)


# ---------------------------------------------------------------- the controller
class BatchAgent:
    """FeedbackPolicy (physics prior + rig identification + ILC + live feedback) + optional residual network."""

    def __init__(self, env: BatchFluteEnv, fb_gain: float = 0.05, ilc_gain: float = 0.5, reg: float = 1.0,
                 fit_iters: int = 6, net=None, scale: float = 0.3, horizon: int = 30, history: int = 0,
                 track_time: float = 0.04, angle_lead: int = 3, rest_angle: float = -1.0, max_err: float = 300.0):
        self.env, self.net, self.scale, self.horizon, self.history = env, net, scale, horizon, history
        self.fb_gain, self.ilc_gain, self.reg, self.fit_iters = fb_gain, ilc_gain, reg, fit_iters
        self.track_time, self.angle_lead, self.rest_angle, self.max_err = track_time, angle_lead, rest_angle, max_err
        B, dev, dt = env.B, env.dev, env.dtype
        self.lead = min(NOM.cmd_delay + int(round(NOM.tau_v / DT)) + 1, LOOKAHEAD - 1)
        self.z = torch.zeros(B, 4, device=dev, dtype=dt)
        self.fit_ok = torch.zeros(B, dtype=torch.bool, device=dev)
        self.fitted = False
        self.offset = torch.zeros(B, env.T + LOOKAHEAD, device=dev, dtype=dt)
        self.pwm_log = torch.zeros(B, env.T, device=dev, dtype=dt)
        self.last = torch.zeros(B, 2, device=dev, dtype=dt)
        self.hist = torch.zeros(B, history, 3, device=dev, dtype=dt)
        self.x_hat = torch.zeros(B, device=dev, dtype=dt)
        self.v_hat = torch.zeros_like(self.x_hat)
        self.q = [torch.zeros_like(self.x_hat) for _ in range(NOM.cmd_delay)]
        if net is not None:
            net.reset_state()

    # model of the controller (nominal, then the identified rig)
    def _v_in(self):
        return NOM.v_max_in * torch.exp(self.z[:, 0])

    def _v_out(self):
        return NOM.v_max_out * torch.exp(self.z[:, 1])

    def _tube(self):
        return NOM.tube_len + self.z[:, 2]

    def _length(self, x):
        return torch.clamp(self._tube() - x + NOM.end_corr, min=0.02)

    def model_cents(self, x):
        c = speed_of_sound(torch.tensor(NOM.temp_c, dtype=x.dtype, device=x.device))
        return hz_to_cents(c / (4.0 * self._length(x))) + 100.0 * self.z[:, 3]

    def _x_for_cents(self, cents):
        f = 440.0 * torch.pow(2.0, cents / 1200.0)
        c = speed_of_sound(torch.tensor(NOM.temp_c, dtype=cents.dtype, device=cents.device))
        return self._tube() + NOM.end_corr - c / (4.0 * f)

    def _fit(self) -> None:
        env = self.env
        measured = env.target_pad + env.prev_err  # the recording of the take just played
        z, rms = batch_fit(self.pwm_log, measured[:, :env.T], iters=self.fit_iters, reg=self.reg)
        self.z = z.to(env.dtype)
        self.fit_ok = torch.isfinite(rms)
        self.fitted = True

    def _features(self, win: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        env, h = self.env, self.horizon
        tgt = win[:, :h]
        mask = torch.isfinite(tgt)
        err = torch.where(mask, torch.clamp((tgt - self.model_cents(self.x_hat)[:, None]) / 100.0, -3.0, 3.0), torch.zeros_like(tgt))
        valid = torch.isfinite(env.fb_seen)
        fb = torch.where(valid, torch.clamp(env.fb_seen / 100.0, -3.0, 3.0), torch.zeros_like(env.fb_seen))
        take = torch.full_like(fb, env.take / (env.takes - 1) if env.takes > 1 else 0.0)
        rng = env.p.angle_range_deg
        angle_off = env.state["theta_readback"] / rng - env.angle_comp / rng
        if self.history:
            rolled = torch.roll(self.hist, -1, 1)
            rolled[:, -1] = torch.stack([fb, self.last[:, 0], self.last[:, 1]], 1)
            self.hist = torch.where(live[:, None, None], rolled, self.hist)
        cols = [err, mask.to(err.dtype), torch.stack([self.v_hat / 0.15, angle_off, fb, valid.to(err.dtype), take], 1),
                self.last, torch.ones_like(fb)[:, None]]
        if self.history:
            cols.append(self.hist.reshape(env.B, -1))
        return torch.cat(cols, 1)

    def act(self) -> torch.Tensor:
        env = self.env
        t = env.t
        win = env.window()
        if t == 0:
            self.x_hat = torch.zeros_like(self.x_hat)
            self.v_hat = torch.zeros_like(self.v_hat)
        live = t < env.length  # padding steps after a shorter piece must not change any memory
        out = None
        if self.net is not None:
            out = self.net(self._features(win, live), live)
            gain_scale = 1.0 + out[:, 2]
        else:
            gain_scale = torch.ones_like(self.x_hat)

        # live feedback: move the plunger belief by what the delayed pitch error says
        if t > 0:
            e = env.fb_seen
            use = torch.isfinite(e) & (torch.clamp(e, -self.max_err, self.max_err).abs() < self.max_err)
            e0 = torch.where(use, e, torch.zeros_like(e))
            dx = e0 * self._length(self.x_hat) * math.log(2.0) / 1200.0
            self.x_hat = torch.clamp(self.x_hat + self.fb_gain * gain_scale * dx * use, 0.0, NOM.stroke)

        # between takes: identify the rig from the take just played (first time only)
        if t == 0 and env.take > 0 and not self.fitted:
            self._fit()
        if t == 0:
            self.q = [torch.zeros_like(self.x_hat) for _ in range(NOM.cmd_delay)]

        # ILC on the previous take's error (not on take 2 when the fit explains take 1)
        i = min(self.lead, LOOKAHEAD - 1)
        pe = torch.clamp(env.prev_err[:, t + i], -1800.0, 1800.0)
        use_ilc = torch.isfinite(pe)
        if env.take == 1:
            use_ilc = use_ilc & ~self.fit_ok
        self.offset[:, t + i] += torch.where(use_ilc, self.ilc_gain * pe, torch.zeros_like(pe))

        # physics prior: aim at the first note at/after the lead time
        mask = torch.isfinite(win)
        aim = win - 100.0 * self.z[:, 3:4] - self.offset[:, t:t + LOOKAHEAD]
        m2 = mask[:, self.lead:]
        has = m2.any(1)
        first = self.lead + m2.to(torch.int8).argmax(1)
        anyw = mask.any(1)
        last = LOOKAHEAD - 1 - torch.flip(mask, [1]).to(torch.int8).argmax(1)
        idx = torch.where(has, first, last)
        c_aim = aim.gather(1, idx[:, None]).squeeze(1)
        x_des = torch.where(has | anyw, self._x_for_cents(torch.nan_to_num(c_aim)), self.x_hat)
        x_des = torch.clamp(x_des, 0.0, NOM.stroke)
        v_in, v_out = self._v_in(), self._v_out()
        v_des = torch.maximum(torch.minimum((x_des - self.x_hat) / self.track_time, v_in), -v_out)
        drive = v_des / torch.where(v_des > 0, v_in, v_out)
        pwm = torch.where(drive.abs() < 0.02, torch.zeros_like(drive), torch.sign(drive) * (NOM.deadband + (1.0 - NOM.deadband) * drive.abs()))
        soon = mask[:, min(self.angle_lead, LOOKAHEAD - 1)] | mask[:, 0]
        angle = torch.where(soon, env.angle_comp / env.p.angle_range_deg, torch.full_like(pwm, self.rest_angle))
        a = torch.stack([pwm, angle], 1)
        if out is not None:
            a = torch.clamp(a + self.scale * out[:, :2], -1.0, 1.0)

        # dead-reckon the PWM actually sent
        self.q.append(a[:, 0])
        u = self.q.pop(0)
        mag = u.abs()
        drv = torch.where(mag < NOM.deadband, torch.zeros_like(u), torch.sign(u) * (mag - NOM.deadband) / (1.0 - NOM.deadband))
        self.v_hat = self.v_hat + (drv * torch.where(drv > 0, v_in, v_out) - self.v_hat) * min(1.0, DT / NOM.tau_v)
        self.x_hat = torch.clamp(self.x_hat + self.v_hat * DT, 0.0, NOM.stroke)
        self.pwm_log[:, t] = a[:, 0]
        self.last = torch.where(live[:, None], a, self.last)
        return a


def run_episodes(env: BatchFluteEnv, agent: BatchAgent) -> torch.Tensor:
    """Play every take; returns env.metrics()."""
    done = False
    with torch.no_grad():
        while not done:
            done = env.step(agent.act())
    return env.metrics()
