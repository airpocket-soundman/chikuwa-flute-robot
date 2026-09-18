"""Yamabiko training sessions in PyTorch (GPU).

The same rig (rig.Rig), homing and valve schedule (session.run_session), recurrent
policy (control.GRUPolicy with carry=True) and reward, written as tensor
operations over all candidates x rigs at once, for evolution strategies.

The numpy stack stays the reference. With the random effects of the rig switched
off the two give the same rewards (tests/test_yamabiko_torch.py); with them on,
the random numbers differ and the two agree statistically.

On CUDA the whole control step (policy, rig, policy update, reward) is captured
once as a CUDA graph and replayed, so one step costs one launch whatever the
batch size. Everything that changes during a step lives in fixed tensors; the
step index is a tensor that the graph advances itself. Rig swaps (a few per
session) are applied outside the graph, in place.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from ..sim import DT
from .control import _C_NOM, _L_NOM, D_NOM, STROKE, GRUPolicy
from .learn import TaskSpec, song_weights
from .rig import INTS, NOMINAL, OVERBLOW_CENTS, QUEUE, RigParams
from .session import PITCH_CLIP, SILENCE_PENALTY, Schedule, Swap, draw_pieces, make_schedule

FIELDS = [f.name for f in dataclasses.fields(RigParams)]
_DB = NOMINAL["deadband"]


def _hz_to_cents(f):
    return 1200.0 * torch.log2(f / 440.0)


def _nominal_cents(x):
    return _hz_to_cents(_C_NOM / (4.0 * torch.clamp(_L_NOM - x, min=0.02)))


def _nominal_x(cents):
    return _L_NOM - _C_NOM / (4.0 * 440.0 * 2.0 ** (cents / 1200.0))


def _pwm_for(x_aim, xb, s_in, s_out):
    v_in = NOMINAL["v_in"] * s_in
    v_out = NOMINAL["v_out"] * s_out
    v_des = torch.minimum(torch.maximum((x_aim - xb) / 0.04, -v_out), v_in)
    drive = torch.where(v_des > 0, v_des / v_in, v_des / v_out)
    pwm = torch.sign(drive) * torch.clamp(_DB + (1.0 - _DB) * drive.abs(), max=1.0)
    return torch.where(drive.abs() < 0.02, 0.0, pwm)


def _shift(q, new):
    """Delay line: q[:, 0] is the newest value."""
    q.copy_(torch.cat([new[:, None], q[:, :-1]], dim=1))


class Player:
    """P candidates x R rigs playing sessions of K songs up to T steps long."""

    def __init__(self, P: int, R: int, K: int, hidden: int, T: int, device="cuda", dtype=torch.float32,
                 graph: bool | None = None):
        self.P, self.R, self.K, self.H, self.T = P, R, K, hidden, T
        self.n = n = P * R
        self.dev = dev = torch.device(device)
        self.dt = dt = dtype
        self.eps = 1e-9 if dtype == torch.float64 else 1e-6

        def z(*shape, dtype=dt):
            return torch.zeros(shape, device=dev, dtype=dtype)

        self.prm = {k: z(n, dtype=torch.long if k in INTS else dt) for k in FIELDS}
        self.w = [z(P, *s) for s in GRUPolicy.shapes(hidden)]
        self.out_scale = torch.tensor(GRUPolicy.OUT_SCALE, device=dev, dtype=dt)
        self.c = {k: torch.tensor(v, device=dev, dtype=dt)  # constants (no host copies inside the graph)
                  for k, v in (("v_in", NOMINAL["v_in"]), ("v_out", NOMINAL["v_out"]), ("up", 1200.0), ("down", -1200.0))}
        # schedule, one row per rig (the same for every candidate)
        self.aim, self.target, self.valve = z(R, T), z(R, T), z(R, T, dtype=torch.bool)
        self.weight = z(R, T)  # reward weight per step (the first note of a song can count more)
        self.home, self.homed_at = z(T, dtype=torch.bool), z(T, dtype=torch.bool)
        self.song = z(T, dtype=torch.long)
        self.karange = torch.arange(K, device=dev)
        self.t = z(1, dtype=torch.long)
        # rig
        self.x_motor, self.x, self.v, self.u_last, self.speed, self.jit = (z(n) for _ in range(6))
        self.pwm_q, self.meas_q = z(n, QUEUE), z(n, QUEUE)
        self.valve_q = z(n, QUEUE, dtype=torch.bool)
        self.air_steps, self.over, self.charge = z(n, dtype=torch.long), z(n, dtype=torch.bool), z(n)
        # policy: dead reckoning, GRU, listening
        self.dv, self.t_in, self.t_out, self.last_pwm, self.pulse = (z(n) for _ in range(5))
        self.dq, self.hist = z(n, QUEUE), z(n, QUEUE)
        self.h, self.out, self.heard_feat = z(P, R, hidden), z(n, GRUPolicy.OUT), z(n, 3)
        self.run = z(n, dtype=torch.long)
        self.reward, self.steps = z(n, K), z(n, K)

        self.graph = dev.type == "cuda" if graph is None else graph
        self._g = None
        if self.graph:
            self._capture()

    # ----------------------------------------------------------------- set-up
    def _capture(self) -> None:
        self.target.fill_(float("nan"))
        self.aim.fill_(float("nan"))
        side = torch.cuda.Stream(self.dev)
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream(self.dev).wait_stream(side)
        self._g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._g):
            self._step()

    def _reset_rig(self, mask=None) -> None:
        """Power on, as rig.Rig.reset (all rigs, or only where `mask`)."""
        if mask is None:
            for a in (self.x_motor, self.x, self.v, self.u_last, self.jit, self.pwm_q, self.air_steps, self.charge):
                a.zero_()
            self.speed.fill_(1.0)
            self.over.fill_(False)
            self.valve_q.fill_(False)
            self.meas_q.fill_(float("nan"))
            return
        for a in (self.x_motor, self.x, self.v, self.u_last, self.jit, self.air_steps, self.charge):
            a.masked_fill_(mask, 0)
        self.speed.masked_fill_(mask, 1.0)
        self.over.masked_fill_(mask, False)
        self.pwm_q.masked_fill_(mask[:, None], 0.0)
        self.valve_q.masked_fill_(mask[:, None], False)
        self.meas_q.masked_fill_(mask[:, None], float("nan"))

    def _params(self, params: RigParams) -> dict:
        return {k: torch.as_tensor(getattr(params, k), device=self.dev,
                                   dtype=torch.long if k in INTS else self.dt) for k in FIELDS}

    def load(self, thetas: np.ndarray, params: RigParams, sched: Schedule, first_weight: float = 1.0) -> None:
        """thetas (P, D); params for all P x R rigs (candidate p uses rows p*R .. p*R+R-1); sched has R rows."""
        thetas = torch.as_tensor(np.atleast_2d(thetas), device=self.dev, dtype=self.dt)
        i = 0
        for w in self.w:
            size = math.prod(w.shape[1:])
            w.copy_(thetas[:, i:i + size].reshape(w.shape))
            i += size
        if i != thetas.shape[1]:
            raise ValueError(f"expected {i} parameters, got {thetas.shape[1]}")
        for k, v in self._params(params).items():
            self.prm[k].copy_(v)
        if sched.n != self.R or sched.k != self.K or sched.T > self.T:
            raise ValueError("schedule does not fit this player")
        T = sched.T

        def put(buf, arr, fill):
            buf.fill_(fill)
            buf[..., :T].copy_(torch.as_tensor(arr, device=self.dev, dtype=buf.dtype))

        put(self.aim, sched.aim, float("nan"))
        put(self.target, sched.target, float("nan"))
        put(self.valve, sched.valve, False)
        put(self.weight, 1.0 + (first_weight - 1.0) * sched.first, 1.0)
        put(self.home, sched.homing, False)
        put(self.song, sched.song, 0)
        homed = sched.homing & ~np.concatenate([sched.homing[1:], [False]])
        put(self.homed_at, homed, False)
        self.t.zero_()
        self._reset_rig()
        for a in (self.dv, self.t_in, self.t_out, self.last_pwm, self.pulse, self.dq, self.hist, self.h, self.out,
                  self.heard_feat, self.run, self.reward, self.steps):
            a.zero_()

    def play(self, thetas: np.ndarray, params: RigParams, sched: Schedule, swap: Swap | None = None,
             first_weight: float = 1.0) -> np.ndarray:
        """Play one session. Returns the per-song reward per rig, (P*R, K), as session.run_session."""
        self.load(thetas, params, sched, first_weight)
        starts = {s: k for k, s in enumerate(sched.starts)}
        new = self._params(swap.params) if swap is not None else None
        for t in range(sched.T):
            k = starts.get(t)
            if swap is not None and k == swap.song and k > 0:
                mask = torch.as_tensor(swap.mask, device=self.dev, dtype=torch.bool)
                for key, v in new.items():
                    self.prm[key].copy_(torch.where(mask, v, self.prm[key]))
                self._reset_rig(mask)
            if self._g is not None:
                self._g.replay()
            else:
                self._step()
        return (self.reward / torch.clamp(self.steps, min=1.0)).cpu().numpy().astype(float)

    # ------------------------------------------------------------------ step
    def _position(self, out):
        s = out * self.out_scale
        return torch.clamp((1.0 + s[:, 0]) * self.t_in - (1.0 + s[:, 1]) * self.t_out + s[:, 2], 0.0, STROKE)

    def _step(self) -> None:
        P, R, H, n, dt = self.P, self.R, self.H, self.n, self.dt
        p, dev = self.prm, self.dev
        t = self.t
        home = self.home.index_select(0, t)          # (1,)
        homed = self.homed_at.index_select(0, t)
        song = self.song.index_select(0, t)
        aim_raw = self.aim.index_select(1, t)[:, 0].repeat(P)
        valve_t = self.valve.index_select(1, t)[:, 0].repeat(P)
        tgt = self.target.index_select(1, t)[:, 0].repeat(P)
        wgt = self.weight.index_select(1, t)[:, 0].repeat(P)

        # policy acts (control.GRUPolicy.act); ignored while homing
        xb_prev = self._position(self.out)
        finite = torch.isfinite(aim_raw)
        aim = torch.where(finite, torch.clamp(torch.nan_to_num(_nominal_x(aim_raw)), 0.0, STROKE), xb_prev)
        feats = torch.cat([
            self.heard_feat,
            torch.stack([self.t_in / 0.1, self.t_out / 0.1, self.dv / 0.15, self.last_pwm, (aim - 0.05) / 0.05,
                         torch.clamp((aim - xb_prev) / 0.02, -3.0, 3.0), valve_t.to(dt),
                         torch.clamp(self.run, max=10).to(dt) / 10.0], dim=1),
            self.out, self.pulse[:, None],
        ], dim=1).view(P, R, GRUPolicy.FEATURES)
        W, U, b, Wo, bo = self.w
        gx = torch.bmm(feats, W) + b[:, None, :]
        gh = torch.bmm(self.h, U)
        zg = torch.sigmoid(gx[..., :H] + gh[..., :H])
        rg = torch.sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H])
        cand = torch.tanh(gx[..., 2 * H:] + rg * gh[..., 2 * H:])
        h_new = (1.0 - zg) * cand + zg * self.h
        out_new = torch.tanh(torch.bmm(torch.cat([h_new, feats], dim=2), Wo) + bo[:, None, :]).reshape(n, -1)
        xb = self._position(out_new)
        s = out_new * self.out_scale
        pwm_act = _pwm_for(torch.where(finite, aim, xb), xb, 1.0 + s[:, 0], 1.0 + s[:, 1])
        self.h.copy_(torch.where(home, self.h, h_new))
        self.out.copy_(torch.where(home, self.out, out_new))
        self.pulse.copy_(torch.where(home, self.pulse, 0.0))
        pwm = torch.where(home, -1.0, pwm_act)
        valve = valve_t & ~home

        # rig (rig.Rig.step)
        _shift(self.pwm_q, torch.clamp(pwm, -1.0, 1.0))
        u = self.pwm_q.gather(1, p["cmd_delay"][:, None])[:, 0]
        late = torch.rand(n, device=dev, dtype=dt) < p["delay_jitter"]
        u = torch.where(late, self.u_last, u)
        self.u_last.copy_(u)
        mag = u.abs()
        threshold = p["deadband"] + torch.where(self.v.abs() < 0.005, p["stiction"], 0.0)
        drive = torch.where(mag < threshold, 0.0, torch.sign(u) * (mag - p["deadband"]) / (1.0 - p["deadband"]))
        drive = torch.sign(drive) * drive.abs() ** p["pwm_curve"]
        v_max = torch.where(drive > 0, p["v_in"] * (1.0 - p["load_slope"] * torch.clamp(self.x_motor / p["stroke"], 0.0, 1.0)),
                            p["v_out"])
        self.speed.copy_(torch.clamp(self.speed + torch.randn(n, device=dev, dtype=dt) * p["speed_drift"] * math.sqrt(DT),
                                     0.8, 1.2))
        v = self.v + (drive * v_max * self.speed - self.v) * torch.clamp(DT / p["tau_v"], max=1.0)
        moved = v * (1.0 + torch.randn(n, device=dev, dtype=dt) * p["motion_noise"])
        xm = self.x_motor + moved * DT
        lo, hi = xm <= 0.0, xm >= p["stroke"]
        xm = torch.minimum(torch.clamp(xm, min=0.0), p["stroke"])
        self.v.copy_(torch.where(lo, torch.clamp(v, min=0.0), torch.where(hi, torch.clamp(v, max=0.0), v)))
        self.x_motor.copy_(xm)
        half = p["backlash"] / 2.0
        self.x.copy_(torch.where(xm - self.x > half, xm - half, torch.where(self.x - xm > half, xm + half, self.x)))

        _shift(self.valve_q, valve)
        flow = self.valve_q.gather(1, p["valve_delay"][:, None])[:, 0]
        self.air_steps.copy_(torch.where(flow, self.air_steps + 1, 0))
        t_air = self.air_steps.to(dt) * DT
        sounding = flow & (t_air >= p["onset_s"] - self.eps)
        k = torch.clamp(DT / p["surge_tau"], max=1.0)
        self.charge.copy_(torch.where(flow, self.charge * (1.0 - k), self.charge + (1.0 - self.charge) * k))
        length = torch.clamp(p["tube_len"] + p["end_corr"] - self.x, min=0.02)
        self.over.copy_(sounding & torch.where(self.over, length < p["overblow_len"] + 0.002, length < p["overblow_len"]))
        self.jit.copy_(self.jit - self.jit * DT / 0.05
                       + p["pitch_jitter"] * math.sqrt(2.0 * DT / 0.05) * torch.randn(n, device=dev, dtype=dt))
        transient = torch.where(sounding, p["onset_cents"] * torch.exp(-torch.clamp(t_air - p["onset_s"], min=0.0) / p["onset_tau"]),
                                0.0)
        cents = (_hz_to_cents((331.3 + 0.606 * p["temp_c"]) / (4.0 * length)) + p["press_cents"]
                 + OVERBLOW_CENTS * self.over.to(dt) + self.jit + transient
                 + torch.where(sounding, p["surge_cents"] * self.charge, 0.0))
        detected = sounding & (torch.rand(n, device=dev, dtype=dt) >= p["dropout"])
        measured = torch.where(detected, cents + p["pitch_noise"] * torch.randn(n, device=dev, dtype=dt), float("nan"))
        octave = detected & (torch.rand(n, device=dev, dtype=dt) < p["octave_err"])
        sign = torch.where(torch.rand(n, device=dev, dtype=dt) < 0.5, self.c["up"], self.c["down"])
        measured = measured + torch.where(octave, sign, 0.0)
        _shift(self.meas_q, measured)
        heard = self.meas_q.gather(1, p["obs_delay"][:, None])[:, 0]

        # policy listens (DeadReckoner.advance + GRUPolicy.update)
        _shift(self.dq, pwm)
        u2 = self.dq[:, NOMINAL["cmd_delay"]]
        mag2 = u2.abs()
        drive2 = torch.where(mag2 < _DB, 0.0, torch.sign(u2) * (mag2 - _DB) / (1.0 - _DB))
        v_nom = torch.where(drive2 > 0, self.c["v_in"], self.c["v_out"])
        self.dv.copy_(self.dv + (drive2 * v_nom - self.dv) * min(1.0, DT / NOMINAL["tau_v"]))
        d = self.dv * DT
        self.t_in.add_(torch.clamp(d, min=0.0))
        self.t_out.add_(torch.clamp(-d, min=0.0))
        self.last_pwm.copy_(pwm)
        _shift(self.hist, self._position(self.out))
        valid = torch.isfinite(heard)
        self.run.copy_(torch.where(valid, self.run + 1, 0))
        hh = torch.nan_to_num(heard, nan=0.0)
        pred = self.hist[:, D_NOM]
        hh = torch.where(hh - _nominal_cents(pred) > 0.5 * (OVERBLOW_CENTS + 1200.0), hh - OVERBLOW_CENTS, hh)
        xh = _nominal_x(hh)
        innov = torch.clamp((xh - pred) / 0.005, -3.0, 3.0)
        self.heard_feat.copy_(torch.stack([valid.to(dt), torch.where(valid, torch.clamp((xh - 0.05) / 0.05, -3.0, 3.0), 0.0),
                                           torch.where(valid, innov, 0.0)], dim=1))

        # end of homing (GRUPolicy.homed)
        for a in (self.dv, self.t_in, self.t_out):
            a.copy_(torch.where(homed, 0.0, a))
        self.dq.copy_(torch.where(homed, 0.0, self.dq))
        self.hist.copy_(torch.where(homed, 0.0, self.hist))
        self.run.copy_(torch.where(homed, 0, self.run))
        self.heard_feat.copy_(torch.where(homed, 0.0, self.heard_feat))
        self.pulse.copy_(torch.where(homed, 1.0, self.pulse))

        # reward (session.run_session)
        note = torch.isfinite(tgt)
        err = torch.clamp(torch.nan_to_num(cents - tgt).abs(), max=PITCH_CLIP) / 100.0
        r = torch.where(note, torch.where(sounding, -err, -SILENCE_PENALTY), 0.0) * wgt
        onehot = (self.karange == song).to(dt)
        self.reward.add_(r[:, None] * onehot)
        self.steps.add_((note.to(dt) * wgt)[:, None] * onehot)
        self.t.add_(1)


_CACHE: dict = {}


def get_player(P: int, R: int, K: int, hidden: int, T: int, device="cuda", dtype=torch.float32) -> Player:
    """A player that fits, reused across generations (capturing the CUDA graph takes a moment)."""
    key = (P, R, K, hidden, str(device), dtype)
    pl = _CACHE.get(key)
    if pl is None or pl.T < T:
        _CACHE.clear()
        pl = _CACHE[key] = Player(P, R, K, hidden, T + 500, device, dtype)
    return pl


def fitness(thetas: np.ndarray, spec: TaskSpec, seed: int, bank=None, device="cuda",
            dtype=torch.float32) -> np.ndarray:
    """learn.fitness on the GPU: the same rigs, songs and swaps for the same seed."""
    thetas = np.atleast_2d(thetas)
    P, R = thetas.shape[0], spec.rigs
    rng = np.random.default_rng(seed)
    base = RigParams.sample(rng, R, spec.spread, spec.harsh)
    sched = make_schedule(draw_pieces(rng, R, spec.songs, bank, spec.progress))
    swap = None
    if spec.swap_prob > 0.0 and spec.songs > 1:
        mask = rng.random(R) < spec.swap_prob
        if mask.any():
            song = int(rng.integers(1, spec.songs))
            swap = Swap(song, np.tile(mask, P), RigParams.sample(rng, R, spec.spread, spec.harsh).tile(P))
    player = get_player(P, R, spec.songs, spec.hidden, sched.T, device, dtype)
    reward = player.play(thetas, base.tile(P), sched, swap, spec.first_weight)
    w = song_weights(spec.songs, spec.weight_growth)
    return (reward @ w / w.sum()).reshape(P, R).mean(axis=1)
