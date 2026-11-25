"""step8: the whole robot as ONE computation graph, one call every 10 ms (PyTorch, exported to ONNX).

    cell = Step8Cell.from_profile(profile)
    state = cell.initial_state()
    out, state = cell(frame, button, shift, heard_cents, heard_valid, use_heard, *state)

Inside, the same networks as the NumPy runtime (flute_rl/yamabiko/step8.py, built from the same exported
arrays): ear -> score memory (written while listening, read back while playing) -> actuator and valve
networks, the conductor (buttons -> idle / listen / play, fan) and the end network.  The mode switch is done
by the graph itself (``torch.where`` on the conductor's choice), the memory W and every recurrent state are
inputs and outputs of the call.  ``scripts/export_step8_onnx.py`` writes it as one ONNX file; the test
checks it against the NumPy runtime step by step.

Differences from the NumPy runtime (all on purpose, none changes the result beyond float rounding):
- every instant the actuator looks ahead to is read from W anew (the NumPy runtime reads each once and keeps
  it); the position code is computed with cos / sin instead of rotating phasors.
- ``heard_cents`` / ``heard_valid`` / ``use_heard`` let the simulator give the pitch while playing, as
  ``Step8.tick(heard=...)`` does; on the rig ``use_heard`` is 0 and the ear hears the frame.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .ear_nn import BIN_CENTS
from .features import WINDOW
from .step8 import END_LOOK, HOMING_STEPS, LEAD_STEPS, MAX_PLAY_STEPS, TAIL_STEPS, WARM_STEPS
from .score_memory import STEP_S

IDLE, LISTEN, PLAY = 0, 1, 2
STATE = ("mode", "end_p", "W", "G", "A", "hi", "lo", "n", "tau", "t", "act_h", "valve_h", "pwm", "open")


def _t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def _mlp(layers, x, last_relu=False):
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1 or last_relu:
            x = torch.relu(x)
    return x


def _logaddexp(a, b):
    m = torch.maximum(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m))


class Step8Cell(nn.Module):
    def __init__(self, ear, memory, actuator, valve, heads, max_cents_above: float = 250.0):
        super().__init__()
        P = lambda name, a: self.register_buffer(name, _t(a))
        # ----- ear
        self.convs = sorted({k.split(".")[1] for k in ear if k.startswith("ear.conv")}, key=lambda s: int(s[4:]))
        self.strides = [int(ear[f"ear.{c}.stride"]) for c in self.convs]
        for c in self.convs:
            P(f"ear_{c}_w", ear[f"ear.{c}.w"]); P(f"ear_{c}_b", ear[f"ear.{c}.b"])
        P("ear_fc0_w", ear["ear.fc0.w"]); P("ear_fc0_b", ear["ear.fc0.b"])
        P("ear_fc2_w", ear["ear.fc2.w"]); P("ear_fc2_b", ear["ear.fc2.b"])
        self.bins = int(ear["ear.bins"]); self.low_cents = float(ear["ear.low_cents"]); self.level = bool(ear["ear.level"])
        # ----- memory
        P("freqs", memory["mem.freqs"])
        self.K = len(memory["mem.freqs"])
        self.mem_low, self.mem_high = float(memory["mem.low"]), float(memory["mem.high"])
        self.beta = float(memory["mem.beta"]) if "mem.beta" in memory else None
        for part in ("enc", "gen", "dec"):
            names = sorted({k[:-2] for k in memory if k.startswith(f"mem.{part}") and k.endswith(".w")},
                           key=lambda s: int(s[len(f"mem.{part}"):]))
            for i, n in enumerate(names):
                P(f"mem_{part}{i}_w", memory[n + ".w"]); P(f"mem_{part}{i}_b", memory[n + ".b"])
            setattr(self, f"n_{part}", len(names))
        self.V = memory[sorted(k for k in memory if k.startswith("mem.enc") and k.endswith(".w"))[-1]].shape[0]
        self.GD = memory[sorted(k for k in memory if k.startswith("mem.gen") and k.endswith(".w"))[-1]].shape[0] - 1 \
            - (1 if self.beta is not None else 0)
        # ----- actuator (twin free, no own valve) and valve networks
        from .residual_numpy import CONSTANTS
        consts = dict(zip(CONSTANTS, np.asarray(actuator["constants"]).tolist()))
        from .features import PITCH_CENTER, PITCH_SCALE
        self.center = float(consts.get("feature_center", PITCH_CENTER)); self.scale = float(consts.get("feature_scale", PITCH_SCALE))
        assert bool(actuator["nn.inputs"][1]) and not np.any(actuator["nn.flags"]), "step8's graph needs a twin-free actuator"
        for k in ("inp.w", "inp.b", "cell.w_ih", "cell.w_hh", "cell.b_ih", "cell.b_hh", "head.0.w", "head.0.b", "head.2.w", "head.2.b"):
            P("act_" + k.replace(".", "_"), actuator["nn." + k])
        for k in ("inp.w", "inp.b", "cell.w_ih", "cell.w_hh", "cell.b_ih", "cell.b_hh", "out.w", "out.b"):
            P("valve_" + k.replace(".", "_"), valve["valve." + k])
        assert bool(valve.get("valve.twin_free", 0)), "step8's graph needs a twin-free valve"
        self.AH, self.VH = actuator["nn.cell.w_hh"].shape[1], valve["valve.cell.w_hh"].shape[1]
        # ----- conductor and end
        for k in ("cond.0.w", "cond.0.b", "cond.2.w", "cond.2.b", "end.0.w", "end.0.b", "end.2.w", "end.2.b"):
            P("s8_" + k.replace(".", "_"), heads["s8." + k])
        self.max_cents_above = max_cents_above
        self.register_buffer("window", torch.tensor(WINDOW, dtype=torch.float32))
        self.register_buffer("end_look", torch.tensor(END_LOOK, dtype=torch.float32))

    @classmethod
    def from_profile(cls, profile):
        load = lambda p: dict(np.load(p))
        return cls(load(profile["ear_runtime_path"]), load(profile["memory_runtime_path"]),
                   load(profile["runtime_path"]), load(profile["valve_runtime_path"]), load(profile["heads_path"]))

    def initial_state(self):
        z = lambda *s: torch.zeros(*s)
        return [z(1), z(1), z(self.V + 1, 2 * self.K), z(self.GD), z(1), torch.full((1,), -1e30), torch.full((1,), -1e30),
                z(1), z(1), z(1), z(self.AH), z(self.VH), z(1), z(1)]

    # ----- parts -------------------------------------------------------------------------------------
    def ear(self, frame):
        x = frame - frame.mean()
        rms = torch.sqrt((x * x).mean())
        h = (x / (rms + 1e-4))[None, None]
        for c, s in zip(self.convs, self.strides):
            h = torch.relu(nn.functional.conv1d(h, getattr(self, f"ear_{c}_w"), getattr(self, f"ear_{c}_b"), stride=s))
        z = h.reshape(-1)
        if self.level:
            z = torch.cat([z, ((20.0 * torch.log10(torch.clamp(rms, min=1e-6)) + 40.0) / 20.0).reshape(1)])
        code = torch.relu(z @ self.ear_fc0_w.T + self.ear_fc0_b)
        out = code @ self.ear_fc2_w.T + self.ear_fc2_b
        logits, v = out[:self.bins], out[self.bins]
        best = torch.argmax(logits)
        idx = best + torch.arange(-2, 3)
        ok = (idx >= 0) & (idx < self.bins)
        lg = torch.where(ok, logits[idx.clamp(0, self.bins - 1)], torch.full((5,), -1e9))
        p = torch.softmax(lg, 0)
        cents = (p * (self.low_cents + BIN_CENTS * idx.to(torch.float32))).sum()
        return code, cents, torch.sigmoid(v)

    def key(self, steps):
        """(len(steps), 2K) position codes of the instants ``steps``."""
        angle = 2 * math.pi * steps[:, None] * STEP_S * self.freqs[None]
        return torch.cat([torch.cos(angle), torch.sin(angle)], 1) / math.sqrt(self.K)

    def mem_mlp(self, part, x):
        return _mlp([(getattr(self, f"mem_{part}{i}_w"), getattr(self, f"mem_{part}{i}_b"))
                     for i in range(getattr(self, f"n_{part}"))], x)

    def read(self, W, G, A, hi, lo, steps):
        """(cents, voice probability, mass) at the instants ``steps`` (S,)."""
        r = self.key(steps) @ W.T                                         # (S, V+1)
        g = G / torch.clamp(A, min=1e-3)
        if self.beta is not None:
            mass = torch.log(torch.clamp(A, min=1e-3))
            g = torch.cat([g, (hi - mass) / self.beta, -(lo - mass) / self.beta])
        out = self.mem_mlp("dec", torch.cat([r, g[None].expand(len(steps), -1)], 1))
        mid, half = (self.mem_low + self.mem_high) / 2, (self.mem_high - self.mem_low) / 2
        return mid + half * out[:, 0], torch.sigmoid(torch.clamp(out[:, 1], -50, 50)), r[:, -1]

    @staticmethod
    def gru(x, h, w_ih, w_hh, b_ih, b_hh):
        gi = w_ih @ x + b_ih; gh = w_hh @ h + b_hh
        n = h.shape[0]
        r = torch.sigmoid(gi[:n] + gh[:n]); z = torch.sigmoid(gi[n:2 * n] + gh[n:2 * n])
        c = torch.tanh(gi[2 * n:] + r * gh[2 * n:])
        return (1 - z) * c + z * h

    # ----- one tick ----------------------------------------------------------------------------------
    def forward(self, frame, button, shift, heard_cents, heard_valid, use_heard,
                mode, end_p, W, G, A, hi, lo, n, tau, t, act_h, valve_h, pwm, open_):
        # conductor: the button, "finished" and the mode before -> this tick's mode and the fan
        x = torch.cat([button, end_p, nn.functional.one_hot(mode.long(), 3).float().reshape(-1)])
        y = torch.relu(x @ self.s8_cond_0_w.T + self.s8_cond_0_b) @ self.s8_cond_2_w.T + self.s8_cond_2_b
        new_mode = torch.argmax(y[:3]).float().reshape(1)
        fan = torch.sigmoid(y[3:4])
        changed = new_mode != mode
        listen, play = new_mode == LISTEN, new_mode == PLAY
        end_p = torch.where(changed, torch.zeros_like(end_p), end_p)
        # entering listen: an empty memory; entering play: the clocks and the networks' states from zero
        start_listen, start_play = changed & listen, changed & play
        W = torch.where(start_listen, torch.zeros_like(W), W)
        G = torch.where(start_listen, torch.zeros_like(G), G)
        A = torch.where(start_listen, torch.zeros_like(A), A)
        hi = torch.where(start_listen, torch.full_like(hi, -1e30), hi)
        lo = torch.where(start_listen, torch.full_like(lo, -1e30), lo)
        n = torch.where(start_listen, torch.zeros_like(n), n)
        tau = torch.where(start_play, torch.zeros_like(tau), tau)
        t = torch.where(start_play, torch.zeros_like(t), t)
        act_h = torch.where(start_play, torch.zeros_like(act_h), act_h)
        valve_h = torch.where(start_play, torch.zeros_like(valve_h), valve_h)
        pwm = torch.where(start_play, torch.zeros_like(pwm), pwm)
        open_ = torch.where(start_play, torch.zeros_like(open_), open_)

        # ear
        code, cents, voiced = self.ear(frame)

        # listening: write the ear's code at instant n
        writing = listen & (n < MAX_PLAY_STEPS)
        k = self.key(n)[0]
        v = torch.cat([self.mem_mlp("enc", code).reshape(-1), torch.ones(1)])
        W = torch.where(writing, W + v[:, None] * k[None], W)
        g = self.mem_mlp("gen", code)
        gd = self.GD
        a = torch.sigmoid(torch.clamp(g[-1:], -50, 50))
        G = torch.where(writing, G + a * g[:gd], G)
        A = torch.where(writing, A + a, A)
        if self.beta is not None:
            la = torch.log(torch.clamp(a, min=1e-12))
            hi = torch.where(writing, _logaddexp(hi, self.beta * g[gd:gd + 1] + la), hi)
            lo = torch.where(writing, _logaddexp(lo, -self.beta * g[gd:gd + 1] + la), lo)
        n = torch.where(writing, n + 1, n)

        # playing: fan spin-up, homing, then the actuator and valve reading the memory
        tau_now = tau
        tau = torch.where(play, tau + 1, tau)
        acting = play & (tau_now >= WARM_STEPS + HOMING_STEPS) & (n >= 10)
        length = LEAD_STEPS + n + TAIL_STEPS
        limit = acting & (t >= length + 4 * TAIL_STEPS)
        acting = acting & ~limit
        # what was heard (the ear, or the simulator's pitch), after the first actuator step
        h_c = torch.where(use_heard > .5, heard_cents, cents.reshape(1))
        h_ok = torch.where(use_heard > .5, (heard_valid > .5).float(), (voiced.reshape(1) >= .5).float()) > .5
        h_ok = h_ok & (t > 0) & (h_c <= self.mem_high + shift + self.max_cents_above)
        heard_n = torch.where(h_ok, (h_c - self.center) / self.scale, torch.zeros_like(h_c))
        # the score at t + WINDOW (the actuator's time: LEAD_STEPS before the song)
        ids = torch.minimum(t + self.window, length - 1)
        u = ids - LEAD_STEPS
        rc, rp, _ = self.read(W, G, A, hi, lo, torch.clamp(u, min=0.0, max=None).minimum(n - 1).clamp(min=0.0))
        sc = (rc + shift - self.center) / self.scale
        sv = ((u >= 0) & (u < n) & (rp >= .5)).float()
        gap = torch.where(h_ok, sc[:1] - heard_n, torch.zeros_like(heard_n))
        now = torch.cat([heard_n, h_ok.float(), gap, pwm])
        xa = torch.cat([sc, sv, now, sv[:1]])
        xa = torch.tanh(self.act_inp_w @ xa + self.act_inp_b)
        act_new = self.gru(xa, act_h, self.act_cell_w_ih, self.act_cell_w_hh, self.act_cell_b_ih, self.act_cell_b_hh)
        hy = act_new @ self.act_head_0_w.T + self.act_head_0_b
        hy = hy * torch.sigmoid(hy)
        pwm_new = torch.tanh(hy @ self.act_head_2_w.T + self.act_head_2_b)
        xv = torch.cat([sc, sv, heard_n, h_ok.float(), gap, pwm_new, open_])
        xv = torch.tanh(self.valve_inp_w @ xv + self.valve_inp_b)
        valve_new = self.gru(xv, valve_h, self.valve_cell_w_ih, self.valve_cell_w_hh, self.valve_cell_b_ih, self.valve_cell_b_hh)
        open_new = torch.sigmoid(valve_new @ self.valve_out_w.T + self.valve_out_b)
        # the end network around the play head
        mass_ref = torch.clamp(self.read(W, G, A, hi, lo, torch.floor(n / 2))[2], min=1e-6)
        ue = torch.clamp(t - LEAD_STEPS + self.end_look, min=0.0)
        _, ep, em = self.read(W, G, A, hi, lo, ue)
        xe = torch.stack([ep, em / mass_ref], 1).reshape(-1)
        end_new = torch.sigmoid(torch.relu(xe @ self.s8_end_0_w.T + self.s8_end_0_b) @ self.s8_end_2_w.T + self.s8_end_2_b)

        act_h = torch.where(acting, act_new, act_h)
        valve_h = torch.where(acting, valve_new, valve_h)
        pwm = torch.where(acting, pwm_new, pwm)
        open_ = torch.where(acting, open_new, open_)
        end_p = torch.where(acting, end_new, torch.where(limit, torch.ones_like(end_p), end_p))
        t = torch.where(acting, t + 1, t)
        homing = play & (tau_now >= WARM_STEPS) & (tau_now < WARM_STEPS + HOMING_STEPS) & (n >= 10)
        out_pwm = torch.where(acting, pwm_new, torch.where(homing, -torch.ones_like(pwm), torch.zeros_like(pwm)))
        out_valve = torch.where(acting, (open_new >= .5).float(), torch.zeros_like(open_))
        outputs = torch.cat([new_mode, out_pwm, out_valve, fan, end_p, cents.reshape(1), voiced.reshape(1)])
        return outputs, new_mode, end_p, W, G, A, hi, lo, n, tau, t, act_h, valve_h, pwm, open_


OUTPUTS = ("mode", "pwm", "valve", "fan", "end", "heard_cents", "heard_voiced")
