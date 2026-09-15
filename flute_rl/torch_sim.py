"""Batched simulator in PyTorch: many rigs advance one control step together (GPU).

Same physics as flute_rl.sim.FluteSim, written as tensor operations over a
batch of rigs so that a whole ES generation (candidates x episodes) can be
simulated at once. With zero measurement noise it reproduces FluteSim
exactly (see tests/test_torch_sim.py); the random numbers differ, so with
noise the two agree statistically, not sample by sample.

The numpy FluteSim stays the reference (and what runs on the UNO Q).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .sim import DT, FluteParams

A4_HZ = 440.0
MAX_CMD_DELAY = 4


class BatchParams:
    """FluteParams fields as (B,) tensors."""

    def __init__(self, params: list[FluteParams], device=None, dtype=torch.float32):
        self.B = len(params)
        for f in dataclasses.fields(FluteParams):
            vals = [getattr(p, f.name) for p in params]
            if f.name in ("cmd_delay", "obs_delay"):
                setattr(self, f.name, torch.tensor(vals, dtype=torch.long, device=device))
            else:
                setattr(self, f.name, torch.tensor(vals, dtype=dtype, device=device))
        if int(self.cmd_delay.max()) > MAX_CMD_DELAY:
            raise ValueError("cmd_delay larger than MAX_CMD_DELAY")


def speed_of_sound(temp_c: torch.Tensor) -> torch.Tensor:
    return 331.3 + 0.606 * temp_c


def hz_to_cents(f: torch.Tensor) -> torch.Tensor:
    return 1200.0 * torch.log2(f / A4_HZ)


class BatchFluteSim:
    """B rigs stepped together. step() takes (B,) PWM and angle commands and returns a dict of (B,) tensors."""

    def __init__(self, params: BatchParams, generator: torch.Generator | None = None):
        self.p = params
        self.gen = generator
        self.reset()

    @property
    def device(self):
        return self.p.tube_len.device

    def reset(self) -> dict:
        p = self.p
        z = torch.zeros_like(p.tube_len)
        self.x_motor, self.x, self.v, self.theta = z.clone(), z.clone(), z.clone(), z.clone()
        self.hist = torch.zeros(p.B, MAX_CMD_DELAY, dtype=z.dtype, device=z.device)  # hist[:, 0] = previous command
        self.t = 0
        return self.observe()

    def pitch_at(self, x: torch.Tensor, theta: torch.Tensor):
        p = self.p
        length = torch.clamp(p.tube_len - x + p.end_corr, min=0.02)
        f = speed_of_sound(p.temp_c) / (4.0 * length)
        d = theta - p.theta_opt_deg
        shrink = 1.0 - p.win_narrowing * torch.clamp(x / p.stroke, 0.0, 1.0)
        lo, hi = p.win_lo_deg * shrink, p.win_hi_deg * shrink
        sounding = (-lo < d) & (d < hi)
        cents = hz_to_cents(f) + p.k_theta * d
        overblown = sounding & (x > p.overblow_x) & (d > 0.5 * hi)
        cents = cents + 1200.0 * overblown.to(cents.dtype)
        return cents, sounding, overblown

    def step(self, pwm: torch.Tensor, angle_cmd: torch.Tensor) -> dict:
        p = self.p
        # plunger: per-rig command delay -> dead band -> velocity lag -> end stops -> backlash
        pwm = torch.clamp(pwm.to(self.x.dtype), -1.0, 1.0)
        buf = torch.cat([pwm[:, None], self.hist], dim=1)
        u = buf.gather(1, p.cmd_delay[:, None]).squeeze(1)
        self.hist = buf[:, :MAX_CMD_DELAY]
        mag = u.abs()
        drive = torch.where(mag < p.deadband, torch.zeros_like(u), torch.sign(u) * (mag - p.deadband) / (1.0 - p.deadband))
        v_cmd = drive * torch.where(drive > 0, p.v_max_in, p.v_max_out)
        self.v = self.v + (v_cmd - self.v) * torch.clamp(DT / p.tau_v, max=1.0)
        self.x_motor = self.x_motor + self.v * DT
        low, high = self.x_motor <= 0.0, self.x_motor >= p.stroke
        self.v = torch.where(low, torch.clamp(self.v, min=0.0), torch.where(high, torch.clamp(self.v, max=0.0), self.v))
        self.x_motor = torch.where(low, torch.zeros_like(self.x_motor), torch.where(high, p.stroke, self.x_motor))
        half = p.backlash / 2.0
        self.x = torch.where(self.x_motor - self.x > half, self.x_motor - half,
                             torch.where(self.x - self.x_motor > half, self.x_motor + half, self.x))

        # servo: quantised target, first-order lag, rate limit
        target = torch.clamp(angle_cmd.to(self.x.dtype), -1.0, 1.0) * p.angle_range_deg
        target = torch.round(target / p.servo_step_deg) * p.servo_step_deg
        d_theta = (target - self.theta) * torch.clamp(DT / p.servo_tau, max=1.0)
        lim = p.servo_rate_dps * DT
        self.theta = self.theta + torch.maximum(torch.minimum(d_theta, lim), -lim)
        self.t += 1
        return self.observe()

    def observe(self) -> dict:
        p = self.p
        cents, sounding, overblown = self.pitch_at(self.x, self.theta)
        readback = torch.round(self.theta / p.servo_step_deg) * p.servo_step_deg
        u = torch.rand(p.B, generator=self.gen, device=self.device, dtype=cents.dtype)
        noise = torch.randn(p.B, generator=self.gen, device=self.device, dtype=cents.dtype)
        heard = sounding & (u >= p.dropout)
        measured = torch.where(heard, cents + noise * p.pitch_noise, torch.full_like(cents, float("nan")))
        return {"x": self.x, "v": self.v, "theta": self.theta, "theta_readback": readback, "cents": cents,
                "sounding": sounding, "overblown": overblown, "measured": measured}


def sample_params(seeds, spread: float = 1.0) -> list[FluteParams]:
    """The same randomised rigs as FluteEnv(seed=s) would build (FluteParams.sample on the env's rng)."""
    return [FluteParams.sample(np.random.default_rng(int(s)), spread) for s in seeds]
