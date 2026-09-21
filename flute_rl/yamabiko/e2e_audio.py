"""Shared causal microphone simulation for E2E cloning and reinforcement learning."""
from __future__ import annotations

import math

import numpy as np

from .e2e_io import FRAME, HOP, SAMPLE_RATE


class RawSelfAudio:
    """16 kHz microphone-like waveform generated from batched rig outputs.

    The frame returned before ``push`` contains only earlier physical results,
    matching the real UNO Q control loop.  Using this exact class for cloning
    and PPO prevents the feedback waveform distribution from changing between
    the two training stages.
    """

    def __init__(self, n: int, rng: np.random.Generator, domain: float = 0.0):
        """Create a causal microphone renderer.

        ``domain=0`` preserves the original clean simulator.  Larger values
        randomize the sensor/room transfer without changing the underlying
        physical pitch.  This separation lets us validate mechanics and raw
        audio robustness independently instead of hiding both in pitch noise.
        """
        self.n, self.rng = n, rng
        self.domain = float(domain)
        if self.domain < 0.0:
            raise ValueError("audio domain strength must be non-negative")
        self.phase = np.zeros(n)
        self.env = np.zeros(n)
        self.buf = np.zeros((n, FRAME), np.float32)
        self.harm = np.column_stack([np.ones(n), rng.uniform(0.05, 0.35, n), rng.uniform(0.01, 0.15, n)])
        self.gain = rng.uniform(0.2, 0.8, n)
        self.noise = 10.0 ** (-rng.uniform(35.0, 60.0, n) / 20.0)
        self.samples = np.arange(1, HOP + 1)
        # Nuisances that the real microphone sees but the ideal acoustic
        # formula does not: room reflection, microphone bandwidth/gain,
        # fan hum, saturation and occasional transport loss.  One set is
        # drawn per rig/session so the controller cannot memorize a single
        # synthetic timbre.
        d = self.domain
        self.echo_gain = d * rng.uniform(0.0, 0.35, n)
        self.echo = np.zeros((n, HOP))
        self.lowpass = 1.0 - d * rng.uniform(0.0, 0.75, n)
        self.filter_state = np.zeros(n)
        self.sensor_gain = np.exp(d * rng.uniform(-1.0, 1.0, n))
        self.clip_level = 1.0 - d * rng.uniform(0.0, 0.55, n)
        self.hum_freq = rng.uniform(80.0, 420.0, n)
        self.hum_amp = d * 10.0 ** (-rng.uniform(28.0, 48.0, n) / 20.0)
        self.hum_phase = rng.uniform(0.0, 2.0 * np.pi, n)
        self.block_dropout = min(0.02, 0.005 * d)

    def reset(self):
        self.phase.fill(0.0)
        self.env.fill(0.0)
        self.buf.fill(0.0)
        self.echo.fill(0.0)
        self.filter_state.fill(0.0)

    def push(self, cents, sounding):
        sounding = np.asarray(sounding, bool)
        f = np.where(sounding, 440.0 * 2.0 ** (np.nan_to_num(cents) / 1200.0), 0.0)
        phase = self.phase[:, None] + 2.0 * np.pi * f[:, None] * self.samples[None, :] / SAMPLE_RATE
        self.phase = np.remainder(phase[:, -1], 2.0 * np.pi)
        y = sum(self.harm[:, k, None] * np.sin((k + 1) * phase) for k in range(3))
        target = sounding.astype(float)
        a = 1.0 - math.exp(-1.0 / (0.02 * SAMPLE_RATE))
        env = np.empty((self.n, HOP))
        e = self.env.copy()
        for j in range(HOP):
            e += a * (target - e)
            env[:, j] = e
        self.env = e
        y = self.gain[:, None] * env * (0.25 * y + 0.03 * self.rng.standard_normal(y.shape))
        y += self.noise[:, None] * self.rng.standard_normal(y.shape)
        if self.domain > 0.0:
            dry = y.copy()
            y += self.echo_gain[:, None] * self.echo
            self.echo = dry
            hum_phase = (self.hum_phase[:, None] + 2.0 * np.pi * self.hum_freq[:, None]
                         * self.samples[None, :] / SAMPLE_RATE)
            y += self.hum_amp[:, None] * np.sin(hum_phase)
            self.hum_phase = np.remainder(hum_phase[:, -1], 2.0 * np.pi)
            # First-order microphone low-pass, with state across control
            # blocks.  The explicit loop is only 160 samples and keeps this
            # dependency-free on the UNO-Q-compatible simulation path.
            filtered = np.empty_like(y)
            s = self.filter_state.copy()
            for j in range(HOP):
                s += self.lowpass * (y[:, j] - s)
                filtered[:, j] = s
            self.filter_state = s
            y = filtered * self.sensor_gain[:, None]
            y = np.clip(y, -self.clip_level[:, None], self.clip_level[:, None])
            lost = self.rng.random(self.n) < self.block_dropout
            y[lost] = 0.0
        self.buf = np.concatenate([self.buf[:, HOP:], y.astype(np.float32)], axis=1)

    def render(self, cents, sounding):
        """Render known traces into causal frames, (n, steps, FRAME)."""
        cents = np.atleast_2d(cents)
        sounding = np.atleast_2d(sounding)
        if cents.shape[0] != self.n:
            raise ValueError("trace batch does not match audio batch")
        self.reset()
        frames = np.empty((self.n, cents.shape[1], FRAME), np.float32)
        for t in range(cents.shape[1]):
            frames[:, t] = self.buf
            self.push(cents[:, t], sounding[:, t])
        return frames
