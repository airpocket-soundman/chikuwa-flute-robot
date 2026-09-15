"""Hearing the robot's own sound inside the batched environment (end-to-end learning, method A).

Instead of the simulator's pitch "measurement" (true pitch + noise), every rig
produces its flute sound 10 ms at a time (8 kHz, timbre randomised per rig,
breath and room noise), and a trained pitch-hearing network (flute_rl.pitchnet,
self ear) listens to the last 32 ms, `obs_delay` steps late. It returns

* the pitch it hears (cents, NaN when it hears no tone),
* its confidence,
* a compact summary of its internal features (PCA of the layer before the
  output), so a policy can use more than the pitch.

The ear's weights stay fixed (method A). Needs PyTorch.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .pitchnet import SELF_EAR, PitchNet, decode, make_dataset

SR = SELF_EAR.sr
HOP = int(round(SR * 0.01))
FRAME = SELF_EAR.frame
MAX_DELAY = 8


class BatchSelfAudio:
    """Phase-continuous flute sound for B rigs, synthesised one control step at a time."""

    def __init__(self, B: int, device, generator: torch.Generator | None = None):
        self.B, self.dev, self.gen = B, device, generator

        def r(lo, hi):
            return lo + (hi - lo) * torch.rand(B, generator=generator, device=device)

        self.amps = torch.stack([torch.ones(B, device=device), r(0.05, 0.4), r(0.02, 0.3), r(0.0, 0.1)], 1)
        self.breath = r(0.02, 0.12)
        self.attack = r(0.01, 0.04)
        self.gain = r(0.3, 1.0)
        self.noise = 0.35 * 10 ** (-r(15.0, 40.0) / 20.0)  # background noise relative to a typical tone
        self.phase = torch.zeros(B, device=device)
        self.env = torch.zeros(B, device=device)
        self.buf = torch.zeros(B, FRAME + MAX_DELAY * HOP, device=device)
        self.n = torch.arange(1, HOP + 1, device=device, dtype=torch.float32)

    def reset(self) -> None:
        self.phase.zero_()
        self.env.zero_()
        self.buf.zero_()

    def push(self, cents: torch.Tensor, sounding: torch.Tensor) -> None:
        on = sounding.to(torch.float32)
        f = torch.where(sounding, 440.0 * torch.pow(2.0, cents.float() / 1200.0), torch.zeros_like(on))
        ph = self.phase[:, None] + 2.0 * math.pi * f[:, None] * self.n[None, :] / SR
        self.phase = torch.remainder(ph[:, -1], 2.0 * math.pi)
        y = torch.zeros(self.B, HOP, device=self.dev)
        for k in range(self.amps.shape[1]):
            ok = ((k + 1) * f < 0.45 * SR).to(torch.float32)
            y = y + (self.amps[:, k] * ok)[:, None] * torch.sin((k + 1) * ph)
        # attack / release envelope, then breath noise that follows the tone
        decay = torch.exp(-self.n[None, :] / (self.attack[:, None] * SR))
        env = on[:, None] - (on[:, None] - self.env[:, None]) * decay
        self.env = env[:, -1]
        white = torch.randn(self.B, HOP, generator=self.gen, device=self.dev)
        y = env * (y + self.breath[:, None] * 2.0 * white)
        # gain applies to the whole microphone signal, so the signal-to-noise ratio stays 15-40 dB
        y = 0.3 * self.gain[:, None] * (y + (self.noise / 0.3)[:, None] * torch.randn(self.B, HOP, generator=self.gen, device=self.dev))
        self.buf = torch.cat([self.buf[:, HOP:], y], 1)

    def frames(self, delay: torch.Tensor) -> torch.Tensor:
        """(B, FRAME): the 32 ms that ended `delay` steps ago, per rig."""
        end = self.buf.shape[1] - delay.clamp(0, MAX_DELAY - 1) * HOP
        idx = end[:, None] - FRAME + torch.arange(FRAME, device=self.dev)[None, :]
        return self.buf.gather(1, idx)


class Ear:
    """A trained self-ear network with a fixed feature summary (method A: weights not trained further)."""

    def __init__(self, path: str, device, n_pca: int = 16, fit_clips: int = 40):
        ck = torch.load(path, map_location=device)
        self.net = PitchNet(SELF_EAR, ck["width"]).to(device).eval()
        self.net.load_state_dict(ck["state"])
        self.dev, self.n_pca = device, n_pca
        # PCA of the layer before the output, fitted on synthetic clips (fixed for the whole run)
        x, _, _ = make_dataset(fit_clips, ("self",), seed=777)
        with torch.no_grad():
            emb = self._embed(torch.from_numpy(x[::3]).to(device))
        self.mean = emb.mean(0, keepdim=True)
        _, s, v = torch.pca_lowrank(emb - self.mean, q=n_pca, center=False)
        self.basis = v[:, :n_pca]
        self.scale = (s[:n_pca] / math.sqrt(len(emb))).clamp_min(1e-6)

    @property
    def dim(self) -> int:
        return 2 + self.n_pca

    def _embed(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames / (frames.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-4)
        return self.net.body(x.unsqueeze(1)).flatten(1)

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor):
        """-> (heard cents (B,), NaN = no tone; features (B, 2 + n_pca): confidence, voiced, PCA)."""
        emb = self._embed(frames)
        logits = self.net.head(emb)
        cents = decode(logits, SELF_EAR)
        conf = torch.sigmoid(logits).max(1).values
        pca = ((emb - self.mean) @ self.basis) / self.scale
        feats = torch.cat([conf[:, None], torch.isfinite(cents).float()[:, None], pca.clamp(-5.0, 5.0)], 1)
        return cents, feats


def check_ear(ear: Ear, n: int = 2000, device=None) -> dict:
    """How well the ear hears the in-environment sound (it was trained on audio.synth_self)."""
    gen = torch.Generator(device=device).manual_seed(0)
    au = BatchSelfAudio(n, device, gen)
    cents = 900.0 + 500.0 * torch.rand(n, generator=gen, device=device)
    on = torch.rand(n, generator=gen, device=device) < 0.8
    for _ in range(6):
        au.push(cents, on)
    heard, _ = ear(au.frames(torch.zeros(n, dtype=torch.long, device=device)))
    ok = on & torch.isfinite(heard)
    err = (heard[ok] - cents[ok]).abs()
    return {"median_cents": float(err.median()), "octave_errors": float((err > 600).float().mean()),
            "voiced_recall": float(ok.float().sum() / on.float().sum()),
            "false_voiced": float((torch.isfinite(heard) & ~on).float().sum() / (~on).float().sum().clamp_min(1))}


def pad_inputs(theta: np.ndarray, arch: str, in_old: int, in_new: int, out_dim: int, hidden: int) -> np.ndarray:
    """Grow a saved network's input with zero weights: the new inputs start with no effect."""
    extra = in_new - in_old
    if extra < 0:
        raise ValueError("new input is smaller")
    parts, i = [], 0
    if arch == "gru":
        shapes = [(in_old, 3 * hidden), (hidden, 3 * hidden), (3 * hidden,), (hidden + in_old, out_dim), (out_dim,)]
    else:
        shapes = [(in_old, hidden), (hidden,), (hidden, out_dim), (out_dim,)]
    for s in shapes:
        n = int(np.prod(s))
        parts.append(theta[i:i + n].reshape(s))
        i += n
    if arch == "gru":
        W, U, b, Wo, bo = parts
        W = np.vstack([W, np.zeros((extra, 3 * hidden))])
        Wo = np.vstack([Wo, np.zeros((extra, out_dim))])
        return np.concatenate([W.ravel(), U.ravel(), b, Wo.ravel(), bo])
    w1, b1, w2, b2 = parts
    w1 = np.vstack([w1, np.zeros((extra, hidden))])
    return np.concatenate([w1.ravel(), b1, w2.ravel(), b2])
