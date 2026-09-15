"""A small network that hears pitch from raw audio frames (needs PyTorch).

First step toward sound-in (end-to-end) policies: before a policy learns to
act on raw sound, check that a network can hear the pitch at least as well as
the hand-written YIN. Frames are centred on each 10 ms control step, like
the simulator.

Output: `n_bins` pitch classes 10 cents apart (CREPE-style soft targets) plus
nothing else; a frame is "unvoiced" when no bin is confident. The pitch is
the confidence-weighted mean around the best bin, so it is finer than a bin.

The numpy-only core of flute_rl does not import this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .audio import SOURCE_KINDS, SR_SELF, SR_SOURCE, room, synth_self, synth_source
from .sim import DT, hz_to_cents
from .targets import make_target, sample_level


@dataclass(frozen=True)
class Ear:
    """Input format and pitch range of one 'ear' (the robot hearing itself, or hearing a model sound)."""
    sr: int
    frame: int      # samples per analysis frame
    f_lo: float     # lowest pitch it reports [Hz]
    f_hi: float     # highest [Hz]
    bin_cents: float = 10.0

    @property
    def c_lo(self) -> float:
        return float(hz_to_cents(self.f_lo))

    @property
    def n_bins(self) -> int:
        return int(np.ceil((float(hz_to_cents(self.f_hi)) - self.c_lo) / self.bin_cents)) + 1


SELF_EAR = Ear(sr=SR_SELF, frame=256, f_lo=500.0, f_hi=3900.0)        # 32 ms frames
SOURCE_EAR = Ear(sr=SR_SOURCE, frame=1024, f_lo=80.0, f_hi=7000.0)     # 64 ms frames (humming needs long ones)


def frames_at_steps(y: np.ndarray, ear: Ear, n_steps: int) -> np.ndarray:
    """(n_steps, frame) windows centred on each control step."""
    hop = int(round(ear.sr * DT))
    pad = ear.frame
    yp = np.concatenate([np.zeros(pad), y, np.zeros(pad)])
    centers = (np.arange(n_steps) + 0.5) * hop + pad
    idx = (centers[:, None] - ear.frame // 2 + np.arange(ear.frame)[None, :]).astype(int)
    return yp[np.clip(idx, 0, len(yp) - 1)].astype(np.float32)


def make_clip(rng: np.random.Generator, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic clip -> (frames, true cents per frame, NaN = silent). kind: 'self' or a source kind."""
    target = make_target(rng, sample_level(rng, 1.0))
    if kind == "self":
        cents = np.where(np.isfinite(target), target + rng.normal(0.0, 30.0), np.nan)  # the flute is often off-target
        over = np.isfinite(cents) & (rng.random() < 0.2) & (np.arange(len(cents)) % 97 < 12)  # occasional overblow
        cents = np.where(over, cents + 1200.0, cents)
        y = room(synth_self(np.nan_to_num(cents), np.isfinite(cents), rng), SR_SELF, rng)
        ear, truth = SELF_EAR, cents
    else:
        y, shift = synth_source(target, kind, rng)
        y = room(y, SR_SOURCE, rng)
        ear, truth = SOURCE_EAR, target + shift
    return frames_at_steps(y, ear, len(truth)), truth.astype(np.float32)


def make_dataset(n_clips: int, kinds: tuple[str, ...], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frames, true cents, index into `kinds` per frame); the kinds take turns clip by clip."""
    rng = np.random.default_rng(seed)
    xs, ys, ks = [], [], []
    for i in range(n_clips):
        x, y = make_clip(rng, kinds[i % len(kinds)])
        xs.append(x)
        ys.append(y)
        ks.append(np.full(len(y), i % len(kinds), dtype=np.int8))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ks)


class PitchNet(nn.Module):
    """Frame (normalised) -> strided 1-D convolutions -> pitch-bin logits."""

    def __init__(self, ear: Ear, width: int = 32):
        super().__init__()
        self.ear = ear
        w = width
        self.body = nn.Sequential(
            nn.Conv1d(1, w, 32, stride=2, padding=15), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w, w, 16, stride=2, padding=7), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w, 2 * w, 8, stride=2, padding=3), nn.BatchNorm1d(2 * w), nn.ReLU(),
            nn.Conv1d(2 * w, 2 * w, 8, stride=2, padding=3), nn.BatchNorm1d(2 * w), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Linear(2 * w * 8, ear.n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / (x.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-4)  # loudness-invariant
        return self.head(self.body(x.unsqueeze(1)).flatten(1))


def soft_targets(cents: torch.Tensor, ear: Ear, sigma_cents: float = 25.0) -> torch.Tensor:
    """Gaussian bumps on the bin axis; all zeros for silent frames."""
    centers = ear.c_lo + ear.bin_cents * torch.arange(ear.n_bins, device=cents.device)
    voiced = torch.isfinite(cents)
    c = torch.where(voiced, cents, torch.zeros_like(cents))
    t = torch.exp(-0.5 * ((centers[None, :] - c[:, None]) / sigma_cents) ** 2)
    return t * voiced[:, None]


def decode(logits: torch.Tensor, ear: Ear, threshold: float = 0.5) -> torch.Tensor:
    """Cents per frame (NaN = unvoiced): weighted mean of +-4 bins around the most confident one
    (a window narrower than the target bump biases the estimate toward the nearest bin)."""
    p = torch.sigmoid(logits)
    best = p.argmax(dim=1)
    offs = torch.arange(-4, 5, device=p.device)
    idx = (best[:, None] + offs[None, :]).clamp(0, ear.n_bins - 1)
    w = p.gather(1, idx)
    centers = ear.c_lo + ear.bin_cents * idx.float()
    cents = (w * centers).sum(1) / w.sum(1).clamp_min(1e-6)
    return torch.where(p.max(dim=1).values >= threshold, cents, torch.full_like(cents, float("nan")))


def pitch_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Median error, octave errors and voicing, the same numbers for the network and for YIN."""
    on, heard = np.isfinite(truth), np.isfinite(pred)
    both = on & heard
    err = np.abs(pred[both] - truth[both])
    fine = err[err < 600.0]
    return {
        "median_cents": float(np.median(fine)) if fine.size else float("nan"),
        "p90_cents": float(np.percentile(fine, 90)) if fine.size else float("nan"),
        "octave_errors": float(np.mean(err >= 600.0)) if err.size else float("nan"),
        "voiced_recall": float(heard[on].mean()) if on.any() else float("nan"),
        "false_voiced": float(heard[~on].mean()) if (~on).any() else 0.0,
    }
