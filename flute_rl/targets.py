"""Target (reference) pitch contours: automatic curriculum and whistle import.

A target is a 1-D array sampled at DT; values are cents relative to A4 and
NaN marks a rest (the flute must be silent).
"""
from __future__ import annotations

import numpy as np

from .sim import DT, hz_to_cents

F_LO, F_HI = 650.0, 1350.0  # playable range assumed for targets [Hz]
MAJOR = (0, 2, 4, 5, 7, 9, 11)
MAX_LEVEL = 4
LEVEL_NAMES = {
    0: "single sustained note",
    1: "two notes (jump)",
    2: "glide between two notes",
    3: "short melody",
    4: "melody with vibrato",
}


def scale_notes(key: int = 0, f_lo: float = F_LO, f_hi: float = F_HI) -> np.ndarray:
    """Major-scale notes of `key` (0 = C) inside [f_lo, f_hi], in cents from A4."""
    m_lo = int(np.ceil(69 + 12 * np.log2(f_lo / 440.0)))
    m_hi = int(np.floor(69 + 12 * np.log2(f_hi / 440.0)))
    return np.array([(m - 69) * 100.0 for m in range(m_lo, m_hi + 1) if (m - key) % 12 in MAJOR])


def make_target(rng: np.random.Generator, level: int, dt: float = DT) -> np.ndarray:
    segs: list[np.ndarray] = []

    def n_steps(sec: float) -> int:
        return max(1, int(round(sec / dt)))

    def rest(sec: float) -> None:
        segs.append(np.full(n_steps(sec), np.nan))

    def hold(c: float, sec: float) -> None:
        segs.append(np.full(n_steps(sec), float(c)))

    notes = scale_notes(key=int(rng.integers(12)))
    # lead-in: every take starts from the home position, and the full stroke takes ~0.7 s
    rest(rng.uniform(0.8, 1.0))

    if level <= 0:
        hold(rng.choice(notes), rng.uniform(0.8, 1.5))
    elif level == 1:
        a, b = rng.choice(notes, 2, replace=False)
        hold(a, rng.uniform(0.4, 0.8))
        if rng.random() < 0.5:
            rest(rng.uniform(0.05, 0.15))
        hold(b, rng.uniform(0.4, 0.8))
    elif level == 2:
        a, b = rng.choice(notes, 2, replace=False)
        hold(a, rng.uniform(0.3, 0.6))
        segs.append(np.linspace(a, b, n_steps(rng.uniform(0.15, 0.4)), endpoint=False))
        hold(b, rng.uniform(0.3, 0.6))
    else:
        count = int(rng.integers(4, 9))
        idx = int(rng.integers(len(notes)))
        for i in range(count):
            dur = float(rng.choice([0.2, 0.3, 0.4, 0.6]))
            seg = np.full(n_steps(dur), notes[idx])
            if level >= 4 and dur >= 0.4:
                t = np.arange(len(seg)) * dt
                onset, rate, depth = 0.15, rng.uniform(5.0, 7.0), rng.uniform(15.0, 40.0)
                seg = seg + np.where(t > onset, depth * np.sin(2 * np.pi * rate * (t - onset)), 0.0)
            segs.append(seg)
            if i < count - 1 and rng.random() < 0.3:
                rest(rng.uniform(0.05, 0.1))
            idx = int(np.clip(idx + rng.integers(-2, 3), 0, len(notes) - 1))

    rest(0.2)
    return np.concatenate(segs)


def sample_level(rng: np.random.Generator, progress: float) -> int:
    """Curriculum: progress in [0, 1] unlocks harder levels; the newest level is sampled twice as often."""
    top = int(np.clip(progress * (MAX_LEVEL + 1), 0, MAX_LEVEL))
    weights = np.ones(top + 1)
    weights[-1] = 2.0
    return int(rng.choice(top + 1, p=weights / weights.sum()))


def from_pitch_track(hz, dt_in: float, dt: float = DT, f_lo: float = F_LO, f_hi: float = F_HI) -> np.ndarray:
    """Convert a whistle pitch track (Hz, NaN/0 = unvoiced) into a target.

    The track is resampled to `dt` and shifted by whole octaves so that its
    median sits in the middle of the playable range, then clipped to it.
    """
    hz = np.asarray(hz, dtype=float)
    voiced = np.isfinite(hz) & (hz > 0)
    if not voiced.any():
        raise ValueError("pitch track has no voiced frames")
    cents_in = np.where(voiced, hz_to_cents(np.where(voiced, hz, 1.0)), np.nan)
    t_out = np.arange(0.0, (len(hz) - 1) * dt_in + 1e-9, dt)
    idx = np.clip(np.round(t_out / dt_in).astype(int), 0, len(hz) - 1)
    out = cents_in[idx]
    lo, hi = float(hz_to_cents(f_lo)), float(hz_to_cents(f_hi))
    shift = 1200.0 * np.round(((lo + hi) / 2 - np.nanmedian(out)) / 1200.0)
    return np.clip(out + shift, lo, hi)
