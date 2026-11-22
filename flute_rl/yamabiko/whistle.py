"""A whistle recording as a target track for the performers (a relative melody copy).

The ear is YIN (flute_rl/pitch.py), not a network: every 10 ms it reads the whistle's pitch, then the track
is cleaned (median smoothing, blips dropped, short gaps bridged) and moved, intervals kept, into the rig's
target range.  NumPy only, so it can run on the UNO Q as well as on the PC.
"""
from __future__ import annotations

import numpy as np

from ..pitch import yin_frame

STEP_S = .01


def whistle_track(pcm: np.ndarray, sr: int, fmin: float = 500.0, fmax: float = 4000.0, frame: int = 640,
                  gate_db: float = -55.0, threshold: float = .2) -> np.ndarray:
    """Pitch of a recording every 10 ms, in cents re A4 (NaN where nothing whistles)."""
    x = np.asarray(pcm, np.float64)
    if x.dtype.kind in "iu" or np.abs(x).max() > 2:
        x = x / 32768.0
    hop = int(round(sr * STEP_S))
    out = []
    for start in range(0, max(0, len(x) - frame), hop):
        w = x[start:start + frame]
        w = w - w.mean()
        spectrum = np.fft.rfft(w)
        spectrum[:int(np.ceil(fmin * .7 * len(w) / sr))] = 0.0     # rumble, fan, hum
        w = np.fft.irfft(spectrum, len(w))
        db = 20 * np.log10(max(float(np.sqrt(np.mean(w * w))), 1e-9))
        if db < gate_db:
            out.append(np.nan)
            continue
        f0, _ = yin_frame(w, sr, fmin, fmax, threshold)
        out.append(1200 * np.log2(f0 / 440.0) if np.isfinite(f0) else np.nan)
    return np.array(out)


def _runs(mask):
    """(start, stop) of every run of True."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def clean_track(cents: np.ndarray, median: int = 5, min_note_steps: int = 6, max_gap_steps: int = 5) -> np.ndarray:
    """Median-smoothed, without blips shorter than ``min_note_steps`` and with gaps up to ``max_gap_steps``
    bridged (a whistle wavers and drops out for a frame or two inside one phrase)."""
    c = np.array(cents, np.float64)
    voiced = np.isfinite(c)
    for a, b in _runs(~voiced):                                   # bridge short gaps inside a phrase
        if a > 0 and b < len(c) and b - a <= max_gap_steps:
            c[a:b] = np.linspace(c[a - 1], c[b], b - a + 2)[1:-1]
    voiced = np.isfinite(c)
    for a, b in _runs(voiced):                                    # drop blips
        if b - a < min_note_steps:
            c[a:b] = np.nan
    half = median // 2
    smooth = c.copy()
    for i in np.flatnonzero(np.isfinite(c)):
        window = c[max(0, i - half):i + half + 1]
        smooth[i] = np.nanmedian(window)
    return smooth


def to_target(cents: np.ndarray, low: float, high: float, lead_steps: int = 60, tail_steps: int = 60,
              quantize: bool = False):
    """(target cents, voice, report): the phrase moved into [low, high] with its intervals kept.  A phrase
    wider than the range is centred and its ends are clipped (reported).  Rests hold the previous pitch, as
    the targets of the performers do; the lead rest holds the first note so the plunger can get there."""
    c = np.asarray(cents, np.float64)
    voiced = np.isfinite(c)
    if not voiced.any():
        raise ValueError("no whistle heard")
    first, last = np.flatnonzero(voiced)[[0, -1]]
    c, voiced = c[first:last + 1], voiced[first:last + 1]
    lo, hi = np.nanpercentile(c, 2), np.nanpercentile(c, 98)
    span, room = hi - lo, high - low
    shift = (low + high) / 2 - (lo + hi) / 2
    moved = c + shift
    if quantize:                                                  # to the semitone grid of the moved phrase
        base = moved[voiced][0]
        moved = base + np.round((moved - base) / 100) * 100
    clipped = float(np.mean((moved[voiced] < low) | (moved[voiced] > high)))
    moved = np.clip(moved, low, high)
    held = moved.copy()
    for i in range(len(held)):
        if not voiced[i]:
            held[i] = held[i - 1] if i else moved[voiced][0]
    target = np.r_[np.full(lead_steps, held[0]), held, np.full(tail_steps, held[-1])]
    voice = np.r_[np.zeros(lead_steps, bool), voiced, np.zeros(tail_steps, bool)]
    report = {"whistle_span_cents": float(span), "range_cents": float(room), "shift_cents": float(shift),
              "clipped_fraction": clipped, "voiced_steps": int(voiced.sum()), "steps": int(len(target))}
    return target.astype(np.float32), voice, report
