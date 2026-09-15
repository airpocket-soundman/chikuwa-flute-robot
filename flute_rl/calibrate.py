"""Fit simulator parameters to measurements of the real flute.

Two sweeps, both plain recordings (WAV) that are analysed with the YIN tracker:

* depth sweep - blow at a fixed angle, recording once per plunger (or stopper)
  depth. A tube closed at the bottom sounds at f = c / (4 (S - x)), where S is
  the acoustic length at depth 0 (tube + end correction) and x the depth. Then
  1 / f = 4 S / c - (4 / c) x is a straight line in x, which gives S and the
  speed of sound c (i.e. the air temperature) directly.
* angle sweep - fixed depth, one recording per blowing angle. The angles that
  sound give the sounding window (edges and centre); the pitch against the
  angle inside the window gives the pitch bend per degree.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np

from .pitch import pitch_track


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(float) / 2**31
    else:
        raise ValueError(f"unsupported sample width {width} (use 16- or 32-bit PCM)")
    return x.reshape(-1, ch).mean(axis=1), sr


@dataclass
class Tone:
    hz: float           # median pitch of the steadiest sounding stretch (NaN if none)
    sounding: float     # fraction of the recording with a detected pitch
    wobble_cents: float


def analyse(x: np.ndarray, sr: int, fmin: float = 200.0, fmax: float = 5000.0) -> Tone:
    x = x / (np.max(np.abs(x)) + 1e-12)
    frame = int(2 ** np.ceil(np.log2(sr * 0.04)))
    _, f0, _ = pitch_track(x, sr, frame=frame, fmin=fmin, fmax=min(fmax, 0.45 * sr), rms_gate=0.02)
    voiced = np.isfinite(f0)
    if not voiced.any():
        return Tone(float("nan"), 0.0, float("nan"))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(int), [0]])))
    a, b = max(zip(edges[::2], edges[1::2]), key=lambda s: s[1] - s[0])  # longest sounding stretch
    seg = f0[a:b]
    hz = float(np.median(seg))
    return Tone(hz, float(voiced.mean()), float(np.std(1200 * np.log2(seg / hz))))


@dataclass
class TubeFit:
    acoustic_len_m: float   # S: acoustic length at depth 0 (tube + end correction)
    speed_of_sound: float   # c [m/s]
    temp_c: float           # the air temperature that c implies
    residual_cents: np.ndarray


def fit_tube(depths_m: np.ndarray, hz: np.ndarray) -> TubeFit:
    """Least squares on 1/f = 4S/c - (4/c) x (depths where the flute sounded)."""
    x, f = np.asarray(depths_m, float), np.asarray(hz, float)
    ok = np.isfinite(f)
    A = np.stack([np.ones(ok.sum()), x[ok]], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, 1.0 / f[ok], rcond=None)
    c = -4.0 / b
    S = a * c / 4.0
    pred = c / (4.0 * (S - x[ok]))
    return TubeFit(float(S), float(c), float((c - 331.3) / 0.606), 1200 * np.log2(f[ok] / pred))


@dataclass
class WindowFit:
    lo_deg: float        # lowest angle that sounds
    hi_deg: float        # highest angle that sounds
    centre_deg: float    # middle of the sounding window
    cents_per_deg: float # pitch bend inside the window
    sounding_angles: np.ndarray


def fit_window(angles_deg: np.ndarray, hz: np.ndarray, sounding: np.ndarray, min_sounding: float = 0.3) -> WindowFit:
    ang, f, s = np.asarray(angles_deg, float), np.asarray(hz, float), np.asarray(sounding, float)
    order = np.argsort(ang)
    ang, f, s = ang[order], f[order], s[order]
    good = (s >= min_sounding) & np.isfinite(f)
    if not good.any():
        raise ValueError("no angle sounded")
    lo, hi = float(ang[good].min()), float(ang[good].max())
    k = float(np.polyfit(ang[good], 1200 * np.log2(f[good] / np.median(f[good])), 1)[0]) if good.sum() >= 2 else float("nan")
    return WindowFit(lo, hi, 0.5 * (lo + hi), k, ang[good])
