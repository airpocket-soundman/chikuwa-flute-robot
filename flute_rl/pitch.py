"""YIN pitch estimation in numpy (front end for whistle input and take analysis).

Runs on the Linux side of the UNO Q without extra dependencies. The whistle
(~1-4 kHz) and the flute (~0.6-1.9 kHz) are nearly sinusoidal, so plain YIN
with parabolic interpolation is accurate to a few cents.
"""
from __future__ import annotations

import numpy as np


def yin_frame(frame: np.ndarray, sr: int, fmin: float = 400.0, fmax: float = 4000.0,
              threshold: float = 0.15) -> tuple[float, float]:
    """Return (f0 [Hz] or NaN if unvoiced, confidence in [0, 1]) for one frame."""
    x = np.asarray(frame, dtype=float)
    x = x - x.mean()
    tau_max = min(int(sr / fmin), len(x) // 2)
    tau_min = max(2, int(sr / fmax))
    n = len(x) - tau_max
    if n <= tau_min:
        raise ValueError("frame too short for fmin")

    # difference function d(tau) = E0 + E_tau - 2 r(tau), autocorrelation via FFT
    nfft = 1 << int(np.ceil(np.log2(len(x) + n)))
    r = np.fft.irfft(np.conj(np.fft.rfft(x[:n], nfft)) * np.fft.rfft(x, nfft), nfft)[: tau_max + 1]
    sq = np.concatenate([[0.0], np.cumsum(x * x)])
    taus = np.arange(tau_max + 1)
    e_tau = sq[taus + n] - sq[taus]
    d = sq[n] + e_tau - 2.0 * r
    d[0] = 0.0

    # cumulative mean normalised difference
    cmnd = np.ones_like(d)
    cums = np.cumsum(d[1:])
    cmnd[1:] = d[1:] * taus[1:] / np.where(cums > 0, cums, 1.0)

    below = np.flatnonzero(cmnd[tau_min:tau_max] < threshold)
    if below.size == 0:
        return float("nan"), float(max(0.0, 1.0 - cmnd[tau_min:tau_max].min()))
    tau = tau_min + int(below[0])
    while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1

    # parabolic interpolation on the raw difference function (less biased than on cmnd)
    if 1 <= tau < tau_max:
        a, b, c = d[tau - 1], d[tau], d[tau + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    return float(sr / (tau + shift)), float(max(0.0, 1.0 - cmnd[tau]))


def pitch_track(signal: np.ndarray, sr: int, frame: int = 1024, hop: int | None = None,
                fmin: float = 400.0, fmax: float = 4000.0, threshold: float = 0.15,
                rms_gate: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame-wise pitch. Returns (times [s], f0 [Hz, NaN = unvoiced], confidence).

    The default hop is 10 ms so the track lines up with the control period.
    """
    x = np.asarray(signal, dtype=float)
    hop = hop or int(round(sr * 0.01))
    starts = np.arange(0, max(1, len(x) - frame + 1), hop)
    f0 = np.full(len(starts), np.nan)
    conf = np.zeros(len(starts))
    for i, s in enumerate(starts):
        fr = x[s:s + frame]
        if len(fr) < frame or np.sqrt(np.mean(fr * fr)) < rms_gate:
            continue
        f0[i], conf[i] = yin_frame(fr, sr, fmin, fmax, threshold)
    times = (starts + frame / 2) / sr
    return times, f0, conf
