"""Audio synthesis for sound-in (end-to-end) learning.

The simulator itself works with pitch values. For policies that listen to
raw sound, this module turns pitch traces into audio with randomised timbre,
so a network has to learn to hear the pitch itself:

* synth_self: the robot's own flute (edge tone) from a simulator log, 8 kHz.
  The flute plays 0.6-1.9 kHz (overblown up to ~3.8 kHz), under the 4 kHz
  Nyquist limit.
* synth_source: a model to imitate (recorder, whistle, humming, bird-like
  tone) from a target contour, 16 kHz. The source is played whole octaves
  away from the flute's range where that is natural (whistle one octave up,
  humming one or two down, bird two or three up); the returned shift tells by
  how much, since the flute should play the target contour itself.
* room: reverberation, background noise and gain, applied to either.

Everything is numpy only and deterministic for a given rng.
"""
from __future__ import annotations

import numpy as np

from .sim import DT, cents_to_hz

SR_SELF = 8000
SR_SOURCE = 16000
SOURCE_KINDS = ("recorder", "whistle", "hum", "bird")


def _per_sample(values: np.ndarray, sr: int) -> np.ndarray:
    """Hold each 10 ms control-frame value for the samples of that frame."""
    return np.repeat(np.asarray(values, dtype=float), int(round(sr * DT)))


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Symmetric Hann smoothing over n samples (used for envelopes and pitch glides)."""
    if n <= 1:
        return x
    k = np.hanning(n + 2)[1:-1]
    return np.convolve(x, k / k.sum(), mode="same")


def _band_noise(n: int, sr: int, lo: float, hi: float, rng: np.random.Generator, tilt: float = 0.0) -> np.ndarray:
    """Noise with energy between lo and hi Hz; tilt > 0 makes it darker (1 = pink)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    shape = ((f >= lo) & (f <= hi)).astype(float) / np.maximum(f, 1.0) ** (tilt / 2.0)
    y = np.fft.irfft(spec * shape, n)
    return y / (np.std(y) + 1e-12)


def _tone(f_hz: np.ndarray, amps: np.ndarray, sr: int, formants=None) -> np.ndarray:
    """Sum of harmonics following a per-sample fundamental (0 = silent), aliasing-free.

    `formants`: optional list of (centre Hz, width in octaves, gain) bumps that
    shape the harmonic amplitudes by their frequency (voice-like timbre)."""
    phase = 2.0 * np.pi * np.cumsum(f_hz) / sr
    y = np.zeros_like(f_hz)
    for k, a in enumerate(amps, start=1):
        fk = k * f_hz
        g = np.where((fk > 0) & (fk < 0.45 * sr), a, 0.0)
        if formants:
            lf = np.log2(np.maximum(fk, 1.0))
            g = g * (0.15 + sum(gain * np.exp(-0.5 * ((lf - np.log2(c)) / w) ** 2) for c, w, gain in formants))
        y += g * np.sin(k * phase)
    return y


def _envelope(on: np.ndarray, sr: int, attack_s: float) -> np.ndarray:
    return np.clip(_smooth(on.astype(float), max(1, int(attack_s * sr))), 0.0, 1.0)


def _onset_chiff(on: np.ndarray, sr: int, rng: np.random.Generator, level: float, lo: float, hi: float) -> np.ndarray:
    """Short noise burst at each note start (the 'chiff' of an edge-tone flute)."""
    starts = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int)])) == 1)
    burst = np.zeros(len(on))
    n = int(0.03 * sr)
    decay = np.exp(-np.arange(n) / (0.008 * sr))
    for s in starts:
        m = min(n, len(on) - s)
        burst[s:s + m] += decay[:m]
    return level * burst * _band_noise(len(on), sr, lo, hi, rng)


def synth_self(cents: np.ndarray, sounding: np.ndarray, rng: np.random.Generator, sr: int = SR_SELF) -> np.ndarray:
    """The robot's own flute sound for a simulator trace (true pitch per control step)."""
    on = _per_sample(np.asarray(sounding, bool), sr) > 0.5
    c = _per_sample(np.where(np.asarray(sounding, bool), cents, 0.0), sr)
    f = np.where(on, cents_to_hz(c), 0.0)
    f = np.where(on, _smooth(f, int(0.004 * sr)), 0.0)
    amps = np.array([1.0, rng.uniform(0.05, 0.4), rng.uniform(0.02, 0.3), rng.uniform(0.0, 0.1)])
    env = _envelope(on, sr, rng.uniform(0.01, 0.04))
    flutter = 1.0 + rng.uniform(0.0, 0.08) * _smooth(rng.standard_normal(len(f)), int(0.02 * sr))
    y = env * flutter * _tone(f, amps, sr)
    y += env * rng.uniform(0.02, 0.12) * _band_noise(len(f), sr, 800.0, 3900.0, rng)  # breath
    y += _onset_chiff(on, sr, rng, rng.uniform(0.1, 0.5), 1000.0, 3900.0)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.5


def synth_source(target: np.ndarray, kind: str, rng: np.random.Generator, sr: int = SR_SOURCE) -> tuple[np.ndarray, float]:
    """A model sound for a target contour (cents per control step, NaN = rest).

    Returns (audio, octave shift in cents): the source sounds at target + shift."""
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind {kind!r}")
    target = np.asarray(target, dtype=float)
    shift = {"recorder": float(rng.choice([0.0, 1200.0])),
             "whistle": 1200.0,
             "hum": float(rng.choice([-1200.0, -2400.0])),
             "bird": float(rng.choice([2400.0, 3600.0]))}[kind]
    top = np.nanmax(target) + shift
    while cents_to_hz(top) > 0.4 * sr:  # keep the fundamental well under Nyquist
        shift -= 1200.0
        top -= 1200.0

    on_f = np.isfinite(target)
    on = _per_sample(on_f, sr) > 0.5
    c = _per_sample(np.where(on_f, target + shift, 0.0), sr)
    # natural pitch drift / jitter, heavier for voices
    drift = {"recorder": 4.0, "whistle": 8.0, "hum": 15.0, "bird": 10.0}[kind]
    c = c + drift * rng.uniform(0.3, 1.0) * _smooth(rng.standard_normal(len(c)), int(0.08 * sr)) * 3.0
    f = np.where(on, cents_to_hz(c), 0.0)
    f = np.where(on, _smooth(f, int(0.006 * sr)), 0.0)

    formants = None
    if kind == "recorder":
        amps = np.array([1.0, rng.uniform(0.1, 0.3), rng.uniform(0.05, 0.25), rng.uniform(0.0, 0.1), rng.uniform(0.0, 0.05)])
        attack, breath, chiff, band = rng.uniform(0.01, 0.03), rng.uniform(0.01, 0.05), rng.uniform(0.2, 0.6), (1500.0, 7000.0)
    elif kind == "whistle":
        amps = np.array([1.0, rng.uniform(0.0, 0.05)])
        attack, breath, chiff, band = rng.uniform(0.02, 0.06), rng.uniform(0.03, 0.15), 0.0, (2000.0, 7500.0)
    elif kind == "hum":
        n_h = 20
        amps = 1.0 / np.arange(1, n_h + 1) ** rng.uniform(1.0, 2.0)
        amps[0] *= rng.uniform(0.3, 1.0)  # a weak fundamental makes octave errors likely, as with real voices
        formants = [(rng.uniform(200, 350), 0.3, 1.0), (rng.uniform(800, 1300), 0.3, rng.uniform(0.2, 0.7)),
                    (rng.uniform(2200, 2900), 0.25, rng.uniform(0.1, 0.4))]
        attack, breath, chiff, band = rng.uniform(0.03, 0.08), rng.uniform(0.0, 0.03), 0.0, (300.0, 4000.0)
    else:  # bird-like: nearly pure, fast amplitude flicker
        amps = np.array([1.0, rng.uniform(0.0, 0.15)])
        attack, breath, chiff, band = rng.uniform(0.003, 0.01), rng.uniform(0.0, 0.03), 0.0, (3000.0, 7900.0)

    env = _envelope(on, sr, attack)
    if kind == "bird":
        env = env * (0.7 + 0.3 * np.sin(2 * np.pi * rng.uniform(20, 40) * np.arange(len(env)) / sr))
    y = env * _tone(f, amps, sr, formants)
    y += env * breath * _band_noise(len(f), sr, band[0], min(band[1], 0.49 * sr), rng)
    if chiff:
        y += _onset_chiff(on, sr, rng, chiff, band[0], min(band[1], 0.49 * sr))
    return y / (np.max(np.abs(y)) + 1e-9) * 0.5, shift


def room(x: np.ndarray, sr: int, rng: np.random.Generator, snr_db: tuple[float, float] = (15.0, 40.0)) -> np.ndarray:
    """Room reverberation, background noise (pinkish) and a random gain."""
    rt60 = rng.uniform(0.05, 0.4)
    n = int(rt60 * sr)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 0.0
    wet = rng.uniform(0.0, 0.3)
    y = x + wet * np.convolve(x, ir / (np.sqrt(np.sum(ir**2)) + 1e-9), mode="full")[: len(x)]
    sig = np.sqrt(np.mean(y**2)) + 1e-12
    noise = _band_noise(len(y), sr, 20.0, 0.49 * sr, rng, tilt=1.0)
    y = y + noise * sig / 10 ** (rng.uniform(*snr_db) / 20.0)
    return np.clip(y * rng.uniform(0.3, 1.0) / (np.max(np.abs(y)) + 1e-9), -1.0, 1.0)
