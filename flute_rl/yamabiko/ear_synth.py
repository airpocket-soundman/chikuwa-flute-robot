"""Synthetic whistles for the common ear (step5), with their true pitch every 10 ms.

A whistle is nearly a sine: notes of random length and pitch (about 500 Hz .. 3.8 kHz), joined by glides or
separated by gaps, with vibrato, slow drift, a soft attack and release, a little second harmonic and breath
noise (a band of noise around the note, and broadband hiss).  ``phrase`` returns 16 kHz samples, the pitch in
cents re A4 at the centre of every causal 20 ms frame the ear hears (one every 10 ms) and its state as in
scripts/yamabiko_ear_data.py: 1 voiced, 0 quiet, -1 unsure (the frame straddles an attack, a release or a
gap) or too quiet to count.  NumPy only.  Real whistles recorded on the rig replace these later; until then they let the training,
the tests and the deployment of the wide ear be built.
"""
from __future__ import annotations

import numpy as np

SR, HOP, FRAME = 16000, 160, 320
LOW_HZ, HIGH_HZ = 500.0, 3800.0
QUIET_DB = -58.0       # frames under this level are "not sounding", as scripts/yamabiko_ear_data.py labels


def _cents(hz):
    return 1200 * np.log2(hz / 440.0)


def _hz(cents):
    return 440.0 * 2 ** (cents / 1200)


def phrase(rng: np.random.Generator, seconds: float = 3.0):
    """(samples float32 in [-1, 1), cents per frame (NaN off), state per frame int8)."""
    n = int(seconds * SR)
    f0 = np.full(n, np.nan)          # instantaneous pitch in cents
    env = np.zeros(n)                # amplitude envelope 0..1 (times the phrase level)
    centre = rng.uniform(_cents(800), _cents(2600))      # this whistler's comfortable middle
    t = int(rng.uniform(.0, .3) * SR)
    cents = centre + rng.normal(0, 300)
    while t < n:
        length = int(rng.uniform(.06, .7) * SR)
        stop = min(n, t + length)
        target = float(np.clip(cents, _cents(LOW_HZ), _cents(HIGH_HZ)))
        f0[t:stop] = target
        attack = int(rng.uniform(.005, .04) * SR); release = int(rng.uniform(.005, .05) * SR)
        e = np.ones(stop - t)
        a = min(attack, len(e)); r = min(release, len(e) - a)
        e[:a] = np.linspace(0, 1, a, endpoint=False) if a else e[:a]
        if r:
            e[len(e) - r:] = np.linspace(1, 0, r)
        env[t:stop] = np.maximum(env[t:stop], e * rng.uniform(.6, 1.0))
        # the next note: a step or a leap, mostly near the middle
        step = rng.choice([rng.normal(0, 150), rng.normal(0, 450), rng.normal(0, 900)], p=[.5, .4, .1])
        cents = .8 * (target + step) + .2 * centre
        join = rng.random()
        if join < .45 and stop < n:                       # legato: glide to the next note, no gap
            glide = int(rng.uniform(.02, .15) * SR)
            g = min(n, stop + glide)
            nxt = float(np.clip(cents, _cents(LOW_HZ), _cents(HIGH_HZ)))
            f0[stop:g] = np.linspace(target, nxt, g - stop)
            env[stop:g] = rng.uniform(.6, 1.0)
            if len(e) and r:                              # no release before a glide
                env[stop - r:stop] = env[stop - r - 1] if stop - r - 1 >= 0 else 1.0
            t = g
        else:                                             # a gap (tonguing or a breath)
            t = stop + int(rng.uniform(.02, .35) * SR)
    # vibrato and drift on the voiced parts
    time = np.arange(n) / SR
    vib = rng.uniform(0, 40) * np.sin(2 * np.pi * rng.uniform(4, 7) * time + rng.uniform(0, 6.3))
    drift = np.cumsum(rng.normal(0, .05, n)); drift -= np.linspace(drift[0], drift[-1], n)
    pitch = f0 + vib + np.clip(drift, -20, 20)
    hz = np.where(np.isfinite(pitch), _hz(np.nan_to_num(pitch, nan=0.0)), 0.0)
    phase = 2 * np.pi * np.cumsum(hz) / SR
    tone = np.sin(phase) + rng.uniform(0, .12) * np.sin(2 * phase + rng.uniform(0, 6.3))
    # breath: noise in a band around the note, and broadband hiss while blowing
    noise = rng.normal(0, 1, n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    mid = np.nanmedian(hz[hz > 0]) if (hz > 0).any() else 1500.0
    band = np.fft.irfft(spec * np.exp(-.5 * ((freqs - mid) / rng.uniform(150, 700)) ** 2), n)
    band /= band.std() + 1e-9
    hiss = noise / (noise.std() + 1e-9)
    blowing = np.convolve(env, np.ones(160) / 160, "same")
    x = env * tone + blowing * (10 ** (rng.uniform(-28, -10) / 20) * band + 10 ** (rng.uniform(-40, -22) / 20) * hiss)
    x = x / (np.abs(x).max() + 1e-9) * 10 ** (rng.uniform(-40, -3) / 20)
    # labels: the causal frame ending at every hop, pitch at its centre
    ends = np.arange(FRAME, n + 1, HOP)
    cents_out = np.full(len(ends), np.nan, np.float32)
    state = np.zeros(len(ends), np.int8)
    for i, e in enumerate(ends):
        w = env[e - FRAME:e]
        seg = x[e - FRAME:e]
        if 20 * np.log10(np.sqrt(np.mean(seg * seg)) + 1e-12) < QUIET_DB:   # as the rig's labels: too quiet
            state[i] = 0
        elif w.min() > .5 and np.isfinite(pitch[e - FRAME:e]).all():
            state[i] = 1
            cents_out[i] = pitch[e - FRAME // 2]
        elif w.max() < .02:
            state[i] = 0
        else:
            state[i] = -1
    return x.astype(np.float32), cents_out, state


def frames(rng: np.random.Generator, count: int, seconds: float = 3.0):
    """``count`` labelled 20 ms frames (int16 (count, 320), cents, state) cut from fresh phrases."""
    xs, cs, ss = [], [], []
    total = 0
    while total < count:
        x, c, s = phrase(rng, seconds)
        pcm = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
        ends = np.arange(FRAME, len(x) + 1, HOP)
        xs.append(np.stack([pcm[e - FRAME:e] for e in ends])); cs.append(c); ss.append(s)
        total += len(ends)
    return (np.concatenate(xs)[:count], np.concatenate(cs)[:count], np.concatenate(ss)[:count])


def noise_frames(rng: np.random.Generator, count: int):
    """``count`` frames with no pitch (int16, state 0): white, pink or brown noise, hiss bursts and clicks from
    about -75 to -35 dBFS, so the ear learns that a quiet or noisy frame is not a note."""
    out = np.empty((count, FRAME), np.int16)
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    for i in range(count):
        spec = np.fft.rfft(rng.normal(0, 1, FRAME))
        colour = rng.choice(3)
        if colour:
            spec = spec / np.maximum(freqs, 20.0) ** (colour / 2)          # pink (1/f) or brown (1/f^2) power
        x = np.fft.irfft(spec, FRAME)
        if rng.random() < .15:                                              # a click
            x[rng.integers(FRAME)] += rng.uniform(5, 30) * x.std()
        x = x / (x.std() + 1e-12) * 10 ** (rng.uniform(-75, -35) / 20)
        out[i] = np.clip(np.round(x * 32767), -32768, 32767)
    return out, np.full(count, np.nan, np.float32), np.zeros(count, np.int8)
