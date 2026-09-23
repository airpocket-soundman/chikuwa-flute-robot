"""Musically meaningful scores for a performance, with uncertainty.

Mean absolute cents mixes a slow move between notes with being out of tune
on a held note, and a few cents of difference in it is usually noise.  These
scores ask what a listener asks:

* ``in_tune``: fraction of sounding time within +-``tolerance`` cents;
* ``reach_ms``: for each note (a run of sounding frames with one target),
  the time from its start until the pitch enters the tolerance and stays for
  ``hold`` frames; notes that never get there count as not reached;
* ``hit_rate``: fraction of notes reached within ``hit_ms``.

``summarize`` bootstraps 95 % intervals over independent units (rig x song),
so a difference is only called real when the intervals separate.
"""
from __future__ import annotations

import numpy as np
import torch

TOLERANCE = 25.0
HOLD = 5
HIT_MS = 300.0
DT_MS = 10.0


def notes(cents: torch.Tensor, voice: torch.Tensor):
    """Yield (row, start, stop) of each note: sounding frames with one target."""
    c, v = cents.cpu().numpy(), voice.cpu().numpy().astype(bool)
    for row in range(c.shape[0]):
        t, steps = 0, c.shape[1]
        while t < steps:
            if not v[row, t]:
                t += 1; continue
            start = t
            while t < steps and v[row, t] and abs(c[row, t] - c[row, start]) < 1.0:
                t += 1
            yield row, start, t


def unit_scores(played, cents, voice, tolerance=TOLERANCE, hold=HOLD, hit_ms=HIT_MS):
    """Per-row (one rig playing one song) scores: in_tune, hit_rate, median reach."""
    error = (played - cents).abs().cpu().numpy()
    v = voice.cpu().numpy().astype(bool)
    rows = error.shape[0]
    in_tune = [(error[r][v[r]] <= tolerance).mean() if v[r].any() else np.nan for r in range(rows)]
    reach = [[] for _ in range(rows)]
    for row, start, stop in notes(cents, voice):
        good = error[row, start:stop] <= tolerance
        found = np.nan
        for t in range(0, len(good) - hold + 1):
            if good[t:t + hold].all():
                found = t * DT_MS; break
        reach[row].append(found)
    hit = [np.mean([(x <= hit_ms) if np.isfinite(x) else 0.0 for x in r]) if r else np.nan for r in reach]
    median_reach = [np.nanmedian(r) if r and np.isfinite(r).any() else np.nan for r in reach]
    return {"in_tune": np.array(in_tune), "hit_rate": np.array(hit), "reach_ms": np.array(median_reach)}


def summarize(units: dict, samples: int = 2000, seed: int = 0):
    """Mean and bootstrap 95 % interval of each score over the units."""
    rng = np.random.default_rng(seed)
    result = {}
    for key, values in units.items():
        values = np.asarray(values, float); values = values[np.isfinite(values)]
        if not len(values):
            result[key] = {"mean": None, "low": None, "high": None}; continue
        boots = [rng.choice(values, len(values)).mean() for _ in range(samples)]
        result[key] = {"mean": float(values.mean()), "low": float(np.percentile(boots, 2.5)),
                       "high": float(np.percentile(boots, 97.5))}
    return result


def merge(unit_list):
    """Concatenate several ``unit_scores`` results (e.g. several songs)."""
    return {key: np.concatenate([u[key] for u in unit_list]) for key in unit_list[0]}
