"""Metrics of a Yamabiko session, per song (see docs/yamabiko.md, "評価指標").

All errors use the pitch the flute really played (the simulator knows it),
not the estimate. An "onset" is a step where the tone starts (valve opened,
tone built up) during a note; the run is the stretch that keeps sounding
over notes after it.

* first1       |error| at the first sounding step of the song's FIRST note: the hardest moment,
               nothing has been heard since the homing wiped the dead reckoning
* onset1       |error| at the first sounding step: pure feed-forward, no feedback can have arrived yet
* onset5       mean |error| over the first 5 sounding steps
* converge     fraction of runs that get within CONVERGE_CENTS ...
* converge_ms  ... and, over those that do, how long it took
* steady       mean |error| after convergence, on settled steps
* mean_abs     mean |error| over all sounding note steps
* sounding     fraction of note steps that sounded
* blowups      runs (per 100) that, after converging, were off by more than BLOWUP_CENTS for 3+ settled steps

"Settled" leaves out the steps right after the target itself moved by a semitone or more
(a slurred note change): chasing a new note is not an error of the position estimate.
* overblow     fraction of sounding note steps that were overblown (a twelfth up)
"""
from __future__ import annotations

import numpy as np

from ..sim import DT

CONVERGE_CENTS = 20.0
BLOWUP_CENTS = 150.0
SETTLE_STEPS = 15      # a step is settled if the target moved less than ...
SETTLE_RANGE = 100.0   # ... a semitone over the last SETTLE_STEPS steps of the run (vibrato stays settled)


def _settled(tgt: np.ndarray) -> np.ndarray:
    w = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(SETTLE_STEPS, tgt[0]), tgt]), SETTLE_STEPS + 1)
    return (w.max(axis=1) - w.min(axis=1)) < SETTLE_RANGE
KEYS = ("mean_abs", "first1", "onset1", "onset5", "converge", "converge_ms", "steady", "sounding", "blowups", "overblow")


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def song_metrics(logs: dict, sched, rigs: np.ndarray | None = None) -> list[dict]:
    """One dict of metrics per song, pooled over the rigs (or only `rigs`)."""
    rows = np.arange(sched.n) if rigs is None else np.asarray(rigs)
    out = []
    for k in range(sched.k):
        cols = sched.song == k
        tgt = sched.target[np.ix_(rows, cols)]
        cents = logs["cents"][np.ix_(rows, cols)]
        snd = logs["sounding"][np.ix_(rows, cols)]
        over = logs["overblown"][np.ix_(rows, cols)]
        note = np.isfinite(tgt)
        err = np.abs(cents - np.nan_to_num(tgt))
        played = note & snd
        acc = {"first1": [], "onset1": [], "onset5": [], "conv": [], "conv_steps": [], "steady": [], "blow": []}
        for i in range(len(rows)):
            for j, (s, e) in enumerate(_runs(played[i])):
                ev = err[i, s:e]
                if j == 0:
                    acc["first1"].append(ev[0])
                acc["onset1"].append(ev[0])
                acc["onset5"].append(ev[:5].mean())
                inside = np.flatnonzero(ev < CONVERGE_CENTS)
                if inside.size:
                    j = int(inside[0])
                    acc["conv"].append(1.0)
                    acc["conv_steps"].append(j)
                    tail, settled = ev[j:], _settled(tgt[i, s:e])[j:]
                    acc["steady"].extend(tail[settled].tolist())
                    bad = (tail > BLOWUP_CENTS) & settled
                    acc["blow"].append(float(any(e2 - s2 >= 3 for s2, e2 in _runs(bad))))
                else:
                    acc["conv"].append(0.0)

        def mean(v):
            return float(np.mean(v)) if len(v) else float("nan")

        out.append({
            "mean_abs": float(err[played].mean()) if played.any() else float("nan"),
            "first1": mean(acc["first1"]),
            "onset1": mean(acc["onset1"]),
            "onset5": mean(acc["onset5"]),
            "converge": mean(acc["conv"]),
            "converge_ms": mean(acc["conv_steps"]) * DT * 1000.0 if acc["conv_steps"] else float("nan"),
            "steady": mean(acc["steady"]),
            "sounding": float(snd[note].mean()) if note.any() else float("nan"),
            "blowups": 100.0 * mean(acc["blow"]) if acc["blow"] else float("nan"),
            "overblow": float(over[played].mean()) if played.any() else float("nan"),
        })
    return out
