"""Evolution strategies for the recurrent policy (numpy only).

One generation: P candidates (antithetic pairs) each play the same R rigs
through a whole session of K songs, with the hidden state carried across
songs. Fitness = mean over rigs of the per-song reward, weighted so that
later songs count more (playing the first song a bit worse to learn the rig
pays off). Some sessions swap the rig at a song boundary without telling the
policy, so it must also learn to drop a stale memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import GRUPolicy
from .rig import Rig, RigParams
from .session import Swap, draw_pieces, make_schedule, run_session


@dataclass
class TaskSpec:
    rigs: int = 8            # rigs per candidate
    songs: int = 5           # songs per session
    hidden: int = 24
    harsh: float = 0.0
    spread: float = 1.0
    swap_prob: float = 0.25  # fraction of rigs swapped in a session (at one random song boundary)
    weight_growth: float = 1.0  # song k weight = 1 + weight_growth * k / (songs - 1)
    first_weight: float = 1.0   # how much more the first note of a song counts in the reward
    progress: float = 0.8


def song_weights(songs: int, growth: float) -> np.ndarray:
    k = np.arange(songs)
    return 1.0 + growth * k / max(songs - 1, 1)


def fitness(thetas: np.ndarray, spec: TaskSpec, seed: int, bank=None) -> np.ndarray:
    """(P,) fitness of each row of `thetas` on one shared session (common rigs, songs and swaps)."""
    thetas = np.atleast_2d(thetas)
    P, R = thetas.shape[0], spec.rigs
    rng = np.random.default_rng(seed)
    base = RigParams.sample(rng, R, spec.spread, spec.harsh)
    sched = make_schedule(draw_pieces(rng, R, spec.songs, bank, spec.progress))
    swap = None
    if spec.swap_prob > 0.0 and spec.songs > 1:
        mask = rng.random(R) < spec.swap_prob
        if mask.any():
            song = int(rng.integers(1, spec.songs))
            swap = Swap(song, np.tile(mask, P), RigParams.sample(rng, R, spec.spread, spec.harsh).tile(P))
    rig = Rig(base.tile(P), np.random.default_rng(seed + 1))
    res = run_session(rig, sched.tile(P), GRUPolicy(thetas, spec.hidden, carry=True), swap=swap,
                      first_weight=spec.first_weight)
    w = song_weights(spec.songs, spec.weight_growth)
    per_rig = res["reward"] @ w / w.sum()
    return per_rig.reshape(P, R).mean(axis=1)


class Adam:
    def __init__(self, dim: int, lr: float):
        self.m, self.v, self.lr, self.t = np.zeros(dim), np.zeros(dim), lr, 0

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Gradient ascent step."""
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        mh = self.m / (1 - 0.9**self.t)
        vh = self.v / (1 - 0.999**self.t)
        return theta + self.lr * mh / (np.sqrt(vh) + 1e-8)


def es_gradient(eps: np.ndarray, scores: np.ndarray, sigma: float) -> np.ndarray:
    """Rank-shaped antithetic estimate. eps: (half, D); scores: (half, 2) for +eps / -eps."""
    ranks = scores.ravel().argsort().argsort().reshape(scores.shape) / (scores.size - 1) - 0.5
    return ((ranks[:, 0] - ranks[:, 1])[:, None] * eps).sum(axis=0) / (len(eps) * sigma)
