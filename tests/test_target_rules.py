import numpy as np

from flute_rl.sim import DT
from flute_rl.targets import GAP_RANGE, MAX_JUMP, MIN_NOTE, make_target


def segments(t):
    on = np.isfinite(t).astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on, [0]])))
    return list(zip(edges[::2], edges[1::2]))


def test_no_short_notes_and_no_big_leaps():
    rng = np.random.default_rng(0)
    for _ in range(300):
        t = make_target(rng, int(rng.integers(5)))
        for a, b in segments(t):
            assert (b - a) * DT >= MIN_NOTE - 1e-9
        v = t[np.isfinite(t)]
        steps = np.abs(np.diff(v))
        assert steps.max(initial=0.0) <= MAX_JUMP + 40.0 + 1e-6  # + the deepest vibrato at a note boundary


def test_articulation():
    rng = np.random.default_rng(1)
    for _ in range(50):
        legato = make_target(rng, 3, articulation="legato")
        assert len(segments(legato)) == 1  # slurred / tied: one continuous sounding stretch
        detached = make_target(rng, 3, articulation="detached")
        segs = segments(detached)
        assert len(segs) >= 4
        for (_, end), (start, _) in zip(segs[:-1], segs[1:]):
            gap = (start - end) * DT
            assert GAP_RANGE[0] - DT <= gap <= GAP_RANGE[1] + DT
