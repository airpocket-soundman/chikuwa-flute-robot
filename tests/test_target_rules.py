import numpy as np

from flute_rl.sim import DT, hz_to_cents
from flute_rl.targets import BIRDS, F_HI, F_LO, GAP_RANGE, MAX_JUMP, MIN_NOTE, bird_song, make_target


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


def test_bird_songs_fit_the_flute():
    rng = np.random.default_rng(2)
    lo, hi = float(hz_to_cents(F_LO)), float(hz_to_cents(F_HI))
    for sp in BIRDS:
        for _ in range(20):
            t = bird_song(rng, sp)
            v = t[np.isfinite(t)]
            assert v.size and v.min() >= lo - 1e-6 and v.max() <= hi + 1e-6
            assert np.isnan(t[0]) and np.isnan(t[-1])  # lead-in and tail rests like the other pieces
    uguisu = bird_song(np.random.default_rng(0), "uguisu")
    assert len(segments(uguisu)) == 2  # "hoo" + "hokekyo"


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
