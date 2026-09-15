import numpy as np
import pytest

from flute_rl.sim import FluteParams, hz_to_cents, x_for_cents
from flute_rl.targets import F_HI, F_LO, MAX_LEVEL, from_pitch_track, make_target, sample_level, scale_notes


def test_scale_notes_in_range():
    notes = scale_notes(key=0)
    assert len(notes) >= 7
    assert notes.min() >= hz_to_cents(F_LO) and notes.max() <= hz_to_cents(F_HI)


@pytest.mark.parametrize("level", range(MAX_LEVEL + 1))
def test_targets_have_rests_and_stay_in_range(level):
    rng = np.random.default_rng(level)
    for _ in range(20):
        t = make_target(rng, level)
        assert np.isnan(t[0]) and np.isnan(t[-1])
        voiced = t[np.isfinite(t)]
        assert voiced.size > 0
        assert voiced.min() >= hz_to_cents(F_LO) - 50 and voiced.max() <= hz_to_cents(F_HI) + 50


def test_target_range_is_reachable_for_randomised_rigs():
    rng = np.random.default_rng(3)
    for _ in range(500):
        p = FluteParams.sample(rng)
        lo = x_for_cents(float(hz_to_cents(F_LO)) - 50, p)
        hi = x_for_cents(float(hz_to_cents(F_HI)) + 50, p)
        assert 0.0 <= lo and hi <= p.stroke


def test_sample_level_respects_progress():
    rng = np.random.default_rng(0)
    assert {sample_level(rng, 0.0) for _ in range(50)} == {0}
    assert max(sample_level(rng, 1.0) for _ in range(200)) == MAX_LEVEL


def test_from_pitch_track_shifts_octaves_into_range():
    hz = np.array([0, 2400, 2400, 2700, np.nan, 3000, 0], dtype=float)  # a high whistle
    t = from_pitch_track(hz, dt_in=0.02)
    voiced = t[np.isfinite(t)]
    assert voiced.min() >= hz_to_cents(F_LO) - 1e-6 and voiced.max() <= hz_to_cents(F_HI) + 1e-6
    assert np.isnan(t[0])
    with pytest.raises(ValueError):
        from_pitch_track([0, np.nan], dt_in=0.01)
