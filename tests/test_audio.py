import numpy as np
import pytest

from flute_rl.audio import SOURCE_KINDS, SR_SELF, SR_SOURCE, room, synth_self, synth_source
from flute_rl.pitch import pitch_track
from flute_rl.sim import cents_to_hz, hz_to_cents

NOTE = float(hz_to_cents(1000.0))


def target():
    return np.concatenate([np.full(20, np.nan), np.full(60, NOTE), np.full(20, np.nan)])


@pytest.mark.parametrize("kind", SOURCE_KINDS)
def test_source_length_silence_and_nyquist(kind):
    y, shift = synth_source(target(), kind, np.random.default_rng(0))
    assert len(y) == 100 * SR_SOURCE // 100
    assert shift % 1200.0 == 0.0 and cents_to_hz(NOTE + shift) < 0.45 * SR_SOURCE
    rest = y[: 15 * SR_SOURCE // 100]
    note = y[40 * SR_SOURCE // 100: 60 * SR_SOURCE // 100]
    assert np.sqrt(np.mean(rest**2)) < 0.1 * np.sqrt(np.mean(note**2))


def test_deterministic_for_seed():
    a, _ = synth_source(target(), "hum", np.random.default_rng(3))
    b, _ = synth_source(target(), "hum", np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_self_sound_pitch_is_recoverable():
    rng = np.random.default_rng(1)
    t = target()
    y = room(synth_self(np.nan_to_num(t, nan=NOTE), np.isfinite(t), rng), SR_SELF, rng, snr_db=(30.0, 30.0))
    _, f0, _ = pitch_track(y, SR_SELF, frame=512, fmin=400, fmax=3900, rms_gate=0.01)
    mid = f0[35:65]
    assert np.nanmedian(np.abs(hz_to_cents(mid) - NOTE)) < 10.0


def test_whistle_is_one_octave_up():
    rng = np.random.default_rng(2)
    y, shift = synth_source(target(), "whistle", rng)
    _, f0, _ = pitch_track(y, SR_SOURCE, frame=1024, fmin=800, fmax=6000, rms_gate=0.01)
    assert shift == 1200.0
    assert np.nanmedian(np.abs(hz_to_cents(f0[35:65]) - (NOTE + 1200.0))) < 15.0
