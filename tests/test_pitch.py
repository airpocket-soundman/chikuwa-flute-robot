import numpy as np
import pytest

from flute_rl.pitch import pitch_track, yin_frame
from flute_rl.sim import hz_to_cents

SR = 16000


def tone(f, dur=0.1, harmonics=(), noise=0.0, seed=0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    x = 0.5 * np.sin(2 * np.pi * f * t)
    for k, amp in harmonics:
        x += amp * np.sin(2 * np.pi * k * f * t)
    return x + noise * np.random.default_rng(seed).standard_normal(len(t))


@pytest.mark.parametrize("f", [650.0, 1000.0, 1350.0, 1900.0])
def test_flute_range_at_16k_within_2_cents(f):
    est, conf = yin_frame(tone(f)[:1024], SR)
    assert abs(float(hz_to_cents(est) - hz_to_cents(f))) < 2.0 and conf > 0.9


@pytest.mark.parametrize("f", [2400.0, 3000.0])
def test_whistle_range_needs_48k(f):
    est, conf = yin_frame(tone(f, sr=48000)[:2048], 48000)
    assert abs(float(hz_to_cents(est) - hz_to_cents(f))) < 1.0 and conf > 0.9


def test_flute_like_tone_with_noise():
    x = tone(900.0, harmonics=[(2, 0.2), (3, 0.1)], noise=0.02)
    est, _ = yin_frame(x[:1024], SR)
    assert abs(float(hz_to_cents(est) - hz_to_cents(900.0))) < 5.0


def test_noise_is_unvoiced():
    x = 0.3 * np.random.default_rng(1).standard_normal(1024)
    est, _ = yin_frame(x, SR)
    assert np.isnan(est)


def test_track_follows_glide_and_gates_silence():
    t = np.arange(int(SR * 0.6)) / SR
    f = np.where(t < 0.3, 800.0, 1200.0)
    x = 0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = np.concatenate([np.zeros(SR // 5), x])
    times, f0, _ = pitch_track(x, SR)
    assert np.allclose(np.diff(times), 0.01)
    assert np.all(np.isnan(f0[times < 0.15]))
    assert abs(np.nanmedian(f0[(times > 0.25) & (times < 0.45)]) - 800.0) < 5.0
    assert abs(np.nanmedian(f0[times > 0.6]) - 1200.0) < 5.0
