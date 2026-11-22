import numpy as np

from flute_rl.yamabiko.whistle import clean_track, to_target, whistle_track

SR = 16000


def synth(notes_hz, note_s=.4, rest_s=.15, noise_db=-50, level_db=-25, seed=0):
    """A whistle: sine notes with vibrato and a little breath noise, rests between them."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(.3 * SR))]
    for hz in notes_hz:
        t = np.arange(int(note_s * SR)) / SR
        f = hz * 2 ** (15 / 1200 * np.sin(2 * np.pi * 5 * t))          # +-15 cent vibrato
        parts.append(10 ** (level_db / 20) * np.sqrt(2) * np.sin(2 * np.pi * np.cumsum(f) / SR))
        parts.append(np.zeros(int(rest_s * SR)))
    x = np.concatenate(parts)
    x += 10 ** (noise_db / 20) * rng.standard_normal(len(x))
    return (x * 32767).astype(np.int16)


def test_whistle_notes_are_read_within_a_few_cents():
    notes = [1500.0, 1700.0, 1900.0]
    cents = clean_track(whistle_track(synth(notes), SR))
    expected = 1200 * np.log2(np.array(notes) / 440)
    voiced = np.isfinite(cents)
    # one run per note
    edges = np.diff(np.r_[0, voiced.astype(int), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    assert len(starts) == 3
    for (a, b), want in zip(zip(starts, stops), expected):
        assert abs(np.median(cents[a:b]) - want) < 10


def test_target_keeps_intervals_and_fits_the_range():
    notes = [1500.0, 1700.0, 1900.0, 1700.0]
    cents = clean_track(whistle_track(synth(notes), SR))
    target, voice, report = to_target(cents, -10.0, 1200.0)
    assert target.min() >= -10 and target.max() <= 1200
    assert report["clipped_fraction"] == 0
    # intervals between the notes as played back equal the whistled ones
    edges = np.diff(np.r_[0, voice.astype(int), 0])
    medians = [np.median(target[a:b]) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    whistled = 1200 * np.log2(np.array(notes) / 440)
    assert np.allclose(np.diff(medians), np.diff(whistled), atol=10)
    assert not voice[:60].any()                     # lead rest for the plunger to get to the first note
