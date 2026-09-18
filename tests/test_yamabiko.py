import numpy as np
import pytest

from flute_rl.sim import DT, hz_to_cents
from flute_rl.targets import make_target
from flute_rl.yamabiko import (OVERBLOW_CENTS, Encoder, ExternalFit, GRUPolicy, Observer, OpenLoop, Oracle, Rig,
                               RigParams, Schedule, Swap, TaskSpec, draw_pieces, fitness, make_schedule, run_session,
                               song_metrics)
from flute_rl.yamabiko.rig import speed_of_sound
from flute_rl.yamabiko.session import HOME_STEPS

QUIET = dict(pitch_noise=0.0, dropout=0.0, pitch_jitter=0.0, octave_err=0.0)


def quiet_rigs(n, **kw):
    return RigParams.nominal(n, **{**QUIET, **kw})


def schedule(n, k, seed=0):
    return make_schedule(draw_pieces(np.random.default_rng(seed), n, k))


def play(params, sched, ctrl, seed=1, **kw):
    return run_session(Rig(params, np.random.default_rng(seed)), sched, ctrl, keep_logs=True, **kw)


def test_pitch_depends_only_on_the_tube_with_the_valve_open():
    p = quiet_rigs(3, onset_cents=0.0)
    rig = Rig(p, np.random.default_rng(0))
    for _ in range(30):
        rig.step(np.array([0.6, 0.8, 1.0]), np.ones(3, bool))
    for _ in range(20):
        out = rig.step(np.zeros(3), np.ones(3, bool))
    assert out["sounding"].all() and (out["x"] > 0.01).all()
    length = p.tube_len + p.end_corr - out["x"]
    expected = hz_to_cents(speed_of_sound(p.temp_c) / (4.0 * length))
    np.testing.assert_allclose(out["cents"], expected, atol=1e-9)
    np.testing.assert_allclose(p.x_for_cents(out["cents"]), out["x"], atol=1e-12)


def test_a_closed_tube_overblows_a_twelfth_not_an_octave():
    p = quiet_rigs(1, onset_cents=0.0, overblow_len=0.5)  # always too short: overblows as soon as it sounds
    rig = Rig(p, np.random.default_rng(0))
    for _ in range(10):
        out = rig.step(np.zeros(1), np.ones(1, bool))
    assert out["overblown"][0]
    assert out["cents"][0] - p.cents_at(out["x"])[0] == pytest.approx(OVERBLOW_CENTS)
    assert OVERBLOW_CENTS == pytest.approx(1901.955, abs=1e-3)


def test_valve_delay_and_tone_onset():
    p = quiet_rigs(1, valve_delay=2, onset_s=0.03)
    rig = Rig(p, np.random.default_rng(0))
    snd = [rig.step(np.zeros(1), np.array([t < 10]))["sounding"][0] for t in range(20)]
    # air arrives 2 steps after the valve opens and the tone needs 3 steps; closing stops it 2 steps later
    assert snd.index(True) == 2 + 3 - 1
    assert snd[11] and not snd[12]


def test_pitch_is_heard_obs_delay_steps_late():
    p = quiet_rigs(1, obs_delay=4)
    rig = Rig(p, np.random.default_rng(0))
    outs = [rig.step(np.array([0.7]), np.ones(1, bool)) for _ in range(30)]
    for t in range(4, 30):
        a, b = outs[t]["heard"][0], outs[t - 4]["measured"][0]
        assert (np.isnan(a) and np.isnan(b)) or a == b


def test_every_song_starts_from_home():
    sched = schedule(4, 3)
    res = play(RigParams.sample(np.random.default_rng(0), 4), sched, OpenLoop())
    for s in sched.starts:  # the rod sits half the backlash away from the screw at the end stop
        assert np.all(res["logs"]["x"][:, s + HOME_STEPS - 1] < 0.001)


def test_valve_opens_ahead_of_each_note():
    sched = schedule(2, 1)
    p = quiet_rigs(2)
    res = play(p, sched, Oracle())
    note = np.isfinite(sched.target)
    snd = res["logs"]["sounding"]
    # on the nominal rig the tone starts exactly with the note and stops with it (the valve schedule is nominal)
    assert np.mean(snd[note]) > 0.99
    assert np.mean(snd[~note]) < 0.01


def test_oracle_and_open_loop_are_accurate_on_the_nominal_rig():
    sched = schedule(4, 2)
    p = quiet_rigs(4, onset_cents=0.0)
    for ctrl, limit in ((Oracle(), 10.0), (OpenLoop(), 25.0), (Encoder(), 10.0)):
        m = song_metrics(play(p, sched, ctrl)["logs"], sched)
        assert all(d["mean_abs"] < limit for d in m), (ctrl.name, m)


def test_the_hand_observer_beats_open_loop_on_random_rigs():
    sched = schedule(24, 2)
    p = RigParams.sample(np.random.default_rng(3), 24)
    ol = song_metrics(play(p, sched, OpenLoop())["logs"], sched)
    ob = song_metrics(play(p, sched, Observer())["logs"], sched)
    assert all(b["mean_abs"] < 0.6 * a["mean_abs"] for a, b in zip(ol, ob))
    assert all(b["converge"] > 0.9 for b in ob)


def test_external_fit_recovers_the_actuator_speeds():
    p = quiet_rigs(2, v_in=0.18, v_out=0.12, onset_cents=0.0)
    rng = np.random.default_rng(0)
    melodies = [[make_target(rng, 3) for _ in range(2)] for _ in range(3)]  # moves both ways
    ctrl = ExternalFit()
    play(p, make_schedule(melodies), ctrl)
    np.testing.assert_allclose(ctrl.theta[:, 0], 1.2, atol=0.05)
    np.testing.assert_allclose(ctrl.theta[:, 1], 0.8, atol=0.05)


def test_external_memory_gets_better_song_by_song():
    sched = schedule(32, 3, seed=5)
    p = RigParams.sample(np.random.default_rng(5), 32)
    ob = song_metrics(play(p, sched, Observer())["logs"], sched)
    ext = song_metrics(play(p, sched, ExternalFit())["logs"], sched)
    assert ext[0]["mean_abs"] == pytest.approx(ob[0]["mean_abs"])  # nothing learned before the first song
    assert ext[2]["mean_abs"] < 0.7 * ob[2]["mean_abs"]
    assert ext[2]["onset1"] < ob[2]["onset1"]


def test_fresh_gru_plays_like_open_loop():
    sched = schedule(3, 2)
    p = RigParams.sample(np.random.default_rng(1), 3)
    theta = GRUPolicy.init_params(np.random.default_rng(0), 8)
    a = play(p, sched, OpenLoop())["logs"]
    b = play(p, sched, GRUPolicy(theta, hidden=8))["logs"]
    np.testing.assert_allclose(a["cents"], b["cents"])


def random_policy(hidden, seed=0):
    rng = np.random.default_rng(seed)
    return GRUPolicy.init_params(rng, hidden) + 0.3 * rng.standard_normal(GRUPolicy.n_params(hidden))


def test_gru_memory_is_carried_across_songs_only_when_asked():
    sched = schedule(2, 2)
    p = quiet_rigs(2)
    theta = random_policy(8)
    carry = play(p, sched, GRUPolicy(theta, hidden=8, carry=True))["logs"]["cents"]
    reset = play(p, sched, GRUPolicy(theta, hidden=8, carry=False))["logs"]["cents"]
    first, second = sched.song == 0, sched.song == 1
    np.testing.assert_allclose(carry[:, first], reset[:, first])
    assert not np.allclose(carry[:, second], reset[:, second])


def test_population_members_play_their_own_rigs():
    sched = schedule(2, 1)
    p = quiet_rigs(2)
    fresh = GRUPolicy.init_params(np.random.default_rng(0), 8)
    thetas = np.stack([random_policy(8), fresh])
    both = play(p.tile(2), sched.tile(2), GRUPolicy(thetas, hidden=8))["logs"]["cents"]
    alone = play(p, sched, OpenLoop())["logs"]["cents"]
    np.testing.assert_allclose(both[2:], alone)  # member 1 (zero read-out) on rigs 2, 3
    assert not np.allclose(both[:2], alone)


def test_swap_puts_a_new_rig_in_only_where_masked():
    sched = schedule(2, 2)
    old, new = quiet_rigs(2), quiet_rigs(2, tube_len=0.160)
    rig = Rig(old, np.random.default_rng(0))
    run_session(rig, sched, OpenLoop(), swap=Swap(1, np.array([True, False]), new))
    assert rig.p.tube_len[0] == pytest.approx(0.160) and rig.p.tube_len[1] == pytest.approx(0.150)


def test_fitness_runs_on_a_tiny_session():
    spec = TaskSpec(rigs=2, songs=2, hidden=4, swap_prob=0.5)
    D = GRUPolicy.n_params(4)
    thetas = np.stack([GRUPolicy.init_params(np.random.default_rng(i), 4) for i in range(2)])
    f = fitness(thetas, spec, seed=0)
    assert f.shape == (2,) and np.all(np.isfinite(f)) and np.all(f <= 0.0)
    assert thetas.shape[1] == D


def test_metrics_onset_convergence_and_censoring():
    T = 40
    target = np.full((2, T), np.nan)
    target[:, 5:25] = 1000.0
    note = np.isfinite(target)
    sched = Schedule(target, np.zeros(T, bool), np.zeros(T, int), [0], target.copy(), note, note, 1)
    cents = np.full((2, T), 1000.0)
    cents[0, 5:25] = 1000.0 + np.array([100, 60, 30, 15, 10] + [5] * 15)
    cents[1, 5:25] = 1000.0 + 50.0  # never converges
    snd = note
    logs = {"cents": cents, "sounding": snd, "overblown": np.zeros((2, T), bool)}
    m = song_metrics(logs, sched)[0]
    assert m["onset1"] == pytest.approx((100 + 50) / 2)
    assert m["first1"] == m["onset1"]  # there is only one note
    assert m["converge"] == pytest.approx(0.5)
    assert m["converge_ms"] == pytest.approx(3 * DT * 1000)
    assert m["steady"] == pytest.approx((15 + 10 + 5 * 15) / 17)
    assert m["sounding"] == pytest.approx(1.0)
