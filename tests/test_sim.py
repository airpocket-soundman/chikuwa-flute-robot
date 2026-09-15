from dataclasses import replace

import numpy as np

from flute_rl.sim import FluteParams, FluteSim, hz_to_cents, x_for_cents


def quiet(**kw):
    return replace(FluteParams(), pitch_noise=0.0, dropout=0.0, **kw)


def test_pitch_rises_as_plunger_goes_in():
    sim = FluteSim(quiet())
    cents = [sim.pitch_at(x, 0.0)[0] for x in np.linspace(0.0, 0.08, 9)]
    assert np.all(np.diff(cents) > 0)


def test_temperature_shifts_about_3_cents_per_degree():
    x = 0.05
    c20 = FluteSim(quiet(temp_c=20.0)).pitch_at(x, 0.0)[0]
    c30 = FluteSim(quiet(temp_c=30.0)).pitch_at(x, 0.0)[0]
    assert 2.5 < (c30 - c20) / 10.0 < 3.5


def test_x_for_cents_inverts_tube_model():
    p = quiet()
    target = float(hz_to_cents(1000.0))
    assert abs(FluteSim(p).pitch_at(x_for_cents(target, p), 0.0)[0] - target) < 1e-6


def test_deadband_blocks_small_pwm():
    sim = FluteSim(quiet())
    for _ in range(50):
        s = sim.step(0.1, 0.0)
    assert s.x == 0.0


def test_command_delay():
    sim = FluteSim(quiet(cmd_delay=2, backlash=0.0))
    s1 = sim.step(1.0, 0.0)
    s2 = sim.step(1.0, 0.0)
    s3 = sim.step(1.0, 0.0)
    assert s1.x == 0.0 and s2.x == 0.0 and s3.x > 0.0


def test_full_speed_matches_actuator_rating():
    sim = FluteSim(quiet())
    for _ in range(40):  # 0.4 s, well past the velocity time constant
        s = sim.step(1.0, 0.0)
    assert abs(s.v - FluteParams().v_max_in) < 0.005


def test_servo_readback_is_quantised():
    p = quiet()
    sim = FluteSim(p)
    for _ in range(30):
        s = sim.step(0.0, 0.33)
    assert abs(s.theta_readback / p.servo_step_deg - round(s.theta_readback / p.servo_step_deg)) < 1e-9
    assert abs(s.theta - 0.33 * p.angle_range_deg) < p.servo_step_deg


def test_sounding_window_and_overblow():
    p = quiet()
    sim = FluteSim(p)
    assert sim.pitch_at(0.05, p.theta_opt_deg)[1]
    assert not sim.pitch_at(0.05, p.theta_opt_deg + 10.0)[1]
    assert not sim.pitch_at(0.05, p.theta_opt_deg - 10.0)[1]
    low, _, _ = sim.pitch_at(0.10, p.theta_opt_deg)
    high, sounding, overblown = sim.pitch_at(0.10, p.theta_opt_deg + 2.5)
    assert sounding and overblown and high - low > 1000


def test_sample_spread_zero_is_nominal():
    assert FluteParams.sample(np.random.default_rng(0), spread=0.0) == FluteParams()


def test_sample_stays_physical():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = FluteParams.sample(rng)
        assert p.cmd_delay >= 0 and p.obs_delay >= 0 and p.dropout >= 0 and p.backlash >= 0
        assert 0.0 < p.deadband < 1.0 and p.win_lo_deg > 0 and p.win_hi_deg > 0


def test_find_sounding_angle_lands_in_window():
    rng = np.random.default_rng(2)
    for _ in range(50):
        p = FluteParams.sample(rng)
        sim = FluteSim(p, rng)
        a = sim.find_sounding_angle(0.07)
        assert sim.pitch_at(0.07, a)[1]
