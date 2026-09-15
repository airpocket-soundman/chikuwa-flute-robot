"""The calibration fits must recover known simulator parameters from synthetic recordings."""
import numpy as np

from flute_rl.audio import room, synth_self
from flute_rl.calibrate import analyse, fit_tube, fit_window
from flute_rl.sim import FluteParams, FluteSim, speed_of_sound

SR = 16000


def record(cents: float, sounding: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.full(100, cents)
    s = np.concatenate([np.zeros(10, bool), np.full(80, sounding), np.zeros(10, bool)])
    return room(synth_self(c, s, rng, sr=SR), SR, rng, snr_db=(30.0, 30.0))


def test_tube_fit_recovers_length_and_temperature():
    p = FluteParams(tube_len=0.140, end_corr=0.004, temp_c=18.0)
    sim = FluteSim(p, np.random.default_rng(0))
    depths = np.array([0.02, 0.035, 0.05, 0.065, 0.08])
    hz = []
    for i, x in enumerate(depths):
        cents, _, _ = sim.pitch_at(x, p.theta_opt_deg)
        hz.append(analyse(record(cents, True, i), SR).hz)
    fit = fit_tube(depths, np.array(hz))
    assert abs(fit.acoustic_len_m - (p.tube_len + p.end_corr)) < 0.001
    assert abs(fit.speed_of_sound - speed_of_sound(p.temp_c)) < 3.0
    assert np.max(np.abs(fit.residual_cents)) < 5.0


def test_actuator_fit_recovers_speed_dead_band_and_curve():
    import dataclasses

    from flute_rl.calibrate import fit_actuator, speed_from_recording

    p = FluteParams(v_max_in=0.09, deadband=0.25, pwm_curve=1.3, tau_v=0.02, cmd_delay=0, backlash=0.0)
    S = p.tube_len + p.end_corr
    c = speed_of_sound(p.temp_c)
    pwms = np.array([0.4, 0.55, 0.7, 0.85, 1.0])
    speeds = []
    for i, u in enumerate(pwms):
        sim = FluteSim(dataclasses.replace(p, pitch_noise=0.0, dropout=0.0), np.random.default_rng(i))
        sim.reset(x0=0.01)
        cents, snd = [], []
        steps = int(0.06 / (p.v_max_in * ((u - p.deadband) / (1 - p.deadband)) ** p.pwm_curve) / 0.01)
        for _ in range(min(max(steps, 20), 300)):
            s = sim.step(u, p.theta_opt_deg / p.angle_range_deg)
            cents.append(s.cents)
            snd.append(s.sounding)
        rng = np.random.default_rng(50 + i)
        y = room(synth_self(np.array(cents), np.array(snd), rng, sr=SR), SR, rng, snr_db=(35.0, 35.0))
        speeds.append(speed_from_recording(y, SR, S, c))
    fit = fit_actuator(pwms, np.array(speeds))
    assert abs(fit.v_max - p.v_max_in) / p.v_max_in < 0.08
    assert abs(fit.deadband - p.deadband) < 0.06
    assert abs(fit.pwm_curve - p.pwm_curve) < 0.3


def test_window_fit_finds_edges_and_bend():
    p = FluteParams(theta_opt_deg=1.5, win_lo_deg=4.0, win_hi_deg=3.0, win_narrowing=0.0, k_theta=12.0)
    sim = FluteSim(p, np.random.default_rng(0))
    angles = np.arange(-6.0, 8.01, 0.5)
    hz, snd = [], []
    for i, a in enumerate(angles):
        cents, sounding, _ = sim.pitch_at(0.04, a)
        t = analyse(record(cents, sounding, 100 + i), SR)
        hz.append(t.hz)
        snd.append(t.sounding)
    fit = fit_window(angles, np.array(hz), np.array(snd))
    assert abs(fit.lo_deg - (p.theta_opt_deg - p.win_lo_deg)) <= 0.51
    assert abs(fit.hi_deg - (p.theta_opt_deg + p.win_hi_deg)) <= 0.51
    assert abs(fit.cents_per_deg - p.k_theta) < 1.5
