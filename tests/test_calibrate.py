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
