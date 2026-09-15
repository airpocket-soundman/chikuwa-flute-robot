import numpy as np

from flute_rl import AdaptivePolicy, FluteEnv, FluteParams, ILCPolicy, fit_rig, rollout
from flute_rl.adapt import apply_z, predict_cents


def test_fit_recovers_actuator_speeds():
    nominal = FluteParams()
    z_true = np.array([np.log(1.18), np.log(0.85), 0.004, 0.2])
    rng = np.random.default_rng(0)
    pwm = np.concatenate([np.full(20, 0.8), np.zeros(30), np.full(15, -0.7), np.zeros(30), np.full(25, 0.6), np.zeros(30)])
    meas = predict_cents(z_true, pwm, nominal) + rng.normal(0.0, 3.0, len(pwm))
    meas[:5] = np.nan  # unvoiced frames are skipped
    z, rms = fit_rig([(pwm, meas)], nominal)
    p = apply_z(z, nominal)
    assert abs(p.v_max_in / nominal.v_max_in - 1.18) < 0.03
    assert abs(p.v_max_out / nominal.v_max_out - 0.85) < 0.05
    assert rms < 10.0


def test_adaptive_policy_beats_classic_ilc_on_take_two():
    env = FluteEnv(progress=0.8, takes=2)
    seeds = range(1_000_000, 1_000_012)
    ada = [rollout(env, AdaptivePolicy(), seed=s)["per_take"] for s in seeds]
    ilc = [rollout(env, ILCPolicy(gain=0.7), seed=s)["per_take"] for s in seeds]
    ada2 = np.nanmean([r[1]["mean_abs_cents"] for r in ada])
    ilc2 = np.nanmean([r[1]["mean_abs_cents"] for r in ilc])
    assert np.nanmean([r[0]["mean_abs_cents"] for r in ada]) > 2.0 * ada2  # take 2 is much better than take 1
    assert ada2 < 0.7 * ilc2
