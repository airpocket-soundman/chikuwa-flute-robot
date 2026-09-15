import numpy as np

from flute_rl import FluteEnv, FluteParams, PhysicsPriorPolicy, ZeroPolicy, rollout


def test_multi_take_length_and_layout():
    env = FluteEnv(level=1, takes=3)
    assert {"prev_err", "prev_mask", "prev_act", "take"} <= set(env.obs_layout)
    log = rollout(env, ZeroPolicy(), seed=0)
    assert len(log["rewards"]) == 3 * len(env.target)
    assert list(np.unique(log["take"])) == [0, 1, 2]
    assert len(log["per_take"]) == 3


def test_single_take_has_no_previous_take_block():
    assert "prev_err" not in FluteEnv(level=1).obs_layout


def test_rig_is_homed_between_takes():
    env = FluteEnv(level=0, takes=2)
    log = rollout(env, PhysicsPriorPolicy(), seed=1)
    first_of_take2 = int(np.argmax(log["take"] == 1))
    assert log["x"][first_of_take2] < 0.005


def test_previous_take_error_is_observed():
    p = FluteParams(dropout=0.0, pitch_noise=0.0)
    target = np.full(60, 1400.0)
    env = FluteEnv(randomize=False, params=p, takes=2, target_fn=lambda rng: target)
    obs, _ = env.reset(seed=0)
    lay = env.obs_layout
    assert obs[lay["prev_mask"]].sum() == 0  # nothing to compare against in take 1
    first_take_err = []
    for _ in range(len(target)):
        obs, _, _, _, info = env.step([0.5, float(obs[lay["angle_comp"]][0])])
        first_take_err.append(info["state"].measured - target[0])
    # obs now belongs to the first step of take 2
    seen = obs[lay["prev_err"]] * 600.0
    expected = np.clip(np.array(first_take_err[: env.lookahead]), -1800, 1800)
    assert obs[lay["take"]][0] == 1.0
    assert np.allclose(seen, expected, atol=1e-3)


def test_classic_ilc_improves_take_by_take():
    from flute_rl import ILCPolicy

    env = FluteEnv(progress=0.8, takes=4)
    runs = [rollout(env, ILCPolicy(), seed=s) for s in range(12)]
    err = [np.nanmean([r["per_take"][i]["mean_abs_cents"] for r in runs]) for i in range(4)]
    assert err[3] < 0.6 * err[0]


def test_physics_prior_repeats_itself_without_learning():
    env = FluteEnv(level=3, takes=3)
    log = rollout(env, PhysicsPriorPolicy(), seed=2)
    errs = [t["mean_abs_cents"] for t in log["per_take"]]
    assert np.nanmax(errs) - np.nanmin(errs) < 15.0  # no improvement mechanism, only noise
