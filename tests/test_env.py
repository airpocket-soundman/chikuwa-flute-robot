import numpy as np

from flute_rl import FluteEnv, FluteParams, PhysicsPriorPolicy, ZeroPolicy, rollout
from flute_rl.targets import make_target


def test_reset_and_step_shapes():
    env = FluteEnv(level=1)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32
    assert "params" in info
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (env.obs_dim,) and isinstance(r, float) and not trunc
    assert set(info["reward_terms"]) == {"pitch", "silence", "rest", "octave", "smooth"}


def test_episode_length_matches_target():
    env = FluteEnv(level=0)
    log = rollout(env, ZeroPolicy(), seed=3)
    assert len(log["rewards"]) == len(log["target"])


def test_seed_is_deterministic():
    env = FluteEnv(level=3)
    a = rollout(env, PhysicsPriorPolicy(), seed=11)
    b = rollout(env, PhysicsPriorPolicy(), seed=11)
    assert np.array_equal(a["target"], b["target"], equal_nan=True)
    assert np.allclose(a["rewards"], b["rewards"])


def test_open_loop_observation_has_no_live_pitch():
    env = FluteEnv(level=1, feedback=False)
    assert "fb_err" not in env.obs_layout


def test_feedback_observation_is_delayed():
    p = FluteParams(obs_delay=4, dropout=0.0, pitch_noise=0.0)
    env = FluteEnv(feedback=True, randomize=False, params=p, target_fn=lambda rng: np.full(80, 1400.0))
    env.reset(seed=0)
    env.angle_comp = p.theta_opt_deg  # make sure it sounds
    errs = []
    for _ in range(30):
        obs, _, _, _, info = env.step([0.0, 0.0])
        s = info["state"]
        errs.append(s.measured - info["target"])
        seen = obs[env.obs_layout["fb_err"]][0] * 600.0
        if len(errs) > p.obs_delay:
            assert abs(seen - np.clip(errs[-1 - p.obs_delay], -1800, 1800)) < 1e-3


def test_rest_and_silence_penalties():
    p = FluteParams(dropout=0.0)
    rest = lambda rng: np.full(30, np.nan)  # noqa: E731
    env = FluteEnv(randomize=False, params=p, target_fn=rest)
    env.reset(seed=0)
    terms = [env.step([0.0, 0.0])[4]["reward_terms"]["rest"] for _ in range(30)]
    assert min(terms) < 0  # sounding during a rest is penalised

    note = lambda rng: np.full(30, 1400.0)  # noqa: E731
    env = FluteEnv(randomize=False, params=p, target_fn=note)
    env.reset(seed=0)
    terms = [env.step([0.0, 1.0])[4]["reward_terms"]["silence"] for _ in range(30)]
    assert terms[-1] < 0  # angle far out of the window: silent during a note


def test_options_target_overrides_generator():
    env = FluteEnv(level=4)
    tgt = make_target(np.random.default_rng(5), 0)
    env.reset(seed=1, options={"target": tgt})
    assert np.array_equal(env.target, tgt, equal_nan=True)


def test_physics_prior_beats_doing_nothing():
    env = FluteEnv(progress=0.8)
    seeds = range(20)
    prior = [rollout(env, PhysicsPriorPolicy(), seed=s) for s in seeds]
    zero = [rollout(env, ZeroPolicy(), seed=s) for s in seeds]
    assert np.mean([r["mean_reward"] for r in prior]) > np.mean([r["mean_reward"] for r in zero]) + 0.3
    assert np.nanmean([r["sounding_rate"] for r in prior]) > 0.7


def test_physics_prior_is_accurate_on_nominal_rig():
    env = FluteEnv(level=0, randomize=False)
    runs = [rollout(env, PhysicsPriorPolicy(), seed=s) for s in range(10)]
    assert np.nanmean([r["mean_abs_cents"] for r in runs]) < 30.0
    assert np.nanmean([r["sounding_rate"] for r in runs]) > 0.9
