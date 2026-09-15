import numpy as np

from flute_rl import MLP, FluteEnv, PhysicsPriorPolicy, ResidualPolicy, rollout


def test_fresh_residual_equals_base():
    env = FluteEnv(level=3)
    net = MLP(env.obs_dim + 2, 2, hidden=8)
    a = rollout(env, ResidualPolicy(PhysicsPriorPolicy(), net), seed=4)
    b = rollout(env, PhysicsPriorPolicy(), seed=4)
    assert np.allclose(a["actions"], b["actions"])


def test_flat_params_round_trip(tmp_path):
    net = MLP(5, 2, hidden=4, rng=np.random.default_rng(0))
    theta = np.random.default_rng(1).standard_normal(net.n_params)
    net.set_flat(theta)
    assert np.allclose(net.get_flat(), theta)
    net.save(tmp_path / "p.npz", scale=0.3)
    loaded, extra = MLP.load(tmp_path / "p.npz")
    x = np.ones(5)
    assert np.allclose(loaded(x), net(x)) and float(extra["scale"]) == 0.3
