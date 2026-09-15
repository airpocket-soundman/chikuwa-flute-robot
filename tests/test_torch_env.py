import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flute_rl import MLP, FluteEnv, FluteParams  # noqa: E402
from flute_rl.policy import GRU  # noqa: E402
from flute_rl.adapt import fit_rig, predict_cents  # noqa: E402
from flute_rl.feedback import FeedbackPolicy, FeedbackResidualPolicy  # noqa: E402
from flute_rl.targets import make_target  # noqa: E402
from flute_rl.torch_env import BatchAgent, BatchFluteEnv, BatchGRU, BatchMLP, batch_fit, rig_from_seed  # noqa: E402


def quiet_rigs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [dataclasses.replace(FluteParams.sample(rng), pitch_noise=0.0, dropout=0.0) for _ in range(n)]


def numpy_actions(p, seed, target, net, history):
    env = FluteEnv(randomize=False, params=p, takes=2, feedback=True)
    obs, _ = env.reset(seed=seed, options={"target": target})
    pol = FeedbackResidualPolicy(FeedbackPolicy(), net, history=history) if net is not None else FeedbackPolicy()
    pol.reset(env)
    acts, done = [], False
    while not done:
        a = pol.act(obs)
        acts.append(np.asarray(a, dtype=float))
        obs, _, done, _, _ = env.step(a)
    return np.array(acts)


@pytest.mark.parametrize("arch,history", [(None, 0), ("mlp", 0), ("mlp", 5), ("gru", 0)])
def test_actions_match_numpy_pipeline(arch, history):
    params = quiet_rigs(5)
    rng = np.random.default_rng(1)
    targets = [make_target(rng, 3) for _ in params]
    seeds = list(range(100, 105))
    in_dim = FeedbackResidualPolicy.feature_dim(30, history)
    thetas, nets = [], []
    for i in range(len(params)):
        if arch is None:
            nets.append(None)
            continue
        cls = GRU if arch == "gru" else MLP
        net = cls(in_dim, 3, hidden=8, rng=np.random.default_rng(i))
        net.set_flat(net.get_flat() + np.random.default_rng(10 + i).normal(0, 0.05, net.n_params))
        nets.append(net)
        thetas.append(net.get_flat())

    rigs = [rig_from_seed(s, params=p) for s, p in zip(seeds, params)]
    env = BatchFluteEnv(rigs, targets, takes=2, dtype=torch.float64)
    bnet = None
    if arch is not None:
        th = torch.tensor(np.array(thetas))
        bnet = (BatchGRU if arch == "gru" else BatchMLP)(th, in_dim, 3, 8)
    agent = BatchAgent(env, net=bnet, history=history)
    acts = []
    done = False
    while not done:
        a = agent.act()
        acts.append(a.numpy().copy())
        done = env.step(a)
    acts = np.array(acts)  # (2*T, B, 2)

    for b in range(len(params)):
        ref = numpy_actions(params[b], seeds[b], targets[b], nets[b], history)
        L = len(targets[b])
        got = np.concatenate([acts[:L, b], acts[env.T:env.T + L, b]])
        np.testing.assert_allclose(got, ref, atol=1e-6)


def test_batch_fit_matches_fit_rig():
    rng = np.random.default_rng(3)
    nominal = FluteParams()
    pwms, meas, zs = [], [], []
    for _ in range(4):
        z = np.array([rng.normal(0, 0.1), rng.normal(0, 0.1), rng.normal(0, 0.004), rng.normal(0, 0.1)])
        pwm = np.concatenate([np.full(20, 0.8), np.zeros(30), np.full(15, -0.7), np.zeros(30), np.full(25, 0.6), np.zeros(40)])
        m = predict_cents(z, pwm, nominal) + rng.normal(0, 3.0, len(pwm))
        m[:5] = np.nan
        pwms.append(pwm)
        meas.append(m)
        zs.append(fit_rig([(pwm, m)], nominal)[0])
    zb, rms = batch_fit(torch.tensor(np.array(pwms)), torch.tensor(np.array(meas)))
    np.testing.assert_allclose(zb.numpy(), np.array(zs), atol=1e-6)
    assert torch.isfinite(rms).all()
