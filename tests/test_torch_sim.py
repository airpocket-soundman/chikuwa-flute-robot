import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flute_rl.sim import FluteParams, FluteSim  # noqa: E402
from flute_rl.torch_sim import BatchFluteSim, BatchParams  # noqa: E402


def quiet(p: FluteParams) -> FluteParams:
    return dataclasses.replace(p, pitch_noise=0.0, dropout=0.0)


def test_matches_numpy_sim_without_noise():
    rng = np.random.default_rng(0)
    params = [quiet(FluteParams.sample(rng)) for _ in range(12)]
    T = 250
    pwm = rng.uniform(-1, 1, (T, len(params))) * (rng.random((T, len(params))) < 0.4)
    ang = np.clip(np.cumsum(rng.normal(0, 0.05, (T, len(params))), axis=0), -1, 1)

    bs = BatchFluteSim(BatchParams(params, dtype=torch.float64))
    sims = [FluteSim(p, np.random.default_rng(1)) for p in params]
    for t in range(T):
        out = bs.step(torch.tensor(pwm[t]), torch.tensor(ang[t]))
        ref = [s.step(pwm[t, i], ang[t, i]) for i, s in enumerate(sims)]
        np.testing.assert_allclose(out["x"].numpy(), [r.x for r in ref], atol=1e-12)
        np.testing.assert_allclose(out["theta"].numpy(), [r.theta for r in ref], atol=1e-9)
        np.testing.assert_allclose(out["cents"].numpy(), [r.cents for r in ref], atol=1e-6)
        assert np.array_equal(out["sounding"].numpy(), [r.sounding for r in ref])
        assert np.array_equal(out["overblown"].numpy(), [r.overblown for r in ref])
        measured = out["measured"].numpy()
        assert np.array_equal(np.isfinite(measured), [r.sounding for r in ref])


def test_noise_statistics():
    p = FluteParams(pitch_noise=5.0, dropout=0.1)
    bs = BatchFluteSim(BatchParams([p] * 4000, dtype=torch.float64), torch.Generator().manual_seed(0))
    o = bs.observe()
    heard = torch.isfinite(o["measured"])
    assert o["sounding"].all()
    assert abs(heard.double().mean().item() - 0.9) < 0.02
    err = (o["measured"][heard] - o["cents"][heard])
    assert abs(err.std().item() - 5.0) < 0.3
