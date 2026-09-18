"""The PyTorch training session reproduces the numpy one (rig random effects switched off)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flute_rl.yamabiko import GRUPolicy, Rig, RigParams, Swap, draw_pieces, make_schedule, run_session  # noqa: E402
from flute_rl.yamabiko.torch_session import Player  # noqa: E402

FIRST_W = 4.0  # the first note of each song counts this much more
QUIET = dict(pitch_jitter=0.0, pitch_noise=0.0, dropout=0.0, octave_err=0.0)


def _quiet(params: RigParams) -> RigParams:
    for k, v in QUIET.items():
        setattr(params, k, np.full(params.n, v))
    return params


def _case(seed=3, P=3, R=4, K=3, hidden=8):
    rng = np.random.default_rng(seed)
    thetas = np.stack([GRUPolicy.init_params(rng, hidden) for _ in range(P)])
    thetas[:, -(hidden + GRUPolicy.FEATURES) * GRUPolicy.OUT - GRUPolicy.OUT:] = 0.3 * rng.standard_normal(
        ((hidden + GRUPolicy.FEATURES) * GRUPolicy.OUT + GRUPolicy.OUT))  # nonzero read-out: the belief moves
    base = _quiet(RigParams.sample(rng, R))
    sched = make_schedule(draw_pieces(rng, R, K))
    mask = np.array([True, False, True, False])
    swap = Swap(2, np.tile(mask, P), _quiet(RigParams.sample(rng, R)).tile(P))
    ref = run_session(Rig(base.tile(P), np.random.default_rng(0)), sched.tile(P),
                      GRUPolicy(thetas, hidden, carry=True), swap=swap, first_weight=FIRST_W)["reward"]
    return thetas, base, sched, swap, ref, (P, R, K, hidden)


def test_torch_session_matches_numpy_cpu():
    thetas, base, sched, swap, ref, (P, R, K, hidden) = _case()
    pl = Player(P, R, K, hidden, sched.T, device="cpu", dtype=torch.float64, graph=False)
    got = pl.play(thetas, base.tile(P), sched, swap, FIRST_W)
    assert np.abs(ref).max() > 0.05
    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_session_matches_numpy_cuda_graph():
    thetas, base, sched, swap, ref, (P, R, K, hidden) = _case()
    pl = Player(P, R, K, hidden, sched.T + 100, device="cuda", dtype=torch.float64)
    for _ in range(2):  # the player is reused across generations
        np.testing.assert_allclose(pl.play(thetas, base.tile(P), sched, swap, FIRST_W), ref, atol=1e-6)
    # float32 (used for training): rounding can flip a threshold now and then, so only close on average
    got = Player(P, R, K, hidden, sched.T, device="cuda", dtype=torch.float32).play(thetas, base.tile(P), sched, swap, FIRST_W)
    assert np.abs(got - ref).mean() < 1e-3
