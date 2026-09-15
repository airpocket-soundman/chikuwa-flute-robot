import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flute_rl import MLP  # noqa: E402
from flute_rl.policy import GRU  # noqa: E402
from flute_rl.targets import make_target  # noqa: E402
from flute_rl.torch_audio import FRAME, HOP, BatchSelfAudio, pad_inputs  # noqa: E402
from flute_rl.torch_env import BatchAgent, BatchFluteEnv, rig_from_seed, run_episodes  # noqa: E402

EAR_PATH = pathlib.Path(__file__).resolve().parents[1] / "runs" / "pitchnet_self.pt"


@pytest.mark.parametrize("cls,arch", [(MLP, "mlp"), (GRU, "gru")])
def test_padded_inputs_do_not_change_the_output(cls, arch):
    net = cls(10, 3, hidden=6, rng=np.random.default_rng(0))
    net.set_flat(net.get_flat() + np.random.default_rng(1).normal(0, 0.2, net.n_params))
    big = cls(14, 3, hidden=6)
    big.set_flat(pad_inputs(net.get_flat(), arch, 10, 14, 3, 6))
    x = np.random.default_rng(2).normal(size=10)
    for _ in range(3):  # several steps so the GRU memory is exercised
        a = net(x)
        b = big(np.concatenate([x, np.random.default_rng(3).normal(size=4)]))
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_audio_frames_follow_the_delay():
    au = BatchSelfAudio(2, "cpu", torch.Generator().manual_seed(0))
    for k in range(6):
        au.push(torch.full((2,), 1000.0), torch.tensor([k >= 3, k >= 3]))
    now = au.frames(torch.tensor([0, 0]))
    late = au.frames(torch.tensor([3, 3]))
    assert now.shape == (2, FRAME)
    # three steps ago the tone had not started yet: much quieter
    assert late[:, -HOP:].pow(2).mean() < 0.2 * now[:, -HOP:].pow(2).mean()


@pytest.mark.skipif(not EAR_PATH.exists(), reason="needs a trained self ear (scripts/train_pitch_net.py --ear self)")
def test_hearing_environment_runs():
    from flute_rl.torch_audio import Ear

    ear = Ear(str(EAR_PATH), "cpu", fit_clips=6)
    rng = np.random.default_rng(0)
    targets = [make_target(rng, 1) for _ in range(3)]
    rigs = [rig_from_seed(s, harsh=1.0) for s in (1, 2, 3)]
    env = BatchFluteEnv(rigs, targets, takes=2, device="cpu", hearing=ear)
    m = run_episodes(env, BatchAgent(env))
    assert m.shape == (3, 2, 5)
    assert torch.isfinite(m[:, :, 1]).all()  # reward per step
    assert env.ear_feat.shape == (3, ear.dim)
