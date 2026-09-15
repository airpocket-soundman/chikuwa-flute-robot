import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flute_rl.pitchnet import SELF_EAR, SOURCE_EAR, PitchNet, decode, frames_at_steps, make_clip, soft_targets  # noqa: E402


@pytest.mark.parametrize("kind,ear", [("self", SELF_EAR), ("recorder", SOURCE_EAR), ("hum", SOURCE_EAR)])
def test_clip_frames_line_up_with_truth(kind, ear):
    x, y = make_clip(np.random.default_rng(0), kind)
    assert x.shape == (len(y), ear.frame) and x.dtype == np.float32
    assert np.isfinite(y).any() and (~np.isfinite(y)).any()


def test_frames_are_centred_on_steps():
    y = np.arange(8000, dtype=float)  # 1 s at 8 kHz
    f = frames_at_steps(y, SELF_EAR, 100)
    assert f.shape == (100, SELF_EAR.frame)
    assert f[50, SELF_EAR.frame // 2] == pytest.approx(50.5 * 80)


def test_decode_recovers_soft_target_pitch():
    cents = torch.tensor([500.0, float("nan"), 1234.0])
    t = soft_targets(cents, SOURCE_EAR)
    logits = torch.logit(t.clamp(1e-4, 1 - 1e-4))
    out = decode(logits, SOURCE_EAR)
    assert torch.isnan(out[1])
    assert abs(out[0].item() - 500.0) < 3.0 and abs(out[2].item() - 1234.0) < 3.0


def test_network_output_shape():
    net = PitchNet(SELF_EAR, width=8)
    out = net(torch.randn(4, SELF_EAR.frame))
    assert out.shape == (4, SELF_EAR.n_bins)
