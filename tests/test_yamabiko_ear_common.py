"""step5's common ear: a wide ear network (flute and whistle), synthetic whistles, and the ear on every path."""
import numpy as np
import pytest
import torch

from flute_rl.yamabiko.ear_nn import BINS, LOW_CENTS, WIDE_BINS, Ear, EarConfig, export, torch_model
from flute_rl.yamabiko.ear_synth import FRAME, frames, phrase
from flute_rl.yamabiko.hw import heard_pitch
from flute_rl.yamabiko.remote import Remote, RemoteError
from test_yamabiko_server import server  # noqa: F401  (the fixture)


def test_wide_ear_exports_its_range_and_matches_torch():
    torch.manual_seed(0)
    model = torch_model(EarConfig.wide()).eval()
    arrays = export(model)
    ear = Ear(arrays)
    assert (ear.bins, ear.low_cents) == (WIDE_BINS, LOW_CENTS)
    x = np.random.default_rng(0).normal(0, .1, FRAME).astype(np.float32)
    with torch.no_grad():
        logits, voiced = model(torch.tensor(x)[None])
    got, v = ear.forward(x)
    assert got.shape == (WIDE_BINS,) and np.allclose(logits[0].numpy(), got, atol=1e-4)
    assert abs(float(voiced) - v) < 1e-4


def test_ear_files_before_step5_still_load():
    arrays = export(torch_model(EarConfig()))
    for key in ("ear.bins", "ear.low_cents"):
        del arrays[key]
    ear = Ear(arrays)
    assert (ear.bins, ear.low_cents) == (BINS, LOW_CENTS)


def test_track_hears_every_10_ms():
    ear = Ear(export(torch_model(EarConfig.wide())))
    out = ear.track(np.zeros(16000, np.int16))
    assert len(out) == (16000 - FRAME) // 160 + 1


def test_synthetic_whistle_labels_agree_with_yin():
    x, cents, state = phrase(np.random.default_rng(3), 3.0)
    assert np.nanmin(cents) > 1200 * np.log2(480 / 440) and np.nanmax(cents) < 1200 * np.log2(3900 / 440)
    ends = np.arange(FRAME, len(x) + 1, 160)
    voiced = np.flatnonzero(state == 1)
    assert len(voiced) > 50
    errors = [abs(heard_pitch(x[ends[i] - FRAME:ends[i]].astype(np.float64))[0] - cents[i]) for i in voiced[::5]]
    assert np.nanmedian(errors) < 10


def test_frames_are_int16_with_labels():
    x, c, s = frames(np.random.default_rng(0), 1000)
    assert x.shape == (1000, FRAME) and x.dtype == np.int16
    assert np.isfinite(c[s == 1]).all() and np.isnan(c[s != 1]).all()


def test_tune_and_calibrate_hear_with_the_ear_network(server, tmp_path):   # noqa: F811
    srv, link = server
    np.savez(tmp_path / "ear.npz", **export(torch_model(EarConfig.wide())))
    with Remote(port=srv.port, forward=False) as rig:
        with pytest.raises(RemoteError, match="unknown ear"):
            rig.tune(ear="cat", warmup_s=.05)
        result = rig.calibrate(out=str(tmp_path / "cal.npz"), plan="yamabiko1_mid", ear="nn",
                               ear_runtime=str(tmp_path / "ear.npz"), warmup_s=.05, homing_seconds=.05)
    assert result["steps"] > 0 and (tmp_path / "cal.npz").exists()
