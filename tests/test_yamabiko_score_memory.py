"""step6's score memory: the rule teacher, the write / read in torch, and the NumPy runtime agreeing with it."""
import numpy as np
import pytest
import torch

from flute_rl.yamabiko.score_memory import (MemoryConfig, MemoryRuntime, export, position_code, teacher,
                                            torch_model)


def test_position_code_is_a_bump_around_the_same_instant():
    cfg = MemoryConfig()
    from flute_rl.yamabiko.score_memory import frequencies
    k = position_code(np.arange(400), frequencies(cfg))
    sim = k @ k[200]
    assert sim[200] == pytest.approx(1.0, abs=1e-5)
    assert abs(sim[230]) < .5 and abs(sim[0]) < .3


def test_teacher_moves_the_whistle_into_the_range_and_holds_rests():
    cfg = MemoryConfig(low=0.0, high=1200.0)
    cents = np.r_[np.full(10, np.nan), np.full(50, 2000.0), np.full(20, np.nan), np.full(50, 2400.0)]
    t, v = teacher(cents, cfg)
    assert t[0] == pytest.approx(t[10])                  # before the first note: the first note
    assert t[10] == pytest.approx(400.0, abs=1) and t[-1] == pytest.approx(800.0, abs=1)
    assert not v[65] and t[65] == pytest.approx(t[59])   # a rest holds the previous pitch
    assert teacher(np.full(20, np.nan), cfg) is None


def test_numpy_runtime_matches_torch():
    torch.manual_seed(0)
    cfg = MemoryConfig(ear_dim=8, value=6, keys=16, global_dim=4, hidden=12)
    model = torch_model(cfg).eval()
    codes = torch.randn(1, 50, 8)
    with torch.no_grad():
        W, G = model.write(codes, torch.ones(1, 50))
        cents, vlog = model.read(W, G, torch.arange(50))
    rt = MemoryRuntime(export(model))
    for c in codes[0].numpy():
        rt.listen(c)
    for i in (0, 17, 49):
        got, p = rt.recall(i)
        assert got == pytest.approx(float(cents[0, i]), abs=1e-3)
        assert p == pytest.approx(float(torch.sigmoid(vlog[0, i])), abs=1e-4)


def test_numpy_runtime_matches_torch_with_extremes():
    torch.manual_seed(1)
    cfg = MemoryConfig(ear_dim=8, value=6, keys=16, global_dim=4, hidden=12, extremes=True)
    model = torch_model(cfg).eval()
    codes = torch.randn(1, 60, 8) * 2
    with torch.no_grad():
        W, G = model.write(codes, torch.ones(1, 60))
        cents, _ = model.read(W, G, torch.arange(60))
        p = model.gen(codes)[0, :, 4]
    assert float(G[0, 4]) <= float(p.max()) + 1e-4 and float(G[0, 5]) >= float(p.min()) - 1e-4
    assert float(G[0, 4]) > float(G[0, 5])
    rt = MemoryRuntime(export(model))
    for c in codes[0].numpy():
        rt.listen(c)
    for i in (0, 31, 59):
        assert rt.recall(i)[0] == pytest.approx(float(cents[0, i]), abs=1e-3)


def test_numpy_runtime_matches_torch_with_histogram_and_key_head():
    torch.manual_seed(2)
    cfg = MemoryConfig(ear_dim=8, value=6, keys=16, global_dim=4, hidden=12, extremes=True, hist_bins=5, key_head=True)
    model = torch_model(cfg).eval()
    codes = torch.randn(1, 70, 8) * 2
    with torch.no_grad():
        W, G = model.write(codes, torch.ones(1, 70))
        cents, vlog = model.read(W, G, torch.arange(70))
        shift = float(model.shift(G)[0])
    assert G.shape[1] == 4 + 2 + 5 and float(G[0, 6:].sum()) == pytest.approx(1.0, abs=1e-5)   # a histogram
    assert float(cents.min()) >= cfg.low - 1e-3 and float(cents.max()) <= cfg.high + 1e-3     # clamped
    rt = MemoryRuntime(export(model))
    for c in codes[0].numpy():
        rt.listen(c)
    for i in (0, 33, 69):
        got, p = rt.recall(i)
        assert got == pytest.approx(float(cents[0, i]), abs=1e-2)
        assert p == pytest.approx(float(torch.sigmoid(vlog[0, i])), abs=1e-4)
    assert np.isfinite(shift)


def test_numpy_runtime_matches_torch_with_fixed_histogram():
    torch.manual_seed(3)
    cfg = MemoryConfig(ear_dim=8, value=6, keys=16, global_dim=4, hidden=12, extremes=True, hist_bins=9,
                       key_head=True, hist_fixed=True)
    model = torch_model(cfg).eval()
    codes = torch.randn(1, 50, 8) * 3
    with torch.no_grad():
        W, G = model.write(codes, torch.ones(1, 50))
        cents, _ = model.read(W, G, torch.arange(50))
    assert G.shape[1] == 4 + 2 + 9 and float(G[0, 6:].sum()) == pytest.approx(1.0, abs=1e-5)
    rt = MemoryRuntime(export(model))
    for c in codes[0].numpy():
        rt.listen(c)
    for i in (0, 25, 49):
        assert rt.recall(i)[0] == pytest.approx(float(cents[0, i]), abs=1e-2)


def test_score_from_recording_runs_the_ear_and_the_memory():
    from flute_rl.yamabiko.ear_nn import Ear, EarConfig, export as export_ear, torch_model as ear_model
    from flute_rl.yamabiko.score_memory import score_from_recording
    torch.manual_seed(0)
    ear = Ear(export_ear(ear_model(EarConfig.wide(level=True))))
    model = torch_model(MemoryConfig(keys=32, hidden=32))
    with torch.no_grad():
        model.dec[-1].bias.copy_(torch.tensor([0.0, 5.0]))          # always "sounding", in the middle
        model.dec[-1].weight.zero_()
    t = np.arange(16000) / 16000
    pcm = (np.sin(2 * np.pi * 1500 * t) * 8000).astype(np.int16)
    target, voice, heard, report = score_from_recording(pcm, ear, export(model), lead_steps=60, tail_steps=60)
    steps = (16000 - 320) // 160 + 1
    assert len(heard) == steps and report["listened_steps"] == steps
    assert len(target) == len(voice) == steps + 120 and not voice[:60].any() and voice[60:60 + steps].all()
    assert np.allclose(target[60:60 + steps], 595.0, atol=1e-3)
