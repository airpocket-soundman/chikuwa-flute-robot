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
