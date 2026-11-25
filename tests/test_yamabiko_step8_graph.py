"""step8 as one graph (PyTorch, for ONNX) gives what the NumPy runtime gives, tick by tick."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from flute_rl.yamabiko.profiles import load_profiles  # noqa: E402

pytestmark = pytest.mark.skipif("step8" not in load_profiles(), reason="no models/step8")


def test_the_graph_follows_the_numpy_runtime():
    from flute_rl.yamabiko.step8 import HOMING_STEPS, WARM_STEPS, Step8
    from flute_rl.yamabiko.step8_graph import Step8Cell
    p = load_profiles()["step8"]
    s8, cell = Step8.from_profile(p), Step8Cell.from_profile(p).eval()
    state = cell.initial_state()
    t = np.arange(16000) / 16000
    x = .3 * np.sin(2 * np.pi * (1500 + 300 * (t > .5)) * t)
    frames = [x[e - 320:e] for e in range(320, len(x) + 1, 160)]
    one = lambda v: torch.tensor([float(v)])

    def graph(frame, button, heard):
        nonlocal state
        hc, hv, use = (heard[0], float(heard[1]), 1.0) if heard else (0.0, 0.0, 0.0)
        f = torch.tensor(np.zeros(320) if frame is None else frame, dtype=torch.float32)
        with torch.no_grad():
            out, *state = cell(f, one(button), one(0), one(hc), one(hv), one(use), *state)
        return out.numpy()

    for i, f in enumerate(frames):
        a, b = s8.tick(f, button=int(i == 0)), graph(f, int(i == 0), None)
        assert a["mode"] == "listen" and int(b[0]) == 1
    rng = np.random.default_rng(0)
    for k in range(WARM_STEPS + HOMING_STEPS + 80):
        heard = (float(rng.uniform(100, 900)), True) if k > WARM_STEPS + HOMING_STEPS else None
        a, b = s8.tick(None, button=int(k == 0), heard=heard), graph(None, int(k == 0), heard)
        assert a["mode"] == ("idle", "listen", "play")[int(b[0])]
        assert a["pwm"] == pytest.approx(float(b[1]), abs=1e-4) and bool(a["valve"]) == bool(b[2] > .5)
