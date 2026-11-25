"""step8 (8a): the buttons switch idle -> listen -> play, the look-ahead reads the memory, and it stops by itself."""
import copy

import numpy as np
import pytest

from flute_rl.yamabiko.profiles import load_profiles

pytestmark = pytest.mark.skipif("step8" not in load_profiles(), reason="no models/step8")


@pytest.fixture(scope="module")
def step8():
    from flute_rl.yamabiko.step8 import Step8
    return Step8.from_profile(load_profiles()["step8"])


def tone(seconds, hz=1500.0):
    t = np.arange(int(seconds * 16000)) / 16000
    x = .3 * np.sin(2 * np.pi * hz * t)
    return [x[e - 320:e] for e in range(320, len(x) + 1, 160)]


def test_buttons_switch_the_modes_and_it_stops_by_itself(step8):
    from flute_rl.yamabiko.step8 import HOMING_STEPS, WARM_STEPS
    s = copy.deepcopy(step8); s.reset()
    quiet = np.zeros(320)
    assert s.tick(quiet)["mode"] == "idle"
    frames = tone(.4) + [quiet] * 30 + tone(.4, 1800.0)
    for i, f in enumerate(frames):
        out = s.tick(f, button=int(i == 0))
        assert out["mode"] == "listen" and out["pwm"] == 0.0 and not out["valve"]
    assert s.memory.n == len(frames)
    modes, ts = [], []
    for k in range(WARM_STEPS + HOMING_STEPS + 600):
        out = s.tick(None, button=int(k == 0), heard=(0.0, False))
        modes.append(out["mode"])
        if out["mode"] != "play":
            break
        if k < WARM_STEPS:
            assert out["pwm"] == 0.0 and not out["valve"] and out["fan"] > .6
        elif k < WARM_STEPS + HOMING_STEPS:
            assert out["pwm"] == -1.0
        else:
            ts.append(out["t"])
    assert modes[0] == "play" and modes[-1] == "idle"            # it stopped by itself
    assert ts == list(range(len(ts))) and len(ts) < len(frames) + 60 + 4 * 60 + 60


def test_the_look_ahead_is_the_memory_read_back(step8):
    from flute_rl.yamabiko.step8 import LEAD_STEPS, Recalled
    s = copy.deepcopy(step8); s.reset()
    for i, f in enumerate(tone(.5)):
        s.tick(f, button=int(i == 0))
    view = Recalled(s.memory, s.memory.n, 12.0, "cents")
    for t in (0, LEAD_STEPS, LEAD_STEPS + 20):
        u = min(max(t - LEAD_STEPS, 0), s.memory.n - 1)
        assert view[t] == pytest.approx(s.memory.recall(u)[0] + 12.0)
    assert list(view[[LEAD_STEPS, LEAD_STEPS + 3]]) == pytest.approx([view[LEAD_STEPS], view[LEAD_STEPS + 3]])
