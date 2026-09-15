"""Software in the loop: scripts/hil.py talking to mcu/hil_host.c must play like the numpy pipeline.

Same noise-free rig and target as the numpy FeedbackResidualPolicy run; the
controller runs in C behind the frame protocol and this side identifies the
rig between takes, as the UNO Q's Linux side will.
"""
import dataclasses
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from flute_rl import MLP, FluteEnv, FluteParams
from flute_rl.feedback import FeedbackPolicy, FeedbackResidualPolicy
from flute_rl.targets import make_target

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import ziglang  # noqa: F401
    HAVE_CC = True
except ImportError:
    HAVE_CC = False


@pytest.mark.skipif(not HAVE_CC, reason="needs a C compiler (pip install ziglang)")
def test_hil_matches_numpy(tmp_path):
    import hil

    history = 5
    rng = np.random.default_rng(4)
    net = MLP(FeedbackResidualPolicy.feature_dim(30, history), 3, hidden=16, rng=rng)
    net.set_flat(net.get_flat() + rng.normal(0, 0.05, net.n_params))
    src = tmp_path / "net.npz"
    net.save(src, scale=0.3, horizon=30, ilc_gain=0.5, feedback=True, fb_gain=0.05, history=history)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "quantize_policy.py"), str(src), "--out",
                    str(tmp_path / "hilnet")], check=True, capture_output=True)
    qnet, _ = MLP.load(tmp_path / "hilnet.npz")
    exe = tmp_path / ("hil_host.exe" if sys.platform == "win32" else "hil_host")
    subprocess.run([sys.executable, "-m", "ziglang", "cc", "-O2", "-I", str(ROOT / "mcu"), "-I", str(tmp_path),
                    '-DHIL_NET_HEADER="hilnet.h"', "-DHIL_NET=HILNET",
                    str(ROOT / "mcu" / "hil_host.c"), str(ROOT / "mcu" / "hil_protocol.c"),
                    str(ROOT / "mcu" / "controller.c"), str(ROOT / "mcu" / "policy_mlp.c"), "-o", str(exe), "-lm"],
                   check=True)

    p = dataclasses.replace(FluteParams.sample(np.random.default_rng(7)), pitch_noise=0.0, dropout=0.0)
    target = make_target(np.random.default_rng(8), 3)

    env = FluteEnv(randomize=False, params=p, takes=3, feedback=True)
    obs, _ = env.reset(seed=11, options={"target": target})
    pol = FeedbackResidualPolicy(FeedbackPolicy(), qnet, history=history)
    pol.reset(env)
    ref, done = [], False
    while not done:
        a = np.asarray(pol.act(obs), dtype=float)
        ref.append(a)
        obs, _, done, _, _ = env.step(a)

    env2 = FluteEnv(randomize=False, params=p, takes=3, feedback=True)
    link = hil.ProcessLink(str(exe), history, 3)
    try:
        got = hil.play_episode(env2, link, seed=11, target=target)["actions"]
    finally:
        link.close()
    np.testing.assert_allclose(got, np.array(ref), atol=2e-3)
