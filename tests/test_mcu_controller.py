"""The C controller (mcu/controller.c) must choose the same actions as the numpy pipeline.

A noise-free rig plays three takes with FeedbackResidualPolicy(FeedbackPolicy(), int8 net).
Every step's inputs (target look-ahead, previous take's error, delayed pitch error,
servo read-back) and the identified rig at each take start are written to a file;
the C controller replays them and its actions are compared with numpy's.
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
BIG = 1e30  # stands for NaN in the text file

try:
    import ziglang  # noqa: F401
    HAVE_CC = True
except ImportError:
    HAVE_CC = False


def fmt(v):
    return f"{BIG if not np.isfinite(v) else v:.17g}"


@pytest.mark.skipif(not HAVE_CC, reason="needs a C compiler (pip install ziglang)")
@pytest.mark.parametrize("history,use_net", [(0, False), (0, True), (5, True)])
def test_c_controller_matches_numpy(tmp_path, history, use_net):
    rng = np.random.default_rng(3)
    in_dim = FeedbackResidualPolicy.feature_dim(30, history)
    header = ""
    net = None
    if use_net:
        net = MLP(in_dim, 3, hidden=16, rng=rng)
        net.set_flat(net.get_flat() + rng.normal(0, 0.05, net.n_params))
        src = tmp_path / "net.npz"
        net.save(src, scale=0.3, horizon=30, ilc_gain=0.5, feedback=True, fb_gain=0.05, history=history)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "quantize_policy.py"), str(src), "--out",
                        str(tmp_path / "ctrlnet")], check=True, capture_output=True)
        net, _ = MLP.load(tmp_path / "ctrlnet.npz")  # numpy plays the same dequantised weights
        header = '#include "ctrlnet.h"\n'

    p = dataclasses.replace(FluteParams.sample(np.random.default_rng(7)), pitch_noise=0.0, dropout=0.0)
    target = make_target(np.random.default_rng(8), 3)
    env = FluteEnv(randomize=False, params=p, takes=3, feedback=True)
    obs, _ = env.reset(seed=11, options={"target": target})
    base = FeedbackPolicy()
    pol = FeedbackResidualPolicy(base, net, history=history) if use_net else base
    pol.reset(env)
    lines, acts, takes = [], [], []
    done = False
    while not done:
        t, take = env.t, env.take
        tw = np.full(50, np.nan)
        seg = env.target[t:t + 50]
        tw[:len(seg)] = seg
        pw = np.full(50, np.nan)
        seg = env.prev_err[t:t + 50]
        pw[:len(seg)] = seg
        fb = env._fb_seen[0]
        rb, comp = env.state.theta_readback, env.angle_comp
        a = np.asarray(pol.act(obs), dtype=float)
        if t == 0:  # identification happened inside act(): record what the take was played with
            takes.append((take, base.z.copy(), int(np.isfinite(base.fit_rms))))
        lines.append(" ".join([str(take), str(t)] + [fmt(v) for v in tw] + [fmt(v) for v in pw] +
                              [fmt(fb), fmt(rb), fmt(comp), fmt(p.angle_range_deg)]))
        acts.append(a)
        obs, _, done, _, _ = env.step(a)

    inp = tmp_path / "inputs.txt"
    zs = "\n".join(f"{k} {int(ok)} " + " ".join(f"{v:.17g}" for v in z) for k, z, ok in takes)
    inp.write_text(f"{len(takes)} {len(lines)}\n{zs}\n" + "\n".join(lines) + "\n")
    netdecl = "static const mlp_int8_t NET = MLP_INT8_FROM(CTRLNET);\n" if use_net else ""
    netptr = "&NET" if use_net else "NULL"
    main = f"""
#include <stdio.h>
#include <math.h>
#include "controller.h"
{header}{netdecl}
static ctrl_t C;
static float nanify(double v) {{ return v > 1e29 ? NAN : (float)v; }}
int main(int argc, char **argv) {{
  FILE *f = fopen(argv[1], "r");
  int nt, ns; fscanf(f, "%d %d", &nt, &ns);
  int tk[8], ok[8]; double z[8][4];
  for (int k = 0; k < nt; ++k) fscanf(f, "%d %d %lf %lf %lf %lf", &tk[k], &ok[k], &z[k][0], &z[k][1], &z[k][2], &z[k][3]);
  ctrl_init(&C, {netptr}, {history}, 3);
  float tw[50], pw[50];
  for (int s = 0; s < ns; ++s) {{
    int take, t; double v, fb, rb, comp, rng;
    fscanf(f, "%d %d", &take, &t);
    for (int j = 0; j < 50; ++j) {{ fscanf(f, "%lf", &v); tw[j] = nanify(v); }}
    for (int j = 0; j < 50; ++j) {{ fscanf(f, "%lf", &v); pw[j] = nanify(v); }}
    fscanf(f, "%lf %lf %lf %lf", &fb, &rb, &comp, &rng);
    if (t == 0) ctrl_begin_take(&C, take, take > 0 ? z[take] : NULL, ok[take]);
    ctrl_obs_t o = {{ tw, pw, nanify(fb), (float)rb, (float)comp, (float)rng }};
    float a0, a1; ctrl_step(&C, &o, &a0, &a1);
    printf("%.9g %.9g\\n", a0, a1);
  }}
  return 0;
}}
"""
    (tmp_path / "main.c").write_text(main)
    exe = tmp_path / ("ctrl.exe" if sys.platform == "win32" else "ctrl")
    subprocess.run([sys.executable, "-m", "ziglang", "cc", "-O2", "-I", str(ROOT / "mcu"), "-I", str(tmp_path),
                    str(tmp_path / "main.c"), str(ROOT / "mcu" / "controller.c"), str(ROOT / "mcu" / "policy_mlp.c"),
                    "-o", str(exe), "-lm"], check=True)
    out = subprocess.run([str(exe), str(inp)], capture_output=True, text=True, check=True).stdout.split()
    got = np.array([float(v) for v in out]).reshape(-1, 2)
    ref = np.array(acts)
    # float32 inputs/outputs and float network arithmetic in C: agreement to ~1e-4
    np.testing.assert_allclose(got, ref, atol=5e-4)
