"""The MCU's C inference must give the same actions as the (dequantised) numpy network."""
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from flute_rl import MLP

ROOT = pathlib.Path(__file__).resolve().parents[1]

try:
    import ziglang  # noqa: F401  (provides a C compiler: python -m ziglang cc)
    HAVE_CC = True
except ImportError:
    HAVE_CC = False


@pytest.mark.skipif(not HAVE_CC, reason="needs a C compiler (pip install ziglang)")
def test_c_inference_matches_numpy(tmp_path):
    rng = np.random.default_rng(0)
    net = MLP(20, 3, hidden=12, rng=rng)
    net.set_flat(rng.normal(0, 0.3, net.n_params))
    src = tmp_path / "net.npz"
    net.save(src, scale=0.3, horizon=30, ilc_gain=0.5, feedback=True, fb_gain=0.05, history=0)
    out = tmp_path / "testnet"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "quantize_policy.py"), str(src), "--out", str(out)], check=True)
    q, _ = MLP.load(out.with_suffix(".npz"))

    xs = rng.normal(0, 1, (5, 20))
    ref = np.array([q(x) for x in xs])
    rows = ",\n".join("{" + ", ".join(f"{v:.9g}f" for v in x) + "}" for x in xs)
    main = f"""
#include <stdio.h>
#include "policy_mlp.h"
#include "testnet.h"
static const float XS[5][20] = {{ {rows} }};
int main(void) {{
  static const mlp_int8_t m = MLP_INT8_FROM(TESTNET);
  float hid[TESTNET_HIDDEN], y[TESTNET_OUT];
  for (int k = 0; k < 5; ++k) {{
    mlp_int8_forward(&m, XS[k], hid, y);
    printf("%.7f %.7f %.7f\\n", y[0], y[1], y[2]);
  }}
  return 0;
}}
"""
    (tmp_path / "main.c").write_text(main)
    exe = tmp_path / ("test.exe" if sys.platform == "win32" else "test")
    subprocess.run([sys.executable, "-m", "ziglang", "cc", "-O2", "-I", str(ROOT / "mcu"), "-I", str(tmp_path),
                    str(tmp_path / "main.c"), str(ROOT / "mcu" / "policy_mlp.c"), "-o", str(exe), "-lm"], check=True)
    got = np.array([[float(v) for v in line.split()]
                    for line in subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines()])
    np.testing.assert_allclose(got, ref, atol=2e-5)
