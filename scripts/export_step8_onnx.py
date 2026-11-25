"""PC: write step8 as ONE ONNX graph (flute_rl/yamabiko/step8_graph.py) and check it against PyTorch.

    python scripts/export_step8_onnx.py --step step8 --out models/step8/step8.onnx

One call is one tick (10 ms).  Inputs: the last 320 microphone samples, the button, the calibration shift,
the simulator's pitch (heard_cents, heard_valid, use_heard: 0 on the rig) and the state; outputs: (mode, pwm,
valve, fan, end, heard_cents, heard_voiced) and the next state.  Open it in Netron to see the whole robot as
one picture.  Checks a few ticks of listening and playing with onnxruntime against the PyTorch cell.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flute_rl.yamabiko.profiles import load_profiles  # noqa: E402
from flute_rl.yamabiko.step8_graph import OUTPUTS, STATE, Step8Cell  # noqa: E402

INPUTS = ("frame", "button", "shift", "heard_cents", "heard_valid", "use_heard")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="step8")
    ap.add_argument("--out", default="models/step8/step8.onnx")
    args = ap.parse_args()
    cell = Step8Cell.from_profile(load_profiles()[args.step]).eval()
    state = cell.initial_state()
    one = lambda v: torch.tensor([float(v)])
    rng = np.random.default_rng(0)
    frame = torch.tensor(np.sin(2 * np.pi * 1500 * np.arange(320) / 16000) * .3, dtype=torch.float32)
    example = (frame, one(1), one(0), one(0), one(0), one(0), *state)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(cell, example, str(out), input_names=list(INPUTS) + [f"in_{s}" for s in STATE],
                      output_names=["outputs"] + [f"out_{s}" for s in STATE], opset_version=17, dynamo=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    st_t, st_o = [s.clone() for s in state], [s.numpy() for s in state]
    worst = 0.0
    plan = [(1, None)] + [(0, None)] * 120 + [(1, None)] + [(0, (float(rng.uniform(0, 900)), 1.0))] * 460
    for b, heard in plan:                         # listen to a tone, then play with a made-up pitch heard
        f = (frame * float(rng.uniform(.5, 1))) if heard is None else torch.zeros(320)
        hc, hv, use = (heard[0], heard[1], 1.0) if heard else (0.0, 0.0, 0.0)
        ins = (f, one(b), one(0), one(hc), one(hv), one(use))
        with torch.no_grad():
            o_t, *st_t = cell(*ins, *st_t)
        res = sess.run(None, {**{k: v.numpy() for k, v in zip(INPUTS, ins)},
                              **{f"in_{s}": v for s, v in zip(STATE, st_o)}})
        o_o, st_o = res[0], res[1:]
        worst = max(worst, float(np.abs(o_o - o_t.numpy()).max()))
    print(f"onnxruntime against PyTorch over {len(plan)} ticks: largest difference {worst:.2e} "
          f"(outputs {', '.join(OUTPUTS)}); last mode {int(o_o[0])}, pwm {o_o[1]:.3f}")
    assert worst < 1e-3, worst


if __name__ == "__main__":
    main()
