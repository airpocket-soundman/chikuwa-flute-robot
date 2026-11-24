"""Write a step's three networks (ear, actuator, valve) as ONNX graphs, for viewing in Netron.

No PyTorch needed: the graphs are built with the onnx package from the ``*_runtime.npz`` weights the UNO Q
runs, then checked with onnxruntime against the NumPy runtimes on random inputs.

    python scripts/export_step_models_onnx.py models/step4            # -> models/step4/onnx/*.onnx

* ear_*.onnx:      samples (1, 320) -> pitch_logits (1, 96), voiced_logit (1,)
* nn_*.onnx:       features (1, F), h (1, H) -> pwm (1, 1), h_next (1, H)      (one 10 ms step)
* valve_*.onnx:    features (1, F), h (1, H) -> open_prob (1, 1), h_next (1, H)

The recurrent networks are one GRUCell step: the hidden state goes out and comes back in on the next 10 ms.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.ear_nn import BINS, FRAME, Ear  # noqa: E402

OPSET = 17


def _init(name, array):
    return numpy_helper.from_array(np.asarray(array, np.float32), name)


def _model(nodes, inputs, outputs, inits, name, doc):
    graph = helper.make_graph(nodes, name, inputs, outputs, inits)
    graph.doc_string = doc
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="chikuwa-flute-robot")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def ear_graph(a, name):
    convs = sorted({k.split(".")[1] for k in a if k.startswith("ear.conv")}, key=lambda s: int(s[4:]))
    fcs = sorted({k.split(".")[1] for k in a if k.startswith("ear.fc")}, key=lambda s: int(s[2:]))
    inits = [numpy_helper.from_array(np.array([1], np.int64), "axis1"),
             _init("eps", np.array(1e-4)),
             numpy_helper.from_array(np.array([0], np.int64), "s0"),
             numpy_helper.from_array(np.array([BINS], np.int64), "s96"),
             numpy_helper.from_array(np.array([BINS + 1], np.int64), "s97")]
    nodes = [  # RMS normalisation, as EarNet.normalize
        helper.make_node("ReduceMean", ["samples"], ["mean"], keepdims=1, axes=[1], name="mean"),
        helper.make_node("Sub", ["samples", "mean"], ["centred"], name="remove_dc"),
        helper.make_node("Mul", ["centred", "centred"], ["sq"], name="square"),
        helper.make_node("ReduceMean", ["sq"], ["ms"], keepdims=1, axes=[1], name="mean_square"),
        helper.make_node("Sqrt", ["ms"], ["rms"], name="rms"),
        helper.make_node("Add", ["rms", "eps"], ["rms_eps"], name="rms_plus_eps"),
        helper.make_node("Div", ["centred", "rms_eps"], ["normalised"], name="normalise"),
        helper.make_node("Unsqueeze", ["normalised", "axis1"], ["x0"], name="to_1_channel"),
    ]
    x = "x0"
    for i, c in enumerate(convs):
        w = a[f"ear.{c}.w"]
        inits += [_init(f"{c}.weight", w), _init(f"{c}.bias", a[f"ear.{c}.b"])]
        nodes += [helper.make_node("Conv", [x, f"{c}.weight", f"{c}.bias"], [f"{c}_out"], name=f"conv{i + 1}",
                                   kernel_shape=[w.shape[2]], strides=[int(a[f"ear.{c}.stride"])]),
                  helper.make_node("Relu", [f"{c}_out"], [f"{c}_relu"], name=f"relu{i + 1}")]
        x = f"{c}_relu"
    nodes.append(helper.make_node("Flatten", [x], ["flat"], axis=1, name="flatten"))
    x = "flat"
    for i, f in enumerate(fcs):
        inits += [_init(f"{f}.weight", a[f"ear.{f}.w"]), _init(f"{f}.bias", a[f"ear.{f}.b"])]
        nodes.append(helper.make_node("Gemm", [x, f"{f}.weight", f"{f}.bias"], [f"{f}_out"], transB=1, name=f"fc{i + 1}"))
        x = f"{f}_out"
        if i < len(fcs) - 1:
            nodes.append(helper.make_node("Relu", [x], [f"{f}_relu"], name=f"fc_relu{i + 1}"))
            x = f"{f}_relu"
    nodes += [helper.make_node("Slice", [x, "s0", "s96", "axis1"], ["pitch_logits"], name="pitch_bins"),
              helper.make_node("Slice", [x, "s96", "s97", "axis1"], ["voiced_2d"], name="voicing"),
              helper.make_node("Squeeze", ["voiced_2d", "axis1"], ["voiced_logit"], name="voiced_scalar")]
    return _model(nodes,
                  [helper.make_tensor_value_info("samples", TensorProto.FLOAT, [1, FRAME])],
                  [helper.make_tensor_value_info("pitch_logits", TensorProto.FLOAT, [1, BINS]),
                   helper.make_tensor_value_info("voiced_logit", TensorProto.FLOAT, [1])],
                  inits, name,
                  "Ear: last 20 ms (320 samples @16 kHz) -> 96 pitch bins of 25 cent (-600..+1775 cent re A4) "
                  "and a voicing logit.")


def gru_graph(a, prefix, head, out_act, out_name, name, doc):
    """inp Linear -> tanh -> GRUCell -> head -> out_act (one step).  head: list of (key, activation or None)."""
    w_in = a[f"{prefix}.inp.w"]
    hid = a[f"{prefix}.cell.w_hh"].shape[1]
    # PyTorch gates are (r, z, n); ONNX GRU wants (z, r, h)
    def zrh(m):
        r, z, n = np.split(m, 3, axis=0)
        return np.concatenate([z, r, n], 0)
    W = zrh(a[f"{prefix}.cell.w_ih"])[None]
    R = zrh(a[f"{prefix}.cell.w_hh"])[None]
    B = np.concatenate([zrh(a[f"{prefix}.cell.b_ih"]), zrh(a[f"{prefix}.cell.b_hh"])])[None]
    inits = [_init("inp.weight", w_in), _init("inp.bias", a[f"{prefix}.inp.b"]),
             _init("gru.W", W), _init("gru.R", R), _init("gru.B", B),
             numpy_helper.from_array(np.array([0], np.int64), "axis0")]
    nodes = [helper.make_node("Gemm", ["features", "inp.weight", "inp.bias"], ["inp_out"], transB=1, name="input_linear"),
             helper.make_node("Tanh", ["inp_out"], ["inp_tanh"], name="input_tanh"),
             helper.make_node("Unsqueeze", ["inp_tanh", "axis0"], ["seq"], name="as_sequence_of_1"),
             helper.make_node("Unsqueeze", ["h", "axis0"], ["h0"], name="h_as_layer_0"),
             helper.make_node("GRU", ["seq", "gru.W", "gru.R", "gru.B", "", "h0"], ["", "h_last"],
                              hidden_size=hid, linear_before_reset=1, name="GRUCell"),
             helper.make_node("Squeeze", ["h_last", "axis0"], ["h_next"], name="h_next")]
    x = "h_next"
    for i, (key, act) in enumerate(head):
        inits += [_init(f"head{i}.weight", a[f"{key}.w"]), _init(f"head{i}.bias", a[f"{key}.b"])]
        nodes.append(helper.make_node("Gemm", [x, f"head{i}.weight", f"head{i}.bias"], [f"head{i}_out"], transB=1,
                                      name=f"head{i + 1}"))
        x = f"head{i}_out"
        if act == "silu":
            nodes += [helper.make_node("Sigmoid", [x], [f"{x}_sig"], name=f"silu{i + 1}_sigmoid"),
                      helper.make_node("Mul", [x, f"{x}_sig"], [f"{x}_silu"], name=f"silu{i + 1}")]
            x = f"{x}_silu"
    nodes.append(helper.make_node(out_act, [x], [out_name], name=out_name))
    return _model(nodes,
                  [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, w_in.shape[1]]),
                   helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, hid])],
                  [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, 1]),
                   helper.make_tensor_value_info("h_next", TensorProto.FLOAT, [1, hid])],
                  inits, name, doc)


def gru_reference(a, prefix, head, out_act, x, h):
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    x = np.tanh(a[f"{prefix}.inp.w"] @ x + a[f"{prefix}.inp.b"])
    gi = a[f"{prefix}.cell.w_ih"] @ x + a[f"{prefix}.cell.b_ih"]
    gh = a[f"{prefix}.cell.w_hh"] @ h + a[f"{prefix}.cell.b_hh"]
    n_ = len(h)
    r, z = sig(gi[:n_] + gh[:n_]), sig(gi[n_:2 * n_] + gh[n_:2 * n_])
    h = (1 - z) * np.tanh(gi[2 * n_:] + r * gh[2 * n_:]) + z * h
    y = h
    for key, act in head:
        y = a[f"{key}.w"] @ y + a[f"{key}.b"]
        if act == "silu":
            y = y * sig(y)
    return (np.tanh(y) if out_act == "Tanh" else sig(y)), h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step_dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.step_dir / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    rng = np.random.default_rng(0)

    for path in sorted(args.step_dir.glob("*_runtime.npz")):
        a = {k: np.asarray(v) for k, v in np.load(path).items()}
        stem = path.name[:-len("_runtime.npz")]
        if any(k.startswith("ear.") for k in a):
            model = ear_graph(a, stem)
            x = rng.normal(0, .1, (1, FRAME)).astype(np.float32)
            got = ort.InferenceSession(model.SerializeToString()).run(None, {"samples": x})
            ref = Ear(a).forward(x[0])
            err = max(np.abs(got[0][0] - ref[0]).max(), abs(got[1][0] - ref[1]))
        elif "nn.cell.w_hh" in a or "valve.cell.w_hh" in a:
            if "nn.cell.w_hh" in a:
                prefix, head, act, oname = "nn", [("nn.head.0", "silu"), ("nn.head.2", None)], "Tanh", "pwm"
                doc = ("Actuator: one 10 ms step. features = target window (10, normalised cents), voice window (10), "
                       "heard, heard valid, target-heard gap, previous pwm, twin position, twin velocity, voice now"
                       + (", target position, heard position" if a["nn.inp.w"].shape[1] == 29 else "")
                       + "; h = GRU state. -> pwm in [-1, 1], next state.")
            else:
                prefix, head, act, oname = "valve", [("valve.out", None)], "Sigmoid", "open_prob"
                doc = ("Valve (tonguing): one 10 ms step. features = target window (10), voice window (10), heard, "
                       "heard valid, gap, actuator pwm, twin position, twin velocity, valve before; h = GRU state. "
                       "-> probability the valve is open, next state.")
            model = gru_graph(a, prefix, head, act, oname, stem, doc)
            f, hid = a[f"{prefix}.inp.w"].shape[1], a[f"{prefix}.cell.w_hh"].shape[1]
            x = rng.normal(0, 1, (1, f)).astype(np.float32); h = rng.normal(0, .5, (1, hid)).astype(np.float32)
            got = ort.InferenceSession(model.SerializeToString()).run(None, {"features": x, "h": h})
            ref = gru_reference(a, prefix, head, act, x[0], h[0])
            err = max(np.abs(got[0][0] - ref[0]).max(), np.abs(got[1][0] - ref[1]).max())
        else:
            continue
        dst = out / f"{stem}.onnx"
        onnx.save(model, dst)
        print(f"{dst}  max |onnx - numpy| = {err:.2e}")
        assert err < 1e-4, "ONNX graph disagrees with the NumPy runtime"


if __name__ == "__main__":
    main()
