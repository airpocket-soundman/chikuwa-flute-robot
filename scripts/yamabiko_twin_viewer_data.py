"""PC: turn the training steps saved by ``yamabiko_nn_train.py --dump-steps`` into the twin viewer's data.

    python scripts/yamabiko_twin_viewer_data.py runs/twin_viewer/nn_f80_t7_tw05 --name nn_f80_t7_tw05 \
        --title "step7c の演奏 NN（新ツイン twin_y1f80_05 で 1500 回）"

Writes docs/twin_viewer/data/<name>/index.json (the run and its saved steps) and step_<n>.json (that step's
batch: every rig's song, the pitch it played, its PWM, plunger position, air and what it heard, and the rig's
parameters against the twin).  Numbers are rounded to keep the files small (cents to 1, PWM and position to
0.001); on/off series are strings of 0 and 1.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RATIOS = ("torque_gain", "coulomb_friction", "viscous_friction", "max_velocity_strokes_s", "torque_tau_s")
SHIFTS = {"deadband": 1.0, "flute_offset_cents": 1.0, "tube_offset_m": 1000.0, "hearing_delay_steps": 1.0}


def bits(a):
    return "".join("1" if x else "0" for x in np.asarray(a).astype(bool))


def ints(a, scale=1.0):
    return [int(round(float(x) * scale)) for x in np.asarray(a)]


def convert(path):
    d = np.load(path)
    cents, voice, pitch = d["cents"], d["voice"].astype(bool), d["pitch"]
    rigs = []
    for i in range(cents.shape[0]):
        err = np.abs(pitch[i] - cents[i])[voice[i]]
        params = {}
        for name in RATIOS:
            if f"rig_{name}" in d.files:
                params[name] = round(float(d[f"rig_{name}"][i] / d[f"rig_twin_{name}"][0]), 3)
        for name, scale in SHIFTS.items():
            if f"rig_{name}" in d.files:
                params[name] = round(float((d[f"rig_{name}"][i] - d[f"rig_twin_{name}"][0]) * scale), 3)
        rigs.append({
            "params": params,
            "mae": round(float(err.mean()), 1) if len(err) else None,
            "hit": round(float(np.mean(err <= 25)), 3) if len(err) else None,
            "cents": ints(cents[i]), "voice": bits(voice[i]), "pitch": ints(pitch[i]),
            "pwm": ints(d["pwm"][i], 1000), "position": ints(d["position"][i], 1000),
            "air": bits(d["air"][i] >= .5), "sounding": bits(d["sounding"][i]),
            "heard": ints(np.where(d["valid"][i], d["heard"][i], 0)), "valid": bits(d["valid"][i]),
        })
    return {"step": int(d["step"]), "loss": round(float(d["loss"]), 4), "mae": round(float(d["mae"]), 1),
            "steps": int(cents.shape[1]), "dt": .01, "rigs": rigs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--log", default=None, help="the training log, for the MAE of every 10th step")
    args = ap.parse_args()
    out = ROOT / "docs" / "twin_viewer" / "data" / args.name
    out.mkdir(parents=True, exist_ok=True)
    steps = []
    for path in sorted(pathlib.Path(args.folder).glob("step_*.npz"), key=lambda p: int(p.stem[5:])):
        data = convert(path)
        name = f"step_{data['step']}.json"
        (out / name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        hits = [r["hit"] for r in data["rigs"] if r["hit"] is not None]
        steps.append({"step": data["step"], "loss": data["loss"], "mae": data["mae"],
                      "hit": round(float(np.mean(hits)), 3), "file": name})
        print(f"{name}: MAE {data['mae']} cents, {(out / name).stat().st_size // 1024} KB")
    curve = []
    if args.log:
        for line in pathlib.Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("step ") and "MAE" in line:
                parts = line.split()
                curve.append({"step": int(parts[1]), "loss": float(parts[3]), "mae": float(parts[5])})
    index = {"name": args.name, "title": args.title, "rigs": 32, "saved": steps, "curve": curve}
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    runs = ROOT / "docs" / "twin_viewer" / "data" / "runs.json"
    listed = json.loads(runs.read_text(encoding="utf-8")) if runs.exists() else []
    listed = [r for r in listed if r["name"] != args.name] + [{"name": args.name, "title": args.title}]
    runs.write_text(json.dumps(listed, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}/index.json and {runs}")


if __name__ == "__main__":
    main()
