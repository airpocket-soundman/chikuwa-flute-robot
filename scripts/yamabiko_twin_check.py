"""PC: how well fitted twins replay real records, one record at a time.

    python scripts/yamabiko_twin_check.py runs/real/twin_y1f80_03.json runs/real/twin_y1f80_04.json \
        --records runs/real/calibration_y1f80_03.npz runs/real/eval/20261123_174554_*.npz

Every record is replayed from the home stop with the twin's values and hearing delay; prints the mean
|heard - twin| over the heard steps (cents) and the median signed offset (heard - twin) per record and twin.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.device import load_logs  # noqa: E402
from flute_rl.yamabiko.fitting import TwinFitter  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402


def twin_params(plant, values, device):
    params = plant.parameters(1, device)
    for name, value in values.items():
        current = getattr(params, name, None)
        if torch.is_tensor(current):
            setattr(params, name, torch.full_like(current, float(value)))
        elif current is not None or hasattr(params, name):
            setattr(params, name, value)
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("twins", nargs="+")
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rows = []
    for path in args.twins:
        report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        plant = DifferentiableMotorFlute(PhysicalPlantConfig.named(report["plant"], temp_c=report.get("temp_c", 25.0)))
        params = twin_params(plant, report["twin"], args.device)
        delay = int(report.get("delay_steps", 0))
        fitter = TwinFitter(plant)
        for record in args.records:
            for i, log in enumerate(load_logs(record, args.device)):
                with torch.no_grad():
                    predicted = fitter.predict(params, log.pwm[:1], delay)
                valid = log.valid[:1]
                diff = (log.heard[:1] - predicted)[valid]
                rows.append((pathlib.Path(path).stem, pathlib.Path(record).stem, i,
                             float(diff.abs().mean()), float(diff.median())))
    for twin, record, i, mae, offset in rows:
        print(f"{twin:16s} {record:40s} {i}  mae {mae:6.1f}  offset {offset:+6.1f}")


if __name__ == "__main__":
    main()
