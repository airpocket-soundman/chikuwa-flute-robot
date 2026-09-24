"""PC: fit the digital twin to a real calibration record and export the runtime.

    python scripts/yamabiko_twin_fit.py runs/real/calibration_01.npz --out runs/real/twin_01

Writes ``<out>.json`` (the twin values and the fit error) and
``<out>_runtime.npz`` (the NumPy performer for the UNO Q, with the twin
inside).  Songs recorded later (scripts/yamabiko_twin_play.py) can be added
with ``--songs`` to refine the twin on real playing; every record starts at
the home stop, so they are fitted the same way, one after another.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.device import PlayLog, load_logs  # noqa: E402
from flute_rl.yamabiko.fitting import TwinFitter  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.residual_numpy import export  # noqa: E402
from flute_rl.yamabiko.residual_performer import ResidualTwinPerformer  # noqa: E402


def concatenate(logs):
    """Several records of one rig as one batch row each (padded with silence)."""
    length = max(log.pwm.shape[1] for log in logs)

    def padded(name):
        parts = []
        for log in logs:
            value = getattr(log, name)
            extra = length - value.shape[1]
            fill = torch.zeros(value.shape[0], extra, dtype=value.dtype, device=value.device)
            parts.append(torch.cat([value, fill], 1))
        return torch.cat(parts, 0)
    return PlayLog(*(padded(name) for name in ("target_cents", "target_voice", "pwm", "valve", "heard", "valid")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calibration")
    ap.add_argument("--songs", nargs="*", default=[], help="play records to fit as well")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="runs/yamabiko_residual_v1.pt")
    ap.add_argument("--fit-steps", type=int, default=300)
    ap.add_argument("--temp", type=float, default=None, help="room temperature [C] if measured")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    config = PhysicalPlantConfig.realistic(**({"temp_c": args.temp} if args.temp is not None else {}))
    plant = DifferentiableMotorFlute(config)
    logs = load_logs(args.calibration, device)
    for path in args.songs:
        logs += load_logs(path, device)
    fit = TwinFitter(plant).fit(concatenate(logs), steps=args.fit_steps)
    twin = fit.parameters
    # Every row is the same rig: keep the calibration row as the twin.
    values = {name: float(value[0]) for name, value in vars(twin).items() if value is not None}
    report = {"twin": values, "heard_mae_cents_per_record": [float(x) for x in fit.heard_mae],
              "delay_steps": int(fit.delay[0]), "records": [args.calibration] + list(args.songs),
              "fit_steps": args.fit_steps, "temp_c": config.temp_c}
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model = ResidualTwinPerformer.from_checkpoint(torch.load(args.checkpoint, map_location=device, weights_only=False),
                                                  plant, device).eval()
    np.savez(str(out) + "_runtime.npz", **export(model, twin, 0))
    print(json.dumps(report, indent=2)); print(f"saved {out}.json and {out}_runtime.npz")


if __name__ == "__main__":
    main()
