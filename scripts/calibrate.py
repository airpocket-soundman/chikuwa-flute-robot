"""Fit the simulator to measurements of the real flute (see flute_rl/calibrate.py).

Depth sweep (a CSV with lines "depth_mm,file.wav"; depth measured from the start position):
    python scripts/calibrate.py tube sweep_depth.csv --bore-mm 10

Angle sweep at one depth (lines "angle_deg,file.wav"):
    python scripts/calibrate.py window sweep_angle.csv

Actuator speed (lines "pwm,file.wav", each a move at constant PWM with the flute sounding;
needs the tube fit's acoustic length and speed of sound):
    python scripts/calibrate.py actuator sweep_pwm.csv --acoustic-mm 150.5 --sound-speed 346.4

Prints each recording's pitch and the fitted values, and the FluteParams values to use.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.calibrate import analyse, fit_actuator, fit_tube, fit_window, read_wav, speed_from_recording  # noqa: E402


def load(csv_path: str) -> list[tuple[float, str]]:
    base = pathlib.Path(csv_path).parent
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if not r or r[0].strip().startswith("#"):
                continue
            wav = pathlib.Path(r[1].strip())
            rows.append((float(r[0]), str(wav if wav.is_absolute() else base / wav)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("tube", "window", "actuator"))
    ap.add_argument("csv")
    ap.add_argument("--bore-mm", type=float, default=10.0, help="inner diameter, to split the acoustic length")
    ap.add_argument("--acoustic-mm", type=float, default=None, help="actuator: acoustic length from the tube fit")
    ap.add_argument("--sound-speed", type=float, default=None, help="actuator: speed of sound from the tube fit")
    args = ap.parse_args()

    rows = load(args.csv)
    if args.kind == "actuator":
        if args.acoustic_mm is None or args.sound_speed is None:
            ap.error("actuator needs --acoustic-mm and --sound-speed (run the tube fit first)")
        speeds = []
        for v, path in rows:
            x, sr = read_wav(path)
            sp = speed_from_recording(x, sr, args.acoustic_mm / 1000.0, args.sound_speed)
            speeds.append(sp)
            print(f"  pwm {v:5.2f}  {pathlib.Path(path).name:24s}  speed {1000 * sp:7.1f} mm/s")
        fit = fit_actuator(np.array([v for v, _ in rows]), np.array(speeds))
        print(f"v_max {1000 * fit.v_max:.1f} mm/s, dead band {fit.deadband:.2f}, PWM curve {fit.pwm_curve:.2f}, "
              f"residuals {np.round(1000 * fit.residual, 1)} mm/s")
        print(f"FluteParams: v_max_in (or v_max_out)={fit.v_max:.4f}, deadband={fit.deadband:.3f}, pwm_curve={fit.pwm_curve:.2f}")
        return
    tones = []
    for v, path in rows:
        x, sr = read_wav(path)
        t = analyse(x, sr)
        tones.append(t)
        print(f"  {v:7.2f}  {pathlib.Path(path).name:24s}  {t.hz:8.1f} Hz  sounding {t.sounding:.2f}  wobble {t.wobble_cents:5.1f} c")
    vals = np.array([v for v, _ in rows])
    hz = np.array([t.hz for t in tones])
    if args.kind == "tube":
        fit = fit_tube(vals / 1000.0, hz)
        end_corr = 0.6 * args.bore_mm / 2 / 1000.0
        print(f"acoustic length at depth 0: {fit.acoustic_len_m * 1000:.1f} mm, speed of sound {fit.speed_of_sound:.1f} m/s "
              f"(= {fit.temp_c:.1f} C), residuals {np.round(fit.residual_cents, 1)} cents")
        print(f"FluteParams: tube_len={fit.acoustic_len_m - end_corr:.4f}, end_corr={end_corr:.4f} "
              f"(end correction taken as 0.6 x radius), temp_c={fit.temp_c:.1f}")
    else:
        fit = fit_window(vals, hz, np.array([t.sounding for t in tones]))
        print(f"sounds from {fit.lo_deg:.1f} to {fit.hi_deg:.1f} deg (centre {fit.centre_deg:.1f}), "
              f"pitch bend {fit.cents_per_deg:.1f} cents/deg")
        half = 0.5 * (fit.hi_deg - fit.lo_deg)
        print(f"FluteParams (angles relative to your zero): theta_opt_deg={fit.centre_deg:.1f}, "
              f"win_lo_deg={half:.1f}, win_hi_deg={half:.1f}, k_theta={fit.cents_per_deg:.1f}")


if __name__ == "__main__":
    main()
