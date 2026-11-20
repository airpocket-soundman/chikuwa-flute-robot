"""The calibration run (no torch here: the UNO Q plays it, the PC fits it in fitting.py)."""
from __future__ import annotations

import numpy as np

# (pwm, steps) segments of the calibration run, about 9 s at 100 Hz.  The
# full-speed runs are long enough (1 s) for the slowest motors to reach their
# top speed; a fast motor simply rests against the far stop meanwhile.
CALIBRATION = ([(level, 25) for level in (.10, .15, .20, .25, .30, .35, .40)] + [(0.0, 10)] +
               [(-1.0, 60), (1.0, 100), (-1.0, 100), (.6, 50), (-.6, 50), (.35, 150), (-.35, 150)])


def calibration_segments(max_pwm: float = 1.0, sweep: float = 1.0):
    """The calibration run, optionally made gentler for a real rig.

    ``max_pwm`` caps every level (the deadband staircase is kept as is) and
    ``sweep`` scales the duration of the slow sweeps, so the plunger stays
    inside the range where the flute sounds cleanly (no overblowing) and
    end-stop hits are softer.  The defaults are the benchmark run.
    """
    segments = []
    for level, steps in CALIBRATION:
        gentle = level if abs(level) <= .4 else max(-max_pwm, min(max_pwm, level))
        if abs(level) == .35 and steps == 150:
            steps = int(round(steps * sweep))
        segments.append((gentle, steps))
    return segments


def calibration_sequence(max_pwm: float = 1.0, sweep: float = 1.0) -> np.ndarray:
    """The run as one PWM value per 10 ms step."""
    return np.concatenate([np.full(steps, level, np.float32) for level, steps in calibration_segments(max_pwm, sweep)])


def calibrate_real(rig, max_pwm: float = 1.0, sweep: float = 1.0, homing_seconds: float = 1.3) -> dict:
    """Home, then play the run on :class:`flute_rl.yamabiko.hw.RealRig` with the valve open.

    Returns the record in the layout of ``device.save_logs`` (one play, index 0), which
    ``device.load_logs`` and scripts/yamabiko_twin_fit.py read.
    """
    rig.resync()
    for _ in range(int(round(homing_seconds / .01))):   # against the home stop, air shut
        rig.step(-1.0, False)
    pwm = calibration_sequence(max_pwm, sweep)
    heard = np.zeros(len(pwm), np.float32)
    valid = np.zeros(len(pwm), bool)
    for t, level in enumerate(pwm):
        h = float(rig.step(float(level), True)["heard"][0])
        valid[t] = np.isfinite(h)
        heard[t] = h if valid[t] else 0.0
    return {"count": 1, "0/target_cents": np.zeros((1, len(pwm)), np.float32),
            "0/target_voice": np.zeros((1, len(pwm)), bool), "0/pwm": pwm[None],
            "0/valve": np.ones((1, len(pwm)), np.float32), "0/heard": heard[None], "0/valid": valid[None]}
