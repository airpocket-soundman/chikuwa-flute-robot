"""The calibration run (no torch here: the UNO Q plays it, the PC fits it in fitting.py)."""
from __future__ import annotations

import numpy as np

# (pwm, steps) segments of the calibration run, about 9 s at 100 Hz.  The
# full-speed runs are long enough (1 s) for the slowest motors to reach their
# top speed; a fast motor simply rests against the far stop meanwhile.
CALIBRATION = ([(level, 25) for level in (.10, .15, .20, .25, .30, .35, .40)] + [(0.0, 10)] +
               [(-1.0, 60), (1.0, 100), (-1.0, 100), (.6, 50), (-.6, 50), (.35, 150), (-.35, 150)])


# The first real rig (PhysicalPlantConfig.yamabiko1): it does not move below |PWM| 0.5 and overblows past
# about 0.7 s of full PWM from home, so every run out stays within 0.55 s of full-PWM travel and each
# returns home (about 10.7 s).
CALIBRATION_YAMABIKO1 = ([(level, 25) for level in (.30, .40, .45, .50, .55, .60)] + [(0.0, 10)] +
                         [(-1.0, 80), (1.0, 55), (0.0, 30), (-1.0, 80),
                          (.8, 50), (0.0, 30), (-.8, 50), (-1.0, 40),
                          (.6, 100), (0.0, 30), (-.6, 100), (-1.0, 40),
                          (.7, 40), (0.0, 20), (.7, 40), (0.0, 20), (-1.0, 80)])
PLANS = {"default": CALIBRATION, "yamabiko1": CALIBRATION_YAMABIKO1}


def calibration_segments(max_pwm: float = 1.0, sweep: float = 1.0, plan: str = "default"):
    """The calibration run, optionally made gentler for a real rig.

    ``max_pwm`` caps every level (the deadband staircase is kept as is) and
    ``sweep`` scales the duration of the slow sweeps, so the plunger stays
    inside the range where the flute sounds cleanly (no overblowing) and
    end-stop hits are softer.  The defaults are the benchmark run.
    """
    segments = []
    if plan != "default":
        return [(max(-max_pwm, min(max_pwm, level)), steps) for level, steps in PLANS[plan]]
    for level, steps in CALIBRATION:
        gentle = level if abs(level) <= .4 else max(-max_pwm, min(max_pwm, level))
        if abs(level) == .35 and steps == 150:
            steps = int(round(steps * sweep))
        segments.append((gentle, steps))
    return segments


def calibration_sequence(max_pwm: float = 1.0, sweep: float = 1.0, plan: str = "default") -> np.ndarray:
    """The run as one PWM value per 10 ms step."""
    return np.concatenate([np.full(steps, level, np.float32)
                           for level, steps in calibration_segments(max_pwm, sweep, plan)])


def calibrate_real(rig, max_pwm: float = 1.0, sweep: float = 1.0, homing_seconds: float = 1.3,
                   plan: str = "default", max_cents: float | None = None) -> dict:
    """Home, then play the run on :class:`flute_rl.yamabiko.hw.RealRig` with the valve open.

    Returns the record in the layout of ``device.save_logs`` (one play, index 0), which
    ``device.load_logs`` and scripts/yamabiko_twin_fit.py read.  ``plan`` picks the PWM sequence
    (:data:`PLANS`); pitch above ``max_cents`` (an overblown flute) is recorded as not heard.
    """
    rig.resync()
    for _ in range(int(round(homing_seconds / .01))):   # against the home stop, air shut
        rig.step(-1.0, False)
    pwm = calibration_sequence(max_pwm, sweep, plan)
    heard = np.zeros(len(pwm), np.float32)
    valid = np.zeros(len(pwm), bool)
    for t, level in enumerate(pwm):
        h = float(rig.step(float(level), True)["heard"][0])
        valid[t] = np.isfinite(h) and (max_cents is None or h <= max_cents)
        heard[t] = h if valid[t] else 0.0
    return {"count": 1, "0/target_cents": np.zeros((1, len(pwm)), np.float32),
            "0/target_voice": np.zeros((1, len(pwm)), bool), "0/pwm": pwm[None],
            "0/valve": np.ones((1, len(pwm)), np.float32), "0/heard": heard[None], "0/valid": valid[None]}
