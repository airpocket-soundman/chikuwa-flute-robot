"""What runs on the UNO Q must not need torch (the UNO Q has none)."""
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_uno_q_modules_load_without_torch():
    code = ("import sys; sys.modules['torch'] = None\n"      # any 'import torch' now fails
            "import flute_rl.yamabiko.hw, flute_rl.yamabiko.residual_numpy, flute_rl.yamabiko.calibration, "
            "flute_rl.yamabiko.server\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
    for script in ("yamabiko_twin_calibrate.py", "yamabiko_twin_play.py", "yamabiko_bench_watch.py", "yamabiko_server.py"):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "import torch" not in source and "fitting import" not in source and "device import" not in source


class FakeRig:
    """Answers every step with a pitch that follows the command; every 7th step is not heard."""

    def __init__(self):
        self.calls = []

    def resync(self):
        pass

    def step(self, pwm, valve):
        self.calls.append((pwm, valve))
        heard = 1000.0 + 100.0 * pwm if valve and len(self.calls) % 7 else float("nan")
        return {"heard": np.array([heard])}


def test_real_calibration_record_has_the_layout_of_save_logs():
    from flute_rl.yamabiko.calibration import calibrate_real, calibration_segments

    rig = FakeRig()
    record = calibrate_real(rig, max_pwm=.6, sweep=.5, homing_seconds=.2)
    steps = sum(n for _, n in calibration_segments(.6, .5))
    assert rig.calls[:20] == [(-1.0, False)] * 20 and len(rig.calls) == 20 + steps
    assert record["count"] == 1
    for name, dtype in (("target_cents", np.float32), ("target_voice", bool), ("pwm", np.float32),
                        ("valve", np.float32), ("heard", np.float32), ("valid", bool)):
        assert record[f"0/{name}"].shape == (1, steps) and record[f"0/{name}"].dtype == dtype
    assert np.abs(record["0/pwm"]).max() <= .6 + 1e-6
    valid = record["0/valid"][0]
    assert 0 < valid.sum() < steps and np.all(record["0/heard"][0][~valid] == 0)
