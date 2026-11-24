"""The classic PID performer (classic_numpy.py), the NumPy rig (sim_numpy.py) and the comparison script.
NumPy only: no torch, no rig."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flute_rl.yamabiko.classic_numpy import CLASSIC_FIELDS, ClassicRuntime, export  # noqa: E402
from flute_rl.yamabiko.profiles import latest, load_profiles  # noqa: E402
from flute_rl.yamabiko.remote import Remote  # noqa: E402
from flute_rl.yamabiko.residual_numpy import WithValve, runtime_for  # noqa: E402
from flute_rl.yamabiko.sim_numpy import NumpyRig, RigImperfections, play, rig_arrays, rigs_around  # noqa: E402
from test_yamabiko_server import server  # noqa: E402,F401  (the fixture)
from yamabiko_compare_sim import evaluate, load_target, runtime_maker  # noqa: E402

TWIN = ROOT / "models/step4/twin_y1f80_04_runtime.npz"


@pytest.fixture(scope="module")
def twin_arrays():
    return dict(np.load(TWIN))


def test_export_and_dispatch(twin_arrays):
    arrays = export(twin_arrays, kp=5.0)
    assert arrays["classic.gains"].shape == (len(CLASSIC_FIELDS),)
    runtime = runtime_for(arrays)
    assert isinstance(runtime, ClassicRuntime) and runtime.g["kp"] == 5.0
    with pytest.raises(ValueError):
        export(twin_arrays, nonsense=1.0)


def test_position_for_inverts_cents_at(twin_arrays):
    runtime = runtime_for(export(twin_arrays))
    for x in (.1, .4, .8):
        assert runtime.position_for(runtime.cents_at(x)) == pytest.approx(x, abs=1e-6)


def test_plays_the_twin_itself(twin_arrays):
    """On a rig that is exactly the twin, without noise, the plunger lands on every note."""
    constants, twin = rig_arrays(twin_arrays)
    rig = NumpyRig(constants, twin, RigImperfections(pitch_noise_cents=0, dropout=0, motor_masking=0))
    cents, voice = load_target("f80_doremi")
    out = play(runtime_for(export(twin_arrays)), rig, cents, voice, start_position=.3)
    err = np.abs(out["emitted"] - cents)
    assert np.median(err[voice]) < 10
    assert out["valve"][voice].mean() > .9 and not out["valve"][:20].any()      # shut in the lead rest


def test_rigs_around_are_reproducible(twin_arrays):
    constants, twin = rig_arrays(twin_arrays)
    a = rigs_around(constants, twin, 3, seed=5); b = rigs_around(constants, twin, 3, seed=5)
    assert [r.t for r in a] == [r.t for r in b] and a[0].t != a[1].t


def test_classic_is_a_step_and_compares(twin_arrays):
    profiles = load_profiles()
    assert "classic" in profiles and latest(profiles) != "classic"       # the newest learned step stays the default
    assert profiles["classic"]["ear"] == "yin" and "valve_runtime" not in profiles["classic"]
    constants, twin = rig_arrays(twin_arrays)
    cents, voice = load_target("f80_doremi")
    m, _ = evaluate(runtime_maker(profiles["classic"]), rigs_around(constants, twin, 4), cents, voice)
    assert m["hit"] > .6 and m["mae"] < 30
    step = runtime_maker(profiles["step4"])()
    assert isinstance(step, WithValve)


def test_server_plays_the_classic_runtime(server, tmp_path, twin_arrays):   # noqa: F811
    srv, link = server
    np.savez(tmp_path / "classic.npz", **export(twin_arrays))
    steps = 60
    cents = np.full(steps, 600.0, np.float32); voice = np.ones(steps, bool); voice[:10] = False
    np.savez(tmp_path / "target.npz", cents=cents, voice=voice)
    with Remote(port=srv.port, forward=False) as rig:
        rig.play(str(tmp_path / "classic.npz"), str(tmp_path / "target.npz"), out=str(tmp_path / "play.npz"),
                 repeats=1, fan=.8, shift_cents=0.0, warmup_s=.1, homing_seconds=.1)
    record = np.load(tmp_path / "play.npz")
    valve = record["0/valve"][0]
    assert valve.shape == (steps,) and not valve[:5].any() and valve[45:].all()   # the rule valve: waits for the plunger, at most 25 steps
    assert np.abs(record["0/pwm"][0]).max() > 0
