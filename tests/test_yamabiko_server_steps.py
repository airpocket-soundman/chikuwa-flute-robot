"""The server plays the step2 / step3 arrangement (actuator network + valve network, YIN or the ear network),
against the fake MCU link of test_yamabiko_server.py (a 1 kHz tone)."""
import numpy as np
import pytest
import torch

from flute_rl.yamabiko.ear_nn import export as export_ear, torch_model
from flute_rl.yamabiko.nn_performer import NNConfig, NNTwinPerformer
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig
from flute_rl.yamabiko.remote import Remote
from flute_rl.yamabiko.residual_numpy import export_nn, export_valve
from flute_rl.yamabiko.valve_nn import ValveNN
from test_yamabiko_server import server  # noqa: F401  (the fixture)


@pytest.fixture()
def files(tmp_path):
    torch.manual_seed(0)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.yamabiko1_fan80())
    twin = plant.parameters(1, "cpu")
    np.savez(tmp_path / "act.npz", **export_nn(NNTwinPerformer(plant, NNConfig(positions=True)), twin))
    valve = ValveNN(plant.config.normalize)
    with torch.no_grad():
        valve.out.bias.fill_(3.0)                       # open
    np.savez(tmp_path / "valve.npz", **export_valve(valve))
    np.savez(tmp_path / "ear.npz", **export_ear(torch_model()))
    steps = 60
    cents = np.full(steps, 1300.0, np.float32); voice = np.ones(steps, bool); voice[:10] = False
    np.savez(tmp_path / "target.npz", cents=cents, voice=voice)
    return tmp_path


@pytest.mark.parametrize("ear", ["yin", "nn"])
def test_play_with_valve_network_and_ear(server, files, ear):   # noqa: F811
    srv, link = server
    with Remote(port=srv.port, forward=False) as rig:
        kw = dict(valve_runtime=str(files / "valve.npz"))
        if ear == "nn":
            kw.update(ear="nn", ear_runtime=str(files / "ear.npz"))
        result = rig.play(str(files / "act.npz"), str(files / "target.npz"), out=str(files / "play.npz"), repeats=1,
                          fan=.8, shift_cents=5.0, warmup_s=.1, homing_seconds=.1, **kw)
    assert result["shift_cents"] == 5.0
    record = np.load(files / "play.npz")
    assert record["0/valve"].shape == (1, 60) and record["0/valve"][0].all()      # the valve network kept it open
    assert record["0/pcm"].size > 0
    assert record["0/target_cents"][0, 0] == pytest.approx(1305.0)                # moved by shift_cents
    if ear == "yin":                                                               # YIN hears the fake 1 kHz tone
        heard = record["0/heard"][0][record["0/valid"][0]]
        assert heard.size and np.median(heard) == pytest.approx(1200 * np.log2(1000 / 440), abs=10)
