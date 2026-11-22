import numpy as np
import torch

from flute_rl.yamabiko.nn_performer import NNConfig, NNTwinPerformer
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig
from flute_rl.yamabiko.residual_numpy import WithValve, export_nn, export_valve, runtime_for
from flute_rl.yamabiko.valve_nn import ValveNN, play_with_valve


def _setup():
    torch.manual_seed(0)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.yamabiko1_fan80())
    actuator = NNTwinPerformer(plant, NNConfig(positions=True)).double()
    valve = ValveNN(plant.config.normalize).double()
    twin = plant.parameters(1, "cpu", torch.float64)
    steps = 100
    cents = torch.full((1, steps), 200.0, dtype=torch.float64); cents[:, 50:] = 600.0
    voice = torch.ones(1, steps, dtype=torch.bool); voice[:, :10] = False; voice[:, 48:52] = False
    return plant, actuator, valve, twin, cents, voice


def test_numpy_runs_both_networks_as_torch_does():
    plant, actuator, valve, twin, cents, voice = _setup()
    runtime = WithValve(runtime_for(export_nn(actuator, twin)), export_valve(valve))
    runtime.prepare(cents[0].numpy(), voice[0].numpy())
    rng = np.random.default_rng(2)
    heard = 300 + 100 * rng.standard_normal(cents.shape[1]); valid = rng.random(cents.shape[1]) > .3
    estimate, fast = actuator.start(1, "cpu", torch.float64, twin)
    hv = torch.zeros(1, valve.config.hidden, dtype=torch.float64)
    h = torch.zeros(1, dtype=torch.float64); v = torch.zeros(1, dtype=torch.bool)
    pwm = torch.zeros(1, dtype=torch.float64); prev = torch.zeros(1, dtype=torch.float64)
    with torch.no_grad():
        for t in range(cents.shape[1]):
            pwm, fast = actuator.act(actuator.features(cents, voice, t, h, v, pwm, estimate, twin), fast)
            p, hv = valve.step(valve.features(cents, voice, t, h, v, pwm, estimate, prev), hv)
            estimate = actuator.motor.step(estimate, pwm, twin)
            got_pwm, got_valve = runtime.command(t)
            assert abs(got_pwm - float(pwm)) < 1e-4, t
            assert got_valve == bool(p[0] >= .5), t
            assert abs(runtime.open - float(p)) < 1e-4, t
            runtime.observe(heard[t], bool(valid[t]))
            h = torch.tensor([heard[t]], dtype=torch.float64); v = torch.tensor([bool(valid[t])]); prev = p


def test_closed_loop_gives_valve_gradients_only():
    plant, actuator, valve, twin, cents, voice = _setup()
    params = plant.parameters(2, "cpu", torch.float64); twin2 = plant.parameters(2, "cpu", torch.float64)
    emitted, air, sounding, cmd = play_with_valve(actuator, valve, plant, cents.expand(2, -1), voice.expand(2, -1),
                                                  params, twin2, torch.Generator().manual_seed(0))
    assert emitted.shape == air.shape == sounding.shape == cmd.shape == (2, cents.shape[1])
    cmd.mean().backward()
    assert valve.out.weight.grad is not None and valve.out.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in actuator.parameters())
