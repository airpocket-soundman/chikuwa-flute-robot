import numpy as np
import torch

import pytest

from flute_rl.yamabiko.nn_performer import NNConfig, NNTwinPerformer
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig
from flute_rl.yamabiko.residual_numpy import NNRuntime, export_nn, runtime_for


def _setup(positions=False, song_slot=0, all_nn=False):
    torch.manual_seed(0)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.yamabiko1())
    model = NNTwinPerformer(plant, NNConfig(positions=positions, song_slot=song_slot, valve=all_nn,
                                            current=all_nn, self_home=all_nn)).double()
    twin = plant.parameters(1, "cpu", torch.float64)
    steps = 120
    cents = torch.full((1, steps), 100.0, dtype=torch.float64); cents[:, 60:] = 300.0
    voice = torch.ones(1, steps, dtype=torch.bool); voice[:, :10] = False
    return plant, model, twin, cents, voice


@pytest.mark.parametrize("positions,song_slot,all_nn", [(False, 0, False), (True, 0, False), (True, 4, False),
                                                        (True, 0, True)])
def test_numpy_runtime_matches_torch_step_by_step(positions, song_slot, all_nn):
    plant, model, twin, cents, voice = _setup(positions, song_slot, all_nn)
    rng = np.random.default_rng(1)
    heard = 100.0 + 50 * rng.standard_normal(cents.shape[1])
    valid = rng.random(cents.shape[1]) > .3
    runtime = runtime_for(export_nn(model, twin))
    assert isinstance(runtime, NNRuntime)
    song = model.empty_song(1, cents.shape[1], "cpu", torch.float64) if song_slot else None
    with torch.no_grad():
        for play in range(2 if song_slot else 1):       # the second play reads the memory of the first
            runtime.prepare(cents[0].numpy(), voice[0].numpy(), keep_song_memory=True)
            estimate, fast = model.start(1, "cpu", torch.float64, twin)
            h = torch.zeros(1, dtype=torch.float64); v = torch.zeros(1, dtype=torch.bool)
            prev = torch.zeros(1, dtype=torch.float64)
            for t in range(cents.shape[1]):
                u, logit, fast = model.act_all(model.features(cents, voice, t, h, v, prev, estimate, twin, song), fast)
                estimate = model.motor.step(estimate, u, twin)
                pwm, valve = runtime.command(t)
                assert abs(pwm - float(u)) < 1e-4, (play, t)
                assert valve == (bool(logit[0] >= 0) if all_nn else bool(voice[0, t]))
                runtime.observe(heard[t], bool(valid[t]))
                h = torch.tensor([heard[t]], dtype=torch.float64); v = torch.tensor([bool(valid[t])]); prev = u
                if song_slot:
                    model.write(song, cents, voice, t, h, v, estimate, u, fast)


def test_perform_and_imitate_run_and_have_gradients():
    plant, model, twin, cents, voice = _setup()
    with torch.no_grad():                # outputs past the deadband, or the plunger never moves (no gradient)
        model.head[-1].bias.fill_(1.5)
    params = plant.parameters(2, "cpu", torch.float64)
    twin2 = plant.parameters(2, "cpu", torch.float64)
    result = model.perform(plant, cents.expand(2, -1), voice.expand(2, -1), params, twin2,
                           torch.Generator().manual_seed(0))
    assert result["pwm"].shape == (2, cents.shape[1])
    result["pitch_cents"].mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    pred = model.imitate(cents, voice, cents, voice, torch.zeros_like(cents), twin)
    assert pred.shape == cents.shape
