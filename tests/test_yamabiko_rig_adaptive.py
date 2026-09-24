import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.deterministic_pipeline import (EncoderlessConfig,  # noqa: E402
                                                       EncoderlessDeterministicPerformer)
from flute_rl.yamabiko.melodies import next_voiced, note_age, random_melodies  # noqa: E402
from flute_rl.yamabiko.physical_plant import (AudibleListener, DifferentiableMotorFlute,  # noqa: E402
                                              PhysicalPlantConfig)
from flute_rl.yamabiko.rig_adaptive import RigAdaptiveConfig, RigAdaptivePerformer  # noqa: E402


def test_deadband_blocks_small_pwm_and_rescales_large_pwm():
    plant = DifferentiableMotorFlute(PhysicalPlantConfig(deadband=.2))
    params = plant.parameters(2, "cpu")
    state = plant.step(plant.initial_state(2, "cpu"), torch.tensor([.15, 1.0]), params)
    assert state.torque[0].item() == 0.0
    free = DifferentiableMotorFlute(PhysicalPlantConfig())
    reference = free.step(free.initial_state(1, "cpu"), torch.ones(1), free.parameters(1, "cpu"))
    assert state.torque[1].item() == pytest.approx(reference.torque.item())


def test_listener_delays_each_rig_by_its_own_steps():
    cfg = PhysicalPlantConfig(hearing_delay_steps=1, hearing_delay_jitter=1)
    plant = DifferentiableMotorFlute(cfg)
    params = plant.parameters(2, "cpu")
    params.hearing_delay_steps = torch.tensor([0, 2])
    listener = AudibleListener(cfg); state = listener.initial_state(2, "cpu")
    heard = []
    for cents in (100.0, 200.0, 300.0):
        value, valid, state = listener.step(torch.full((2,), cents), torch.ones(2, dtype=torch.bool), params, state)
        heard.append(value)
    assert heard[-1][0].item() == 300.0 and heard[-1][1].item() == 100.0


def test_melody_helpers_hold_rests_and_count_note_age():
    cents, voice = random_melodies(np.random.default_rng(1), 4, 400)
    assert cents.shape == voice.shape == (4, 400) and not voice[:, :20].any()
    aim = next_voiced(cents, voice)
    first = int(torch.nonzero(voice[0])[0])
    assert aim[0, 0] == cents[0, first]
    age = note_age(voice)
    assert age[0, first] == 1 and (age[~voice] == 0).all()


def test_encoderless_performer_tracks_pitch_without_plant_state():
    torch.manual_seed(0)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    params = plant.parameters(8, "cpu", spread=1.0, generator=torch.Generator().manual_seed(3))
    cents, voice = random_melodies(np.random.default_rng(2), 8, 500)
    with torch.no_grad():
        result, memory = EncoderlessDeterministicPerformer(plant).perform(
            cents, voice, params, generator=torch.Generator().manual_seed(0))
        blind, _ = EncoderlessDeterministicPerformer(plant, EncoderlessConfig(observer_gain=0.0,
                                                     observer_velocity_gain=0.0)).perform(
            cents, voice, params, generator=torch.Generator().manual_seed(0))
    settled = voice & (note_age(voice) > 30)
    error = (result["pitch_cents"] - cents).abs()[settled].mean()
    blind_error = (blind["pitch_cents"] - cents).abs()[settled].mean()
    assert error < 80 and error < blind_error
    assert memory.theta.shape == (8, 2)


def test_rig_adaptive_performer_carries_memory_and_backpropagates():
    torch.manual_seed(0)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    params = plant.parameters(3, "cpu", spread=1.0, generator=torch.Generator().manual_seed(4))
    model = RigAdaptivePerformer(RigAdaptiveConfig(fast_hidden=16, planner_hidden=16, homing_steps=5))
    cents, voice = random_melodies(np.random.default_rng(5), 3, 40, lead_rest=(2, 4))
    result, memory = model.perform(plant, cents, voice, params, generator=torch.Generator().manual_seed(0))
    assert result["pitch_cents"].shape == (3, 40) and memory.shape == (3, 16)
    assert memory.abs().sum() > 0
    (result["pitch_cents"] - cents).abs().mean().backward()
    assert model.planner.net[0].weight.grad.abs().sum() > 0
    restored = RigAdaptivePerformer.from_checkpoint(model.checkpoint())
    torch.testing.assert_close(restored.core.head[0].weight, model.core.head[0].weight)


def test_varied_melodies_repeat_notes_and_keep_default_songs():
    cents, voice = random_melodies(np.random.default_rng(3), 32, 800, varied=True)
    onsets = voice[:, 1:] & ~voice[:, :-1]
    rows, steps = torch.nonzero(onsets, as_tuple=True)
    same = [(cents[r, s + 1] == cents[r, s - 8]).item() for r, s in zip(rows.tolist(), steps.tolist()) if s >= 8]
    assert any(same)
    first = random_melodies(np.random.default_rng(3), 4, 300)
    second = random_melodies(np.random.default_rng(3), 4, 300, varied=False)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_timeline_pitch_residual_zero_stores_the_ear_pitch():
    from flute_rl.yamabiko.beat_grid import BeatGridConfig, BeatTimelineRecallNet
    torch.manual_seed(0)
    timeline = BeatTimelineRecallNet(BeatGridConfig(), pitch_residual=0.0)
    stored = torch.randn(2, 7, 2 * BeatGridConfig().memory_hidden + 2)
    decoded = timeline.decode(stored)
    torch.testing.assert_close(decoded[..., 0], stored[..., -2])


def test_motor_scale_range_keeps_other_draws_and_calibration_measures_it():
    base = DifferentiableMotorFlute(PhysicalPlantConfig.realistic(motor_scale_range=None))
    wide = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    a = base.parameters(64, "cpu", spread=1.0, generator=torch.Generator().manual_seed(1))
    b = wide.parameters(64, "cpu", spread=1.0, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a.tube_offset_m, b.tube_offset_m)
    ratio = b.max_velocity_strokes_s / a.max_velocity_strokes_s
    torch.testing.assert_close(b.torque_gain / a.torque_gain, ratio)
    assert ratio.min() >= .4 - 1e-6 and ratio.max() <= 2.5 + 1e-6
    with torch.no_grad():
        measured = EncoderlessDeterministicPerformer(wide).measure_motor(b, torch.Generator().manual_seed(0))
    true = b.max_velocity_strokes_s / wide.config.max_velocity_strokes_s
    assert np.corrcoef(measured.numpy(), true.numpy())[0, 1] > .7


def test_home_margin_keeps_the_lowest_note_reachable_on_every_rig():
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    params = plant.parameters(512, "cpu", spread=1.0, generator=torch.Generator().manual_seed(2))
    home = plant.cents_at(torch.zeros(512), params)
    assert home.max() < plant.config.low_cents


def test_black_box_device_matches_simulator_without_gradient_and_logs_observables():
    from flute_rl.yamabiko.adaptive_memory import AdaptiveMemoryConfig, AdaptiveMemoryPerformer
    from flute_rl.yamabiko.device import BlackBoxDevice
    torch.manual_seed(0)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    params = plant.parameters(3, "cpu", spread=1.0, generator=torch.Generator().manual_seed(1))
    model = AdaptiveMemoryPerformer(AdaptiveMemoryConfig(fast_hidden=16, planner_hidden=16, homing_steps=5))
    cents, voice = random_melodies(np.random.default_rng(0), 3, 50, lead_rest=(2, 4))
    simulated, memory = model.perform(plant, cents, voice, params, generator=torch.Generator().manual_seed(2))
    rig = BlackBoxDevice(plant, params, torch.Generator().manual_seed(2))
    played, _ = model.play(rig, cents, voice)
    torch.testing.assert_close(simulated["pitch_cents"], played["pitch_cents"])
    assert simulated["pitch_cents"].requires_grad and not played["pitch_cents"].requires_grad
    assert len(rig.logs) == 1 and set(vars(rig.logs[0])) == {"target_cents", "target_voice", "pwm", "valve",
                                                             "heard", "valid"}
    assert memory["song"].shape == (3, 50, 4)
    wiped = model.forget(memory, "motor", rows=torch.tensor([True, False, False]))
    assert wiped["motor"][0].abs().sum() == 0 and torch.equal(wiped["motor"][1], memory["motor"][1])


def test_world_model_pitch_is_monotone_and_trains_from_logs():
    from flute_rl.yamabiko.device import BlackBoxDevice
    from flute_rl.yamabiko.world_model import DeviceWorldModel, WorldModelConfig
    torch.manual_seed(0)
    world = DeviceWorldModel(WorldModelConfig(hidden=16))
    context = torch.randn(1, world.config.context)
    position = torch.linspace(0, 1, 50)
    pitch = world.pitch(position, context.expand(50, -1))
    assert (pitch[1:] >= pitch[:-1] - 1e-6).all()
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    rig = BlackBoxDevice(plant, plant.parameters(2, "cpu"), torch.Generator().manual_seed(0))
    state = rig.reset("cpu", homing_steps=5); records = {"pwm": [], "heard": [], "valid": [], "emitted": []}
    for t in range(30):
        pwm = torch.full((2,), .6 if t < 20 else -.6)
        heard, valid, state, emitted = rig.step(state, pwm, torch.ones(2))
        for key, value in (("pwm", pwm), ("heard", heard), ("valid", valid), ("emitted", emitted)):
            records[key].append(value)
    stacked = {k: torch.stack(v, 1) for k, v in records.items()}
    rig.record(torch.zeros(2, 30), torch.zeros(2, 30, dtype=torch.bool), stacked["pwm"], torch.ones(2, 30),
               stacked["heard"], stacked["valid"], stacked["emitted"])
    loss, mae = world.loss(rig.logs, world.encoder(rig.logs))
    loss.backward()
    assert torch.isfinite(loss) and world.speed.increments.weight.grad.abs().sum() > 0


def test_residual_performer_starts_as_deterministic_twin_and_learns():
    from flute_rl.yamabiko.residual_performer import ResidualConfig, ResidualTwinPerformer
    torch.manual_seed(0)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic())
    params = plant.parameters(3, "cpu", spread=1.0, generator=torch.Generator().manual_seed(1))
    cents, voice = random_melodies(np.random.default_rng(0), 3, 120)
    model = ResidualTwinPerformer(plant, ResidualConfig(hidden=16))
    played, memory = model.perform(plant, cents, voice, params, None, torch.Generator().manual_seed(0), twin=params)
    with torch.no_grad():
        reference, _ = EncoderlessDeterministicPerformer(plant).perform(
            cents, voice, params, None, torch.Generator().manual_seed(0), twin=params)
    torch.testing.assert_close(played["pitch_cents"], reference["pitch_cents"])
    assert memory["song"].shape == (3, 120, 4)
    (played["pitch_cents"] - cents).abs()[voice].mean().backward()
    assert model.planner[-1].weight.grad.abs().sum() > 0 and model.head[-1].weight.grad.abs().sum() > 0


def test_numpy_runtime_matches_torch_residual_performer():
    from flute_rl.yamabiko.device import SimulatorEnv
    from flute_rl.yamabiko.residual_numpy import Runtime, export
    from flute_rl.yamabiko.residual_performer import ResidualConfig, ResidualTwinPerformer
    torch.manual_seed(1)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.realistic(pitch_noise_cents=0.0, dropout=0.0))
    params = plant.parameters(1, "cpu", spread=1.0, generator=torch.Generator().manual_seed(2))
    model = ResidualTwinPerformer(plant, ResidualConfig(hidden=16))
    for module in (model.planner[-1], model.head[-1]):  # give the residuals something to do
        torch.nn.init.normal_(module.weight, std=.3); torch.nn.init.normal_(module.bias, std=.1)
    cents, voice = random_melodies(np.random.default_rng(3), 1, 150, varied=True)
    with torch.no_grad():
        reference, memory = model.perform(plant, cents, voice, params, None, torch.Generator().manual_seed(0),
                                          twin=params)
    runtime = Runtime(export(model, params)); runtime.prepare(cents[0].numpy(), voice[0].numpy())
    env = SimulatorEnv(plant, params, torch.Generator().manual_seed(0)); state = env.reset("cpu")
    played = []
    for t in range(150):
        pwm, valve = runtime.command(t)
        heard, valid, state, emitted = env.step(state, torch.tensor([pwm], dtype=torch.float32),
                                                torch.tensor([float(valve)]))
        runtime.observe(float(heard[0]), bool(valid[0])); played.append(float(emitted[0]))
    assert np.abs(np.array(played) - reference["pitch_cents"][0].numpy()).max() < .05
    assert np.abs(runtime.song - memory["song"][0].numpy()).max() < 1e-4
