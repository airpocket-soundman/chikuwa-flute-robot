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
