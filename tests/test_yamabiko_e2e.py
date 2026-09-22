import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flute_rl.yamabiko.e2e import (  # noqa: E402
    FRAME,
    HOP,
    E2EConfig,
    E2EImitator,
    E2ERuntime,
    frame_audio,
)
from flute_rl.yamabiko.e2e_numpy import NumpyE2ERuntime, NumpyE2EModel, export_numpy  # noqa: E402
from flute_rl.yamabiko.e2e_audio import RawSelfAudio  # noqa: E402
from flute_rl.yamabiko.e2e_stages import E2EStageProbes  # noqa: E402
from flute_rl.yamabiko.e2e_io import SAMPLE_RATE  # noqa: E402
from flute_rl.yamabiko.staged_nn import (ErrorComparator, FeedForwardPolicy,
                                         FeedbackResidualPolicy, ReferenceMemory,
                                         AcousticFeedbackResidual, MotorTrajectoryController,
                                         MotorAudioWorldModel,
                                         StagedConfig, TargetPositionPlanner)  # noqa: E402
from flute_rl.yamabiko.physical_plant import (DifferentiableMotorFlute,
                                              PhysicalPlantConfig)  # noqa: E402
from flute_rl.yamabiko.beat_grid import BeatGridConfig, BeatTimelineRecallNet, TempoBeatNet  # noqa: E402
from flute_rl.yamabiko.composite import YamabikoComposite  # noqa: E402
from flute_rl.yamabiko.deterministic_pipeline import (DeterministicEar,
                                                       DeterministicTimelineMemory,
                                                       DeterministicYamabikoPipeline)  # noqa: E402
from flute_rl.yamabiko.hw import PITCH_FRAME, SR  # noqa: E402
from flute_rl.sim import DT  # noqa: E402


def tiny_model():
    return E2EImitator(E2EConfig(audio_width=8, audio_dim=12, reference_hidden=8, controller_hidden=16))


def test_pc_profile_expands_capacity_and_adaptation_memory():
    small = E2EConfig()
    pc = E2EConfig.pc()
    assert pc.audio_width > small.audio_width
    assert pc.audio_dim > small.audio_dim
    assert pc.reference_hidden > small.reference_hidden
    assert pc.controller_hidden > small.controller_hidden
    assert pc.adaptation_dim > small.adaptation_dim
    assert pc.adaptation_dim < pc.controller_hidden


def test_stage_probes_expose_reference_self_and_error_separately():
    config = E2EConfig(audio_width=8, audio_dim=12, reference_hidden=8, controller_hidden=16)
    probes = E2EStageProbes(config)
    memory = torch.randn(5, 2 * config.reference_hidden + config.audio_dim + 2)
    own = torch.randn(5, config.audio_dim + 2)
    reference, self_pitch, error = probes(memory, own)
    assert reference.shape == (5, 2)
    assert self_pitch.shape == error.shape == (5,)
    for parameter in probes.comparator.parameters():
        parameter.data.zero_()
    reference, self_pitch, error = probes(memory, own)
    torch.testing.assert_close(error, reference[:, 0] - self_pitch)


def test_e2e_audio_contract_matches_the_mcu_stream_and_control_clock():
    assert SAMPLE_RATE == SR == 16_000
    assert FRAME == PITCH_FRAME == 320
    assert HOP == round(SR * DT) == 160


def test_raw_audio_framing_is_causal_for_self_sound():
    x = np.arange(10 * HOP, dtype=np.float32) / 1000.0
    causal = frame_audio(x, 10, causal=True)
    reference = frame_audio(x, 10, causal=False)
    assert causal.shape == reference.shape == (10, FRAME)
    assert torch.count_nonzero(causal[0]) == 0
    assert causal[1, -1] == pytest.approx((HOP - 1) / 1000.0)
    assert reference[1, FRAME // 2] == pytest.approx(1.5 * HOP / 1000.0)


def test_randomized_microphone_domain_stays_causal_finite_and_bounded():
    audio = RawSelfAudio(2, np.random.default_rng(7), domain=1.0)
    assert np.count_nonzero(audio.buf) == 0
    audio.push(np.array([1200.0, 1600.0]), np.ones(2, bool))
    assert np.count_nonzero(audio.buf[:, :HOP]) == 0  # result enters only the newest causal half-frame
    assert np.isfinite(audio.buf).all()
    assert np.max(np.abs(audio.buf)) <= 1.0
    with pytest.raises(ValueError):
        RawSelfAudio(1, np.random.default_rng(0), domain=-0.1)


def test_one_model_maps_both_raw_waveforms_to_three_actuator_outputs():
    model = tiny_model()
    reference = torch.randn(2, 7, FRAME)
    own = torch.randn(2, 6, FRAME)
    raw, state, attention = model(reference, own, torch.tensor([7, 4]))
    assert raw.shape == (2, 6, 5)  # PWM, valve, done, target profile, learned self position
    assert state.shape == (2, 16)
    assert attention.shape == (2, 6, 7)
    assert torch.allclose(attention[1, :, 4:], torch.zeros_like(attention[1, :, 4:]))
    assert torch.allclose(attention.sum(-1), torch.ones(2, 6), atol=1e-6)


def test_action_loss_trains_the_shared_audio_encoder_end_to_end():
    model = tiny_model()
    raw, _, _ = model(torch.randn(1, 5, FRAME), torch.randn(1, 5, FRAME))
    raw.square().mean().backward()
    grad = model.audio.body[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_checkpoint_runtime_keeps_adaptation_between_references(tmp_path):
    model = tiny_model()
    path = tmp_path / "policy.pt"
    torch.save(model.checkpoint(), path)
    runtime = E2ERuntime.load(path)
    audio = np.random.default_rng(0).normal(size=5 * HOP).astype(np.float32)
    runtime.start_reference(audio)
    runtime.act(np.zeros(FRAME, np.float32))
    learned = runtime.state.clone()
    assert torch.count_nonzero(learned) > 0
    runtime.start_reference(audio, keep_adaptation=True)
    assert torch.equal(runtime.state, learned)
    runtime.start_reference(audio, keep_adaptation=False)
    assert torch.count_nonzero(runtime.state) == 0


def test_runtime_actions_are_safe_ranges(tmp_path):
    model = tiny_model()
    runtime = E2ERuntime(model)
    runtime.start_reference(np.zeros(3 * HOP, np.float32))
    pwm, valve, done = runtime.act(np.zeros(FRAME, np.float32))
    assert -1.0 <= pwm <= 1.0
    assert isinstance(valve, bool)
    assert 0.0 <= done <= 1.0


def numpy_model(model):
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    return NumpyE2EModel(arrays, vars(model.config))


def test_uno_q_numpy_audio_and_recurrent_outputs_match_pytorch():
    torch.manual_seed(4)
    model = tiny_model().eval()
    portable = numpy_model(model)
    reference = torch.randn(1, 6, FRAME)
    own = torch.randn(1, FRAME)
    with torch.no_grad():
        tm, tk, mask = model.encode_reference(reference)
        state = model.initial_state(1)
        prev = torch.zeros(1, 2)
        traw, tstate, tattn = model.step(own, tm, tk, mask, state, prev, torch.ones(1))
    nm, nk, nmask = portable.encode_reference(reference.numpy())
    nraw, nstate, nattn = portable.step(own.numpy(), nm, nk, nmask, portable.initial_state(),
                                        np.zeros((1, 2), np.float32), True)
    np.testing.assert_allclose(nm, tm.numpy(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(nraw, traw.numpy(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(nstate, tstate.numpy(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(nattn, tattn.numpy(), atol=2e-5, rtol=2e-5)


def test_exported_numpy_runtime_needs_no_torch_checkpoint(tmp_path):
    model = tiny_model().eval()
    path = tmp_path / "policy.npz"
    export_numpy(model, path, source="test")
    runtime = NumpyE2ERuntime.load(path)
    runtime.start_reference(np.zeros(4 * HOP, np.float32))
    pwm, valve, done = runtime.act(np.zeros(FRAME, np.float32))
    assert path.stat().st_size < 1_000_000
    assert -1 <= pwm <= 1 and isinstance(valve, bool) and 0 <= done <= 1


def test_separated_neural_stages_have_explicit_boundaries():
    config = StagedConfig(memory_hidden=8, control_hidden=12, comparator_hidden=8)
    memory = ReferenceMemory(config)
    feedforward = FeedForwardPolicy(config)
    comparator = ErrorComparator(config)
    feedback = FeedbackResidualPolicy(config)
    ear = torch.randn(2, 7, 2)
    decoded, stored = memory(ear)
    assert stored.shape == (2, 7, 16) and decoded.shape == (2, 7, 2)
    actions = feedforward(decoded)
    assert actions.shape == (2, 7, 2)
    error = comparator(decoded[:, 0], ear[:, 0])
    state = torch.zeros(2, config.control_hidden)
    corrected, state = feedback.step(decoded[:, 0], ear[:, 0], error, actions[:, 0],
                                     torch.zeros(2, 2), state)
    assert corrected.shape == (2, 2) and state.shape == (2, config.control_hidden)
    assert torch.all((-1 <= corrected[:, 0]) & (corrected[:, 0] <= 1))


def test_position_planner_treats_voice_input_as_a_logit():
    torch.manual_seed(23)
    planner = TargetPositionPlanner().eval()
    pitch = torch.tensor([[[0.2]]])
    confident = planner(torch.cat([pitch, torch.tensor([[[6.0]]])], -1))
    very_confident = planner(torch.cat([pitch, torch.tensor([[[12.0]]])], -1))
    torch.testing.assert_close(confident, very_confident, atol=1e-4, rtol=1e-4)


def test_minimal_physical_plant_has_torque_rise_motion_and_linear_flute():
    cfg = PhysicalPlantConfig()
    plant = DifferentiableMotorFlute(cfg)
    params = plant.parameters(1, "cpu")
    state = plant.initial_state(1, "cpu")
    first = plant.step(state, torch.ones(1), params)
    assert 0 < first.torque.item() < cfg.torque_gain
    for _ in range(100):
        first = plant.step(first, torch.ones(1), params)
    assert first.position.item() > .1
    cents, sounding = plant.flute(first, torch.ones(1), params)
    assert cents.item() == pytest.approx(cfg.low_cents + cfg.pitch_span_cents * first.position.item(), abs=1e-4)
    assert sounding.item()
    _, silent = plant.flute(first, torch.zeros(1), params)
    assert not silent.item()


def test_motor_physics_is_deterministic_for_fixed_parameters_and_inputs():
    plant = DifferentiableMotorFlute()
    params = plant.parameters(2, "cpu", spread=0.0)
    a = plant.initial_state(2, "cpu")
    b = plant.initial_state(2, "cpu")
    for pwm in torch.linspace(-1.0, 1.0, 80):
        command = pwm.repeat(2)
        a = plant.step(a, command, params)
        b = plant.step(b, command, params)
    torch.testing.assert_close(a.position, b.position)
    torch.testing.assert_close(a.velocity, b.velocity)
    torch.testing.assert_close(a.torque, b.torque)


def test_physical_controllers_have_separate_plan_and_feedback_boundaries():
    batch, steps = 3, 12
    planner_executor = MotorTrajectoryController(hidden=8)
    feedback = AcousticFeedbackResidual(hidden=8)
    plan = torch.linspace(.1, .9, steps).repeat(batch, 1)
    voice = torch.ones_like(plan)
    base_state = planner_executor.initial_state(batch, "cpu")
    base_pwm, valve_logit, base_state = planner_executor.step(
        plan, voice, torch.zeros(batch), base_state, 0)
    feedback_state = feedback.initial_state(batch, "cpu")
    residual, error, feedback_state = feedback.step(
        plan[:, 0], plan[:, 0] - .1, torch.ones(batch, dtype=torch.bool), voice[:, 0],
        base_pwm, torch.zeros(batch), torch.zeros(batch), feedback_state)
    assert base_pwm.shape == valve_logit.shape == residual.shape == error.shape == (batch,)
    assert base_state.shape == (batch, 8)
    assert feedback_state.shape == (batch, 16)
    assert torch.all(residual.abs() <= feedback.limit)


def test_motor_audio_world_model_exposes_only_pwm_to_pitch_relation():
    model = MotorAudioWorldModel(hidden=8)
    state = model.initial_state(3, "cpu")
    pitch, state = model.step(torch.tensor([-.5, 0., .5]), state)
    assert pitch.shape == (3,) and state.shape == (3, 8)
    assert torch.all((pitch >= 0) & (pitch <= 1))


def test_deterministic_modules_connect_with_tempo_stretch_and_articulation():
    samples = torch.arange(FRAME) / SAMPLE_RATE
    frequency = 440.0 * 2 ** (1300.0 / 1200.0)
    frame = (.25 * torch.sin(2 * torch.pi * frequency * samples)).repeat(1, 80, 1)
    ear = DeterministicEar()(frame)
    assert ear.shape == (1, 80, 2)
    assert ear[..., 1].bool().all()
    memory = DeterministicTimelineMemory()
    stored = memory.remember(ear, torch.tensor([120.0]))
    stored.onset[:, 20] = True
    recalled = memory.recall(stored, tempo_scale=2, articulation_frames=4)
    assert recalled.pitch.shape == (1, 160)
    assert not recalled.voice[0, 40:44].any()
    assert recalled.bpm.item() == pytest.approx(60.0)


def test_fully_deterministic_reference_pipeline_runs_end_to_end():
    samples = torch.arange(FRAME) / SAMPLE_RATE
    frequency = 440.0 * 2 ** (1100.0 / 1200.0)
    frames = (.2 * torch.sin(2 * torch.pi * frequency * samples)).repeat(1, 30, 1)
    profile, position, result = DeterministicYamabikoPipeline()(frames)
    assert profile.pitch.shape == position.shape == (1, 60)
    assert result["pwm"].shape == result["pitch_cents"].shape == (1, 60)
    assert torch.isfinite(result["pitch_cents"]).all()


def test_neural_composite_connects_every_stage_without_true_plant_state_input():
    ear = tiny_model()
    beat_config = BeatGridConfig(input_dim=ear.config.audio_dim + 2, beat_hidden=8,
                                 memory_hidden=8, clock_hidden=8, aligner_hidden=8)
    composite = YamabikoComposite(
        ear, TempoBeatNet(beat_config), BeatTimelineRecallNet(beat_config, direct_pitch=True),
        TargetPositionPlanner(), MotorTrajectoryController(hidden=8),
        ErrorComparator(StagedConfig(comparator_hidden=8)),
        AcousticFeedbackResidual(hidden=8), tempo_scale=2, articulation_frames=4)
    reference = torch.randn(1, 20, FRAME) * .02
    profile, performances = composite(reference, torch.tensor([20]), repetitions=2)
    assert profile.target.shape == (1, 40, 2)
    assert profile.position.shape == (1, 40)
    assert profile.playback_bpm.shape == (1,)
    assert len(performances) == 2
    assert performances[0]["pwm"].shape == (1, 40)
    assert performances[1]["slow_context"].shape == (1, 8)
    restored = YamabikoComposite.from_checkpoint(composite.checkpoint())
    restored_profile = restored.listen(reference, torch.tensor([20]))
    torch.testing.assert_close(restored_profile.position, profile.position)
