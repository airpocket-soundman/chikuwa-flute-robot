"""Deterministic reference modules for every Yamabiko inference boundary.

These modules are deliberately non-learned.  They share semantic contracts
with the neural stages and provide debuggable baselines, oracle tests, and a
way to replace one gate at a time without changing the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig


@dataclass
class DeterministicProfile:
    pitch: torch.Tensor       # normalized pitch, (B,T)
    voice: torch.Tensor       # boolean, (B,T)
    onset: torch.Tensor       # boolean, (B,T)
    bpm: torch.Tensor         # (B,)


class DeterministicEar(nn.Module):
    """Windowed-FFT pitch/voice baseline for 16 kHz, 320-sample frames."""

    def __init__(self, sample_rate=16_000, pitch_center_cents=1300.0, pitch_scale_cents=600.0,
                 rms_threshold=.008):
        super().__init__(); self.sample_rate = sample_rate
        self.pitch_center_cents, self.pitch_scale_cents = pitch_center_cents, pitch_scale_cents
        self.rms_threshold = rms_threshold

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(frames.shape[-1], device=frames.device, dtype=frames.dtype)
        spectrum = torch.fft.rfft(frames * window).abs()
        frequencies = torch.fft.rfftfreq(frames.shape[-1], 1 / self.sample_rate).to(frames.device)
        useful = (frequencies >= 500) & (frequencies <= 1600)
        local = spectrum[..., useful]
        frequency = frequencies[useful][local.argmax(-1)]
        cents = 1200.0 * torch.log2(frequency.clamp_min(1.0) / 440.0)
        pitch = ((cents - self.pitch_center_cents) / self.pitch_scale_cents).clamp(-1, 1)
        rms = frames.square().mean(-1).sqrt()
        voice = rms >= self.rms_threshold
        return torch.stack([pitch, voice.to(frames.dtype)], -1)


class DeterministicTempoBeat(nn.Module):
    """Autocorrelation tempo baseline over pitch/voice event strength."""

    def __init__(self, frame_rate=100.0, minimum_bpm=40, maximum_bpm=240):
        super().__init__(); self.frame_rate = frame_rate
        self.minimum_bpm, self.maximum_bpm = minimum_bpm, maximum_bpm

    def forward(self, ear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        voice = ear[..., 1]
        pitch_change = torch.nn.functional.pad((ear[:, 1:, 0] - ear[:, :-1, 0]).abs(), (1, 0))
        onset = torch.nn.functional.pad((voice[:, 1:] - voice[:, :-1]).clamp_min(0), (1, 0))
        strength = onset + pitch_change
        min_lag = max(1, round(self.frame_rate * 60 / self.maximum_bpm))
        max_lag = min(strength.shape[1] - 1, round(self.frame_rate * 60 / self.minimum_bpm))
        scores = []
        for lag in range(min_lag, max_lag + 1):
            scores.append((strength[:, lag:] * strength[:, :-lag]).mean(1))
        score = torch.stack(scores, 1)
        lag = score.argmax(1) + min_lag
        bpm = self.frame_rate * 60.0 / lag.to(ear.dtype)
        time = torch.arange(ear.shape[1], device=ear.device, dtype=ear.dtype)[None]
        phase = torch.remainder(time * bpm[:, None] / (60 * self.frame_rate), 1.0)
        return bpm, phase


class DeterministicTimelineMemory(nn.Module):
    """Exact semantic memory with deterministic tempo scaling/articulation."""

    def remember(self, ear: torch.Tensor, bpm: torch.Tensor) -> DeterministicProfile:
        voice = ear[..., 1] >= .5
        changed = torch.nn.functional.pad((ear[:, 1:, 0] - ear[:, :-1, 0]).abs() > .02, (1, 0))
        onset = torch.nn.functional.pad((voice[:, 1:] & ~voice[:, :-1]), (1, 0)) | (changed & voice)
        return DeterministicProfile(ear[..., 0].clone(), voice.clone(), onset, bpm.clone())

    def recall(self, profile: DeterministicProfile, tempo_scale=2, articulation_frames=4) -> DeterministicProfile:
        if int(tempo_scale) != tempo_scale or tempo_scale < 1:
            raise ValueError("deterministic baseline currently requires an integer tempo_scale >= 1")
        scale = int(tempo_scale)
        pitch = profile.pitch.repeat_interleave(scale, 1)
        voice = profile.voice.repeat_interleave(scale, 1)
        onset = profile.onset.repeat_interleave(scale, 1)
        for row in range(len(voice)):
            starts = torch.nonzero(onset[row], as_tuple=False)[:, 0]
            for start in starts.tolist():
                voice[row, start:min(len(voice[row]), start + articulation_frames)] = False
        return DeterministicProfile(pitch, voice, onset, profile.bpm / scale)


class DeterministicPositionPlanner(nn.Module):
    """Nominal flute inverse from normalized musical pitch to stroke fraction.

    The linear flute maps pitch straight to stroke.  The closed tube inverts
    f = c / 4(L - x) with the nominal rig; per-rig intercept/slope errors are
    left to feedback.
    """

    def __init__(self, plant: DifferentiableMotorFlute | None = None,
                 pitch_center_cents=1300.0, pitch_scale_cents=600.0):
        super().__init__(); self.plant = plant
        self.pitch_center_cents, self.pitch_scale_cents = pitch_center_cents, pitch_scale_cents

    def forward(self, profile: DeterministicProfile) -> torch.Tensor:
        if self.plant is None or self.plant.config.flute_model == "linear":
            return ((profile.pitch + 1.0) * .5).clamp(0, 1)
        pitch = profile.pitch
        nominal = self.plant.parameters(pitch.shape[0], pitch.device, pitch.dtype)
        nominal = type(nominal)(*(value[:, None] if value is not None else None
                                  for value in vars(nominal).values()))
        cents = self.pitch_center_cents + self.pitch_scale_cents * pitch
        return self.plant.position_for_cents(cents, nominal)


class DeterministicMotorController(nn.Module):
    """Explicit PD reference controller for simulator/oracle diagnostics."""

    def __init__(self, kp=6.0, kd=1.0):
        super().__init__(); self.kp, self.kd = kp, kd

    def forward(self, target_position, actual_position, actual_velocity):
        return (self.kp * (target_position - actual_position) - self.kd * actual_velocity).clamp(-1, 1)


class DeterministicComparator(nn.Module):
    def forward(self, target_pitch, heard_pitch, valid=None):
        error = target_pitch - heard_pitch
        return error if valid is None else torch.where(valid, error, torch.zeros_like(error))


@dataclass
class PIDState:
    integral: torch.Tensor
    previous_error: torch.Tensor


class DeterministicFeedback(nn.Module):
    """Bounded PID residual baseline using only audible signed pitch error."""

    def __init__(self, kp=2.0, ki=.08, kd=.25, limit=.45):
        super().__init__(); self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit

    def initial_state(self, batch, device, dtype=torch.float32):
        zero = torch.zeros(batch, device=device, dtype=dtype)
        return PIDState(zero, zero.clone())

    def step(self, error, valid, state: PIDState):
        error = torch.where(valid, error, torch.zeros_like(error))
        integral = (state.integral + error).clamp(-2, 2)
        derivative = error - state.previous_error
        residual = (self.kp * error + self.ki * integral + self.kd * derivative).clamp(-self.limit, self.limit)
        return residual, PIDState(integral, error)


class DeterministicYamabikoPipeline(nn.Module):
    """Fully connected, non-learned oracle/reference implementation.

    The PD controller intentionally reads simulator position/velocity, so this
    class is a diagnostic upper baseline rather than the encoder-less deployed
    path.  Every other stage uses observable audio/semantic values.
    """

    def __init__(self, config: PhysicalPlantConfig = PhysicalPlantConfig(), tempo_scale=2,
                 articulation_frames=4):
        super().__init__(); self.ear = DeterministicEar()
        self.tempo = DeterministicTempoBeat(); self.memory = DeterministicTimelineMemory()
        self.plant = DifferentiableMotorFlute(config)
        self.planner = DeterministicPositionPlanner(self.plant); self.controller = DeterministicMotorController()
        self.comparator = DeterministicComparator(); self.feedback = DeterministicFeedback(limit=config.feedback_limit)
        self.tempo_scale, self.articulation_frames = tempo_scale, articulation_frames

    def listen(self, reference_frames: torch.Tensor):
        ear = self.ear(reference_frames); bpm, _ = self.tempo(ear)
        stored = self.memory.remember(ear, bpm)
        profile = self.memory.recall(stored, self.tempo_scale, self.articulation_frames)
        return profile, self.planner(profile)

    def perform(self, profile: DeterministicProfile, position_plan: torch.Tensor, parameters=None):
        batch, steps = position_plan.shape; device, dtype = position_plan.device, position_plan.dtype
        parameters = parameters or self.plant.parameters(batch, device, dtype)
        physical = self.plant.initial_state(batch, device, dtype)
        feedback_state = self.feedback.initial_state(batch, device, dtype)
        outputs = {key: [] for key in ("pwm", "position", "pitch_cents", "error")}
        for t in range(steps):
            cents, sounding = self.plant.flute(physical, profile.voice[:, t].to(dtype), parameters)
            heard = ((cents - self.plant.config.low_cents) /
                     self.plant.config.pitch_span_cents).clamp(0, 1)
            target = ((profile.pitch[:, t] + 1.0) * .5).clamp(0, 1)
            error = self.comparator(target, heard, sounding)
            residual, feedback_state = self.feedback.step(error, sounding, feedback_state)
            base = self.controller(position_plan[:, t], physical.position, physical.velocity)
            pwm = (base + residual).clamp(-1, 1)
            physical = self.plant.step(physical, pwm, parameters)
            emitted, _ = self.plant.flute(physical, profile.voice[:, t].to(dtype), parameters)
            for key, value in (("pwm", pwm), ("position", physical.position),
                               ("pitch_cents", emitted), ("error", error)):
                outputs[key].append(value)
        return {key: torch.stack(value, 1) for key, value in outputs.items()}

    def forward(self, reference_frames: torch.Tensor, parameters=None):
        profile, position = self.listen(reference_frames)
        return profile, position, self.perform(profile, position, parameters)


def period_ms(cents: torch.Tensor) -> torch.Tensor:
    return 1000.0 / (440.0 * torch.pow(2.0, cents / 1200.0))


@dataclass
class RigMemory:
    """Per-rig flute coefficients in period space: T[ms] = a - b * x_model.

    ``a`` (intercept) is the period at the home end stop and ``b`` (slope) the
    period change per stroke *as the robot's own motor model counts stroke*,
    so it also absorbs a proportional motor-speed error.  Kept across plays.
    """
    theta: torch.Tensor      # (B, 2) = [a, b]
    variance: torch.Tensor   # (B, 2) estimate variances for [a, b]


@dataclass
class EncoderlessConfig:
    homing_steps: int = 130          # full stroke at the slowest randomized motor
    # With the pitch observer a and b cancel while sounding, and even the true
    # coefficients gain <2 cents in the audit; the online estimate costs more
    # than it gains, so coefficient learning is opt-in.
    learn_coefficients: bool = False
    lead_steps: int = 0              # start moving toward a note this many steps early
    # Distance-aware anticipation (Planner timing): start each move
    # lag + distance / speed before the note, keeping at least `hold_fraction`
    # of the previous note.  speed is in strokes/s; None disables it.
    anticipation_speed: float | None = 1.6
    anticipation_lag_steps: float = 6.0
    hold_fraction: float = .5
    note_threshold_cents: float = 50.0
    kp: float = 12.0                 # PD on the estimated position
    kd: float = 2.0
    deadband_compensation: float | None = .25  # PWM offset; None = the plant's nominal deadband
    ki: float = .0                   # integral of the position error (static friction)
    integral_limit: float = .5
    observer_gain: float = .6        # heard-pitch correction of the position estimate (alpha)
    observer_velocity_gain: float = .2  # ... and of its velocity (beta, per step)
    settled_period_ms: float = .004  # |dT| per step below which a heard frame is settled
    settled_velocity: float = .05
    slope_min_move: float = .05      # dead-reckoned move needed to observe the slope
    observation_variance: tuple = (.0004, .004)
    process_variance: tuple = (1e-6, 1e-6)
    prior_variance: tuple = (.004, .02)


class EncoderlessDeterministicPerformer(nn.Module):
    """Pitch-as-position-sensor control without an encoder.

    Deterministic counterpart of the planned neural Planner/Controller/Feedback:

    * Planner: target cents -> period -> x = (a - T) / b with the remembered
      coefficients, aiming at the next note during rests (pre-positioning).
    * Controller: PD on an estimated position.  The estimate is a nominal motor
      model driven by the applied PWM (dead reckoning, with nominal deadband
      compensation) and reset by homing to the end stop before each play.
    * Feedback: while sounding, the heard period is converted back to a
      position with the same (a, b) and corrects the estimate (observer).  In
      that loop a and b cancel, so the pitch itself is servoed.  The
      coefficients matter for silent moves (slope) and the first note after
      homing (intercept); both are estimated from settled heard notes and kept.

    It never reads the plant's position, velocity or parameters.
    """

    def __init__(self, plant: DifferentiableMotorFlute, config: EncoderlessConfig = EncoderlessConfig()):
        super().__init__()
        from .physical_plant import AudibleListener
        self.plant, self.config = plant, config
        self.listener = AudibleListener(plant.config)
        self.model = DifferentiableMotorFlute(plant.config)  # nominal parameters only
        self.learn = config.learn_coefficients

    def nominal_memory(self, batch, device, dtype=torch.float32) -> RigMemory:
        cfg = self.plant.config
        zero = torch.zeros(batch, device=device, dtype=dtype)
        if cfg.flute_model == "linear":  # secant of the linear flute in period space
            a = period_ms(zero + cfg.low_cents)
            b = a - period_ms(zero + cfg.high_cents)
        else:
            nominal = self.plant.parameters(batch, device, dtype)
            a = self.plant.period_s(zero, nominal) * 1000.0
            b = a - self.plant.period_s(zero + 1.0, nominal) * 1000.0
        variance = torch.tensor(self.config.prior_variance, device=device, dtype=dtype).expand(batch, 2).clone()
        return RigMemory(torch.stack([a, b], 1), variance)

    def _anticipate(self, aim, voice, a, b, motor_ratio=None):
        """Switch the aim to each next note early enough to arrive on time.

        Notes are segmented first: a new note starts where the target moves
        more than ``note_threshold_cents`` from the current note's first value.
        A heard (neural) target wobbles a few cents every frame and glides
        through leaps; without segmentation every wobble would look like a new
        note and the anticipation would be lost.  The early part aims at the
        next note's median.
        """
        cfg = self.config
        ratio = torch.ones(aim.shape[0], device=aim.device) if motor_ratio is None else motor_ratio
        result = aim.clone()
        for row in range(aim.shape[0]):
            values = aim[row]
            starts, anchor = [0], values[0].item()
            for t in range(1, len(values)):
                if abs(values[t].item() - anchor) > cfg.note_threshold_cents:
                    starts.append(t); anchor = values[t].item()
            bounds = starts + [len(values)]
            medians = [values[bounds[k]:bounds[k + 1]].median() for k in range(len(starts))]
            positions = [((a[row] - period_ms(m)) / b[row].clamp_min(.2)).clamp(0, 1).item() for m in medians]
            for k in range(1, len(starts)):
                onset = starts[k]
                distance = abs(positions[k] - positions[k - 1])
                steps_per_stroke = 1.0 / (cfg.anticipation_speed * ratio[row].item() * self.plant.config.dt)
                lead = int(round(cfg.anticipation_lag_steps + distance * steps_per_stroke))
                earliest = starts[k - 1] + int(cfg.hold_fraction * (onset - starts[k - 1]))
                start = max(earliest, onset - lead)
                result[row, start:onset] = medians[k]
        return result

    @staticmethod
    def _kalman(value, variance, observation, observation_variance, use):
        gain = variance / (variance + observation_variance)
        return (torch.where(use, value + gain * (observation - value), value),
                torch.where(use, (1 - gain) * variance, variance))

    def measure_motor(self, parameters, generator: torch.Generator | None = None, steps: int = 70,
                      skip: int = 20):
        """Learning mode: estimate this rig's motor speed relative to the nominal model.

        Home, then drive out at full PWM with the valve open, convert the heard
        period back to position with the nominal flute, and fit the slope of
        the middle of the run.  Only heard pitch is used.
        """
        cfg = self.config
        batch = parameters.torque_gain.shape[0]; device = parameters.torque_gain.device
        dtype = parameters.torque_gain.dtype
        memory = self.nominal_memory(batch, device, dtype)
        a, b = memory.theta[:, 0], memory.theta[:, 1]
        state = self.plant.initial_state(batch, device, dtype)
        for _ in range(cfg.homing_steps):
            state = self.plant.step(state, -torch.ones(batch, device=device, dtype=dtype), parameters)
        listener = self.listener.initial_state(batch, device, dtype)
        nominal = self.model.parameters(batch, device, dtype)
        model_state = self.model.initial_state(batch, device, dtype)
        xs, oks, model_xs = [], [], []
        delay = self.plant.config.hearing_delay_steps + 1
        for _ in range(steps):
            state = self.plant.step(state, torch.ones(batch, device=device, dtype=dtype), parameters)
            model_state = self.model.step(model_state, torch.ones(batch, device=device, dtype=dtype), nominal)
            emitted, sounding = self.plant.flute(state, torch.ones(batch, device=device, dtype=dtype), parameters)
            heard, valid, listener = self.listener.step(emitted, sounding, parameters, listener, generator)
            x = (a - period_ms(heard)) / b
            xs.append(x); oks.append(valid & (x > .02) & (x < .9)); model_xs.append(model_state.position)
        # Heard pitch is `delay` steps late; skip the acceleration phase.
        x = torch.stack(xs, 1)[:, delay + skip:]; ok = torch.stack(oks, 1)[:, delay + skip:].to(dtype)
        model_x = torch.stack(model_xs, 1)[:, skip:skip + x.shape[1]]
        model_ok = ok * (model_x < .9).to(dtype)

        def slope(values, weights):
            t = torch.arange(values.shape[1], device=device, dtype=dtype)[None].expand_as(values)
            total = weights.sum(1).clamp_min(1)
            mt, mv = (t * weights).sum(1) / total, (values * weights).sum(1) / total
            return (((t - mt[:, None]) * (values - mv[:, None]) * weights).sum(1) /
                    (((t - mt[:, None]) ** 2 * weights).sum(1).clamp_min(1e-6)))

        # Same frames, same commands: the ratio cancels the shared acceleration phase.
        return (slope(x, ok) / slope(model_x, model_ok).clamp_min(1e-4)).clamp(.2, 5.0)

    def perform(self, cents, voice, parameters, memory: RigMemory | None = None,
                generator: torch.Generator | None = None, motor_ratio: torch.Tensor | None = None):
        from .melodies import next_voiced
        cfg = self.config
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        memory = memory or self.nominal_memory(batch, device, dtype)
        a, b = memory.theta[:, 0].clone(), memory.theta[:, 1].clone()
        var_a = memory.variance[:, 0] + cfg.process_variance[0]
        var_b = memory.variance[:, 1] + cfg.process_variance[1]
        nominal = self.model.parameters(batch, device, dtype)
        if motor_ratio is not None:  # measured in learning mode: scale the internal motor model
            nominal.torque_gain = nominal.torque_gain * motor_ratio
            nominal.max_velocity_strokes_s = nominal.max_velocity_strokes_s * motor_ratio
        plant_state = self.plant.initial_state(batch, device, dtype)
        # Start somewhere unknown inside the stroke, then home to the low end stop.
        plant_state.position = torch.rand(batch, generator=generator, device=device, dtype=dtype) * .8
        for _ in range(cfg.homing_steps):
            plant_state = self.plant.step(plant_state, -torch.ones(batch, device=device, dtype=dtype), parameters)
        estimate = self.model.initial_state(batch, device, dtype)
        estimate.torque = plant_state.torque.clone()  # both pressed against the stop
        listener = self.listener.initial_state(batch, device, dtype)
        aim = next_voiced(cents, voice)
        if cfg.anticipation_speed is not None:
            aim = self._anticipate(aim, voice, a, b, motor_ratio)
        if cfg.lead_steps > 0:
            aim = torch.cat([aim[:, cfg.lead_steps:], aim[:, -1:].expand(-1, cfg.lead_steps)], 1)
        delay = self.plant.config.hearing_delay_steps + 1
        x_history = torch.zeros(batch, delay, device=device, dtype=dtype)
        voice_history = torch.zeros(batch, delay, device=device, dtype=torch.bool)
        dead_reckoned = torch.zeros(batch, device=device, dtype=dtype)  # since homing, uncorrected
        anchor_period = torch.zeros(batch, device=device, dtype=dtype)
        anchor_reckoned = torch.zeros(batch, device=device, dtype=dtype)
        has_anchor = torch.zeros(batch, device=device, dtype=torch.bool)
        first_note = torch.ones(batch, device=device, dtype=torch.bool)
        previous_period = torch.zeros(batch, device=device, dtype=dtype)
        band = (self.plant.config.deadband if cfg.deadband_compensation is None
                else cfg.deadband_compensation)
        integral = torch.zeros(batch, device=device, dtype=dtype)
        heard_recently = torch.zeros(batch, device=device, dtype=torch.bool)
        logs = {key: [] for key in ("pitch_cents", "heard", "pwm", "command", "estimate")}
        for t in range(steps):
            x_cmd = ((a - period_ms(aim[:, t])) / b.clamp_min(.2)).clamp(0, 1)
            position_error = x_cmd - estimate.position
            integral = torch.where(heard_recently, integral + cfg.ki * position_error, .9 * integral)
            integral = integral.clamp(-cfg.integral_limit, cfg.integral_limit)
            u = (cfg.kp * position_error - cfg.kd * estimate.velocity + integral).clamp(-1, 1)
            pwm = torch.where(u.abs() > .01, torch.sign(u) * (band + (1 - band) * u.abs()), torch.zeros_like(u))
            plant_state = self.plant.step(plant_state, pwm, parameters)
            before = estimate.position
            estimate = self.model.step(estimate, pwm, nominal)
            dead_reckoned = dead_reckoned + (estimate.position - before)
            emitted, sounding = self.plant.flute(plant_state, voice[:, t].to(dtype), parameters)
            heard, valid, listener = self.listener.step(emitted, sounding, parameters, listener, generator)
            x_history = torch.cat([x_history[:, 1:], estimate.position[:, None]], 1)
            voice_history = torch.cat([voice_history[:, 1:], voice[:, t, None]], 1)
            period = period_ms(heard)
            usable = valid & voice_history[:, 0]
            heard_recently = usable
            # Observer: the heard pitch left the flute `delay` steps ago.
            innovation = torch.where(usable, (a - period) / b.clamp_min(.2) - x_history[:, 0],
                                     torch.zeros_like(heard))
            shift = cfg.observer_gain * innovation
            estimate.position = (estimate.position + shift).clamp(0, 1)
            estimate.velocity = estimate.velocity + cfg.observer_velocity_gain * innovation / self.plant.config.dt
            x_history = (x_history + shift[:, None]).clamp(0, 1)
            if self.learn:
                settled = (usable & ((period - previous_period).abs() < cfg.settled_period_ms) &
                           (estimate.velocity.abs() < cfg.settled_velocity))
                # Intercept: the first settled note after homing, a = T + b * x.
                first = settled & first_note
                a, var_a = self._kalman(a, var_a, period + b * dead_reckoned, cfg.observation_variance[0], first)
                first_note = first_note & ~first
                # Slope: settled-to-settled period change over a dead-reckoned move.
                moved = dead_reckoned - anchor_reckoned
                observe = settled & has_anchor & (moved.abs() > cfg.slope_min_move)
                slope = -(period - anchor_period) / torch.where(observe, moved, torch.ones_like(moved))
                b, var_b = self._kalman(b, var_b, slope,
                                        cfg.observation_variance[1] / moved.abs().clamp_min(.05), observe)
                anchor_period = torch.where(settled, period, anchor_period)
                anchor_reckoned = torch.where(settled, dead_reckoned, anchor_reckoned)
                has_anchor = has_anchor | settled
            previous_period = torch.where(valid, period, previous_period)
            for key, value in (("pitch_cents", emitted), ("heard", heard), ("pwm", pwm),
                               ("command", x_cmd), ("estimate", estimate.position)):
                logs[key].append(value)
        result = {key: torch.stack(value, 1) for key, value in logs.items()}
        return result, RigMemory(torch.stack([a, b], 1), torch.stack([var_a, var_b], 1))
