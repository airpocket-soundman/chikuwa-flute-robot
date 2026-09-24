"""Deterministic twin controller as the skeleton, a neural network on top.

The encoder-less deterministic controller holds a note within a cent (its
pitch observer corrects the position estimate from what is heard) but cannot
learn a song; the neural performer learns to anticipate leaps from repeating
a song but drifts on held notes.  This performer keeps the deterministic
skeleton, driven by the rig's digital twin, and lets a network add two
bounded residuals:

* a planning offset to the aimed stroke position (timing and anticipation),
* a PWM offset to the controller output.

Both read a time-indexed song memory (as in :mod:`adaptive_memory`), so the
performer can improve over repeats of a song.  The residual heads start at
zero, so before training it plays exactly like "deterministic + twin".  The
whole loop is differentiable and trains as one graph inside the twin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .adaptive_memory import SONG_LOOKAHEAD, Writer, AdaptiveMemoryConfig
from .deterministic_pipeline import EncoderlessDeterministicPerformer, period_ms
from .melodies import next_voiced
from .physical_plant import AudibleListener, DifferentiableMotorFlute
from .rig_adaptive import WINDOW, future_window, normalize


@dataclass(frozen=True)
class ResidualConfig:
    hidden: int = 64
    song_slot: int = 4
    song_rate: float = .5
    max_aim_offset: float = .08    # strokes
    max_pwm_offset: float = .3


class ResidualTwinPerformer(nn.Module):
    def __init__(self, plant: DifferentiableMotorFlute, config: ResidualConfig = ResidualConfig()):
        super().__init__()
        self.config, self.plant = config, plant
        self.skeleton = EncoderlessDeterministicPerformer(plant)
        song_features = len(SONG_LOOKAHEAD) * config.song_slot
        self.planner = nn.Sequential(nn.Linear(2 * len(WINDOW) + song_features + 2, config.hidden), nn.SiLU(),
                                     nn.Linear(config.hidden, config.hidden), nn.SiLU(), nn.Linear(config.hidden, 1))
        self.cell = nn.GRUCell(9 + song_features, config.hidden)
        self.head = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.SiLU(), nn.Linear(config.hidden, 1))
        self.writer = Writer(AdaptiveMemoryConfig(fast_hidden=config.hidden), config.song_slot, config.song_rate)
        for last in (self.planner[-1], self.head[-1]):  # start as the plain deterministic controller
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def initial_memory(self, batch, steps, device, dtype=torch.float32):
        return {"song": torch.zeros(batch, steps, self.config.song_slot, device=device, dtype=dtype)}

    @staticmethod
    def forget_song(memory):
        return None if memory is None else {"song": torch.zeros_like(memory["song"])}

    def perform(self, plant, cents, voice, parameters, memory=None, generator=None, twin=None,
                write=True, write_memory=None):
        """Play on the simulator ``parameters`` using ``twin`` (the fitted rig) as the skeleton's model."""
        if write_memory is not None:
            write = write_memory
        cfg, det = self.config, self.skeleton
        dcfg = det.config
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        twin = parameters if twin is None else twin
        memory = self.initial_memory(batch, steps, device, dtype) if memory is None else dict(memory)
        if memory["song"].shape[1] != steps:
            memory["song"] = torch.zeros(batch, steps, cfg.song_slot, device=device, dtype=dtype)
        coefficients = det.twin_memory(twin)
        a, b = coefficients.theta[:, 0], coefficients.theta[:, 1].clamp_min(.2)
        ratio = twin.max_velocity_strokes_s / plant.config.max_velocity_strokes_s
        aim = det._anticipate(next_voiced(cents, voice), voice, a, b, ratio)
        listener = AudibleListener(plant.config)
        state = plant.initial_state(batch, device, dtype)
        with torch.no_grad():
            state.position = torch.rand(batch, generator=generator, device=device, dtype=dtype) * .8
        for _ in range(dcfg.homing_steps):
            state = plant.step(state, -torch.ones(batch, device=device, dtype=dtype), parameters)
        estimate = det.model.initial_state(batch, device, dtype)
        estimate.torque = state.torque.clone()
        heard_state = listener.initial_state(batch, device, dtype)
        span = 6
        delay = twin.hearing_delay_steps.clamp(0, span - 1) if twin.hearing_delay_steps is not None else \
            torch.full((batch,), plant.config.hearing_delay_steps, device=device, dtype=torch.long)
        heard_index = (span - 1 - delay)[:, None]
        x_history = torch.zeros(batch, span, device=device, dtype=dtype)
        voice_history = torch.zeros(batch, span, device=device, dtype=torch.bool)
        band = (twin.deadband + .03).clamp(max=.6) if twin.deadband is not None else torch.full((batch,), .25, device=device)
        fast = torch.zeros(batch, cfg.hidden, device=device, dtype=dtype)
        heard = torch.zeros(batch, device=device, dtype=dtype)
        valid = torch.zeros(batch, device=device, dtype=torch.bool)
        allow = float(write)
        song = [memory["song"][:, t] for t in range(steps)]
        played, pwms = [], []
        for t in range(steps):
            read = torch.cat([song[min(t + k, steps - 1)] for k in SONG_LOOKAHEAD], 1)
            base_aim = ((a - period_ms(aim[:, t])) / b).clamp(0, 1)
            planner_in = torch.cat([future_window(cents, voice, t), read, base_aim[:, None],
                                    estimate.position[:, None]], 1)
            x_cmd = (base_aim + cfg.max_aim_offset * torch.tanh(self.planner(planner_in)[:, 0])).clamp(0, 1)
            u = (dcfg.kp * (x_cmd - estimate.position) - dcfg.kd * estimate.velocity).clamp(-1, 1)
            pwm_det = torch.sign(u) * (band + (1 - band) * u.abs()) * (u.abs() > .01).to(dtype)
            heard_n = torch.where(valid, normalize(heard), torch.zeros_like(heard))
            features = torch.stack([x_cmd, estimate.position, estimate.velocity, heard_n, valid.to(dtype),
                                    torch.where(valid, normalize(cents[:, t]) - heard_n, torch.zeros_like(heard)),
                                    voice[:, t].to(dtype), pwm_det, base_aim], 1)
            fast = self.cell(torch.cat([features, read], 1), fast)
            pwm = (pwm_det + cfg.max_pwm_offset * torch.tanh(self.head(fast)[:, 0])).clamp(-1, 1)
            state = plant.step(state, pwm, parameters)
            estimate = det.model.step(estimate, pwm, twin)
            emitted, sounding = plant.flute(state, voice[:, t].to(dtype), parameters)
            heard, valid, heard_state = listener.step(emitted, sounding, parameters, heard_state, generator)
            x_history = torch.cat([x_history[:, 1:], estimate.position[:, None]], 1)
            voice_history = torch.cat([voice_history[:, 1:], voice[:, t, None]], 1)
            usable = valid & voice_history.gather(1, heard_index)[:, 0]
            innovation = torch.where(usable, (a - period_ms(heard)) / b - x_history.gather(1, heard_index)[:, 0],
                                     torch.zeros_like(heard))
            shift = dcfg.observer_gain * innovation
            estimate.position = (estimate.position + shift).clamp(0, 1)
            estimate.velocity = estimate.velocity + dcfg.observer_velocity_gain * innovation / plant.config.dt
            x_history = (x_history + shift[:, None]).clamp(0, 1)
            then = max(0, t - 2)
            write_features = torch.stack([torch.where(valid, normalize(heard), torch.zeros_like(heard)),
                                          valid.to(dtype), normalize(cents[:, then]), voice[:, then].to(dtype),
                                          x_cmd, pwm], 1)
            song[then] = self.writer(write_features, fast, song[then],
                                     torch.full((batch,), allow, device=device, dtype=dtype))
            played.append(emitted); pwms.append(pwm)
        memory["song"] = torch.stack(song, 1)
        return {"pitch_cents": torch.stack(played, 1), "pwm": torch.stack(pwms, 1)}, memory

    def checkpoint(self, **metadata):
        return {"format": "yamabiko-residual-twin-v1", "config": asdict(self.config),
                "state_dict": self.state_dict(), **metadata}

    @classmethod
    def from_checkpoint(cls, checkpoint, plant, device="cpu"):
        model = cls(plant, ResidualConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
