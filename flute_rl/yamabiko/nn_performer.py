"""A performer whose network does the feedback too: no PD loop, no pitch observer.

:class:`~.residual_performer.ResidualTwinPerformer` keeps a hand-written skeleton (a PD loop on a position
estimate that a pitch observer corrects) and lets a network nudge it.  On the first real rig the skeleton
dithers: every small correction has to jump the static-friction deadband, so the plunger shakes around the
note, and the motor's noise then hides the flute.  Here one recurrent network maps, every 10 ms,

* the target track ahead (WINDOW), normalised by the rig's feature range,
* what it heard on the previous step (pitch, whether it heard anything, target minus heard),
* its own previous command,
* the twin's open-loop guess of position and velocity (the motor model driven by the commands only; it
  is never corrected from pitch: that correction is the network's job),

to the PWM directly.  It starts by imitating a residual performer (``imitate``) and is then trained
through the simulator on pitch error plus a smoothness cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .adaptive_memory import AdaptiveMemoryConfig, Writer
from .deterministic_pipeline import EncoderlessDeterministicPerformer
from .features import SONG_LOOKAHEAD, WINDOW
from .physical_plant import AudibleListener, DifferentiableMotorFlute

FEATURES = 2 * len(WINDOW) + 7     # window, heard (3), previous pwm, twin position / velocity, voice now


@dataclass(frozen=True)
class NNConfig:
    hidden: int = 64
    # also the twin's stroke position of the target now and of what was heard (closed-tube inverse), so the
    # feedback "heard here, aiming there" is a difference the network does not have to learn to compute
    positions: bool = False
    # a song memory (as the residual performer's): one slot per step of the song, written while playing
    # and read SONG_LOOKAHEAD steps ahead on the next play of the same song (0: none)
    song_slot: int = 0
    song_rate: float = .5

    @property
    def features(self) -> int:
        return FEATURES + (2 if self.positions else 0) + len(SONG_LOOKAHEAD) * self.song_slot


class NNTwinPerformer(nn.Module):
    def __init__(self, plant: DifferentiableMotorFlute, config: NNConfig = NNConfig()):
        super().__init__()
        self.config, self.plant = config, plant
        self.motor = EncoderlessDeterministicPerformer(plant).model     # the twin's motor model, open loop
        self.inp = nn.Linear(config.features, config.hidden)
        self.cell = nn.GRUCell(config.hidden, config.hidden)
        self.head = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.SiLU(), nn.Linear(config.hidden, 1))
        if config.song_slot:
            self.writer = Writer(AdaptiveMemoryConfig(fast_hidden=config.hidden), config.song_slot, config.song_rate)

    # ----- one step --------------------------------------------------------------------------------
    def features(self, cents, voice, t, heard, valid, prev_pwm, estimate, twin=None, song=None):
        normalize = self.plant.config.normalize
        steps = cents.shape[1]
        ids = [min(t + k, steps - 1) for k in WINDOW]
        dtype = cents.dtype
        heard_n = torch.where(valid, normalize(heard), torch.zeros_like(heard))
        gap = torch.where(valid, normalize(cents[:, t]) - heard_n, torch.zeros_like(heard))
        parts = [normalize(cents[:, ids]), voice[:, ids].to(dtype),
                 torch.stack([heard_n, valid.to(dtype), gap, prev_pwm, estimate.position,
                              estimate.velocity, voice[:, t].to(dtype)], 1)]
        if self.config.positions:
            aim = self.plant.position_for_cents(cents[:, t], twin)
            at = torch.where(valid, self.plant.position_for_cents(heard, twin), torch.zeros_like(heard))
            parts.append(torch.stack([aim, at], 1))
        if self.config.song_slot:
            parts.append(torch.cat([song[min(t + k, steps - 1)] for k in SONG_LOOKAHEAD], 1))
        return torch.cat(parts, 1)

    def empty_song(self, batch, steps, device, dtype):
        return [torch.zeros(batch, self.config.song_slot, device=device, dtype=dtype) for _ in range(steps)]

    def write(self, song, cents, voice, t, heard, valid, estimate, pwm, fast):
        """Remember what happened two steps ago (the heard pitch has caught up by now)."""
        normalize = self.plant.config.normalize
        then = max(0, t - 2); dtype = cents.dtype
        feats = torch.stack([torch.where(valid, normalize(heard), torch.zeros_like(heard)), valid.to(dtype),
                             normalize(cents[:, then]), voice[:, then].to(dtype), estimate.position, pwm], 1)
        song[then] = self.writer(feats, fast, song[then], torch.ones_like(heard))

    def act(self, x, fast):
        fast = self.cell(torch.tanh(self.inp(x)), fast)
        return torch.tanh(self.head(fast)[:, 0]), fast

    def start(self, batch, device, dtype, twin):
        estimate = self.motor.initial_state(batch, device, dtype)
        estimate.torque = -twin.torque_gain.to(dtype)                     # homing leaves it pressed home
        return estimate, torch.zeros(batch, self.config.hidden, device=device, dtype=dtype)

    # ----- closed loop on the simulator ------------------------------------------------------------
    def perform(self, plant, cents, voice, parameters, twin, generator=None, start_position=None, song=None):
        """Play ``cents``/``voice`` (B, T) on the simulated rigs ``parameters``; ``twin`` is the fitted rig
        whose motor model gives the open-loop guess.  Returns pitch_cents (emitted) and pwm, (B, T), and the
        song memory after the play (pass it back as ``song`` to play the same song again)."""
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        listener = AudibleListener(plant.config)
        state = plant.initial_state(batch, device, dtype)
        if start_position is not None:
            state.position = start_position
        for _ in range(130):                                              # homing, as on the rig
            state = plant.step(state, -torch.ones(batch, device=device, dtype=dtype), parameters)
        heard_state = listener.initial_state(batch, device, dtype)
        estimate, fast = self.start(batch, device, dtype, twin)
        heard = torch.zeros(batch, device=device, dtype=dtype)
        valid = torch.zeros(batch, device=device, dtype=torch.bool)
        pwm = torch.zeros(batch, device=device, dtype=dtype)
        band = (twin.deadband if twin.deadband is not None else torch.zeros(batch, device=device)).to(dtype)
        if self.config.song_slot:
            song = list(song) if song is not None else self.empty_song(batch, steps, device, dtype)
        played, pwms = [], []
        for t in range(steps):
            x = self.features(cents, voice, t, heard, valid, pwm, estimate, twin, song)
            pwm, fast = self.act(x, fast)
            state = plant.step(state, pwm, parameters)
            estimate = self.motor.step(estimate, pwm, twin)
            emitted, sounding = plant.flute(state, voice[:, t].to(dtype), parameters)
            heard, valid, heard_state = listener.step(emitted, sounding, parameters, heard_state, generator)
            if plant.config.motor_masking > 0:
                lost = torch.rand(valid.shape, generator=generator, device=device) < plant.config.motor_masking
                valid = valid & ~((pwm.abs() > band) & lost)
            if self.config.song_slot:
                self.write(song, cents, voice, t, heard, valid, estimate, pwm, fast)
            played.append(emitted); pwms.append(pwm)
        return {"pitch_cents": torch.stack(played, 1), "pwm": torch.stack(pwms, 1), "song": song}

    # ----- imitation of a recorded play ------------------------------------------------------------
    def imitate(self, cents, voice, heard, valid, pwm, twin):
        """The network's commands along a recorded closed-loop play (teacher forcing): it sees what the
        teacher saw and did, step by step.  heard/valid[t] is what was heard after the command of step t.
        Returns the predicted PWM, (B, T)."""
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        estimate, fast = self.start(batch, device, dtype, twin)
        h = torch.zeros(batch, device=device, dtype=dtype)
        v = torch.zeros(batch, device=device, dtype=torch.bool)
        prev = torch.zeros(batch, device=device, dtype=dtype)
        song = self.empty_song(batch, steps, device, dtype) if self.config.song_slot else None
        out = []
        for t in range(steps):
            x = self.features(cents, voice, t, h, v, prev, estimate, twin, song)
            u, fast = self.act(x, fast)
            out.append(u)
            prev = pwm[:, t]
            estimate = self.motor.step(estimate, prev, twin)
            h, v = heard[:, t], valid[:, t]
        return torch.stack(out, 1)

    def checkpoint(self, **metadata):
        return {"format": "yamabiko-nn-twin-v1", "config": asdict(self.config), "state_dict": self.state_dict(),
                **metadata}

    @classmethod
    def from_checkpoint(cls, checkpoint, plant, device="cpu", **upgrade):
        """Load a checkpoint; ``upgrade`` (e.g. song_slot=4) adds inputs: their weights start at zero, so the
        upgraded network plays exactly as the loaded one until it learns to use them."""
        config = NNConfig(**{**checkpoint["config"], **upgrade})
        model = cls(plant, config).to(device)
        state = dict(checkpoint["state_dict"])
        w = state["inp.weight"]
        if w.shape[1] < config.features:
            grown = torch.zeros(w.shape[0], config.features, dtype=w.dtype, device=w.device)
            grown[:, :w.shape[1]] = w
            state["inp.weight"] = grown
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("writer.") for k in missing), (missing, unexpected)
        return model
