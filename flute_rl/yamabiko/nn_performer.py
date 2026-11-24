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
from .features import PREV_LOOK, SONG_LOOKAHEAD, WINDOW
from .physical_plant import AudibleListener, DifferentiableMotorFlute

RIG_AUX = ("torque_gain", "max_velocity_strokes_s", "deadband", "flute_offset_cents", "tube_offset_m")
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
    # every actuator command from the network: it also opens and shuts the valve (a second output), it
    # hears the motor current (0 when an end switch has cut it), and it homes the plunger itself during the
    # rest before the first note instead of a fixed homing run
    valve: bool = False
    current: bool = False
    self_home: bool = False
    # step7: no input from the twin's physics (no open-loop position / velocity of its motor model, and no
    # ``positions``): the recurrent state has to keep track of where the plunger is from what it hears
    twin_free: bool = False
    # practice (iterative learning control): the record of the previous play of the same song (pitch error
    # where heard, heard or not, the PWM) is read PREV_LOOK steps ahead; the network learns to correct
    # what went wrong there last time.  The record is data outside the network, kept by the runtime.
    prev_play: bool = False
    # step7's rig memory: a small vector kept across songs on the same rig (not per song), written slowly
    # while playing from what was heard against the target; it carries the rig's quirks into the next song
    rig_memory: int = 0
    rig_rate: float = .02

    @property
    def features(self) -> int:
        return len(self.layout())

    def layout(self) -> list:
        """The names of the inputs, in order (checkpoints with another layout are mapped by name)."""
        names = [f"cents+{k}" for k in WINDOW] + [f"voice+{k}" for k in WINDOW] + ["heard", "valid", "gap", "prev_pwm"]
        if not self.twin_free:
            names += ["twin_position", "twin_velocity"]
        names.append("voice_now")
        if self.positions and not self.twin_free:
            names += ["aim_position", "heard_position"]
        names += [f"song{k}.{i}" for k in SONG_LOOKAHEAD for i in range(self.song_slot)]
        if self.prev_play:
            names += ([f"prev_err+{k}" for k in PREV_LOOK] + [f"prev_valid+{k}" for k in PREV_LOOK]
                      + [f"prev_pwm+{k}" for k in PREV_LOOK] + ["has_prev"])
        if self.current:
            names.append("current")
        names += [f"rig{i}" for i in range(self.rig_memory)]
        return names

    @property
    def outputs(self) -> int:
        return 2 if self.valve else 1


class NNTwinPerformer(nn.Module):
    def __init__(self, plant: DifferentiableMotorFlute, config: NNConfig = NNConfig()):
        super().__init__()
        self.config, self.plant = config, plant
        self.motor = EncoderlessDeterministicPerformer(plant).model     # the twin's motor model, open loop
        self.inp = nn.Linear(config.features, config.hidden)
        self.cell = nn.GRUCell(config.hidden, config.hidden)
        self.head = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.SiLU(), nn.Linear(config.hidden, config.outputs))
        if config.song_slot:
            self.writer = Writer(AdaptiveMemoryConfig(fast_hidden=config.hidden), config.song_slot, config.song_rate)
        if config.rig_memory:
            self.rigmem = Writer(AdaptiveMemoryConfig(fast_hidden=config.hidden), config.rig_memory, config.rig_rate)
            # training only (not exported): guess the rig from the memory, so the memory learns to hold it
            self.rig_aux = nn.Linear(config.rig_memory, len(RIG_AUX))

    # ----- one step --------------------------------------------------------------------------------
    def features(self, cents, voice, t, heard, valid, prev_pwm, estimate, twin=None, song=None, current=None,
                 rig=None):
        normalize = self.plant.config.normalize
        steps = cents.shape[1]
        ids = [min(t + k, steps - 1) for k in WINDOW]
        dtype = cents.dtype
        heard_n = torch.where(valid, normalize(heard), torch.zeros_like(heard))
        gap = torch.where(valid, normalize(cents[:, t]) - heard_n, torch.zeros_like(heard))
        now = [heard_n, valid.to(dtype), gap, prev_pwm]
        if not self.config.twin_free:
            now += [estimate.position, estimate.velocity]
        parts = [normalize(cents[:, ids]), voice[:, ids].to(dtype), torch.stack(now + [voice[:, t].to(dtype)], 1)]
        if self.config.positions and not self.config.twin_free:
            aim = self.plant.position_for_cents(cents[:, t], twin)
            at = torch.where(valid, self.plant.position_for_cents(heard, twin), torch.zeros_like(heard))
            parts.append(torch.stack([aim, at], 1))
        if self.config.song_slot:
            parts.append(torch.cat([song[min(t + k, steps - 1)] for k in SONG_LOOKAHEAD], 1))
        if self.config.prev_play:
            look = [min(t + k, steps - 1) for k in PREV_LOOK]
            parts += [song["err"][:, look], song["valid"][:, look], song["pwm"][:, look], song["has"][:, None]]
        if self.config.current:
            parts.append((prev_pwm.abs() if current is None else current)[:, None])
        if self.config.rig_memory:
            parts.append(rig if rig is not None else torch.zeros(cents.shape[0], self.config.rig_memory,
                                                                 device=cents.device, dtype=dtype))
        return torch.cat(parts, 1)

    @staticmethod
    def rig_answer(parameters, twin):
        """What the rig memory should be able to tell (training only): the rig against the twin, scaled to ~1."""
        log = lambda a, b: torch.log(a / b)
        return torch.stack([log(parameters.torque_gain, twin.torque_gain) / .25,
                            log(parameters.max_velocity_strokes_s, twin.max_velocity_strokes_s) / .25,
                            (parameters.deadband - twin.deadband) / .1,
                            (parameters.flute_offset_cents - twin.flute_offset_cents) / 30.0,
                            (parameters.tube_offset_m - twin.tube_offset_m) / .003], 1)

    def write_rig(self, rig, cents, voice, t, heard, valid, estimate, pwm, fast):
        """The rig memory learns from two steps ago (the heard pitch has caught up), only while it hears."""
        normalize = self.plant.config.normalize
        then = max(0, t - 2); dtype = cents.dtype
        position = torch.zeros_like(heard) if self.config.twin_free else estimate.position
        feats = torch.stack([torch.where(valid, normalize(heard), torch.zeros_like(heard)), valid.to(dtype),
                             normalize(cents[:, then]), voice[:, then].to(dtype), position, pwm], 1)
        return self.rigmem(feats, fast, rig, valid.to(dtype))

    def empty_song(self, batch, steps, device, dtype):
        if self.config.prev_play:                         # no previous play yet
            zero = torch.zeros(batch, steps, device=device, dtype=dtype)
            return {"err": zero, "valid": zero, "pwm": zero, "has": torch.zeros(batch, device=device, dtype=dtype)}
        return [torch.zeros(batch, self.config.song_slot, device=device, dtype=dtype) for _ in range(steps)]

    def write(self, song, cents, voice, t, heard, valid, estimate, pwm, fast):
        """Remember what happened two steps ago (the heard pitch has caught up by now)."""
        normalize = self.plant.config.normalize
        then = max(0, t - 2); dtype = cents.dtype
        feats = torch.stack([torch.where(valid, normalize(heard), torch.zeros_like(heard)), valid.to(dtype),
                             normalize(cents[:, then]), voice[:, then].to(dtype), estimate.position, pwm], 1)
        song[then] = self.writer(feats, fast, song[then], torch.ones_like(heard))

    def act(self, x, fast):
        pwm, _, fast = self.act_all(x, fast)
        return pwm, fast

    def act_all(self, x, fast):
        """(pwm, valve logit or None, new hidden state)."""
        fast = self.cell(torch.tanh(self.inp(x)), fast)
        out = self.head(fast)
        return torch.tanh(out[:, 0]), (out[:, 1] if self.config.valve else None), fast

    def start(self, batch, device, dtype, twin):
        estimate = self.motor.initial_state(batch, device, dtype)
        if not self.config.self_home:
            estimate.torque = -twin.torque_gain.to(dtype)                 # homing leaves it pressed home
        return estimate, torch.zeros(batch, self.config.hidden, device=device, dtype=dtype)

    # ----- closed loop on the simulator ------------------------------------------------------------
    def perform(self, plant, cents, voice, parameters, twin, generator=None, start_position=None, song=None,
                rig=None):
        """Play ``cents``/``voice`` (B, T) on the simulated rigs ``parameters``; ``twin`` is the fitted rig
        whose motor model gives the open-loop guess.  Returns pitch_cents (emitted) and pwm, (B, T), and the
        song memory after the play (pass it back as ``song`` to play the same song again)."""
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        listener = AudibleListener(plant.config)
        cfg = self.config
        state = plant.initial_state(batch, device, dtype)
        if start_position is not None:
            state.position = start_position
        elif cfg.self_home:                                               # wherever the last play left it
            state.position = torch.rand(batch, generator=generator, device=device, dtype=dtype)
        if not cfg.self_home:
            for _ in range(130):                                          # homing, as on the rig
                state = plant.step(state, -torch.ones(batch, device=device, dtype=dtype), parameters)
        current = torch.zeros(batch, device=device, dtype=dtype)
        delay = plant.config.valve_delay_steps
        valve_queue = [torch.zeros(batch, device=device, dtype=dtype) for _ in range(delay)]
        valve_soft = []
        heard_state = listener.initial_state(batch, device, dtype)
        estimate, fast = self.start(batch, device, dtype, twin)
        heard = torch.zeros(batch, device=device, dtype=dtype)
        valid = torch.zeros(batch, device=device, dtype=torch.bool)
        pwm = torch.zeros(batch, device=device, dtype=dtype)
        band = (twin.deadband if twin.deadband is not None else torch.zeros(batch, device=device)).to(dtype)
        if self.config.song_slot:
            song = list(song) if song is not None else self.empty_song(batch, steps, device, dtype)
        if cfg.prev_play:
            song = song if song is not None else self.empty_song(batch, steps, device, dtype)
            rec_err, rec_valid = [], []
        if cfg.rig_memory:                               # carried over from the songs before on this rig
            rig = rig if rig is not None else torch.zeros(batch, cfg.rig_memory, device=device, dtype=dtype)
        played, pwms = [], []
        for t in range(steps):
            x = self.features(cents, voice, t, heard, valid, pwm, estimate, twin, song, current, rig)
            pwm, logit, fast = self.act_all(x, fast)
            state = plant.step(state, pwm, parameters)
            estimate = self.motor.step(estimate, pwm, twin)
            # the motor current: what the command drives, nothing once an end switch has cut it
            at_end = ((state.position <= 0) & (pwm < 0)) | ((state.position >= 1) & (pwm > 0))
            current = torch.where(at_end, torch.zeros_like(pwm), pwm.abs())
            if cfg.valve:
                valve_queue.append(torch.sigmoid(logit))
                opened = valve_queue.pop(0)                               # the air follows a few steps late
                valve_soft.append(opened)
                valve_now = (opened >= .5).to(dtype)
            else:
                valve_now = voice[:, t].to(dtype)
            emitted, sounding = plant.flute(state, valve_now, parameters)
            heard, valid, heard_state = listener.step(emitted, sounding, parameters, heard_state, generator)
            if plant.config.motor_masking > 0:
                lost = torch.rand(valid.shape, generator=generator, device=device) < plant.config.motor_masking
                valid = valid & ~((pwm.abs() > band) & lost)
            if self.config.song_slot:
                self.write(song, cents, voice, t, heard, valid, estimate, pwm, fast)
            if cfg.rig_memory:
                rig = self.write_rig(rig, cents, voice, t, heard, valid, estimate, pwm, fast)
            if cfg.prev_play:
                normalize = plant.config.normalize
                rec_err.append(torch.where(valid, normalize(heard) - normalize(cents[:, t]), torch.zeros_like(heard)))
                rec_valid.append(valid.to(dtype))
            played.append(emitted); pwms.append(pwm)
        if cfg.prev_play:                                 # this play's record, for the next play (data, no grad)
            song = {"err": torch.stack(rec_err, 1).detach(), "valid": torch.stack(rec_valid, 1),
                    "pwm": torch.stack(pwms, 1).detach(), "has": torch.ones(batch, device=device, dtype=dtype)}
        result = {"pitch_cents": torch.stack(played, 1), "pwm": torch.stack(pwms, 1), "song": song, "rig": rig}
        if cfg.valve:
            result["valve"] = torch.stack(valve_soft, 1)                  # open probability the air follows
        return result

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
        song = self.empty_song(batch, steps, device, dtype) if (self.config.song_slot or self.config.prev_play) else None
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
        old = NNConfig(**checkpoint["config"]).layout()
        if old != config.layout()[:len(old)]:                             # inputs dropped or moved: map by name
            w = state["inp.weight"]
            moved = torch.zeros(w.shape[0], config.features, dtype=w.dtype, device=w.device)
            for j, name in enumerate(config.layout()):
                if name in old:
                    moved[:, j] = w[:, old.index(name)]
            state["inp.weight"] = moved
        last = f"head.{len(model.head) - 1}"
        if state[f"{last}.weight"].shape[0] < config.outputs:             # a new output (the valve) at zero
            w, b = state[f"{last}.weight"], state[f"{last}.bias"]
            state[f"{last}.weight"] = torch.cat([w, torch.zeros(config.outputs - w.shape[0], w.shape[1],
                                                                 dtype=w.dtype, device=w.device)])
            state[f"{last}.bias"] = torch.cat([b, torch.zeros(config.outputs - b.shape[0], dtype=b.dtype,
                                                              device=b.device)])
        w = state["inp.weight"]
        if w.shape[1] < config.features:
            grown = torch.zeros(w.shape[0], config.features, dtype=w.dtype, device=w.device)
            grown[:, :w.shape[1]] = w
            state["inp.weight"] = grown
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith(("writer.", "rigmem.", "rig_aux.")) for k in missing), (missing, unexpected)
        return model
