"""Planner / Controller / Feedback with three separately gated memories.

The rig-adaptive performer (:mod:`rig_adaptive`) keeps one slow memory in
which motor, flute and song experience mix.  This model splits it by how long
each kind of knowledge stays true:

* ``motor``: a vector, forgotten only when the motor is swapped;
* ``flute``: a vector, forgotten only when the flute is swapped;
* ``song``: a time-indexed trace (one small slot per 10 ms step of the
  song), forgotten on a new song and kept across repeats of the same song.
  Step ``t`` writes slot ``t``; later plays read slots ``t .. t+SONG_LOOKAHEAD``,
  the neural analogue of iterative learning control.

Each memory can be frozen (``write`` flags), so the three adaptations can be
measured one at a time.  Which information goes where is not supervised: a
part's memory is wiped exactly when that part changes, so only knowledge
that survives the other parts' swaps pays off in its own memory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .physical_plant import DifferentiableMotorFlute
from .rig_adaptive import WINDOW, future_window, normalize

PARTS = ("motor", "flute", "song")
SONG_LOOKAHEAD = (0, 5, 10, 20)


@dataclass(frozen=True)
class AdaptiveMemoryConfig:
    fast_hidden: int = 96
    planner_hidden: int = 96
    rig_hidden: int = 16          # motor and flute memories
    song_slot: int = 4            # song memory per 10 ms step
    rig_rate: float = .02         # max fraction of a rig memory renewed per step
    song_rate: float = .5         # max fraction of a song slot renewed when written
    homing_steps: int = 130
    deadband_compensation: float = .25

    @property
    def memory_features(self) -> int:
        return 2 * self.rig_hidden + len(SONG_LOOKAHEAD) * self.song_slot


class Planner(nn.Module):
    def __init__(self, cfg: AdaptiveMemoryConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * len(WINDOW) + cfg.memory_features, cfg.planner_hidden), nn.SiLU(),
                                 nn.Linear(cfg.planner_hidden, cfg.planner_hidden), nn.SiLU(),
                                 nn.Linear(cfg.planner_hidden, 1))

    def forward(self, window, memory):
        return torch.sigmoid(self.net(torch.cat([window, memory], 1)))[:, 0]


class Core(nn.Module):
    FEATURES = 9

    def __init__(self, cfg: AdaptiveMemoryConfig):
        super().__init__()
        self.band = cfg.deadband_compensation
        self.cell = nn.GRUCell(self.FEATURES + cfg.memory_features, cfg.fast_hidden)
        self.head = nn.Sequential(nn.Linear(cfg.fast_hidden + 3, cfg.fast_hidden), nn.SiLU(),
                                  nn.Linear(cfg.fast_hidden, 1))

    def forward(self, aim, aim_future, previous_aim, heard, valid, target_now, voice_now,
                previous_pwm, memory, state):
        heard_n = torch.where(valid, normalize(heard), torch.zeros_like(heard))
        features = torch.stack([aim, aim_future, aim - previous_aim, heard_n, valid.to(aim.dtype),
                                torch.where(valid, normalize(target_now) - heard_n, torch.zeros_like(heard)),
                                voice_now.to(aim.dtype), previous_pwm, aim_future - aim], 1)
        state = self.cell(torch.cat([features, memory], 1), state)
        u = torch.tanh(self.head(torch.cat([state, aim[:, None], aim_future[:, None],
                                            previous_pwm[:, None]], 1)))[:, 0]
        return self.band * torch.tanh(u / .02) + (1 - self.band) * u, state


class Writer(nn.Module):
    """Slow, gated update of one memory from what was just heard."""

    FEATURES = 6

    def __init__(self, cfg: AdaptiveMemoryConfig, size: int, rate: float):
        super().__init__()
        self.rate = rate
        self.candidate = nn.Sequential(nn.Linear(self.FEATURES + cfg.fast_hidden + size, 64), nn.SiLU(),
                                       nn.Linear(64, size))
        self.gate = nn.Linear(self.FEATURES + cfg.fast_hidden, 1)

    def forward(self, features, fast, current, allow):
        x = torch.cat([features, fast], 1)
        proposal = torch.tanh(self.candidate(torch.cat([x, current], 1)))
        rate = self.rate * torch.sigmoid(self.gate(x)) * allow[:, None]
        return current + rate * (proposal - current)


class AdaptiveMemoryPerformer(nn.Module):
    def __init__(self, config: AdaptiveMemoryConfig = AdaptiveMemoryConfig()):
        super().__init__()
        self.config = config
        self.planner, self.core = Planner(config), Core(config)
        self.writers = nn.ModuleDict({
            "motor": Writer(config, config.rig_hidden, config.rig_rate),
            "flute": Writer(config, config.rig_hidden, config.rig_rate),
            "song": Writer(config, config.song_slot, config.song_rate)})

    # ----- memory bookkeeping -------------------------------------------------
    def initial_memory(self, batch, steps, device, dtype=torch.float32):
        cfg = self.config
        return {"motor": torch.zeros(batch, cfg.rig_hidden, device=device, dtype=dtype),
                "flute": torch.zeros(batch, cfg.rig_hidden, device=device, dtype=dtype),
                "song": torch.zeros(batch, steps, cfg.song_slot, device=device, dtype=dtype)}

    @staticmethod
    def forget(memory, *parts, rows=None):
        """Wipe the named parts (optionally only for some rigs: boolean ``rows``)."""
        if memory is None:
            return None
        memory = dict(memory)
        for part in parts:
            if rows is None:
                memory[part] = torch.zeros_like(memory[part])
            else:
                shape = (-1,) + (1,) * (memory[part].dim() - 1)
                memory[part] = torch.where(rows.view(shape), torch.zeros_like(memory[part]), memory[part])
        return memory

    def forget_song(self, memory):
        return self.forget(memory, "song")

    # ----- one play ----------------------------------------------------------
    def perform(self, plant: DifferentiableMotorFlute, cents, voice, parameters, memory=None,
                generator: torch.Generator | None = None, target_delay: int = 2, write=True,
                write_memory=None):
        """Play on the differentiable simulator (pre-training and scoring)."""
        from .device import SimulatorEnv
        if write_memory is not None:  # compatibility with RigAdaptivePerformer callers
            write = write_memory
        return self.play(SimulatorEnv(plant, parameters, generator), cents, voice, memory, write, target_delay)

    def play(self, env, cents, voice, memory=None, write=True, target_delay: int = 2):
        """Home, then play one song on any environment (simulator, black box, world model).

        Only heard pitch reaches the model.  ``write``: True/False or
        {part: bool or (B,) bool tensor}.  If ``env`` keeps logs (a black-box
        device), the observable record of the play is appended to them.
        """
        cfg = self.config
        batch, steps = cents.shape; device, dtype = cents.device, cents.dtype
        memory = self.initial_memory(batch, steps, device, dtype) if memory is None else dict(memory)
        if memory["song"].shape[1] != steps:  # a different song length starts a fresh trace
            memory["song"] = torch.zeros(batch, steps, cfg.song_slot, device=device, dtype=dtype)
        allow = {}
        for part in PARTS:
            flag = write if isinstance(write, bool) else write.get(part, True)
            allow[part] = (flag.to(dtype) if torch.is_tensor(flag)
                           else torch.full((batch,), float(flag), device=device, dtype=dtype))
        state = env.reset(device, dtype, cfg.homing_steps)
        fast = torch.zeros(batch, cfg.fast_hidden, device=device, dtype=dtype)
        heard = torch.zeros(batch, device=device, dtype=dtype)
        valid = torch.zeros(batch, device=device, dtype=torch.bool)
        previous_pwm = torch.zeros(batch, device=device, dtype=dtype)
        previous_aim = torch.zeros(batch, device=device, dtype=dtype)
        song = [memory["song"][:, t] for t in range(steps)]   # per-step slots, written in place
        logs = {key: [] for key in ("pitch_cents", "pwm", "aim", "valve", "heard", "valid")}
        for t in range(steps):
            read = torch.cat([memory["motor"], memory["flute"],
                              torch.cat([song[min(t + k, steps - 1)] for k in SONG_LOOKAHEAD], 1)], 1)
            aim = self.planner(future_window(cents, voice, t), read)
            aim_future = self.planner(future_window(cents, voice, min(t + 10, steps - 1)), read)
            pwm, fast = self.core(aim, aim_future, previous_aim, heard, valid, cents[:, t], voice[:, t],
                                  previous_pwm, read, fast)
            valve = voice[:, t].to(dtype)
            heard, valid, state, emitted = env.step(state, pwm, valve)
            then = max(0, t - target_delay)
            heard_n = torch.where(valid, normalize(heard), torch.zeros_like(heard))
            features = torch.stack([heard_n, valid.to(dtype), normalize(cents[:, then]),
                                    voice[:, then].to(dtype), aim, pwm], 1)
            gate = valid.to(dtype)
            memory["motor"] = self.writers["motor"](features, fast, memory["motor"], allow["motor"] * gate)
            memory["flute"] = self.writers["flute"](features, fast, memory["flute"], allow["flute"] * gate)
            # The heard pitch belongs to the slot `target_delay` steps back.
            song[then] = self.writers["song"](features, fast, song[then], allow["song"])
            for key, value in (("pitch_cents", emitted), ("pwm", pwm), ("aim", aim), ("valve", valve),
                               ("heard", heard), ("valid", valid)):
                logs[key].append(value)
            previous_pwm, previous_aim = pwm, aim
        memory["song"] = torch.stack(song, 1)
        result = {key: torch.stack(value, 1) for key, value in logs.items()}
        if hasattr(env, "record"):
            env.record(cents, voice, result["pwm"], result["valve"], result["heard"], result["valid"],
                       result["pitch_cents"])
        return result, memory

    def checkpoint(self, **metadata):
        return {"format": "yamabiko-adaptive-memory-v1", "config": asdict(self.config),
                "state_dict": self.state_dict(), **metadata}

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        model = cls(AdaptiveMemoryConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
