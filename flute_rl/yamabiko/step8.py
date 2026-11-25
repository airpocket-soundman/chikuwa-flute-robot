"""step8 (first stage, 8a): the whole robot as one object called every 10 ms.

    out = Step8(profile).tick(frame, button)     # frame: the last 320 microphone samples, button: 0 / 1

Inside, the networks trained in steps 5 to 7 are joined, and two small new ones decide what to do:

  ear (step5)        every frame -> pitch heard, and its hidden code
  memory (step6)     while listening the code is written into W; while playing the score is read back from it
  actuator + valve   (step7) PWM and air from the score read back, the pitch heard and what they did before
  conductor          a tiny MLP: button, "finished" and the mode it chose before -> idle / listen / play,
                     and the fan's duty
  end                a small MLP: from how much was written around the play head and whether it sounded,
                     "the song has finished"

Nothing between listening and playing is an array: the actuator's look-ahead reads the memory at the instants it
needs (each read once, then kept).  Still rules for now (as the plan says, kept outside the networks first):
the fan's spin-up time and homing before the song (``WARM_STEPS`` with the valve shut, then ``HOMING_STEPS`` at
full reverse), the lead before the first note and the shift measured by calibration.  The safety limits stay
in the server and the firmware.

For the simulator, ``tick(..., heard=(cents, valid))`` replaces the ear's hearing while playing (the simulator
gives the pitch, not the sound, as scripts/yamabiko_compare_sim.py).
"""
from __future__ import annotations

import numpy as np

from .ear_nn import decode
from .residual_numpy import WithValve, runtime_for, sigmoid
from .score_memory import MemoryRuntime

IDLE, LISTEN, PLAY = 0, 1, 2
MODES = ("idle", "listen", "play")
LEAD_STEPS, TAIL_STEPS = 60, 60          # as score_from_recording
WARM_STEPS, HOMING_STEPS = 300, 130      # as the server: fan spin-up (valve shut), then homing at full reverse
END_LOOK = (-80, -60, -40, -20, 0, 10, 20, 40, 60, 80, 120)   # instants around the play head the end network reads
MAX_PLAY_STEPS = 60 * 100                # a listening longer than this is cut (a limit, as the plan's 60 s)


def end_features(reads, u, mass_ref):
    """The end network's input at play head ``u`` (memory instants): voice probability and the written mass
    (relative to the middle of the song) at ``u + END_LOOK`` (not before 0).  ``reads(i)`` -> (cents, prob, mass)."""
    f = []
    for o in END_LOOK:
        _, p, m = reads(max(u + o, 0))
        f += [p, m / mass_ref]
    return np.array(f, np.float32)


class Recalled:
    """The score as the actuator sees it (index t of its time, ``LEAD_STEPS`` before the song), read from the
    memory only when first asked for.  Supports ``len`` and integer or list indexing, as the arrays it stands
    in for."""

    def __init__(self, memory, listened, shift, field):
        self.memory, self.n, self.shift, self.field = memory, listened, shift, field
        self.cache = memory_cache(memory)

    def __len__(self):
        return LEAD_STEPS + self.n + TAIL_STEPS

    def _one(self, t):
        u = int(t) - LEAD_STEPS
        cents, p, _ = self.cache(min(max(u, 0), self.n - 1))
        if self.field == "cents":
            return cents + self.shift
        return bool(0 <= u < self.n and p >= .5)

    def __getitem__(self, idx):
        if np.ndim(idx) == 0:
            return self._one(idx)
        return np.array([self._one(i) for i in idx], np.float64 if self.field == "cents" else bool)


def memory_cache(memory):
    """memory.read(i) remembered per instant (shared by every view of the same memory)."""
    if not hasattr(memory, "_reads"):
        memory._reads = {}

    def read(i):
        i = int(i)
        if i not in memory._reads:
            memory._reads[i] = memory.read(i)
        return memory._reads[i]
    return read


class Step8:
    def __init__(self, ear, memory_arrays, actuator_arrays, valve_arrays, heads, max_cents_above: float = 250.0):
        self.ear = ear
        self.memory_arrays = memory_arrays
        self.actuator_arrays, self.valve_arrays = actuator_arrays, valve_arrays
        self.h = {k: np.asarray(v) for k, v in heads.items()}
        self.high = float(memory_arrays["mem.high"])
        self.max_cents_above = max_cents_above
        self.reset()

    @classmethod
    def from_profile(cls, profile: dict):
        from .ear_nn import Ear
        load = lambda p: dict(np.load(p))
        return cls(Ear(load(profile["ear_runtime_path"])), load(profile["memory_runtime_path"]),
                   load(profile["runtime_path"]), load(profile["valve_runtime_path"]), load(profile["heads_path"]))

    def reset(self):
        self.mode = IDLE
        self.end_p = 0.0
        self.memory = None
        self.player = None
        self.tau = 0

    # ----- the two new small networks ----------------------------------------------------------------
    def conductor(self, button):
        h = self.h
        x = np.r_[float(button), self.end_p, np.eye(3)[self.mode]].astype(np.float32)
        y = np.maximum(h["s8.cond.0.w"] @ x + h["s8.cond.0.b"], 0.0)
        out = h["s8.cond.2.w"] @ y + h["s8.cond.2.b"]
        return int(np.argmax(out[:3])), float(sigmoid(out[3]))

    def end_net(self, u):
        x = end_features(self.reads, u, self.mass_ref)
        y = np.maximum(self.h["s8.end.0.w"] @ x + self.h["s8.end.0.b"], 0.0)
        return float(sigmoid(self.h["s8.end.2.w"] @ y + self.h["s8.end.2.b"])[0])

    # ----- every 10 ms -----------------------------------------------------------------------------------
    def tick(self, frame, button=0, shift: float = 0.0, heard=None):
        mode, fan = self.conductor(button)
        if mode != self.mode:
            self.end_p = 0.0
            if mode == LISTEN:
                self.memory = MemoryRuntime(self.memory_arrays)
            if mode == PLAY:
                self._start_play(shift)
            self.mode = mode
        out = {"mode": MODES[self.mode], "fan": fan, "pwm": 0.0, "valve": False, "end": self.end_p}
        hear = self.mode == LISTEN or (self.mode == PLAY and heard is None)
        if hear and frame is not None:
            x = np.asarray(frame)
            x = x / 32768.0 if x.dtype.kind in "iu" else x
            (logits, v), code = self.ear._run(x)
            cents, p = decode(logits, v, self.ear.low_cents)
            if self.mode == LISTEN:
                if self.memory.n < MAX_PLAY_STEPS:
                    self.memory.listen(code)
            elif heard is None:
                heard = (cents, p >= self.ear.threshold)
        if self.mode == PLAY and self.player is not None:
            out.update(self._play_step(heard))
        return out

    def _start_play(self, shift):
        self.tau = 0
        if self.memory is None or self.memory.n < 10:
            self.player = None
            return
        n = self.memory.n
        self.reads = memory_cache(self.memory)
        self.mass_ref = max(self.reads(n // 2)[2], 1e-6)
        self.player = WithValve(runtime_for(dict(self.actuator_arrays)), dict(self.valve_arrays))
        cents, voice = Recalled(self.memory, n, shift, "cents"), Recalled(self.memory, n, shift, "voice")
        self.player.prepare(np.zeros(len(cents)), np.zeros(len(cents), bool), keep_song_memory=False)
        for part in (self.player, self.player.act):           # the look-ahead reads the memory, no array
            part.cents, part.voice = cents, voice
        self.max_cents = self.high + shift + self.max_cents_above
        self.t = 0

    def _play_step(self, heard):
        tau = self.tau; self.tau += 1
        if tau < WARM_STEPS:
            return {"pwm": 0.0, "valve": False}
        if tau < WARM_STEPS + HOMING_STEPS:
            return {"pwm": -1.0, "valve": False}
        if self.t > 0 and heard is not None:
            h, ok = heard
            ok = bool(ok) and np.isfinite(h) and h <= self.max_cents
            self.player.observe(h if ok else 0.0, ok)
        t = self.t
        if t >= len(self.player.cents) + 4 * TAIL_STEPS:     # a limit, in case the end network never fires
            self.end_p = 1.0
            return {"pwm": 0.0, "valve": False, "end": 1.0, "t": t}
        pwm, valve = self.player.command(t)
        self.end_p = self.end_net(t - LEAD_STEPS)
        self.t += 1
        return {"pwm": float(pwm), "valve": bool(valve), "end": self.end_p, "t": t}
