"""A session: power on, then K different songs in a row on the same rigs.

Before each song the plunger is homed (driven out against the end stop with
the valve shut), so every song starts from a known position. The
valve follows the score with a fixed lead: articulation is not what this
rig is about, the plunger position is.

`run_session` steps the rigs and a controller through the whole timeline
and returns the per-song reward (for training) and, if asked, the logs (for
metrics). Rigs can be swapped (a different flute put in) at a song boundary
without telling the controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim import DT
from ..targets import make_target, sample_level
from .rig import NOMINAL, Rig, RigParams

HOME_STEPS = 100    # 1 s pulling out against the end stop covers the whole stroke
AIM_LEAD = 5        # nominal command delay + velocity lag + 1 [steps]
VALVE_OPEN_LEAD = NOMINAL["valve_delay"] + int(round(NOMINAL["onset_s"] / DT)) - 1  # valve opens this early
VALVE_CLOSE_LEAD = NOMINAL["valve_delay"]
PITCH_CLIP = 300.0
SILENCE_PENALTY = 2.0


@dataclass
class Schedule:
    target: np.ndarray   # (n, T) cents, NaN = the flute must be silent
    homing: np.ndarray   # (T,) bool
    song: np.ndarray     # (T,) song index
    starts: list         # first (homing) step of each song
    aim: np.ndarray      # (n, T) the note to steer toward now (first note AIM_LEAD steps ahead), NaN = hold
    valve: np.ndarray    # (n, T) bool, valve command (to the flute)
    first: np.ndarray    # (n, T) bool, the steps of the FIRST note of each song
    k: int = field(default=0)

    @property
    def n(self) -> int:
        return self.target.shape[0]

    @property
    def T(self) -> int:
        return self.target.shape[1]

    def tile(self, reps: int) -> "Schedule":
        return Schedule(np.tile(self.target, (reps, 1)), self.homing, self.song, self.starts,
                        np.tile(self.aim, (reps, 1)), np.tile(self.valve, (reps, 1)),
                        np.tile(self.first, (reps, 1)), self.k)


def make_schedule(pieces: list[list[np.ndarray]]) -> Schedule:
    """pieces[k][i]: target of song k for rig i. Songs are padded with rests to a common length."""
    k_songs, n = len(pieces), len(pieces[0])
    blocks, homing, song, starts, t = [], [], [], [], 0
    for k, songs in enumerate(pieces):
        length = max(len(s) for s in songs)
        block = np.full((n, HOME_STEPS + length), np.nan)
        for i, s in enumerate(songs):
            block[i, HOME_STEPS:HOME_STEPS + len(s)] = s
        blocks.append(block)
        homing.append(np.arange(block.shape[1]) < HOME_STEPS)
        song.append(np.full(block.shape[1], k))
        starts.append(t)
        t += block.shape[1]
    first = np.concatenate([_first_note(b) for b in blocks], axis=1)
    target = np.concatenate(blocks, axis=1)
    homing = np.concatenate(homing)
    T = target.shape[1]
    note = np.isfinite(target)

    # next note at or after each step (index T = none)
    idx = np.where(note, np.arange(T), T)
    nxt = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    look = nxt[:, np.minimum(np.arange(T) + AIM_LEAD, T - 1)]
    padded = np.concatenate([target, np.full((n, 1), np.nan)], axis=1)
    aim = np.take_along_axis(padded, look, axis=1)

    def ahead(lead):
        return np.concatenate([note[:, lead:], np.zeros((n, lead), bool)], axis=1)

    valve = (ahead(VALVE_OPEN_LEAD) | ahead(VALVE_CLOSE_LEAD)) & ~homing[None, :]
    return Schedule(target, homing, np.concatenate(song), starts, aim, valve, first, k_songs)


def _first_note(block: np.ndarray) -> np.ndarray:
    """The steps of the first note of one song block: after the homing, up to the first rest."""
    note = np.isfinite(block)
    started = np.concatenate([np.zeros((len(block), 1), bool), note[:, :-1]], axis=1)
    return note & (np.cumsum(note & ~started, axis=1) == 1)


def draw_pieces(rng: np.random.Generator, n: int, k: int, bank: list[np.ndarray] | None = None,
                progress: float = 0.8) -> list[list[np.ndarray]]:
    """k songs for n rigs: from the bank if given, otherwise newly generated."""
    if bank is not None:
        return [[bank[int(j)] for j in rng.integers(0, len(bank), n)] for _ in range(k)]
    return [[make_target(rng, sample_level(rng, progress)) for _ in range(n)] for _ in range(k)]


@dataclass
class Swap:
    song: int              # the new rig is in place from the start of this song
    mask: np.ndarray       # (n,) which rigs are swapped
    params: RigParams      # the new rigs (used where mask)


def run_session(rig: Rig, sched: Schedule, ctrl, keep_logs: bool = False, swap: Swap | None = None,
                first_weight: float = 1.0) -> dict:
    """Play the whole session. Returns per-song reward per rig, (n, K), and the logs if asked.

    `first_weight` > 1 counts the first note of each song that much more: it is the note played
    with no pitch heard since the homing, so it is where a memory of the rig has to pay off."""
    n, T, K = sched.n, sched.T, sched.k
    reward = np.zeros((n, K))
    steps = np.zeros((n, K))
    logs = None
    if keep_logs:
        logs = {k: np.zeros((n, T)) for k in ("x", "cents", "heard")}
        logs.update({k: np.zeros((n, T), bool) for k in ("sounding", "overblown")})
    ctrl.begin(sched, rig)
    starts = set(sched.starts)
    for t in range(T):
        if t in starts:
            k = int(sched.song[t])
            if swap is not None and swap.song == k and k > 0:
                rig.swap(swap.mask, swap.params)
            ctrl.song_start(k)
        home = bool(sched.homing[t])
        if home:
            pwm = np.full(n, -1.0)
            valve = np.zeros(n, bool)
        else:
            pwm = ctrl.act(t)
            valve = sched.valve[:, t]
        out = rig.step(pwm, valve)
        ctrl.update(t, pwm, out)
        if home and (t + 1 == T or not sched.homing[t + 1]):
            ctrl.homed()

        tgt = sched.target[:, t]
        note = np.isfinite(tgt)
        if note.any():
            k = int(sched.song[t])
            err = np.minimum(np.abs(np.nan_to_num(out["cents"] - tgt)), PITCH_CLIP) / 100.0
            r = np.where(out["sounding"], -err, -SILENCE_PENALTY)
            w = 1.0 + (first_weight - 1.0) * sched.first[:, t]
            reward[:, k] += np.where(note, r * w, 0.0)
            steps[:, k] += note * w
        if logs is not None:
            for key in logs:
                logs[key][:, t] = out[key]
    ctrl.finish()
    result = {"reward": reward / np.maximum(steps, 1.0)}
    if logs is not None:
        result["logs"] = logs
        result["schedule"] = sched
    if hasattr(ctrl, "probe"):
        result["probe"] = ctrl.probe
    return result
