"""Dependency-free audio contract shared by E2E training and UNO Q inference."""
from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16_000
CONTROL_HZ = 100
HOP = SAMPLE_RATE // CONTROL_HZ       # 160 new samples per control step
FRAME = 320                           # latest 20 ms, same window as yamabiko.hw


def audio_float(audio) -> np.ndarray:
    """Mono PCM as float32 in [-1, 1], accepting firmware int16 recordings."""
    x = np.asarray(audio).reshape(-1)
    if np.issubdtype(x.dtype, np.integer):
        return (x.astype(np.float32) / 32768.0)
    x = x.astype(np.float32, copy=False)
    if x.size and np.max(np.abs(x)) > 2.0:  # float containers holding int16 values
        x = x / 32768.0
    return x


def frame_audio_numpy(audio, steps: int | None = None, *, causal: bool = False) -> np.ndarray:
    """Raw 20 ms windows on the firmware's 10 ms sample grid.

    Complete references use centred windows.  Live self audio is causal: the
    frame at action t ends at the start of that action.
    """
    x = audio_float(audio)
    if steps is None:
        steps = max(1, int(math.ceil(len(x) / HOP)))
    xp = np.concatenate([np.zeros(FRAME, np.float32), x, np.zeros(FRAME, np.float32)])
    if causal:
        ends = np.arange(steps) * HOP + FRAME
        idx = ends[:, None] - FRAME + np.arange(FRAME)[None, :]
    else:
        centers = ((np.arange(steps) + 0.5) * HOP).astype(int) + FRAME
        idx = centers[:, None] - FRAME // 2 + np.arange(FRAME)[None, :]
    return xp[np.clip(idx, 0, len(xp) - 1)].astype(np.float32, copy=False)
