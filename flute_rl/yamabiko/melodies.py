"""Random playable melodies in cents for physical-control training/evaluation.

The downstream stages (Planner, Controller, Feedback) consume the remembered
100 Hz target: pitch in cents plus a voicing flag.  These generators produce
that contract directly, at the 0.5x replay tempo, with a short valve-off gap
at each note start so a move does not sound as an unintended glissando.
"""
from __future__ import annotations

import numpy as np
import torch

ARTICULATION_FRAMES = 4


def random_melodies(rng: np.random.Generator, batch: int, steps: int, device="cpu",
                    low_cents=700.0, high_cents=1900.0, lead_rest=(20, 60)):
    """Return ``(cents, voice)`` of shape ``(B, T)``; cents hold during rests."""
    cents = np.zeros((batch, steps), np.float32)
    voice = np.zeros((batch, steps), np.float32)
    semitones = int((high_cents - low_cents) // 100)
    for row in range(batch):
        t = int(rng.integers(*lead_rest))
        note = int(rng.integers(0, semitones + 1))
        cents[row, :t] = low_cents + 100 * note
        while t < steps:
            length = int(rng.integers(48, 131))
            stop = min(steps, t + length)
            if rng.random() < .18:  # rest
                cents[row, t:stop] = cents[row, t - 1]
            else:
                step = int(rng.choice([-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 12, -12]))
                note = int(np.clip(note + step, 0, semitones))
                cents[row, t:stop] = low_cents + 100 * note
                voice[row, min(stop, t + ARTICULATION_FRAMES):stop] = 1.0
            t = stop
    return (torch.from_numpy(cents).to(device), torch.from_numpy(voice).to(device).bool())


def next_voiced(cents: torch.Tensor, voice: torch.Tensor) -> torch.Tensor:
    """Target of the current or next sounding note (for pre-positioning in rests)."""
    result = cents.clone()
    upcoming = cents[:, -1].clone()
    for t in range(cents.shape[1] - 1, -1, -1):
        upcoming = torch.where(voice[:, t], cents[:, t], upcoming)
        result[:, t] = upcoming
    return result


def note_age(voice: torch.Tensor) -> torch.Tensor:
    """Steps since the current sounding note began (0 while silent)."""
    age = torch.zeros(voice.shape, dtype=torch.long, device=voice.device)
    running = torch.zeros(voice.shape[0], dtype=torch.long, device=voice.device)
    for t in range(voice.shape[1]):
        running = torch.where(voice[:, t], running + 1, torch.zeros_like(running))
        age[:, t] = running
    return age


def pitch_errors(played: torch.Tensor, target: torch.Tensor, voice: torch.Tensor, settle_steps=30):
    """MAE over all sounding frames and over frames after ``settle_steps`` of a note."""
    error = (played - target).abs()
    settled = voice & (note_age(voice) > settle_steps)
    return (error[voice].mean().item(), error[settled].mean().item())
