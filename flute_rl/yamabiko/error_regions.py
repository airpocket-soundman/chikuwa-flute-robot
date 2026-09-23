"""Split a performance's pitch error by what the mechanism allows.

A slow plunger cannot be at the right place on every sounding frame: after a
leap it is still travelling, and to be on time for the next note it must
leave early.  These regions are fixed by the score and the rig's top speed
alone, not by what a controller did, so every controller is judged on the
same frames:

* ``transit``: the rig could not have arrived yet, even starting at the
  previous note's onset and moving at full speed (plus ``rise_s``).
* ``departing``: to reach the next note on time at full speed, the plunger
  must already be leaving this one.
* ``core``: everything else, where the pitch should be right.

The rig parameters are evaluation labels only; no model reads them.
"""
from __future__ import annotations

import torch

from .physical_plant import DifferentiableMotorFlute, PhysicalPlantParameters


def _per_rig(params: PhysicalPlantParameters) -> PhysicalPlantParameters:
    return type(params)(*(value[:, None] if value is not None and value.dim() == 1 else value
                          for value in vars(params).values()))


def error_regions(plant: DifferentiableMotorFlute, params: PhysicalPlantParameters,
                  cents: torch.Tensor, voice: torch.Tensor, rise_s: float = .15):
    """Return boolean ``(transit, departing, core)`` masks over sounding frames."""
    dt = plant.config.dt
    position = plant.position_for_cents(cents, _per_rig(params))
    speed = params.max_velocity_strokes_s[:, None]
    batch, steps = cents.shape
    changed = torch.zeros_like(voice)
    changed[:, 1:] = (position[:, 1:] - position[:, :-1]).abs() > 1e-4
    # Distance and elapsed time since the latest target change.
    previous = position.clone(); since = torch.zeros_like(position)
    last_value, last_change = position[:, 0].clone(), torch.zeros(batch, device=cents.device)
    for t in range(steps):
        last_value = torch.where(changed[:, t], position[:, t - 1] if t else last_value, last_value)
        last_change = torch.where(changed[:, t], torch.full_like(last_change, t), last_change)
        previous[:, t] = last_value; since[:, t] = t - last_change
    need_arrive = (position - previous).abs() / speed + rise_s
    transit = voice & (since * dt < need_arrive)
    # Distance to and time until the next target change.
    upcoming = position.clone(); until = torch.full_like(position, float("inf"))
    next_value = position[:, -1].clone(); next_change = torch.full((batch,), float("inf"), device=cents.device)
    for t in range(steps - 1, -1, -1):
        if t + 1 < steps:
            next_value = torch.where(changed[:, t + 1], position[:, t + 1], next_value)
            next_change = torch.where(changed[:, t + 1], torch.full_like(next_change, t + 1), next_change)
        upcoming[:, t] = next_value; until[:, t] = next_change - t
    need_leave = (upcoming - position).abs() / speed + rise_s
    departing = voice & ~transit & (until * dt < need_leave)
    core = voice & ~transit & ~departing
    return transit, departing, core


def region_errors(played, cents, voice, masks):
    """Mean absolute error and share of the total error per region."""
    error = (played - cents).abs()
    total = error[voice].sum().clamp_min(1e-9)
    result = {"all": float(error[voice].mean())}
    for name, mask in zip(("transit", "departing", "core"), masks):
        result[name] = float(error[mask].mean()) if mask.any() else None
        result[f"{name}_frames"] = float(mask.sum() / voice.sum().clamp_min(1))
        result[f"{name}_error_share"] = float(error[mask].sum() / total)
    return result
