"""The steps of the performer that live side by side (models/<step>/profile.json), any of them runnable.

Each step folder holds its trained runtime and a profile:

    {"name": "step1", "title": "...", "runtime": "nn_f80_02_runtime.npz", "ear": "yin", "fan": 0.8,
     "low_cents": -10, "high_cents": 1200, "valve_us": [1560, 1820], "doc": "docs/step1.md"}

The code is shared: the runtime file says which performer it is (residual_numpy.runtime_for), the profile
says how to run it.  No NumPy or torch here, so the UNO Q's web page can read the profiles.
"""
from __future__ import annotations

import json
import pathlib

MODELS = pathlib.Path(__file__).resolve().parents[2] / "models"


def load_profiles(root: pathlib.Path | str = MODELS) -> dict:
    """{name: profile}, sorted by name, with ``runtime_path`` resolved to the file in the step's folder."""
    out = {}
    for path in sorted(pathlib.Path(root).glob("*/profile.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile.setdefault("name", path.parent.name)
        profile["runtime_path"] = str(path.parent / profile["runtime"])
        profile.setdefault("ear", "yin")
        out[profile["name"]] = profile
    return out


def latest(profiles: dict) -> str | None:
    """The newest step (the last name in order)."""
    return list(profiles)[-1] if profiles else None
