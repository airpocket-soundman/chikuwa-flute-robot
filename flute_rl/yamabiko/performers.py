"""Load any trained downstream performer from its checkpoint."""
from __future__ import annotations

import pathlib

import torch

from .adaptive_memory import AdaptiveMemoryPerformer
from .rig_adaptive import RigAdaptivePerformer


def load_performer(path, device="cpu"):
    """Return ``(model, checkpoint)``; both performers share ``perform``'s signature."""
    checkpoint = torch.load(pathlib.Path(path), map_location=device, weights_only=False)
    if checkpoint.get("format") == "yamabiko-adaptive-memory-v1":
        model = AdaptiveMemoryPerformer.from_checkpoint(checkpoint, device)
    else:
        model = RigAdaptivePerformer.from_checkpoint(checkpoint, device)
    return model.eval(), checkpoint
