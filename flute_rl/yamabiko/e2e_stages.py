"""Diagnostic modules for staged E2E development.

These probes make reference memory and target/self comparison measurable
before either representation is allowed to drive the actuator controller.
They consume learned latent features, not hand-written pitch estimates.
"""
from __future__ import annotations

import torch
from torch import nn

from .e2e import E2EConfig


class E2EStageProbes(nn.Module):
    """Gate-2 reference-memory and Gate-4 error-comparison readouts."""

    def __init__(self, config: E2EConfig):
        super().__init__()
        feature_dim = config.audio_dim + 2
        memory_dim = 2 * config.reference_hidden + feature_dim
        hidden = config.controller_hidden
        self.reference = nn.Sequential(
            nn.Linear(memory_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2)
        )
        self.self_audio = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.comparator = nn.Sequential(
            nn.Linear(memory_dim + feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, reference_memory: torch.Tensor, own_feature: torch.Tensor):
        reference = self.reference(reference_memory)
        own_pitch = self.self_audio(own_feature).squeeze(-1)
        # The interpretable difference is the main path.  A small learned
        # residual may compensate representation bias without being able to
        # replace target/self comparison with an unrelated shortcut.
        residual = self.comparator(torch.cat([reference_memory, own_feature], dim=-1)).squeeze(-1)
        error = reference[:, 0] - own_pitch + 0.1 * residual
        return reference, own_pitch, error
