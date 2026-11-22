"""The ear as a network (step3): 20 ms of raw microphone samples to the flute's pitch and whether it sounds.

It replaces YIN (flute_rl/pitch.py) in front of the performers.  A small 1-D convolutional network reads the
320 samples (normalised by their RMS) the rig hears every 10 ms and scores 96 pitch bins of 25 cent from
-600 to +1800 cent re A4 (about 310..1250 Hz) plus a voicing logit; the pitch is the probability-weighted
mean of the bins around the best one.  Trained on the rig's own recordings (scripts/yamabiko_ear_train.py),
labels found offline with more time and context (scripts/yamabiko_ear_data.py).  The NumPy version below has
no torch, for the UNO Q.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BIN_CENTS, LOW_CENTS, BINS = 25.0, -600.0, 96
FRAME = 320


def bin_cents():
    return LOW_CENTS + BIN_CENTS * np.arange(BINS)


@dataclass(frozen=True)
class EarConfig:
    channels: tuple = (24, 32, 48)
    hidden: int = 128


def torch_model(config: EarConfig = EarConfig()):
    import torch
    from torch import nn

    class EarNet(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            c1, c2, c3 = cfg.channels
            self.conv = nn.Sequential(nn.Conv1d(1, c1, 32, stride=4), nn.ReLU(),       # 320 -> 73
                                      nn.Conv1d(c1, c2, 8, stride=2), nn.ReLU(),       # 73 -> 33
                                      nn.Conv1d(c2, c3, 5, stride=2), nn.ReLU())       # 33 -> 15
            self.fc = nn.Sequential(nn.Linear(c3 * 15, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, BINS + 1))

        @staticmethod
        def normalize(x):
            x = x - x.mean(1, keepdim=True)
            return x / (x.pow(2).mean(1, keepdim=True).sqrt() + 1e-4)

        def forward(self, x):
            """x: (B, 320) float samples in [-1, 1).  Returns (pitch logits (B, 96), voicing logit (B,))."""
            out = self.fc(self.conv(self.normalize(x)[:, None]).flatten(1))
            return out[:, :BINS], out[:, BINS]

        def checkpoint(self, **metadata):
            return {"format": "yamabiko-ear-v1", "config": asdict(self.config), "state_dict": self.state_dict(),
                    **metadata}

    return EarNet(config)


def decode(logits, voiced_logit):
    """(cents, voiced probability) from the network's outputs (NumPy or torch arrays of one frame or many)."""
    logits = np.asarray(logits, np.float64)
    single = logits.ndim == 1
    logits = np.atleast_2d(logits)
    best = logits.argmax(1)
    cents = np.empty(len(logits))
    centres = bin_cents()
    for i, b in enumerate(best):
        lo, hi = max(0, b - 2), min(BINS, b + 3)
        p = np.exp(logits[i, lo:hi] - logits[i, lo:hi].max())
        cents[i] = (p * centres[lo:hi]).sum() / p.sum()
    voiced = 1.0 / (1.0 + np.exp(-np.asarray(voiced_logit, np.float64)))
    return (cents[0], float(np.atleast_1d(voiced)[0])) if single else (cents, voiced)


def export(model):
    arrays = {}
    for i, layer in enumerate(model.conv):
        if hasattr(layer, "weight"):
            arrays[f"ear.conv{i}.w"] = layer.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"ear.conv{i}.b"] = layer.bias.detach().cpu().numpy().astype(np.float32)
            arrays[f"ear.conv{i}.stride"] = np.array(layer.stride[0])
    for i, layer in enumerate(model.fc):
        if hasattr(layer, "weight"):
            arrays[f"ear.fc{i}.w"] = layer.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"ear.fc{i}.b"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    return arrays


class Ear:
    """The ear network on NumPy: ``hear(samples)`` -> (cents or NaN, voiced probability)."""

    def __init__(self, arrays, voiced_threshold: float = .5):
        self.a = {k: np.asarray(v) for k, v in arrays.items()}
        self.convs = sorted({k.split(".")[1] for k in self.a if k.startswith("ear.conv")}, key=lambda s: int(s[4:]))
        self.fcs = sorted({k.split(".")[1] for k in self.a if k.startswith("ear.fc")}, key=lambda s: int(s[2:]))
        self.threshold = voiced_threshold

    def forward(self, x):
        x = np.asarray(x, np.float32)
        x = x - x.mean()
        x = x / (np.sqrt(np.mean(x * x)) + 1e-4)
        h = x[None, :]                                           # (channels, length)
        for name in self.convs:
            w, b, s = self.a[f"ear.{name}.w"], self.a[f"ear.{name}.b"], int(self.a[f"ear.{name}.stride"])
            k = w.shape[2]
            n = (h.shape[1] - k) // s + 1
            idx = np.arange(k)[None, :] + s * np.arange(n)[:, None]          # (n, k)
            cols = h[:, idx]                                                 # (cin, n, k)
            h = np.maximum(np.einsum("oik,ink->on", w, cols) + b[:, None], 0.0)
        z = h.reshape(-1)
        for i, name in enumerate(self.fcs):
            z = self.a[f"ear.{name}.w"] @ z + self.a[f"ear.{name}.b"]
            if i < len(self.fcs) - 1:
                z = np.maximum(z, 0.0)
        return z[:BINS], z[BINS]

    def hear(self, samples):
        """samples: the last 320 samples as floats in [-1, 1) (or int16)."""
        x = np.asarray(samples)
        if x.dtype.kind in "iu":
            x = x / 32768.0
        logits, v = self.forward(x)
        cents, voiced = decode(logits, v)
        return (cents if voiced >= self.threshold else float("nan")), voiced
