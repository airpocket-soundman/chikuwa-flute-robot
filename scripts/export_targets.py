"""Write training targets as WAV files (sine at the target pitch, silence at rests) and an overview plot.

    python scripts/export_targets.py --bank runs/target_bank.npz --count 20 --out runs/target_preview
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.sim import DT, cents_to_hz  # noqa: E402
from flute_rl.targets import load_bank  # noqa: E402

SR = 22050


def render(target: np.ndarray, sr: int = SR) -> np.ndarray:
    """Phase-continuous sine following the target, with 5 ms fades at note starts/ends."""
    n = int(len(target) * DT * sr)
    idx = np.minimum((np.arange(n) / sr / DT).astype(int), len(target) - 1)
    c = target[idx]
    on = np.isfinite(c)
    f = np.where(on, cents_to_hz(np.where(on, c, 0.0)), 0.0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    amp = on.astype(float)
    k = np.ones(int(0.005 * sr)) / int(0.005 * sr)
    amp = np.convolve(amp, k, mode="same")
    return 0.3 * amp * np.sin(phase)


def write_wav(path: pathlib.Path, y: np.ndarray, sr: int = SR) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="runs/target_bank.npz")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--out", default="runs/target_preview")
    args = ap.parse_args()

    bank = load_bank(args.bank)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(bank[:args.count]):
        write_wav(out / f"target_{i:04d}.wav", render(t))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = min(args.count, 20)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 1.1 * rows), sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        t = bank[i]
        ax.plot(np.arange(len(t)) * DT, cents_to_hz(t), lw=1.5)
        ax.set_ylabel(f"#{i}", rotation=0, labelpad=18)
        ax.set_ylim(600, 1400)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel("time [s]  (y: Hz, gaps = silence)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=90)

    durs = []
    for t in bank:
        on = np.isfinite(t)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
        durs += list((edges[1::2] - edges[::2]) * DT)
    print(f"wrote {min(args.count, len(bank))} WAV files and overview.png to {out}")
    print(f"{len(bank)} pieces, sounding segments: shortest {min(durs):.2f} s, median {np.median(durs):.2f} s")


if __name__ == "__main__":
    main()
