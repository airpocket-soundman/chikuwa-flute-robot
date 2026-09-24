"""Write a reference phrase as a target file for scripts/yamabiko_twin_play.py.

    python scripts/yamabiko_twin_target.py --phrase large-leaps --out runs/real/target_leaps.npz
    python scripts/yamabiko_twin_target.py --list
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_yamabiko_fitting import TAIL  # noqa: E402
from evaluate_yamabiko_hybrid_lab import note_sequence, sample_specs  # noqa: E402
from evaluate_yamabiko_integrated import pad, score_track  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrase", default="four-step")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    specs = {s[0]: s for s in sample_specs()}
    if args.list:
        for key, spec in specs.items():
            print(f"{key:18s} {spec[1]}: {spec[2]}")
        return
    _, title, _, notes, durations = specs[args.phrase]
    cents, voice = score_track(note_sequence(notes, durations))
    cents, voice = pad(cents, voice, len(cents) + TAIL)
    out = args.out or f"runs/real/target_{args.phrase}.npz"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cents=cents.astype(np.float32), voice=voice)
    print(f"{title}: {len(cents)} steps ({len(cents) / 100:.1f} s at 0.5x replay) -> {out}")


if __name__ == "__main__":
    main()
