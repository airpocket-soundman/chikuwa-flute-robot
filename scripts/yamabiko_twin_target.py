"""Write a reference phrase as a target file for scripts/yamabiko_twin_play.py.

    python scripts/yamabiko_twin_target.py --phrase large-leaps --out runs/real/target_leaps.npz
    python scripts/yamabiko_twin_target.py --list
    python scripts/yamabiko_twin_target.py --phrase chromatic --plant yamabiko1    # moved into the rig's range

With ``--plant`` the phrase is moved to start at the rig's lowest target note and, when it spans more than
the rig's range, its intervals are scaled down to fit (rounded to 10 cent).
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
    ap.add_argument("--plant", default=None, help="fit the phrase into this rig's range (e.g. yamabiko1)")
    args = ap.parse_args()
    specs = {s[0]: s for s in sample_specs()}
    if args.list:
        for key, spec in specs.items():
            print(f"{key:18s} {spec[1]}: {spec[2]}")
        return
    _, title, _, notes, durations = specs[args.phrase]
    if args.plant:
        from flute_rl.yamabiko.physical_plant import PhysicalPlantConfig
        cfg = PhysicalPlantConfig.named(args.plant)
        sounding = [n for n in notes if n is not None]
        lo, span = min(sounding), max(sounding) - min(sounding)
        scale = min(1.0, (cfg.high_cents - cfg.low_cents) / span) if span else 1.0
        notes = [None if n is None else float(round(cfg.low_cents + (n - lo) * scale, -1)) for n in notes]
        print(f"{args.plant}: notes {notes} (intervals x{scale:.2f})")
    cents, voice = score_track(note_sequence(notes, durations))
    cents, voice = pad(cents, voice, len(cents) + TAIL)
    out = args.out or f"runs/real/target_{args.phrase}.npz"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cents=cents.astype(np.float32), voice=voice)
    print(f"{title}: {len(cents)} steps ({len(cents) / 100:.1f} s at 0.5x replay) -> {out}")


if __name__ == "__main__":
    main()
