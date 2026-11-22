"""Write a reference phrase as a target file for scripts/yamabiko_twin_play.py.

    python scripts/yamabiko_twin_target.py --phrase large-leaps --out runs/real/target_leaps.npz
    python scripts/yamabiko_twin_target.py --list
    python scripts/yamabiko_twin_target.py --phrase chromatic --plant yamabiko1    # moved into the rig's range

With ``--plant`` the phrase is transposed into the middle of the rig's range with every interval kept (a
relative melody copy); a phrase wider than the range is refused.  The play can transpose it again to the
pitch the flute sounds that day (``relative`` in the server's play).
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
    ap.add_argument("--notes", default=None,
                    help="a phrase of your own instead: cents re A4, comma separated, 'r' for a rest")
    ap.add_argument("--note-steps", type=int, default=40, help="--notes: length of each note [10 ms steps]")
    ap.add_argument("--plant", default=None, help="fit the phrase into this rig's range (e.g. yamabiko1)")
    args = ap.parse_args()
    specs = {s[0]: s for s in sample_specs()}
    if args.list:
        for key, spec in specs.items():
            print(f"{key:18s} {spec[1]}: {spec[2]}")
        return
    if args.notes:
        notes = [None if n.strip() == "r" else float(n) for n in args.notes.split(",")]
        title, durations = f"notes {args.notes}", [args.note_steps] * len(notes)
        args.phrase = args.phrase if args.out else "custom"
    else:
        _, title, _, notes, durations = specs[args.phrase]
    if args.plant:
        # relative melody copy: keep every interval, only transpose into the rig's range (centred)
        from flute_rl.yamabiko.physical_plant import PhysicalPlantConfig
        cfg = PhysicalPlantConfig.named(args.plant)
        sounding = [n for n in notes if n is not None]
        lo, span = min(sounding), max(sounding) - min(sounding)
        room = cfg.high_cents - cfg.low_cents
        if span > room:
            raise SystemExit(f"the phrase spans {span:.0f} cent, the rig only {room:.0f}: pick a narrower one")
        shift = cfg.low_cents + (room - span) / 2 - lo
        notes = [None if n is None else float(round(n + shift)) for n in notes]
        scale = 1.0
        print(f"{args.plant}: notes {notes} (intervals x{scale:.2f})")
    cents, voice = score_track(note_sequence(notes, durations))
    cents, voice = pad(cents, voice, len(cents) + TAIL)
    out = args.out or f"runs/real/target_{args.phrase}.npz"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cents=cents.astype(np.float32), voice=voice)
    print(f"{title}: {len(cents)} steps ({len(cents) / 100:.1f} s at 0.5x replay) -> {out}")


if __name__ == "__main__":
    main()
