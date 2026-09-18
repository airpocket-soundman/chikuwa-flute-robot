"""Compare controllers of Yamabiko No.1 song by song after power on (see docs/yamabiko.md).

    python scripts/yamabiko_compare.py --rigs 200 --songs 5
    python scripts/yamabiko_compare.py --policy runs/yamabiko_gru.npz --swap-song 3 --json runs/compare.json

Rows are controllers, columns are songs since power on. --swap-song k puts a
different rig in (every rig) at the start of song k, without telling anyone.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.yamabiko import (Encoder, ExternalFit, GRUPolicy, Observer, OpenLoop, Oracle, Rig, RigParams,  # noqa: E402
                               Swap, draw_pieces, make_schedule, run_session, song_metrics)
from flute_rl.yamabiko.metrics import KEYS  # noqa: E402

LABELS = {
    "mean_abs": "mean |cents|",
    "first1": "first note of the song, first step |cents|",
    "onset1": "onset, first step |cents|",
    "onset5": "onset, first 5 steps |cents|",
    "converge": "converged (within 20 cents)",
    "converge_ms": "time to converge [ms]",
    "steady": "steady |cents|",
    "sounding": "sounding rate",
    "blowups": "blow-ups per 100 onsets",
    "overblow": "overblown rate",
}


def controllers(policy: str | None):
    yield OpenLoop
    yield Encoder
    yield Oracle
    yield Observer
    yield ExternalFit
    if policy:
        d = np.load(policy)
        theta, hidden = d["theta"], int(d["hidden"])
        yield lambda: GRUPolicy(theta, hidden, carry=True)
        yield lambda: GRUPolicy(theta, hidden, carry=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigs", type=int, default=200)
    ap.add_argument("--songs", type=int, default=5)
    ap.add_argument("--harsh", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--policy", type=str, default=None, help="trained recurrent policy (.npz)")
    ap.add_argument("--swap-song", type=int, default=0,
                    help="put new rigs in just before this song, counted from 1 like the table (0: never)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    if args.swap_song and not 2 <= args.swap_song <= args.songs:
        ap.error("--swap-song must be between 2 and --songs")

    rng = np.random.default_rng(args.seed)
    params = RigParams.sample(rng, args.rigs, harsh=args.harsh)
    sched = make_schedule(draw_pieces(rng, args.rigs, args.songs))
    swap = None
    if args.swap_song:
        swap = Swap(args.swap_song - 1, np.ones(args.rigs, bool), RigParams.sample(rng, args.rigs, harsh=args.harsh))

    results = {}
    for make in controllers(args.policy):
        ctrl = make()
        res = run_session(Rig(params, np.random.default_rng(args.seed + 1)), sched, ctrl, keep_logs=True, swap=swap)
        results[ctrl.name] = song_metrics(res["logs"], sched)
        print(f"{ctrl.name} done", file=sys.stderr, flush=True)

    head = "".join(f"{'song ' + str(k + 1):>9s}" for k in range(args.songs))
    for key in KEYS:
        print(f"\n{LABELS[key]}{'  (swap before song ' + str(args.swap_song) + ')' if swap else ''}")
        print(f"{'':14s}{head}")
        for name, songs in results.items():
            print(f"{name:14s}" + "".join(f"{d[key]:9.2f}" if key in ('converge', 'sounding', 'overblow')
                                          else f"{d[key]:9.1f}" for d in songs))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"args": vars(args), "results": results}, indent=1))


if __name__ == "__main__":
    main()
