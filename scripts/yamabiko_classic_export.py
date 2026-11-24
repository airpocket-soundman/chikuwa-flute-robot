"""PC: make the classic PID performer as a step folder (models/classic/), runnable like any learned step.

    python scripts/yamabiko_classic_export.py                                   # twin of step4, default gains
    python scripts/yamabiko_classic_export.py --twin runs/real/twin_y1f80_05_runtime.npz --set kp=15 ki=3
    python scripts/yamabiko_classic_export.py --tune 120                        # random search on the simulator first

The runtime is flute_rl/yamabiko/classic_numpy.ClassicRuntime: the twin's formulas for the plan and the dead
reckoning, a pitch observer, PID on position and a rule valve.  No network, so the profile has no
valve_runtime and hears with YIN.  ``--tune N`` tries N random changes of the main gains on simulated rigs
around the twin (flute_rl/yamabiko/sim_numpy.py; doremi and scale, 8 rigs) and keeps the best, then checks it
on 16 other rigs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flute_rl.yamabiko.classic_numpy import CLASSIC_FIELDS, DEFAULT_GAINS, export  # noqa: E402
from flute_rl.yamabiko.residual_numpy import runtime_for  # noqa: E402
from flute_rl.yamabiko.sim_numpy import rig_arrays, rigs_around  # noqa: E402
from yamabiko_compare_sim import evaluate, load_target  # noqa: E402

SEARCH = dict(kp=(2, 30), ki=(0, 10), kd=(0, 4), hold_tol=(.002, .02), pitch_ki=(0, .15), valve_gate=(.015, .15))


def tune(twin_arrays, gains, tries, seed=0):
    constants, twin = rig_arrays(twin_arrays)
    songs = [load_target(n) for n in ("f80_doremi", "f80_scale")]

    def score(g, rig_seed=1234, count=8):
        return float(np.mean([evaluate(lambda: runtime_for(export(twin_arrays, **g)),
                                       rigs_around(constants, twin, count, .12, rig_seed), c, v)[0]["mae"]
                              for c, v in songs]))
    rng = np.random.default_rng(seed)
    best, best_score = dict(gains), score(gains)
    print(f"start: mae {best_score:.1f} cent")
    for i in range(tries):
        g = dict(best)
        for k, (lo, hi) in SEARCH.items():
            if rng.random() < .5:
                g[k] = float(np.clip(g[k] * np.exp(rng.normal(0, .35)) if g[k] > 0 else rng.uniform(lo, hi), lo, hi))
        g["hold_release"] = max(g["hold_release"], 2 * g["hold_tol"])
        s = score(g)
        if s < best_score:
            best, best_score = g, s
            print(f"  try {i}: mae {s:.1f} cent  " + " ".join(f"{k}={g[k]:.4g}" for k in SEARCH))
    print(f"kept: mae {best_score:.1f} cent on the tuning rigs, {score(best, 99, 16):.1f} on 16 other rigs "
          f"(start {score(gains, 99, 16):.1f})")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin", default=str(ROOT / "models/step4/twin_y1f80_04_runtime.npz"),
                    help="a twin runtime .npz (twin and controller constants)")
    ap.add_argument("--out", default=str(ROOT / "models/classic"))
    ap.add_argument("--set", nargs="*", default=[], help="gains as name=value")
    ap.add_argument("--tune", type=int, default=0, help="random-search tries on the simulator (0: none)")
    ap.add_argument("--profile-from", default=str(ROOT / "models/step4/profile.json"),
                    help="fan, valve pulses, pitch range and whistle options are copied from this profile")
    args = ap.parse_args()
    twin_arrays = dict(np.load(args.twin))
    gains = dict(DEFAULT_GAINS)
    for item in args.set:
        k, v = item.split("=")
        if k not in CLASSIC_FIELDS:
            raise SystemExit(f"unknown gain {k}; one of {', '.join(CLASSIC_FIELDS)}")
        gains[k] = float(v)
    if args.tune:
        gains = tune(twin_arrays, gains, args.tune)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "classic_runtime.npz", **export(twin_arrays, **gains))
    base = json.loads(pathlib.Path(args.profile_from).read_text(encoding="utf-8"))
    profile = {"name": out.name,
               "title": f"{out.name}: 比較用の古典制御（PID・観測器・規則の弁、NN なし、耳は YIN）",
               "doc": "docs/classic.md", "runtime": "classic_runtime.npz", "ear": "yin",
               **{k: base[k] for k in ("fan", "low_cents", "high_cents", "valve_us", "whistle") if k in base},
               "twin": pathlib.Path(args.twin).name, "gains": gains}
    (out / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for target in sorted(pathlib.Path(args.profile_from).parent.glob("target_*.npz")):
        shutil.copy(target, out / target.name)
    print(f"wrote {out}/classic_runtime.npz and profile.json")


if __name__ == "__main__":
    main()
