"""PC or UNO Q: whole step8 episodes in the NumPy simulator, against step7c playing the rule's score.

    python scripts/yamabiko_step8_sim.py --step models/step8 --songs 8 --rigs 8

One episode: 0.5 s idle (room noise) -> the button -> a whistle (synthetic, or cut from the rig's held-out
recordings with --real) heard through the ear into the memory -> the button -> step8 plays it on a simulated
rig (its fan spin-up and homing, then the actuator and valve networks reading the memory) -> step8 stops by
itself.  The simulator gives the pitch while playing (the ear hears the whistle only), as
scripts/yamabiko_compare_sim.py.

Per episode, against the rule pipeline's score of the same whistle (score_memory.teacher, with the player's
lead and tail):
  hit / hit_in_key   share of settled note time within 25 cent (in key: after the median offset)
  end_s              when step8 stopped minus when the score ends (s; negative = too early)
and the same hits for step7c playing the rule's score itself (the array score, today's step7c), and step7c
playing the score recalled from the memory as one array (score_from_recording, today's step6/7 copy).
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flute_rl.yamabiko.ear_nn import FRAME, Ear  # noqa: E402
from flute_rl.yamabiko.ear_synth import phrase  # noqa: E402
from flute_rl.yamabiko.profiles import load_profiles  # noqa: E402
from flute_rl.yamabiko.residual_numpy import WithValve, runtime_for  # noqa: E402
from flute_rl.yamabiko.score_memory import MemoryConfig, teacher  # noqa: E402
from flute_rl.yamabiko.sim_numpy import play, rig_arrays, rigs_around  # noqa: E402
from flute_rl.yamabiko.step8 import (HOMING_STEPS, LEAD_STEPS, PLAY, TAIL_STEPS, WARM_STEPS, Step8)  # noqa: E402

HOP = 160
WHISTLE = {"max_gap_steps": 2}


def frames_of(x):
    x = np.round(np.asarray(x, np.float64) * 32767) / 32768.0
    return [x[e - FRAME:e] for e in range(FRAME, len(x) + 1, HOP)]


def settled(c, v):
    keep = np.zeros(len(c), bool); run = 0
    for i in range(len(c)):
        run = run + 1 if v[i] else 0
        keep[i] = run > 30 and abs(c[i] - c[max(0, i - 15)]) < 50
    return keep


def framed(c, v):
    return (np.r_[np.full(LEAD_STEPS, c[0]), c, np.full(TAIL_STEPS, c[-1])],
            np.r_[np.zeros(LEAD_STEPS, bool), v, np.zeros(TAIL_STEPS, bool)])


def hits(emitted, ref_c, keep):
    n = min(len(emitted), len(ref_c))
    e = np.full(len(ref_c), np.nan); e[:n] = emitted[:n]
    k = keep & np.isfinite(e)
    if not k.any():
        return 0.0, 0.0
    err = e[k] - ref_c[k]
    return float(np.mean(np.abs(err) <= 25)), float(np.mean(np.abs(err - np.median(err)) <= 25))


def whistles(args, rng):
    """(name, samples float, cents per frame) of the whistles to copy."""
    out = []
    if args.real:
        from flute_rl.yamabiko.whistle import whistle_track
        for path in sorted(pathlib.Path(args.real).glob("*.npz")):
            if not any(h in path.stem for h in args.real_holdout):
                continue
            d = np.load(path)
            if "pcm" not in d.files or ("kind" in d.files and str(d["kind"]) != "whistle"):
                continue
            pcm = d["pcm"]
            track = np.r_[np.nan, whistle_track(pcm, 16000)]
            for k in range(args.songs):
                sec = rng.uniform(4, 12)
                start = int(rng.integers(0, max(1, len(pcm) - int(sec * 16000))))
                seg = pcm[start:start + int(sec * 16000)]
                f0 = start // HOP
                out.append((f"{path.stem}#{k}", seg / 32768.0, track[f0:f0 + len(frames_of(seg))]))
    else:
        for k in range(args.songs):
            x, cents, _ = phrase(rng, float(rng.uniform(4, 12)))
            x = x + rng.normal(0, 10 ** (rng.uniform(-75, -60) / 20), len(x)).astype(np.float32)
            out.append((f"synth#{k}", x, cents))
    return out


def episode(step8, frames, rig, room_rng):
    """Listen to ``frames`` then play on ``rig`` until step8 stops: emitted pitch per actuator step, end step."""
    s8 = copy.deepcopy(step8)
    s8.reset()
    for _ in range(50):                                        # idle with the room
        s8.tick(room_rng.normal(0, 3e-4, FRAME))
    for i, f in enumerate(frames):
        out = s8.tick(f, button=1 if i == 0 else 0)
    assert out["mode"] == "listen", out
    emitted, heard, end_at, s8_last_t = {}, None, None, 0
    rig.reset(rig.rng.uniform(0, .8))
    for k in range(WARM_STEPS + HOMING_STEPS + 60 * 100):
        out = s8.tick(None, button=1 if k == 0 else 0, heard=heard)
        if out["mode"] != "play":
            end_at = s8_last_t
            break
        c, sounding, h, valid = rig.step(out["pwm"], out["valve"])
        heard = (h, valid)
        if "t" in out:
            emitted[out["t"]] = c
            s8_last_t = out["t"]
        elif k >= WARM_STEPS + HOMING_STEPS:
            s8_last_t = k
    t_max = max(emitted) + 1 if emitted else 0
    em = np.full(t_max, np.nan)
    for t, c in emitted.items():
        em[t] = c
    return em, end_at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="step8")
    ap.add_argument("--baseline", default="step7c")
    ap.add_argument("--songs", type=int, default=8)
    ap.add_argument("--rigs", type=int, default=8)
    ap.add_argument("--real", default=None, help="folder of rig recordings: cut whistles from the held-out takes")
    ap.add_argument("--real-holdout", nargs="*", default=["20261124_111404", "20261124_111830"])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    profiles = load_profiles()
    p8, base = profiles[args.step], profiles[args.baseline]
    step8 = Step8.from_profile(p8)
    constants, twin = rig_arrays(p8["twin_runtime_path"] if "twin_runtime_path" in p8 else p8["runtime_path"])
    rng = np.random.default_rng(args.seed)
    cfg = MemoryConfig(low=p8.get("low_cents", -10.0), high=p8.get("high_cents", 1200.0))
    rows = []
    for name, x, cents in whistles(args, rng):
        frames = frames_of(x)
        t = teacher(np.asarray(cents[:len(frames)], np.float64), cfg, WHISTLE)
        if t is None or t[1].sum() < 50:
            continue
        ref_c, ref_v = framed(*t)
        keep = settled(ref_c, ref_v)
        should_end = LEAD_STEPS + int(np.flatnonzero(t[1])[-1]) + 1 + TAIL_STEPS
        # the memory recalled as one array (step6/7's copy) from the same listening
        listened = copy.deepcopy(step8); listened.reset()
        for i, f in enumerate(frames):
            listened.tick(f, button=1 if i == 0 else 0)
        mem = listened.memory
        rec = np.array([mem.recall(i) for i in range(mem.n)])
        arr_c, arr_v = framed(rec[:, 0], rec[:, 1] >= .5)
        for r, rig in enumerate(rigs_around(constants, twin, args.rigs, seed=900 + len(rows))):
            em, end_at = episode(step8, frames, rig, np.random.default_rng(r))
            h8 = hits(em, ref_c, keep)
            outs = {}
            for label, (c, v) in (("rule", (ref_c, ref_v)), ("memory_array", (arr_c, arr_v))):
                runtime = WithValve(runtime_for(dict(np.load(base["runtime_path"]))), dict(np.load(base["valve_runtime_path"])))
                rig2 = rigs_around(constants, twin, args.rigs, seed=900 + len(rows))[r]
                o = play(runtime, rig2, c, v)
                outs[label] = hits(o["emitted"], ref_c, keep)
            rows.append({"song": name, "rig": r, "step8_hit": h8[0], "step8_in_key": h8[1],
                         "rule_hit": outs["rule"][0], "rule_in_key": outs["rule"][1],
                         "array_hit": outs["memory_array"][0], "array_in_key": outs["memory_array"][1],
                         "end_s": None if end_at is None else (end_at - should_end) / 100})
        last = rows[-1]
        print(f"{name}: step8 {last['step8_hit']:.2f}/{last['step8_in_key']:.2f}  rule {last['rule_hit']:.2f}/"
              f"{last['rule_in_key']:.2f}  array {last['array_hit']:.2f}/{last['array_in_key']:.2f}  end {last['end_s']}",
              flush=True)
    keys = ["step8_hit", "step8_in_key", "array_hit", "array_in_key", "rule_hit", "rule_in_key"]
    ends = np.array([r["end_s"] for r in rows if r["end_s"] is not None])
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update(episodes=len(rows), ended=len(ends), end_mean_s=float(ends.mean()) if len(ends) else None,
                   end_abs_mean_s=float(np.abs(ends).mean()) if len(ends) else None)
    print(json.dumps(summary, indent=2))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
