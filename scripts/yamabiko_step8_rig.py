"""PC: step8 versions on the rig, from a recorded whistle to the flute, with practice ("again").

    python scripts/yamabiko_step8_rig.py --steps step8p step8p2 --plays 3

The whistles are the rig's held-out recordings (not trained on): from each, ``--crops`` stretches of
``--seconds`` with the most whistling.  For every version and stretch: one calibration per version (the
flute sounds ~10 s), then the server's step8 listens to the stretch in place of the microphone
(``step8_start(clip_pcm16_b64=...)``), plays what it heard and stops by itself; then ``--plays`` - 1 times
"again" (practice).  Every play's heard pitch is compared with the rule's score of the same stretch
(score_memory.teacher on YIN, moved by the calibration's shift, the player's lead and tail), on the settled note
time: hit within 25 cent, in key (after the median offset), and the stop time against the score's end.
The flute sounds all through.  Writes runs/real/eval/<stamp>_step8_rig.json.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flute_rl.yamabiko.profiles import load_profiles  # noqa: E402
from flute_rl.yamabiko.remote import Remote  # noqa: E402
from flute_rl.yamabiko.score_memory import MemoryConfig, teacher  # noqa: E402
from flute_rl.yamabiko.step8 import LEAD_STEPS, TAIL_STEPS  # noqa: E402
from flute_rl.yamabiko.whistle import whistle_track  # noqa: E402

UNOQ_ROOT = "/home/arduino/yamabiko"
HOP, FRAME, SR = 160, 320, 16000
WHISTLE = {"max_gap_steps": 2}


def remote_path(profile, key):
    return f"{UNOQ_ROOT}/models/{pathlib.Path(profile[key]).parent.name}/{pathlib.Path(profile[key]).name}"


def stretches(folders, holdout, crops, seconds):
    """(name, pcm int16, YIN cents per 10 ms frame) of the stretches with the most whistling."""
    out = []
    for stem in holdout:
        path = next(iter(sorted(q for f in folders for q in pathlib.Path(f).glob(f"{stem}*.npz")
                                if "_play" not in q.stem)), None)
        if path is None:
            print(f"no recording {stem}", flush=True); continue
        pcm = np.load(path)["pcm"].astype(np.int16)
        track = np.r_[np.nan, whistle_track(pcm, SR)]            # frame i ends at sample 320 + 160 i
        n = int(seconds * 100)
        voiced = np.isfinite(track).astype(float)
        score = np.convolve(voiced, np.ones(n), "valid")
        taken = []
        for _ in range(crops):
            order = np.argsort(-score)
            start = next((int(s) for s in order if all(abs(int(s) - t) >= n for t in taken)), None)
            if start is None:
                break
            taken.append(start)
            seg = pcm[start * HOP:start * HOP + FRAME + (n - 1) * HOP]
            out.append((f"{stem}#{len(taken)}", seg, track[start:start + n]))
    return out


def settled(c, v):
    keep = np.zeros(len(c), bool); run = 0
    for i in range(len(c)):
        run = run + 1 if v[i] else 0
        keep[i] = run > 30 and abs(c[i] - c[max(0, i - 15)]) < 50
    return keep


def fetch(path, into):
    local = into / pathlib.Path(path).name
    subprocess.run(["scp", "-q", f"unoq:{path}", str(local)], check=True)
    return local


def measure(npz, ref_c, ref_v, shift, end_should):
    d = np.load(npz)
    h, v = d["0/heard"][0], d["0/valid"][0].astype(bool)
    n = min(len(h), len(ref_c))
    keep = settled(ref_c, ref_v)[:n] & v[:n]
    if keep.sum() < 10:
        return {"hit": None, "hit_in_key": None, "steps": int(len(h))}
    err = h[:n][keep] - (ref_c[:n][keep] + shift)
    return {"hit": round(float(np.mean(np.abs(err) <= 25)), 3),
            "hit_in_key": round(float(np.mean(np.abs(err - np.median(err)) <= 25)), 3),
            "mae_in_key": round(float(np.abs(err - np.median(err)).mean()), 1),
            "end_s": round((len(h) - end_should) / 100, 2), "steps": int(len(h))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=["step8p", "step8p2"])
    ap.add_argument("--host", default="192.168.50.160")
    ap.add_argument("--whistles", nargs="+", default=["runs/real/whistle_data", "runs/real/whistle"])
    ap.add_argument("--holdout", nargs="*", default=["20261124_111404", "20261124_111830", "20261125_115203"])
    ap.add_argument("--crops", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--plays", type=int, default=3)
    args = ap.parse_args()
    profiles = load_profiles()
    songs = stretches(args.whistles, args.holdout, args.crops, args.seconds)
    print(f"{len(songs)} stretches: {[s[0] for s in songs]}", flush=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = ROOT / "runs" / "real" / "eval" / f"{stamp}_step8_rig"; work.mkdir(parents=True, exist_ok=True)
    rig = Remote(args.host, forward=False, timeout=30)
    rows = []
    for step in args.steps:
        p = profiles[step]
        paths = {k: remote_path(p, k + "_path") for k in ("ear_runtime", "memory_runtime", "runtime", "valve_runtime", "heads")}
        tune = rig.call("tune", timeout=60, runtime=paths["runtime"], fan=p.get("fan", .8), ear="nn",
                        ear_runtime=paths["ear_runtime"])
        shift = float(tune["shift_cents"])
        cfg = MemoryConfig(low=p.get("low_cents", -10.0), high=p.get("high_cents", 1200.0))
        print(f"{step}: calibrated, home {tune['home_cents']:.0f} cent, shift {shift:+.0f}", flush=True)
        for name, pcm, track in songs:
            t = teacher(track.astype(np.float64), cfg, WHISTLE)
            if t is None:
                continue
            ref_c = np.r_[np.full(LEAD_STEPS, t[0][0]), t[0], np.full(TAIL_STEPS, t[0][-1])]
            ref_v = np.r_[np.zeros(LEAD_STEPS, bool), t[1], np.zeros(TAIL_STEPS, bool)]
            end_should = LEAD_STEPS + int(np.flatnonzero(t[1])[-1]) + 1 + TAIL_STEPS
            for play in range(1, args.plays + 1):
                out = f"{UNOQ_ROOT}/runs/real/eval/{stamp}_{step}_{name.replace('#', '_')}_play{play}.npz"
                extra = {"clip_pcm16_b64": base64.b64encode(pcm.astype("<i2").tobytes()).decode()} if play == 1 else {}
                rig.call("step8_start", start="listen" if play == 1 else "again", shift_cents=shift, out=out,
                         max_seconds=90, **paths, **extra)
                while True:
                    st = rig.call("step8_status")
                    if not st["active"]:
                        result = st["result"] or {}
                        break
                    time.sleep(.2)
                if not result.get("played"):
                    print(f"  {name} play {play}: nothing played ({result})", flush=True)
                    break
                m = measure(fetch(out, work), ref_c, ref_v, shift, end_should)
                rows.append({"step": step, "song": name, "play": play, **m, "stopped": result.get("stopped")})
                print(f"  {name} play {play}: {m}", flush=True)
                time.sleep(1.0)
    summary = {}
    for step in args.steps:
        for play in range(1, args.plays + 1):
            r = [x for x in rows if x["step"] == step and x["play"] == play and x["hit"] is not None]
            if r:
                summary[f"{step} play {play}"] = {k: round(float(np.mean([x[k] for x in r])), 3)
                                                  for k in ("hit", "hit_in_key", "mae_in_key", "end_s")} | {"songs": len(r)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (work.parent / f"{stamp}_step8_rig.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2,
                                                                    ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
