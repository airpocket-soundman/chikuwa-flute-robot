"""PC: play fixed phrases with one or more steps (models/<step>/) on the rig and compare them.

    python scripts/yamabiko_step_eval.py --steps step1 step2 --targets f80_doremi f80_scale --repeats 2

For every step: one calibration (home note and a sweep, the flute sounds about 10 s), then each target played
``--repeats`` times with the step's profile (actuator runtime, valve network if any, fan, valve pulses).
The targets are models/<step>/target_<name>.npz or runs/real/target_<name>.npz (pushed to the UNO Q first).
Per play, relative to the key it was played in (the median offset):
  rel   mean |heard - target| on the score's notes          interval  error of the intervals between notes
  hit   share of heard note time within 25 cent             glide     share of heard time off by > 50 cent
  cover share of the score's note time the flute was heard  flips     PWM sign changes
Writes runs/real/eval/<stamp>_<step>_<target>.{npz,png} and a summary json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.profiles import load_profiles  # noqa: E402
from flute_rl.yamabiko.remote import Remote  # noqa: E402

REMOTE = "/home/arduino/yamabiko/"


def metrics(h, v, tc, tv, pwm):
    m = v & tv
    if m.sum() < 10:
        return {"rel": None, "interval": None, "hit": 0.0, "glide": None, "cover": float(m.sum() / max(tv.sum(), 1)),
                "flips": int((np.diff(np.sign(pwm)) != 0).sum()), "key": None}
    err = h - tc
    key = float(np.median(err[m]))
    rel = np.abs(err - key)
    starts = np.flatnonzero(np.diff(np.r_[0, tv.astype(int)]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[tv.astype(int), 0]) == -1) + 1
    settled = tv.copy()
    for a in starts:
        settled[a:a + 30] = False
    notes = []
    for a, b in zip(starts, ends):
        k = v[a:b] & settled[a:b]
        notes.append((np.median(h[a:b][k]) if k.sum() >= 5 else np.nan, tc[a]))
    played, wanted = np.array([n[0] for n in notes]), np.array([n[1] for n in notes])
    iv = np.abs(np.diff(played) - np.diff(wanted))
    return {"rel": float(rel[m].mean()), "interval": float(np.nanmean(iv)) if np.isfinite(iv).any() else None,
            "hit": float((rel[m] < 25).mean()), "glide": float((rel[v] > 50).mean()) if v.any() else None,
            "cover": float(m.sum() / max(tv.sum(), 1)), "flips": int((np.diff(np.sign(pwm)) != 0).sum()), "key": key}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.50.160")
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", default=["f80_doremi", "f80_scale"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--candidates", default="runs/cand", help="also steps in this folder (<name>/profile.json)")
    args = ap.parse_args()
    profiles = {**load_profiles(), **load_profiles(args.candidates)}
    for name in args.steps:                      # candidates live outside models/: push their folders too
        folder = pathlib.Path(profiles[name]["runtime_path"]).parent
        if folder.parent.resolve() != pathlib.Path("models").resolve():
            subprocess.run(["ssh", "unoq", f"mkdir -p {REMOTE}models/{name}"], check=True)
            for f in folder.glob("*"):
                if f.suffix in (".json", ".npz"):
                    subprocess.run(["scp", "-q", str(f), f"unoq:{REMOTE}models/{name}/{f.name}"], check=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path("runs/real/eval"); out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh", "unoq", f"mkdir -p {REMOTE}runs/real/eval"], check=True)
    for t in args.targets:
        local = pathlib.Path(f"runs/real/target_{t}.npz")
        subprocess.run(["scp", "-q", str(local), f"unoq:{REMOTE}runs/real/target_{t}.npz"], check=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rig = Remote(args.host, forward=False)
    summary = {}
    try:
        for name in args.steps:
            p = profiles[name]
            runtime = f"{REMOTE}models/{name}/{p['runtime']}"
            valve = f"{REMOTE}models/{name}/{p['valve_runtime']}" if p.get("valve_runtime") else None
            rig.valve_positions(*p["valve_us"], move_ms=20)
            tune = rig.tune(runtime=runtime, fan=p["fan"])
            print(f"{name}: home {440 * 2 ** (tune['home_cents'] / 1200):.1f} Hz, shift {tune['shift_cents']:+.1f} cent", flush=True)
            for t in args.targets:
                rec = f"{REMOTE}runs/real/eval/{stamp}_{name}_{t}.npz"
                rig.play(runtime, f"{REMOTE}runs/real/target_{t}.npz", out=rec, repeats=args.repeats, fan=p["fan"],
                         shift_cents=tune["shift_cents"], warmup_s=2.0, ear=p.get("ear", "yin"), valve_runtime=valve)
                local = out / f"{stamp}_{name}_{t}.npz"
                subprocess.run(["scp", "-q", "unoq:" + rec, str(local)], check=True)
                d = np.load(local)
                n = int(d["count"])
                fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), squeeze=False)
                for k in range(n):
                    h, v, tc, tv, pwm, valve_log = (d[f"{k}/{x}"][0] for x in ("heard", "valid", "target_cents",
                                                                               "target_voice", "pwm", "valve"))
                    m = metrics(h, v, tc, tv, pwm)
                    summary.setdefault(name, {}).setdefault(t, []).append(m)
                    line = " ".join(f"{a} {b:.2f}" if isinstance(b, float) else f"{a} {b}" for a, b in m.items())
                    print(f"  {t} play {k + 1}: {line}", flush=True)
                    ax = axes[k, 0]; tt = np.arange(len(tc)) * .01
                    key = m["key"] or 0.0
                    ax.plot(tt, np.where(tv, tc, np.nan), "k", lw=2)
                    ax.plot(tt[v], h[v] - key, ".", ms=3)
                    ax.fill_between(tt, -150, -150 + 60 * valve_log, color="tab:green", alpha=.3, step="mid")
                    a2 = ax.twinx(); a2.plot(tt, pwm, "r", lw=.5, alpha=.5); a2.set_ylim(-1.1, 1.1)
                    ax.set_ylim(-160, 1400); ax.grid(alpha=.3); ax.set_title(f"{name} {t} play {k + 1}: {line}", fontsize=8)
                plt.tight_layout(); plt.savefig(out / f"{stamp}_{name}_{t}.png", dpi=65); plt.close()
    finally:
        rig.stop(force=True)
        s = rig.status()
        print("after:", {k: s[k] for k in ("fan_switch", "valve_open", "applied_pwm")})
        rig.close()
    (out / f"{stamp}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("mean per step and target:")
    for name, per in summary.items():
        for t, ms in per.items():
            mean = {k: np.mean([m[k] for m in ms if m[k] is not None]) for k in ("rel", "interval", "hit", "glide", "cover", "flips")}
            print(f"  {name:6s} {t:12s} " + " ".join(f"{k} {v:.2f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()
