"""What did a learned feedback policy learn about *when to trust the ear*?

For each step it records the feedback-gain scale the network chose (1 = the
classic gain, 0 = ignore what it hears, 2 = double), and groups the steps by
situation: rest, note onset (the plunger is still moving to a new note),
steady note, and whether the delayed pitch is available at all. It also
groups by the rig's listening delay, to see whether the network adapts to it.

    python scripts/trust_analysis.py runs/fb_gru.npz --episodes 60
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl import FluteEnv  # noqa: E402
from flute_rl.feedback import FeedbackPolicy, FeedbackResidualPolicy  # noqa: E402
from flute_rl.policy import load_net  # noqa: E402

ONSET_STEPS = 15  # steps after a target change counted as "note onset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=3_000_000)
    ap.add_argument("--harsh", type=float, default=0.0, help="0..1: rig effects the controllers do not model")
    args = ap.parse_args()

    net, meta = load_net(args.net)
    rows = []  # (take, situation, fb_valid, delay, gain_scale)
    for i in range(args.episodes):
        env = FluteEnv(progress=0.8, takes=args.takes, feedback=True, harsh=args.harsh)
        base = FeedbackPolicy(fb_gain=float(meta["fb_gain"]), ilc_gain=float(meta["ilc_gain"]))
        pol = FeedbackResidualPolicy(base, net, scale=float(meta["scale"]), horizon=int(meta["horizon"]),
                                     history=int(meta.get("history", 0)))
        obs, _ = env.reset(seed=args.seed0 + i)
        pol.reset(env)
        tgt = env.target
        change = np.zeros(len(tgt), bool)
        prev = np.nan
        for t, v in enumerate(tgt):
            if np.isfinite(v) and not (np.isfinite(prev) and abs(v - prev) < 1.0):
                change[t:t + ONSET_STEPS] = True
            prev = v
        done = False
        while not done:
            t, take = env.t, env.take
            valid = obs[env.obs_layout["fb_valid"]][0] > 0.5
            a = pol.act(obs)
            if not np.isfinite(tgt[t]):
                sit = "rest"
            elif change[t]:
                sit = "note onset"
            else:
                sit = "steady note"
            rows.append((take, sit, valid, env.params.obs_delay, base.gain_scale))
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc

    take = np.array([r[0] for r in rows])
    sit = np.array([r[1] for r in rows])
    valid = np.array([r[2] for r in rows])
    delay = np.array([r[3] for r in rows])
    gain = np.array([r[4] for r in rows])
    print(f"{args.net}: feedback-gain scale chosen by the network (1 = classic)")
    print(f"{'':14s}" + "".join(f"take {k + 1:<10d}" for k in range(args.takes)))
    for s in ("rest", "note onset", "steady note"):
        for vv, lab in ((True, "heard"), (False, "not heard")):
            cells = []
            for k in range(args.takes):
                m = (take == k) & (sit == s) & (valid == vv)
                cells.append(f"{gain[m].mean():5.2f} (n={m.sum():5d})" if m.any() else " " * 15)
            print(f"{s + ', ' + lab:26s}" + "  ".join(cells))
    print("by listening delay (steady notes, heard):")
    for d in np.unique(delay):
        m = (delay == d) & (sit == "steady note") & valid
        m0 = m & (take == 0)
        print(f"  delay {d} steps: gain scale {gain[m].mean():5.2f}  (take 1 only {gain[m0].mean():5.2f}, n={m.sum()})")


if __name__ == "__main__":
    main()
