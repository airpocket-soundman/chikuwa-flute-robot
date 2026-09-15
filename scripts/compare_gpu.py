"""Compare live-feedback policies on the GPU, including ones that hear through the ear network.

Every model plays the same rigs and targets (built like FluteEnv(seed) would),
with the same noise seed. Prints mean |cents| per take with a 95% bootstrap
interval, the paired difference against the first model, and the sounding rates.

    python scripts/compare_gpu.py --harsh 1.0 --model "hand-made=classic" --model "hand-made, ear=classic+ear" \
        --model "GRU=runs/harsh_gru.npz" --model "GRU, ear=runs/hear_gru.npz"

"classic" is the hand-made feedback (no network); "+ear" makes a model hear through
runs/pitchnet_self.pt (networks trained with --hearing hear automatically).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.policy import load_net  # noqa: E402
from flute_rl.sim import FluteParams, FluteSim  # noqa: E402
from flute_rl.targets import make_target, sample_level  # noqa: E402
from flute_rl.torch_env import BatchAgent, BatchFluteEnv, BatchGRU, BatchMLP, rig_from_seed, run_episodes  # noqa: E402

DEFAULT_EAR = "runs/pitchnet_self.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help='"label=path.npz", "label=classic", or add "+ear"')
    ap.add_argument("--harsh", type=float, default=1.0)
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=2_000_000)
    ap.add_argument("--progress", type=float, default=0.8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seeds = range(args.seed0, args.seed0 + args.episodes)
    rigs = [rig_from_seed(s, harsh=args.harsh) for s in seeds]
    targets = []
    for s in seeds:  # replay FluteEnv.reset's use of its rng so the targets match compare.py
        r = np.random.default_rng(s)
        FluteSim(FluteParams.sample(r, harsh=args.harsh), r)
        targets.append(make_target(r, sample_level(r, args.progress)))

    ears = {}
    results = {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        hear = path.endswith("+ear")
        path = path[:-4] if hear else path
        net, kw = None, {}
        if path != "classic":
            n, meta = load_net(path)
            kw = dict(fb_gain=float(meta["fb_gain"]), ilc_gain=float(meta["ilc_gain"]), scale=float(meta["scale"]),
                      horizon=int(meta["horizon"]), history=int(meta.get("history", 0)))
            ear_path = str(meta.get("hearing", "")) or (DEFAULT_EAR if hear else "")
            hear = bool(ear_path)
            theta = torch.tensor(np.tile(n.get_flat(), (len(rigs), 1)), device=args.device, dtype=torch.float32)
            net = (BatchGRU if n.arch == "gru" else BatchMLP)(theta, n.in_dim, n.out_dim, n.hidden)
        ear = None
        if hear:
            from flute_rl.torch_audio import Ear  # noqa: E402
            ear_path = ear_path if path != "classic" else DEFAULT_EAR
            ears.setdefault(ear_path, Ear(ear_path, args.device))
            ear = ears[ear_path]
        gen = torch.Generator(device=args.device).manual_seed(0)
        env = BatchFluteEnv(rigs, targets, takes=args.takes, device=args.device, generator=gen, hearing=ear)
        results[label] = run_episodes(env, BatchAgent(env, net=net, **kw)).cpu().numpy()  # (B, takes, 4)
        print(f"done: {label}", flush=True)

    rng = np.random.default_rng(0)

    def ci(x):
        x = x[np.isfinite(x)]
        m = rng.choice(x, (2000, len(x))).mean(1)
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    names = list(results)
    w = max(len(n) for n in names) + 2
    print(f"\n{args.episodes} rigs x {args.takes} takes, harsh {args.harsh}\nmean |cents| per take [95% CI]")
    for n in names:
        cells = [f"{np.nanmean(results[n][:, k, 0]):6.1f} [{ci(results[n][:, k, 0])[0]:5.1f},{ci(results[n][:, k, 0])[1]:5.1f}]"
                 for k in range(args.takes)]
        print(f"  {n:{w}s}" + "  ".join(cells))
    ref = names[0]
    print(f"paired difference vs '{ref}' (negative = better)")
    for n in names[1:]:
        cells = []
        for k in range(args.takes):
            d = results[n][:, k, 0] - results[ref][:, k, 0]
            lo, hi = ci(d)
            cells.append(f"{np.nanmean(d):+6.1f} [{lo:+5.1f},{hi:+5.1f}]")
        print(f"  {n:{w}s}" + "  ".join(cells))
    for title, j in (("reward/step", 1), ("sounding rate in notes", 2), ("sounding inside short gaps", 3)):
        print(title)
        for n in names:
            print(f"  {n:{w}s}" + "  ".join(f"{np.nanmean(results[n][:, k, j]):.3f}" for k in range(args.takes)))


if __name__ == "__main__":
    main()
