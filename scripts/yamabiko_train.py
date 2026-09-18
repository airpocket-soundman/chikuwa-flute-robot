"""Train the recurrent policy of Yamabiko No.1 with evolution strategies (numpy only).

Each generation, every candidate plays the same session: power on, then
--songs different songs on --rigs random rigs, hidden state carried across
songs. Later songs weigh more (--weight-growth), and some rigs are swapped at
a random song boundary without telling the policy (--swap-prob).

Heavy: run it on a training machine, e.g.

    python scripts/yamabiko_train.py --gens 300 --pop 64 --rigs 8 --songs 5 --workers 16 --out runs/yamabiko_gru.npz

With a CUDA GPU (PyTorch), all candidates x rigs are played at once
(flute_rl/yamabiko/torch_session.py), which allows many more rigs per candidate:

    python scripts/yamabiko_train.py --device cuda --gens 300 --pop 256 --rigs 64 --songs 5 --out runs/yamabiko_gru.npz

Evaluation (every --eval-every generations) prints, per song, the mean |cents|,
the first-step onset error and the error on the first note of the song, of the policy with memory carried, the same
weights with memory cleared at every song, and the external-memory reference.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time
from contextlib import nullcontext
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.targets import load_bank, make_bank, save_bank  # noqa: E402
from flute_rl.yamabiko import (ExternalFit, GRUPolicy, Rig, RigParams, TaskSpec, draw_pieces, fitness,  # noqa: E402
                               make_schedule, run_session, song_metrics)
from flute_rl.yamabiko.learn import Adam, es_gradient, song_weights  # noqa: E402

_W: dict = {}


def _init(bank_file):
    _W["bank"] = load_bank(bank_file) if bank_file else None


def _fitness(task):
    thetas, spec, seed = task
    return fitness(thetas, spec, seed, _W["bank"])


def evaluate(theta: np.ndarray, spec: TaskSpec, rigs: int, seed: int) -> dict:
    """Per-song metrics on fixed evaluation rigs and newly generated songs (not from the training bank)."""
    rng = np.random.default_rng(seed)
    params = RigParams.sample(rng, rigs, spec.spread, spec.harsh)
    sched = make_schedule(draw_pieces(rng, rigs, spec.songs, None, spec.progress))
    out = {}
    for name, ctrl in (("gru", GRUPolicy(theta, spec.hidden, carry=True)),
                       ("gru_reset", GRUPolicy(theta, spec.hidden, carry=False)),
                       ("external_fit", ExternalFit())):
        res = run_session(Rig(params, np.random.default_rng(seed + 1)), sched, ctrl, keep_logs=True,
                          first_weight=spec.first_weight)
        w = song_weights(spec.songs, spec.weight_growth)
        out[name] = {"songs": song_metrics(res["logs"], sched), "score": float((res["reward"] @ w / w.sum()).mean())}
    return out


def report(gen: int, ev: dict) -> None:
    for name, r in ev.items():
        cells = "  ".join(f"{d['mean_abs']:6.1f}/{d['onset1']:5.1f}/{d['first1']:5.1f}" for d in r["songs"])
        print(f"  gen {gen:4d} {name:12s} score {r['score']:+.4f}  |cents|/onset1/first1 per song: {cells}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=64, help="population size (even)")
    ap.add_argument("--rigs", type=int, default=8, help="rigs per candidate (shared by all candidates)")
    ap.add_argument("--songs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--memory-steps", type=float, default=0.0,
                    help="chrono initialisation of the GRU update gate up to this many steps (0: plain bias 1)")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.005)
    ap.add_argument("--harsh", type=float, default=0.0)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--swap-prob", type=float, default=0.25)
    ap.add_argument("--weight-growth", type=float, default=1.0, help="song k weight = 1 + growth * k / (songs - 1)")
    ap.add_argument("--first-weight", type=float, default=1.0,
                    help="how much more the first note of each song counts in the reward")
    ap.add_argument("--bank", type=int, default=1000, help="train on a fixed set of this many songs (0: new songs)")
    ap.add_argument("--bank-file", type=str, default="runs/target_bank.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu", help="cpu (numpy, --workers processes) or cuda (PyTorch)")
    ap.add_argument("--eval-rigs", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default=None, help="continue from a saved policy")
    ap.add_argument("--out", type=str, default="runs/yamabiko_gru.npz")
    args = ap.parse_args()
    if args.pop % 2 or (args.device == "cpu" and args.pop < args.workers):
        ap.error("--pop must be even and at least --workers")

    spec = TaskSpec(rigs=args.rigs, songs=args.songs, hidden=args.hidden, harsh=args.harsh, spread=args.spread,
                    swap_prob=args.swap_prob, weight_growth=args.weight_growth, first_weight=args.first_weight)
    rng = np.random.default_rng(args.seed)
    theta = GRUPolicy.init_params(rng, args.hidden, args.memory_steps)
    if args.init:
        d = np.load(args.init)
        if int(d["hidden"]) != args.hidden:
            ap.error("--hidden does not match the --init policy")
        theta = d["theta"]
    opt = Adam(theta.size, args.lr)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bank_file = None
    if args.bank:
        bank_path = pathlib.Path(args.bank_file)
        if not bank_path.exists() or len(load_bank(bank_path)) != args.bank:
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            save_bank(bank_path, make_bank(args.bank))
        bank_file = str(bank_path)
    print(f"params {theta.size}  pop {args.pop}  rigs {args.rigs}  songs {args.songs}  "
          f"{'workers ' + str(args.workers) if args.device == 'cpu' else 'device ' + args.device}", flush=True)
    if args.device == "cpu":
        pool = Pool(args.workers, initializer=_init, initargs=(bank_file,))

        def generation_fitness(thetas, seed):
            chunks = np.array_split(np.arange(args.pop), args.workers)
            return np.concatenate(pool.map(_fitness, [(thetas[c], spec, seed) for c in chunks]))
    else:
        from flute_rl.yamabiko import torch_session

        pool = nullcontext()
        bank = load_bank(bank_file) if bank_file else None

        def generation_fitness(thetas, seed):
            return torch_session.fitness(thetas, spec, seed, bank, device=args.device)

    eval_seed = 10_000_019
    best = -np.inf
    with pool:
        half = args.pop // 2
        for g in range(1, args.gens + 1):
            t0 = time.time()
            seed = int(rng.integers(0, 2**31 - 1))
            eps = rng.standard_normal((half, theta.size))
            thetas = (theta[None, :] + args.sigma * np.stack([eps, -eps], axis=1)).reshape(args.pop, -1)
            f = generation_fitness(thetas, seed)
            scores = f.reshape(half, 2)
            grad = es_gradient(eps, scores, args.sigma) - args.weight_decay * theta / args.sigma
            theta = opt.step(theta, grad)
            print(f"gen {g:4d}  pop mean {scores.mean():+.4f}  best {scores.max():+.4f}  ({time.time() - t0:.1f}s)", flush=True)
            if g % args.eval_every == 0 or g == args.gens:
                ev = evaluate(theta, spec, args.eval_rigs, eval_seed)
                report(g, ev)
                if ev["gru"]["score"] > best:
                    best = ev["gru"]["score"]
                    np.savez(out, theta=theta, hidden=args.hidden, gen=g, score=best,
                             spec=np.array(str(dataclasses.asdict(spec))))
                    print(f"  saved {out} (best so far)", flush=True)


if __name__ == "__main__":
    main()
