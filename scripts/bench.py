"""Latency / throughput benchmark of every component that runs on the Linux side.

Run this on the UNO Q itself to replace the PC numbers in the design notes:

    python3 scripts/bench.py
"""
from __future__ import annotations

import pathlib
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl import MLP, FluteEnv, PhysicsPriorPolicy, ResidualPolicy, rollout  # noqa: E402
from flute_rl.pitch import yin_frame  # noqa: E402


def timeit(fn, n: int) -> float:
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> None:
    print(f"python {platform.python_version()}  numpy {np.__version__}  machine {platform.machine()}  {platform.processor()}")
    env = FluteEnv(progress=0.8, takes=3)
    obs, _ = env.reset(seed=0)
    base = PhysicsPriorPolicy()
    base.reset(env)
    rows = []

    rows.append(("env.step (simulator)", timeit(lambda: env.step([0.3, 0.0]) if env.t < len(env.target) - 1 else env.reset(), 2000)))
    rows.append(("physics prior act", timeit(lambda: base.act(obs), 2000)))
    for hidden in (32, 64, 128):
        net = MLP(env.obs_dim + 2, 2, hidden=hidden)
        pol = ResidualPolicy(base, net)
        rows.append((f"residual policy act (hidden {hidden}, {net.n_params} params)", timeit(lambda: pol.act(obs), 2000)))
    net = MLP(env.obs_dim + 2, 2, hidden=32)
    x = np.concatenate([obs, [0.0, 0.0]])
    batch = np.tile(x, (1000, 1))
    rows.append(("MLP forward, batch 1000 (whole-take profile)", timeit(lambda: net(batch), 200)))

    sr = 16000
    t = np.arange(1024) / sr
    frame = 0.5 * np.sin(2 * np.pi * 1000 * t)
    rows.append(("YIN pitch, 1024-sample frame @16 kHz", timeit(lambda: yin_frame(frame, sr), 500)))

    pol = ResidualPolicy(PhysicsPriorPolicy(), MLP(env.obs_dim + 2, 2, hidden=32))
    t0 = time.perf_counter()
    steps = 0
    for s in range(5):
        steps += len(rollout(env, pol, seed=s)["rewards"])
    per_step = (time.perf_counter() - t0) / steps
    rows.append(("rollout, per simulated step (env + policy)", per_step))

    for name, sec in rows:
        print(f"{name:52s} {sec * 1e6:10.1f} us")
    gen = per_step * 16 * 4 * steps / 5
    print(f"\nES generation estimate (pop 16 x 4 episodes, 3 takes): {gen:.1f} s")


if __name__ == "__main__":
    main()
