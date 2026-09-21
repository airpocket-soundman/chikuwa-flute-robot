"""Benchmark the dependency-free E2E policy on UNO Q Linux.

Pass criterion for live control is a comfortably sub-10 ms p99 action time.
Reference encoding happens before playing and is reported separately.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.yamabiko.e2e_io import FRAME, SAMPLE_RATE  # noqa: E402
from flute_rl.yamabiko.e2e_numpy import NumpyE2ERuntime  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("policy", nargs="?", default="runs/yamabiko_e2e.npz")
    ap.add_argument("--reference-seconds", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 0.05, int(SAMPLE_RATE * args.reference_seconds)).astype(np.float32)
    own = rng.normal(0.0, 0.05, (args.steps, FRAME)).astype(np.float32)
    runtime = NumpyE2ERuntime.load(args.policy)

    t0 = time.perf_counter()
    n = runtime.start_reference(reference)
    reference_ms = 1000.0 * (time.perf_counter() - t0)
    for i in range(10):
        runtime.act(own[i])
    times = np.empty(args.steps)
    for i in range(args.steps):
        t0 = time.perf_counter()
        runtime.act(own[i])
        times[i] = 1000.0 * (time.perf_counter() - t0)
    print(f"policy: {args.policy}  reference: {n} frames / {reference_ms:.1f} ms")
    print(f"action: median {np.median(times):.3f} ms  p95 {np.percentile(times, 95):.3f} ms  "
          f"p99 {np.percentile(times, 99):.3f} ms  max {times.max():.3f} ms")
    if np.percentile(times, 99) >= 10.0:
        raise SystemExit("FAIL: p99 does not fit the 10 ms control period")
    print("PASS: p99 fits the 10 ms control period")


if __name__ == "__main__":
    main()
