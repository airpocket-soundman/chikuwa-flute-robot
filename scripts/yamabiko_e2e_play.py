"""Play a raw whistle recording with the single E2E Yamabiko model.

Unlike yamabiko_play.py, this path never calls YIN, never constructs a pitch
track or score, and never uses the tube/actuator model.  The network receives
the complete raw demonstration and the latest raw microphone samples, and
directly emits PWM, valve and finished probability.

Mechanical homing and firmware PWM limiting remain outside the model as safety
operations.  A checkpoint must first be trained with train_yamabiko_e2e.py.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
import time

import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from yamabiko_collect import add_rig_args, close_rig, open_rig  # noqa: E402
from yamabiko_play import wait_trigger  # noqa: E402

from flute_rl.sim import DT  # noqa: E402
from flute_rl.yamabiko.e2e_io import FRAME  # noqa: E402
from flute_rl.yamabiko.session import HOME_STEPS  # noqa: E402


def record_raw(link, seconds: float) -> np.ndarray:
    """Record a fixed raw window.  No onset, silence or pitch decision is made."""
    print(f"whistle now ({seconds:g} s raw recording)")
    start = link.audio.end
    until = time.monotonic() + seconds
    while (left := until - time.monotonic()) > 0:
        link.poll(min(left, 0.02))
    return link.audio.span(start, link.audio.end)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_rig_args(ap)
    ap.add_argument("--policy", default="runs/yamabiko_e2e.npz",
                    help="UNO Q NumPy .npz (recommended), or a PyTorch .pt checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--songs", type=int, default=0, help="0 = until Ctrl-C")
    ap.add_argument("--record-seconds", type=float, default=6.0)
    ap.add_argument("--tail-seconds", type=float, default=2.0)
    ap.add_argument("--done", type=float, default=0.8, help="model finished-probability threshold")
    ap.add_argument("--done-hold", type=int, default=5, help="consecutive finished frames")
    ap.add_argument("--key", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if pathlib.Path(args.policy).suffix == ".npz":
        from flute_rl.yamabiko.e2e_numpy import NumpyE2ERuntime
        runtime = NumpyE2ERuntime.load(args.policy)
    else:
        from flute_rl.yamabiko.e2e import E2ERuntime
        runtime = E2ERuntime.load(args.policy, args.device)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = pathlib.Path(args.out or f"runs/real/e2e_play_{stamp}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    references, rows = [], []

    link, rig = open_rig(args)
    rig.pitch_enabled = False  # raw PCM goes to the model; do not spend the 10 ms budget on YIN
    try:
        # Keep the control clock and microphone stream alive while the fan settles.
        for _ in range(int(round(args.settle / DT))):
            rig.step(np.array([0.0]), np.array([False]))
        song = 0
        while args.songs <= 0 or song < args.songs:
            wait_trigger(link, args.key)
            reference = record_raw(link, args.record_seconds)
            references.append(reference)
            reference_steps = runtime.start_reference(reference, keep_adaptation=True)

            # A fixed end-stop operation is a safety invariant, not musical
            # interpretation.  The learned recurrent rig memory is retained.
            rig.resync()
            for _ in range(HOME_STEPS):
                rig.step(np.array([-1.0]), np.array([False]))

            maximum = reference_steps + int(round(args.tail_seconds / DT))
            finished = 0
            for t in range(maximum):
                raw_self = link.audio.latest(FRAME).astype(np.float32)
                pwm, valve, done = runtime.act(raw_self)
                rig.step(np.array([pwm]), np.array([valve]))
                rows.append((song, t, pwm, valve, done, link.audio.end))
                finished = finished + 1 if done >= args.done else 0
                if finished >= args.done_hold:
                    break
            rig.safe()
            np.savez_compressed(
                out_path,
                actions=np.asarray(rows, dtype=float),
                **{f"reference_{i}": x for i, x in enumerate(references)},
            )
            print(f"song {song}: {t + 1} control steps, saved {out_path}")
            song += 1
    finally:
        close_rig(link, rig)


if __name__ == "__main__":
    main()
