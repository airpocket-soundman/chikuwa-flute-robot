"""Export a trained E2E PyTorch checkpoint to the UNO Q NumPy runtime."""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_numpy import NumpyE2ERuntime, export_numpy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="runs/yamabiko_e2e.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    src = pathlib.Path(args.checkpoint)
    out = pathlib.Path(args.out) if args.out else src.with_suffix(".npz")
    checkpoint = torch.load(src, map_location="cpu")
    model = E2EImitator.from_checkpoint(checkpoint).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    export_numpy(model, out, source=str(src))
    runtime = NumpyE2ERuntime.load(out)
    size = out.stat().st_size
    print(f"saved {out} ({size / 1024:.1f} KiB, {sum(v.size for v in runtime.model.a.values()):,} float32 values)")


if __name__ == "__main__":
    main()
