"""Train the pitch-hearing network on synthetic audio and compare it with YIN (needs PyTorch).

    python scripts/train_pitch_net.py --ear self --clips 1500 --epochs 15 --out runs/pitchnet_self.pt
    python scripts/train_pitch_net.py --ear source --kinds recorder --out runs/pitchnet_recorder.pt
    python scripts/train_pitch_net.py --ear source --kinds recorder,whistle,hum,bird --out runs/pitchnet_sources.pt

Training and test clips come from different seeds. YIN is scored on the
same test frames with the same metrics.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.pitch import yin_frame  # noqa: E402
from flute_rl.pitchnet import SELF_EAR, SOURCE_EAR, PitchNet, decode, make_dataset, pitch_metrics, soft_targets  # noqa: E402
from flute_rl.sim import hz_to_cents  # noqa: E402


def _make(task):
    n, kinds, seed = task
    return make_dataset(n, kinds, seed)


def build(n_clips: int, kinds: tuple[str, ...], seed: int, workers: int):
    chunks = [(max(1, n_clips // workers), kinds, seed * 1000 + i) for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_make, chunks)
    return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))


def yin_track(frames: np.ndarray, ear, limit: int) -> np.ndarray:
    out = np.full(min(limit, len(frames)), np.nan)
    for i in range(len(out)):
        fr = frames[i]
        if np.sqrt(np.mean(fr * fr)) < 0.01:
            continue
        f0, _ = yin_frame(fr, ear.sr, ear.f_lo, min(ear.f_hi, 0.45 * ear.sr))
        out[i] = hz_to_cents(f0) if np.isfinite(f0) else np.nan
    return out


def report(label: str, m: dict) -> None:
    print(f"  {label:8s} median {m['median_cents']:5.1f} c  p90 {m['p90_cents']:6.1f} c  octave errors {m['octave_errors']:.3f}  "
          f"voiced recall {m['voiced_recall']:.3f}  false voiced {m['false_voiced']:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ear", choices=("self", "source"), default="self")
    ap.add_argument("--kinds", default="recorder", help="source kinds, comma separated (ear=source)")
    ap.add_argument("--clips", type=int, default=1500)
    ap.add_argument("--test-clips", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yin-frames", type=int, default=20000)
    ap.add_argument("--init", default=None, help="continue from a saved network (adding timbres step by step)")
    ap.add_argument("--out", default="runs/pitchnet.pt")
    args = ap.parse_args()

    ear = SELF_EAR if args.ear == "self" else SOURCE_EAR
    kinds = ("self",) if args.ear == "self" else tuple(args.kinds.split(","))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    xtr, ytr, _ = build(args.clips, kinds, 1, args.workers)
    xte, yte, kte = build(args.test_clips, kinds, 2, args.workers)
    print(f"ear {args.ear} {kinds}: {len(xtr)} train / {len(xte)} test frames of {ear.frame} samples @ {ear.sr} Hz "
          f"({time.time() - t0:.0f}s to synthesise), device {dev}", flush=True)

    net = PitchNet(ear, args.width).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev)["state"])
        print(f"continuing from {args.init}", flush=True)
    print(f"parameters: {sum(p.numel() for p in net.parameters())}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    Xtr, Ytr = torch.from_numpy(xtr), torch.from_numpy(ytr)
    Xte = torch.from_numpy(xte).to(dev)
    lossf = torch.nn.BCEWithLogitsLoss()

    def test_pred() -> np.ndarray:
        net.eval()
        with torch.no_grad():
            out = torch.cat([decode(net(Xte[i:i + 4096]), ear) for i in range(0, len(Xte), 4096)])
        net.train()
        return out.cpu().numpy()

    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            b = perm[i:i + args.batch]
            x, y = Xtr[b].to(dev, non_blocking=True), Ytr[b].to(dev, non_blocking=True)
            loss = lossf(net(x), soft_targets(y, ear))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = pitch_metrics(test_pred(), yte)
        print(f"epoch {ep:2d} loss {tot / (len(perm) // args.batch):.4f}", flush=True)
        report("network", m)

    pred = test_pred()
    if len(kinds) > 1:  # per timbre: shows whether an earlier timbre was forgotten
        for i, k in enumerate(kinds):
            report(k, pitch_metrics(pred[kte == i], yte[kte == i]))
    n = min(args.yin_frames, len(xte))
    report("network", pitch_metrics(pred[:n], yte[:n]))
    report("YIN", pitch_metrics(yin_track(xte, ear, n), yte[:n]))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "ear": args.ear, "width": args.width, "kinds": kinds}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
