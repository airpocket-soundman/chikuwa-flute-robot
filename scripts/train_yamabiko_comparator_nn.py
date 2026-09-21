"""Continue only the signed-error comparator without changing passed stages."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_audio import RawSelfAudio  # noqa: E402
from flute_rl.yamabiko.staged_nn import ErrorComparator, FeedForwardPolicy, FeedbackResidualPolicy, ReferenceMemory, StagedConfig  # noqa: E402
from train_yamabiko_staged_nn import comparator_metrics, make_samples, train_comparator  # noqa: E402


@torch.inference_mode()
def encoded_training_set(samples, ear, memory, rng, device):
    targets, owns, truths = [], [], []
    offsets = np.asarray([-400, -200, -100, -50, 50, 100, 200, 400], np.float32)
    for item in samples:
        n = len(item.target); cents = item.target[:, 0].numpy() * 600.0 + 1300.0
        voiced = item.target[:, 1].numpy() > .5
        offset = rng.choice(offsets, n); actual = np.where(voiced, cents + offset, 0.0)
        frames = RawSelfAudio(1, rng, domain=1.0).render(actual[None], voiced[None])[0]
        own = ear.audio_features(torch.from_numpy(frames).to(device))[:, -2:]
        _, stored = memory(item.ear[None].to(device), torch.tensor([n], device=device))
        target = memory.decode(stored)[0]
        observed = np.r_[0.0, actual[:-1]]; valid = voiced & np.r_[False, voiced[:-1]]
        targets.append(target[torch.from_numpy(valid).to(device)].cpu())
        owns.append(own[torch.from_numpy(valid).to(device)].cpu())
        truths.append(torch.from_numpy(((cents - observed) / 600.0).astype(np.float32))[valid])
    return torch.cat(targets), torch.cat(owns), torch.cat(truths)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="runs/yamabiko_staged_nn.pt"); ap.add_argument("--out", default=None)
    ap.add_argument("--report", default="runs/yamabiko_comparator_report.json")
    ap.add_argument("--songs", type=int, default=256); ap.add_argument("--valid-songs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--seed", type=int, default=9137)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = args.out or args.init
    ck = torch.load(args.init, map_location=args.device); cfg = StagedConfig(**ck["config"])
    memory, comparator = ReferenceMemory(cfg).to(args.device), ErrorComparator(cfg).to(args.device)
    memory.load_state_dict(ck["memory"]); comparator.load_state_dict(ck["comparator"]); memory.eval()
    ear = E2EImitator.from_checkpoint(torch.load(ck["ear_checkpoint"], map_location=args.device), args.device).eval()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    train = make_samples(ear, rng, args.songs, args.device, .8, random_rig=False)
    valid = make_samples(ear, np.random.default_rng(args.seed + 100_000), args.valid_songs, args.device, .8, random_rig=False)
    target, own, truth = encoded_training_set(train, ear, memory, rng, args.device)
    optimizer = torch.optim.AdamW(comparator.parameters(), lr=1e-3, weight_decay=1e-6)
    for step in range(1, args.steps + 1):
        ids = torch.from_numpy(rng.integers(len(truth), size=2048)).long()
        pred = comparator(target[ids].to(args.device), own[ids].to(args.device))[:, 0]
        loss = F.smooth_l1_loss(pred, truth[ids].to(args.device))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 200 == 0: print(f"step {step:4d} loss {float(loss.detach()):.5f}", flush=True)
    metrics = comparator_metrics(comparator.eval(), ear, memory, valid, np.random.default_rng(args.seed + 200_000), args.device)
    metrics["pass"] = metrics["error_mae_cents"] < 60 and metrics["error_sign_accuracy"] >= .9
    ck["comparator"] = comparator.state_dict(); ck.setdefault("metrics", {})["comparator_continued"] = metrics
    torch.save(ck, out); pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"saved {out}")


if __name__ == "__main__": main()
