"""PC: train the ear network (flute_rl/yamabiko/ear_nn.py) on the rig's recordings and compare it with YIN.

    python scripts/yamabiko_ear_data.py                          # once: runs/ear/dataset.npz
    python scripts/yamabiko_ear_train.py --out runs/ear/ear_01

Voiced frames teach the pitch (cross-entropy on 25-cent bins, a Gaussian of one bin around the label), quiet
frames teach "not sounding"; unsure frames teach nothing.  Every batch is augmented: gain, white noise and a
50..110 Hz rumble like the fan's.  On the held-out recordings it is compared with the rig's YIN ear
(hw.heard_pitch, the same 20 ms): share of labelled voiced frames heard within 25 cent, median error, and
voicing errors.  Writes <out>.pt, <out>.json and <out>_runtime.npz (NumPy, for the UNO Q).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.ear_nn import BIN_CENTS, BINS, LOW_CENTS, Ear, EarConfig, decode, export, torch_model  # noqa: E402
from flute_rl.yamabiko.hw import heard_pitch  # noqa: E402

SR = 16000


def augment(x, gen):
    b, n = x.shape
    t = torch.arange(n, device=x.device) / SR
    gain = 10 ** (torch.empty(b, 1, device=x.device).uniform_(-18, 12, generator=gen) / 20)
    rms = x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-4)
    snr = 10 ** (torch.empty(b, 1, device=x.device).uniform_(10, 40, generator=gen) / 20)
    noise = torch.randn(b, n, device=x.device, generator=gen) * rms / snr
    f = torch.empty(b, 1, device=x.device).uniform_(50, 110, generator=gen)
    phase = torch.rand(b, 1, device=x.device, generator=gen) * 6.283
    rumble = torch.sin(6.283 * f * t[None] + phase) * rms * torch.empty(b, 1, device=x.device).uniform_(0, 1.5, generator=gen)
    return (x + noise + rumble) * gain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/ear/dataset.npz")
    ap.add_argument("--out", default="runs/ear/ear_01")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); dev = args.device
    torch.manual_seed(args.seed)
    d = np.load(args.data)
    x = d["x"].astype(np.float32) / 32768.0
    cents, state, split = d["cents"], d["state"], d["split"]
    use = state >= 0
    train = np.flatnonzero(use & (split == 0)); test = np.flatnonzero(split == 1)
    X = torch.tensor(x, device=dev)
    target_bin = torch.tensor(np.nan_to_num((cents - LOW_CENTS) / BIN_CENTS, nan=0.0), device=dev, dtype=torch.float32)
    voiced = torch.tensor(state == 1, device=dev)
    model = torch_model(EarConfig()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    gen = torch.Generator(dev).manual_seed(args.seed)
    grid = torch.arange(BINS, device=dev, dtype=torch.float32)
    started = time.time()
    for epoch in range(args.epochs):
        perm = torch.tensor(np.random.default_rng(epoch).permutation(train), device=dev)
        total = 0.0
        for i in range(len(train) // args.batch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            logits, vlogit = model(augment(X[idx], gen))
            v = voiced[idx]
            soft = torch.exp(-.5 * (grid[None] - target_bin[idx][:, None]) ** 2)
            soft = soft / soft.sum(1, keepdim=True)
            pitch_loss = -(soft[v] * F.log_softmax(logits[v], 1)).sum(1).mean() if v.any() else 0.0
            loss = pitch_loss + F.binary_cross_entropy_with_logits(vlogit, v.float())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            total += float(loss)
        print(f"epoch {epoch + 1:2d} loss {total / (len(train) // args.batch):.4f}  {time.time() - started:5.0f}s", flush=True)

    model.eval()
    with torch.no_grad():
        outs = [model(X[torch.tensor(test[i:i + 4096], device=dev)]) for i in range(0, len(test), 4096)]
    logits = torch.cat([o[0] for o in outs]).cpu().numpy(); vlog = torch.cat([o[1] for o in outs]).cpu().numpy()
    nn_cents, nn_voiced = decode(logits, vlog)
    yin = np.array([heard_pitch(x[i])[0] for i in test])
    lab, st = cents[test], state[test]
    report = {"test_frames": int(len(test)), "labelled_voiced": int((st == 1).sum()), "quiet": int((st == 0).sum()),
              "unsure": int((st == -1).sum())}
    for name, c, vprob in (("nn", nn_cents, nn_voiced), ("yin", yin, np.isfinite(yin).astype(float))):
        says = vprob >= .5
        err = np.abs(c - lab)
        m = st == 1
        report[name] = {
            "within_25c_of_voiced": float(np.mean(says[m] & (err[m] < 25))),
            "median_error_cents": float(np.nanmedian(np.where(says[m], err[m], np.nan))),
            "missed_voiced": float(np.mean(~says[m])),
            "false_voiced_on_quiet": float(np.mean(says[st == 0])) if (st == 0).any() else None,
            "heard_on_unsure": float(np.mean(says[st == -1])) if (st == -1).any() else None}
    print(json.dumps(report, indent=2))
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(report=report), str(out) + ".pt")
    out.with_suffix(".json").write_text(json.dumps({**report, **vars(args)}, indent=2), encoding="utf-8")
    arrays = export(model)
    np.savez(str(out) + "_runtime.npz", **arrays)
    # the NumPy ear must agree with torch
    ear = Ear(arrays)
    for i in test[:50]:
        with torch.no_grad():
            lt, vt = model(X[int(i)][None])
        ln, vn = ear.forward(x[i])
        assert np.allclose(lt.cpu().numpy()[0], ln, atol=1e-3) and abs(float(vt) - vn) < 1e-3
    print(f"saved {out}.pt, {out}.json and {out}_runtime.npz (NumPy ear checked against torch)")


if __name__ == "__main__":
    main()
