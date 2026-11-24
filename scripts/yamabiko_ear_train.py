"""PC: train the ear network (flute_rl/yamabiko/ear_nn.py) on the rig's recordings and compare it with YIN.

    python scripts/yamabiko_ear_data.py                          # once: runs/ear/dataset.npz
    python scripts/yamabiko_ear_train.py --out runs/ear/ear_01
    python scripts/yamabiko_ear_train.py --wide --synth 300000 --demo runs/ear/demo --out runs/ear/ear_02   # step5

``--wide`` hears the flute and a whistle (178 bins, about 310 Hz .. 4 kHz).  ``--synth N`` adds N synthetic
whistle frames (flute_rl/yamabiko/ear_synth.py) to the training set, a quiet frame of the rig's recordings
under each, and scores a separate synthetic test set.  ``--demo DIR`` also scores the 16 kHz WAVs in DIR
(the demo videos: a whistle from a phone, then the flute), labelled offline like the dataset.

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
from flute_rl.yamabiko.ear_nn import BIN_CENTS, Ear, EarConfig, decode, export, torch_model  # noqa: E402
from flute_rl.yamabiko.ear_synth import frames as synth_frames, noise_frames  # noqa: E402
from flute_rl.yamabiko.hw import heard_pitch  # noqa: E402

SR = 16000


def augment(x, gen, gain_db=(-18, 12)):
    b, n = x.shape
    t = torch.arange(n, device=x.device) / SR
    gain = 10 ** (torch.empty(b, 1, device=x.device).uniform_(*gain_db, generator=gen) / 20)
    rms = x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-4)
    snr = 10 ** (torch.empty(b, 1, device=x.device).uniform_(10, 40, generator=gen) / 20)
    noise = torch.randn(b, n, device=x.device, generator=gen) * rms / snr
    f = torch.empty(b, 1, device=x.device).uniform_(50, 110, generator=gen)
    phase = torch.rand(b, 1, device=x.device, generator=gen) * 6.283
    rumble = torch.sin(6.283 * f * t[None] + phase) * rms * torch.empty(b, 1, device=x.device).uniform_(0, 1.5, generator=gen)
    return (x + noise + rumble) * gain


def demo_frames(folder):
    """Labelled frames of every 16 kHz WAV in ``folder`` (labels as yamabiko_ear_data.py, up to 4 kHz)."""
    import wave
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import yamabiko_ear_data as data
    data.FMAX = 4000.0
    parts = []
    for path in sorted(pathlib.Path(folder).glob("*.wav")):
        with wave.open(str(path)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
        parts.append(data.build(pcm))
    return tuple(np.concatenate(z) for z in zip(*parts))


def score(name, x, lab, st, model, low, dev):
    """Network vs the rig's YIN on labelled frames (int16 or float x)."""
    xf = x.astype(np.float32) / 32768.0 if x.dtype.kind in "iu" else x
    with torch.no_grad():
        outs = [model(torch.tensor(xf[i:i + 4096], device=dev)) for i in range(0, len(xf), 4096)]
    logits = torch.cat([o[0] for o in outs]).cpu().numpy(); vlog = torch.cat([o[1] for o in outs]).cpu().numpy()
    nn_cents, nn_voiced = decode(logits, vlog, low)
    yin = np.array([heard_pitch(f)[0] for f in xf])
    report = {"frames": int(len(x)), "labelled_voiced": int((st == 1).sum()), "quiet": int((st == 0).sum()),
              "unsure": int((st == -1).sum())}
    for who, c, vprob in (("nn", nn_cents, nn_voiced), ("yin", yin, np.isfinite(yin).astype(float))):
        says = vprob >= .5
        err = np.abs(c - lab)
        m = st == 1
        report[who] = {
            "within_25c_of_voiced": float(np.mean(says[m] & (err[m] < 25))) if m.any() else None,
            "median_error_cents": float(np.nanmedian(np.where(says[m], err[m], np.nan))) if m.any() else None,
            "missed_voiced": float(np.mean(~says[m])) if m.any() else None,
            "false_voiced_on_quiet": float(np.mean(says[st == 0])) if (st == 0).any() else None,
            "heard_on_unsure": float(np.mean(says[st == -1])) if (st == -1).any() else None}
    print(f"{name}: nn {report['nn']['within_25c_of_voiced']}, yin {report['yin']['within_25c_of_voiced']} "
          f"(within 25 cent of {report['labelled_voiced']} voiced frames)", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/ear/dataset.npz")
    ap.add_argument("--out", default="runs/ear/ear_01")
    ap.add_argument("--wide", action="store_true", help="flute and whistle range (step5)")
    ap.add_argument("--synth", type=int, default=0, help="synthetic whistle frames to train on")
    ap.add_argument("--synth-test", type=int, default=30000)
    ap.add_argument("--noise", type=int, default=0, help="pitchless noise frames to train on (not sounding)")
    ap.add_argument("--level", action="store_true", help="feed the frame level to the network as well")
    ap.add_argument("--demo", default=None, help="folder of 16 kHz WAVs to score as well")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); dev = args.device
    torch.manual_seed(args.seed)
    config = EarConfig.wide(level=args.level) if args.wide else EarConfig(level=args.level)
    d = np.load(args.data)
    n_rig = len(d["x"])
    x = d["x"].astype(np.float32) / 32768.0
    cents, state, split = d["cents"], d["state"], d["split"]
    if args.synth:
        rng = np.random.default_rng(args.seed + 1000)
        sx, sc, ss = synth_frames(rng, args.synth)
        quiet = np.flatnonzero((state == 0) & (split == 0))
        bg = d["x"][rng.choice(quiet, len(sx))].astype(np.float32) / 32768.0       # the room and the rig
        sx = sx.astype(np.float32) / 32768.0 + bg * rng.uniform(0, 3, (len(sx), 1)).astype(np.float32)
        x = np.concatenate([x, np.clip(sx, -1, 1)])
        cents = np.concatenate([cents, sc]); state = np.concatenate([state, ss])
        split = np.concatenate([split, np.zeros(len(sx), split.dtype)])
    if args.noise:
        nx, nc, ns = noise_frames(np.random.default_rng(args.seed + 3000), args.noise)
        x = np.concatenate([x, nx.astype(np.float32) / 32768.0])
        cents = np.concatenate([cents, nc]); state = np.concatenate([state, ns])
        split = np.concatenate([split, np.zeros(len(nx), split.dtype)])
    use = state >= 0
    train = np.flatnonzero(use & (split == 0))
    test = np.flatnonzero(split[:n_rig] == 1)
    X = torch.tensor(x, device=dev)
    target_bin = torch.tensor(np.nan_to_num((cents - config.low_cents) / BIN_CENTS, nan=0.0), device=dev,
                              dtype=torch.float32)
    voiced = torch.tensor(state == 1, device=dev)
    model = torch_model(config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    gen = torch.Generator(dev).manual_seed(args.seed)
    grid = torch.arange(config.bins, device=dev, dtype=torch.float32)
    started = time.time()
    for epoch in range(args.epochs):
        perm = torch.tensor(np.random.default_rng(epoch).permutation(train), device=dev)
        total = 0.0
        for i in range(len(train) // args.batch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            logits, vlogit = model(augment(X[idx], gen, (-6, 6) if args.level else (-18, 12)))
            v = voiced[idx]
            soft = torch.exp(-.5 * (grid[None] - target_bin[idx][:, None]) ** 2)
            soft = soft / soft.sum(1, keepdim=True)
            pitch_loss = -(soft[v] * F.log_softmax(logits[v], 1)).sum(1).mean() if v.any() else 0.0
            loss = pitch_loss + F.binary_cross_entropy_with_logits(vlogit, v.float())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            total += float(loss)
        print(f"epoch {epoch + 1:2d} loss {total / (len(train) // args.batch):.4f}  {time.time() - started:5.0f}s", flush=True)

    model.eval()
    report = {"config": {"low_cents": config.low_cents, "bins": config.bins},
              "flute": score("flute (held-out rig recordings)", x[test], cents[test], state[test], model,
                             config.low_cents, dev)}
    if args.synth:
        tx, tc, ts = synth_frames(np.random.default_rng(args.seed + 2000), args.synth_test)
        report["synthetic_whistle"] = score("synthetic whistle", tx, tc, ts, model, config.low_cents, dev)
    if args.demo:
        dx, dc, ds = demo_frames(args.demo)
        high = (ds == 1) & (dc > 1800)          # above the flute's range: the whistle
        report["demo_whistle"] = score("demo videos, whistle (> 1250 Hz)", dx, dc,
                                       np.where((ds == 1) & ~high, -1, ds).astype(np.int8), model, config.low_cents, dev)
        report["demo_all"] = score("demo videos, all", dx, dc, ds, model, config.low_cents, dev)
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
