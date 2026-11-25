"""PC: train step8's two small new networks (flute_rl/yamabiko/step8.py) and write their NumPy file.

    python scripts/yamabiko_step8_train.py --ear models/step5/ear_06.pt --memory runs/memory/mem_05.pt \
        --out models/step8/step8_heads

conductor   a tiny GRU: (button pressed this tick, "finished" from the end network) -> idle / listen / play and
            the fan's duty.  Taught on random sequences of presses: idle -press-> listen -press-> play
            -finished or press-> idle; the fan at 0.8 while playing, off otherwise (as the server today).
end         a small MLP: from the memory read around the play head (voice probability and how much was written
            there, step8.end_features), "the song has finished": the play head is past the last note heard
            plus the tail.  Taught on synthetic whistles (and the rig's whistles with --real) heard by the ear
            and written into the memory, exactly as the NumPy memory reads them.
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
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flute_rl.yamabiko.score_memory import MemoryConfig, torch_model  # noqa: E402
from flute_rl.yamabiko.step8 import END_LOOK, TAIL_STEPS  # noqa: E402
from yamabiko_memory_train import load_ear, make, real_items  # noqa: E402

FAN = .8


def conductor_episode(rng, length=600):
    """Random presses and "finished" signals, and the modes and fan they should give.  "finished" is only ever
    above 0 while playing (step8 clears it when the mode changes); it stays high once it fires."""
    button = (rng.random(length) < .03).astype(np.float32)
    end = np.zeros(length, np.float32)
    mode = np.zeros(length, np.int64)
    m, fired = 0, False
    for t in range(length):
        if m == 2:
            if not fired and rng.random() < .02:
                fired = True
            end[t] = rng.uniform(.6, 1.0) if fired else (rng.uniform(0, .45) if rng.random() < .3 else 0.0)
        if button[t]:
            m, fired = (m + 1) % 3, False
        elif m == 2 and end[t] >= .5:
            m, fired = 0, False
        mode[t] = m
    return button, end, mode


class Conductor(nn.Module):
    """(button, finished, the mode chosen before as one-hot) -> 3 mode logits and the fan's logit."""

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, button, end, prev):
        return self.net(torch.cat([torch.stack([button, end], -1), F.one_hot(prev, 3).float()], -1))


def train_conductor(device, steps=2000):
    """Taught on the steps of random sequences (the mode before is the true one), checked by running whole
    sequences on its own choices."""
    model = Conductor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(3)
    for step in range(steps):
        b, e, m = zip(*[conductor_episode(rng) for _ in range(32)])
        b, e, m = (torch.tensor(np.stack(x), device=device) for x in (b, e, m))
        prev = torch.cat([torch.zeros_like(m[:, :1]), m[:, :-1]], 1)
        out = model(b, e, prev)
        loss = F.cross_entropy(out[..., :3].reshape(-1, 3), m.reshape(-1))             + F.binary_cross_entropy_with_logits(out[..., 3], (m == 2).float() * FAN)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        b, e, m = zip(*[conductor_episode(np.random.default_rng(99 + k), 2000) for k in range(16)])
        b, e, m = (torch.tensor(np.stack(x), device=device) for x in (b, e, m))
        prev = torch.zeros(len(m), dtype=torch.long, device=device)
        chosen, fan = [], []
        for t in range(m.shape[1]):                              # closed loop: its own choice fed back
            out = model(b[:, t], e[:, t], prev)
            prev = out[:, :3].argmax(-1)
            chosen.append(prev); fan.append(torch.sigmoid(out[:, 3]))
        chosen, fan = torch.stack(chosen, 1), torch.stack(fan, 1)
        acc = float((chosen == m).float().mean())
        fan_err = float((fan - (m == 2).float() * FAN).abs().max())
    return model, {"mode_accuracy": acc, "fan_max_error": fan_err}


def memory_reads(memory, items, device, extra=300):
    """Per item: (prob, mass) read at every instant 0 .. n + extra, the n listened, the last voiced instant."""
    out = []
    with torch.no_grad():
        for c, _, v in items:
            n = len(c)
            ear = c.to(device).float()[None]
            W, G = memory.write(ear, torch.ones(1, n, device=device))
            steps = torch.arange(n + extra + max(END_LOOK) + 1, device=device)
            cents, vlog = memory.read(W, G, steps)
            mass = torch.einsum("bk,tk->bt", W[:, -1], memory.code(steps))[0]
            voiced = np.flatnonzero(v.numpy())
            out.append((torch.sigmoid(vlog[0]).cpu().numpy(), mass.cpu().numpy(), n, int(voiced[-1])))
    return out


def end_data(reads, rng, per_song=200):
    xs, ys = [], []
    look = np.array(END_LOOK)
    for prob, mass, n, last in reads:
        ref = max(mass[n // 2], 1e-6)
        boundary = last + 1 + TAIL_STEPS
        # many instants near the boundary, some anywhere
        u = np.r_[rng.integers(0, n + 250, per_song // 2), boundary + rng.integers(-80, 80, per_song // 2)]
        u = np.clip(u, 0, len(prob) - max(END_LOOK) - 1)
        x = np.concatenate([np.stack([prob[np.maximum(u + o, 0)], mass[np.maximum(u + o, 0)] / ref], 1) for o in look], 1)
        xs.append(x); ys.append((u >= boundary).astype(np.float32))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


def end_timing(end, reads, device):
    """Seconds between where the end network first fires and where it should (last note + tail), per song."""
    err = []
    look = np.array(END_LOOK)
    with torch.no_grad():
        for prob, mass, n, last in reads:
            ref = max(mass[n // 2], 1e-6)
            u = np.arange(0, len(prob) - max(END_LOOK) - 1)
            x = np.concatenate([np.stack([prob[np.maximum(u + o, 0)], mass[np.maximum(u + o, 0)] / ref], 1)
                                for o in look], 1).astype(np.float32)
            p = torch.sigmoid(end(torch.tensor(x, device=device)))[:, 0].cpu().numpy()
            fired = np.flatnonzero(p >= .5)
            at = fired[0] if len(fired) else len(u)
            err.append((at - (last + 1 + TAIL_STEPS)) / 100)
    err = np.array(err)
    return {"mean_s": float(err.mean()), "abs_mean_s": float(np.abs(err).mean()),
            "early_over_0.3s": float(np.mean(err < -.3)), "late_over_0.5s": float(np.mean(err > .5)), "songs": len(err)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ear", default="models/step5/ear_06.pt")
    ap.add_argument("--memory", required=True)
    ap.add_argument("--real", nargs="*", default=["runs/real/whistle_data", "runs/real/whistle"])
    ap.add_argument("--songs", type=int, default=400)
    ap.add_argument("--out", default="models/step8/step8_heads")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); dev = args.device
    started = time.time()
    torch.manual_seed(0)
    conductor, cond_report = train_conductor(dev)
    print(f"conductor: {cond_report}  {time.time() - started:.0f}s", flush=True)

    ck = torch.load(args.memory, map_location=dev, weights_only=False)
    memory = torch_model(MemoryConfig(**ck["config"])).to(dev).eval()
    memory.load_state_dict(ck["state_dict"])
    ear = load_ear(args.ear, dev)
    train = make(np.random.default_rng(21), args.songs, (3.0, 20.0), ear, memory.config, dev)
    test = make(np.random.default_rng(22), 64, (3.0, 20.0), ear, memory.config, dev)
    real_test = []
    if args.real:
        real_train, real_test = real_items(args.real, [], ear, memory.config, dev, np.random.default_rng(23), crops=4)
        train += real_train
    rng = np.random.default_rng(5)
    x, y = end_data(memory_reads(memory, train, dev), rng)
    print(f"end data: {len(y)} instants from {len(train)} songs  {time.time() - started:.0f}s", flush=True)
    end = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(end.parameters(), lr=3e-3)
    xt, yt = torch.tensor(x, device=dev), torch.tensor(y, device=dev)
    for step in range(3000):
        i = torch.randint(0, len(yt), (1024,), device=dev)
        loss = F.binary_cross_entropy_with_logits(end(xt[i])[:, 0], yt[i])
        opt.zero_grad(); loss.backward(); opt.step()
    report = {"conductor": cond_report, "end_synthetic": end_timing(end, memory_reads(memory, test, dev), dev),
              "memory": args.memory, "ear": args.ear}
    print(json.dumps(report, indent=2), flush=True)

    g = lambda t: t.detach().cpu().numpy().astype(np.float32)
    arrays = {"s8.cond.0.w": g(conductor.net[0].weight), "s8.cond.0.b": g(conductor.net[0].bias),
              "s8.cond.2.w": g(conductor.net[2].weight), "s8.cond.2.b": g(conductor.net[2].bias),
              "s8.end.0.w": g(end[0].weight), "s8.end.0.b": g(end[0].bias),
              "s8.end.2.w": g(end[2].weight), "s8.end.2.b": g(end[2].bias)}
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out) + ".npz", **arrays)
    torch.save({"conductor": conductor.state_dict(), "end": end.state_dict(), "report": report}, str(out) + ".pt")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}.npz, .pt and .json")


if __name__ == "__main__":
    main()
