"""PC: step8b — the score memory fine-tuned through the playing (memory -> actuator -> simulated rigs).

    python scripts/yamabiko_step8b_train.py --memory runs/memory/mem_07.pt --actuator models/step7c/nn_f80_t7_tw05.pt \
        --twin runs/real/twin_y1f80_05.json --out runs/memory/mem_07_play

Every step, 128 whistles (synthetic, and cut from the rig's recordings) are heard by the ear and written into
the memory; the memory reads the score back, the actuator network (frozen) plays it on 128 rigs drawn around
the twin, and the pitch the flute gives is compared with the rule's score of the same whistle (the
note time 0.3 s after each onset, as the play scores of yamabiko_memory_train.py).  The loss is that play
error, plus the memory's own loss (reading back the rule's score) so it stays a score.  Only the memory
learns.  The score read back is laid out as step8 plays it (step8.Recalled: the lead before, the tail after).
Scored before and after on held-out whistles: hit within 25 cent of the rule's score, also in key.
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flute_rl.yamabiko.nn_performer import NNTwinPerformer  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.score_memory import MemoryConfig, MemoryRuntime, export, torch_model  # noqa: E402
from flute_rl.yamabiko.step8 import LEAD_STEPS, TAIL_STEPS  # noqa: E402
from yamabiko_memory_train import batch as make_batch, load_ear, make, real_items  # noqa: E402
from yamabiko_twin_finetune import load_twin, repeat, rigs_around  # noqa: E402


def laid_out(values, n, total):
    """(B, L) per-instant values -> (B, total) in the player's time: index clamp(t - LEAD, 0, n - 1)."""
    t = torch.arange(total, device=values.device)[None] - LEAD_STEPS
    idx = torch.minimum(t.clamp(min=0).expand(len(n), -1), (n[:, None] - 1))
    inside = (t >= 0) & (t < n[:, None])
    return values.gather(1, idx), inside


def settled(ref_c, ref_v):
    """Voiced instants 0.3 s into their note, not in a glide (as yamabiko_memory_train.play_scores)."""
    c, v = ref_c.cpu().numpy(), ref_v.cpu().numpy()
    keep = np.zeros_like(v)
    for b in range(len(c)):
        run = 0
        for i in range(c.shape[1]):
            run = run + 1 if v[b, i] else 0
            keep[b, i] = run > 30 and abs(c[b, i] - c[b, max(0, i - 15)]) < 50
    return torch.tensor(keep, device=ref_c.device)


def play(memory, performer, plant, items, rigs, twin, device, generator, grad=True):
    ear, target, voice, mask = make_batch(items, device)
    n = mask.sum(1).long()
    W, G = memory.write(ear, mask)
    cents, vlog = memory.read(W, G, torch.arange(ear.shape[1], device=device))
    total = LEAD_STEPS + int(n.max()) + TAIL_STEPS
    in_c, inside = laid_out(cents, n, total)
    in_v, _ = laid_out((torch.sigmoid(vlog) >= .5).float(), n, total)
    ref_c, _ = laid_out(target, n, total)
    ref_v, _ = laid_out(voice, n, total)
    in_v, ref_v = (in_v > .5) & inside, (ref_v > .5) & inside
    with torch.set_grad_enabled(grad):
        result = performer.perform(plant, in_c, in_v, rigs, twin, generator)
    return result["pitch_cents"], ref_c, ref_v, (cents, vlog, target, voice, mask)


def memory_loss(cents, vlog, target, voice, mask, beta):
    pitch = (F.smooth_l1_loss(cents / 100, target / 100, beta=beta, reduction="none") * (1 + voice) * mask).sum() / mask.sum()
    return pitch + (F.binary_cross_entropy_with_logits(vlog, voice, reduction="none") * mask).sum() / mask.sum()


def score(memory, performer, plant, items, rigs, twin, device):
    hits, keyed = [], []
    with torch.no_grad():
        for i in range(0, len(items), len(rigs.torque_gain)):
            chunk = items[i:i + len(rigs.torque_gain)]
            r = type(rigs)(**{k: None if v is None else v[:len(chunk)] for k, v in vars(rigs).items()})
            tw = repeat(twin, len(chunk))
            pitch, ref_c, ref_v, _ = play(memory, performer, plant, chunk, r, tw, device, torch.Generator(device).manual_seed(0), False)
            keep = settled(ref_c, ref_v)
            err = (pitch - ref_c)
            for b in range(len(chunk)):
                k = keep[b].bool()
                if k.sum() < 10:
                    continue
                e = err[b][k]
                hits.append(float((e.abs() <= 25).float().mean()))
                keyed.append(float(((e - e.median()).abs() <= 25).float().mean()))
    return {"hit": float(np.mean(hits)), "hit_in_key": float(np.mean(keyed)), "songs": len(hits)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", required=True)
    ap.add_argument("--ear", default="models/step5/ear_06.pt")
    ap.add_argument("--actuator", default="models/step7c/nn_f80_t7_tw05.pt")
    ap.add_argument("--twin", default="runs/real/twin_y1f80_05.json")
    ap.add_argument("--real", nargs="*", default=["runs/real/whistle_data", "runs/real/whistle"])
    ap.add_argument("--real-holdout", nargs="*", default=["20261124_111404", "20261124_111830"])
    ap.add_argument("--songs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--real-share", type=float, default=.4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--memory-weight", type=float, default=1.0, help="weight of the memory's own loss")
    ap.add_argument("--spread", type=float, default=.12)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); dev = args.device
    torch.manual_seed(args.seed)
    started = time.time()
    ck = torch.load(args.memory, map_location=dev, weights_only=False)
    memory = torch_model(MemoryConfig(**ck["config"])).to(dev)
    memory.load_state_dict(ck["state_dict"])
    beta = float(ck.get("report", {}).get("config", {}).get("beta", .25))
    ear = load_ear(args.ear, dev)
    twin, temp_c, plant_name = load_twin(args.twin, dev)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.named(plant_name, **({"temp_c": temp_c} if temp_c else {})))
    performer = NNTwinPerformer.from_checkpoint(torch.load(args.actuator, map_location=dev, weights_only=False), plant, dev).eval()
    for p in performer.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(args.seed)
    train = make(rng, args.songs, (3.0, 10.0), ear, memory.config, dev)
    test = make(np.random.default_rng(args.seed + 50), 64, (3.0, 10.0), ear, memory.config, dev)
    real_train, real_test = real_items(args.real, args.real_holdout, ear, memory.config, dev, np.random.default_rng(args.seed + 7),
                                       seconds=(3.0, 10.0))
    print(f"data: {len(train)} synthetic, {len(real_train)} real for training; {len(test)} + {len(real_test)} held out  "
          f"{time.time() - started:.0f}s", flush=True)
    eval_rigs = rigs_around(twin, args.batch, args.spread, torch.Generator(dev).manual_seed(7777))
    report = {"config": vars(args), "before": {"synthetic": score(memory, performer, plant, test, eval_rigs, twin, dev),
                                                "real": score(memory, performer, plant, real_test, eval_rigs, twin, dev)}}
    print(f"before: {report['before']}  {time.time() - started:.0f}s", flush=True)
    opt = torch.optim.AdamW(memory.parameters(), lr=args.lr, weight_decay=1e-5)
    gen = torch.Generator(dev).manual_seed(args.seed)
    best = {"score": -1, "step": 0}
    best_state = {k: v.detach().clone() for k, v in memory.state_dict().items()}
    history = []
    for step in range(1, args.steps + 1):
        k = int(round(args.batch * args.real_share)) if real_train else 0
        chosen = [train[i] for i in rng.choice(len(train), args.batch - k)] + [real_train[i] for i in rng.choice(len(real_train), k)]
        rigs = rigs_around(twin, args.batch, args.spread, gen)
        pitch, ref_c, ref_v, (cents, vlog, target, voice, mask) = play(memory, performer, plant, chosen, rigs,
                                                                       repeat(twin, args.batch), dev, gen)
        keep = settled(ref_c, ref_v).bool()
        play_loss = F.smooth_l1_loss(pitch[keep] / 100, ref_c[keep] / 100, beta=.25)
        loss = play_loss + args.memory_weight * memory_loss(cents, vlog, target, voice, mask, beta)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(memory.parameters(), 1.0); opt.step()
        if step % 10 == 0 or step == 1:
            print(f"step {step:4d} loss {loss.item():.4f} play {play_loss.item():.4f} "
                  f"hit {(((pitch - ref_c).abs() <= 25) & keep).sum().item() / max(keep.sum().item(), 1):.3f}  "
                  f"{time.time() - started:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            s = {"synthetic": score(memory, performer, plant, test, eval_rigs, twin, dev),
                 "real": score(memory, performer, plant, real_test, eval_rigs, twin, dev)}
            history.append({"step": step, **s})
            value = s["synthetic"]["hit"] + s["real"]["hit"]
            print(f"eval {step}: {s}", flush=True)
            if value > best["score"]:
                best = {"score": value, "step": step, **s}
                best_state = {k: v.detach().clone() for k, v in memory.state_dict().items()}
    memory.load_state_dict(best_state)
    report.update(kept=best, history=history)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory.checkpoint(report=report), str(out) + ".pt")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    arrays = export(memory)
    np.savez(str(out) + "_runtime.npz", **arrays)
    rt = MemoryRuntime(arrays)
    c, _, _ = test[0]
    for code in c.float().numpy():
        rt.listen(code)
    print(f"kept step {best['step']}: {best}\nsaved {out}.pt, .json and _runtime.npz")


if __name__ == "__main__":
    main()
