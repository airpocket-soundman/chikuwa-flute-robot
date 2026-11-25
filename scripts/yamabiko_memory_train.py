"""PC: train step6's score memory (flute_rl/yamabiko/score_memory.py) on synthetic whistles heard by the ear.

    python scripts/yamabiko_memory_train.py --ear models/step5/ear_04.pt --out runs/memory/mem_01 \
        --demo runs/ear/demo --play runs/real/twin_y1f80_04.json --actuator models/step4/nn_f80_t4_s06.pt

Synthetic whistles (flute_rl/yamabiko/ear_synth.py) of 3..20 s go through the ear network (its hidden code
every 10 ms); the memory writes them and is taught to read back the rule pipeline's target (score_memory
.teacher) at every instant.  Scored on held-out whistles of 5, 10, 20 and 30 s (how long a song the memory
holds), on the demo videos' real whistles (teacher from YIN, as the whistle copy does today), and with
``--play`` by playing the recalled targets with an actuator network on rigs around a twin, against the same
melody played from the rule's targets.  Writes <out>.pt, <out>.json and <out>_runtime.npz.
``--extra DIR`` adds songs cut from more human whistles in the rig recordings' form (scripts/yamabiko_archive_whistles
.py: the Silbo Gomero corpus under the rig's room) as ``--extra-share`` of every batch; its takes with "_test"
in the stem are scored apart (``extra_holdout``), so ``real_holdout`` stays the rig's own held-out whistles.
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
from flute_rl.yamabiko.ear_nn import EarConfig, torch_model as ear_model  # noqa: E402
from flute_rl.yamabiko.ear_synth import FRAME, HOP, phrase  # noqa: E402
from flute_rl.yamabiko.score_memory import MemoryConfig, MemoryRuntime, export, teacher, teacher_shift, torch_model  # noqa: E402

WHISTLE = {"max_gap_steps": 2}     # the profiles' whistle cleaning


def load_ear(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = {k: (tuple(v) if k == "channels" else v) for k, v in ck["config"].items()}
    model = ear_model(EarConfig(**cfg)).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model


def ear_codes(ear, x, device):
    """(frames, hidden) codes of every causal 20 ms frame, one every 10 ms, of float samples x."""
    ends = np.arange(FRAME, len(x) + 1, HOP)
    frames = np.stack([x[e - FRAME:e] for e in ends]).astype(np.float32)
    frames = np.round(frames * 32767) / 32768.0                   # as the int16 microphone
    with torch.no_grad():
        out = [ear.embed(torch.tensor(frames[i:i + 8192], device=device)) for i in range(0, len(frames), 8192)]
    return torch.cat(out).half().cpu()


def make(rng, count, seconds, ear, config, device):
    items = []
    while len(items) < count:
        sec = float(rng.uniform(*seconds)) if isinstance(seconds, tuple) else float(seconds)
        x, cents, _ = phrase(rng, sec)
        x = x + rng.normal(0, 10 ** (rng.uniform(-75, -60) / 20), len(x)).astype(np.float32)   # the room
        t = teacher(cents, config, WHISTLE)
        if t is None:
            continue
        codes = ear_codes(ear, x, device)
        n = min(len(codes), len(t[0]))
        target = torch.tensor(t[0][:n]); target.shift = teacher_shift(cents, config, WHISTLE)
        items.append((codes[:n], target, torch.tensor(t[1][:n])))
    return items


def real_items(folders, holdout, ear, config, device, rng, crops=8, seconds=(4.0, 20.0), kinds=("whistle",)):
    """Songs cut from whistles recorded on the rig (the page's takes of kind whistle, and the whistle copy's
    recordings): ``crops`` stretches of ``seconds`` per take, the teacher made from YIN as the whistle copy does.
    Returns (training items, held-out items); the takes named in ``holdout`` only give held-out items."""
    from flute_rl.yamabiko.whistle import whistle_track
    train, test = [], []
    for folder in folders:
        for path in sorted(pathlib.Path(folder).glob("*.npz")):
            if "_play" in path.stem or "_target" in path.stem:
                continue
            d = np.load(path)
            if "pcm" not in d.files or ("kind" in d.files and str(d["kind"]) not in kinds):
                continue
            pcm = d["pcm"]
            codes = ear_codes(ear, pcm / 32768.0, device)
            n = len(codes)
            track = np.r_[np.nan, whistle_track(pcm, 16000)][:n]   # whistle_track centres one hop later
            track = np.r_[track, np.full(n - len(track), np.nan)]
            held = any(h in path.stem for h in holdout)
            r = np.random.default_rng(abs(hash(path.stem)) % 2**32) if held else rng
            for _ in range(crops):
                length = int(min(n, r.uniform(*seconds) * 100))
                start = int(r.integers(0, max(1, n - length + 1)))
                t = teacher(track[start:start + length], config, WHISTLE)
                if t is None or t[1].sum() < 50:
                    continue
                target = torch.tensor(t[0]); target.shift = teacher_shift(track[start:start + length], config, WHISTLE)
                item = (codes[start:start + length], target, torch.tensor(t[1]))
                (test if held else train).append(item)
    return train, test


def batch(items, device):
    L = max(len(c) for c, _, _ in items)
    B = len(items)
    ear = torch.zeros(B, L, items[0][0].shape[1], device=device)
    target = torch.zeros(B, L, device=device); voice = torch.zeros(B, L, device=device)
    mask = torch.zeros(B, L, device=device)
    for i, (c, t, v) in enumerate(items):
        n = len(c)
        ear[i, :n] = c.to(device).float(); target[i, :n] = t.to(device); voice[i, :n] = v.to(device).float()
        mask[i, :n] = 1
    target.shifts = torch.tensor([float(getattr(t, "shift", 0.0)) for _, t, _ in items], device=device)
    return ear, target, voice, mask


def recall(model, items, device):
    out = []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            ear, _, _, mask = batch(items[i:i + 16], device)
            W, G = model.write(ear, mask)
            cents, vlog = model.read(W, G, torch.arange(ear.shape[1], device=device))
            for j, (c, _, _) in enumerate(items[i:i + 16]):
                out.append((cents[j, :len(c)].cpu().numpy(), torch.sigmoid(vlog[j, :len(c)]).cpu().numpy()))
    return out


def scores(model, items, device):
    errs, within, offset_free, voice_ok = [], [], [], []
    for (c, t, v), (rc, rv) in zip(items, recall(model, items, device)):
        t, v = t.numpy(), v.numpy().astype(bool)
        d = rc[v] - t[v]
        errs.append(np.abs(d).mean()); within.append(np.mean(np.abs(d) < 25))
        offset_free.append(np.abs(d - np.median(d)).mean()); voice_ok.append(np.mean((rv >= .5) == v))
    return {"mae_cents": float(np.mean(errs)), "within_25c": float(np.mean(within)),
            "mae_after_offset": float(np.mean(offset_free)), "voice_agree": float(np.mean(voice_ok)),
            "songs": len(items)}


def demo_items(folder, ear, config, device):
    """The demo videos' whistles (up to the first long silence after whistling), teacher from YIN."""
    import wave
    from flute_rl.yamabiko.whistle import whistle_track
    items, names = [], []
    for path in sorted(pathlib.Path(folder).glob("*.wav")):
        with wave.open(str(path)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
        x = pcm / 32768.0
        track = whistle_track(pcm, 16000)
        high = np.flatnonzero(np.isfinite(track) & (track > 1600))
        gaps = np.diff(high) > 100
        end = (high[np.argmax(gaps)] if gaps.any() else high[-1]) + 20
        codes = ear_codes(ear, x, device)
        aligned = np.r_[np.nan, track][:len(codes)]                 # whistle_track's frames centre one hop later
        t = teacher(aligned[:end], config, WHISTLE)
        items.append((codes[:end], torch.tensor(t[0]), torch.tensor(t[1]))); names.append(path.stem)
    return items, names


def play_scores(model, items, device, twin_path, actuator_path, rigs=16):
    """Play every item's recalled target and its teacher target with the actuator network; hit within 25 cent
    of the teacher's melody (first 0.3 s of every note left out), both with the player's lead and tail."""
    from flute_rl.yamabiko.nn_performer import NNTwinPerformer
    from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig
    from yamabiko_twin_finetune import load_twin, repeat, rigs_around
    twin, temp_c, plant_name = load_twin(twin_path, device)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.named(plant_name, **({"temp_c": temp_c} if temp_c else {})))
    performer = NNTwinPerformer.from_checkpoint(torch.load(actuator_path, map_location=device, weights_only=False),
                                                plant, device).eval()
    rig_set = rigs_around(twin, rigs, .12, torch.Generator(device).manual_seed(7777))
    batch_twin = repeat(twin, rigs)

    def settled(c, v):
        """Voiced steps 0.3 s or more into their voiced run, not in a glide (moving < 50 cent in 0.15 s)."""
        keep = np.zeros(len(c), bool)
        run = 0
        for i in range(len(c)):
            run = run + 1 if v[i] else 0
            keep[i] = run > 30 and abs(c[i] - c[max(0, i - 15)]) < 50
        return keep

    def framed(c, v, lead=60, tail=60):
        return np.r_[np.full(lead, c[0]), c, np.full(tail, c[-1])].astype(np.float32), np.r_[np.zeros(lead, bool), v, np.zeros(tail, bool)]

    out = {"rule": [], "memory": [], "rule_in_key": [], "memory_in_key": []}
    for (c, t, v), (rc, rv) in zip(items, recall(model, items, device)):
        ref_c, ref_v = framed(t.numpy(), v.numpy().astype(bool))
        keep = settled(ref_c, ref_v)
        for name, (pc, pv) in (("rule", (ref_c, ref_v)), ("memory", framed(rc, rv >= .5))):
            cents = torch.tensor(pc, device=device)[None].expand(rigs, -1).contiguous()
            voice = torch.tensor(pv, device=device)[None].expand(rigs, -1).contiguous()
            with torch.no_grad():
                result = performer.perform(plant, cents, voice, rig_set, batch_twin, torch.Generator(device).manual_seed(0))
            pitch = result["pitch_cents"].cpu().numpy()
            err = pitch - ref_c[None]
            if not keep.any():
                continue
            out[name].append(float((np.abs(err)[:, keep] <= 25).mean()))
            key = np.median(err[:, keep], axis=1, keepdims=True)     # the key it played in, as yamabiko_step_eval
            out[name + "_in_key"].append(float((np.abs(err - key)[:, keep] <= 25).mean()))
    return {k: float(np.mean(v)) for k, v in out.items()} | {"songs": len(items)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ear", default="models/step5/ear_04.pt")
    ap.add_argument("--out", default="runs/memory/mem_01")
    ap.add_argument("--songs", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--keys", type=int, default=256)
    ap.add_argument("--max-hz", type=float, default=12.0)
    ap.add_argument("--demo", default=None)
    ap.add_argument("--play", default=None, help="twin json: also play recalled targets in the simulator")
    ap.add_argument("--actuator", default="models/step4/nn_f80_t4_s06.pt")
    ap.add_argument("--real", nargs="*", default=[], help="folders of whistles recorded on the rig")
    ap.add_argument("--real-holdout", nargs="*", default=[], help="stems of recordings kept for the test only")
    ap.add_argument("--real-share", type=float, default=.4, help="share of each batch cut from real whistles")
    ap.add_argument("--extra", nargs="*", default=[], help="folders of more human whistles (runs/archive/silbo_whistles)")
    ap.add_argument("--extra-share", type=float, default=.25, help="share of each batch cut from the extra whistles")
    ap.add_argument("--extra-crops", type=int, default=2, help="songs cut from every extra take")
    ap.add_argument("--value", type=int, default=64, help="numbers written per instant")
    ap.add_argument("--beta", type=float, default=.25, help="smooth-L1 knee of the pitch loss, in 100 cents")
    ap.add_argument("--key-aux", type=float, default=0.0,
                    help="weight of teaching the memory's pitch-like number (the one its soft highest / lowest "
                         "keep) the whistle's own pitch, in 1000 cents (training only; needs --extremes)")
    ap.add_argument("--extremes", action="store_true", help="also keep a soft highest and lowest pitch (the key)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); dev = args.device
    torch.manual_seed(args.seed)
    config = MemoryConfig(keys=args.keys, max_hz=args.max_hz, extremes=args.extremes, value=args.value)
    ear = load_ear(args.ear, dev)
    started = time.time()
    train = make(np.random.default_rng(args.seed), args.songs, (3.0, 20.0), ear, config, dev)
    tests = {s: make(np.random.default_rng(args.seed + 100 + s), 32, float(s), ear, config, dev) for s in (5, 10, 20, 30)}
    real_train, real_test = (real_items(args.real, args.real_holdout, ear, config, dev, np.random.default_rng(args.seed + 7))
                             if args.real else ([], []))
    extra_train, extra_test = (real_items(args.extra, ["_test"], ear, config, dev, np.random.default_rng(args.seed + 9),
                                          crops=args.extra_crops) if args.extra else ([], []))
    print(f"data: {len(train)} songs, {sum(len(c) for c, _, _ in train) / 6000:.0f} min; real {len(real_train)} songs "
          f"for training, {len(real_test)} held out; extra {len(extra_train)} / {len(extra_test)}  "
          f"{time.time() - started:.0f}s", flush=True)
    model = torch_model(config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    rng = np.random.default_rng(args.seed + 1)
    for step in range(1, args.steps + 1):
        k = int(round(args.batch * args.real_share)) if real_train else 0
        e = int(round(args.batch * args.extra_share)) if extra_train else 0
        chosen = [train[i] for i in rng.choice(len(train), args.batch - k - e)]
        chosen += [real_train[i] for i in rng.choice(len(real_train), k)] if k else []
        chosen += [extra_train[i] for i in rng.choice(len(extra_train), e)] if e else []
        ear_b, target, voice, mask = batch(chosen, dev)
        W, G = model.write(ear_b, mask)
        cents, vlog = model.read(W, G, torch.arange(ear_b.shape[1], device=dev))
        pitch = (F.smooth_l1_loss(cents / 100, target / 100, beta=args.beta, reduction="none") * (1 + voice) * mask).sum() / mask.sum()
        loss = pitch + (F.binary_cross_entropy_with_logits(vlog, voice, reduction="none") * mask).sum() / mask.sum()
        if args.key_aux:                              # the key: p of every heard note ~ the whistle's pitch
            p = model.gen(ear_b)[..., config.global_dim]
            raw = (target - target.shifts[:, None]) / 1000
            on = voice * mask * (target > config.low + 1) * (target < config.high - 1)   # not where the rule clipped
            loss = loss + args.key_aux * (F.smooth_l1_loss(p, raw, beta=.05, reduction="none") * on).sum() / on.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps:
            s10 = scores(model, tests[10], dev)
            sr = scores(model, real_test, dev) if real_test else None
            print(f"step {step:5d} loss {loss.item():.4f}  10 s: mae {s10['mae_cents']:.1f} within25 {s10['within_25c']:.3f}"
                  f" voice {s10['voice_agree']:.3f}"
                  + (f"  real: mae {sr['mae_cents']:.1f} after offset {sr['mae_after_offset']:.1f} voice {sr['voice_agree']:.3f}" if sr else "")
                  + f"  {time.time() - started:.0f}s", flush=True)
    report = {"config": vars(args), "by_length": {f"{s}s": scores(model, items, dev) for s, items in tests.items()}}
    if args.demo:
        items, names = demo_items(args.demo, ear, config, dev)
        report["demo"] = {n: scores(model, [it], dev) for n, it in zip(names, items)}
    if real_test:
        report["real_holdout"] = scores(model, real_test, dev)
    if extra_test:
        report["extra_holdout"] = scores(model, extra_test, dev)
    if args.play:
        report["play_10s"] = play_scores(model, tests[10][:12], dev, args.play, args.actuator)
        if real_test:
            report["play_real"] = play_scores(model, real_test[:12], dev, args.play, args.actuator)
        if extra_test:
            report["play_extra"] = play_scores(model, extra_test[:12], dev, args.play, args.actuator)
        if args.demo:
            report["play_demo"] = play_scores(model, items, dev, args.play, args.actuator)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(report=report), str(out) + ".pt")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    arrays = export(model)
    np.savez(str(out) + "_runtime.npz", **arrays)
    # the NumPy memory must agree with torch on one song
    c, _, _ = tests[5][0]
    rt = MemoryRuntime(arrays)
    for code in c.float().numpy():
        rt.listen(code)
    tc, _ = recall(model, [tests[5][0]], dev)[0]
    got = np.array([rt.recall(i)[0] for i in range(0, len(c), 37)])
    assert np.allclose(got, tc[::37], atol=.5), np.abs(got - tc[::37]).max()
    print(f"saved {out}.pt, {out}.json and {out}_runtime.npz (NumPy memory checked against torch)")


if __name__ == "__main__":
    main()
