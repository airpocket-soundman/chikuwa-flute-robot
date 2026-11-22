"""PC: train the valve network (tonguing) beside a frozen actuator network, around one fitted twin.

    python scripts/yamabiko_valve_train.py runs/real/twin_y1f80_01.json --actuator models/step1/nn_f80_02.pt \
        --out runs/real/valve_f80_01

The valve learns to let the air through only where the score has a note and the plunger is within ``--tol``
cent of it (the glide between notes stays silent), shutting for rests and repeated notes ahead of the
servo's delay.  Reported on 48 fixed rigs around the twin, against the rule "open when the score has a
note": the share of sounding time within 25 cent of the target, the share sounding off by more than 50 cent
(glides), and the share of the score's notes that sounded.  Writes <out>.pt, <out>.json, <out>_runtime.npz.
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from flute_rl.yamabiko.features import WINDOW  # noqa: E402
from flute_rl.yamabiko.nn_performer import NNTwinPerformer  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.residual_numpy import export_valve  # noqa: E402
from flute_rl.yamabiko.valve_nn import ValveConfig, ValveNN, play_with_valve  # noqa: E402
from yamabiko_twin_finetune import load_twin, repeat, rigs_around, songs  # noqa: E402


class RuleValve(torch.nn.Module):
    """The step1 rule as a stand-in valve network: open when the score has a note now."""

    config = ValveConfig(hidden=1)

    def __init__(self, normalize):
        super().__init__()
        self.valve = ValveNN(normalize)

    def features(self, *args):
        return self.valve.features(*args)

    def step(self, x, h):
        return x[:, len(WINDOW)], h                    # voice[t] (the voice block starts after the cents)


def travel_target(plant, twin, cents, voice, speed_share=.8, cap=40, min_leap=150.0):
    """Open on the score's notes, except the start of a note reached by a leap of more than ``min_leap``
    cent: shut for the time the plunger needs to get there (twin's top speed x ``speed_share``, at most
    ``cap`` steps).  Rests and the score's re-articulation gaps stay shut.  It depends only on the score and
    the twin, never on what is heard: the flute must sound for the actuator to hear where it is."""
    want = voice.clone()
    flat = cents.reshape(-1)
    one = type(twin)(**{k: (v[:1].expand(flat.numel()).clone() if torch.is_tensor(v) and v.dim() == 1 else v)
                        for k, v in vars(twin).items()})
    pos = plant.position_for_cents(flat, one).reshape(cents.shape)          # (B, T) stroke positions
    speed = (twin.max_velocity_strokes_s[:1] * speed_share).item() * plant.config.dt   # strokes per step
    v = voice.cpu().numpy(); c = cents.cpu().numpy(); x = pos.detach().cpu().numpy()
    w = want.cpu().numpy()
    for b in range(v.shape[0]):
        starts = np.flatnonzero(np.diff(np.r_[0, v[b].astype(int)]) == 1)
        last = None
        for a in starts:
            if last is not None and abs(c[b, a] - c[b, last]) > min_leap:
                steps = int(min(cap, np.ceil(abs(x[b, a] - x[b, last]) / max(speed, 1e-6))))
                w[b, a:a + steps] = False
            last = a
    return torch.as_tensor(w, device=voice.device)


def scores(emitted, sounding, cents, voice):
    error = (emitted - cents).abs()
    s, v = sounding & voice, voice
    on = float((error[s] < 25).float().mean()) if s.any() else 0.0
    glide = float(((error > 50) & sounding).float().sum() / sounding.float().sum().clamp_min(1))
    coverage = float(s.float().sum() / v.float().sum().clamp_min(1))
    return {"on_target": on, "glide": glide, "coverage": coverage}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("twin")
    ap.add_argument("--actuator", required=True, help="the frozen actuator network (NNTwinPerformer .pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=40.0, help="--target heard: open only within this many cent")
    ap.add_argument("--target", choices=["travel", "heard"], default="travel",
                    help="travel: shut for rests, re-articulation and the travel of a leap (score and twin only); "
                         "heard: open only where the flute sounds within --tol (it learns to stay shut)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--song-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--chatter", type=float, default=.05, help="cost of the mean |change| of the valve command")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--spread", type=float, default=.12)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    twin, temp_c, plant_name = load_twin(args.twin, device)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.named(plant_name, **({"temp_c": temp_c} if temp_c else {})))
    actuator = NNTwinPerformer.from_checkpoint(torch.load(args.actuator, map_location=device, weights_only=False),
                                               plant, device).eval()
    for p in actuator.parameters():
        p.requires_grad_(False)
    valve = ValveNN(plant.config.normalize).to(device)
    generator = torch.Generator(device).manual_seed(args.seed)
    eval_rigs = rigs_around(twin, 48, args.spread, torch.Generator(device).manual_seed(7777))
    eval_twin = repeat(twin, 48)
    eval_cents, eval_voice = songs(np.random.default_rng(900), 48, 500, device, plant.config)
    batch_twin = repeat(twin, args.batch)

    def evaluate(model):
        with torch.no_grad():
            emitted, air, sounding, cmd = play_with_valve(actuator, model, plant, eval_cents, eval_voice, eval_rigs,
                                                          eval_twin, torch.Generator(device).manual_seed(0))
        return scores(emitted, sounding, eval_cents, eval_voice)

    rule = evaluate(RuleValve(plant.config.normalize).to(device))
    print("rule  (open on the score's notes): " + json.dumps({k: round(v, 3) for k, v in rule.items()}), flush=True)
    best, history, started = None, [], time.time()
    optimizer = torch.optim.AdamW(valve.parameters(), lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .05)
    for step in range(1, args.steps + 1):
        rigs = rigs_around(twin, args.batch, args.spread, generator)
        cents, voice = songs(rng, args.batch, args.song_steps, device, plant.config)
        emitted, air, sounding, cmd = play_with_valve(actuator, valve, plant, cents, voice, rigs, batch_twin, generator)
        if args.target == "travel":
            want = travel_target(plant, batch_twin, cents, voice).to(air.dtype)
        else:
            want = (voice & ((emitted - cents).abs() < args.tol)).to(air.dtype)
        d = plant.config.valve_delay_steps
        loss = (F.binary_cross_entropy(air[:, d:].clamp(1e-5, 1 - 1e-5), want[:, d:])
                + args.chatter * (cmd[:, 1:] - cmd[:, :-1]).abs().mean())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(valve.parameters(), 1.0); optimizer.step(); schedule.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:4d} loss {float(loss):.4f}  {time.time() - started:6.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(valve)
            score = m["glide"] + (1 - m["coverage"]) * .5 + (1 - m["on_target"])
            history.append({"step": step, **m})
            print(f"eval {step}: " + json.dumps({k: round(v, 3) for k, v in m.items()}) + f" score {score:.3f}", flush=True)
            if best is None or score < best["score"]:
                best = {"step": step, "score": score, **m}
                best_state = {k: v.detach().clone() for k, v in valve.state_dict().items()}
    valve.load_state_dict(best_state)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(valve.checkpoint(best=best, twin=args.twin, actuator=args.actuator), str(out) + ".pt")
    report = {"twin": args.twin, "actuator": args.actuator, "rule": rule, "kept": best, "history": history,
              **{k: v for k, v in vars(args).items() if k not in ("twin", "actuator")}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(str(out) + "_runtime.npz", **export_valve(valve))
    print(f"kept step {best['step']}: " + json.dumps({k: round(v, 3) for k, v in best.items() if k != 'step'})
          + "  rule: " + json.dumps({k: round(v, 3) for k, v in rule.items()}))
    print(f"saved {out}.pt, {out}.json and {out}_runtime.npz")


if __name__ == "__main__":
    main()
