"""PC: retrain the residual network around one fitted twin and export the runtime.

    python scripts/yamabiko_twin_finetune.py runs/real/twin_01.json --out runs/real/twin_01_tuned

The pre-trained residual (scripts/train_yamabiko_residual.py) learned on rigs
over the whole guessed range.  Here it learns again, starting from those
weights, on rigs drawn around the twin that scripts/yamabiko_twin_fit.py
fitted to the real rig: the skeleton always runs on that twin, and the rig it
plays is the twin with its motor, friction and deadband off by up to
``--spread`` (the real rig is unknown within the fit error), the tube a couple
of millimetres off and the hearing delay off by one step now and then.
Gradients flow through the simulator only; the real rig is never differentiated.

The weights are kept only if they beat the pre-trained ones on a fixed set of
rigs around the twin, so the result is never worse there than the input
checkpoint.  Writes ``<out>.pt``, ``<out>.json`` (before / after) and
``<out>_runtime.npz`` (the NumPy performer for the UNO Q, as yamabiko_twin_fit.py).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flute_rl.yamabiko.physical_plant import (DifferentiableMotorFlute, PhysicalPlantConfig,  # noqa: E402
                                              PhysicalPlantParameters)
from flute_rl.yamabiko.residual_numpy import export  # noqa: E402
from flute_rl.yamabiko.residual_performer import ResidualTwinPerformer  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from flute_rl.yamabiko.melodies import random_melodies  # noqa: E402

SCALED = ("torque_gain", "torque_tau_s", "inertia", "coulomb_friction", "viscous_friction",
          "max_velocity_strokes_s", "deadband")
SHIFTED = {"flute_offset_cents": 5.0, "tube_offset_m": .002, "temp_offset_c": 1.0}


def load_twin(path, device):
    """The twin of yamabiko_twin_fit.py's ``<out>.json`` as one batch row, and the room temperature."""
    report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    fields = {name: torch.tensor([value], device=device, dtype=torch.float32)
              for name, value in report["twin"].items()}
    if "hearing_delay_steps" in fields:
        fields["hearing_delay_steps"] = fields["hearing_delay_steps"].round().long()
    return PhysicalPlantParameters(**fields), report.get("temp_c"), report.get("plant", "realistic")


def songs(rng, batch, steps, device, config):
    """train_yamabiko_residual's mix of melodies, inside the rig's target range."""
    heavy = int(round(batch * .3))
    span = dict(low_cents=config.low_cents, high_cents=config.high_cents)
    a = random_melodies(rng, batch - heavy, steps, device, varied=True, **span)
    b = random_melodies(rng, heavy, steps, device, varied=True, rest_probability=.45, repeats=10, **span)
    return torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]])


def repeat(twin, batch):
    return type(twin)(**{name: None if value is None else value.expand(batch).clone()
                         for name, value in vars(twin).items()})


def rigs_around(twin, batch, spread, generator, delay_flip=.3):
    """Rigs the real one could be, given the fitted twin (one row)."""
    device = twin.torque_gain.device
    uniform = lambda: torch.rand(batch, device=device, generator=generator) * 2 - 1
    fields = dict(vars(repeat(twin, batch)))
    for name in SCALED:
        if fields[name] is not None:
            fields[name] = fields[name] * (1 + spread * uniform())
    for name, amount in SHIFTED.items():
        if fields[name] is not None:
            fields[name] = fields[name] + amount * uniform()
    fields["torque_tau_s"] = fields["torque_tau_s"].clamp_min(.011)
    if fields["deadband"] is not None:
        fields["deadband"] = fields["deadband"].clamp(0, .6)
    if fields["hearing_delay_steps"] is not None:
        flip = torch.rand(batch, device=device, generator=generator) < delay_flip
        step = torch.where(torch.rand(batch, device=device, generator=generator) < .5, -1, 1)
        fields["hearing_delay_steps"] = torch.where(flip, (fields["hearing_delay_steps"] + step).clamp_min(0),
                                                    fields["hearing_delay_steps"])
    return type(twin)(**fields)


def episode(model, plant, params, twin, cents, voice, generator, smooth=0.0):
    """Two plays of the songs (the second with the song memory of the first); the loss is the pitch error
    plus ``smooth`` x the mean |change of PWM| per step, so the plunger moves in fewer, smoother strokes
    instead of dithering around a note.  Returns (loss, [MAE play 1, play 2], [mean |dPWM| play 1, 2])."""
    memory, loss, maes, tvs = None, 0.0, [], []
    for weight in (.3, 1.0):
        result, memory = model.perform(plant, cents, voice, params, memory, generator, twin=twin)
        error = result["pitch_cents"] - cents
        tv = (result["pwm"][:, 1:] - result["pwm"][:, :-1]).abs().mean()
        loss = loss + weight * (F.smooth_l1_loss(error[voice] / 100, torch.zeros_like(error[voice]), beta=.2)
                                + smooth * tv)
        maes.append(float(error[voice].abs().mean())); tvs.append(float(tv))
    return loss / 1.3, maes, tvs


def evaluate(model, plant, rigs, twin, eval_songs, device):
    with torch.no_grad():
        model.eval()
        _, maes, tvs = episode(model, plant, rigs, twin, *eval_songs, torch.Generator(device).manual_seed(0))
        model.train()
    return maes, tvs


def score(maes, tvs, smooth):
    """What the kept weights minimise: play-2 error [cent] plus the smoothness term in the same units."""
    return maes[1] + 100 * smooth * tvs[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("twin", help="<out>.json of scripts/yamabiko_twin_fit.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="runs/yamabiko_residual_v1.pt")
    ap.add_argument("--spread", type=float, default=.12, help="motor, friction and deadband off by up to this fraction")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--song-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-rigs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="weight of the mean |change of PWM| per step (0.3: about as much as the pitch error)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    twin, temp_c, plant_name = load_twin(args.twin, device)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.named(plant_name, **({"temp_c": temp_c} if temp_c is not None else {})))
    start = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ResidualTwinPerformer.from_checkpoint(start, plant, device)

    eval_rigs = rigs_around(twin, args.eval_rigs, args.spread, torch.Generator(device).manual_seed(7777))
    eval_twin = repeat(twin, args.eval_rigs)
    eval_songs = songs(np.random.default_rng(900), args.eval_rigs, 500, device, plant.config)
    before, before_tv = evaluate(model, plant, eval_rigs, eval_twin, eval_songs, device)
    print(f"pre-trained: play1 {before[0]:.1f} play2 {before[1]:.1f} cents (MAE on rigs around the twin),"
          f" mean |dPWM| {before_tv[1]:.3f}", flush=True)
    best = {"step": 0, "play1": before[0], "play2": before[1], "dpwm": before_tv[1],
            "score": score(before, before_tv, args.smooth)}
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .1)
    generator = torch.Generator(device).manual_seed(args.seed)
    batch_twin = repeat(twin, args.batch)
    history, started = [], time.time()
    for step in range(1, args.steps + 1):
        rigs = rigs_around(twin, args.batch, args.spread, generator)
        cents, voice = songs(rng, args.batch, args.song_steps, device, plant.config)
        loss, maes, tvs = episode(model, plant, rigs, batch_twin, cents, voice, generator, args.smooth)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); schedule.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:4d} loss {float(loss.detach()):.4f} MAE {maes[0]:6.1f} {maes[1]:6.1f} |dPWM| {tvs[1]:.3f}"
                  f"  {time.time() - started:6.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            maes, tvs = evaluate(model, plant, eval_rigs, eval_twin, eval_songs, device)
            history.append({"step": step, "maes": maes, "dpwm": tvs})
            print(f"eval {step}: play1 {maes[0]:.1f} play2 {maes[1]:.1f} |dPWM| {tvs[1]:.3f}", flush=True)
            if score(maes, tvs, args.smooth) < best["score"]:
                best = {"step": step, "play1": maes[0], "play2": maes[1], "dpwm": tvs[1],
                        "score": score(maes, tvs, args.smooth)}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    kept = "pre-trained (no step did better)" if best["step"] == 0 else f"step {best['step']}"
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(best=best, seed=args.seed, twin=args.twin, spread=args.spread,
                                start=args.checkpoint), str(out) + ".pt")
    report = {"twin": args.twin, "start": args.checkpoint, "kept": kept,
              "before": {"play1": before[0], "play2": before[1], "dpwm": before_tv[1]}, "after": best,
              "history": history,
              **{k: v for k, v in vars(args).items() if k not in ("twin", "checkpoint")}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(str(out) + "_runtime.npz", **export(model, twin, 0))
    print(f"kept {kept}: play1 {before[0]:.1f} -> {best['play1']:.1f}, play2 {before[1]:.1f} -> {best['play2']:.1f},"
          f" |dPWM| {before_tv[1]:.3f} -> {best['dpwm']:.3f}")
    print(f"saved {out}.pt, {out}.json and {out}_runtime.npz")


if __name__ == "__main__":
    main()
