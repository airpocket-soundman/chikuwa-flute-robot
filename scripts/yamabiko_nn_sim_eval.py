"""PC: play fixed phrases with actuator networks on rigs drawn around a twin (no rig, no sound).

    python scripts/yamabiko_nn_sim_eval.py runs/real/twin_y1f80_04.json \
        --nets models/step3/nn_f80_t3_01.pt runs/real/nn_f80_t4_s06.pt --targets f80_doremi f80_scale f80_chromatic

Per network and phrase, over ``--rigs`` rigs (spread as in training):
  hit    share of note time within 25 cent of the score, the first 0.3 s of every note left out
  mae    mean |pitch - score| over the notes
  flips  PWM sign changes (|PWM| > 0.05) per play         dpwm  mean |change of PWM| per step
The targets are models/<step>/target_<name>.npz or runs/real/target_<name>.npz.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flute_rl.yamabiko.nn_performer import NNTwinPerformer  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from yamabiko_twin_finetune import load_twin, repeat, rigs_around  # noqa: E402

ONSET = 30   # steps (0.3 s) left out at the start of every note


def target_file(name):
    for path in sorted(ROOT.glob(f"models/*/target_{name}.npz")) + [ROOT / f"runs/real/target_{name}.npz"]:
        if path.exists():
            return path
    raise FileNotFoundError(name)


def settled(cents, voice):
    """Note steps after the first ONSET steps of each note."""
    keep = np.zeros_like(voice)
    since = 0
    for t in range(len(cents)):
        new = voice[t] and (t == 0 or not voice[t - 1] or abs(cents[t] - cents[t - 1]) > 1)
        since = 0 if new else since + 1
        keep[t] = voice[t] and since >= ONSET
    return keep


def flips(pwm):
    s = np.sign(np.where(np.abs(pwm) > .05, pwm, 0))
    out = []
    for row in s:
        row = row[row != 0]
        out.append(int((row[1:] != row[:-1]).sum()))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("twin")
    ap.add_argument("--nets", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", default=["f80_doremi", "f80_scale", "f80_chromatic"])
    ap.add_argument("--rigs", type=int, default=48)
    ap.add_argument("--spread", type=float, default=.12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    twin, temp_c, plant_name = load_twin(args.twin, device)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.named(plant_name, **({"temp_c": temp_c} if temp_c else {})))
    rigs = rigs_around(twin, args.rigs, args.spread, torch.Generator(device).manual_seed(7777))
    batch_twin = repeat(twin, args.rigs)
    print(f"{'network':22s} {'target':15s} {'hit':>6s} {'mae':>6s} {'flips':>6s} {'dpwm':>6s}")
    for net in args.nets:
        model = NNTwinPerformer.from_checkpoint(torch.load(net, map_location=device, weights_only=False),
                                                plant, device).eval()
        for name in args.targets:
            data = np.load(target_file(name))
            cents_np, voice_np = data["cents"].astype(np.float32), data["voice"].astype(bool)
            cents = torch.as_tensor(cents_np, device=device)[None].expand(args.rigs, -1).contiguous()
            voice = torch.as_tensor(voice_np, device=device)[None].expand(args.rigs, -1).contiguous()
            with torch.no_grad():
                result = model.perform(plant, cents, voice, rigs, batch_twin, torch.Generator(device).manual_seed(0))
            pitch = result["pitch_cents"].cpu().numpy(); pwm = result["pwm"].cpu().numpy()
            error = np.abs(pitch - cents_np[None])
            keep = settled(cents_np, voice_np)
            hit = float((error[:, keep] <= 25).mean())
            mae = float(error[:, voice_np].mean())
            dpwm = float(np.abs(np.diff(pwm, axis=1)).mean())
            print(f"{pathlib.Path(net).stem:22s} {name:15s} {hit:6.1%} {mae:6.1f} {flips(pwm):6.1f} {dpwm:6.3f}")


if __name__ == "__main__":
    main()
