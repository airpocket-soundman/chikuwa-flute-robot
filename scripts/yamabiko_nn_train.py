"""PC: train the network-only performer (flute_rl/yamabiko/nn_performer.py) around one fitted twin.

    python scripts/yamabiko_nn_train.py runs/real/twin_y1_02.json --teacher runs/real/twin_y1_02_tuned.pt \
        --out runs/real/nn_y1_01

1. imitation: the teacher (a residual performer: skeleton + network) plays songs on rigs drawn around the
   twin; the network learns its commands from what the teacher heard (teacher forcing)
2. closed loop: the network plays the simulator itself and learns from pitch error plus ``--smooth`` x the
   mean |change of PWM| per step, with the rig's static friction and the motor hiding the flute
   (PhysicalPlantConfig.yamabiko1)

The weights kept are the ones with the best score (play error + 100 x smooth x mean |dPWM|) on 48 fixed
rigs around the twin; the teacher's score is printed for comparison.  Writes ``<out>.pt``, ``<out>.json``
and ``<out>_runtime.npz`` (NumPy, for the server's play).
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
from flute_rl.yamabiko.nn_performer import NNConfig, NNTwinPerformer  # noqa: E402
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.residual_numpy import export_nn  # noqa: E402
from flute_rl.yamabiko.residual_performer import ResidualTwinPerformer  # noqa: E402
from yamabiko_twin_finetune import load_twin, repeat, rigs_around, songs  # noqa: E402


def teacher_play(teacher, plant, rigs, twin, cents, voice, generator):
    trace = []
    with torch.no_grad():
        result, _ = teacher.perform(plant, cents, voice, rigs, None, generator, twin=twin, trace=trace)
    heard = torch.stack([h for h, _ in trace], 1); valid = torch.stack([v for _, v in trace], 1)
    return result, heard, valid


def measure(result, cents, voice):
    error = result["pitch_cents"] - cents
    tv = (result["pwm"][:, 1:] - result["pwm"][:, :-1]).abs().mean()
    return error, tv


def valve_miss(result, voice):
    """Mean fraction of steps the air does not follow the score (open in a rest, shut in a note)."""
    if "valve" not in result:
        return torch.zeros((), device=voice.device)
    v = voice.to(result["valve"].dtype)
    return (v * (1 - result["valve"]) + (1 - v) * result["valve"]).mean()


def with_lead(cents, voice, rng, lead):
    """A rest of ``lead`` (lo, hi) steps before every song, the target held at its first note: the network
    homes the plunger and gets to that note in it."""
    if not lead:
        return cents, voice
    n = int(rng.integers(*lead))
    first = cents[:, :1].expand(-1, n)
    rest = torch.zeros(voice.shape[0], n, dtype=voice.dtype, device=voice.device)
    return torch.cat([first, cents], 1), torch.cat([rest, voice], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("twin")
    ap.add_argument("--teacher", required=True, help="residual performer checkpoint to imitate first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imitate-steps", type=int, default=150)
    ap.add_argument("--soft-deadband", type=float, default=0.0,
                    help="train through a softplus deadband edge of this width (evaluation keeps the exact one)")
    ap.add_argument("--positions", action="store_true", help="also feed the twin's position of target and heard")
    ap.add_argument("--all-nn", action="store_true",
                    help="every actuator command from the network: valve output, motor current input, self homing")
    ap.add_argument("--valve-weight", type=float, default=1.0, help="weight of the valve-follows-the-score cost")
    ap.add_argument("--song-slot", type=int, default=0,
                    help="practice: a song memory of this size per step (from --init, its weights start at 0)")
    ap.add_argument("--plays", type=int, default=1,
                    help="practice: plays of the same song on the same rig per episode, the song memory kept "
                         "between them; later plays weigh more")
    ap.add_argument("--eval-plays", type=int, default=5)
    ap.add_argument("--twin-free", action="store_true",
                    help="step7: drop the twin's position / velocity / positions inputs (from --init by name)")
    ap.add_argument("--init", default=None, help="start from this NN checkpoint (then imitation is skipped)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--smooth", type=float, default=.3)
    ap.add_argument("--spread", type=float, default=.12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--song-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = args.device
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    twin, temp_c, plant_name = load_twin(args.twin, device)
    plant = DifferentiableMotorFlute(PhysicalPlantConfig.named(plant_name, **({"temp_c": temp_c} if temp_c else {})))
    train_plant = DifferentiableMotorFlute(PhysicalPlantConfig.named(
        plant_name, soft_deadband=args.soft_deadband, **({"temp_c": temp_c} if temp_c else {})))
    teacher = ResidualTwinPerformer.from_checkpoint(torch.load(args.teacher, map_location=device, weights_only=False),
                                                    plant, device).eval()
    if args.init:
        upgrade = dict(valve=True, current=True, self_home=True) if args.all_nn else {}
        if args.twin_free:
            upgrade.update(twin_free=True, positions=False)
        if args.song_slot:
            upgrade.update(song_slot=args.song_slot)
        model = NNTwinPerformer.from_checkpoint(torch.load(args.init, map_location=device, weights_only=False),
                                                plant, device, **upgrade)
        args.imitate_steps = 0
    else:
        model = NNTwinPerformer(plant, NNConfig(positions=args.positions and not args.twin_free,
                                                twin_free=args.twin_free)).to(device)
    generator = torch.Generator(device).manual_seed(args.seed)
    eval_rigs = rigs_around(twin, 48, args.spread, torch.Generator(device).manual_seed(7777))
    eval_twin = repeat(twin, 48)
    lead = (120, 200) if model.config.self_home else None
    eval_cents, eval_voice = with_lead(*songs(np.random.default_rng(900), 48, 500, device, plant.config),
                                       np.random.default_rng(901), lead)
    batch_twin = repeat(twin, args.batch)

    practice = bool(model.config.song_slot)

    def evaluate():
        """(MAE, |dPWM|, score) of the last play; with a song memory the same songs are played ``--eval-plays``
        times on the same rigs and ``evaluate.per_play`` keeps every play's MAE (practice should lower it)."""
        song, per_play = None, []
        with torch.no_grad():
            model.eval()
            for play in range(args.eval_plays if practice else 1):
                result = model.perform(plant, eval_cents, eval_voice, eval_rigs, eval_twin,
                                       torch.Generator(device).manual_seed(play), song=song)
                song = result["song"]
                error, tv = measure(result, eval_cents, eval_voice)
                per_play.append(round(float(error[eval_voice].abs().mean()), 2))
            model.train()
        evaluate.per_play = per_play
        mae = float(np.mean(per_play[1:])) if practice else per_play[0]       # practice: plays 2.. count
        miss = float(valve_miss(result, eval_voice))
        evaluate.miss = miss
        return mae, float(tv), mae + 100 * args.smooth * float(tv) + 100 * args.valve_weight * miss

    evaluate.miss = 0.0
    t_result, _, _ = teacher_play(teacher, plant, eval_rigs, eval_twin, eval_cents, eval_voice,
                                  torch.Generator(device).manual_seed(0))
    t_error, t_tv = measure(t_result, eval_cents, eval_voice)
    teacher_eval = {"mae": float(t_error[eval_voice].abs().mean()), "dpwm": float(t_tv)}
    print(f"teacher: MAE {teacher_eval['mae']:.1f} cents, mean |dPWM| {teacher_eval['dpwm']:.3f}", flush=True)

    history, started = [], time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    for step in range(1, args.imitate_steps + 1):
        rigs = rigs_around(twin, args.batch, args.spread, generator)
        cents, voice = songs(rng, args.batch, args.song_steps, device, plant.config)
        result_t, heard, valid = teacher_play(teacher, plant, rigs, batch_twin, cents, voice, generator)
        pred = model.imitate(cents, voice, heard, valid, result_t["pwm"], batch_twin)
        loss = F.mse_loss(pred, result_t["pwm"])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(f"imitate {step:4d} mse {float(loss):.4f}  {time.time() - started:6.0f}s", flush=True)
    mae, tv, score = evaluate()
    print(f"{'start' if args.init else 'after imitation'}: MAE {mae:.1f} cents, mean |dPWM| {tv:.3f}"
          f"{f'  per play {evaluate.per_play}' if practice else ''}", flush=True)
    best = {"step": 0, "mae": mae, "dpwm": tv, "score": score, "per_play": evaluate.per_play}
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history.append({"step": 0, "mae": mae, "dpwm": tv})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * .3, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .03)
    for step in range(1, args.steps + 1):
        rigs = rigs_around(twin, args.batch, args.spread, generator)
        cents, voice = with_lead(*songs(rng, args.batch, args.song_steps, device, plant.config), rng, lead)
        song, loss, weights = None, 0.0, np.linspace(.3, 1.0, args.plays) if args.plays > 1 else np.ones(1)
        for weight in weights:                         # the same songs on the same rigs, the memory kept
            result = model.perform(train_plant, cents, voice, rigs, batch_twin, generator, song=song)
            song = result["song"]
            error, tv = measure(result, cents, voice)
            miss = valve_miss(result, voice)
            loss = loss + float(weight) * (F.smooth_l1_loss(error[voice] / 100, torch.zeros_like(error[voice]), beta=.2)
                                           + args.smooth * tv + args.valve_weight * miss)
        loss = loss / float(weights.sum())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); schedule.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:4d} loss {float(loss):.4f} MAE {float(error[voice].abs().mean()):6.1f} |dPWM| {float(tv):.3f}"
                  f" valve miss {float(miss):.3f}"
                  f"  {time.time() - started:6.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            mae, tv_e, score = evaluate()
            history.append({"step": step, "mae": mae, "dpwm": tv_e, "per_play": evaluate.per_play})
            print(f"eval {step}: MAE {mae:.1f} |dPWM| {tv_e:.3f} valve miss {evaluate.miss:.3f} score {score:.1f}"
                  f"{f'  per play {evaluate.per_play}' if practice else ''}", flush=True)
            if score < best["score"]:
                best = {"step": step, "mae": mae, "dpwm": tv_e, "score": score, "per_play": evaluate.per_play}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(best=best, twin=args.twin, teacher=args.teacher), str(out) + ".pt")
    report = {"twin": args.twin, "teacher": args.teacher, "teacher_eval": teacher_eval, "kept": best,
              "history": history, **{k: v for k, v in vars(args).items() if k not in ("twin", "teacher")}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(str(out) + "_runtime.npz", **export_nn(model, twin, 0))
    print(f"kept step {best['step']}: MAE {best['mae']:.1f} cents, mean |dPWM| {best['dpwm']:.3f} "
          f"(teacher {teacher_eval['mae']:.1f}, {teacher_eval['dpwm']:.3f})")
    print(f"saved {out}.pt, {out}.json and {out}_runtime.npz")


if __name__ == "__main__":
    main()
