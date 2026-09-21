"""Train the one-model raw-audio Yamabiko policy by behaviour cloning.

This is the practical first stage for the E2E policy.  A simulator oracle
creates actuator demonstrations, but the student only receives raw reference
audio and raw self audio.  Pitch, target contours, rig parameters and simulated
position are never model inputs.  Several songs from the same random rig are
presented in sequence and the controller state is carried between them.

After cloning, the same model can be fine-tuned with a policy-gradient method;
the inference interface and checkpoint format do not change.

Example (GPU):
    python scripts/train_yamabiko_e2e.py --steps 2000 --batch 8 --songs 3 \
        --out runs/yamabiko_e2e.pt
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.audio import SOURCE_KINDS, room, synth_source  # noqa: E402
from flute_rl.targets import make_target, sample_level  # noqa: E402
from flute_rl.yamabiko import Oracle, Rig, RigParams, make_schedule  # noqa: E402
from flute_rl.yamabiko.e2e import E2EConfig, E2EImitator, SAMPLE_RATE, frame_audio  # noqa: E402
from flute_rl.yamabiko.e2e_audio import RawSelfAudio  # noqa: E402


@dataclass
class Example:
    reference: torch.Tensor
    own: torch.Tensor
    actions: torch.Tensor
    done: torch.Tensor
    profile: torch.Tensor
    position: torch.Tensor
    position_valid: torch.Tensor
    ear_pitch: torch.Tensor
    rig_target: torch.Tensor


def teacher_song(rng: np.random.Generator, rig: Rig, progress: float, audio_domain: float = 0.0) -> Example:
    """One oracle rollout, exposed to the student only as waveforms/actions."""
    target = make_target(rng, sample_level(rng, progress))
    schedule = make_schedule([[target]])
    controller = Oracle()
    controller.begin(schedule, rig)
    actions, cents, sounding, profiles, positions = [], [], [], [], []
    nominal = RigParams.nominal(1)
    profile = 0.0
    for t in range(schedule.T):
        home = bool(schedule.homing[t])
        if home:
            pwm, valve = np.array([-1.0]), np.array([False])
        else:
            pwm = np.asarray(controller.act(t), dtype=float)
            valve = schedule.valve[:, t]
        out = rig.step(pwm, valve)
        controller.update(t, pwm, out)
        if home and (t + 1 == schedule.T or not schedule.homing[t + 1]):
            controller.homed()
        if not home:
            aim = schedule.aim[0, t]
            if np.isfinite(aim):
                profile = float(nominal.x_for_cents(aim)[0] / nominal.stroke[0])
            actions.append((float(pwm[0]), float(valve[0])))
            profiles.append(np.clip(profile, 0.0, 1.0))
            positions.append(float(out["x"][0] / rig.p.stroke[0]))
            cents.append(float(out["cents"][0]))
            sounding.append(bool(out["sounding"][0]))
    controller.finish()

    kind = str(rng.choice(SOURCE_KINDS))
    reference_audio, _ = synth_source(target, kind, rng, sr=SAMPLE_RATE)
    reference_audio = room(reference_audio, SAMPLE_RATE, rng)
    steps = len(actions)
    done = np.zeros(steps, np.float32)
    done[-1] = 1.0
    return Example(
        frame_audio(reference_audio, steps),
        torch.from_numpy(RawSelfAudio(1, rng, domain=audio_domain).render(np.asarray(cents)[None],
                                                                         np.asarray(sounding)[None])[0]),
        torch.tensor(actions, dtype=torch.float32),
        torch.from_numpy(done),
        torch.tensor(profiles, dtype=torch.float32),
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(sounding, dtype=torch.float32),
        torch.tensor((np.asarray(cents) - 1300.0) / 600.0, dtype=torch.float32),
        torch.tensor([(rig.p.tube_len[0] - 0.150) / 0.008,
                      (rig.p.v_in[0] - 0.150) / 0.030,
                      (rig.p.v_out[0] - 0.150) / 0.030,
                      rig.p.press_cents[0] / 10.0], dtype=torch.float32),
    )


def teacher_sessions(rng: np.random.Generator, batch: int, songs: int, progress: float,
                     audio_domain: float = 0.0):
    sessions = []
    for _ in range(batch):
        params = RigParams.sample(rng, 1, harsh=1.0)
        rig = Rig(params, np.random.default_rng(int(rng.integers(2**31))))
        sessions.append([teacher_song(rng, rig, progress, audio_domain) for _ in range(songs)])
    return sessions


def collate(examples: list[Example], device):
    lengths = torch.tensor([len(x.actions) for x in examples], device=device)
    T = int(lengths.max())

    def pad(name, tail):
        out = torch.zeros(len(examples), T, *tail, device=device)
        for i, ex in enumerate(examples):
            value = getattr(ex, name).to(device)
            out[i, :len(value)] = value
        return out

    reference = pad("reference", (examples[0].reference.shape[-1],))
    own = pad("own", (examples[0].own.shape[-1],))
    actions = pad("actions", (2,))
    done = pad("done", ())
    profile = pad("profile", ())
    position = pad("position", ())
    position_valid = pad("position_valid", ())
    ear_pitch = pad("ear_pitch", ())
    mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
    return reference, own, actions, done, profile, position, position_valid, ear_pitch, lengths, mask


def batch_loss(model, sessions, device, self_dropout: float = 0.5, autoregressive: bool = False,
               later_take_weight: float = 1.0):
    state = None
    total = torch.zeros((), device=device)
    total_weight = 0.0
    metrics = {"pwm": 0.0, "valve": 0.0, "done": 0.0}
    for k in range(len(sessions[0])):
        batch = collate([s[k] for s in sessions], device)
        reference, own, actions, done, profile, position, position_valid, ear_pitch, lengths, mask = batch
        own_for_ear = own
        # Some sequences must start and continue without self sound.  Otherwise
        # cloning finds a circular shortcut: "open only after I hear myself",
        # which can never emit the first note at deployment.  The unmuted rows
        # still train the feedback pathway in the same model.
        if self_dropout > 0.0:
            muted = torch.rand(len(own), device=device) < self_dropout
            own = torch.where(muted[:, None, None], torch.zeros_like(own), own)
        # Keep action history physically consistent with the oracle self audio.
        # The valve no longer depends on this history (it has its own aligned
        # feed-forward head), so the old teacher-forcing valve collapse cannot
        # occur. PPO later closes the remaining deployment-distribution gap.
        history = None if autoregressive else actions
        raw, state, _ = model(reference, own, lengths, teacher_actions=history, state=state)
        pwm_error = (torch.tanh(raw[..., 0]) - actions[..., 0]) ** 2
        pwm_weight = torch.ones_like(actions[..., 0])
        pwm_loss = (pwm_error * pwm_weight)[mask].sum() / pwm_weight[mask].sum()
        valve_target = actions[..., 1][mask]
        positives = valve_target.sum().clamp_min(1.0)
        pos_weight = ((valve_target.numel() - positives) / positives).clamp(0.5, 4.0)
        valve_loss = F.binary_cross_entropy_with_logits(raw[..., 1][mask], valve_target,
                                                         pos_weight=pos_weight)
        # Ending is sparse.  Give the single positive frame equal total weight
        # to all non-ending frames without encoding any note/onset knowledge.
        done_each = F.binary_cross_entropy_with_logits(raw[..., 2], done, reduction="none")
        weights = torch.where(done > 0.5, lengths[:, None].float(), torch.ones_like(done))
        done_loss = (done_each * weights)[mask].sum() / weights[mask].sum()
        profile_loss = ((raw[..., 3] - profile) ** 2)[mask].mean()
        pos_mask = mask & position_valid.bool()
        position_loss = ((raw[..., 4] - position) ** 2)[pos_mask].mean()
        # Preserve a physically meaningful internal ear under microphone and
        # room randomization.  True pitch is a training-only auxiliary label;
        # deployment still receives only the raw waveform.
        ear_raw = model.ear(model.audio(own_for_ear))
        ear_pitch_loss = F.smooth_l1_loss(ear_raw[..., 0][pos_mask], ear_pitch[pos_mask])
        ear_voice_loss = F.binary_cross_entropy_with_logits(ear_raw[..., 1][mask], position_valid[mask])
        rig_target = torch.stack([s[k].rig_target for s in sessions]).to(device)
        rig_prediction = model.rig_estimator(state[:, :model.config.adaptation_dim])
        rig_loss = F.smooth_l1_loss(rig_prediction, rig_target)
        take_weight = later_take_weight ** k
        take_loss = (pwm_loss + 0.5 * valve_loss + 0.2 * done_loss
                     + 0.5 * profile_loss + 0.5 * position_loss + 0.15 * rig_loss
                     + 0.25 * ear_pitch_loss + 0.05 * ear_voice_loss)
        total = total + take_weight * take_loss
        total_weight += take_weight
        metrics["pwm"] += float(pwm_loss.detach())
        metrics["valve"] += float(valve_loss.detach())
        metrics["done"] += float(done_loss.detach())
    n = len(sessions[0])
    return total / total_weight, {key: value / n for key, value in metrics.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--songs", type=int, default=3)
    ap.add_argument("--progress", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr-scale", type=float, default=0.1,
                    help="learning-rate multiplier for the pretrained audio encoder and neural ear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/yamabiko_e2e.pt")
    ap.add_argument("--init", default=None, help="continue from an E2E .pt checkpoint")
    ap.add_argument("--tiny", action="store_true", help="small model for tests and pipeline checks")
    ap.add_argument("--pc", action="store_true", help="capacity-first PC model; ignored with --init")
    ap.add_argument("--self-dropout", type=float, default=0.5,
                    help="fraction of sequences with self audio muted, forcing feedforward control")
    ap.add_argument("--pool", type=int, default=0,
                    help="pre-generate this many sessions and resample them (0 regenerates every step)")
    ap.add_argument("--autoregressive", action="store_true",
                    help="feed the policy's own prior actions for exposure-bias fine-tuning")
    ap.add_argument("--later-take-weight", type=float, default=1.0,
                    help="multiply each later take's loss by this factor (for cross-take adaptation)")
    ap.add_argument("--audio-domain", type=float, default=0.0,
                    help="raw microphone/room randomization strength (0=legacy clean, 1=sim-to-real)")
    args = ap.parse_args()
    if args.later_take_weight <= 0:
        ap.error("--later-take-weight must be positive")
    if args.audio_domain < 0:
        ap.error("--audio-domain must be non-negative")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.init:
        model = E2EImitator.from_checkpoint(torch.load(args.init, map_location=args.device), args.device)
    else:
        config = (E2EConfig(audio_width=8, audio_dim=16, reference_hidden=16,
                            controller_hidden=32, adaptation_dim=8) if args.tiny
                  else E2EConfig.pc() if args.pc else E2EConfig())
        model = E2EImitator(config).to(args.device)
    ear_ids = {id(p) for module in (model.audio, model.ear) for p in module.parameters()}
    policy_params = [p for p in model.parameters() if id(p) not in ear_ids]
    ear_params = [p for p in model.parameters() if id(p) in ear_ids]
    optimizer = torch.optim.AdamW([
        {"params": policy_params, "lr": args.lr},
        {"params": ear_params, "lr": args.lr * args.encoder_lr_scale},
    ], weight_decay=1e-5)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}  device: {args.device}")
    pool = teacher_sessions(rng, args.pool, args.songs, args.progress, args.audio_domain) if args.pool else None
    for step in range(1, args.steps + 1):
        if pool is None:
            sessions = teacher_sessions(rng, args.batch, args.songs, args.progress, args.audio_domain)
        else:
            selected = rng.choice(len(pool), size=args.batch, replace=len(pool) < args.batch)
            sessions = [pool[int(i)] for i in selected]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = batch_loss(model, sessions, args.device, args.self_dropout, args.autoregressive,
                                   args.later_take_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best:
            best = value
            checkpoint = model.checkpoint()
            checkpoint.update({"step": step, "loss": value, "training": "oracle-behaviour-cloning"})
            torch.save(checkpoint, out)
        if step == 1 or step % 10 == 0:
            print(f"step {step:5d} loss {value:.4f}  pwm {metrics['pwm']:.4f}  "
                  f"valve {metrics['valve']:.4f}  done {metrics['done']:.4f}  best {best:.4f}", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
