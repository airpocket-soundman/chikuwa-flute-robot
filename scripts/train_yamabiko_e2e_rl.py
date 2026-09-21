"""Closed-loop recurrent PPO training for the raw-audio Yamabiko model.

The policy sees only the raw demonstration and its own causal microphone
waveform. Simulator pitch/position are used exclusively by the reward and
metrics. The same unknown rig and reference are played repeatedly; controller
state is carried across takes so gradients can teach adaptation itself.

Start from behaviour cloning for useful exploration:

  python scripts/train_yamabiko_e2e.py --steps 2000 --out runs/yamabiko_e2e_bc.pt
  python scripts/train_yamabiko_e2e_rl.py --init runs/yamabiko_e2e_bc.pt \
      --iters 300 --out runs/yamabiko_e2e.pt
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.audio import SOURCE_KINDS, room, synth_source  # noqa: E402
from flute_rl.targets import make_target, sample_level  # noqa: E402
from flute_rl.yamabiko import Rig, RigParams  # noqa: E402
from flute_rl.yamabiko.e2e import E2EConfig, E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_audio import RawSelfAudio  # noqa: E402
from flute_rl.yamabiko.e2e_io import FRAME, SAMPLE_RATE, frame_audio_numpy  # noqa: E402
from flute_rl.yamabiko.session import HOME_STEPS  # noqa: E402

PITCH_CLIP_CENTS = 1200.0  # preserve a gradient when the raw policy is several semitones wrong
SILENCE_PENALTY = 13.0     # must remain worse than the largest sounding pitch error (-12)
REST_LEAK_PENALTY = 0.5


@dataclass
class Take:
    reference: torch.Tensor
    own: torch.Tensor
    actions: torch.Tensor
    latent_pwm: torch.Tensor
    old_logp: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    done: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor
    mae: np.ndarray
    sounding: np.ndarray
    target: np.ndarray
    true_cents: np.ndarray
    true_sounding: np.ndarray
    reference_audio: list[np.ndarray]


def _normal_logp(value, mean, std):
    return -0.5 * ((value - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2.0 * math.pi)


def _bernoulli_logp(value, logits):
    return -F.binary_cross_entropy_with_logits(logits, value, reduction="none")


def _pad_reference(targets, rng):
    lengths = np.array([len(t) for t in targets], int)
    refs = np.zeros((len(targets), lengths.max(), FRAME), np.float32)
    audios = []
    for i, target in enumerate(targets):
        kind = str(rng.choice(SOURCE_KINDS))
        audio, _ = synth_source(target, kind, rng, sr=SAMPLE_RATE)
        audio = room(audio, SAMPLE_RATE, rng)
        audios.append(audio)
        refs[i, :len(target)] = frame_audio_numpy(audio, len(target))
    return refs, lengths, audios


def discounted_returns(reward: np.ndarray, mask: np.ndarray, gamma: float, take_weight: float):
    out = np.zeros_like(reward, np.float32)
    running = np.zeros(len(reward), np.float32)
    for t in range(reward.shape[1] - 1, -1, -1):
        running = np.where(mask[:, t], take_weight * reward[:, t] + gamma * running, 0.0)
        out[:, t] = running
    return out


@torch.no_grad()
def rollout(model, rng, batch: int, takes: int, progress: float, harsh: float, pwm_std: float,
            gamma: float, device, deterministic: bool = False, mute_self: bool = False,
            audio_domain: float = 0.0):
    targets = [make_target(rng, sample_level(rng, progress)) for _ in range(batch)]
    reference_np, lengths_np, reference_audio = _pad_reference(targets, rng)
    T = reference_np.shape[1]
    target_pad = np.full((batch, T), np.nan)
    for i, target in enumerate(targets):
        target_pad[i, :len(target)] = target
    mask_np = np.arange(T)[None, :] < lengths_np[:, None]
    reference = torch.from_numpy(reference_np).to(device)
    lengths = torch.from_numpy(lengths_np).to(device)
    memory, keys, memory_mask = model.encode_reference(reference, lengths)
    params = RigParams.sample(rng, batch, harsh=harsh)
    rig = Rig(params, np.random.default_rng(int(rng.integers(2**31))))
    audio = RawSelfAudio(batch, rng, domain=audio_domain)
    state = model.initial_state(batch, device)
    result = []

    for take_index in range(takes):
        for _ in range(HOME_STEPS):
            rig.step(-np.ones(batch), np.zeros(batch, bool))
        audio.reset()
        previous = torch.zeros(batch, 2, device=device)
        own_frames = np.zeros((batch, T, FRAME), np.float32)
        actions = np.zeros((batch, T, 2), np.float32)
        latent = np.zeros((batch, T), np.float32)
        logps = np.zeros((batch, T), np.float32)
        rewards = np.zeros((batch, T), np.float32)
        true_cents = np.full((batch, T), np.nan, np.float32)
        true_sounding = np.zeros((batch, T), bool)
        abs_sum, heard_count, sounding_count, note_count = (np.zeros(batch) for _ in range(4))
        last_pwm = np.zeros(batch)

        for t in range(T):
            active = mask_np[:, t]
            own_np = np.zeros_like(audio.buf) if mute_self else audio.buf.copy()
            own_frames[:, t] = own_np
            own = torch.from_numpy(own_np).to(device)
            old_state = state
            raw, new_state, _ = model.step(own, memory, keys, memory_mask, state, previous,
                                           torch.ones(batch, device=device) if t == 0 else None, t)
            active_t = torch.from_numpy(active).to(device)[:, None]
            state = torch.where(active_t, new_state, old_state)
            if deterministic:
                z = raw[:, 0]
                valve = (raw[:, 1] >= 0).float()
            else:
                z = raw[:, 0] + pwm_std * torch.randn(batch, device=device)
                valve = torch.bernoulli(torch.sigmoid(raw[:, 1]))
            pwm = torch.tanh(z)
            action = torch.stack([pwm, valve], 1)
            action = torch.where(active_t, action, torch.zeros_like(action))
            logp = _normal_logp(z, raw[:, 0], pwm_std) + _bernoulli_logp(valve, raw[:, 1])
            previous = torch.where(active_t, action, previous)
            pwm_np, valve_np = action[:, 0].cpu().numpy(), action[:, 1].cpu().numpy().astype(bool)
            out = rig.step(pwm_np, valve_np)
            audio.push(out["cents"], out["sounding"])
            true_cents[:, t] = out["cents"]
            true_sounding[:, t] = out["sounding"] & active

            tgt = target_pad[:, t]
            note = np.isfinite(tgt) & active
            snd = out["sounding"] & active
            error = np.abs(np.nan_to_num(out["cents"] - tgt))
            reward = np.where(note,
                              np.where(snd, -np.minimum(error, PITCH_CLIP_CENTS) / 100.0, -SILENCE_PENALTY),
                              np.where(snd, -REST_LEAK_PENALTY, 0.0))
            reward -= 0.01 * (pwm_np - last_pwm) ** 2
            reward *= active
            both = note & snd
            abs_sum += np.where(both, error, 0.0)
            heard_count += both
            sounding_count += both
            note_count += note
            last_pwm = pwm_np
            actions[:, t] = action.cpu().numpy()
            latent[:, t] = z.cpu().numpy()
            logps[:, t] = logp.cpu().numpy()
            rewards[:, t] = reward

        # Meta-RL objective: the first attempt may be exploratory; accurate
        # later attempts are the goal.  A much smaller first-take weight is
        # needed to teach the recurrent state to acquire/use rig information.
        weight = 0.25 + 1.75 * take_index / max(takes - 1, 1)
        returns = discounted_returns(rewards, mask_np, gamma, weight)
        done = np.zeros((batch, T), np.float32)
        done[np.arange(batch), lengths_np - 1] = 1.0
        result.append(Take(reference.cpu(), torch.from_numpy(own_frames), torch.from_numpy(actions),
                           torch.from_numpy(latent), torch.from_numpy(logps), torch.from_numpy(rewards),
                           torch.from_numpy(returns), torch.from_numpy(done), lengths.cpu(),
                           torch.from_numpy(mask_np), np.divide(abs_sum, heard_count,
                                                                out=np.full(batch, np.nan), where=heard_count > 0),
                           sounding_count / np.maximum(note_count, 1), target_pad.copy(), true_cents,
                           true_sounding, reference_audio))
    return result


def ppo_update(model, optimizer, takes: list[Take], device, pwm_std: float, clip: float,
               entropy_weight: float, done_weight: float, epochs: int):
    # A global return mean confounds action quality with sequence time: early
    # frames always have a longer (more negative) reward-to-go.  Use a baseline
    # across rigs at each take/time, then normalize only the residuals.
    advantages = []
    for take in takes:
        ret = take.returns.to(device)
        mask = take.mask.to(device)
        count_t = mask.sum(0).clamp_min(1)
        baseline_t = (ret * mask).sum(0) / count_t
        advantages.append((ret - baseline_t[None]) * mask)
    valid_advantages = torch.cat([adv[take.mask.to(device)] for adv, take in zip(advantages, takes)])
    std = valid_advantages.std().clamp_min(1e-6)
    stats = {}
    for _ in range(epochs):
        state = None
        optimizer.zero_grad(set_to_none=True)
        policy_sum = entropy_sum = done_sum = torch.zeros((), device=device)
        count = 0
        for take, advantage_raw in zip(takes, advantages):
            ref, own, actions = take.reference.to(device), take.own.to(device), take.actions.to(device)
            raw, state, _ = model(ref, own, take.lengths.to(device), teacher_actions=actions, state=state)
            mask = take.mask.to(device)
            z, old_logp = take.latent_pwm.to(device), take.old_logp.to(device)
            logp = _normal_logp(z, raw[..., 0], pwm_std) + _bernoulli_logp(actions[..., 1], raw[..., 1])
            ratio = torch.exp((logp - old_logp).clamp(-10.0, 10.0))
            advantage = advantage_raw / std
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantage
            policy_sum = policy_sum - torch.minimum(unclipped, clipped)[mask].sum()
            p = torch.sigmoid(raw[..., 1]).clamp(1e-6, 1 - 1e-6)
            entropy_sum = entropy_sum + (-(p * p.log() + (1 - p) * (1 - p).log()))[mask].sum()
            done_sum = done_sum + F.binary_cross_entropy_with_logits(raw[..., 2][mask],
                                                                      take.done.to(device)[mask], reduction="sum")
            count += int(mask.sum())
        loss = policy_sum / count - entropy_weight * entropy_sum / count + done_weight * done_sum / count
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        stats = {"loss": float(loss.detach()), "policy": float(policy_sum.detach() / count),
                 "entropy": float(entropy_sum.detach() / count), "done": float(done_sum.detach() / count)}
    return stats


def summary(takes):
    def mae(t):
        finite = np.isfinite(t.mae)
        return float(np.mean(t.mae[finite])) if finite.any() else float("nan")

    return "  ".join(f"take {i + 1}: {mae(t):.1f} cents, sound {np.mean(t.sounding):.2f}, "
                      f"reward {float(t.rewards[t.mask].mean()):.3f}" for i, t in enumerate(takes))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default=None, help="behaviour-cloned .pt checkpoint (recommended)")
    ap.add_argument("--out", default="runs/yamabiko_e2e.pt")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder-lr-scale", type=float, default=0.05,
                    help="learning-rate multiplier for the pretrained neural ear")
    ap.add_argument("--pwm-std", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--entropy", type=float, default=0.002)
    ap.add_argument("--done-weight", type=float, default=0.05)
    ap.add_argument("--progress", type=float, default=0.8)
    ap.add_argument("--harsh", type=float, default=1.0)
    ap.add_argument("--audio-domain", type=float, default=0.0,
                    help="raw microphone/room randomization strength (0=legacy clean, 1=sim-to-real)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-batch", type=int, default=0,
                    help="fixed validation set size (0 uses --batch)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.audio_domain < 0:
        ap.error("--audio-domain must be non-negative")
    eval_batch = args.eval_batch or args.batch

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.init:
        checkpoint = torch.load(args.init, map_location=args.device)
        model = E2EImitator.from_checkpoint(checkpoint, args.device)
    else:
        model = E2EImitator(E2EConfig()).to(args.device)
    ear_ids = {id(p) for module in (model.audio, model.ear) for p in module.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in ear_ids], "lr": args.lr},
        {"params": [p for p in model.parameters() if id(p) in ear_ids],
         "lr": args.lr * args.encoder_lr_scale},
    ], weight_decay=1e-6)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        model.eval()
        evaluation = rollout(model, np.random.default_rng(args.seed + 100_000), eval_batch, args.takes,
                             args.progress, args.harsh, args.pwm_std, args.gamma, args.device, deterministic=True,
                             audio_domain=args.audio_domain)
        muted = rollout(model, np.random.default_rng(args.seed + 100_000), eval_batch, args.takes,
                        args.progress, args.harsh, args.pwm_std, args.gamma, args.device,
                        deterministic=True, mute_self=True, audio_domain=args.audio_domain)
        print(f"feedback: {summary(evaluation)}\nmuted:    {summary(muted)}")
        return
    best = -float("inf")
    for iteration in range(1, args.iters + 1):
        model.eval()
        data = rollout(model, rng, args.batch, args.takes, args.progress, args.harsh, args.pwm_std,
                       args.gamma, args.device, audio_domain=args.audio_domain)
        model.train()
        stats = ppo_update(model, optimizer, data, args.device, args.pwm_std, args.clip,
                           args.entropy, args.done_weight, args.epochs)
        score = float(np.mean([t.rewards[t.mask].mean().item() for t in data]))
        if iteration == 1 or iteration % args.eval_every == 0:
            model.eval()
            seed = args.seed + 100_000
            evaluation = rollout(model, np.random.default_rng(seed), eval_batch, args.takes, args.progress,
                                 args.harsh, args.pwm_std, args.gamma, args.device, deterministic=True,
                                 audio_domain=args.audio_domain)
            muted = rollout(model, np.random.default_rng(seed), eval_batch, args.takes, args.progress,
                            args.harsh, args.pwm_std, args.gamma, args.device, deterministic=True,
                            mute_self=True, audio_domain=args.audio_domain)
            eval_score = float(np.mean([t.rewards[t.mask].mean().item() for t in evaluation]))
            if eval_score > best:
                best = eval_score
                checkpoint = model.checkpoint()
                checkpoint.update({"iteration": iteration, "score": eval_score,
                                   "training_score": score, "training": "recurrent-ppo-raw-audio"})
                torch.save(checkpoint, out)
            print(f"iter {iteration:4d} loss {stats['loss']:.4f} best {best:.3f}\n"
                  f"  feedback: {summary(evaluation)}\n  muted:    {summary(muted)}", flush=True)
    print(f"saved best policy to {out}")


if __name__ == "__main__":
    main()
