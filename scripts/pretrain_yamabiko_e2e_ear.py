"""Pretrain the internal neural ear from raw, timbre-randomised waveforms.

The labels are used only as a training signal.  At deployment the ear is part
of the single E2E policy and receives raw waveform samples; no pitch tracker or
pitch sequence is present in the runtime pipeline.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.audio import SOURCE_KINDS, room, synth_self, synth_source  # noqa: E402
from flute_rl.targets import make_target, sample_level  # noqa: E402
from flute_rl.yamabiko.e2e import E2EConfig, E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_audio import RawSelfAudio  # noqa: E402
from flute_rl.yamabiko.e2e_io import SAMPLE_RATE, frame_audio_numpy  # noqa: E402

PITCH_CENTER = 1300.0
PITCH_SCALE = 600.0


def make_dataset(rng: np.random.Generator, songs: int, progress: float, audio_domain: float = 0.0,
                 reference_repeats: int = 1):
    frames, pitch, voiced = [], [], []
    for _ in range(songs):
        target = make_target(rng, sample_level(rng, progress))
        kind = str(rng.choice(SOURCE_KINDS))
        source, _ = synth_source(target, kind, rng, sr=SAMPLE_RATE)
        source = room(source, SAMPLE_RATE, rng)
        on = np.isfinite(target)
        source_frames = frame_audio_numpy(source, len(target))
        for _ in range(reference_repeats):
            frames.append(source_frames)
            pitch.append(np.where(on, (np.nan_to_num(target) - PITCH_CENTER) / PITCH_SCALE, 0.0))
            voiced.append(on.astype(np.float32))

        # The same ear must understand the robot's own edge-tone timbre.
        self_audio = synth_self(np.nan_to_num(target), on, rng, sr=SAMPLE_RATE)
        self_audio = room(self_audio, SAMPLE_RATE, rng)
        frames.append(frame_audio_numpy(self_audio, len(target), causal=True))
        # A causal frame at action t ends before action t and therefore holds
        # the previous physical result.  The old current-t label was one step
        # wrong exactly at note changes.
        heard_target = np.concatenate([[0.0], np.nan_to_num(target[:-1])])
        heard_on = np.concatenate([[False], on[:-1]])
        pitch.append(np.where(heard_on, (heard_target - PITCH_CENTER) / PITCH_SCALE, 0.0))
        voiced.append(heard_on.astype(np.float32))

        if audio_domain > 0.0:
            # Cover the signed errors used by Gate 4, not only near-target
            # oracle sound.  Piecewise offsets force genuine pitch reading.
            offset = np.empty(len(target), np.float32)
            at = 0
            choices = np.array([-300.0, -100.0, -50.0, 0.0, 50.0, 100.0, 300.0])
            while at < len(target):
                stop = min(len(target), at + int(rng.integers(12, 61)))
                offset[at:stop] = float(rng.choice(choices)); at = stop
            actual = np.where(on, np.nan_to_num(target) + offset, 0.0)
            raw = RawSelfAudio(1, rng, domain=audio_domain).render(
                actual[None], on[None])[0]
            raw_heard = np.concatenate([[0.0], actual[:-1]])
            frames.append(raw)
            pitch.append(np.where(heard_on, (raw_heard - PITCH_CENTER) / PITCH_SCALE, 0.0))
            voiced.append(heard_on.astype(np.float32))
    return (torch.from_numpy(np.concatenate(frames).astype(np.float32)),
            torch.from_numpy(np.concatenate(pitch).astype(np.float32)),
            torch.from_numpy(np.concatenate(voiced).astype(np.float32)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--songs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--progress", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/yamabiko_e2e_ear.pt")
    ap.add_argument("--pc", action="store_true", help="use the capacity-first PC model")
    ap.add_argument("--audio-domain", type=float, default=0.0,
                    help="also train on causal randomized microphone audio")
    ap.add_argument("--validation-songs", type=int, default=64)
    ap.add_argument("--reference-repeats", type=int, default=1,
                    help="sampling weight for cross-timbre reference hearing")
    ap.add_argument("--init", default=None, help="continue the neural ear from an E2E checkpoint")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    x, pitch, voiced = make_dataset(rng, args.songs, args.progress, args.audio_domain, args.reference_repeats)
    vx, vpitch, vvoiced = make_dataset(np.random.default_rng(args.seed + 100_000),
                                       args.validation_songs, args.progress, args.audio_domain,
                                       args.reference_repeats)
    model = (E2EImitator.from_checkpoint(torch.load(args.init, map_location=args.device), args.device)
             if args.init else E2EImitator(E2EConfig.pc() if args.pc else E2EConfig()).to(args.device))
    optimizer = torch.optim.AdamW(list(model.audio.parameters()) + list(model.ear.parameters()), lr=args.lr)
    best = float("inf")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"dataset {len(x):,} frames  parameters {sum(p.numel() for p in model.parameters()):,}")
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(rng.integers(len(x), size=args.batch)).long()
        xb, pb, vb = x[idx].to(args.device), pitch[idx].to(args.device), voiced[idx].to(args.device)
        z = model.ear(model.audio(xb))
        pitch_loss = F.smooth_l1_loss(z[:, 0][vb > 0.5], pb[vb > 0.5])
        voice_loss = F.binary_cross_entropy_with_logits(z[:, 1], vb)
        loss = pitch_loss + 0.25 * voice_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best:
            best = value
            checkpoint = model.checkpoint()
            checkpoint.update({"training": "neural-ear-pretraining", "ear_loss": value, "ear_step": step})
            torch.save(checkpoint, out)
        if step == 1 or step % 50 == 0:
            mae = float(((z[:, 0][vb > 0.5] - pb[vb > 0.5]).abs().mean() * PITCH_SCALE).detach())
            accuracy = float(((z[:, 1] >= 0) == (vb >= 0.5)).float().mean())
            print(f"step {step:4d} loss {value:.4f} pitch_mae {mae:.1f} cents voice {accuracy:.3f}", flush=True)
    print(f"saved {out}")
    best_model = E2EImitator.from_checkpoint(torch.load(out, map_location=args.device), args.device).eval()
    with torch.inference_mode():
        total_error = voiced_count = correct = total = 0
        for start in range(0, len(vx), args.batch):
            xb = vx[start:start + args.batch].to(args.device)
            pb = vpitch[start:start + args.batch].to(args.device)
            vb = vvoiced[start:start + args.batch].to(args.device)
            z = best_model.ear(best_model.audio(xb))
            mask = vb > 0.5
            total_error += float((z[:, 0][mask] - pb[mask]).abs().sum() * PITCH_SCALE)
            voiced_count += int(mask.sum())
            correct += int(((z[:, 1] >= 0) == mask).sum())
            total += len(xb)
    print(f"heldout pitch_mae {total_error / max(voiced_count, 1):.1f} cents  "
          f"voice_accuracy {correct / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
