"""Render an objective held-out E2E listening example.

The script evaluates a fixed unseen batch, selects the sample nearest the
batch median three-take error (not the best sample), and writes the exact
reference waveform seen by the model plus audible renders of its true
simulated output pitch for all takes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from export_targets import write_wav  # noqa: E402
from flute_rl.audio import synth_self  # noqa: E402
from flute_rl.yamabiko.e2e import E2EImitator  # noqa: E402
from flute_rl.yamabiko.e2e_io import SAMPLE_RATE  # noqa: E402
from train_yamabiko_e2e_rl import rollout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/yamabiko_e2e_pc_domain_rl.pt")
    ap.add_argument("--out", default="runs/e2e_listening")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--audio-domain", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    checkpoint = torch.load(args.model, map_location=args.device)
    model = E2EImitator.from_checkpoint(checkpoint, args.device).eval()
    takes = rollout(model, np.random.default_rng(args.seed + 100_000), args.batch, 3, 0.8, 1.0,
                    0.15, 0.98, args.device, deterministic=True, audio_domain=args.audio_domain)
    per_sample = np.nanmean(np.stack([take.mae for take in takes]), axis=0)
    median = float(np.nanmedian(per_sample))
    index = int(np.nanargmin(np.abs(per_sample - median)))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_wav(out / "reference.wav", takes[0].reference_audio[index], SAMPLE_RATE)
    length = int(takes[0].lengths[index])
    target = takes[0].target[index, :length]
    target_on = np.isfinite(target)
    target_audio = synth_self(np.nan_to_num(target), target_on, np.random.default_rng(8999), sr=SAMPLE_RATE)
    write_wav(out / "target_flute_timbre.wav", target_audio, SAMPLE_RATE)
    entries = []
    for k, take in enumerate(takes, 1):
        length = int(take.lengths[index])
        y = synth_self(take.true_cents[index, :length], take.true_sounding[index, :length],
                       np.random.default_rng(9000 + k), sr=SAMPLE_RATE)
        name = f"performance_take{k}.wav"
        write_wav(out / name, y, SAMPLE_RATE)
        entries.append({"take": k, "wav": name, "mean_abs_cents": round(float(take.mae[index]), 1),
                        "sounding_rate": round(float(take.sounding[index]), 3)})
    manifest = {"selection": "sample nearest median three-take MAE in fixed held-out batch",
                "model": args.model, "seed": args.seed, "batch": args.batch,
                "audio_domain": args.audio_domain, "selected_index": index,
                "batch_median_three_take_mae": round(median, 1), "reference": "reference.wav",
                "target_flute_timbre": "target_flute_timbre.wav",
                "performances": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
