"""PC: score several (ear, memory) pairs on the same real whistles, recalled and played.

    python scripts/yamabiko_memory_compare.py \
        --pair models/step8/ear_06.pt runs/memory/mem_06.pt --pair runs/ear/ear_08.pt runs/memory/mem_09.pt ...

The memory reads its ear's hidden code, so every memory is scored with the ear it was trained on.  The sets
(all left out of every training here):
  rig    the rig's held-out whistles (20261124_111404, 20261124_111830; 8 songs cut from each)
  demo   the demo videos' whistles (runs/ear/demo)
  mlend  every 8th MLEnd whistler (runs/archive/mlend_whistles/*_test_*: other people, other microphones)
Per set: the recalled score against the rule's (mae, mae after the key, within 25 cent, voice agreement) and
the share of settled note time within 25 cent when the recalled score is played on 16 rigs around the twin
(memory, memory in its own key) next to the same melody played from the rule's score (rule).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import yamabiko_memory_train as mt  # noqa: E402
from flute_rl.yamabiko.score_memory import MemoryConfig, torch_model  # noqa: E402


def load_memory(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    config = MemoryConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()})
    model = torch_model(config).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", metavar=("EAR", "MEMORY"), required=True)
    ap.add_argument("--twin", default="runs/real/twin_y1f80_05.json")
    ap.add_argument("--actuator", default="models/step7c/nn_f80_t7_tw05.pt")
    ap.add_argument("--mlend-songs", type=int, default=40)
    ap.add_argument("--rig-holdout", nargs="+", default=["20261124_111404", "20261124_111830"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    table = {}
    for ear_path, mem_path in args.pair:
        ear = mt.load_ear(ear_path, dev)
        model, config = load_memory(mem_path, dev)
        _, rig = mt.real_items(["runs/real/whistle_data", "runs/real/whistle"], args.rig_holdout,
                               ear, config, dev, np.random.default_rng(7))
        _, mlend = mt.real_items(["runs/archive/mlend_whistles"], ["_test"], ear, config, dev,
                                 np.random.default_rng(9), crops=1)
        mlend = mlend[:args.mlend_songs]
        demo, _ = mt.demo_items("runs/ear/demo", ear, config, dev)
        name = pathlib.Path(mem_path).stem
        row = {}
        for set_name, items in (("rig", rig), ("demo", demo), ("mlend", mlend)):
            with torch.no_grad():
                rec = mt.scores(model, items, dev)
                play = mt.play_scores(model, items, dev, args.twin, args.actuator)
            row[set_name] = {"recall": rec, "play": play}
            print(f"{name:10s} {set_name:6s} songs {len(items):3d}  mae {rec['mae_cents']:6.1f}  after key "
                  f"{rec['mae_after_offset']:5.1f}  within25 {rec['within_25c']:.2f}  voice {rec['voice_agree']:.2f}  |  "
                  f"play memory {play['memory']:.2f} (in key {play['memory_in_key']:.2f})  rule {play['rule']:.2f}",
                  flush=True)
        table[name] = row
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
