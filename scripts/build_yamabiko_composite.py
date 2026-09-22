"""Bundle all adopted Gate checkpoints into one connected model checkpoint."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.composite import YamabikoComposite  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ear", default="runs/yamabiko_e2e_pc_ear_timbre_v2.pt")
    ap.add_argument("--timeline-position", default="runs/yamabiko_timeline_position_v5.pt")
    ap.add_argument("--comparator", default="runs/yamabiko_staged_nn_comparator.pt")
    ap.add_argument("--physical", default="runs/yamabiko_physical_fast_v2.pt")
    ap.add_argument("--out", default="runs/yamabiko_connected_composite_v1.pt")
    ap.add_argument("--report", default="runs/yamabiko_connected_composite_v1_report.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model = YamabikoComposite.from_checkpoints(
        args.ear, args.timeline_position, args.comparator, args.physical, args.device).eval()
    torch.save(model.checkpoint(), args.out)
    parameters = sum(p.numel() for p in model.parameters())
    report = {
        "format": "yamabiko-connected-composite-v1", "parameters": parameters,
        "tempo_scale": model.tempo_scale, "articulation_gap_ms": model.articulation_frames * 10,
        "reference_audio_to_actuator_connected": True,
        "self_audio_neural_ear_connected": True,
        "error_comparator_connected": True,
        "adaptive_feedback_connected": True,
        "deterministic_motor_flute_connected": True,
        "real_rig_validated": False,
        "sources": {"ear": args.ear, "timeline_position": args.timeline_position,
                    "comparator": args.comparator, "physical": args.physical},
    }
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); print(f"saved {args.out}")


if __name__ == "__main__": main()
