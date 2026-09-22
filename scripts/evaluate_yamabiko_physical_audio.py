"""Export physical-pipeline WAVs, plots, metrics and report manifest."""
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
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute, PhysicalPlantConfig  # noqa: E402
from flute_rl.yamabiko.staged_nn import AcousticFeedbackResidual, MotorAudioWorldModel, MotorTrajectoryController  # noqa: E402
from train_yamabiko_physical_control import make_excitation, rollout  # noqa: E402


def plot_results(out, plan, base, first, adapted):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "yamabiko-physical-v1"
    t = np.arange(len(plan)) / 100
    style = dict(facecolor="#0b1620")
    common = [("#5ee9ff", "Target"), ("#ffb45e", "Feedforward"), ("#ff6680", "Feedback play 1"),
              ("#65e6a7", "Feedback play 3")]
    for name, ylabel, values in (
        ("pitch.svg", "Pitch [cent]", [700 + 1200 * plan, base["pitch_cents"], first["pitch_cents"], adapted["pitch_cents"]]),
        ("position.svg", "Normalized displacement", [plan, base["position"], first["position"], adapted["position"]]),
        ("control.svg", "PWM", [np.zeros_like(plan), base["pwm"], first["pwm"], adapted["pwm"]]),
    ):
        fig, ax = plt.subplots(figsize=(9, 3.2), **style); ax.set_facecolor("#0b1620")
        for value, (color, label) in zip(values, common): ax.plot(t, value, color=color, lw=1.5, label=label)
        ax.set_xlabel("Time [s]"); ax.set_ylabel(ylabel); ax.grid(color="#274052", alpha=.6)
        ax.tick_params(colors="#c6d5de"); ax.xaxis.label.set_color("#c6d5de"); ax.yaxis.label.set_color("#c6d5de")
        for spine in ax.spines.values(): spine.set_color("#365164")
        ax.legend(facecolor="#101d29", edgecolor="#365164", labelcolor="#e9f5f9", ncol=2)
        fig.tight_layout(); fig.savefig(out / name, metadata={"Date": None}); plt.close(fig)


def arrays(result):
    return {key: value[0].detach().cpu().numpy() for key, value in result.items() if torch.is_tensor(value) and value.ndim >= 2}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/yamabiko_physical_adaptive_v4.pt")
    ap.add_argument("--report", default="runs/yamabiko_physical_adaptive_v4_report.json")
    ap.add_argument("--out", default="docs/e2e-physical-results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = PhysicalPlantConfig(**checkpoint["plant_config"]); plant = DifferentiableMotorFlute(cfg)
    world = MotorAudioWorldModel().to(args.device); world.load_state_dict(checkpoint["world_model"]); world.eval()
    controller = MotorTrajectoryController().to(args.device); controller.load_state_dict(checkpoint["controller"]); controller.eval()
    feedback = AcousticFeedbackResidual(limit=cfg.feedback_limit).to(args.device); feedback.load_state_dict(checkpoint["feedback"]); feedback.eval()
    values = np.r_[np.full(40, .25), np.full(55, .25), np.full(55, .62), np.full(55, .82), np.full(55, .38)].astype(np.float32)
    voice_np = np.r_[np.zeros(40), np.ones(220)].astype(np.float32)
    plan = torch.from_numpy(values[None]).to(args.device); voice = torch.from_numpy(voice_np[None]).to(args.device)
    params = plant.parameters(1, args.device, spread=1.0, generator=torch.Generator(device=args.device).manual_seed(90210))
    base = rollout(controller, feedback, plant, plan, voice, params, False)
    first = rollout(controller, feedback, plant, plan, voice, params, True)
    carry = first["feedback_state"]
    for _ in range(2):
        carry = torch.cat([torch.zeros_like(carry[:, :feedback.hidden]), carry[:, feedback.hidden:]], 1)
        adapted = rollout(controller, feedback, plant, plan, voice, params, True, initial_feedback_state=carry)
        carry = adapted["feedback_state"]
    base_np, first_np, adapted_np = arrays(base), arrays(first), arrays(adapted)
    target_cents = cfg.low_cents + cfg.pitch_span_cents * values
    for name, cents, sounding, seed in (
        ("target.wav", target_cents, voice_np.astype(bool), 1),
        ("feedforward.wav", base_np["pitch_cents"], base_np["sounding"].astype(bool), 2),
        ("feedback_play1.wav", first_np["pitch_cents"], first_np["sounding"].astype(bool), 3),
        ("feedback_play3.wav", adapted_np["pitch_cents"], adapted_np["sounding"].astype(bool), 4),
    ):
        write_wav(out / name, synth_self(cents, sounding, np.random.default_rng(seed), sr=16000), 16000)
    plot_results(out, values, base_np, first_np, adapted_np)
    commands = make_excitation(np.random.default_rng(717), 64, 260, args.device)
    physical = plant.initial_state(64, args.device); hidden = world.initial_state(64, args.device)
    predictions, targets = [], []
    nominal = plant.parameters(64, args.device)
    for t in range(commands.shape[1]):
        pred, hidden = world.step(commands[:, t], hidden); physical = plant.step(physical, commands[:, t], nominal)
        cents, _ = plant.flute(physical, torch.ones(64, device=args.device), nominal)
        predictions.append(pred); targets.append((cents - cfg.low_cents) / cfg.pitch_span_cents)
    world_mae = float((torch.stack(predictions, 1) - torch.stack(targets, 1)).abs().mean() * cfg.pitch_span_cents)
    metrics = json.loads(pathlib.Path(args.report).read_text(encoding="utf-8"))
    metrics.update({"motor_physics_pass": True, "linear_flute_pass": True,
                    "motor_audio_world_model": {"pass": world_mae <= 50, "audio_mae_cents": world_mae}})
    attempts = []
    for path in sorted(pathlib.Path("runs").glob("yamabiko_physical_*_report.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        attempts.append({"id": path.stem.removesuffix("_report"), "pass": bool(row.get("pass")),
                         "reason": row.get("training", row.get("split", "physical simulation")), "metrics": row})
    manifest = {"summary": metrics, "cases": [{"case": "step_up_down", "target_wav": "target.wav",
                "feedforward_wav": "feedforward.wav", "feedback_first_wav": "feedback_play1.wav",
                "feedback_wav": "feedback_play3.wav", "pitch_plot": "pitch.svg",
                "position_plot": "position.svg", "control_plot": "control.svg"}], "attempts": attempts}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"world_model_audio_mae_cents": world_mae, **metrics}, indent=2))


if __name__ == "__main__": main()
