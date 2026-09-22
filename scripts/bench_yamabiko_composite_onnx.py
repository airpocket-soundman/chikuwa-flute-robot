"""Run the full connected ONNX pipeline with a deterministic hardware simulator."""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import onnxruntime as ort


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


class Plant:
    def __init__(self, config): self.config = config; self.reset()
    def reset(self): self.position = self.velocity = self.torque = 0.0
    def flute(self, valve):
        low, high = self.config["low_cents"], self.config["high_cents"]
        return low + (high - low) * self.position, valve >= .5
    def step(self, pwm):
        cfg = self.config; dt = cfg["dt"]
        alpha = min(dt / cfg["torque_tau_s"], 1.0)
        self.torque += alpha * (cfg["torque_gain"] * np.clip(pwm, -1, 1) - self.torque)
        friction = cfg["coulomb_friction"] * np.tanh(self.velocity / .015) + cfg["viscous_friction"] * self.velocity
        acceleration = (self.torque - friction) / cfg["inertia"]
        vmax = cfg["max_velocity_strokes_s"]
        self.velocity = float(np.clip(self.velocity + dt * acceleration, -vmax, vmax))
        raw = self.position + dt * self.velocity
        self.position = float(np.clip(raw, 0, 1))
        if (raw <= 0 and self.velocity < 0) or (raw >= 1 and self.velocity > 0): self.velocity = 0.0


class Renderer:
    def __init__(self): self.previous = np.zeros(160, np.float32); self.phase = 0.0
    def step(self, cents, sounding):
        frequency = 440.0 * 2.0 ** (cents / 1200.0)
        index = np.arange(1, 161, dtype=np.float32)
        angle = self.phase + 2 * np.pi * frequency * index / 16000.0
        fresh = ((.24 * np.sin(angle) + .05 * np.sin(2 * angle)) * float(sounding)).astype(np.float32)
        frame = np.concatenate([self.previous, fresh])[None]
        self.previous, self.phase = fresh, float(angle[-1] % (2 * np.pi))
        return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", nargs="?", default="runs/yamabiko_composite_onnx")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(); root = pathlib.Path(args.model_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    options = ort.SessionOptions(); options.intra_op_num_threads = args.threads; options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    listen = ort.InferenceSession(str(root / manifest["graphs"]["listen"]), options,
                                  providers=["CPUExecutionProvider"])
    control = ort.InferenceSession(str(root / manifest["graphs"]["control"]), options,
                                   providers=["CPUExecutionProvider"])
    reference = np.load(root / "demo_reference_frames.npy")
    t0 = time.perf_counter(); target, position, bpm, _ = listen.run(None, {"reference_frames": reference})
    listen_ms = 1000 * (time.perf_counter() - t0); target = target.astype(np.float32); position = position.astype(np.float32)
    steps = target.shape[1]; slow = np.zeros((1, manifest["feedback_hidden"]), np.float32)
    takes, all_times = [], []
    for take in range(args.repetitions):
        plant, renderer = Plant(manifest["plant_config"]), Renderer(); pwm = valve = 0.0
        error = heard = np.zeros(1, np.float32)
        controller_state = np.zeros((1, manifest["controller_hidden"]), np.float32)
        feedback_state = np.concatenate([np.zeros_like(slow), slow], 1)
        emitted, voiced, step_times = [], [], []
        for t in range(steps):
            cents, sounding = plant.flute(valve); frame = renderer.step(cents, sounding)
            ids = [t, min(t + 5, steps - 1), min(t + 20, steps - 1), min(t + 50, steps - 1)]
            feeds = {"plan_taps": position[:, ids], "target": target[:, t],
                     "target_future_pitch": target[:, ids[2], 0], "self_frame": frame,
                     "sounding": np.array([sounding], np.float32),
                     "previous_pwm": np.array([pwm], np.float32), "previous_error": error,
                     "previous_heard": heard, "controller_state": controller_state,
                     "feedback_state": feedback_state}
            tic = time.perf_counter(); outputs = control.run(None, feeds)
            step_times.append(1000 * (time.perf_counter() - tic))
            (pwm_a, valve_a, _, _, _, _, _, error, heard,
             controller_state, feedback_state) = outputs
            pwm, valve = float(pwm_a[0]), float(valve_a[0]); plant.step(pwm)
            played, is_voiced = plant.flute(valve); emitted.append(played); voiced.append(is_voiced)
        slow = feedback_state[:, manifest["feedback_hidden"]:].copy()
        wanted = 1300.0 + 600.0 * target[0, :, 0]
        valid = (sigmoid(target[0, :, 1]) >= .5) & np.asarray(voiced)
        mae = float(np.mean(np.abs(np.asarray(emitted)[valid] - wanted[valid]))) if valid.any() else float("nan")
        all_times.extend(step_times); takes.append({"take": take + 1, "pitch_mae_cents": mae,
                                                    "step_p99_ms": float(np.percentile(step_times, 99))})
    try:
        import resource
        peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except ImportError:
        peak_rss_kib = None
    result = {"format": "yamabiko-uno-q-onnx-hil-v1", "machine": __import__("platform").machine(),
              "onnxruntime": ort.__version__, "listen_ms": listen_ms, "playback_steps": steps,
              "listen_onnx_bytes": (root / manifest["graphs"]["listen"]).stat().st_size,
              "control_onnx_bytes": (root / manifest["graphs"]["control"]).stat().st_size,
              "peak_rss_kib": peak_rss_kib,
              "step_median_ms": float(np.median(all_times)), "step_p95_ms": float(np.percentile(all_times, 95)),
              "step_p99_ms": float(np.percentile(all_times, 99)), "realtime_10ms_pass": bool(np.percentile(all_times, 99) < 10),
              "takes": takes, "hardware": "deterministic motor/flute simulator on UNO Q Linux",
              "real_motor_used": False}
    print(json.dumps(result, indent=2))
    if args.out: pathlib.Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
