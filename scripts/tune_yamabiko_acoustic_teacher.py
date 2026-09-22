"""Grid-search the public-audio-only bootstrap controller (diagnostic tool)."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.physical_plant import DifferentiableMotorFlute
from scripts.train_yamabiko_physical_control import make_plans


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plant = DifferentiableMotorFlute()
    plan, voice = make_plans(np.random.default_rng(91), 128, 260, device)
    params = plant.parameters(128, device)
    for lookahead in (20, 25, 30, 35, 40, 50):
        for kp in (3.0, 4.0, 5.0, 6.0):
            for kd in (.4, .5, .65, .8, 1.0):
                physical = plant.initial_state(128, device)
                previous = torch.zeros(128, device=device)
                errors = []
                for t in range(plan.shape[1]):
                    cents, _ = plant.flute(physical, torch.ones(128, device=device), params)
                    heard = (cents - plant.config.low_cents) / plant.config.pitch_span_cents
                    velocity = (heard - previous) / plant.config.dt if t else torch.zeros_like(heard)
                    desired = plan[:, min(t + lookahead, plan.shape[1] - 1)]
                    pwm = (kp * (desired - heard) - kd * velocity).clamp(-1, 1)
                    physical = plant.step(physical, pwm, params)
                    errors.append((physical.position - plan[:, t]).abs())
                    previous = heard
                mae = torch.stack(errors, 1)[:, 20:].mean() * 100
                print(f"{float(mae):7.3f}  look={lookahead:2d} kp={kp:3.1f} kd={kd:4.2f}")


if __name__ == "__main__":
    main()
