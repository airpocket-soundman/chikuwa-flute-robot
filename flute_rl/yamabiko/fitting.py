"""Fit an unknown rig: a short calibration run and a digital twin.

F1 (learning mode on the rig): after homing, the valve stays open while a
fixed PWM sequence runs (deadband staircase, full-speed runs, medium runs,
slow sweeps).  Only commands and heard pitch are recorded.

F2 (on the PC): the differentiable simulator is used as the *shape* of the
rig and its parameters are fitted to that record, one parameter set per rig.
The home run fixes the start (position 0, pressed against the stop), so the
record is replayed forward from there; the fitted prefix grows to the whole
record.  The integer hearing delay is chosen by fitting each candidate.
Nothing but the record is used; the rig's true values only score the fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .device import PlayLog
from .physical_plant import DifferentiableMotorFlute, PhysicalPlantParameters

# (pwm, steps) segments of the calibration run, about 9 s at 100 Hz.  The
# full-speed runs are long enough (1 s) for the slowest motors to reach their
# top speed; a fast motor simply rests against the far stop meanwhile.
CALIBRATION = ([(level, 25) for level in (.10, .15, .20, .25, .30, .35, .40)] + [(0.0, 10)] +
               [(-1.0, 60), (1.0, 100), (-1.0, 100), (.6, 50), (-.6, 50), (.35, 150), (-.35, 150)])

FITTED = ("torque_gain", "torque_tau_s", "coulomb_friction", "viscous_friction",
          "max_velocity_strokes_s", "deadband", "tube_offset_m", "temp_offset_c", "flute_offset_cents")
LOG_SCALED = ("torque_gain", "torque_tau_s", "coulomb_friction", "viscous_friction", "max_velocity_strokes_s")


def calibration_pwm(batch, device, dtype=torch.float32):
    pwm = torch.cat([torch.full((steps,), level) for level, steps in CALIBRATION])
    return pwm.to(device=device, dtype=dtype)[None].expand(batch, -1).contiguous()


def run_calibration(rig):
    """Play the calibration run on a rig (black box or real); returns its log."""
    device = rig._parameters.torque_gain.device if hasattr(rig, "_parameters") else "cpu"
    pwm = calibration_pwm(rig.batch, device)
    with torch.no_grad():
        state = rig.reset(device)
        valve = torch.ones_like(pwm)
        heard, valid, emitted = [], [], []
        for t in range(pwm.shape[1]):
            h, v, state, e = rig.step(state, pwm[:, t], valve[:, t])
            heard.append(h); valid.append(v); emitted.append(e)
        heard, valid, emitted = (torch.stack(x, 1) for x in (heard, valid, emitted))
    rig.record(torch.zeros_like(pwm), torch.zeros_like(pwm, dtype=torch.bool), pwm, valve, heard, valid, emitted)
    return rig.logs[-1]


@dataclass
class TwinFit:
    parameters: PhysicalPlantParameters
    delay: torch.Tensor          # (B,) integer hearing delay chosen per rig
    heard_mae: torch.Tensor      # (B,) cents on the fitted record


class TwinFitter:
    def __init__(self, plant: DifferentiableMotorFlute):
        self.plant = plant

    def _unpack(self, raw, nominal):
        values = {}
        for name in FITTED:
            base = getattr(nominal, name)
            if name in LOG_SCALED:
                values[name] = base * torch.exp(raw[name])
            elif name == "deadband":
                values[name] = torch.sigmoid(raw[name]) * .6
            else:
                values[name] = base + raw[name] * {"tube_offset_m": .01, "temp_offset_c": 8.0,
                                                   "flute_offset_cents": 20.0}[name]
        fields = {k: v for k, v in vars(nominal).items()}
        fields.update(values)
        return type(nominal)(**fields)

    def predict(self, params, pwm, delay):
        """Heard pitch (cents) of the twin for the logged commands, from the home stop."""
        plant = self.plant
        batch, steps = pwm.shape
        state = plant.initial_state(batch, pwm.device, pwm.dtype)
        state.torque = -params.torque_gain  # pressed against the home stop after homing
        emitted = []
        for t in range(steps):
            state = plant.step(state, pwm[:, t], params)
            emitted.append(plant.cents_at(state.position, params))
        emitted = torch.stack(emitted, 1)
        shifted = torch.cat([emitted[:, :1].expand(-1, int(delay)), emitted[:, :steps - int(delay)]], 1) \
            if delay else emitted
        return shifted

    def fit(self, log: PlayLog, delays=(0, 1, 2, 3), steps: int = 300, lr: float = .05) -> TwinFit:
        batch, length = log.pwm.shape
        device, dtype = log.pwm.device, log.pwm.dtype
        nominal = self.plant.parameters(batch, device, dtype)
        best = None
        for delay in delays:
            raw = {name: torch.zeros(batch, device=device, dtype=dtype, requires_grad=True) for name in FITTED}
            with torch.no_grad():
                raw["deadband"].fill_(float(torch.logit(torch.tensor(max(self.plant.config.deadband, .02) / .6))))
            optimizer = torch.optim.Adam(raw.values(), lr=lr)
            for step in range(steps):
                horizon = min(length, 40 + (length - 40) * step // max(1, int(steps * .6)))
                params = self._unpack(raw, nominal)
                predicted = self.predict(params, log.pwm[:, :horizon], delay)
                mask = log.valid[:, :horizon]
                error = (predicted - log.heard[:, :horizon]) / 100
                loss = F.smooth_l1_loss(error[mask], torch.zeros_like(error[mask]), beta=.2)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            with torch.no_grad():
                params = self._unpack(raw, nominal)
                predicted = self.predict(params, log.pwm, delay)
                error = ((predicted - log.heard).abs() * log.valid).sum(1) / log.valid.sum(1).clamp_min(1)
            candidate = TwinFit(params, torch.full((batch,), delay, device=device, dtype=torch.long), error)
            if best is None:
                best = candidate
            else:
                better = candidate.heard_mae < best.heard_mae
                merged = {k: (torch.where(better, getattr(candidate.parameters, k), getattr(best.parameters, k))
                              if torch.is_tensor(getattr(best.parameters, k)) and getattr(best.parameters, k).dim() == 1
                              else getattr(best.parameters, k))
                          for k in vars(best.parameters)}
                best = TwinFit(type(best.parameters)(**merged), torch.where(better, candidate.delay, best.delay),
                               torch.minimum(candidate.heard_mae, best.heard_mae))
        best.parameters.hearing_delay_steps = best.delay
        return best


def cached_twin(plant, true_parameters, seed, fit_steps, device, cache_dir="runs/twins"):
    """Calibrate the benchmark rigs and fit their twins once; reuse the result.

    The cache key is the rig seed, count and fit steps.  Only the twin (and
    the chosen delays) is stored, never the rigs' true values.
    """
    import pathlib
    from .device import BlackBoxDevice
    rigs = true_parameters.torque_gain.shape[0]
    path = pathlib.Path(cache_dir) / f"twin_seed{seed}_rigs{rigs}_steps{fit_steps}.pt"
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        return type(true_parameters)(**saved["twin"])
    rig = BlackBoxDevice(plant, true_parameters, torch.Generator(device).manual_seed(seed + 1))
    fit = TwinFitter(plant).fit(run_calibration(rig), steps=fit_steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"twin": {k: v for k, v in vars(fit.parameters).items()}, "heard_mae": fit.heard_mae}, path)
    return fit.parameters
