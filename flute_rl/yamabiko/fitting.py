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

from .calibration import CALIBRATION, calibration_segments  # noqa: F401  (re-exported)
from .device import PlayLog
from .physical_plant import DifferentiableMotorFlute, PhysicalPlantParameters

FITTED = ("torque_gain", "torque_tau_s", "coulomb_friction", "viscous_friction",
          "max_velocity_strokes_s", "deadband", "tube_offset_m", "temp_offset_c", "flute_offset_cents")
LOG_SCALED = ("torque_gain", "torque_tau_s", "coulomb_friction", "viscous_friction", "max_velocity_strokes_s")


def calibration_pwm(batch, device, dtype=torch.float32, max_pwm: float = 1.0, sweep: float = 1.0):
    pwm = torch.cat([torch.full((steps,), level) for level, steps in calibration_segments(max_pwm, sweep)])
    return pwm.to(device=device, dtype=dtype)[None].expand(batch, -1).contiguous()


def run_calibration(rig, max_pwm: float = 1.0, sweep: float = 1.0):
    """Play the calibration run on a rig (black box or real); returns its log."""
    device = rig._parameters.torque_gain.device if hasattr(rig, "_parameters") else getattr(rig, "device", "cpu")
    pwm = calibration_pwm(rig.batch, device, max_pwm=max_pwm, sweep=sweep)
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
        """Fit every rig once per candidate delay, all candidates in one batch (the delay only shifts the
        predicted track), and keep the candidate with the smallest heard error per rig."""
        batch, length = log.pwm.shape
        device, dtype = log.pwm.device, log.pwm.dtype
        k = len(delays)
        rows = lambda x: x.repeat(k, *([1] * (x.dim() - 1)))          # (k*B, ...), delay-major
        pwm, heard, valid = rows(log.pwm), rows(log.heard), rows(log.valid)
        delay_of_row = torch.tensor(delays, device=device).repeat_interleave(batch)
        nominal = self.plant.parameters(k * batch, device, dtype)
        raw = {name: torch.zeros(k * batch, device=device, dtype=dtype, requires_grad=True) for name in FITTED}
        with torch.no_grad():
            raw["deadband"].fill_(float(torch.logit(torch.tensor(max(self.plant.config.deadband, .02) / .6))))
        optimizer = torch.optim.Adam(raw.values(), lr=lr)

        def shifted(params, upto):
            emitted = self.predict(params, pwm[:, :upto], 0)
            parts = []
            for j, delay in enumerate(delays):
                part = emitted[j * batch:(j + 1) * batch]
                if delay:
                    part = torch.cat([part[:, :1].expand(-1, int(delay)), part[:, :upto - int(delay)]], 1)
                parts.append(part)
            return torch.cat(parts, 0)

        for step in range(steps):
            horizon = min(length, 40 + (length - 40) * step // max(1, int(steps * .6)))
            params = self._unpack(raw, nominal)
            predicted = shifted(params, horizon)
            mask = valid[:, :horizon]
            error = (predicted - heard[:, :horizon]) / 100
            # the mean over each row's own valid steps, summed over rows: every candidate learns as if alone
            per_row = F.smooth_l1_loss(error, torch.zeros_like(error), beta=.2, reduction="none") * mask
            loss = (per_row.sum(1) / mask.sum(1).clamp_min(1)).sum()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            params = self._unpack(raw, nominal)
            predicted = shifted(params, length)
            error = ((predicted - heard).abs() * valid).sum(1) / valid.sum(1).clamp_min(1)
        error = error.view(k, batch)
        pick = error.argmin(0)                                         # (B,) best candidate per rig
        index = pick * batch + torch.arange(batch, device=device)
        chosen = {name: (value[index] if torch.is_tensor(value) and value.dim() == 1 and value.shape[0] == k * batch
                         else value) for name, value in vars(params).items()}
        best = TwinFit(type(params)(**chosen), delay_of_row[index], error.min(0).values)
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
