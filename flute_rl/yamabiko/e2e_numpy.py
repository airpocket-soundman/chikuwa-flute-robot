"""Pure-NumPy inference for the Yamabiko E2E model on Arduino UNO Q Linux.

This module intentionally does not import PyTorch.  It implements exactly the
layers used by e2e.E2EImitator so a training checkpoint can be exported to one
portable ``.npz`` file and run with the NumPy package already used by the real
rig software.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .e2e_io import FRAME, HOP, SAMPLE_RATE, audio_float, frame_audio_numpy


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _silu(x):
    return x * _sigmoid(x)


def _conv1d(x, weight, bias, stride: int, padding: int):
    x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    k = weight.shape[2]
    windows = np.lib.stride_tricks.sliding_window_view(x, k, axis=2)[:, :, ::stride, :]
    # (N,C,L,K) x (O,C,K) -> (N,O,L); tensordot reaches BLAS for the large axis.
    y = np.tensordot(windows, weight, axes=([1, 3], [1, 2])).transpose(0, 2, 1)
    return y + bias[None, :, None]


def _group_norm(x, weight, bias, groups: int = 4, eps: float = 1e-5):
    n, c, length = x.shape
    z = x.reshape(n, groups, c // groups, length)
    mean = z.mean(axis=(2, 3), keepdims=True)
    var = z.var(axis=(2, 3), keepdims=True)
    z = ((z - mean) / np.sqrt(var + eps)).reshape(x.shape)
    return z * weight[None, :, None] + bias[None, :, None]


def _linear(x, weight, bias=None):
    y = x @ weight.T
    return y if bias is None else y + bias


def _gru_step(x, h, w_ih, w_hh, b_ih, b_hh):
    gi = _linear(x, w_ih, b_ih)
    gh = _linear(h, w_hh, b_hh)
    ir, iz, inn = np.split(gi, 3, axis=-1)
    hr, hz, hn = np.split(gh, 3, axis=-1)
    reset = _sigmoid(ir + hr)
    update = _sigmoid(iz + hz)
    new = np.tanh(inn + reset * hn)
    return (1.0 - update) * new + update * h


def _gru_sequence(x, arrays, suffix=""):
    """PyTorch-compatible one-layer GRU, x (N,T,F), output (N,T,H)."""
    key = lambda name: arrays[f"reference.{name}_l0{suffix}"]
    w_ih, w_hh = key("weight_ih"), key("weight_hh")
    b_ih, b_hh = key("bias_ih"), key("bias_hh")
    n, steps = x.shape[:2]
    h = np.zeros((n, w_hh.shape[1]), np.float32)
    out = np.zeros((n, steps, h.shape[1]), np.float32)
    order = range(steps - 1, -1, -1) if suffix else range(steps)
    for t in order:
        h = _gru_step(x[:, t], h, w_ih, w_hh, b_ih, b_hh)
        out[:, t] = h
    return out


class NumpyE2EModel:
    """Numerically equivalent inference implementation of E2EImitator."""

    def __init__(self, arrays: dict[str, np.ndarray], config: dict):
        self.a = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
        self.config = dict(config)

    def audio(self, frames):
        x = np.asarray(frames, np.float32)
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True))
        x = x / (rms + 1e-3)
        x = x[:, None, :]
        for conv, norm, stride, pad in ((0, 1, 2, 31), (3, 4, 2, 4), (6, 7, 2, 3), (9, 10, 2, 2)):
            x = _conv1d(x, self.a[f"audio.body.{conv}.weight"], self.a[f"audio.body.{conv}.bias"], stride, pad)
            x = _silu(_group_norm(x, self.a[f"audio.body.{norm}.weight"], self.a[f"audio.body.{norm}.bias"]))
        # FRAME=320 becomes length 20, so AdaptiveAvgPool1d(4) is four equal bins.
        if x.shape[-1] % 4:
            bins = [x[..., math.floor(i * x.shape[-1] / 4):math.ceil((i + 1) * x.shape[-1] / 4)].mean(-1)
                    for i in range(4)]
            x = np.stack(bins, -1)
        else:
            x = x.reshape(x.shape[0], x.shape[1], 4, x.shape[-1] // 4).mean(-1)
        x = _linear(x.reshape(len(x), -1), self.a["audio.proj.weight"], self.a["audio.proj.bias"])
        x = x + np.log1p(100.0 * rms)
        return x.reshape(*shape, -1).astype(np.float32, copy=False)

    def encode_reference(self, frames):
        z = self.audio_features(frames)
        if z.ndim == 2:
            z = z[None]
        forward = _gru_sequence(z, self.a)
        reverse = _gru_sequence(z, self.a, "_reverse")
        memory = np.concatenate([forward, reverse, z], axis=-1)
        keys = _linear(memory, self.a["memory_key.weight"])
        mask = np.ones(memory.shape[:2], bool)
        return memory, keys, mask

    def audio_features(self, frames):
        z = self.audio(frames)
        ear = _linear(z, self.a["ear.weight"], self.a["ear.bias"])
        return np.concatenate([z, ear], axis=-1)

    def initial_state(self, batch=1):
        return np.zeros((batch, int(self.config["controller_hidden"])), np.float32)

    def step(self, self_frame, memory, keys, mask, state, previous, new_song=False, reference_step=None):
        own = self.audio_features(np.asarray(self_frame, np.float32).reshape(len(state), 1, FRAME))[:, 0]
        pulse = np.full((len(state), 1), float(new_song), np.float32)
        if new_song:
            state = state.copy()
            state[:, int(self.config.get("adaptation_dim", 16)):] = 0.0
        query = _linear(state, self.a["query.weight"])
        score = np.sum(keys * query[:, None, :], axis=-1) / math.sqrt(keys.shape[-1])
        score = np.where(mask, score, -1e30)
        score = score - np.max(score, axis=1, keepdims=True)
        attention = np.exp(score) * mask
        attention /= np.maximum(attention.sum(axis=1, keepdims=True), 1e-12)
        attention_context = np.sum(attention[:, :, None] * memory, axis=1)
        if reference_step is None:
            aligned = attention_context
            future_aligned = aligned
            context = attention_context
        else:
            index = np.asarray(reference_step, dtype=int).reshape(-1)
            if len(index) == 1:
                index = np.repeat(index, len(memory))
            index = np.minimum(index, mask.sum(1) - 1).clip(0)
            aligned = memory[np.arange(len(memory)), index]
            future_index = np.minimum(index + 10, mask.sum(1) - 1)
            future_aligned = memory[np.arange(len(memory)), future_index]
            context = 0.5 * (attention_context + aligned)
        profile_hidden = _silu(_linear(aligned, self.a["profile.0.weight"], self.a["profile.0.bias"]))
        profile = _sigmoid(_linear(profile_hidden, self.a["profile.2.weight"], self.a["profile.2.bias"]))
        future_profile_hidden = _silu(_linear(future_aligned, self.a["profile.0.weight"], self.a["profile.0.bias"]))
        future_profile = _sigmoid(_linear(future_profile_hidden, self.a["profile.2.weight"],
                                          self.a["profile.2.bias"]))
        self_position_hidden = _silu(_linear(own, self.a["self_position.0.weight"],
                                             self.a["self_position.0.bias"]))
        self_position = _sigmoid(_linear(self_position_hidden, self.a["self_position.2.weight"],
                                         self.a["self_position.2.bias"]))
        target_pitch = aligned[:, -2:-1]
        future_pitch = future_aligned[:, -2:-1]
        motor_features = np.concatenate([profile, self_position, profile - self_position,
                                         future_profile, future_profile - profile,
                                         target_pitch, future_pitch, future_pitch - target_pitch], axis=1)
        inp = np.concatenate([own, context, previous, motor_features, pulse], axis=1)
        state = _gru_step(inp, state, self.a["controller.weight_ih"], self.a["controller.weight_hh"],
                          self.a["controller.bias_ih"], self.a["controller.bias_hh"])
        hidden = _silu(_linear(np.concatenate([state, context, own, motor_features], axis=1),
                               self.a["action.0.weight"], self.a["action.0.bias"]))
        residual = _linear(hidden, self.a["action.2.weight"], self.a["action.2.bias"])
        plan_hidden = _silu(_linear(aligned, self.a["planner.0.weight"], self.a["planner.0.bias"]))
        plan = _linear(plan_hidden, self.a["planner.2.weight"], self.a["planner.2.bias"])
        valve_hidden = _silu(_linear(aligned, self.a["valve.0.weight"], self.a["valve.0.bias"]))
        valve = _linear(valve_hidden, self.a["valve.2.weight"], self.a["valve.2.bias"])[:, 0]
        raw = np.stack((plan[:, 0] + 0.5 * residual[:, 0], valve,
                        plan[:, 1] + residual[:, 2], profile[:, 0], self_position[:, 0]), axis=1)
        return raw, state.astype(np.float32, copy=False), attention


class NumpyE2ERuntime:
    """UNO Q stateful runtime; recurrent rig memory survives new references."""

    def __init__(self, model: NumpyE2EModel):
        self.model = model
        self.state = model.initial_state()
        self.previous = np.zeros((1, 2), np.float32)
        self.memory = self.keys = self.mask = None
        self.first = False
        self.position = 0

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["metadata"]))
        if meta.get("format") != "yamabiko-e2e-numpy-v1":
            raise ValueError("not a Yamabiko NumPy E2E checkpoint")
        arrays = {k: data[k] for k in data.files if k != "metadata"}
        return cls(NumpyE2EModel(arrays, meta["config"]))

    def start_reference(self, audio, *, keep_adaptation=True):
        frames = frame_audio_numpy(audio)
        self.memory, self.keys, self.mask = self.model.encode_reference(frames)
        if not keep_adaptation:
            self.state.fill(0.0)
        self.previous.fill(0.0)
        self.first = True
        self.position = 0
        return len(frames)

    def act(self, latest_self_audio):
        if self.memory is None:
            raise RuntimeError("start_reference must be called before act")
        x = audio_float(latest_self_audio)
        if len(x) < FRAME:
            x = np.concatenate([np.zeros(FRAME - len(x), np.float32), x])
        raw, self.state, _ = self.model.step(x[-FRAME:][None], self.memory, self.keys, self.mask,
                                             self.state, self.previous, self.first, self.position)
        pwm = float(np.tanh(raw[0, 0]))
        valve_p = float(_sigmoid(raw[0, 1]))
        done_p = float(_sigmoid(raw[0, 2]))
        self.previous[:] = (pwm, valve_p)
        self.first = False
        self.position += 1
        return pwm, valve_p >= 0.5, done_p


def export_numpy(model, path, *, source: str | None = None):
    """Write one dependency-free float32 deployment file from E2EImitator."""
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    meta = {"format": "yamabiko-e2e-numpy-v1", "config": vars(model.config), "sample_rate": SAMPLE_RATE,
            "frame": FRAME, "hop": HOP, "source": source}
    np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **arrays)
