"""step6: the score kept in weights.  While listening, the ear's code of every 10 ms is written into a matrix;
while playing, the target of any instant is read back from it.  No pitch track or score array between them.

Writing (listening, step s):   W <- W + v_s k_s^T          v_s = enc(ear code of frame s) with a constant 1 appended
                               G <- G + a_s gen(ear code of frame s),  A <- A + a_s   (G / A is read)
Reading (playing, instant t):  r_t = W k_t                 (the values written near t, and how much was written)
                               target, voice = dec(r_t, G)

``k_s`` is the instant's position code: cos and sin of 2 pi f s for fixed random frequencies f up to ``max_hz``
(unit length), so ``k_s . k_t`` is a bump around s = t whose width is about 1 / max_hz.  The appended 1 makes
the last row of W the bump's mass, which the decoder uses to normalise.  ``G`` is the mean of another code
over the whole listening, weighted by a learned gate (the notes, not the rests): it tells the decoder where the whistle sits, to move it into the flute's range.
With ``extremes`` the memory also keeps a soft highest and lowest of a learned pitch-like number p_s over the
listening (hi = log(sum a_s exp(beta p_s) / A) / beta, lo the same with -p): the rule moves the whistle by the
middle of its range, which a mean cannot tell.  Two running sums, so their size does not grow with the song.
Nothing in W grows with the length of the song except the number of terms summed: long songs blur.

The targets taught are those of the rule pipeline today (flute_rl/yamabiko/whistle.py: clean_track, then
to_target's shift into [low, high] and rests holding the previous pitch), on the whistle's true pitch, in the
whistle's own time (no lead or tail: the player adds its fixed lead).  ``MemoryRuntime`` is the NumPy version.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .whistle import clean_track

STEP_S = .01


@dataclass(frozen=True)
class MemoryConfig:
    ear_dim: int = 128
    value: int = 64
    keys: int = 256              # frequencies; the key has 2 * keys numbers
    max_hz: float = 12.0
    global_dim: int = 32
    hidden: int = 256
    seed: int = 0
    low: float = -10.0           # the flute's target range (cents re A4), as the profiles
    high: float = 1200.0
    extremes: bool = False       # also keep a soft highest and lowest pitch (mem_05 on)
    beta: float = 10.0


def frequencies(config: MemoryConfig) -> np.ndarray:
    return np.random.default_rng(config.seed).uniform(0.0, config.max_hz, config.keys)


def position_code(steps: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """(len(steps), 2 * keys) unit-length codes of the instants ``steps`` (10 ms each)."""
    angle = 2 * np.pi * np.asarray(steps, np.float64)[:, None] * STEP_S * freqs[None]
    return (np.concatenate([np.cos(angle), np.sin(angle)], 1) / np.sqrt(len(freqs))).astype(np.float32)


def teacher(cents: np.ndarray, config: MemoryConfig, whistle: dict | None = None):
    """The rule pipeline's target in the whistle's time: (target cents, voice) or None when nothing whistles."""
    c = clean_track(cents, **(whistle or {}))
    voiced = np.isfinite(c)
    if not voiced.any():
        return None
    lo, hi = np.nanpercentile(c, 2), np.nanpercentile(c, 98)
    moved = np.clip(c + (config.low + config.high) / 2 - (lo + hi) / 2, config.low, config.high)
    held = moved.copy()
    first = np.flatnonzero(voiced)[0]
    held[:first] = moved[first]
    for i in range(first + 1, len(held)):
        if not voiced[i]:
            held[i] = held[i - 1]
    return held.astype(np.float32), voiced


def torch_model(config: MemoryConfig = MemoryConfig()):
    import torch
    from torch import nn

    class ScoreMemory(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.register_buffer("freqs", torch.tensor(frequencies(cfg), dtype=torch.float32))
            self.enc = nn.Sequential(nn.Linear(cfg.ear_dim, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.value))
            extra = 1 if cfg.extremes else 0
            self.gen = nn.Sequential(nn.Linear(cfg.ear_dim, cfg.hidden), nn.ReLU(),
                                     nn.Linear(cfg.hidden, cfg.global_dim + extra + 1))
            self.dec = nn.Sequential(nn.Linear(cfg.value + 1 + cfg.global_dim + 2 * extra, cfg.hidden), nn.ReLU(),
                                     nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 2))

        def code(self, steps):
            angle = 2 * torch.pi * steps[:, None].float() * STEP_S * self.freqs[None]
            return torch.cat([torch.cos(angle), torch.sin(angle)], 1) / len(self.freqs) ** .5

        def write(self, ear, mask):
            """ear (B, L, ear_dim) codes heard, mask (B, L) the steps listened to -> (W (B, value+1, 2K), G)."""
            B, L, _ = ear.shape
            v = torch.cat([self.enc(ear), torch.ones(B, L, 1, device=ear.device)], 2) * mask[..., None]
            k = self.code(torch.arange(L, device=ear.device))
            W = torch.einsum("blv,lk->bvk", v, k)
            g = self.gen(ear)
            weight = torch.sigmoid(g[..., -1:]) * mask[..., None]          # how much this step says about the key
            mass = weight.sum(1).clamp_min(1e-3)
            gd = self.config.global_dim
            G = (g[..., :gd] * weight).sum(1) / mass
            if self.config.extremes:                                        # soft highest and lowest p
                p, b = g[..., gd], self.config.beta
                logw = torch.log(weight[..., 0].clamp_min(1e-12))
                hi = (torch.logsumexp(b * p + logw, 1) - torch.log(mass[:, 0])) / b
                lo = -(torch.logsumexp(-b * p + logw, 1) - torch.log(mass[:, 0])) / b
                G = torch.cat([G, hi[:, None], lo[:, None]], 1)
            return W, G

        def read(self, W, G, steps):
            """-> (target cents (B, T), voice logit (B, T)) at the instants ``steps`` (T,)."""
            r = torch.einsum("bvk,tk->btv", W, self.code(steps))
            out = self.dec(torch.cat([r, G[:, None].expand(-1, r.shape[1], -1)], 2))
            mid, half = (self.config.low + self.config.high) / 2, (self.config.high - self.config.low) / 2
            return mid + half * out[..., 0], out[..., 1]

        def checkpoint(self, **metadata):
            return {"format": "yamabiko-score-memory-v1", "config": asdict(self.config),
                    "state_dict": self.state_dict(), **metadata}

    return ScoreMemory(config)


def export(model) -> dict:
    a = {"mem.freqs": model.freqs.cpu().numpy().astype(np.float64),
         "mem.low": np.array(model.config.low), "mem.high": np.array(model.config.high)}
    if model.config.extremes:
        a["mem.beta"] = np.array(model.config.beta)
    for part in ("enc", "gen", "dec"):
        for i, layer in enumerate(getattr(model, part)):
            if hasattr(layer, "weight"):
                a[f"mem.{part}{i}.w"] = layer.weight.detach().cpu().numpy().astype(np.float32)
                a[f"mem.{part}{i}.b"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    return a


class MemoryRuntime:
    """NumPy: ``listen(ear_code)`` every 10 ms while the whistle sounds, then ``recall(step)``."""

    def __init__(self, arrays):
        self.a = {k: np.asarray(v) for k, v in arrays.items()}
        self.freqs = self.a["mem.freqs"]
        self.low, self.high = float(self.a["mem.low"]), float(self.a["mem.high"])
        value = self.a[self._last("enc") + ".w"].shape[0]
        self.W = np.zeros((value + 1, 2 * len(self.freqs)), np.float32)
        # the position code turns by a fixed angle every step: keep it as phasors and rotate them
        self.turn = np.exp(2j * np.pi * self.freqs * STEP_S)
        self.phasor = np.ones(len(self.freqs), np.complex128)
        self.scale = 1 / np.sqrt(len(self.freqs))
        self.beta = float(self.a["mem.beta"]) if "mem.beta" in self.a else None
        self.G = np.zeros(self.a[self._last("gen") + ".w"].shape[0] - 1 - (self.beta is not None), np.float64)
        self.A = 0.0
        self.hi = self.lo = -np.inf                      # log sum a exp(+-beta p)
        self.n = 0

    def _last(self, part):
        return max((k[:-2] for k in self.a if k.startswith(f"mem.{part}") and k.endswith(".w")),
                   key=lambda s: int(s[len(f"mem.{part}"):]))

    def _mlp(self, part, x):
        names = sorted({k[:-2] for k in self.a if k.startswith(f"mem.{part}") and k.endswith(".w")},
                       key=lambda s: int(s[len(f"mem.{part}"):]))
        for i, name in enumerate(names):
            x = self.a[name + ".w"] @ x + self.a[name + ".b"]
            if i < len(names) - 1:
                x = np.maximum(x, 0.0)
        return x

    def listen(self, ear_code):
        k = (np.concatenate([self.phasor.real, self.phasor.imag]) * self.scale).astype(np.float32)
        v = np.append(self._mlp("enc", np.asarray(ear_code, np.float32)), np.float32(1.0)).astype(np.float32)
        self.W += np.outer(v, k)
        self.phasor *= self.turn
        if self.n % 1000 == 999:
            self.phasor /= np.abs(self.phasor)
        g = self._mlp("gen", np.asarray(ear_code, np.float32))
        a = 1 / (1 + np.exp(-np.clip(g[-1], -50.0, 50.0)))
        gd = len(self.G)
        self.G += a * g[:gd]
        self.A += a
        if self.beta is not None:
            la = np.log(max(a, 1e-12))
            self.hi = np.logaddexp(self.hi, self.beta * float(g[gd]) + la)
            self.lo = np.logaddexp(self.lo, -self.beta * float(g[gd]) + la)
        self.n += 1

    def recall(self, step):
        """(target cents, voice probability) of the instant ``step`` (10 ms) of the song heard."""
        r = self.W @ position_code(np.array([step]), self.freqs)[0]
        g = self.G / max(self.A, 1e-3)
        if self.beta is not None:
            mass = np.log(max(self.A, 1e-3))
            g = np.r_[g, (self.hi - mass) / self.beta, -(self.lo - mass) / self.beta]
        out = self._mlp("dec", np.concatenate([r, g]).astype(np.float32))
        mid, half = (self.low + self.high) / 2, (self.high - self.low) / 2
        return float(mid + half * out[0]), float(1 / (1 + np.exp(-np.clip(out[1], -50.0, 50.0))))


def score_from_recording(pcm, ear, memory_arrays, lead_steps: int = 60, tail_steps: int = 60):
    """The whistle copy of step6: every 10 ms of a 16 kHz recording goes through the ear network (``ear``, an
    ear_nn.Ear) into the memory, then the whole target is read back from it.  Returns (target cents, voice,
    heard cents (the ear's own pitch, for the page), report).  The player's fixed lead and tail are added (the
    lead holds the first recalled pitch while the plunger gets there); nothing after the last recalled note
    but the tail is kept."""
    from .ear_nn import FRAME, decode
    x = np.asarray(pcm)
    x = x / 32768.0 if x.dtype.kind in "iu" else x.astype(np.float64)
    memory = MemoryRuntime(memory_arrays)
    heard = []
    for end in range(FRAME, len(x) + 1, 160):
        (logits, v), code = ear._run(x[end - FRAME:end])
        cents, p = decode(logits, v, ear.low_cents)
        heard.append(cents if p >= ear.threshold else np.nan)
        memory.listen(code)
    recalled = np.array([memory.recall(i) for i in range(memory.n)])
    if not len(recalled) or not (recalled[:, 1] >= .5).any():
        raise ValueError("no whistle heard")
    cents, voice = recalled[:, 0], recalled[:, 1] >= .5
    last = np.flatnonzero(voice)[-1]
    cents, voice = cents[:last + 1], voice[:last + 1]
    first = cents[np.flatnonzero(voice)[0]]
    target = np.r_[np.full(lead_steps, first), cents, np.full(tail_steps, cents[-1])].astype(np.float32)
    voice = np.r_[np.zeros(lead_steps, bool), voice, np.zeros(tail_steps, bool)]
    report = {"memory": "score_memory", "listened_steps": int(memory.n), "voiced_steps": int(voice.sum()),
              "steps": int(len(target))}
    return target, voice, np.array(heard), report
