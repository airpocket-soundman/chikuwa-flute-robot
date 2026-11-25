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

The key (how far the whistle is moved into the flute's range) was the largest error left (mem_09: about 30 cent
per song, against 12 cent after it).  With ``hist_bins`` the memory also keeps a soft histogram of the whistle's
height: every step adds a_s softmax(h_s) over ``hist_bins`` learned bins (one more running sum of a fixed size),
from which the 2nd and 98th percentiles the rule uses can be read, untroubled by a few octave slips.  With
``key_head`` a separate small network reads the key (the shift, in cents) from G, the extremes and the
histogram, taught directly with the rule's shift as well, and the decoder gives the whistle's own pitch: the
target is clamp(pitch + shift, low, high), the rule's sum and clip as a fixed part of the graph.
With ``hist_fixed`` the histogram's bins are fixed bumps on the pitch axis (HIST_LO .. HIST_HI) of the
extremes' p (taught to be the whistle's pitch / 1000 with the trainer's --key-aux), instead of learned bins:
the key head then reads where on the pitch axis the whistle sat (mem_10's learned bins left it guessing).

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
    hist_bins: int = 0           # a soft histogram of the whistle's height (mem_10 on)
    key_head: bool = False       # the key read by its own network; the decoder gives the whistle's pitch
    hist_fixed: bool = False     # histogram bins fixed on the pitch axis of p (needs extremes)


KEY_MID, KEY_HALF = -1300.0, 1300.0      # the shift's scale: whistles sit above the flute (shift -2600 .. 0)
RAW_MID, RAW_HALF = 1700.0, 1700.0       # the whistle's pitch (cents re A4): 0 .. 3400 = 440 Hz .. 3.1 kHz
HIST_LO, HIST_HI = -600.0, 4200.0        # the fixed histogram's pitch axis (cents re A4): 310 Hz .. 4.9 kHz


def hist_centres(bins: int) -> np.ndarray:
    return np.linspace(HIST_LO, HIST_HI, bins)


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


def teacher_shift(cents: np.ndarray, config: MemoryConfig, whistle: dict | None = None) -> float:
    """How far the rule moves the whistle (cents added): the middle of the flute's range minus the middle of
    the whistle's (2nd and 98th percentiles), as ``teacher``.  0 when nothing whistles."""
    c = clean_track(cents, **(whistle or {}))
    if not np.isfinite(c).any():
        return 0.0
    return float((config.low + config.high) / 2 - (np.nanpercentile(c, 2) + np.nanpercentile(c, 98)) / 2)


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
            learned_bins = 0 if cfg.hist_fixed else cfg.hist_bins
            self.gen = nn.Sequential(nn.Linear(cfg.ear_dim, cfg.hidden), nn.ReLU(),
                                     nn.Linear(cfg.hidden, cfg.global_dim + extra + learned_bins + 1))
            if cfg.hist_fixed:
                self.register_buffer("centres", torch.tensor(hist_centres(cfg.hist_bins), dtype=torch.float32))
            g_dim = cfg.global_dim + 2 * extra + cfg.hist_bins
            self.dec = nn.Sequential(nn.Linear(cfg.value + 1 + g_dim, cfg.hidden), nn.ReLU(),
                                     nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 2))
            if cfg.key_head:
                self.key = nn.Sequential(nn.Linear(g_dim, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.hidden),
                                         nn.ReLU(), nn.Linear(cfg.hidden, 1))

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
            if self.config.hist_bins:                                       # where the whistle sits, as a histogram
                at = gd + (1 if self.config.extremes else 0)
                if self.config.hist_fixed:                                  # bumps on the pitch axis of p
                    width = (HIST_HI - HIST_LO) / (self.config.hist_bins - 1)
                    d = (1000 * g[..., gd:gd + 1] - self.centres) / width
                    h = torch.softmax(-.5 * d * d, -1)
                else:
                    h = torch.softmax(g[..., at:at + self.config.hist_bins], -1)
                G = torch.cat([G, (h * weight).sum(1) / mass], 1)
            return W, G

        def shift(self, G):
            """The key: cents the whistle is moved by (key_head only)."""
            return KEY_MID + KEY_HALF * self.key(G)[:, 0]

        def read(self, W, G, steps):
            """-> (target cents (B, T), voice logit (B, T)) at the instants ``steps`` (T,)."""
            r = torch.einsum("bvk,tk->btv", W, self.code(steps))
            out = self.dec(torch.cat([r, G[:, None].expand(-1, r.shape[1], -1)], 2))
            if self.config.key_head:
                pitch = RAW_MID + RAW_HALF * out[..., 0]
                return torch.clamp(pitch + self.shift(G)[:, None], self.config.low, self.config.high), out[..., 1]
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
    a["mem.hist_bins"] = np.array(model.config.hist_bins)
    a["mem.hist_fixed"] = np.array(int(model.config.hist_fixed))
    parts = ("enc", "gen", "dec") + (("key",) if model.config.key_head else ())
    for part in parts:
        for i, layer in enumerate(getattr(model, part)):
            if hasattr(layer, "weight"):
                a[f"mem.{part}{i}.w"] = layer.weight.detach().cpu().numpy().astype(np.float32)
                a[f"mem.{part}{i}.b"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    return a


class MemoryRuntime:
    """NumPy: ``listen(ear_code)`` every 10 ms while the whistle sounds, then ``recall(step)``.

    For the UNO Q (2026-11-25: 8.7 ms per write and 6.1 ms per read into a 129 x 8192 W, and its NumPy's matrix
    products too slow to batch them): W k_t = sum_s v_s (k_s . k_t), and k_s . k_t = g(t - s) with
    g(d) = sum_f cos(2 pi f d dt) / keys depends on the difference only.  So writing keeps the values v_s (129
    numbers) and reading is a convolution of them with g, done for many instants at once with FFTs
    (``kernel`` / ``reads_from_kernel``, in pieces small enough for step8's 10 ms).  The same sums as W; ``W`` is
    still there, made from the values when something asks for it (the PC, the tests)."""

    def __init__(self, arrays):
        self.a = {k: np.asarray(v) for k, v in arrays.items()}
        self.freqs = self.a["mem.freqs"]
        self.low, self.high = float(self.a["mem.low"]), float(self.a["mem.high"])
        value = self.a[self._last("enc") + ".w"].shape[0]
        self._W = np.zeros((value + 1, 2 * len(self.freqs)), np.float32)
        self._V = []                                     # every value written (v_s with its 1), in order
        self._in_W = 0                                   # how many of them W holds
        # the position code turns by a fixed angle every step: keep it as phasors and rotate them
        self.turn = np.exp(2j * np.pi * self.freqs * STEP_S)
        self.phasor = np.ones(len(self.freqs), np.complex128)
        self.scale = 1 / np.sqrt(len(self.freqs))
        self.beta = float(self.a["mem.beta"]) if "mem.beta" in self.a else None
        self.bins = int(self.a["mem.hist_bins"]) if "mem.hist_bins" in self.a else 0
        self.fixed = bool(int(self.a["mem.hist_fixed"])) if "mem.hist_fixed" in self.a else False
        self.key = any(k.startswith("mem.key") for k in self.a)
        learned = 0 if self.fixed else self.bins
        self.G = np.zeros(self.a[self._last("gen") + ".w"].shape[0] - 1 - (self.beta is not None) - learned, np.float64)
        self.H = np.zeros(self.bins, np.float64)
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

    @property
    def W(self):
        """The written matrix sum_s v_s k_s^T (made from the values kept, as far as it is behind)."""
        if self._in_W < len(self._V):
            steps = np.arange(self._in_W, len(self._V))
            self._W += np.stack(self._V[self._in_W:]).T @ position_code(steps, self.freqs)
            self._in_W = len(self._V)
        return self._W

    def _flush(self):
        pass

    def listen(self, ear_code):
        v = np.append(self._mlp("enc", np.asarray(ear_code, np.float32)), np.float32(1.0)).astype(np.float32)
        self._V.append(v)
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
        if self.bins:
            at = gd + (self.beta is not None)
            if self.fixed:
                width = (HIST_HI - HIST_LO) / (self.bins - 1)
                d = (1000 * float(g[gd]) - hist_centres(self.bins)) / width
                h = -.5 * d * d
            else:
                h = g[at:at + self.bins].astype(np.float64)
            h = np.exp(h - h.max())
            self.H += a * h / h.sum()
        self.n += 1

    def recall(self, step):
        """(target cents, voice probability) of the instant ``step`` (10 ms) of the song heard."""
        return self.read(step)[:2]

    def read(self, step):
        """(target cents, voice probability, mass): ``mass`` is how much was written around ``step`` (the last
        row of W read back), about the same inside the listening and near 0 past its end (step8's end)."""
        self._flush()
        return self._decode(self.W @ position_code(np.array([step]), self.freqs)[0])

    def kernel(self, lags):
        """g(d) for the differences ``lags`` (consecutive is fastest): the position codes' dot product."""
        lags = np.asarray(lags, np.int64)
        if not len(lags):
            return np.zeros(0)
        first = int(lags[0])
        if np.array_equal(lags, np.arange(first, first + len(lags))):
            z = np.exp(2j * np.pi * self.freqs * STEP_S * first) * self.turn ** np.arange(len(lags))[:, None]
            return z.real.sum(1) / len(self.freqs)
        return np.cos(2 * np.pi * lags[:, None] * STEP_S * self.freqs[None]).sum(1) / len(self.freqs)

    def kernel_lags(self, steps_to):
        """The differences t - s the reads of instants 0 .. steps_to - 1 need (as ``reads_from_kernel`` wants)."""
        return np.arange(-(self.n - 1), steps_to)

    def reads_from_kernel(self, g, steps_to, cols=None):
        """W k_t for t = 0 .. steps_to - 1, by one FFT convolution of the values kept with g (from ``kernel`` of
        ``kernel_lags(steps_to)``): rows of value + 1 numbers, as ``W @ position_code``.  ``cols``: only these
        of the value + 1 numbers (to do it in pieces)."""
        V = np.stack(self._V).astype(np.float64)            # (n, value + 1)
        if cols is not None:
            V = V[:, cols]
        n = len(V)
        size = 1 << int(np.ceil(np.log2(n + len(g))))
        y = np.fft.irfft(np.fft.rfft(V, size, axis=0) * np.fft.rfft(g, size)[:, None], size, axis=0)
        return y[n - 1:n - 1 + steps_to].astype(np.float32)   # y[m] = sum_s v_s g(m - (n - 1) - s): m = t + n - 1

    def _mlp_rows(self, part, X):
        names = sorted({k[:-2] for k in self.a if k.startswith(f"mem.{part}") and k.endswith(".w")},
                       key=lambda s: int(s[len(f"mem.{part}"):]))
        for i, name in enumerate(names):
            X = X @ self.a[name + ".w"].T + self.a[name + ".b"]
            if i < len(names) - 1:
                X = np.maximum(X, 0.0)
        return X

    def decode_many(self, rows):
        """``read`` of instants whose W k_t rows are given, all at once (one matrix product per layer)."""
        rows = np.asarray(rows, np.float32)
        if not len(rows):
            return []
        g = self._global()
        X = np.concatenate([rows, np.repeat(g[None].astype(np.float32), len(rows), 0)], 1)
        out = self._mlp_rows("dec", X)
        voice = 1 / (1 + np.exp(-np.clip(out[:, 1], -50.0, 50.0)))
        if self.key:
            shift = KEY_MID + KEY_HALF * float(self._mlp("key", g.astype(np.float32))[0])
            cents = np.clip(RAW_MID + RAW_HALF * out[:, 0] + shift, self.low, self.high)
        else:
            cents = (self.low + self.high) / 2 + (self.high - self.low) / 2 * out[:, 0]
        return [(float(c), float(v), float(m)) for c, v, m in zip(cents, voice, rows[:, -1])]

    def _global(self):
        """What the decoder reads besides W k_t: the mean code G, the extremes and the histogram."""
        g = self.G / max(self.A, 1e-3)
        if self.beta is not None:
            mass = np.log(max(self.A, 1e-3))
            g = np.r_[g, (self.hi - mass) / self.beta, -(self.lo - mass) / self.beta]
        if self.bins:
            g = np.r_[g, self.H / max(self.A, 1e-3)]
        return g

    def read_many(self, steps):
        """``read`` of every instant in ``steps`` (consecutive is fastest), with one matrix product."""
        steps = np.asarray(steps, np.int64)
        if not len(steps):
            return []
        self._flush()
        first = int(steps.min())
        span = int(steps.max()) - first + 1
        if span <= 4 * len(steps):                       # turn the phasors from the first instant
            z = np.exp(2j * np.pi * self.freqs * STEP_S * first) * self.turn ** np.arange(span)[:, None]
            codes = (np.concatenate([z.real, z.imag], 1) * self.scale).astype(np.float32)[steps - first]
        else:
            codes = position_code(steps, self.freqs)
        r = codes @ self.W.T
        return [self._decode(row) for row in r]

    def _decode(self, r):
        g = self.G / max(self.A, 1e-3)
        if self.beta is not None:
            mass = np.log(max(self.A, 1e-3))
            g = np.r_[g, (self.hi - mass) / self.beta, -(self.lo - mass) / self.beta]
        if self.bins:
            g = np.r_[g, self.H / max(self.A, 1e-3)]
        out = self._mlp("dec", np.concatenate([r, g]).astype(np.float32))
        voice = float(1 / (1 + np.exp(-np.clip(out[1], -50.0, 50.0))))
        if self.key:
            shift = KEY_MID + KEY_HALF * float(self._mlp("key", g.astype(np.float32))[0])
            return float(np.clip(RAW_MID + RAW_HALF * out[0] + shift, self.low, self.high)), voice, float(r[-1])
        mid, half = (self.low + self.high) / 2, (self.high - self.low) / 2
        return float(mid + half * out[0]), voice, float(r[-1])


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
    rows = memory.reads_from_kernel(memory.kernel(memory.kernel_lags(memory.n)), memory.n)   # all at once (fast)
    recalled = np.array([r[:2] for r in memory.decode_many(rows)])
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
