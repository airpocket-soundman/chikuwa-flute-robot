"""numpy-only policies (no deep-learning framework needed, so they run on the
Linux side of the UNO Q as-is)."""
from __future__ import annotations

import numpy as np


class MLP:
    """One-hidden-layer tanh network with a flat parameter vector (for ES)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32, rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.in_dim, self.out_dim, self.hidden = in_dim, out_dim, hidden
        self.shapes = [(in_dim, hidden), (hidden,), (hidden, out_dim), (out_dim,)]
        w1 = rng.standard_normal((in_dim, hidden)) / np.sqrt(in_dim)
        # zero output layer: a fresh residual network adds nothing to the base policy
        self.set_flat(np.concatenate([w1.ravel(), np.zeros(hidden), np.zeros(hidden * out_dim), np.zeros(out_dim)]))

    @property
    def n_params(self) -> int:
        return int(sum(np.prod(s) for s in self.shapes))

    def get_flat(self) -> np.ndarray:
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    def set_flat(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        parts, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            parts.append(theta[i:i + n].reshape(s))
            i += n
        self.w1, self.b1, self.w2, self.b2 = parts

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)

    def save(self, path, **extra) -> None:
        np.savez(path, theta=self.get_flat(), in_dim=self.in_dim, out_dim=self.out_dim, hidden=self.hidden, **extra)

    @classmethod
    def load(cls, path) -> tuple["MLP", dict]:
        d = dict(np.load(path))
        net = cls(int(d.pop("in_dim")), int(d.pop("out_dim")), int(d.pop("hidden")))
        net.set_flat(d.pop("theta"))
        return net, d


class ResidualPolicy:
    """base policy + scale * MLP([obs, base_action]) (residual learning on a physics prior)."""

    def __init__(self, base, net: MLP, scale: float = 0.3):
        self.base, self.net, self.scale = base, net, scale

    def reset(self, env) -> None:
        self.base.reset(env)

    def act(self, obs: np.ndarray) -> np.ndarray:
        b = np.asarray(self.base.act(obs), dtype=float)
        r = self.net(np.concatenate([obs, b]))
        return np.clip(b + self.scale * r, -1.0, 1.0).astype(np.float32)


class MLPPolicy:
    """Pure learned policy (no prior)."""

    def __init__(self, net: MLP):
        self.net = net

    def reset(self, env) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.net(obs).astype(np.float32)
