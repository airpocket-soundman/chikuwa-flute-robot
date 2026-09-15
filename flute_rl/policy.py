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


class ModelResidualPolicy:
    """base + scale * MLP(features), with features built from the base controller's own model.

    The base must be a PhysicsPriorPolicy (or subclass): its dead-reckoned
    plunger state gives the pitch it expects to be playing now, so the network
    sees "how far each upcoming target note is from where I am" instead of raw
    pitches. The PWM actually sent (base + residual) is fed back into the
    base's dead reckoning (and, for AdaptivePolicy, into its identification log).

    Features (2 * horizon + 4): expected error of the next `horizon` target
    frames [100 cents, clipped], their note mask, the dead-reckoned velocity,
    the servo read-back relative to the sounding angle, and the base action.
    """

    def __init__(self, base, net: MLP, scale: float = 0.3, horizon: int = 30):
        self.base, self.net, self.scale, self.horizon = base, net, scale, horizon

    @staticmethod
    def feature_dim(horizon: int = 30) -> int:
        return 2 * horizon + 4

    def reset(self, env) -> None:
        self.base.reset(env)
        self.lay = env.obs_layout

    def act(self, obs: np.ndarray) -> np.ndarray:
        from .env import CENTER_CENTS, CENTS_SCALE

        b = np.asarray(self.base.act(obs), dtype=float)
        base = self.base
        base.rewind()
        h = self.horizon
        tgt = obs[self.lay["target"]][:h] * CENTS_SCALE + CENTER_CENTS
        mask = obs[self.lay["target_mask"]][:h] > 0.5
        err = np.where(mask, np.clip((tgt - base.model_cents(base.x_hat)) / 100.0, -3.0, 3.0), 0.0)
        angle_off = obs[self.lay["angle_readback"]][0] - obs[self.lay["angle_comp"]][0]
        feats = np.concatenate([err, mask.astype(float), [base.v_hat / 0.15, angle_off], b])
        a = np.clip(b + self.scale * self.net(feats), -1.0, 1.0)
        base.commit(a[0])
        return a.astype(np.float32)


class MLPPolicy:
    """Pure learned policy (no prior)."""

    def __init__(self, net: MLP):
        self.net = net

    def reset(self, env) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.net(obs).astype(np.float32)
