"""Controllers for the Yamabiko rig, all batched over rigs.

Every controller steers the plunger toward the position that the *nominal*
tube model gives for the next note, using a "belief" of where the plunger
is. They differ only in where that belief comes from:

* OpenLoop     dead reckoning of the PWM it sent, nominal actuator model (no sensor)
* Encoder      the true position (a position sensor), nominal tube model
* Oracle       the true position and this rig's true tube model (upper bound)
* Observer     dead reckoning corrected by the delayed pitch (hand-written);
               forgets everything at each song
* ExternalFit  Observer + a least-squares fit of this rig's actuator speeds and
               tube offset from all songs so far, kept OUTSIDE any model
               (the "external memory" reference)
* GRUPolicy    a recurrent network: all it knows about this rig lives in its
               hidden state, which is carried across songs (reset only at power on)

Interface (called by session.run_session):
    begin(sched, rig), song_start(k), act(t) -> pwm, update(t, pwm, out), homed(), finish()
"""
from __future__ import annotations

import numpy as np

from ..sim import DT, hz_to_cents
from .rig import NOMINAL, OVERBLOW_CENTS, QUEUE, speed_of_sound

D_NOM = NOMINAL["obs_delay"]
STROKE = NOMINAL["stroke"]
_C_NOM = float(speed_of_sound(NOMINAL["temp_c"]))
_L_NOM = NOMINAL["tube_len"] + NOMINAL["end_corr"]


def nominal_cents(x):
    """Pitch the nominal tube gives at plunger position x."""
    return hz_to_cents(_C_NOM / (4.0 * np.maximum(_L_NOM - np.asarray(x, dtype=float), 0.02)))


def nominal_x(cents):
    """Plunger position that the nominal tube needs for `cents` (NaN stays NaN)."""
    f = 440.0 * 2.0 ** (np.asarray(cents, dtype=float) / 1200.0)
    return _L_NOM - _C_NOM / (4.0 * f)


def pwm_for(x_aim, x_belief, s_in=1.0, s_out=1.0, track_time: float = 0.04) -> np.ndarray:
    """PWM that the nominal actuator model (speeds scaled by s_in / s_out) says moves the plunger to x_aim."""
    v_in = NOMINAL["v_in"] * s_in
    v_out = NOMINAL["v_out"] * s_out
    v_des = np.clip((x_aim - x_belief) / track_time, -v_out, v_in)
    drive = np.where(v_des > 0, v_des / v_in, v_des / v_out)
    db = NOMINAL["deadband"]
    pwm = np.sign(drive) * np.minimum(db + (1.0 - db) * np.abs(drive), 1.0)
    return np.where(np.abs(drive) < 0.02, 0.0, pwm)


class DeadReckoner:
    """Integrates the PWM that was sent through the nominal actuator model.

    Travel in each direction is kept apart (t_in, t_out), so a model can scale
    the two speeds separately: position = s_in * t_in - s_out * t_out."""

    def __init__(self, n: int):
        self.n = n
        self.q = np.zeros((n, QUEUE))
        self.reset()

    def reset(self) -> None:
        self.v = np.zeros(self.n)
        self.t_in = np.zeros(self.n)
        self.t_out = np.zeros(self.n)
        self.q[:] = 0.0

    def advance(self, pwm: np.ndarray) -> None:
        self.q[:, 1:] = self.q[:, :-1].copy()
        self.q[:, 0] = pwm
        u = self.q[:, NOMINAL["cmd_delay"]]
        db = NOMINAL["deadband"]
        mag = np.abs(u)
        drive = np.where(mag < db, 0.0, np.sign(u) * (mag - db) / (1.0 - db))
        v_max = np.where(drive > 0, NOMINAL["v_in"], NOMINAL["v_out"])
        self.v = self.v + (drive * v_max - self.v) * min(1.0, DT / NOMINAL["tau_v"])
        d = self.v * DT
        self.t_in = self.t_in + np.maximum(d, 0.0)
        self.t_out = self.t_out + np.maximum(-d, 0.0)


class OpenLoop:
    name = "open_loop"

    def begin(self, sched, rig) -> None:
        self.sched, self.rig, self.n = sched, rig, sched.n
        self.dr = DeadReckoner(self.n)
        self.true_x = np.zeros(self.n)
        self.init()

    def init(self) -> None:
        pass

    def song_start(self, k: int) -> None:
        pass

    def homed(self) -> None:
        self.dr.reset()

    def finish(self) -> None:
        pass

    def belief(self) -> np.ndarray:
        return np.clip(self.dr.t_in - self.dr.t_out, 0.0, STROKE)

    def scales(self):
        return 1.0, 1.0

    def aim_x(self, t: int, xb: np.ndarray) -> np.ndarray:
        a = self.sched.aim[:, t]
        return np.where(np.isfinite(a), np.clip(np.nan_to_num(nominal_x(a)), 0.0, STROKE), xb)

    def act(self, t: int) -> np.ndarray:
        xb = self.belief()
        s_in, s_out = self.scales()
        return pwm_for(self.aim_x(t, xb), xb, s_in, s_out)

    def update(self, t: int, pwm: np.ndarray, out: dict) -> None:
        self.dr.advance(pwm)
        self.true_x = out["x"]


class Encoder(OpenLoop):
    """A position sensor on the plunger, but the tube model is still the nominal one."""
    name = "encoder"

    def belief(self) -> np.ndarray:
        return self.true_x


class Oracle(Encoder):
    """Position sensor and this rig's true tube model: the upper bound."""
    name = "oracle"

    def aim_x(self, t: int, xb: np.ndarray) -> np.ndarray:
        a = self.sched.aim[:, t]
        x = self.rig.p.x_for_cents(np.nan_to_num(a))
        return np.where(np.isfinite(a), np.clip(x, 0.0, STROKE), xb)


class Observer(OpenLoop):
    """Delayed observer: the pitch heard now was played D_NOM steps ago, so it is compared with the belief
    of that time and the whole belief history is shifted by (gain x) the difference. Forgets at each song.

    Frames are skipped during the onset transient and when the pitch jumps from the previous frame
    (a one-frame octave mistake of the estimator). An overblown tone (a twelfth up) is still a valid
    position reading once the twelfth is taken off."""
    name = "observer"

    def __init__(self, gain: float = 0.3, gate_steps: int = 3, max_jump: float = 150.0, max_err: float = 1000.0):
        self.gain, self.gate_steps, self.max_jump, self.max_err = gain, gate_steps, max_jump, max_err

    def init(self) -> None:
        self.dx = np.zeros(self.n)
        self.hist = np.zeros((self.n, QUEUE))
        self.run = np.zeros(self.n, int)
        self.prev_heard = np.full(self.n, np.nan)

    def homed(self) -> None:
        super().homed()
        self.dx[:] = 0.0
        self.hist[:] = 0.0
        self.run[:] = 0
        self.prev_heard[:] = np.nan

    def model(self) -> np.ndarray:
        return self.dr.t_in - self.dr.t_out

    def belief(self) -> np.ndarray:
        return np.clip(self.model() + self.dx, 0.0, STROKE)

    def update(self, t: int, pwm: np.ndarray, out: dict) -> None:
        super().update(t, pwm, out)
        self.hist[:, 1:] = self.hist[:, :-1].copy()
        self.hist[:, 0] = self.belief()
        self.listen(out["heard"])

    def listen(self, heard: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply the correction. Returns (which rigs used this frame, the fundamental pitch heard)."""
        valid = np.isfinite(heard)
        self.run = np.where(valid, self.run + 1, 0)
        steady = np.abs(np.nan_to_num(heard - self.prev_heard, nan=np.inf)) < self.max_jump
        self.prev_heard = heard
        pred = self.hist[:, D_NOM]
        h = np.nan_to_num(heard, nan=0.0)
        h = np.where(h - nominal_cents(pred) > 0.5 * (OVERBLOW_CENTS + 1200.0), h - OVERBLOW_CENTS, h)
        use = (valid & steady & (self.run > self.gate_steps)
               & (np.abs(h - nominal_cents(pred)) < self.max_err))
        corr = np.where(use, self.gain * (nominal_x(h) - pred), 0.0)
        self.dx += corr
        self.hist += corr[:, None]
        return use, h


class ExternalFit(Observer):
    """Observer + rig identification kept outside any network.

    Every frame the observer uses is also a sample of
        x_heard(t) = s_in * t_in(t - D) - s_out * t_out(t - D) + c
    (x_heard: the position the nominal tube gives for the heard pitch). After
    each song, (s_in, s_out, c) are refitted by regularised least squares over
    all songs so far (older songs weighted down by `forget`) and used as the
    motion model of the next song."""
    name = "external_fit"
    PRIOR = np.array([1.0, 1.0, 0.0])
    PRIOR_INFO = np.diag([1.0 / 0.2**2, 1.0 / 0.2**2, 1.0 / 0.01**2])
    NOISE_M = 0.0005  # position noise of one heard frame, model mismatch included [m]

    def __init__(self, forget: float = 0.7, **kw):
        super().__init__(**kw)
        self.forget = forget

    def init(self) -> None:
        super().init()
        n = self.n
        self.theta = np.tile(self.PRIOR, (n, 1))
        self.S = np.zeros((n, 3, 3))
        self.s = np.zeros((n, 3))
        self.tin_hist = np.zeros((n, QUEUE))
        self.tout_hist = np.zeros((n, QUEUE))

    def model(self) -> np.ndarray:
        return self.theta[:, 0] * self.dr.t_in - self.theta[:, 1] * self.dr.t_out + self.theta[:, 2]

    def scales(self):
        return self.theta[:, 0], self.theta[:, 1]

    def update(self, t: int, pwm: np.ndarray, out: dict) -> None:
        OpenLoop.update(self, t, pwm, out)
        for hist, val in ((self.tin_hist, self.dr.t_in), (self.tout_hist, self.dr.t_out),
                          (self.hist, self.belief())):
            hist[:, 1:] = hist[:, :-1].copy()
            hist[:, 0] = val
        use, fundamental = self.listen(out["heard"])
        if use.any():
            X = np.stack([self.tin_hist[:, D_NOM], -self.tout_hist[:, D_NOM], np.ones(self.n)], axis=1)
            y = nominal_x(fundamental)
            w = use / self.NOISE_M**2
            self.S += w[:, None, None] * X[:, :, None] * X[:, None, :]
            self.s += (w * y)[:, None] * X

    def homed(self) -> None:
        if self.S.any():  # a song has been played: refit
            A = self.S + self.PRIOR_INFO
            b = self.s + self.PRIOR_INFO @ self.PRIOR
            th = np.linalg.solve(A, b[:, :, None])[:, :, 0]
            self.theta = np.column_stack([np.clip(th[:, 0], 0.5, 1.6), np.clip(th[:, 1], 0.5, 1.6),
                                          np.clip(th[:, 2], -0.02, 0.02)])
            self.S *= self.forget
            self.s *= self.forget
        super().homed()
        self.tin_hist[:] = 0.0
        self.tout_hist[:] = 0.0


class GRUPolicy(OpenLoop):
    """Recurrent network. Its outputs (a, b, c) define the belief
        position = (1 + a) * t_in - (1 + b) * t_out + c
    and scale the actuator speeds used for steering. Nothing about the rig is
    stored outside the hidden state: the dead reckoning is the fixed nominal
    model and is cleared at every homing.

    `thetas` is (P, n_params): P population members, each playing the same
    n / P rigs (for evolution strategies). `carry=False` clears the hidden state
    at every song (the ablation)."""
    name = "gru"
    FEATURES = 15
    OUT = 3
    OUT_SCALE = np.array([0.4, 0.4, 0.015])

    def __init__(self, thetas: np.ndarray, hidden: int = 24, carry: bool = True, record: bool = False):
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        self.P, self.H, self.carry, self.record = thetas.shape[0], hidden, carry, record
        F, H, O = self.FEATURES, hidden, self.OUT
        parts, i = [], 0
        for shape in self.shapes(hidden):
            size = int(np.prod(shape))
            parts.append(thetas[:, i:i + size].reshape((self.P,) + shape))
            i += size
        if i != thetas.shape[1]:
            raise ValueError(f"expected {i} parameters, got {thetas.shape[1]}")
        self.W, self.U, self.b, self.Wo, self.bo = parts
        if not carry:
            self.name = "gru_reset"

    @staticmethod
    def shapes(hidden: int) -> list[tuple]:
        F, H, O = GRUPolicy.FEATURES, hidden, GRUPolicy.OUT
        return [(F, 3 * H), (H, 3 * H), (3 * H,), (H + F, O), (O,)]

    @classmethod
    def n_params(cls, hidden: int) -> int:
        return int(sum(np.prod(s) for s in cls.shapes(hidden)))

    @classmethod
    def init_params(cls, rng: np.random.Generator, hidden: int, memory_steps: float = 0.0) -> np.ndarray:
        """Random recurrent weights, zero read-out: a fresh network plays exactly like OpenLoop.

        memory_steps > 1: "chrono" initialisation of the update gate, bias = log(T - 1) with T drawn
        log-uniformly from 2 .. memory_steps, so the units keep their state for a few steps up to a
        whole session. With the plain bias of 1 every unit forgets within a few steps, and evolution
        strategies (small parameter noise) do not find the long time constants a rig memory needs."""
        F, H, O = cls.FEATURES, hidden, cls.OUT
        b = np.zeros(3 * H)
        if memory_steps > 1.0:
            b[:H] = np.log(np.exp(rng.uniform(np.log(2.0), np.log(memory_steps), H)) - 1.0)
        else:
            b[:H] = 1.0  # update gate biased toward keeping the memory
        return np.concatenate([(rng.standard_normal((F, 3 * H)) / np.sqrt(F)).ravel(),
                               (rng.standard_normal((H, 3 * H)) / np.sqrt(H)).ravel(),
                               b, np.zeros((H + F) * O), np.zeros(O)])

    def init(self) -> None:
        if self.n % self.P:
            raise ValueError("number of rigs must be a multiple of the population size")
        n, self.R = self.n, self.n // self.P
        self.h = np.zeros((self.P, self.R, self.H))
        self.out = np.zeros((n, self.OUT))
        self.hist = np.zeros((n, QUEUE))
        self.run = np.zeros(n, int)
        self.heard_feat = np.zeros((n, 3))
        self.last_pwm = np.zeros(n)
        self.pulse = np.zeros(n)
        self.probe = []

    def _snapshot(self, k: int) -> None:
        p = self.rig.p
        self.probe.append({
            "song": k, "h": self.h.reshape(self.n, self.H).copy(),
            "s_in": p.v_in / NOMINAL["v_in"], "s_out": p.v_out / NOMINAL["v_out"],
            "dL_mm": 1000.0 * (p.tube_len + p.end_corr - _L_NOM), "temp_c": p.temp_c.copy(),
            "press_cents": p.press_cents.copy(), "obs_delay": p.obs_delay.astype(float),
            "cmd_delay": p.cmd_delay.astype(float), "deadband": p.deadband.copy(),
        })

    def song_start(self, k: int) -> None:
        if self.record:
            self._snapshot(k)  # what the network holds after songs 0..k-1
        if not self.carry:
            self.h[:] = 0.0

    def finish(self) -> None:
        if self.record:
            self._snapshot(self.sched.k)

    def homed(self) -> None:
        super().homed()
        self.hist[:] = 0.0
        self.run[:] = 0
        self.heard_feat[:] = 0.0
        self.pulse[:] = 1.0

    def _position(self, out: np.ndarray) -> np.ndarray:
        a, b, c = (out * self.OUT_SCALE).T
        return np.clip((1.0 + a) * self.dr.t_in - (1.0 + b) * self.dr.t_out + c, 0.0, STROKE)

    def act(self, t: int) -> np.ndarray:
        xb_prev = self._position(self.out)
        a_raw = self.sched.aim[:, t]
        aim = np.where(np.isfinite(a_raw), np.clip(np.nan_to_num(nominal_x(a_raw)), 0.0, STROKE), xb_prev)
        feats = np.column_stack([
            self.heard_feat,
            self.dr.t_in / 0.1, self.dr.t_out / 0.1, self.dr.v / 0.15, self.last_pwm,
            (aim - 0.05) / 0.05, np.clip((aim - xb_prev) / 0.02, -3.0, 3.0),
            self.sched.valve[:, t].astype(float), np.minimum(self.run, 10) / 10.0,
            self.out, self.pulse,
        ]).reshape(self.P, self.R, self.FEATURES)
        H = self.H
        gx = feats @ self.W + self.b[:, None, :]
        gh = self.h @ self.U
        z = 1.0 / (1.0 + np.exp(-(gx[..., :H] + gh[..., :H])))
        r = 1.0 / (1.0 + np.exp(-(gx[..., H:2 * H] + gh[..., H:2 * H])))
        cand = np.tanh(gx[..., 2 * H:] + r * gh[..., 2 * H:])
        self.h = (1.0 - z) * cand + z * self.h
        o = np.concatenate([self.h, feats], axis=2) @ self.Wo + self.bo[:, None, :]
        self.out = np.tanh(o).reshape(self.n, self.OUT)
        self.pulse[:] = 0.0
        xb = self._position(self.out)
        a, b, _ = (self.out * self.OUT_SCALE).T
        aim = np.where(np.isfinite(a_raw), aim, xb)
        return pwm_for(aim, xb, 1.0 + a, 1.0 + b)

    def update(self, t: int, pwm: np.ndarray, out: dict) -> None:
        super().update(t, pwm, out)
        self.last_pwm = np.asarray(pwm, dtype=float)
        self.hist[:, 1:] = self.hist[:, :-1].copy()
        self.hist[:, 0] = self._position(self.out)
        heard = out["heard"]
        valid = np.isfinite(heard)
        self.run = np.where(valid, self.run + 1, 0)
        h = np.nan_to_num(heard, nan=0.0)
        # an overblown tone is a twelfth up: fold it back (the same stateless rule as the hand observer)
        h = np.where(h - nominal_cents(self.hist[:, D_NOM]) > 0.5 * (OVERBLOW_CENTS + 1200.0), h - OVERBLOW_CENTS, h)
        xh = nominal_x(h)
        innov =np.clip((xh - self.hist[:, D_NOM]) / 0.005, -3.0, 3.0)
        self.heard_feat = np.column_stack([valid.astype(float), np.where(valid, np.clip((xh - 0.05) / 0.05, -3.0, 3.0), 0.0),
                                           np.where(valid, innov, 0.0)])
