"""The real Yamabiko No.1 rig, driven through the MCU firmware (mcu/yamabiko_fw) from the UNO Q's Linux side.

`RealRig.step(pwm, valve)` has the same meaning as `Rig.step` of the simulator for one rig: it sends the
command, waits for the end of the 10 ms control step and returns what was heard. So the controllers of
control.py run on the real rig unchanged, and the logs of the real rig have the same shape as the
simulator's (for scripts/yamabiko_fit.py).

What the simulator knows and the real rig does not: the plunger position and the true pitch. `out["x"]` is
NaN, `out["cents"]` / `out["sounding"]` are what was heard. The delay from the sound to `heard` is whatever
the real chain adds (MCU, link, the pitch window); nothing is added here.

Needs numpy (Debian: `sudo apt install python3-numpy`). The firmware owns Serial1 (/dev/ttyHS1):
arduino-router must be stopped.
"""
from __future__ import annotations

import json
import os
import select
import struct
import time
from dataclasses import dataclass

import numpy as np

from ..pitch import yin_frame
from ..sim import DT, hz_to_cents

SR = 16000             # stream rate of the firmware [Hz]
MAGIC = b"\xa5\x5a"
PITCH_FRAME = 320      # 20 ms window for the pitch heard at each step
PITCH_FMIN, PITCH_FMAX = 400.0, 4000.0
RMS_GATE_DB = -60.0    # quieter windows count as silence (the room is about -64 dBFS)
STATUS = struct.Struct("<IIHHHBBhhH")
FLAG_BUTTON, FLAG_VALVE, FLAG_FAN, FLAG_TIMEOUT, FLAG_SERVO_SILENT = 1, 2, 4, 8, 16


@dataclass
class Status:
    ms: int
    frame48: int
    isense_mv: int
    isense_peak_mv: int
    fan_rpm: int
    buttons: int
    flags: int
    pwm: int
    servo_pos: int
    cmd_age_ms: int
    t_rx: float


class Audio:
    """The 16 kHz stream, kept whole (gaps filled with zeros), indexed by the firmware's sample index."""

    def __init__(self, seconds: float = 900.0):
        self.buf = np.zeros(int(seconds * SR), np.int16)
        self.base = None      # sample index of buf[0]
        self.end = 0          # one past the newest sample (sample index)
        self.gaps = 0

    def add(self, first: int, samples: np.ndarray) -> None:
        if self.base is None:
            self.base = first
            self.end = first
        if first > self.end:
            self.gaps += 1
        i0, i1 = first - self.base, first - self.base + len(samples)
        if i1 > len(self.buf):  # full: drop the oldest half
            keep = len(self.buf) // 2
            shift = (self.end - self.base) - keep
            self.buf[:keep] = self.buf[shift:shift + keep]
            self.base += shift
            i0, i1 = first - self.base, first - self.base + len(samples)
        if i0 >= 0:
            self.buf[i0:i1] = samples
        self.end = max(self.end, first + len(samples))

    def latest(self, n: int) -> np.ndarray:
        """The newest n samples as floats in [-1, 1)."""
        if self.base is None or self.end - self.base < n:
            return np.zeros(n)
        i1 = self.end - self.base
        return self.buf[i1 - n:i1].astype(float) / 32768.0

    def span(self, start: int, stop: int) -> np.ndarray:
        """Samples [start, stop) by sample index, as int16 (zeros where not kept)."""
        out = np.zeros(max(0, stop - start), np.int16)
        if self.base is None:
            return out
        a, b = max(start, self.base), min(stop, self.end)
        if b > a:
            out[a - start:b - start] = self.buf[a - self.base:b - self.base]
        return out


class Link:
    """The firmware's packet link (see the protocol at the top of mcu/yamabiko_fw/yamabiko_fw.ino)."""

    def __init__(self, dev: str = "/dev/ttyHS1", baud: int = 921600):
        import termios  # Linux only; the rest of this module also loads on the PC
        self.fd = os.open(dev, os.O_RDWR | os.O_NOCTTY)
        attr = termios.tcgetattr(self.fd)
        attr[0] = attr[1] = attr[3] = 0
        attr[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attr[4] = attr[5] = getattr(termios, f"B{baud}")
        attr[6][termios.VMIN] = 1
        attr[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attr)
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        self.buf = bytearray()
        self.bad = 0
        self.audio = Audio()
        self.status: Status | None = None
        self.texts: list[str] = []

    def close(self) -> None:
        os.close(self.fd)

    # ---- commands
    def send(self, data: bytes) -> None:
        os.write(self.fd, data)

    def start(self) -> None:
        self.send(b"S")

    def stop(self) -> None:
        self.send(b"X")

    def command(self, pwm: float, valve: bool, brake: bool = False) -> None:
        p = int(round(max(-1.0, min(1.0, float(pwm))) * 1000.0))
        self.send(b"C" + struct.pack("<hB", p, (1 if valve else 0) | (2 if brake else 0)))

    def fan(self, on: bool, duty: float) -> None:
        self.send(b"F" + struct.pack("<BH", 1 if on else 0, int(round(max(0.0, min(1.0, duty)) * 1000))))

    def valve_positions(self, open_pos: int, closed_pos: int, move_ms: int) -> None:
        self.send(b"K" + struct.pack("<HHH", open_pos, closed_pos, move_ms))

    def limit(self, max_pwm: float) -> None:
        self.send(b"L" + struct.pack("<H", int(round(max(0.0, min(1.0, max_pwm)) * 1000))))

    # ---- receiving
    def _packet(self):
        b = self.buf
        while True:
            i = b.find(MAGIC)
            if i < 0:
                self.bad += max(0, len(b) - 1)
                del b[:max(0, len(b) - 1)]
                return None
            if i:
                self.bad += i
                del b[:i]
            if len(b) < 5:
                return None
            n = b[3] | (b[4] << 8)
            if n > 4096:
                self.bad += 1
                del b[:1]
                continue
            if len(b) < 6 + n:
                return None
            if (sum(b[2:5 + n]) & 0xFF) != b[5 + n]:
                self.bad += 1
                del b[:1]
                continue
            typ, payload = chr(b[2]), bytes(b[5:5 + n])
            del b[:6 + n]
            return typ, payload

    def poll(self, timeout: float) -> None:
        """Take in everything that arrives within `timeout` seconds (returns early once something came)."""
        if not select.select([self.fd], [], [], max(0.0, timeout))[0]:
            return
        self.buf += os.read(self.fd, 65536)
        t = time.monotonic()
        while (p := self._packet()) is not None:
            typ, payload = p
            if typ == "A":
                _, first = struct.unpack_from("<HI", payload)
                self.audio.add(first, np.frombuffer(payload[6:], np.int16))
            elif typ == "Y" and len(payload) == STATUS.size:
                self.status = Status(*STATUS.unpack(payload), t)
            elif typ == "T":
                self.texts.append(payload.decode(errors="replace"))

    def drain(self, seconds: float) -> None:
        until = time.monotonic() + seconds
        while (left := until - time.monotonic()) > 0:
            self.poll(left)


def heard_pitch(x: np.ndarray, sr: int = SR) -> tuple[float, float, float]:
    """(cents or NaN, YIN confidence, level [dBFS]) of one window."""
    rms = float(np.sqrt(np.mean(x * x)))
    db = 20.0 * np.log10(max(rms, 1e-9))
    if db < RMS_GATE_DB:
        return float("nan"), 0.0, db
    f0, conf = yin_frame(x, sr, PITCH_FMIN, PITCH_FMAX)
    return (float(hz_to_cents(f0)) if np.isfinite(f0) else float("nan")), conf, db


class RealRig:
    """One real rig with the step interface of the simulator's Rig (arrays of shape (1,))."""

    LOG_KEYS = ("t_step", "pwm", "valve", "heard", "conf", "level_db", "isense_mv", "isense_peak_mv",
                "fan_rpm", "servo_pos", "sample", "late_ms")

    def __init__(self, link: Link, flip: bool = False, max_pwm: float = 1.0):
        self.link, self.flip, self.max_pwm = link, flip, max_pwm
        self.t_next = None
        self.late_steps = 0
        self.log = {k: [] for k in self.LOG_KEYS}
        self.t0 = None

    def start(self) -> None:
        self.link.limit(self.max_pwm)
        self.link.start()
        until = time.monotonic() + 2.0
        while self.link.audio.base is None and time.monotonic() < until:
            self.link.poll(0.05)
        if self.link.audio.base is None:
            raise RuntimeError("no audio from the MCU: is mcu/yamabiko_fw flashed and arduino-router stopped?")
        self.link.drain(0.1)                     # the first 20 ms after a restart are unsettled
        self.t_next = time.monotonic() + DT
        self.t0 = time.monotonic()

    def send(self, pwm: float, valve: bool) -> None:
        self.link.command(-pwm if self.flip else pwm, valve)

    def step(self, pwm, valve) -> dict:
        p = float(np.asarray(pwm).reshape(-1)[0])
        v = bool(np.asarray(valve).reshape(-1)[0])
        self.send(p, v)
        while (left := self.t_next - time.monotonic()) > 0:
            self.link.poll(left)
        late = -left
        if late > DT:                            # fell behind by a whole step: start counting afresh
            self.late_steps += 1
            self.t_next = time.monotonic()
        self.t_next += DT
        cents, conf, db = heard_pitch(self.link.audio.latest(PITCH_FRAME))
        st = self.link.status
        row = {"t_step": time.monotonic() - self.t0, "pwm": p, "valve": v, "heard": cents, "conf": conf,
               "level_db": db, "isense_mv": st.isense_mv if st else -1, "isense_peak_mv": st.isense_peak_mv if st else -1,
               "fan_rpm": st.fan_rpm if st else -1, "servo_pos": st.servo_pos if st else -1,
               "sample": self.link.audio.end, "late_ms": 1000.0 * late}
        for k in self.LOG_KEYS:
            self.log[k].append(row[k])
        h = np.array([cents])
        return {"x": np.array([np.nan]), "cents": h, "sounding": np.isfinite(h), "overblown": np.zeros(1, bool),
                "measured": h, "heard": h}

    def resync(self) -> None:
        """Restart the step clock (after waiting outside the control loop)."""
        self.t_next = time.monotonic() + DT

    def safe(self) -> None:
        """Plunger stopped, air shut."""
        try:
            self.link.command(0.0, False)
        except OSError:
            pass


class SessionLog:
    """Everything of one run on the real rig, saved as one .npz that scripts/yamabiko_fit.py reads.

    Per step: the RealRig log plus the schedule (target, aim, homing, song) and the controller's name.
    The audio of the whole run is kept too, so the pitch can be estimated again offline."""

    def __init__(self, meta: dict):
        self.meta = dict(meta)
        self.rows: dict[str, list] = {"target": [], "aim": [], "homing": [], "song": [], "sched_valve": []}
        self.whistles: list[np.ndarray] = []

    def add_step(self, target: float, aim: float, homing: bool, song: int, sched_valve: bool) -> None:
        for k, v in (("target", target), ("aim", aim), ("homing", homing), ("song", song), ("sched_valve", sched_valve)):
            self.rows[k].append(v)

    def save(self, path, rig: RealRig) -> None:
        arrays = {k: np.asarray(v) for k, v in rig.log.items()}
        arrays.update({k: np.asarray(v) for k, v in self.rows.items()})
        s = arrays["sample"]
        if len(s):
            start = int(s[0]) - PITCH_FRAME
            arrays["audio"] = rig.link.audio.span(start, int(s[-1]))
            arrays["audio_start"] = np.array(start)
        for i, w in enumerate(self.whistles):
            arrays[f"whistle_{i}"] = w
        meta = {**self.meta, "late_steps": rig.late_steps, "audio_gaps": rig.link.audio.gaps, "link_bad": rig.link.bad,
                "sr": SR, "dt": DT, "pitch_frame": PITCH_FRAME}
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def hold(rig: RealRig, log: "SessionLog", seconds: float, valve: bool = False, song: int = -1) -> None:
    """Plunger stopped (pwm 0) for a while, still stepping and logging (fan noise, rests between songs)."""
    for _ in range(int(round(seconds / DT))):
        rig.step(0.0, valve)
        log.add_step(float("nan"), float("nan"), False, song, valve)


def play_song(rig: RealRig, ctrl, sched, k: int, log: SessionLog, first: bool, explore=None) -> None:
    """One song on the real rig, stepped like session.run_session: homing (plunger out, valve shut), then
    the controller. `first`: the controller is begun here (power on); later songs keep its state.
    `explore(pwm) -> pwm` perturbs the controller's command (data collection); the controller is told
    what was actually sent."""
    if first:
        ctrl.begin(sched, rig)
    else:
        ctrl.sched = sched
    ctrl.song_start(k)
    for t in range(sched.T):
        home = bool(sched.homing[t])
        if home:
            pwm, valve = np.full(1, -1.0), np.zeros(1, bool)
        else:
            pwm = np.asarray(ctrl.act(t), dtype=float).reshape(1)
            if explore is not None:
                pwm = np.clip(explore(pwm), -1.0, 1.0)
            valve = sched.valve[:, t]
        out = rig.step(pwm, valve)
        ctrl.update(t, pwm, out)
        if home and (t + 1 == sched.T or not sched.homing[t + 1]):
            ctrl.homed()
        log.add_step(float(sched.target[0, t]), float(sched.aim[0, t]), home, k, bool(sched.valve[0, t]))
