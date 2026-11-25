"""The page the touch panel (M5Stack Tab5, tab5/yamabiko_panel/) shows, as bytes, and what it asks for.

The web page's process (scripts/yamabiko_web.py) makes the page from its status every time it changes and
sends it through the command server (cmd ``panel``) and the MCU (``U`` / ``V`` commands) to the panel, which
reads it over I2C from the MCU (mcu/yamabiko_fw/yamabiko_fw.ino, "Panel").  The panel's requests come back
the same way: presses of the EXEC button go straight into the MCU's button count (so they work while the
server plays), the rest are events the web process acts on.

Page, little endian (the panel's reader is tab5/yamabiko_panel/page.h; keep the two the same):
  u8  format (FORMAT)
  u8  phase (PHASES)                 u8  flags (FLAG_*)
  u8  step index                     u8  number of steps
  i8  archive clip playing (-1 none) u8  number of archive clips
  u16 recording limit [0.1 s]        u16 length of the score [0.1 s] (the play's length)
  i16 calibration: home pitch [cent] (NO_VALUE: not calibrated)
  i16 calibration: today's shift [cent]
  u8  number of practice plays, then u16 error of each [0.1 cent] (0xFFFF: none)
  str message, str per step (name), str per archive clip (title)     str = u8 length + UTF-8
  u16 number of points, then i16 score [cent] per point, then i16 heard (the last play, in the score's key)
      per point; NO_VALUE where the score rests or nothing was heard.  Points are 10 ms apart x ``stride``.
  u8  stride (steps of 10 ms per point)

Events (the bytes the panel writes after register 0x02):
  EV_STEP index        choose a step              EV_CALIBRATE        calibrate
  EV_ARCHIVE index     play an archive clip
  EV_RECORD            record (a new song)    EV_FINISH           end the recording and play    EV_AGAIN   again
The panel's "stop" is not an event: it is a press of the EXEC button in the MCU, which reaches the server even
while it plays (the board's button stops everything and resets, 2026-11-25).
"""
from __future__ import annotations

import math
import struct

FORMAT = 1
NO_VALUE = -32768
PHASES = ["idle", "prepare", "tune", "record", "analyse", "play", "done", "saved", "error"]
FLAG_BUSY, FLAG_HAS_SONG, FLAG_COLLECT, FLAG_SERVER_BUSY, FLAG_ARCHIVE = 1, 2, 4, 8, 16
EV_STEP, EV_CALIBRATE, EV_ARCHIVE = 0x10, 0x11, 0x12
EV_RECORD, EV_FINISH, EV_AGAIN = 0x13, 0x14, 0x15     # the panel's record / end and play / again
EV_RESET = 0x16                                         # stop everything and reset (as the board's button)
MAX_POINTS = 240
MAX_BYTES = 3072                       # PANEL_MAX in the firmware
LIMITS = {"message": 150, "step": 24, "archive": 72}


def _str(text, limit):
    """u8 length + UTF-8, cut on a character boundary."""
    b = (text or "").encode("utf-8")[:limit]
    while b:
        try:
            b.decode("utf-8")
            break
        except UnicodeDecodeError:
            b = b[:-1]
    return bytes([len(b)]) + b


def _i16(v):
    if v is None or not math.isfinite(v):
        return NO_VALUE
    return int(max(-32767, min(32767, round(v))))


def thin(values, voice, max_points=MAX_POINTS):
    """(points, stride): every ``stride``-th value (NO_VALUE where ``voice`` is off), at most max_points."""
    n = len(values)
    stride = max(1, math.ceil(n / max_points))
    out = []
    for i in range(0, n, stride):
        v = values[i]
        ok = (voice is None or (i < len(voice) and voice[i])) and v is not None
        out.append(_i16(v) if ok else NO_VALUE)
    return out, stride


def encode(status, archive=(), heard=None) -> bytes:
    """The page from the web page's status (WhistleJob.status()), its archive list and the last play's pitch
    (in the score's key; None: the status' own ``heard``)."""
    phase = status.get("phase", "idle")
    steps = status.get("steps") or []
    names = [s["name"] for s in steps]
    flags = ((FLAG_BUSY if status.get("busy") else 0) | (FLAG_HAS_SONG if status.get("has_song") else 0)
             | (FLAG_COLLECT if status.get("collect") else 0)
             | (FLAG_SERVER_BUSY if status.get("busy") and phase in ("tune", "play") else 0)
             | (FLAG_ARCHIVE if status.get("archive") else 0))
    ids = [a["id"] for a in archive]
    arc = ids.index(status["archive"]) if status.get("archive") in ids else -1
    target, voice = status.get("target") or [], status.get("voice") or []
    tuning = status.get("tuning") or {}
    practice = status.get("practice") or []
    out = bytearray(struct.pack("<BBBBBbBHHhhB", FORMAT, PHASES.index(phase) if phase in PHASES else 0, flags,
                                names.index(status["step"]) if status.get("step") in names else 0, len(names),
                                arc, len(ids),
                                int(round(10 * float(status.get("max_seconds") or status.get("collect_seconds") or 30))),
                                int(round(len(target))) // 10 if target else 0,
                                _i16(tuning.get("home_cents")), _i16(tuning.get("shift_cents")),
                                min(len(practice), 32)))
    for e in practice[-32:]:
        out += struct.pack("<H", 0xFFFF if e is None else int(round(min(6553.4, e) * 10)))
    out += _str(status.get("message"), LIMITS["message"])
    for name in names:
        out += _str(name, LIMITS["step"])
    for a in archive:
        out += _str(a.get("title", a["id"]), LIMITS["archive"])
    points, stride = thin(target, voice)
    played = heard if heard is not None else (status.get("heard") or [])
    played_points, _ = thin(list(played) + [None] * max(0, len(target) - len(played)), voice)
    room = (MAX_BYTES - len(out) - 3) // 4
    points, played_points = points[:room], played_points[:room]
    out += struct.pack("<H", len(points)) + struct.pack(f"<{len(points)}h", *points)
    out += struct.pack(f"<{len(played_points)}h", *played_points) + bytes([stride])
    return bytes(out)


def decode(page: bytes) -> dict:
    """The page back as a dict (the tests, and a model of the panel's reader)."""
    head = struct.Struct("<BBBBBbBHHhhB")
    (fmt, phase, flags, step, n_steps, arc, n_arc, rec_ds, play_ds, home, shift, n_pr) = head.unpack_from(page)
    k = head.size
    practice = []
    for _ in range(n_pr):
        v = struct.unpack_from("<H", page, k)[0]
        practice.append(None if v == 0xFFFF else v / 10)
        k += 2

    def text():
        nonlocal k
        n = page[k]
        s = page[k + 1:k + 1 + n].decode("utf-8")
        k += 1 + n
        return s
    message = text()
    steps = [text() for _ in range(n_steps)]
    archive = [text() for _ in range(n_arc)]
    n = struct.unpack_from("<H", page, k)[0]
    k += 2
    target = list(struct.unpack_from(f"<{n}h", page, k))
    k += 2 * n
    heard = list(struct.unpack_from(f"<{n}h", page, k))
    k += 2 * n
    stride = page[k]
    none = lambda a: [None if v == NO_VALUE else v for v in a]  # noqa: E731
    return {"format": fmt, "phase": PHASES[phase], "flags": flags, "step": step, "steps": steps,
            "archive_playing": arc, "archive": archive, "record_s": rec_ds / 10, "play_s": play_ds / 10,
            "home_cents": None if home == NO_VALUE else home, "shift_cents": None if shift == NO_VALUE else shift,
            "practice": practice, "message": message, "target": none(target), "heard": none(heard),
            "stride": stride}
