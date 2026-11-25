"""PC: drive the real rig through the UNO Q's command server (flute_rl/yamabiko/server.py) over USB.

    from flute_rl.yamabiko.remote import Remote
    with Remote() as rig:                 # runs `adb forward tcp:8765 tcp:8765` first
        print(rig.status())
        rig.fan(True, .5); rig.actuator(.3); time.sleep(1); rig.stop()
        pcm, sr = rig.record(2.0)          # int16 array of the microphone

A background thread sends a heartbeat, so the outputs stay as set while this object lives; they go back to
safe when it is closed (or the program dies), as the server takes them back after its lease.
"""
from __future__ import annotations

import base64
import itertools
import json
import os
import shutil
import socket
import subprocess
import threading

import numpy as np

REMOTE_DIR = "/home/arduino/yamabiko"


def adb_path():
    return os.environ.get("ADB") or shutil.which("adb") or "adb"


class RemoteError(RuntimeError):
    pass


class Remote:
    def __init__(self, host="127.0.0.1", port=8765, forward=True, heartbeat_s=.5, timeout=10.0):
        if forward:
            subprocess.run([adb_path(), "forward", f"tcp:{port}", f"tcp:{port}"], check=True, capture_output=True)
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.file = self.sock.makefile("rb")
        self.ids = itertools.count(1)
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.beat = threading.Thread(target=self._heartbeat, args=(heartbeat_s,), daemon=True)
        self.beat.start()

    # ----- plumbing -------------------------------------------------------------------------------
    def call(self, cmd, timeout=None, **args):
        """One request, one reply (dict).  Raises RemoteError when the server says no."""
        with self.lock:
            rid = next(self.ids)
            self.sock.sendall((json.dumps({"id": rid, "cmd": cmd, **args}) + "\n").encode())
            old = self.sock.gettimeout()
            if timeout is not None:
                self.sock.settimeout(timeout)
            try:
                while True:
                    line = self.file.readline()
                    if not line:
                        raise RemoteError("the server closed the connection")
                    reply = json.loads(line)
                    if reply.get("id") == rid:
                        break
            finally:
                self.sock.settimeout(old)
        if not reply.pop("ok"):
            raise RemoteError(reply.get("error"))
        reply.pop("id", None)
        return reply

    def _heartbeat(self, period):
        while not self.closed.wait(period):
            try:
                self.call("ping")
            except (OSError, RemoteError, ValueError):
                return

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.call("stop")
        except (OSError, RemoteError, ValueError):
            pass
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----- commands -------------------------------------------------------------------------------
    def info(self):
        return self.call("info")["lines"]

    def status(self):
        return self.call("status")

    def actuator(self, pwm, brake=False):
        return self.call("actuator", pwm=float(pwm), brake=bool(brake))

    def limit(self, max_pwm):
        return self.call("limit", max_pwm=float(max_pwm))

    def valve(self, open_):
        return self.call("valve", open=bool(open_))

    def valve_positions(self, open_, closed, move_ms=300):
        return self.call("valve_positions", open=int(open_), closed=int(closed), move_ms=int(move_ms))

    def fan(self, on, duty=.7):
        return self.call("fan", on=bool(on), duty=float(duty))

    def stop(self, force=False):
        return self.call("stop", force=bool(force))

    def mic(self, window_s=.25):
        return self.call("mic", window_s=float(window_s))

    def record(self, seconds):
        """(int16 samples, sample rate) of the next `seconds` of the microphone."""
        reply = self.call("record", timeout=seconds + 10, seconds=float(seconds))
        return np.frombuffer(base64.b64decode(reply["pcm16_b64"]), "<i2").copy(), int(reply["sr"])

    def audio_mark(self):
        """The microphone's sample index now (server cmd_audio_mark)."""
        return int(self.call("audio_mark")["index"])

    def audio_since(self, start, stop=None):
        """(int16 samples, sample rate, stop index) from ``start`` (an audio_mark) to ``stop`` or now."""
        reply = self.call("audio_span", start=int(start), stop=stop)
        return np.frombuffer(base64.b64decode(reply["pcm16_b64"]), "<i2").copy(), int(reply["sr"]), int(reply["stop"])

    def calibrate(self, out=f"{REMOTE_DIR}/runs/real/calibration.npz", **kw):
        """Calibration run on the UNO Q; the record stays there (adb pull it).  kw: max_pwm, sweep, fan, flip."""
        return self.call("calibrate", timeout=60, out=out, **kw)

    def tune(self, runtime=None, **kw):
        """Today's flute: home note and one sweep (server cmd_tune); its shift_cents goes to play()."""
        return self.call("tune", timeout=60, runtime=runtime, **kw)

    def panel(self, page=None, version=0):
        """Show a page on the touch panel (flute_rl/yamabiko/panel.py) and get what it asked for."""
        import base64
        req = {} if page is None else {"page_b64": base64.b64encode(page).decode(), "version": int(version)}
        return self.call("panel", **req)["events"]

    def play(self, runtime, target, out=None, repeats=1, **kw):
        """Play a target with a NumPy runtime; both paths are on the UNO Q (push them first with push())."""
        return self.call("play", timeout=600, runtime=runtime, target=target, out=out, repeats=int(repeats), **kw)


def push(local, remote):
    subprocess.run([adb_path(), "push", str(local), remote], check=True, capture_output=True)


def pull(remote, local):
    subprocess.run([adb_path(), "pull", remote, str(local)], check=True, capture_output=True)
