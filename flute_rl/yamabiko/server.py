"""UNO Q: one process owns the MCU link and takes commands from the PC (JSON lines over TCP).

    python3 scripts/yamabiko_server.py [--port 8765] [--max-pwm 1.0]

The PC reaches it through USB with ``adb forward tcp:8765 tcp:8765`` (flute_rl/yamabiko/remote.py does that).
Every request is one JSON object on one line, ``{"id": 1, "cmd": "actuator", "pwm": 0.3}``; every reply is
one line, ``{"id": 1, "ok": true, ...}`` or ``{"id": 1, "ok": false, "error": "..."}``.  ``help`` lists the
commands.

Safety: the outputs (actuator, valve, fan) belong to the connection that set them.  They go back to safe
(actuator 0, valve shut, fan off) when that connection closes or sends nothing for ``lease`` seconds
(default 2; remote.py sends a heartbeat).  While the server runs it sends the actuator / valve command every
10 ms, so the firmware's own 100 ms watchdog only fires if the server dies.  ``--max-pwm`` caps the actuator
(``limit`` changes the cap, up to 1.0).

On start the server sets the two GPIO lines arduino-router's unit sets (the MCU does not start its sketch
after a boot without them, and arduino-router is off), checks that mcu/yamabiko_fw answers and flashes
``build_fw`` (compiled on the UNO Q, see docs/yamabiko_real.md) if it does not.
"""
from __future__ import annotations

import base64
import json
import pathlib
import selectors
import socket
import subprocess
import time
import traceback

import numpy as np

from .hw import (FLAG_BUTTON, FLAG_FAN, FLAG_SERVO_SILENT, FLAG_TIMEOUT, FLAG_VALVE, PITCH_FRAME, SR, Link,
                 RealRig, heard_pitch)

STEP_S = 0.01
HOME = pathlib.Path("/home/arduino")
FLASH = [str(HOME / ".arduino15/packages/arduino/tools/remoteocd/0.0.4-rc.4/remoteocd"), "upload", "-f",
         str(HOME / ".arduino15/packages/arduino/hardware/zephyr/0.53.1/variants/arduino_uno_q_stm32u585xx/"
                    "flash_sketch.cfg"),
         str(HOME / "yamabiko/build_fw/yamabiko_fw.ino.elf-zsk.bin")]


# What arduino-router's unit does around its start: line 37 low puts the micro in a ready state, line 70 high
# ends the boot animation.  Without them (arduino-router is off) the sketch in flash does not run after a boot.
READY_LINES = ["gpioset", "-c", "/dev/gpiochip1", "-t0", "37=0", "70=1"]


def mcu_ready_lines(log=print):
    try:
        subprocess.run(READY_LINES, check=True, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"could not set the MCU ready lines: {exc}")


def ask_info(link, timeout=1.0):
    """The firmware's two text lines (INFO ..., IO ...), or [] when it does not answer."""
    link.texts.clear()
    link.send(b"i")
    until = time.monotonic() + timeout
    while time.monotonic() < until and len(link.texts) < 2:
        link.poll(.05)
    return list(link.texts)


def ensure_firmware(link, log=print):
    if any(t.startswith("INFO fw yamabiko_fw") for t in ask_info(link)):
        return "running"
    log("yamabiko_fw does not answer: flashing build_fw")
    subprocess.run(FLASH, check=True, capture_output=True, env={"TMPDIR": "/tmp", "HOME": str(HOME), "PATH": "/usr/bin:/bin"})
    time.sleep(1.0)
    link.drain(.2)
    if any(t.startswith("INFO fw yamabiko_fw") for t in ask_info(link, 2.0)):
        return "flashed"
    raise RuntimeError("yamabiko_fw does not answer even after flashing (is arduino-router holding /dev/ttyHS1?)")


CURRENT_FULL_MV = 20.0   # isense_mv of the plunger running at full PWM (0.1 ohm shunt): the networks' scale
AUDIO_STALL_S = 0.5   # no audio this long while serving: restart the stream (see Server.audio_watch)

class Client:
    def __init__(self, sock, addr):
        self.sock, self.addr = sock, addr
        self.buf = b""
        self.out = bytearray()          # reply bytes the socket has not taken yet (a recording over Wi-Fi)
        self.last_seen = time.monotonic()


class Server:
    """The command loop.  ``link`` is a hw.Link (or anything with its interface, for tests)."""

    def __init__(self, link, host="127.0.0.1", port=8765, max_pwm=1.0, lease=2.0, log=print):
        self.link, self.max_pwm, self.lease, self.log = link, max_pwm, lease, log
        self.sel = selectors.DefaultSelector()
        self.listener = socket.create_server((host, port))
        self.listener.setblocking(False)
        self.port = self.listener.getsockname()[1]
        self.sel.register(self.listener, selectors.EVENT_READ, None)
        self.pwm, self.brake, self.valve = 0.0, False, False
        self.fan_on, self.fan_duty = False, 0.0
        self.owner: Client | None = None
        self.peak_hold, self.peak_seen = 0, None
        self.audio_seen_end, self.audio_seen_t, self.audio_restarts = None, time.monotonic(), 0
        self.jobs = []                     # (client, id, kind, data) finished in the loop
        self.running = True
        self.next_step = time.monotonic()

    # ----- outputs ---------------------------------------------------------------------------------
    def safe(self, why=""):
        if self.pwm or self.valve or self.fan_on or self.brake:
            self.log(f"outputs to safe{': ' + why if why else ''}")
        self.pwm, self.brake, self.valve = 0.0, False, False
        # always: calibrate and play switch the fan on through the link, not through fan_on
        self.link.fan(False, 0.0)
        self.fan_on, self.fan_duty = False, 0.0
        self.link.command(0.0, False)
        self.owner = None

    def take(self, client):
        if self.owner is not None and self.owner is not client:
            raise RuntimeError("another connection is driving the outputs")
        self.owner = client

    # ----- commands -------------------------------------------------------------------------------
    def status(self):
        st = self.link.status
        cents, conf, db = heard_pitch(self.link.audio.latest(PITCH_FRAME))
        out = {"pwm": self.pwm, "brake": self.brake, "valve": self.valve, "fan": self.fan_on, "fan_duty": self.fan_duty,
               "max_pwm": self.max_pwm, "mic_db": round(db, 1),
               "pitch_hz": round(440 * 2 ** (cents / 1200), 1) if np.isfinite(cents) else None,
               "pitch_conf": round(conf, 3), "audio_gaps": self.link.audio.gaps, "link_bad": self.link.bad}
        if st is not None:
            out.update({"mcu_ms": st.ms, "applied_pwm": st.pwm / 1000, "isense_mv": st.isense_mv,
                        "isense_peak_mv": st.isense_peak_mv, "fan_rpm": st.fan_rpm, "button_count": st.buttons,
                        "button_held": bool(st.flags & FLAG_BUTTON), "valve_open": bool(st.flags & FLAG_VALVE),
                        "fan_switch": bool(st.flags & FLAG_FAN), "fw_timeout": bool(st.flags & FLAG_TIMEOUT),
                        "servo_answering": not st.flags & FLAG_SERVO_SILENT, "servo_pos": st.servo_pos,
                        "status_age_s": round(time.monotonic() - st.t_rx, 3),
                        # the highest motor current since the previous status request (any client), so a
                        # 20 ms pulse is not missed between polls
                        "isense_peak_hold_mv": max(self.peak_hold, st.isense_peak_mv),
                        "audio_restarts": self.audio_restarts})
            self.peak_hold = 0
        return out

    def handle(self, client, req):
        cmd = req.get("cmd")
        if cmd == "help":
            return {"commands": sorted(name[4:] for name in dir(self) if name.startswith("cmd_"))}
        fn = getattr(self, f"cmd_{cmd}", None)
        if fn is None:
            raise ValueError(f"unknown cmd {cmd!r} (try help)")
        return fn(client, req)

    def cmd_ping(self, client, req):
        return {"t": time.time()}

    def cmd_info(self, client, req):
        return {"lines": ask_info(self.link)}

    def cmd_status(self, client, req):
        return self.status()

    def cmd_actuator(self, client, req):
        pwm = float(req.get("pwm", 0.0))
        if abs(pwm) > self.max_pwm + 1e-9:
            raise ValueError(f"|pwm| {abs(pwm)} is over the cap {self.max_pwm} (raise it with limit)")
        self.take(client)
        self.pwm, self.brake = pwm, bool(req.get("brake", False))
        return {"pwm": self.pwm, "brake": self.brake}

    def cmd_limit(self, client, req):
        self.max_pwm = min(max(float(req["max_pwm"]), 0.0), 1.0)
        self.link.limit(self.max_pwm)
        if abs(self.pwm) > self.max_pwm:
            self.pwm = float(np.clip(self.pwm, -self.max_pwm, self.max_pwm))
        return {"max_pwm": self.max_pwm}

    def cmd_valve(self, client, req):
        self.take(client)
        self.valve = bool(req.get("open", False))
        return {"valve": self.valve}

    def cmd_valve_positions(self, client, req):
        self.link.valve_positions(int(req["open"]), int(req["closed"]), int(req.get("move_ms", 300)))
        return {}

    def cmd_fan(self, client, req):
        self.take(client)
        self.fan_on, self.fan_duty = bool(req.get("on", False)), float(np.clip(req.get("duty", .7), 0, 1))
        self.link.fan(self.fan_on, self.fan_duty)
        return {"fan": self.fan_on, "duty": self.fan_duty}

    def cmd_stop(self, client, req):
        if self.owner is not None and self.owner is not client and not req.get("force", False):
            raise RuntimeError("another connection is driving the outputs (force: true to stop anyway)")
        self.safe("stop")
        return {}

    def cmd_mic(self, client, req):
        audio = self.link.audio
        have = 0 if audio.base is None else audio.end - audio.base
        n = min(int(round(float(req.get("window_s", PITCH_FRAME / SR)) * SR)), have)
        if n < 64:
            raise RuntimeError("no audio from the MCU yet")
        x = audio.latest(n)
        cents, conf, db = heard_pitch(self.link.audio.latest(PITCH_FRAME))
        spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x))))
        lo = max(1, int(100 * len(x) / SR))
        return {"rms_db": round(20 * np.log10(max(float(np.sqrt(np.mean(x * x))), 1e-9)), 1),
                "peak_db": round(20 * np.log10(max(float(np.abs(x).max()), 1e-9)), 1),
                "fft_peak_hz": round((int(np.argmax(spec[lo:])) + lo) * SR / len(x), 1),
                "pitch_hz": round(440 * 2 ** (cents / 1200), 1) if np.isfinite(cents) else None,
                "pitch_conf": round(conf, 3)}

    def cmd_record(self, client, req):
        seconds = float(req.get("seconds", 1.0))
        if not 0 < seconds <= 30:
            raise ValueError("seconds must be in (0, 30]")
        start = self.link.audio.end
        self.jobs.append((client, req.get("id"), "record", (start, start + int(round(seconds * SR)))))
        return None                                       # answered by the loop when the audio is in

    def cmd_calibrate(self, client, req):
        from .calibration import calibrate_real
        self.take(client)
        rig = self._rig(req)
        self.fan_on, self.fan_duty = True, float(req.get("fan", .7))
        self.link.fan(True, self.fan_duty)
        try:
            for _ in range(int(round(float(req.get("warmup_s", 3.0)) / STEP_S))):   # fan spin-up, valve shut
                rig.step(0.0, False)
            record = calibrate_real(rig, max_pwm=float(req.get("max_pwm", self.max_pwm)),
                                    sweep=float(req.get("sweep", 1.0)), homing_seconds=float(req.get("homing_seconds", 1.3)),
                                    plan=str(req.get("plan", "default")), max_cents=req.get("max_cents"))
        finally:
            self.safe("calibration done")
        out = req.get("out")
        if out:
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **record)
        valid = record["0/valid"][0]
        heard = record["0/heard"][0][valid]
        return {"steps": int(valid.size), "voiced": int(valid.sum()), "late_steps": rig.late_steps, "out": out,
                "pitch_cents": [float(heard.min()), float(heard.max())] if valid.any() else None}

    def cmd_play(self, client, req):
        from .residual_numpy import runtime_for
        self.take(client)
        runtime = runtime_for(dict(np.load(req["runtime"])))
        if req.get("valve_runtime"):                   # a separate valve network beside the actuator's
            from .residual_numpy import WithValve
            runtime = WithValve(runtime, dict(np.load(req["valve_runtime"])))
        if req.get("ear", "yin") != "yin":
            raise ValueError(f"ear {req.get('ear')!r} is not available yet (yin only)")
        target = np.load(req["target"])
        cents, voice = target["cents"].astype(np.float64), target["voice"].astype(bool)
        rig = self._rig(req)
        self.fan_on, self.fan_duty = True, float(req.get("fan", .7))
        self.link.fan(True, self.fan_duty)
        # the fan spins up with the valve shut (no sound) before the first note; the steps keep the
        # firmware's 100 ms command watchdog fed
        for _ in range(int(round(float(req.get("warmup_s", 3.0)) / STEP_S))):
            rig.step(0.0, False)
        # heard above this is the flute overblowing, not a position the twin knows: count it as not heard
        max_cents = float(req.get("max_cents", cents[voice].max() + 250 if voice.any() else np.inf))
        errors, records = [], {}
        try:
            shift = float(req.get("shift_cents", 0.0))   # today's offset, measured by cmd_tune
            cents = cents + shift
            max_cents += shift
            self_home = getattr(runtime, "self_home", False)
            if self_home:
                # every actuator command from the network: it gets a rest before the song (the target held
                # at its first note, voice off) to home the plunger and open the valve itself
                lead = int(round(float(req.get("lead_seconds", 1.6)) / STEP_S))
                cents = np.r_[np.full(lead, cents[0]), cents]
                voice = np.r_[np.zeros(lead, bool), voice]
            for play in range(int(req.get("repeats", 1))):
                rig.resync()
                if not self_home:
                    for _ in range(int(round(float(req.get("homing_seconds", 1.3)) / STEP_S))):
                        rig.step(-1.0, False)
                if play == 0 and req.get("relative", False) and "shift_cents" not in req:
                    # a relative melody copy: sound the home note, and move the whole target by how far the
                    # flute sounds from the twin's home today (the plunger stays pressed home meanwhile)
                    heard_home = []
                    for k in range(int(round(float(req.get("listen_seconds", .8)) / STEP_S))):
                        h = float(rig.step(-.3, True)["heard"][0])
                        if k * STEP_S > .3 and np.isfinite(h) and h <= max_cents:
                            heard_home.append(h)
                    rig.step(0.0, False)
                    if len(heard_home) >= 10:
                        shift = float(np.clip(np.median(heard_home) - runtime.cents_at(0.0), -150, 150))
                        cents = cents + shift
                        max_cents += shift
                runtime.prepare(cents, voice, keep_song_memory=play > 0)
                pwm_log, heard_log, valid_log, valve_log = [], [], [], []
                audio_start = self.link.audio.end
                for t in range(len(cents)):
                    pwm, valve = runtime.command(t)
                    h = float(rig.step(pwm, valve)["heard"][0]); ok = bool(np.isfinite(h)) and h <= max_cents
                    if self_home:
                        runtime.observe(h if ok else 0.0, ok, max(rig.log["isense_mv"][-1], 0) / CURRENT_FULL_MV)
                    else:
                        runtime.observe(h if ok else 0.0, ok)
                    valve_log.append(bool(valve))
                    pwm_log.append(pwm); heard_log.append(h if ok else 0.0); valid_log.append(ok)
                heard, valid = np.array(heard_log), np.array(valid_log)
                mask = valid & voice
                errors.append(float(np.abs(heard[mask] - cents[mask]).mean()) if mask.any() else None)
                records.update({f"{play}/pwm": np.array(pwm_log, np.float32)[None],
                                f"{play}/heard": heard.astype(np.float32)[None], f"{play}/valid": valid[None],
                                f"{play}/valve": np.array(valve_log, np.float32)[None],
                                f"{play}/target_cents": cents.astype(np.float32)[None], f"{play}/target_voice": voice[None],
                                f"{play}/pcm": self.link.audio.span(audio_start, self.link.audio.end)})
        finally:
            self.safe("play done")
        out = req.get("out")
        if out:
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, count=len(errors), shift_cents=shift, sr=SR, **records)
        return {"heard_error_cents": errors, "late_steps": rig.late_steps, "out": out, "shift_cents": shift}

    def cmd_tune(self, client, req):
        """Today's flute: the home note and one sweep out to the far end and back at ``pwm`` (the end switch
        cuts the current).  Returns the heard track, the home and top pitch, and ``shift_cents``: how far the
        home note sounds from the runtime's twin (plays take it as ``shift_cents``)."""
        from .residual_numpy import runtime_for
        self.take(client)
        rig = self._rig(req)
        self.fan_on, self.fan_duty = True, float(req.get("fan", .8))
        self.link.fan(True, self.fan_duty)
        pwm = float(req.get("pwm", .6))
        heard, phase = [], []

        def step(command, valve, name):
            h = float(rig.step(command, valve)["heard"][0])
            heard.append(h if np.isfinite(h) else None); phase.append(name)
            return rig.log["isense_peak_mv"][-1]

        try:
            for _ in range(int(round(float(req.get("warmup_s", 3.0)) / STEP_S))):
                rig.step(0.0, False)
            for _ in range(int(round(1.3 / STEP_S))):
                rig.step(-1.0, False)
            for _ in range(80):
                step(-.3, True, "home")
            for direction, name in ((1, "out"), (-1, "in")):
                for k in range(500):                    # 5 s at most; the end switch cuts the current
                    peak = step(direction * pwm, True, name)
                    if k > 30 and 0 <= peak <= 3:
                        break
                for _ in range(30):
                    step(0.0, True, name + "_end")
        finally:
            self.safe("tune done")
        home = [h for h, p in zip(heard, phase) if p == "home" and h is not None][20:]
        if len(home) < 10:
            raise RuntimeError("the home note was not heard: is the flute sounding?")
        home_cents = float(np.median(home))
        top = [h for h, p in zip(heard, phase) if p == "out_end" and h is not None]
        sweep = [h for h, p in zip(heard, phase) if p in ("out", "in") and h is not None]
        twin_home = float(runtime_for(dict(np.load(req["runtime"]))).cents_at(0.0)) if req.get("runtime") else home_cents
        return {"heard": [None if h is None else round(h, 1) for h in heard], "phase": phase,
                "home_cents": home_cents, "top_cents": float(np.median(top)) if top else None,
                "sweep_max_cents": float(np.percentile(sweep, 98)) if sweep else None,
                "shift_cents": float(np.clip(home_cents - twin_home, -150, 150))}

    def _rig(self, req):
        rig = RealRig(self.link, flip=bool(req.get("flip", False)), max_pwm=min(float(req.get("max_pwm", self.max_pwm)),
                                                                                self.max_pwm))
        rig.start()
        return rig

    # ----- loop -----------------------------------------------------------------------------------
    def reply(self, client, rid, ok, payload=None, error=None):
        msg = {"id": rid, "ok": ok}
        if ok:
            msg.update(payload or {})
        else:
            msg["error"] = error
        client.out += (json.dumps(msg) + "\n").encode()
        self.flush(client)

    def flush(self, client):
        """Send what the socket takes now; the rest waits for EVENT_WRITE, so the 10 ms loop never blocks."""
        try:
            while client.out:
                n = client.sock.send(client.out[:262144])
                del client.out[:n]
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            self.drop(client)
            return
        try:
            self.sel.modify(client.sock, selectors.EVENT_READ | (selectors.EVENT_WRITE if client.out else 0), client)
        except (KeyError, ValueError):
            pass

    def drop(self, client):
        try:
            self.sel.unregister(client.sock)
        except (KeyError, ValueError):
            pass
        client.sock.close()
        if self.owner is client:
            self.safe("connection closed")

    def accept(self):
        sock, addr = self.listener.accept()
        sock.setblocking(False)
        client = Client(sock, addr)
        self.sel.register(sock, selectors.EVENT_READ, client)

    def read(self, client):
        try:
            data = client.sock.recv(65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b""
        if not data:
            self.drop(client)
            return
        client.last_seen = time.monotonic()
        client.buf += data
        while b"\n" in client.buf:
            line, client.buf = client.buf.split(b"\n", 1)
            if not line.strip():
                continue
            rid = None
            try:
                req = json.loads(line)
                rid = req.get("id")
                result = self.handle(client, req)
                if result is not None:
                    self.reply(client, rid, True, result)
            except Exception as exc:      # noqa: BLE001  (every error goes back to the caller)
                self.reply(client, rid, False, error=f"{type(exc).__name__}: {exc}")
                if not isinstance(exc, (ValueError, RuntimeError, KeyError)):
                    self.log(traceback.format_exc())

    def finish_jobs(self):
        for job in list(self.jobs):
            client, rid, kind, (start, stop) = job
            if self.link.audio.end >= stop:
                self.jobs.remove(job)
                pcm = self.link.audio.span(start, stop)
                self.reply(client, rid, True, {"sr": SR, "samples": int(pcm.size),
                                               "pcm16_b64": base64.b64encode(pcm.astype("<i2").tobytes()).decode()})

    def tick(self):
        for key, events in self.sel.select(timeout=0):
            if key.data is None:
                self.accept()
                continue
            if events & selectors.EVENT_WRITE:
                self.flush(key.data)
            if events & selectors.EVENT_READ and key.data.sock.fileno() >= 0:
                self.read(key.data)
        if self.owner is not None and time.monotonic() - self.owner.last_seen > self.lease:
            self.safe(f"no message for {self.lease} s")
        now = time.monotonic()
        if now >= self.next_step:
            self.link.command(self.pwm, self.valve, self.brake)
            self.next_step = max(self.next_step + STEP_S, now - STEP_S)
        self.link.poll(max(0.0, min(.002, self.next_step - time.monotonic())))
        st = self.link.status
        if st is not None and st.t_rx != self.peak_seen:
            self.peak_seen = st.t_rx
            self.peak_hold = max(self.peak_hold, st.isense_peak_mv)
        self.audio_watch(now)
        self.finish_jobs()

    def audio_watch(self, now):
        """Restart the stream when no audio came for AUDIO_STALL_S.  The command bytes have no framing, so
        one lost byte can make the firmware read a data byte as 'X' (stop streaming): pwm 0.6 is 0x0258,
        and 0x58 is 'X'.  Recordings waiting on the old stream are answered with an error."""
        a = self.link.audio
        if a.end != self.audio_seen_end:
            self.audio_seen_end, self.audio_seen_t = a.end, now
            return
        if now - self.audio_seen_t < AUDIO_STALL_S:
            return
        self.audio_restarts += 1
        self.log(f"no audio for {AUDIO_STALL_S} s: restarting the stream ({self.audio_restarts})")
        self.link.start()
        self.audio_seen_end, self.audio_seen_t = None, now
        for client, rid, kind, _ in self.jobs:
            self.reply(client, rid, False, error="RuntimeError: the audio stream restarted")
        self.jobs.clear()

    def serve(self):
        self.link.limit(self.max_pwm)
        self.link.start()
        self.log(f"yamabiko server on port {self.port}, max pwm {self.max_pwm}, lease {self.lease} s")
        try:
            while self.running:
                self.tick()
        finally:
            self.safe("server stopping")
            self.link.stop()

    def close(self):
        for key in list(self.sel.get_map().values()):
            key.fileobj.close()
        self.sel.close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 lets the PC connect over Wi-Fi without adb (no authentication: trusted LAN only)")
    ap.add_argument("--dev", default="/dev/ttyHS1")
    ap.add_argument("--max-pwm", type=float, default=1.0)
    ap.add_argument("--lease", type=float, default=2.0)
    ap.add_argument("--no-flash", action="store_true", help="do not flash build_fw when the firmware is silent")
    ap.add_argument("--wait", type=float, default=0.0, help="keep trying to open the link this long (at boot)")
    args = ap.parse_args(argv)
    until = time.monotonic() + args.wait
    while True:
        try:
            link = Link(args.dev)
            break
        except OSError as exc:              # e.g. arduino-router still holds /dev/ttyHS1
            if time.monotonic() >= until:
                raise SystemExit(f"cannot open {args.dev}: {exc} (stop arduino-router)")
            print(f"cannot open {args.dev} yet: {exc}", flush=True)
            time.sleep(5)
    mcu_ready_lines()
    time.sleep(.5)
    if not args.no_flash:
        print(f"firmware: {ensure_firmware(link)}", flush=True)
    server = Server(link, host=args.host, port=args.port, max_pwm=args.max_pwm, lease=args.lease,
                    log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        link.close()
