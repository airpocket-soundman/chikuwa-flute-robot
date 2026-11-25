"""PC: talk to the Tab5 panel (tab5/yamabiko_panel) over its USB serial: its state, and its functions by command.

    python scripts/tab5_cli.py status                # what the panel holds (page version, phase, I2C counts, ...)
    python scripts/tab5_cli.py page                  # the page it shows (steps, archive clips, practice, message)
    python scripts/tab5_cli.py press 1               # the EXEC button once (record / end / stop); press 2: again
    python scripts/tab5_cli.py step 12               # choose the 13th step;  archive 0: play the first clip
    python scripts/tab5_cli.py calibrate | tap 150 200 | tab archive | bright 200 | log off | redraw | help
    python scripts/tab5_cli.py watch 60              # print what the panel logs (taps, presses, events) for 60 s

The port is the first USB serial with Espressif's id (0x303A), or ``--port``.  Every command is one line, every
answer one JSON line (the panel's serial API, "serial API" in yamabiko_panel.ino).  ``press``, ``step``,
``archive`` and ``calibrate`` reach the rig: the flute can sound.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import serial
from serial.tools import list_ports


def find_port():
    for p in list_ports.comports():
        if p.vid == 0x303A:
            return p.device
    raise SystemExit("no Tab5 on USB (Espressif 0x303A); give --port")


def opened(port):
    """The port opened without moving DTR / RTS (moving them restarts the ESP32-P4)."""
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 115200, .2
    s.dtr = s.rts = False
    s.open()
    return s


def ask(port, line, timeout=10.0):
    """Send ``line`` and return the JSON answer; sent again every 2 s (opening the port can restart the panel,
    which then takes a few seconds to come up)."""
    with opened(port) as s:
        until, sent, buf = time.time() + timeout, 0.0, b""
        while time.time() < until:
            if time.time() - sent > 2.0:
                s.write((line + "\n").encode())
                sent = time.time()
            buf += s.read(4096)
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                raw = raw.strip()
                if raw.startswith(b"{"):
                    try:
                        return json.loads(raw)
                    except ValueError:
                        pass
    raise SystemExit(f"no answer to {line!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="+")
    ap.add_argument("--port", default=None)
    args = ap.parse_args()
    port = args.port or find_port()
    if args.command[0] == "watch":
        seconds = float(args.command[1]) if len(args.command) > 1 else 30.0
        with opened(port) as s:
            until, buf = time.time() + seconds, b""
            while time.time() < until:
                buf += s.read(4096)
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    text = raw.decode(errors="replace").strip()
                    if text:
                        print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)
        return
    print(json.dumps(ask(port, " ".join(args.command)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
