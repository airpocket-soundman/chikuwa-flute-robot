"""Talk to mcu/sai1_mic_test on the UNO Q's Linux side, standard library only.

The sketch owns Serial1, which is /dev/ttyHS1 here, so arduino-router must be stopped:

    sudo systemctl stop arduino-router arduino-router-serial
    python3 scripts/sai1_capture.py info
    python3 scripts/sai1_capture.py capture --out /tmp/sai1.raw

The capture is the left slot as S32_LE, one channel. Analyse it where numpy lives:

    python scripts/mic_test.py --from-raw sai1.raw --channels 1 --rate 48000
"""
from __future__ import annotations

import argparse
import os
import sys
import termios
import time
from pathlib import Path


def open_link(dev: str, baud: int) -> int:
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY)
    attr = termios.tcgetattr(fd)
    speed = getattr(termios, f"B{baud}")
    attr[0] = 0                                   # iflag
    attr[1] = 0                                   # oflag
    attr[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attr[3] = 0                                   # lflag
    attr[4] = attr[5] = speed
    attr[6][termios.VMIN] = 0
    attr[6][termios.VTIME] = 1                    # 0.1 s read timeout
    termios.tcsetattr(fd, termios.TCSANOW, attr)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


class Reader:
    def __init__(self, fd: int, timeout: float):
        self.fd, self.timeout, self.buf = fd, timeout, b""

    def _fill(self, deadline: float) -> None:
        chunk = os.read(self.fd, 65536)
        if chunk:
            self.buf += chunk
        elif time.monotonic() > deadline:
            raise TimeoutError(f"no reply; got {self.buf[:200]!r}")

    def line(self) -> str:
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.buf:
            self._fill(deadline)
        head, self.buf = self.buf.split(b"\n", 1)
        return head.decode(errors="replace").rstrip("\r")

    def exact(self, n: int) -> bytes:
        deadline = time.monotonic() + self.timeout
        while len(self.buf) < n:
            self._fill(deadline)
        head, self.buf = self.buf[:n], self.buf[n:]
        return head


def command(fd: int, cmd: str, out: Path | None, timeout: float) -> None:
    os.write(fd, cmd.encode())
    r = Reader(fd, timeout)
    while True:
        line = r.line()
        if line.startswith("DATA "):
            data = r.exact(int(line.split()[1]))
            if out:
                out.write_bytes(data)
                print(f"DATA {len(data)} bytes -> {out}")
            continue
        if line == "END":
            return
        if line:
            print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["info", "stats", "capture"])
    p.add_argument("--dev", default="/dev/ttyHS1")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--out", type=Path, default=Path("/tmp/sai1.raw"))
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout", type=float, default=5.0)
    args = p.parse_args()

    fd = open_link(args.dev, args.baud)
    cmd = {"info": "i", "stats": "h", "capture": "c"}[args.what]
    try:
        for k in range(args.repeat):
            out = args.out
            if args.repeat > 1:
                out = out.with_name(f"{out.stem}_{k}{out.suffix}")
            command(fd, cmd, out if cmd == "c" else None, args.timeout)
    except TimeoutError as e:
        sys.exit(f"{e}\nIs arduino-router stopped and the sketch flashed?")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
