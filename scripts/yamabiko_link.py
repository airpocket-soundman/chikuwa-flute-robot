"""Receive the microphone stream of mcu/yamabiko_link on the UNO Q's Linux side, standard library only.

The sketch owns Serial1 (/dev/ttyHS1 here), so arduino-router must be stopped first:

    sudo systemctl stop arduino-router arduino-router-serial
    python3 scripts/yamabiko_link.py info
    python3 scripts/yamabiko_link.py latency --seconds 10
    python3 scripts/yamabiko_link.py record --seconds 3 --out /tmp/link.raw

`latency` measures how old the newest sample of each audio packet is when this process has it.
The MCU and Linux clocks are tied together by pings: the MCU answers each ping with the number
of SAI frames captured when it read the ping, so that frame was captured between sending the ping
and getting the answer. The pings with the shortest round trip pin the clock offset best; the
drift between the two clocks is fitted over the whole run. The error bound of the offset is
printed with the result.

`record` writes the 16 kHz stream as S32_LE mono, for the PC:

    python scripts/mic_test.py --from-raw link.raw --channels 1 --rate 16000

`Link.packets()` is the route into a controller or network on this side: it yields the audio
as it arrives, with the arrival time and the sample index of the first sample.
"""
from __future__ import annotations

import argparse
import array
import os
import select
import statistics
import struct
import sys
import termios
import time
from dataclasses import dataclass
from pathlib import Path

FS_SAI = 48000.0      # nominal SAI frame rate
DECIM = 3             # the stream runs at FS_SAI / DECIM
FIR_DELAY = 15        # group delay of the MCU's 31-tap anti-alias filter [48 kHz frames]
MAGIC = b"\xa5\x5a"


@dataclass
class Audio:
    seq: int
    first: int                # 16 kHz index of samples[0]
    samples: array.array      # int16
    t_rx: float               # time.monotonic() when the packet was complete


@dataclass
class Pong:
    token: int
    frame48: int
    overruns: int
    max_loop_us: int
    t_rx: float


class Link:
    def __init__(self, dev: str = "/dev/ttyHS1", baud: int = 921600):
        self.fd = os.open(dev, os.O_RDWR | os.O_NOCTTY)
        attr = termios.tcgetattr(self.fd)
        attr[0] = attr[1] = attr[3] = 0
        attr[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attr[4] = attr[5] = getattr(termios, f"B{baud}")
        attr[6][termios.VMIN] = 1        # return as soon as a byte is there
        attr[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attr)
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        self.buf = bytearray()
        self.bad = 0                     # bytes skipped while resynchronising + checksum failures
        self.tokens: dict[int, float] = {}

    def close(self) -> None:
        try:
            os.write(self.fd, b"X")
        finally:
            os.close(self.fd)

    def send(self, data: bytes) -> None:
        os.write(self.fd, data)

    def ping(self, token: int) -> None:
        self.tokens[token] = time.monotonic()
        os.write(self.fd, b"P" + struct.pack("<I", token))

    def _packet(self):
        """One parsed packet from the buffer, or None if it holds no complete packet."""
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

    def packets(self, until: float | None = None):
        """Yield Audio, Pong and str (text) as they arrive, and None after 50 ms of silence, until
        time.monotonic() passes `until`."""
        while until is None or time.monotonic() < until:
            if not select.select([self.fd], [], [], 0.05)[0]:
                yield None                   # nothing for 50 ms: lets the caller act and stop
                continue
            chunk = os.read(self.fd, 65536)
            t = time.monotonic()
            self.buf += chunk
            while (p := self._packet()) is not None:
                typ, payload = p
                if typ == "A":
                    seq, first = struct.unpack_from("<HI", payload)
                    yield Audio(seq, first, array.array("h", payload[6:]), t)
                elif typ == "P":
                    yield Pong(*struct.unpack("<IIII", payload), t)
                elif typ == "T":
                    yield payload.decode(errors="replace")


def fit_line(x: list[float], y: list[float]) -> tuple[float, float]:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx if sxx else 0.0
    return slope, my - slope * mx


def pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def cmd_info(link: Link) -> None:
    link.send(b"i")
    got = 0
    for p in link.packets(until=time.monotonic() + 1.0):
        if isinstance(p, str):
            print(p)
            got += 1
            if got == 2:
                return
    sys.exit("no reply: is arduino-router stopped and mcu/yamabiko_link flashed?")


def cmd_latency(link: Link, seconds: float, ping_every: float) -> None:
    link.send(b"S")
    time.sleep(0.05)
    t_end = time.monotonic() + seconds
    audio: list[tuple[int, float]] = []   # (48 kHz frame of the newest sample, arrival time)
    pongs: list[tuple[int, float, float]] = []  # (frame48, t_sent, t_rx)
    gaps = expected = 0
    last_seq = None
    overruns = max_loop = 0
    token = 1
    settled = next_ping = time.monotonic() + 0.2   # skip the start: packets queued before reading began
    for p in link.packets(until=t_end):
        now = time.monotonic()
        if now >= next_ping:
            link.ping(token)
            token += 1
            next_ping = now + ping_every
        if isinstance(p, Audio) and p.t_rx >= settled:
            if last_seq is not None and p.seq != (last_seq + 1) & 0xFFFF:
                gaps += 1
            last_seq = p.seq
            expected += 1
            newest = (p.first + len(p.samples) - 1) * DECIM + (DECIM - 1)
            audio.append((newest, p.t_rx))
        elif isinstance(p, Pong):
            t_sent = link.tokens.pop(p.token, None)
            if t_sent is not None:
                pongs.append((p.frame48, t_sent, p.t_rx))
            overruns, max_loop = p.overruns, p.max_loop_us
    link.send(b"X")
    if len(pongs) < 5 or not audio:
        sys.exit(f"too little data: {len(pongs)} pongs, {len(audio)} packets")

    # Clock tie: frame f was captured at about t_sent + rtt/2. Keep the quarter with the shortest
    # round trips, fit time = a + f / fs, and bound the error by half their longest round trip.
    rtts = sorted(t1 - t0 for _, t0, t1 in pongs)
    cut = rtts[max(0, len(rtts) // 4 - 1)]
    best = [(f, (t0 + t1) / 2) for f, t0, t1 in pongs if t1 - t0 <= cut]
    slope, icpt = fit_line([f for f, _ in best], [t for _, t in best])
    fs = 1.0 / slope if slope > 0 else FS_SAI
    bound = cut / 2

    lat = [(t - (icpt + slope * f)) * 1000.0 for f, t in audio]
    arrivals = [b[1] - a[1] for a, b in zip(audio, audio[1:])]
    print(f"packets {len(audio)}  seq gaps {gaps}  link errors {link.bad}  MCU overruns {overruns}"
          f"  MCU max loop {max_loop} us")
    print(f"pings {len(pongs)}  round trip min {rtts[0]*1000:.2f}  median {statistics.median(rtts)*1000:.2f}"
          f"  max {rtts[-1]*1000:.2f} ms")
    print(f"SAI rate against the Linux clock {fs:.1f} Hz ({(fs / FS_SAI - 1) * 1e6:+.0f} ppm)")
    print(f"packet spacing median {statistics.median(arrivals)*1000:.2f}  p99 {pct(arrivals, 0.99)*1000:.2f}"
          f"  max {max(arrivals)*1000:.2f} ms")
    print(f"age of the newest sample on arrival [ms], clock tie +-{bound*1000:.2f}:")
    print(f"  min {min(lat):.2f}  median {statistics.median(lat):.2f}  p95 {pct(lat, 0.95):.2f}"
          f"  p99 {pct(lat, 0.99):.2f}  max {max(lat):.2f}")
    over = "  ".join(f"> {ms} ms: {sum(v > ms for v in lat)}" for ms in (5, 10, 20))
    print(f"  late packets  {over}  (of {len(lat)})")
    fir = FIR_DELAY / FS_SAI * 1000.0
    print(f"  + the anti-alias filter delays the sound by {fir:.2f} ms more (not in the numbers above)")


def cmd_record(link: Link, seconds: float, out: Path) -> None:
    link.send(b"S")
    t_end = time.monotonic() + seconds + 0.05
    data = array.array("i")
    first = None
    for p in link.packets(until=t_end):
        if isinstance(p, Audio):
            if first is None:
                first = p.first
            data.extend(s << 16 for s in p.samples)
    link.send(b"X")
    out.write_bytes(data.tobytes())
    print(f"{len(data)} samples at {FS_SAI / DECIM:.0f} Hz -> {out} (S32_LE mono), first index {first}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["info", "latency", "record"])
    p.add_argument("--dev", default="/dev/ttyHS1")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--ping-every", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=Path("/tmp/link.raw"))
    args = p.parse_args()

    link = Link(args.dev, args.baud)
    try:
        if args.what == "info":
            cmd_info(link)
        elif args.what == "latency":
            cmd_latency(link, args.seconds, args.ping_every)
        else:
            cmd_record(link, args.seconds, args.out)
    finally:
        link.close()


if __name__ == "__main__":
    main()
