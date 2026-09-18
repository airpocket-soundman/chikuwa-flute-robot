"""Compare how late the microphone reaches a Linux process: MCU SAI1 + UART link vs Linux MI2S0 (ALSA).

Two ICS-43434 side by side hear the same sound: one on the MCU (mcu/yamabiko_link), one on MI2S0.

`record` (on the UNO Q, standard library only) reads both streams at once and logs when each piece
arrives, and pings the MCU so its sample clock can be tied to the Linux clock:

    python3 scripts/latency_compare.py record --seconds 20 --out /tmp/cmp

`analyze` (on the PC, numpy) finds where the MI2S0 stream lines up with the MCU stream, which gives
the capture time of every MI2S0 sample, and then reports for both routes how old the newest sample
is when the process has it (the same measure as scripts/yamabiko_link.py latency):

    python scripts/latency_compare.py analyze cmp

Play something with sharp onsets near the two microphones while recording
(python scripts/latency_compare.py chirps --seconds 25, on the PC).
"""
from __future__ import annotations

import argparse
import array
import os
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FS = 48000
DECIM = 3
FIR_DELAY = 15        # the MCU's anti-alias filter [48 kHz frames]
ALSA_PERIOD = 960     # the patched q6asm capture period: 7680 bytes of S32_LE stereo
ALSA_BUFFER = 7680


# ---------------------------------------------------------------- record (UNO Q)

def cmd_record(args) -> None:
    from yamabiko_link import Link

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    if args.clock_playback:  # IchiPing clocks MI2S0 by playing silence
        procs.append(subprocess.Popen(
            ["aplay", "-q", "-D", args.play_dev, "-t", "raw", "-f", "S32_LE", "-c", "2", "-r", str(FS),
             f"--period-size={args.play_period}", f"--buffer-size={4 * args.play_period}", "/dev/zero"]))
        time.sleep(0.3)
    rec = subprocess.Popen(
        ["arecord", "-q", "-D", args.rec_dev, "-t", "raw", "-f", "S32_LE", "-c", "2", "-r", str(FS),
         f"--period-size={args.rec_period}", f"--buffer-size={args.rec_buffer}"],
        stdout=subprocess.PIPE, bufsize=0)
    procs.append(rec)

    link = Link(args.dev, args.baud)
    link.send(b"S")
    mcu_audio = array.array("h")
    mcu_log, alsa_log, pongs = [], [], []
    alsa_bytes = 0
    t_end = time.monotonic() + args.seconds
    next_ping, token = time.monotonic() + 0.2, 1
    with open(out / "alsa.raw", "wb") as alsa_file:
        while time.monotonic() < t_end:
            ready = select.select([link.fd, rec.stdout.fileno()], [], [], 0.1)[0]
            now = time.monotonic()
            if now >= next_ping:
                link.ping(token)
                token += 1
                next_ping = now + args.ping_every
            if rec.stdout.fileno() in ready:
                chunk = os.read(rec.stdout.fileno(), 1 << 16)
                t = time.monotonic()
                if not chunk:
                    sys.exit("arecord stopped")
                alsa_file.write(chunk)
                alsa_bytes += len(chunk)
                alsa_log.append((alsa_bytes, t))
            if link.fd in ready:
                chunk = os.read(link.fd, 1 << 16)
                t = time.monotonic()
                link.buf += chunk
                while (p := link._packet()) is not None:
                    typ, payload = p
                    if typ == "A":
                        seq, first = struct.unpack_from("<HI", payload)
                        s = array.array("h", payload[6:])
                        if first != len(mcu_audio):
                            sys.exit(f"MCU stream gap at {first} (have {len(mcu_audio)})")
                        mcu_audio.extend(s)
                        mcu_log.append((first + len(s), t))
                    elif typ == "P":
                        tok, frame48, _, _ = struct.unpack("<IIII", payload)
                        t0 = link.tokens.pop(tok, None)
                        if t0 is not None:
                            pongs.append((frame48, t0, t))
    link.close()
    for p in procs:
        p.terminate()
    (out / "mcu.raw").write_bytes(mcu_audio.tobytes())
    with open(out / "log.txt", "w") as f:
        for n, t in mcu_log:
            f.write(f"M {n} {t:.6f}\n")
        for n, t in alsa_log:
            f.write(f"A {n} {t:.6f}\n")
        for fr, t0, t1 in pongs:
            f.write(f"P {fr} {t0:.6f} {t1:.6f}\n")
    print(f"MCU {len(mcu_audio)} samples in {len(mcu_log)} packets, MI2S0 {alsa_bytes} bytes in "
          f"{len(alsa_log)} reads, {len(pongs)} pings -> {out}")


# ---------------------------------------------------------------- chirps (PC)

def cmd_chirps(args) -> None:
    import io
    import math
    import random
    import wave
    import winsound

    rate, n = 48000, int(48000 * args.seconds)
    frames = bytearray(2 * n)
    burst, every = int(rate * 0.03), int(rate * args.every)
    rng = random.Random(1)
    for start in range(int(rate * 0.5), n - burst, every):
        start += rng.randrange(0, every // 4)            # irregular, so the match cannot slip a period
        for i in range(burst):
            t = i / rate
            g = math.sin(math.pi * i / burst)
            v = int(32767 * args.amp * g * math.sin(2 * math.pi * (500 * t + (5500 / 0.03) * t * t / 2)))
            struct.pack_into("<h", frames, 2 * (start + i), v)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(bytes(frames))
    winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)


# ---------------------------------------------------------------- analyze (PC)

def cmd_analyze(args) -> None:
    import numpy as np

    d = Path(args.dir)
    mcu = np.frombuffer((d / "mcu.raw").read_bytes(), "<i2").astype(float) / 2 ** 15
    alsa = np.frombuffer((d / "alsa.raw").read_bytes(), "<i4").reshape(-1, 2).astype(float) / 2 ** 31
    rows = {"M": [], "A": [], "P": []}
    for line in (d / "log.txt").read_text().splitlines():
        k, *v = line.split()
        rows[k].append([float(x) for x in v])
    M, A, P = (np.array(rows[k]) for k in "MAP")

    # MCU clock: SAI frame -> Linux time, from the quarter of pings with the shortest round trip
    rtt = P[:, 2] - P[:, 1]
    keep = rtt <= np.quantile(rtt, 0.25)
    b, a = np.polyfit(P[keep, 0], (P[keep, 1] + P[keep, 2]) / 2, 1)
    bound = rtt[keep].max() / 2

    # MCU output sample i came out when SAI frame 3i+2 was in; the linear-phase filter makes it
    # the sound of frame 3i+2-FIR_DELAY. On the 48 kHz grid of acoustic time: y[f] = sound of frame f.
    shift = FIR_DELAY - (DECIM - 1)
    y = np.repeat(mcu, DECIM)[shift:]
    lat_mcu = (M[:, 1] - (a + b * (DECIM * (M[:, 0] - 1) + DECIM - 1 - FIR_DELAY))) * 1000

    # MI2S0: the slot with the microphone, lined up with y: x[j] = y[j - L(j)]
    ch = int(np.argmax(np.abs(alsa).mean(axis=0)))
    x = alsa[:, ch] - alsa[:, ch].mean()
    y = y - y.mean()
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n)), n)
    lag = int(np.argmax(np.abs(c)))
    lag = lag - n if lag > n // 2 else lag
    # refine in 1 s windows: the two sample clocks drift apart
    W, R = FS, 100
    pos, offs = [], []
    for s in range(0, len(x) - W, W):
        s_y = s - lag - R
        if s_y < 0 or s_y + W + 2 * R > len(y):
            continue
        seg = x[s:s + W]
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:
            continue
        cc = np.correlate(y[s_y:s_y + W + 2 * R], seg, mode="valid")
        k = int(np.argmax(np.abs(cc)))
        fk = float(k)
        if 0 < k < len(cc) - 1:
            den = cc[k - 1] - 2 * cc[k] + cc[k + 1]
            fk += 0.5 * (cc[k - 1] - cc[k + 1]) / den if den else 0.0
        pos.append(s + W / 2)
        offs.append(lag + R - fk)
    pos, offs = np.array(pos), np.array(offs)
    if len(offs) >= 2:
        g1, g0 = np.polyfit(pos, offs, 1)
    else:
        g1, g0 = 0.0, float(lag)
    resid = np.abs(offs - (g0 + g1 * pos)).max() if len(offs) else float("nan")

    j = A[:, 0] / 8 - 1                               # newest MI2S0 sample of each read
    lat_alsa = (A[:, 1] - (a + b * (j - (g0 + g1 * j)))) * 1000
    spacing = np.diff(A[:, 1]) * 1000

    def show(name, v):
        v = v[len(v) // 20:]                          # drop the first 5 %: start-up
        print(f"  {name:6s} n {len(v):5d}  min {v.min():6.2f}  median {np.median(v):6.2f}  "
              f"p95 {np.quantile(v, .95):6.2f}  p99 {np.quantile(v, .99):6.2f}  max {v.max():6.2f} ms")

    print(f"clock tie +-{bound * 1000:.2f} ms; MI2S0 slot {ch}, level {20 * np.log10(np.std(x) + 1e-12):.1f} dBFS; "
          f"lined up in {len(offs)} windows, worst residual {resid:.1f} frames; clocks differ by {g1 * 1e6:+.0f} ppm")
    print(f"MI2S0 reads: every {np.median(spacing):.2f} ms (median), {len(A)} reads, "
          f"{np.median(np.diff(A[:, 0])) / 8:.0f} frames each")
    print("age of the newest sample when the process has it (sound at the microphone -> Python):")
    show("MCU", lat_mcu)
    show("MI2S0", lat_alsa)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    r = sub.add_parser("record")
    r.add_argument("--seconds", type=float, default=20.0)
    r.add_argument("--out", default="/tmp/cmp")
    r.add_argument("--dev", default="/dev/ttyHS1")
    r.add_argument("--baud", type=int, default=921600)
    r.add_argument("--ping-every", type=float, default=0.05)
    r.add_argument("--rec-dev", default="hw:0,1")
    r.add_argument("--rec-period", type=int, default=ALSA_PERIOD)
    r.add_argument("--rec-buffer", type=int, default=ALSA_BUFFER)
    r.add_argument("--clock-playback", action="store_true", help="play silence on MI2S0 to drive its clocks")
    r.add_argument("--play-dev", default="hw:0,0")
    r.add_argument("--play-period", type=int, default=480)
    c = sub.add_parser("chirps")
    c.add_argument("--seconds", type=float, default=25.0)
    c.add_argument("--every", type=float, default=0.4)
    c.add_argument("--amp", type=float, default=0.5)
    an = sub.add_parser("analyze")
    an.add_argument("dir")
    args = p.parse_args()
    {"record": cmd_record, "chirps": cmd_chirps, "analyze": cmd_analyze}[args.what](args)


if __name__ == "__main__":
    main()
