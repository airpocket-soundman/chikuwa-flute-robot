"""Check an I2S microphone (ICS-43434) by recording and tracking the pitch.

Run it on the machine the microphone is wired to (a Raspberry Pi for the first
try, the UNO Q's Linux side later). It records with arecord, so the only
dependency beyond numpy is alsa-utils.

    python3 scripts/mic_test.py --device hw:0,0 --seconds 3

What it answers, in order:

    1. does the microphone put out anything at all
    2. is the L/R pin wired the way the SAI/I2S side expects
    3. does flute_rl.pitch read a sensible pitch off it

Record a steady tone (a tuning fork, a phone app, or the flute itself) while it
runs. With nothing playing it still reports the noise floor, which is worth
having: that noise is the position sensor's noise in this project.

The first 20 ms are dropped. The ICS-43434 outputs nothing for 32,768 SCK
cycles (10.7 ms at 3.072 MHz) after the clock starts and needs up to 20 ms
before its sensitivity has settled (DS-000069 Rev 1.2, Table 5).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flute_rl.pitch import pitch_track  # noqa: E402

SETTLE_S = 0.020
FULL_SCALE = 2.0 ** 23  # 24-bit data, MSB-aligned in a 32-bit slot


def record(device: str, rate: int, seconds: float, channels: int) -> bytes:
    cmd = ["arecord", "-D", device, "-f", "S32_LE", "-c", str(channels),
           "-r", str(rate), "-d", str(int(np.ceil(seconds))), "-t", "raw", "-q"]
    print("$ " + " ".join(cmd))
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("arecord not found. Install alsa-utils.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"arecord failed ({e.returncode}):\n{e.stderr.decode(errors='replace')}")
    return out


def to_float(raw: bytes, channels: int) -> np.ndarray:
    """Raw S32_LE frames -> (samples, channels) in [-1, 1)."""
    x = np.frombuffer(raw, dtype="<i4")
    x = x[: len(x) - len(x) % channels].reshape(-1, channels)
    return (x >> 8) / FULL_SCALE


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x * x)))
    return 20.0 * np.log10(rms) if rms > 0 else -np.inf


def report_channels(sig: np.ndarray) -> None:
    print("\n-- channels " + "-" * 52)
    for c in range(sig.shape[1]):
        ch = sig[:, c]
        name = {0: "left  (slot 0)", 1: "right (slot 1)"}.get(c, f"slot {c}")
        silent = "all zero" if not np.any(ch) else f"peak {np.max(np.abs(ch)):.5f}"
        print(f"  {name}: {dbfs(ch):7.1f} dBFS   {silent}")

    if sig.shape[1] < 2:
        return
    left, right = sig[:, 0], sig[:, 1]
    live = [np.any(left), np.any(right)]
    if not any(live):
        print("\n  Nothing on either slot. Check VDD, then that SCK and WS are"
              "\n  actually running, then the SD wire. Power VDD before the clocks.")
    elif all(live) and np.allclose(left, right, atol=1e-6):
        print("\n  Both slots carry the same samples: the frame sync is off by"
              "\n  a slot, or two microphones share the line.")
    elif live[0] and not live[1]:
        print("\n  Left only, right silent: LR is tied low. This is what the"
              "\n  design expects (docs/i2s_mic_trial.md).")
    elif live[1] and not live[0]:
        print("\n  Right only, left silent: LR is tied high. Move it to GND,"
              "\n  or read slot 1 instead.")


def report_pitch(mono: np.ndarray, rate: int) -> None:
    times, f0, conf = pitch_track(mono, rate)
    voiced = ~np.isnan(f0)
    print("\n-- pitch " + "-" * 55)
    print(f"  frames {len(f0)}, voiced {int(voiced.sum())}"
          f" ({100.0 * voiced.mean():.0f}%)")
    if not voiced.any():
        print("  No pitch found. With a tone playing, check the level above:"
              "\n  pitch_track gates frames below 1e-3 RMS (-60 dBFS).")
        return
    v = f0[voiced]
    med = float(np.median(v))
    cents = 1200.0 * np.log2(v / med)
    print(f"  median {med:8.2f} Hz   spread {np.std(cents):5.1f} cents"
          f"   confidence {np.mean(conf[voiced]):.2f}")
    print(f"  range  {np.min(v):8.2f} - {np.max(v):.2f} Hz")
    if 650.0 <= med <= 1350.0:
        print("  In the flute's range (650-1350 Hz).")
    else:
        print("  Outside the flute's range (650-1350 Hz); fine for a bench tone.")
    print("\n  The spread in cents is what matters: it is the noise of this"
          "\n  project's only position sensor.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="hw:0,0", help="ALSA device (aplay -l / arecord -l)")
    p.add_argument("--rate", type=int, default=48000,
                   help="48000 keeps the ICS-43434 in high-performance mode (23-51.6 kHz)")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--channel", type=int, default=0, help="which slot to track (0 = left)")
    p.add_argument("--from-raw", type=Path, help="read S32_LE raw instead of recording")
    p.add_argument("--save", type=Path, help="write the raw capture here")
    args = p.parse_args()

    if args.rate < 23000 or args.rate > 51600:
        print(f"warning: {args.rate} Hz is outside the ICS-43434's high-performance"
              " range (23-51.6 kHz)", file=sys.stderr)

    raw = args.from_raw.read_bytes() if args.from_raw else record(
        args.device, args.rate, args.seconds, args.channels)
    if args.save:
        args.save.write_bytes(raw)
        print(f"raw capture -> {args.save}")

    sig = to_float(raw, args.channels)
    settle = int(SETTLE_S * args.rate)
    if len(sig) <= settle:
        sys.exit(f"only {len(sig)} samples; need more than the {settle} dropped to settle")
    sig = sig[settle:]
    print(f"\n{len(sig)} samples at {args.rate} Hz "
          f"({len(sig) / args.rate:.2f} s, first {SETTLE_S * 1e3:.0f} ms dropped)")

    report_channels(sig)
    report_pitch(sig[:, min(args.channel, sig.shape[1] - 1)], args.rate)


if __name__ == "__main__":
    main()
