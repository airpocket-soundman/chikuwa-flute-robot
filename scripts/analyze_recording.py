"""Measure the pitch of a flute recording (WAV) and relate it to the tube length.

Prints each sounding stretch with its pitch (Hz, note name, cents off), and the
acoustic length a tube closed at the bottom would need for that pitch, so a
recording can be compared with the tube model (f = c / 4L):

    python scripts/analyze_recording.py take.wav --tube-mm 60 --bore-mm 10 --temp 25

WAV must be PCM (16/24/32-bit integer or 32-bit float), mono or stereo; convert
phone recordings first (e.g. ffmpeg -i rec.m4a -ac 1 -ar 48000 take.wav).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl.pitch import pitch_track  # noqa: E402
from flute_rl.sim import speed_of_sound  # noqa: E402

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        x = np.where(x >= 1 << 23, x - (1 << 24), x).astype(float) / (1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(float) / 2**31
    else:
        raise ValueError(f"unsupported sample width {width}")
    return x.reshape(-1, ch).mean(axis=1), sr


def note_name(f: float) -> str:
    m = 69 + 12 * np.log2(f / 440.0)
    k = int(round(m))
    return f"{NAMES[k % 12]}{k // 12 - 1} {100 * (m - k):+.0f}c"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--fmin", type=float, default=200.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--temp", type=float, default=25.0, help="air temperature [C]")
    ap.add_argument("--tube-mm", type=float, default=None, help="physical length of the tube, for comparison")
    ap.add_argument("--bore-mm", type=float, default=None, help="inner diameter, for the end correction")
    ap.add_argument("--min-sec", type=float, default=0.08, help="ignore sounding stretches shorter than this")
    args = ap.parse_args()

    x, sr = read_wav(args.wav)
    x = x / (np.max(np.abs(x)) + 1e-12)
    frame = int(2 ** np.ceil(np.log2(sr * 0.04)))  # ~40 ms frames
    t, f0, conf = pitch_track(x, sr, frame=frame, fmin=args.fmin, fmax=min(args.fmax, 0.45 * sr), rms_gate=0.02)
    c = speed_of_sound(args.temp)
    print(f"{args.wav}: {len(x) / sr:.1f} s @ {sr} Hz, speed of sound {c:.1f} m/s")
    if args.tube_mm:
        corr = 0.6 * args.bore_mm / 2 if args.bore_mm else 0.0
        l_eff = (args.tube_mm + corr) / 1000.0
        print(f"closed-bottom tube {args.tube_mm:.0f} mm (+{corr:.1f} mm end correction): expected "
              f"{c / (4 * l_eff):.0f} Hz, then {3 * c / (4 * l_eff):.0f} / {5 * c / (4 * l_eff):.0f} Hz")

    voiced = np.isfinite(f0)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(int), [0]])))
    rows = 0
    for a, b in zip(edges[::2], edges[1::2]):
        if (t[b - 1] - t[a]) < args.min_sec:
            continue
        seg = f0[a:b]
        f = float(np.median(seg))
        spread = float(np.std(1200 * np.log2(seg / f)))
        print(f"  {t[a]:6.2f}-{t[b - 1]:6.2f} s  {f:7.1f} Hz  {note_name(f):>10s}  wobble {spread:5.1f} c  "
              f"-> closed tube acoustic length {1000 * c / (4 * f):6.1f} mm")
        rows += 1
    if rows == 0:
        print("  no steady tone found (try a louder recording, or lower --min-sec)")
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    band = (freqs > args.fmin) & (freqs < min(args.fmax, sr / 2))
    peaks = np.argsort(spec[band])[::-1]
    top, used = [], []
    for i in peaks:
        fpk = freqs[band][i]
        if all(abs(1200 * np.log2(fpk / u)) > 100 for u in used):
            used.append(fpk)
            top.append(fpk)
        if len(top) == 5:
            break
    print("strongest spectral peaks: " + ", ".join(f"{p:.0f} Hz" for p in sorted(top)))


if __name__ == "__main__":
    main()
