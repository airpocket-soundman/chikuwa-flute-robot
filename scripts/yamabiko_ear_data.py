"""PC: build the ear network's dataset from the rig's recordings (step3).

    python scripts/yamabiko_ear_data.py --out runs/ear/dataset.npz

Every 10 ms of every recording (motion tests, calibration-free plays with their audio) becomes one example:
the input is the causal 20 ms the ear hears on the rig (320 samples ending at that instant), the label is
the pitch found offline with time to spare: a 64 ms window centred on the input, rumble removed, YIN and
the strongest spectral peak in agreement, then a median over neighbouring frames.  Frames where the two
disagree are kept as "unsure" (no pitch loss); quiet frames are unvoiced.  Recordings are split by file
into train / test (--test-every).
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.pitch import yin_frame  # noqa: E402

SR, HOP, FRAME, WIDE = 16000, 160, 320, 1024
FMIN, FMAX = 300.0, 2000.0


def recordings():
    """(name, pcm int16) of every recording with audio, cut at chunk boundaries."""
    files = sorted(glob.glob("runs/real/motion/*.npz") + glob.glob("runs/real/motion/full/*.npz")
                   + glob.glob("runs/real/eval/*.npz") + glob.glob("runs/real/cycle_rec_01.npz"))
    for f in files:
        d = np.load(f)
        name = pathlib.Path(f).relative_to("runs/real").as_posix()
        if "pcm" in d.files:
            pcm = d["pcm"]
            if "chunk_n" in d.files:
                start = 0
                for i, n in enumerate(d["chunk_n"]):
                    yield f"{name}#{i}", pcm[start:start + int(n)]
                    start += int(n)
            else:
                yield name, pcm
        else:
            for k in range(int(d["count"])):
                if f"{k}/pcm" in d.files:
                    yield f"{name}#{k}", d[f"{k}/pcm"]


def highpass(x, cutoff=250.0):
    s = np.fft.rfft(x - x.mean())
    s[:int(np.ceil(cutoff * len(x) / SR))] = 0
    return np.fft.irfft(s, len(x))


def label(x):
    """Offline pitch of a centred wide window: (cents or nan, state) with state 1 voiced, 0 quiet, -1 unsure."""
    w = highpass(x)
    db = 20 * np.log10(max(np.sqrt(np.mean(w * w)), 1e-9))
    if db < -58:
        return np.nan, 0
    f0, conf = yin_frame(w, SR, FMIN, FMAX, .15)
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) + 1e-12
    f = np.fft.rfftfreq(len(w), 1 / SR)
    m = np.flatnonzero((f > FMIN) & (f < FMAX))
    j = m[np.argmax(spec[m])]
    a, b, c = np.log(spec[j - 1:j + 2])
    peak = f[j] + .5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
    prom = 20 * np.log10(spec[j] / np.median(spec[m]))
    if not np.isfinite(f0) or conf < .7 or prom < 20:
        return np.nan, -1
    if abs(1200 * np.log2(f0 / peak)) > 30:
        return np.nan, -1
    return 1200 * np.log2(f0 / 440.0), 1


def build(pcm):
    x = pcm.astype(np.float64) / 32768.0
    X, cents, state = [], [], []
    for t in range(WIDE, len(x) - WIDE // 2, HOP):              # t = the instant the ear decides
        centre = t - FRAME // 2
        c, s = label(x[centre - WIDE // 2:centre + WIDE // 2])
        X.append(pcm[t - FRAME:t]); cents.append(c); state.append(s)
    cents, state = np.array(cents), np.array(state)
    # median over +-2 frames among voiced neighbours; a lone voiced frame becomes unsure
    smooth = cents.copy()
    for i in np.flatnonzero(state == 1):
        win = cents[max(0, i - 2):i + 3]
        win = win[np.isfinite(win)]
        if len(win) < 3:
            state[i], smooth[i] = -1, np.nan
        else:
            smooth[i] = np.median(win)
    return np.array(X, np.int16), smooth.astype(np.float32), state.astype(np.int8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/ear/dataset.npz")
    ap.add_argument("--test-every", type=int, default=5, help="every n-th recording goes to the test split")
    args = ap.parse_args()
    parts, names = [], []
    for i, (name, pcm) in enumerate(recordings()):
        if len(pcm) < 2 * WIDE:
            continue
        X, c, s = build(pcm)
        split = np.full(len(X), 1 if i % args.test_every == args.test_every - 1 else 0, np.int8)
        source = np.full(len(X), len(names), np.int32)
        parts.append((X, c, s, split, source)); names.append(name)
        print(f"{name:55s} {len(X):6d} frames  voiced {np.mean(s == 1):.2f} unsure {np.mean(s == -1):.2f}"
              f"  {'test' if split[0] else 'train'}", flush=True)
    X, c, s, split, source = (np.concatenate(z) for z in zip(*parts))
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=X, cents=c, state=s, split=split, source=source, names=np.array(names))
    print(f"saved {out}: {len(X)} frames ({len(X) / 100 / 60:.1f} min), voiced {np.mean(s == 1):.2f}, "
          f"unsure {np.mean(s == -1):.2f}, test {np.mean(split == 1):.2f}")


if __name__ == "__main__":
    main()
