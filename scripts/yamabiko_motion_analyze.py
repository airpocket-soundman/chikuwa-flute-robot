"""PC: analyse recordings of scripts/yamabiko_motion_test.py (pitch at every rest, pitch tracks while moving).

    python scripts/yamabiko_motion_analyze.py runs/real/motion/*.npz [--fig runs/real/motion/fig]

Pitch at rest: the strongest spectral peak (300..2000 Hz) over the quiet part of each rest.
Pitch while moving: STFT peak in 340..1500 Hz after dividing out the motor's median spectrum.
Prints a summary per file and writes one figure per file.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FMIN, FMAX = 300.0, 2000.0


class Rec:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.path = pathlib.Path(path)
        self.d = d
        self.program = str(d["program"])
        self.sr = int(d["sr"])
        self.pcm = d["pcm"].astype(float) / 32768
        starts = np.concatenate([[0], np.cumsum(d["chunk_n"])[:-1]])
        self.chunks = list(zip(d["chunk_t0"], starts, d["chunk_n"]))
        self.log = d["log"]
        self.cmd_t, self.cmd_pwm, self.cmd_label = d["cmd_t"], d["cmd_pwm"], d["cmd_label"]
        self.mark_t, self.mark_label = d["mark_t"], d["mark_label"]

    def audio(self, t1, t2):
        out = []
        for t0, s0, n in self.chunks:
            a = int(max(0, round((t1 - t0) * self.sr))); b = int(min(n, round((t2 - t0) * self.sr)))
            if b > a:
                out.append(self.pcm[s0 + a:s0 + b])
        return np.concatenate(out) if out else np.zeros(0)

    def rest_pitch(self, t1, t2):
        """(Hz or NaN, level dB, prominence dB): the strongest spectral peak in FMIN..FMAX over the window.
        (YIN loses the loud, clean tones of a strong fan, so the peak of a Hann-windowed FFT is used.)"""
        x = self.audio(t1, t2)
        if len(x) < 512:
            return np.nan, np.nan, 0.0
        db = 20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-9))
        sp = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) + 1e-12
        f = np.fft.rfftfreq(len(x), 1 / self.sr)
        m = np.flatnonzero((f > FMIN) & (f < FMAX))
        j = m[np.argmax(sp[m])]
        prom = 20 * np.log10(sp[j] / np.median(sp[m]))
        a, b, c = np.log(sp[j - 1:j + 2])
        hz = f[j] + .5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
        return (float(hz) if prom > 25 and db > -50 else np.nan), float(db), float(prom)

    def rests(self):
        """(label, t_start, t_end, hz, db, voiced) for every rest mark."""
        out = []
        later = np.sort(np.concatenate([self.cmd_t, self.mark_t, [self.log[-1, 0]]]))
        for t, lab in zip(self.mark_t, self.mark_label):
            if not str(lab).startswith("rest_"):
                continue
            nxt = later[later > t + .05]
            t_end = (nxt[0] if nxt.size else t + .5) - .03
            t1 = max(t + .15, t_end - 1.5) if "_f" in str(lab) else t + .15   # fan steps: once it has settled
            hz, db, v = self.rest_pitch(t1, t_end)
            out.append((str(lab)[5:], t, t_end, hz, db, v))
        return out

    def track(self, nper=2048, hop=256):
        from scipy.signal import stft
        f, t, Z = stft(self.pcm, self.sr, nperseg=nper, noverlap=nper - hop)
        # sample index -> PC time through the chunk table
        times = np.empty_like(t)
        for i, tt in enumerate(t):
            k = int(tt * self.sr)
            for t0, s0, n in self.chunks:
                if s0 <= k < s0 + n:
                    times[i] = t0 + (k - s0) / self.sr
                    break
            else:
                times[i] = np.nan
        P = np.abs(Z) ** 2
        moving = np.interp(times, self.log[:, 0], self.log[:, 2]) > 4
        floor = np.where(moving, np.median(P[:, moving], axis=1, keepdims=True) if moving.any() else 1,
                         np.median(P[:, ~moving], axis=1, keepdims=True))
        R = 10 * np.log10(P / np.maximum(floor, 1e-20))
        band = (f > 340) & (f < 1500)
        fb, Rb = f[band], R[band]
        j = np.argmax(Rb, axis=0)
        hz = np.where(Rb[j, np.arange(len(j))] > 12, fb[j], np.nan)
        return times, hz, moving


def cents(hz, ref=440.0):
    return 1200 * np.log2(np.asarray(hz, float) / ref)


def analyse(path, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = Rec(path)
    print(f"\n== {r.path.name} ({r.program}) ==")
    rests = r.rests()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    if r.program == "sweep":
        took = dict(zip(r.d["took_label"], r.d["took_s"]))
        outs = [v for k, v in took.items() if k.startswith("out")]
        ins = [v for k, v in took.items() if k.startswith("in")]
        print(f"stroke time out {np.mean(outs):.2f}+-{np.std(outs):.3f} s, in {np.mean(ins):.2f}+-{np.std(ins):.3f} s"
              f"  (limit {r.d['args']})"[:120])
        for lab, t, te, hz, db, v in rests:
            print(f"  rest after {lab:5s}: {hz:7.1f} Hz  {db:6.1f} dB  prominence {v:.0f} dB")
        times, hz, moving = r.track()
        for lab, t in zip(r.mark_label, r.mark_t):
            lab = str(lab)
            if lab.startswith(("out", "in")):
                k = (times >= t) & (times < t + took.get(lab, 2) + .3)
                axes[1].plot(times[k] - t, hz[k], "." if lab.startswith("out") else "x", ms=3, label=lab)
        axes[1].set_xlabel("s since the stroke started"); axes[1].legend(ncol=6, fontsize=7)
        axes[0].plot(times, hz, ".", ms=2)
    elif r.program == "steps":
        reps = {}
        for lab, t, te, hz, db, v in rests:
            rep, kind = lab.split("_")
            reps.setdefault(rep, []).append((kind, hz, db, v))
        for rep, rows in reps.items():
            seq = [h for _, h, _, _ in rows]
            axes[1].plot(cents(seq), ".-", label=rep)
            outs = [h for k, h, _, _ in rows if k.startswith("out")]
            ins = [h for k, h, _, _ in rows if k.startswith("in")]
            def brk(hs):
                c = cents(hs)
                for i in range(1, len(c)):
                    if not np.isfinite(c[i]) or abs(c[i] - c[i - 1]) > 400:
                        return i
                return None
            b = brk(outs)
            print(f"  {rep}: out {len(outs)} steps {outs[0]:.0f} -> {np.nanmax(outs[:b] if b else outs):.0f} Hz,"
                  f" first break at out step {b}; in {len(ins)} steps, ends {ins[-1]:.0f} Hz")
            c = cents(outs[:b] if b else outs)
            d = np.diff(c)
            print(f"      out cents/step: median {np.nanmedian(d):.1f}, min {np.nanmin(d):.1f}, max {np.nanmax(d):.1f}")
            ci = cents(ins)
            di = np.diff(ci)
            di = di[np.isfinite(di) & (np.abs(di) < 300)]
            if di.size:
                print(f"      in  cents/step (fundamental only): median {np.median(di):.1f}, min {di.min():.1f}, max {di.max():.1f}")
        # spread across repeats at each out step
        outs_by_rep = [np.array([h for k, h, _, _ in rows if k.startswith("out")]) for rows in reps.values()]
        n = min(len(o) for o in outs_by_rep)
        M = cents(np.array([o[:n] for o in outs_by_rep]))
        sd = np.nanstd(M, axis=0)
        print(f"  spread across repeats per out step (cents sd): {np.array2string(sd, precision=0, max_line_width=200)}")
        axes[1].set_xlabel("rest index (out steps, then in steps)"); axes[1].set_ylabel("cents re A4"); axes[1].legend()
        axes[0].plot([x[1] for x in rests], cents([x[3] for x in rests]), ".")
    elif r.program == "fan":
        table = {}
        for lab, t, te, hz, db, v in rests:
            pos, f = lab.split("_f")
            table.setdefault(pos, {})[int(f)] = (hz, db, v)
        levels = sorted({f for row in table.values() for f in row})
        print("  position  " + "".join(f"{f:>9d}%" for f in levels) + "   (Hz; cents re 50% below)")
        for pos, row in table.items():
            hz = [row.get(f, (np.nan,))[0] for f in levels]
            c = cents(hz, hz[0]) if np.isfinite(hz[0]) else [np.nan] * len(hz)
            print(f"  {pos:8s}  " + "".join(f"{h:10.1f}" for h in hz))
            print(f"  {'':8s}  " + "".join(f"{x:+10.0f}" for x in c))
            axes[1].plot(levels, hz, "o-", label=pos)
        axes[1].set_xlabel("fan %"); axes[1].set_ylabel("Hz"); axes[1].legend()
        axes[0].plot([x[1] for x in rests], [x[3] for x in rests], ".")
    else:
        by_t = {}
        for lab, t, te, hz, db, v in rests:
            if lab.startswith("r"):
                by_t.setdefault(int(lab.split("_t")[1]), []).append(hz)
        took = r.d["took_s"]
        print(f"  homing time after each run: {np.mean(took):.2f}+-{np.std(took):.3f} s")
        for ms, hs in sorted(by_t.items()):
            c = cents(hs)
            print(f"  {ms:4d} ms out: {np.array2string(np.array(hs), precision=0)} Hz, mean {np.nanmean(hs):.1f} Hz,"
                  f" sd {np.nanstd(c):.1f} cents")
            axes[1].plot([ms] * len(hs), hs, "o")
        homes = [x for x in rests]
        axes[1].set_xlabel("run time out [ms]"); axes[1].set_ylabel("Hz at rest")
        axes[0].plot([x[1] for x in rests], [x[3] for x in rests], ".")
    axes[0].plot(r.log[:, 0], r.log[:, 2] * 20 + 300, lw=.5, color="orange", label="isense x20 + 300")
    axes[0].set_xlabel("s"); axes[0].set_ylabel("Hz"); axes[0].legend(); axes[0].grid(alpha=.3); axes[1].grid(alpha=.3)
    axes[0].set_title(r.path.name)
    figdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(figdir / (r.path.stem + ".png"), dpi=80); plt.close()
    return r, rests


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--fig", default="runs/real/motion/fig")
    args = ap.parse_args()
    for p in args.files:
        analyse(p, pathlib.Path(args.fig))


if __name__ == "__main__":
    main()
