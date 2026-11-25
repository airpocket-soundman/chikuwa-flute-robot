"""PC: whistle to the rig's microphone, and the rig plays the melody back (a relative copy).

    python scripts/yamabiko_whistle_copy.py --seconds 5                  # record, analyse, play (newest step)
    python scripts/yamabiko_whistle_copy.py --step step1                 # a given step (models/<step>/)
    python scripts/yamabiko_whistle_copy.py --seconds 5 --dry-run        # record and analyse only
    python scripts/yamabiko_whistle_copy.py --wav take.wav --dry-run     # a recording instead of the mic
    python scripts/yamabiko_whistle_copy.py --target runs/archive_demo/<run>/target.npz   # a score made elsewhere

The ear is YIN (flute_rl/yamabiko/whistle.py); the hand is a performer runtime already on the UNO Q (the
network-only NNTwinPerformer by default).  The whistle is recorded with the fan off and the valve shut,
moved into the rig's range with its intervals kept, and played with ``relative`` so it follows the pitch the
flute sounds today.  Writes runs/real/whistle/<stamp>_*.{wav,npz,png}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.profiles import latest, load_profiles  # noqa: E402
from flute_rl.yamabiko.remote import Remote  # noqa: E402
from flute_rl.yamabiko.whistle import clean_track, to_target, whistle_track  # noqa: E402

REMOTE_DIR = "/home/arduino/yamabiko/runs/real/whistle/"


def read_wav(path):
    with wave.open(str(path)) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), "<i2").reshape(-1, ch)[:, 0]
    return x, sr


def write_wav(path, pcm, sr):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(np.asarray(pcm, "<i2").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.50.160")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--wav", default=None, help="analyse this recording instead of recording the mic")
    ap.add_argument("--target", default=None,
                    help="play this score (cents, voice) instead of listening, e.g. from scripts/archive_demo.py")
    ap.add_argument("--step", default=None, help="models/<step>/profile.json to run (default: the newest)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--quantize", action="store_true", help="snap the copy to semitones of its first note")
    ap.add_argument("--dry-run", action="store_true", help="record and analyse, do not play")
    args = ap.parse_args()
    profiles = load_profiles()
    profile = profiles[args.step or latest(profiles)]
    remote_runtime = f"/home/arduino/yamabiko/models/{profile['name']}/{profile['runtime']}"
    remote_valve = (f"/home/arduino/yamabiko/models/{profile['name']}/{profile['valve_runtime']}"
                    if profile.get("valve_runtime") else None)
    print(f"{profile['name']}: {profile.get('title', '')}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path("runs/real/whistle"); out.mkdir(parents=True, exist_ok=True)
    rig = None
    if args.target:
        given = np.load(args.target)
        target, voice = given["cents"].astype(np.float32), given["voice"].astype(bool)
        raw = cents = given["source_cents"] if "source_cents" in given.files else np.where(voice, target, np.nan)
        report = {"target": args.target, "steps": int(len(target))}
        pcm, sr = np.zeros(1600, np.int16), 16000
    elif args.wav:
        pcm, sr = read_wav(args.wav)
    else:
        rig = Remote(args.host, forward=False)
        rig.stop(force=True)                           # fan off, valve shut: only the whistle in the mic
        for k in (3, 2, 1):
            print(f"{k} ...", flush=True); time.sleep(1)
        print(f"whistle now ({args.seconds:.0f} s)", flush=True)
        pcm, sr = rig.record(args.seconds)
        print("got it", flush=True)
    if not args.target:
        write_wav(out / f"{stamp}_whistle.wav", pcm, sr)
        raw = whistle_track(pcm, sr)
        cents = clean_track(raw, **profile.get("whistle", {}))
        try:
            target, voice, report = to_target(cents, profile["low_cents"], profile["high_cents"],
                                              quantize=args.quantize)
        except ValueError as exc:
            level = 20 * np.log10(max(float(np.sqrt(np.mean((pcm / 32768.0) ** 2))), 1e-9))
            raise SystemExit(f"{exc}: the recording peaks at {level:.0f} dBFS; whistle closer to the microphone")
    print(json.dumps(report))
    np.savez(out / f"{stamp}_target.npz", cents=target, voice=voice, whistle_cents=raw)

    played = None
    if not args.dry_run:
        rig = rig or Remote(args.host, forward=False)
        remote_target = REMOTE_DIR + f"{stamp}_target.npz"
        subprocess.run(["ssh", "unoq", f"mkdir -p {REMOTE_DIR}"], check=True)
        subprocess.run(["scp", "-q", str(out / f"{stamp}_target.npz"), "unoq:" + remote_target], check=True)
        rig.valve_positions(*profile["valve_us"], move_ms=20)
        tuning = rig.tune(runtime=remote_runtime, fan=profile["fan"])
        print(f"today's flute: home {440 * 2 ** (tuning['home_cents'] / 1200):.1f} Hz, shift {tuning['shift_cents']:+.1f} cent")
        result = rig.play(remote_runtime, remote_target, out=REMOTE_DIR + f"{stamp}_play.npz", repeats=args.repeats,
                          fan=profile["fan"], shift_cents=tuning["shift_cents"], ear=profile.get("ear", "yin"),
                          valve_runtime=remote_valve)
        print(json.dumps({k: v for k, v in result.items() if k != "out"}))
        subprocess.run(["scp", "-q", "unoq:" + REMOTE_DIR + f"{stamp}_play.npz", str(out)], check=True)
        played = np.load(out / f"{stamp}_play.npz")
    if rig is not None:
        s = rig.status()
        print("after:", {k: s[k] for k in ("fan_switch", "valve_open", "applied_pwm")})
        rig.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = 2 + (played["count"] if played is not None else 0)
    fig, axes = plt.subplots(int(rows), 1, figsize=(13, 2.8 * rows), squeeze=False)
    t = np.arange(len(raw)) * .01
    axes[0, 0].plot(t, raw, ".", ms=2, color="gray", label="whistle (YIN)")
    axes[0, 0].plot(t, cents, lw=1.5, label="cleaned"); axes[0, 0].legend(); axes[0, 0].set_ylabel("cent")
    axes[0, 0].set_title("whistle")
    tt = np.arange(len(target)) * .01
    axes[1, 0].plot(tt, np.where(voice, target, np.nan), "k", lw=2)
    axes[1, 0].set_title(f"target in the rig's range (shift {report.get('shift_cents', 0):+.0f} cent)")
    if played is not None:
        for p in range(int(played["count"])):
            h, v, tc, tv = (played[f"{p}/{k}"][0] for k in ("heard", "valid", "target_cents", "target_voice"))
            ax = axes[2 + p, 0]
            ax.plot(tt[:len(tc)], np.where(tv, tc, np.nan), "k", lw=2, label="target")
            ax.plot(tt[:len(h)][v], h[v], ".", ms=3, label="played")
            m = v & tv
            key = np.median((h - tc)[m]) if m.any() else 0.0
            ax.set_title(f"play {p + 1}: key {key:+.0f} cent, relative MAE {np.abs(h - tc - key)[m].mean():.0f} cent")
            ax.legend()
    for ax in axes[:, 0]:
        ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out / f"{stamp}.png", dpi=70)
    print(f"saved {out / stamp}_*")


if __name__ == "__main__":
    main()
