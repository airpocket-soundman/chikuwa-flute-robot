"""Yamabiko No.1 on the real rig: listen to a whistle, then play it back with the trained network
(runs on the UNO Q's Linux side).

    sudo systemctl stop arduino-router arduino-router-serial     # the firmware owns /dev/ttyHS1
    python3 scripts/yamabiko_play.py --policy runs/yamabiko_gru.npz

Press the EXEC button (or Enter with --key) and whistle; the recording stops after --silence seconds
without a tone. The pitch track becomes the song (octaves folded into the flute's range, as in
targets.from_pitch_track), the plunger is homed and the network plays it. Its memory of the rig is kept
from song to song and only cleared when this program starts (power on), as in training.

--demo N plays N random songs instead of whistles (no button, no whistling), to try the rig.
--listen only records whistles and shows the songs they become (nothing moves; the microphone alone).
--auto starts recording at once instead of waiting for the button.
Each run is logged like yamabiko_collect.py, so what is played is also training data.

A network trained on a fitted rig (YAMABIKO_RIG=... yamabiko_train.py) must be played with the same
--rig: the dead reckoning inside the controller uses the nominal rig.
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import select
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from yamabiko_collect import add_rig_args, close_rig, open_rig  # noqa: E402


def wait_trigger(link, use_key: bool) -> None:
    """Until the EXEC button is pressed (or Enter with use_key). The MCU has stopped the plunger and
    shut the valve by now (no commands for 100 ms)."""
    if use_key:
        print("Enter to record a whistle")
    else:
        print("press EXEC to record a whistle")
    start = link.status.buttons if link.status else None
    while True:
        link.poll(0.02)
        if use_key and select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return
        st = link.status
        if not use_key and st is not None:
            if start is None:
                start = st.buttons
            elif st.buttons != start:
                return


def band_db(x) -> float:
    """Level of a window in the whistle band (400-5000 Hz) [dBFS]: breath on the microphone and handling
    noise (tens of Hz) do not count."""
    import numpy as np
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), 1.0 / 16000)
    band = (f >= 400.0) & (f <= 5000.0)
    power = 2.0 * np.sum(np.abs(X[band]) ** 2) / (len(x) * np.sum(w * w))
    return float(10.0 * np.log10(max(power, 1e-18)))


def noise_floor(link) -> float:
    """The quieter 20 % of the last second, in the whistle band, and never above -45 dBFS."""
    import numpy as np
    x = link.audio.latest(16000)
    levels = [band_db(x[i:i + 320]) for i in range(0, 16000, 320)]
    return min(-45.0, float(np.percentile(levels, 20)))


def record_whistle(link, max_s: float, silence_s: float, floor_db: float, wait_s: float = 5.0):
    """Audio from the first sound 10 dB over the floor (room and fan) in the whistle band until `silence_s`
    without one (at most `max_s`), as int16."""
    import numpy as np

    from flute_rl.yamabiko.hw import SR
    print("whistle now")
    t0 = time.monotonic()
    start = end = None
    quiet_since = None
    while True:
        link.poll(0.01)
        db = band_db(link.audio.latest(320))
        now = time.monotonic()
        loud = db > floor_db + 10.0
        if start is None:
            if loud:
                start = link.audio.end - 3200                    # keep 0.2 s before the first tone
            elif now - t0 > wait_s:
                return None
            continue
        if loud:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        if (quiet_since is not None and now - quiet_since > silence_s) or link.audio.end - start > max_s * SR:
            end = link.audio.end
            break
    return link.audio.span(start, end)


def whistle_to_song(audio, floor_db: float = -70.0, bridge: int = 3):
    """Pitch track (10 ms hop) -> target of the flute, leading and trailing silence cut off.

    Frames are voiced down to 10 dB over the noise floor (a whistle from 30 cm is quiet), and gaps of up
    to `bridge` frames inside a phrase are filled from both sides: a frame the estimator missed is not a
    rest for the valve."""
    import numpy as np

    from flute_rl.pitch import pitch_track
    from flute_rl.targets import from_pitch_track
    from flute_rl.yamabiko.hw import SR
    gate = 10.0 ** ((floor_db + 10.0) / 20.0)
    _, f0, _ = pitch_track(audio.astype(float) / 32768.0, SR, frame=512, hop=160, fmin=400.0, fmax=4000.0,
                           rms_gate=gate)
    voiced = np.flatnonzero(np.isfinite(f0))
    if voiced.size < 20:                                          # less than 0.2 s of tone
        return None
    f0 = f0[voiced[0]:voiced[-1] + 1]
    on = np.isfinite(f0)
    i = 0
    while i < len(f0):
        if on[i]:
            i += 1
            continue
        j = i
        while j < len(f0) and not on[j]:
            j += 1
        if j - i <= bridge and i > 0 and j < len(f0):              # geometric mean of the neighbours
            f0[i:j] = np.sqrt(f0[i - 1] * f0[j])
        i = j
    return from_pitch_track(f0, 0.01)


def describe(audio, target) -> None:
    """What was heard and what the flute will play."""
    import numpy as np

    from flute_rl.pitch import pitch_track
    from flute_rl.yamabiko.hw import SR
    _, f0, _ = pitch_track(audio.astype(float) / 32768.0, SR, frame=512, hop=160, fmin=400.0, fmax=4000.0)
    v = f0[np.isfinite(f0)]
    hz = 440.0 * 2.0 ** (target[np.isfinite(target)] / 1200.0)
    notes = np.isfinite(target)
    n_notes = int(np.sum(notes[1:] & ~notes[:-1]) + notes[0])
    print(f"  whistle {len(audio) / SR:.1f} s, pitch {np.percentile(v, 5):.0f}-{np.percentile(v, 95):.0f} Hz "
          f"(median {np.median(v):.0f})")
    print(f"  song {len(target) * 0.01:.1f} s, {n_notes} phrase(s), {notes.mean() * 100:.0f} % sounding, "
          f"flute {hz.min():.0f}-{hz.max():.0f} Hz (shifted {1200 * np.log2(np.median(hz) / np.median(v)):+.0f} cents)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_rig_args(ap)
    ap.add_argument("--policy", default="runs/yamabiko_gru.npz")
    ap.add_argument("--songs", type=int, default=0, help="stop after this many songs (0 = until Ctrl-C)")
    ap.add_argument("--demo", type=int, default=0, help="play this many random songs instead of whistles")
    ap.add_argument("--key", action="store_true", help="Enter instead of the EXEC button")
    ap.add_argument("--auto", action="store_true", help="record at once, no button")
    ap.add_argument("--listen", action="store_true", help="record whistles and show the songs, do not play")
    ap.add_argument("--max-whistle", type=float, default=8.0, help="longest whistle [s]")
    ap.add_argument("--wait", type=float, default=5.0, help="give up if no whistle starts within this long [s]")
    ap.add_argument("--silence", type=float, default=1.0, help="the whistle ends after this long without a tone [s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.listen:
        args.fan = 0.0                                            # nothing moves, nothing blows
    if args.rig:
        os.environ["YAMABIKO_RIG"] = str(pathlib.Path(args.rig).resolve())

    import numpy as np

    from flute_rl.sim import DT
    from flute_rl.targets import make_target, sample_level
    from flute_rl.yamabiko import GRUPolicy, make_schedule
    from flute_rl.yamabiko.hw import SessionLog, hold, play_song
    from flute_rl.yamabiko.rig import NOMINAL, RIG_FILE

    d = np.load(args.policy)
    ctrl = GRUPolicy(d["theta"][None, :], int(d["hidden"]), carry=True)
    rng = np.random.default_rng(args.seed)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(args.out or f"runs/real/play_{stamp}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = SessionLog({"kind": "play", "date": stamp, "args": vars(args), "controller": "gru",
                      "policy": str(args.policy), "rig_file": RIG_FILE, "nominal": NOMINAL})

    link, rig = open_rig(args)
    try:
        hold(rig, log, args.settle, song=-1)
        floor_db = max(-70.0, noise_floor(link))                  # room + fan
        print(f"noise floor {floor_db:.1f} dBFS (400-5000 Hz)")
        k = 0
        while args.songs <= 0 or k < args.songs:
            if args.demo:
                if k >= args.demo:
                    break
                target = make_target(rng, sample_level(rng, 0.8))
            else:
                if not args.auto:
                    wait_trigger(link, args.key)
                audio = record_whistle(link, args.max_whistle, args.silence, floor_db, args.wait)
                target = whistle_to_song(audio, floor_db) if audio is not None else None
                if target is None:
                    print("no whistle heard")
                    floor_db = max(-70.0, noise_floor(link))
                    print(f"noise floor {floor_db:.1f} dBFS (400-5000 Hz)")
                    if args.auto and args.songs > 0:
                        k += 1
                    continue
                log.whistles.append(audio)
                log.meta.setdefault("songs", []).append([None if not np.isfinite(v) else round(float(v), 1)
                                                         for v in target])
                describe(audio, target)
                if args.listen:
                    log.save(out, rig)
                    k += 1
                    continue
            sched = make_schedule([[target]])
            print(f"song {k}: {np.isfinite(target).sum() * DT:.1f} s of notes")
            rig.resync()
            play_song(rig, ctrl, sched, k, log, first=(k == 0))
            rig.safe()
            heard = np.asarray(rig.log["heard"][-sched.T:])
            note = np.isfinite(sched.target[0])
            both = note & np.isfinite(heard)
            err = np.abs(heard[both] - sched.target[0][both])
            print(f"  heard on {both.sum()}/{note.sum()} note steps, median |error| "
                  f"{np.median(err) if err.size else float('nan'):.0f} cents, late steps {rig.late_steps}")
            log.save(out, rig)
            k += 1
    finally:
        close_rig(link, rig)
        log.save(out, rig)
        print("saved", out)


if __name__ == "__main__":
    main()
