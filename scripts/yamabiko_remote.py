"""PC: drive the real rig from a terminal through the UNO Q's command server (USB, adb forward).

    python scripts/yamabiko_remote.py status
    python scripts/yamabiko_remote.py watch --seconds 10          # status twice a second
    python scripts/yamabiko_remote.py actuator 0.3 --hold 2       # then back to 0
    python scripts/yamabiko_remote.py fan on --duty 0.5 --hold 5
    python scripts/yamabiko_remote.py valve open --hold 3
    python scripts/yamabiko_remote.py mic --watch 10              # level and pitch, 4 times a second
    python scripts/yamabiko_remote.py record 3 --out whistle.wav
    python scripts/yamabiko_remote.py calibrate --max-pwm 0.6 --sweep 0.5 --pull runs/real/
    python scripts/yamabiko_remote.py info | limit 0.8 | stop

Outputs set here go back to safe when the command ends (the connection closes).  The server must be running
on the UNO Q: python3 scripts/yamabiko_server.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import wave

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.remote import REMOTE_DIR, Remote, RemoteError, pull, push  # noqa: E402


def show(obj):
    print(json.dumps(obj, ensure_ascii=False))


def hold(rig, seconds, every=.5):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        st = rig.status()
        print(f"  applied pwm {st.get('applied_pwm', 0):+.2f}  isense {st.get('isense_mv')} mV  fan {st.get('fan_rpm')} rpm"
              f"  valve {'open' if st.get('valve_open') else 'shut'}", flush=True)
        time.sleep(min(every, max(0.0, until - time.monotonic())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status"); sub.add_parser("info"); sub.add_parser("stop")
    p = sub.add_parser("watch"); p.add_argument("--seconds", type=float, default=10)
    p = sub.add_parser("actuator"); p.add_argument("pwm", type=float); p.add_argument("--hold", type=float, default=1.0)
    p.add_argument("--brake", action="store_true")
    p = sub.add_parser("limit"); p.add_argument("max_pwm", type=float)
    p = sub.add_parser("fan"); p.add_argument("state", choices=["on", "off"]); p.add_argument("--duty", type=float, default=.7)
    p.add_argument("--hold", type=float, default=3.0)
    p = sub.add_parser("valve"); p.add_argument("state", choices=["open", "closed"]); p.add_argument("--hold", type=float, default=3.0)
    p = sub.add_parser("mic"); p.add_argument("--watch", type=float, default=0.0)
    p = sub.add_parser("record"); p.add_argument("seconds", type=float); p.add_argument("--out", default="record.wav")
    p = sub.add_parser("calibrate"); p.add_argument("--max-pwm", type=float, default=.6); p.add_argument("--sweep", type=float, default=1.0)
    p.add_argument("--fan", type=float, default=.7); p.add_argument("--flip", action="store_true")
    p.add_argument("--out", default=f"{REMOTE_DIR}/runs/real/calibration.npz"); p.add_argument("--pull", default=None, help="local folder")
    p = sub.add_parser("play"); p.add_argument("runtime"); p.add_argument("target", help="local .npz files, pushed first")
    p.add_argument("--repeats", type=int, default=3); p.add_argument("--fan", type=float, default=.7); p.add_argument("--flip", action="store_true")
    p.add_argument("--out", default=f"{REMOTE_DIR}/runs/real/play.npz"); p.add_argument("--pull", default=None, help="local folder")
    args = ap.parse_args()

    with Remote(port=args.port) as rig:
        if args.cmd in ("status", "info", "stop"):
            show(getattr(rig, args.cmd)())
        elif args.cmd == "watch":
            until = time.monotonic() + args.seconds
            while time.monotonic() < until:
                show(rig.status()); time.sleep(.5)
        elif args.cmd == "actuator":
            show(rig.actuator(args.pwm, args.brake)); hold(rig, args.hold)
        elif args.cmd == "limit":
            show(rig.limit(args.max_pwm))
        elif args.cmd == "fan":
            show(rig.fan(args.state == "on", args.duty)); hold(rig, args.hold)
        elif args.cmd == "valve":
            show(rig.valve(args.state == "open")); hold(rig, args.hold)
        elif args.cmd == "mic":
            until = time.monotonic() + args.watch
            while True:
                show(rig.mic())
                if time.monotonic() >= until:
                    break
                time.sleep(.25)
        elif args.cmd == "record":
            pcm, sr = rig.record(args.seconds)
            with wave.open(args.out, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
            peak = int(abs(pcm.astype(int)).max()) if pcm.size else 0
            print(f"saved {args.out}: {pcm.size / sr:.2f} s, peak {peak} of 32768")
        elif args.cmd == "calibrate":
            show(rig.calibrate(out=args.out, max_pwm=args.max_pwm, sweep=args.sweep, fan=args.fan, flip=args.flip))
            if args.pull:
                pull(args.out, args.pull); print(f"pulled to {args.pull}")
        elif args.cmd == "play":
            remote = {}
            for name in ("runtime", "target"):
                local = pathlib.Path(getattr(args, name))
                remote[name] = f"{REMOTE_DIR}/runs/real/{local.name}"
                push(local, remote[name])
            show(rig.play(remote["runtime"], remote["target"], out=args.out, repeats=args.repeats, fan=args.fan,
                          flip=args.flip))
            if args.pull:
                pull(args.out, args.pull); print(f"pulled to {args.pull}")


if __name__ == "__main__":
    try:
        main()
    except RemoteError as exc:
        sys.exit(f"error from the rig: {exc}")
    except ConnectionRefusedError:
        sys.exit("no server: start it on the UNO Q with  python3 scripts/yamabiko_server.py")
