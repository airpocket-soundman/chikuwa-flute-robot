"""PC: play archive clips (runs/archive_pack/, on the UNO Q) on the rig through the play page, with practice.

    python scripts/yamabiko_archive_rig.py --step step7_practice --clips silbo_01_001 tit_1147321 --plays 3

For every clip: the page's "play with the flute" (archive_play), then ``--plays`` - 1 times "again" (the board
button's double press: the song memory is kept), waiting for each play to end.  Prints the relative error of
every play (the page's practice list) and saves the page's plays (heard pitch per play, in the score's key) to
runs/real/archive/<stamp>_<step>_<clip>.json.  The flute sounds: a calibration (about 10 s) the first time,
then every play.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def call(host, path, body=None):
    req = urllib.request.Request(f"http://{host}:8080/api/whistle/{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def wait_idle(host, timeout=300):
    until = time.time() + timeout
    time.sleep(1.5)
    while time.time() < until:
        s = call(host, "status")
        if not s["busy"]:
            return s
        time.sleep(1.0)
    raise TimeoutError("the page stayed busy")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.50.160")
    ap.add_argument("--step", default="step7_practice")
    ap.add_argument("--clips", nargs="+", default=["silbo_01_001", "tit_1147321"])
    ap.add_argument("--plays", type=int, default=3)
    args = ap.parse_args()
    out = ROOT / "runs/real/archive"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    s = call(args.host, "status")
    if s["busy"]:
        raise SystemExit("the page is busy")
    if s["step"] != args.step:
        call(args.host, "step", {"step": args.step})
    results = {}
    for clip in args.clips:
        call(args.host, "archive_play", {"id": clip})
        s = wait_idle(args.host)
        for _ in range(args.plays - 1):
            if s["phase"] == "error":
                break
            call(args.host, "replay", {})
            s = wait_idle(args.host)
        plays = call(args.host, "plays")
        results[clip] = {"practice": s.get("practice"), "message": s.get("message"), "phase": s.get("phase")}
        (out / f"{stamp}_{args.step}_{clip}.json").write_text(json.dumps({**plays, **results[clip]}), encoding="utf-8")
        print(f"{clip}: relative error per play {s.get('practice')} ({s.get('phase')}: {s.get('message')})", flush=True)
    s = call(args.host, "status")
    print("after:", {k: s.get(k) for k in ("phase", "busy")})


if __name__ == "__main__":
    main()
