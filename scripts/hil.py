"""Hardware-in-the-loop: the simulator runs here, the controller runs on the MCU (roadmap stage 3).

The simulator plays the rig; every 10 ms it sends the MCU what the rig heard
(the delayed pitch error) and the servo read-back, and applies the PWM and
angle the MCU answers. Between takes this side identifies the rig from the
recording (flute_rl.adapt.fit_rig, as on the UNO Q's Linux side) and sends the
next take's targets, the previous take's errors and the fitted rig. The frame
format is in mcu/hil_protocol.h.

Links:
  --link process:<exe>      mcu/hil_host.c compiled on the PC (no hardware; for testing)
  --link serial:<port>      the MCU over a serial port (needs pyserial), e.g. serial:COM5

    python scripts/hil.py --link process:runs/hil_host.exe --episodes 3 --harsh 1.0
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flute_rl import FluteEnv, FluteParams  # noqa: E402
from flute_rl.adapt import fit_rig  # noqa: E402


def frame(kind: bytes, payload: bytes) -> bytes:
    return b"\xa5" + kind + struct.pack("<H", len(payload)) + payload + bytes([sum(payload) & 0xFF])


class ProcessLink:
    def __init__(self, exe: str, history: int, takes: int):
        self.p = subprocess.Popen([exe, str(history), str(takes)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def write(self, data: bytes) -> None:
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def read(self, n: int) -> bytes:
        return self.p.stdout.read(n)

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


class SerialLink:
    def __init__(self, port: str, baud: int):
        import serial  # pyserial

        self.s = serial.Serial(port, baud, timeout=1.0)
        time.sleep(2.0)  # boards may reset when the port opens

    def write(self, data: bytes) -> None:
        self.s.write(data)

    def read(self, n: int) -> bytes:
        return self.s.read(n)

    def close(self) -> None:
        self.s.close()


def read_frame(link) -> tuple[bytes, bytes]:
    while link.read(1) != b"\xa5":
        pass
    kind = link.read(1)
    (n,) = struct.unpack("<H", link.read(2))
    payload = link.read(n)
    if (sum(payload) & 0xFF) != link.read(1)[0]:
        raise IOError("checksum mismatch from the MCU")
    return kind, payload


def play_episode(env: FluteEnv, link, seed: int, target=None) -> dict:
    """One episode (env.takes takes) with the controller on the other side of `link`."""
    obs, _ = env.reset(seed=seed, options={"target": target} if target is not None else None)
    n = len(env.target)
    z, rms = np.zeros(4), float("nan")
    history, done, actions = [], False, []
    while not done:
        if env.t == 0:  # take setup
            has_z = 0
            if env.take > 0:
                history.append((np.array(pwm_log), env.target + env.prev_err))
                if len(history) == 1:  # identify once, after the first take (as AdaptivePolicy does)
                    z, rms = fit_rig(history, FluteParams(), z0=z)
                    has_z = 1
            head = struct.pack("<BBB4ff fH".replace(" ", ""), env.take, int(np.isfinite(rms)), has_z, *z.astype(np.float32),
                               env.angle_comp, env.params.angle_range_deg, n)
            body = np.concatenate([env.target, env.prev_err[:n]]).astype("<f4").tobytes()
            link.write(frame(b"T", head + body))
            kind, _ = read_frame(link)
            if kind != b"K":
                raise IOError("take setup not acknowledged")
            pwm_log = []
        fb = env._fb_seen[0]
        link.write(frame(b"S", struct.pack("<ff", fb, env.state.theta_readback)))
        kind, payload = read_frame(link)
        a = np.array(struct.unpack("<ff", payload), dtype=float)
        pwm_log.append(a[0])
        actions.append(a)
        obs, _, done, _, _ = env.step(a)
    return {"actions": np.array(actions)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", required=True, help="process:<exe> or serial:<port>")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--history", type=int, default=0, help="history length the MCU's network was built with")
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--harsh", type=float, default=0.0)
    ap.add_argument("--seed0", type=int, default=5_000_000)
    args = ap.parse_args()

    kind, where = args.link.split(":", 1)
    link = ProcessLink(where, args.history, args.takes) if kind == "process" else SerialLink(where, args.baud)
    env = FluteEnv(progress=0.8, takes=args.takes, feedback=True, harsh=args.harsh)
    try:
        for i in range(args.episodes):
            t0 = time.time()
            play_episode(env, link, args.seed0 + i)
            print(f"episode {i + 1}: {len(env.target) * env.takes} steps in {time.time() - t0:.1f}s")
    finally:
        link.close()


if __name__ == "__main__":
    main()
