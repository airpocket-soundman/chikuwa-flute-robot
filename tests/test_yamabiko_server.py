"""The UNO Q command server and its PC client, against a fake MCU link (no hardware, no torch)."""
import pathlib
import sys
import threading
import time

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flute_rl.yamabiko.hw import SR, Audio, Status  # noqa: E402
from flute_rl.yamabiko.remote import Remote, RemoteError  # noqa: E402
from flute_rl.yamabiko.server import Server  # noqa: E402


class FakeLink:
    """Streams a 1 kHz tone at 16 kHz in real time and reports the last command as the firmware would."""

    def __init__(self):
        self.audio, self.texts, self.bad = Audio(60.0), [], 0
        self.status, self.sent = None, []
        self.pwm, self.valve, self.fan_state, self.max_pwm = 0, False, (False, 0.0), 1.0
        self.t0, self.made = time.monotonic(), 0
        self.buttons = 7          # the board button's press count

    def send(self, data):
        self.sent.append(data)
        if data == b"i":
            self.texts += ["INFO fw yamabiko_fw fake", "IO fake"]

    def start(self): self.sent.append(b"S")
    def stop(self): self.sent.append(b"X")
    def limit(self, m): self.max_pwm = m
    def fan(self, on, duty): self.fan_state = (on, duty)
    def valve_positions(self, *a): self.sent.append(("K",) + a)

    def command(self, pwm, valve, brake=False):
        self.pwm, self.valve = int(round(max(-self.max_pwm, min(self.max_pwm, pwm)) * 1000)), valve

    def poll(self, timeout):
        time.sleep(timeout)
        due = int((time.monotonic() - self.t0) * SR)
        if due > self.made:
            n = np.arange(self.made, due)
            self.audio.add(self.made, (8000 * np.sin(2 * np.pi * 1000 * n / SR)).astype(np.int16))
            self.made = due
        flags = (2 if self.valve else 0) | (4 if self.fan_state[0] else 0) | 16
        self.status = Status(int(1000 * (time.monotonic() - self.t0)), 0, 3, 5, 1200 if self.fan_state[0] else 0,
                             self.buttons, flags, self.pwm, -1, 0, time.monotonic())

    def drain(self, s): self.poll(s)


@pytest.fixture()
def server():
    link = FakeLink()
    srv = Server(link, port=0, max_pwm=.6, lease=.4, log=lambda s: None)
    thread = threading.Thread(target=srv.serve, daemon=True)
    thread.start()
    yield srv, link
    srv.running = False
    thread.join(2)
    srv.close()


def test_outputs_status_and_mic(server):
    srv, link = server
    with Remote(port=srv.port, forward=False) as rig:
        assert rig.info()[0].startswith("INFO fw yamabiko_fw")
        rig.actuator(.3); rig.fan(True, .5); rig.valve(True)
        time.sleep(.1)
        st = rig.status()
        assert st["applied_pwm"] == pytest.approx(.3) and st["valve_open"] and st["fan_switch"] and st["fan_rpm"] == 1200
        assert st["button_count"] == 7 and st["pitch_hz"] == pytest.approx(1000, abs=5)
        mic = rig.mic()
        assert mic["fft_peak_hz"] == pytest.approx(1000, abs=10) and mic["rms_db"] > -30
        with pytest.raises(RemoteError, match="cap"):
            rig.actuator(.9)
        rig.limit(.9); rig.actuator(.9)
        pcm, sr = rig.record(.5)
        assert sr == SR and pcm.size == SR // 2 and np.abs(pcm).max() > 7000
        with pytest.raises(RemoteError, match="unknown"):
            rig.call("nonsense")
    time.sleep(.1)                                   # closing the connection makes everything safe
    assert link.pwm == 0 and not link.valve and link.fan_state == (False, 0.0)


def test_lease_and_one_driver(server):
    srv, link = server
    a = Remote(port=srv.port, forward=False, heartbeat_s=60)        # no heartbeat in time
    b = Remote(port=srv.port, forward=False)
    try:
        a.actuator(.2); a.fan(True, .4)
        with pytest.raises(RemoteError, match="another connection"):
            b.actuator(.1)
        time.sleep(.8)                               # a's lease (0.4 s) runs out
        assert link.pwm == 0 and link.fan_state == (False, 0.0)
        b.actuator(.1)                               # now b may drive
        time.sleep(.05)
        assert link.pwm == 100
    finally:
        a.close(); b.close()


def test_audio_follows_a_restarted_stream():
    audio = Audio(5.0)
    audio.add(1_000_000, np.ones(80, np.int16))
    audio.add(1_000_080, np.ones(80, np.int16))
    audio.add(0, np.full(80, 2, np.int16))          # 'S' again: the firmware counts from 0
    audio.add(80, np.full(80, 3, np.int16))
    assert audio.base == 0 and audio.end == 160
    assert np.allclose(audio.latest(160) * 32768, np.r_[np.full(80, 2), np.full(80, 3)])
