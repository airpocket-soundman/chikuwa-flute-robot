import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))

from flute_rl.yamabiko import panel  # noqa: E402
from flute_rl.yamabiko.hw import Link  # noqa: E402
from flute_rl.yamabiko.server import Server  # noqa: E402
from test_yamabiko_server import FakeLink  # noqa: E402

STATUS = {
    "phase": "done", "busy": False, "has_song": True, "collect": False, "step": "step7c",
    "steps": [{"name": "classic"}, {"name": "step7c"}, {"name": "step8"}],
    "message": "できました（練習 2 回目）。2 回押すともう一度、1 回押すと新しく録音します",
    "practice": [61.4, None, 38.0], "tuning": {"home_cents": 105.2, "shift_cents": -12.0},
    "target": [100.0] * 300 + [500.0] * 300, "voice": [False] * 50 + [True] * 550,
    "archive": "tit_285813",
}
ARCHIVE = [{"id": "silbo_01_001", "title": "シルボ・ゴメロ「おじいさんは何歳？」"}, {"id": "tit_285813", "title": "シジュウカラ：さえずり"}]


def test_page_round_trip():
    heard = [None] * 60 + [110.0] * 540
    page = panel.encode(STATUS, ARCHIVE, heard)
    assert len(page) <= panel.MAX_BYTES
    d = panel.decode(page)
    assert d["phase"] == "done" and d["steps"] == ["classic", "step7c", "step8"] and d["step"] == 1
    assert d["archive"][0].startswith("シルボ") and d["archive_playing"] == 1
    assert d["practice"] == [61.4, None, 38.0]
    assert d["home_cents"] == 105 and d["shift_cents"] == -12
    assert d["play_s"] == 6.0                                          # 600 steps of 10 ms
    assert d["flags"] & panel.FLAG_HAS_SONG and d["flags"] & panel.FLAG_ARCHIVE and not d["flags"] & panel.FLAG_BUSY
    assert d["stride"] == 3 and len(d["target"]) == 200
    assert d["target"][0] is None and d["target"][-1] == 500           # rests are NO_VALUE
    assert d["heard"][10] is None and d["heard"][-1] == 110


def test_message_cut_on_a_character():
    page = panel.encode(dict(STATUS, message="あ" * 200), [], None)
    assert panel.decode(page)["message"] == "あ" * 50                   # 150 bytes = 50 characters


def test_empty_status():
    d = panel.decode(panel.encode({"phase": "idle", "steps": []}))
    assert d["phase"] == "idle" and d["target"] == [] and d["home_cents"] is None


def test_link_sends_the_page_in_pieces():
    sent = []
    link = Link.__new__(Link)
    link.send = sent.append
    page = bytes(range(70))
    link.panel(page, 5)
    assert [s[:1] for s in sent] == [b"U", b"U", b"U", b"V"]
    assert all(len(s) == 1 + 2 + 32 for s in sent[:3])
    assert struct.unpack("<H", sent[2][1:3])[0] == 64 and sent[2][3:9] == bytes(range(64, 70))
    assert struct.unpack("<HH", sent[3][1:]) == (70, 5)


def test_server_panel_command():
    link = FakeLink()
    srv = Server(link, port=0)
    try:
        link.panel_events.append(bytes([panel.EV_STEP, 2]))
        out = srv.cmd_panel(None, {"page_b64": "AAEC", "version": 9})
        assert out == {"events": [[panel.EV_STEP, 2]]} and link.panel_pages == [(b"\x00\x01\x02", 9)]
        assert srv.cmd_panel(None, {}) == {"events": []}
    finally:
        srv.close()


class FakeWhistle:
    def __init__(self):
        self.profiles = {"classic": {}, "step7c": {}}
        self.did, self.state, self.plays = [], {}, []

    def choose(self, name): self.did.append(("step", name))
    def calibrate(self): self.did.append(("calibrate",))
    def archive_list(self): return [{"id": "a"}, {"id": "b"}]
    def archive_play(self, item_id): self.did.append(("archive", item_id))
    def _set(self, **kw): self.state.update(kw)


def test_bridge_acts_on_events():
    import yamabiko_web as web
    w = FakeWhistle()
    bridge = web.PanelBridge(w, 0)
    bridge._act([[panel.EV_STEP, 1], [panel.EV_CALIBRATE], [panel.EV_ARCHIVE, 1], [panel.EV_ARCHIVE, 9], []])
    assert w.did == [("step", "step7c"), ("calibrate",), ("archive", "b")]
