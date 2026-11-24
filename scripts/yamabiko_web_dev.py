"""PC: the web page against a fake MCU (no rig, no sound), to work on the page.

    python scripts/yamabiko_web_dev.py [--port 8080]      # then open http://127.0.0.1:8080/

The command server runs here on the fake link of tests/test_yamabiko_server.py (a 1 kHz tone at 16 kHz in
real time); ``POST /dev/press`` presses the fake board button (the page's EXEC button).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import threading
from http.server import ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
from flute_rl.yamabiko.profiles import load_profiles  # noqa: E402
from flute_rl.yamabiko.server import Server  # noqa: E402
from test_yamabiko_server import FakeLink  # noqa: E402
import yamabiko_web as web  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    link = FakeLink()
    srv = Server(link, port=0, max_pwm=1.0, lease=2.0, log=lambda s: print("server:", s, flush=True))
    threading.Thread(target=srv.serve, daemon=True).start()
    bench = web.Bench(srv.port)
    whistle = web.WhistleJob(bench, srv.port, load_profiles())
    whistle.watch_button()
    handler = web.make_handler(bench, whistle)

    class Dev(handler):
        def do_POST(self):
            if self.path == "/dev/press":
                link.buttons += 1
                return self._json(200, {"buttons": link.buttons})
            return super().do_POST()

    print(f"fake rig on {srv.port}, page on http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Dev).serve_forever()


if __name__ == "__main__":
    main()
