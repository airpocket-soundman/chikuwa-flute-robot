"""UNO Q: the command server the PC drives over USB (see flute_rl/yamabiko/server.py).

    python3 scripts/yamabiko_server.py [--port 8765] [--max-pwm 0.6] [--lease 2]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.server import main  # noqa: E402

if __name__ == "__main__":
    main()
