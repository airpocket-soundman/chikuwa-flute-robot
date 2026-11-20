"""UNO Q: watch the microphone and the A board's inputs, for a first power-up of the boards.

    python3 scripts/yamabiko_bench_watch.py [seconds]

Every 0.25 s it prints the level (RMS and peak of the last 20 ms / 250 ms), the strongest frequency, the
pitch the controller would hear (hw.py's 20 ms YIN), the EXEC button (presses so far, held), the fan's
speed and the status flags. The actuator is held at 0 and the valve closed. The whole 16 kHz stream is
saved to /tmp/bench_watch.raw (int16) for the PC. Needs mcu/yamabiko_fw and arduino-router stopped;
nothing else may have /dev/ttyHS1 open.
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flute_rl.yamabiko.hw import FLAG_BUTTON, FLAG_SERVO_SILENT, FLAG_TIMEOUT, Link, heard_pitch  # noqa: E402

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15
link = Link(); link.limit(0.0); link.start()
t0 = time.monotonic(); nxt = t0; first = None
try:
    while time.monotonic() - t0 < seconds:
        link.command(0.0, False)
        link.poll(0.01)
        if first is None and link.audio.base is not None:
            first = link.audio.base
        if time.monotonic() < nxt:
            continue
        nxt += 0.25
        cents, conf, db = heard_pitch(link.audio.latest(320))
        hz = 440 * 2 ** (cents / 1200) if np.isfinite(cents) else float("nan")
        x = link.audio.latest(4000)
        peak = 20 * np.log10(max(np.abs(x).max(), 1e-9))
        spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x))))
        fpk_hz = (np.argmax(spec[25:]) + 25) * 16000 / len(x)
        st = link.status
        if st is None:
            print("no status yet", flush=True)
            continue
        f = st.flags
        print(f"{time.monotonic()-t0:5.2f}s  level {db:6.1f} peak {peak:6.1f} dBFS  fft {fpk_hz:6.0f} Hz  "
              f"pitch {hz:7.1f} Hz conf {conf:.2f}  btn_count {st.buttons} {'HELD' if f & FLAG_BUTTON else '----'}  "
              f"fan {st.fan_rpm} rpm{' TIMEOUT' if f & FLAG_TIMEOUT else ''}"
              f"{' SERVO_SILENT' if f & FLAG_SERVO_SILENT else ''}  gaps {link.audio.gaps}", flush=True)
finally:
    link.command(0.0, False); link.stop()
    if first is not None:
        link.audio.span(first, link.audio.end).tofile("/tmp/bench_watch.raw")
    link.close()
