"""PC: a few archive clips made ready for the play page (scripts/yamabiko_web.py): runs/archive_pack/ (not in git).

    python scripts/archive_pack.py                   # the default set: Silbo Gomero sentences and Japanese tit
    python scripts/archive_pack.py --list            # what is in the pack
    UNOQ=ssh bash scripts/uno_q_push.sh              # copies runs/archive_pack/ to the UNO Q with the rest

Fetch the archives first (scripts/archive_fetch.py).  Every item is a folder runs/archive_pack/<id>/ with
  source.wav       the archive clip (16 kHz mono): what the page plays as the example
  source_slow.wav  the clip at the speed the flute plays it (pitch kept)
  target.npz       the score for the flute (cents, voice, source_cents), in the profiles' range, intervals
                   scaled down when the clip is wider than the flute and time stretched by ``slow``
  meta.json        title, what it says (Silbo), credit, licence, the source page, slow, interval scale
The page plays a score as it is (it does not listen to the clip again), so any step can play it.
The sounds stay on the PC and the UNO Q: several xeno-canto recordings are CC BY-NC-ND (no edited copies
published), so the pack is not committed or put on the public pages.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import archive_demo as demo  # noqa: E402
from flute_rl.yamabiko.whistle import to_target  # noqa: E402

PACK = ROOT / "runs/archive_pack"
SILBO = ROOT / "runs/archive/silbo_gomero/sentences/sentences"
XC = ROOT / "runs/archive/xeno_canto/parus_minor"
LOW, HIGH = -10.0, 1200.0          # every profile's range (models/*/profile.json)

# (id, kind, file, start s, seconds, slow, title).  start None: the loudest-voiced window of that length.
DEFAULT = [
    ("silbo_01_001", "silbo", SILBO / "clips/01-sentence-001.wav", 0.0, 8.0, 2.0, "シルボ・ゴメロ「おじいさんは何歳？」"),
    ("silbo_03_024", "silbo", SILBO / "clips/03-sentence-024.wav", 0.0, 8.0, 2.0, "シルボ・ゴメロ「ラ・ゴメラ島」"),
    ("silbo_03_010", "silbo", SILBO / "clips/03-sentence-010.wav", 0.0, 8.0, 2.0,
     "シルボ・ゴメロ「アントニオという子だけがまだ来ていない」"),
    ("silbo_04_003", "silbo", SILBO / "clips/04-sentence-003.wav", 0.0, 8.0, 4.0,
     "シルボ・ゴメロ「今朝は何時に起きた？」（4 倍ゆっくり）"),
    ("tit_1147321", "bird", XC / "XC1147321.mp3", 0.0, 2.3, 4.0, "シジュウカラ：上がっていく短い声"),
    ("tit_285813", "bird", XC / "XC285813.mp3", None, 3.0, 4.0, "シジュウカラ：さえずり"),
    ("tit_285663", "bird", XC / "XC285663.mp3", None, 3.0, 4.0, "シジュウカラ：地鳴き"),
]
SILBO_CREDIT = ("Silbo Gomero Speech Corpus (OpenSLR SLR137), A. Jakubiak; recordings: Silbo Gomero Teaching "
                "Project", "https://creativecommons.org/licenses/by-nc-sa/4.0/", "https://openslr.org/137/")


def busiest(x, sr, kind, seconds):
    """Start (s) of the ``seconds`` window with the most pitched 10 ms frames."""
    raw, clean = demo.listen(x, sr, kind)
    v = np.isfinite(clean).astype(float)
    n = int(seconds / .01)
    if len(v) <= n:
        return 0.0
    counts = np.convolve(v, np.ones(n), mode="valid")
    return float(np.argmax(counts) * .01)


def credit_of(kind, path):
    if kind == "silbo":
        return SILBO_CREDIT
    manifest = json.loads((path.parent / "manifest.json").read_text("utf-8"))
    rec = next((r for r in manifest if f"XC{r['id']}" == path.stem), {})
    return (f"XC{rec.get('id', path.stem)} {rec.get('common_name', '')} ({rec.get('species', '')}), "
            f"{rec.get('recordist', '')}, {rec.get('place', '')}, {rec.get('date', '')}, xeno-canto",
            rec.get("licence", ""), rec.get("page", ""))


def make(item_id, kind, path, start, seconds, slow, title):
    if not path.exists():
        print(f"  skip {item_id}: {path.relative_to(ROOT)} is not there (scripts/archive_fetch.py)")
        return None
    x, sr = demo.read_audio(path)
    if start is None:
        start = busiest(x, sr, kind, seconds)
    x = x[int(start * sr):int((start + seconds) * sr)]
    x = x / max(np.abs(x).max(), 1e-9) * .9
    raw, clean = demo.listen(x, sr, kind)
    fitted, scale = demo.fit(clean, LOW, HIGH, "compress", slow)
    target, voice, report = to_target(fitted, LOW, HIGH)
    folder = PACK / item_id
    folder.mkdir(parents=True, exist_ok=True)
    y = demo.resample(x, sr)
    demo.write_wav(folder / "source.wav", y, demo.SR_OUT)
    demo.write_wav(folder / "source_slow.wav", np.clip(demo.stretch(y, slow), -1, 1), demo.SR_OUT)
    np.savez(folder / "target.npz", cents=target, voice=voice, source_cents=raw)
    credit, licence, page = credit_of(kind, path)
    meta = {"id": item_id, "kind": kind, "title": title, "says": demo.transcription(path) if kind == "silbo" else "",
            "credit": credit, "licence": licence, "page": page, "file": path.name, "start": round(start, 2),
            "seconds": round(len(x) / sr, 2), "slow": slow, "interval_scale": round(scale, 3),
            "play_seconds": round(len(target) * .01, 1), "voiced_steps": int(voice.sum())}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--clean", action="store_true", help="remove the pack first")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if not args.list:
        if args.clean and PACK.exists():
            shutil.rmtree(PACK)
        for item in DEFAULT:
            meta = make(*item)
            if meta:
                print(f"  {meta['id']:14s} {meta['seconds']:4.1f} s from {meta['start']:5.1f} s, x{meta['slow']:.0f} slower, "
                      f"intervals x{meta['interval_scale']:.2f} -> {meta['play_seconds']:.1f} s  {meta['says']}")
    for m in sorted(PACK.glob("*/meta.json")) if PACK.exists() else []:
        meta = json.loads(m.read_text("utf-8"))
        print(f"{meta['id']:14s} {meta['title']}  ({meta['credit'][:60]})")


if __name__ == "__main__":
    main()
