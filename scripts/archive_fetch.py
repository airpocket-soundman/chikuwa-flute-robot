"""PC: fetch the sound archives for the whistled-language and bird demos into runs/archive/ (not in git).

    python scripts/archive_fetch.py silbo                    # Silbo Gomero Speech Corpus (OpenSLR SLR137), ~0.3 GB
    python scripts/archive_fetch.py tit --pages 2            # Japanese tit (Parus minor) from xeno-canto, Japan, quality A
    python scripts/archive_fetch.py tit --query "Parus minor q:A type:song"

silbo  runs/archive/silbo_gomero/{words,sentences,fragments}/*.wav plus the transcriptions.  CC BY-NC-SA 4.0
       (A. Jakubiak, Silbo Gomero Teaching Project; cite Interspeech 2023, doi 10.21437/Interspeech.2023-989).
tit    runs/archive/xeno_canto/<species>/XC<id>.mp3 and manifest.json (recordist, place, date, type, licence of
       each recording).  The licences are per recording, mostly CC BY-NC-ND or BY-NC-SA: use them for analysis and
       demos with the credit, do not commit or publish edited copies of an ND one.  xeno-canto's API v3 needs a
       personal key, so this reads the public search page instead (one request per page, then one per file).

The audio is data, not code: nothing in these folders is run.  Skips files that are already there.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runs/archive"
SLR137_MIRRORS = ("https://openslr.trmal.net/resources/137/", "https://openslr.elda.org/resources/137/")
SLR137_FILES = ("README.txt", "words.zip", "sentences.zip", "fragments.zip")
AGENT = "chikuwa-flute-robot archive fetch (research, non-commercial)"


def fetch(url: str, path: pathlib.Path) -> bool:
    """Download ``url`` to ``path`` unless it is there; True when the file is there afterwards."""
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=120) as r, \
                open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except OSError as exc:
        print(f"  failed {url}: {exc}")
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


def silbo(args):
    folder = ARCHIVE / "silbo_gomero"
    for name in SLR137_FILES:
        if not any(fetch(m + name, folder / name) for m in SLR137_MIRRORS):
            raise SystemExit(f"could not fetch {name}")
        print(f"  {name}: {(folder / name).stat().st_size / 1e6:.1f} MB")
        if name.endswith(".zip"):
            part = folder / name[:-4]
            if not part.exists():
                with zipfile.ZipFile(folder / name) as z:
                    for member in z.namelist():               # no absolute paths or .. in the archive
                        if member.startswith(("/", "\\")) or ".." in pathlib.PurePath(member).parts:
                            raise SystemExit(f"unsafe path in {name}: {member}")
                    z.extractall(part)
            print(f"    {sum(1 for _ in part.rglob('*.wav'))} wav files in {part.relative_to(ROOT)}")


ROW = re.compile(r"<tr\s*>(.*?)</tr>", re.S)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
TAG = re.compile(r"<[^>]+>")


def text(cell: str) -> str:
    return html.unescape(TAG.sub("", cell)).strip()


def parse_page(page: str) -> list[dict]:
    """Rows of a xeno-canto search page: id, length, recordist, date, time, country, place, type, licence."""
    rows = []
    for row in ROW.findall(page):
        m = re.search(r"https://xeno-canto.org/(\d+)/download", row)
        if not m:
            continue
        cells = CELL.findall(row)
        lic = re.search(r'href="(https://creativecommons.org/licenses/[^"]+)"', row)
        sci = re.search(r"<span class='sci-name'>([^<]+)</span>", row)
        common = re.search(r"<span class=\"common-name\"><a[^>]*>([^<]+)</a>", row)
        rows.append({"id": int(m.group(1)), "species": sci.group(1).strip() if sci else "",
                     "common_name": common.group(1).strip() if common else "",
                     "length": text(cells[2]) if len(cells) > 2 else "",
                     "recordist": text(cells[3]) if len(cells) > 3 else "",
                     "date": text(cells[4]) if len(cells) > 4 else "",
                     "time": text(cells[5]) if len(cells) > 5 else "",
                     "country": text(cells[6]) if len(cells) > 6 else "",
                     "place": text(cells[7]) if len(cells) > 7 else "",
                     "type": text(cells[9]) if len(cells) > 9 else "",
                     "licence": lic.group(1) if lic else "",
                     "page": f"https://xeno-canto.org/{m.group(1)}",
                     "download": f"https://xeno-canto.org/{m.group(1)}/download"})
    return rows


def tit(args):
    folder = ARCHIVE / "xeno_canto" / args.folder
    manifest_path = folder / "manifest.json"
    manifest = {r["id"]: r for r in json.loads(manifest_path.read_text("utf-8"))} if manifest_path.exists() else {}
    for pg in range(1, args.pages + 1):
        url = "https://xeno-canto.org/explore?" + urllib.parse.urlencode({"query": args.query, "pg": pg})
        req = urllib.request.Request(url, headers={"User-Agent": AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            rows = parse_page(r.read().decode("utf-8", "replace"))
        print(f"page {pg}: {len(rows)} recordings")
        if not rows:
            break
        for row in rows[:max(0, args.limit - len(manifest))] if args.limit else rows:
            path = folder / f"XC{row['id']}.mp3"
            if fetch(row["download"], path):
                row["file"] = path.name
                manifest[row["id"]] = row
                print(f"  XC{row['id']} {row['type']:24.24s} {row['length']:>6s} {row['recordist']}")
            time.sleep(args.pause)
        if args.limit and len(manifest) >= args.limit:
            break
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(sorted(manifest.values(), key=lambda r: r["id"]), ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"{len(manifest)} recordings in {folder.relative_to(ROOT)} (manifest.json has each licence)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("silbo")
    t = sub.add_parser("tit")
    t.add_argument("--query", default="Parus minor q:A cnt:Japan", help="xeno-canto search")
    t.add_argument("--folder", default="parus_minor")
    t.add_argument("--pages", type=int, default=2, help="search pages to read (30 recordings each)")
    t.add_argument("--limit", type=int, default=0, help="stop after this many recordings (0: all on the pages)")
    t.add_argument("--pause", type=float, default=1.0, help="seconds between downloads")
    args = ap.parse_args()
    {"silbo": silbo, "tit": tit}[args.what](args)


if __name__ == "__main__":
    main()
