"""Autoroute a board with Freerouting: export Specctra DSN, route headless, import the SES, refill zones.

    "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" route.py BOARD.kicad_pcb FREEROUTING.jar [passes]

Freerouting is not part of the repository (https://github.com/freerouting/freerouting, v2.4.1 was used).
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pcbnew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sexpr import dump, parse  # noqa: E402


def stripped_copy(pcb: Path) -> Path:
    """Copy of the board without tracks, vias and zones. Removing them through pcbnew breaks its
    SWIG wrappers for the rest of the process, so they are dropped from the file instead."""
    tree = parse(pcb.read_text(encoding="utf-8"))
    tree[:] = [x for x in tree if not (isinstance(x, list) and x and x[0] in ("segment", "via", "arc", "zone"))]
    tmp = pcb.with_name("_unrouted.kicad_pcb")
    tmp.write_text(dump(tree) + "\n", encoding="utf-8")
    shutil.copy(pcb.with_suffix(".kicad_pro"), tmp.with_suffix(".kicad_pro"))  # net classes live here
    return tmp


def make_gnd_pours(board):
    """GND pours on both copper layers over the whole board (not yet added to it)."""
    bb = board.GetBoardEdgesBoundingBox()
    gnd = board.FindNet("GND")
    zones = []
    for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
        z = pcbnew.ZONE(board)
        z.SetLayer(layer)
        z.SetNet(gnd)
        ol = z.Outline()
        ol.NewOutline()
        for x, y in [(bb.GetLeft(), bb.GetTop()), (bb.GetRight(), bb.GetTop()),
                     (bb.GetRight(), bb.GetBottom()), (bb.GetLeft(), bb.GetBottom())]:
            ol.Append(x, y)
        z.SetLocalClearance(pcbnew.FromMM(0.3))
        z.SetMinThickness(pcbnew.FromMM(0.25))
        z.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
        z.SetThermalReliefGap(pcbnew.FromMM(0.4))
        z.SetThermalReliefSpokeWidth(pcbnew.FromMM(0.5))
        zones.append(z)
    return zones


def main(pcb: str, jar: str, passes: str = "100"):
    pcb = Path(pcb).resolve()
    dsn, ses = pcb.with_suffix(".dsn"), pcb.with_suffix(".ses")
    clean = stripped_copy(pcb)  # GND is routed with tracks too; the pours are added back afterwards
    board = pcbnew.LoadBoard(str(clean))
    if not pcbnew.ExportSpecctraDSN(board, str(dsn)):
        raise SystemExit("DSN export failed")
    # Freerouting is deterministic for a given setup; different thread counts search in a different
    # order, so try a few and keep the best. Its log goes to a temporary folder.
    best = None
    with tempfile.TemporaryDirectory() as logdir:
        for threads in (1, 2, 4, 8, 3, 6):
            out = subprocess.run(["java", "-jar", jar, "-de", str(dsn), "-do", str(ses), "-mp", passes,
                                  "-mt", str(threads), "--gui.enabled=false"], cwd=logdir, check=True,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
            m = re.findall(r"final score: [\d.]+ \((\d+) unrouted and (\d+) violations\)", out)
            score = tuple(map(int, m[-1])) if m else (999, 999)
            print(f"freerouting -mt {threads}: {score[0]} unrouted, {score[1]} violations")
            if best is None or score < best[0]:
                best = (score, ses.read_bytes())
            if score == (0, 0):
                break
    ses.write_bytes(best[1])
    board = pcbnew.LoadBoard(str(clean))
    zones = make_gnd_pours(board)
    if not pcbnew.ImportSpecctraSES(board, str(ses)):
        raise SystemExit("SES import failed")
    for z in zones:
        board.Add(z)
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(str(pcb))
    clean.unlink(missing_ok=True)
    clean.with_suffix(".kicad_pro").unlink(missing_ok=True)
    clean.with_suffix(".kicad_prl").unlink(missing_ok=True)
    dsn.unlink(missing_ok=True)
    ses.unlink(missing_ok=True)
    print("routed", pcb.name, "tracks", len(board.GetTracks()))


if __name__ == "__main__":
    main(*sys.argv[1:])
