"""Put the board name and revision on an already routed board without regenerating it
(regenerating means routing again). Replaces an earlier title.
    python add_title.py yamabiko_a/yamabiko_a.kicad_pcb
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pcbnew  # noqa: E402

from design import SILK_TITLE  # noqa: E402
from gen_boards import place_title, title_lines, write_project  # noqa: E402

pcb = Path(sys.argv[1]).resolve()
name = pcb.stem
# drop an earlier title in the file: removing items with pcbnew breaks its wrappers
src = pcb.read_text(encoding="utf-8")
src = re.sub(r'\n\t\(gr_text "' + re.escape(SILK_TITLE[name]) + r'(?: |\\n)Rev [^"]*".*?\n\t\)', "", src, flags=re.S)
pcb.write_text(src, encoding="utf-8")
board = pcbnew.LoadBoard(str(pcb))
print(name, "title size %.1f angle %d lines %d at (%.2f, %.2f)" % place_title(board, title_lines(name)))
board.Save(str(pcb))
write_project(pcb.parent, name)  # saving the board rewrote the project file with default net classes
