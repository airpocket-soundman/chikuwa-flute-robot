#!/usr/bin/env bash
# Build both boards from design.py: generate, autoroute (Freerouting), finish the leftovers, check.
#   gen/build.sh FREEROUTING.jar REPORT_DIR
set -e
cd "$(dirname "$0")/.."
KP="/c/Program Files/KiCad/10.0/bin/python.exe"
"$KP" gen/gen_boards.py 2>&1 | grep -v "memory leak"
for b in a b; do
  "$KP" gen/route.py yamabiko_$b/yamabiko_$b.kicad_pcb "$1" 300 2>&1 | grep -E "^freerouting|^routed|Error"
  "$KP" gen/finish_route.py yamabiko_$b/yamabiko_$b.kicad_pcb 2>&1 | grep -E "^routed|^no path|^finished|Error"
done
gen/check.sh "$2"
