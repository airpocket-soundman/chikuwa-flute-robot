#!/usr/bin/env bash
# Gerbers and drill files for JLCPCB (2 layers), one zip per board, into fab/:
#   gen/fab.sh
# Zones are refilled before plotting; silkscreen is clipped to the solder mask openings.
set -e
K="/c/Program Files/KiCad/10.0/bin/kicad-cli.exe"
cd "$(dirname "$0")/.."
for b in a b; do
  d=yamabiko_$b; out=fab/$d; rm -rf "$out"; mkdir -p "$out"
  "$K" pcb export gerbers --check-zones --subtract-soldermask --no-x2 \
    -l F.Cu,B.Cu,F.Mask,B.Mask,F.SilkS,B.SilkS,Edge.Cuts -o "$out/" $d/$d.kicad_pcb >/dev/null
  "$K" pcb export drill --format excellon --excellon-units mm --excellon-zeros-format decimal \
    --excellon-separate-th --generate-map --map-format gerberx2 -o "$out/" $d/$d.kicad_pcb >/dev/null
  rm -f "fab/${d}_gerber.zip"
  (cd "$out" && python -c "import sys, zipfile, pathlib; z = zipfile.ZipFile(sys.argv[1], 'w', zipfile.ZIP_DEFLATED); [z.write(p, p.name) for p in sorted(pathlib.Path('.').iterdir()) if p.is_file()]" "../${d}_gerber.zip")
  echo "fab/${d}_gerber.zip: $(ls "$out" | tr '\n' ' ')"
done
