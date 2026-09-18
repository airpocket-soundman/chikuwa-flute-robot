#!/usr/bin/env bash
# ERC + DRC (with schematic parity) for both boards, summary to stdout, reports and layout plots to $1
R="$1"; mkdir -p "$R"
K="/c/Program Files/KiCad/10.0/bin/kicad-cli.exe"; KP="/c/Program Files/KiCad/10.0/bin/python.exe"
cd "$(dirname "$0")/.."
for b in a b; do
  d=yamabiko_$b
  "$K" sch erc --format json -o "$R/erc_$b.json" $d/$d.kicad_sch >/dev/null 2>&1
  "$K" pcb drc --schematic-parity --refill-zones --format json -o "$R/drc_$b.json" $d/$d.kicad_pcb >/dev/null 2>&1
  "$KP" gen/plot_layout.py dump $d/$d.kicad_pcb "$R/lay_$b.json" 2>/dev/null
  python gen/plot_layout.py plot "$R/lay_$b.json" "$R/lay_$b.png"
done
python - "$R" <<'PY'
import json, sys, collections
R = sys.argv[1]
for b in "ab":
    e = json.load(open(f"{R}/erc_{b}.json", encoding="utf-8"))
    ev = [v for s in e.get("sheets", []) for v in s.get("violations", [])]
    r = json.load(open(f"{R}/drc_{b}.json", encoding="utf-8"))
    print(f"== board {b}: ERC {len(ev)}  unconnected {len(r.get('unconnected_items', []))}")
    for key in ("violations", "schematic_parity"):
        c = collections.Counter((v.get("type"), v.get("severity")) for v in r.get(key, []))
        print("  ", key, dict(c))
    seen = set()
    for v in ev + r.get("violations", []) + r.get("schematic_parity", []):
        if v.get("severity") == "error":
            t = v.get("type") + " | " + " | ".join(i.get("description", "") for i in v.get("items", []))
            if t not in seen:
                seen.add(t); print("     ", t)
PY
