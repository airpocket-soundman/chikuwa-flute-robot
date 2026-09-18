"""Generate the KiCad 10 projects of the Yamabiko No.1 boards from design.py.

Run with KiCad's Python (it needs pcbnew):
    "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" hardware/gen/gen_boards.py

Writes hardware/lib (project symbols and footprints), hardware/yamabiko_a and hardware/yamabiko_b
(schematic, board with placement and nets, project settings). Routing is done afterwards.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pcbnew  # noqa: E402

from design import A_PARTS, B_EDGE, B_PARTS, POWER_NETS, REV, SILK_TITLE, WIDE_NETS  # noqa: E402
from sexpr import KICAD, Sym, dump, find, find_all, flat_symbol, fmt_num, parse, symbol_pins  # noqa: E402

HW = HERE.parent
LIB = HW / "lib"
TEMPLATE = KICAD / "template" / "Arduino_Uno"
S = Sym


def uid() -> str:
    return str(uuid.uuid4())


# ------------------------------------------------------------------ project library

TB6643_SYM = """
(symbol "TB6643KQ" (pin_names (offset 1.016)) (exclude_from_sim no) (in_bom yes) (on_board yes)
  (property "Reference" "U" (at 0 11.43 0) (effects (font (size 1.27 1.27))))
  (property "Value" "TB6643KQ" (at 0 -11.43 0) (effects (font (size 1.27 1.27))))
  (property "Footprint" "yamabiko:Toshiba_HSIP7_P2.54mm" (at 0 -13.97 0) (effects (font (size 1.27 1.27)) (hide yes)))
  (property "Datasheet" "https://akizukidenshi.com/goodsaffix/TB6643KQ_datasheet_ja_20110621.pdf" (at 0 0 0) (effects (font (size 1.27 1.27)) (hide yes)))
  (property "Description" "Toshiba DC motor full-bridge driver, 10-45 V, 1.5 A average, HSIP7" (at 0 0 0) (effects (font (size 1.27 1.27)) (hide yes)))
  (symbol "TB6643KQ_0_1"
    (rectangle (start -7.62 7.62) (end 7.62 -7.62) (stroke (width 0.254) (type default)) (fill (type background))))
  (symbol "TB6643KQ_1_1"
    (pin input line (at -10.16 2.54 0) (length 2.54) (name "IN1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
    (pin input line (at -10.16 -2.54 0) (length 2.54) (name "IN2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    (pin output line (at 10.16 2.54 180) (length 2.54) (name "OUT1" (effects (font (size 1.27 1.27)))) (number "3" (effects (font (size 1.27 1.27)))))
    (pin power_in line (at 0 -10.16 90) (length 2.54) (name "GND" (effects (font (size 1.27 1.27)))) (number "4" (effects (font (size 1.27 1.27)))))
    (pin output line (at 10.16 -2.54 180) (length 2.54) (name "OUT2" (effects (font (size 1.27 1.27)))) (number "5" (effects (font (size 1.27 1.27)))))
    (pin no_connect line (at -10.16 -5.08 0) (length 2.54) (name "NC" (effects (font (size 1.27 1.27)))) (number "6" (effects (font (size 1.27 1.27)))))
    (pin power_in line (at 0 10.16 270) (length 2.54) (name "VM" (effects (font (size 1.27 1.27)))) (number "7" (effects (font (size 1.27 1.27)))))))
"""


def fet_gds(kind: str):
    """Device:Q_PMOS / Q_NMOS with numeric pins 1 = G, 2 = D, 3 = S (TO-251 / TO-220 order)."""
    d = flat_symbol("Device", kind)
    new = f"{kind}_GDS"

    def walk(n):
        if isinstance(n, list):
            if n and n[0] == "symbol" and isinstance(n[1], str):
                n[1] = n[1].replace("Device:" + kind, new).replace(kind + "_", new + "_", 1)
            if n and n[0] == "number":
                n[1] = {"G": "1", "D": "2", "S": "3"}.get(n[1], n[1])
            for x in n:
                walk(x)
    walk(d)
    d[1] = new
    return d


def hsip7_footprint() -> str:
    pads = []
    for i in range(7):
        shape = "rect" if i == 0 else "oval"
        pads.append(f'\t(pad "{i + 1}" thru_hole {shape} (at {fmt_num(i * 2.54)} 0) (size 1.7 2.4) (drill 1.0) '
                    f'(layers "*.Cu" "*.Mask") (remove_unused_layers no))')
    return "\n".join([
        '(footprint "Toshiba_HSIP7_P2.54mm" (version 20241229) (generator "yamabiko") (layer "F.Cu")',
        '\t(descr "Toshiba HSIP7-P-2.54A, 7 leads in one row at 2.54 mm (TB6643KQ). The back tab is tied to pin 4 inside: keep it insulated")',
        '\t(property "Reference" "REF**" (at 7.62 -3.6 0) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))',
        '\t(property "Value" "Toshiba_HSIP7_P2.54mm" (at 7.62 3.6 0) (layer "F.Fab") (effects (font (size 1 1) (thickness 0.15))))',
        '\t(fp_rect (start -2.3 -2.3) (end 17.54 1.9) (stroke (width 0.12) (type solid)) (fill no) (layer "F.SilkS"))',
        '\t(fp_line (start -2.3 -1.3) (end 17.54 -1.3) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
        '\t(fp_rect (start -2.2 -2.2) (end 17.44 1.8) (stroke (width 0.1) (type solid)) (fill no) (layer "F.Fab"))',
        '\t(fp_rect (start -2.55 -2.55) (end 17.79 2.15) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))',
        *pads,
        ")",
    ]) + "\n"


def write_library() -> dict:
    """Write hardware/lib and return {lib_id: symbol definition} for the project symbols."""
    (LIB / "yamabiko.pretty").mkdir(parents=True, exist_ok=True)
    (LIB / "yamabiko.pretty" / "Toshiba_HSIP7_P2.54mm.kicad_mod").write_text(hsip7_footprint(), encoding="utf-8")
    syms = {"TB6643KQ": parse(TB6643_SYM), "Q_PMOS_GDS": fet_gds("Q_PMOS"), "Q_NMOS_GDS": fet_gds("Q_NMOS")}
    lib = [S("kicad_symbol_lib"), [S("version"), 20241209], [S("generator"), "yamabiko"],
           [S("generator_version"), "10.0"]] + [syms[k] for k in syms]
    (LIB / "yamabiko.kicad_sym").write_text(dump(lib) + "\n", encoding="utf-8")
    out = {}
    for k, v in syms.items():
        d = parse(dump(v))
        d[1] = "yamabiko:" + k
        out["yamabiko:" + k] = d
    return out


# ------------------------------------------------------------------ schematic

def sym_def(lib_id: str, custom: dict):
    if lib_id in custom:
        return custom[lib_id]
    lib, name = lib_id.split(":", 1)
    return flat_symbol(lib, name)


def lib_id_of(p) -> str:
    if p.renumber:
        return "yamabiko:" + p.sym.split(":")[1] + "_GDS"
    return p.sym


def snap(v: float) -> float:
    return round(v / 2.54) * 2.54


def prop(name, value, x, y, hide=False, angle=0):
    eff = [S("effects"), [S("font"), [S("size"), 1.27, 1.27]]]
    if hide:
        eff.append([S("hide"), S("yes")])
    return [S("property"), name, value, [S("at"), x, y, angle], eff]


def write_schematic(path: Path, project: str, parts, custom: dict, title: str) -> dict:
    root = uid()
    used, instances, extras = {}, [], []
    uuids = {}
    pwr_n = [0]

    def power_symbol(net, x, y, pin_angle):
        pwr_n[0] += 1
        lid = "power:" + net
        used[lid] = sym_def(lid, custom)
        natural = 270 if net == "GND" else 90
        desired = (pin_angle + 180) % 360
        rot = int((desired - natural) % 360)
        ref = f"#PWR{pwr_n[0]:03d}"
        u = uid()
        instances.append([S("symbol"), [S("lib_id"), lid], [S("at"), x, y, rot], [S("unit"), 1],
                          [S("exclude_from_sim"), S("no")], [S("in_bom"), S("yes")], [S("on_board"), S("yes")],
                          [S("dnp"), S("no")], [S("uuid"), u],
                          prop("Reference", ref, x, y + 3.81, hide=True), prop("Value", net, x, y - 3.81 if natural == 90 else y + 5.08),
                          prop("Footprint", "", x, y, hide=True), prop("Datasheet", "", x, y, hide=True),
                          [S("pin"), "1", [S("uuid"), uid()]],
                          [S("instances"), [S("project"), project, [S("path"), "/" + root, [S("reference"), ref], [S("unit"), 1]]]]])

    for p in parts:
        lid = lib_id_of(p)
        d = sym_def(lid, custom)
        used[lid] = d
        u = uid()
        uuids[p.ref] = u
        for unit in range(1, p.units + 1):
            X, Y = snap(p.sch[0] + (unit - 1) * 25.4), snap(p.sch[1])
            pins = symbol_pins(d, unit)
            props = {q[1]: q for q in find_all(d, "property")}

            def lp(name, default):
                q = props.get(name)
                if not q:
                    return default
                at = find(q, "at")
                return X + float(at[1]), Y - float(at[2])
            rx, ry = lp("Reference", (X, Y - 5))
            vx, vy = lp("Value", (X, Y + 5))
            ref_txt = p.ref
            node = [S("symbol"), [S("lib_id"), lid], [S("at"), X, Y, 0], [S("unit"), unit],
                    [S("exclude_from_sim"), S("no")], [S("in_bom"), S("no") if p.ref.startswith("#") or p.sym.startswith("Mechanical:") else S("yes")],
                    [S("on_board"), S("no") if p.ref.startswith("#") else S("yes")],
                    [S("dnp"), S("yes") if p.dnp else S("no")],
                    [S("uuid"), u if unit == 1 else uid()],
                    prop("Reference", ref_txt, rx, ry, hide=p.ref.startswith("#")),
                    prop("Value", p.value, vx, vy),
                    prop("Footprint", p.fp, X, Y, hide=True),
                    prop("Datasheet", "", X, Y, hide=True)]
            if p.akizuki:
                node.append(prop("Akizuki", p.akizuki, X, Y, hide=True))
            for num in pins:
                node.append([S("pin"), num, [S("uuid"), uid()]])
            node.append([S("instances"), [S("project"), project, [S("path"), "/" + root, [S("reference"), p.ref], [S("unit"), unit]]]])
            instances.append(node)
            for num, (px, py, pa) in pins.items():
                x, y = round(X + px, 4), round(Y - py, 4)
                if num not in p.pins:
                    if p.ref.startswith("#") or not pins:
                        continue
                    net = None
                else:
                    net = p.pins[num]
                if net is None:
                    extras.append([S("no_connect"), [S("at"), x, y], [S("uuid"), uid()]])
                elif net in POWER_NETS:
                    power_symbol(net, x, y, pa)
                else:
                    ang = int((pa + 180) % 360)
                    just = {0: "left", 90: "left", 180: "right", 270: "right"}[ang]
                    extras.append([S("label"), net, [S("at"), x, y, ang], [S("fields_autoplaced"), S("yes")],
                                   [S("effects"), [S("font"), [S("size"), 1.27, 1.27]], [S("justify"), S(just), S("bottom")]],
                                   [S("uuid"), uid()]])

    sch = [S("kicad_sch"), [S("version"), 20250114], [S("generator"), "eeschema"], [S("generator_version"), "9.0"],
           [S("uuid"), root], [S("paper"), "A3"],
           [S("title_block"), [S("title"), title], [S("date"), "2026-09-18"], [S("rev"), REV],
            [S("company"), "chikuwa-flute-robot"],
            [S("comment"), 1, "Generated by hardware/gen/gen_boards.py from hardware/gen/design.py"]],
           [S("lib_symbols")] + list(used.values())]
    sch += instances + extras
    sch.append([S("sheet_instances"), [S("path"), "/", [S("page"), "1"]]])
    sch.append([S("embedded_fonts"), S("no")])
    path.write_text(dump(sch) + "\n", encoding="utf-8")
    return {"root": root, "uuids": uuids}


# ------------------------------------------------------------------ PCB

_IO = pcbnew.PCB_IO_KICAD_SEXPR()
FP_DIRS = {"yamabiko": LIB / "yamabiko.pretty"}


def load_fp(fpid: str):
    lib, name = fpid.split(":", 1)
    d = FP_DIRS.get(lib, KICAD / "footprints" / f"{lib}.pretty")
    fp = _IO.FootprintLoad(str(d), name)
    if fp is None:
        raise KeyError(fpid)
    fp.SetFPID(pcbnew.LIB_ID(lib, name))
    return fp


def net_name(n: str) -> str:
    return n if n in POWER_NETS else "/" + n


KICAD_CLI = KICAD.parent.parent / "bin" / "kicad-cli.exe"


def schematic_nets(sch: Path, parts) -> dict:
    """{(ref, pad): net} from the netlist KiCad exports for the schematic, checked against design.py."""
    tmp = sch.with_name("_netlist.net")
    subprocess.run([str(KICAD_CLI), "sch", "export", "netlist", "--format", "kicadsexpr", "-o", str(tmp), str(sch)],
                   check=True, capture_output=True)
    tree = parse(tmp.read_text(encoding="utf-8"))
    tmp.unlink()
    pad_nets = {}
    for net in find_all(find(tree, "nets"), "net"):
        name = find(net, "name")[1]
        for node in find_all(net, "node"):
            pad_nets[(find(node, "ref")[1], find(node, "pin")[1])] = name
    bad = []
    for p in parts:
        if p.ref.startswith("#"):
            continue
        for pin, n in p.pins.items():
            got = pad_nets.get((p.ref, pin))
            want = net_name(n) if n else None
            if n and got != want:
                bad.append(f"{p.ref}.{pin}: schematic {got}, design {want}")
            if not n and got and not got.startswith("unconnected-"):
                bad.append(f"{p.ref}.{pin}: schematic {got}, design unconnected")
    if bad:
        raise SystemExit("schematic does not match design.py:\n  " + "\n  ".join(bad))
    return pad_nets


TEXT = 0.8        # silkscreen text height [mm]
PIN_TEXT = 0.8    # connector pin names [mm]
GAP = 0.25        # text to courtyard [mm]


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _box(bb, grow=0.0):
    return (bb.GetLeft() / 1e6 - grow, bb.GetTop() / 1e6 - grow, bb.GetRight() / 1e6 + grow, bb.GetBottom() / 1e6 + grow)


def place_labels(board, parts):
    """Reference (every part) and value (resistors, capacitors) on the silkscreen, each put where it
    overlaps the least with pads, silkscreen drawings, labels already placed and the board edge."""
    obstacles = []
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            obstacles.append(_box(pad.GetBoundingBox(), 0.2))
        for g in fp.GraphicalItems():
            if g.GetLayer() == pcbnew.F_SilkS:
                obstacles.append(_box(g.GetBoundingBox(), 0.1))
    edge = _box(board.GetBoardEdgesBoundingBox(), -0.3)
    want_value = {p.ref for p in parts if re.match(r"[RC]\d", p.ref)}
    by_ref = {p.ref: p for p in parts}
    # connectors first: a pin name next to every pad, on the side given in design.py
    for fp in board.GetFootprints():
        p = by_ref.get(fp.GetReference())
        if not p or not p.pin_names:
            continue
        x0, y0, x1, y1 = _box(fp.GetCourtyard(pcbnew.F_CrtYd).BBox())
        pads = {pd.GetNumber(): pd.GetPosition() for pd in fp.Pads()}
        for i, name in enumerate(p.pin_names, start=1):
            pos = pads[str(i)]
            px, py = pos.x / 1e6, pos.y / 1e6
            t = pcbnew.PCB_TEXT(fp)
            t.SetText(name)
            t.SetLayer(pcbnew.F_SilkS)
            t.SetTextSize(pcbnew.VECTOR2I_MM(PIN_TEXT * 0.8, PIN_TEXT))  # narrow: 4 letters within the 2.5 mm pitch
            t.SetTextThickness(pcbnew.FromMM(0.1))
            t.SetTextAngle(pcbnew.EDA_ANGLE(0, pcbnew.DEGREES_T))
            t.SetVertJustify(pcbnew.GR_TEXT_V_ALIGN_CENTER)
            if p.pin_side in ("above", "below"):
                t.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)
                y = y0 - PIN_TEXT / 2 - GAP if p.pin_side == "above" else y1 + PIN_TEXT / 2 + GAP
                t.SetPosition(pcbnew.VECTOR2I_MM(px, y))
            else:  # a column of pads: the names stand on end so each fits within the 2.5 mm pitch
                left = p.pin_side == "left"
                t.SetTextAngle(pcbnew.EDA_ANGLE(90, pcbnew.DEGREES_T))
                t.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)
                x = x0 - PIN_TEXT / 2 - GAP if left else x1 + PIN_TEXT / 2 + GAP
                t.SetPosition(pcbnew.VECTOR2I_MM(x, py))
            fp.Add(t)
            obstacles.append(_box(t.GetBoundingBox()))
    # connectors with a device name first: their labels are the longest and the most useful
    fps = sorted(board.GetFootprints(),
                 key=lambda f: (not (by_ref.get(f.GetReference()) and by_ref[f.GetReference()].device), f.GetReference()))
    for fp in fps:
        cb = fp.GetCourtyard(pcbnew.F_CrtYd).BBox()
        x0, y0, x1, y1 = _box(cb)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        p = by_ref.get(fp.GetReference())
        if p and not p.silk_ref:
            fp.Reference().SetLayer(pcbnew.F_Fab)
            fp.Value().SetLayer(pcbnew.F_Fab)
            continue
        if p and p.device:  # reference and the device that plugs in, as one label
            fp.Reference().SetVisible(False)
            lab = pcbnew.PCB_TEXT(fp)
            lab.SetText(f"{p.ref} {p.device}")
            fp.Add(lab)
            texts = [(lab, True)]
        else:
            texts = [(fp.Reference(), True)]
        if fp.GetReference() in want_value:
            texts.append((fp.Value(), False))
        else:
            fp.Value().SetLayer(pcbnew.F_Fab)
        for t, is_ref in texts:
            t.SetLayer(pcbnew.F_SilkS)
            t.SetVisible(True)
            t.SetTextSize(pcbnew.VECTOR2I_MM(TEXT, TEXT))
            t.SetTextThickness(pcbnew.FromMM(0.12))
            t.SetTextAngle(pcbnew.EDA_ANGLE(0, pcbnew.DEGREES_T))
            t.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)
            t.SetVertJustify(pcbnew.GR_TEXT_V_ALIGN_CENTER)
            t.SetPosition(pcbnew.VECTOR2I_MM(cx, cy))
            tw = t.GetBoundingBox().GetWidth() / 1e6
            h = TEXT / 2 + GAP
            above, below = (cx, y0 - h), (cx, y1 + h)
            left, right = (x0 - tw / 2 - GAP, cy), (x1 + tw / 2 + GAP, cy)
            corners = [(x0 + tw / 2, y0 - h), (x1 - tw / 2, y0 - h), (x0 + tw / 2, y1 + h), (x1 - tw / 2, y1 + h)]
            d = TEXT * 0.6  # beside the part, nudged up or down so a reference and a value can share a side
            if is_ref:
                order = [above, below, (right[0], cy - d), (left[0], cy - d), left, right] + corners + [(cx, cy)]
            else:
                order = [below, above, (right[0], cy + d), (left[0], cy + d), right, left] + corners + [(cx, cy)]
            best = None
            for rank, (x, y) in enumerate(order):
                t.SetPosition(pcbnew.VECTOR2I_MM(x, y))
                tb = _box(t.GetBoundingBox())
                cost = sum(_overlap(tb, o) for o in obstacles)
                if tb[0] < edge[0] or tb[1] < edge[1] or tb[2] > edge[2] or tb[3] > edge[3]:
                    cost += 100.0
                cost += rank * 0.01
                if best is None or cost < best[0]:
                    best = (cost, x, y, tb)
            t.SetPosition(pcbnew.VECTOR2I_MM(best[1], best[2]))
            obstacles.append(best[3])


TITLE_SIZES = (1.5, 1.2, 1.0, 0.8)  # board name and revision [mm], largest that fits
EDGE_GAP = 0.8                      # title to board edge [mm]


def _inside(outline, box, gap):
    """Every point on the rim of box grown by gap lies inside the board outline (sampled every 0.5 mm)."""
    x0, y0, x1, y1 = box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap
    nx, ny = max(1, int((x1 - x0) / 0.5)), max(1, int((y1 - y0) / 0.5))
    pts = [(x0 + (x1 - x0) * i / nx, y) for i in range(nx + 1) for y in (y0, y1)]
    pts += [(x, y0 + (y1 - y0) * j / ny) for j in range(ny + 1) for x in (x0, x1)]
    return all(outline.Contains(pcbnew.VECTOR2I_MM(x, y)) for x, y in pts)


def place_title(board, lines):
    """Board name and revision on the front silkscreen, in the largest free spot: away from pads,
    silkscreen, courtyards and the board edge. Tracks underneath are fine (they are under the mask).
    Tries one line first, then the name and the revision on two lines."""
    obstacles = []
    for fp in board.GetFootprints():
        obstacles.append(_box(fp.GetCourtyard(pcbnew.F_CrtYd).BBox(), 0.2))
        for pad in fp.Pads():
            obstacles.append(_box(pad.GetBoundingBox(), 0.3))
        for item in list(fp.GraphicalItems()) + [fp.Reference(), fp.Value()]:
            if item.GetLayer() == pcbnew.F_SilkS and (not hasattr(item, "IsVisible") or item.IsVisible()):
                obstacles.append(_box(item.GetBoundingBox(), 0.3))
    for d in board.GetDrawings():
        if d.GetLayer() == pcbnew.F_SilkS:
            obstacles.append(_box(d.GetBoundingBox(), 0.3))
    outline = pcbnew.SHAPE_POLY_SET()
    board.GetBoardPolygonOutlines(outline, False)
    edge = _box(board.GetBoardEdgesBoundingBox(), -EDGE_GAP)
    cx, cy = (edge[0] + edge[2]) / 2, (edge[1] + edge[3]) / 2
    t = pcbnew.PCB_TEXT(board)
    t.SetLayer(pcbnew.F_SilkS)
    t.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)
    t.SetVertJustify(pcbnew.GR_TEXT_V_ALIGN_CENTER)
    for size in TITLE_SIZES:
        for text in (" ".join(lines), "\n".join(lines)):
            t.SetText(text)
            t.SetTextSize(pcbnew.VECTOR2I_MM(size, size))
            t.SetTextThickness(pcbnew.FromMM(size * 0.15))
            for angle in (0, 90):
                t.SetTextAngle(pcbnew.EDA_ANGLE(angle, pcbnew.DEGREES_T))
                best = None
                for i in range(int((edge[3] - edge[1]) / 0.25) + 1):
                    y = edge[1] + i * 0.25
                    for j in range(int((edge[2] - edge[0]) / 0.25) + 1):
                        x = edge[0] + j * 0.25
                        t.SetPosition(pcbnew.VECTOR2I_MM(x, y))
                        tb = _box(t.GetBoundingBox())
                        if tb[0] < edge[0] or tb[1] < edge[1] or tb[2] > edge[2] or tb[3] > edge[3]:
                            continue
                        if any(_overlap(tb, o) for o in obstacles) or not _inside(outline, tb, EDGE_GAP):
                            continue
                        dist = (x - cx) ** 2 + (y - cy) ** 2  # the free spot nearest the middle of the board
                        if best is None or dist < best[0]:
                            best = (dist, x, y)
                if best:
                    t.SetPosition(pcbnew.VECTOR2I_MM(best[1], best[2]))
                    board.Add(t)
                    return size, angle, text.count("\n") + 1, best[1], best[2]
    raise RuntimeError(f"no room for the title {lines!r}")


def title_lines(name: str) -> tuple:
    return (SILK_TITLE[name], f"Rev {REV}")


def write_pcb(path: Path, parts, sch_info, base: Path | None, edge=None, project="", pad_nets=None):
    # load every footprint before touching a board: after LoadBoard/Remove the loader hands back raw pointers
    loaded = {p.ref: load_fp(p.fp) for p in parts if p.fp and p.pcb is not None}
    board = pcbnew.BOARD()
    if base:
        # copy the outline only: removing footprints from a loaded board breaks pcbnew's SWIG wrappers
        tpl = pcbnew.LoadBoard(str(base))
        for d in tpl.GetDrawings():
            if d.GetLayer() == pcbnew.Edge_Cuts:
                board.Add(d.Duplicate())
    else:
        x0, y0, x1, y1 = edge
        rect = pcbnew.PCB_SHAPE(board, pcbnew.SHAPE_T_RECT)
        rect.SetStart(pcbnew.VECTOR2I_MM(x0, y0))
        rect.SetEnd(pcbnew.VECTOR2I_MM(x1, y1))
        rect.SetLayer(pcbnew.Edge_Cuts)
        rect.SetWidth(pcbnew.FromMM(0.1))
        board.Add(rect)
    nets = {}
    for name in sorted(set(pad_nets.values())):
        ni = pcbnew.NETINFO_ITEM(board, name)
        board.Add(ni)
        nets[name] = ni
    for p in parts:
        if not p.fp or p.pcb is None:
            continue
        fp = loaded[p.ref]
        fp.SetReference(p.ref)
        fp.SetValue(p.value)
        x, y, rot = p.pcb
        fp.SetPosition(pcbnew.VECTOR2I_MM(x, y))
        fp.SetOrientationDegrees(rot)
        fp.SetPath(pcbnew.KIID_PATH("/" + sch_info["uuids"][p.ref]))
        if hasattr(fp, "SetSheetfile"):
            fp.SetSheetfile(path.with_suffix(".kicad_sch").name)
            fp.SetSheetname("/")
        size, thick = pcbnew.VECTOR2I_MM(0.8, 0.8), pcbnew.FromMM(0.12)
        fp.Reference().SetTextSize(size)
        fp.Reference().SetTextThickness(thick)
        if re.match(r"[RC]\d", p.ref):  # values of resistors and capacitors go on the silkscreen
            val = fp.Value()
            val.SetLayer(pcbnew.F_SilkS)
            val.SetVisible(True)
            val.SetTextSize(size)
            val.SetTextThickness(thick)
            fp.Reference().SetLayer(pcbnew.F_Fab)  # only the value on the silkscreen: they would overlap
        if p.akizuki:
            fp.SetField("Akizuki", p.akizuki)
            fld = fp.GetField("Akizuki")
            if fld:
                fld.SetVisible(False)
        if p.dnp and hasattr(fp, "SetDNP"):
            fp.SetDNP(True)
        board.Add(fp)
        for pad in fp.Pads():
            n = pad_nets.get((p.ref, pad.GetNumber()))
            if n:
                pad.SetNet(nets[n])
    place_labels(board, parts)
    place_title(board, title_lines(project))
    # the GND pours are added by route.py after routing
    board.Save(str(path))
    return board


# ------------------------------------------------------------------ project files

def write_project(d: Path, name: str):
    pro = json.loads((TEMPLATE / "Arduino_Uno.kicad_pro").read_text(encoding="utf-8"))
    pro["meta"]["filename"] = f"{name}.kicad_pro"
    ns = pro.setdefault("net_settings", {})
    classes = ns.get("classes") or []
    default = next((c for c in classes if c.get("name") == "Default"), None) or {"name": "Default"}
    default.update({"clearance": 0.2, "track_width": 0.3, "via_diameter": 0.8, "via_drill": 0.4})
    power = dict(default)
    power.update({"name": "Power", "track_width": 0.6, "clearance": 0.25, "priority": 0})
    default["priority"] = 2147483647
    ns["classes"] = [default, power]
    ns["netclass_patterns"] = [{"netclass": "Power", "pattern": pat} for pat in
                               sorted({net_name(n) for n in WIDE_NETS})]
    sev = pro.setdefault("board", {}).setdefault("design_settings", {}).setdefault("rule_severities", {})
    sev.update({"silk_overlap": "warning", "silk_over_copper": "warning", "silk_edge_clearance": "warning",
                "starved_thermal": "warning"})
    (d / f"{name}.kicad_pro").write_text(json.dumps(pro, indent=2), encoding="utf-8")
    (d / "fp-lib-table").write_text(
        '(fp_lib_table\n  (version 7)\n'
        '  (lib (name "yamabiko")(type "KiCad")(uri "${KIPRJMOD}/../lib/yamabiko.pretty")(options "")(descr "Yamabiko project footprints"))\n'
        ')\n', encoding="utf-8")
    (d / "sym-lib-table").write_text(
        '(sym_lib_table\n  (version 7)\n'
        '  (lib (name "yamabiko")(type "KiCad")(uri "${KIPRJMOD}/../lib/yamabiko.kicad_sym")(options "")(descr "Yamabiko project symbols"))\n'
        ')\n', encoding="utf-8")


def main():
    custom = write_library()
    for name, parts, base, edge, title in [
        ("yamabiko_a", A_PARTS, TEMPLATE / "Arduino_Uno.kicad_pcb", None, "Yamabiko No.1 A: UNO Q driver shield"),
        ("yamabiko_b", B_PARTS, None, B_EDGE, "Yamabiko No.1 B: microphone board on the Breakout Carrier"),
    ]:
        d = HW / name
        d.mkdir(parents=True, exist_ok=True)
        write_project(d, name)
        info = write_schematic(d / f"{name}.kicad_sch", name, parts, custom, title)
        pad_nets = schematic_nets(d / f"{name}.kicad_sch", parts)
        write_pcb(d / f"{name}.kicad_pcb", parts, info, base, edge, name, pad_nets)
        write_project(d, name)  # saving the board rewrote the project file with default net classes
        print("wrote", d)


if __name__ == "__main__":
    main()
