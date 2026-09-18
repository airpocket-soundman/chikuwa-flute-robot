"""Minimal S-expression reader/writer for KiCad files, plus symbol-library helpers."""
from __future__ import annotations

import re
from pathlib import Path

KICAD = Path(r"C:\Program Files\KiCad\10.0\share\kicad")

_TOKEN = re.compile(r'\s*(?:(\()|(\))|("(?:[^"\\]|\\.)*")|([^\s()"]+))')


class Sym(str):
    """A bare (unquoted) atom."""


def parse(text: str):
    stack, cur, pos = [], [], 0
    while True:
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            break
        pos = m.end()
        lp, rp, s, a = m.groups()
        if lp:
            stack.append(cur)
            cur = []
        elif rp:
            done = cur
            cur = stack.pop()
            cur.append(done)
        elif s is not None:
            cur.append(s[1:-1].replace('\\"', '"').replace("\\\\", "\\"))
        else:
            cur.append(Sym(a))
    return cur[0]


def q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def dump(node, indent: int = 0) -> str:
    if isinstance(node, list):
        if not node:
            return "()"
        simple = all(not isinstance(x, list) for x in node)
        if simple and len(node) <= 8:
            return "(" + " ".join(dump(x) for x in node) + ")"
        pad = "\t" * (indent + 1)
        head = dump(node[0])
        parts = [head]
        for x in node[1:]:
            if isinstance(x, list):
                parts.append("\n" + pad + dump(x, indent + 1))
            else:
                parts.append(" " + dump(x))
        return "(" + "".join(parts) + "\n" + "\t" * indent + ")"
    if isinstance(node, Sym):
        return str(node)
    if isinstance(node, bool):
        return "yes" if node else "no"
    if isinstance(node, (int, float)):
        return fmt_num(node)
    return q(node)


def fmt_num(v) -> str:
    if isinstance(v, int):
        return str(v)
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def find(node, key):
    for x in node:
        if isinstance(x, list) and x and x[0] == key:
            return x
    return None


def find_all(node, key):
    return [x for x in node if isinstance(x, list) and x and x[0] == key]


_lib_cache: dict[str, list] = {}


def load_lib(lib: str, name: str):
    """Return the raw symbol node `name` from library `lib` (handles .kicad_symdir)."""
    base = KICAD / "symbols"
    d = base / f"{lib}.kicad_symdir"
    path = d / f"{name}.kicad_sym" if d.is_dir() else base / f"{lib}.kicad_sym"
    key = str(path)
    if key not in _lib_cache:
        _lib_cache[key] = parse(path.read_text(encoding="utf-8"))
    for s in find_all(_lib_cache[key], "symbol"):
        if s[1] == name:
            return s
    raise KeyError(f"{lib}:{name}")


def flat_symbol(lib: str, name: str, custom: dict | None = None):
    """Symbol definition ready for lib_symbols: derived symbols flattened, top name 'lib:name'."""
    if custom and f"{lib}:{name}" in custom:
        node = parse(custom[f"{lib}:{name}"])
    else:
        node = load_lib(lib, name)
    ext = find(node, "extends")
    if ext:
        parent = flat_symbol(lib, ext[1], custom)
        props = {p[1]: p for p in find_all(node, "property")}
        out = [Sym("symbol"), f"{lib}:{name}"]
        for x in parent[2:]:
            if isinstance(x, list) and x[0] == "property" and x[1] in props:
                out.append(props.pop(x[1]))
            elif isinstance(x, list) and x[0] == "symbol":
                sub = list(x)
                sub[1] = sub[1].replace(ext[1] + "_", name + "_", 1)
                out.append(sub)
            else:
                out.append(x)
        out[2:2] = []
        out.extend(props.values())
        return out
    out = list(node)
    out[1] = f"{lib}:{name}"
    return out


def symbol_pins(defn, unit: int):
    """{pin number: (x, y, angle)} in library coordinates (y up) for `unit` (+ common unit 0)."""
    pins = {}
    base = defn[1].split(":", 1)[1]
    for sub in find_all(defn, "symbol"):
        m = re.match(re.escape(base) + r"_(\d+)_(\d+)$", sub[1])
        if not m:
            continue
        u = int(m.group(1))
        if u not in (0, unit):
            continue
        for p in find_all(sub, "pin"):
            at = find(p, "at")
            num = find(p, "number")[1]
            pins[num] = (float(at[1]), float(at[2]), float(at[3]) if len(at) > 3 else 0.0)
    return pins


if __name__ == "__main__":
    for lib, name in [("Transistor_FET", "2N7000"), ("Device", "Q_PMOS"), ("Device", "Q_NMOS"),
                      ("Regulator_Linear", "L7806"), ("74xx", "74LS125"), ("Connector", "Barrel_Jack"),
                      ("Switch", "SW_Push"), ("Device", "D_Schottky"), ("power", "GND")]:
        d = flat_symbol(lib, name)
        units = sorted({int(re.search(r"_(\d+)_\d+$", s[1]).group(1)) for s in find_all(d, "symbol")})
        fp = [p[2] for p in find_all(d, "property") if p[1] == "Footprint"]
        print(lib, name, "units", units, "fp", fp)
        for u in units or [1]:
            print("   unit", u, symbol_pins(d, u))
