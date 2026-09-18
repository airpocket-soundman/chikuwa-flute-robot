"""Plot a board's placement (outline, courtyards, pads, references, tracks) for review.

Two steps, because pcbnew lives in KiCad's Python and matplotlib in the normal one:
    "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" plot_layout.py dump BOARD.kicad_pcb OUT.json
    python plot_layout.py plot OUT.json OUT.png
"""
import json
import sys


def dump(pcb, out):
    import pcbnew
    b = pcbnew.LoadBoard(pcb)
    mm = lambda v: v / 1e6  # noqa: E731
    data = {"edges": [], "fps": [], "tracks": []}
    for d in b.GetDrawings():
        if d.GetLayer() == pcbnew.Edge_Cuts:
            data["edges"].append([[mm(p.x), mm(p.y)] for p in (d.GetStart(), d.GetEnd())] + [d.GetShapeStr()])
            if d.GetShapeStr() == "Rect":
                s, e = d.GetStart(), d.GetEnd()
                data["edges"][-1] = [[mm(s.x), mm(s.y)], [mm(e.x), mm(e.y)], "Rect"]
            if d.GetShapeStr() == "Arc":
                pts = [d.GetStart(), d.GetArcMid(), d.GetEnd()]
                data["edges"][-1] = [[mm(p.x), mm(p.y)] for p in pts] + ["Arc"]
    for fp in b.GetFootprints():
        bb = fp.GetCourtyard(pcbnew.F_CrtYd).BBox()
        pads = [[mm(p.GetPosition().x), mm(p.GetPosition().y), mm(p.GetSize(pcbnew.F_Cu).x), mm(p.GetSize(pcbnew.F_Cu).y),
                 p.GetNetname(), p.GetNumber()] for p in fp.Pads()]
        data["fps"].append({"ref": fp.GetReference(), "crt": [mm(bb.GetX()), mm(bb.GetY()), mm(bb.GetWidth()), mm(bb.GetHeight())],
                            "pads": pads, "pos": [mm(fp.GetPosition().x), mm(fp.GetPosition().y)]})
    for t in b.GetTracks():
        if t.GetClass() == "PCB_TRACK":
            data["tracks"].append([mm(t.GetStart().x), mm(t.GetStart().y), mm(t.GetEnd().x), mm(t.GetEnd().y),
                                   mm(t.GetWidth()), t.GetLayerName()])
    json.dump(data, open(out, "w"))


def plot(js, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    d = json.load(open(js))
    fig, ax = plt.subplots(figsize=(14, 11))
    for e in d["edges"]:
        if e[-1] == "Rect":
            (x0, y0), (x1, y1) = e[0], e[1]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="k", lw=1.5))
        else:
            xs = [p[0] for p in e[:-1]]
            ys = [p[1] for p in e[:-1]]
            ax.plot(xs, ys, "k-", lw=1.5)
    for t in d["tracks"]:
        ax.plot([t[0], t[2]], [t[1], t[3]], "-", color="#c33" if t[5] == "F.Cu" else "#36c", lw=max(0.6, t[4] * 3), alpha=0.7)
    for f in d["fps"]:
        x, y, w, h = f["crt"]
        if w > 0:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, color="#999", lw=0.8))
        for px, py, sx, sy, net, num in f["pads"]:
            if not net and sx > 2.5:
                ax.add_patch(Circle((px, py), sx / 2, color="#bbb", alpha=0.5))
                continue
            ax.add_patch(Rectangle((px - sx / 2, py - sy / 2), sx, sy, color="#d4a017" if num != "1" else "#b35c00", alpha=0.8))
            ax.text(px, py, net.replace("/", "")[:9], fontsize=4.2, ha="center", va="center", color="#222")
        ax.text(x + w / 2, y + h / 2 if w else f["pos"][1], f["ref"], fontsize=8, ha="center", va="bottom", color="#1d6b57", weight="bold")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.autoscale()
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.savefig(out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    {"dump": dump, "plot": plot}[sys.argv[1]](*sys.argv[2:])
