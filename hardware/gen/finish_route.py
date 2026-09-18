"""Route what the autorouter left unconnected with a small two-layer grid maze router (A*).

    "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" finish_route.py BOARD.kicad_pcb

For every net that is still split into islands, the router joins the island holding the net's first
pad to the nearest other island, over a 0.1 mm grid on F.Cu and B.Cu, keeping the net class
clearance to every other net's pads, tracks and vias and to the board edge. Vias cost extra.
The GND pours are refilled afterwards.
"""
import heapq
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pcbnew

G = 0.1            # grid [mm]
EDGE = 0.5         # copper to board edge [mm]
VIA_COST = 40.0    # in grid steps


def mm(v):
    return v / 1e6


def dist_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class Grid:
    def __init__(self, board):
        bb = board.GetBoardEdgesBoundingBox()
        self.x0, self.y0 = mm(bb.GetLeft()), mm(bb.GetTop())
        self.nx = int(mm(bb.GetWidth()) / G) + 1
        self.ny = int(mm(bb.GetHeight()) / G) + 1
        self.board = board
        # cells closer than EDGE to the outline (or outside it)
        outline = pcbnew.SHAPE_POLY_SET()
        board.GetBoardPolygonOutlines(outline, True)
        ch = outline.Outline(0)
        pts = [(mm(ch.CPoint(k).x), mm(ch.CPoint(k).y)) for k in range(ch.PointCount())]
        segs = list(zip(pts, pts[1:] + pts[:1]))
        self.edge = bytearray(self.nx * self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.xy(i, j)
                if not outline.Contains(pcbnew.VECTOR2I_MM(x, y)) or                         min(dist_seg(x, y, *a, *b) for a, b in segs) < EDGE:
                    self.edge[i * self.ny + j] = 1

    def cell(self, x, y):
        return int(round((x - self.x0) / G)), int(round((y - self.y0) / G))

    def xy(self, i, j):
        return self.x0 + i * G, self.y0 + j * G


def copper_items(board):
    """(net, layer set, shape) for every piece of copper: shape = ('seg', ax, ay, bx, by, r) or ('box', x0, y0, x1, y1)."""
    items = []
    for fp in board.GetFootprints():
        for p in fp.Pads():
            bb = p.GetBoundingBox()
            layers = {l for l in (pcbnew.F_Cu, pcbnew.B_Cu) if p.IsOnLayer(l)}
            if p.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH:
                layers = {pcbnew.F_Cu, pcbnew.B_Cu}
            items.append((p.GetNetCode(), layers, ("box", mm(bb.GetLeft()), mm(bb.GetTop()), mm(bb.GetRight()), mm(bb.GetBottom()))))
    for t in board.GetTracks():
        if t.GetClass() == "PCB_VIA":
            c = t.GetPosition()
            r = mm(t.GetWidth(pcbnew.F_Cu)) / 2
            items.append((t.GetNetCode(), {pcbnew.F_Cu, pcbnew.B_Cu}, ("seg", mm(c.x), mm(c.y), mm(c.x), mm(c.y), r)))
        else:
            s, e = t.GetStart(), t.GetEnd()
            items.append((t.GetNetCode(), {t.GetLayer()}, ("seg", mm(s.x), mm(s.y), mm(e.x), mm(e.y), mm(t.GetWidth()) / 2)))
    return items


def rasterize(grid, items, net, keep):
    """blocked[layer][i][j]: a track centre at the cell would come within `keep` of foreign copper."""
    blocked = {l: bytearray(grid.nx * grid.ny) for l in (pcbnew.F_Cu, pcbnew.B_Cu)}
    own = {l: bytearray(grid.nx * grid.ny) for l in (pcbnew.F_Cu, pcbnew.B_Cu)}
    for n, layers, shp in items:
        if shp[0] == "box":
            _, x0, y0, x1, y1 = shp
            pad = 0.0 if n == net else keep
            ax, ay, bx, by = x0 - pad, y0 - pad, x1 + pad, y1 + pad
            dist = None
        else:
            _, sx, sy, ex, ey, r = shp
            pad = r + (0.0 if n == net else keep)
            ax, ay, bx, by = min(sx, ex) - pad, min(sy, ey) - pad, max(sx, ex) + pad, max(sy, ey) + pad
            dist = (sx, sy, ex, ey, pad)
        i0, j0 = grid.cell(ax, ay)
        i1, j1 = grid.cell(bx, by)
        for i in range(max(0, i0), min(grid.nx, i1 + 1)):
            for j in range(max(0, j0), min(grid.ny, j1 + 1)):
                x, y = grid.xy(i, j)
                if dist and dist_seg(x, y, *dist[:4]) > dist[4]:
                    continue
                for l in layers:
                    if n == net and n != 0:
                        own[l][i * grid.ny + j] = 1
                    else:
                        blocked[l][i * grid.ny + j] = 1
    for l in blocked:
        for k in range(len(grid.edge)):
            if grid.edge[k]:
                blocked[l][k] = 1
    return blocked, own


def astar(grid, blocked, own_src, own_dst, via_ok):
    L = (pcbnew.F_Cu, pcbnew.B_Cu)
    ny = grid.ny
    starts = [(l, k) for l in L for k in range(len(own_src[l])) if own_src[l][k]]
    goals = {(l, k) for l in L for k in range(len(own_dst[l])) if own_dst[l][k]}
    if not starts or not goals:
        return None
    gx = sum(k // ny for _, k in goals) / len(goals)
    gy = sum(k % ny for _, k in goals) / len(goals)

    def h(k):
        return math.hypot(k // ny - gx, k % ny - gy) * 0.9

    dist, prev, pq = {}, {}, []
    for s in starts:
        dist[s] = 0.0
        heapq.heappush(pq, (h(s[1]), 0.0, s))
    steps = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while pq:
        _, d, (l, k) = heapq.heappop(pq)
        if d > dist.get((l, k), 1e18):
            continue
        if (l, k) in goals:
            path = [(l, k)]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1]
        i, j = divmod(k, ny)
        nbrs = []
        for di, dj, c in steps:
            ii, jj = i + di, j + dj
            if 0 <= ii < grid.nx and 0 <= jj < ny:
                nbrs.append(((l, ii * ny + jj), c))
        other = L[1] if l == L[0] else L[0]
        if via_ok[k]:
            nbrs.append(((other, k), VIA_COST))
        for nb, c in nbrs:
            nl, nk = nb
            if blocked[nl][nk] and not own_dst[nl][nk] and not own_src[nl][nk]:
                continue
            nd = d + c
            if nd < dist.get(nb, 1e18):
                dist[nb] = nd
                prev[nb] = (l, k)
                heapq.heappush(pq, (nd + h(nk), nd, nb))
    return None


def add_path(board, grid, path, net, width, via_d, via_drill):
    ny = grid.ny
    pts = [(l, *grid.xy(*divmod(k, ny))) for l, k in path]
    # collapse straight runs
    runs = []
    for p in pts:
        if runs and runs[-1][-1][0] == p[0]:
            runs[-1].append(p)
        else:
            runs.append([p])
    for r_i, run in enumerate(runs):
        corner = [run[0]]
        for a, b, c in zip(run, run[1:], run[2:]):
            if (b[1] - a[1], b[2] - a[2]) != (c[1] - b[1], c[2] - b[2]):
                corner.append(b)
        corner.append(run[-1])
        for a, b in zip(corner, corner[1:]):
            if (a[1], a[2]) == (b[1], b[2]):
                continue
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(pcbnew.VECTOR2I_MM(a[1], a[2]))
            t.SetEnd(pcbnew.VECTOR2I_MM(b[1], b[2]))
            t.SetWidth(pcbnew.FromMM(width))
            t.SetLayer(a[0])
            t.SetNetCode(net)
            board.Add(t)
        if r_i + 1 < len(runs):
            v = pcbnew.PCB_VIA(board)
            v.SetPosition(pcbnew.VECTOR2I_MM(run[-1][1], run[-1][2]))
            v.SetWidth(pcbnew.FromMM(via_d))
            v.SetDrill(pcbnew.FromMM(via_drill))
            v.SetNetCode(net)
            board.Add(v)


def unconnected(pcb: Path):
    """[(net name, (x, y), (x, y))] from a KiCad DRC run: the ratsnest is not reachable from Python."""
    cli = Path(pcbnew.__file__).resolve().parents[2] / "kicad-cli.exe"
    with tempfile.TemporaryDirectory() as d:
        rep = Path(d) / "drc.json"
        subprocess.run([str(cli), "pcb", "drc", "--format", "json", "-o", str(rep), str(pcb)],
                       check=False, capture_output=True)
        r = json.loads(rep.read_text(encoding="utf-8"))
    out = []
    for v in r.get("unconnected_items", []):
        a, b = v["items"][:2]
        m = re.search(r"\[(.+?)\]", a["description"])
        out.append((m.group(1) if m else "", (a["pos"]["x"], a["pos"]["y"]), (b["pos"]["x"], b["pos"]["y"])))
    return out


def main(pcb):
    pcb = Path(pcb).resolve()
    pro = json.loads(pcb.with_suffix(".kicad_pro").read_text(encoding="utf-8"))
    classes = {c["name"]: c for c in pro["net_settings"]["classes"]}
    done = 0
    for _ in range(20):
        todo = unconnected(pcb)
        if not todo:
            break
        name, pa, pb = todo[0]
        board = pcbnew.LoadBoard(str(pcb))
        grid = Grid(board)
        ni = board.FindNet(name)
        if ni is None:
            print("unknown net", name)
            break
        n = ni.GetNetCode()
        nc = classes[ni.GetNetClassName()]  # the NETCLASS object is not wrapped: read the project file
        width = nc["track_width"]
        clear = max(c["clearance"] for c in classes.values())  # the other net may be in a wider class
        via_d, via_drill = nc["via_diameter"], nc["via_drill"]
        items = copper_items(board)
        blocked, _ = rasterize(grid, items, n, clear + width / 2 + 0.02)
        vb, _ = rasterize(grid, items, n, clear + via_d / 2 + 0.02)
        via_ok = bytearray(0 if vb[pcbnew.F_Cu][k] or vb[pcbnew.B_Cu][k] else 1 for k in range(grid.nx * grid.ny))
        src = {l: bytearray(grid.nx * grid.ny) for l in blocked}
        dst = {l: bytearray(grid.nx * grid.ny) for l in blocked}
        for buf, (x, y) in ((src, pa), (dst, pb)):
            i, j = grid.cell(x, y)
            for l in buf:
                buf[l][i * grid.ny + j] = 1
        path = astar(grid, blocked, src, dst, via_ok)
        if not path:
            print("no path for", name, pa, pb)
            break
        add_path(board, grid, path, n, width, via_d, via_drill)
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        board.Save(str(pcb))
        done += 1
        print("routed", name, pa, "->", pb, "in", len(path), "grid steps")
    print("finished", done, "connection(s)")


if __name__ == "__main__":
    main(sys.argv[1])
