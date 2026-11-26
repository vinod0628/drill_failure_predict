"""Click picking for Identify.

One hardware (GPU) selection over a small square around the click returns the
visible cells of every actor there; it costs one selection render, also for
block models with tens of millions of faces. The candidates are then resolved
exactly on the CPU: polygons by testing whether the click falls inside them
(front-most wins), lines and points by their on-screen distance, so thin
strings and drillholes can be picked without pixel-perfect aim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

if TYPE_CHECKING:
    from .layers import Layer

RADIUS = 6            # pixels searched around the click
SNAP = 2.5            # anything this close (pixels) counts as "under the cursor"
POINT_BONUS = 2.0     # a point marker drawn on a line wins over the line when about as close
MAX_LINE_CELLS = 5000
MAX_POLY_CELLS = 2000


@dataclass
class PickHit:
    layer: "Layer"
    actor: object
    mesh: pv.PolyData
    cell_id: int
    kind: str                  # "vertex", "line" or "polygon"
    point_ids: np.ndarray      # the cell's point ids
    weights: np.ndarray        # interpolation weights of the picked spot over point_ids
    local: np.ndarray          # picked position, scene-local coordinates
    distance_px: float

    def cell_value(self, name: str):
        return self.mesh.cell_data[name][self.cell_id]

    def interpolate(self, name: str) -> float:
        v = np.asarray(self.mesh.point_data[name], dtype=np.float64)[self.point_ids]
        return float(np.dot(self.weights, v))

    def nearest_point_value(self, name: str):
        return self.mesh.point_data[name][self.point_ids[int(np.argmax(self.weights))]]

    def cell_points(self) -> np.ndarray:
        return np.asarray(self.mesh.points[self.point_ids], dtype=np.float64)


# ---------------------------------------------------------------------------
def _projector(renderer):
    """World (scene-local) -> (display x, display y, normalised depth)."""
    cam = renderer.GetActiveCamera()
    m = cam.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)
    mat = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
    w, h = renderer.GetSize()
    ox, oy = renderer.GetOrigin()

    def project(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        hom = np.c_[pts, np.ones(len(pts))] @ mat.T
        ndc = hom[:, :3] / hom[:, 3:4]
        return np.column_stack([(ndc[:, 0] + 1) * 0.5 * w + ox, (ndc[:, 1] + 1) * 0.5 * h + oy, ndc[:, 2]])

    return project


def _select(renderer, x0, y0, x1, y1) -> list[tuple[object, np.ndarray]]:
    w, h = renderer.GetSize()
    ox, oy = renderer.GetOrigin()
    x0, x1 = max(int(x0), ox), min(int(x1), ox + w - 1)
    y0, y1 = max(int(y0), oy), min(int(y1), oy + h - 1)
    if x1 < x0 or y1 < y0:
        return []
    sel = vtk.vtkHardwareSelector()
    sel.SetRenderer(renderer)
    sel.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
    sel.SetArea(x0, y0, x1, y1)
    result = sel.Select()
    out = []
    if result is None:
        return out
    for i in range(result.GetNumberOfNodes()):
        node = result.GetNode(i)
        prop = node.GetProperties().Get(vtk.vtkSelectionNode.PROP())
        ids = node.GetSelectionList()
        if prop is not None and ids is not None and ids.GetNumberOfTuples():
            out.append((prop, vtk_to_numpy(ids).astype(np.int64)))
    return out


def _uniform_step(off: np.ndarray, chunk: int = 1 << 20) -> int | None:
    """Points per cell when every cell has the same size (checked exactly, in chunks)."""
    n = len(off) - 1
    if n <= 0:
        return None
    step = int(off[1] - off[0])
    if int(off[-1] - off[0]) != n * step:
        return None
    for s in range(0, n, chunk):
        d = np.diff(off[s:min(n, s + chunk) + 1])
        if not np.all(d == step):
            return None
    return step


class _CellLookup:
    """Point ids of polydata cells, read straight from the cell arrays (no BuildCells).

    It also maps OpenGL primitive ids back to cells: a mapper that is not a
    vtkPolyDataMapper (PyVista's DataSetMapper) reports the primitive drawn at a
    pixel - a point of a vertex cell, a segment of a polyline, a triangle of a
    polygon - numbered over verts, then lines, then polygons, then strips.
    """

    _PRIMS_PER_CELL = {"vertex": 0, "line": 1, "polygon": 2, "strip": 2}  # n points -> n - this

    def __init__(self, poly):
        self.blocks = []
        start = prim = 0
        for kind, arr in (("vertex", poly.GetVerts()), ("line", poly.GetLines()),
                          ("polygon", poly.GetPolys()), ("strip", poly.GetStrips())):
            n = arr.GetNumberOfCells()
            if n:
                off = vtk_to_numpy(arr.GetOffsetsArray())
                con = vtk_to_numpy(arr.GetConnectivityArray())
                sub = self._PRIMS_PER_CELL[kind]
                step = _uniform_step(off)
                if step is not None:
                    per, cum = max(step - sub, 0), None
                    n_prims = per * n
                else:
                    per = None
                    cum = np.cumsum(np.maximum(np.diff(off) - sub, 0))
                    n_prims = int(cum[-1]) if len(cum) else 0
                self.blocks.append(dict(lo=start, hi=start + n, kind=kind, off=off, con=con,
                                        plo=prim, phi=prim + n_prims, per=per, cum=cum))
                prim += n_prims
            start += n

    def get(self, cid: int):
        for b in self.blocks:
            if b["lo"] <= cid < b["hi"]:
                k = cid - b["lo"]
                return b["kind"], b["con"][b["off"][k]:b["off"][k + 1]].astype(np.int64)
        return None, None

    def cell_of_primitive(self, pid: int) -> int | None:
        for b in self.blocks:
            if b["plo"] <= pid < b["phi"]:
                q = pid - b["plo"]
                k = q // b["per"] if b["per"] else int(np.searchsorted(b["cum"], q, side="right"))
                return b["lo"] + int(k)
        return None


_lookup_cache: dict[str, tuple[int, _CellLookup]] = {}


def _lookup_for(poly) -> _CellLookup:
    key = poly.GetAddressAsString("vtkObject")
    cached = _lookup_cache.get(key)
    if cached is not None and cached[0] == poly.GetMTime():
        return cached[1]
    lk = _CellLookup(poly)
    if len(_lookup_cache) > 16:
        _lookup_cache.clear()
    _lookup_cache[key] = (poly.GetMTime(), lk)
    return lk


def _closest_on_segments(p2: np.ndarray, click: np.ndarray):
    """Closest point to ``click`` on the polyline through 2D points p2: (dist, seg, t)."""
    a, b = p2[:-1], p2[1:]
    ab = b - a
    L = np.einsum("ij,ij->i", ab, ab)
    t = np.where(L > 0, np.einsum("ij,ij->i", click - a, ab) / np.where(L > 0, L, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = np.linalg.norm(a + t[:, None] * ab - click, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), k, float(t[k])


def _evaluate(kind, ids, pts3, proj, click):
    """(distance px, depth, weights, local point) of a cell relative to the click."""
    s = proj(pts3)
    p2, z = s[:, :2], s[:, 2]
    n = len(ids)
    if kind == "vertex" or n == 1:
        d = np.linalg.norm(p2 - click, axis=1)
        k = int(np.argmin(d))
        w = np.zeros(n)
        w[k] = 1.0
        return float(d[k]), float(z[k]), w, pts3[k]
    if kind == "line":
        d, k, t = _closest_on_segments(p2, click)
        w = np.zeros(n)
        w[k], w[k + 1] = 1.0 - t, t
        return d, float(z[k] * (1 - t) + z[k + 1] * t), w, pts3[k] * (1 - t) + pts3[k + 1] * t
    # Polygon: fan triangulation, barycentric test of the click in screen space.
    best = None
    for j in range(1, n - 1):
        tri = (0, j, j + 1)
        a, b, c = p2[0], p2[j], p2[j + 1]
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (click[0] - c[0]) + (c[0] - b[0]) * (click[1] - c[1])) / den
        l2 = ((c[1] - a[1]) * (click[0] - c[0]) + (a[0] - c[0]) * (click[1] - c[1])) / den
        lam = np.array([l1, l2, 1.0 - l1 - l2])
        if lam.min() >= -1e-6:
            w = np.zeros(n)
            w[list(tri)] = lam
            depth = float(np.dot(lam, z[list(tri)]))
            return 0.0, depth, w, w @ pts3
    # Click outside the polygon (within the search square): nearest edge.
    ring = np.vstack([p2, p2[:1]])
    d, k, t = _closest_on_segments(ring, click)
    k2 = (k + 1) % n
    w = np.zeros(n)
    w[k], w[k2] = 1.0 - t, t
    best = (d, float(z[k] * (1 - t) + z[k2] * t), w, w @ pts3)
    return best


def pick(renderer, layers: list["Layer"], x: int, y: int, radius: int = RADIUS) -> PickHit | None:
    """Identify the layer object under display position (x, y), or None."""
    owners = {}
    for layer in layers:
        if not layer.visible:
            continue
        for a in layer.actors:
            owners[a.GetAddressAsString("vtkObject")] = (layer, a)
    if not owners:
        return None
    proj = _projector(renderer)
    click = np.array([float(x), float(y)])
    cands: list[tuple[float, float, PickHit]] = []   # (distance px, depth, hit)
    for prop, ids in _select(renderer, x - radius, y - radius, x + radius, y + radius):
        owner = owners.get(prop.GetAddressAsString("vtkObject"))
        if owner is None:
            continue
        layer, actor = owner
        poly = actor.GetMapper().GetInput()
        if poly is None or not isinstance(poly, vtk.vtkPolyData):
            continue
        mesh = pv.wrap(poly)
        pts = mesh.points
        lookup = _lookup_for(poly)
        mapper = actor.GetMapper()
        if not mapper.IsA("vtkPolyDataMapper"):  # ids are OpenGL primitives, not cells
            cells = {lookup.cell_of_primitive(int(i)) for i in ids}
            ids = np.array(sorted(c for c in cells if c is not None), dtype=np.int64)
        n_line = n_poly = 0
        for cid in ids:
            kind, pids = lookup.get(int(cid))
            if kind is None or kind == "strip" or len(pids) == 0:
                continue
            if kind == "polygon":
                n_poly += 1
                if n_poly > MAX_POLY_CELLS or len(pids) < 3:
                    continue
            else:
                n_line += 1
                if n_line > MAX_LINE_CELLS:
                    continue
            d, depth, w, local = _evaluate(kind, pids, np.asarray(pts[pids], dtype=np.float64), proj, click)
            if d <= radius:
                cands.append((d, depth, PickHit(layer, actor, mesh, int(cid), kind, pids, w,
                                                np.asarray(local), d)))
    return _choose(cands)


DEPTH_EPS = 2e-3  # normalised depth: lines this close to a surface count as lying on it


def _choose(cands: list[tuple[float, float, PickHit]]) -> PickHit | None:
    """Best candidate: lines / points near the cursor win over the surface under it, unless
    the surface hides them; otherwise the front-most surface; otherwise the nearest thing."""
    if not cands:
        return None
    front = min((z for d, z, h in cands if h.kind == "polygon" and d == 0.0), default=np.inf)
    visible = [c for c in cands if c[2].kind == "polygon" or c[1] <= front + DEPTH_EPS]

    def key(c):
        d, z, h = c
        if d > SNAP:
            return (1, d, z)
        # Within snap distance: the closest line / point first, then surfaces front to back.
        # A point marker gets a small head start, so clicking a string vertex picks the point.
        if h.kind == "polygon":
            return (0, SNAP, z)
        return (0, max(0.0, d - POINT_BONUS) if h.kind == "vertex" else d, z)

    return min(visible or cands, key=key)[2]
