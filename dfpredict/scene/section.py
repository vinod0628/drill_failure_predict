"""Cross-sections: cut the scene along a line drawn in a 2D view.

The section line is a polyline drawn in the plane of a 2D view (plan XY, or the
XZ / YZ sections). The section surface is that line swept along the view
direction (``axis``): a vertical curtain for a line drawn in plan. Each line
segment A->B is one flat panel of the curtain.

Section coordinates: ``u`` is the distance along the line from its first point
(chainage), ``v`` the coordinate along the view direction (elevation for a
plan line). Everything here works in scene-local coordinates; the section view
adds the datum back for its labels.

What each layer contributes:

* polygons (surfaces / solids): the lines where the panel cuts them;
* lines and points (strings, drillholes): the parts within ``half_width`` of the
  panel (the corridor), projected onto it;
* block models: every shown block the panel passes through, as a filled
  rectangle (the cut of an axis-aligned box by a panel parallel to ``axis``).

The cut geometry keeps the source mesh's cell / point arrays, so it can be drawn
with a copy of the source mapper: same colours, colour map and range as the
main view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from ..i18n import N_

AXIS_NAMES = (N_("X (East)"), N_("Y (North)"), N_("Z (Elevation)"))   # shown as tr(AXIS_NAMES[i])
START_MARK, END_MARK = "A", "A'"          # section line ends: A - A'
SECTION_NAME = f"{START_MARK} - {END_MARK}"

# Draw order in the section view (z of the flat geometry; the camera looks down -z).
DEPTH_BLOCKS, DEPTH_SURFACE, DEPTH_LINES, DEPTH_POINTS = 0.0, 1.0, 2.0, 3.0


@dataclass
class Panel:
    """One straight segment of the section line."""

    a: np.ndarray        # start (scene-local)
    t: np.ndarray        # unit direction along the segment (perpendicular to the axis)
    n: np.ndarray        # unit normal of the panel (perpendicular to t and the axis)
    length: float
    u0: float            # chainage at the start


@dataclass
class SectionLine:
    """A section polyline drawn in a 2D view; ``axis`` is the view direction (0 X, 1 Y, 2 Z)."""

    points: np.ndarray   # (k, 3) scene-local vertices
    axis: int
    panels: list[Panel] = field(init=False, default_factory=list)

    def __post_init__(self):
        pts = np.asarray(self.points, dtype=np.float64).reshape(-1, 3).copy()
        pts[:, self.axis] = 0.0          # only the in-plane position matters
        self.points, _removed = remove_duplicate_points(pts)
        e = np.zeros(3)
        e[self.axis] = 1.0
        self.panels = []
        u = 0.0
        for a, b in zip(self.points[:-1], self.points[1:]):
            d = b - a
            length = float(np.linalg.norm(d))
            t = d / length
            n = np.cross(t, e)
            self.panels.append(Panel(a, t, n / np.linalg.norm(n), length, u))
            u += length

    @property
    def valid(self) -> bool:
        return len(self.panels) > 0

    @property
    def length(self) -> float:
        return sum(p.length for p in self.panels)

    @property
    def chainages(self) -> np.ndarray:
        """Chainage of every vertex (bends included)."""
        return np.array([0.0] + [p.u0 + p.length for p in self.panels])

    @property
    def in_plane_axes(self) -> tuple[int, int]:
        return tuple(i for i in range(3) if i != self.axis)

    def bearing(self) -> float | None:
        """Plan bearing of the first-to-last direction (clockwise from north), plan sections only."""
        if self.axis != 2 or len(self.points) < 2:
            return None
        d = self.points[-1] - self.points[0]
        return math.degrees(math.atan2(d[0], d[1])) % 360.0


DUPLICATE_TOL = 0.05     # metres: a segment shorter than this means a duplicated node


def remove_duplicate_points(points, tol: float = DUPLICATE_TOL) -> tuple[np.ndarray, int]:
    """Drop nodes that repeat the node before them: the segment to it is shorter than ``tol``
    (0.05 m), i.e. (nearly) zero length. Works on (k, 2) or (k, 3) arrays; returns (points, number removed)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy(), 0
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= tol:
            keep.append(i)
    return pts[keep], len(pts) - len(keep)


def duplicate_rows(points, tol: float = DUPLICATE_TOL) -> list[int]:
    """Indices of nodes whose segment from the node before is shorter than ``tol``."""
    pts = np.asarray(points, dtype=np.float64)
    return [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) < tol]


# ---------------------------------------------------------------------------------
def to_section(points: np.ndarray, panel: Panel, axis: int, depth: float) -> np.ndarray:
    """Scene-local 3D points (on or near ``panel``) -> flat section points (u, v, depth)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.empty((len(p), 3))
    out[:, 0] = panel.u0 + (p - panel.a) @ panel.t
    out[:, 1] = p[:, axis]
    out[:, 2] = depth
    return out


def _panel_planes(panel: Panel, half_width: float | None) -> np.ndarray:
    """Half-spaces ``n . p <= c`` (rows: nx, ny, nz, c) bounding the panel's corridor."""
    t, n, a = panel.t, panel.n, panel.a
    rows = [np.r_[-t, -(t @ a)], np.r_[t, t @ a + panel.length]]
    if half_width is not None:
        rows += [np.r_[n, n @ a + half_width], np.r_[-n, -(n @ a) + half_width]]
    return np.array(rows)


def clip_segments(p0: np.ndarray, p1: np.ndarray, planes: np.ndarray):
    """Clip segments p0->p1 to the convex region ``planes`` (Liang-Barsky).

    Returns (keep mask, t0, t1): the kept part of segment i runs from parameter
    t0[i] to t1[i] (0 = p0, 1 = p1).
    """
    m = len(p0)
    t0, t1 = np.zeros(m), np.ones(m)
    keep = np.ones(m, dtype=bool)
    d = p1 - p0
    for row in planes:
        nrm, c = row[:3], row[3]
        f0 = p0 @ nrm - c            # <= 0 inside
        dn = d @ nrm
        par = np.abs(dn) < 1e-12
        keep &= ~(par & (f0 > 1e-9))
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = -f0 / dn
        entering = ~par & (dn < 0)
        leaving = ~par & (dn > 0)
        t0 = np.where(entering, np.maximum(t0, tt), t0)
        t1 = np.where(leaving, np.minimum(t1, tt), t1)
    keep &= t1 >= t0 - 1e-12
    return keep, t0, t1


def points_in(points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return np.all(p @ planes[:, :3].T <= planes[:, 3] + 1e-9, axis=1)


# --- reading VTK cells ---------------------------------------------------------------
def _cells(ca: vtk.vtkCellArray) -> tuple[np.ndarray, np.ndarray]:
    """(offsets, connectivity) of a vtkCellArray as numpy arrays."""
    if ca is None or ca.GetNumberOfCells() == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return (vtk_to_numpy(ca.GetOffsetsArray()).astype(np.int64),
            vtk_to_numpy(ca.GetConnectivityArray()).astype(np.int64))


def line_edges(mesh: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
    """Every 2-point edge of the mesh's line cells: (edges (m, 2) point ids, cell id of each).

    Cell ids count verts first, then lines (VTK's cell order for poly data)."""
    off, conn = _cells(mesh.GetLines())
    if len(conn) < 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    j = np.arange(len(conn) - 1)
    starts = np.zeros(len(conn), dtype=bool)
    starts[off[:-1]] = True
    j = j[~starts[j + 1]]                      # j and j+1 belong to the same cell
    cell = np.searchsorted(off, j, side="right") - 1
    n_verts = mesh.GetVerts().GetNumberOfCells() if mesh.GetVerts() is not None else 0
    return np.column_stack([conn[j], conn[j + 1]]), cell + n_verts


def vert_points(mesh: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
    """Point ids of the vertex cells and the cell id of each."""
    off, conn = _cells(mesh.GetVerts())
    if not len(conn):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cell = np.searchsorted(off, np.arange(len(conn)), side="right") - 1
    return conn, cell


def _copy_arrays(src, dst, index: np.ndarray) -> None:
    for name in src.keys():
        arr = np.asarray(src[name])
        if arr.ndim and len(arr) > (index.max(initial=-1)):
            dst[name] = arr[index]


def _interp_point_arrays(src, dst, i0, i1, w) -> None:
    for name in src.keys():
        arr = np.asarray(src[name])
        if not arr.ndim:
            continue
        a, b = arr[i0], arr[i1]
        if arr.dtype.kind in "fc":
            ww = w.reshape((-1,) + (1,) * (arr.ndim - 1))
            dst[name] = (a + (b - a) * ww).astype(arr.dtype)
        else:  # integer / colour data: take the nearer end
            dst[name] = np.where((w < 0.5).reshape((-1,) + (1,) * (arr.ndim - 1)), a, b)


# --- cutting -------------------------------------------------------------------------
def cut_lines(mesh: pv.PolyData, panel: Panel, axis: int, half_width: float | None,
              depth: float = DEPTH_LINES) -> pv.PolyData | None:
    """Line cells of ``mesh`` clipped to the panel's corridor, flattened to section coords.

    The result is a set of 2-point lines carrying the source cell and point data."""
    edges, cell = line_edges(mesh)
    if not len(edges):
        return None
    pts = np.asarray(mesh.points, dtype=np.float64)
    p0, p1 = pts[edges[:, 0]], pts[edges[:, 1]]
    keep, t0, t1 = clip_segments(p0, p1, _panel_planes(panel, half_width))
    if not keep.any():
        return None
    p0, p1, t0, t1 = p0[keep], p1[keep], t0[keep], t1[keep]
    e, cell = edges[keep], cell[keep]
    d = p1 - p0
    a, b = p0 + d * t0[:, None], p0 + d * t1[:, None]
    m = len(a)
    flat = np.empty((2 * m, 3))
    flat[0::2] = to_section(a, panel, axis, depth)
    flat[1::2] = to_section(b, panel, axis, depth)
    lines = np.column_stack([np.full(m, 2), np.arange(0, 2 * m, 2), np.arange(1, 2 * m, 2)]).ravel()
    out = pv.PolyData(flat.astype(np.float32), lines=lines)
    _copy_arrays(mesh.cell_data, out.cell_data, cell)
    w = np.empty(2 * m)
    w[0::2], w[1::2] = t0, t1
    i0 = np.repeat(e[:, 0], 2)
    i1 = np.repeat(e[:, 1], 2)
    _interp_point_arrays(mesh.point_data, out.point_data, i0, i1, w)
    return out


def cut_points(mesh: pv.PolyData, panel: Panel, axis: int, half_width: float,
               depth: float = DEPTH_POINTS) -> pv.PolyData | None:
    """Vertex cells of ``mesh`` inside the corridor, projected onto the panel."""
    ids, cell = vert_points(mesh)
    if not len(ids):
        return None
    pts = np.asarray(mesh.points, dtype=np.float64)[ids]
    inside = points_in(pts, _panel_planes(panel, half_width))
    if not inside.any():
        return None
    ids, cell = ids[inside], cell[inside]
    out = pv.PolyData(to_section(pts[inside], panel, axis, depth).astype(np.float32))
    _copy_arrays(mesh.point_data, out.point_data, ids)
    _copy_arrays(mesh.cell_data, out.cell_data, cell)
    return out


def cut_polygons(mesh: pv.PolyData, panel: Panel, axis: int,
                 depth: float = DEPTH_SURFACE) -> pv.PolyData | None:
    """Where the panel cuts the polygons of ``mesh`` (surfaces / solids): section lines."""
    if mesh.GetPolys() is None or mesh.GetPolys().GetNumberOfCells() == 0:
        return None
    bounds_ok = _bounds_reach(mesh, panel)
    if not bounds_ok:
        return None
    only = pv.PolyData()
    only.SetPoints(mesh.GetPoints())
    only.SetPolys(mesh.GetPolys())
    for name in mesh.cell_data.keys():       # polygon cells only: drop verts / lines data
        arr = np.asarray(mesh.cell_data[name])
        n_other = mesh.n_cells - mesh.GetPolys().GetNumberOfCells()
        only.cell_data[name] = arr[n_other:] if n_other else arr
    for name in mesh.point_data.keys():
        only.point_data[name] = np.asarray(mesh.point_data[name])
    plane = vtk.vtkPlane()
    plane.SetOrigin(*panel.a)
    plane.SetNormal(*panel.n)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(only)
    cutter.GenerateTrianglesOff()
    cutter.Update()
    lines = pv.wrap(cutter.GetOutput())
    if lines.n_points == 0:
        return None
    return cut_lines(lines, panel, axis, None, depth)


def _bounds_reach(mesh, panel: Panel) -> bool:
    """Quick reject: does the mesh's bounding box reach the panel's plane at all?"""
    b = mesh.bounds
    corners = np.array([[x, y, z] for x in b[0:2] for y in b[2:4] for z in b[4:6]])
    d = (corners - panel.a) @ panel.n
    return bool(d.min() <= 1e-9 and d.max() >= -1e-9)


def block_rectangles(centroids_local: np.ndarray, sizes: np.ndarray, panel: Panel, axis: int,
                     depth: float = DEPTH_BLOCKS) -> tuple[np.ndarray, pv.PolyData | None]:
    """Blocks the panel passes through, as rectangles (u from..to, v bottom..top).

    Returns (indices of those blocks, quad mesh with one cell per block in that order).
    """
    i1, i2 = [i for i in range(3) if i != axis]
    a, t = panel.a, panel.t
    c = np.asarray(centroids_local, dtype=np.float64)
    h = np.asarray(sizes, dtype=np.float64) / 2.0
    # Quick reject on the distance from the panel's plane.
    reach = np.abs(panel.n[i1]) * h[:, i1] + np.abs(panel.n[i2]) * h[:, i2]
    dist = (c[:, i1] - a[i1]) * panel.n[i1] + (c[:, i2] - a[i2]) * panel.n[i2]
    cand = np.flatnonzero(np.abs(dist) < reach)
    if not len(cand):
        return cand, None
    c, h = c[cand], h[cand]
    s0 = np.full(len(cand), 0.0)
    s1 = np.full(len(cand), panel.length)
    for i in (i1, i2):
        lo, hi = c[:, i] - h[:, i] - a[i], c[:, i] + h[:, i] - a[i]
        if abs(t[i]) < 1e-12:
            ok = (lo <= 0) & (hi >= 0)
            s1 = np.where(ok, s1, -1.0)
            continue
        ta, tb = lo / t[i], hi / t[i]
        s0 = np.maximum(s0, np.minimum(ta, tb))
        s1 = np.minimum(s1, np.maximum(ta, tb))
    ok = s1 - s0 > 1e-6
    if not ok.any():
        return cand[:0], None
    cand, c, h, s0, s1 = cand[ok], c[ok], h[ok], s0[ok], s1[ok]
    m = len(cand)
    u0, u1 = panel.u0 + s0, panel.u0 + s1
    v0, v1 = c[:, axis] - h[:, axis], c[:, axis] + h[:, axis]
    pts = np.empty((4 * m, 3))
    pts[0::4, 0], pts[0::4, 1] = u0, v0
    pts[1::4, 0], pts[1::4, 1] = u1, v0
    pts[2::4, 0], pts[2::4, 1] = u1, v1
    pts[3::4, 0], pts[3::4, 1] = u0, v1
    pts[:, 2] = depth
    k = np.arange(0, 4 * m, 4)
    faces = np.column_stack([np.full(m, 4), k, k + 1, k + 2, k + 3]).ravel()
    return cand, pv.PolyData(pts.astype(np.float32), faces=faces)


# --- drawing with the source mapper ----------------------------------------------------
def clone_actor(src: vtk.vtkActor, mesh: vtk.vtkPolyData, line_width: float | None = None,
                lit: bool = False, scale: float = 1.0) -> vtk.vtkActor:
    """A new actor drawing ``mesh`` the way ``src`` is drawn (colour mapping and style).

    ``lit``: keep the source lighting and opacity (otherwise flat, opaque colours).
    ``scale``: factor for line widths and point sizes (prints)."""
    sm = src.GetMapper()
    # Mappers in "use cell / point data" mode colour by the *active* scalars: make the same
    # arrays active on the copy (the arrays were copied over, but not which one is active).
    src_in = sm.GetInput()
    if src_in is not None:
        for get in ("GetCellData", "GetPointData"):
            s = getattr(src_in, get)().GetScalars()
            if s is not None and s.GetName() and getattr(mesh, get)().HasArray(s.GetName()):
                getattr(mesh, get)().SetActiveScalars(s.GetName())
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(mesh)
    m.SetScalarVisibility(sm.GetScalarVisibility())
    m.SetScalarMode(sm.GetScalarMode())
    m.SetColorMode(sm.GetColorMode())
    name = sm.GetArrayName()
    if name:
        m.SelectColorArray(name)
    m.SetLookupTable(sm.GetLookupTable())
    m.SetScalarRange(sm.GetScalarRange())
    m.SetUseLookupTableScalarRange(sm.GetUseLookupTableScalarRange())
    m.SetInterpolateScalarsBeforeMapping(sm.GetInterpolateScalarsBeforeMapping())
    a = vtk.vtkActor()
    a.SetMapper(m)
    p = a.GetProperty()
    p.DeepCopy(src.GetProperty())
    if not lit:
        p.LightingOff()      # flat colours: the section is a drawing, not a lit scene
        p.SetOpacity(1.0)
    p.SetRepresentationToSurface()
    if line_width is not None:
        p.SetLineWidth(max(p.GetLineWidth(), line_width))
    if scale != 1.0:
        p.SetLineWidth(p.GetLineWidth() * scale)
        p.SetPointSize(p.GetPointSize() * scale)
    a.PickableOff()
    return a


# --- the whole scene ------------------------------------------------------------------
@dataclass
class SectionResult:
    actors: list = field(default_factory=list)          # vtkActors in section coordinates
    labels: list = field(default_factory=list)          # (u, v, depth, text): drillhole names
    scalar_bars: list = field(default_factory=list)     # (title, lookup table)
    legends: list = field(default_factory=list)         # (title, [(text, rgb)]): drillhole codes
    counts: dict = field(default_factory=dict)          # layer name -> pieces drawn
    bounds: tuple | None = None                         # (u0, u1, v0, v1) of the drawn geometry

    @property
    def empty(self) -> bool:
        return not self.actors


def _merge_bounds(b, mesh):
    mb = mesh.bounds
    nb = (mb[0], mb[1], mb[2], mb[3])
    if b is None:
        return nb
    return (min(b[0], nb[0]), max(b[1], nb[1]), min(b[2], nb[2]), max(b[3], nb[3]))


def build_section(scene, line: SectionLine, half_width: float) -> SectionResult:
    """Cut every visible layer of ``scene`` along ``line``."""
    from .layers import BlockModelLayer, DrillholeLayer

    res = SectionResult()
    if not line.valid or scene.origin is None:
        return res
    ax = line.axis
    for layer in scene.layers:
        if not layer.visible:
            continue
        pieces = 0
        if isinstance(layer, BlockModelLayer):
            src = next((a for a in layer.actors if isinstance(a, vtk.vtkActor)), None)
            if src is None:
                continue
            mask = layer.visible_mask()
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            # mined blocks shown gray / transparent: drawn like their actor in the view
            mined_src = getattr(layer, "_mined_actor", None)
            mined = layer.mined_mask() if mined_src is not None else None
            groups = [(idx, src, layer.block_cell_data)]
            if mined is not None:
                groups = [(idx[~mined[idx]], src, layer.block_cell_data),
                          (idx[mined[idx]], mined_src, layer.mined_cell_data)]
            for gidx, gsrc, cell_data in groups:
                if not len(gidx):
                    continue
                cen = layer.data.centroids[gidx] - scene.origin
                sizes = layer.data.sizes[gidx]
                for panel in line.panels:
                    which, rect = block_rectangles(cen, sizes, panel, ax)
                    if rect is None:
                        continue
                    for key, arr in cell_data(gidx[which]).items():
                        rect.cell_data[key] = arr
                    res.actors.append(clone_actor(gsrc, rect))
                    res.bounds = _merge_bounds(res.bounds, rect)
                    pieces += 1
            if pieces and layer.color_by and not layer.data.is_categorical(layer.color_by):
                res.scalar_bars.append((layer.color_by, src.GetMapper().GetLookupTable()))
            res.counts[layer.name] = pieces
            continue

        hole_top: dict[int, tuple[float, float]] = {}
        rows_cut: list[np.ndarray] = []          # drillhole interval rows in the section
        interval_actor = None
        for k, actor in enumerate(layer.actors):
            if not isinstance(actor, vtk.vtkActor) or not actor.GetVisibility():
                continue
            # Later actors of a layer draw over earlier ones (drillhole intervals over the trace).
            dz = 0.05 * k
            # drillholes drawn as cylinders keep their path: the section shows the hole as a thick line
            # along it, projected from the whole corridor (not the outline where the plane cuts the cylinder)
            path = getattr(actor, "_gv_lines", None)
            data = path if path is not None else (actor.GetMapper().GetInput() if actor.GetMapper() is not None
                                                  else None)
            if not isinstance(data, vtk.vtkPolyData):
                continue
            mesh = pv.wrap(data)
            width = getattr(actor, "_gv_section_width", None)
            for panel in line.panels:
                outs = [None if path is not None else cut_polygons(mesh, panel, ax, DEPTH_SURFACE + dz),
                        cut_lines(mesh, panel, ax, half_width, DEPTH_LINES + dz),
                        cut_points(mesh, panel, ax, half_width, DEPTH_POINTS + dz)]
                for kind, out in zip(("poly", "line", "point"), outs):
                    if out is None or out.n_points == 0:
                        continue
                    res.actors.append(clone_actor(actor, out, 2.0 if kind == "poly" else width))
                    res.bounds = _merge_bounds(res.bounds, out)
                    pieces += 1
                    if kind == "line" and "gv_hole" in out.cell_data:
                        _hole_tops(out, hole_top)
                    if kind == "line" and "gv_row" in out.cell_data:
                        rows_cut.append(np.asarray(out.cell_data["gv_row"]))
                        interval_actor = actor
        if isinstance(layer, DrillholeLayer) and rows_cut:
            _interval_legend(layer, np.concatenate(rows_cut), interval_actor, res)
        if isinstance(layer, DrillholeLayer) and layer.show_labels:
            for h, (u, v) in hole_top.items():
                if 0 <= h < len(layer._trace_holes):
                    res.labels.append((u, v, DEPTH_POINTS + 1.0, layer._trace_holes[h]))
        if pieces:
            res.counts[layer.name] = pieces
    return res


def _interval_legend(layer, rows: np.ndarray, actor, res: SectionResult) -> None:
    """Legend of the drillhole interval field for the intervals in the section: the codes
    present (with descriptions) for a categorical field, a colour bar for a numeric one."""
    if not (layer.table and layer.field_name):
        return
    title = f"{layer.field_name} ({layer.table})"
    if actor.GetMapper().GetArrayName() == layer.field_name:     # numeric: colour map
        res.scalar_bars.append((title, actor.GetMapper().GetLookupTable()))
        return
    col = layer.data.intervals[layer.table][layer.field_name]
    codes = col.iloc[np.unique(rows)]
    codes = sorted({str(c) for c in codes if c == c and str(c) != ""})
    desc = layer.data.code_descriptions(layer.table, layer.field_name)
    entries = []
    for c in codes:
        text = f"{c}  {desc[c]}" if desc.get(c) else c
        entries.append((text, tuple(layer.code_color(c))))
    if entries:
        res.legends.append((title, entries))


def _hole_tops(out: pv.PolyData, tops: dict) -> None:
    """Highest point of each drillhole in a cut piece (where its name is shown)."""
    holes = np.asarray(out.cell_data["gv_hole"])
    pts = np.asarray(out.points)
    ends = pts.reshape(-1, 2, 3)            # cut_lines makes 2-point lines in order
    best = np.where((ends[:, 0, 1] >= ends[:, 1, 1])[:, None], ends[:, 0], ends[:, 1])
    for h in np.unique(holes):
        sel = best[holes == h]
        k = int(np.argmax(sel[:, 1]))
        u, v = float(sel[k, 0]), float(sel[k, 1])
        if int(h) not in tops or v > tops[int(h)][1]:
            tops[int(h)] = (u, v)
