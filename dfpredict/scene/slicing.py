"""3D slice of the main view: every object is shown only between two parallel planes.

The slab is set along one axis (the plane it is parallel to: XY - between two elevations, XZ - between two northings,
YZ - between two eastings), from ``lo`` to ``hi`` in world coordinates:

- surfaces, solids, drillholes, grids, points, MT sections ... are cut by two clipping planes on their mappers (the GPU
  drops what lies outside; moving the planes is instant);
- block models show whole blocks: the blocks whose centres lie in the slab (``Scene.slab_mask``), so the slab's faces
  show the blocks' own colours; until a block model is drawn again for the new slab its faces are clipped too;
- solids get section faces at both planes (their cut, filled in the solid's colour), open surfaces (topography, faults)
  a trace line there, so the two sections are seen in the 3D view itself;
- labels (hole names, ...) are hidden while slicing: they are not cut by the planes.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

from ..i18n import N_

PLANES = {"XY": 2, "XZ": 1, "YZ": 0}          # plane -> the axis the slab is measured along
AXIS_NAMES = ("X", "Y", "Z")
PLANE_TEXT = {"XY": N_("XY - between two elevations (Z)"),          # shown: tr()
              "XZ": N_("XZ - between two northings (Y)"),
              "YZ": N_("YZ - between two eastings (X)")}
TRACE_WIDTH = 3.0


def plane_of(axis: int) -> str:
    return next(p for p, a in PLANES.items() if a == axis)


class Slicer:
    def __init__(self, scene):
        self.scene = scene
        self.axis: int | None = None
        self.lo = self.hi = 0.0                  # world coordinates along the axis
        self.faces = True                        # section faces of solids at both planes
        self._planes: vtk.vtkPlaneCollection | None = None
        self._face_actors: list = []

    # --- state ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.axis is not None

    def state(self) -> dict | None:
        if not self.active:
            return None
        return {"plane": plane_of(self.axis), "from": self.lo, "to": self.hi, "faces": self.faces}

    def set(self, axis: int, lo: float, hi: float, faces: bool | None = None, update: bool = True,
            render: bool = True) -> None:
        """Show only what lies between ``lo`` and ``hi`` (world) along ``axis`` (0 X, 1 Y, 2 Z). ``update`` False:
        only the clipping planes move (fast, while a slider is dragged); call :meth:`update` when it stops."""
        lo, hi = sorted((float(lo), float(hi)))
        was = self.active
        self.axis, self.lo, self.hi = int(axis), lo, hi
        if faces is not None:
            self.faces = bool(faces)
        n = np.zeros(3)
        n[self.axis] = 1.0
        point = np.zeros(3)
        point[self.axis] = 0.5 * (lo + hi)
        self.scene.slab = (point, n, 0.5 * (hi - lo))
        self._planes = self._make_planes()
        self.clip_all()
        if not was:
            self._labels()
        if update:
            self.update(render=False)
        if render:
            self.scene.render()

    def clear(self, render: bool = True) -> None:
        """Show everything again."""
        was = self.active
        self.axis = None
        self.scene.slab = None
        self._planes = None
        self._remove_faces()
        if was:
            self.clip_all()
            self._labels()
            self._redraw_blocks()
        if render:
            self.scene.render()

    def forget(self) -> None:
        """The scene was cleared (its actors are gone): drop the state without touching them."""
        self.axis = None
        self._planes = None
        self._face_actors = []

    # --- clipping ---------------------------------------------------------------
    def _make_planes(self) -> vtk.vtkPlaneCollection:
        """Two planes in scene-local coordinates; VTK keeps the side each normal points to."""
        pc = vtk.vtkPlaneCollection()
        o = self.scene.origin if self.scene.origin is not None else np.zeros(3)
        for value, sign in ((self.lo, 1.0), (self.hi, -1.0)):
            p = np.zeros(3)
            p[self.axis] = value - o[self.axis]
            n = np.zeros(3)
            n[self.axis] = sign
            pl = vtk.vtkPlane()
            pl.SetOrigin(*p)
            pl.SetNormal(*n)
            pc.AddItem(pl)
        return pc

    def clip_all(self) -> None:
        for layer in self.scene.layers:
            self.clip_layer(layer)

    def clip_layer(self, layer) -> None:
        """Clipping planes on the layer's 3D actors (called after every rebuild of a layer). A block model drawn for
        the current slab is not clipped: it shows whole blocks."""
        from .layers import BlockModelLayer

        planes = self._planes
        if planes is not None and isinstance(layer, BlockModelLayer) and getattr(layer, "_built_slab", None) is \
                self.scene.slab:
            planes = None
        for a in layer.actors:
            if not isinstance(a, vtk.vtkActor):
                continue
            m = a.GetMapper()
            if m is None:
                continue
            if planes is None:
                m.RemoveAllClippingPlanes()
            else:
                m.SetClippingPlanes(planes)

    def _labels(self) -> None:
        for layer in self.scene.layers:
            layer.apply_visibility()            # hides / shows the label actors (Layer.apply_visibility)

    # --- block models and section faces -----------------------------------------
    def update(self, render: bool = True) -> None:
        """Draw the block models for the current slab and the section faces at both planes."""
        self._redraw_blocks(visible_only=True)
        self._build_faces()
        if render:
            self.scene.render()

    def _redraw_blocks(self, visible_only: bool = False) -> None:
        from .layers import BlockModelLayer

        for layer in self.scene.layers:
            if not isinstance(layer, BlockModelLayer) or getattr(layer, "_built_slab", None) is self.scene.slab:
                continue
            if visible_only and not layer.visible:
                continue                         # drawn when it is shown again (update after the layer change)
            layer.rebuild()

    def _remove_faces(self) -> None:
        for a in self._face_actors:
            self.scene.plotter.remove_actor(a, render=False)
        self._face_actors = []

    def _build_faces(self) -> None:
        from .layers import SurfaceLayer

        self._remove_faces()
        if not self.active or not self.faces:
            return
        o = self.scene.origin if self.scene.origin is not None else np.zeros(3)
        for layer in self.scene.layers:
            if not isinstance(layer, SurfaceLayer) or not layer.visible or not layer.actors:
                continue
            mesh = pv.wrap(layer.actors[0].GetMapper().GetInput())
            if mesh is None or mesh.n_cells == 0:
                continue
            closed = _closed_cells(layer, mesh)
            for value in (self.lo, self.hi):
                p = np.zeros(3)
                p[self.axis] = value - o[self.axis]
                for poly, is_fill in section_faces(mesh, self.axis, p[self.axis], closed):
                    kw = dict(scalars="rgb", rgb=True, opacity=layer.opacity, pickable=False, render=False,
                              reset_camera=False, show_scalar_bar=False,
                              name=f"slice-face-{id(layer)}-{value}-{int(is_fill)}")
                    if is_fill:
                        kw.update(ambient=0.35, diffuse=0.65, specular=0.0)
                    else:
                        kw.update(line_width=TRACE_WIDTH, lighting=False)
                    self._face_actors.append(self.scene.plotter.add_mesh(poly, **kw))

    # --- camera -----------------------------------------------------------------
    def content_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """World bounds of what the slab shows: the visible layers reaching into it, cut to the slab along its axis.
        Regional 2D grids (geophysics, DEM) count only when nothing else is in the slab - they would make the model
        a speck."""
        Grid2DLayer = ()        # no regional 2D grid layers in this application

        if not self.active:
            return None
        a = self.axis
        o = self.scene.origin if self.scene.origin is not None else np.zeros(3)
        model, regional = [], []
        for layer in self.scene.layers:
            if not layer.visible:
                continue
            b = layer.world_bounds()
            if b is None or b[1][a] < self.lo or b[0][a] > self.hi:
                continue
            inside = []                          # the drawn points in the slab (not the whole layer)
            for act in layer.actors:
                if isinstance(act, vtk.vtkActor) and act.GetMapper() is not None:
                    data = act.GetMapper().GetInput()
                    if data is not None and data.GetNumberOfPoints():
                        p = np.asarray(pv.wrap(data).points, dtype=np.float64) + o
                        inside.append(p[(p[:, a] >= self.lo) & (p[:, a] <= self.hi)])
            inside = [p for p in inside if len(p)]
            if inside:
                p = np.concatenate(inside)
                b = (p.min(axis=0), p.max(axis=0))
            (regional if isinstance(layer, Grid2DLayer or ()) else model).append(b)
        boxes = model or regional
        if not boxes:
            return None
        lo = np.min([b[0] for b in boxes], axis=0).astype(np.float64)
        hi = np.max([b[1] for b in boxes], axis=0).astype(np.float64)
        lo[a], hi[a] = self.lo, self.hi
        return lo, hi

    def zoom_to(self) -> None:
        """Fit the view to what the slab shows, keeping the view direction."""
        b = self.content_bounds()
        if b is None:
            return
        lo, hi = self.scene.to_local(b[0]), self.scene.to_local(b[1])
        self.scene.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        self.scene.render()

    def view_face(self, side: str) -> None:
        """Look at the section at ``side`` ("from" or "to") from outside the slab, square to it."""
        if not self.active:
            return
        sc = self.scene
        b = self.content_bounds()
        if b is None:
            return
        lo, hi = (sc.to_local(b[0]), sc.to_local(b[1]))
        a = self.axis
        c = 0.5 * (lo + hi)
        sign = -1.0 if side == "from" else 1.0
        c[a] = lo[a] if side == "from" else hi[a]
        d = float(np.linalg.norm(hi - lo)) or 10.0
        e = np.zeros(3)
        e[a] = 1.0
        from .scene import VIEW_3D

        if sc.view != VIEW_3D:
            sc.set_view(VIEW_3D)
        cam = sc.plotter.camera
        cam.focal_point = tuple(c)
        cam.position = tuple(c + sign * d * e)
        cam.up = (0.0, 1.0, 0.0) if a == 2 else (0.0, 0.0, 1.0)
        sc.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        sc.render()


def watertight(triangles: np.ndarray) -> bool:
    """Every edge is shared by exactly two triangles (a closed solid, whatever its file says)."""
    t = np.asarray(triangles)
    if t.ndim != 2 or len(t) < 4:
        return False
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _closed_cells(layer, mesh: pv.PolyData) -> np.ndarray:
    """Per cell: the triangle belongs to a closed trisolation (a solid)."""
    out = np.zeros(mesh.n_cells, dtype=bool)
    try:
        closed = {(obj.id, tri.id) for obj, tri in layer.data.iter_trisolations()
                  if tri.closed or watertight(tri.triangles)}
        obj = np.asarray(mesh.cell_data["object"])
        tid = np.asarray(mesh.cell_data["trisolation"])
    except (KeyError, AttributeError):
        return out
    for o_, t_ in closed:
        out |= (obj == o_) & (tid == t_)
    return out


def section_faces(mesh: pv.PolyData, axis: int, value: float, closed: np.ndarray | None = None):
    """The cut of a triangle mesh at ``value`` (scene-local) along ``axis``: [(polydata, is_fill)]. Closed parts are
    filled (a polygon per solid, in its colour), open parts give trace lines. Uses the mesh's ``rgb`` cell colours."""
    lo, hi = mesh.bounds[2 * axis], mesh.bounds[2 * axis + 1]
    if not (lo <= value <= hi):
        return []
    work = mesh.copy(deep=False)
    if "rgb" not in work.cell_data:
        work.cell_data["rgb"] = np.full((work.n_cells, 3), 160, dtype=np.uint8)
    work.cell_data["_closed"] = (closed if closed is not None and len(closed) == work.n_cells
                                 else np.zeros(work.n_cells, dtype=bool)).astype(np.uint8)
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    origin = [0.0, 0.0, 0.0]
    origin[axis] = value
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(work)
    cutter.Update()
    cut = pv.wrap(cutter.GetOutput())
    if cut is None or cut.n_cells == 0:
        return []
    rgb = np.asarray(cut.cell_data["rgb"])
    is_closed = np.asarray(cut.cell_data["_closed"]).astype(bool)
    out = []
    if (~is_closed).any():                       # open surfaces: the trace
        tr_ = cut.extract_cells(np.flatnonzero(~is_closed)).extract_surface(algorithm="dataset_surface")
        if tr_.n_cells:
            out.append((tr_, False))
    fills = []
    if is_closed.any():                          # solids: fill the cut, one colour at a time
        keys = rgb[is_closed]
        for colour in np.unique(keys, axis=0):
            sel = np.flatnonzero(is_closed & np.all(rgb == colour, axis=1))
            part = cut.extract_cells(sel).extract_surface(algorithm="dataset_surface")
            tri = vtk.vtkContourTriangulator()
            tri.SetInputData(part)
            tri.TriangulationErrorDisplayOff()
            tri.Update()
            poly = pv.wrap(tri.GetOutput())
            if poly is None or poly.n_cells == 0:
                continue
            poly.cell_data["rgb"] = np.tile(colour.astype(np.uint8), (poly.n_cells, 1))
            fills.append(poly)
    if fills:
        out.insert(0, (fills[0].append_polydata(*fills[1:]) if len(fills) > 1 else fills[0], True))
    return out
