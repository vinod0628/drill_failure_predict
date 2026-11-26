"""Display layers: each wraps loaded data and builds its VTK actors.

All geometry is shifted by the scene datum (``scene.origin``) before it goes
to the GPU, because float32 vertex positions jitter at mining-grid
coordinates (e.g. Y = 7,000,000). Values shown to the user add it back.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyvista as pv
import vtk

from ..i18n import N_, tr
from ..io.block_model import BlockModel
from ..io.drillholes import DrillholeDB
from ..io.surpac_dtm import DtmFile
from ..io.surpac_ssi import StyleSet
from ..io.surpac_str import StringFile
from .block_faces import exposed_faces, face_quads

if TYPE_CHECKING:
    from .picking import PickHit
    from .scene import Scene

RGB = tuple[float, float, float]
DEPLETION_ATTRS = ("mined_fraction", "status")   # computed by the reserve tools, not in the file

# Distinct default colours for new layers and categorical values.
PALETTE: list[RGB] = [
    (0.96, 0.62, 0.20), (0.30, 0.69, 0.95), (0.45, 0.82, 0.40), (0.93, 0.33, 0.33),
    (0.75, 0.52, 0.95), (0.98, 0.85, 0.30), (0.25, 0.85, 0.80), (0.95, 0.50, 0.75),
    (0.65, 0.65, 0.65), (0.60, 0.80, 0.25), (0.85, 0.55, 0.40), (0.40, 0.55, 0.95),
]
_layer_colors = itertools.cycle(PALETTE)
# Mined shapes, one colour per month: none of them red / orange / grey (the block status colours).
MONTH_COLORS: list[RGB] = [
    (0.30, 0.69, 0.95), (0.45, 0.82, 0.40), (0.75, 0.52, 0.95), (0.98, 0.85, 0.30),
    (0.25, 0.85, 0.80), (0.95, 0.50, 0.75), (0.40, 0.55, 0.95), (0.60, 0.80, 0.25),
]

COLORMAPS = ["turbo", "jet", "viridis", "plasma", "coolwarm", "RdYlGn_r", "Spectral_r"]
# How blocks mined by the recorded cuts are shown (block model property "Mined blocks"). A block
# counts as mined when at least half of it is (mined_fraction >= MINED_SHARE).
MINED_DISPLAYS = {"show": N_("Show"), "gray": N_("Gray"), "transparent": N_("Transparent"),  # shown: tr(text)
                  "hidden": N_("Hidden")}
MINED_SHARE = 0.5
MINED_GRAY = 1.0            # share of gray in a mined block's colour: 1 = plain gray
MINED_OPACITY = 0.25        # opacity of a mined block shown transparent


def _to_u8(rgb) -> np.ndarray:
    return np.clip(np.asarray(rgb, dtype=np.float64) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def file_signature(path: Path) -> tuple[int, int] | None:
    """(modification time ns, size) of a file, or None when it is missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def fmt_num(v: Any, decimals: int = 3) -> str:
    """Readable number: thousands separators, no trailing zeros, '-' for missing."""
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return tr("yes") if v else tr("no")
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        s = str(v)
        return s if s.strip() else "-"
    if not math.isfinite(f):
        return "-"
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    s = f"{f:,.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _xyz_rows(world) -> list[tuple[str, str]]:
    return [("X", f"{world[0]:,.3f}"), ("Y", f"{world[1]:,.3f}"), ("Z", f"{world[2]:,.3f}")]


def _polyline(points: np.ndarray, closed: bool = False) -> pv.PolyData:
    n = len(points)
    ids = list(range(n)) + ([0] if closed and n > 2 else [])
    return pv.PolyData(np.asarray(points, dtype=np.float32), lines=np.array([len(ids)] + ids))


def _box_edges(lo: np.ndarray, hi: np.ndarray) -> pv.PolyData:
    c = np.array([[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    lines = np.array([[2, a, b] for a, b in edges]).ravel()
    return pv.PolyData(c.astype(np.float32), lines=lines)


@dataclass
class IdentifyResult:
    """What Identify shows for a picked object.

    ``rows`` are (label, value) pairs; a row with value None is a section heading.
    ``highlight`` is geometry in scene-local coordinates drawn over the object (lines /
    points), ``fill`` surfaces drawn see-through over it, and ``marker`` the scene-local
    point marked (default: the picked spot).
    """
    title: str
    rows: list[tuple[str, str | None]]
    highlight: pv.PolyData | None = None
    fill: pv.PolyData | None = None
    marker: np.ndarray | None = None
    extra: dict = field(default_factory=dict)   # e.g. {"hole_id": ..., "depth": ...} for drillholes


@dataclass
class HoleLog:
    """Everything the drillhole log window shows for one hole (world coordinates)."""

    hole_id: str
    collar: np.ndarray                  # (3,)
    length: float
    depths: np.ndarray                  # (P,) along the path
    path: np.ndarray                    # (P, 3)
    intervals: list                     # (from, to, code / value text, description, rgb)
    table: str | None = None
    field_name: str | None = None
    collar_azimuth: float | None = None
    collar_dip: float | None = None
    dirs: np.ndarray | None = None      # (P, 3) unit direction (E, N, Up) at each of ``depths``
    # numeric fields of the hole's interval tables (assays): name -> [(from, to, value), ...]
    grades: dict = field(default_factory=dict)

    def azimuth_dip(self, depth: float) -> tuple[float, float] | None:
        """Azimuth (clockwise from north) and dip (internal: negative down; shown with
        ``display_dip``: 0 = straight down) of the hole at ``depth``."""
        if self.dirs is None or not len(self.dirs):
            return None
        d = np.array([np.interp(depth, self.depths, self.dirs[:, k]) for k in range(3)])
        n = np.linalg.norm(d)
        if n < 1e-12:
            return None
        e, nn, up = d / n
        az = math.degrees(math.atan2(e, nn)) % 360.0 if math.hypot(e, nn) > 1e-9 else 0.0
        return az, math.degrees(math.asin(max(-1.0, min(1.0, up))))


def _watertight(faces: np.ndarray) -> bool:
    """Every edge shared by exactly two triangles (a closed solid)."""
    if len(faces) < 4:
        return False
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _boundary_lines(points: np.ndarray, faces: np.ndarray) -> pv.PolyData | None:
    """Edges used by one triangle only (the outline of an open surface)."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    b = uniq[counts == 1]
    if not len(b):
        return None
    return pv.PolyData(points.astype(np.float32), lines=np.column_stack([np.full(len(b), 2), b]).ravel())


@dataclass(eq=False)
class Layer:
    name: str
    path: Path | None
    kind: str = "layer"
    visible: bool = True
    opacity: float = 1.0
    color: RGB = field(default_factory=lambda: next(_layer_colors))
    actors: list = field(default_factory=list, repr=False)
    scene: "Scene | None" = field(default=None, repr=False)
    # Files the layer was loaded from (passed to the loader again on reload), the part of
    # a multi-layer file (e.g. "faces" of a DXF), and every file the data depends on with
    # its signature at load time; Refresh reloads the layer when any of them changes.
    inputs: list = field(default_factory=list, repr=False)
    part: str = ""
    sources: list = field(default_factory=list, repr=False)
    signature: dict = field(default_factory=dict, repr=False)
    failed_signature: dict = field(default_factory=dict, repr=False)  # files whose reload failed
    source: str = ""                  # "database": read from the project database, not from files
    db_signature: object = field(default=None, repr=False)   # of the database tables when read

    # Display settings saved in a workspace and kept when the layer is reloaded.
    SETTINGS = ("visible", "opacity", "color")

    # --- settings ---------------------------------------------------------
    def settings(self) -> dict:
        out = {}
        for k in self.SETTINGS:
            v = getattr(self, k)
            if isinstance(v, tuple):
                v = [float(x) for x in v]
            elif isinstance(v, dict):
                v = self._dict_setting(k, v)
            elif isinstance(v, (np.floating, np.integer)):
                v = v.item()
            out[k] = v
        return out

    def apply_settings(self, d: dict) -> None:
        """Apply saved settings (before the layer is built); unknown or stale values are ignored."""
        for k in self.SETTINGS:
            if k not in d or d[k] is None and k not in self._nullable:
                continue
            v = d[k]
            if not self._setting_ok(k, v):
                continue
            if k in ("color", "trace_color", "edge_color") or (k == "clim" and v is not None):
                v = tuple(float(x) for x in v)
            elif k == "category_colors":
                v = {str(a): tuple(float(x) for x in b) for a, b in v.items()}
            setattr(self, k, v)

    _nullable = ()   # settings where None is a meaningful saved value

    def _dict_setting(self, key: str, value: dict):
        """A dict setting as saved in the workspace (default: name -> colour)."""
        return {str(a): [float(x) for x in b] for a, b in value.items()}

    def _setting_ok(self, key: str, value) -> bool:
        return True

    @property
    def removable(self) -> bool:
        """Layers made by the app from the database (the drillholes of the project database,
        the mined shapes) stay in the list - they could not be brought back; hide them instead."""
        return self.source != "database" and getattr(self, "kind", "") != "Mined shapes"

    # --- files --------------------------------------------------------------
    def track_files(self, inputs, sources, before: dict | None = None) -> None:
        """Remember where the layer came from. ``before`` holds signatures taken before
        reading, so a file that changes while it is being read is reloaded next time."""
        self.inputs = [Path(p) for p in inputs]
        self.sources = list(dict.fromkeys(Path(p) for p in sources if p is not None))
        before = before or {}
        self.signature = {p: before.get(p) or file_signature(p) for p in self.sources}

    # --- lifecycle -------------------------------------------------------
    def build(self, scene: "Scene") -> None:
        self.scene = scene
        self.rebuild()

    def rebuild(self) -> None:
        self.remove_actors()
        self._build_actors()
        slicer = getattr(self.scene, "slicer", None)
        if slicer is not None:
            slicer.clip_layer(self)              # the 3D slice's clipping planes on the new actors
        self.apply_visibility()

    def _build_actors(self) -> None:
        raise NotImplementedError

    def remove_actors(self) -> None:
        if self.scene is not None:
            for a in self.actors:
                self.scene.plotter.remove_actor(a, render=False)
        self.actors = []

    def _add(self, mesh: pv.DataSet, **kw):
        kw.setdefault("render", False)
        kw.setdefault("reset_camera", False)
        kw.setdefault("show_scalar_bar", False)
        kw.setdefault("name", f"{id(self)}-{len(self.actors)}")
        actor = self.scene.plotter.add_mesh(mesh, **kw)
        actor._geoviewer_layer = self  # used by picking to find the layer
        self.actors.append(actor)
        return actor

    # --- display properties ---------------------------------------------
    def apply_visibility(self) -> None:
        # labels (2D actors) are not cut by the 3D slice's planes: hidden while the view is sliced
        sliced = self.scene is not None and getattr(self.scene, "slab", None) is not None
        for a in self.actors:
            a.SetVisibility(self.visible and not (sliced and isinstance(a, vtk.vtkActor2D)))

    def set_visible(self, v: bool) -> None:
        self.visible = v
        self.apply_visibility()

    # The layer's legend (colour bar) is shown only while it is the active layer in the TOC.
    legend_active = False

    def set_legend_active(self, on: bool) -> None:
        if bool(on) != self.legend_active:
            self.legend_active = bool(on)
            self.apply_visibility()

    def set_opacity(self, v: float) -> None:
        self.opacity = float(v)
        for a in self.actors:
            a.GetProperty().SetOpacity(self.opacity)

    def set_color(self, rgb: RGB) -> None:
        self.color = tuple(rgb)
        self.rebuild()

    # --- info -----------------------------------------------------------
    def world_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        raise NotImplementedError

    def info(self) -> list[tuple[str, str]]:
        rows = [(tr("Type"), tr(self.kind))]     # no file path: the files are in the workspace folder
        b = self.world_bounds()
        if b is not None:
            lo, hi = b
            rows += [
                (tr("X range"), f"{lo[0]:,.2f} .. {hi[0]:,.2f}"),
                (tr("Y range"), f"{lo[1]:,.2f} .. {hi[1]:,.2f}"),
                (tr("Z range"), f"{lo[2]:,.2f} .. {hi[2]:,.2f}"),
            ]
        return rows

    # --- identify --------------------------------------------------------
    def identify(self, hit: "PickHit") -> IdentifyResult:
        return IdentifyResult(tr(self.kind), self._picked_rows(hit))

    def _picked_rows(self, hit: "PickHit") -> list[tuple[str, str | None]]:
        return [(tr("Picked location"), None)] + _xyz_rows(self.scene.to_world(hit.local))

    def _local(self, world) -> np.ndarray:
        return self.scene.to_local(world)


# ---------------------------------------------------------------------------
@dataclass(eq=False)
class StringLayer(Layer):
    data: StringFile | None = None
    styles: StyleSet | None = None
    kind: str = N_("Strings")
    line_width: float = 2.0
    use_styles: bool = True
    show_points: bool = False

    role: str = ""          # geological role (fault, horizon, unconformity, ...; see geo_roles.ROLES)

    SETTINGS = Layer.SETTINGS + ("line_width", "use_styles", "show_points", "role")

    def _segment_colors(self) -> np.ndarray:
        sf = self.data
        if self.use_styles and self.styles is not None:
            cache = {}
            cols = []
            for sn in sf.seg_string:
                if sn not in cache:
                    cache[sn] = _to_u8(self.styles.string_style(int(sn)).line_color)
                cols.append(cache[sn])
            return np.array(cols, dtype=np.uint8).reshape(-1, 3)
        return np.tile(_to_u8(self.color), (sf.n_segments, 1))

    def _build_actors(self) -> None:
        sf = self.data
        if sf.n_segments == 0:
            return
        origin = self.scene.origin
        mask = sf.point_mask
        # Map record index -> compact point index.
        new_index = np.cumsum(mask) - 1
        pts = (sf.records_xyz[mask] - origin).astype(np.float32)

        lengths = sf.seg_end - sf.seg_start
        line_segs = np.flatnonzero(lengths >= 2)
        vert_segs = np.flatnonzero(lengths == 1)
        cols = self._segment_colors()

        if len(line_segs):
            parts = []
            for s in line_segs:
                ids = new_index[sf.seg_start[s]:sf.seg_end[s]]
                parts.append(np.concatenate([[len(ids)], ids]))
            mesh = pv.PolyData(pts, lines=np.concatenate(parts).astype(np.int64))
            mesh.cell_data["rgb"] = cols[line_segs]
            mesh.cell_data["string"] = sf.seg_string[line_segs]
            mesh.cell_data["segment"] = line_segs
            self._add(mesh, scalars="rgb", rgb=True, line_width=self.line_width,
                      opacity=self.opacity, lighting=False)
        if len(vert_segs) or self.show_points:
            segs = np.arange(sf.n_segments) if self.show_points else vert_segs
            ids = np.concatenate([new_index[sf.seg_start[s]:sf.seg_end[s]] for s in segs])
            seg_of_pt = np.concatenate([np.full(sf.seg_end[s] - sf.seg_start[s], s) for s in segs])
            pmesh = pv.PolyData(pts[ids])
            pmesh.point_data["rgb"] = cols[seg_of_pt]
            pmesh.point_data["string"] = sf.seg_string[seg_of_pt]
            pmesh.point_data["record"] = np.concatenate(
                [np.arange(sf.seg_start[s], sf.seg_end[s]) for s in segs])
            self._add(pmesh, scalars="rgb", rgb=True, point_size=max(4.0, self.line_width * 2.5),
                      render_points_as_spheres=True, opacity=self.opacity, lighting=False)

    def world_bounds(self):
        return self.data.bounds() if self.data.point_mask.any() else None

    def info(self):
        sf = self.data
        return super().info() + [
            (tr("Points"), f"{int(sf.point_mask.sum()):,}"),
            (tr("Segments"), f"{sf.n_segments:,}"),
            (tr("String numbers"), f"{len(sf.string_numbers):,}"),
            (tr("Style file"), str(self.styles.path.name) if self.styles and self.styles.path else "-"),
        ]

    def identify(self, hit):
        sf = self.data
        world = self.scene.to_world(hit.local)
        if "segment" in hit.mesh.cell_data:
            s = int(hit.cell_value("segment"))
        else:  # a point marker: identify that one point
            return self._identify_point(hit, int(hit.nearest_point_value("record")))
        start, end = int(sf.seg_start[s]), int(sf.seg_end[s])
        pts = sf.records_xyz[start:end]
        k = int(np.argmin(np.linalg.norm(pts - world, axis=1)))
        sn = int(sf.seg_string[s])
        seg_no = int(np.count_nonzero(sf.seg_string[:s] == sn)) + 1
        n_seg = int(np.count_nonzero(sf.seg_string == sn))
        closed = sf.segment_is_closed(s)
        d = np.diff(pts, axis=0)
        rows = [
            (tr("String"), f"{sn}"),
            (tr("Segment"), tr("{i} of {n}", i=seg_no, n=n_seg)),
            (tr("Points"), f"{len(pts):,}"),
            (tr("Closed"), tr("yes") if closed else tr("no")),
        ]
        if len(pts) > 1:
            rows += [(tr("Length 3D"), f"{np.linalg.norm(d, axis=1).sum():,.2f} m"),
                     (tr("Length plan"), f"{np.linalg.norm(d[:, :2], axis=1).sum():,.2f} m")]
        if closed:
            x, y = pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean()
            area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
            rows.append((tr("Plan area"), f"{area:,.1f} m²"))
        desc = sf.records_desc[start + k].strip()
        rows += [(tr("Nearest point"), None),
                 (tr("Point"), tr("{i} of {n}", i=k + 1, n=len(pts)))] + _xyz_rows(pts[k])
        rows.append((tr("Description"), desc or "-"))
        rows += self._picked_rows(hit)
        hl = _polyline(self._local(pts)) if len(pts) > 1 else pv.PolyData(self._local(pts).astype(np.float32))
        return IdentifyResult(tr("String {sn}, segment {seg}", sn=sn, seg=seg_no), rows, hl)

    def _identify_point(self, hit, rec: int) -> IdentifyResult:
        sf = self.data
        s = int(np.searchsorted(sf.seg_start, rec, side="right") - 1)
        start, end = int(sf.seg_start[s]), int(sf.seg_end[s])
        sn = int(sf.seg_string[s])
        seg_no = int(np.count_nonzero(sf.seg_string[:s] == sn)) + 1
        n_seg = int(np.count_nonzero(sf.seg_string == sn))
        xyz = sf.records_xyz[rec]
        rows = [(tr("String"), f"{sn}")]
        if n_seg > 1:
            rows.append((tr("Segment"), tr("{i} of {n}", i=seg_no, n=n_seg)))
        if end - start > 1:
            rows.append((tr("Point"), tr("{i} of {n}", i=rec - start + 1, n=end - start)))
        rows += _xyz_rows(xyz)
        desc = sf.records_desc[rec].strip()
        fields = [f.strip() for f in desc.split(",")] if desc else []
        if len(fields) > 1:
            rows += [(tr("Description"), None)] + [(f"D{i + 1}", f or "-") for i, f in enumerate(fields)]
        else:
            rows.append((tr("Description"), desc or "-"))
        local = self._local(xyz)
        return IdentifyResult(tr("Point, string {sn}", sn=sn), rows, None, marker=local)


# ---------------------------------------------------------------------------
@dataclass(eq=False)
class SurfaceLayer(Layer):
    data: DtmFile | None = None
    styles: StyleSet | None = None
    kind: str = N_("Surface / Solid")
    use_styles: bool = True
    smooth: bool = False
    show_edges: bool = False
    edge_color: RGB = (0.1, 0.1, 0.1)

    role: str = ""          # geological role (fault, horizon, unconformity, ...; see geo_roles.ROLES)

    SETTINGS = Layer.SETTINGS + ("use_styles", "smooth", "show_edges", "role")

    def _object_color(self, obj_id: int) -> RGB:
        if self.use_styles and self.styles is not None and "TRIOBJECTS" in self.styles.sections:
            return self.styles.object_style(obj_id).face_color
        return self.color

    def _build_actors(self) -> None:
        dtm = self.data
        faces, colors, objs, tris, tids = [], [], [], [], []
        for obj, tri in dtm.iter_trisolations():
            faces.append(tri.triangles)
            colors.append(np.tile(_to_u8(self._object_color(obj.id)), (tri.n_triangles, 1)))
            objs.append(np.full(tri.n_triangles, obj.id))
            tris.append(np.full(tri.n_triangles, tri.id))
            tids.append(tri.triangle_ids)
        if not faces:
            return
        f = np.concatenate(faces)
        used, inv = np.unique(f, return_inverse=True)
        pts = (dtm.vertices[used] - self.scene.origin).astype(np.float32)
        cells = np.column_stack([np.full(len(f), 3), inv.reshape(-1, 3)]).ravel()
        mesh = pv.PolyData(pts, faces=cells)
        mesh.cell_data["rgb"] = np.concatenate(colors)
        mesh.cell_data["object"] = np.concatenate(objs)
        mesh.cell_data["trisolation"] = np.concatenate(tris)
        mesh.cell_data["triangle"] = np.concatenate(tids)
        if self.smooth:
            mesh = mesh.compute_normals(cell_normals=False, split_vertices=True,
                                        feature_angle=35.0, consistent_normals=False)
        self._add(
            mesh, scalars="rgb", rgb=True, opacity=self.opacity,
            smooth_shading=self.smooth, show_edges=self.show_edges, edge_color=self.edge_color,
            specular=0.25, specular_power=12.0, ambient=0.15, diffuse=0.85,
        )

    def world_bounds(self):
        v = self.data.vertices
        used = np.unique(np.concatenate([t.triangles.ravel() for _, t in self.data.iter_trisolations()]))
        p = v[used]
        return p.min(axis=0), p.max(axis=0)

    def info(self):
        d = self.data
        closed = sum(t.closed for _, t in d.iter_trisolations())
        total = sum(1 for _ in d.iter_trisolations())
        return super().info() + [
            (tr("String file"), d.str_name),
            (tr("Objects"), f"{len(d.objects):,}"),
            (tr("Trisolations"), tr("{total:,} ({closed} closed)", total=total, closed=closed)),
            (tr("Triangles"), f"{d.n_triangles:,}"),
            (tr("Style file"), str(self.styles.path.name) if self.styles and self.styles.path else "-"),
        ]

    def identify(self, hit):
        """The whole surface / solid (Surpac object) under the cursor, not one triangle."""
        obj_id = int(hit.cell_value("object"))
        parts = [t for o, t in self.data.iter_trisolations() if o.id == obj_id and t.n_triangles]
        v = self.data.vertices
        faces = np.concatenate([t.triangles for t in parts])
        used, inv = np.unique(faces, return_inverse=True)
        pts = v[used]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        local_faces = inv.reshape(-1, 3)

        area = plan = volume = 0.0
        n_closed = 0
        open_faces = []
        offset = 0
        for t in parts:
            f = local_faces[offset:offset + t.n_triangles]
            offset += t.n_triangles
            a, b, c = (pts[f[:, i]] - lo for i in range(3))  # relative: no precision loss
            cr = np.cross(b - a, c - a)
            area += 0.5 * float(np.linalg.norm(cr, axis=1).sum())
            if t.closed or _watertight(f):
                n_closed += 1
                volume += abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())) / 6.0
            else:
                plan += 0.5 * float(np.abs(cr[:, 2]).sum())
                open_faces.append(f)
        kind = (tr("Solid") if n_closed == len(parts) else
                tr("Surface") if n_closed == 0 else tr("Solid and surface"))
        n_tris = (tr("{total:,} ({closed} closed)", total=len(parts), closed=n_closed)
                  if 0 < n_closed < len(parts) else f"{len(parts):,}")
        rows = [(tr("Object"), f"{obj_id}"), (tr("Kind"), kind),
                (tr("Trisolations"), n_tris),
                (tr("Triangles"), f"{len(faces):,}"),
                (tr("Surface area"), f"{area:,.1f} m²")]
        if open_faces:
            rows.append((tr("Plan area"), f"{plan:,.1f} m²"))
        if n_closed:
            rows.append((tr("Volume"), f"{volume:,.1f} m³"))
        rows += [(tr("Extent"), None),
                 (tr("X range"), f"{lo[0]:,.2f} .. {hi[0]:,.2f}"),
                 (tr("Y range"), f"{lo[1]:,.2f} .. {hi[1]:,.2f}"),
                 (tr("Z range"), f"{lo[2]:,.2f} .. {hi[2]:,.2f}")]
        rows += self._picked_rows(hit)

        local_pts = (pts - self.scene.origin).astype(np.float32)
        fill = pv.PolyData(local_pts, faces=np.column_stack([np.full(len(local_faces), 3), local_faces]).ravel())
        outline = _boundary_lines(local_pts, np.concatenate(open_faces)) if open_faces else None
        return IdentifyResult(tr("{kind} object {id}", kind=kind, id=obj_id), rows, outline, fill=fill)


# ---------------------------------------------------------------------------
@dataclass(eq=False)
class BlockModelLayer(Layer):
    data: BlockModel | None = None
    kind: str = N_("Block model")
    color_by: str | None = None
    cmap: str = "turbo"
    hide_null: bool = True
    value_min: float | None = None     # filter: show blocks with value >= min
    value_max: float | None = None     # filter: show blocks with value <= max
    clim: tuple[float, float] | None = None
    show_edges: bool = False             # block edges off by default (user request); a setting per layer
    edge_color: RGB = (0.05, 0.05, 0.05)
    # constraint: conditions (see scene/constraints.py) combined by "and", "or" or an expression
    constraints: list = field(default_factory=list)
    constraint_logic: str = "and"
    mined_display: str = "gray"          # mined blocks: "show", "gray", "transparent", "hidden" (MINED_DISPLAYS)
    _grid = None
    n_shown: int = 0
    n_faces: int = 0
    _dtm_cache: dict = field(default_factory=dict, repr=False)   # DTM layer name -> (key, kind, values)
    _bar_slot: int | None = None         # makes the colour bar title unique among block models

    SETTINGS = Layer.SETTINGS + ("color_by", "cmap", "hide_null", "value_min", "value_max", "clim",
                                 "show_edges", "constraints", "constraint_logic", "mined_display")
    _nullable = ("color_by", "value_min", "value_max", "clim")

    def _setting_ok(self, key, value):
        from .constraints import valid_condition

        if key == "color_by":                  # not by the mined attributes: see mined_display
            return value is None or (value in self.data.attribute_names and value not in DEPLETION_ATTRS)
        if key == "mined_display":
            return value in MINED_DISPLAYS
        if key == "constraints":
            return isinstance(value, list) and all(valid_condition(c) for c in value)
        if key == "constraint_logic":
            return isinstance(value, str) and bool(value.strip())
        return True

    def apply_settings(self, d: dict) -> None:
        d = dict(d)
        if d.get("constrain_solid") and "constraints" not in d:   # workspaces from before (one DTM)
            d["constraints"] = [{"kind": "dtm", "layer": d["constrain_solid"],
                                 "mode": d.get("constrain_mode") or "inside"}]
        super().apply_settings(d)
        self.constraints = [dict(c) for c in self.constraints]

    # --- constraints (DTMs and block values, combined by and / or) ---------------------------
    def dtm_layer(self, name: str | None):
        """The scene's DTM layer called ``name`` (None when it is not loaded)."""
        if not name or self.scene is None:
            return None
        return next((lay for lay in self.scene.layers if isinstance(lay, SurfaceLayer) and lay.name == name),
                    None)

    def _dtm_data(self, name: str):
        """(kind, values) for a DTM, cached: a solid gives the inside mask of the block
        centroids, a surface the surface height at each centroid (NaN off it)."""
        from .solid import all_triangles, closed_mesh, points_inside, surface_heights

        lay = self.dtm_layer(name)
        if lay is None:
            return None
        key = (id(lay.data), id(self.data))
        hit = self._dtm_cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        mesh = closed_mesh(lay.data)
        if mesh is not None:
            kind, values = "solid", points_inside(self.data.centroids, *mesh)
        else:
            tri = all_triangles(lay.data)
            kind = "surface"
            values = (surface_heights(self.data.centroids, *tri) if tri is not None
                      else np.full(self.data.n_blocks, np.nan))
        self._dtm_cache[name] = (key, kind, values)
        return kind, values

    def condition_mask(self, cond: dict) -> np.ndarray | None:
        """Blocks meeting one condition; None for a DTM that is not loaded (not applied)."""
        from .constraints import value_mask

        if cond.get("kind") == "value":
            return value_mask(self.data, cond)
        data = self._dtm_data(cond.get("layer"))
        if data is None:
            return None
        kind, values = data
        mode = cond.get("mode", "inside")
        if kind == "solid":
            return values if mode != "outside" else ~values
        z = self.data.centroids[:, 2]
        with np.errstate(invalid="ignore"):
            return (z > values) if mode == "above" else (z < values)

    def constraint_mask(self) -> np.ndarray | None:
        """Blocks meeting the whole constraint; None when there is none. A condition on a DTM
        that is not loaded is left out (counts as met)."""
        from .constraints import combine

        if not self.constraints:
            return None
        masks = []
        for cond in self.constraints:
            m = self.condition_mask(cond)
            masks.append(m if m is not None else np.ones(self.data.n_blocks, dtype=bool))
        try:
            return combine(masks, self.constraint_logic)
        except ValueError:
            return combine(masks, "and")             # a broken expression: all conditions

    def missing_dtms(self) -> list[str]:
        return [c["layer"] for c in self.constraints
                if c.get("kind") == "dtm" and self.dtm_layer(c.get("layer")) is None]

    def constraints_changed(self) -> bool:
        """True when a DTM used by the constraint was reloaded / removed / added since the
        last build (the constraint must be evaluated again)."""
        for cond in self.constraints:
            if cond.get("kind") != "dtm":
                continue
            lay = self.dtm_layer(cond.get("layer"))
            hit = self._dtm_cache.get(cond.get("layer"))
            now = id(lay.data) if lay is not None else None
            before = hit[0][0] if hit is not None else None
            if now != before:
                return True
        return False

    def constraint_summary(self) -> str:
        from .constraints import describe, letter, logic_text

        if not self.constraints:
            return tr("none")
        if len(self.constraints) == 1:
            return describe(self.constraints[0])
        lines = [f"{letter(i)}: {describe(c)}" for i, c in enumerate(self.constraints)]
        return "\n".join(lines + [f"→ {logic_text(self.constraint_logic, len(self.constraints))}"])

    def __post_init__(self):
        if self.color_by is None and self.data is not None:
            names = [n for n in self.data.attribute_names if self.data.valid_mask(n).any()]
            prefer = [n for n in names if not self.data.is_categorical(n) and "class" not in n.lower()]
            self.color_by = (prefer or names or [None])[0]

    @property
    def grid(self):
        if self._grid is None:
            self._grid = self.data.grid()
        return self._grid

    def value_range(self, name: str | None = None) -> tuple[float, float] | None:
        name = name or self.color_by
        if name is None:
            return None
        v = self.data.attributes[name][self.data.valid_mask(name)]
        if len(v) == 0:
            return None
        return float(v.min()), float(v.max())

    def visible_mask(self) -> np.ndarray:
        bm = self.data
        m = np.ones(bm.n_blocks, dtype=bool)
        if self.color_by is not None:
            valid = bm.valid_mask(self.color_by)
            if self.hide_null:
                m &= valid
            if not bm.is_categorical(self.color_by):
                v = bm.attributes[self.color_by]
                if self.value_min is not None:
                    m &= ~valid | (v >= self.value_min)
                if self.value_max is not None:
                    m &= ~valid | (v <= self.value_max)
        if self.scene is not None:
            m &= self.scene.slab_mask(bm.centroids)
        keep = self.constraint_mask()
        if keep is not None:
            m &= keep
        if self.mined_display == "hidden":
            mined = self.mined_mask()
            if mined is not None:
                m &= ~mined
        return m

    def mined_mask(self) -> np.ndarray | None:
        """Blocks counted as mined (at least MINED_SHARE of them mined by the recorded cuts);
        None when nothing is mined (or the reserve is not computed)."""
        f = self.data.attributes.get("mined_fraction")
        if f is None:
            return None
        m = np.nan_to_num(np.asarray(f, dtype=np.float32), nan=0.0) >= MINED_SHARE
        return m if m.any() else None

    def block_rgb(self, blocks: np.ndarray) -> np.ndarray:
        """The colour each block is drawn with (single colour, category or colour map), uint8 RGB."""
        name = self.color_by
        if name is None:
            return np.tile(_to_u8(self.color), (len(blocks), 1)).astype(np.uint8)
        data = self.block_cell_data(blocks)
        if "rgb" in data:
            return data["rgb"]
        from matplotlib import colormaps

        v = data[name]
        lo, hi = self.clim or self.value_range(name) or (0.0, 1.0)
        t = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0.0, 1.0)
        rgb = (colormaps[self.cmap](np.nan_to_num(t))[:, :3] * 255).round().astype(np.uint8)
        rgb[~np.isfinite(v)] = _to_u8((0.45, 0.45, 0.45))
        return rgb

    def mined_cell_data(self, blocks: np.ndarray) -> dict[str, np.ndarray]:
        """Per-cell arrays for drawing mined ``blocks``: plain gray (mined_display "gray") or
        their own colour (drawn semi-transparent for "transparent")."""
        rgb = self.block_rgb(blocks).astype(np.float32)
        if self.mined_display == "gray":
            rgb = rgb * (1.0 - MINED_GRAY) + 128.0 * MINED_GRAY
        return {"block": blocks, "rgb": rgb.round().astype(np.uint8)}

    def _build_actors(self) -> None:
        mask = self.visible_mask()
        self._built_slab = getattr(self.scene, "slab", None)   # the 3D slice drawn for
        self.n_shown = int(mask.sum())
        self._mined_actor = None
        mined = self.mined_mask() if self.mined_display in ("gray", "transparent") else None
        if mined is not None:
            mined = mined & mask
            if not mined.any():
                mined = None
        if mined is None:
            groups = [(*exposed_faces(self.grid, mask), False)]
        elif self.mined_display == "transparent":         # each set closed: no holes behind see-through blocks
            groups = [(*exposed_faces(self.grid, mask & ~mined), False), (*exposed_faces(self.grid, mined), True)]
        else:
            blocks, dirs = exposed_faces(self.grid, mask)
            sel = mined[blocks]
            groups = [(blocks[~sel], dirs[~sel], False), (blocks[sel], dirs[sel], True)]
        self.n_faces = int(sum(len(g[0]) for g in groups))
        kw = dict(opacity=self.opacity, show_edges=self.show_edges, edge_color=self.edge_color,
                  ambient=0.2, diffuse=0.8, specular=0.1)
        for blocks, dirs, is_mined in groups:
            if len(blocks) == 0:
                continue
            pts, cells = face_quads(self.grid, blocks, dirs, self.scene.origin)
            mesh = pv.PolyData(pts, faces=cells)
            if is_mined:
                for key, arr in self.mined_cell_data(blocks).items():
                    mesh.cell_data[key] = arr
                op = self.opacity * (MINED_OPACITY if self.mined_display == "transparent" else 1.0)
                self._add(mesh, scalars="rgb", rgb=True, **{**kw, "opacity": op})
                self._mined_actor = self.actors[-1]
                continue
            for key, arr in self.block_cell_data(blocks).items():
                mesh.cell_data[key] = arr
            self._add_blocks(mesh, kw)

    def _add_blocks(self, mesh, kw: dict) -> None:
        bm = self.data
        if self.color_by is None:
            self._add(mesh, color=self.color, **kw)
            return
        name = self.color_by
        if bm.is_categorical(name):
            self._add(mesh, scalars="rgb", rgb=True, **kw)
            return
        clim = self.clim or self.value_range(name) or (0.0, 1.0)
        self._add(
            mesh, scalars=name, cmap=self.cmap, clim=clim, nan_color=(0.45, 0.45, 0.45),
            show_scalar_bar=True,
            scalar_bar_args=dict(title=self.scalar_bar_title, vertical=True, n_labels=6,
                                 fmt="%.2f", color=self.scene.foreground, position_x=0.055, position_y=0.30,
                                 height=0.40, width=0.05, title_font_size=14, label_font_size=12),
            **kw,
        )

    def block_cell_data(self, blocks: np.ndarray) -> dict[str, np.ndarray]:
        """Per-cell arrays for drawing ``blocks`` (one cell per entry): the block index and
        the colouring array (``rgb`` for a categorical attribute, the values otherwise)."""
        out = {"block": blocks}
        name = self.color_by
        if name is None:
            return out
        bm = self.data
        vals = bm.attributes[name][blocks]
        if bm.is_categorical(name):
            cats = bm.categories[name]
            fixed = bm.category_rgb.get(name) or []
            lut = np.array([_to_u8(fixed[i] if i < len(fixed) else PALETTE[i % len(PALETTE)])
                            for i in range(max(len(cats), 1))])
            rgb = np.where((vals >= 0)[:, None], lut[np.clip(vals, 0, None)], _to_u8((0.5, 0.5, 0.5)))
            out["rgb"] = rgb.astype(np.uint8)
        else:
            v = vals.astype(np.float32).copy()
            v[~bm.valid_mask(name)[blocks]] = np.nan
            out[name] = v
        return out

    def apply_visibility(self) -> None:
        super().apply_visibility()
        # The colour bar shows only for a visible layer that is active in the TOC.
        if self.scene is not None and self.color_by:
            try:
                self.scene.plotter.scalar_bars[self.scalar_bar_title].SetVisibility(
                    self.visible and self.legend_active)
            except (KeyError, AttributeError):
                pass

    @property
    def scalar_bar_title(self) -> str:
        # Unique per layer (pyvista keys scalar bars by title) but short on screen: trailing
        # spaces, a different number for every block model view in the scene.
        if self._bar_slot is None:
            used = {lay._bar_slot for lay in (self.scene.layers if self.scene else [])
                    if isinstance(lay, BlockModelLayer) and lay is not self}
            self._bar_slot = next(i for i in range(1, 10_000) if i not in used)
        return f"{self.color_by}" + " " * self._bar_slot

    def remove_actors(self) -> None:
        if self.scene is not None and self.color_by:
            try:
                self.scene.plotter.remove_scalar_bar(self.scalar_bar_title, render=False)
            except (KeyError, ValueError, AttributeError):
                pass
        super().remove_actors()

    def world_bounds(self):
        return self.data.bounds()

    def info(self):
        bm = self.data
        sizes = ", ".join("x".join(f"{v:g}" for v in s) for s in bm.block_sizes()[:6])
        return super().info() + [
            (tr("Blocks"), f"{bm.n_blocks:,}"),
            (tr("Block sizes"), sizes),
            (tr("Attributes"), ", ".join(bm.attribute_names)),
            (tr("Shown blocks"), f"{self.n_shown:,}"),
            (tr("Drawn faces"), f"{self.n_faces:,}"),
        ]

    def attribute_text(self, name: str, i: int) -> str:
        bm = self.data
        v = bm.attributes[name][i]
        if bm.is_categorical(name):
            cats = bm.categories[name]
            return cats[int(v)] if 0 <= int(v) < len(cats) else "-"
        if not np.isfinite(v) or v == np.float32(bm.null_value):  # same test as valid_mask
            return tr("null")
        return fmt_num(float(v), 4)

    def identify(self, hit):
        bm = self.data
        i = int(hit.cell_value("block"))
        c, s = bm.centroids[i], bm.sizes[i].astype(np.float64)
        rows = [(tr("Block"), tr("{i:,} of {n:,}", i=i + 1, n=bm.n_blocks)),
                (tr("Centroid"), None)] + _xyz_rows(c) + [
                (tr("Size"), " x ".join(fmt_num(v) for v in s))]
        if bm.attribute_names:
            rows.append((tr("Attributes"), None))
            for name in bm.attribute_names:
                rows.append((name, self.attribute_text(name, i)))
        rows += self._picked_rows(hit)
        lo, hi = self._local(c - s / 2.0), self._local(c + s / 2.0)
        return IdentifyResult(tr("Block {i:,}", i=i + 1), rows, _box_edges(lo, hi))


# ---------------------------------------------------------------------------
SECTION_TRACE_WIDTH = 2.5         # px: drillholes in cross-sections are thick lines along the hole path
SECTION_INTERVAL_WIDTH = 7.0


@dataclass(eq=False)
class DrillholeLayer(Layer):
    data: DrillholeDB | None = None
    kind: str = N_("Drillholes")
    table: str | None = None          # interval table used for colouring
    field_name: str | None = None     # column in that table
    cmap: str = "turbo"
    diameter: float = 4.0              # metres: the holes are cylinders of this diameter; 0 = lines
    trace_color: RGB = (0.55, 0.55, 0.55)
    show_labels: bool = True
    category_colors: dict[str, RGB] = field(default_factory=dict)
    # drilling area shown (drillhole_collar.area_id): AREA_AUTO = the areas of the holes being drilled or
    # planned (all holes when there are none), AREA_ALL, or one area id
    area: str = "auto"
    all_data: DrillholeDB | None = field(default=None, repr=False)     # every hole; ``data`` = those shown
    _trace_holes: list = field(default_factory=list, repr=False)

    AREA_AUTO, AREA_ALL = "auto", "*"
    SETTINGS = Layer.SETTINGS + ("table", "field_name", "cmap", "diameter", "show_labels",
                                 "category_colors", "area")

    def _setting_ok(self, key, value):
        db = self.data
        if key == "table":
            return value in db.intervals
        if key == "field_name":
            return self.table is not None and value in self.field_columns(self.table)
        return True

    def __post_init__(self):
        db = self.data
        if db is None:
            return
        self.all_data = db
        self.data = self.shown_data()
        if self.table is None and db.intervals:
            prefer = [t for t in db.intervals if t.lower() in ("geology", "lithology", "litho", "lith")]
            self.table = (prefer or list(db.intervals))[0]
        if self.table is not None and self.field_name is None:
            cols = self.field_columns(self.table)
            prefer = [c for c in cols if c.lower() in ("rock", "lith", "lithology", "litho", "code")]
            self.field_name = (prefer or cols or [None])[0]

    # --- drilling areas ------------------------------------------------------------------------
    def shown_areas(self) -> list[str] | None:
        """The areas shown (None: every hole). A database with holes of several areas - e.g. the
        drilling history of distant areas, kept for the risk prediction - shows by default only the
        areas being drilled or planned."""
        db = self.all_data
        counts = db.area_counts() if db is not None else {}
        if not counts or self.area == self.AREA_ALL:
            return None
        if self.area in counts:
            return [self.area]
        live = db.live_areas()
        return live if live and len(counts) > 1 else None

    def shown_data(self) -> DrillholeDB:
        areas = self.shown_areas()
        return self.all_data if areas is None else self.all_data.only_areas(areas)

    def set_area(self, area: str) -> None:
        self.area = area or self.AREA_AUTO
        self.data = self.shown_data()
        self.rebuild()

    def apply_settings(self, d: dict) -> None:
        if self.all_data is not None:
            self.data = self.all_data                # the saved table / field are checked against every hole
        super().apply_settings(d)
        if self.all_data is not None:
            self.data = self.shown_data()

    def field_columns(self, table: str) -> list[str]:
        df = self.data.intervals[table]
        skip = {"hole_id", "depth_from", "depth_to", "samp_id", "x_from", "y_from", "z_from",
                "x_to", "y_to", "z_to"}
        return [c for c in df.columns if c not in skip and df[c].notna().any()]

    def _build_actors(self) -> None:
        db = self.data
        origin = self.scene.origin
        # Hole traces (thin, grey).
        pts_all, depths, lines, k = [], [], [], 0
        collars, labels = [], []
        self._trace_holes = []
        for hid in db.hole_ids:
            try:
                tr = db.trace(hid)
            except KeyError:
                continue
            p, d = tr.path(step=5.0, with_depths=True)
            pts_all.append(p - origin)
            depths.append(d)
            lines.append(np.concatenate([[len(p)], np.arange(k, k + len(p))]))
            k += len(p)
            collars.append(tr.collar - origin)
            labels.append(hid)
            self._trace_holes.append(hid)
        if not pts_all:
            return
        trace = pv.PolyData(np.vstack(pts_all).astype(np.float32), lines=np.concatenate(lines))
        trace.cell_data["gv_hole"] = np.arange(len(self._trace_holes))
        trace.point_data["gv_depth"] = np.concatenate(depths)
        radius = max(0.0, float(self.diameter or 0.0)) / 2          # the layer's Diameter, metres
        if radius > 0:                                  # the trace a little thinner than the intervals
            act = self._add(trace.tube(radius=radius * 0.6, n_sides=8).triangulate(), color=self.trace_color,
                            opacity=self.opacity)
            act._gv_lines = trace                       # sections draw the path, not the cylinder outline
        else:
            act = self._add(trace, color=self.trace_color, line_width=1.5, lighting=False, opacity=self.opacity)
        act._gv_section_width = SECTION_TRACE_WIDTH

        # Intervals coloured by the chosen field.
        if self.table and self.field_name:
            # Each interval follows the hole path (curved holes: a point every 2 m).
            rows, ipts, ideps, counts = db.interval_paths(self.table)
            ok = np.isfinite(ipts).all(axis=1)
            keep = np.add.reduceat(ok, np.r_[0, np.cumsum(counts)[:-1]]) == counts if len(counts) else ok[:0]
            df = db.intervals[self.table].iloc[rows[keep]] if len(rows) else db.intervals[self.table].iloc[:0]
            n = len(df)
            if n:
                own = np.repeat(keep, counts)
                pts = (ipts[own] - origin).astype(np.float32)
                c = counts[keep]
                starts = np.r_[0, np.cumsum(c)[:-1]]
                cells = np.concatenate([np.r_[k, np.arange(s, s + k)] for s, k in zip(starts, c)])
                mesh = pv.PolyData(pts, lines=cells.astype(np.int64))
                mesh.cell_data["gv_row"] = rows[keep]  # position in the full interval table
                mesh.point_data["gv_depth"] = ideps[own]
                col = df[self.field_name]
                if radius > 0:                          # cylinders (made below, once the colours are set)
                    kw = dict(opacity=self.opacity, smooth_shading=True)
                else:
                    kw = dict(line_width=5.0, render_lines_as_tubes=True, opacity=self.opacity)
                numeric = col.dtype.kind in "fi" and col.nunique() > 12
                if numeric:
                    v = col.to_numpy(dtype=np.float32)
                    mesh.cell_data[self.field_name] = v
                    fin = v[np.isfinite(v)]
                    clim = (float(fin.min()), float(fin.max())) if len(fin) else (0, 1)
                    lines = mesh
                    if radius > 0:
                        mesh = mesh.tube(radius=radius, n_sides=12).triangulate()   # triangles: pickable; data kept
                    act = self._add(mesh, scalars=self.field_name, cmap=self.cmap, clim=clim,
                                    nan_color=self.trace_color, **kw)
                    act._gv_lines, act._gv_section_width = lines, SECTION_INTERVAL_WIDTH
                else:
                    codes = col.astype(str).where(col.notna(), "")
                    rgb = np.array([_to_u8(self.code_color(c)) for c in codes], dtype=np.uint8)
                    mesh.cell_data["rgb"] = rgb
                    lines = mesh
                    if radius > 0:
                        mesh = mesh.tube(radius=radius, n_sides=12).triangulate()   # triangles: pickable; data kept
                    act = self._add(mesh, scalars="rgb", rgb=True, **kw)
                    act._gv_lines, act._gv_section_width = lines, SECTION_INTERVAL_WIDTH

        if self.show_labels and collars:
            act = self.scene.plotter.add_point_labels(
                np.array(collars, dtype=np.float32), labels, font_size=11, text_color=self.scene.foreground,
                shape=None, show_points=False,  # hole IDs only; no collar marker
                always_visible=False, render=False, name=f"{id(self)}-labels",
            )
            act._geoviewer_layer = self
            self.actors.append(act)

    def style_colors(self) -> dict:
        """Rock code -> colour from the database's drillhole_styles, when colouring by rock."""
        if self.data is not None and (self.field_name or "").lower() in ("rock", "lith", "lithology", "litho"):
            return getattr(self.data, "rock_colors", {}) or {}
        return {}

    def code_color(self, code: str) -> RGB:
        if code == "":
            return self.trace_color
        styled = self.style_colors()
        if code in styled:                       # the database colour wins (shared by everyone)
            self.category_colors[code] = tuple(styled[code])
            return self.category_colors[code]
        if code not in self.category_colors:
            self.category_colors[code] = PALETTE[len(self.category_colors) % len(PALETTE)]
        return self.category_colors[code]

    def legend(self) -> list[tuple[str, str, RGB]]:
        """(code, description, colour) for the current categorical field."""
        if not (self.table and self.field_name):
            return []
        desc = self.data.code_descriptions(self.table, self.field_name)
        return [(c, desc.get(c, ""), rgb) for c, rgb in sorted(self.category_colors.items())]

    def world_bounds(self):
        c = self.data.collars
        if c.empty:
            return None
        xyz = c[["x", "y", "z"]].to_numpy(dtype=np.float64)
        md = np.nan_to_num(c["max_depth"].to_numpy(dtype=np.float64), nan=0.0)
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        lo = lo.copy()
        lo[2] -= md.max(initial=0.0)
        return lo, hi

    def info(self):
        db = self.data
        if self.source == "database" and db.collars.empty:
            return super().info() + [(tr("Holes"), tr("none yet - add them with Drillholes (Ctrl+D) or import "
                                                      "drillhole_collar, drillhole_survey, ..."))]
        total = len(self.all_data.collars) if self.all_data is not None else len(db.collars)
        areas = self.shown_areas() if self.all_data is not None else None
        holes = f"{len(db.collars):,}" if areas is None else tr(
            "{n:,} of {total:,} (area {areas})", n=len(db.collars), total=total, areas=", ".join(areas))
        return super().info() + [
            (tr("Holes"), holes),
            (tr("Survey records"), f"{len(db.surveys):,}"),
            (tr("Interval tables"), ", ".join(f"{k} ({len(v):,})" for k, v in db.intervals.items())),
            (tr("Warnings"), "; ".join(db.warnings) or "-"),
        ]

    _INTERNAL = {"hole_id", "depth_from", "depth_to", "x_from", "y_from", "z_from", "x_to", "y_to", "z_to"}

    def _interval_rows(self, table: str, row: pd.Series) -> list[tuple[str, str | None]]:
        f, t = float(row["depth_from"]), float(row["depth_to"])
        rows = [(tr("{table} interval", table=table), None), (tr("From"), f"{f:,.2f} m"),
                (tr("To"), f"{t:,.2f} m"), (tr("Length"), f"{t - f:,.2f} m")]
        for c in row.index:
            if c in self._INTERNAL:
                continue
            v = row[c]
            text = fmt_num(v, 4) if not pd.isna(v) else "-"
            if c == self.field_name and table == self.table and not pd.isna(v):
                desc = self.data.code_descriptions(table, c).get(str(v), "")
                text += f"  ({desc})" if desc else ""
            rows.append((c, text))
        # notes are not loaded with the drillholes: read this interval's from the database now
        for c, v in self.data.interval_notes(table, str(row["hole_id"]), f, t).items():
            rows.append((c, str(v) if v not in (None, "") else "-"))
        return rows

    def identify(self, hit):
        db = self.data
        depth = float(hit.interpolate("gv_depth"))
        row = None
        if "gv_row" in hit.mesh.cell_data:  # an interval of the colouring table
            row = db.intervals[self.table].iloc[int(hit.cell_value("gv_row"))]
            hid = str(row["hole_id"])
        else:                                # the hole trace
            hid = self._trace_holes[int(hit.cell_value("gv_hole"))]
            if self.table:
                df = db.intervals[self.table]
                m = (df["hole_id"] == hid) & (df["depth_from"] <= depth) & (df["depth_to"] > depth)
                if m.any():
                    row = df[m].iloc[0]
        trace = db.trace(hid)
        maxd = trace.max_depth if np.isfinite(trace.max_depth) else trace.md[-1]
        rows = [(tr("Hole"), hid), (tr("Depth at pick"), f"{depth:,.2f} m"), (tr("Max depth"), f"{maxd:,.2f} m"),
                (tr("Collar"), None)] + _xyz_rows(trace.collar)
        if row is not None:
            rows += self._interval_rows(self.table, row)
            a, b = float(row["depth_from"]), float(row["depth_to"])
            seg = trace.positions_at(np.linspace(a, b, max(2, int(np.ceil((b - a) / 2.0)) + 1)))  # follows a bend
        else:
            seg = trace.path(step=5.0)
        rows += self._picked_rows(hit)
        return IdentifyResult(tr("Hole {hole}", hole=hid), rows, _polyline(self._local(seg)),
                              extra={"hole_id": hid, "depth": depth})

    # --- drillhole log window ------------------------------------------------------------
    def interval_rgb(self, values) -> list[tuple[float, float, float]]:
        """Colours of interval values of the current field, as drawn in the 3D view."""
        name = self.field_name
        actor = next((a for a in self.actors if hasattr(a, "GetMapper") and a.GetMapper() is not None
                      and hasattr(a.GetMapper(), "GetArrayName") and a.GetMapper().GetArrayName() == name),
                     None)
        if actor is not None:                      # numeric field: the actor's colour map
            lut = actor.GetMapper().GetLookupTable()
            out, rgb = [], [0.0, 0.0, 0.0]
            for v in values:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    f = float("nan")
                if np.isfinite(f):
                    lut.GetColor(f, rgb)
                    out.append(tuple(rgb))
                else:
                    out.append(tuple(self.trace_color))
            return out
        return [tuple(self.code_color("" if pd.isna(v) else str(v))) for v in values]

    def grade_fields(self) -> list[tuple[str, str, str]]:
        """(name shown, table, column) of every numeric field of the interval tables (assays
        first); the table is added to the name when two tables have a field of that name."""
        db = self.data
        skip = {"hole_id", "depth_from", "depth_to", "x_from", "y_from", "z_from", "x_to", "y_to", "z_to"}
        found = []
        order = sorted(db.intervals, key=lambda t: (not any(k in t.lower() for k in ("assay", "sample")), t))
        for t in order:
            df = db.intervals[t]
            found += [(c, t) for c in df.columns
                      if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
        names = [c for c, _t in found]
        return [(c if names.count(c) == 1 else f"{c} ({t})", t, c) for c, t in found]

    def hole_grades(self, hid: str) -> dict:
        """name -> [(from, to, value), ...] of the hole, in depth order (values as stored)."""
        out = {}
        for name, t, c in self.grade_fields():
            df = self.data.intervals[t]
            h = df[df["hole_id"] == hid].sort_values("depth_from")
            out[name] = [(float(a), float(b), float(v)) for a, b, v in
                         h[["depth_from", "depth_to", c]].itertuples(index=False) if not pd.isna(v)]
        return out

    def hole_log(self, hid: str) -> HoleLog:
        db = self.data
        tr = db.trace(hid)
        length = float(tr.max_depth) if np.isfinite(tr.max_depth) and tr.max_depth > 0 else float(tr.md[-1])
        depths = np.unique(np.r_[np.arange(0.0, length, 1.0), tr.md[tr.md <= length], length])
        intervals = []
        if self.table and self.field_name:
            df = db.intervals[self.table]
            h = df[df["hole_id"] == hid].sort_values("depth_from")
            vals = h[self.field_name].tolist()
            colours = self.interval_rgb(vals)
            desc = db.code_descriptions(self.table, self.field_name)
            for (a, b, v), rgb in zip(h[["depth_from", "depth_to", self.field_name]].itertuples(index=False),
                                      colours):
                text = "" if pd.isna(v) else fmt_num(v, 3) if isinstance(v, (int, float, np.number)) else str(v)
                intervals.append((float(a), float(b), text, desc.get(str(v), "") if not pd.isna(v) else "", rgb))
        dirs = tr.direction_at(depths)          # follows the path between survey stations
        log = HoleLog(hid, np.asarray(tr.collar, dtype=np.float64), length, depths, tr.positions_at(depths),
                      intervals, self.table, self.field_name, dirs=dirs, grades=self.hole_grades(hid))
        log.collar_azimuth, log.collar_dip = log.azimuth_dip(0.0)
        return log


# ---------------------------------------------------------------------------
@dataclass(eq=False)
class MinedShapesLayer(Layer):
    """The mined shapes (outline x From/To) of the production log, see-through, one colour
    per month. Made by the Reserve window from production.json; not saved in the workspace."""
    shapes: list = field(default_factory=list)       # MinedShape, those up to the "as of" date
    stats: dict = field(default_factory=dict)        # shape id -> (Totals, blocks touched)
    unit: str = "g/t"
    kind: str = N_("Mined shapes")
    opacity: float = 0.45
    saved = False                                    # not a file layer: left out of the workspace

    def month_colors(self) -> dict[str, RGB]:
        from .production import month_of

        months = sorted({month_of(s.date) for s in self.shapes})
        return {m: MONTH_COLORS[i % len(MONTH_COLORS)] for i, m in enumerate(months)}

    def _build_actors(self) -> None:
        from .production import month_of, prism_mesh

        colors = self.month_colors()
        meshes = []
        for sh in self.shapes:
            m = prism_mesh(sh, self.scene.origin)
            if m is None or m.n_cells == 0:
                continue
            m.cell_data["shape"] = np.full(m.n_cells, sh.id, dtype=np.int32)
            m.cell_data["rgb"] = np.tile(_to_u8(colors[month_of(sh.date)]), (m.n_cells, 1))
            meshes.append(m)
        if not meshes:
            return
        mesh = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
        mesh = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh
        self._add(mesh, scalars="rgb", rgb=True, opacity=self.opacity, show_edges=False,
                  ambient=0.3, diffuse=0.7, specular=0.1)
        edges = mesh.extract_feature_edges(boundary_edges=False, non_manifold_edges=False,
                                           feature_edges=True, manifold_edges=False, feature_angle=30)
        if edges.n_cells:
            self._add(edges, scalars="rgb", rgb=True, line_width=1.5, lighting=False)

    def world_bounds(self):
        if not self.shapes:
            return None
        b = [s.bounds() for s in self.shapes]
        return np.min([lo for lo, _ in b], axis=0), np.max([hi for _, hi in b], axis=0)

    def info(self):
        return super().info() + [(tr("Shapes"), f"{len(self.shapes):,}")]

    def identify(self, hit):
        from .production import AXIS_WORD, metal_text

        try:
            sid = int(hit.cell_value("shape"))
        except (KeyError, IndexError, TypeError):
            return super().identify(hit)
        sh = next((s for s in self.shapes if s.id == sid), None)
        if sh is None:
            return super().identify(hit)
        if sh.is_solid:
            rows = [(tr("Name"), sh.name or "-"), (tr("Date"), sh.date), (tr("Mined solid"), sh.source or "-"),
                    (tr("Volume"), f"{sh.volume():,.0f} m³"),
                    (tr("Elevation"), f"{sh.from_:,.2f} .. {sh.to:,.2f}"),
                    (tr("Triangles"), f"{len(sh.triangles):,}")]
        else:
            ia = sh.axes[2]
            rows = [(tr("Name"), sh.name or "-"), (tr("Date"), sh.date),
                    (tr("From"), f"{sh.from_:,.2f}  ({tr(AXIS_WORD[ia])})"), (tr("To"), f"{sh.to:,.2f}"),
                    (tr("Thickness"), f"{sh.depth:,.2f} m"), (tr("Outline area"), f"{sh.area():,.1f} m²")]
        st = self.stats.get(sid)
        if st is not None:
            tot, blocks = st
            rows += [(tr("Mined ore"), f"{tot.tonnes:,.0f} t"), (tr("Grade"), f"{tot.grade:,.2f} {self.unit}"),
                     (tr("Metal"), metal_text(tot.metal, self.unit)), (tr("Blocks touched"), f"{blocks:,}")]
        if sh.note:
            rows.append((tr("Note"), sh.note))
        rows += self._picked_rows(hit)
        return IdentifyResult(tr("Mined shape {name}", name=sh.name or sid), rows)
