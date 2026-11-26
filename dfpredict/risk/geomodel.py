"""The workspace's geological model, as the drilling risk tools read it ahead of the bit (spec 3.2, 5.6).

The model comes from the surface layers of the workspace and their **role** (see ``scene/geo_roles.py``,
after 3D MPM Studio's layer roles):

* ``horizon`` / ``unconformity`` surfaces are stratigraphic contacts. Ordered top to bottom they split the
  ground into units: the unit of a point is the number of contacts above it. A horizon named like
  ``horizon_H2_U2-U3`` gives the units above (U2) and below (U3); otherwise the units are named in the
  Geological model dialog (``drillrisk/geomodel.json`` in the workspace folder).
* ``formation`` solids (closed meshes) give the formation directly where they exist (converted to voxels).
* ``fault`` surfaces give fault crossings and the distance to the nearest fault.
* ``topography`` gives the ground surface (above it: air).

Every surface is gridded once (``GRID`` m): unit look-ups, thickness and the local dip of contacts are
then simple array reads, fast enough for many trial paths. ``uncertainty`` is the error band ±u of spec
5.6: from the distance to model control (the holes the model was built from) and widened below units
whose thickness varies widely. No Qt here.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

GRID = 10.0                    # m, plan grid of the gridded surfaces
CONFIG_FILE = "geomodel.json"  # in <workspace>/drillrisk/
DEFAULTS = {"u_near": 8.0, "u_far": 30.0, "d_far": 250.0, "fault_zone": 12.0, "thickness_factor": 0.10,
            "voxel": [5.0, 5.0, 2.0]}
_UNIT_PAIR = re.compile(r"([^_\-\s]+)-([^_\-\s]+)$")


def unit_pair(name: str) -> tuple[str, str] | None:
    """('U2', 'U3') from a horizon name like 'horizon_H2_U2-U3' (None when the name has no pair)."""
    m = _UNIT_PAIR.search(str(name))
    return (m.group(1), m.group(2)) if m else None


# ---------------------------------------------------------------------------------------------
@dataclass
class Grid2D:
    """A surface gridded on the model's plan grid: z (ny, nx), filled outside its footprint."""

    name: str
    x0: float
    y0: float
    d: float
    z: np.ndarray
    covered: np.ndarray                       # (ny, nx) inside the surface's own footprint

    def sample(self, x, y) -> np.ndarray:
        """Bilinear elevation at plan points (clamped to the grid)."""
        ny, nx = self.z.shape
        fx = np.clip((np.asarray(x, np.float64) - self.x0) / self.d, 0, nx - 1.000001)
        fy = np.clip((np.asarray(y, np.float64) - self.y0) / self.d, 0, ny - 1.000001)
        i, j = fx.astype(int), fy.astype(int)
        tx, ty = fx - i, fy - j
        z = self.z
        return ((z[j, i] * (1 - tx) + z[j, i + 1] * tx) * (1 - ty)
                + (z[j + 1, i] * (1 - tx) + z[j + 1, i + 1] * tx) * ty)

    def gradient(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """dz/dx, dz/dy at plan points (central differences of the grid)."""
        e = self.d
        gx = (self.sample(np.asarray(x) + e, y) - self.sample(np.asarray(x) - e, y)) / (2 * e)
        gy = (self.sample(x, np.asarray(y) + e) - self.sample(x, np.asarray(y) - e)) / (2 * e)
        return gx, gy


def grid_surface(name: str, vertices: np.ndarray, triangles: np.ndarray, x0, y0, d, nx, ny) -> Grid2D:
    """Grid a triangulated (single-valued) surface: linear inside its triangles, nearest outside."""
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, np.float64)
    used = np.unique(np.asarray(triangles).ravel())
    gx = x0 + np.arange(nx) * d
    gy = y0 + np.arange(ny) * d
    X, Y = np.meshgrid(gx, gy)
    z = np.full(X.shape, np.nan)
    c = v[used, :2].mean(axis=0)
    try:
        from matplotlib.tri import LinearTriInterpolator, Triangulation

        tri = Triangulation(v[:, 0] - c[0], v[:, 1] - c[1], np.asarray(triangles, dtype=np.int64))
        z = np.ma.filled(LinearTriInterpolator(tri, v[:, 2])(X - c[0], Y - c[1]), np.nan)
    except Exception:  # noqa: BLE001 - overlapping triangles (e.g. a folded surface): over the vertices
        from scipy.interpolate import LinearNDInterpolator

        z = LinearNDInterpolator(v[used, :2] - c, v[used, 2])(X - c[0], Y - c[1])
    z = np.asarray(z, np.float64)
    covered = np.isfinite(z)
    if not covered.all():
        if covered.any():
            tree = cKDTree(np.column_stack([X[covered], Y[covered]]))
            _, k = tree.query(np.column_stack([X[~covered], Y[~covered]]))
            z[~covered] = z[covered][k]
        else:                                   # a surface without area on the grid: its vertices' nearest
            tree = cKDTree(v[used, :2])
            _, k = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
            z = v[used, 2][k].reshape(X.shape)
    return Grid2D(name, float(x0), float(y0), float(d), z, covered)


@dataclass
class Fault:
    name: str
    mesh: object                       # pyvista PolyData (world coordinates - origin)
    origin: np.ndarray

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        import pyvista as pv

        pts = pv.PolyData(np.asarray(points, np.float64) - self.origin)
        out = pts.compute_implicit_distance(self.mesh)
        return np.asarray(out["implicit_distance"], np.float64)


@dataclass
class Contact:
    depth: float                       # along the path (m)
    upper: str
    lower: str
    name: str                          # horizon / contact name
    z: float


@dataclass
class Crossing:
    depth: float
    fault: str
    xyz: np.ndarray


# ---------------------------------------------------------------------------------------------
@dataclass
class GeoModel:
    horizons: list[Grid2D] = field(default_factory=list)          # top to bottom
    units: list[str] = field(default_factory=list)                # len(horizons) + 1, top to bottom
    unconformities: set = field(default_factory=set)              # names of horizons that are unconformities
    faults: list[Fault] = field(default_factory=list)
    topography: Grid2D | None = None
    solids: dict | None = None                                    # voxels: {"origin", "size", "index" (nx,ny,nz)}
    solid_names: list[str] = field(default_factory=list)
    control_xy: np.ndarray | None = None                          # (K, 2) model control (hole traces)
    cards: dict = field(default_factory=dict)                     # formation -> property card (dict)
    config: dict = field(default_factory=lambda: dict(DEFAULTS))
    version: str = ""
    sources: list[str] = field(default_factory=list)

    # --- state ------------------------------------------------------------------------------
    @property
    def ok(self) -> bool:
        return bool(self.horizons) or self.solids is not None

    def describe(self) -> str:
        parts = []
        if self.horizons:
            parts.append(f"{len(self.horizons)} contacts, {len(self.units)} units ({' / '.join(self.units)})")
        if self.solids is not None:
            parts.append(f"{len(self.solid_names)} formation solids")
        parts.append(f"{len(self.faults)} faults")
        if self.topography is not None:
            parts.append("topography")
        return ", ".join(parts)

    # --- look-ups ------------------------------------------------------------------------------
    def horizon_z(self, x, y, dz: float | np.ndarray = 0.0) -> np.ndarray:
        """(H, N) contact elevations at plan points, shifted by ``dz`` (model error trials)."""
        if not self.horizons:
            return np.zeros((0, len(np.atleast_1d(x))))
        return np.stack([h.sample(x, y) for h in self.horizons]) + dz

    def unit_index(self, points: np.ndarray, dz: float | np.ndarray = 0.0) -> np.ndarray:
        """Index into ``units`` of each point (-1: above the topography). Formation solids win
        where they exist (their index is offset by len(units))."""
        p = np.atleast_2d(np.asarray(points, np.float64))
        if self.horizons:
            hz = self.horizon_z(p[:, 0], p[:, 1], dz)
            idx = (hz > p[:, 2][None, :]).sum(axis=0)
            # contacts may cross (thin-out): keep the order monotonic by taking the deepest contact above
        else:
            idx = np.zeros(len(p), dtype=int)
        if self.solids is not None:
            sid = self._solid_index(p)
            m = sid >= 0
            idx = np.where(m, len(self.units) + sid, idx)
        if self.topography is not None:
            idx = np.where(p[:, 2] > self.topography.sample(p[:, 0], p[:, 1]) + 0.5, -1, idx)
        return idx.astype(int)

    def unit_name(self, idx: int) -> str:
        if idx < 0:
            return "air"
        if idx < len(self.units):
            return self.units[idx]
        k = idx - len(self.units)
        return self.solid_names[k] if k < len(self.solid_names) else f"solid {k}"

    @property
    def all_units(self) -> list[str]:
        return list(self.units) + list(self.solid_names)

    def _solid_index(self, p: np.ndarray) -> np.ndarray:
        s = self.solids
        ijk = np.floor((p - s["origin"]) / s["size"]).astype(int)
        shape = np.array(s["index"].shape)
        inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
        out = np.full(len(p), -1, dtype=int)
        if inside.any():
            q = ijk[inside]
            out[inside] = s["index"][q[:, 0], q[:, 1], q[:, 2]]
        return out

    def bedding_normal(self, points: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """Unit normal (E, N, Up) of the bedding at each point: from the contact below the point's
        unit (the one above for the lowest unit)."""
        p = np.atleast_2d(np.asarray(points, np.float64))
        n = np.tile([0.0, 0.0, 1.0], (len(p), 1))
        if not self.horizons:
            return n
        if idx is None:
            idx = self.unit_index(p)
        k = np.clip(np.where(idx < 0, 0, idx), 0, len(self.horizons) - 1)
        for h_i, h in enumerate(self.horizons):
            m = k == h_i
            if m.any():
                gx, gy = h.gradient(p[m, 0], p[m, 1])
                v = np.column_stack([-gx, -gy, np.ones(m.sum())])
                n[m] = v / np.linalg.norm(v, axis=1, keepdims=True)
        return n

    def contact_dip(self, points: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        n = self.bedding_normal(points, idx)
        return np.degrees(np.arccos(np.clip(np.abs(n[:, 2]), 0, 1)))

    def column_at(self, x: float, y: float) -> list[tuple[str, float, float]]:
        """The units at a plan point, top to bottom: (unit, z_top, z_base); z_top of the first is the
        topography (or +inf), z_base of the last -inf."""
        zs = [float(h.sample([x], [y])[0]) for h in self.horizons]
        top = float(self.topography.sample([x], [y])[0]) if self.topography is not None else math.inf
        bounds = [top] + zs + [-math.inf]
        return [(u, bounds[i], bounds[i + 1]) for i, u in enumerate(self.units)]

    # --- along a path ----------------------------------------------------------------------------
    def contacts_along(self, xyz: np.ndarray, depths: np.ndarray, dz: float = 0.0) -> list[Contact]:
        """Where the unit changes along a path (sampled at ``depths``)."""
        idx = self.unit_index(xyz, dz)
        out = []
        for k in np.nonzero(np.diff(idx))[0]:
            a, b = int(idx[k]), int(idx[k + 1])
            if a < 0 or b < 0:
                continue
            name = ""
            if a < len(self.units) and b < len(self.units) and self.horizons:
                name = self.horizons[min(a, b, len(self.horizons) - 1)].name if abs(b - a) == 1 else "contact"
            out.append(Contact(float((depths[k] + depths[k + 1]) / 2), self.unit_name(a), self.unit_name(b), name,
                               float((xyz[k, 2] + xyz[k + 1, 2]) / 2)))
        return out

    def fault_distances(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,) distance to the nearest fault surface and (N,) its index (-1: no faults)."""
        p = np.atleast_2d(points)
        if not self.faults:
            return np.full(len(p), np.inf), np.full(len(p), -1)
        d = np.stack([np.abs(f.signed_distance(p)) for f in self.faults])
        k = np.argmin(d, axis=0)
        return d[k, np.arange(len(p))], k

    def fault_crossings(self, xyz: np.ndarray, depths: np.ndarray) -> list[Crossing]:
        """Where a path passes through a fault surface (sign change of the signed distance close to
        the surface)."""
        out = []
        if len(xyz) < 2:
            return out
        step = float(np.median(np.diff(depths))) if len(depths) > 1 else 1.0
        for f in self.faults:
            sd = f.signed_distance(xyz)
            s = np.sign(sd)
            for k in np.nonzero(s[:-1] * s[1:] < 0)[0]:
                if abs(sd[k]) < 3 * step and abs(sd[k + 1]) < 3 * step:
                    t = sd[k] / (sd[k] - sd[k + 1])
                    out.append(Crossing(float(depths[k] + t * (depths[k + 1] - depths[k])), f.name,
                                        xyz[k] + t * (xyz[k + 1] - xyz[k])))
        return sorted(out, key=lambda c: c.depth)

    # --- error band (spec 5.6) ------------------------------------------------------------------
    def control_distance(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(points)
        if self.control_xy is None or not len(self.control_xy):
            return np.full(len(p), float(self.config.get("d_far", 250.0)))
        from scipy.spatial import cKDTree

        d, _ = cKDTree(self.control_xy).query(p[:, :2])
        return d

    def uncertainty(self, points: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """±u (m) at each point: near / far from model control, widened below units whose thickness
        varies widely (errors add up with depth through every variable unit)."""
        c = self.config
        p = np.atleast_2d(points)
        d = self.control_distance(p)
        u = c["u_near"] + (c["u_far"] - c["u_near"]) * np.clip(d / max(c["d_far"], 1.0), 0, 1)
        if idx is None:
            idx = self.unit_index(p)
        add = np.zeros(len(p))
        for k, unit in enumerate(self.units):
            card = self.cards.get(unit) or {}
            tmin, tmax = _num(card.get("thick_min")), _num(card.get("thick_max"))
            if tmin is None or tmax is None:
                continue
            var = (c["thickness_factor"] * max(tmax - tmin, 0.0)) ** 2
            add += np.where(idx > k, var, 0.0)                  # every variable unit above the point
        return np.sqrt(u ** 2 + add)

    def to_json(self) -> dict:
        return {"units": self.units, "unconformities": sorted(self.unconformities), **self.config}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# --- building ------------------------------------------------------------------------------------
def config_path(folder: str | Path) -> Path:
    return Path(folder) / "drillrisk" / CONFIG_FILE


def load_config(folder: str | Path | None) -> dict:
    if folder is None:
        return {}
    p = config_path(folder)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        log.warning("cannot read %s", p)
        return {}


def save_config(folder: str | Path, cfg: dict) -> Path:
    p = config_path(folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


@dataclass
class SurfaceInput:
    """A surface of the model: name, role, vertices (world) and triangles."""

    name: str
    role: str
    vertices: np.ndarray
    triangles: np.ndarray


def surfaces_from_layers(layers) -> list[SurfaceInput]:
    """The surfaces with a geological-model role among the scene's layers (visible or not)."""
    from ..scene.geo_roles import MODEL_ROLES
    from ..scene.layers import SurfaceLayer

    out = []
    for lay in layers:
        role = getattr(lay, "role", "")
        if not isinstance(lay, SurfaceLayer) or role not in MODEL_ROLES or lay.data is None:
            continue
        tris = [t.triangles for _, t in lay.data.iter_trisolations()]
        if not tris:
            continue
        out.append(SurfaceInput(lay.name, role, np.asarray(lay.data.vertices, np.float64), np.concatenate(tris)))
    return out


def surfaces_from_files(paths_roles: list[tuple[str | Path, str]]) -> list[SurfaceInput]:
    """Read surfaces from files (DTM / DXF / GOCAD) with given roles - for tests and scripts."""
    from ..scene.loaders import load_paths
    from ..scene.layers import SurfaceLayer

    out = []
    for p, role in paths_roles:
        layers, errors = load_paths([Path(p)])
        if errors:
            raise ValueError("; ".join(errors))
        for lay in layers:
            if isinstance(lay, SurfaceLayer):
                tris = [t.triangles for _, t in lay.data.iter_trisolations()]
                out.append(SurfaceInput(Path(p).stem, role, np.asarray(lay.data.vertices, np.float64),
                                        np.concatenate(tris)))
    return out


def build_model(surfaces: list[SurfaceInput], config: dict | None = None, cards: dict | None = None,
                control_xy: np.ndarray | None = None) -> GeoModel:
    """Build the model from its surfaces. ``config`` (geomodel.json) may give the unit names and the
    order of the contacts; otherwise they come from the horizon names and mean elevations."""
    import pyvista as pv

    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (config or {}).items() if v is not None})
    contacts = [s for s in surfaces if s.role in ("horizon", "unconformity")]
    faults = [s for s in surfaces if s.role in ("fault", "fold_axis")]
    topo = [s for s in surfaces if s.role == "topography"]
    solids = [s for s in surfaces if s.role == "formation"]
    allv = [s.vertices[np.unique(s.triangles.ravel())] for s in contacts + topo + solids + faults]
    if not allv:
        return GeoModel(config=cfg, cards=cards or {})
    pts = np.concatenate(allv)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    d = float(cfg.get("grid", GRID))
    x0, y0 = math.floor(lo[0] / d) * d - d, math.floor(lo[1] / d) * d - d
    nx = int(math.ceil((hi[0] - x0) / d)) + 2
    ny = int(math.ceil((hi[1] - y0) / d)) + 2
    grids = [(s, grid_surface(s.name, s.vertices, s.triangles, x0, y0, d, nx, ny)) for s in contacts]
    order = cfg.get("order")
    if order:
        rank = {n: i for i, n in enumerate(order)}
        grids.sort(key=lambda g: rank.get(g[0].name, 1e9 - float(np.median(g[1].z))))
    else:
        grids.sort(key=lambda g: -float(np.median(g[1].z[g[1].covered] if g[1].covered.any() else g[1].z)))
    model = GeoModel(config=cfg, cards=dict(cards or {}), control_xy=control_xy)
    model.horizons = [g for _, g in grids]
    model.unconformities = {s.name for s, _ in grids if s.role == "unconformity"}
    model.units = _unit_names([s.name for s, _ in grids], cfg.get("units"))
    model.sources = [s.name for s in surfaces]
    if topo:
        model.topography = grid_surface(topo[0].name, topo[0].vertices, topo[0].triangles, x0, y0, d, nx, ny)
    origin = np.array([x0, y0, 0.0])
    for s in faults:
        used, inv = np.unique(s.triangles.ravel(), return_inverse=True)
        cells = np.column_stack([np.full(len(s.triangles), 3), inv.reshape(-1, 3)]).ravel()
        mesh = pv.PolyData(s.vertices[used] - origin, faces=cells).clean()
        model.faults.append(Fault(s.name, mesh, origin))
    if solids:
        model.solids, model.solid_names = _voxelize(solids, cfg.get("voxel", DEFAULTS["voxel"]), lo, hi)
    model.version = _version(surfaces)
    return model


def _unit_names(horizon_names: list[str], configured) -> list[str]:
    n = len(horizon_names) + 1
    if configured and len(configured) == n:
        return [str(u) for u in configured]
    names: list[str | None] = [None] * n
    for i, h in enumerate(horizon_names):
        pair = unit_pair(h)
        if pair:
            names[i] = names[i] or pair[0]
            names[i + 1] = pair[1]
    return [u or f"Unit {i + 1}" for i, u in enumerate(names)]


def _voxelize(solids: list[SurfaceInput], size, lo, hi):
    """Formation solids to voxels (spec 3.2: e.g. 5 x 5 x 2 m): index of the solid at each voxel, -1 none."""
    import pyvista as pv

    size = np.asarray(size, np.float64)
    origin = np.floor(lo / size) * size
    shape = np.maximum(np.ceil((hi - origin) / size).astype(int), 1)
    if int(np.prod(shape)) > 6_000_000:                    # keep it light: coarser voxels for big models
        f = (np.prod(shape) / 6_000_000) ** (1 / 3)
        size = size * f
        shape = np.maximum(np.ceil((hi - origin) / size).astype(int), 1)
    centers = np.stack(np.meshgrid(*[origin[a] + (np.arange(shape[a]) + 0.5) * size[a] for a in range(3)],
                                   indexing="ij"), axis=-1).reshape(-1, 3)
    index = np.full(len(centers), -1, dtype=np.int16)
    names = []
    o = origin
    for k, s in enumerate(solids):
        used, inv = np.unique(s.triangles.ravel(), return_inverse=True)
        cells = np.column_stack([np.full(len(s.triangles), 3), inv.reshape(-1, 3)]).ravel()
        mesh = pv.PolyData(s.vertices[used] - o, faces=cells).clean()
        b = mesh.bounds
        sel = np.all((centers - o >= [b[0], b[2], b[4]]) & (centers - o <= [b[1], b[3], b[5]]), axis=1)
        if not sel.any():
            names.append(s.name)
            continue
        cloud = pv.PolyData(centers[sel] - o)
        try:
            inside = cloud.select_interior_points(mesh, check_surface=False)["selected_points"].astype(bool)
        except AttributeError:                                       # older pyvista
            inside = cloud.select_enclosed_points(mesh, check_surface=False)["SelectedPoints"].astype(bool)
        ids = np.nonzero(sel)[0][inside]
        index[ids] = k
        names.append(s.name)
    return {"origin": origin, "size": size, "index": index.reshape(shape)}, names


def _version(surfaces) -> str:
    import hashlib

    h = hashlib.sha1()
    for s in surfaces:
        h.update(s.name.encode("utf-8"))
        h.update(s.role.encode("utf-8"))
        h.update(np.round(np.nan_to_num(s.vertices[:: max(1, len(s.vertices) // 200)]), 2).tobytes())
    return h.hexdigest()[:8]
