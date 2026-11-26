"""The underground survey in the view: the drives built from the survey (each pit's centre
line swept with its rectangular profile), the centre lines and the stations.

The drives are made from the survey tables, not from files: whenever a station or a setup
changes, the layer is read again and the drives follow. The geology mapped in the pits (rocks on
the walls and the back, structures as discs tilted to their dip, samples as narrow bands; see
scene/pit_geology.py) is drawn on the drives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from ..i18n import N_, tr
from .layers import IdentifyResult, Layer, _to_u8, _xyz_rows

SURVEY_LAYER_NAME = "Survey (database)"
FAMILY_COLORS = [(0.95, 0.62, 0.22), (0.35, 0.70, 0.95), (0.50, 0.82, 0.42), (0.85, 0.50, 0.90),
                 (0.95, 0.85, 0.35), (0.35, 0.85, 0.80), (0.95, 0.45, 0.45), (0.65, 0.65, 0.95)]
STATUS_COLORS = {"known": (0.25, 0.55, 1.0), "entered": (0.72, 0.55, 1.0), "closed": (0.35, 0.85, 0.40),
                 "open": (0.98, 0.80, 0.25), "failed": (0.95, 0.30, 0.30), "unresolved": (0.60, 0.60, 0.60)}
PIT_STYLE_DEFAULTS = {"visible": True, "color": None, "centerline": True, "stations": True}
DESIGN_COLOR = (0.30, 0.90, 0.45)           # designed pits (route designer): green, as in its preview
DESIGN_STYLE_DEFAULTS = {"visible": True, "color": None, "centerline": True}
STATUS_TEXT = {"known": N_("known (entered)"), "entered": N_("coordinates given (already calculated)"),
               "closed": N_("computed, traverse closed"), "open": N_("computed, not checked"),
               "failed": N_("computed, closure failed"), "unresolved": N_("no coordinates yet")}   # shown: tr(text)
PROFILE_TEXT = {"back": N_("{w:g} x {h:g} m, stations in the backs"),                       # shown: tr(text, w=, h=)
                "floor": N_("{w:g} x {h:g} m, stations in the floor"),
                "middle": N_("{w:g} x {h:g} m, stations in the middle")}


@dataclass
class SurveyData:
    """What the layer draws: the pits, the stations with coordinates and each pit's paths
    (lists of station names, in order)."""
    pits: list
    stations: dict                          # name -> survey_store.Station
    paths: dict                             # pit id -> [[station, station, ...], ...]
    source: str = ""
    warnings: list = field(default_factory=list)
    designs: list = field(default_factory=list)  # survey_store.DesignPit: drives designed, not surveyed
    geology: dict = field(default_factory=dict)  # pit id -> pit_geology_store.PitGeology (mapped geology)
    rock_colors: dict = field(default_factory=dict)  # rock code -> "#rrggbb" (drillhole_styles)
    assay_fields: list = field(default_factory=list)  # the pit samples' assay fields

    def pit(self, pid):
        return next((p for p in self.pits if p.id == pid), None)

    def design(self, did):
        return next((d for d in self.designs if d.id == did), None)

    def family(self, pid) -> int:
        """Index of the main pit a pit belongs to (its branches share the main pit's colour)."""
        by_id = {p.id: p for p in self.pits}
        p, seen = by_id.get(pid), set()
        while p is not None and p.parent_id is not None and p.parent_id in by_id and p.id not in seen:
            seen.add(p.id)
            p = by_id[p.parent_id]
        mains = [q.id for q in self.pits if q.parent_id is None]
        return mains.index(p.id) if p is not None and p.id in mains else 0

    def main_of(self, pid):
        """Id of the main pit that ``pid`` belongs to (itself for a main pit)."""
        by_id = {p.id: p for p in self.pits}
        p, seen = by_id.get(pid), set()
        while p is not None and p.parent_id is not None and p.parent_id in by_id and p.id not in seen:
            seen.add(p.id)
            p = by_id[p.parent_id]
        return p.id if p is not None else pid

    def ancestors(self, pid) -> list:
        """The pit's parents, nearest first."""
        by_id = {p.id: p for p in self.pits}
        out, p = [], by_id.get(pid)
        while p is not None and p.parent_id is not None and p.parent_id in by_id and p.parent_id not in out:
            out.append(p.parent_id)
            p = by_id[p.parent_id]
        return out

    def children(self, pid) -> list:
        return [p for p in self.pits if p.parent_id == pid]

    def xyz(self, name):
        s = self.stations.get(name)
        return None if s is None or None in (s.x, s.y, s.z) else np.array([s.x, s.y, s.z], dtype=np.float64)


def hex_rgb(color: str, default=(0.6, 0.6, 0.6)) -> tuple:
    """"#rrggbb" -> (r, g, b) in 0..1 (``default`` when it is not one)."""
    h = str(color or "").strip().lstrip("#")
    try:
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) if len(h) == 6 else default
    except ValueError:
        return default


def pit_paths(setups: dict) -> dict:
    """pit id -> its centre lines: the setups' legs (at -> foresight) joined where one leg
    starts at the previous leg's end (a new line where they do not)."""
    out: dict = {}
    for pid, rows in setups.items():
        lines: list = []
        for r in rows:
            if not r.at or not r.fs:
                continue
            if lines and lines[-1][-1] == r.at:
                lines[-1].append(r.fs)
            else:
                lines.append([r.at, r.fs])
        out[pid] = lines
    return out


# the four walls of the first leg (corners 0-3 of a point: bottom left, bottom right, top
# right, top left); each further leg is the same 4 vertices on
_WALLS = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)
_EAST = np.array([1.0, 0.0])


def _ray_hits(c, u, wall):
    """Parameters t where the plan ray ``c + t u`` crosses the polyline ``wall`` (N, 2)."""
    a, e = wall[:-1], wall[1:] - wall[:-1]
    den = u[0] * e[:, 1] - u[1] * e[:, 0]
    ok = np.abs(den) > 1e-12
    w = a - c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / den
        s = (w[:, 0] * u[1] - w[:, 1] * u[0]) / den
    return t[ok & (s >= -1e-9) & (s <= 1 + 1e-9)]


@dataclass
class OtherDrive:
    """Another drive that a drive starts or ends on: its centre line, profile and plan walls."""
    P: np.ndarray                      # centre line (K, 3)
    width: float
    height: float
    level: str
    walls: list                        # plan polylines: left and right wall

    def contains(self, pts: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        """Which of the points (N, 3) are inside this drive (within its walls, between its floor
        and its roof there; ``tol`` m of slack on every side)."""
        a, b = self.P[:-1], self.P[1:]
        ab = b[:, :2] - a[:, :2]
        L2 = np.maximum((ab ** 2).sum(axis=1), 1e-18)
        rel = pts[:, None, :2] - a[None, :, :2]                              # (N, legs, 2)
        s = np.clip((rel * ab[None]).sum(axis=2) / L2[None], 0.0, 1.0)
        near = a[None, :, :2] + s[..., None] * ab[None]
        dist = np.hypot(*(pts[:, None, :2] - near).transpose(2, 0, 1))      # (N, legs)
        leg = dist.argmin(axis=1)
        i = np.arange(len(pts))
        zc = a[leg, 2] + s[i, leg] * (b[leg, 2] - a[leg, 2])
        floor = zc + {"back": -self.height, "middle": -self.height / 2.0, "floor": 0.0}.get(self.level, -self.height)
        return ((dist[i, leg] <= self.width / 2.0 + tol) & (pts[:, 2] >= floor - tol)
                & (pts[:, 2] <= floor + self.height + tol))


def _cut_ring(ring, p0, p1, walls, limit: float = 0.95) -> None:
    """Move an end profile (``ring``, its 4 corners, changed in place) along its leg (from the
    end point ``p0`` towards ``p1``) out of the other drive(s) it starts in - so drives meeting
    at a junction neither overlap nor leave a gap.

    ``walls``: OtherDrive objects (or bare plan wall polylines: a cut in plan only). In plan,
    each corner goes to where its own line meets the other drive's wall (a drive meeting the
    wall at an angle gets a slanted end lying on the wall). In 3D, a corner whose line is no
    longer inside the other drive there - above its roof or below its floor, as where an
    inclined drive meets a level one - stops where its line leaves the other drive instead
    (back to the end point at most), so the wedge of drive above the roof / below the floor is
    kept. Never moves more than ``limit`` of the leg."""
    d = p1[:2] - p0[:2]
    length = float(np.hypot(*d))
    if length <= 1e-9:
        return
    u = d / length
    others = [w for w in walls if isinstance(w, OtherDrive)]
    polys = [w for w in walls if not isinstance(w, OtherDrive)] + [p for o in others for p in o.walls]
    out = [ts[ts > 1e-9].min() for ts in (_ray_hits(p0[:2], u, w) for w in polys) if (ts > 1e-9).any()]
    if not out:
        return
    t0 = float(min(out))                              # where the centre line leaves the other drive
    slope = (p1[2] - p0[2]) / length
    step = np.array([u[0], u[1], slope])

    def inside(c, ts):
        pts = c[None, :] + np.asarray(ts, float)[:, None] * step[None, :]
        return np.any([o.contains(pts) for o in others], axis=0) if others else np.ones(len(pts), bool)

    for j in range(4):
        c = ring[j].copy()
        ts = np.concatenate([_ray_hits(c[:2], u, w) for w in polys])
        t = float(ts[np.argmin(np.abs(ts - t0))]) if ts.size else t0
        t = min(max(t, -length), limit * length)
        if t > 0 and others:
            tw = t
            probe = np.linspace(tw - 1e-4, 0.0, 61)       # from the wall back to the end point
            ins = inside(c, probe)
            if not ins[0]:                                # past the other drive's roof / floor at the wall
                k = int(np.argmax(ins)) if ins.any() else -1
                if k <= 0:
                    t = 0.0                               # never inside: the drive keeps its end here
                else:
                    lo, hi = probe[k], probe[k - 1]       # inside at lo, outside at hi: find the edge
                    for _ in range(30):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if inside(c, [mid])[0] else (lo, mid)
                    t = lo
        ring[j] = c + t * step


def _other(data: "SurveyData", pit, path, origin):
    """``pit``'s drive along ``path`` (station names), as drawn without any cut, as an
    OtherDrive; None without 2 points."""
    pts = [x for x in (data.xyz(n) for n in path) if x is not None]
    if len(pts) < 2:
        return None
    P = np.array(pts) - (0.0 if origin is None else origin)
    V = drive_mesh(P, pit.profile_width, pit.profile_height, pit.station_level)[0].reshape(-1, 4, 3)
    return OtherDrive(P, pit.profile_width, pit.profile_height, pit.station_level,
                      [V[:, 0, :2].copy(), V[:, 1, :2].copy()])


def parent_walls(data: "SurveyData", pit, origin=None) -> list:
    """The drives a branch starts in: the parent's drive along each parent line that passes the
    branch's start station partway (not at the line's first or last station - a branch carrying
    on from the end of its parent stays joined at the station); [] for a main pit."""
    parent = data.pit(pit.parent_id) if pit.parent_id is not None and pit.start_station else None
    if parent is None:
        return []
    out = [_other(data, parent, path, origin) for path in data.paths.get(parent.id, [])
           if pit.start_station in path[1:-1]]
    return [o for o in out if o is not None]


def junction_walls(data: "SurveyData", pit, path, origin=None) -> list:
    """The drives a pit's line ends in: when the line ends at a station where 2 or more of its
    branches start (a T or crossing, e.g. a decline reaching a level that runs both ways), the
    branches' drives there; else [] (a single branch carrying on stays joined at the station)."""
    end = path[-1]
    if any(other is not path and end in other[:-1] for other in data.paths.get(pit.id, [])):
        return []                                     # the pit itself carries on from there
    lines = [(c, q) for c in data.children(pit.id) if c.start_station == end
             for q in data.paths.get(c.id, []) if q and q[0] == end]
    if len({c.id for c, _q in lines}) < 2:
        return []
    out = [_other(data, c, q, origin) for c, q in lines]
    return [o for o in out if o is not None]


def pit_drives(data: "SurveyData", pit, origin=None) -> list:
    """The pit's drives: [(centre line (K, 3), vertices, quads)] for each of its lines. Where
    drives meet they do not overlap: a branch leaving its parent partway starts on the parent's
    wall, and a line ending where its branches run both ways ends on their wall."""
    out = []
    for path in data.paths.get(pit.id, []):
        pts = [x for x in (data.xyz(n) for n in path) if x is not None]
        if len(pts) < 2:
            continue
        P = np.array(pts) - (0.0 if origin is None else origin)
        start = (parent_walls(data, pit, origin)
                 if path[0] == pit.start_station and data.xyz(path[0]) is not None else [])
        end = junction_walls(data, pit, path, origin) if data.xyz(path[-1]) is not None else []
        V, Q = drive_mesh(P, pit.profile_width, pit.profile_height, pit.station_level,
                          start_walls=start, end_walls=end)
        out.append((P, V, Q))
    return out


def _drives_volume(data: "SurveyData", pit) -> float:
    """Volume (m³) of a pit's drives: the closed tubes as drawn (cross-section x length where a tube is
    not closed)."""
    if pit is None:
        return 0.0
    first = next((x for path in data.paths.get(pit.id, []) for x in (data.xyz(n) for n in path) if x is not None),
                 None)                                # local coordinates: no precision lost far from 0
    total = 0.0
    for P, V, Q in pit_drives(data, pit, first):
        mesh = pv.PolyData(np.asarray(V, np.float64), faces=np.hstack([np.full((len(Q), 1), 4), Q]).ravel())
        mesh = mesh.triangulate().clean()
        if mesh.n_cells and mesh.n_open_edges == 0:
            total += abs(float(mesh.volume))
        else:
            total += pit.profile_width * pit.profile_height * float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return total


def _tube_volume(V, Q, fallback: float) -> float:
    mesh = pv.PolyData(np.asarray(V, np.float64), faces=np.hstack([np.full((len(Q), 1), 4), Q]).ravel())
    mesh = mesh.triangulate().clean()
    return abs(float(mesh.volume)) if mesh.n_cells and mesh.n_open_edges == 0 else fallback


def design_volume(d) -> float:
    """Volume (m³) of a designed pit's drive (the closed tube along its centre line)."""
    P = np.asarray(d.points, np.float64)
    if len(P) < 2:
        return 0.0
    P = P - P[0]
    V, Q = drive_mesh(P, d.profile_width, d.profile_height, d.station_level)
    return _tube_volume(V, Q, d.profile_width * d.profile_height * float(np.linalg.norm(np.diff(P, axis=0),
                                                                                       axis=1).sum()))


def drive_mesh(points: np.ndarray, width: float, height: float, level: str = "back", start_walls=None,
               end_walls=None):
    """A closed rectangular tube along ``points`` (K, 3): walls vertical, ``width`` x ``height``,
    the points in the back (roof), the floor or the middle of the profile. Bends are mitred so
    the walls keep their width. Returns (vertices (4K, 3), quads (M, 4)).

    ``start_walls`` / ``end_walls`` (OtherDrive objects, or plan wall polylines): the tube starts /
    ends where it leaves that other drive (on its wall, or its roof / floor) instead of at the
    first / last point.

    Done with array operations for all points at once (no loop over the points)."""
    P = np.asarray(points, dtype=np.float64)
    k = len(P)
    seg = P[1:, :2] - P[:-1, :2]
    lens = np.sqrt(seg[:, 0] ** 2 + seg[:, 1] ** 2)
    good = lens > 1e-9
    if good.all():                                       # the usual case: no vertical legs
        dirs = seg / lens[:, None]
    else:
        # a vertical leg keeps the direction of the last leg before it that has one (the very
        # first legs, if vertical, point east): forward-fill by the index of the last good leg
        unit = np.zeros_like(seg)
        unit[good] = seg[good] / lens[good, None]
        last = np.maximum.accumulate(np.where(good, np.arange(k - 1), -1))
        dirs = np.where((last >= 0)[:, None], unit[np.maximum(last, 0)], _EAST)
    # at each point: the mitre between the leg before (A) and the leg after (B)
    A = np.vstack([dirs[:1], dirs])                      # point i: leg i-1 (the first point: leg 0)
    B = np.vstack([dirs, dirs[-1:]])                     # point i: leg i (the last point: the last leg)
    d = A + B
    n = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    ok = n > 1e-9
    if ok.all():
        d = d / n[:, None]
    else:                                                # a U-turn: the direction of the leg after
        d = np.where(ok[:, None], d / np.where(ok, n, 1.0)[:, None], B)
    # to the right of the direction of travel, stretched at bends so the walls keep the width
    cos = d[:, 0] * B[:, 0] + d[:, 1] * B[:, 1]          # dot(right of d, right of B)
    half = (width / 2.0) / np.maximum(cos, 0.35)
    rx, ry = d[:, 1] * half, -d[:, 0] * half
    off = {"back": -height, "middle": -height / 2.0, "floor": 0.0}.get(level, -height)
    V = np.repeat(P[:, None, :], 4, axis=1)              # (K, 4 corners, 3)
    V[:, :, 2] += off
    V[:, 2:, 2] += height
    V[:, (0, 3), 0] -= rx[:, None]
    V[:, (0, 3), 1] -= ry[:, None]
    V[:, (1, 2), 0] += rx[:, None]
    V[:, (1, 2), 1] += ry[:, None]
    limit = 0.45 if start_walls and end_walls and k == 2 else 0.95      # both ends of one leg: never cross
    if start_walls:
        _cut_ring(V[0], P[0], P[1], start_walls, limit)
    if end_walls:
        _cut_ring(V[-1], P[-1], P[-2], end_walls, limit)
    e = (k - 1) * 4
    walls = (_WALLS[None, :, :] + 4 * np.arange(k - 1, dtype=np.int64)[:, None, None]).reshape(-1, 4)
    caps = np.array([[3, 2, 1, 0], [e, e + 1, e + 2, e + 3]], dtype=np.int64)     # end caps
    return V.reshape(-1, 3), np.vstack([walls, caps])


@dataclass(eq=False)
class SurveyLayer(Layer):
    data: SurveyData | None = None
    kind: str = N_("Survey")
    opacity: float = 0.55
    show_drives: bool = True
    show_labels: bool = True
    show_designs: bool = True                       # the "Designed Pits" group switched on
    show_geology: bool = True                       # mapped rocks on the walls and the back
    show_structures: bool = True                    # mapped structures (discs tilted to their dip)
    show_samples: bool = True                       # pit samples (narrow bands on the walls)
    structure_size: float = 1.0                     # m: radius of a structure's disc
    # per pit (key: pit id as text): visible, color (None = the parent's / the main pit's
    # default), and for main pits centerline / stations (for the main pit and its branches);
    # per designed pit (key "d" + its id): visible, color (None = DESIGN_COLOR), centerline
    pit_styles: dict = field(default_factory=dict)

    SETTINGS = Layer.SETTINGS + ("show_drives", "show_labels", "show_designs", "pit_styles", "show_geology",
                                 "show_structures", "show_samples", "structure_size")

    def _dict_setting(self, key, value):
        if key != "pit_styles":
            return super()._dict_setting(key, value)
        out = {}
        for pid, st in value.items():
            st = dict(st)
            if st.get("color") is not None:
                st["color"] = [float(x) for x in st["color"]]
            out[str(pid)] = st
        return out

    def _setting_ok(self, key, value):
        return isinstance(value, dict) if key == "pit_styles" else super()._setting_ok(key, value)

    # --- per pit display ----------------------------------------------------------------
    def pit_style(self, pid) -> dict:
        st = dict(PIT_STYLE_DEFAULTS)
        saved = self.pit_styles.get(str(pid))
        if isinstance(saved, dict):
            st.update({k: saved[k] for k in PIT_STYLE_DEFAULTS if k in saved})
        if st["color"] is not None:
            try:
                st["color"] = tuple(float(x) for x in st["color"])[:3]
            except (TypeError, ValueError):
                st["color"] = None
        return st

    def set_pit_style(self, pid, **kw) -> None:
        st = dict(self.pit_styles.get(str(pid)) or {})
        for k, v in kw.items():
            if k in PIT_STYLE_DEFAULTS:
                st[k] = list(v) if k == "color" and v is not None else v
        self.pit_styles[str(pid)] = st
        if self.scene is not None:
            self.rebuild()

    def pit_shown(self, pid) -> bool:
        """Drawn: the pit and every pit it branches from are switched on."""
        d = self.data
        return self.pit_style(pid)["visible"] and all(self.pit_style(a)["visible"] for a in d.ancestors(pid))

    def pit_color(self, pid) -> tuple:
        """The pit's own colour, else the nearest parent's, else its main pit's default."""
        d = self.data
        for q in [pid] + d.ancestors(pid):
            c = self.pit_style(q)["color"]
            if c is not None:
                return c
        return FAMILY_COLORS[d.family(pid) % len(FAMILY_COLORS)]

    def default_color(self, pid) -> tuple:
        """The colour the pit has without a colour of its own (inherited / default)."""
        d = self.data
        for q in d.ancestors(pid):
            c = self.pit_style(q)["color"]
            if c is not None:
                return c
        return FAMILY_COLORS[d.family(pid) % len(FAMILY_COLORS)]

    # --- designed pits ------------------------------------------------------------------
    def design_style(self, did) -> dict:
        st = dict(DESIGN_STYLE_DEFAULTS)
        saved = self.pit_styles.get(f"d{did}")
        if isinstance(saved, dict):
            st.update({k: saved[k] for k in DESIGN_STYLE_DEFAULTS if k in saved})
        if st["color"] is not None:
            try:
                st["color"] = tuple(float(x) for x in st["color"])[:3]
            except (TypeError, ValueError):
                st["color"] = None
        return st

    def set_design_style(self, did, **kw) -> None:
        st = dict(self.pit_styles.get(f"d{did}") or {})
        for k, v in kw.items():
            if k in DESIGN_STYLE_DEFAULTS:
                st[k] = list(v) if k == "color" and v is not None else v
        self.pit_styles[f"d{did}"] = st
        if self.scene is not None:
            self.rebuild()

    def set_designs_centerline(self, on: bool) -> None:
        """The centre lines of all designed pits on / off."""
        for x in (self.data.designs if self.data is not None else []):
            st = dict(self.pit_styles.get(f"d{x.id}") or {})
            st["centerline"] = bool(on)
            self.pit_styles[f"d{x.id}"] = st
        if self.scene is not None:
            self.rebuild()

    def set_show_designs(self, on: bool) -> None:
        self.show_designs = bool(on)
        if self.scene is not None:
            self.rebuild()

    def design_shown(self, did) -> bool:
        return self.show_designs and self.design_style(did)["visible"]

    def design_color(self, did) -> tuple:
        return self.design_style(did)["color"] or DESIGN_COLOR

    def design_volume(self, did) -> float:
        d = self.data.design(did) if self.data is not None else None
        if d is None:
            return 0.0
        cache = self.data.__dict__.setdefault("_volumes", {})
        if ("d", did) not in cache:
            cache[("d", did)] = design_volume(d)
        return cache[("d", did)]

    def design_bounds(self, did=None):
        """World bounds of a designed pit (None: of all of them); None without points."""
        d = self.data
        ds = [x for x in (d.designs if d is not None else []) if did is None or x.id == did]
        pts = [p for x in ds for p in x.points]
        if not pts:
            return None
        a = np.asarray(pts, float)
        pad = max(max(x.profile_width, x.profile_height) for x in ds)
        return a.min(axis=0) - pad, a.max(axis=0) + pad

    def pit_family(self, pid) -> set:
        """The pit and all its branches (and theirs)."""
        d = self.data
        ids, todo = set(), [pid]
        while todo:
            q = todo.pop()
            if q not in ids:
                ids.add(q)
                todo += [c.id for c in d.children(q)]
        return ids

    def pit_volume(self, pid, branches: bool = False) -> float:
        """Excavated volume (m³) of the pit's drives as drawn - a branch counted from where it leaves its
        parent's wall, so junctions are not counted twice; with ``branches`` also all its branches."""
        d = self.data
        if d is None:
            return 0.0
        cache = d.__dict__.setdefault("_volumes", {})
        ids = self.pit_family(pid) if branches else {pid}
        total = 0.0
        for q in ids:
            if q not in cache:
                cache[q] = _drives_volume(d, d.pit(q))
            total += cache[q]
        return total

    def pit_bounds(self, pid):
        """World bounds of a pit and its branches (None without coordinates)."""
        d = self.data
        ids = self.pit_family(pid)
        pts = [d.xyz(n) for q in ids for path in d.paths.get(q, []) for n in path]
        pts = [x for x in pts if x is not None]
        if not pts:
            return None
        a = np.array(pts)
        p = d.pit(pid)
        pad = max(p.profile_width, p.profile_height) if p else 5.0
        return a.min(axis=0) - pad, a.max(axis=0) + pad

    @property
    def removable(self) -> bool:                    # comes from the database: hide it instead
        return False

    def set_opacity(self, v: float) -> None:
        """The drives' opacity (the mapped geology stays solid)."""
        self.opacity = float(v)
        for a in self.actors:
            if not getattr(a, "_geoviewer_solid", False):
                a.GetProperty().SetOpacity(self.opacity)

    # --- mapped geology ----------------------------------------------------------------------
    def rock_rgb(self, code: str) -> tuple:
        """A rock's colour: drillhole_styles, else a palette colour by the code's place."""
        from .layers import PALETTE

        d = self.data
        c = (d.rock_colors or {}).get(code) if d is not None else None
        if c:
            return hex_rgb(c)
        codes = sorted({r.rock for g in (d.geology.values() if d else []) for r in g.intervals}
                       - set((d.rock_colors or {}) if d else ()))
        return PALETTE[codes.index(code) % len(PALETTE)] if code in codes else (0.6, 0.6, 0.6)

    def axis(self, pid):
        """The pit's chainage axis (cached until the data change); None without two stations."""
        from .pit_geology import pit_axis

        d = self.data
        cache = d.__dict__.setdefault("_axes", {})
        if pid not in cache:
            cache[pid] = pit_axis(d, pid)
        return cache[pid]

    def _build_geology(self, o) -> None:
        from ..io.pit_geology_store import STRUCTURE_COLORS
        from .pit_geology import interval_bands, sample_ribbon, structure_disc

        d = self.data
        self._geo_items = []
        parts = {"rocks": ([], [], [], []), "structures": ([], [], [], []), "samples": ([], [], [], [])}

        def put(kind, V, F, item, rgb):
            verts, faces, ids, cols = parts[kind]
            base = sum(len(v) for v in verts)
            verts.append(V - o)
            faces.append(F + base)
            ids += [len(self._geo_items)] * len(F)
            cols += [_to_u8(rgb)] * len(F)

        sample_rgb = hex_rgb("#1de9b6")
        for pid, geo in sorted(d.geology.items()):
            if d.pit(pid) is None or not self.pit_shown(pid):
                continue
            ax = self.axis(pid)
            if ax is None:
                continue
            if self.show_geology:
                for i, r in enumerate(geo.intervals):
                    bands = interval_bands(ax, r)
                    if bands:
                        for _w, V, F in bands:
                            put("rocks", V, F, None, self.rock_rgb(r.rock))
                        self._geo_items.append(("interval", pid, i))
            if self.show_structures:
                for i, s in enumerate(geo.structures):
                    got = structure_disc(ax, s, max(float(self.structure_size), 0.05))
                    if got is not None:
                        put("structures", *got, None, hex_rgb(STRUCTURE_COLORS.get(s.kind, "#b0b6bd")))
                        self._geo_items.append(("structure", pid, i))
            if self.show_samples:
                for i, s in enumerate(geo.samples):
                    got = sample_ribbon(ax, s)
                    if got is not None:
                        put("samples", *got, None, sample_rgb)
                        self._geo_items.append(("sample", pid, i))
        for kind, (verts, faces, ids, cols) in parts.items():
            if not verts:
                continue
            F = np.vstack(faces)
            n = F.shape[1]
            mesh = pv.PolyData(np.vstack(verts).astype(np.float64), faces=np.hstack([np.full((len(F), 1), n), F]).ravel())
            mesh.cell_data["geo"] = np.asarray(ids, dtype=np.int64)
            mesh.cell_data["rgb"] = np.asarray(cols, dtype=np.uint8)
            act = self._add(mesh, scalars="rgb", rgb=True, show_edges=False, ambient=0.35,
                            lighting=kind != "samples")
            act._geoviewer_solid = True

    def _build_actors(self) -> None:
        d = self.data
        if d is None:
            return
        o = self.scene.origin
        verts, quads, q_pit, q_rgb = [], [], [], []
        line_pts, line_cells, l_pit, l_rgb = [], [], [], []
        base = 0
        lbase = 0
        for p in d.pits:
            if not self.pit_shown(p.id):
                continue
            rgb = _to_u8(self.pit_color(p.id))
            centre = self.pit_style(d.main_of(p.id))["centerline"]
            for P, V, Q in pit_drives(d, p, o):          # a branch starts at its parent's wall
                if self.show_drives:
                    verts.append(V)
                    quads.append(Q + base)
                    base += len(V)
                    q_pit += [p.id] * len(Q)
                    q_rgb += [rgb] * len(Q)
                if not centre:
                    continue
                line_pts.append(P)
                line_cells.append(np.r_[len(P), np.arange(lbase, lbase + len(P))])
                lbase += len(P)
                l_pit.append(p.id)
                l_rgb.append(rgb)
        if verts:
            Q = np.vstack(quads)
            faces = np.hstack([np.full((len(Q), 1), 4), Q]).ravel()
            mesh = pv.PolyData(np.vstack(verts).astype(np.float64), faces=faces)
            mesh.cell_data["pit"] = np.asarray(q_pit, dtype=np.int64)
            mesh.cell_data["rgb"] = np.asarray(q_rgb, dtype=np.uint8)
            self._add(mesh, scalars="rgb", rgb=True, opacity=self.opacity, show_edges=False, ambient=0.25)
        if line_pts:
            lines = pv.PolyData(np.vstack(line_pts).astype(np.float32), lines=np.concatenate(line_cells))
            lines.cell_data["pit"] = np.asarray(l_pit, dtype=np.int64)
            lines.cell_data["rgb"] = np.asarray(l_rgb, dtype=np.uint8)
            self._add(lines, scalars="rgb", rgb=True, line_width=2.5, lighting=False)
        self._build_designs(o)
        if d.geology:
            self._build_geology(o)

        def station_shown(s):
            pid = s.pit_id if d.pit(s.pit_id) is not None else None
            if pid is None:
                return True
            return self.pit_shown(pid) and self.pit_style(d.main_of(pid))["stations"]

        names = [n for n, s in d.stations.items() if d.xyz(n) is not None and s.active and station_shown(s)]
        if names:
            pts = np.array([d.xyz(n) for n in names]) - o
            cloud = pv.PolyData(pts.astype(np.float32))
            cloud.cell_data["station"] = np.arange(len(names))
            cloud.cell_data["rgb"] = np.array([_to_u8(STATUS_COLORS.get(d.stations[n].status, (0.6, 0.6, 0.6)))
                                               for n in names], dtype=np.uint8)
            self._station_names = names
            self._add(cloud, scalars="rgb", rgb=True, point_size=9, render_points_as_spheres=True)
            if self.show_labels:
                act = self.scene.plotter.add_point_labels(
                    pts.astype(np.float32), names, font_size=10, text_color=self.scene.foreground, shape=None,
                    show_points=False, always_visible=False, render=False, name=f"{id(self)}-labels")
                act._geoviewer_layer = self
                self.actors.append(act)

    def _build_designs(self, o) -> None:
        """The designed pits: their drives (the same see-through look as the surveyed ones) and centre lines."""
        verts, quads, q_id, q_rgb = [], [], [], []
        line_pts, line_cells, l_id, l_rgb = [], [], [], []
        base = lbase = 0
        for d in self.data.designs:
            if len(d.points) < 2 or not self.design_shown(d.id):
                continue
            P = np.asarray(d.points, np.float64) - o
            rgb = _to_u8(self.design_color(d.id))
            if self.show_drives:
                V, Q = drive_mesh(P, d.profile_width, d.profile_height, d.station_level)
                verts.append(V)
                quads.append(Q + base)
                base += len(V)
                q_id += [d.id] * len(Q)
                q_rgb += [rgb] * len(Q)
            if not self.design_style(d.id)["centerline"]:
                continue
            line_pts.append(P)
            line_cells.append(np.r_[len(P), np.arange(lbase, lbase + len(P))])
            lbase += len(P)
            l_id.append(d.id)
            l_rgb.append(rgb)
        if verts:
            Q = np.vstack(quads)
            mesh = pv.PolyData(np.vstack(verts), faces=np.hstack([np.full((len(Q), 1), 4), Q]).ravel())
            mesh.cell_data["design"] = np.asarray(q_id, dtype=np.int64)
            mesh.cell_data["rgb"] = np.asarray(q_rgb, dtype=np.uint8)
            self._add(mesh, scalars="rgb", rgb=True, opacity=self.opacity, show_edges=False, ambient=0.25)
        if line_pts:
            lines = pv.PolyData(np.vstack(line_pts).astype(np.float32), lines=np.concatenate(line_cells))
            lines.cell_data["design"] = np.asarray(l_id, dtype=np.int64)
            lines.cell_data["rgb"] = np.asarray(l_rgb, dtype=np.uint8)
            self._add(lines, scalars="rgb", rgb=True, line_width=2.5, lighting=False)

    def world_bounds(self):
        d = self.data
        pts = [d.xyz(n) for n in d.stations] if d is not None else []
        pts = [p for p in pts if p is not None] + ([np.asarray(q, float) for x in d.designs for q in x.points]
                                                   if d is not None else [])
        if not pts:
            return None
        a = np.array(pts)
        pad = max((p.profile_height for p in d.pits), default=5.0)
        return a.min(axis=0) - pad, a.max(axis=0) + pad

    def info(self):
        d = self.data
        if d is None or (not d.pits and not d.designs):
            return super().info() + [(tr("Pits"), tr("none yet - add them with the Pit Manager (Ctrl+U)"))]
        mains = sum(1 for p in d.pits if p.parent_id is None)
        known = sum(1 for s in d.stations.values() if s.status == "known")
        rows = super().info() + [(tr("Main pits"), f"{mains:,}"), (tr("Branches"), f"{len(d.pits) - mains:,}"),
                                 (tr("Stations"), tr("{n:,} ({known} known)", n=len(d.stations), known=known))]
        if d.designs:
            rows.append((tr("Designed pits"), f"{len(d.designs):,}"))
        return rows

    def pit_length(self, pid) -> float:
        d = self.data
        total = 0.0
        for path in d.paths.get(pid, []):
            pts = [d.xyz(n) for n in path]
            pts = [p for p in pts if p is not None]
            total += sum(float(np.hypot(*(b - a)[:2])) for a, b in zip(pts, pts[1:]))
        return total

    def identify(self, hit):
        d = self.data
        mesh = hit.mesh
        if "geo" in mesh.cell_data and getattr(self, "_geo_items", None):
            k = int(hit.cell_value("geo"))
            if 0 <= k < len(self._geo_items):
                from .pit_geology import geology_identify

                got = geology_identify(self, *self._geo_items[k])
                if got is not None:
                    title, rows = got
                    return IdentifyResult(title, rows + self._picked_rows(hit))
        if "station" in mesh.cell_data and getattr(self, "_station_names", None):
            name = self._station_names[int(hit.cell_value("station"))]
            s = d.stations[name]
            p = d.pit(s.pit_id)
            status = tr(STATUS_TEXT[s.status]) if s.status in STATUS_TEXT else (s.status or "-")
            rows = [(tr("Station"), name), (tr("Pit"), f"{p.name} ({p.code})" if p else "-"),
                    (tr("Status"), status), (tr("Coordinates"), None)]
            rows += _xyz_rows((s.x, s.y, s.z))
            if s.note:
                rows.append((tr("Note"), s.note))
            return IdentifyResult(tr("Station {name}", name=name), rows, marker=self.scene.to_local(np.array([s.x, s.y, s.z])))
        if "design" in mesh.cell_data:
            x = d.design(int(hit.cell_value("design")))
            if x is not None:
                rows = design_rows(self, x) + self._picked_rows(hit)
                return IdentifyResult(tr("Designed pit {name}", name=x.name), rows)
        if "pit" in mesh.cell_data:
            p = d.pit(int(hit.cell_value("pit")))
            if p is not None:
                parent = d.pit(p.parent_id) if p.parent_id is not None else None
                n = sum(1 for s in d.stations.values() if s.pit_id == p.id)
                rows = [(tr("Pit"), p.name), (tr("Code"), p.code),
                        (tr("Branch of"), parent.name if parent else tr("- (main pit)")),
                        (tr("Starts at"), p.start_station or "-"),
                        (tr("Length (plan)"), f"{self.pit_length(p.id):,.1f} m"),
                        (tr("Stations"), f"{n:,}"),
                        (tr("Profile"), tr(PROFILE_TEXT[p.station_level], w=p.profile_width, h=p.profile_height))]
                rows += self._picked_rows(hit)
                return IdentifyResult(tr("Pit {name}", name=p.name), rows)
        return super().identify(hit)


def design_rows(layer: SurveyLayer, x) -> list:
    """(label, text) rows describing a designed pit."""
    d = layer.data
    pit = d.pit(x.start_pit_id) if x.start_pit_id is not None else None
    rows = [(tr("Designed pit"), x.name)]
    rows.append((tr("Starts at"), (x.start_name or "-") + (f"  ({pit.name})" if pit is not None else "")))
    rows.append((tr("Goes to"), x.target_name or "-"))
    rows.append((tr("Length"), f"{x.length:,.1f} m"))
    rows.append((tr("Height change"), f"{x.climb:+,.1f} m"))
    if x.steepest is not None:
        from ..ai.route_planner import ratio_text

        rows.append((tr("Steepest part"), ratio_text(x.steepest)))
        if x.gradient:
            rows.append((tr("Steepest gradient allowed"), ratio_text(x.gradient)))
    rows.append((tr("Legs"), f"{max(len(x.points) - 1, 0):,}"))
    rows.append((tr("Profile"), tr(PROFILE_TEXT.get(x.station_level, PROFILE_TEXT["middle"]), w=x.profile_width,
                                   h=x.profile_height)))
    rows.append((tr("Volume"), tr("{v:,.0f} m³", v=layer.design_volume(x.id))))
    if x.created_at:
        rows.append((tr("Saved"), x.created_at.replace("T", " ")))
    if x.note:
        rows.append((tr("Note"), x.note))
    return rows
