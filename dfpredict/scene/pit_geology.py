"""Where the geology mapped in a pit is: chainage along the drive -> a point on a wall.

A pit's **chainage** is measured along its centre line (slope distance, as a tape is pulled along
the drive) from its start: 0 m is the entrance of a main pit, and the junction station of a
branch (the parent's station it starts at). Survey stations get their chainage from the
traverse, so a position measured from a station ("P1-03+4.5": 4.5 m past station P1-03) is turned
into a chainage. Past the last station the drive goes on along the last leg (the face is usually
ahead of the survey).

The walls are those of the drawn drive (scene/survey.py ``drive_mesh``: vertical walls, the
rectangular profile, mitred at bends), so the rocks, structures and samples sit on the drive in
the view. Left / right are as seen looking forward (toward the face). No Qt here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from ..i18n import tr
from ..io.pit_geology_store import DEFAULT_SAMPLE_HEIGHT, plane_normal

# corners of a profile (survey.drive_mesh): 0 bottom left, 1 bottom right, 2 top right, 3 top left
_EDGES = {"left": (0, 3), "right": (1, 2), "back": (3, 2)}
BAND_OFFSET = 0.04          # m: rocks drawn just outside the drive's walls (seen through them)
SAMPLE_OFFSET = 0.09        # m: samples just outside the rocks
SAMPLE_RIBBON = 0.25        # m: how high a sample's ribbon is drawn on the wall


@dataclass
class PitAxis:
    """A pit's centre line from its start (chainage 0) with its profile."""
    pit_id: int
    names: list                 # station names along the line
    points: np.ndarray          # (K, 3) world coordinates of the stations
    width: float
    height: float
    level: str                  # where the stations are in the profile: back / floor / middle

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64)
        d = np.linalg.norm(np.diff(self.points, axis=0), axis=1) if len(self.points) > 1 else np.zeros(0)
        self.chain = np.r_[0.0, np.cumsum(d)]
        from .survey import drive_mesh

        V, _Q = drive_mesh(self.points, self.width, self.height, self.level)
        self.corners = V.reshape(-1, 4, 3)          # (K, 4, 3) the profile's corners at each station

    @property
    def length(self) -> float:
        return float(self.chain[-1])

    def station_chainage(self) -> dict:
        """Station name -> chainage (m)."""
        return {n: float(c) for n, c in zip(self.names, self.chain)}

    def _end_dir(self, first: bool) -> np.ndarray:
        a, b = (self.points[0], self.points[1]) if first else (self.points[-2], self.points[-1])
        d = b - a
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def corners_at(self, m) -> np.ndarray:
        """The profile's 4 corners (n, 4, 3) at chainages ``m`` (array or number): along the drive's
        walls, and on along the first / last leg before 0 m / past the last station."""
        m = np.atleast_1d(np.asarray(m, dtype=np.float64))
        c, C = self.chain, self.corners
        i = np.clip(np.searchsorted(c, m, side="right") - 1, 0, len(c) - 2)
        span = c[i + 1] - c[i]
        t = np.where(span > 1e-9, (m - c[i]) / np.where(span > 1e-9, span, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)[:, None, None]
        out = C[i] * (1.0 - t) + C[i + 1] * t
        past = m > c[-1]
        if past.any():
            out[past] = C[-1][None] + (m[past] - c[-1])[:, None, None] * self._end_dir(False)[None, None, :]
        before = m < 0
        if before.any():
            out[before] = C[0][None] + m[before][:, None, None] * self._end_dir(True)[None, None, :]
        return out

    def forward_at(self, m: float) -> np.ndarray:
        """Unit direction of the drive (3D) at chainage ``m``."""
        c = self.chain
        if m >= c[-1]:
            return self._end_dir(False)
        if m <= 0:
            return self._end_dir(True)
        i = int(np.clip(np.searchsorted(c, m, side="right") - 1, 0, len(c) - 2))
        d = self.points[i + 1] - self.points[i]
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else self._end_dir(False)

    def bearing_at(self, m: float) -> float:
        """Plan bearing (degrees clockwise from north) of the drive at chainage ``m``; a vertical leg
        takes the bearing of the drawn walls there."""
        q = self.corners_at(m)[0]
        right = q[1, :2] - q[0, :2]                  # bottom left -> bottom right
        if np.hypot(*right) < 1e-9:
            f = self.forward_at(m)
            return math.degrees(math.atan2(f[0], f[1])) % 360.0
        return (math.degrees(math.atan2(right[0], right[1])) - 90.0) % 360.0

    def wall_point(self, m: float, wall: str, height: float | None = None, across: float = 0.5) -> np.ndarray:
        """The point at chainage ``m`` on ``wall`` (left / right / back / face), ``height`` m above the
        floor (None: half way up a wall; the back is always at the top). ``across``: where on the back
        or the face (0 = left edge, 1 = right edge)."""
        q = self.corners_at(m)[0]
        H = self.height
        h = H / 2.0 if height is None else float(height)
        if wall in ("left", "right"):
            b, t = _EDGES[wall]
            return q[b] + (q[t] - q[b]) * (h / H if H > 0 else 0.0)
        if wall == "back":
            return q[3] + (q[2] - q[3]) * across
        bottom = q[0] + (q[1] - q[0]) * across           # the face: across the floor, then up
        return bottom + np.array([0.0, 0.0, h])

    def outward(self, m: float, wall: str) -> np.ndarray:
        """Unit vector out of the drive through ``wall`` at chainage ``m``."""
        q = self.corners_at(m)[0]
        if wall == "back":
            return np.array([0.0, 0.0, 1.0])
        if wall == "face":
            return self.forward_at(m)
        d = (q[0] - q[1]) if wall == "left" else (q[1] - q[0])
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def axis_names(paths: list, start: str | None) -> list:
    """The stations of a pit's centre line from its start: the line beginning at the start
    station (else the first), continued by the lines that begin where it ends."""
    if not paths:
        return []
    lines = [list(p) for p in paths if len(p) >= 2]
    if not lines:
        return []
    first = next((p for p in lines if start and p[0] == start), lines[0])
    names = list(first)
    rest = [p for p in lines if p is not first]
    grown = True
    while grown and rest:
        grown = False
        for p in rest:
            if p[0] == names[-1] and not set(p[1:]) & set(names):
                names += p[1:]
                rest.remove(p)
                grown = True
                break
    return names


def pit_axis(data, pid) -> PitAxis | None:
    """The chainage axis of a pit of a survey.SurveyData (None: fewer than two stations with
    coordinates from its start). It stops at the first station without coordinates."""
    p = data.pit(pid)
    if p is None:
        return None
    names = axis_names(data.paths.get(pid, []), p.start_station)
    pts, kept = [], []
    for n in names:
        xyz = data.xyz(n)
        if xyz is None:
            break
        pts.append(xyz)
        kept.append(n)
    if len(pts) < 2:
        return None
    return PitAxis(pid, kept, np.array(pts), float(p.profile_width), float(p.profile_height), p.station_level)


_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


def parse_chainage(text, stations: dict | None = None) -> float | None:
    """A chainage from what was typed: metres ("23.5"), or a station and metres past it
    ("P1-03+4.5", "P1-03 + 4.5", "P1-03-2" for 2 m before it, "P1-03" alone). ``stations``:
    name -> chainage. None for empty; ValueError for anything else."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return None if isinstance(text, float) and math.isnan(text) else float(text)
    t = str(text).strip().replace(",", ".").replace("＋", "+").replace("－", "-")
    if not t:
        return None
    if _NUMBER.match(t):
        return float(t)
    stations = stations or {}
    low = t.lower()
    for name in sorted(stations, key=len, reverse=True):     # the longest name that starts the text
        if low.startswith(name.lower()):
            rest = t[len(name):].replace(" ", "")
            if not rest:
                return float(stations[name])
            if rest[0] in "+-" and _NUMBER.match(rest[1:] or "x") and not rest[1:].startswith(("+", "-")):
                off = float(rest[1:])
                return float(stations[name]) + (off if rest[0] == "+" else -off)
    raise ValueError(tr("not a chainage: {t!r} (type metres from the start, e.g. 23.5, or a station and "
                        "metres past it, e.g. P1-03+4.5)", t=t))


def station_offset_text(m: float, stations: dict) -> str:
    """The chainage as "station + metres" from the nearest station at or before it
    ("P1-03+4.50"); "" without stations."""
    before = [(c, n) for n, c in stations.items() if c <= m + 1e-9]
    if not before:
        return ""
    c, n = max(before)
    return f"{n}+{m - c:.2f}"


# --- what the view draws ----------------------------------------------------------------------------
def _span(axis: PitAxis, a: float, b: float) -> np.ndarray:
    """Chainages from a to b with every station between (so a band follows the bends)."""
    inner = axis.chain[(axis.chain > a + 1e-9) & (axis.chain < b - 1e-9)]
    return np.r_[a, inner, b]


def _strip(axis: PitAxis, ms: np.ndarray, wall: str, h0: float | None = None, h1: float | None = None,
           offset: float = BAND_OFFSET) -> tuple:
    """Vertices (2n, 3) and quads (n-1, 4) of a band on ``wall`` over the chainages ``ms`` (from
    ``h0`` to ``h1`` m above the floor on a wall; the whole back), moved ``offset`` out of the drive."""
    Q = axis.corners_at(ms)                              # (n, 4, 3)
    if wall == "back":
        e0, e1 = Q[:, 3].copy(), Q[:, 2].copy()
        out = np.tile([0.0, 0.0, 1.0], (len(ms), 1))
    else:
        b, t = _EDGES[wall]
        H = axis.height if axis.height > 0 else 1.0
        lo = 0.0 if h0 is None else h0 / H
        hi = 1.0 if h1 is None else h1 / H
        e0 = Q[:, b] + (Q[:, t] - Q[:, b]) * lo
        e1 = Q[:, b] + (Q[:, t] - Q[:, b]) * hi
        d = (Q[:, 0] - Q[:, 1]) if wall == "left" else (Q[:, 1] - Q[:, 0])
        n = np.linalg.norm(d, axis=1, keepdims=True)
        out = np.where(n > 1e-9, d / np.where(n > 1e-9, n, 1.0), 0.0)
    e0 = e0 + out * offset
    e1 = e1 + out * offset
    k = len(ms)
    V = np.vstack([e0, e1])
    j = np.arange(k - 1)
    quads = np.column_stack([j, j + 1, k + j + 1, k + j])
    return V, quads


def disc(center, normal, radius: float, n: int = 24) -> tuple:
    """A flat disc as a triangle fan: vertices (n + 1, 3), triangles (n, 3)."""
    nv = np.asarray(normal, dtype=np.float64)
    nv = nv / (np.linalg.norm(nv) or 1.0)
    ref = np.array([0.0, 0.0, 1.0]) if abs(nv[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(nv, ref)
    u /= np.linalg.norm(u)
    v = np.cross(nv, u)
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ring = np.asarray(center)[None, :] + radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
    V = np.vstack([np.asarray(center, dtype=np.float64)[None, :], ring])
    k = np.arange(n)
    T = np.column_stack([np.zeros(n, dtype=np.int64), 1 + k, 1 + (k + 1) % n])
    return V, T


def interval_bands(axis: PitAxis, r) -> list:
    """[(wall, vertices, quads)] of a rock interval (one band per wall it is on)."""
    if r.frm is None or r.to is None or r.to <= r.frm:
        return []
    ms = _span(axis, r.frm, r.to)
    return [(w, *_strip(axis, ms, w)) for w in r.walls()]


def structure_disc(axis: PitAxis, s, radius: float) -> tuple | None:
    """(vertices, triangles) of a structure's disc at its place, tilted as measured (a level disc
    on its wall when no dip is measured)."""
    if s.at is None:
        return None
    p = axis.wall_point(s.at, s.wall, s.height) + axis.outward(s.at, s.wall) * (BAND_OFFSET * 2)
    if s.dip is not None and s.dip_direction is not None:
        n = plane_normal(s.dip_direction, s.dip)
    else:
        n = axis.outward(s.at, s.wall)
    return disc(p, n, radius)


def sample_ribbon(axis: PitAxis, s) -> tuple | None:
    """(vertices, quads) of a sample: a narrow band on its wall (or along the middle of the back),
    centred at its height."""
    if s.frm is None or s.to is None or s.to <= s.frm:
        return None
    ms = _span(axis, s.frm, s.to)
    if s.wall in ("left", "right"):
        h = DEFAULT_SAMPLE_HEIGHT if s.height is None else s.height
        h = min(max(h, SAMPLE_RIBBON / 2), max(axis.height - SAMPLE_RIBBON / 2, SAMPLE_RIBBON / 2))
        return _strip(axis, ms, s.wall, h - SAMPLE_RIBBON / 2, h + SAMPLE_RIBBON / 2, SAMPLE_OFFSET)
    if s.wall == "back":
        Q = axis.corners_at(ms)
        mid = (Q[:, 3] + Q[:, 2]) / 2.0
        across = Q[:, 2] - Q[:, 3]
        n = np.linalg.norm(across, axis=1, keepdims=True)
        across = across / np.where(n > 1e-9, n, 1.0) * (SAMPLE_RIBBON / 2)
        up = np.array([0.0, 0.0, SAMPLE_OFFSET])
        k = len(ms)
        V = np.vstack([mid - across + up, mid + across + up])
        j = np.arange(k - 1)
        return V, np.column_stack([j, j + 1, k + j + 1, k + j])
    # the face: a short vertical band at the chainage, across the middle of the face
    m = s.to
    f = axis.forward_at(m)
    h = DEFAULT_SAMPLE_HEIGHT if s.height is None else s.height
    c = axis.wall_point(m, "face", h) + f * SAMPLE_OFFSET
    q = axis.corners_at(m)[0]
    side = q[1] - q[0]
    side = side / (np.linalg.norm(side) or 1.0) * (SAMPLE_RIBBON / 2)
    up = np.array([0.0, 0.0, max(s.to - s.frm, SAMPLE_RIBBON) / 2])
    V = np.array([c - side - up, c + side - up, c + side + up, c - side + up])
    return V, np.array([[0, 1, 2, 3]])


# --- traces on the unfolded wall sketch ---------------------------------------------------------------
def trace_direction(dip_direction: float, dip: float, bearing: float, wall: str) -> tuple | None:
    """Direction (along the drive, across) of a plane's trace on a wall of a drive heading
    ``bearing``: on a wall (along, up), on the back (along, to the left); None when the plane is
    parallel to the wall (no trace)."""
    n = np.array(plane_normal(dip_direction, dip))
    b = math.radians(bearing)
    f = np.array([math.sin(b), math.cos(b), 0.0])         # forward
    r = np.array([math.cos(b), -math.sin(b), 0.0])        # right
    up = np.array([0.0, 0.0, 1.0])
    d = np.cross(up if wall == "back" else r, n)
    along, other = float(d @ f), float(d @ (-r if wall == "back" else up))
    size = math.hypot(along, other)
    if size < 1e-6:
        return None
    return along / size, other / size


# --- Identify ---------------------------------------------------------------------------------------
def _m(v) -> str:
    return "-" if v is None else f"{v:,.2f} m"


def geology_identify(layer, kind: str, pid: int, i: int):
    """(title, rows) Identify shows for a mapped rock, structure or sample of a pit."""
    from ..io.pit_geology_store import (INTERVAL_WALLS, SAMPLE_KINDS, STRUCTURE_KINDS, WALLS,
                                        orientation_text, strike_of)

    d = layer.data
    p = d.pit(pid)
    geo = d.geology.get(pid)
    if p is None or geo is None:
        return None
    ax = layer.axis(pid)
    st = ax.station_chainage() if ax is not None else {}
    pit = f"{p.name} ({p.code})"

    def where(m):
        t = station_offset_text(m, st) if m is not None else ""
        return _m(m) + (f"  ({t})" if t else "")

    def who(on, by):
        return [(tr("Mapped on"), on)] * bool(on) + [(tr("By"), by)] * bool(by)

    if kind == "interval" and i < len(geo.intervals):
        r = geo.intervals[i]
        rows = [(tr("Pit"), pit), (tr("Rock"), r.rock), (tr("From"), where(r.frm)), (tr("To"), where(r.to)),
                (tr("Length"), _m(r.to - r.frm) if None not in (r.frm, r.to) else "-"),
                (tr("Where"), tr(INTERVAL_WALLS.get(r.wall, r.wall)))]
        for label, v in ((tr("Description"), r.description), (tr("Alteration"), r.alteration),
                         (tr("Mineralization"), r.mineralization)):
            if v:
                rows.append((label, v))
        return tr("Rock {rock} in {pit}", rock=r.rock, pit=p.name), rows + who(r.mapped_on, r.geologist)
    if kind == "structure" and i < len(geo.structures):
        s = geo.structures[i]
        name = tr(STRUCTURE_KINDS.get(s.kind, s.kind))
        rows = [(tr("Pit"), pit), (tr("Type"), name), (tr("Chainage"), where(s.at)),
                (tr("Where"), tr(WALLS.get(s.wall, s.wall))),
                (tr("Height above floor"), _m(s.height) if s.height is not None else tr("- (half way up)"))]
        if s.dip is not None and s.dip_direction is not None:
            rows += [(tr("Dip direction / dip"), orientation_text(s.dip_direction, s.dip)),
                     (tr("Strike"), f"{strike_of(s.dip_direction):03.0f}°")]
        if s.thickness is not None:
            rows.append((tr("Thickness"), _m(s.thickness)))
        for label, v in ((tr("Filling"), s.fill), (tr("Description"), s.description)):
            if v:
                rows.append((label, v))
        return tr("{kind} in {pit}", kind=name, pit=p.name), rows + who(s.mapped_on, s.geologist)
    if kind == "sample" and i < len(geo.samples):
        s = geo.samples[i]
        rows = [(tr("Pit"), pit), (tr("Sample"), s.name), (tr("Type"), tr(SAMPLE_KINDS.get(s.kind, s.kind))),
                (tr("From"), where(s.frm)), (tr("To"), where(s.to)), (tr("Length"), _m(s.length)),
                (tr("Where"), tr(WALLS.get(s.wall, s.wall)))]
        if s.width_cm is not None or s.depth_cm is not None:
            rows.append((tr("Channel (width x depth)"), tr("{w} x {d} cm", w=f"{s.width_cm:g}" if s.width_cm is not None
                                                           else "?", d=f"{s.depth_cm:g}" if s.depth_cm is not None
                                                           else "?")))
        if s.weight_kg is not None:
            rows.append((tr("Weight"), f"{s.weight_kg:g} kg"))
        vals = [(k, v) for k, v in s.values.items() if v is not None]
        if vals:
            rows.append((tr("Assays"), None))
            rows += [(k, f"{v:g}") for k, v in vals]
        if s.note:
            rows.append((tr("Note"), s.note))
        rows += [(tr("Taken on"), s.taken_on)] * bool(s.taken_on) + [(tr("By"), s.sampler)] * bool(s.sampler)
        return tr("Sample {name} in {pit}", name=s.name, pit=p.name), rows
    return None
