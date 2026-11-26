"""A new drive (decline, ramp, access) from a station to a target, designed automatically.
Offline; no Qt.

The route must stay within a gradient limit (e.g. 1 in 7), keep a clearance from old workings
(the recorded mined shapes), from the existing drives and - when asked - from the ore body, and
should be as short as possible (development metres cost money).

Method
    1. A 3D search grid covers the start, the target and room around them. The cells are sized so that
       the start and the target are exactly grid nodes; levels are spaced so that one plan step with
       one level change is never steeper than the limit - every path on the grid keeps the gradient.
    2. Grid nodes inside the obstacles, grown by the clearances (and a little safety margin), are
       blocked (mined shapes and the ore solid by the point-in-solid test, drives by the distance to
       their centre lines). Nodes near the start and the target are kept free, so the route can
       leave and reach the drives there.
    3. A* search finds the shortest path. A search state is a node and the heading it was reached
       with; a step may turn at most 45°, so bends are gradual (no hairpins, no knots), and each 45°
       turned costs a little, so among routes of the same length the straightest wins. Length stays
       the main aim: the heuristic max(plan distance, height difference / gradient) never
       overestimates, because a drive must be at least that long to change that height.
    4. The grid path is straightened into a few legs (string pulling): a leg is kept as long as it
       stays within the gradient, away from the obstacles, and its bends stay gradual (<= 50°). A leg
       that the grid (with its safety margin) rejects is tested again exactly - the distance from its
       centre line to the drives, old workings and ore - so a straight drive that truly keeps the
       clearances is never bent because of the grid.
    5. The design is measured exactly: length, gradient of each leg, bends, clearances (distance
       from the centre line to the obstacles, less half the profile).

When the target is high above or far below the start, the direct line is too steep: the route
then winds (switchbacks) to gain length - this is shown and explained.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from ..i18n import N_, tr, trn
from . import BAD, INFO, OK, WARN

MAX_NODES = 150_000           # search grid size (nodes); the cell grows until the grid fits


@dataclass
class Obstacles:
    voids: list = field(default_factory=list)            # mined shapes (MinedShape: .contains, .bounds)
    drives: list = field(default_factory=list)           # [(points (K, 3), half size m, name)]
    ore: tuple | None = None                             # (vertices, triangles) of the ore solid


@dataclass
class RouteRequest:
    start: tuple
    target: tuple
    gradient: float = 1.0 / 7.0          # steepest allowed slope (rise / run)
    void_clearance: float = 10.0         # m from old workings (wall to wall)
    drive_clearance: float = 10.0        # m from other drives
    ore_standoff: float | None = None    # m outside the ore solid; None: the route may cross the ore
    width: float = 5.0                   # profile (for the clearances and the waste tonnes)
    height: float = 5.0
    cell: float | None = None            # search cell (m); None: chosen from the size of the job
    start_name: str = ""
    target_name: str = ""
    min_radius: float = 15.0             # smallest bend radius (m): trucks, and room between the legs of a loop
    start_bearing: float | None = None   # direction the drive leaves the start (degrees from north, clockwise);
                                         # None: automatic (the search chooses)
    # internal (a design that carries on from a lead-in leg): the start's free zone, heading and lead-in point
    start_free: float | None = None
    start_heading: int | None = None
    lead_from: tuple | None = None


@dataclass
class Leg:
    a: np.ndarray
    b: np.ndarray
    plan: float
    length: float
    gradient: float                       # rise / run (signed: + up)


@dataclass
class Route:
    ok: bool
    message: str = ""
    points: np.ndarray | None = None      # (K, 3) the stations of the design
    legs: list = field(default_factory=list)
    length: float = 0.0                   # 3D
    plan_length: float = 0.0
    climb: float = 0.0                    # target z - start z
    steepest: float = 0.0                 # largest |gradient| of a leg
    direct: float = 0.0                   # straight-line plan distance start - target
    needed: float = 0.0                   # shortest plan length the gradient allows
    sharpest_bend: float = 0.0            # degrees (largest change of direction in plan)
    bend_station: int = 0
    void_clear: float | None = None       # smallest clearance (m) to old workings (None: none near)
    drive_clear: float | None = None
    ore_clear: float | None = None        # + outside / - inside the ore
    cell: float = 0.0
    explored: int = 0
    findings: list = field(default_factory=list)
    request: RouteRequest | None = None
    free_r: float = 0.0                   # around the start / target clearances are not counted
    self_clear: float | None = None       # smallest clearance between non-neighbouring parts of the route
    radius: float = 0.0                   # bend radius the design was made with
    radius_reduced: bool = False          # the asked radius left no way through: tighter bends were used
    start_point: object = None            # StartPoint when the start was chosen along the drives
    meets: list = field(default_factory=list)   # [(0 start / 1 target, drive name, centre distance m)]: an end
                                                # inside another drive's clearance - the new drive meets it there


# --- geometry helpers -------------------------------------------------------------------------------
def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p (N, 3) to the segment a-b."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 <= 1e-18:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _poly_dist(p: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        out = np.minimum(out, _seg_dist(p, a, b))
    if len(pts) == 1:
        out = np.linalg.norm(p - pts[0], axis=1)
    return out


def _dilate(mask: np.ndarray, r: tuple) -> np.ndarray:
    """Grow a 3D boolean mask by r cells along each axis (box)."""
    out = mask.copy()
    for ax, n in enumerate(r):
        if n <= 0:
            continue
        cur = out.copy()
        for k in range(1, n + 1):
            sl_a = [slice(None)] * 3
            sl_b = [slice(None)] * 3
            sl_a[ax], sl_b[ax] = slice(k, None), slice(None, -k)
            cur[tuple(sl_a)] |= out[tuple(sl_b)]
            cur[tuple(sl_b)] |= out[tuple(sl_a)]
        out = cur
    return out


class _Grid:
    def __init__(self, lo: np.ndarray, hi: np.ndarray, cx: float, cy: float, dz: float):
        self.lo, self.cx, self.cy, self.dz = lo, cx, cy, dz
        self.cell = max(cx, cy)
        self.nx = int(math.ceil((hi[0] - lo[0]) / cx)) + 1
        self.ny = int(math.ceil((hi[1] - lo[1]) / cy)) + 1
        self.nz = int(math.ceil((hi[2] - lo[2]) / dz)) + 1

    @property
    def size(self) -> int:
        return self.nx * self.ny * self.nz

    def coords(self) -> np.ndarray:
        i, j, k = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij")
        return np.stack([self.lo[0] + i * self.cx, self.lo[1] + j * self.cy, self.lo[2] + k * self.dz],
                        -1).reshape(-1, 3)

    def node(self, p) -> tuple:
        i = int(np.clip(round((p[0] - self.lo[0]) / self.cx), 0, self.nx - 1))
        j = int(np.clip(round((p[1] - self.lo[1]) / self.cy), 0, self.ny - 1))
        k = int(np.clip(round((p[2] - self.lo[2]) / self.dz), 0, self.nz - 1))
        return i, j, k

    def xyz(self, i, j, k) -> np.ndarray:
        return np.array([self.lo[0] + i * self.cx, self.lo[1] + j * self.cy, self.lo[2] + k * self.dz])


def _layout(req: RouteRequest) -> _Grid:
    s, t = np.asarray(req.start, float), np.asarray(req.target, float)
    g = req.gradient
    direct = math.hypot(*(t[:2] - s[:2]))
    needed = abs(t[2] - s[2]) / g
    extra = max(needed - direct, 0.0)
    margin = max(60.0, 0.3 * direct, 0.35 * extra, 3 * max(req.void_clearance, req.drive_clearance))
    lo = np.minimum(s, t) - [margin, margin, 0.0]
    hi = np.maximum(s, t) + [margin, margin, 0.0]
    lo[2] -= 15.0
    hi[2] += 15.0
    cell = req.cell or max(2.0, min(req.width, 5.0))
    def fit(d, c):                          # a cell size that puts the target exactly on a node
        n = round(abs(d) / c)
        return abs(d) / n if n >= 1 else c

    while True:
        cx, cy = fit(t[0] - s[0], cell), fit(t[1] - s[1], cell)
        dz = min(cx, cy) * g
        rise = abs(t[2] - s[2])
        # levels spaced so the target is exactly on one (and never steeper). A target less than half a
        # level above / below the start stays on the start's level (the last leg makes up the few cm):
        # spacing the levels by a tiny height difference would need thousands of them and blow up the grid
        if rise >= 0.5 * dz:
            dz = rise / math.ceil(rise / dz)
        # the start is exactly a grid node, and so (by the cell sizes) is the target
        a = s - np.ceil((s - lo) / [cx, cy, dz]) * [cx, cy, dz]
        grid = _Grid(a, hi, cx, cy, dz)
        if grid.size <= MAX_NODES:
            return grid
        cell *= 1.2


def free_radius(req: "RouteRequest", cell: float, hs: float = 0.0) -> float:
    """Around the start and the target the route may pass close to the drive it leaves (or joins).

    ``hs`` is the half size of that drive: the zone must reach past the whole band the drive blocks
    (its half size + the new drive's half size + the clearance + the grid margin), or it leaves a ring
    around the junction that only some directions happen to cross - and the route bends for nothing."""
    return max(req.drive_clearance + hs, req.void_clearance) + 0.5 * math.hypot(req.width, req.height) + cell


def end_half_size(obs: "Obstacles", p) -> float:
    """Half size of the existing drive(s) that the point p is on (0 when it is on none)."""
    p = np.asarray(p, float)[None]
    return max((hs for pts, hs, _name in obs.drives
                if len(pts) >= 2 and _poly_dist(p, np.asarray(pts, float))[0] <= 1.0), default=0.0)


def end_contacts(req: "RouteRequest", obs: "Obstacles", p) -> list:
    """The drives whose clearance band holds the point p but that p is not on: [(name, centre distance, half size)].
    A new drive ending there meets that drive (their walls are closer than the clearance asked)."""
    half = 0.5 * math.hypot(req.width, req.height)
    q = np.asarray(p, float)[None]
    out = []
    for pts, hs, name in obs.drives:
        if len(pts) < 2:
            continue
        d = float(_poly_dist(q, np.asarray(pts, float))[0])
        if 1.0 < d < hs + half + req.drive_clearance:
            out.append((name, d, hs))
    return out


def end_zones(req: "RouteRequest", obs: "Obstacles", cell: float) -> list:
    """(point, radius) of the free zones around the start and the target. An end inside the clearance
    band of a drive it is not on gets a zone that reaches out of that band: the route meets that drive
    there (reported) instead of having to squeeze into the band from its far end."""
    out = []
    for k, p in enumerate((req.start, req.target)):
        r = free_radius(req, cell, end_half_size(obs, p))
        for _name, d, hs in end_contacts(req, obs, p):
            r = max(r, free_radius(req, cell, hs) + d)
        if k == 0 and req.start_free is not None:
            r = req.start_free                  # the end of a lead-in leg: not on a drive
        out.append((np.asarray(p, float), r))
    return out


def _blocked(grid: _Grid, req: RouteRequest, obs: Obstacles) -> np.ndarray:
    P = grid.coords()
    shape = (grid.nx, grid.ny, grid.nz)
    half = 0.5 * math.hypot(req.width, req.height)
    margin = 0.75 * grid.cell                # the route runs between grid nodes: keep a little extra
    blocked = np.zeros(len(P), dtype=bool)
    # old workings: inside a mined shape, then grown by the clearance (+ half the profile)
    void = np.zeros(len(P), dtype=bool)
    reach = req.void_clearance + half + margin
    for sh in obs.voids:
        lo, hi = sh.bounds()
        near = np.all((P >= np.asarray(lo) - reach) & (P <= np.asarray(hi) + reach), axis=1)
        if near.any():
            idx = np.flatnonzero(near)
            void[idx[sh.contains(P[idx])]] = True
    if void.any():
        r = (int(math.ceil(reach / grid.cx)), int(math.ceil(reach / grid.cy)), int(math.ceil(reach / grid.dz)))
        blocked |= _dilate(void.reshape(shape), r).reshape(-1)
    # other drives: distance to their centre lines
    for pts, hs, _name in obs.drives:
        pts = np.asarray(pts, float)
        r = hs + half + req.drive_clearance + margin
        lo, hi = pts.min(axis=0) - r, pts.max(axis=0) + r
        near = np.flatnonzero(np.all((P >= lo) & (P <= hi), axis=1))
        if len(near):
            blocked[near[_poly_dist(P[near], pts) < r]] = True
    # the ore body (when the route must stay out of it)
    if obs.ore is not None and req.ore_standoff is not None:
        from ..scene.solid import points_inside

        ore = points_inside(P, *obs.ore).reshape(shape)
        reach = req.ore_standoff + half + margin
        r = (int(math.ceil(reach / grid.cx)), int(math.ceil(reach / grid.cy)), int(math.ceil(reach / grid.dz)))
        blocked |= _dilate(ore, r).reshape(-1)
    # leave room around the start and the target (the route leaves an existing drive there)
    for p, free_r in end_zones(req, obs, grid.cell):
        blocked[np.linalg.norm(P - p, axis=1) <= free_r] = False
    return blocked.reshape(shape)


EXACT_STEP = 0.5              # m between the points checked along a straight leg (exact clearance test)


class _Exact:
    """Exact clearance test of a straight leg: the distance from its centre line to the drives, the old
    workings and the ore - no grid, no grid margin. The grid (with its margin for the coarse cells) is
    only for the search; a straight leg that truly keeps the clearances is accepted.

    The points are checked every EXACT_STEP m; the distance changes by at most half a step between two
    of them, so half a step is added to each clearance."""

    def __init__(self, req: RouteRequest, obs: Obstacles):
        half = 0.5 * math.hypot(req.width, req.height)
        pad = 0.5 * EXACT_STEP
        self.drives = [(np.asarray(pts, float), hs + half + req.drive_clearance + pad)
                       for pts, hs, _name in obs.drives if len(pts)]
        self.void_reach = req.void_clearance + half + pad
        self.voids = [[sh, None] for sh in obs.voids]           # mesh distance built when first needed
        self.ore = obs.ore if (obs.ore is not None and req.ore_standoff is not None) else None
        self.ore_reach = (req.ore_standoff or 0.0) + half + pad
        self._ore_f = None

    @staticmethod
    def _distance_function(vertices, triangles):
        import pyvista as pv
        import vtk

        faces = np.hstack([np.full((len(triangles), 1), 3), np.asarray(triangles, dtype=np.int64)]).ravel()
        f = vtk.vtkImplicitPolyDataDistance()
        f.SetInput(pv.PolyData(np.asarray(vertices, dtype=np.float64), faces=faces))
        return f

    @staticmethod
    def _far(f, P: np.ndarray, reach: float) -> bool:
        return all(abs(f.EvaluateFunction(*map(float, p))) >= reach for p in P)

    def clear(self, P: np.ndarray) -> bool:
        for pts, reach in self.drives:
            if (_poly_dist(P, pts) < reach).any():
                return False
        for item in self.voids:
            sh = item[0]
            lo, hi = sh.bounds()
            near = P[np.all((P >= np.asarray(lo) - self.void_reach) & (P <= np.asarray(hi) + self.void_reach), axis=1)]
            if not len(near):
                continue
            if np.any(sh.contains(near)):
                return False
            if item[1] is None:
                from ..scene.production import prism_mesh

                mesh = prism_mesh(sh)
                if mesh is None or mesh.n_cells == 0:
                    item[1] = False
                    continue
                mesh = mesh.triangulate()
                item[1] = self._distance_function(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
            if item[1] is not False and not self._far(item[1], near, self.void_reach):
                return False
        if self.ore is not None:
            from ..scene.solid import points_inside

            V = np.asarray(self.ore[0], float)
            near = P[np.all((P >= V.min(axis=0) - self.ore_reach) & (P <= V.max(axis=0) + self.ore_reach), axis=1)]
            if len(near):
                if np.any(points_inside(near, *self.ore)):
                    return False
                if self._ore_f is None:
                    self._ore_f = self._distance_function(*self.ore)
                if not self._far(self._ore_f, near, self.ore_reach):
                    return False
        return True


def _leg_ok(a, b, grid: _Grid, blocked: np.ndarray, g: float, free, exact: _Exact | None = None) -> bool:
    plan = math.hypot(b[0] - a[0], b[1] - a[1])
    if plan <= 1e-9 or abs(b[2] - a[2]) > g * plan * (1 + 1e-6):
        return plan <= 1e-9 and abs(b[2] - a[2]) < 1e-6
    n = max(2, int(math.ceil(math.dist(a, b) / (min(grid.cx, grid.cy) / 3))))
    grid_ok = True
    for s in np.linspace(0.0, 1.0, n + 1):
        p = a + (b - a) * s
        if any(np.linalg.norm(p - q) <= r for q, r in free):
            continue
        i, j, k = grid.node(p)
        if blocked[i, j, k]:
            grid_ok = False
            break
    if grid_ok or exact is None:           # the grid (with its margin) is the safe side: clear there is clear
        return grid_ok
    n = max(2, int(math.ceil(math.dist(a, b) / EXACT_STEP)))
    P = a + (b - a) * np.linspace(0.0, 1.0, n + 1)[:, None]
    keep = np.ones(len(P), dtype=bool)
    for q, r in free:
        keep &= np.linalg.norm(P - q, axis=1) > r
    return exact.clear(P[keep]) if keep.any() else True


DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]   # headings, 45° apart
MAX_TURN = 1                  # a step may turn at most 45°: bends stay drivable (no hairpins, no knots)
TURN_COST = 0.1               # cost of each 45° of turning, in cells: among routes of the same length the
                              # straightest wins; length stays the main aim


def _astar(grid: _Grid, blocked: np.ndarray, s: tuple, t: tuple, g: float, run_steps: int = 1,
           start_heading: int | None = None):
    """Shortest drivable path on the grid (list of (i, j, k)) and the number of states explored.

    A state is a grid node, the heading it was reached with and how many steps it has gone straight
    (up to ``run_steps``). The route may go straight on, or turn 45° left or right once it has gone
    ``run_steps`` steps straight - so bends have at least the radius those steps give (one level down,
    level or one up each step). Cost = 3D length + a small cost per 45° turned."""
    nx, ny, nz = grid.nx + 2, grid.ny + 2, grid.nz + 2           # padded: no bounds checks
    pad = np.ones((nx, ny, nz), dtype=bool)
    pad[1:-1, 1:-1, 1:-1] = blocked
    wall = pad.reshape(-1).tolist()
    sy, sz = ny * nz, nz
    c, cx, cy, dz = grid.cell, grid.cx, grid.cy, grid.dz
    steps = []                                                    # per heading: (offset, cost) for dk = -1, 0, 1
    for di, dj in DIRS:
        h = math.hypot(di * cx, dj * cy)
        steps.append([(di * sy + dj * sz + dk, math.hypot(h, dk * dz)) for dk in (-1, 0, 1)])
    turn_cost = TURN_COST * c
    allowed = {h: [(d, min((d - h) % 8, (h - d) % 8)) for d in range(8)
                   if min((d - h) % 8, (h - d) % 8) <= MAX_TURN] for h in range(8)}
    allowed[8] = [(d, 0) for d in range(8)]                       # the start: any direction
    R = max(1, int(run_steps))                                    # straight steps before the next turn
    NS = 9 * (R + 1)                                              # states per node: heading x straight run

    def idx(n):
        return (n[0] + 1) * sy + (n[1] + 1) * sz + (n[2] + 1)

    start, goal = idx(s), idx(t)
    wall[start] = wall[goal] = False
    ii, jj, kk = np.meshgrid(np.arange(nx) - 1, np.arange(ny) - 1, np.arange(nz) - 1, indexing="ij")
    plan = np.hypot((ii - t[0]) * cx, (jj - t[1]) * cy)
    climb = np.abs(kk - t[2]) * dz / g
    hgt = np.maximum(plan, climb).reshape(-1).tolist()           # never overestimates: exact shortest
    s0 = start * NS + (8 if start_heading is None else start_heading) * (R + 1) + R   # 8: any direction
    best = {s0: 0.0}
    came = {}
    heap = [(hgt[start], 0.0, s0)]
    closed = set()
    explored = 0
    end = None
    while heap:
        _f, gs, st = heapq.heappop(heap)
        if st in closed:
            continue
        node, rest = divmod(st, NS)
        h, run = divmod(rest, R + 1)
        if node == goal:
            end = st
            break
        closed.add(st)
        explored += 1
        for d, turns in allowed[h]:
            if turns and run < R:                                 # not yet straight long enough to turn
                continue
            extra = turns * turn_cost
            nrun = 1 if turns else min(run + 1, R)
            for off, cost in steps[d]:
                nb = node + off
                if wall[nb]:
                    continue
                ns = nb * NS + d * (R + 1) + nrun
                if ns in closed:
                    continue
                ng = gs + cost + extra
                if ng < best.get(ns, math.inf):
                    best[ns] = ng
                    came[ns] = st
                    heapq.heappush(heap, (ng + hgt[nb], ng, ns))
    if end is None:
        return None, explored
    path, st = [], end
    while True:
        node = st // NS
        i, rest = divmod(node, sy)
        j, k = divmod(rest, sz)
        path.append((i - 1, j - 1, k - 1))
        if st == s0:
            break
        st = came[st]
    return path[::-1], explored


def _bend(a, b, c) -> float:
    """Change of direction in plan at b (degrees) going a -> b -> c."""
    u, v = b[:2] - a[:2], c[:2] - b[:2]
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(u @ v / (nu * nv), -1.0, 1.0))))


MAX_BEND = 50.0               # degrees: the straightened route keeps its bends gradual


def _straighten(pts: np.ndarray, grid: _Grid, blocked: np.ndarray, g: float, free,
                exact: _Exact | None = None, prefix=None) -> np.ndarray:
    """String pulling: from each kept point, the farthest point that a valid straight leg reaches
    (within the gradient, clear of the obstacles, and not bending more than MAX_BEND). ``prefix``: the
    point before the first one (a lead-in leg), for the first bend."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        prev = out[-2] if len(out) >= 2 else prefix
        while j > i + 1:
            if _leg_ok(pts[i], pts[j], grid, blocked, g, free, exact) and \
                    (prev is None or _bend(prev, pts[i], pts[j]) <= MAX_BEND) and \
                    (j == len(pts) - 1 or _bend(pts[i], pts[j], pts[j + 1]) <= MAX_BEND):
                break
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


# --- measuring a design ---------------------------------------------------------------------------------
def measure(route: Route, req: RouteRequest, obs: Obstacles) -> None:
    pts = route.points
    route.legs = []
    for a, b in zip(pts[:-1], pts[1:]):
        plan = math.hypot(*(b[:2] - a[:2]))
        route.legs.append(Leg(a, b, plan, float(np.linalg.norm(b - a)), (b[2] - a[2]) / plan if plan > 1e-9 else 0.0))
    route.length = sum(lg.length for lg in route.legs)
    route.plan_length = sum(lg.plan for lg in route.legs)
    route.climb = float(pts[-1][2] - pts[0][2])
    route.steepest = max((abs(lg.gradient) for lg in route.legs), default=0.0)
    route.direct = math.hypot(*(pts[-1][:2] - pts[0][:2]))
    route.needed = abs(route.climb) / req.gradient
    bend, at = 0.0, 0
    for k in range(1, len(pts) - 1):
        u, v = pts[k][:2] - pts[k - 1][:2], pts[k + 1][:2] - pts[k][:2]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-9 and nv > 1e-9:
            ang = math.degrees(math.acos(float(np.clip(u @ v / (nu * nv), -1.0, 1.0))))
            if ang > bend:
                bend, at = ang, k
    route.sharpest_bend, route.bend_station = bend, at
    # clearances along the centre line (every 2 m), skipping the ends next to the start / target
    samples = []
    for lg in route.legs:
        m = max(1, int(math.ceil(lg.length / 2.0)))
        samples += [lg.a + (lg.b - lg.a) * s for s in np.linspace(0, 1, m, endpoint=False)]
    samples.append(pts[-1])
    S = np.array(samples) if samples else pts
    half = 0.5 * math.hypot(req.width, req.height)
    end_r = (route.free_r or free_radius(req, route.cell or 5.0)) + (route.cell or 5.0)
    away = (np.linalg.norm(S - pts[0], axis=1) > end_r) & (np.linalg.norm(S - pts[-1], axis=1) > end_r)
    route.void_clear = _clearance_to_shapes(S[away], obs.voids, half) if away.any() and obs.voids else None
    if obs.drives and away.any():
        d = np.full(int(away.sum()), np.inf)
        for dpts, hs, _name in obs.drives:
            d = np.minimum(d, _poly_dist(S[away], np.asarray(dpts, float)) - hs - half)
        route.drive_clear = float(d.min()) if np.isfinite(d.min()) else None
    if obs.ore is not None:
        route.ore_clear = _clearance_to_mesh(S, *obs.ore, half)


def _clearance_to_mesh(S: np.ndarray, vertices, triangles, half: float) -> float | None:
    """Smallest signed distance (outside +, inside -) from the samples to a closed mesh, less half the
    profile."""
    import pyvista as pv
    import vtk

    faces = np.hstack([np.full((len(triangles), 1), 3), np.asarray(triangles, dtype=np.int64)]).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=np.float64), faces=faces)
    f = vtk.vtkImplicitPolyDataDistance()
    f.SetInput(mesh)
    best = math.inf
    for p in S:
        best = min(best, f.EvaluateFunction(*map(float, p)))
    return None if not math.isfinite(best) else best - half


def _clearance_to_shapes(S: np.ndarray, shapes: list, half: float) -> float | None:
    from ..scene.production import prism_mesh

    best = math.inf
    for sh in shapes:
        lo, hi = sh.bounds()
        if not np.any(np.all((S >= np.asarray(lo) - 200) & (S <= np.asarray(hi) + 200), axis=1)):
            continue
        mesh = prism_mesh(sh)
        if mesh is None or mesh.n_cells == 0:
            continue
        mesh = mesh.triangulate()
        tri = mesh.faces.reshape(-1, 4)[:, 1:]
        c = _clearance_to_mesh(S, mesh.points, tri, 0.0)
        if c is not None:
            best = min(best, c)
    return None if not math.isfinite(best) else best - half


# --- the design ------------------------------------------------------------------------------------------
def plan_route(req: RouteRequest, obs: Obstacles) -> Route:
    s, t = np.asarray(req.start, float), np.asarray(req.target, float)
    if req.gradient <= 0:
        return Route(False, tr("The gradient limit must be more than 0."), request=req)
    if np.linalg.norm(t - s) < 1.0:
        return Route(False, tr("The start and the target are at the same place."), request=req)
    if req.start_bearing is not None:
        return _plan_with_direction(req, obs)
    radius = max(req.min_radius, 0.0)
    best = None
    for attempt in range(3):                       # wider bends when the route comes too close to itself
        route = _design(req, obs, s, t, radius)
        if not route.ok and best is None and radius > 0:
            tight = _design(req, obs, s, t, 0.0)   # no room for that radius: the tightest gradual bends
            if tight.ok:
                tight.radius_reduced = True
                route = tight
        if not route.ok:
            if best is None:
                route.findings = []
                return route
            break
        need = req.drive_clearance - 0.5
        if best is None or (route.self_clear or 1e9) > (best.self_clear or 1e9):
            best = route
        if route.self_clear is None or route.self_clear >= need:
            break
        radius *= 1.6
    best.findings = findings(best, req)
    return best


# --- a start direction set by the user ----------------------------------------------------------------
def bearing_vector(bearing: float) -> np.ndarray:
    """Plan unit vector (east, north, 0) of a bearing in degrees (from north, clockwise)."""
    b = math.radians(bearing)
    return np.array([math.sin(b), math.cos(b), 0.0])


def bearing_of(a, b) -> float:
    """Bearing (degrees from north, clockwise, 0-360) from a to b in plan."""
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0


def _heading_index(bearing: float) -> int:
    """The search heading (DIRS index, 45° apart, 0 = east, counter-clockwise) nearest to a bearing."""
    return int(round(((90.0 - bearing) % 360.0) / 45.0)) % 8


def _leg_clear(a, b, exact: _Exact, free) -> bool:
    """Exact clearance of the straight leg a-b (outside the free zones)."""
    n = max(2, int(math.ceil(math.dist(a, b) / EXACT_STEP)))
    P = a + (b - a) * np.linspace(0.0, 1.0, n + 1)[:, None]
    keep = np.ones(len(P), dtype=bool)
    for q, r in free:
        keep &= np.linalg.norm(P - q, axis=1) > r
    return exact.clear(P[keep]) if keep.any() else True


def _finish(route: Route, req: RouteRequest, obs: Obstacles) -> Route:
    """Measure a design made of ``route.points`` and explain it."""
    free = end_zones(req, obs, route.cell or 5.0)
    route.free_r = max(r for _p, r in free)
    route.meets = [(k, name, d) for k, p in enumerate((req.start, req.target))
                   for name, d, _hs in end_contacts(req, obs, p)]
    route.ok = True
    route.request = req
    measure(route, req, obs)
    route.self_clear = self_clearance(route.points, req)
    route.findings = findings(route, req)
    return route


def straight_route(req: RouteRequest, obs: Obstacles) -> Route | None:
    """The single straight leg start - target when it keeps the gradient and the clearances, else None."""
    s, t = np.asarray(req.start, float), np.asarray(req.target, float)
    plan = math.hypot(*(t[:2] - s[:2]))
    if plan < 1e-6 or abs(t[2] - s[2]) > req.gradient * plan * (1 + 1e-6):
        return None
    cell = req.cell or max(2.0, min(req.width, 5.0))
    if not _leg_clear(s, t, _Exact(req, obs), end_zones(req, obs, cell)):
        return None
    return _finish(Route(True, points=np.array([s, t]), cell=cell, radius=req.min_radius), req, obs)


def _plan_with_direction(req: RouteRequest, obs: Obstacles) -> Route:
    """The drive leaves the start in the bearing asked: a straight lead-in leg in that direction (long
    enough to get clear of the drive it leaves and to start a bend), then the search carries on from
    its end with that heading."""
    from dataclasses import replace

    s, t = np.asarray(req.start, float), np.asarray(req.target, float)
    u = bearing_vector(req.start_bearing)
    plan = math.hypot(*(t[:2] - s[:2]))
    if plan > 1e-6 and math.degrees(math.acos(float(np.clip((t - s)[:2] @ u[:2] / plan, -1, 1)))) < 1.0:
        r = straight_route(req, obs)             # the target lies in that direction: straight if clear
        if r is not None:
            return r
    cell = req.cell or max(2.0, min(req.width, 5.0))
    lead = max(free_radius(req, cell, end_half_size(obs, s)), req.min_radius, 2 * cell)
    rise = float(t[2] - s[2])                   # the lead-in takes its share of the height change
    half = 0.5 * math.hypot(req.width, req.height)

    def lead_end(L):
        dz = float(np.clip(rise * L / max(plan, abs(rise) / req.gradient, L), -req.gradient * L, req.gradient * L))
        return s + L * u + np.array([0.0, 0.0, dz])

    def in_a_band(p):                           # where the search grid would block it (drives + grid margin)
        return any(_poly_dist(p[None], np.asarray(pts, float))[0] < hs + half + req.drive_clearance + 0.75 * cell
                   for pts, hs, _name in obs.drives if len(pts))

    p1 = lead_end(lead)
    for _ in range(int(3 * lead / cell)):       # carry on until the lead-in is out of the drives' clearance
        if not in_a_band(p1):
            break
        lead += cell
        p1 = lead_end(lead)
    if not _leg_clear(s, p1, _Exact(req, obs), end_zones(req, obs, cell)):
        return Route(False, tr("Leaving the start towards {b:.0f}°, the drive runs into an existing drive, old "
                               "workings or the ore within {l:.0f} m - choose another start direction or "
                               "Automatic.", b=req.start_bearing, l=lead), request=req)
    if np.linalg.norm(t - p1) < 1.0:
        return _finish(Route(True, points=np.array([s, t]), cell=cell, radius=req.min_radius), req, obs)
    sub = replace(req, start=tuple(p1), start_bearing=None, start_free=0.75 * cell,
                  start_heading=_heading_index(req.start_bearing), lead_from=tuple(s))
    r = plan_route(sub, obs)
    if not r.ok:
        r.request = req
        return r
    route = Route(True, points=np.vstack([s[None], r.points]), cell=r.cell, explored=r.explored,
                  radius=r.radius, radius_reduced=r.radius_reduced)
    return _finish(route, req, obs)


def _design(req: RouteRequest, obs: Obstacles, s: np.ndarray, t: np.ndarray, radius: float) -> Route:
    grid = _layout(req)
    blocked = _blocked(grid, req, obs)
    sn, tn = grid.node(s), grid.node(t)
    step = min(grid.cx, grid.cy)
    run_steps = max(1, int(math.ceil(radius / (1.307 * step))))     # a 45° turn every n steps: radius ~1.3 n cells
    path, explored = _astar(grid, blocked, sn, tn, req.gradient, run_steps, req.start_heading)
    lead = None if req.lead_from is None else np.asarray(req.lead_from, float)
    route = Route(False, cell=grid.cell, explored=explored, request=req, radius=radius)
    if path is None:
        route.message = tr("No route found inside the search area: the gradient limit, the bend radius and the "
                           "clearances leave no way through. Try a steeper limit, smaller clearances or bend radius, "
                           "or another target.")
        route.direct = math.hypot(*(t[:2] - s[:2]))
        route.needed = abs(t[2] - s[2]) / req.gradient
        return route
    pts = np.array([grid.xyz(*n) for n in path])
    pts[0] = s                                              # a grid node already (the grid is aligned on it)
    free = end_zones(req, obs, grid.cell)
    free_r = max(r for _p, r in free)
    exact = _Exact(req, obs)                # straight legs: the true clearances (the grid keeps a margin)
    # join the exact target from the last grid point that reaches it within the gradient
    def joins(m):                           # within the gradient, clear, and no sharp turn onto the target
        prev = pts[m - 1] if m else lead
        return _leg_ok(pts[m], t, grid, blocked, req.gradient, free, exact) and \
            (prev is None or _bend(prev, pts[m], t) <= MAX_BEND)

    m = len(pts) - 1
    while m > 0 and not joins(m):
        m -= 1
    if m == 0 and not _leg_ok(pts[0], t, grid, blocked, req.gradient, free, exact):
        m = len(pts) - 1                    # keep the whole path; a too steep end is reported, not hidden
    pts = np.vstack([pts[:m + 1], t[None]])
    route.points = _straighten(pts, grid, blocked, req.gradient, free, exact, lead)
    route.free_r = free_r
    route.meets = [(k, name, d) for k, p in enumerate((req.start, req.target)) for name, d, _hs in end_contacts(req, obs, p)]
    route.ok = True
    measure(route, req, obs)
    route.self_clear = self_clearance(route.points, req)
    return route


def self_clearance(points: np.ndarray, req: RouteRequest) -> float | None:
    """Smallest wall-to-wall distance between parts of the route that are far apart along it (the legs
    of a loop), or None when no parts are far enough apart along it to compare."""
    samples, ch = [], []
    run = 0.0
    for a, b in zip(points[:-1], points[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(L / 2.0)))
        for u in np.linspace(0.0, 1.0, n, endpoint=False):
            samples.append(a + (b - a) * u)
            ch.append(run + L * u)
        run += L
    samples.append(points[-1])
    ch.append(run)
    S, C = np.array(samples), np.array(ch)
    size = 2.0 * 0.5 * math.hypot(req.width, req.height)
    apart = np.pi * max(req.min_radius, size) + 2 * (size + req.drive_clearance)   # neighbours along the route
    best = math.inf
    for i in range(len(S)):
        far = C - C[i] > apart
        if far.any():
            d = np.linalg.norm(S[far] - S[i], axis=1).min() - size
            best = min(best, float(d))
    return None if not math.isfinite(best) else best


# --- the best start point along the existing drives ----------------------------------------------------
BEND_PENALTY = 10.0           # m: a bend counts as this many metres more when start points are compared,
                              # so a straight drive wins unless a bent one is clearly shorter
CANDIDATE_STEP = 2.5          # m between the start points tried along the drives
BENT_TRIES = 5                # start points (the most promising) designed in full when no straight drive wins
MAX_ATTEMPTS = 20             # ... trying at most this many (a start direction can rule many out)


@dataclass
class StartPoint:
    point: np.ndarray             # on the centre line of an existing drive
    line: int                     # index of the line (drive) it is on
    drive: str                    # the drive's name
    after: str                    # the station before it on that line ...
    before: str                   # ... and the one after it
    chainage: float               # m from ``after`` along the line


def start_points(lines: list, step: float = CANDIDATE_STEP, target=None, bearing: float | None = None,
                 only=None) -> list:
    """Points every ``step`` m along the existing drives' centre lines ``lines`` = [(points (K, 3), station
    names, drive name)]; with ``bearing`` also the exact points from which the target lies in that direction.
    ``only``: the indices of the lines to search (None: all)."""
    out = []
    u = None if bearing is None else bearing_vector(bearing)
    for li, (pts, names, drive) in enumerate(lines):
        if only is not None and li not in only:
            continue
        pts = np.asarray(pts, float)
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            L = float(np.linalg.norm(b - a))
            if L < 1e-6:
                continue
            fr = list(np.arange(0.0, L, step) / L) + ([1.0] if k == len(pts) - 2 else [])
            if u is not None and target is not None:            # the target straight ahead in that direction
                d, w = b - a, np.asarray(target, float) - a
                den = d[0] * u[1] - d[1] * u[0]
                if abs(den) > 1e-9:
                    mu = (w[0] * u[1] - w[1] * u[0]) / den
                    c = a + mu * d
                    if 0.0 <= mu <= 1.0 and (np.asarray(target, float) - c)[:2] @ u[:2] > 0:
                        fr.append(mu)
            for f in fr:
                if f >= 1.0 - 1e-9:                             # at the station that ends the segment
                    out.append(StartPoint(b.copy(), li, drive, names[k + 1], names[k], 0.0))
                else:
                    out.append(StartPoint(a + f * (b - a), li, drive, names[k], names[k + 1], f * L))
    return out


def _safe(r: Route, req: RouteRequest) -> bool:
    """The design keeps the gradient and all the clearances asked."""
    if not r.ok or r.steepest > req.gradient * 1.001:
        return False
    for clear, need in ((r.void_clear, req.void_clearance), (r.drive_clear, req.drive_clearance),
                        (r.self_clear, req.drive_clearance)):
        if clear is not None and clear < need - 0.5:
            return False
    if req.ore_standoff is not None and r.ore_clear is not None and r.ore_clear < req.ore_standoff - 0.5:
        return False
    return True


def _clear_of_other_drives(points: np.ndarray, sp: StartPoint, req: RouteRequest, obs: Obstacles,
                           t_zone: float) -> bool:
    """Near the start the planner ignores the drives (it leaves one there): the route must still keep the
    clearance from the *other* drives there (not the one it starts on), up to the target's zone."""
    half = 0.5 * math.hypot(req.width, req.height)
    S = []
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / EXACT_STEP)))
        S += [a + (b - a) * f for f in np.linspace(0.0, 1.0, n, endpoint=False)]
    S = np.array(S + [points[-1]])
    S = S[np.linalg.norm(S - points[-1], axis=1) > t_zone]
    if not len(S):
        return True
    for li, (pts, hs, _name) in enumerate(obs.drives):
        if li == sp.line or not len(pts):
            continue
        if (_poly_dist(S, np.asarray(pts, float)) < hs + half + req.drive_clearance - 0.5).any():
            return False
    return True


def best_start(req: RouteRequest, obs: Obstacles, lines: list, only=None) -> Route:
    """The route from the best point anywhere along the existing drives to the target.

    ``lines`` = [(points, station names, drive name)], in the same order as ``obs.drives`` (all of them are
    obstacles); ``only``: the indices of the lines the start may be on (None: any). Best means:
    safe (the gradient and every clearance kept, and the new junction clear of the other drives), then
    straight, then short - a route's score is its length plus BEND_PENALTY m per bend. Every point every
    CANDIDATE_STEP m is tried as a straight drive; the most promising points are also designed in full
    (with bends) when that could beat the best straight drive."""
    from dataclasses import replace

    t = np.asarray(req.target, float)
    g = req.gradient
    half = 0.5 * math.hypot(req.width, req.height)
    cands = start_points(lines, max(CANDIDATE_STEP, 0.5 * (req.cell or 5.0)), t, req.start_bearing, only)
    # the new junction must be clear of the other drives (not at, or next to, an existing junction)
    safe_pts = []
    void_check = _Exact(req, Obstacles(obs.voids, [], obs.ore))
    for sp in cands:
        p = sp.point[None]
        if any(li != sp.line and len(pts) and _poly_dist(p, np.asarray(pts, float))[0] < hs + half + req.drive_clearance
               for li, (pts, hs, _n) in enumerate(obs.drives)):
            continue
        if not void_check.clear(p):
            continue
        safe_pts.append(sp)
    if not safe_pts:
        return Route(False, tr("No safe start point on the existing drives: every point is too close to another "
                               "drive, old workings or the ore."), request=req)

    def request_from(sp):
        return replace(req, start=tuple(sp.point), start_name=_start_name(sp))

    def lower_bound(sp):
        plan = math.hypot(*(t[:2] - sp.point[:2]))
        return max(math.dist(sp.point, t), abs(t[2] - sp.point[2]) / g if g > 0 else plan)

    t_zone = end_zones(req, obs, req.cell or 5.0)[1][1]
    tried = []                                   # (score, route, start point)
    # 1. straight drives (in the start direction asked, when one is set)
    u = None if req.start_bearing is None else bearing_vector(req.start_bearing)
    straight = []
    for sp in safe_pts:
        d = t - sp.point
        plan = math.hypot(d[0], d[1])
        if plan < 1.0 or abs(d[2]) > g * plan * (1 + 1e-6):
            continue
        if u is not None and d[:2] @ u[:2] < plan * math.cos(math.radians(1.0)):
            continue
        straight.append((float(np.linalg.norm(d)), sp))
    straight.sort(key=lambda x: x[0])
    for _length, sp in straight:
        r = straight_route(replace(request_from(sp), start_bearing=None), obs)
        if r is not None and _safe(r, req) and _clear_of_other_drives(r.points, sp, req, obs, t_zone):
            tried.append((r.length, r, sp))
            break                                # the shortest straight drive that is clear
    best_score = tried[0][0] if tried else math.inf
    # 2. routes with bends from the most promising points, when they could still beat it
    picked, found = [], 0
    for sp in sorted(safe_pts, key=lower_bound):
        if lower_bound(sp) + BEND_PENALTY >= best_score or found >= BENT_TRIES or len(picked) >= MAX_ATTEMPTS:
            break
        if any(np.linalg.norm(sp.point - q.point) < 25.0 for q in picked):
            continue                             # spread the tries along the drives
        picked.append(sp)
        r = plan_route(request_from(sp), obs)
        if not _safe(r, req) or not _clear_of_other_drives(r.points, sp, req, obs, t_zone):
            continue
        found += 1
        score = r.length + BEND_PENALTY * (len(r.legs) - 1)
        tried.append((score, r, sp))
        best_score = min(best_score, score)
    if not tried:
        return Route(False, tr("No safe route from any point of the existing drives: the gradient limit and the "
                               "clearances leave no way through. Try a steeper limit, smaller clearances or "
                               "another target."), request=req)
    tried.sort(key=lambda x: x[0])
    score, route, sp = tried[0]
    why = (tr("a straight drive is possible from there") if len(route.legs) == 1 else
           tr("no straight drive is possible from any of them (too steep, something in the way or not in the "
              "direction asked), so the shortest safe route was chosen"))
    where = (tr("at station {a} on {drive}", a=sp.after, drive=sp.drive) if sp.chainage < 0.05 else
             tr("on {drive}, {d:.1f} m from {a} towards {b}", drive=sp.drive, d=sp.chainage, a=sp.after, b=sp.before))
    route.findings.insert(0, (INFO, tr("Best start point: {where} (X {x:,.2f}, Y {y:,.2f}, Z {z:,.2f}). {n:,} points "
                                       "along the existing drives were compared; {why}.", where=where,
                                       x=sp.point[0], y=sp.point[1], z=sp.point[2], n=len(cands), why=why)))
    if len(tried) > 1:
        _s2, r2, sp2 = tried[1]
        route.findings.insert(1, (INFO, trn("Next best: from {name} on {drive}, {m:,.0f} m in {n} leg.",
                                            "Next best: from {name} on {drive}, {m:,.0f} m in {n} legs.",
                                            len(r2.legs), name=_start_name(sp2), drive=sp2.drive, m=r2.length)))
    route.start_point = sp
    route.compared = len(cands)
    return route


def _start_name(sp: StartPoint) -> str:
    """A start point as "P1A-02 +23.4 m" (from that station along the drive)."""
    return sp.after if sp.chainage < 0.05 else f"{sp.after} +{sp.chainage:.1f} m"


def ratio_text(g: float) -> str:
    """A gradient as "1 in 7.2 (13.9 %)"."""
    if abs(g) < 1e-6:
        return tr("level")
    return tr("1 in {n:.1f} ({p:.1f} %)", n=1.0 / abs(g), p=abs(g) * 100.0)


def findings(r: Route, req: RouteRequest) -> list:
    out = []
    up = tr("climbs {h:.1f} m", h=r.climb) if r.climb > 0.05 else (tr("goes down {h:.1f} m", h=-r.climb)
                                                                    if r.climb < -0.05 else tr("stays level"))
    out.append((OK, trn("A route was found: {length:,.0f} m of drive in {n} straight leg; it {up}.",
                        "A route was found: {length:,.0f} m of drive in {n} straight legs; it {up}.", len(r.legs),
                        length=r.length, up=up)))
    limit = req.gradient
    if r.steepest <= limit * 1.001:
        out.append((OK, tr("Steepest part {g}: within the limit of {limit}.", g=ratio_text(r.steepest),
                           limit=ratio_text(limit))))
    elif r.steepest <= limit * 1.05:
        out.append((WARN, tr("Steepest part {g}: just over the limit of {limit} (at the end next to the target) - "
                             "adjust the last leg.", g=ratio_text(r.steepest), limit=ratio_text(limit))))
    else:
        out.append((BAD, tr("Steepest part {g}: over the limit of {limit}.", g=ratio_text(r.steepest),
                            limit=ratio_text(limit))))
    if r.needed > r.direct * 1.05:
        out.append((INFO, tr("A straight drive to the target would be {g}, steeper than the limit of {limit}. To change "
                             "height by {h:.0f} m within the limit the drive must be at least {n:,.0f} m long, so the "
                             "route makes a loop to gain that length. It is {length:,.0f} m: {p:.0%} more than that "
                             "minimum (for the bends and the clearances).",
                             g=ratio_text(abs(r.climb) / max(r.direct, 1e-9)), limit=ratio_text(limit),
                             h=abs(r.climb), n=r.needed, length=r.length,
                             p=max(r.plan_length / max(r.needed, 1e-9) - 1, 0.0))))
    elif r.plan_length > 1.05 * max(r.direct, r.needed):
        out.append((INFO, tr("The route is {p:.0%} longer than the straight line ({d:,.0f} m), to keep clear of the "
                             "existing drives and old workings and to keep the bends gradual.",
                             p=r.plan_length / max(r.direct, 1e-9) - 1, d=r.direct)))
    for clear, need, ok_txt, near_txt in (
            (r.void_clear, req.void_clearance, N_VOID_OK, N_VOID_NEAR),
            (r.drive_clear, req.drive_clearance, N_DRIVE_OK, N_DRIVE_NEAR)):
        if clear is None:
            continue
        if clear >= need - 0.5:
            out.append((OK, tr(ok_txt, c=clear)))
        else:
            out.append((WARN if clear > 0 else BAD, tr(near_txt, c=max(clear, 0.0), need=need)))
    if r.self_clear is not None:
        if r.self_clear >= req.drive_clearance - 0.5:
            out.append((OK, tr("Its loop keeps {c:.0f} m of rock between its own legs.", c=r.self_clear)))
        else:
            out.append((WARN, tr("Its loop leaves only {c:.0f} m of rock between its own legs (asked: {need:.0f} m) - "
                                 "try a larger bend radius or another target.", c=max(r.self_clear, 0.0),
                                 need=req.drive_clearance)))
    if r.ore_clear is not None:
        if req.ore_standoff is None:
            out.append((INFO, tr("The route passes through the ore body (allowed).") if r.ore_clear < 0 else
                        tr("The route stays outside the ore body ({c:.0f} m at the closest).", c=r.ore_clear)))
        elif r.ore_clear >= req.ore_standoff - 0.5:
            out.append((OK, tr("It stays {c:.0f} m outside the ore body at the closest (asked: {need:.0f} m).",
                               c=r.ore_clear, need=req.ore_standoff)))
        else:
            out.append((WARN, tr("It comes within {c:.0f} m of the ore body (asked: {need:.0f} m).",
                                 c=max(r.ore_clear, 0.0), need=req.ore_standoff)))
    if r.sharpest_bend > 100:
        out.append((WARN, tr("Sharpest bend {a:.0f}° at station {k}: a tight turn for trucks - plan a curve or a "
                             "wider turn there.", a=r.sharpest_bend, k=r.bend_station + 1)))
    elif r.sharpest_bend > 0 and not r.radius_reduced:
        out.append((OK, tr("Bends: {a:.0f}° at most, designed for a radius of {r:.0f} m.", a=r.sharpest_bend,
                           r=r.radius)))
    if req.start_bearing is not None and len(r.points) >= 2:
        out.append((INFO, tr("Leaves the start towards {b:.0f}° (as asked).", b=bearing_of(r.points[0], r.points[1]))))
    for end, name, d in r.meets:
        out.append((WARN, tr(N_MEETS[end], d=d, drive=name, need=req.drive_clearance)))
    if r.radius_reduced:
        out.append((WARN, tr("There is no way out with bends of {r:.0f} m radius (the start is boxed in by other "
                             "drives), so tighter bends were used - check the turns, or start from another station.",
                             r=req.min_radius)))
    return out


N_VOID_OK = N_("Keeps {c:.0f} m away from old workings at the closest.")
N_VOID_NEAR = N_("Comes within {c:.0f} m of old workings (asked: {need:.0f} m).")
N_DRIVE_OK = N_("Keeps {c:.0f} m away from the other drives at the closest.")
N_DRIVE_NEAR = N_("Comes within {c:.0f} m of another drive (asked: {need:.0f} m).")
N_MEETS = (N_("The start is only {d:.1f} m from the centre line of drive {drive} (closer than the {need:.0f} m "
              "clearance asked), so the new drive meets {drive} there - choose another start if they must stay "
              "apart."),
           N_("The target is only {d:.1f} m from the centre line of drive {drive} (closer than the {need:.0f} m "
              "clearance asked), so the new drive meets {drive} there - move the target if they must stay apart."))


def write_str(path, points: np.ndarray, name: str = "route") -> None:
    """The design as a Surpac string file (one string, number 1, station names in D1)."""
    import datetime as _dt
    from pathlib import Path

    lines = [f"{Path(path).stem},{_dt.date.today().strftime('%d-%b-%y')},GeoViewer route design,",
             "0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000"]
    for k, (x, y, z) in enumerate(points):
        lines.append(f"1, {y:.3f}, {x:.3f}, {z:.3f}, {name.replace(',', ' ')}-{k + 1:02d}")
    lines += ["0, 0.000, 0.000, 0.000,", "0, 0.000, 0.000, 0.000, END"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="latin-1", errors="replace")


def write_csv(path, route: Route, name: str = "route") -> None:
    from pathlib import Path

    rows = ["station,x,y,z,leg_length_m,leg_gradient_pct"]
    for k, p in enumerate(route.points):
        lg = route.legs[k - 1] if k > 0 else None
        rows.append(f"{name}-{k + 1:02d},{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},"
                    f"{'' if lg is None else f'{lg.length:.2f}'},{'' if lg is None else f'{lg.gradient * 100:.2f}'}")
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8-sig")
