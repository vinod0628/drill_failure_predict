"""Underground survey data: pits (main pits and their branches), stations, instrument setups
and their shots, kept in the project database (tables survey_*), and the traverse computation.

Tables (made by the database set-up when missing):

    survey_pits      id, name, code (short prefix of its station names, e.g. P1, P1A), parent_id
                     (empty: a main pit; else the pit it branches from), start_station,
                     adjust (1: spread a passing closure - Bowditch), note, and the drive's
                     rectangular profile drawn along its centre line: profile_width,
                     profile_height (m), station_level (where the stations sit in the profile:
                     "floor" / "middle" / "back")
    survey_stations  name (key), pit_id, kind ("known": coordinates entered, e.g. an entrance;
                     "station": computed), x, y, z, status, active, note
    survey_setups    id, pit_id, seq (order in the pit), survey_date, surveyor, instrument,
                     at_station, hi, bs_station or bs_bearing + bs_bearing_type (method "angles":
                     total station), or fs_bearing + fs_bearing_type (method "bearing": the leg's
                     direction measured directly, e.g. by compass), or fs_x + fs_y + fs_z (method
                     "coords": the foresight station's coordinates typed in, already calculated
                     elsewhere - they are used as they are, like a known station), note
    survey_shots     id, setup_id, role ("BS" / "FS"), face ("L" / "R"), target, ha, za, sd, ht
    survey_settings  setting, value (closure tolerance, grid convergence, magnetic declination)
    survey_designs   id, name, start_pit_id (the pit the design starts on, if any), start_name,
                     target_name, profile_width, profile_height, station_level, gradient (the
                     steepest slope allowed, rise / run), steepest, length, climb (m), params (the
                     other settings of the design, JSON), note, created_at - a designed pit (a
                     drive designed with the route designer, not surveyed)
    survey_design_points  id, design_id, seq, x, y, z - its centre line, in order

Angles are stored in decimal degrees; bearings clockwise from grid north. The engineer edits
one row per setup (``SetupRow``, like a field book line); it is stored as shots. After every
change the whole network is computed (``compute``) and the stations' coordinates are stored,
so the view reads them directly. No Qt here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..i18n import N_, tr

PITS, STATIONS, SETUPS, SHOTS, SETTINGS = ("survey_pits", "survey_stations", "survey_setups", "survey_shots",
                                           "survey_settings")
DESIGNS, DESIGN_POINTS = "survey_designs", "survey_design_points"
TABLES = (PITS, STATIONS, SETUPS, SHOTS, SETTINGS, DESIGNS, DESIGN_POINTS)
BEARING_TYPES = ("grid", "true", "magnetic")
DEFAULT_SETTINGS = {"closure_ratio": 10000.0, "convergence": 0.0, "declination": 0.0}
FACE_TOLERANCE = 30.0 / 3600.0          # degrees: face left / right readings further apart are flagged
BS_CHECK_TOLERANCE = 0.05               # metres: backsight distance / height off the stored leg by more


def _ddl(mysql: bool) -> dict[str, str]:
    opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if mysql else ""
    return {
        PITS: f"CREATE TABLE {PITS} (id INTEGER NOT NULL PRIMARY KEY, name VARCHAR(100) NOT NULL, "
              f"code VARCHAR(16), parent_id INTEGER, start_station VARCHAR(64), adjust SMALLINT, note TEXT, "
              f"profile_width DOUBLE, profile_height DOUBLE, station_level VARCHAR(10), "
              f"created_at VARCHAR(19)){opts}",
        STATIONS: f"CREATE TABLE {STATIONS} (name VARCHAR(64) NOT NULL PRIMARY KEY, pit_id INTEGER, "
                  f"kind VARCHAR(10) NOT NULL, x DOUBLE, y DOUBLE, z DOUBLE, status VARCHAR(12), "
                  f"active SMALLINT, note TEXT, updated_at VARCHAR(19)){opts}",
        SETUPS: f"CREATE TABLE {SETUPS} (id INTEGER NOT NULL PRIMARY KEY, pit_id INTEGER NOT NULL, "
                f"seq INTEGER NOT NULL, survey_date VARCHAR(10), surveyor VARCHAR(100), instrument VARCHAR(100), "
                f"at_station VARCHAR(64) NOT NULL, hi DOUBLE, bs_station VARCHAR(64), bs_bearing DOUBLE, "
                f"bs_bearing_type VARCHAR(10), fs_bearing DOUBLE, fs_bearing_type VARCHAR(10), fs_x DOUBLE, fs_y DOUBLE, "
                f"fs_z DOUBLE, note TEXT){opts}",
        SHOTS: f"CREATE TABLE {SHOTS} (id INTEGER NOT NULL PRIMARY KEY, setup_id INTEGER NOT NULL, "
               f"role VARCHAR(2) NOT NULL, face CHAR(1) NOT NULL, target VARCHAR(64), ha DOUBLE, za DOUBLE, "
               f"sd DOUBLE, ht DOUBLE){opts}",
        SETTINGS: f"CREATE TABLE {SETTINGS} (setting VARCHAR(100) NOT NULL PRIMARY KEY, value TEXT){opts}",
        DESIGNS: f"CREATE TABLE {DESIGNS} (id INTEGER NOT NULL PRIMARY KEY, name VARCHAR(100) NOT NULL, "
                 f"start_pit_id INTEGER, start_name VARCHAR(100), target_name VARCHAR(100), profile_width DOUBLE, "
                 f"profile_height DOUBLE, station_level VARCHAR(10), gradient DOUBLE, steepest DOUBLE, "
                 f"length DOUBLE, climb DOUBLE, params TEXT, note TEXT, created_at VARCHAR(19)){opts}",
        DESIGN_POINTS: f"CREATE TABLE {DESIGN_POINTS} (id INTEGER NOT NULL PRIMARY KEY, design_id INTEGER NOT NULL, "
                       f"seq INTEGER NOT NULL, x DOUBLE, y DOUBLE, z DOUBLE){opts}",
    }


# Fields each table must have (added as empty columns when missing - the set-up only adds).
_FIELDS = {
    PITS: [("name", "VARCHAR(100)"), ("code", "VARCHAR(16)"), ("parent_id", "INTEGER"), ("start_station", "VARCHAR(64)"),
           ("adjust", "SMALLINT"), ("note", "TEXT"), ("profile_width", "DOUBLE"), ("profile_height", "DOUBLE"),
           ("station_level", "VARCHAR(10)"), ("created_at", "VARCHAR(19)")],
    STATIONS: [("pit_id", "INTEGER"), ("kind", "VARCHAR(10)"), ("x", "DOUBLE"), ("y", "DOUBLE"), ("z", "DOUBLE"),
               ("status", "VARCHAR(12)"), ("active", "SMALLINT"), ("note", "TEXT"), ("updated_at", "VARCHAR(19)")],
    SETUPS: [("pit_id", "INTEGER"), ("seq", "INTEGER"), ("survey_date", "VARCHAR(10)"), ("surveyor", "VARCHAR(100)"),
             ("instrument", "VARCHAR(100)"), ("at_station", "VARCHAR(64)"), ("hi", "DOUBLE"),
             ("bs_station", "VARCHAR(64)"), ("bs_bearing", "DOUBLE"), ("bs_bearing_type", "VARCHAR(10)"),
             ("fs_bearing", "DOUBLE"), ("fs_bearing_type", "VARCHAR(10)"), ("fs_x", "DOUBLE"), ("fs_y", "DOUBLE"),
             ("fs_z", "DOUBLE"), ("note", "TEXT")],
    SHOTS: [("setup_id", "INTEGER"), ("role", "VARCHAR(2)"), ("face", "CHAR(1)"), ("target", "VARCHAR(64)"),
            ("ha", "DOUBLE"), ("za", "DOUBLE"), ("sd", "DOUBLE"), ("ht", "DOUBLE")],
    SETTINGS: [("setting", "VARCHAR(100)"), ("value", "TEXT")],
    DESIGNS: [("name", "VARCHAR(100)"), ("start_pit_id", "INTEGER"), ("start_name", "VARCHAR(100)"),
              ("target_name", "VARCHAR(100)"), ("profile_width", "DOUBLE"), ("profile_height", "DOUBLE"),
              ("station_level", "VARCHAR(10)"), ("gradient", "DOUBLE"), ("steepest", "DOUBLE"), ("length", "DOUBLE"),
              ("climb", "DOUBLE"), ("params", "TEXT"), ("note", "TEXT"), ("created_at", "VARCHAR(19)")],
    DESIGN_POINTS: [("design_id", "INTEGER"), ("seq", "INTEGER"), ("x", "DOUBLE"), ("y", "DOUBLE"), ("z", "DOUBLE")],
}
_INDEXES = ((STATIONS, "idx_survey_stations_pit", ["pit_id"]), (SETUPS, "idx_survey_setups_pit", ["pit_id", "seq"]),
            (SHOTS, "idx_survey_shots_setup", ["setup_id"]),
            (DESIGN_POINTS, "idx_survey_design_points", ["design_id", "seq"]))


# --- angles ---------------------------------------------------------------------------------------
def parse_angle(text) -> float | None:
    """Degrees from "120 30 00", "120°30′00″", "120-30-00", "120:30:00" or decimal "120.5".
    None for empty; ValueError for anything else."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return None if isinstance(text, float) and math.isnan(text) else float(text)
    t = str(text).strip()
    if not t:
        return None
    parts = [p for p in re.split(r"[\s°º'′\"″:dms-]+", t.replace("''", '"')) if p]
    if len(parts) == 1 and not re.search(r"[\s°º'′\"″:-]", t):
        return float(parts[0])                         # decimal degrees
    if not 1 <= len(parts) <= 3:
        raise ValueError(tr("not an angle: {t!r} (use e.g. 120 30 00 or 120.5)", t=t))
    d, m, s = (list(map(float, parts)) + [0.0, 0.0])[:3]
    if m >= 60 or s >= 60 or m < 0 or s < 0:
        raise ValueError(tr("minutes and seconds must be below 60: {t!r}", t=t))
    sign = -1.0 if t.startswith("-") or d < 0 else 1.0
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


def format_dms(deg, seconds: int = 0) -> str:
    """120.5 -> "120°30′00″" (seconds rounded to ``seconds`` decimals)."""
    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return ""
    sign = "-" if deg < 0 else ""
    total = round(abs(float(deg)) * 3600.0, seconds)
    if 0 <= float(deg) < 360.0 and total >= 360 * 3600.0:      # 359°59′59.7″ rounds to 0°, not 360°
        total = 0.0
    d, rest = divmod(total, 3600.0)
    m, s = divmod(rest, 60.0)
    sec = f"{s:0{3 + seconds if seconds else 2}.{seconds}f}"
    return f"{sign}{int(d)}°{int(m):02d}′{sec}″"


def bearing_of(dx: float, dy: float) -> float:
    """Bearing (degrees clockwise from north) of the direction (dE, dN)."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _norm(a: float) -> float:
    return a % 360.0


# --- what the engineer edits: one row per setup -----------------------------------------------
@dataclass
class SetupRow:
    """One instrument setup as a field book line (face right readings optional)."""
    at: str
    hi: float | None = None
    bs_station: str | None = None          # backsight to a station with coordinates ...
    bs_bearing: float | None = None        # ... or a known bearing (degrees) to a reference mark
    bs_bearing_type: str = "grid"
    bs_ha: float | None = None             # empty: the circle was set to the backsight bearing
    bs_za: float | None = None
    bs_sd: float | None = None
    bs_ht: float | None = None
    fs: str | None = None
    fs_ha: float | None = None
    fs_za: float | None = None
    fs_sd: float | None = None
    fs_ht: float | None = None
    fs_bearing: float | None = None        # method "bearing": the leg's direction measured directly ...
    fs_bearing_type: str = "magnetic"      # ... by compass (magnetic), or a true / grid bearing
    fs_x: float | None = None              # method "coords": the foresight's coordinates typed in
    fs_y: float | None = None              # (already calculated, e.g. by another program) - used
    fs_z: float | None = None              # as they are, no readings needed
    bs_ha_r: float | None = None           # face right
    bs_za_r: float | None = None
    fs_ha_r: float | None = None
    fs_za_r: float | None = None
    date: str = ""
    surveyor: str = ""
    instrument: str = ""
    note: str = ""
    id: int | None = None

    @property
    def method(self) -> str:
        """"coords" when the foresight's coordinates were typed in, "bearing" when its direction
        was measured directly, else "angles"."""
        if any(v is not None for v in (self.fs_x, self.fs_y, self.fs_z)):
            return "coords"
        return "bearing" if self.fs_bearing is not None else "angles"


@dataclass
class Pit:
    id: int
    name: str
    parent_id: int | None = None
    start_station: str | None = None
    adjust: bool = True
    note: str = ""
    code: str = ""
    profile_width: float = 5.0          # m - the drive's profile is a rectangle
    profile_height: float = 5.0         # m
    station_level: str = "back"         # the stations are in the "back" (roof), the "floor" or the "middle"


STATION_LEVELS = {"back": N_("Backs (roof)"), "floor": N_("Floor"), "middle": N_("Middle")}  # tr() where shown


@dataclass
class DesignPit:
    """A designed pit: a drive designed with the route designer (its centre line and profile), kept
    apart from the surveyed pits - nothing is surveyed or computed from it."""
    id: int
    name: str
    points: list                        # [(x, y, z)] centre line, in order
    profile_width: float = 5.0
    profile_height: float = 5.0
    station_level: str = "middle"       # where the points are in the profile (as Pit.station_level)
    start_pit_id: int | None = None     # the surveyed pit it starts on (None: not known)
    start_name: str = ""
    target_name: str = ""
    gradient: float | None = None       # the steepest slope allowed (rise / run)
    steepest: float | None = None       # the steepest leg designed
    length: float = 0.0                 # m, along the drive
    climb: float = 0.0                  # m, end minus start
    params: dict = field(default_factory=dict)
    note: str = ""
    created_at: str = ""


def design_measures(points) -> tuple:
    """(length along the drive, end minus start height, steepest leg as rise / run) of a centre line
    [(x, y, z)] - as the route designer measures a design."""
    length, steepest = 0.0, 0.0
    for a, b in zip(points[:-1], points[1:]):
        dx, dy, dz = (float(b[k]) - float(a[k]) for k in range(3))
        plan = math.hypot(dx, dy)
        length += math.sqrt(plan * plan + dz * dz)
        if plan > 1e-9:
            steepest = max(steepest, abs(dz) / plan)
    climb = float(points[-1][2]) - float(points[0][2]) if len(points) >= 2 else 0.0
    return length, climb, steepest


def next_code(pits: list, parent_id: int | None) -> str:
    """The next pit code: main pits P1, P2, ...; branches the parent's code + A, B, ... (and
    a digit after a letter: P1A1)."""
    used = {p.code for p in pits}
    if parent_id is None:
        n = 1
        while f"P{n}" in used:
            n += 1
        return f"P{n}"
    parent = next((p for p in pits if p.id == parent_id), None)
    base = parent.code if parent is not None and parent.code else f"P{parent_id}"
    marks = "ABCDEFGHJKLMNPQRSTUVWXYZ" if base[-1].isdigit() else [str(i) for i in range(1, 100)]
    return next((base + m for m in marks if base + m not in used), f"{base}-{len(used) + 1}")


def next_station_name(code: str, names) -> str:
    """The next station name of a pit: <code>-01, <code>-02, ... (the first number not in use)."""
    names = set(names)
    n = 1
    while f"{code}-{n:02d}" in names:
        n += 1
    return f"{code}-{n:02d}"


@dataclass
class Station:
    name: str
    pit_id: int | None = None
    kind: str = "station"                  # "known" (entered) or "station" (computed)
    x: float | None = None
    y: float | None = None
    z: float | None = None
    status: str = ""
    active: bool = True
    note: str = ""


# --- the computation ----------------------------------------------------------------------------------
@dataclass
class Leg:
    pit_id: int
    setup_index: int
    frm: str
    to: str
    bearing: float
    hd: float
    dz: float
    closing: bool = False                  # the foresight was a station with coordinates already
    entered: bool = False                  # the foresight's coordinates were typed in (method "coords")


@dataclass
class PitResult:
    pit_id: int
    length: float = 0.0
    closure: tuple | None = None           # (dE, dN, dZ, linear, ratio, passed, onto station)
    adjusted: bool = False
    messages: list = field(default_factory=list)


@dataclass
class Network:
    coords: dict                            # station -> (x, y, z)
    status: dict                            # station -> status
    legs: list
    pits: dict                              # pit id -> PitResult
    messages: list                          # [(pit id, setup index, text)] warnings / errors


def _mean_face(left, right, za: bool = False):
    """Mean of face left / right readings (right turned back by 180°, or 360° - ZA)."""
    if left is None:
        return None, None
    if right is None:
        return left, None
    r = (360.0 - right) if za else _norm(right - 180.0)
    diff = (r - left + 180.0) % 360.0 - 180.0
    return _norm(left + diff / 2.0) if not za else left + diff / 2.0, abs(diff)


def grid_bearing(value: float, kind: str, settings: dict) -> float:
    """A bearing as grid bearing: true = grid + convergence; magnetic = true - declination."""
    conv = float(settings.get("convergence", 0.0) or 0.0)
    decl = float(settings.get("declination", 0.0) or 0.0)
    if kind == "true":
        return _norm(value - conv)
    if kind == "magnetic":
        return _norm(value + decl - conv)
    return _norm(value)


def _leg(pid, i, row, at, brg, za, coords, results, chains, legs, settings):
    """Add the leg at -> foresight (grid bearing ``brg``, zenith angle ``za``). Returns the new
    station's coordinates, or None when the foresight closes onto a station with coordinates
    (the misclosure is then recorded on the pit)."""
    hd = row.fs_sd * math.sin(math.radians(za))
    dz = row.fs_sd * math.cos(math.radians(za)) + (row.hi or 0.0) - (row.fs_ht or 0.0)
    new = (at[0] + hd * math.sin(math.radians(brg)), at[1] + hd * math.cos(math.radians(brg)), at[2] + dz)
    closing = row.fs in coords
    leg = Leg(pid, i, row.at, row.fs, brg, hd, dz, closing)
    legs.append(leg)
    chains[pid].append(leg)
    results[pid].length += hd
    if not closing:
        return new
    have = coords[row.fs]
    d = (new[0] - have[0], new[1] - have[1], new[2] - have[2])
    lin = math.hypot(d[0], d[1])
    ratio = results[pid].length / lin if lin > 1e-9 else math.inf
    results[pid].closure = (*d, lin, ratio, ratio >= float(settings["closure_ratio"]), row.fs)
    return None


def compute(pits: list, stations: list, setups: dict, settings: dict | None = None) -> Network:
    """The coordinates of every station from the known stations and the setups.

    ``setups``: pit id -> [SetupRow] in order. Pits are worked parents first; a setup is
    computed when its station and its backsight have coordinates (passes repeat until nothing
    more can be done, so branches and loops in any order work). A foresight onto a station that
    has coordinates already closes the pit's traverse: the misclosure and ratio are reported and,
    when the pit's ``adjust`` is on and the ratio passes, spread over its legs (Bowditch)."""
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    coords: dict = {}
    status: dict = {}
    for s in stations:
        if s.kind == "known" and None not in (s.x, s.y, s.z):
            coords[s.name] = (float(s.x), float(s.y), float(s.z))
            status[s.name] = "known"
    by_id = {p.id: p for p in pits}
    depth = {}

    def level(p):
        seen = set()
        d = 0
        while p.parent_id is not None and p.parent_id in by_id and p.id not in seen:
            seen.add(p.id)
            p = by_id[p.parent_id]
            d += 1
        return d

    for p in pits:
        depth[p.id] = level(p)
    order = sorted(pits, key=lambda p: (depth[p.id], p.id))
    results = {p.id: PitResult(p.id) for p in pits}
    messages: list = []
    legs: list = []
    done: set = set()
    computed_by: dict = {}                  # station -> pit id that computed it
    chains: dict = {p.id: [] for p in pits}  # pit id -> legs in order

    # method "coords": the typed coordinates are fixed points, like known stations - set them
    # first, so traverses can close on them and continue from them in any order
    entered: dict = {}                      # (pit id, setup index) -> the station it gave
    for p in order:
        for i, row in enumerate(setups.get(p.id, [])):
            if row.method != "coords" and getattr(row, "ui_method", None) != "coords":
                continue
            xyz = (row.fs_x, row.fs_y, row.fs_z)
            if not row.fs or None in xyz:
                messages.append((p.id, i, tr("foresight {fs}: X, Y and Z are needed", fs=row.fs or '?')))
                continue
            if row.fs in coords:
                if status.get(row.fs) == "known":
                    msg = tr("station {fs} is a known station; these coordinates are not used", fs=row.fs)
                else:
                    msg = tr("station {fs} is given coordinates in another setup; these coordinates are not used",
                             fs=row.fs)
                messages.append((p.id, i, msg))
                continue
            coords[row.fs] = tuple(float(v) for v in xyz)
            status[row.fs] = "entered"
            computed_by[row.fs] = p.id
            entered[(p.id, i)] = row.fs

    progress = True
    while progress:
        progress = False
        for p in order:
            for i, row in enumerate(setups.get(p.id, [])):
                key = (p.id, i)
                if key in done or not row.at:
                    continue
                if row.at not in coords:
                    continue
                at = coords[row.at]
                if row.method == "coords" or getattr(row, "ui_method", None) == "coords":
                    done.add(key)
                    progress = True
                    if key in entered:              # the leg at -> the typed station: drawn, not adjusted
                        b = coords[row.fs]
                        dx, dy, dz = b[0] - at[0], b[1] - at[1], b[2] - at[2]
                        lg = Leg(p.id, i, row.at, row.fs, bearing_of(dx, dy), math.hypot(dx, dy), dz, entered=True)
                        legs.append(lg)
                        chains[p.id].clear()         # a fixed point: a later closure checks only the
                        results[p.id].length = 0.0   # legs measured after it
                    continue
                if row.fs_bearing is not None:      # method "bearing": the leg's direction measured directly
                    done.add(key)
                    progress = True
                    if not row.fs or row.fs_za is None or not row.fs_sd:
                        messages.append((p.id, i, tr("foresight {fs}: bearing, ZA and SD are needed",
                                                     fs=row.fs or '?')))
                        continue
                    brg = grid_bearing(row.fs_bearing, row.fs_bearing_type or "magnetic", settings)
                    self_leg = _leg(p.id, i, row, at, brg, row.fs_za, coords, results, chains, legs, settings)
                    if self_leg is not None:
                        coords[row.fs], status[row.fs], computed_by[row.fs] = self_leg, "open", p.id
                    continue
                # orientation: the grid bearing of the backsight
                if row.bs_station:
                    if row.bs_station not in coords:
                        continue
                    b = coords[row.bs_station]
                    if (b[0], b[1]) == (at[0], at[1]):
                        messages.append((p.id, i, tr("backsight {bs} is at the same place as {at}",
                                                     bs=row.bs_station, at=row.at)))
                        done.add(key)
                        continue
                    bs_brg = bearing_of(b[0] - at[0], b[1] - at[1])
                elif row.bs_bearing is not None:
                    bs_brg = grid_bearing(row.bs_bearing, row.bs_bearing_type or "grid", settings)
                else:
                    messages.append((p.id, i, tr("setup at {at}: no backsight station or bearing", at=row.at)))
                    done.add(key)
                    continue
                done.add(key)
                progress = True
                bs_ha, bs_diff = _mean_face(row.bs_ha, row.bs_ha_r)
                if bs_ha is None:
                    bs_ha = bs_brg                   # the circle was set to the backsight bearing
                if bs_diff is not None and bs_diff > FACE_TOLERANCE:
                    messages.append((p.id, i, tr("backsight faces differ by {s:.0f}″", s=bs_diff * 3600)))
                # backsight check against the stored leg
                if row.bs_station and row.bs_sd and row.bs_za is not None:
                    za, _d = _mean_face(row.bs_za, row.bs_za_r, za=True)
                    hd = row.bs_sd * math.sin(math.radians(za))
                    dz = row.bs_sd * math.cos(math.radians(za)) + (row.hi or 0.0) - (row.bs_ht or 0.0)
                    b = coords[row.bs_station]
                    hd_c = math.hypot(b[0] - at[0], b[1] - at[1])
                    dz_c = b[2] - at[2]
                    if abs(hd - hd_c) > BS_CHECK_TOLERANCE or abs(dz - dz_c) > BS_CHECK_TOLERANCE:
                        messages.append((p.id, i, tr("backsight check at {at}: distance to {bs} {hd:.3f} m "
                                                     "(stored {hd_c:.3f}), height {dz:+.3f} m (stored {dz_c:+.3f})",
                                                     at=row.at, bs=row.bs_station, hd=hd, hd_c=hd_c, dz=dz,
                                                     dz_c=dz_c)))
                # foresight
                if not row.fs or row.fs_ha is None or row.fs_za is None or not row.fs_sd:
                    if row.fs:
                        messages.append((p.id, i, tr("foresight {fs}: HA, ZA and SD are needed", fs=row.fs)))
                    continue
                fs_ha, fs_diff = _mean_face(row.fs_ha, row.fs_ha_r)
                za, za_diff = _mean_face(row.fs_za, row.fs_za_r, za=True)
                for diff, what in ((fs_diff, N_("foresight HA faces differ by {s:.0f}″")),
                                   (za_diff, N_("foresight ZA faces differ by {s:.0f}″"))):
                    if diff is not None and diff > FACE_TOLERANCE:
                        messages.append((p.id, i, tr(what, s=diff * 3600)))
                brg = _norm(bs_brg + (fs_ha - bs_ha))
                new = _leg(p.id, i, row, at, brg, za, coords, results, chains, legs, settings)
                if new is not None:
                    coords[row.fs], status[row.fs], computed_by[row.fs] = new, "open", p.id

    # closures: Bowditch over the pit's legs from its start to the closing leg
    for p in order:
        res = results[p.id]
        if res.closure is None:
            continue
        de, dn, dz, lin, ratio, passed, onto = res.closure
        own = [lg for lg in chains[p.id] if not lg.closing]
        total = res.length
        if passed and p.adjust and total > 0:
            run = 0.0
            for lg in own:
                run += lg.hd
                f = run / total
                x, y, z = coords[lg.to]
                coords[lg.to] = (x - de * f, y - dn * f, z - dz * f)
            res.adjusted = True
        for lg in own:
            status[lg.to] = "closed" if passed else "failed"
        if not passed:
            res.messages.append(tr("closure 1:{ratio:,.0f} is worse than 1:{limit:,.0f}", ratio=ratio,
                                     limit=float(settings['closure_ratio'])))

    # stations nothing could reach
    for s in stations:
        if s.name not in coords and s.active:
            status[s.name] = "unresolved"
    for p in order:
        for i, row in enumerate(setups.get(p.id, [])):
            if (p.id, i) not in done and row.at:
                if row.at not in coords:
                    msg = tr("setup at {at} not computed: station {at} has no coordinates yet", at=row.at)
                else:
                    msg = tr("setup at {at} not computed: backsight {bs} has no coordinates yet", at=row.at,
                             bs=row.bs_station)
                messages.append((p.id, i, msg))
    return Network(coords, status, legs, results, messages)


# --- storage ---------------------------------------------------------------------------------------------
def _num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _txt(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


class SurveyStore:
    """Read and change the survey data of a project database (a database ``Target``)."""

    def __init__(self, target):
        self.target = target

    # --- tables ------------------------------------------------------------------------------
    def prepare(self, db=None) -> list[str]:
        """Make the missing survey tables / fields and indexes (only adds). Returns what was done."""
        if db is None:
            with self.target.connect() as con:
                return self.prepare(con)
        notes = []
        ddl = _ddl(db.is_mysql)
        for name in TABLES:
            stored = db.table_name(name)
            if stored is None:
                db.execute(ddl[name])
                db.con.commit()
                notes.append(f"made table {name}")
                continue
            have = {c.lower() for c in db.columns(stored)}
            for col, kind in _FIELDS[name]:
                if col not in have:
                    db.execute(f"ALTER TABLE {db.q(stored)} ADD COLUMN {db.q(col)} {kind}")
                    db.con.commit()
                    notes.append(f"added field {col} to {name}")
        for table, idx, cols in _INDEXES:
            db.ensure_index(table, idx, cols)
        db.con.commit()
        return notes

    # --- settings --------------------------------------------------------------------------
    def settings(self) -> dict:
        out = dict(DEFAULT_SETTINGS)
        with self.target.connect(create=False) as db:
            for k, v in db.query(f"SELECT setting, value FROM {SETTINGS}"):
                n = _num(v)
                if n is not None:
                    out[str(k)] = n
        return out

    def set_settings(self, values: dict) -> None:
        with self.target.connect() as db, db.transaction():
            for k, v in values.items():
                db.execute(f"DELETE FROM {SETTINGS} WHERE setting = ?", (k,))
                db.execute(f"INSERT INTO {SETTINGS} (setting, value) VALUES (?, ?)", (k, str(float(v))))

    # --- pits ---------------------------------------------------------------------------------
    def pits(self) -> list[Pit]:
        with self.target.connect(create=False) as db:
            rows = db.query(f"SELECT id, name, parent_id, start_station, adjust, note, code, profile_width, "
                            f"profile_height, station_level FROM {PITS} ORDER BY id")
        return [Pit(int(i), str(n or ""), int(p) if p is not None else None, _txt(s), bool(a if a is not None else 1),
                    str(nt or ""), str(c or f"P{i}"), _num(w) or 5.0, _num(h) or 5.0,
                    lv if lv in STATION_LEVELS else "back")
                for i, n, p, s, a, nt, c, w, h, lv in rows]

    def add_pit(self, name: str, parent_id: int | None = None, start_station: str | None = None,
                known: tuple | None = None, code: str | None = None) -> int:
        """A new main pit (``known`` = entrance X, Y, Z) or branch (``parent_id``, starting at a
        station of the parent). Returns its id."""
        import datetime as _dt

        name = _txt(name)
        if not name:
            raise ValueError(tr("Type a name for the pit"))
        start = _txt(start_station)
        code = _txt(code) or next_code(self.pits(), parent_id)
        with self.target.connect() as db:
            if db.scalar(f"SELECT COUNT(*) FROM {PITS} WHERE name = ?", (name,)):
                raise ValueError(tr("There is already a pit named {name}", name=name))
            pid = int(db.scalar(f"SELECT MAX(id) FROM {PITS}", default=0) or 0) + 1
            with db.transaction():
                db.execute(f"INSERT INTO {PITS} (id, name, code, parent_id, start_station, adjust, note, created_at) "
                           "VALUES (?, ?, ?, ?, ?, 1, NULL, ?)",
                           (pid, name, code, parent_id, start, _dt.datetime.now().isoformat(timespec="seconds")))
                if known is not None and start:
                    if db.scalar(f"SELECT COUNT(*) FROM {STATIONS} WHERE name = ?", (start,)):
                        raise ValueError(tr("There is already a station named {name}", name=start))
                    x, y, z = (float(v) for v in known)
                    db.execute(f"INSERT INTO {STATIONS} (name, pit_id, kind, x, y, z, status, active) "
                               "VALUES (?, ?, 'known', ?, ?, ?, 'known', 1)", (start, pid, x, y, z))
        return pid

    def update_pit(self, pid: int, name: str | None = None, adjust: bool | None = None, note: str | None = None,
                   code: str | None = None, profile: tuple | None = None) -> None:
        """Change a pit; ``profile`` = (width, height, station level) of its rectangular profile."""
        with self.target.connect() as db, db.transaction():
            if profile is not None:
                w, h, level = profile
                if not (_num(w) and _num(h)) or float(w) <= 0 or float(h) <= 0:
                    raise ValueError(tr("The profile width and height must be more than 0"))
                db.execute(f"UPDATE {PITS} SET profile_width = ?, profile_height = ?, station_level = ? "
                           "WHERE id = ?", (float(w), float(h), level if level in STATION_LEVELS else "back", pid))
            if code is not None and _txt(code):
                db.execute(f"UPDATE {PITS} SET code = ? WHERE id = ?", (_txt(code), pid))
            if name is not None:
                name = _txt(name)
                if not name:
                    raise ValueError(tr("Type a name for the pit"))
                if db.scalar(f"SELECT COUNT(*) FROM {PITS} WHERE name = ? AND id <> ?", (name, pid)):
                    raise ValueError(tr("There is already a pit named {name}", name=name))
                db.execute(f"UPDATE {PITS} SET name = ? WHERE id = ?", (name, pid))
            if adjust is not None:
                db.execute(f"UPDATE {PITS} SET adjust = ? WHERE id = ?", (1 if adjust else 0, pid))
            if note is not None:
                db.execute(f"UPDATE {PITS} SET note = ? WHERE id = ?", (_txt(note), pid))

    def delete_pit(self, pid: int) -> list[int]:
        """Delete a pit, its branches (all levels), their setups, shots, computed stations and
        mapped geology. Returns the ids deleted."""
        from .pit_geology_store import PitGeologyStore

        pits = self.pits()
        ids, todo = [], [pid]
        while todo:
            p = todo.pop()
            ids.append(p)
            todo += [q.id for q in pits if q.parent_id == p]
        with self.target.connect() as db, db.transaction():
            for p in ids:
                setup_ids = [r[0] for r in db.query(f"SELECT id FROM {SETUPS} WHERE pit_id = ?", (p,))]
                for s in setup_ids:
                    db.execute(f"DELETE FROM {SHOTS} WHERE setup_id = ?", (s,))
                db.execute(f"DELETE FROM {SETUPS} WHERE pit_id = ?", (p,))
                db.execute(f"DELETE FROM {STATIONS} WHERE pit_id = ?", (p,))
                db.execute(f"DELETE FROM {PITS} WHERE id = ?", (p,))
            PitGeologyStore.delete_pits(db, ids)
        return ids

    # --- stations -------------------------------------------------------------------------
    def stations(self) -> list[Station]:
        with self.target.connect(create=False) as db:
            rows = db.query(f"SELECT name, pit_id, kind, x, y, z, status, active, note FROM {STATIONS} ORDER BY name")
        return [Station(str(n), int(p) if p is not None else None, str(k or "station"), _num(x), _num(y), _num(z),
                        str(st or ""), bool(a if a is not None else 1), str(nt or ""))
                for n, p, k, x, y, z, st, a, nt in rows]

    def set_known(self, name: str, x, y, z, pit_id: int | None = None) -> None:
        """Enter (or change) a known station's coordinates."""
        name = _txt(name)
        if not name:
            raise ValueError(tr("Type a station name"))
        vals = [_num(v) for v in (x, y, z)]
        if None in vals:
            raise ValueError(tr("X, Y and Z must be numbers"))
        with self.target.connect() as db, db.transaction():
            if db.scalar(f"SELECT COUNT(*) FROM {STATIONS} WHERE name = ?", (name,)):
                db.execute(f"UPDATE {STATIONS} SET kind = 'known', x = ?, y = ?, z = ?, status = 'known' "
                           "WHERE name = ?", (*vals, name))
            else:
                db.execute(f"INSERT INTO {STATIONS} (name, pit_id, kind, x, y, z, status, active) "
                           "VALUES (?, ?, 'known', ?, ?, ?, 'known', 1)", (name, pit_id, *vals))

    # --- setups --------------------------------------------------------------------------------
    def setups(self, pit_id: int | None = None) -> dict[int, list[SetupRow]]:
        """Pit id -> its setups in order, as field book rows."""
        where, params = ("WHERE pit_id = ?", (pit_id,)) if pit_id is not None else ("", ())
        with self.target.connect(create=False) as db:
            rows = db.query(f"SELECT id, pit_id, seq, survey_date, surveyor, instrument, at_station, hi, bs_station, "
                            f"bs_bearing, bs_bearing_type, note, fs_bearing, fs_bearing_type, fs_x, fs_y, fs_z FROM {SETUPS} "
                            f"{where} "
                            f"ORDER BY pit_id, seq", params)
            ids = [r[0] for r in rows]
            shots: dict = {}
            if ids:
                marks = ", ".join("?" * len(ids)) if len(ids) < 900 else None
                q = (f"SELECT setup_id, role, face, target, ha, za, sd, ht FROM {SHOTS} WHERE setup_id IN ({marks})"
                     if marks else f"SELECT setup_id, role, face, target, ha, za, sd, ht FROM {SHOTS}")
                for sid, role, face, target, ha, za, sd, ht in db.query(q, ids if marks else ()):
                    shots.setdefault(int(sid), []).append((role, face, target, _num(ha), _num(za), _num(sd), _num(ht)))
        out: dict[int, list[SetupRow]] = {}
        for sid, pid, _seq, date, surveyor, instr, at, hi, bs, brg, btype, note, fbrg, ftype, fx, fy, fz in rows:
            r = SetupRow(str(at), _num(hi), _txt(bs), _num(brg), str(btype or "grid"), date=str(date or ""),
                         surveyor=str(surveyor or ""), instrument=str(instr or ""), note=str(note or ""), id=int(sid),
                         fs_bearing=_num(fbrg), fs_bearing_type=str(ftype or "magnetic"),
                         fs_x=_num(fx), fs_y=_num(fy), fs_z=_num(fz))
            for role, face, target, ha, za, sd, ht in shots.get(int(sid), []):
                if role == "BS":
                    if face == "R":
                        r.bs_ha_r, r.bs_za_r = ha, za
                    else:
                        r.bs_ha, r.bs_za, r.bs_sd, r.bs_ht = ha, za, sd, ht
                elif role == "FS":
                    if face == "R":
                        r.fs_ha_r, r.fs_za_r = ha, za
                    else:
                        r.fs, r.fs_ha, r.fs_za, r.fs_sd, r.fs_ht = _txt(target), ha, za, sd, ht
            out.setdefault(int(pid), []).append(r)
        return out

    def save_setups(self, pit_id: int, rows: list[SetupRow]) -> None:
        """Replace a pit's setups (and their shots) with ``rows``, in this order; new foresight
        stations are added to the pit. Then the whole network is computed and stored."""
        with self.target.connect() as db, db.transaction():
            self._write_setups(db, pit_id, rows)
        self.recompute()

    @staticmethod
    def _write_setups(db, pit_id: int, rows: list[SetupRow]) -> None:
        """``save_setups`` inside the caller's transaction (no computing)."""
        old = [r[0] for r in db.query(f"SELECT id FROM {SETUPS} WHERE pit_id = ?", (pit_id,))]
        for s in old:
            db.execute(f"DELETE FROM {SHOTS} WHERE setup_id = ?", (s,))
        db.execute(f"DELETE FROM {SETUPS} WHERE pit_id = ?", (pit_id,))
        sid = int(db.scalar(f"SELECT MAX(id) FROM {SETUPS}", default=0) or 0)
        shot_id = int(db.scalar(f"SELECT MAX(id) FROM {SHOTS}", default=0) or 0)
        known = {r[0] for r in db.query(f"SELECT name FROM {STATIONS}")}
        shots = []
        for seq, r in enumerate(rows, start=1):
            sid += 1
            coords_mode = r.method == "coords"
            bearing_mode = coords_mode or r.fs_bearing is not None      # no backsight
            db.execute(f"INSERT INTO {SETUPS} (id, pit_id, seq, survey_date, surveyor, instrument, at_station, hi, "
                       "bs_station, bs_bearing, bs_bearing_type, note, fs_bearing, fs_bearing_type, fs_x, fs_y, "
                       "fs_z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (sid, pit_id, seq, _txt(r.date), _txt(r.surveyor), _txt(r.instrument), _txt(r.at),
                        _num(r.hi), None if bearing_mode else _txt(r.bs_station),
                        None if bearing_mode else _num(r.bs_bearing), r.bs_bearing_type or "grid", _txt(r.note),
                        None if coords_mode else _num(r.fs_bearing), r.fs_bearing_type or "magnetic",
                        *((_num(r.fs_x), _num(r.fs_y), _num(r.fs_z)) if coords_mode else (None, None, None))))
            if not coords_mode and any(v is not None for v in (r.bs_ha, r.bs_za, r.bs_sd)):
                shot_id += 1
                shots.append((shot_id, sid, "BS", "L", _txt(r.bs_station), r.bs_ha, r.bs_za, r.bs_sd, r.bs_ht))
            if not coords_mode and (r.bs_ha_r is not None or r.bs_za_r is not None):
                shot_id += 1
                shots.append((shot_id, sid, "BS", "R", _txt(r.bs_station), r.bs_ha_r, r.bs_za_r, None, None))
            if r.fs:
                shot_id += 1
                readings = (None,) * 4 if coords_mode else (r.fs_ha, r.fs_za, r.fs_sd, r.fs_ht)
                shots.append((shot_id, sid, "FS", "L", _txt(r.fs), *readings))   # coords: the name only
                if not coords_mode and (r.fs_ha_r is not None or r.fs_za_r is not None):
                    shot_id += 1
                    shots.append((shot_id, sid, "FS", "R", _txt(r.fs), r.fs_ha_r, r.fs_za_r, None, None))
                if _txt(r.fs) not in known:
                    db.execute(f"INSERT INTO {STATIONS} (name, pit_id, kind, status, active) "
                               "VALUES (?, ?, 'station', 'open', 1)", (_txt(r.fs), pit_id))
                    known.add(_txt(r.fs))
        db.executemany(f"INSERT INTO {SHOTS} (id, setup_id, role, face, target, ha, za, sd, ht) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", shots)

    # --- designed pits -------------------------------------------------------------------------
    def designs(self) -> list[DesignPit]:
        """The designed pits (none when the tables are not there yet)."""
        import json

        with self.target.connect(create=False) as db:
            if db.table_name(DESIGNS) is None or db.table_name(DESIGN_POINTS) is None:
                return []
            rows = db.query(f"SELECT id, name, start_pit_id, start_name, target_name, profile_width, profile_height, "
                            f"station_level, gradient, steepest, length, climb, params, note, created_at FROM {DESIGNS} "
                            f"ORDER BY id")
            pts: dict = {}
            for did, x, y, z in db.query(f"SELECT design_id, x, y, z FROM {DESIGN_POINTS} ORDER BY design_id, seq"):
                xyz = (_num(x), _num(y), _num(z))
                if None not in xyz:
                    pts.setdefault(int(did), []).append(xyz)
        out = []
        for i, n, sp, sn, tn, w, h, lv, g, st, ln, cl, prm, nt, at in rows:
            try:
                params = json.loads(prm) if prm else {}
            except (TypeError, ValueError):
                params = {}
            out.append(DesignPit(int(i), str(n or ""), pts.get(int(i), []), _num(w) or 5.0, _num(h) or 5.0,
                                 lv if lv in STATION_LEVELS else "middle", int(sp) if sp is not None else None,
                                 str(sn or ""), str(tn or ""), _num(g), _num(st), _num(ln) or 0.0, _num(cl) or 0.0,
                                 params if isinstance(params, dict) else {}, str(nt or ""), str(at or "")))
        return out

    def add_design(self, d: DesignPit) -> int:
        """Store a designed pit (its id is given here). Returns the id."""
        import datetime as _dt
        import json

        name = _txt(d.name)
        if not name:
            raise ValueError(tr("Type a name for the designed pit"))
        if len(d.points) < 2:
            raise ValueError(tr("A designed pit needs at least 2 points"))
        with self.target.connect() as db:
            if db.table_name(DESIGNS) is None or db.table_name(DESIGN_POINTS) is None:
                self.prepare(db)
            if db.scalar(f"SELECT COUNT(*) FROM {DESIGNS} WHERE name = ?", (name,)):
                raise ValueError(tr("There is already a designed pit named {name}", name=name))
            did = int(db.scalar(f"SELECT MAX(id) FROM {DESIGNS}", default=0) or 0) + 1
            pid = int(db.scalar(f"SELECT MAX(id) FROM {DESIGN_POINTS}", default=0) or 0)
            with db.transaction():
                db.execute(f"INSERT INTO {DESIGNS} (id, name, start_pit_id, start_name, target_name, profile_width, "
                           "profile_height, station_level, gradient, steepest, length, climb, params, note, created_at) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (did, name, d.start_pit_id, _txt(d.start_name), _txt(d.target_name), float(d.profile_width),
                            float(d.profile_height), d.station_level if d.station_level in STATION_LEVELS else "middle",
                            _num(d.gradient), _num(d.steepest), float(d.length), float(d.climb),
                            json.dumps(d.params or {}), _txt(d.note),
                            d.created_at or _dt.datetime.now().isoformat(timespec="seconds")))
                db.executemany(f"INSERT INTO {DESIGN_POINTS} (id, design_id, seq, x, y, z) VALUES (?, ?, ?, ?, ?, ?)",
                               [(pid + k, did, k, float(p[0]), float(p[1]), float(p[2]))
                                for k, p in enumerate(d.points, start=1)])
        d.id = did
        return did

    def update_design(self, did: int, name: str, note: str = "", profile: tuple | None = None,
                      points: list | None = None) -> None:
        """Change a designed pit: name, note, ``profile`` = (width, height, station level) and its centre line
        ``points`` [(x, y, z)] (length, height change and steepest leg follow)."""
        name = _txt(name)
        if not name:
            raise ValueError(tr("Type a name for the designed pit"))
        if points is not None:
            points = [tuple(float(v) for v in p) for p in points]
            if len(points) < 2:
                raise ValueError(tr("A designed pit needs at least 2 points"))
        with self.target.connect() as db, db.transaction():
            if db.scalar(f"SELECT COUNT(*) FROM {DESIGNS} WHERE name = ? AND id <> ?", (name, did)):
                raise ValueError(tr("There is already a designed pit named {name}", name=name))
            db.execute(f"UPDATE {DESIGNS} SET name = ?, note = ? WHERE id = ?", (name, _txt(note), did))
            if profile is not None:
                w, h, level = profile
                if not (_num(w) and _num(h)) or float(w) <= 0 or float(h) <= 0:
                    raise ValueError(tr("The profile width and height must be more than 0"))
                db.execute(f"UPDATE {DESIGNS} SET profile_width = ?, profile_height = ?, station_level = ? WHERE id = ?",
                           (float(w), float(h), level if level in STATION_LEVELS else "middle", did))
            if points is not None:
                length, climb, steepest = design_measures(points)
                db.execute(f"UPDATE {DESIGNS} SET length = ?, climb = ?, steepest = ? WHERE id = ?",
                           (length, climb, steepest, did))
                db.execute(f"DELETE FROM {DESIGN_POINTS} WHERE design_id = ?", (did,))
                pid = int(db.scalar(f"SELECT MAX(id) FROM {DESIGN_POINTS}", default=0) or 0)
                db.executemany(f"INSERT INTO {DESIGN_POINTS} (id, design_id, seq, x, y, z) VALUES (?, ?, ?, ?, ?, ?)",
                               [(pid + k, did, k, *p) for k, p in enumerate(points, start=1)])

    def rename_design(self, did: int, name: str) -> None:
        name = _txt(name)
        if not name:
            raise ValueError(tr("Type a name for the designed pit"))
        with self.target.connect() as db, db.transaction():
            if db.scalar(f"SELECT COUNT(*) FROM {DESIGNS} WHERE name = ? AND id <> ?", (name, did)):
                raise ValueError(tr("There is already a designed pit named {name}", name=name))
            db.execute(f"UPDATE {DESIGNS} SET name = ? WHERE id = ?", (name, did))

    def delete_design(self, did: int) -> None:
        with self.target.connect() as db, db.transaction():
            db.execute(f"DELETE FROM {DESIGN_POINTS} WHERE design_id = ?", (did,))
            db.execute(f"DELETE FROM {DESIGNS} WHERE id = ?", (did,))

    def next_design_name(self, base: str = "D") -> str:
        """D1, D2, ... - the first not in use."""
        used = {d.name for d in self.designs()}
        n = 1
        while f"{base}{n}" in used:
            n += 1
        return f"{base}{n}"

    # --- the network ---------------------------------------------------------------------------
    def network(self) -> Network:
        return compute(self.pits(), self.stations(), self.setups(), self.settings())

    def recompute(self) -> Network:
        """Compute every station and store the coordinates and status (known stations untouched)."""
        import datetime as _dt

        net = self.network()
        now = _dt.datetime.now().isoformat(timespec="seconds")
        with self.target.connect() as db, db.transaction():
            for s in self.stations():
                if s.kind == "known":
                    continue
                xyz = net.coords.get(s.name)
                st = net.status.get(s.name, "unresolved")
                if xyz is None:
                    db.execute(f"UPDATE {STATIONS} SET x = NULL, y = NULL, z = NULL, status = ?, updated_at = ? "
                               "WHERE name = ?", (st, now, s.name))
                else:
                    db.execute(f"UPDATE {STATIONS} SET x = ?, y = ?, z = ?, status = ?, updated_at = ? WHERE name = ?",
                               (*xyz, st, now, s.name))
        return net
