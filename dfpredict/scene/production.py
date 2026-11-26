"""Reserve management: mined shapes and block model depletion.

A *mined shape* is an outline drawn in a 2D view (plan XY, or the XZ / YZ sections)
swept across that view from ``from_`` to ``to`` (e.g. elevation 180 -> 186 m for a
plan outline). Every shape has a date. The shapes are the permanent record: they
are saved in the prod_* tables of the project database (geodatabase.sqlite in the
workspace folder, or MySQL); the block model files are never changed.

Depletion is derived from the shapes. Each *reserve block* (a block whose centre is
inside the reserve solid, e.g. ore1) is represented by n x n x n sample points; a
point is mined by the first shape (by date) that contains it. So a block can be
mined bit by bit, overlapping shapes are never counted twice, and

    mined fraction of a block (as of a date) = share of its points mined by then
    produced tonnes = sum(block tonnes x mined fraction), remaining = total - produced

with tonnes = block volume x density and the grade tonnage-weighted.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..i18n import N_, tr

LEGACY_DB = "production.sqlite"                                 # first SQLite version: imported once
LEGACY_FILE = "production.json"                                 # JSON version: imported once
PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}   # (u axis, v axis, depth axis)
SOLID_PLANE = "3d"                                              # a mined solid (not drawn in a 2D view)
AXIS_WORD = {0: N_("Easting (X)"), 1: N_("Northing (Y)"), 2: N_("Elevation (Z)")}   # shown with tr()
ORE_LEVEL_SHARE = 0.05                                          # ore range: levels with >= 5 % of the richest
STATUS_CATEGORIES = ["Unmined", "Partly mined", "Mined"]
STATUS_COLORS = [(0.62, 0.65, 0.70), (1.0, 0.70, 0.15), (0.85, 0.22, 0.20)]


def day_of(date: str) -> int:
    return _dt.date.fromisoformat(date).toordinal()


def today() -> str:
    return _dt.date.today().isoformat()


# --- one mined shape ------------------------------------------------------------------------
@dataclass
class MinedShape:
    id: int
    date: str                      # "YYYY-MM-DD"
    name: str
    plane: str                     # "xy" / "xz" / "yz": the view the outline was drawn in
    points: list                   # [[u, v], ...] world coordinates in that view
    from_: float                   # start along the depth axis (e.g. floor elevation)
    to: float                      # end along the depth axis (e.g. roof elevation)
    note: str = ""
    # A mined shape is either an outline swept From - To (kind "prism", drawn in a 2D view) or
    # a closed solid loaded from a Surpac .dtm (kind "solid"; From / To are its lowest / highest Z).
    kind: str = "prism"
    vertices: object = field(default=None, repr=False, compare=False)    # solid: (N, 3) world coordinates
    triangles: object = field(default=None, repr=False, compare=False)   # solid: (M, 3) vertex indices
    source: str = ""                                                      # solid: the file it came from

    @property
    def is_solid(self) -> bool:
        return self.kind == "solid"

    @classmethod
    def from_solid(cls, sid: int, date: str, name: str, vertices, triangles, source: str = "",
                   note: str = "") -> "MinedShape":
        """A mined shape from a closed solid (only the vertices its triangles use are kept)."""
        v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        used, t = np.unique(t, return_inverse=True)
        v, t = v[used], t.reshape(-1, 3)
        return cls(sid, date, name, SOLID_PLANE, [], float(v[:, 2].min()), float(v[:, 2].max()), note,
                   kind="solid", vertices=v, triangles=t, source=source)

    @property
    def axes(self) -> tuple[int, int, int]:
        return PLANES.get(self.plane, (0, 1, 2))

    @property
    def lo(self) -> float:
        return min(self.from_, self.to)

    @property
    def hi(self) -> float:
        return max(self.from_, self.to)

    @property
    def depth(self) -> float:
        return self.hi - self.lo

    @property
    def day(self) -> int:
        return day_of(self.date)

    def outline(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float64).reshape(-1, 2)

    def volume(self) -> float:
        """m³: of the solid (closed triangles), or outline area x thickness."""
        if not self.is_solid:
            return self.area() * self.depth
        p = self.vertices[self.triangles]
        return float(abs(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) / 6.0)

    def area(self) -> float:
        if self.is_solid:
            return 0.0
        p = self.outline()
        x, y = p[:, 0], p[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World bounding box (lo, hi)."""
        if self.is_solid:
            return self.vertices.min(axis=0), self.vertices.max(axis=0)
        p = self.outline()
        iu, iv, ia = self.axes
        lo, hi = np.zeros(3), np.zeros(3)
        lo[iu], hi[iu] = p[:, 0].min(), p[:, 0].max()
        lo[iv], hi[iv] = p[:, 1].min(), p[:, 1].max()
        lo[ia], hi[ia] = self.lo, self.hi
        return lo, hi

    def in_outline(self, pts: np.ndarray) -> np.ndarray:
        """Points (N, 3) whose in-plane position is inside the outline (depth ignored)."""
        iu, iv, _ = self.axes
        return point_in_polygon(pts[:, iu], pts[:, iv], self.outline())

    def contains(self, pts: np.ndarray) -> np.ndarray:
        if self.is_solid:
            from .solid import points_inside

            return points_inside(pts, self.vertices, self.triangles)
        _, _, ia = self.axes
        a = pts[:, ia]
        return (a >= self.lo) & (a <= self.hi) & self.in_outline(pts)

    def to_json(self) -> dict:
        d = asdict(self)
        d["from"] = d.pop("from_")
        return d

    @classmethod
    def from_json(cls, d: dict) -> "MinedShape":
        return cls(id=int(d["id"]), date=str(d["date"]), name=str(d.get("name", "")),
                   plane=str(d.get("plane", "xy")), points=[[float(a), float(b)] for a, b in d["points"]],
                   from_=float(d["from"]), to=float(d["to"]), note=str(d.get("note", "")))


def point_in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test of points (x, y) against a closed polygon (K, 2)."""
    inside = np.zeros(len(x), dtype=bool)
    k = len(poly)
    if k < 3:
        return inside
    xj, yj = poly[-1]
    for xi, yi in poly:
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cond & (x < xc)
        xj, yj = xi, yi
    return inside


# --- the production records (prod_* tables of the project database) ------------------------------
PRODUCTION_FORMAT = "GeoViewer production"
PRODUCTION_VERSION = 3

# table names: the project database (prod_*), and production.sqlite of the first SQLite version
PROD_TABLES = {"settings": "prod_settings", "shapes": "prod_mined_shapes"}
_POINTS_V2 = "prod_mined_shape_points"      # version 2 kept one row per outline corner: upgraded on open
_OLD_TABLES = {"settings": "production_settings", "shapes": "mined_shapes", "points": "mined_shape_points"}
OUTLINE_DECIMALS = 3                        # outline corners are kept to the millimetre


def outline_text(points) -> str:
    """The outline as compact JSON text: [[u, v], [u, v], ...] in the view's world coordinates.

    One text per shape (not one row per corner): a shape is read and written as a single
    row, and the text is readable in any SQL tool (MySQL can query it with its JSON functions)."""
    return json.dumps([[round(float(u), OUTLINE_DECIMALS), round(float(v), OUTLINE_DECIMALS)] for u, v in points],
                      separators=(",", ":"))


def solid_blob(vertices, triangles) -> bytes:
    """A solid as compressed bytes: point / triangle counts, float64 X Y Z, int32 indices."""
    import struct
    import zlib

    v = np.ascontiguousarray(vertices, dtype="<f8").reshape(-1, 3)
    t = np.ascontiguousarray(triangles, dtype="<i4").reshape(-1, 3)
    return zlib.compress(struct.pack("<4sII", b"GVS1", len(v), len(t)) + v.tobytes() + t.tobytes(), 6)


def solid_from_blob(blob) -> tuple[np.ndarray, np.ndarray] | None:
    """(vertices, triangles) from ``solid_blob`` bytes; None for none / unreadable."""
    import struct
    import zlib

    if not blob:
        return None
    try:
        raw = zlib.decompress(bytes(blob))
        tag, nv, nt = struct.unpack_from("<4sII", raw)
        if tag != b"GVS1":
            return None
        off = struct.calcsize("<4sII")
        v = np.frombuffer(raw, dtype="<f8", count=nv * 3, offset=off).reshape(-1, 3).astype(np.float64)
        t = np.frombuffer(raw, dtype="<i4", count=nt * 3, offset=off + nv * 24).reshape(-1, 3).astype(np.int64)
        return v, t
    except (zlib.error, struct.error, ValueError):
        return None


def outline_points(text) -> list:
    """[[u, v], ...] from an outline text ([] for an empty or unreadable one)."""
    try:
        return [[float(u), float(v)] for u, v in json.loads(text or "[]")]
    except (TypeError, ValueError):
        return []


def _schema(mysql: bool) -> list[str]:
    """Portable DDL (SQLite and MySQL): VARCHAR keys, DOUBLE coordinates, ISO dates as text."""
    opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if mysql else ""
    t = PROD_TABLES
    return [
        f"""CREATE TABLE IF NOT EXISTS {t['settings']} (
            setting VARCHAR(100) NOT NULL PRIMARY KEY,
            value TEXT){opts}""",
        f"""CREATE TABLE IF NOT EXISTS {t['shapes']} (
            id INTEGER NOT NULL PRIMARY KEY,
            mined_on VARCHAR(10) NOT NULL,
            name VARCHAR(200) NOT NULL,
            view_plane VARCHAR(2) NOT NULL,
            depth_from DOUBLE NOT NULL,
            depth_to DOUBLE NOT NULL,
            outline {_outline_type(mysql)} NOT NULL,
            shape_kind VARCHAR(10),
            solid {_blob_type(mysql)},
            source_file VARCHAR(255),
            note TEXT,
            created_at VARCHAR(19),
            updated_at VARCHAR(19)){opts}""",
    ]


def _blob_type(mysql: bool) -> str:
    return "LONGBLOB" if mysql else "BLOB"         # a mined solid, compressed (solid_blob)


def _outline_type(mysql: bool) -> str:
    return "MEDIUMTEXT" if mysql else "TEXT"       # MySQL TEXT stops at 64 KB; MEDIUMTEXT holds 16 MB


# Fields each production table must have (added as empty columns when missing).
_PROD_FIELDS = {
    "settings": [("setting", "VARCHAR(100)"), ("value", "TEXT")],
    "shapes": [("id", "INTEGER"), ("mined_on", "VARCHAR(10)"), ("name", "VARCHAR(200)"), ("view_plane", "VARCHAR(2)"),
               ("depth_from", "DOUBLE"), ("depth_to", "DOUBLE"), ("outline", None), ("shape_kind", "VARCHAR(10)"),
               ("solid", "BLOB"), ("source_file", "VARCHAR(255)"), ("note", "TEXT"),
               ("created_at", "VARCHAR(19)"), ("updated_at", "VARCHAR(19)")],
}


def _upgrade(db) -> list[str]:
    """Bring existing production tables up to date - only adding, never dropping:
    missing fields are added as empty columns, and outlines kept one row per corner
    (version 2, prod_mined_shape_points) are copied into the empty outline cells
    (that table is left as it is). Returns what was done."""
    notes = []
    for key, fields in _PROD_FIELDS.items():
        t = PROD_TABLES[key]
        if not db.has_table(t):
            continue
        have = {c.lower() for c in db.columns(t)}
        for name, kind in fields:
            if name not in have:
                kind = (_blob_type(db.is_mysql) if kind == "BLOB" else kind) or _outline_type(db.is_mysql)
                db.execute(f"ALTER TABLE {t} ADD COLUMN {name} {kind}")
                db.con.commit()
                notes.append(tr("added field {name} to {table}", name=name, table=t))
    ts = PROD_TABLES["shapes"]
    if db.has_table(ts) and db.has_table(_POINTS_V2):
        empty = {int(r[0]) for r in db.query(f"SELECT id FROM {ts} WHERE outline IS NULL OR outline = ''")}
        pts: dict[int, list] = {}
        if empty:
            for sid, u, v in db.query(f"SELECT shape_id, u, v FROM {_POINTS_V2} ORDER BY shape_id, seq"):
                if int(sid) in empty:
                    pts.setdefault(int(sid), []).append([float(u), float(v)])
        if pts:
            with db.transaction():
                for sid, corners in pts.items():
                    db.execute(f"UPDATE {ts} SET outline = ? WHERE id = ?", (outline_text(corners), sid))
            notes.append(tr("copied the outlines of {n} mined shapes into {table}.outline", n=len(pts), table=ts))
    if db.has_table(ts):
        db.ensure_index(ts, "idx_prod_mined_shapes_mined_on", ["mined_on"])   # by date: as-of, monthly
    if notes:
        with db.transaction():
            ProductionLog._put_setting(db, "version", str(PRODUCTION_VERSION))
    return notes


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _read_records(db, names: dict) -> tuple[list, dict] | None:
    """(shapes, reserve setup) from the production tables ``names`` of ``db``; None when the
    tables are not there. Raises ValueError for tables that are not GeoViewer's."""
    if not db.has_table(names["settings"]):
        return None
    fmt = db.scalar(f"SELECT value FROM {names['settings']} WHERE setting = ?", ("format",))
    if fmt != PRODUCTION_FORMAT:
        raise ValueError(tr("{table} in {where} does not hold GeoViewer production records",
                            table=names["settings"], where=db.label))
    reserve = {}
    for k, v in db.query(f"SELECT setting, value FROM {names['settings']} WHERE setting LIKE ?", ("reserve.%",)):
        try:
            reserve[k[len("reserve."):]] = json.loads(v)
        except (TypeError, ValueError):
            reserve[k[len("reserve."):]] = v
    pts: dict[int, list] = {}                     # one row per corner (production.sqlite, version 2)
    corners = names.get("points") or _POINTS_V2
    if db.has_table(corners):
        for sid, u, v in db.query(f"SELECT shape_id, u, v FROM {corners} ORDER BY shape_id, seq"):
            pts.setdefault(int(sid), []).append([float(u), float(v)])
    have = {c.lower() for c in db.columns(names["shapes"])}
    fields = ("id", "mined_on", "name", "view_plane", "depth_from", "depth_to", "outline", "note",
              "shape_kind", "solid", "source_file")
    cols = ", ".join(c if c in have else "NULL" for c in fields)   # a missing field reads as empty (the init adds it)
    shapes = []
    for i, d, n, pl, a, b, o, note, kind, blob, src in db.query(f"SELECT {cols} FROM {names['shapes']} ORDER BY id"):
        solid = solid_from_blob(blob) if kind == "solid" else None
        if solid is not None:
            sh = MinedShape.from_solid(int(i), str(d or ""), str(n or ""), *solid, str(src or ""), str(note or ""))
        else:
            sh = MinedShape(int(i), str(d or ""), str(n or ""), str(pl or "xy"),
                            outline_points(o) or pts.get(int(i), []), float(a or 0.0), float(b or 0.0), str(note or ""))
        shapes.append(sh)
    return shapes, reserve


@dataclass
class ProductionLog:
    """The mined shapes and the reserve setup of one workspace folder.

    Kept in the prod_* tables of the project database (geodatabase.sqlite in the folder, or
    the MySQL database chosen in Settings). Every change is written at once in one
    transaction; a shape is one row, its outline a JSON text (see ``outline_text``). With SQLite the
    production tables are copied to geodatabase.sqlite.bak before the first change of a session.
    Records of an earlier version are imported the first time: production.sqlite,
    production.json, or (when switching to MySQL) geodatabase.sqlite of the folder."""

    folder: Path
    shapes: list = field(default_factory=list)
    reserve: dict = field(default_factory=dict)     # block_model, solid, density, density_value, grade, unit
    imported_from: str | None = None                # where the records were imported from
    upgraded: list = field(default_factory=list)    # what reading brought up to date (fields added, ...)
    target: object = None                           # database.Target; None: by the current setting
    _backed_up: bool = field(default=False, repr=False)
    _exists: bool = field(default=False, repr=False)

    @property
    def db(self):
        """The Target the records are kept in."""
        if self.target is None:
            from ..io.database import project_target

            self.target = project_target(self.folder)
        return self.target

    @property
    def where(self) -> str:
        """For messages: the database file, or the MySQL database."""
        return self.db.label

    @property
    def exists(self) -> bool:
        """The production tables have been written (False before the first save)."""
        return self._exists

    # --- reading -------------------------------------------------------------------------------
    @classmethod
    def load(cls, folder, target=None) -> "ProductionLog":
        """The records of ``target`` (default: the project database of ``folder``). Records
        of earlier versions found in ``folder`` are moved in the first time (``folder``
        None: none are looked for)."""
        log = cls(Path(folder) if folder is not None else Path("."), target=target)
        t = log.db
        if t.exists():
            with t.connect(create=False) as db:
                found = _read_records(db, PROD_TABLES)
                if found is not None:
                    try:
                        log.upgraded = _upgrade(db)   # missing fields, outline rows -> JSON text
                    except Exception:  # noqa: BLE001 - e.g. read-only: read as it is, upgrade next time
                        pass
            if found is not None:
                log.shapes, log.reserve = found
                log._exists = True
                return log
        legacy = _legacy_records(log.folder, t) if folder is not None else None
        if legacy is not None:
            log.imported_from, log.shapes, log.reserve = legacy
            log.save()
        return log

    def ensure_tables(self) -> list[str]:
        """Make the production tables when they are not there yet and add missing fields to
        existing ones (never dropping anything). Returns what was done. Call after ``load``
        (which first moves in records of earlier versions)."""
        if self._exists:
            with self.db.connect() as db:
                return _upgrade(db)
        self._open().close()
        self._exists = True
        return [tr("made the production tables")]

    # --- writing -------------------------------------------------------------------------------
    def _open(self):
        """The database with the production tables (made when missing); an SQLite file is
        backed up once per session before its first change."""
        from ..io.database import PROD_PREFIX, backup_sqlite_tables

        t = self.db
        if not t.is_mysql:
            if t.exists() and not self._backed_up:
                try:                                # the prod_* tables only: small whatever the drillholes
                    backup_sqlite_tables(t.path, t.path.with_name(t.path.name + ".bak"), PROD_PREFIX)
                except Exception:  # noqa: BLE001 - a failed backup must not stop the save
                    pass
            t.path.parent.mkdir(parents=True, exist_ok=True)
        self._backed_up = True
        db = t.connect()
        try:
            for ddl in _schema(db.is_mysql):
                db.execute(ddl)
            _upgrade(db)
            with db.transaction():
                self._put_setting(db, "format", PRODUCTION_FORMAT)
                self._put_setting(db, "version", str(PRODUCTION_VERSION))
        except BaseException:
            db.close()
            raise
        return db

    @staticmethod
    def _put_setting(db, key: str, value: str) -> None:
        t = PROD_TABLES["settings"]
        db.execute(f"DELETE FROM {t} WHERE setting = ?", (key,))
        db.execute(f"INSERT INTO {t} (setting, value) VALUES (?, ?)", (key, value))

    def _write_setup(self, db) -> None:
        db.execute(f"DELETE FROM {PROD_TABLES['settings']} WHERE setting LIKE ?", ("reserve.%",))
        for k, v in (self.reserve or {}).items():
            self._put_setting(db, f"reserve.{k}", json.dumps(v))

    @staticmethod
    def _write_shape(db, sh: "MinedShape") -> None:
        ts = PROD_TABLES["shapes"]
        now = _now()
        blob = solid_blob(sh.vertices, sh.triangles) if sh.is_solid else None
        row = (sh.date, sh.name, sh.plane, float(sh.from_), float(sh.to), outline_text(sh.points), sh.note,
               sh.kind, blob, sh.source or None)
        if db.scalar(f"SELECT COUNT(*) FROM {ts} WHERE id = ?", (sh.id,)):
            db.execute(f"UPDATE {ts} SET mined_on = ?, name = ?, view_plane = ?, depth_from = ?, depth_to = ?, "
                       "outline = ?, note = ?, shape_kind = ?, solid = ?, source_file = ?, updated_at = ? "
                       "WHERE id = ?", (*row, now, sh.id))
        else:
            db.execute(f"INSERT INTO {ts} (id, mined_on, name, view_plane, depth_from, depth_to, outline, note, "
                       "shape_kind, solid, source_file, created_at, updated_at) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (sh.id, *row, now, now))

    @staticmethod
    def _delete_shape(db, sid: int) -> None:
        db.execute(f"DELETE FROM {PROD_TABLES['shapes']} WHERE id = ?", (sid,))

    def save(self) -> str:
        """Write the reserve setup and every shape (the database then holds exactly these)."""
        with self._open() as db, db.transaction():
            self._write_setup(db)
            ids = {s.id for s in self.shapes}
            for (sid,) in db.query(f"SELECT id FROM {PROD_TABLES['shapes']}"):
                if sid not in ids:
                    self._delete_shape(db, sid)
            for sh in self.shapes:
                self._write_shape(db, sh)
        self._exists = True
        return self.where

    def save_setup(self) -> None:
        with self._open() as db, db.transaction():
            self._write_setup(db)
        self._exists = True

    def put(self, sh: "MinedShape") -> None:
        """Add a new shape or replace the one with the same id, in memory and in the database."""
        with self._open() as db, db.transaction():
            self._write_setup(db)
            self._write_shape(db, sh)
        self._exists = True
        for i, old in enumerate(self.shapes):
            if old.id == sh.id:
                self.shapes[i] = sh
                break
        else:
            self.shapes.append(sh)

    def remove(self, sid: int) -> None:
        with self._open() as db, db.transaction():
            self._delete_shape(db, sid)
        self.shapes = [s for s in self.shapes if s.id != sid]

    def next_id(self) -> int:
        """One more than the highest id in memory or in the database (another user may have
        added shapes to a shared MySQL database)."""
        top = max((s.id for s in self.shapes), default=0)
        if self._exists:
            try:
                with self.db.connect(create=False) as db:
                    top = max(top, int(db.scalar(f"SELECT MAX(id) FROM {PROD_TABLES['shapes']}", default=0) or 0))
            except Exception:  # noqa: BLE001
                pass
        return top + 1

    def sorted_shapes(self) -> list:
        return sorted(self.shapes, key=lambda s: (s.date, s.id))

    def get(self, sid: int) -> MinedShape | None:
        return next((s for s in self.shapes if s.id == sid), None)


def production_signature(target) -> tuple | None:
    """A fingerprint of the production records (table_updates versions, or row count and the
    latest change without it); None when the database cannot be reached."""
    from ..io.database import PROD_PREFIX, change_signature

    try:
        with target.connect(create=False) as db:
            tracked = change_signature(db, PROD_PREFIX)
            if tracked is not None:
                return ("tracked",) + tracked
            t = PROD_TABLES["shapes"]
            if not db.has_table(t):
                return ("none",)
            return tuple(db.query(f"SELECT COUNT(*), MAX(updated_at) FROM {t}")[0])
    except Exception:  # noqa: BLE001
        return None


def _legacy_records(folder: Path, target) -> tuple[str, list, dict] | None:
    """Records kept elsewhere by an earlier version or setting: (where, shapes, reserve).

    In order: geodatabase.sqlite of the folder (when the project database is MySQL),
    production.sqlite, then production.json (a newer production.json.tmp is preferred:
    it holds a save that was written but never swapped in)."""
    from ..io.database import PROJECT_DB, Database, is_sqlite_file

    candidates = []
    if target.is_mysql:
        candidates.append((folder / PROJECT_DB, PROD_TABLES))
    candidates.append((folder / LEGACY_DB, _OLD_TABLES))
    for path, names in candidates:
        if not is_sqlite_file(path):
            continue
        try:
            with Database.sqlite(path, create=False) as db:
                found = _read_records(db, names)
        except Exception:  # noqa: BLE001 - a damaged old file: try the next one
            continue
        if found is not None:
            return (str(path), *found)
    js = []
    for p in (folder / LEGACY_FILE, folder / (LEGACY_FILE + ".tmp")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("format") == PRODUCTION_FORMAT:
            js.append((p.stat().st_mtime_ns, p, data))
    if not js:
        return None
    _, p, data = max(js, key=lambda t: t[0])
    return str(p), [MinedShape.from_json(d) for d in data.get("shapes", [])], dict(data.get("reserve", {}))


# --- depletion ---------------------------------------------------------------------------------
@dataclass
class Totals:
    tonnes: float
    metal: float            # tonnes x grade (e.g. grams for g/t)

    @property
    def grade(self) -> float:
        return self.metal / self.tonnes if self.tonnes > 0 else 0.0


class BlockIndex:
    """The blocks bucketed on a plan (XY) grid, so the blocks near a box are found by looking
    at the grid cells under it instead of scanning every block: the cost of a query grows
    with the size of the box (a mining cut), not with the size of the block model."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo, self.hi = lo, hi
        self.n = len(lo)
        if not self.n:
            return
        c = (lo[:, :2] + hi[:, :2]) / 2
        size = hi - lo
        self.half = size.max(axis=0) / 2                    # largest block half-size per axis
        typical = float(np.median(size[:, :2].max(axis=1)))
        self.cell = max(typical * 4.0, 1e-6)                # a few blocks per cell
        self.origin = c.min(axis=0)
        ij = np.floor((c - self.origin) / self.cell).astype(np.int64)
        self.nx, self.ny = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
        key = ij[:, 0] * self.ny + ij[:, 1]
        self.order = np.argsort(key, kind="stable")
        self.keys = key[self.order]

    def query(self, qlo, qhi) -> np.ndarray:
        """Indices (sorted) of the blocks overlapping the box (qlo, qhi)."""
        if not self.n:
            return np.zeros(0, dtype=np.int64)
        qlo, qhi = np.asarray(qlo, dtype=np.float64), np.asarray(qhi, dtype=np.float64)
        # a block overlaps the box only if its centre is within the box grown by the largest half-size
        a = np.floor((qlo[:2] - self.half[:2] - self.origin) / self.cell).astype(np.int64)
        b = np.floor((qhi[:2] + self.half[:2] - self.origin) / self.cell).astype(np.int64)
        a = np.maximum(a, 0)
        b = np.minimum(b, [self.nx - 1, self.ny - 1])
        if (a > b).any():
            return np.zeros(0, dtype=np.int64)
        ix = np.arange(a[0], b[0] + 1)
        start = np.searchsorted(self.keys, ix * self.ny + a[1], side="left")
        stop = np.searchsorted(self.keys, ix * self.ny + b[1], side="right")
        parts = [self.order[s0:s1] for s0, s1 in zip(start, stop) if s1 > s0]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        rows = np.concatenate(parts)
        exact = np.all((self.hi[rows] >= qlo) & (self.lo[rows] <= qhi), axis=1)
        return np.sort(rows[exact])


SAMPLES_PER_AXIS = 2      # sample points per block: 2 x 2 x 2 = 8 (each stands for 1/8 of the block)


class Depletion:
    """Mined state of the reserve blocks from the mined shapes (sample-point based)."""

    def __init__(self, bm, reserve: np.ndarray, density: np.ndarray, grade: np.ndarray, n: int = SAMPLES_PER_AXIS):
        """``reserve``: indices of the reserve blocks; ``density`` / ``grade``: per block (all blocks).
        ``n``: sample points per axis (n x n x n per block)."""
        self.bm = bm
        self.idx = np.asarray(reserve, dtype=np.int64)
        self.n = n
        f = (np.arange(n) + 0.5) / n - 0.5
        self.offsets = np.stack(np.meshgrid(f, f, f, indexing="ij"), -1).reshape(-1, 3)   # (P, 3)
        c = bm.centroids[self.idx]
        s = bm.sizes[self.idx].astype(np.float64)
        self.lo, self.hi = c - s / 2, c + s / 2
        self.tonnes = s.prod(axis=1) * np.asarray(density, dtype=np.float64)[self.idx]
        self.grade = np.asarray(grade, dtype=np.float64)[self.idx]
        self.point_shape = np.full((len(self.idx), len(self.offsets)), -1, dtype=np.int32)
        self.shape_day: dict[int, int] = {}
        self.index = BlockIndex(self.lo, self.hi)
        self._totals: dict | None = None        # shape id -> (Totals, blocks), worked out once per apply()

    # --- sample points ------------------------------------------------------------------
    def points(self, rows: np.ndarray) -> np.ndarray:
        """Sample points of reserve rows, (len(rows) * P, 3)."""
        c = self.bm.centroids[self.idx[rows]]
        s = self.bm.sizes[self.idx[rows]].astype(np.float64)
        return (c[:, None, :] + self.offsets[None] * s[:, None, :]).reshape(-1, 3)

    def candidates(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Reserve rows whose block overlaps the box (lo, hi) (sorted)."""
        return self.index.query(lo, hi)

    # --- the mined state ------------------------------------------------------------------
    def apply(self, shapes: list) -> None:
        """Mine the points shape by shape, earliest date first; a point keeps its first shape."""
        self.point_shape.fill(-1)
        self.shape_day = {}
        self._totals = None
        for sh in sorted(shapes, key=lambda s: (s.date, s.id)):
            self.shape_day[sh.id] = sh.day
            rows = self.candidates(*sh.bounds())
            if not len(rows):
                continue
            inside = sh.contains(self.points(rows)).reshape(len(rows), -1)
            cur = self.point_shape[rows]
            new = inside & (cur < 0)
            cur[new] = sh.id
            self.point_shape[rows] = cur

    def mined_points(self, as_of: int | None = None) -> np.ndarray:
        """Boolean (R, P): points mined up to the day ``as_of`` (all mined points when None)."""
        mined = self.point_shape >= 0
        if as_of is None or not self.shape_day:
            return mined
        late = [sid for sid, d in self.shape_day.items() if d > as_of]
        if late:
            mined &= ~np.isin(self.point_shape, late)
        return mined

    def fraction(self, as_of: int | None = None) -> np.ndarray:
        return self.mined_points(as_of).mean(axis=1)

    def summary(self, as_of: int | None = None) -> dict:
        f = self.fraction(as_of)
        total = Totals(float(self.tonnes.sum()), float((self.tonnes * self.grade).sum()))
        t = self.tonnes * f
        produced = Totals(float(t.sum()), float((t * self.grade).sum()))
        remaining = Totals(total.tonnes - produced.tonnes, total.metal - produced.metal)
        return {"total": total, "produced": produced, "remaining": remaining}

    def shape_totals(self, sid: int) -> tuple[Totals, int]:
        """What one shape mined (points it was the first to mine), and blocks it touched."""
        return self.all_shape_totals().get(int(sid), (Totals(0.0, 0.0), 0))

    def all_shape_totals(self) -> dict:
        """Shape id -> (Totals, blocks touched) for every shape, in one pass over the mined
        points (not one pass per shape); kept until the next apply()."""
        if self._totals is None:
            rows, cols = np.nonzero(self.point_shape >= 0)
            out: dict = {}
            if len(rows):
                sids = self.point_shape[rows, cols]
                per_point = self.tonnes[rows] / self.point_shape.shape[1]
                ids, inv = np.unique(sids, return_inverse=True)
                tonnes = np.bincount(inv, weights=per_point, minlength=len(ids))
                metal = np.bincount(inv, weights=per_point * self.grade[rows], minlength=len(ids))
                pairs = np.unique(inv.astype(np.int64) * len(self.idx) + rows)       # (shape, block) once
                blocks = np.bincount(pairs // len(self.idx), minlength=len(ids))
                out = {int(i): (Totals(float(t), float(m)), int(b)) for i, t, m, b in zip(ids, tonnes, metal, blocks)}
            self._totals = out
        return self._totals

    # --- a shape being entered (not saved yet) --------------------------------------------
    def preview(self, sh: MinedShape) -> dict:
        """New tonnes the shape would mine, tonnes in it already mined by earlier shapes,
        blocks touched, and how much of the shape's volume is reserve (ore fill)."""
        rows = self.candidates(*sh.bounds())
        res = {"new": Totals(0.0, 0.0), "already": Totals(0.0, 0.0), "blocks": 0, "fill": 0.0}
        volume = sh.volume()                            # outline area x thickness, or the solid's volume
        if not len(rows) or volume <= 0:
            return res
        P = len(self.offsets)
        inside = sh.contains(self.points(rows)).reshape(len(rows), P)
        earlier = self.point_shape[rows] >= 0
        if sh.id in self.shape_day:                     # editing a saved shape: its own points are free
            earlier &= self.point_shape[rows] != sh.id
        tp = self.tonnes[rows] / P                      # tonnes per point
        g = self.grade[rows]
        new = inside & ~earlier
        old = inside & earlier
        tn = (new * tp[:, None]).sum(axis=1)
        to = (old * tp[:, None]).sum(axis=1)
        res["new"] = Totals(float(tn.sum()), float((tn * g).sum()))
        res["already"] = Totals(float(to.sum()), float((to * g).sum()))
        res["blocks"] = int((inside.any(axis=1)).sum())
        vol = self.bm.sizes[self.idx[rows]].astype(np.float64).prod(axis=1) / P
        res["fill"] = float(min(1.0, (inside * vol[:, None]).sum() / volume))
        return res

    def profile(self, sh: MinedShape, bins: int = 80):
        """Reserve tonnes inside the outline, level by level along the depth axis.

        Returns (edges (bins+1,), ore tonnes (bins,), already-mined tonnes (bins,), ore range
        (lo, hi) or None) - the ore range is where reserve points lie inside the outline."""
        iu, iv, ia = sh.axes
        lo, hi = sh.bounds()
        lo[ia], hi[ia] = -1e18, 1e18                     # the whole depth range
        rows = self.candidates(lo, hi)
        if not len(rows):
            return None
        P = len(self.offsets)
        pts = self.points(rows)
        inside = sh.in_outline(pts)
        if not inside.any():
            return None
        a = pts[inside, ia]
        tp = np.repeat(self.tonnes[rows] / P, P)[inside]
        mined = (self.point_shape[rows].reshape(-1) >= 0)[inside]
        if sh.id in self.shape_day:
            mined &= self.point_shape[rows].reshape(-1)[inside] != sh.id
        a0, a1 = float(a.min()), float(a.max())
        # Levels a whole number of sample-point steps thick, centred on the points (other
        # widths give stripes: levels holding one or two rows of points by turns).
        step = float(np.median(self.bm.sizes[self.idx[rows]][:, ia])) / self.n
        n_steps = int(np.ceil((a1 - a0) / step)) + 1
        width = step * max(1, int(np.ceil(n_steps / bins)))
        pad = 3
        start = a0 - step / 2 - pad * width
        count = int(np.ceil((a1 + step / 2 - start) / width)) + pad
        edges = start + width * np.arange(count + 1)
        ore, _ = np.histogram(a, bins=edges, weights=tp)
        old, _ = np.histogram(a[mined], bins=edges, weights=tp[mined])
        # The ore range: from the lowest to the highest level holding a real amount of ore
        # (5 % of the richest level), so a thin sliver at the tip of the ore is left out.
        sig = np.flatnonzero(ore >= ORE_LEVEL_SHARE * ore.max())
        return edges, ore, old, (float(edges[sig[0]]), float(edges[sig[-1] + 1]))

    # --- attributes for the block model (as of a date) -----------------------------------
    def block_attributes(self, as_of: int | None = None) -> dict:
        """``mined_fraction`` (0..1, NaN off the reserve) and ``status`` (Unmined / Partly
        mined / Mined, null off the reserve) for every block of the model."""
        nb = self.bm.n_blocks
        frac = np.full(nb, np.nan, dtype=np.float32)
        f = self.fraction(as_of)
        frac[self.idx] = f
        status = np.full(nb, -1, dtype=np.int32)
        status[self.idx] = np.where(f <= 0, 0, np.where(f >= 1 - 1e-9, 2, 1))
        return {"mined_fraction": frac, "status": status}


def reserve_blocks(bm, inside: np.ndarray | None, density_name: str | None, density_value: float,
                   grade_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(reserve indices, density per block, grade per block). Reserve blocks are inside the
    reserve solid (all blocks when ``inside`` is None) with a valid grade (and density)."""
    ok = bm.valid_mask(grade_name).copy()
    if density_name:
        ok &= bm.valid_mask(density_name)
        dens = bm.attributes[density_name].astype(np.float64)
    else:
        dens = np.full(bm.n_blocks, float(density_value))
    if inside is not None:
        ok &= inside
    return np.flatnonzero(ok), dens, bm.attributes[grade_name].astype(np.float64)


def metal_text(metal: float, unit: str) -> str:
    """Contained metal: g/t grades -> kg; % grades -> tonnes."""
    if unit == "%":
        return f"{metal / 100.0:,.0f} t"
    return f"{metal / 1000.0:,.1f} kg"


# --- geometry for display ---------------------------------------------------------------------
def prism_mesh(sh: MinedShape, origin=None):
    """The shape as a closed solid (the outline swept from ``from`` to ``to``), triangles,
    in scene coordinates (``origin`` subtracted). None when the outline is degenerate."""
    import pyvista as pv

    if sh.is_solid:                                    # a mined solid: its own triangles
        o = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
        faces = np.hstack([np.full((len(sh.triangles), 1), 3), sh.triangles]).ravel()
        return pv.PolyData(sh.vertices - o, faces=faces)

    p2 = sh.outline()
    if len(p2) > 3 and np.allclose(p2[0], p2[-1]):
        p2 = p2[:-1]
    if len(p2) < 3 or sh.area() <= 0:
        return None
    iu, iv, ia = sh.axes
    depth = max(sh.depth, 0.01)
    pts = np.zeros((len(p2), 3))
    pts[:, iu], pts[:, iv], pts[:, ia] = p2[:, 0], p2[:, 1], sh.lo
    if origin is not None:
        pts -= np.asarray(origin, dtype=np.float64)
    cap = pv.PolyData(pts.astype(np.float64), faces=np.array([len(p2), *range(len(p2))])).triangulate()
    vec = [0.0, 0.0, 0.0]
    vec[ia] = depth
    solid = cap.extrude(vec, capping=True).triangulate().clean()
    # extrude leaves the caps and walls facing different ways: make every face point outward
    # (a proper closed solid: right volume, right for export)
    return solid.compute_normals(auto_orient_normals=True, consistent_normals=True, cell_normals=True,
                                 point_normals=False)


def month_of(date: str) -> str:
    return date[:7]


MONTH_ABBR = [N_("Jan"), N_("Feb"), N_("Mar"), N_("Apr"), N_("May"), N_("Jun"), N_("Jul"), N_("Aug"), N_("Sep"),
              N_("Oct"), N_("Nov"), N_("Dec")]


def month_label(month: str) -> str:
    try:
        d = _dt.date.fromisoformat(month + "-01")
    except ValueError:
        return month
    return tr("{month} {year}", month=tr(MONTH_ABBR[d.month - 1]), year=d.year)
