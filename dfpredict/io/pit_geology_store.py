"""Pit geology: what the geologist maps in an underground drive (a pit), kept in the project
database next to the survey (tables survey_geo_*, so Refresh and change tracking see them).

How it is recorded (the usual practice of underground / adit mapping):

* every position is a **chainage**: metres along the drive from the pit's start - 0 m is the
  entrance of a main pit, and the junction station of a branch (the station it starts at, on its
  parent). In the drive it is measured with a tape from a survey station, so it can also be
  typed as "station + metres" (P1-03+4.5; see scene/pit_geology.py);
* **left / right wall** are as seen looking forward, toward the face (chainage increasing);
  the **back** is the roof, the **face** the end of the drive;
* rocks are mapped as intervals From - To along the drive, on both walls and the back
  (wall "all") or on one of them;
* planar structures (bedding, contacts, faults, joints, veins, foliation, shear zones) are
  measured at a point: chainage, wall, height above the floor, **dip direction and dip**
  (written 135∠45 in Chinese practice: 倾向∠倾角), thickness, filling;
* samples (channel, chip, grab or panel) From - To on a wall at a height, with the channel's
  width and depth, the weight and the assay values (one numeric field per element, as many as
  needed - the same idea as drillhole_assay).

Tables (made by the database set-up when missing; only adds):

    survey_geo_intervals   id, pit_id, seq, from_m, to_m, wall, rock, description, alteration,
                           mineralization, mapped_on (YYYY-MM-DD), geologist
    survey_geo_structures  id, pit_id, seq, at_m, wall, height, kind, dip_direction, dip,
                           thickness, fill, description, mapped_on, geologist
    survey_geo_samples     id, pit_id, seq, sample, from_m, to_m, wall, height, kind, width_cm,
                           depth_cm, weight_kg, taken_on, sampler, note, + one REAL field per
                           assay value (gold, silver, ...)

Rock codes and colours are the project's: drillhole_styles / drillhole_translation, edited in
the Drillhole Manager's Rock colours tab. No Qt here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..i18n import N_, tr

INTERVALS, STRUCTURES, SAMPLES = "survey_geo_intervals", "survey_geo_structures", "survey_geo_samples"
GEO_TABLES = (INTERVALS, STRUCTURES, SAMPLES)

# where a record is: wall keys -> text shown (tr() where shown)
WALLS = {"left": N_("Left wall"), "right": N_("Right wall"), "back": N_("Back (roof)"), "face": N_("Face")}
INTERVAL_WALLS = {"all": N_("Both walls and back"), "left": N_("Left wall"), "right": N_("Right wall"),
                  "back": N_("Back (roof)")}
SAMPLE_WALLS = {"left": N_("Left wall"), "right": N_("Right wall"), "back": N_("Back (roof)"), "face": N_("Face")}
STRUCTURE_KINDS = {"bedding": N_("Bedding"), "contact": N_("Contact"), "fault": N_("Fault"), "joint": N_("Joint"),
                   "vein": N_("Vein"), "foliation": N_("Foliation / schistosity"), "shear": N_("Shear zone"),
                   "other": N_("Other")}
# their colours in the view and on the wall sketch
STRUCTURE_COLORS = {"bedding": "#4fb4ff", "contact": "#ffffff", "fault": "#ff4f4f", "joint": "#ffd21a",
                    "vein": "#ff9b30", "foliation": "#5fd38d", "shear": "#e070ff", "other": "#b0b6bd"}
SAMPLE_KINDS = {"channel": N_("Channel"), "chip": N_("Chip"), "grab": N_("Grab"), "panel": N_("Panel")}

DEFAULT_SAMPLE_HEIGHT = 1.0       # m above the floor: a channel cut at waist height when none is given

_ID = "INTEGER NOT NULL PRIMARY KEY"
_COMMON = [("pit_id", "INTEGER"), ("seq", "INTEGER")]
_FIELDS = {
    INTERVALS: _COMMON + [("from_m", "DOUBLE"), ("to_m", "DOUBLE"), ("wall", "VARCHAR(10)"), ("rock", "VARCHAR(32)"),
                          ("description", "TEXT"), ("alteration", "VARCHAR(200)"), ("mineralization", "VARCHAR(200)"),
                          ("mapped_on", "VARCHAR(10)"), ("geologist", "VARCHAR(100)")],
    STRUCTURES: _COMMON + [("at_m", "DOUBLE"), ("wall", "VARCHAR(10)"), ("height", "DOUBLE"), ("kind", "VARCHAR(16)"),
                           ("dip_direction", "DOUBLE"), ("dip", "DOUBLE"), ("thickness", "DOUBLE"),
                           ("fill", "VARCHAR(200)"), ("description", "TEXT"), ("mapped_on", "VARCHAR(10)"),
                           ("geologist", "VARCHAR(100)")],
    SAMPLES: _COMMON + [("sample", "VARCHAR(64)"), ("from_m", "DOUBLE"), ("to_m", "DOUBLE"), ("wall", "VARCHAR(10)"),
                        ("height", "DOUBLE"), ("kind", "VARCHAR(16)"), ("width_cm", "DOUBLE"), ("depth_cm", "DOUBLE"),
                        ("weight_kg", "DOUBLE"), ("taken_on", "VARCHAR(10)"), ("sampler", "VARCHAR(100)"),
                        ("note", "TEXT")],
}
_FIXED_SAMPLE_COLUMNS = {"id"} | {c for c, _k in _FIELDS[SAMPLES]}
_INDEXES = tuple((t, f"idx_{t}_pit", ["pit_id", "seq"]) for t in GEO_TABLES)


def _ddl(table: str, mysql: bool) -> str:
    opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if mysql else ""
    cols = ", ".join(f"{c} {k}" for c, k in _FIELDS[table])
    return f"CREATE TABLE {table} (id {_ID}, {cols}){opts}"


# --- values ---------------------------------------------------------------------------------------
def _num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _txt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()


def _none(v):
    t = _txt(v)
    return t or None


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def orientation_text(dip_direction, dip) -> str:
    """135∠45 (dip direction ∠ dip, whole degrees); "" when not measured."""
    if dip_direction is None or dip is None:
        return ""
    return f"{float(dip_direction) % 360:03.0f}∠{float(dip):.0f}"


def strike_of(dip_direction: float) -> float:
    """The strike (right-hand rule: the dip is to the right of the strike) of a dip direction."""
    return (float(dip_direction) - 90.0) % 360.0


def dip_direction_of(strike: float) -> float:
    """The dip direction of a strike given by the right-hand rule."""
    return (float(strike) + 90.0) % 360.0


def plane_normal(dip_direction: float, dip: float) -> tuple:
    """Upward unit normal (east, north, up) of a plane given as dip direction / dip (degrees)."""
    a, d = math.radians(float(dip_direction)), math.radians(float(dip))
    return (math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d))


def next_sample_name(name: str) -> str:
    """The sample number after ``name``: PS-009 -> PS-010, 12 -> 13; "" -> ""."""
    m = re.match(r"^(.*?)(\d+)(\D*)$", name or "")
    if not m:
        return ""
    head, num, tail = m.groups()
    return f"{head}{int(num) + 1:0{len(num)}d}{tail}"


# --- records ------------------------------------------------------------------------------------
@dataclass
class GeoInterval:
    """A rock mapped along the drive from ``frm`` to ``to`` (chainage, m)."""
    frm: float | None
    to: float | None
    wall: str = "all"
    rock: str = ""
    description: str = ""
    alteration: str = ""
    mineralization: str = ""
    mapped_on: str = ""
    geologist: str = ""
    id: int | None = None

    def walls(self) -> tuple:
        return ("left", "back", "right") if self.wall == "all" else (self.wall,)


@dataclass
class GeoStructure:
    """A planar structure measured at chainage ``at`` on a wall (height above the floor)."""
    at: float | None
    wall: str = "left"
    height: float | None = None
    kind: str = "joint"
    dip_direction: float | None = None
    dip: float | None = None
    thickness: float | None = None     # m
    fill: str = ""
    description: str = ""
    mapped_on: str = ""
    geologist: str = ""
    id: int | None = None


@dataclass
class GeoSample:
    """A sample cut from ``frm`` to ``to`` (chainage) on a wall at ``height`` above the floor."""
    name: str
    frm: float | None
    to: float | None
    wall: str = "left"
    height: float | None = None
    kind: str = "channel"
    width_cm: float | None = None
    depth_cm: float | None = None
    weight_kg: float | None = None
    values: dict = field(default_factory=dict)     # assay field -> value
    taken_on: str = ""
    sampler: str = ""
    note: str = ""
    id: int | None = None

    @property
    def length(self) -> float | None:
        if self.frm is None or self.to is None:
            return None
        return abs(self.to - self.frm)


@dataclass
class PitGeology:
    intervals: list = field(default_factory=list)
    structures: list = field(default_factory=list)
    samples: list = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.intervals or self.structures or self.samples)

    def extent(self) -> float:
        """The furthest chainage used (0 without records)."""
        vals = [v for r in self.intervals for v in (r.frm, r.to)] + [s.at for s in self.structures] + \
               [v for s in self.samples for v in (s.frm, s.to)]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else 0.0


# --- checks ------------------------------------------------------------------------------------------
def sorted_geology(geo: PitGeology) -> PitGeology:
    """Records in chainage order (as they are saved)."""
    def key(v):
        return float("inf") if v is None else v
    return PitGeology(sorted(geo.intervals, key=lambda r: (key(r.frm), key(r.to))),
                      sorted(geo.structures, key=lambda s: key(s.at)),
                      sorted(geo.samples, key=lambda s: (key(s.frm), s.name)))


def check_geology(geo: PitGeology, length: float | None = None, assay_fields=(),
                  rows: dict | None = None) -> tuple[list, list]:
    """(errors, warnings) of a pit's geology before it is saved, each "where: what". Errors stop the
    save; warnings do not. ``length``: the pit's surveyed length along the drive (None: not known).
    ``rows``: "intervals" / "structures" / "samples" -> the row number shown for each record (an
    Excel row); records are numbered 1, 2, ... otherwise."""
    errors, warnings = [], []
    rows = rows or {}
    rocks, structs, samples = tr("Rocks"), tr("Structures"), tr("Samples")
    keys = {rocks: "intervals", structs: "structures", samples: "samples"}

    def num(label, i):
        given = rows.get(keys[label])
        return given[i] if given is not None and i < len(given) else i + 1

    def where(label, i):
        return tr("{kind} row {n}", kind=label, n=num(label, i))

    def chain(w, name, v):
        if v is None:
            errors.append(f"{w}: " + tr("{name} is needed", name=name))
            return False
        if v < 0:
            errors.append(f"{w}: " + tr("{name} must be 0 m or more (0 m is the start of the pit)", name=name))
            return False
        if length is not None and v > length + 1e-6:
            warnings.append(f"{w}: " + tr("{name} {v:.2f} m is beyond the last station ({length:.2f} m); it is placed "
                                          "along the last leg", name=name, v=v, length=length))
        return True

    def date(w, v):
        if v and not _DATE.match(v):
            errors.append(f"{w}: " + tr("the date {v!r} is not YYYY-MM-DD", v=v))

    for i, r in enumerate(geo.intervals):
        w = where(rocks, i)
        ok = chain(w, tr("From"), r.frm) & chain(w, tr("To"), r.to)
        if ok and r.to <= r.frm:
            errors.append(f"{w}: " + tr("To must be more than From"))
        if not r.rock:
            errors.append(f"{w}: " + tr("choose the rock"))
        if r.wall not in INTERVAL_WALLS:
            errors.append(f"{w}: " + tr("unknown wall {v!r}", v=r.wall))
        date(w, r.mapped_on)
    # overlapping rocks on the same wall
    spans = {k: [] for k in ("left", "back", "right")}
    for i, r in enumerate(geo.intervals):
        if r.frm is None or r.to is None or r.to <= r.frm or r.wall not in INTERVAL_WALLS:
            continue
        for k in r.walls():
            spans[k].append((r.frm, r.to, i))
    seen = set()
    for k, items in spans.items():
        items.sort()
        for (a0, a1, i), (b0, b1, j) in zip(items, items[1:]):
            if b0 < a1 - 1e-6 and (min(i, j), max(i, j)) not in seen:
                seen.add((min(i, j), max(i, j)))
                errors.append(tr("Rocks rows {a} and {b} overlap on the {wall} ({b0:.2f} - {a1:.2f} m)",
                                 a=num(rocks, i), b=num(rocks, j), wall=tr(WALLS[k]).lower(), b0=b0, a1=a1))

    for i, s in enumerate(geo.structures):
        w = where(structs, i)
        chain(w, tr("Chainage"), s.at)
        if s.wall not in WALLS:
            errors.append(f"{w}: " + tr("unknown wall {v!r}", v=s.wall))
        if s.kind not in STRUCTURE_KINDS:
            errors.append(f"{w}: " + tr("unknown structure type {v!r}", v=s.kind))
        if s.dip is None or s.dip_direction is None:
            if s.dip is not None or s.dip_direction is not None:
                errors.append(f"{w}: " + tr("type both the dip direction and the dip (or neither)"))
            else:
                warnings.append(f"{w}: " + tr("no dip direction / dip measured"))
        if s.dip is not None and not 0 <= s.dip <= 90:
            errors.append(f"{w}: " + tr("the dip must be 0° to 90°"))
        if s.dip_direction is not None and not 0 <= s.dip_direction <= 360:
            errors.append(f"{w}: " + tr("the dip direction must be 0° to 360°"))
        if s.height is not None and s.height < 0:
            errors.append(f"{w}: " + tr("the height above the floor must be 0 m or more"))
        if s.thickness is not None and s.thickness < 0:
            errors.append(f"{w}: " + tr("the thickness must be 0 m or more"))
        date(w, s.mapped_on)

    names: dict = {}
    for i, s in enumerate(geo.samples):
        w = where(samples, i)
        if not s.name:
            errors.append(f"{w}: " + tr("type the sample number"))
        elif s.name in names:
            errors.append(f"{w}: " + tr("sample number {name} is used twice (rows {a} and {b})", name=s.name,
                                        a=num(samples, names[s.name]), b=num(samples, i)))
        else:
            names[s.name] = i
        ok = chain(w, tr("From"), s.frm) & chain(w, tr("To"), s.to)
        if ok and s.to <= s.frm:
            errors.append(f"{w}: " + tr("To must be more than From"))
        if s.wall not in SAMPLE_WALLS:
            errors.append(f"{w}: " + tr("unknown wall {v!r}", v=s.wall))
        if s.kind not in SAMPLE_KINDS:
            errors.append(f"{w}: " + tr("unknown sample type {v!r}", v=s.kind))
        for k, label in (("height", N_("Height (m)")), ("width_cm", N_("Width (cm)")), ("depth_cm", N_("Depth (cm)")),
                         ("weight_kg", N_("Weight (kg)"))):
            v = getattr(s, k)
            if v is not None and v < 0:
                errors.append(f"{w}: " + tr("{name} must be 0 or more", name=tr(label)))
        for k, v in s.values.items():
            if v is not None and v < 0:
                errors.append(f"{w}: " + tr("{name} must be 0 or more", name=k))
        date(w, s.taken_on)
    return errors, warnings


# --- storage ----------------------------------------------------------------------------------------------
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class PitGeologyStore:
    """Read and change the pit geology of a project database (a database ``Target``)."""

    def __init__(self, target):
        self.target = target

    def prepare(self, db=None) -> list[str]:
        """Make the missing tables / fields and indexes (only adds). A new samples table gets the
        assay fields of drillhole_assay (the same elements are assayed). Returns what was done."""
        if db is None:
            with self.target.connect() as con:
                return self.prepare(con)
        notes = []
        for name in GEO_TABLES:
            stored = db.table_name(name)
            if stored is None:
                db.execute(_ddl(name, db.is_mysql))
                db.con.commit()
                notes.append(f"made table {name}")
                if name == SAMPLES:
                    for f in self._drillhole_assay_fields(db):
                        db.add_column(SAMPLES, f, "real")
                        db.con.commit()
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

    @staticmethod
    def _drillhole_assay_fields(db) -> list[str]:
        try:
            from .drillhole_store import ASSAY, DrillholeStore

            if not db.has_table(ASSAY):
                return []
            info = DrillholeStore(None)._info(db, db.table_name(ASSAY))
            return [f for f in (info.value_fields if info else []) if f.lower() not in _FIXED_SAMPLE_COLUMNS
                    and _FIELD_NAME.match(f)]
        except Exception:  # noqa: BLE001 - no assay fields to copy: none
            return []

    # --- assay fields of the samples -----------------------------------------------------------------
    def assay_fields(self, db=None) -> list[str]:
        """The samples table's assay fields (every column that is not a fixed one), in table order."""
        if db is None:
            with self.target.connect(create=False) as con:
                return self.assay_fields(con)
        stored = db.table_name(SAMPLES)
        if stored is None:
            return []
        return [c for c in db.columns(stored) if c.lower() not in _FIXED_SAMPLE_COLUMNS]

    def add_field(self, name: str) -> str:
        name = _txt(name)
        if not _FIELD_NAME.match(name):
            raise ValueError(tr("A field name starts with a letter and has only letters, digits and _ "
                                "(e.g. gold, cu_pct)"))
        if name.lower() in _FIXED_SAMPLE_COLUMNS:
            raise ValueError(tr("{name} is a fixed field of the samples", name=name))
        with self.target.connect() as db:
            if name.lower() in {c.lower() for c in db.columns(SAMPLES)}:
                raise ValueError(tr("The samples already have a field {name}", name=name))
            db.add_column(SAMPLES, name, "real")
            db.con.commit()
        return name

    def remove_field(self, name: str) -> None:
        with self.target.connect() as db:
            if name not in self.assay_fields(db):
                raise ValueError(tr("{name} is not an assay field", name=name))
            db.drop_column(SAMPLES, name)
            db.con.commit()

    # --- read --------------------------------------------------------------------------------------
    def geology(self, pit_id: int | None = None) -> dict[int, PitGeology]:
        """Pit id -> its geology (every pit with records, or only ``pit_id``)."""
        out: dict[int, PitGeology] = {}
        where, params = ("WHERE pit_id = ?", (pit_id,)) if pit_id is not None else ("", ())
        with self.target.connect(create=False) as db:
            if not all(db.has_table(t) for t in GEO_TABLES):
                return out
            for row in db.query(f"SELECT id, pit_id, from_m, to_m, wall, rock, description, alteration, "
                                f"mineralization, mapped_on, geologist FROM {INTERVALS} {where} "
                                f"ORDER BY pit_id, seq, id",
                                params):
                i, pid, a, b, wall, rock, desc, alt, mine, on, who = row
                out.setdefault(int(pid), PitGeology()).intervals.append(GeoInterval(
                    _num(a), _num(b), _txt(wall) or "all", _txt(rock), _txt(desc), _txt(alt), _txt(mine), _txt(on),
                    _txt(who), int(i)))
            for row in db.query(f"SELECT id, pit_id, at_m, wall, height, kind, dip_direction, dip, thickness, fill, "
                                f"description, mapped_on, geologist FROM {STRUCTURES} {where} ORDER BY pit_id, seq, id",
                                params):
                i, pid, at, wall, h, kind, dd, dip, th, fill, desc, on, who = row
                out.setdefault(int(pid), PitGeology()).structures.append(GeoStructure(
                    _num(at), _txt(wall) or "left", _num(h), _txt(kind) or "other", _num(dd), _num(dip), _num(th),
                    _txt(fill), _txt(desc), _txt(on), _txt(who), int(i)))
            fields = self.assay_fields(db)
            extra = "".join(f", {db.q(f)}" for f in fields)
            for row in db.query(f"SELECT id, pit_id, sample, from_m, to_m, wall, height, kind, width_cm, depth_cm, "
                                f"weight_kg, taken_on, sampler, note{extra} FROM {SAMPLES} {where} "
                                f"ORDER BY pit_id, seq, id", params):
                i, pid, name, a, b, wall, h, kind, wcm, dcm, kg, on, who, note = row[:14]
                vals = {f: _num(v) for f, v in zip(fields, row[14:])}
                out.setdefault(int(pid), PitGeology()).samples.append(GeoSample(
                    _txt(name), _num(a), _num(b), _txt(wall) or "left", _num(h), _txt(kind) or "channel", _num(wcm),
                    _num(dcm), _num(kg), vals, _txt(on), _txt(who), _txt(note), int(i)))
        return out

    def sample_names(self, exclude_pit: int | None = None) -> dict[str, int]:
        """Sample number -> pit id of every saved sample (but those of ``exclude_pit``)."""
        with self.target.connect(create=False) as db:
            if not db.has_table(SAMPLES):
                return {}
            rows = db.query(f"SELECT sample, pit_id FROM {SAMPLES}")
        return {_txt(n): int(p) for n, p in rows if _txt(n) and (exclude_pit is None or int(p) != exclude_pit)}

    # --- write --------------------------------------------------------------------------------------
    def save(self, pit_id: int, geo: PitGeology) -> None:
        """Replace the pit's geology with ``geo`` (one transaction; rows kept in the order given)."""
        with self.target.connect() as db:
            fields = self.assay_fields(db)
            with db.transaction():
                self._write(db, pit_id, geo, fields)

    @staticmethod
    def _next_id(db, table) -> int:
        return int(db.scalar(f"SELECT MAX(id) FROM {table}", default=0) or 0) + 1

    def _write(self, db, pit_id: int, geo: PitGeology, fields: list[str]) -> None:
        for t in GEO_TABLES:
            db.execute(f"DELETE FROM {t} WHERE pit_id = ?", (pit_id,))
        n = self._next_id(db, INTERVALS)
        db.executemany(f"INSERT INTO {INTERVALS} (id, pit_id, seq, from_m, to_m, wall, rock, description, alteration, "
                       "mineralization, mapped_on, geologist) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [(n + k, pit_id, k + 1, r.frm, r.to, r.wall, _none(r.rock), _none(r.description),
                         _none(r.alteration), _none(r.mineralization), _none(r.mapped_on), _none(r.geologist))
                        for k, r in enumerate(geo.intervals)])
        n = self._next_id(db, STRUCTURES)
        db.executemany(f"INSERT INTO {STRUCTURES} (id, pit_id, seq, at_m, wall, height, kind, dip_direction, dip, "
                       "thickness, fill, description, mapped_on, geologist) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                       "?, ?, ?)",
                       [(n + k, pit_id, k + 1, s.at, s.wall, s.height, s.kind, s.dip_direction, s.dip, s.thickness,
                         _none(s.fill), _none(s.description), _none(s.mapped_on), _none(s.geologist))
                        for k, s in enumerate(geo.structures)])
        n = self._next_id(db, SAMPLES)
        cols = "".join(f", {db.q(f)}" for f in fields)
        marks = ", ?" * len(fields)
        db.executemany(f"INSERT INTO {SAMPLES} (id, pit_id, seq, sample, from_m, to_m, wall, height, kind, width_cm, "
                       f"depth_cm, weight_kg, taken_on, sampler, note{cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                       f"?, ?, ?, ?{marks})",
                       [(n + k, pit_id, k + 1, s.name, s.frm, s.to, s.wall, s.height, s.kind, s.width_cm, s.depth_cm,
                         s.weight_kg, _none(s.taken_on), _none(s.sampler), _none(s.note),
                         *[_num(s.values.get(f)) for f in fields])
                        for k, s in enumerate(geo.samples)])

    @staticmethod
    def delete_pits(db, pit_ids) -> None:
        """Delete the geology of these pits (inside the caller's transaction; tables may be missing)."""
        for t in GEO_TABLES:
            if db.has_table(t):
                for p in pit_ids:
                    db.execute(f"DELETE FROM {t} WHERE pit_id = ?", (p,))
