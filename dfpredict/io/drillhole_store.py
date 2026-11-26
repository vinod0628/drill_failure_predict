"""Editing the drillholes of the project database (storage for the Drillhole Manager).

Tables (prefix drillhole_), made with these columns when missing:

    drillhole_collar   hole_id, x, y, z, max_depth, organization, drilling_manager, geology_surveyor,
                       start_date, end_date (YYYY-MM-DD), state (1 In work, 2 Completed with ore hit,
                       3 Completed without ore hit, 4 Failed - see HOLE_STATES)
    drillhole_survey   hole_id, depth, azimuth, dip            (dip: 0 = straight down, 90 = horizontal)
    drillhole_geology  hole_id, depth_from, depth_to, rock, note    (note: core description, may be empty)
    drillhole_assay    hole_id, depth_from, depth_to, then one number column per assay (gold, silver, ...)
    drillhole_styles   rock, color                              (color: "#RRGGBB")
    drillhole_translation  table_name, field_name, code, description   (e.g. geology / rock / SH / Shale)

plus the drilling records shared with the Drilling Failure Prediction system (drilling fields of the
collar and the geology, drillhole_run, _bit, _hole_size, _occurrence, _failure and the reference
tables drillhole_area, _formation, _rig, _measure) - see drilling_schema.py, made by ``prepare`` too.

Tables imported by the user keep their own column names: the hole id may be "BHID",
the depths "from" / "to", ... (recognised as when reading). A hole is saved in one
transaction: its collar row and all its survey / geology / assay rows together.
No Qt here.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass, field

import pandas as pd

from ..i18n import N_, every_translation, tr, trc
from . import drilling_schema as ds
from .colors import parse_color
from .drillholes import _AZI, _DIP, _FROM, _HOLE, _MAXDEPTH, _SDEPTH, _TO, _X, _Y, _Z, _find

COLLAR, SURVEY, GEOLOGY, ASSAY = "drillhole_collar", "drillhole_survey", "drillhole_geology", "drillhole_assay"
# Information about a hole kept in its collar row (all may be empty).
HOLE_STATES = ds.HOLE_STATES   # 1 In work ... 4 Failed, 5 Planned, 6 Completed; English: tr() where shown
# drillhole_collar.state is stored as this small number and always shown as the text
COLLAR_INFO = (("organization", "VARCHAR(100)"), ("drilling_manager", "VARCHAR(100)"),
               ("geology_surveyor", "VARCHAR(100)"), ("start_date", "VARCHAR(10)"), ("end_date", "VARCHAR(10)"),
               ("state", "SMALLINT"))
STYLES, TRANSLATION = "drillhole_styles", "drillhole_translation"
HOLE_TABLES = (SURVEY, GEOLOGY, ASSAY)        # the per-hole tables the manager edits, in tab order
_ROCK = ("rock", "lith", "lithology", "litho", "code")
_NOTE = ("note", "notes", "comment", "comments", "remarks")
_NOT_HOLE_TABLES = {STYLES, TRANSLATION, "drillhole_meta"} | {t.name for t in ds.REFERENCE_TABLES}
DRILLING_TABLES = tuple(t.name for t in ds.RECORD_TABLES)   # a hole's drilling records (Drillhole Manager tabs)
_NUMERIC = ("INT", "REAL", "DOUB", "FLOA", "DEC", "NUM")
_ROLE_ORDER = ("x", "y", "z", "max_depth", "depth", "azimuth", "dip", "from", "to", "rock", "note")


def _ddl(mysql: bool) -> dict[str, str]:
    opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if mysql else ""
    hid = "hole_id VARCHAR(64) NOT NULL"
    return {
        COLLAR: f"CREATE TABLE {COLLAR} ({hid} PRIMARY KEY, x DOUBLE, y DOUBLE, z DOUBLE, max_depth DOUBLE, "
                + ", ".join(f"{n} {t}" for n, t in COLLAR_INFO) + f"){opts}",
        SURVEY: f"CREATE TABLE {SURVEY} ({hid}, depth DOUBLE NOT NULL, azimuth DOUBLE, dip DOUBLE){opts}",
        GEOLOGY: f"CREATE TABLE {GEOLOGY} ({hid}, depth_from DOUBLE NOT NULL, depth_to DOUBLE NOT NULL, "
                 f"rock VARCHAR(64), note TEXT){opts}",
        ASSAY: f"CREATE TABLE {ASSAY} ({hid}, depth_from DOUBLE NOT NULL, depth_to DOUBLE NOT NULL, "
               f"gold DOUBLE, silver DOUBLE){opts}",
        STYLES: f"CREATE TABLE {STYLES} (rock VARCHAR(64) NOT NULL PRIMARY KEY, color VARCHAR(7) NOT NULL){opts}",
        TRANSLATION: f"CREATE TABLE {TRANSLATION} (table_name VARCHAR(64), field_name VARCHAR(64), "
                     f"code VARCHAR(64), description VARCHAR(255)){opts}",
    }


# Fields each table must have, as (role, name to add, type). A field counts as present when a
# column fills its role under any recognised name (hole id "BHID", depths "from" / "to", ...);
# role None: that exact name. Assay grade fields are the user's and are never added back.
_HOLE_ROLE = ("hole", "hole_id", "VARCHAR(64)")
_REQUIRED = {
    COLLAR: [_HOLE_ROLE, ("x", "x", "DOUBLE"), ("y", "y", "DOUBLE"), ("z", "z", "DOUBLE"),
             ("max_depth", "max_depth", "DOUBLE")] + [(None, n, t) for n, t in COLLAR_INFO],
    SURVEY: [_HOLE_ROLE, ("depth", "depth", "DOUBLE"), ("azimuth", "azimuth", "DOUBLE"), ("dip", "dip", "DOUBLE")],
    GEOLOGY: [_HOLE_ROLE, ("from", "depth_from", "DOUBLE"), ("to", "depth_to", "DOUBLE"),
              ("rock", "rock", "VARCHAR(64)"), ("note", "note", "TEXT")],
    ASSAY: [_HOLE_ROLE, ("from", "depth_from", "DOUBLE"), ("to", "depth_to", "DOUBLE")],
    STYLES: [(None, "rock", "VARCHAR(64)"), (None, "color", "VARCHAR(7)")],
    TRANSLATION: [(None, "table_name", "VARCHAR(64)"), (None, "field_name", "VARCHAR(64)"),
                  (None, "code", "VARCHAR(64)"), (None, "description", "VARCHAR(255)")],
}


def hex_color(rgb) -> str:
    """'#rrggbb' from RGB floats 0..1."""
    return "#" + "".join(f"{max(0, min(255, round(float(c) * 255))):02x}" for c in rgb[:3])


def rgb_of(text: str, default=(0.6, 0.6, 0.6)) -> tuple[float, float, float]:
    """RGB floats from '#rrggbb' (default for anything else)."""
    t = str(text or "").strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{6}", t):
        return tuple(int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return default


@dataclass
class TableInfo:
    """A drillhole table as it is in the database: its columns and which column is what."""

    name: str
    columns: list[str]                       # in table order
    numeric: dict[str, bool]                 # column -> holds numbers
    hole: str | None                         # the hole id column
    roles: dict[str, str] = field(default_factory=dict)   # x / y / z / max_depth / depth / azimuth / dip /
                                                          # from / to / rock -> column

    @property
    def fields(self) -> list[str]:
        """The columns edited for a hole (all but the hole id), in a natural order: X, Y, Z,
        length / depth, azimuth, dip / from, to, rock - then any other columns as stored."""
        known = [self.roles[k] for k in _ROLE_ORDER if k in self.roles]
        return known + [c for c in self.columns if c != self.hole and c not in known]

    @property
    def depth_columns(self) -> list[str]:
        return [self.roles[k] for k in ("from", "to", "depth") if k in self.roles]

    @property
    def value_fields(self) -> list[str]:
        """Fields other than the hole id and the depths (assays, rock, azimuth, ...)."""
        return [c for c in self.fields if c not in self.depth_columns]


@dataclass
class HoleData:
    """One hole as edited: its collar values and its rows in each per-hole table
    (DataFrames with the table's own column names, without the hole id column)."""

    collar: dict
    tables: dict[str, pd.DataFrame]


def _role_columns(name: str, columns: list[str]) -> tuple[str | None, dict[str, str]]:
    hole = _find(columns, _HOLE, False, "hole id")
    roles: dict[str, str] = {}

    def put(key, names):
        c = _find([x for x in columns if x != hole], names, False, key)
        if c is not None and c not in roles.values():
            roles[key] = c

    if name == COLLAR:
        for key, names in (("x", _X), ("y", _Y), ("z", _Z), ("max_depth", _MAXDEPTH)):
            put(key, names)
    elif name == SURVEY:
        for key, names in (("depth", _SDEPTH), ("azimuth", _AZI), ("dip", _DIP)):
            put(key, names)
    else:
        put("from", _FROM)
        put("to", _TO)
        if name == GEOLOGY:
            put("rock", _ROCK)
            put("note", _NOTE)
    return hole, roles


def _num(v):
    """A float, or None for blank / not a number."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip().replace(",", "")
        if not v:
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _text(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


class DrillholeStore:
    """Read and change the drillholes of a project database (a database ``Target``)."""

    def __init__(self, target):
        self.target = target

    # --- tables ----------------------------------------------------------------------------
    def prepare(self) -> list[str]:
        """Make what is missing - never drops a table, a column or a row:
        * missing tables are created with the standard columns;
        * existing tables get the fields they lack (see _REQUIRED), as empty columns;
        * a drillhole_styles from Surpac (code, graphics_colour, ...) gets rock / color
          filled where they are empty;
        * the hole id columns get an index (a hole's rows found without reading whole tables).
        Returns what was done, for the user ([] when nothing was missing)."""
        notes = []
        with self.target.connect() as db:
            ddl = _ddl(db.is_mysql)
            for name in (COLLAR, SURVEY, GEOLOGY, ASSAY, STYLES, TRANSLATION):
                stored = db.table_name(name)
                if stored is None:
                    db.execute(ddl[name])
                    db.con.commit()
                    notes.append(tr("made table {name}", name=name))
                else:
                    notes += self._add_missing_fields(db, stored, name)
            n = self._fill_styles(db)
            if n:
                notes.append(tr("filled rock / color of {n} {table} rows from the Surpac colours", n=n, table=STYLES))
            notes += ds.prepare(db)                    # the drilling records + predefined measures and rigs
            for t in self._hole_tables(db):
                info = self._info(db, t)
                if info.hole:
                    db.ensure_index(t, f"idx_{t}_hole", [info.hole])
            db.con.commit()
        return notes

    def _add_missing_fields(self, db, table: str, standard: str) -> list[str]:
        info = self._info(db, table)
        low = {c.lower() for c in info.columns}
        notes = []
        for role, name, kind in _REQUIRED[standard]:
            present = (info.hole is not None if role == "hole" else
                       role in info.roles if role is not None else name.lower() in low)
            if not present and name.lower() not in low:
                db.execute(f"ALTER TABLE {db.q(table)} ADD COLUMN {db.q(name)} {kind}")
                db.con.commit()
                notes.append(tr("added field {name} to {table}", name=name, table=table))
        return notes

    def _fill_styles(self, db) -> int:
        """Rows of a Surpac style table for geology rock codes: rock / color filled from
        code / graphics_colour where they are empty (nothing else is changed)."""
        cols = {c.lower(): c for c in db.columns(STYLES)}
        colour = next((cols[c] for c in ("graphics_colour", "plotting_colour", "line_colour", "colour")
                       if c in cols), None)
        if "code" not in cols or colour is None:
            return 0
        where = f"({db.q('rock')} IS NULL OR {db.q('rock')} = '')"
        params = []
        if "table_name" in cols and "field_name" in cols:
            where += (f" AND LOWER({db.q(cols['table_name'])}) = 'geology' AND LOWER({db.q(cols['field_name'])}) "
                      f"IN ({', '.join('?' * len(_ROCK))})")
            params = list(_ROCK)
        rows = db.query(f"SELECT {db.q(cols['code'])}, {db.q(colour)} FROM {STYLES} WHERE {where}", params)
        n = 0
        with db.transaction():
            for code, spec in rows:
                if _text(code):
                    db.execute(f"UPDATE {STYLES} SET {db.q('rock')} = ?, {db.q('color')} = ? "
                               f"WHERE {db.q(cols['code'])} = ? AND {where}",
                               [_text(code), hex_color(parse_color(str(spec))), code] + params)
                    n += 1
        return n

    def _hole_tables(self, db) -> list[str]:
        """Every drillhole_ table holding rows per hole (not styles / translation)."""
        out = []
        for t in db.tables():
            low = t.lower()
            if low.startswith("drillhole_") and low not in _NOT_HOLE_TABLES:
                out.append(t)
        return out

    def _info(self, db, table: str) -> TableInfo:
        types = db.column_types(table)
        cols = list(types)
        hole, roles = _role_columns(table.lower(), cols)
        numeric = {c: any(k in t for k in _NUMERIC) for c, t in types.items()}
        return TableInfo(table, cols, numeric, hole, roles)

    def info(self, table: str) -> TableInfo | None:
        with self.target.connect(create=False) as db:
            name = db.table_name(table)
            return self._info(db, name) if name else None

    # --- holes -----------------------------------------------------------------------------
    def hole_ids(self) -> list[str]:
        with self.target.connect(create=False) as db:
            info = self._info(db, COLLAR)
            rows = db.query(f"SELECT {db.q(info.hole)} FROM {COLLAR}")
        return sorted((str(r[0]) for r in rows if r[0] is not None), key=_natural)

    def load_hole(self, hid: str) -> HoleData:
        with self.target.connect(create=False) as db:
            ci = self._info(db, COLLAR)
            rows = db.query(f"SELECT * FROM {COLLAR} WHERE {db.q(ci.hole)} = ?", (hid,))
            collar = dict(zip(ci.columns, rows[0])) if rows else {}
            collar.pop(ci.hole, None)
            tables = {}
            for t in HOLE_TABLES + tuple(d for d in DRILLING_TABLES if db.has_table(d)):
                ti = self._info(db, t)
                cur = db.con.cursor()
                try:
                    cur.execute(db._sql(f"SELECT * FROM {t} WHERE {db.q(ti.hole)} = ?"), (hid,))
                    names = [d[0] for d in cur.description]
                    df = pd.DataFrame.from_records([tuple(r) for r in cur.fetchall()], columns=names)
                finally:
                    cur.close()
                df = df.drop(columns=[ti.hole])
                key = ti.depth_columns[:1]
                if key and len(df):
                    df = df.sort_values(key, kind="stable").reset_index(drop=True)
                tables[t] = df
        return HoleData(collar, tables)

    def add_hole(self, hid: str, collar: dict) -> None:
        hid = str(hid).strip()
        if not hid:
            raise ValueError(tr("Type a hole id"))
        with self.target.connect() as db:
            ci = self._info(db, COLLAR)
            if db.scalar(f"SELECT COUNT(*) FROM {COLLAR} WHERE {db.q(ci.hole)} = ?", (hid,)):
                raise ValueError(tr("Hole {hid} is already in the database", hid=hid))
            cols = [ci.hole] + [c for c in collar if c in ci.columns and c != ci.hole]
            vals = [hid] + [self._value(ci, c, collar[c]) for c in cols[1:]]
            with db.transaction():
                db.execute(f"INSERT INTO {COLLAR} ({', '.join(db.q(c) for c in cols)}) "
                           f"VALUES ({', '.join('?' * len(cols))})", vals)

    def remove_hole(self, hid: str) -> int:
        """Delete the hole from every per-hole table; returns the number of rows removed."""
        n = 0
        with self.target.connect() as db, db.transaction():
            for t in self._hole_tables(db):
                info = self._info(db, t)
                if info.hole:
                    n += int(db.scalar(f"SELECT COUNT(*) FROM {db.q(t)} WHERE {db.q(info.hole)} = ?", (hid,)) or 0)
                    db.execute(f"DELETE FROM {db.q(t)} WHERE {db.q(info.hole)} = ?", (hid,))
        return n

    def save_hole(self, hid: str, data: HoleData, new_id: str | None = None) -> str:
        """Write the hole: its collar values and, for each table in ``data.tables``, exactly
        those rows (sorted by depth). ``new_id`` renames the hole in every table. All in one
        transaction. Returns the hole id it is saved under."""
        new_id = str(new_id).strip() if new_id is not None else hid
        if not new_id:
            raise ValueError(tr("The hole id cannot be empty"))
        with self.target.connect() as db:
            ci = self._info(db, COLLAR)
            if new_id != hid and db.scalar(f"SELECT COUNT(*) FROM {COLLAR} WHERE {db.q(ci.hole)} = ?", (new_id,)):
                raise ValueError(tr("Hole {hid} is already in the database", hid=new_id))
            with db.transaction():
                if new_id != hid:
                    for t in self._hole_tables(db):
                        info = self._info(db, t)
                        if info.hole:
                            db.execute(f"UPDATE {db.q(t)} SET {db.q(info.hole)} = ? WHERE {db.q(info.hole)} = ?",
                                       (new_id, hid))
                sets = [c for c in data.collar if c in ci.columns and c != ci.hole]
                if sets:
                    db.execute(f"UPDATE {COLLAR} SET {', '.join(f'{db.q(c)} = ?' for c in sets)} "
                               f"WHERE {db.q(ci.hole)} = ?",
                               [self._value(ci, c, data.collar[c]) for c in sets] + [new_id])
                for t, df in data.tables.items():
                    self._write_rows(db, self._info(db, t), new_id, df)
        return new_id

    def _write_rows(self, db, ti: TableInfo, hid: str, df: pd.DataFrame) -> int:
        """Replace the hole's rows in one per-hole table with ``df`` (sorted by depth); inside
        the caller's transaction. Returns the rows written."""
        t = ti.name
        db.execute(f"DELETE FROM {db.q(t)} WHERE {db.q(ti.hole)} = ?", (hid,))
        cols = [c for c in df.columns if c in ti.columns and c != ti.hole]
        key = [c for c in ti.depth_columns[:1] if c in cols]
        if key:
            df = df.assign(_k=[_num(v) for v in df[key[0]]]).sort_values("_k", kind="stable")
        rows = [[hid] + [self._value(ti, c, v) for c, v in zip(cols, r)]
                for r in df[cols].itertuples(index=False, name=None)]
        if rows:
            names = ", ".join(db.q(c) for c in [ti.hole] + cols)
            db.executemany(f"INSERT INTO {db.q(t)} ({names}) VALUES ({', '.join('?' * (len(cols) + 1))})", rows)
        return len(rows)

    @staticmethod
    def _value(info: TableInfo, col: str, v):
        if info.name.lower() == COLLAR and col.lower() == "state":
            code = state_code(v)
            return code if info.numeric.get(col) or code is None else HOLE_STATES[code]
        return _num(v) if info.numeric.get(col) else _text(v)

    # --- rock colours ----------------------------------------------------------------------
    def styles(self) -> dict[str, str]:
        """Rock code -> '#rrggbb' from drillhole_styles."""
        with self.target.connect(create=False) as db:
            if not db.has_table(STYLES):
                return {}
            return {str(r): str(c) for r, c in db.query(f"SELECT rock, color FROM {STYLES}") if r is not None}

    def set_style(self, rock: str, color: str) -> None:
        rock = _text(rock)
        if not rock:
            raise ValueError("Type a rock code")
        with self.target.connect() as db, db.transaction():
            db.execute(f"DELETE FROM {STYLES} WHERE rock = ?", (rock,))
            db.execute(f"INSERT INTO {STYLES} (rock, color) VALUES (?, ?)", (rock, color.lower()))

    def set_description(self, rock: str, description: str) -> None:
        """The description of a rock code (drillhole_translation, table geology / field rock)."""
        rock = _text(rock)
        if not rock:
            raise ValueError("Type a rock code")
        with self.target.connect() as db, db.transaction():
            db.execute(f"DELETE FROM {TRANSLATION} WHERE LOWER(table_name) = 'geology' AND LOWER(field_name) = "
                       "'rock' AND code = ?", (rock,))
            if _text(description):
                db.execute(f"INSERT INTO {TRANSLATION} (table_name, field_name, code, description) "
                           "VALUES ('geology', 'rock', ?, ?)", (rock, _text(description)))

    def export_excel(self, path) -> dict[str, int]:
        """All holes to an Excel workbook with the sheets collar, geology, survey and assay, plus
        the drilling records (run, hole_size, bit, occurrence, failure) that have rows (every column
        as stored, sorted by hole and depth; header row frozen). Returns the
        number of rows per sheet. Raises PermissionError when the file is open in Excel."""
        import openpyxl  # noqa: F401 - the Excel writer pandas uses (imported here so it is bundled)
        from openpyxl.utils import get_column_letter

        frames: dict[str, pd.DataFrame] = {}
        with self.target.connect(create=False) as db:
            extra = tuple((t.short, t.name) for t in ds.RECORD_TABLES)
            for sheet, table in (("collar", COLLAR), ("geology", GEOLOGY), ("survey", SURVEY), ("assay", ASSAY)) + extra:
                name = db.table_name(table)
                if (sheet, table) in extra and (name is None or not db.scalar(f"SELECT COUNT(*) FROM {db.q(name)}")):
                    continue                           # the drilling sheets only when they have rows
                if name is None:
                    frames[sheet] = pd.DataFrame()
                    continue
                info = self._info(db, name)
                df = db.read_frame(name)
                if info.hole and len(df):                  # by hole (WD2 before WD10), then depth
                    holes = df[info.hole].tolist()
                    dcol = info.depth_columns[0] if info.depth_columns else None
                    depth = [_num(v) for v in df[dcol]] if dcol else [0.0] * len(df)
                    order = sorted(range(len(df)), key=lambda i: (_natural(holes[i]), depth[i] or 0.0))
                    df = df.iloc[order]
                for c in df.columns:
                    if table == COLLAR and c.lower() == "state":
                        df[c] = [state_text(v) or None for v in df[c]]     # shown as text, not the code
                frames[sheet] = df.reset_index(drop=True)
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for sheet, df in frames.items():
                df.to_excel(xw, sheet_name=sheet, index=False)
                ws = xw.sheets[sheet]
                ws.freeze_panes = "A2"
                for i, col in enumerate(df.columns, start=1):
                    width = max([len(str(col))] + [len(str(v)) for v in df[col].head(500) if v is not None])
                    ws.column_dimensions[get_column_letter(i)].width = min(60, max(8, width + 2))
        return {sheet: len(df) for sheet, df in frames.items()}

    def remove_style(self, rock: str) -> None:
        with self.target.connect() as db, db.transaction():
            db.execute(f"DELETE FROM {STYLES} WHERE rock = ?", (rock,))

    def rock_codes(self) -> list[tuple[str, str, int]]:
        """(code, description, intervals using it) for every rock code in the geology
        table, the styles and the translation table."""
        with self.target.connect(create=False) as db:
            counts: dict[str, int] = {}
            gi = self._info(db, GEOLOGY) if db.has_table(GEOLOGY) else None
            if gi is not None and "rock" in gi.roles:
                rc = db.q(gi.roles["rock"])
                for code, n in db.query(f"SELECT {rc}, COUNT(*) FROM {GEOLOGY} GROUP BY {rc}"):
                    if _text(code):
                        counts[_text(code)] = int(n)
            desc: dict[str, str] = {}
            if db.has_table(TRANSLATION):
                tt = db.read_frame(TRANSLATION)
                low = {c.lower(): c for c in tt.columns}
                if {"code", "description"} <= set(low):
                    m = pd.Series(True, index=tt.index)
                    if "field_name" in low:
                        m &= tt[low["field_name"]].astype(str).str.lower().isin(_ROCK)
                    desc = {str(c): str(d) for c, d in zip(tt.loc[m, low["code"]], tt.loc[m, low["description"]])}
            if db.has_table(STYLES):
                for (code,) in db.query(f"SELECT rock FROM {STYLES}"):
                    if _text(code):
                        counts.setdefault(_text(code), 0)
        return [(c, desc.get(c, ""), counts[c]) for c in sorted(counts, key=_natural)]

    # --- drilling records shared with the risk prediction (drilling_schema.py) -----------------
    def hole_areas(self) -> dict[str, str]:
        """Hole -> its drilling area ('' when none)."""
        with self.target.connect(create=False) as db:
            ci = self._info(db, COLLAR)
            col = next((c for c in ci.columns if c.lower() == "area_id"), None)
            if col is None:
                return {}
            return {str(h): (str(a) if a is not None else "")
                    for h, a in db.query(f"SELECT {db.q(ci.hole)}, {db.q(col)} FROM {COLLAR}") if h is not None}

    def reference(self, table: str) -> pd.DataFrame:
        """All rows of a reference table (drillhole_area, _formation, _rig, _measure)."""
        with self.target.connect(create=False) as db:
            df = ds.read_table(db, ds.table(table))
        key = ds.table(table).key[0]
        if table == ds.MEASURE.name or not len(df):
            return df
        return df.iloc[sorted(range(len(df)), key=lambda i: _natural(df[key].iloc[i]))].reset_index(drop=True)

    def save_reference(self, table: str, df: pd.DataFrame) -> None:
        with self.target.connect() as db:
            ds.save_reference(db, ds.table(table), df)

    def default_reference(self, table: str) -> pd.DataFrame:
        """The predefined rows of a reference table (measures, example rigs; empty for the others)."""
        t = ds.table(table)
        return pd.DataFrame(ds.default_reference().get(t.name, []), columns=t.names)

    def choices(self) -> dict[str, dict[str, str]]:
        """The codes of the columns filled from other tables: area_id, rig_model, formation, and the
        rock classes of the formation cards (main_rock, second_rock)."""
        with self.target.connect(create=False) as db:
            out = {"area_id": {a: a for a in ds.area_ids(db)}}
            for col, t in (("rig_model", ds.RIG), ("formation", ds.FORMATION)):
                name = db.table_name(t.name)
                rows = db.query(f"SELECT {db.q(col)} FROM {db.q(name)}") if name else []
                out[col] = {str(r[0]): str(r[0]) for r in sorted(rows, key=lambda r: _natural(r[0]))
                            if r[0] is not None}
            rocks = ds.rock_classes(db)
        out["main_rock"] = out["second_rock"] = rocks
        return out

    def rig(self, model: str | None) -> dict | None:
        """The specification of a rig model (None when it is not in drillhole_rig)."""
        if not _text(model):
            return None
        df = self.reference(ds.RIG.name)
        rows = df[df["rig_model"].astype(str) == str(model).strip()]
        return rows.iloc[0].to_dict() if len(rows) else None

    def measures(self) -> list[dict]:
        """The measure library (drillhole_measure rows, by expert order)."""
        df = self.reference(ds.MEASURE.name)
        if len(df):
            df = df.assign(_o=pd.to_numeric(df["expert_order"], errors="coerce").fillna(999)).sort_values(
                ["_o", "code"], kind="stable").drop(columns="_o")
        return df.to_dict("records")

    def export_drilling_csv(self, folder, areas: list[str] | None = None) -> dict[str, int]:
        """The drilling history as CSV files (the exchange format of the risk prediction)."""
        return ds.export_csv(self.target, folder, areas)

    def import_drilling_csv(self, folder, replace_areas: bool = False):
        """CSV files of a drilling history into the database: (rows per file, issues)."""
        self.prepare()
        return ds.import_csv(self.target, folder, replace_areas)

    # --- assay fields ------------------------------------------------------------------------
    def assay_fields(self) -> list[str]:
        info = self.info(ASSAY)
        return info.value_fields if info else []

    def add_field(self, name: str) -> None:
        name = _check_field_name(name)
        with self.target.connect() as db:
            if name.lower() in {c.lower() for c in db.columns(ASSAY)}:
                raise ValueError(tr("The assay table already has a field {name}", name=name))
            db.add_column(ASSAY, name, "real")
            db.con.commit()

    def rename_field(self, old: str, new: str) -> None:
        new = _check_field_name(new)
        with self.target.connect() as db:
            info = self._info(db, ASSAY)
            if old not in info.value_fields:
                raise ValueError(tr("{name} is not an assay field", name=old))
            if new.lower() in {c.lower() for c in info.columns if c != old}:
                raise ValueError(tr("The assay table already has a field {name}", name=new))
            db.rename_column(ASSAY, old, new)
            db.con.commit()

    def remove_field(self, name: str) -> None:
        with self.target.connect() as db:
            info = self._info(db, ASSAY)
            if name not in info.value_fields:
                raise ValueError(tr("{name} is not an assay field", name=name))
            db.drop_column(ASSAY, name)
            db.con.commit()


def state_code(value) -> int | None:
    """The stored code of a hole state from its code or its text, English or translated
    (None: empty / unknown)."""
    t = _text(value)
    if t is None:
        return None
    n = _num(t)
    if n is not None and n == int(n):
        n = ds.LEGACY_STATES.get(int(n), int(n))          # 6 "Completed" of an earlier version
        if n in HOLE_STATES:
            return n
    low = t.lower()
    if low == "completed":
        return ds.NO_ORE_HIT
    found = next((k for k, v in HOLE_STATES.items() if low in (v.lower(), tr(v).lower())), None)
    if found is None:                         # typed in another language (e.g. an Excel template)
        found = next((k for k, v in HOLE_STATES.items() if low in {x.lower() for x in every_translation(v)}), None)
    return found


def state_text(value) -> str:
    """The text shown for a stored hole state, in the current language ("" for empty, the
    value itself if unknown)."""
    code = state_code(value)
    return tr(HOLE_STATES[code]) if code is not None else (_text(value) or "")


def _check_field_name(name: str) -> str:
    name = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
        raise ValueError(tr("A field name starts with a letter and has only letters, digits and _ "
                            "(e.g. gold, ag_ppm)"))
    if name.lower() in {"hole_id", "depth_from", "depth_to", "from", "to"}:
        raise ValueError(tr("{name} is used for the hole and the depths", name=name))
    return name


def _natural(s: str):
    """Sort key: WD2 before WD10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


# --- checks before saving ------------------------------------------------------------------------
_DATE_ERRORS = {"start_date": N_("Collar: start date must be a date like 2026-03-31"),
                "end_date": N_("Collar: end date must be a date like 2026-03-31")}


def check_hole(collar_info: TableInfo, collar: dict, tables: dict[str, tuple[TableInfo, pd.DataFrame]],
               row_numbers: dict[str, list[int]] | None = None, context: dict | None = None):
    """(errors, warnings) for a hole about to be saved; errors stop the save.

    ``tables``: table -> (its TableInfo, the rows as edited). ``row_numbers``: table -> the
    number shown for each row (e.g. its row in an Excel sheet); else 1, 2, ... ``context``: for
    the drilling records, e.g. max_rpm of the hole's rig."""
    errors, warnings = [], []
    label = {SURVEY: trc("drillhole", "Survey"), GEOLOGY: tr("Geology"), ASSAY: tr("Assay")}
    for key in ("x", "y", "z"):
        c = collar_info.roles.get(key)
        if c and _num(collar.get(c)) is None:
            errors.append(tr("Collar: {field} must be a number", field=c))
    low = {c.lower(): c for c in collar_info.columns}
    dates = {}
    for name in ("start_date", "end_date"):
        v = _text(collar.get(low.get(name, name)))
        if v is not None:
            try:
                dates[name] = _dt.date.fromisoformat(v)
            except ValueError:
                errors.append(tr(_DATE_ERRORS[name]))
    if len(dates) == 2 and dates["end_date"] < dates["start_date"]:
        errors.append(tr("Collar: the end date is before the start date"))
    state = _text(collar.get(low.get("state", "state")))
    if state is not None and state_code(state) is None:
        errors.append(tr("Collar: state must be one of: {states}",
                         states=", ".join(tr(v) for v in HOLE_STATES.values())))
    md_col = collar_info.roles.get("max_depth")
    max_depth = _num(collar.get(md_col)) if md_col else None
    if max_depth is not None and (max_depth < 0 or max_depth == 0 and state_code(state) not in (ds.PLANNED,
                                                                                                ds.IN_WORK)):
        errors.append(tr("Collar: the hole length must be more than 0"))
    drilling = {c.name: collar.get(low[c.name]) for c in ds.COLLAR_ADDED if c.name in low}
    if drilling:                                  # the drilling fields: ranges and codes
        _, found = ds.validate(ds.COLLAR, pd.DataFrame([drilling]))
        for i in found:
            (errors if i.level == "error" else warnings).append(tr("Collar: {message}", message=i.message))
    for t, (info, df) in tables.items():
        if t in DRILLING_TABLES or t == GEOLOGY:      # the drilling records (drilling_schema.py)
            part = df if t != GEOLOGY else df[[c.name for c in ds.GEOLOGY_ADDED if c.name in df.columns]]
            _, found = ds.validate(ds.table(t), part, {"max_depth": max_depth, **(context or {})})
            shown = (row_numbers or {}).get(t)
            for i in found:
                if shown and i.row is not None and i.row < len(shown):
                    i.row = shown[i.row] - 1
                (errors if i.level == "error" else warnings).append(i.text())
            if t in DRILLING_TABLES:
                continue
        name = label.get(t, t)
        r = info.roles
        numbers = (row_numbers or {}).get(t) or list(range(1, len(df) + 1))
        if t == SURVEY:
            seen = set()
            for i, row in zip(numbers, df.to_dict("records")):
                d = _num(row.get(r.get("depth")))
                if d is None or d < 0:
                    errors.append(tr("{name} row {i}: depth must be a number, 0 or more", name=name, i=i))
                elif d in seen:
                    errors.append(tr("{name} row {i}: depth {d:g} is in the table twice", name=name, i=i, d=d))
                else:
                    seen.add(d)
                    if max_depth is not None and d > max_depth:
                        warnings.append(tr("{name} row {i}: depth {d:g} is past the end of the hole ({max_depth:g} m)",
                                         name=name, i=i, d=d, max_depth=max_depth))
                a = _num(row.get(r.get("azimuth")))
                if "azimuth" in r and (a is None or not 0 <= a <= 360):
                    errors.append(tr("{name} row {i}: azimuth must be 0 to 360", name=name, i=i))
                p = _num(row.get(r.get("dip")))
                if "dip" in r and (p is None or not 0 <= p <= 180):
                    errors.append(tr("{name} row {i}: dip must be 0 (straight down) to 180 (straight up)",
                                     name=name, i=i))
            continue
        if "from" not in r or "to" not in r:
            continue
        spans = []
        for i, row in zip(numbers, df.to_dict("records")):
            a, b = _num(row.get(r["from"])), _num(row.get(r["to"]))
            if a is None or b is None:
                errors.append(tr("{name} row {i}: From and To must be numbers", name=name, i=i))
                continue
            if a < 0 or b <= a:
                errors.append(tr("{name} row {i}: From must be 0 or more and less than To ({a:g} - {b:g})",
                                 name=name, i=i, a=a, b=b))
                continue
            if max_depth is not None and b > max_depth + 1e-9:
                warnings.append(tr("{name} row {i}: {b:g} m is past the end of the hole ({max_depth:g} m)",
                                    name=name, i=i, b=b, max_depth=max_depth))
            for c in info.value_fields:
                v = row.get(c)
                if info.numeric.get(c) and _text(v) is not None and _num(v) is None:
                    errors.append(tr("{name} row {i}: {field} must be a number (or empty)", name=name, i=i, field=c))
            spans.append((a, b, i))
        spans.sort()
        for (a1, b1, i1), (a2, b2, i2) in zip(spans, spans[1:]):
            if a2 < b1 - 1e-9:
                warnings.append(tr("{name}: rows {i1} and {i2} overlap ({a1:g}-{b1:g} and {a2:g}-{b2:g})",
                                    name=name, i1=i1, i2=i2, a1=a1, b1=b1, a2=a2, b2=b2))
    if any(t in DRILLING_TABLES for t in tables):     # across the hole's drilling records
        values = {c.name: collar.get(low.get(c.name, c.name)) for c in ds.COLLAR.cols}
        values["state"] = state_code(state)
        for i in ds.check_drilling(values, {t: df for t, (_info, df) in tables.items()}):
            (errors if i.level == "error" else warnings).append(i.text())
    return errors, warnings
