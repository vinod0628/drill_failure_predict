"""Reading and writing the drilling records in the project database (SQLite or MySQL).

The drillhole records are the drillhole tables shared with GeoViewer and 3D MPM Studio (io/drilling_schema.py):
they are read into memory in the exchange format (dip negative downward, final_depth, result) and written
back as stored, keeping the columns only GeoViewer writes (organization, assays, ...). The risk system's own
data is in risk_rock, risk_rock_code and risk_warning. Built on GeoViewer's portable
:class:`~dfpredict.io.database.Database` (``?`` placeholders, quoted names, one transaction per save).
``RiskStore.prepare`` only ADDS missing tables and columns - it never drops anything, like the other stores
of the project database. No Qt here.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..i18n import tr
from ..io import drilling_schema as ds
from ..io.database import Target, _db_value
from . import codes
from .schema import ALL_TABLES, BY_NAME, COLLAR, FAILURE, HOLE_TABLES, OCCURRENCE, RUN, SHARED_PREFIX, WARNING, Table

# CSV column names recognised on import (spec 3.1 writes them in capitals: HOLE_ID, FROM, TO ...).
ALIASES = {
    "hole_id": ("hole_id", "holeid", "hole", "bhid", "dhid"),
    "depth_from": ("depth_from", "from", "from_m", "mfrom", "depthfrom", "run_from"),
    "depth_to": ("depth_to", "to", "to_m", "mto", "depthto", "run_to"),
    "rock": ("rock", "rock_code", "lith", "lithology"),
    "size": ("size", "hole_size", "size_code"),
    "diameter": ("diameter", "diameter_mm", "dia"),
    "bit_type": ("bit_type", "bit", "type"),
    "type": ("type", "occurrence", "occ_type"),
    "fail_code": ("fail_code", "failcode", "code"),
    "pump_press": ("pump_press", "pressure", "pump_pressure"),
    "flow_in": ("flow_in", "flow", "flowin"),
    "return_pct": ("return_pct", "return", "returns"),
    "result": ("result", "status"),
    "area_id": ("area_id", "area"),
    "planned_depth": ("planned_depth", "plan_depth", "target_depth"),
    "final_depth": ("final_depth", "max_depth", "eoh", "depth_reached"),
}
CSV_FILES = {t.name: f"{t.name}.csv" for t in ALL_TABLES}


def _norm(name: str) -> str:
    return re.sub(r"[\s()\[\]\-.]", "", str(name)).lower()


def _clean(v):
    """A value for the database: NaN / '' -> NULL (blank stays blank, never 0)."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    if isinstance(v, float) and math.isnan(v):
        return None
    return _db_value(v)


@dataclass
class Issue:
    """A problem found by the entry checks (spec 7.3)."""

    level: str              # "error" (not saved) or "warning" (saved, the user is told)
    table: str
    row: int | None         # 0-based row in the table being checked
    column: str | None
    message: str

    def text(self) -> str:
        where = f" {tr('row')} {self.row + 1}" if self.row is not None else ""
        return f"{self.table}{where}: {self.message}"


@dataclass
class RiskData:
    """All drilling records in memory (for training, statistics and prediction)."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def hole(self, name: str, hole_id: str) -> pd.DataFrame:
        df = self.tables[name]
        return df[df["hole_id"] == hole_id]

    @property
    def collar(self) -> pd.DataFrame:
        return self.tables["collar"]


class RiskStore:
    """The drilling records of one project database: the shared drillhole tables (io/drilling_schema.py, read
    and written in the exchange format) and the risk system's own risk_* tables."""

    def __init__(self, target: Target):
        self.target = target

    # --- set-up --------------------------------------------------------------------------------
    def prepare(self) -> list[str]:
        """Make missing tables / columns: the shared drillhole tables (GeoViewer's drillhole store, which also
        moves the records of an earlier version's risk_* tables in) and the risk system's own tables
        (+ an index on each hole id, the predefined rock properties and rock codes in empty tables).
        Returns what was done."""
        from ..io.drillhole_store import DrillholeStore

        notes = DrillholeStore(self.target).prepare()
        with self.target.connect() as db:
            existing = {t.lower(): t for t in db.tables()}
            for t in ALL_TABLES:
                if t.shared:
                    continue
                name = existing.get(t.sql_name.lower())
                if name is None:
                    db.execute(self._ddl(t, db.is_mysql))
                    notes.append(f"made table {t.sql_name}")
                    name = t.sql_name
                else:
                    have = {c.lower() for c in db.columns(name)}
                    for c in t.cols:
                        if c.name.lower() not in have:
                            db.execute(f"ALTER TABLE {db.q(name)} ADD COLUMN {db.q(c.name)} {c.sql}")
                            notes.append(f"added {name}.{c.name}")
                if t.per_hole:
                    db.ensure_index(name, f"ix_{t.sql_name}_hole", ["hole_id"])
            if not db.is_mysql:
                db.con.commit()
            # the predefined reference data, in a table that is still empty (user edits are never replaced)
            for name, rows in default_reference().items():
                t = BY_NAME[name]
                if t.shared:
                    continue                          # measures, rigs: filled by the shared set-up
                if not db.scalar(f"SELECT COUNT(*) FROM {db.q(t.sql_name)}", default=0):
                    with db.transaction():
                        self._insert(db, t, pd.DataFrame(rows))
                    notes.append(f"filled {t.sql_name} with {len(rows)} predefined rows")
        return notes

    @staticmethod
    def _ddl(t: Table, mysql: bool) -> str:
        q = (lambda n: f"`{n}`") if mysql else (lambda n: f'"{n}"')      # e.g. "group" is an SQL keyword
        cols = []
        for c in t.cols:
            null = " NOT NULL" if c.required and c.name in t.key else ""
            cols.append(f"{q(c.name)} {c.sql}{null}")
        if len(t.key) == 1 and not t.per_hole or t.name == "failure":
            cols.append(f"PRIMARY KEY ({q(t.key[0])})")
        opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if mysql else ""
        return f"CREATE TABLE {t.sql_name} ({', '.join(cols)}){opts}"

    # --- reading --------------------------------------------------------------------------------
    @staticmethod
    def _names(t: Table) -> list[str]:
        """The columns of a table's records in memory (collar: + state, the hole's stored status)."""
        return t.names + (["state"] if t is COLLAR else [])

    def _read(self, db, t: Table, hole_ids=None) -> pd.DataFrame:
        if t.shared:
            df = ds.to_exchange(ds.table(t.sql_name), ds.read_table(db, ds.table(t.sql_name), hole_ids))
            return self._typed(t, df.reindex(columns=self._names(t)))
        name = db.table_name(t.sql_name)
        if name is None:
            return pd.DataFrame(columns=t.names)
        sql = f"SELECT {', '.join(db.q(c) for c in t.names)} FROM {db.q(name)}"
        params: tuple = ()
        if hole_ids is not None and t.per_hole:
            ids = [str(h) for h in hole_ids]
            sql += f" WHERE hole_id IN ({', '.join('?' * len(ids))})" if ids else " WHERE 1 = 0"
            params = tuple(ids)
        return self._typed(t, pd.DataFrame.from_records(db.query(sql, params), columns=t.names))

    def read(self, table: str | Table, hole_id: str | None = None) -> pd.DataFrame:
        t = BY_NAME[table] if isinstance(table, str) else table
        with self.target.connect(create=False) as db:
            ids = [hole_id] if hole_id is not None and (t.per_hole or t is COLLAR) else None
            return self._read(db, t, ids)

    @staticmethod
    def _typed(t: Table, df: pd.DataFrame) -> pd.DataFrame:
        for c in t.cols:
            if c.kind in ("real", "int") and c.name in df:
                df[c.name] = pd.to_numeric(df[c.name], errors="coerce")
            elif c.name in df:
                df[c.name] = df[c.name].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v))
                                            else ds._code_text(v) if c.kind == "code" else str(v))
        sort = [k for k in ("hole_id", "depth_from", "depth") if k in df]
        if sort:
            df = df.sort_values(sort, kind="stable").reset_index(drop=True)
        return df

    def load_all(self) -> RiskData:
        """Every table at once (one connection)."""
        out = {}
        with self.target.connect(create=False) as db:
            for t in ALL_TABLES:
                out[t.name] = self._read(db, t)
        apply_reference(out)
        return RiskData(out)

    def hole(self, hole_id: str) -> dict[str, pd.DataFrame]:
        with self.target.connect(create=False) as db:
            out = {"collar": self._read(db, COLLAR, [hole_id])}
            for t in HOLE_TABLES:
                out[t.name] = self._read(db, t, [hole_id])
        return out

    def summary(self) -> dict:
        """Counts for the dashboard: holes by result, failures by code, occurrences by type."""
        with self.target.connect(create=False) as db:
            def grouped(table, col):
                name = db.table_name(table)
                if name is None:
                    return {}
                return {str(k): int(n) for k, n in db.query(
                    f"SELECT {db.q(col)}, COUNT(*) FROM {db.q(name)} GROUP BY {db.q(col)}") if k is not None}
            result: dict[str, int] = {}
            for state, n in grouped(ds.COLLAR.name, "state").items():
                r = ds.result_of_state(state)
                if r is not None:
                    result[r] = result.get(r, 0) + n
            return {"result": result, "fail_code": grouped(FAILURE.sql_name, "fail_code"),
                    "occurrence": grouped(OCCURRENCE.sql_name, "type"), "area": grouped(COLLAR.sql_name, "area_id"),
                    "runs": sum(grouped(RUN.sql_name, "hole_id").values()) if db.table_name(RUN.sql_name) else 0}

    # --- writing --------------------------------------------------------------------------------
    def _rows(self, t: Table, df: pd.DataFrame) -> list[list]:
        cols = t.names
        df = df.reindex(columns=cols)
        return [[_clean(v) for v in row] for row in df.itertuples(index=False, name=None)]

    @staticmethod
    def _stored(db, t: Table, df: pd.DataFrame) -> pd.DataFrame:
        """Records of a shared table (exchange format, this program's columns) -> as stored. The collar's
        result becomes the state (a completed hole keeps 'with / without ore hit' stored by GeoViewer)."""
        sch = ds.table(t.sql_name)
        df = df[[c for c in df.columns if c in t.names]].copy()      # only the columns the risk system knows
        cur = {}
        if t is COLLAR and len(df):
            name = db.table_name(sch.name)
            sc = ds.stored_columns(db, name)
            if "state" in sc:
                cur = {str(h): s for h, s in db.query(f"SELECT {db.q(sc['hole_id'])}, {db.q(sc['state'])} "
                                                      f"FROM {db.q(name)}")}
        out = ds.from_exchange(sch, df, cur)
        return out[[c for c in out.columns if sch.has(c)]]

    def _write_shared(self, db, t: Table, df: pd.DataFrame, hole_ids=None) -> None:
        """Replace the rows of these holes (all holes in ``df`` when None) in a shared per-hole table. Columns the
        risk system does not know keep their values in the rows with the same depth."""
        sch = ds.table(t.sql_name)
        rows = self._stored(db, t, df) if len(df) else pd.DataFrame(columns=["hole_id"])
        holes = [str(h) for h in (hole_ids if hole_ids is not None else rows["hole_id"].dropna().unique())]
        name = db.table_name(sch.name)
        sc = ds.stored_columns(db, name)
        others = [c for c in db.columns(name) if c not in {sc[k] for k in rows.columns if k in sc} and c != sc["hole_id"]]
        if others:                                   # keep them: hole by hole
            for hid in holes:
                g = rows[rows["hole_id"].astype(str) == hid]
                ds.write_hole_rows(db, sch, hid, g.drop(columns="hole_id"))
            return
        for part in ds._chunks(holes, 400):
            db.execute(f"DELETE FROM {db.q(name)} WHERE {db.q(sc['hole_id'])} IN ({', '.join('?' * len(part))})", part)
        ds.insert_rows(db, sch, rows[rows["hole_id"].astype(str).isin(set(holes))], name)

    def save_hole(self, collar: dict, tables: dict[str, pd.DataFrame], old_id: str | None = None) -> None:
        """Save one hole: its collar row and all rows of the given per-hole tables, in one transaction
        (``old_id``: the hole's id before a rename). The fields GeoViewer keeps of the hole (organization, assays,
        notes of the core, ...) stay as they are."""
        hid = str(collar["hole_id"]).strip()
        with self.target.connect() as db:
            if old_id and old_id != hid and db.scalar(
                    f"SELECT COUNT(*) FROM {db.q(ds.COLLAR.name)} WHERE hole_id = ?", (hid,), default=0):
                raise ValueError(tr("Hole {hid} is already in the database", hid=hid))
            with db.transaction():
                if old_id and old_id != hid:          # every table follows the rename (also GeoViewer's own)
                    self._rename(db, old_id, hid)
                row = {k: v for k, v in collar.items() if k != "state"}
                rec = self._stored(db, COLLAR, pd.DataFrame([{**row, "hole_id": hid}])).iloc[0].to_dict()
                ds.upsert_collar(db, rec)
                for name, df in tables.items():
                    t = BY_NAME[name]
                    df = (df.copy() if df is not None else pd.DataFrame(columns=t.names))
                    df["hole_id"] = hid
                    self._write_shared(db, t, df, [hid])

    @staticmethod
    def _rename(db, old: str, new: str) -> None:
        for name in db.tables():
            low = name.lower()
            if low.startswith(SHARED_PREFIX) or low == WARNING.sql_name:
                sc = ds.stored_columns(db, name) if low.startswith(SHARED_PREFIX) else {"hole_id": "hole_id"}
                if "hole_id" in sc:
                    db.execute(f"UPDATE {db.q(name)} SET {db.q(sc['hole_id'])} = ? WHERE {db.q(sc['hole_id'])} = ?",
                               (new, old))

    def _insert(self, db, t: Table, df: pd.DataFrame) -> None:
        if t.shared:
            ds.insert_rows(db, ds.table(t.sql_name), self._stored(db, t, df))
            return
        rows = self._rows(t, df)
        if rows:
            marks = ", ".join("?" * len(t.names))
            db.executemany(f"INSERT INTO {db.q(t.sql_name)} ({', '.join(db.q(c) for c in t.names)}) VALUES ({marks})",
                           rows)

    def delete_hole(self, hole_id: str) -> None:
        """Remove the hole from every drillhole table (GeoViewer's assays too) and its warnings."""
        with self.target.connect() as db:
            with db.transaction():
                ds.remove_holes(db, [hole_id])
                if db.table_name(WARNING.sql_name):
                    db.execute(f"DELETE FROM {db.q(WARNING.sql_name)} WHERE hole_id = ?", (hole_id,))

    def save_reference(self, table: str | Table, df: pd.DataFrame) -> None:
        """Replace all rows of a reference table (areas, formation cards, rigs, rock properties, ...)."""
        t = BY_NAME[table] if isinstance(table, str) else table
        with self.target.connect() as db:
            if t.shared:
                ds.save_reference(db, ds.table(t.sql_name), df[[c for c in df.columns if c in t.names]])
                return
            with db.transaction():
                db.execute(f"DELETE FROM {db.q(t.sql_name)}")
                self._insert(db, t, df)

    def append_rows(self, table: str | Table, df: pd.DataFrame) -> None:
        t = BY_NAME[table] if isinstance(table, str) else table
        with self.target.connect() as db:
            with db.transaction():
                self._insert(db, t, df)

    def replace_all(self, data: dict[str, pd.DataFrame]) -> None:
        """Bulk write (demo data): reference tables and the risk system's own tables are replaced as a whole; the
        holes in ``data`` get its collar values and, in each per-hole table given, their rows replaced. Holes that
        are not in ``data`` are kept (e.g. GeoViewer's)."""
        with self.target.connect() as db:
            with db.transaction():
                holes = set()
                for name, df in data.items():
                    if (BY_NAME[name].per_hole or BY_NAME[name] is COLLAR) and "hole_id" in df:
                        holes |= set(df["hole_id"].dropna().astype(str))
                for name, df in data.items():
                    t = BY_NAME[name]
                    if t is COLLAR:
                        for rec in self._stored(db, t, df).to_dict("records"):
                            ds.upsert_collar(db, rec)
                    elif t.shared and t.per_hole:
                        self._write_shared(db, t, df, sorted(holes))
                    elif t.shared:
                        name_ = db.table_name(t.sql_name)
                        db.execute(f"DELETE FROM {db.q(name_)}")
                        ds.insert_rows(db, ds.table(t.sql_name), df[[c for c in df.columns if c in t.names]], name_)
                    else:
                        db.execute(f"DELETE FROM {db.q(t.sql_name)}")
                        self._insert(db, t, df)

    def save_warnings(self, rows: list[dict]) -> None:
        if rows:
            self.append_rows("warning", pd.DataFrame(rows))

    # --- CSV exchange (spec 3.4: CSV UTF-8) -------------------------------------------------------
    def export_csv(self, folder: str | Path, areas: list[str] | None = None) -> list[Path]:
        """All drilling records as CSV files: the shared tables in the exchange format (drilling_schema.export_csv),
        plus rock.csv, rock_code.csv and warning.csv."""
        folder = Path(folder)
        ds.export_csv(self.target, folder, areas)
        out = [folder / f"{t.short}.csv" for t in ds.SHARED_TABLES]
        with self.target.connect(create=False) as db:
            for t in ALL_TABLES:
                if not t.shared:
                    p = folder / CSV_FILES[t.name]
                    self._read(db, t).to_csv(p, index=False, encoding="utf-8")
                    out.append(p)
        return out

    def import_csv(self, folder: str | Path, replace: bool = False) -> tuple[dict[str, int], list]:
        """Import ``<table>.csv`` files of a folder. The shared tables (drilling_schema.import_csv): a hole in the
        files gets their collar values and its rows replaced in each file with rows for it; reference rows replace
        rows with the same key. ``replace``: the holes of the areas in collar.csv that are in none of the files are
        removed too. rock.csv / rock_code.csv replace rows with the same key. Returns rows read per table and the
        problems found (rows with errors are skipped)."""
        folder = Path(folder)
        counts, issues = ds.import_csv(self.target, folder, replace_areas=replace)
        frames = {}
        for t in ALL_TABLES:
            p = folder / CSV_FILES[t.name]
            if t.shared or t is WARNING or not p.exists():
                continue
            raw = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            df = map_columns(t, raw)
            bad, found = validate(t, df)
            issues += found
            frames[t.name] = df.drop(index=sorted(bad)).reset_index(drop=True) if bad else df
            counts[t.name] = len(frames[t.name])
        if frames:
            current = self.load_all()
            with self.target.connect() as db:
                with db.transaction():
                    for name, df in frames.items():
                        t = BY_NAME[name]
                        old = current[name]
                        keep = old[~old[t.key[0]].astype(str).isin(set(df[t.key[0]].astype(str)))]
                        db.execute(f"DELETE FROM {db.q(t.sql_name)}")
                        self._insert(db, t, pd.concat([keep, df], ignore_index=True))
        return counts, issues


# --- column mapping and entry checks ------------------------------------------------------------
def map_columns(t: Table, raw: pd.DataFrame) -> pd.DataFrame:
    """The columns of a CSV mapped to the table's columns (names matched ignoring case, spaces and
    the aliases above); values typed; unknown columns dropped."""
    normed = {_norm(c): c for c in raw.columns}
    out = pd.DataFrame(index=raw.index)
    for c in t.cols:
        src = None
        for alias in (c.name,) + ALIASES.get(c.name, ()):
            if _norm(alias) in normed:
                src = normed[_norm(alias)]
                break
        if src is None:
            out[c.name] = None
            continue
        s = raw[src]
        if c.kind in ("real", "int"):
            out[c.name] = pd.to_numeric(s.where(s.astype(str).str.strip() != "", None), errors="coerce")
        else:
            s = s.astype(str).str.strip().replace("", None)
            if c.kind in ("code", "yn") and c.choices and c.name not in ("area_id", "rig_model", "formation"):
                s = s.map(lambda v: _match_code(v, c.choices))
            out[c.name] = s
    return out


def _match_code(v, choices: dict):
    if v is None:
        return None
    s = str(v).strip()
    for k in choices:
        if s.lower() == str(k).lower():
            return k
    for k, text in choices.items():               # the text instead of the code ("Success", "Yes")
        if s.lower() == str(text).lower() or s.lower() == tr(text).lower():
            return k
    if s.lower() in ("yes", "true", "1") and "Y" in choices:
        return "Y"
    if s.lower() in ("no", "false", "0") and "N" in choices:
        return "N"
    if s.lower() in ("fail", "failure") and "failed" in choices:
        return "failed"
    return s


def validate(t: Table, df: pd.DataFrame, context: dict | None = None) -> tuple[set[int], list[Issue]]:
    """Entry checks (spec 7.3) on the rows of one table: required fields, physical ranges, codes,
    From < To, interval continuity (runs: RUN_FROM must equal the previous RUN_TO), overlaps,
    consistency. ``context``: the hole's collar (final / planned depth) and rig (max RPM).
    Returns the rows with errors (not to be saved) and every issue found."""
    context = context or {}
    bad: set[int] = set()
    issues: list[Issue] = []
    title = tr(t.title)

    def err(i, col, msg):
        bad.add(i)
        issues.append(Issue("error", title, i, col, msg))

    def warn(i, col, msg):
        issues.append(Issue("warning", title, i, col, msg))

    for i, row in enumerate(df.itertuples(index=False)):
        r = row._asdict()
        for c in t.cols:
            v = r.get(c.name)
            blank = v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())
            if blank:
                if c.required:
                    err(i, c.name, tr("{field} is required", field=tr(c.label)))
                continue
            if c.kind in ("real", "int"):
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    err(i, c.name, tr("{field}: '{value}' is not a number", field=tr(c.label), value=v))
                    continue
                if (c.lo is not None and x < c.lo) or (c.hi is not None and x > c.hi):
                    err(i, c.name, tr("{field} = {value} is outside {lo} - {hi} {unit}", field=tr(c.label), value=x,
                                      lo=c.lo, hi=c.hi, unit=c.unit))
            elif c.kind in ("code", "yn") and c.choices and str(v) not in c.choices:
                err(i, c.name, tr("{field}: unknown code '{value}'", field=tr(c.label), value=v))
        if t.interval:
            f, to = r.get("depth_from"), r.get("depth_to")
            if f is not None and to is not None and not (pd.isna(f) or pd.isna(to)) and float(to) <= float(f):
                err(i, "depth_to", tr("To ({to}) must be greater than From ({f})", to=to, f=f))
        if t is RUN and context.get("max_rpm") and r.get("rpm") is not None and not pd.isna(r.get("rpm")):
            if float(r["rpm"]) > float(context["max_rpm"]):
                err(i, "rpm", tr("RPM {rpm} is above the rig maximum {max}", rpm=r["rpm"], max=context["max_rpm"]))
        if t is RUN:
            ret, rem = r.get("return_pct"), str(r.get("remarks") or "").lower()
            if ret is not None and not pd.isna(ret) and float(ret) >= 100 and "loss" in rem:
                warn(i, "return_pct", tr("A loss is noted but the return is 100%: please check"))
    # intervals: continuity / overlaps, per hole
    if t.interval and len(df) and "hole_id" in df:
        for hid, g in df.reset_index().groupby("hole_id", sort=False):
            g = g.dropna(subset=["depth_from", "depth_to"]).sort_values("depth_from")
            prev_to, prev_i = None, None
            for _, row in g.iterrows():
                i = int(row["index"])
                if prev_to is not None:
                    gap = float(row["depth_from"]) - prev_to
                    if t is RUN and abs(gap) > 0.005:
                        err(i, "depth_from", tr("Run From {f} m must equal the previous run's To {t} m ({kind})",
                                                f=row["depth_from"], t=prev_to,
                                                kind=tr("gap") if gap > 0 else tr("overlap")))
                    elif t is not OCCURRENCE and gap < -0.005:
                        err(i, "depth_from", tr("Overlaps the interval above (it ends at {t} m)", t=prev_to))
                prev_to = max(prev_to or 0.0, float(row["depth_to"])) if t is OCCURRENCE else float(row["depth_to"])
                prev_i = i
            del prev_i
    depth = context.get("final_depth")
    if depth is not None and not pd.isna(depth) and len(df):
        col = "depth_to" if t.interval else ("fail_depth" if t is FAILURE else ("depth" if "depth" in df else None))
        if col and col in df:
            for i, v in enumerate(df[col]):
                if v is not None and not pd.isna(v) and float(v) > float(depth) + 0.01:
                    warn(i, col, tr("{v} m is below the hole's depth ({d} m)", v=v, d=depth))
    return bad, issues


def default_reference() -> dict[str, list[dict]]:
    """The predefined rows of the reference tables (rock properties, rock codes, measures, example rigs)."""
    from . import rockmech

    rock_names = {k: p.name for k, p in rockmech.DEFAULTS.items()}
    return {
        "rock": rockmech.default_rows(),
        "rock_code": [{"code": k, "rock_class": v, "description": rock_names.get(v, v)}
                      for k, v in codes.DEFAULT_ROCK_CODES.items()],
        "measure": codes.measure_rows(),
        "rig": [dict(r) for r in DEFAULT_RIGS],
    }


DEFAULT_RIGS = [   # generic wireline core rigs (examples: replace with your fleet)
    {"rig_model": "Core rig 1000 m", "depth_PQ": 350, "depth_HQ": 700, "depth_NQ": 1000, "depth_BQ": 1300,
     "max_rpm": 1300, "pump_lpm": 120, "pull_kn": 90, "max_torque": 2400},
    {"rig_model": "Core rig 600 m", "depth_PQ": 200, "depth_HQ": 420, "depth_NQ": 620, "depth_BQ": 800,
     "max_rpm": 1000, "pump_lpm": 90, "pull_kn": 60, "max_torque": 1600},
]


def apply_reference(tables: dict) -> None:
    """Use the database's rock properties, rock codes and measure library from now on."""
    from . import rockmech

    def rows(name):
        df = tables.get(name)
        return df.to_dict("records") if df is not None and len(df) else []
    rockmech.apply_table(rows("rock"))
    codes.apply_rock_codes(rows("rock_code"))
    codes.apply_measures(rows("measure"))


def rated_depth(rig: dict | None, size: str | None) -> float | None:
    if not rig or not size:
        return None
    v = rig.get(f"depth_{size}")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 and not math.isnan(v) else None


def today() -> str:
    return _dt.date.today().isoformat()


def fail_family(code: str) -> str:
    return codes.family_of(code)
