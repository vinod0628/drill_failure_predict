"""Drillholes from an Excel workbook: the template to fill in, reading and checking a filled-in
workbook, and writing it to the project database. No Qt here.

The workbook has the sheets collar, survey, geology and assay (as the export writes them, so an
exported workbook can be edited and imported again). Columns are found by name: the database's
own names, or the usual other names (BHID, from, to, ...). For a hole already in the database
the import either replaces its data with the file's - the collar values given, and its survey /
geology / assay rows in each sheet that has rows for it (other tables are kept) - or skips it.
An assay column the database does not have yet is added as a new assay field. Nothing is
written while any error is found; all is written in one transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..i18n import N_, tr, trc
from .drillhole_store import (ASSAY, COLLAR, GEOLOGY, HOLE_STATES, HOLE_TABLES, SURVEY, DrillholeStore, TableInfo,
                              _check_field_name, _natural, _role_columns, check_hole)
from .excel_template import Column, Sheet, cell_date, cell_text, read_workbook, write_template

SHEETS = {"collar": COLLAR, "survey": SURVEY, "geology": GEOLOGY, "assay": ASSAY}     # sheet -> table
_DATES = ("start_date", "end_date")

SHEET_HELP = {
    "collar": N_("One row per hole: where it starts and how long it is."),
    "survey": N_("One row per survey reading down a hole (direction of the hole at that depth)."),
    "geology": N_("One row per logged interval: From, To and the rock."),
    "assay": N_("One row per sample interval: From, To and the grades (one column per assay)."),
}
# (table, role or column name) -> (meaning, example, required, kind)
_SPEC = {
    (COLLAR, "x"): (N_("Easting of the collar (m)."), "512340.50", True, "number"),
    (COLLAR, "y"): (N_("Northing of the collar (m)."), "4621875.20", True, "number"),
    (COLLAR, "z"): (N_("Elevation of the collar (m)."), "1240.30", True, "number"),
    (COLLAR, "max_depth"): (N_("Length of the hole (m)."), "250", False, "number"),
    (COLLAR, "organization"): (N_("Organization that drilled the hole."), "", False, "text"),
    (COLLAR, "drilling_manager"): (N_("Drilling manager."), "", False, "text"),
    (COLLAR, "geology_surveyor"): (N_("Geologist who logged the hole."), "", False, "text"),
    (COLLAR, "start_date"): (N_("Day drilling started."), "2026-03-01", False, "date"),
    (COLLAR, "end_date"): (N_("Day drilling finished."), "2026-03-12", False, "date"),
    (COLLAR, "state"): (N_("State of the hole."), "", False, "list"),
    (SURVEY, "depth"): (N_("Depth down the hole of the reading (m)."), "0", True, "number"),
    (SURVEY, "azimuth"): (N_("Azimuth: degrees clockwise from north, 0 to 360."), "45", False, "number"),
    (SURVEY, "dip"): (N_("Dip: degrees from straight down - 0 = straight down, 90 = horizontal."), "30", False,
                      "number"),
    (GEOLOGY, "from"): (N_("From: start of the interval down the hole (m)."), "0", True, "number"),
    (GEOLOGY, "to"): (N_("To: end of the interval down the hole (m)."), "12.5", True, "number"),
    (GEOLOGY, "rock"): (N_("Rock code, e.g. GR (its colour is set in Rock colours)."), "GR", False, "text"),
    (GEOLOGY, "note"): (N_("Description of the core (may be empty)."), "", False, "text"),
    (ASSAY, "from"): (N_("From: start of the sample down the hole (m)."), "0", True, "number"),
    (ASSAY, "to"): (N_("To: end of the sample down the hole (m)."), "1.5", True, "number"),
}


# --- the template ----------------------------------------------------------------------------------
def template_sheets(store: DrillholeStore) -> list[Sheet]:
    """The sheets of the template, with the database's own columns (its assay fields too)."""
    out = []
    for sheet, table in SHEETS.items():
        info = store.info(table)
        role_of = {c: k for k, c in info.roles.items()}
        cols = [Column(info.hole, N_("Hole ID - the same in every sheet."), "WD101", True, "text")]
        for c in info.fields:
            role = role_of.get(c)
            spec = _SPEC.get((table, role)) or _SPEC.get((table, c.lower()))
            if spec is None:
                kind = "number" if info.numeric.get(c) else "text"
                spec = ((N_("Grade (empty = not assayed)."), "0.85", False, "number") if table == ASSAY else
                        (N_("A field of your database."), "", False, kind))
            meaning, example, required, kind = spec
            choices = tuple(tr(v) for v in HOLE_STATES.values()) if kind == "list" else ()
            cols.append(Column(c, meaning, example or (choices[0] if choices else ""), required, kind, choices))
        out.append(Sheet(sheet, SHEET_HELP[sheet], cols))
    return out


def save_template(store: DrillholeStore, path) -> None:
    store.prepare()
    write_template(path, tr("GeoViewer - drillholes to import"), [
        tr("1. Fill in the sheets collar, survey, geology and assay: one row per line of data, the "
           "column names in the first row as they are."),
        tr("2. The hole ID links the sheets: write it on every row. Orange columns are required."),
        tr("3. In GeoViewer: Drillhole Manager - Import from Excel, choose this file, check, Import."),
        tr("A sheet may be left empty. More assay columns (e.g. copper) may be added: they become new "
           "assay fields."),
    ], template_sheets(store))


# --- reading and checking --------------------------------------------------------------------------
@dataclass
class HolePlan:
    hid: str
    new: bool                                       # not in the database yet
    collar: dict | None = None                      # database column -> value (None: no collar row)
    collar_row: int | None = None
    records: dict = field(default_factory=dict)     # table -> [values by database column]
    rows: dict = field(default_factory=dict)        # table -> the Excel row of each record
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)      # table -> DataFrame to write (built after reading)


@dataclass
class DrillholeImport:
    path: str
    holes: dict = field(default_factory=dict)       # hole id -> HolePlan (natural order)
    new_fields: list = field(default_factory=list)  # assay fields to add
    errors: list = field(default_factory=list)      # problems not of one hole (sheets, columns)
    warnings: list = field(default_factory=list)

    def todo(self, replace: bool) -> list[HolePlan]:
        return [h for h in self.holes.values() if h.new or replace]

    def problems(self, replace: bool) -> tuple[list[str], list[str]]:
        errors, warnings = list(self.errors), list(self.warnings)
        for h in self.todo(replace):
            errors += h.errors
            warnings += h.warnings
        return errors, warnings

    def summary(self, replace: bool) -> list[str]:
        new = sum(1 for h in self.holes.values() if h.new)
        old = len(self.holes) - new
        lines = [tr("Holes in the file: {n:,} ({new:,} new, {old:,} already in the database)", n=len(self.holes),
                    new=new, old=old)]
        if old:
            lines.append(tr("Holes already in the database: replaced with the file's data") if replace else
                         tr("Holes already in the database: skipped"))
        todo = self.todo(replace)
        n = {t: sum(len(h.records.get(t, [])) for h in todo) for t in HOLE_TABLES}
        lines.append(tr("Rows to import: survey {survey:,}, geology {geology:,}, assay {assay:,}",
                        survey=n[SURVEY], geology=n[GEOLOGY], assay=n[ASSAY]))
        if self.new_fields:
            lines.append(tr("New assay fields: {names}", names=", ".join(self.new_fields)))
        return lines


def _map_columns(table: str, info: TableInfo, headers: list[str]):
    """(the sheet's hole id header, {header: database column}, headers not in the table)."""
    hole_h, roles_h = _role_columns(table, headers)
    role_of_h = {h: k for k, h in roles_h.items()}
    low = {c.lower(): c for c in info.columns}
    mapping, unknown, used = {}, [], set()
    for h in headers:
        if h == hole_h:
            continue
        c = low.get(h.lower())
        if c is None and h in role_of_h:
            c = info.roles.get(role_of_h[h])
        if c is None or c == info.hole or c in used:
            unknown.append(h)
            continue
        mapping[h] = c
        used.add(c)
    return hole_h, mapping, unknown


def _cell(info: TableInfo, col: str, v):
    low = col.lower()
    if info.name.lower() == COLLAR and low in _DATES:
        return cell_date(v)
    if info.numeric.get(col) and isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return cell_text(v)


def read_import(store: DrillholeStore, path) -> DrillholeImport:
    """Read and check a filled-in workbook (nothing is written)."""
    store.prepare()
    plan = DrillholeImport(str(path))
    book = read_workbook(path)
    infos = {t: store.info(t) for t in SHEETS.values()}
    existing = set(store.hole_ids())
    with store.target.connect(create=False) as db:
        stored = db.read_frame(COLLAR)
    ci = infos[COLLAR]
    old_collars = {str(r[ci.hole]): {k: v for k, v in r.items() if k != ci.hole}
                   for r in stored.to_dict("records") if r.get(ci.hole) is not None}
    found = {s: book.get(s) or book.get(t) for s, t in SHEETS.items()}
    if not any(found.values()):
        plan.errors.append(tr("The workbook has none of the sheets collar, survey, geology and assay."))
        return plan
    holes: dict[str, HolePlan] = {}
    for sheet, table in SHEETS.items():
        sr = found[sheet]
        if sr is None:
            continue
        info = infos[table]
        hole_h, mapping, unknown = _map_columns(table, info, sr.headers)
        if hole_h is None:
            if sr.rows:
                plan.errors.append(tr("Sheet {sheet}: no hole ID column (e.g. hole_id)", sheet=sr.name))
            continue
        for h in unknown:
            if table == ASSAY:
                try:
                    name = _check_field_name(h)
                except ValueError as e:
                    plan.errors.append(tr("Sheet {sheet}, column {column}: {problem}", sheet=sr.name, column=h,
                                          problem=e))
                    continue
                if name.lower() in {c.lower() for c in info.columns}:
                    plan.warnings.append(tr("Sheet {sheet}: column {column} is in the sheet twice - the second "
                                            "is left out", sheet=sr.name, column=h))
                    continue
                plan.new_fields.append(name)
                info = TableInfo(info.name, info.columns + [name], {**info.numeric, name: True}, info.hole,
                                 info.roles)
                mapping[h] = name
            else:
                plan.warnings.append(tr("Sheet {sheet}: column {column} is not in the database - left out",
                                        sheet=sr.name, column=h))
        infos[table] = info
        for n, rec in sr.rows:
            hid = cell_text(rec.get(hole_h))
            if not hid:
                plan.errors.append(tr("Sheet {sheet} row {row}: no hole ID", sheet=sr.name, row=n))
                continue
            values = {c: _cell(info, c, rec.get(h)) for h, c in mapping.items()}
            hp = holes.get(hid)
            if hp is None:
                hp = holes[hid] = HolePlan(hid, hid not in existing)
            if table == COLLAR:
                if hp.collar is not None:
                    hp.errors.append(tr("Sheet {sheet} row {row}: hole {hid} is in the sheet twice (also row {first})",
                                        sheet=sr.name, row=n, hid=hid, first=hp.collar_row))
                    continue
                hp.collar, hp.collar_row = values, n
            else:
                hp.records.setdefault(table, []).append(values)
                hp.rows.setdefault(table, []).append(n)
    label = {SURVEY: trc("drillhole", "Survey"), GEOLOGY: tr("Geology"), ASSAY: tr("Assay")}
    for hid in sorted(holes, key=_natural):
        hp = holes[hid]
        if hp.new and hp.collar is None:
            where = ", ".join(label[t] for t in hp.records)
            hp.errors.append(tr("Hole {hid} ({where}) is not in the database and has no row in the collar sheet",
                                hid=hid, where=where))
        collar = {**old_collars.get(hid, {}), **(hp.collar or {})}
        hp.tables = {t: pd.DataFrame(recs, columns=infos[t].fields) for t, recs in hp.records.items()}
        cinfo = ci if not (hp.new and hp.collar is None) else TableInfo(ci.name, [], {}, ci.hole, {})  # said above
        errors, warnings = check_hole(cinfo, collar,{t: (infos[t], df) for t, df in hp.tables.items()}, hp.rows)
        hp.errors += [tr("Hole {hid}: {problem}", hid=hid, problem=e) for e in errors]
        hp.warnings += [tr("Hole {hid}: {problem}", hid=hid, problem=w) for w in warnings]
        plan.holes[hid] = hp
    return plan


# --- writing ---------------------------------------------------------------------------------------
def apply_import(store: DrillholeStore, plan: DrillholeImport, replace: bool) -> dict:
    """Write the plan (it must have no errors for ``replace``): new holes added, holes in the
    database replaced (``replace``) or skipped. One transaction (new assay fields are added
    first). Returns the counts: added, updated, skipped and the rows per table."""
    errors, _w = plan.problems(replace)
    if errors:
        raise ValueError(tr("The workbook has errors - nothing was imported."))
    todo = plan.todo(replace)
    counts = {"added": 0, "updated": 0, "skipped": len(plan.holes) - len(todo), SURVEY: 0, GEOLOGY: 0, ASSAY: 0}
    with store.target.connect() as db:
        stored = db.table_name(ASSAY)
        have = {c.lower() for c in db.columns(stored)}
        for f in plan.new_fields:
            if f.lower() not in have:
                db.add_column(stored, f, "real")
        db.con.commit()
        ci = store._info(db, COLLAR)
        infos = {t: store._info(db, t) for t in HOLE_TABLES}
        with db.transaction():
            for h in todo:
                vals = {c: v for c, v in (h.collar or {}).items() if c in ci.columns and c != ci.hole}
                if h.new:
                    cols = [ci.hole] + list(vals)
                    db.execute(f"INSERT INTO {COLLAR} ({', '.join(db.q(c) for c in cols)}) "
                               f"VALUES ({', '.join('?' * len(cols))})",
                               [h.hid] + [store._value(ci, c, v) for c, v in vals.items()])
                    counts["added"] += 1
                else:
                    if vals:
                        db.execute(f"UPDATE {COLLAR} SET {', '.join(f'{db.q(c)} = ?' for c in vals)} "
                                   f"WHERE {db.q(ci.hole)} = ?",
                                   [store._value(ci, c, v) for c, v in vals.items()] + [h.hid])
                    counts["updated"] += 1
                for t, df in h.tables.items():
                    counts[t] += store._write_rows(db, infos[t], h.hid, df)
    return counts


def result_text(counts: dict) -> str:
    return tr("Imported - holes added: {added:,}, replaced: {updated:,}, skipped: {skipped:,}; rows: survey "
              "{survey:,}, geology {geology:,}, assay {assay:,}", added=counts["added"], updated=counts["updated"],
              skipped=counts["skipped"], survey=counts[SURVEY], geology=counts[GEOLOGY], assay=counts[ASSAY])
