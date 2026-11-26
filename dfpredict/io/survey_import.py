"""Underground survey from an Excel workbook: the template to fill in, reading and checking a
filled-in workbook (the traverse is computed before anything is written), and writing it to
the project database. No Qt here.

The workbook has two sheets:

    pits     one row per pit: name, code, parent (empty: a main pit), start station, the entrance's
             X / Y / Z (main pits), the profile, where the stations sit in it, Bowditch on / off
    setups   one row per instrument setup, in order, each naming its pit; the direction to the
             foresight by backsight + angles (total station), by the bearing of the leg (compass)
             or by its X, Y, Z typed in (already calculated) - as in the Survey window

A pit is found by its name (a setup's pit, and a branch's parent, by name or code). For a pit
already in the database the import either replaces it - its header from the pits sheet (empty
cells keep what is stored) and its setups when the setups sheet has rows for it - or skips it.
Nothing is written while any error is found; all is written in one transaction.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field, replace as _copy

from ..i18n import N_, tr
from .excel_template import (Column, Sheet, cell_date, cell_number, cell_text, header_map, read_workbook,
                             write_template)
from .survey_store import (BEARING_TYPES, PITS, STATION_LEVELS, STATIONS, Pit, SetupRow, Station, SurveyStore,
                           compute, next_code, parse_angle)

METHODS = ("angles", "bearing", "coords")
_YES, _NO = ("yes", "y", "1", "true", "on", "是"), ("no", "n", "0", "false", "off", "否")

_A, _N, _T, _D, _L = "angle", "number", "text", "date", "list"
PIT_COLUMNS = [
    Column("name", N_("Name of the pit as it is called underground. A pit already in the database is found by "
                      "its name."), "North decline", True, _T, width=22),
    Column("code", N_("Short code that numbers its stations, e.g. P1 -> P1-01, P1-02 (empty: given "
                      "automatically)."), "P1", False, _T),
    Column("parent", N_("For a branch: the name or code of the pit it leaves from. Empty: a main pit."), "", False,
           _T, width=18),
    Column("start_station", N_("Main pit: its entrance station (exact coordinates known). Branch: the station of "
                               "the parent pit where it starts."), "E1", True, _T, width=16),
    Column("entrance_x", N_("Main pit: Easting of the entrance station (m)."), "1000.000", False, _N),
    Column("entrance_y", N_("Main pit: Northing of the entrance station (m)."), "5000.000", False, _N),
    Column("entrance_z", N_("Main pit: Elevation of the entrance station mark (m)."), "250.000", False, _N),
    Column("profile_width", N_("Width of the drive's rectangular profile (m), default 5."), "5", False, _N),
    Column("profile_height", N_("Height of the drive's profile (m), default 5."), "5", False, _N),
    Column("station_level", N_("Where the stations are in the profile: back (roof), floor or middle "
                               "(default back)."), "back", False, _L, tuple(STATION_LEVELS)),
    Column("adjust", N_("Spread a passing closure over the legs (Bowditch): yes or no (default yes)."), "yes",
           False, _L, ("yes", "no")),
    Column("note", N_("Note (may be empty)."), "", False, _T, width=24),
]
SETUP_COLUMNS = [
    Column("pit", N_("Name or code of the pit the setup belongs to. A pit's setups are used in the order of "
                     "the rows."), "North decline", True, _T, width=18),
    Column("at", N_("Station the instrument stands on."), "E1", True, _T),
    Column("method", N_("How the foresight is found: angles = backsight + angles (total station), bearing = "
                        "bearing of the leg (compass), coords = its X, Y, Z typed in (already calculated). "
                        "Empty: from the columns filled in."), "angles", False, _L, METHODS),
    Column("fs", N_("Foresight: the new station the setup measures to."), "P1-01", False, _T),
    Column("hi", N_("Height of the instrument above the station mark (m)."), "1.500", False, _N),
    Column("date", N_("Day of the survey."), "2026-03-31", False, _D),
    Column("surveyor", N_("Surveyor."), "", False, _T),
    Column("instrument", N_("Instrument."), "", False, _T),
    Column("bs_station", N_("Angles: backsight station (a station with coordinates) ..."), "", False, _T),
    Column("bs_bearing", N_("... or a known bearing to a reference mark instead."), "45 00 00", False, _A),
    Column("bs_bearing_type", N_("Kind of that bearing (default grid)."), "grid", False, _L, BEARING_TYPES),
    Column("bs_ha", N_("Angles: horizontal angle read on the backsight (empty: the circle was set to the "
                       "backsight bearing)."), "0 00 00", False, _A),
    Column("bs_za", N_("Angles: zenith angle to the backsight (for the backsight check)."), "", False, _A),
    Column("bs_sd", N_("Angles: slope distance to the backsight (m), for the check."), "", False, _N),
    Column("bs_ht", N_("Angles: target height on the backsight (m)."), "", False, _N),
    Column("fs_ha", N_("Angles: horizontal angle read on the foresight."), "120 30 00", False, _A),
    Column("fs_za", N_("Angles and bearing: zenith angle to the foresight (90 = level)."), "95 00 00", False, _A),
    Column("fs_sd", N_("Angles and bearing: slope distance to the foresight (m)."), "50.000", False, _N),
    Column("fs_ht", N_("Angles and bearing: target height on the foresight (m)."), "1.400", False, _N),
    Column("fs_bearing", N_("Bearing: the leg's bearing, measured directly (e.g. by compass)."), "", False, _A),
    Column("fs_bearing_type", N_("Kind of that bearing (default magnetic)."), "magnetic", False, _L,
           ("magnetic", "true", "grid")),
    Column("fs_x", N_("Coords: Easting of the foresight (m), already calculated."), "", False, _N),
    Column("fs_y", N_("Coords: Northing of the foresight (m)."), "", False, _N),
    Column("fs_z", N_("Coords: Elevation of the foresight (m)."), "", False, _N),
    Column("bs_ha_r", N_("Face right: horizontal angle on the backsight (optional, averaged with face left)."),
           "", False, _A),
    Column("bs_za_r", N_("Face right: zenith angle on the backsight (optional)."), "", False, _A),
    Column("fs_ha_r", N_("Face right: horizontal angle on the foresight (optional)."), "", False, _A),
    Column("fs_za_r", N_("Face right: zenith angle on the foresight (optional)."), "", False, _A),
    Column("note", N_("Note (may be empty)."), "", False, _T, width=24),
]
SHEETS = [Sheet("pits", N_("One row per pit (main pits and branches)."), PIT_COLUMNS),
          Sheet("setups", N_("One row per instrument setup, in the order they were made."), SETUP_COLUMNS)]
_ANGLES = {c.name for c in SETUP_COLUMNS if c.kind == _A}
_NUMBERS = {c.name for c in SETUP_COLUMNS if c.kind == _N}


def save_template(path) -> None:
    write_template(path, tr("GeoViewer - survey to import"), [
        tr("1. Sheet pits: one row per pit. A main pit needs its entrance station and the entrance's X, Y, Z; "
           "a branch names its parent pit and the station where it starts."),
        tr("2. Sheet setups: one row per instrument setup, in order. Fill in the columns of its method: "
           "angles (total station), bearing (compass) or coords (X, Y, Z already calculated)."),
        tr("3. In GeoViewer: Survey - Import from Excel, choose this file, check, Import."),
        tr("Angles as 120 30 00 (degrees minutes seconds) or 120.5. Orange columns are required. For a pit "
           "already in the database, empty cells in the pits sheet keep what is stored."),
    ], SHEETS)


# --- reading and checking --------------------------------------------------------------------------
@dataclass
class PitPlan:
    name: str
    id: int                                        # its id in the database (new pits: the id it gets)
    existing: Pit | None = None
    row: int | None = None                         # its row in the pits sheet (None: only setups given)
    values: dict = field(default_factory=dict)     # pit fields given in the sheet (code, parent_id, ...)
    entrance: tuple | None = None                  # main pit: the entrance's X, Y, Z given
    setups: list = field(default_factory=list)     # [SetupRow] in order
    setup_rows: list = field(default_factory=list)  # their rows in the setups sheet
    errors: list = field(default_factory=list)

    @property
    def new(self) -> bool:
        return self.existing is None


@dataclass
class SurveyImport:
    path: str
    pits: list = field(default_factory=list)       # [PitPlan], parents before branches
    errors: list = field(default_factory=list)     # not of one pit
    warnings: list = field(default_factory=list)
    max_id: int = 0                                # the pits' highest id when read (new ids follow it)
    db_pits: list = field(default_factory=list)
    db_stations: list = field(default_factory=list)
    db_setups: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    _net: dict = field(default_factory=dict)       # replace? -> (warnings, stations computed, without)

    def todo(self, replace: bool) -> list[PitPlan]:
        return [p for p in self.pits if p.new or replace]

    def problems(self, replace: bool) -> tuple[list[str], list[str]]:
        errors = list(self.errors)
        for p in self.todo(replace):
            errors += p.errors
        warnings = list(self.warnings)
        if not errors:
            warnings += self._simulate(replace)[0]
        return errors, warnings

    def summary(self, replace: bool) -> list[str]:
        new = sum(1 for p in self.pits if p.new)
        old = len(self.pits) - new
        lines = [tr("Pits in the file: {n:,} ({new:,} new, {old:,} already in the database)", n=len(self.pits),
                    new=new, old=old)]
        if old:
            lines.append(tr("Pits already in the database: replaced with the file's data") if replace else
                         tr("Pits already in the database: skipped"))
        todo = self.todo(replace)
        lines.append(tr("Setups to import: {n:,}", n=sum(len(p.setups) for p in todo)))
        if not any(p.errors for p in todo) and not self.errors and todo:
            _w, ok, missing = self._simulate(replace)
            lines.append(tr("Stations computed: {ok:,}; without coordinates: {missing:,}", ok=ok, missing=missing))
        return lines

    # the network as it will be after the import: its messages for the imported pits
    def _final(self, replace: bool):
        todo = {p.id: p for p in self.todo(replace)}
        pits = []
        for q in self.db_pits:
            pp = todo.get(q.id)
            pits.append(_copy(q, **pp.values) if pp is not None else q)
        for p in todo.values():
            if p.new:
                pits.append(Pit(p.id, p.name, **p.values))
        stations = [s for s in self.db_stations]
        names = {s.name: i for i, s in enumerate(stations)}
        for p in todo.values():
            if p.entrance is not None:
                start = p.values.get("start_station") or (p.existing.start_station if p.existing else None)
                st = Station(start, p.id, "known", *p.entrance, "known")
                if start in names:
                    stations[names[start]] = st
                else:
                    stations.append(st)
        setups = dict(self.db_setups)
        for p in todo.values():
            if p.setups:
                setups[p.id] = p.setups
        return pits, stations, setups

    def _simulate(self, replace: bool):
        if replace in self._net:
            return self._net[replace]
        pits, stations, setups = self._final(replace)
        net = compute(pits, stations, setups, self.settings)
        warnings, ok, missing = [], 0, 0
        for p in self.todo(replace):
            for pid, i, text in net.messages:
                if pid == p.id:
                    row = p.setup_rows[i] if i < len(p.setup_rows) and p.setups else None
                    warnings.append(tr("Pit {pit}, setups row {row}: {problem}", pit=p.name, row=row, problem=text)
                                    if row else tr("Pit {pit}: {problem}", pit=p.name, problem=text))
            res = net.pits.get(p.id)
            if res is not None:
                warnings += [tr("Pit {pit}: {problem}", pit=p.name, problem=m) for m in res.messages]
            for r in p.setups:
                if r.fs:
                    if r.fs in net.coords:
                        ok += 1
                    else:
                        missing += 1
        self._net[replace] = (warnings, ok, missing)
        return self._net[replace]


def _level(v) -> str | None:
    t = (cell_text(v) or "").lower()
    if not t:
        return None
    for k, label in STATION_LEVELS.items():
        if t in (k, label.lower(), tr(label).lower()) or t == tr(label).split(" ")[0].lower():
            return k
    raise ValueError(tr("must be one of: {values}", values=", ".join(STATION_LEVELS)))


def _yes_no(v) -> bool | None:
    t = (cell_text(v) or "").lower()
    if not t:
        return None
    if t in _YES or t == tr("yes").lower():
        return True
    if t in _NO or t == tr("no").lower():
        return False
    raise ValueError(tr("must be yes or no"))


def _choice(v, allowed) -> str | None:
    t = (cell_text(v) or "").lower()
    if not t:
        return None
    if t not in allowed:
        raise ValueError(tr("must be one of: {values}", values=", ".join(allowed)))
    return t


def _date(v) -> str | None:
    t = cell_date(v)
    if t is None:
        return None
    try:
        _dt.date.fromisoformat(t)
    except ValueError:
        raise ValueError(tr("must be a date like 2026-03-31")) from None
    return t


def read_import(store: SurveyStore, path) -> SurveyImport:
    """Read and check a filled-in workbook (nothing is written)."""
    store.prepare()
    plan = SurveyImport(str(path))
    plan.db_pits = store.pits()
    plan.db_stations = store.stations()
    plan.db_setups = store.setups()
    plan.settings = store.settings()
    plan.max_id = max((p.id for p in plan.db_pits), default=0)
    book = read_workbook(path)
    s_pits, s_setups = book.get("pits"), book.get("setups")
    if s_pits is None and s_setups is None:
        plan.errors.append(tr("The workbook has neither of the sheets pits and setups."))
        return plan
    by_name = {p.name.lower(): p for p in plan.db_pits}
    by_code = {p.code.lower(): p for p in plan.db_pits if p.code}
    plans: dict[str, PitPlan] = {}                 # lower-case name -> plan
    next_id = plan.max_id

    # --- the pits sheet
    parent_refs: dict[str, tuple[int, str]] = {}
    cols = _columns(s_pits.headers, PIT_COLUMNS, s_pits.name, plan) if s_pits else {}
    for n, rec in (s_pits.rows if s_pits else []):
        get = {c: rec.get(h) for c, h in cols.items()}
        name = cell_text(get.get("name"))
        if not name:
            plan.errors.append(tr("Sheet {sheet} row {row}: the pit has no name", sheet=s_pits.name, row=n))
            continue
        if name.lower() in plans:
            plan.errors.append(tr("Sheet {sheet} row {row}: pit {name} is in the sheet twice (also row {first})",
                                  sheet=s_pits.name, row=n, name=name, first=plans[name.lower()].row))
            continue
        old = by_name.get(name.lower())
        if old is None:
            next_id += 1
        pp = PitPlan(old.name if old else name, old.id if old else next_id, old, n)
        plans[name.lower()] = pp
        where = tr("Sheet {sheet} row {row}", sheet=s_pits.name, row=n)
        v = pp.values

        def bad(column, e, where=where, pp=pp):
            pp.errors.append(tr("{where}, {column}: {problem}", where=where, column=column, problem=e))

        for key in ("code", "start_station", "note"):
            t = cell_text(get.get(key))
            if t is not None:
                v[key] = t
        for key, conv in (("station_level", _level), ("adjust", _yes_no)):
            try:
                x = conv(get.get(key))
            except ValueError as e:
                bad(key, e)
                continue
            if x is not None:
                v[key] = x
        for key in ("profile_width", "profile_height"):
            try:
                x = cell_number(get.get(key))
            except ValueError:
                bad(key, tr("must be a number"))
                continue
            if x is not None:
                if x <= 0:
                    bad(key, tr("must be more than 0"))
                else:
                    v[key] = x
        xyz = []
        for key in ("entrance_x", "entrance_y", "entrance_z"):
            try:
                xyz.append(cell_number(get.get(key)))
            except ValueError:
                bad(key, tr("must be a number"))
                xyz.append(math.nan)
        if any(x is not None for x in xyz):
            if None in xyz:
                bad("entrance_x / y / z", tr("give all three of X, Y and Z"))
            elif not any(math.isnan(x) for x in xyz):
                pp.entrance = tuple(xyz)
        ref = cell_text(get.get("parent"))
        if ref is not None:
            parent_refs[name.lower()] = (n, ref)

    # --- parents (a pit of the file or of the database, by name or code) and the codes, in passes:
    # a branch may name its parent by the code it is only given here (P1)
    codes = {p.code.lower(): p.id for p in plan.db_pits if p.code}
    given = {}
    for pp in plans.values():
        c = pp.values.get("code")
        if c:
            other = codes.get(c.lower())
            if other is not None and other != pp.id:
                pp.errors.append(tr("Sheet {sheet} row {row}: code {code} is used by another pit", sheet=s_pits.name,
                                    row=pp.row, code=c))
            codes[c.lower()] = pp.id
            given[pp.id] = c
    names = {p.name.lower(): p.id for p in plan.db_pits}
    names.update({k: pp.id for k, pp in plans.items()})
    pits_now = [_copy(q, code=given[q.id]) if q.id in given else q for q in plan.db_pits]
    pits_now += [Pit(pp.id, pp.name, code=given[pp.id]) for pp in plans.values() if pp.new and pp.id in given]
    pending = dict(parent_refs)
    to_code = {pp.id: pp for pp in plans.values() if pp.new and pp.id not in given}
    progress = True
    while progress:
        progress = False
        for key, (n, ref) in list(pending.items()):
            pid = names.get(ref.lower()) or codes.get(ref.lower())
            if pid is None:
                continue
            del pending[key]
            progress = True
            if pid == plans[key].id:
                plans[key].errors.append(tr("Sheet {sheet} row {row}: a pit cannot branch from itself",
                                            sheet=s_pits.name, row=n))
            else:
                plans[key].values["parent_id"] = pid
        for pid, pp in list(to_code.items()):
            parent = pp.values.get("parent_id")
            if pp.name.lower() in pending or parent in to_code:      # its parent (and its code) first
                continue
            code = pp.values["code"] = next_code(pits_now, parent)
            codes[code.lower()] = pid
            pits_now.append(Pit(pid, pp.name, parent, code=code))
            del to_code[pid]
            progress = True
    for key, (n, ref) in pending.items():
        plans[key].errors.append(tr("Sheet {sheet} row {row}: parent pit {ref} is neither in the file nor in the "
                                    "database", sheet=s_pits.name, row=n, ref=ref))
    parent_of = {p.id: p.parent_id for p in plan.db_pits}
    parent_of.update({pp.id: pp.values.get("parent_id", pp.existing.parent_id if pp.existing else None)
                      for pp in plans.values()})
    depth = {}
    for pid in parent_of:
        seen, d, p = set(), 0, pid
        while parent_of.get(p) is not None and p not in seen:
            seen.add(p)
            p = parent_of[p]
            d += 1
        depth[pid] = d if p not in seen else -1          # -1: a loop of parents
    order = sorted(plans.values(), key=lambda pp: (max(depth[pp.id], 0), pp.id))

    # --- per pit: what it needs as a main pit or a branch
    starts = {s.name: s for s in plan.db_stations}
    used_starts: dict[str, str] = {}
    for pp in order:
        where = tr("Sheet {sheet} row {row}", sheet=s_pits.name, row=pp.row)
        if depth[pp.id] < 0:
            pp.errors.append(tr("{where}: its parents lead back to it (a loop of branches)", where=where))
            continue
        main = parent_of[pp.id] is None
        start = pp.values.get("start_station") or (pp.existing.start_station if pp.existing else None)
        if not start:
            pp.errors.append(tr("{where}: start_station is needed", where=where))
        if main:
            if pp.new and pp.entrance is None:
                pp.errors.append(tr("{where}: a main pit needs the entrance's X, Y and Z (entrance_x, entrance_y, "
                                    "entrance_z)", where=where))
            if start and pp.entrance is not None:
                st = starts.get(start)
                if st is not None and (st.kind != "known" or st.pit_id != pp.id):
                    pp.errors.append(tr("{where}: station {name} is already in the database (another pit's "
                                        "station)", where=where, name=start))
                if start in used_starts:
                    pp.errors.append(tr("{where}: entrance station {name} is also the entrance of pit {other}",
                                        where=where, name=start, other=used_starts[start]))
                used_starts[start] = pp.name
        elif pp.entrance is not None:
            plan.warnings.append(tr("{where}: a branch starts at its parent's station - entrance_x / y / z are not "
                                    "used", where=where))
            pp.entrance = None

    # --- the setups sheet
    if s_setups is not None:
        cols = _columns(s_setups.headers, SETUP_COLUMNS, s_setups.name, plan)
        file_codes = {pp.values["code"].lower(): pp for pp in plans.values() if pp.values.get("code")}
        for n, rec in s_setups.rows:
            get = {c: rec.get(h) for c, h in cols.items()}
            where = tr("Sheet {sheet} row {row}", sheet=s_setups.name, row=n)
            ref = cell_text(get.get("pit"))
            pp = None
            if ref:
                pp = plans.get(ref.lower()) or file_codes.get(ref.lower())
                if pp is None:
                    old = by_name.get(ref.lower()) or by_code.get(ref.lower())
                    if old is not None:
                        pp = plans.setdefault(old.name.lower(), PitPlan(old.name, old.id, old))
            if pp is None:
                plan.errors.append(tr("{where}: pit {ref} is neither in the pits sheet nor in the database",
                                      where=where, ref=ref) if ref else tr("{where}: the pit is missing", where=where))
                continue
            row, problems = _setup(get)
            pp.errors += [tr("{where}, {column}: {problem}", where=where, column=c, problem=e) for c, e in problems]
            pp.setups.append(row)
            pp.setup_rows.append(n)
    # pits named only in the setups sheet go after the others, parents first
    plan.pits = sorted(plans.values(), key=lambda pp: (max(depth.get(pp.id, 0), 0), pp.id))
    return plan


def _columns(headers, columns, sheet, plan) -> dict[str, str]:
    found = header_map(headers, [c.name for c in columns])
    known = set(found.values())
    for h in headers:
        if h not in known:
            plan.warnings.append(tr("Sheet {sheet}: column {column} is not one of the template's - left out",
                                    sheet=sheet, column=h))
    return found


def _setup(get: dict) -> tuple[SetupRow, list]:
    """A setups row as a SetupRow, and its problems [(column, text)]."""
    problems, v = [], {}
    for key in _ANGLES:
        try:
            v[key] = parse_angle(get.get(key))
        except ValueError as e:
            problems.append((key, e))
            v[key] = None
    for key in _NUMBERS:
        try:
            v[key] = cell_number(get.get(key))
        except ValueError:
            problems.append((key, tr("must be a number")))
            v[key] = None
    text = {k: cell_text(get.get(k)) for k in ("at", "fs", "bs_station", "surveyor", "instrument", "note")}
    kinds = {}
    for key, allowed in (("method", METHODS), ("bs_bearing_type", BEARING_TYPES), ("fs_bearing_type", BEARING_TYPES)):
        try:
            kinds[key] = _choice(get.get(key), allowed)
        except ValueError as e:
            problems.append((key, e))
            kinds[key] = None
    try:
        date = _date(get.get("date"))
    except ValueError as e:
        problems.append(("date", e))
        date = None
    if not text["at"]:
        problems.append(("at", tr("the station the instrument stands on is needed")))
    row = SetupRow(text["at"] or "", v["hi"], text["bs_station"], v["bs_bearing"], kinds["bs_bearing_type"] or "grid",
                   v["bs_ha"], v["bs_za"], v["bs_sd"], v["bs_ht"], text["fs"], v["fs_ha"], v["fs_za"], v["fs_sd"],
                   v["fs_ht"], v["fs_bearing"], kinds["fs_bearing_type"] or "magnetic", v["fs_x"], v["fs_y"],
                   v["fs_z"], v["bs_ha_r"], v["bs_za_r"], v["fs_ha_r"], v["fs_za_r"], date or "",
                   text["surveyor"] or "", text["instrument"] or "", text["note"] or "")
    method = kinds["method"] or row.method
    if method == "angles":                         # only the columns of its method count
        row.fs_bearing = row.fs_x = row.fs_y = row.fs_z = None
        if row.bs_station and row.bs_bearing is not None:
            problems.append(("bs_bearing", tr("give a backsight station or a known bearing, not both")))
        elif not row.bs_station and row.bs_bearing is None:
            problems.append(("bs_station", tr("a backsight station or a known bearing (bs_bearing) is needed")))
    elif method == "bearing":
        row.fs_x = row.fs_y = row.fs_z = None
        row.bs_station = row.bs_bearing = row.bs_ha = row.bs_za = row.bs_sd = row.bs_ht = None
        row.bs_ha_r = row.bs_za_r = row.fs_ha = row.fs_ha_r = None
        if row.fs_bearing is None:
            problems.append(("fs_bearing", tr("the bearing of the leg is needed (method bearing)")))
    else:
        row.fs_bearing = row.bs_station = row.bs_bearing = None
        if None in (row.fs_x, row.fs_y, row.fs_z):
            problems.append(("fs_x / fs_y / fs_z", tr("X, Y and Z of the foresight are needed (method coords)")))
        if not row.fs:
            problems.append(("fs", tr("the foresight station's name is needed (method coords)")))
    return row, problems


# --- writing ---------------------------------------------------------------------------------------
def apply_import(store: SurveyStore, plan: SurveyImport, replace: bool) -> dict:
    """Write the plan (it must have no errors): new pits added, pits in the database replaced
    (``replace``) or skipped; one transaction, then the network is computed and stored.
    Returns the counts: added, updated, skipped, setups."""
    errors, _w = plan.problems(replace)
    if errors:
        raise ValueError(tr("The workbook has errors - nothing was imported."))
    todo = plan.todo(replace)
    counts = {"added": 0, "updated": 0, "skipped": len(plan.pits) - len(todo), "setups": 0}
    now = _dt.datetime.now().isoformat(timespec="seconds")
    with store.target.connect() as db:
        if int(db.scalar(f"SELECT MAX(id) FROM {PITS}", default=0) or 0) != plan.max_id:
            raise ValueError(tr("The pits in the database changed since the file was checked - check it again."))
        with db.transaction():
            for pp in todo:
                v = pp.values
                parent = v.get("parent_id", pp.existing.parent_id if pp.existing else None)
                start = v.get("start_station") or (pp.existing.start_station if pp.existing else None)
                if pp.new:
                    db.execute(f"INSERT INTO {PITS} (id, name, code, parent_id, start_station, adjust, note, "
                               "profile_width, profile_height, station_level, created_at) "
                               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (pp.id, pp.name, v.get("code"), parent, start, 0 if v.get("adjust") is False else 1,
                                v.get("note"), v.get("profile_width", 5.0), v.get("profile_height", 5.0),
                                v.get("station_level", "back"), now))
                    counts["added"] += 1
                else:
                    sets = {k: (1 if x else 0) if k == "adjust" else x for k, x in v.items()}
                    if sets:
                        db.execute(f"UPDATE {PITS} SET {', '.join(f'{db.q(k)} = ?' for k in sets)} WHERE id = ?",
                                   [*sets.values(), pp.id])
                    counts["updated"] += 1
                if parent is None and pp.entrance is not None:
                    x, y, z = pp.entrance
                    if db.scalar(f"SELECT COUNT(*) FROM {STATIONS} WHERE name = ?", (start,)):
                        db.execute(f"UPDATE {STATIONS} SET kind = 'known', pit_id = ?, x = ?, y = ?, z = ?, "
                                   "status = 'known' WHERE name = ?", (pp.id, x, y, z, start))
                    else:
                        db.execute(f"INSERT INTO {STATIONS} (name, pit_id, kind, x, y, z, status, active) "
                                   "VALUES (?, ?, 'known', ?, ?, ?, 'known', 1)", (start, pp.id, x, y, z))
                if pp.setups:
                    store._write_setups(db, pp.id, pp.setups)
                    counts["setups"] += len(pp.setups)
    store.recompute()
    return counts


def result_text(counts: dict) -> str:
    return tr("Imported - pits added: {added:,}, replaced: {updated:,}, skipped: {skipped:,}; setups: {setups:,}",
              **counts)
