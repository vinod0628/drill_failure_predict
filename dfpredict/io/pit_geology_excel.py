"""Pit geology and Excel: the template to fill in, the export (all pits' rocks, structures and
samples), and reading / checking a filled-in workbook before it is written. No Qt here.

The workbook has three data sheets, one row per record, each row naming its pit:

    rocks       pit, from_m, to_m, wall, rock, description, alteration, mineralization,
                mapped_on, geologist
    structures  pit, at_m, wall, height, kind, dip_direction, dip, thickness, fill, description,
                mapped_on, geologist
    samples     pit, sample, from_m, to_m, wall, height, kind, width_cm, depth_cm, weight_kg,
                one column per assay field (gold, silver, ...), taken_on, sampler, note

and a reference sheet ``stations`` (pit, station, chainage_m; not read back). Chainages are
metres from the pit's start (0 m = the entrance of a main pit, the junction station of a
branch) or a station and taped metres as text ("P1-03+4.5"). Walls, types and kinds may be the
data keys (left, fault, channel), their English or Chinese names. A pit is found by its name or
code and must be in the database (pits are added in the Pit Manager).

For a pit that already has geology the import either replaces it - the rocks, structures or
samples of each sheet that has rows for it; the others are kept - or skips it. A samples column
that is not an assay field yet becomes one. Nothing is written while any error is found; all is
written in one transaction.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

from ..i18n import N_, tr
from .excel_template import Column, Sheet, cell_date, cell_number, cell_text, read_workbook, write_template
from .pit_geology_store import (_FIELD_NAME, _FIXED_SAMPLE_COLUMNS, GEO_TABLES, INTERVAL_WALLS, SAMPLE_KINDS,
                                SAMPLE_WALLS, SAMPLES, STRUCTURE_KINDS, WALLS, GeoInterval, GeoSample, GeoStructure,
                                PitGeology, PitGeologyStore, check_geology, sorted_geology)
from .survey_store import SurveyStore, parse_angle

KINDS = ("intervals", "structures", "samples")              # PitGeology's lists
SHEET_OF = {"intervals": "rocks", "structures": "structures", "samples": "samples"}
_SHEET_NAMES = {"intervals": ("rocks", "rock", "intervals", "geology", "岩性"),
                "structures": ("structures", "structure", "构造"),
                "samples": ("samples", "sample", "样品")}

# names a cell may use besides the key and its (translated) label
_WALL_ALIASES = {"all": ("both", "all walls", "walls and back", "两壁一顶", "全部"),
                 "left": ("l", "lw", "left", "左", "左壁", "左帮"),
                 "right": ("r", "rw", "right", "右", "右壁", "右帮"),
                 "back": ("b", "roof", "top", "顶", "顶板", "顶部", "拱顶"),
                 "face": ("f", "掌子面", "工作面")}
_KIND_ALIASES = {"bedding": ("层理",), "contact": ("接触面", "接触带"), "fault": ("断层",), "joint": ("节理",),
                 "vein": ("脉", "矿脉"), "foliation": ("schistosity", "cleavage", "片理", "劈理"),
                 "shear": ("shear", "剪切带"), "other": ("其他",),
                 "channel": ("刻槽", "刻槽样"), "chip": ("刻块", "刻块样"), "grab": ("拣块", "拣块样"),
                 "panel": ("剥层", "剥层样")}

_T, _N, _A, _D, _L = "text", "number", "angle", "date", "list"
_PIT = Column("pit", N_("Name or code of the pit (as in the Pit Manager)."), "North decline", True, _T, width=18)
ROCK_COLUMNS = [
    _PIT,
    Column("from_m", N_("Where the rock starts: metres from the pit's start (0 m = the entrance; a branch: its "
                        "junction station), or a station and metres past it, e.g. P1-03+4.5."), "0", True, _T,
           width=12),
    Column("to_m", N_("Where the rock ends (metres, or a station and metres past it)."), "P1-02+4.5", True, _T,
           width=12),
    Column("wall", N_("Where: all = both walls and the back, or left, right, back (left / right looking toward "
                      "the face). Empty: all."), "all", False, _L, tuple(INTERVAL_WALLS)),
    Column("rock", N_("Rock code (the project's codes: Drillhole Manager > Rock colours)."), "GR", True, _T),
    Column("description", N_("Colour, grain size, texture, structure, minerals ..."), "", False, _T, width=30),
    Column("alteration", N_("Alteration, e.g. silicification, sericite."), "", False, _T, width=18),
    Column("mineralization", N_("Mineralization, e.g. pyrite 2 %, visible gold."), "", False, _T, width=18),
    Column("mapped_on", N_("Day it was mapped."), "2026-03-31", False, _D),
    Column("geologist", N_("Geologist."), "", False, _T, width=14),
]
STRUCTURE_COLUMNS = [
    _PIT,
    Column("at_m", N_("Where it was measured: metres from the pit's start, or a station and metres past it, e.g. "
                      "P1-03+4.5."), "23.4", True, _T, width=12),
    Column("wall", N_("Wall: left, right, back or face (left / right looking toward the face)."), "left", True, _L,
           tuple(WALLS)),
    Column("height", N_("Height above the floor (m); empty: half way up the wall."), "1.5", False, _N),
    Column("kind", N_("Type of structure."), "fault", True, _L, tuple(STRUCTURE_KINDS)),
    Column("dip_direction", N_("Dip direction, 0-360° clockwise from north (the 135 of 135∠45)."), "135", False, _A,
           width=14),
    Column("dip", N_("Dip below the horizontal, 0-90° (the 45 of 135∠45)."), "45", False, _A),
    Column("thickness", N_("Width of a fault zone, vein ... (m); empty: a plane."), "", False, _N),
    Column("fill", N_("Filling, e.g. clay gouge, quartz."), "", False, _T, width=16),
    Column("description", N_("Description (may be empty)."), "", False, _T, width=24),
    Column("mapped_on", N_("Day it was measured."), "2026-03-31", False, _D),
    Column("geologist", N_("Geologist."), "", False, _T, width=14),
]
_SAMPLE_HEAD = [
    _PIT,
    Column("sample", N_("Sample number (unique in the project)."), "PS-001", True, _T),
    Column("from_m", N_("Where the sample starts (metres, or a station and metres past it)."), "30", True, _T,
           width=12),
    Column("to_m", N_("Where the sample ends (metres, or a station and metres past it)."), "31.5", True, _T,
           width=12),
    Column("wall", N_("Wall: left, right, back or face (left / right looking toward the face)."), "left", True, _L,
           tuple(SAMPLE_WALLS)),
    Column("height", N_("Height of the channel above the floor (m); empty: 1 m."), "1.0", False, _N),
    Column("kind", N_("Type of sample (empty: channel)."), "channel", False, _L, tuple(SAMPLE_KINDS)),
    Column("width_cm", N_("Channel width (cm)."), "10", False, _N),
    Column("depth_cm", N_("Channel depth (cm)."), "3", False, _N),
    Column("weight_kg", N_("Sample weight (kg)."), "3.2", False, _N),
]
_SAMPLE_TAIL = [
    Column("taken_on", N_("Day it was taken."), "2026-03-31", False, _D),
    Column("sampler", N_("Sampler."), "", False, _T, width=14),
    Column("note", N_("Note (may be empty)."), "", False, _T, width=24),
]
_CHAIN_COLUMNS = {"from_m", "to_m", "at_m"}
# other headers that mean the same column (matched like header_map: case, spaces and _ ignored)
_ALIASES = {"from_m": ("from",), "to_m": ("to",), "at_m": ("at", "chainage"), "kind": ("type",),
            "mapped_on": ("date",), "taken_on": ("date",), "sample": ("sample_id", "number"), "wall": ("where",)}


def sample_columns(assay_fields=()) -> list:
    return _SAMPLE_HEAD + [Column(f, N_("Assay value (empty = not assayed)."), "", False, _N) for f in assay_fields] \
        + _SAMPLE_TAIL


def sheets(assay_fields=()) -> list[Sheet]:
    return [Sheet("rocks", N_("One row per rock interval along the drive."), ROCK_COLUMNS),
            Sheet("structures", N_("One row per structure measured (bedding, fault, joint, vein ...)."),
                  STRUCTURE_COLUMNS),
            Sheet("samples", N_("One row per sample, with its assays (one column per assay field)."),
                  sample_columns(assay_fields))]


def _norm(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# --- the template and the export -------------------------------------------------------------------
def _pits_and_stations(survey: SurveyStore):
    """[(pit, {station: chainage})] in the Pit Manager's order (main pits, then branches)."""
    from ..scene.pit_geology import pit_axis
    from ..scene.survey import SurveyData, pit_paths

    pits = survey.pits()
    data = SurveyData(pits, {s.name: s for s in survey.stations()}, pit_paths(survey.setups()), "")
    out = []
    for p in sorted(pits, key=lambda q: (q.parent_id is not None, q.id)):
        ax = pit_axis(data, p.id)
        out.append((p, ax.station_chainage() if ax is not None else {}))
    return out


def _write_stations(wb, survey: SurveyStore) -> int:
    """The reference sheet: every pit's stations and their chainage."""
    from openpyxl.styles import Font, PatternFill

    ws = wb.create_sheet("stations")
    for c, h in enumerate(("pit", "code", "station", "chainage_m"), start=1):
        cell = ws.cell(1, c, h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="5A6470")
    r = 1
    for p, stations in _pits_and_stations(survey):
        for name, m in sorted(stations.items(), key=lambda kv: kv[1]):
            r += 1
            ws.cell(r, 1, p.name)
            ws.cell(r, 2, p.code)
            ws.cell(r, 3, name)
            ws.cell(r, 4, round(float(m), 3)).number_format = "0.000"
    ws.freeze_panes = "A2"
    for letter, w in zip("ABCD", (22, 10, 14, 12)):
        ws.column_dimensions[letter].width = w
    return r - 1


def save_template(store: PitGeologyStore, path) -> None:
    """An empty workbook to fill in (the project's assay fields as columns of the samples sheet),
    and the stations sheet to look the chainages up in."""
    import openpyxl

    store.prepare()
    write_template(path, tr("GeoViewer - pit geology to import"), [
        tr("1. Sheets rocks, structures and samples: one row per record, each naming its pit (name or code, as in "
           "the Pit Manager). The pits must be in the database already."),
        tr("2. Chainages: metres from the pit's start - 0 m is the entrance of a main pit, the junction station "
           "of a branch - or a station and metres past it as text, e.g. P1-03+4.5. The sheet stations lists the "
           "stations and their chainage."),
        tr("3. Left / right wall as seen looking toward the face. Orientation as dip direction and dip "
           "(135∠45: dip direction 135, dip 45)."),
        tr("4. In GeoViewer: Pit Geology - Import from Excel, choose this file, check, Import."),
        tr("Orange columns are required. A sheet may be left empty. More assay columns (e.g. copper) may be added "
           "to the samples sheet: they become new assay fields."),
    ], sheets(store.assay_fields()))
    wb = openpyxl.load_workbook(Path(path))
    _write_stations(wb, SurveyStore(store.target))
    wb.save(Path(path))


def export_excel(store: PitGeologyStore, path) -> dict[str, int]:
    """Every pit's geology to a workbook the import reads back (sheets rocks, structures,
    samples, and the stations sheet). Returns the rows per sheet. Raises PermissionError when
    the file is open in Excel."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    survey = SurveyStore(store.target)
    pits = _pits_and_stations(survey)
    geology = store.geology()
    fields = store.assay_fields()
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    counts = {}
    for sh in sheets(fields):
        ws = wb.create_sheet(sh.name)
        names = [c.name for c in sh.columns]
        for c, col in enumerate(sh.columns, start=1):
            cell = ws.cell(1, c, col.name)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="B4532A" if col.required else "2F5D8A")
            ws.column_dimensions[get_column_letter(c)].width = col.width or max(10, len(col.name) + 3)
        ws.freeze_panes = "A2"
        r = 1
        for p, _st in pits:
            geo = geology.get(p.id)
            if geo is None:
                continue
            for rec in getattr(geo, {"rocks": "intervals"}.get(sh.name, sh.name)):
                r += 1
                for c, name in enumerate(names, start=1):
                    v = p.name if name == "pit" else _record_value(rec, name, fields)
                    if v is None or v == "":
                        continue
                    cell = ws.cell(r, c, round(v, 4) if isinstance(v, float) else v)
                    if name in _CHAIN_COLUMNS:
                        cell.number_format = "0.00"
        counts[sh.name] = r - 1
    counts["stations"] = _write_stations(wb, survey)
    wb.save(Path(path))
    return counts


_ATTR = {"from_m": "frm", "to_m": "to", "at_m": "at", "sample": "name"}


def _record_value(rec, name: str, assay_fields=()):
    if isinstance(rec, GeoSample) and name in assay_fields:
        return rec.values.get(name)
    return getattr(rec, _ATTR.get(name, name), None)


# --- reading and checking --------------------------------------------------------------------------
@dataclass
class PitPlan:
    pid: int
    name: str
    stored: PitGeology                                # in the database (empty: nothing mapped yet)
    length: float | None = None                       # surveyed length (None: not surveyed far enough)
    given: dict = field(default_factory=dict)         # kind -> [records] from the file
    rows: dict = field(default_factory=dict)          # kind -> their rows in the sheet
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def new(self) -> bool:
        return self.stored.is_empty()

    def final(self) -> PitGeology:
        """The pit's geology after the import: the file's records for each sheet that has rows for
        it, the stored ones for the others."""
        return PitGeology(*[self.given.get(k, getattr(self.stored, k)) for k in KINDS])


@dataclass
class GeologyImport:
    path: str
    pits: list = field(default_factory=list)          # [PitPlan] in the order met in the file
    errors: list = field(default_factory=list)        # not of one pit
    warnings: list = field(default_factory=list)
    new_fields: list = field(default_factory=list)    # assay fields to add
    pit_names: dict = field(default_factory=dict)     # pit id -> name
    db_samples: dict = field(default_factory=dict)    # sample number -> pit id (in the database)
    snapshot: tuple = ()                              # the geology tables when read (rows, highest id)

    def todo(self, replace: bool) -> list[PitPlan]:
        return [p for p in self.pits if p.new or replace]

    def problems(self, replace: bool) -> tuple[list[str], list[str]]:
        errors, warnings = list(self.errors), list(self.warnings)
        todo = self.todo(replace)
        for p in todo:
            errors += p.errors
            warnings += p.warnings
        # sample numbers are unique in the project: the database's as they will be, then the file's
        owner = dict(self.db_samples)
        replaced = {p.pid for p in todo if "samples" in p.given}
        owner = {n: pid for n, pid in owner.items() if pid not in replaced}
        for p in todo:
            for s, row in zip(p.given.get("samples", []), p.rows.get("samples", [])):
                other = owner.get(s.name)
                if s.name and other is not None and other != p.pid:
                    errors.append(tr("Sheet {sheet} row {row}: sample {name} is already in pit {pit}", sheet="samples",
                                     row=row, name=s.name, pit=self.pit_names.get(other, "?")))
                elif s.name:
                    owner[s.name] = p.pid
        return errors, warnings

    def summary(self, replace: bool) -> list[str]:
        new = sum(1 for p in self.pits if p.new)
        old = len(self.pits) - new
        lines = [tr("Pits in the file: {n:,} ({new:,} with no geology yet, {old:,} already mapped)",
                    n=len(self.pits), new=new, old=old)]
        if old:
            lines.append(tr("Pits already mapped: their rocks, structures or samples replaced by the file's (each "
                            "sheet that has rows for them)") if replace else tr("Pits already mapped: skipped"))
        todo = self.todo(replace)
        n = {k: sum(len(p.given.get(k, [])) for p in todo) for k in KINDS}
        lines.append(tr("Records to import: rocks {rocks:,}, structures {structures:,}, samples {samples:,}",
                        rocks=n["intervals"], structures=n["structures"], samples=n["samples"]))
        if self.new_fields:
            lines.append(tr("New assay fields: {names}", names=", ".join(self.new_fields)))
        return lines


def _choice(v, keys: dict) -> str | None:
    """A wall / type cell as its key: the key, its English or translated name, or an alias."""
    t = (cell_text(v) or "").strip()
    if not t:
        return None
    low = _norm(t)
    for k, label in keys.items():
        names = (k, label, tr(label), *_WALL_ALIASES.get(k, ()), *_KIND_ALIASES.get(k, ()))
        if low in {_norm(x) for x in names} or t in names:
            return k
    raise ValueError(tr("must be one of: {values}", values=", ".join(keys)))


def _date(v) -> str:
    t = cell_date(v)
    if t is None:
        return ""
    try:
        _dt.date.fromisoformat(t)
    except ValueError:
        raise ValueError(tr("must be a date like 2026-03-31")) from None
    return t


def _snapshot(db) -> tuple:
    out = []
    for t in GEO_TABLES:
        if db.has_table(t):
            out.append(tuple(db.query(f"SELECT COUNT(*), MAX(id) FROM {t}")[0]))
        else:
            out.append((0, None))
    return tuple(out)


def _columns(headers, columns, sheet, plan, assays: dict | None = None) -> dict[str, str]:
    """Column name -> the sheet's header for it. ``assays`` (samples): the sheet's other headers
    as assay fields, header -> field name (filled in here)."""
    wanted = {c.name for c in columns}
    found, used = {}, set()
    by_norm = {_norm(h): h for h in headers}
    for name in [c.name for c in columns]:
        for alias in (name, *_ALIASES.get(name, ())):
            h = by_norm.get(_norm(alias))
            if h is not None and h not in used:
                found[name] = h
                used.add(h)
                break
    for h in headers:
        if h in used:
            continue
        if assays is not None and _norm(h) not in {_norm(w) for w in wanted}:
            assays[h] = h
            continue
        plan.warnings.append(tr("Sheet {sheet}: column {column} is not one of the template's - left out",
                                sheet=sheet, column=h))
    return found


def read_import(store: PitGeologyStore, path) -> GeologyImport:
    """Read and check a filled-in workbook (nothing is written)."""
    from ..scene.pit_geology import parse_chainage

    store.prepare()
    plan = GeologyImport(str(path))
    survey = SurveyStore(store.target)
    pits = _pits_and_stations(survey)
    plan.pit_names = {p.id: p.name for p, _s in pits}
    lookup: dict[str, int] = {}
    for p, _s in pits:                                  # codes first, so a name wins over a code
        if p.code:
            lookup[p.code.strip().lower()] = p.id
    for p, _s in pits:
        lookup[p.name.strip().lower()] = p.id
    stations = {p.id: s for p, s in pits}
    stored = store.geology()
    fields = store.assay_fields()
    field_of = {f.lower(): f for f in fields}
    with store.target.connect(create=False) as db:
        plan.snapshot = _snapshot(db)
    plan.db_samples = store.sample_names()
    book = read_workbook(path)
    found = {k: next((book[n] for n in names if n in book), None) for k, names in _SHEET_NAMES.items()}
    if not any(found.values()):
        plan.errors.append(tr("The workbook has none of the sheets rocks, structures and samples."))
        return plan

    by_pid: dict[int, PitPlan] = {}
    unknown: set = set()
    specs = {"intervals": ROCK_COLUMNS, "structures": STRUCTURE_COLUMNS, "samples": sample_columns()}
    for kind in KINDS:
        sr = found[kind]
        if sr is None:
            continue
        sheet = sr.name
        assays: dict | None = {} if kind == "samples" else None
        cols = _columns(sr.headers, specs[kind], sheet, plan, assays)
        if "pit" not in cols:
            if sr.rows:
                plan.errors.append(tr("Sheet {sheet}: no pit column", sheet=sheet))
            continue
        # samples: the other columns are assay fields (new ones only when they hold numbers)
        assay_of: dict[str, str] = {}
        for h in (assays or {}):
            f = field_of.get(h.lower())
            if f is None and h.lower() in {x.lower() for x in plan.new_fields}:
                f = next(x for x in plan.new_fields if x.lower() == h.lower())
            if f is None:
                values = [rec.get(h) for _n, rec in sr.rows if rec.get(h) is not None]
                if not values:
                    plan.warnings.append(tr("Sheet {sheet}: column {column} is empty - left out", sheet=sheet,
                                            column=h))
                    continue
                if not _FIELD_NAME.match(h) or h.lower() in _FIXED_SAMPLE_COLUMNS:
                    plan.errors.append(tr("Sheet {sheet}, column {column}: {problem}", sheet=sheet, column=h,
                                          problem=tr("an assay field name starts with a letter and has only letters, "
                                                     "digits and _ (e.g. gold, cu_pct)")))
                    continue
                if any(_number_or_none(v) is _BAD for v in values):
                    plan.warnings.append(tr("Sheet {sheet}: column {column} is not an assay field and holds text - "
                                            "left out", sheet=sheet, column=h))
                    continue
                f = h
                plan.new_fields.append(f)
            assay_of[h] = f
        for n, rec in sr.rows:
            pit_text = cell_text(rec.get(cols["pit"]))
            if not pit_text:
                plan.errors.append(tr("Sheet {sheet} row {row}: no pit", sheet=sheet, row=n))
                continue
            pid = lookup.get(pit_text.strip().lower())
            if pid is None:
                if pit_text.lower() not in unknown:
                    unknown.add(pit_text.lower())
                    plan.errors.append(tr("Sheet {sheet} row {row}: pit {pit} is not in the database - add it in the "
                                          "Pit Manager first", sheet=sheet, row=n, pit=pit_text))
                continue
            pp = by_pid.get(pid)
            if pp is None:
                pp = by_pid[pid] = PitPlan(pid, plan.pit_names[pid], stored.get(pid) or PitGeology())
                st = stations.get(pid) or {}
                pp.length = max(st.values()) if st else None
                plan.pits.append(pp)
            problems: list = []
            bad: set = set()                            # columns already reported

            def get(name, conv, rec=rec, problems=problems, bad=bad):
                h = cols.get(name)
                if h is None:
                    return None
                try:
                    return conv(rec.get(h))
                except ValueError as e:
                    problems.append(tr("column {column}: {problem}", column=h, problem=e))
                    bad.add(name)
                    return None

            def needed(name, value, problems=problems, bad=bad):
                if value is None and name not in bad:
                    problems.append(tr("column {column}: {problem}", column=cols.get(name, name),
                                       problem=tr("is needed")))

            def chain(v, pid=pid):
                if isinstance(v, str):
                    return parse_chainage(v, stations.get(pid))
                return cell_number(v)

            text = (lambda v: cell_text(v) or "")
            if kind == "intervals":
                r = GeoInterval(get("from_m", chain), get("to_m", chain),
                                get("wall", lambda v: _choice(v, INTERVAL_WALLS)) or "all",
                                get("rock", text), get("description", text), get("alteration", text),
                                get("mineralization", text), get("mapped_on", _date), get("geologist", text))
            elif kind == "structures":
                wall = get("wall", lambda v: _choice(v, WALLS))
                skind = get("kind", lambda v: _choice(v, STRUCTURE_KINDS))
                needed("wall", wall)
                needed("kind", skind)
                r = GeoStructure(get("at_m", chain), wall or "left", get("height", cell_number), skind or "other",
                                 get("dip_direction", parse_angle), get("dip", parse_angle),
                                 get("thickness", cell_number), get("fill", text), get("description", text),
                                 get("mapped_on", _date), get("geologist", text))
            else:
                wall = get("wall", lambda v: _choice(v, SAMPLE_WALLS))
                needed("wall", wall)
                values = {}
                for h, f in assay_of.items():
                    try:
                        values[f] = cell_number(rec.get(h))
                    except ValueError:
                        problems.append(tr("column {column}: {problem}", column=h,
                                           problem=tr("not a number: {t!r}", t=cell_text(rec.get(h)))))
                r = GeoSample(get("sample", text), get("from_m", chain), get("to_m", chain), wall or "left",
                              get("height", cell_number),
                              get("kind", lambda v: _choice(v, SAMPLE_KINDS)) or "channel",
                              get("width_cm", cell_number), get("depth_cm", cell_number),
                              get("weight_kg", cell_number), values, get("taken_on", _date), get("sampler", text),
                              get("note", text))
            for e in problems:
                pp.errors.append(tr("Sheet {sheet} row {row}: {problem}", sheet=sheet, row=n, problem=e))
            pp.given.setdefault(kind, []).append(r)
            pp.rows.setdefault(kind, []).append(n)

    for pp in plan.pits:
        # check what the file gives (the kept records were checked when they were saved)
        geo = PitGeology(*[pp.given.get(k, []) for k in KINDS])
        errors, warnings = check_geology(geo, pp.length, fields + plan.new_fields, pp.rows)
        pp.errors += [tr("Pit {pit}: {problem}", pit=pp.name, problem=e) for e in errors]
        pp.warnings += [tr("Pit {pit}: {problem}", pit=pp.name, problem=w) for w in warnings]
    return plan


_BAD = object()


def _number_or_none(v):
    try:
        return cell_number(v)
    except ValueError:
        return _BAD


# --- writing ---------------------------------------------------------------------------------------
def apply_import(store: PitGeologyStore, plan: GeologyImport, replace: bool) -> dict:
    """Write the plan (it must have no errors for ``replace``): pits with no geology get the
    file's, mapped pits get theirs replaced sheet by sheet (``replace``) or are skipped. New assay
    fields are added first, then one transaction. Returns the counts: added, updated, skipped
    and the records per kind."""
    errors, _w = plan.problems(replace)
    if errors:
        raise ValueError(tr("The workbook has errors - nothing was imported."))
    todo = plan.todo(replace)
    counts = {"added": 0, "updated": 0, "skipped": len(plan.pits) - len(todo), **{k: 0 for k in KINDS}}
    with store.target.connect() as db:
        if _snapshot(db) != plan.snapshot:
            raise ValueError(tr("The pit geology in the database changed since the file was checked - check it "
                                "again."))
        have = {f.lower() for f in store.assay_fields(db)}
        for f in plan.new_fields:
            if f.lower() not in have:
                db.add_column(SAMPLES, f, "real")
        db.con.commit()
        fields = store.assay_fields(db)
        with db.transaction():
            for pp in todo:
                store._write(db, pp.pid, sorted_geology(pp.final()), fields)
                counts["added" if pp.new else "updated"] += 1
                for k in KINDS:
                    counts[k] += len(pp.given.get(k, []))
    return counts


def result_text(counts: dict) -> str:
    return tr("Imported - pits added: {added:,}, replaced: {updated:,}, skipped: {skipped:,}; rocks {rocks:,}, "
              "structures {structures:,}, samples {samples:,}", added=counts["added"], updated=counts["updated"],
              skipped=counts["skipped"], rocks=counts["intervals"], structures=counts["structures"],
              samples=counts["samples"])
