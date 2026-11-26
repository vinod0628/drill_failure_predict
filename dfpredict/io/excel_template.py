"""Excel templates to fill in and import (drillholes, survey): writing a template workbook and
reading the filled-in sheets back. No Qt here.

A template has one sheet per table, its first row the column names (data names, kept in English
so the import finds them in any language), and an instructions sheet in the user's language:
what one row is, every column's meaning, whether it is required, and an example. Column notes
on the headers, lists to pick from (states, bearing kinds ...), number checks, and text format
for ids and angles (so Excel does not turn "120:30:00" into a time) help while typing.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..i18n import tr

FORMATTED_ROWS = 2000                # rows of the data sheets given text / date formats and checks
_HEAD_FILL, _REQ_FILL, _HEAD_TEXT = "2F5D8A", "B4532A", "FFFFFF"


@dataclass
class Column:
    name: str                        # the header: the data name the import looks for
    help: str                        # its meaning (English, N_-marked; translated when written)
    example: str = ""
    required: bool = False
    kind: str = "text"               # "text", "number", "date", "angle" or "list"
    choices: tuple = ()              # kind "list": the values to pick from
    width: int = 0                   # column width (0: from the header / example)


@dataclass
class Sheet:
    name: str                        # data name, e.g. "collar"
    help: str                        # what one row is (N_-marked)
    columns: list[Column] = field(default_factory=list)


def write_template(path, title: str, intro: list[str], sheets: list[Sheet]) -> None:
    """An empty workbook to fill in: one sheet per ``Sheet`` (header row only) and an
    instructions sheet first. ``title`` / ``intro`` are already translated. Raises
    PermissionError when the file is open in Excel."""
    import openpyxl
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = openpyxl.Workbook()
    info = wb.active
    info.title = _sheet_title(tr("Instructions"))
    bold = Font(bold=True)
    head_font = Font(bold=True, color=_HEAD_TEXT)
    info["A1"] = title
    info["A1"].font = Font(bold=True, size=14)
    r = 3
    for line in intro:
        info.cell(r, 1, line)
        r += 1
    r += 1
    heads = [tr("Sheet"), tr("Column"), tr("Required"), tr("Meaning"), tr("Example")]
    for c, h in enumerate(heads, start=1):
        cell = info.cell(r, c, h)
        cell.font = head_font
        cell.fill = PatternFill("solid", fgColor=_HEAD_FILL)
    info.freeze_panes = info.cell(r + 1, 1)
    for sh in sheets:
        r += 1
        info.cell(r, 1, sh.name).font = bold
        info.cell(r, 4, tr(sh.help)).alignment = Alignment(wrap_text=True, vertical="top")
        for col in sh.columns:
            r += 1
            info.cell(r, 2, col.name).font = bold if col.required else Font()
            info.cell(r, 3, tr("yes") if col.required else "")
            info.cell(r, 4, _meaning(col)).alignment = Alignment(wrap_text=True, vertical="top")
            info.cell(r, 5, col.example)
    for c, w in enumerate((14, 20, 10, 80, 22), start=1):
        info.column_dimensions[get_column_letter(c)].width = w

    for sh in sheets:
        ws = wb.create_sheet(_sheet_title(sh.name))
        ws.freeze_panes = "A2"
        for c, col in enumerate(sh.columns, start=1):
            letter = get_column_letter(c)
            cell = ws.cell(1, c, col.name)
            cell.font = head_font
            cell.fill = PatternFill("solid", fgColor=_REQ_FILL if col.required else _HEAD_FILL)
            note = _meaning(col) + (f"\n{tr('Example')}: {col.example}" if col.example else "")
            if col.required:
                note = tr("Required.") + " " + note
            cell.comment = Comment(note, "GeoViewer", width=320, height=140)
            ws.column_dimensions[letter].width = col.width or min(40, max(10, len(col.name) + 4,
                                                                          len(col.example) + 2))
            rng = f"{letter}2:{letter}{FORMATTED_ROWS + 1}"
            if col.kind in ("text", "angle", "list"):
                for row in ws.iter_rows(min_row=2, max_row=FORMATTED_ROWS + 1, min_col=c, max_col=c):
                    row[0].number_format = "@"
            elif col.kind == "date":
                for row in ws.iter_rows(min_row=2, max_row=FORMATTED_ROWS + 1, min_col=c, max_col=c):
                    row[0].number_format = "yyyy-mm-dd"
            if col.kind == "list" and col.choices:
                dv = DataValidation(type="list", formula1='"' + ",".join(col.choices) + '"', allow_blank=True)
                dv.error = tr("Pick one of: {values}", values=", ".join(col.choices))
                dv.errorTitle = col.name
                dv.showErrorMessage = True
                ws.add_data_validation(dv)
                dv.add(rng)
            elif col.kind == "number":
                dv = DataValidation(type="decimal", operator="between", formula1="-1E+15", formula2="1E+15",
                                    allow_blank=True)
                dv.error = tr("{column} must be a number", column=col.name)
                dv.errorTitle = col.name
                dv.showErrorMessage = True
                ws.add_data_validation(dv)
                dv.add(rng)
    wb.active = 0
    path = Path(path)
    wb.save(path)


def _meaning(col: Column) -> str:
    text = tr(col.help)
    if col.kind == "list" and col.choices:
        text += " " + tr("One of: {values}.", values=", ".join(col.choices))
    elif col.kind == "date":
        text += " " + tr("A date, e.g. 2026-03-31.")
    elif col.kind == "angle":
        text += " " + tr("Degrees: 120 30 00 (degrees minutes seconds) or 120.5.")
    return text


def _sheet_title(name: str) -> str:
    bad = set('[]:*?/\\')
    return "".join("_" if ch in bad else ch for ch in name)[:31] or "sheet"


# --- reading the filled-in workbook --------------------------------------------------------------
@dataclass
class SheetRows:
    name: str                        # as in the workbook
    headers: list[str]               # the first row's column names (empty ones left out)
    rows: list[tuple[int, dict]]     # (row number in Excel, header -> value) for rows with a value


def read_workbook(path) -> dict[str, SheetRows]:
    """Every sheet: lower-case name -> its rows. Blank rows are skipped; values come as Excel
    holds them (numbers, text, dates)."""
    import openpyxl

    wb = openpyxl.load_workbook(Path(path), read_only=True, data_only=True)
    out: dict[str, SheetRows] = {}
    try:
        for ws in wb.worksheets:
            it = ws.iter_rows(values_only=True)
            first = next(it, None)
            if not first:
                continue
            headers = [(i, str(h).strip()) for i, h in enumerate(first) if h is not None and str(h).strip()]
            rows = []
            for n, values in enumerate(it, start=2):
                rec = {}
                for i, h in headers:
                    v = values[i] if i < len(values) else None
                    if isinstance(v, str):
                        v = v.strip() or None
                    if v is not None:
                        rec[h] = v
                if rec:
                    rows.append((n, rec))
            out[ws.title.strip().lower()] = SheetRows(ws.title, [h for _i, h in headers], rows)
    finally:
        wb.close()
    return out


def cell_text(v) -> str | None:
    """A cell as text: 101.0 -> "101", a date -> "2026-03-31", blanks -> None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
    if isinstance(v, _dt.datetime):
        return v.date().isoformat() if v.time() == _dt.time() else v.isoformat(sep=" ")
    if isinstance(v, _dt.date):
        return v.isoformat()
    s = str(v).strip()
    return s or None


def cell_date(v) -> str | None:
    """A date cell as YYYY-MM-DD (text is kept as typed, so the checks can name it)."""
    if isinstance(v, (_dt.datetime, _dt.date)):
        return (v.date() if isinstance(v, _dt.datetime) else v).isoformat()
    return cell_text(v)


def cell_number(v):
    """A number, None for blank; ValueError for anything else."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    return float(s)


def header_map(headers: list[str], wanted) -> dict[str, str]:
    """Wanted column name -> the workbook's header for it (matched ignoring case, spaces and _)."""
    def norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    have = {norm(h): h for h in headers}
    return {w: have[norm(w)] for w in wanted if norm(w) in have}
