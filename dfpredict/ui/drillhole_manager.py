"""Drillhole Manager: add / remove holes and edit their collar, survey, geology and assay
rows in the project database, the rock colours (drillhole_styles) and the assay fields - and
the drilling records the risk prediction learns from (drilling_schema.py, the same tables as
the Drilling Failure Prediction system): the drilling fields of the collar and the geology,
runs, hole sizes, bits, occurrences and the failure of each hole, and the reference tables
(areas, formation cards, rigs, measure library). The drilling history can be read from and
written to CSV files (the exchange format of the risk prediction).

A hole is edited in place and written with Save (all its tables in one transaction);
rock colours and assay fields are written at once. After every write ``changed`` is
emitted so the drillhole layer is read again.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..i18n import N_, tr, trc
from ..io import drilling_schema as ds
from ..io.drillhole_store import (ASSAY, COLLAR, DRILLING_TABLES, GEOLOGY, HOLE_STATES, SURVEY, DrillholeStore,
                                  HoleData, check_hole, state_code)
from . import theme
from .icons import icon

PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
           "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3"]
COLLAR_LABELS = {"organization": N_("Organization"), "drilling_manager": N_("Drilling manager"),
                 "geology_surveyor": N_("Geology surveyor"), "start_date": N_("Start date"),
                 "end_date": N_("End date"), "state": N_("State")}          # tr() where shown
HEADERS = {"depth": N_("Depth (m)"), "azimuth": N_("Azimuth (°)"), "dip": N_("Dip (°, 0 = down)"),
           "from": N_("From (m)"), "to": N_("To (m)"), "rock": N_("Rock"), "note": N_("Note"),
           "x": N_("X (East)"), "y": N_("Y (North)"), "z": N_("Z (Elevation)"), "max_depth": N_("Length (m)")}


DRILLING_TITLES = {ds.RUN.name: N_("Runs"), ds.HOLE_SIZE.name: N_("Hole size"), ds.BIT.name: N_("Bit"),
                   ds.OCCURRENCE.name: N_("Occurrences"), ds.FAILURE.name: N_("Failure")}
DRILLING_HINTS = {
    ds.RUN.name: N_("One row per core run (1.5-6 m): From = the previous run's To. Next run copies the last run's "
                    "settings. Per-run values (return, pump pressure, rod drop ...) record the warning signs of problems."),
    ds.HOLE_SIZE.name: N_("Hole size by depth. Each reduction usually marks casing being set."),
    ds.BIT.name: N_("Bit type by depth (and matrix / series)."),
    ds.OCCURRENCE.name: N_("Cavities, fault zones, broken rock, clay, water, losses - solved or not, with the measures "
                           "taken (Measures ... picks them from the library)."),
    ds.FAILURE.name: N_("Only for a failed hole: one row with the failure depth, code and severity, and the measures "
                        "tried. Set the state of the collar to Failed."),
}
RUN_COPY = ("rpm", "rop", "flow_in", "pump_press", "wob", "torque", "fluid_type")   # Next run: from the last run


def _header(role: str, default: str) -> str:
    """The column title of a role (translated), else ``default``."""
    return tr(HEADERS[role]) if role in HEADERS else default


def schema_col(table: str | None, col: str) -> ds.Col | None:
    """The shared schema's description of a column (None for columns it does not know)."""
    try:
        t = ds.table(table or "")
    except KeyError:
        return None
    return next((c for c in t.cols if c.name == str(col).lower()), None)


def col_title(c: ds.Col) -> str:
    return tr(c.label) + (f" ({c.unit})" if c.unit else "")


def choice_text(choices: dict, code) -> str:
    """How a code is shown: its text (translated), or the code itself."""
    if code is None or code == "":
        return ""
    key = ds._code_text(code)
    text = choices.get(key)
    return tr(text) if text and text != key else key


def _swatch(color: str, size: int = 14) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    return QIcon(pm)


def rock_label(code: str, desc: str) -> str:
    """How a rock is shown: "ST (Schist)", or just the code without a description."""
    return f"{code} ({desc})" if desc else code


def rock_code(label: str) -> str:
    """The code from a shown rock ("ST (Schist)" -> "ST")."""
    return str(label or "").split(" (", 1)[0].strip()


def _text(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, float):
        return f"{v:.6g}" if abs(v) < 1e6 else f"{v:.3f}"
    return str(v)


class _DateEdit(QDateEdit):
    """A date that may be empty (shown blank): the empty date is the minimum date."""

    EMPTY = QDate(1900, 1, 1)

    def __init__(self, iso: str = ""):
        super().__init__()
        self.setCalendarPopup(True)
        self.setDisplayFormat("yyyy-MM-dd")
        self.setMinimumDate(self.EMPTY)
        self.setSpecialValueText(" ")                 # the minimum date shows as blank
        d = QDate.fromString(iso or "", "yyyy-MM-dd")
        self.setDate(d if d.isValid() else self.EMPTY)
        self.setToolTip(tr("Pick a date from the calendar; clear it with Delete"))

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selectedText() == self.text():
            self.setDate(self.EMPTY)
            return
        super().keyPressEvent(e)

    def value(self) -> str:
        return "" if self.date() == self.EMPTY else self.date().toString("yyyy-MM-dd")


def _widget_value(w) -> str:
    if isinstance(w, _DateEdit):
        return w.value()
    if isinstance(w, QComboBox):
        if w.isEditable():                            # a list of known values, or a new one typed in
            i = w.findText(w.currentText())
            code = w.itemData(i) if i >= 0 else None
            return str(code) if code is not None else w.currentText().strip()
        code = w.currentData()
        return "" if code is None else str(code)
    return w.text()


class _Delegate(QStyledItemDelegate):
    """Editors per column: numbers get a number field, the rock column a list of codes."""

    def __init__(self, grid):
        super().__init__(grid.table)
        self.grid = grid

    def createEditor(self, parent, option, index):
        col = self.grid.columns[index.column()]
        if col in self.grid.choices:                  # a code: pick it from the list
            box = QComboBox(parent)
            box.addItem("", None)
            for code, text in self.grid.choices[col].items():
                box.addItem(choice_text(self.grid.choices[col], code), code)
            current = index.data(Qt.UserRole)
            if current not in (None, "") and box.findData(current) < 0:
                box.addItem(str(current), current)    # a code that is not in the list: kept
            return box
        if col == self.grid.rock_column:
            box = QComboBox(parent)                   # registered rocks only (Rock colours tab)
            for code, color, desc in self.grid.rock_codes:
                box.addItem(_swatch(color), rock_label(code, desc), code)
            current = rock_code(index.data() or "")
            if current and box.findData(current) < 0:  # a code that is not registered: kept, marked
                box.insertItem(0, tr("{code} (not registered)", code=current), current)
            return box
        ed = QLineEdit(parent)
        if self.grid.info.numeric.get(col):
            v = QDoubleValidator(ed)
            v.setNotation(QDoubleValidator.StandardNotation)
            ed.setValidator(v)
        return ed

    def setEditorData(self, editor, index):
        text = index.data() or ""
        col = self.grid.columns[index.column()]
        if isinstance(editor, QComboBox):
            code = index.data(Qt.UserRole) if col in self.grid.choices else rock_code(text)
            i = editor.findData(code) if code not in (None, "") else 0
            editor.setCurrentIndex(max(i, 0))
            editor.showPopup()                        # one click opens the list
        else:
            editor.setText(text)

    def setModelData(self, editor, model, index):
        col = self.grid.columns[index.column()]
        if isinstance(editor, QComboBox) and col in self.grid.choices:
            code = editor.currentData()
            model.setData(index, "" if code is None else str(code), Qt.UserRole)
            model.setData(index, choice_text(self.grid.choices[col], code))
        elif isinstance(editor, QComboBox):
            model.setData(index, self.grid.label_of(editor.currentData() or ""))
        else:
            model.setData(index, editor.text())


class _Grid(QWidget):
    """One per-hole table (survey, geology, assay, or a drilling record table) as an editable
    grid. Columns the shared drilling schema knows get its titles, units and tips; code columns
    (``choices``) are picked from a list, shown as their text and kept as the code."""

    edited = Signal()

    def __init__(self, hint: str, extra_buttons=(), copy_columns=()):
        super().__init__()
        self.info = None
        self.columns: list[str] = []
        self.rock_column = None
        self.rock_codes: list[tuple[str, str, str]] = []      # (code, colour, description)
        self.choices: dict[str, dict] = {}                     # column -> code -> text
        self.copy_columns = tuple(copy_columns)                # Next run: values taken from the row above
        self._filling = False
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setItemDelegate(_Delegate(self))
        self.table.itemChanged.connect(self._changed)
        add = QPushButton(icon("table-row-plus-after"), tr("Add row"))
        add.setToolTip(tr("Add a row at the end (From = the last To)"))
        add.clicked.connect(self.add_row)
        ins = QPushButton(icon("table-row-plus-before"), tr("Insert row"))
        ins.setToolTip(tr("Insert a row above the selected one"))
        ins.clicked.connect(self.insert_row)
        rem = QPushButton(icon("table-row-remove"), tr("Delete rows"))
        rem.setToolTip(tr("Delete the selected rows"))
        rem.clicked.connect(self.delete_rows)
        row = QHBoxLayout()
        if self.copy_columns:
            nxt = QPushButton(icon("skip-next-outline"), tr("Next run"))
            nxt.setToolTip(tr("Add the next run: From = the last To, the settings of the last run copied"))
            nxt.clicked.connect(self.next_row)
            row.addWidget(nxt)
        for b in (add, ins, rem):
            row.addWidget(b)
        row.addStretch(1)
        for b in extra_buttons:
            row.addWidget(b)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.hint = QLabel(hint)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(self.hint)
        v.addWidget(self.table, 1)
        v.addLayout(row)

    def set_data(self, info, df: pd.DataFrame, rock_codes=(), choices: dict | None = None) -> None:
        self._filling = True
        self.info = info
        self.columns = info.fields
        self.rock_column = info.roles.get("rock")
        self.rock_codes = list(rock_codes)
        self.choices = {}
        role_of = {c: k for k, c in info.roles.items()}
        t = self.table
        t.clear()
        t.setColumnCount(len(self.columns))
        cols = [schema_col(info.name, c) for c in self.columns]
        for i, (c, sc) in enumerate(zip(self.columns, cols)):
            if sc is not None and sc.kind in ("code", "yn") and c != self.rock_column and sc.choices is not None:
                self.choices[c] = dict((choices or {}).get(sc.name) or sc.choices)
            title = _header(role_of.get(c, ""), col_title(sc) if sc is not None else c)
            head = QTableWidgetItem(title)
            if sc is not None and sc.tip:
                head.setToolTip(tr(sc.tip))
            t.setHorizontalHeaderItem(i, head)
        head = t.horizontalHeader()
        note = info.roles.get("note") or next((c for c, sc in zip(self.columns, cols)
                                               if sc is not None and sc.kind == "note"), None)
        wide = len(self.columns) > 6                  # many fields: fixed widths, scrolled sideways
        for i, (c, sc) in enumerate(zip(self.columns, cols)):  # a note gets the room; the others a fixed width
            stretch = (c == note or note is None) and not wide
            head.setSectionResizeMode(i, QHeaderView.Stretch if stretch else QHeaderView.Interactive)
            if not stretch:
                if c == self.rock_column:
                    width = 190
                elif c == note:
                    width = 220
                elif sc is not None and sc.width:
                    width = sc.width
                else:
                    width = 120 if c in self.choices else 95
                t.setColumnWidth(i, width)
        head.setStretchLastSection(wide)
        t.setRowCount(len(df))
        colors = {c: col for c, col, _d in self.rock_codes}
        for r, rec in enumerate(df.to_dict("records")):
            for c, col in enumerate(self.columns):
                v = rec.get(col)
                if col in self.choices:
                    code = ds._code_text(v) if not ds.is_blank(v) else ""
                    it = QTableWidgetItem(choice_text(self.choices[col], code))
                    it.setData(Qt.UserRole, code)
                    t.setItem(r, c, it)
                    continue
                it = QTableWidgetItem(_text(v))
                if info.numeric.get(col):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == self.rock_column and it.text():
                    code = it.text()
                    it.setText(self.label_of(code))
                    if code in colors:
                        it.setIcon(_swatch(colors[code]))
                t.setItem(r, c, it)
        self._filling = False

    def label_of(self, code: str) -> str:
        desc = {c: d for c, _col, d in self.rock_codes}
        return rock_label(code, desc.get(code, "")) if code else ""

    def _cell(self, r: int, c: int) -> str:
        it = self.table.item(r, c)
        if it is None:
            return ""
        if self.columns[c] in self.choices:
            code = it.data(Qt.UserRole)
            return str(code).strip() if code not in (None, "") else it.text().strip()
        return it.text().strip()

    def frame(self) -> pd.DataFrame:
        """The rows as edited (rock cells give back the code only, code cells their code)."""
        t = self.table
        rock = self.columns.index(self.rock_column) if self.rock_column in self.columns else -1
        rows = []
        for r in range(t.rowCount()):
            vals = [self._cell(r, c) for c in range(len(self.columns))]
            if rock >= 0:
                vals[rock] = rock_code(vals[rock])
            if any(vals):
                rows.append([v or None for v in vals])
        return pd.DataFrame(rows, columns=self.columns)

    def _changed(self, item) -> None:
        if self._filling:
            return
        if self.columns[item.column()] == self.rock_column:
            colors = {c: col for c, col, _d in self.rock_codes}
            code = rock_code(item.text())
            self._filling = True
            item.setText(self.label_of(code))
            item.setIcon(_swatch(colors[code]) if code in colors else QIcon())
            self._filling = False
        self.edited.emit()

    def _new_row(self, at: int, copy: bool = False) -> None:
        self._filling = True
        self.table.insertRow(at)
        for c in range(len(self.columns)):
            it = QTableWidgetItem("")
            if self.columns[c] in self.choices:
                it.setData(Qt.UserRole, "")
            self.table.setItem(at, c, it)
        r = self.info.roles
        if at > 0 and "from" in r and "to" in r:          # continue from the row above
            prev = self.table.item(at - 1, self.columns.index(r["to"]))
            if prev is not None and prev.text():
                self.table.item(at, self.columns.index(r["from"])).setText(prev.text())
        if at > 0 and copy:
            for col in self.copy_columns:
                if col in self.columns:
                    c = self.columns.index(col)
                    src, dst = self.table.item(at - 1, c), self.table.item(at, c)
                    if src is not None:
                        dst.setText(src.text())
                        dst.setData(Qt.UserRole, src.data(Qt.UserRole))
        self._filling = False
        self.table.setCurrentCell(at, 0)
        self.edited.emit()

    def add_row(self) -> None:
        if self.info is not None:
            self._new_row(self.table.rowCount())

    def next_row(self) -> None:
        if self.info is not None:
            self._new_row(self.table.rowCount(), copy=True)

    def insert_row(self) -> None:
        if self.info is not None:
            self._new_row(max(self.table.currentRow(), 0))

    def delete_rows(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        if rows:
            self.edited.emit()

    def set_cell(self, row: int, col: str, value: str) -> None:
        """Set one cell (e.g. the measures picked from the library)."""
        if col in self.columns and 0 <= row < self.table.rowCount():
            c = self.columns.index(col)
            if self.table.item(row, c) is None:
                self.table.setItem(row, c, QTableWidgetItem(""))
            self.table.item(row, c).setText(value)

    def value(self, row: int, col: str) -> str:
        return self._cell(row, self.columns.index(col)) if col in self.columns and row >= 0 else ""


class MeasurePicker(QDialog):
    """Pick measure codes from the library (drillhole_measure), in the order they were tried."""

    def __init__(self, parent, measures: list[dict], selected: str = "", fail_codes=()):
        super().__init__(parent)
        self.setWindowTitle(tr("Measures"))
        self.resize(720, 520)
        chosen = [s.strip() for s in str(selected or "").replace(",", ";").split(";") if s.strip()]
        self.order = list(chosen)
        v = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("Search measures ..."))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        v.addWidget(self.search)
        self.fail_codes = {c for c in fail_codes if c}
        self.only = QCheckBox(tr("Only measures for {codes}", codes=", ".join(sorted(self.fail_codes))))
        self.only.setVisible(bool(self.fail_codes))
        self.only.setChecked(bool(self.fail_codes))
        self.only.toggled.connect(self._filter)
        v.addWidget(self.only)
        self.list = QListWidget()
        for m in measures:
            code = str(m.get("code") or "")
            fails = str(m.get("failures") or "")
            stage = tr("prevention") if str(m.get("stage") or "").upper() == "P" else tr("response")
            title = tr(str(m.get("title") or code))
            it = QListWidgetItem(f"{code}  ·  {title}   [{fails}]  {stage}")
            it.setData(Qt.UserRole, code)
            it.setData(Qt.UserRole + 1, {x.strip().upper() for x in fails.replace(";", ",").split(",") if x.strip()})
            it.setToolTip(tr(str(m.get("text") or "")))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if code in chosen else Qt.Unchecked)
            self.list.addItem(it)
        self.list.itemChanged.connect(self._ticked)
        v.addWidget(self.list, 1)
        self.picked = QLabel("")
        self.picked.setWordWrap(True)
        self.picked.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(self.picked)
        b = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b.accepted.connect(self.accept)
        b.rejected.connect(self.reject)
        v.addWidget(b)
        self._filter()
        self._show_order()

    def _ticked(self, it: QListWidgetItem) -> None:
        code = it.data(Qt.UserRole)
        if it.checkState() == Qt.Checked and code not in self.order:
            self.order.append(code)
        elif it.checkState() != Qt.Checked and code in self.order:
            self.order.remove(code)
        self._show_order()

    def _show_order(self) -> None:
        self.picked.setText(tr("In the order tried: {codes}", codes="; ".join(self.order) or "-"))

    def _filter(self, *_a) -> None:
        t = self.search.text().strip().lower()
        only = self.only.isChecked() and bool(self.fail_codes)
        for i in range(self.list.count()):
            it = self.list.item(i)
            hide = bool(t) and t not in (it.text() + " " + it.toolTip()).lower()
            if only and not (it.data(Qt.UserRole + 1) & self.fail_codes) and it.checkState() != Qt.Checked:
                hide = True
            it.setHidden(hide)

    def value(self) -> str:
        return ";".join(self.order)


class _NewHoleDialog(QDialog):
    def __init__(self, parent, collar_info, areas=(), area: str = ""):
        super().__init__(parent)
        self.setWindowTitle(tr("Add drillhole"))
        self.info = collar_info
        f = QFormLayout(self)
        self.hid = QLineEdit()
        self.hid.setPlaceholderText(tr("e.g. WD101"))
        f.addRow(tr("Hole ID"), self.hid)
        self.edits = {}
        for key in ("x", "y", "z", "max_depth"):
            col = collar_info.roles.get(key)
            if col:
                e = QLineEdit()
                e.setValidator(QDoubleValidator(e))
                f.addRow(tr(HEADERS[key]), e)
                self.edits[col] = e
        low = {c.lower(): c for c in collar_info.columns}
        if "state" in low:
            st = QComboBox()
            st.addItem("", None)
            for code, text in HOLE_STATES.items():
                st.addItem(tr(text), code)
            f.addRow(tr("State"), st)
            self.edits[low["state"]] = st
        if "area_id" in low:
            ar = QComboBox()
            ar.setEditable(True)
            ar.addItem("")
            ar.addItems(list(areas))
            ar.setCurrentText(area or "")
            ar.setToolTip(tr("The drilling area (site sheet in Drilling reference > Areas)"))
            f.addRow(tr("Area"), ar)
            self.edits[low["area_id"]] = ar
        note = QLabel(tr("The new hole has no survey yet: it is drawn straight down until survey rows "
                         "are added."))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow(note)
        b = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b.accepted.connect(self.accept)
        b.rejected.connect(self.reject)
        f.addRow(b)

    def values(self) -> tuple[str, dict]:
        return self.hid.text().strip(), {c: _widget_value(e) for c, e in self.edits.items()}


class DrillholeManager(QDialog):
    changed = Signal()            # the database was written: read the drillhole layer again

    def __init__(self, store: DrillholeStore, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Drillhole Manager"))
        self.resize(1240, 760)
        self.store = store
        self.hid: str | None = None
        self.dirty = False
        self.collar_info = None
        self.codes: dict[str, dict] = {}            # area_id / rig_model / formation / rock classes -> choices
        self.ref_grids: dict[str, _Grid] = {}
        self.ref_dirty: set[str] = set()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._holes_page(), icon("format-list-bulleted"), tr("Drillholes"))
        self.tabs.addTab(self._colours_page(), icon("palette-outline"), tr("Rock colours"))
        self.tabs.addTab(self._reference_page(), icon("book-open-variant"), tr("Drilling reference"))
        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {theme.MUTED};")
        self.status.setWordWrap(True)
        export = QPushButton(icon("microsoft-excel"), tr("Export to Excel ..."))
        export.setToolTip(tr("Write all holes to an Excel file: sheets collar, geology, survey and assay "
                             "(and the drilling records that have rows)"))
        export.clicked.connect(self.export_excel)
        imp = QPushButton(icon("database-import-outline"), tr("Import from Excel ..."))
        imp.setToolTip(tr("Add or update many holes at once from an Excel workbook (a template to fill in is "
                          "provided)"))
        imp.clicked.connect(self.import_excel)
        csv_in = QPushButton(icon("file-delimited-outline"), tr("Import drilling history ..."))
        csv_in.setToolTip(tr("Read a drilling history - CSV files collar, survey, geology, run, bit, hole_size, "
                             "occurrence, failure, area, formation, rig, measure (the format of the Drilling "
                             "Failure Prediction system) - into the database"))
        csv_in.clicked.connect(self.import_history)
        csv_out = QPushButton(icon("export"), tr("Export drilling history ..."))
        csv_out.setToolTip(tr("Write the drilling records as CSV files (all areas or one), e.g. for the Drilling Failure "
                              "Prediction system or another project"))
        csv_out.clicked.connect(self.export_history)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        for b in (csv_in, csv_out, imp, export, close):
            bottom.addWidget(b)
        v = QVBoxLayout(self)
        v.addWidget(self.tabs, 1)
        v.addLayout(bottom)

    # --- pages ---------------------------------------------------------------------------------
    def _holes_page(self) -> QWidget:
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("Find hole ..."))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.area_filter = QComboBox()
        self.area_filter.setToolTip(tr("Show the holes of one drilling area"))
        self.area_filter.currentIndexChanged.connect(lambda _i: self._filter(self.search.text()))
        self.holes = QListWidget()
        self.holes.currentTextChanged.connect(self._hole_clicked)
        self.count = QLabel("")
        self.count.setStyleSheet(f"color: {theme.MUTED};")
        b_add = QPushButton(icon("plus"), tr("Add hole"))
        b_add.clicked.connect(self.add_hole)
        b_rem = QPushButton(icon("delete-outline"), tr("Remove hole"))
        b_rem.clicked.connect(self.remove_hole)
        row = QHBoxLayout()
        row.addWidget(b_add)
        row.addWidget(b_rem)
        lv.addWidget(self.search)
        lv.addWidget(self.area_filter)
        lv.addWidget(self.holes, 1)
        lv.addWidget(self.count)
        lv.addLayout(row)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel(tr("Select a hole"))
        self.title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        rv.addWidget(self.title)
        self.hole_tabs = QTabWidget()
        self.collar_page = QScrollArea()
        self.collar_page.setWidgetResizable(True)
        inner = QWidget()
        cv = QVBoxLayout(inner)
        g_hole = QGroupBox(tr("Hole"))
        self.collar_form = QFormLayout(g_hole)
        g_drill = QGroupBox(tr("Drilling records"))
        dv = QVBoxLayout(g_drill)
        hint = QLabel(tr("Area, rig, plan and site of the hole - with the runs, bits, hole sizes, occurrences and "
                         "failure in the other tabs they form the drilling history used by the Drilling Failure "
                         "Prediction system (the same database tables)."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        dv.addWidget(hint)
        self.drill_form = QFormLayout()
        dv.addLayout(self.drill_form)
        g_other = QGroupBox(tr("Other fields"))
        self.other_form = QFormLayout(g_other)
        self.other_group = g_other
        for g in (g_hole, g_drill, g_other):
            cv.addWidget(g)
        cv.addStretch(1)
        self.collar_page.setWidget(inner)
        self.hole_tabs.addTab(self.collar_page, tr("Collar"))
        self.g_survey = _Grid(tr("One row per survey station. Dip in degrees from straight down: "
                                 "0 = straight down, 90 = horizontal. Azimuth clockwise from north."))
        self.g_geology = _Grid(tr("One row per interval: From and To down the hole, the rock (pick from the list - "
                                  "rocks are registered in the Rock colours tab) and a note on the core "
                                  "(may be empty)."))
        b_fadd = QPushButton(icon("table-column-plus-after"), tr("Add field"))
        b_fadd.setToolTip(tr("Add an assay field (e.g. copper) to the assay table, for all holes"))
        b_fadd.clicked.connect(self.add_field)
        b_fren = QPushButton(icon("rename-box"), tr("Rename field"))
        b_fren.clicked.connect(self.rename_field)
        b_frem = QPushButton(icon("table-column-remove"), tr("Remove field"))
        b_frem.setToolTip(tr("Remove an assay field and its values from all holes"))
        b_frem.clicked.connect(self.remove_field)
        self.g_assay = _Grid(tr("One row per sample interval: From, To and the grades. Empty = not assayed."),
                             (b_fadd, b_fren, b_frem))
        for g, name in ((self.g_survey, trc("drillhole", "Survey")), (self.g_geology, tr("Geology")),
                        (self.g_assay, tr("Assay"))):
            g.edited.connect(self._mark_dirty)
            self.hole_tabs.addTab(g, name)
        self.drill_grids: dict[str, _Grid] = {}           # the drilling records, one tab each
        for t in DRILLING_TABLES:
            extra = ()
            if t in (ds.OCCURRENCE.name, ds.FAILURE.name):
                b = QPushButton(icon("book-open-variant"), tr("Measures ..."))
                b.setToolTip(tr("Pick the measures of the selected row from the measure library"))
                b.clicked.connect(lambda _c=False, name=t: self.pick_measures(name))
                extra = (b,)
            g = _Grid(tr(DRILLING_HINTS[t]), extra, RUN_COPY if t == ds.RUN.name else ())
            g.edited.connect(self._mark_dirty)
            self.drill_grids[t] = g
            self.hole_tabs.addTab(g, tr(DRILLING_TITLES[t]))
        rv.addWidget(self.hole_tabs, 1)
        self.b_discard = QPushButton(icon("undo"), tr("Discard changes"))
        self.b_discard.clicked.connect(lambda: self._show_hole(self.hid))
        self.b_save = QPushButton(icon("content-save-outline"), tr("Save hole"))
        self.b_save.setToolTip(tr("Write this hole (collar, survey, geology, assay and the drilling records) to the "
                                  "database"))
        self.b_save.clicked.connect(self.save_hole)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.b_discard)
        row.addWidget(self.b_save)
        rv.addLayout(row)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([230, 1010])
        w = QWidget()
        QVBoxLayout(w).addWidget(split)
        return w

    def _colours_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        hint = QLabel(tr("The colour of each rock code (drillhole_geology's rock) in the 3D view, the "
                         "sections and the drillhole log. Double-click a row or use Change colour."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(hint)
        self.t_rocks = QTableWidget(0, 4)
        self.t_rocks.setHorizontalHeaderLabels([tr("Colour"), tr("Rock"), tr("Description"), tr("Intervals")])
        self.t_rocks.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.t_rocks.setSelectionMode(QAbstractItemView.SingleSelection)
        self.t_rocks.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.t_rocks.verticalHeader().setVisible(False)
        self.t_rocks.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.t_rocks.cellDoubleClicked.connect(lambda r, _c: self.change_colour())
        v.addWidget(self.t_rocks, 1)
        b_col = QPushButton(icon("palette-outline"), tr("Change colour ..."))
        b_col.clicked.connect(self.change_colour)
        b_add = QPushButton(icon("plus"), tr("Add rock code ..."))
        b_add.clicked.connect(self.add_rock)
        b_desc = QPushButton(icon("text-box-edit-outline"), tr("Change description ..."))
        b_desc.clicked.connect(self.change_description)
        b_rem = QPushButton(icon("delete-outline"), tr("Remove colour"))
        b_rem.setToolTip(tr("Remove the colour of the selected rock code (an automatic colour is used then)"))
        b_rem.clicked.connect(self.remove_colour)
        row = QHBoxLayout()
        for b in (b_col, b_add, b_desc, b_rem):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        return w

    def _reference_page(self) -> QWidget:
        """Areas, formation cards, rigs and the measure library: the reference tables of the drilling
        records, one sub-tab each, saved table by table."""
        w = QWidget()
        v = QVBoxLayout(w)
        self.ref_tabs = QTabWidget()
        for t in ds.REFERENCE_TABLES:
            save = QPushButton(icon("content-save-outline"), tr("Save table"))
            save.clicked.connect(lambda _c=False, name=t.name: self.save_reference(name))
            revert = QPushButton(icon("undo"), tr("Revert"))
            revert.setToolTip(tr("Show the table as it is in the database again"))
            revert.clicked.connect(lambda _c=False, name=t.name: self.load_reference(name))
            buttons = [revert, save]
            if ds.default_reference().get(t.name):
                dflt = QPushButton(icon("restore"), tr("Restore defaults ..."))
                dflt.setToolTip(tr("Add the predefined rows that are missing, or reset the table to them"))
                dflt.clicked.connect(lambda _c=False, name=t.name: self.restore_reference(name))
                buttons.insert(0, dflt)
            g = _Grid(tr(t.help) if t.help else "", buttons)
            g.edited.connect(lambda name=t.name: self._ref_edited(name))
            self.ref_grids[t.name] = g
            self.ref_tabs.addTab(g, tr(t.title))
        v.addWidget(self.ref_tabs, 1)
        return w

    # --- loading -------------------------------------------------------------------------------
    def start(self) -> bool:
        """Prepare the tables and fill the window; False (with a message) when the database
        cannot be used."""
        try:
            notes = self.store.prepare()
            self.collar_info = self.store.info(COLLAR)
            self.codes = self.store.choices()
            self._fill_holes()
            self._fill_rocks()
            for t in ds.REFERENCE_TABLES:
                self.load_reference(t.name)
        except Exception as e:  # noqa: BLE001 - unreachable server, read-only file, ...
            QMessageBox.warning(self.parent(), tr("Drillhole Manager"),
                                tr("The project database cannot be used:\n{database}\n\n{error}",
                                   database=self.store.target.label, error=e))
            return False
        where = self.store.target.short
        self.status.setText(tr("Database: {where}", where=where) + (f"  ({'; '.join(notes)})" if notes else ""))
        if notes:
            self.changed.emit()
        return True

    def _fill_holes(self, select: str | None = None) -> None:
        ids = self.store.hole_ids()
        self.hole_area = self.store.hole_areas()
        areas = sorted({a for a in self.hole_area.values() if a}, key=ds.natural_key)
        keep = self.area_filter.currentData()
        self.area_filter.blockSignals(True)
        self.area_filter.clear()
        self.area_filter.addItem(tr("All areas"), None)
        for a in areas:
            self.area_filter.addItem(tr("Area {area}", area=a), a)
        self.area_filter.setCurrentIndex(max(self.area_filter.findData(keep), 0) if keep else 0)
        self.area_filter.setVisible(bool(areas))
        self.area_filter.blockSignals(False)
        self.holes.blockSignals(True)
        self.holes.clear()
        self.holes.addItems(ids)
        self.holes.blockSignals(False)
        self._filter(self.search.text())
        target = select if select in ids else (ids[0] if ids and self.hid not in ids else self.hid)
        if target:
            items = self.holes.findItems(target, Qt.MatchExactly)
            if items:
                self.holes.setCurrentItem(items[0])
        if not ids:
            self._show_hole(None)

    def _filter(self, text: str) -> None:
        t = text.strip().lower()
        area = self.area_filter.currentData()
        shown = 0
        for i in range(self.holes.count()):
            it = self.holes.item(i)
            hide = bool(t) and t not in it.text().lower() or area is not None and \
                getattr(self, "hole_area", {}).get(it.text(), "") != area
            it.setHidden(hide)
            shown += not hide
        n = self.holes.count()
        self.count.setText(tr("{n:,} holes", n=n) if shown == n else tr("{shown:,} of {n:,} holes", shown=shown, n=n))

    def _rock_codes(self) -> list[tuple[str, str, str]]:
        styles = self.store.styles()
        return [(c, styles.get(c, "#9aa3ad"), d) for c, d, _n in self.store.rock_codes()]

    def _fill_rocks(self) -> None:
        styles = self.store.styles()
        rows = self.store.rock_codes()
        t = self.t_rocks
        t.setRowCount(len(rows))
        for r, (code, desc, n) in enumerate(rows):
            color = styles.get(code)
            sw = QTableWidgetItem(color.upper() if color else tr("automatic"))
            if color:
                sw.setIcon(_swatch(color, 18))
            t.setItem(r, 0, sw)
            t.setItem(r, 1, QTableWidgetItem(code))
            t.setItem(r, 2, QTableWidgetItem(desc))
            cnt = QTableWidgetItem(f"{n:,}")
            cnt.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(r, 3, cnt)
        t.resizeColumnToContents(0)

    # --- one hole --------------------------------------------------------------------------------
    def _hole_clicked(self, hid: str) -> None:
        if not hid or hid == self.hid:
            return
        if not self._leave_hole():
            items = self.holes.findItems(self.hid or "", Qt.MatchExactly)
            self.holes.blockSignals(True)
            if items:
                self.holes.setCurrentItem(items[0])
            self.holes.blockSignals(False)
            return
        self._show_hole(hid)

    def _leave_hole(self) -> bool:
        """True when the current hole may be left (saved, discarded or not changed)."""
        if not self.dirty:
            return True
        ans = QMessageBox.question(self, tr("Drillhole Manager"), tr("Save the changes to hole {hid}?", hid=self.hid),
                                   QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if ans == QMessageBox.Save:
            return self.save_hole()
        return ans == QMessageBox.Discard

    def _show_hole(self, hid: str | None) -> None:
        self.hid = hid
        for form in (self.collar_form, self.drill_form, self.other_form):
            while form.rowCount():
                form.removeRow(0)
        self.collar_edits = {}
        if hid is None:
            self.title.setText(tr("No holes yet - click Add hole"))
            for g in (self.g_survey, self.g_geology, self.g_assay, *self.drill_grids.values()):
                g.table.setRowCount(0)
            self.other_group.setVisible(False)
            self._set_dirty(False)
            return
        data = self.store.load_hole(hid)
        self.hole_id_edit = QLineEdit(hid)
        self.hole_id_edit.setToolTip(tr("Change to rename the hole in every table"))
        self.hole_id_edit.textEdited.connect(self._mark_dirty)
        self.collar_form.addRow(tr("Hole ID"), self.hole_id_edit)
        role_of = {c: k for k, c in self.collar_info.roles.items()}
        drilling = {c.name for c in ds.COLLAR_ADDED}
        known = {c.name for c in ds.COLLAR.cols}
        fields = self.collar_info.fields
        order = {c.name: i for i, c in enumerate(ds.COLLAR.cols)}       # the schema's order, then the user's own
        fields = sorted(fields, key=lambda c: (order.get(role_of.get(c, c.lower()), 999), fields.index(c)))
        others = 0
        for col in fields:
            kind = col.lower()
            sc = ds.COLLAR.col(role_of.get(col, kind)) if role_of.get(col, kind) in known else None
            e = self._collar_editor(col, sc, data.collar.get(col))
            if sc is not None and sc.tip:
                e.setToolTip(tr(sc.tip))
            label = _header(role_of.get(col, ""), tr(COLLAR_LABELS[kind]) if kind in COLLAR_LABELS
                            else col_title(sc) if sc is not None else col)
            form = self.drill_form if kind in drilling else self.collar_form if sc is not None else self.other_form
            others += form is self.other_form
            form.addRow(label, e)
            self.collar_edits[col] = e
        self.other_group.setVisible(bool(others))
        codes = self._rock_codes()
        geo_choices = {"formation": self.codes.get("formation", {})}
        for g, t in ((self.g_survey, SURVEY), (self.g_geology, GEOLOGY), (self.g_assay, ASSAY)):
            g.set_data(self.store.info(t), data.tables[t], codes, geo_choices)
        titles = [(SURVEY, tr("Survey ({n})", n=len(data.tables[SURVEY]))),
                  (GEOLOGY, tr("Geology ({n})", n=len(data.tables[GEOLOGY]))),
                  (ASSAY, tr("Assay ({n})", n=len(data.tables[ASSAY])))]
        for t, g in self.drill_grids.items():
            info = self.store.info(t)
            df = data.tables.get(t)
            if info is not None:
                g.set_data(info, df if df is not None else pd.DataFrame(columns=info.fields))
            n = len(df) if df is not None else 0
            titles.append((t, f"{tr(DRILLING_TITLES[t])} ({n})" if n else tr(DRILLING_TITLES[t])))
        for i, (_t, text) in enumerate(titles, start=1):
            self.hole_tabs.setTabText(i, text)
        self._set_dirty(False)

    def _collar_editor(self, col: str, sc: ds.Col | None, value) -> QWidget:
        """The editor of one collar field: a date, a list of codes (state; area and rig from the
        reference tables, a new one may be typed), a number or a text."""
        kind = col.lower()
        text = _text(value)
        if kind in ("start_date", "end_date"):
            e = _DateEdit(text)
            e.dateChanged.connect(self._mark_dirty)
        elif kind == "state":
            e = QComboBox()                         # shown as text, kept as its code
            e.addItem("", None)
            for code, name in HOLE_STATES.items():
                e.addItem(tr(name), code)
            code = state_code(value)
            e.setCurrentIndex(max(e.findData(code), 0) if code is not None else 0)
            e.currentIndexChanged.connect(self._mark_dirty)
        elif sc is not None and sc.kind == "code" and kind in ("area_id", "rig_model"):
            e = QComboBox()
            e.setEditable(True)
            e.addItem("")
            e.addItems(list(self.codes.get(kind, {})))
            e.setCurrentText(text)
            e.currentTextChanged.connect(self._mark_dirty)
        else:
            e = QLineEdit(text)
            if self.collar_info.numeric.get(col):
                e.setValidator(QDoubleValidator(e))
            e.textEdited.connect(self._mark_dirty)
        return e

    def _mark_dirty(self, *_a) -> None:
        self._set_dirty(True)

    def _set_dirty(self, on: bool) -> None:
        self.dirty = on and self.hid is not None
        self.b_save.setEnabled(self.dirty)
        self.b_discard.setEnabled(self.dirty)
        if self.hid is not None:
            self.title.setText(tr("Hole {hid}   • not saved", hid=self.hid) if self.dirty
                               else tr("Hole {hid}", hid=self.hid))

    def save_hole(self) -> bool:
        if self.hid is None:
            return True
        collar = {c: _widget_value(e) for c, e in self.collar_edits.items()}
        grids = {SURVEY: self.g_survey, GEOLOGY: self.g_geology, ASSAY: self.g_assay}
        grids.update({t: g for t, g in self.drill_grids.items() if g.info is not None})
        tables = {t: (g.info, g.frame()) for t, g in grids.items()}
        low = {c.lower(): c for c in collar}
        rig = self.store.rig(collar.get(low.get("rig_model", ""), "")) if "rig_model" in low else None
        context = {"max_rpm": rig.get("max_rpm")} if rig else {}
        errors, warnings = check_hole(self.collar_info, collar, tables, context=context)
        if errors:
            more = "\n" + tr("... and {n} more", n=len(errors) - 15) if len(errors) > 15 else ""
            QMessageBox.warning(self, tr("Save hole"), tr("Please correct these first:") + "\n\n"
                                + "\n".join(errors[:15]) + more)
            return False
        if warnings:
            more = "\n" + tr("... and {n} more", n=len(warnings) - 15) if len(warnings) > 15 else ""
            ans = QMessageBox.question(self, tr("Save hole"), tr("Please check:") + "\n\n" + "\n".join(warnings[:15])
                                       + more + "\n\n" + tr("Save anyway?"))
            if ans != QMessageBox.Yes:
                return False
        new_id = self.hole_id_edit.text().strip()
        try:
            saved = self.store.save_hole(self.hid, HoleData(collar, {t: df for t, (_i, df) in tables.items()}), new_id)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Save hole"), tr("The hole was not saved:\n{error}", error=e))
            return False
        self.dirty = False
        renamed = saved != self.hid
        self.hid = None if renamed else self.hid
        if renamed:
            self._fill_holes(select=saved)
        else:
            self._show_hole(saved)
        self._fill_rocks()
        self.codes = self.store.choices()               # a new area / rig typed in
        self.status.setText(tr("Saved hole {hid}", hid=saved))
        self.changed.emit()
        return True

    def add_hole(self) -> None:
        if not self._leave_hole():
            return
        dlg = _NewHoleDialog(self, self.collar_info, list(self.codes.get("area_id", {})),
                             self.area_filter.currentData() or "")
        while dlg.exec():
            hid, values = dlg.values()
            try:
                self.store.add_hole(hid, values)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, tr("Add drillhole"), str(e))
                continue
            self.dirty = False
            self._fill_holes(select=hid)
            self.status.setText(tr("Added hole {hid}", hid=hid))
            self.changed.emit()
            return

    def remove_hole(self) -> None:
        if self.hid is None:
            return
        ans = QMessageBox.question(self, tr("Remove drillhole"),
                                   tr("Remove hole {hid} with all its rows (survey, geology, assay and drilling "
                                      "records)?\n\nThis cannot be undone.", hid=self.hid))
        if ans != QMessageBox.Yes:
            return
        hid = self.hid
        n = self.store.remove_hole(hid)
        self.dirty = False
        self.hid = None
        self._fill_holes()
        self._fill_rocks()
        self.status.setText(tr("Removed hole {hid} ({n} rows)", hid=hid, n=n))
        self.changed.emit()

    # --- assay fields ------------------------------------------------------------------------
    def _field_change(self, fn, *args, done: str) -> None:
        if not self._leave_hole():
            return
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Assay fields"), str(e))
            return
        self._show_hole(self.hid)
        self.status.setText(done)
        self.changed.emit()

    def add_field(self) -> None:
        name, ok = QInputDialog.getText(self, tr("Add assay field"), tr("Name of the new field (e.g. copper, ag_ppm):"))
        if ok and name.strip():
            self._field_change(self.store.add_field, name.strip(),
                               done=tr("Added assay field {name}", name=name.strip()))

    def _pick_field(self, title: str) -> str | None:
        fields = self.store.assay_fields()
        if not fields:
            QMessageBox.information(self, title, tr("The assay table has no fields yet."))
            return None
        name, ok = QInputDialog.getItem(self, title, tr("Field:"), fields, 0, False)
        return name if ok else None

    def rename_field(self) -> None:
        old = self._pick_field(tr("Rename assay field"))
        if old:
            new, ok = QInputDialog.getText(self, tr("Rename assay field"), tr("New name for {name}:", name=old),
                                           text=old)
            if ok and new.strip() and new.strip() != old:
                self._field_change(self.store.rename_field, old, new.strip(),
                                   done=tr("Renamed {old} to {new}", old=old, new=new.strip()))

    def remove_field(self) -> None:
        name = self._pick_field(tr("Remove assay field"))
        if name and QMessageBox.question(self, tr("Remove assay field"),
                                         tr("Remove the field {name} and its values from ALL holes?\n\n"
                                            "This cannot be undone.", name=name)) == QMessageBox.Yes:
            self._field_change(self.store.remove_field, name, done=tr("Removed assay field {name}", name=name))

    # --- rock colours ------------------------------------------------------------------------
    def _selected_rock(self) -> str | None:
        r = self.t_rocks.currentRow()
        return self.t_rocks.item(r, 1).text() if r >= 0 and self.t_rocks.item(r, 1) else None

    def change_colour(self) -> None:
        code = self._selected_rock()
        if code is None:
            return
        cur = self.store.styles().get(code, "#9aa3ad")
        c = QColorDialog.getColor(QColor(cur), self, tr("Colour of {code}", code=code))
        if c.isValid():
            self.store.set_style(code, c.name())
            self._rocks_changed(tr("Colour of {code} set to {color}", code=code, color=c.name().upper()))

    def add_rock(self) -> None:
        code, ok = QInputDialog.getText(self, tr("Add rock code"),
                                        tr("Rock code (as used in the geology rows, e.g. GR):"))
        code = code.strip()
        if not ok or not code:
            return
        desc, ok = QInputDialog.getText(self, tr("Add rock code"),
                                        tr("Description of {code} (e.g. Granite):", code=code))
        if not ok:
            return
        used = set(self.store.styles().values())
        color = next((p for p in PALETTE if p not in used), PALETTE[0])
        c = QColorDialog.getColor(QColor(color), self, tr("Colour of {code}", code=code))
        if c.isValid():
            self.store.set_style(code, c.name())
            self.store.set_description(code, desc)
            self._rocks_changed(tr("Added rock code {code}", code=code))

    def change_description(self) -> None:
        code = self._selected_rock()
        if code is None:
            return
        r = self.t_rocks.currentRow()
        cur = self.t_rocks.item(r, 2).text() if self.t_rocks.item(r, 2) else ""
        desc, ok = QInputDialog.getText(self, tr("Rock description"), tr("Description of {code}:", code=code),
                                        text=cur)
        if ok:
            self.store.set_description(code, desc)
            self._rocks_changed(tr("Description of {code} set", code=code))

    def export_excel(self) -> None:
        """All holes to an Excel workbook (sheets collar, geology, survey, assay)."""
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog

        if self.dirty and QMessageBox.question(
                self, tr("Export to Excel"), tr("Hole {hid} has changes that are not saved. Save them first?\n\n"
                                                "(Unsaved changes are not in the export.)", hid=self.hid)
        ) == QMessageBox.Yes and not self.save_hole():
            return
        start = str(Path(self._folder()) / "drillholes.xlsx")
        path, _ = QFileDialog.getSaveFileName(self, tr("Export drillholes to Excel"), start,
                                              tr("Excel workbook (*.xlsx)"))
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            counts = self.store.export_excel(path)
        except PermissionError:
            QMessageBox.warning(self, tr("Export to Excel"),
                                tr("{path}\n\nThe file cannot be written - is it open in Excel? Close it and "
                                   "try again.", path=path))
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Export to Excel"), tr("The export failed:\n{error}", error=e))
            return
        self.status.setText(tr("Exported to {file}: {counts}", file=Path(path).name,
                               counts=", ".join(f"{k} {v:,}" for k, v in counts.items())))

    def _folder(self) -> str:
        """Where the Excel file dialogs start: the project folder (home for MySQL)."""
        from pathlib import Path

        t = self.store.target
        return str(t.path.parent if not t.is_mysql else Path.home())

    def import_excel(self) -> None:
        """Holes from an Excel workbook (the template filled in, or an export edited)."""
        from ..io import drillhole_import as di
        from .excel_import import ExcelImportDialog

        if not self._leave_hole():
            return
        dlg = ExcelImportDialog(
            self, title=tr("Import drillholes from Excel"),
            intro=tr("Add many holes at once: save the template, fill in its sheets collar, survey, geology and "
                     "assay in Excel, then choose the file here (a workbook from Export to Excel works too). "
                     "A hole already in the database gets the file's collar values, and its survey, geology "
                     "or assay rows are replaced in each sheet that has rows for it - or it is skipped. New "
                     "assay columns become assay fields."),
            existing=tr("Holes already in the database"), folder=self._folder(),
            template_name="drillholes_template.xlsx",
            save_template=lambda path: di.save_template(self.store, path),
            read=lambda path: di.read_import(self.store, path),
            apply=lambda plan, replace: di.apply_import(self.store, plan, replace),
            result_text=di.result_text)
        if dlg.exec() != QDialog.Accepted:
            return
        self.dirty = False
        self.hid = None                                 # show the (maybe replaced) hole again
        self.collar_info = self.store.info(COLLAR)
        self._fill_holes()
        self._fill_rocks()
        self.status.setText(dlg.result_message())
        self.changed.emit()

    def remove_colour(self) -> None:
        code = self._selected_rock()
        if code is not None and code in self.store.styles():
            self.store.remove_style(code)
            self._rocks_changed(tr("Removed the colour of {code}", code=code))

    def _rocks_changed(self, text: str) -> None:
        keep = self._selected_rock()
        self._fill_rocks()
        for r in range(self.t_rocks.rowCount()):
            if self.t_rocks.item(r, 1).text() == keep:
                self.t_rocks.selectRow(r)
        self.g_geology.rock_codes = self._rock_codes()
        if self.hid is not None and not self.dirty:
            self._show_hole(self.hid)                  # rock labels / swatches with the new colours
        self.status.setText(text)
        self.changed.emit()

    # --- drilling reference (areas, formation cards, rigs, measures) -------------------------
    def _ref_choices(self) -> dict:
        return {"main_rock": self.codes.get("main_rock", {}), "second_rock": self.codes.get("second_rock", {})}

    def load_reference(self, name: str) -> None:
        g = self.ref_grids[name]
        info = self.store.info(name)
        if info is None:
            return
        g.set_data(info, self.store.reference(name), choices=self._ref_choices())
        self.ref_dirty.discard(name)
        self._ref_title(name)

    def _ref_edited(self, name: str) -> None:
        self.ref_dirty.add(name)
        self._ref_title(name)

    def _ref_title(self, name: str) -> None:
        t = ds.table(name)
        i = list(self.ref_grids).index(name)
        n = self.ref_grids[name].table.rowCount()
        self.ref_tabs.setTabText(i, f"{tr(t.title)} ({n})" + (" •" if name in self.ref_dirty else ""))

    def save_reference(self, name: str) -> bool:
        t = ds.table(name)
        df = self.ref_grids[name].frame()
        _bad, issues = ds.validate(t, df, choices=self._ref_choices())
        key = t.key[0]
        keys = [str(k) for k in df[key].dropna()] if key in df else []
        dup = sorted({k for k in keys if keys.count(k) > 1})
        errors = [i.text() for i in issues if i.level == "error"]
        if dup:
            errors.append(tr("{field} is in the table twice: {values}", field=tr(t.col(key).label), values=", ".join(dup)))
        if errors:
            more = "\n" + tr("... and {n} more", n=len(errors) - 15) if len(errors) > 15 else ""
            QMessageBox.warning(self, tr("Save table"), tr("Please correct these first:") + "\n\n"
                                + "\n".join(errors[:15]) + more)
            return False
        try:
            self.store.save_reference(name, df)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Save table"), tr("The table was not saved:\n{error}", error=e))
            return False
        self.codes = self.store.choices()
        self.load_reference(name)
        self.status.setText(tr("Saved {table}: {n} rows", table=tr(t.title), n=len(df)))
        self.changed.emit()
        return True

    def restore_reference(self, name: str) -> None:
        t = ds.table(name)
        box = QMessageBox(self)
        box.setWindowTitle(tr("Restore defaults"))
        box.setText(tr("{table}: add the predefined rows that are missing, or replace the whole table with "
                       "them? (Save table writes the result.)", table=tr(t.title)))
        add = box.addButton(tr("Add missing"), QMessageBox.AcceptRole)
        reset = box.addButton(tr("Reset to defaults"), QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() not in (add, reset):
            return
        dflt = self.store.default_reference(name)
        if box.clickedButton() is add:
            cur = self.ref_grids[name].frame()
            have = set(cur[t.key[0]].astype(str)) if len(cur) else set()
            dflt = pd.concat([cur, dflt[~dflt[t.key[0]].astype(str).isin(have)]], ignore_index=True)
        self.ref_grids[name].set_data(self.store.info(name), dflt, choices=self._ref_choices())
        self._ref_edited(name)

    # --- measures of an occurrence / failure ----------------------------------------------
    OCC_FAILS = {"cavity": ("CV", "LC"), "fault": ("CL", "SP", "LC"), "broken": ("CL", "CB", "SP"),
                 "clay": ("CL", "SP"), "water": ("WI",), "loss": ("LC",)}

    def pick_measures(self, name: str) -> None:
        g = self.drill_grids[name]
        row = g.table.currentRow()
        if row < 0:
            QMessageBox.information(self, tr("Measures"), tr("Select a row first (add one with Add row)."))
            return
        if name == ds.FAILURE.name:
            fails = (g.value(row, "fail_code"),)
        else:
            fails = self.OCC_FAILS.get(g.value(row, "type"), ())
        dlg = MeasurePicker(self, self.store.measures(), g.value(row, "measures"), fails)
        if dlg.exec():
            g.set_cell(row, "measures", dlg.value())
            self._mark_dirty()

    # --- the drilling history as CSV files -----------------------------------------------------
    def import_history(self) -> None:
        if not self._leave_hole():
            return
        folder = QFileDialog.getExistingDirectory(self, tr("Folder with the drilling history CSV files"), self._folder())
        if not folder:
            return
        names = {f"{t.short}.csv" for t in ds.SHARED_TABLES}
        files = sorted(p.name for p in Path(folder).glob("*.csv") if p.name.lower() in names)
        if not files:
            QMessageBox.information(self, tr("Import drilling history"),
                                    tr("No drilling history files in this folder (collar.csv, survey.csv, geology.csv, "
                                       "run.csv, bit.csv, hole_size.csv, occurrence.csv, failure.csv, area.csv, "
                                       "formation.csv, rig.csv, measure.csv)."))
            return
        box = QMessageBox(self)
        box.setWindowTitle(tr("Import drilling history"))
        box.setText(tr("Import {n} files: {files}\n\nMerge: a hole in the files gets their collar values and, in each "
                       "file with rows for it, its rows replaced; other holes are kept.\nReplace areas: as Merge, and "
                       "the holes of the areas in collar.csv that are in none of the files are removed.",
                       n=len(files), files=", ".join(files)))
        merge = box.addButton(tr("Merge"), QMessageBox.AcceptRole)
        replace = box.addButton(tr("Replace areas"), QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() not in (merge, replace):
            return
        from PySide6.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            counts, issues = self.store.import_drilling_csv(folder, replace_areas=box.clickedButton() is replace)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, tr("Import drilling history"), tr("The import failed:\n{error}", error=e))
            return
        QApplication.restoreOverrideCursor()
        errs = [i for i in issues if i.level == "error"]
        text = tr("Imported: {counts}", counts=", ".join(f"{k} {v:,}" for k, v in counts.items()))
        if errs:
            text += "\n\n" + tr("{n} rows skipped because of errors, e.g.:", n=len(errs)) + "\n" + \
                "\n".join(i.text() for i in errs[:12])
        QMessageBox.information(self, tr("Import drilling history"), text)
        self.dirty = False
        self.hid = None
        self.collar_info = self.store.info(COLLAR)
        self.codes = self.store.choices()
        self._fill_holes()
        self._fill_rocks()
        for t in ds.REFERENCE_TABLES:
            self.load_reference(t.name)
        self.status.setText(text.split("\n")[0])
        self.changed.emit()

    def export_history(self) -> None:
        if self.dirty and QMessageBox.question(
                self, tr("Export drilling history"), tr("Hole {hid} has changes that are not saved. Save them first?",
                                                        hid=self.hid)) == QMessageBox.Yes and not self.save_hole():
            return
        areas = list(self.codes.get("area_id", {}))
        area = None
        if areas:
            items = [tr("All areas")] + areas
            pick, ok = QInputDialog.getItem(self, tr("Export drilling history"), tr("Holes of:"), items, 0, False)
            if not ok:
                return
            area = None if pick == items[0] else pick
        folder = QFileDialog.getExistingDirectory(self, tr("Export the drilling history to folder"), self._folder())
        if not folder:
            return
        try:
            counts = self.store.export_drilling_csv(folder, [area] if area else None)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Export drilling history"), tr("The export failed:\n{error}", error=e))
            return
        self.status.setText(tr("Exported to {file}: {counts}", file=folder,
                               counts=", ".join(f"{k} {v:,}" for k, v in counts.items())))

    def closeEvent(self, e) -> None:
        if not self._leave_hole():
            e.ignore()
            return
        if self.ref_dirty:
            names = ", ".join(tr(ds.table(n).title) for n in sorted(self.ref_dirty))
            ans = QMessageBox.question(self, tr("Drillhole Manager"),
                                       tr("{tables}: the changes are not saved. Close anyway?", tables=names))
            if ans != QMessageBox.Yes:
                e.ignore()
                return
        super().closeEvent(e)
