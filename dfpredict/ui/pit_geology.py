"""Pit Geology: the geologist's mapping of a pit (an underground drive), typed in as it is
recorded in the drive, with a live wall sketch.

* Every position is a chainage in metres from the pit's start: 0 m is the entrance of a main pit
  and the junction station of a branch. It can be typed as metres (23.5) or as a station and
  the metres taped past it (P1-03+4.5) - the station's chainage comes from the survey.
* Left / right wall are as seen looking toward the face.
* Rocks: From - To, where (both walls and the back, or one of them), the rock (the project's
  rock codes and colours, see Drillhole Manager > Rock colours), description, alteration,
  mineralization.
* Structures: chainage, wall, height above the floor, type, dip direction and dip (135∠45),
  thickness, filling, description.
* Samples: number, From - To, wall, height, type, channel width x depth, weight, the assays.

Nothing is written until Save; Save writes the whole pit in one transaction after the checks
(io/pit_geology_store.check_geology). The drives in the view show the saved geology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QDate, QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QImage, QPageLayout, QPageSize, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..i18n import N_, tr
from ..io.pit_geology_store import (INTERVAL_WALLS, SAMPLE_KINDS, SAMPLE_WALLS, STRUCTURE_KINDS, WALLS, GeoInterval,
                                    GeoSample, GeoStructure, PitGeology, PitGeologyStore, check_geology,
                                    next_sample_name, sorted_geology)
from ..io.survey_store import SurveyStore, format_dms, parse_angle
from ..scene.pit_geology import parse_chainage, pit_axis, station_offset_text
from ..scene.survey import SurveyData, pit_paths
from . import theme
from .drillhole_log import PAPER, LogStyle
from .drillhole_manager import _swatch, rock_code, rock_label
from .icons import icon
from .pit_sketch import SketchData, WallSketch, paint_sketch

PRINT_DPI = 300


@dataclass
class Col:
    key: str
    title: str                  # N_() text, tr() where shown
    kind: str                   # chain / num / angle / choice / rock / text / date
    choices: dict | None = None
    tip: str = ""               # N_() text


def interval_columns() -> list:
    return [Col("frm", N_("From (m)"), "chain", tip=N_("Chainage where the rock starts: metres from the pit's start, "
                                                       "or a station + metres (P1-03+4.5)")),
            Col("to", N_("To (m)"), "chain", tip=N_("Chainage where the rock ends")),
            Col("wall", N_("Where"), "choice", INTERVAL_WALLS,
                N_("On both walls and the back, or only on one of them (left / right looking toward the face)")),
            Col("rock", N_("Rock"), "rock", tip=N_("The project's rock codes (Drillhole Manager > Rock colours)")),
            Col("description", N_("Description"), "text",
                N_("Colour, grain size, texture, structure, minerals ...")),
            Col("alteration", N_("Alteration"), "text", tip=N_("e.g. silicification, sericite, chlorite")),
            Col("mineralization", N_("Mineralization"), "text",
                tip=N_("e.g. pyrite 2 %, visible gold, quartz veinlets")),
            Col("mapped_on", N_("Date"), "date"),
            Col("geologist", N_("Geologist"), "text")]


def structure_columns() -> list:
    return [Col("at", N_("Chainage (m)"), "chain", tip=N_("Where it was measured: metres from the pit's start, "
                                                           "or a station + metres (P1-03+4.5)")),
            Col("wall", N_("Wall"), "choice", WALLS, N_("Left / right as seen looking toward the face")),
            Col("height", N_("Height (m)"), "num", tip=N_("Height above the floor where it was measured "
                                                          "(empty: half way up the wall)")),
            Col("kind", N_("Type"), "choice", STRUCTURE_KINDS),
            Col("dip_direction", N_("Dip direction (°)"), "angle",
                tip=N_("Direction the plane dips toward, 0-360° clockwise from north (the 135 of 135∠45)")),
            Col("dip", N_("Dip (°)"), "angle", tip=N_("Angle of the plane below the horizontal, 0-90° (the 45 of "
                                                      "135∠45)")),
            Col("thickness", N_("Thickness (m)"), "num", tip=N_("Width of a fault zone, vein, ... (empty: a plane)")),
            Col("fill", N_("Filling"), "text", tip=N_("e.g. clay gouge, quartz, calcite")),
            Col("description", N_("Description"), "text"),
            Col("mapped_on", N_("Date"), "date"),
            Col("geologist", N_("Geologist"), "text")]


def sample_columns(assay_fields=()) -> list:
    return ([Col("name", N_("Sample"), "text", tip=N_("The sample number (unique in the project)")),
             Col("frm", N_("From (m)"), "chain", tip=N_("Chainage where the sample starts")),
             Col("to", N_("To (m)"), "chain", tip=N_("Chainage where the sample ends")),
             Col("wall", N_("Wall"), "choice", SAMPLE_WALLS),
             Col("height", N_("Height (m)"), "num", tip=N_("Height of the channel above the floor (empty: 1 m)")),
             Col("kind", N_("Type"), "choice", SAMPLE_KINDS),
             Col("width_cm", N_("Width (cm)"), "num", tip=N_("Channel width")),
             Col("depth_cm", N_("Depth (cm)"), "num", tip=N_("Channel depth")),
             Col("weight_kg", N_("Weight (kg)"), "num")]
            + [Col("a:" + f, f, "num", tip=N_("Assay value")) for f in assay_fields]
            + [Col("taken_on", N_("Date"), "date"), Col("sampler", N_("Sampler"), "text"),
               Col("note", N_("Note"), "text")])


def _shown_title(col: Col) -> str:
    return col.title if col.key.startswith("a:") else tr(col.title)


def _fmt_num(v, decimals=None) -> str:
    if v is None:
        return ""
    if decimals is not None:
        return f"{v:.{decimals}f}"
    return f"{v:g}"


# --- the tables --------------------------------------------------------------------------------------
class _Delegate(QStyledItemDelegate):
    def __init__(self, grid: "GeoGrid"):
        super().__init__(grid.table)
        self.grid = grid

    def createEditor(self, parent, option, index):
        col = self.grid.cols[index.column()]
        if col.kind == "choice":
            box = QComboBox(parent)
            for k, v in col.choices.items():
                box.addItem(tr(v), k)
            return box
        if col.kind == "rock":
            box = QComboBox(parent)                   # the registered rocks; a new code can be typed
            box.setEditable(True)
            box.setInsertPolicy(QComboBox.NoInsert)
            for code, color, desc in self.grid.owner.rock_codes:
                box.addItem(_swatch(color), rock_label(code, desc), code)
            return box
        ed = QLineEdit(parent)
        if col.kind == "num":
            v = QDoubleValidator(ed)
            v.setNotation(QDoubleValidator.StandardNotation)
            ed.setValidator(v)
        elif col.kind == "date":
            ed.setPlaceholderText("YYYY-MM-DD")
        elif col.kind == "chain":
            ed.setPlaceholderText(tr("m, or P1-03+4.5"))
        return ed

    def setEditorData(self, editor, index):
        value = index.data(Qt.UserRole)
        if isinstance(editor, QComboBox):
            i = editor.findData(value)
            if i < 0 and value:                       # a rock code that is not registered: kept
                editor.insertItem(0, tr("{code} (not registered)", code=value), value)
                i = 0
            editor.setCurrentIndex(max(i, 0))
            editor.showPopup()
        else:
            editor.setText(index.data() or "")
            editor.selectAll()

    def setModelData(self, editor, model, index):
        col = self.grid.cols[index.column()]
        if isinstance(editor, QComboBox):
            value = editor.currentData()
            if editor.isEditable():                   # a rock: the item chosen, or the code typed
                text = editor.currentText().strip()
                i = editor.findText(text)
                value = editor.itemData(i) if i >= 0 else (rock_code(text) or None)
            self.grid.set_cell(index.row(), index.column(), value)
            return
        text = editor.text().strip()
        try:
            value = self.grid.parse(col, text)
        except ValueError as e:
            self.grid.owner.flash(str(e))
            return
        self.grid.set_cell(index.row(), index.column(), value)


class GeoGrid(QWidget):
    """One table of the pit's geology (rocks, structures or samples)."""

    edited = Signal()
    row_selected = Signal(int)

    def __init__(self, owner: "PitGeologyWindow", kind: str, cols: list, hint: str, extra_buttons=()):
        super().__init__()
        self.owner = owner
        self.kind = kind                # interval / structure / sample
        self.cols = cols
        self._filling = False
        self.table = QTableWidget(0, len(cols))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.AnyKeyPressed | QAbstractItemView.SelectedClicked)
        self.table.setItemDelegate(_Delegate(self))
        self.table.itemChanged.connect(self._changed)
        self.table.currentCellChanged.connect(lambda r, *_: self.row_selected.emit(r))
        self._headers()
        add = QPushButton(icon("table-row-plus-after"), tr("Add row"))
        add.setToolTip(tr("Add a row at the end, carrying on from the last one"))
        add.clicked.connect(self.add_row)
        ins = QPushButton(icon("table-row-plus-before"), tr("Insert row"))
        ins.setToolTip(tr("Insert a row above the selected one"))
        ins.clicked.connect(self.insert_row)
        rem = QPushButton(icon("table-row-remove"), tr("Delete rows"))
        rem.setToolTip(tr("Delete the selected rows"))
        rem.clicked.connect(self.delete_rows)
        row = QHBoxLayout()
        for b in (add, ins, rem):
            row.addWidget(b)
        row.addStretch(1)
        for b in extra_buttons:
            row.addWidget(b)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        h = QLabel(hint)
        h.setWordWrap(True)
        h.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(h)
        v.addWidget(self.table, 1)
        v.addLayout(row)

    def _headers(self) -> None:
        self.table.setColumnCount(len(self.cols))
        for j, c in enumerate(self.cols):
            it = QTableWidgetItem(_shown_title(c))
            if c.tip:
                it.setToolTip(tr(c.tip))
            self.table.setHorizontalHeaderItem(j, it)
        widths = {"chain": 80, "num": 74, "angle": 92, "choice": 130, "rock": 120, "text": 150, "date": 88}
        for j, c in enumerate(self.cols):
            self.table.setColumnWidth(j, widths.get(c.kind, 100))

    def set_columns(self, cols: list) -> None:
        """New columns (the samples' assay fields changed); the rows typed are kept."""
        rows = self.values()
        self.cols = cols
        self._headers()
        self.set_values(rows)

    # --- values --------------------------------------------------------------------------------
    def parse(self, col: Col, text: str):
        if not text:
            return None if col.kind != "text" else ""
        if col.kind == "chain":
            return parse_chainage(text, self.owner.stations_m)
        if col.kind == "angle":
            v = parse_angle(text)
            return v
        if col.kind == "num":
            try:
                return float(text.replace(",", "."))
            except ValueError:
                raise ValueError(tr("not a number: {t!r}", t=text)) from None
        if col.kind == "date":
            d = QDate.fromString(text, "yyyy-MM-dd")
            if not d.isValid():
                raise ValueError(tr("the date {v!r} is not YYYY-MM-DD", v=text))
            return text
        return text

    def display(self, col: Col, value) -> str:
        if value is None:
            return ""
        if col.kind == "chain":
            return _fmt_num(value, 2)
        if col.kind == "angle":                       # 135, 135.5; 135°30′15″ when typed so
            return f"{value:g}" if abs(value - round(value, 2)) < 1e-9 else format_dms(value)
        if col.kind == "num":
            return _fmt_num(value)
        if col.kind == "choice":
            return tr(col.choices.get(value, value)) if value else ""
        if col.kind == "rock":
            desc = self.owner.rock_desc.get(value, "")
            return rock_label(value, desc) if value else ""
        return str(value)

    def set_cell(self, r: int, j: int, value) -> None:
        col = self.cols[j]
        it = self.table.item(r, j)
        if it is None:
            it = QTableWidgetItem()
            self.table.setItem(r, j, it)
        was = self._filling
        self._filling = True
        it.setData(Qt.UserRole, value)
        it.setText(self.display(col, value))
        if col.kind == "rock" and value:
            color = self.owner.rock_colors.get(value)
            it.setIcon(_swatch(color) if color else _swatch("#8a9096"))
        if col.kind == "chain" and value is not None:
            off = station_offset_text(value, self.owner.stations_m)
            it.setToolTip(off)
        self._filling = was
        if not self._filling:
            self.edited.emit()

    def values(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            row = {}
            for j, c in enumerate(self.cols):
                it = self.table.item(r, j)
                v = it.data(Qt.UserRole) if it is not None else None
                row[c.key] = v if v is not None or c.kind != "text" else ""
            out.append(row)
        return out

    def set_values(self, rows: list[dict]) -> None:
        self._filling = True
        self.table.setRowCount(0)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for j, c in enumerate(self.cols):
                self.set_cell(r, j, row.get(c.key))
        self._filling = False

    def _changed(self, _item) -> None:
        if not self._filling:
            self.edited.emit()

    # --- rows ------------------------------------------------------------------------------------
    def _new_row(self, before: dict | None) -> dict:
        o = self.owner
        row = {c.key: None for c in self.cols}
        row.update({"mapped_on": o.default_date(), "geologist": o.default_geologist(),
                    "taken_on": o.default_date(), "sampler": o.default_geologist()})
        if self.kind == "interval":
            row.update(wall="all", frm=(before or {}).get("to") or (0.0 if before is None else None))
            if before:
                row["wall"] = before.get("wall") or "all"
        elif self.kind == "structure":
            row.update(wall="left", kind="joint")
            if before:
                row.update(at=before.get("at"), wall=before.get("wall") or "left", kind=before.get("kind") or "joint")
        else:
            row.update(wall="left", kind="channel", frm=0.0)
            if before:
                b = before
                row.update(name=next_sample_name(b.get("name") or ""), frm=b.get("to"), wall=b.get("wall") or "left",
                           height=b.get("height"), kind=b.get("kind") or "channel", width_cm=b.get("width_cm"),
                           depth_cm=b.get("depth_cm"))
                if b.get("frm") is not None and b.get("to") is not None:
                    row["to"] = b["to"] + (b["to"] - b["frm"])           # the same length as the last one
            else:
                row["to"] = 1.0
        return row

    def add_row(self) -> None:
        rows = self.values()
        rows.append(self._new_row(rows[-1] if rows else None))
        self.set_values(rows)
        self.table.setCurrentCell(len(rows) - 1, 0)
        self.edited.emit()

    def insert_row(self) -> None:
        rows = self.values()
        r = self.table.currentRow()
        if r < 0:
            self.add_row()
            return
        rows.insert(r, self._new_row(rows[r - 1] if r > 0 else None))
        self.set_values(rows)
        self.table.setCurrentCell(r, 0)
        self.edited.emit()

    def delete_rows(self) -> None:
        sel = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not sel:
            return
        rows = self.values()
        for r in sel:
            del rows[r]
        self.set_values(rows)
        self.edited.emit()


# --- records <-> rows ---------------------------------------------------------------------------------
def _rows_of(records, kind) -> list[dict]:
    out = []
    for r in records:
        d = dict(r.__dict__)
        if kind == "sample":
            for k, v in r.values.items():
                d["a:" + k] = v
        out.append(d)
    return out


def _records_of(rows, kind) -> list:
    out = []
    for d in rows:
        if kind == "interval":
            out.append(GeoInterval(d.get("frm"), d.get("to"), d.get("wall") or "all", d.get("rock") or "",
                                   d.get("description") or "", d.get("alteration") or "",
                                   d.get("mineralization") or "", d.get("mapped_on") or "", d.get("geologist") or ""))
        elif kind == "structure":
            out.append(GeoStructure(d.get("at"), d.get("wall") or "left", d.get("height"), d.get("kind") or "other",
                                    d.get("dip_direction"), d.get("dip"), d.get("thickness"), d.get("fill") or "",
                                    d.get("description") or "", d.get("mapped_on") or "", d.get("geologist") or ""))
        else:
            vals = {k[2:]: v for k, v in d.items() if k.startswith("a:")}
            out.append(GeoSample(d.get("name") or "", d.get("frm"), d.get("to"), d.get("wall") or "left",
                                 d.get("height"), d.get("kind") or "channel", d.get("width_cm"), d.get("depth_cm"),
                                 d.get("weight_kg"), vals, d.get("taken_on") or "", d.get("sampler") or "",
                                 d.get("note") or ""))
    return out


# --- the window ---------------------------------------------------------------------------------------
class PitGeologyWindow(QDialog):
    changed = Signal()                  # the database was written: read the survey layer again

    def __init__(self, target, parent=None, workspace_folder=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Pit Geology"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
        self.resize(1360, 860)
        self.target = target
        self.workspace_folder = workspace_folder
        self.survey = SurveyStore(target)
        self.store = PitGeologyStore(target)
        self.data: SurveyData | None = None
        self.geology: dict = {}
        self.assay_fields: list = []
        self.rock_codes: list = []      # (code, colour, description)
        self.rock_colors: dict = {}
        self.rock_desc: dict = {}
        self.pid = None
        self.axis = None
        self.stations_m: dict = {}
        self.dirty = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh_sketch)

        # left: the pits
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._pit_clicked)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel(tr("Pits")))
        lv.addWidget(self.tree, 1)
        hint = QLabel(tr("Pits are added in the Pit Manager (Ctrl+U)."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        lv.addWidget(hint)
        self.b_import = QPushButton(icon("file-excel-outline"), tr("Import from Excel ..."))
        self.b_import.setToolTip(tr("Rocks, structures and samples of many pits at once from an Excel workbook (the "
                                    "template filled in, or an export edited)"))
        self.b_import.clicked.connect(self.import_excel)
        self.b_export = QPushButton(icon("microsoft-excel"), tr("Export to Excel ..."))
        self.b_export.setToolTip(tr("Every pit's rocks, structures and samples to an Excel workbook (sheets rocks, "
                                    "structures, samples and the stations)"))
        self.b_export.clicked.connect(self.export_excel)
        lv.addWidget(self.b_import)
        lv.addWidget(self.b_export)

        # top of the middle: the pit, where 0 m is, the stations, the defaults for new rows
        self.title = QLabel(tr("Select a pit"))
        self.title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        self.origin = QLabel("")
        self.origin.setWordWrap(True)
        self.origin.setStyleSheet(f"color: {theme.MUTED};")
        self.station_box = QComboBox()
        self.station_box.setToolTip(tr("The survey stations and their chainage - choose one to show it on the sketch"))
        self.station_box.activated.connect(self._goto_station)
        self.f_geologist = QLineEdit(QSettings().value("pit_geology/geologist", "", str))
        self.f_geologist.setPlaceholderText(tr("Name"))
        self.f_geologist.setToolTip(tr("Filled in on new rows"))
        self.f_geologist.editingFinished.connect(
            lambda: QSettings().setValue("pit_geology/geologist", self.f_geologist.text().strip()))
        self.f_date = QLineEdit(QDate.currentDate().toString("yyyy-MM-dd"))
        self.f_date.setMaximumWidth(100)
        self.f_date.setToolTip(tr("Filled in on new rows (YYYY-MM-DD)"))
        head = QHBoxLayout()
        head.addWidget(QLabel(tr("Stations")))
        head.addWidget(self.station_box, 1)
        head.addSpacing(16)
        head.addWidget(QLabel(tr("Geologist")))
        head.addWidget(self.f_geologist, 1)
        head.addWidget(QLabel(tr("Date")))
        head.addWidget(self.f_date)

        # the three tables
        self.g_rocks = GeoGrid(self, "interval", interval_columns(),
                               tr("Rocks along the drive, From - To in metres from the pit's start (or a station + "
                                  "metres, e.g. P1-03+4.5). Where: both walls and the back, or one of them."))
        self.g_structs = GeoGrid(self, "structure", structure_columns(),
                                 tr("Planar structures measured with the compass: dip direction and dip (135∠45 = dips "
                                    "45° toward 135°). Angles can be typed 135 or 135 30 00."))
        b_fields = QPushButton(icon("table-column-plus-after"), tr("Assay fields ..."))
        b_fields.setToolTip(tr("Add or remove an assay field of the samples (gold, silver, cu_pct, ...)"))
        b_fields.clicked.connect(self._fields_menu)
        self.b_fields = b_fields
        self.g_samples = GeoGrid(self, "sample", sample_columns(), tr("Samples cut in the drive: number, From - To, "
                                                                       "wall and height, channel size, weight and the "
                                                                       "assays."), (b_fields,))
        self.tabs = QTabWidget()
        self.tabs.addTab(self.g_rocks, icon("layers-outline"), tr("Rocks"))
        self.tabs.addTab(self.g_structs, icon("angle-acute"), tr("Structures"))
        self.tabs.addTab(self.g_samples, icon("flask-outline"), tr("Samples"))
        for g in (self.g_rocks, self.g_structs, self.g_samples):
            g.edited.connect(self._edited)
            g.row_selected.connect(lambda r, g=g: self._row_selected(g, r))
        self.tabs.currentChanged.connect(lambda _i: self._row_selected(self._grid(), self._grid().table.currentRow()))

        # the sketch
        self.sketch = WallSketch()
        self.sketch.picked.connect(self._sketch_picked)
        b_fit = QPushButton(icon("arrow-expand-horizontal"), tr("Whole pit"))
        b_fit.setToolTip(tr("Show the whole pit (or double-click the sketch)"))
        b_fit.clicked.connect(self.sketch.fit_all)
        b_png = QPushButton(icon("image-outline"), tr("Save image ..."))
        b_png.setToolTip(tr("Save the sketch as shown (same chainage range) as a 300 dpi PNG on white, A4 landscape"))
        b_png.clicked.connect(self.save_image_dialog)
        b_print = QPushButton(icon("printer-outline"), tr("Print ..."))
        b_print.setToolTip(tr("Print the sketch as shown (same chainage range), A4 landscape"))
        b_print.clicked.connect(self.print_dialog)
        sk_head = QHBoxLayout()
        sk_hint = QLabel(tr("Wall sketch - wheel zooms, drag pans, double-click shows the whole pit, click a record "
                            "to select it"))
        sk_hint.setStyleSheet(f"color: {theme.MUTED};")
        sk_head.addWidget(sk_hint, 1)
        for b in (b_fit, b_png, b_print):
            sk_head.addWidget(b)
        sk = QWidget()
        sv = QVBoxLayout(sk)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.addLayout(sk_head)
        sv.addWidget(self.sketch, 1)

        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(self.title)
        tv.addWidget(self.origin)
        tv.addLayout(head)
        tv.addWidget(self.tabs, 1)
        mid = QSplitter(Qt.Vertical)
        mid.addWidget(top)
        mid.addWidget(sk)
        mid.setSizes([470, 330])
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(mid)
        split.setSizes([210, 1150])

        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {theme.MUTED};")
        self.b_save = QPushButton(icon("content-save-outline"), tr("Save pit geology"))
        self.b_save.setToolTip(tr("Write this pit's rocks, structures and samples to the database; the drives in the "
                                  "view show them"))
        self.b_save.clicked.connect(self.save)
        self.b_discard = QPushButton(icon("undo"), tr("Discard changes"))
        self.b_discard.setToolTip(tr("Forget the changes to this pit since it was last saved"))
        self.b_discard.clicked.connect(lambda: self.show_pit(self.pid, force=True))
        close = QPushButton(tr("Close"))
        close.setToolTip(tr("Close the window (asks to save changes first)"))
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.b_discard)
        bottom.addWidget(self.b_save)
        bottom.addWidget(close)
        v = QVBoxLayout(self)
        v.addWidget(split, 1)
        v.addWidget(self.msg)
        v.addLayout(bottom)
        self._set_dirty(False)

    # --- data ------------------------------------------------------------------------------------
    def start(self, pid=None) -> bool:
        """Read the pits and their geology; False (after saying why) when the database cannot be read."""
        try:
            self.store.prepare()
            self._load()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Pit Geology"), tr("Cannot read the pits:\n{e}", e=e))
            return False
        self._fill_tree(pid)
        return True

    def _load(self) -> None:
        pits = self.survey.pits()
        self.data = SurveyData(pits, {s.name: s for s in self.survey.stations()}, pit_paths(self.survey.setups()),
                               str(getattr(self.target, "label", "")))
        self.geology = self.store.geology()
        self.assay_fields = self.store.assay_fields()
        self.rock_codes, self.rock_colors, self.rock_desc = [], {}, {}
        try:
            from ..io.drillhole_store import DrillholeStore

            dh = DrillholeStore(self.target)
            self.rock_colors = dh.styles()
            for code, desc, _n in dh.rock_codes():
                self.rock_codes.append((code, self.rock_colors.get(code, "#8a9096"), desc))
                self.rock_desc[code] = desc
        except Exception:  # noqa: BLE001 - no rock tables yet: codes only from the geology
            pass
        self.g_samples.set_columns(sample_columns(self.assay_fields))

    def _fill_tree(self, select=None) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        items = {}
        pits = self.data.pits
        for p in sorted(pits, key=lambda q: (q.parent_id is not None, q.id)):
            g = self.geology.get(p.id)
            n = (len(g.intervals) + len(g.structures) + len(g.samples)) if g else 0
            it = QTreeWidgetItem([f"{p.name}  ({p.code})" + (f"  · {n}" if n else "")])
            it.setData(0, Qt.UserRole, p.id)
            it.setToolTip(0, tr("{n} records mapped", n=n) if n else tr("nothing mapped yet"))
            if p.parent_id is None:
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
            items[p.id] = it
        for p in pits:
            parent = items.get(p.parent_id)
            (parent.addChild if parent is not None else self.tree.addTopLevelItem)(items[p.id])
        self.tree.expandAll()
        self.tree.blockSignals(False)
        target = select if select in items else (self.pid if self.pid in items else (pits[0].id if pits else None))
        if target is not None:
            self.tree.setCurrentItem(items[target])
            self.show_pit(target, force=True)
        else:
            self.show_pit(None, force=True)

    def _pit_clicked(self, item, _prev) -> None:
        pid = item.data(0, Qt.UserRole) if item is not None else None
        if pid == self.pid:
            return
        if not self._leave():
            self.tree.blockSignals(True)
            self._select_item(self.pid)
            self.tree.blockSignals(False)
            return
        self.show_pit(pid, force=True)

    def _select_item(self, pid) -> None:
        todo = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while todo:
            it = todo.pop()
            if it.data(0, Qt.UserRole) == pid:
                self.tree.setCurrentItem(it)
                return
            todo += [it.child(i) for i in range(it.childCount())]

    def select_pit(self, pid) -> None:
        """Show a pit (asks to save the one being edited first)."""
        if pid == self.pid or not self._leave():
            return
        self.tree.blockSignals(True)
        self._select_item(pid)
        self.tree.blockSignals(False)
        self.show_pit(pid, force=True)

    def show_pit(self, pid, force: bool = False) -> None:
        self.pid = pid
        p = self.data.pit(pid) if (self.data is not None and pid is not None) else None
        on = p is not None
        for w in (self.tabs, self.b_save, self.b_discard, self.station_box):
            w.setEnabled(on)
        if not on:
            self.title.setText(tr("Select a pit") if self.data and self.data.pits else
                               tr("No pits yet - add them in the Pit Manager (Ctrl+U)"))
            self.origin.setText("")
            for g in (self.g_rocks, self.g_structs, self.g_samples):
                g.set_values([])
            self.sketch.set_data(None)
            self._set_dirty(False)
            return
        self.axis = pit_axis(self.data, pid)
        self.stations_m = self.axis.station_chainage() if self.axis is not None else {}
        self.title.setText(tr("{name} ({code})", name=p.name, code=p.code))
        self.origin.setText(self._origin_text(p))
        self.station_box.clear()
        for n, m in sorted(self.stations_m.items(), key=lambda kv: kv[1]):
            self.station_box.addItem(f"{n}   {m:,.2f} m", m)
        geo = self.geology.get(pid) or PitGeology()
        self.g_rocks.set_values(_rows_of(geo.intervals, "interval"))
        self.g_structs.set_values(_rows_of(geo.structures, "structure"))
        self.g_samples.set_values(_rows_of(geo.samples, "sample"))
        self._set_dirty(False)
        self._update_tabs()
        self._refresh_sketch(keep_view=False)
        self._check()

    def _origin_text(self, p) -> str:
        parent = self.data.pit(p.parent_id) if p.parent_id is not None else None
        if parent is not None:
            zero = tr("0 m = junction station {st} on {parent}", st=p.start_station or "?", parent=parent.name)
        else:
            zero = tr("0 m = entrance station {st}", st=p.start_station or "?")
        if self.axis is None:
            return zero + " · " + tr("not surveyed far enough yet (fewer than two stations with coordinates): "
                                     "chainages are kept but cannot be placed in the view")
        last = self.axis.names[-1]
        return " · ".join([zero, tr("surveyed to {st} at {m:,.2f} m", st=last, m=self.axis.length),
                           tr("profile {w:g} x {h:g} m", w=p.profile_width, h=p.profile_height),
                           tr("left / right looking toward the face")])

    def _goto_station(self, i: int) -> None:
        m = self.station_box.itemData(i)
        if m is not None:
            self.sketch.show_chainage(float(m))

    # --- defaults for new rows --------------------------------------------------------------------
    def default_geologist(self) -> str:
        return self.f_geologist.text().strip()

    def default_date(self) -> str:
        t = self.f_date.text().strip()
        return t if QDate.fromString(t, "yyyy-MM-dd").isValid() else ""

    # --- edits -------------------------------------------------------------------------------------
    def _grid(self) -> GeoGrid:
        return self.tabs.currentWidget()

    def flash(self, text: str) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {theme.DANGER};")
        QTimer.singleShot(6000, lambda: (self.status.setStyleSheet(f"color: {theme.MUTED};")))

    def current_geology(self) -> PitGeology:
        return PitGeology(_records_of(self.g_rocks.values(), "interval"),
                          _records_of(self.g_structs.values(), "structure"),
                          _records_of(self.g_samples.values(), "sample"))

    def _edited(self) -> None:
        self._set_dirty(True)
        self._update_tabs()
        self._timer.start()

    def _update_tabs(self) -> None:
        for i, (g, name) in enumerate(((self.g_rocks, tr("Rocks")), (self.g_structs, tr("Structures")),
                                       (self.g_samples, tr("Samples")))):
            n = g.table.rowCount()
            self.tabs.setTabText(i, f"{name} ({n})" if n else name)

    def _set_dirty(self, on: bool) -> None:
        self.dirty = bool(on)
        self.b_save.setEnabled(self.pid is not None and self.dirty)
        self.b_discard.setEnabled(self.pid is not None and self.dirty)
        self.status.setStyleSheet(f"color: {theme.MUTED};")
        self.status.setText(tr("Changes not saved yet") if self.dirty else "")

    def _sketch_data(self) -> SketchData | None:
        p = self.data.pit(self.pid) if self.data is not None and self.pid is not None else None
        if p is None:
            return None
        geo = self.current_geology()
        colors = dict(self.rock_colors)
        from ..scene.layers import PALETTE

        missing = sorted({r.rock for r in geo.intervals if r.rock and r.rock not in colors})
        for i, code in enumerate(missing):                     # not registered: a palette colour
            colors[code] = QColor.fromRgbF(*PALETTE[i % len(PALETTE)]).name()
        ax = self.axis
        return SketchData(f"{p.name} ({p.code})", p.profile_width, p.profile_height, ax.length if ax else None,
                          dict(self.stations_m), geo, colors, dict(self.rock_desc),
                          ax.bearing_at if ax is not None else None)

    def _refresh_sketch(self, keep_view: bool = True) -> None:
        self.sketch.set_data(self._sketch_data(), keep_view=keep_view)
        self._check()

    def _check(self) -> tuple[list, list]:
        if self.pid is None:
            self.msg.setText("")
            return [], []
        geo = self.current_geology()
        errors, warnings = check_geology(geo, self.axis.length if self.axis is not None else None, self.assay_fields)
        try:
            others = self.store.sample_names(exclude_pit=self.pid)
        except Exception:  # noqa: BLE001
            others = {}
        names = {p.id: p.name for p in self.data.pits}
        for s in geo.samples:
            if s.name in others:
                errors.append(tr("Sample {name} is already in pit {pit}", name=s.name,
                                 pit=names.get(others[s.name], "?")))
        lines = [f"<span style='color:{theme.DANGER}'>✖ {e}</span>" for e in errors[:6]]
        lines += [f"<span style='color:#f0b429'>⚠ {w}</span>" for w in warnings[:max(0, 6 - len(lines))]]
        more = len(errors) + len(warnings) - len(lines)
        if more > 0:
            lines.append(tr("... and {n} more", n=more))
        self.msg.setText("<br>".join(lines))
        return errors, warnings

    def _row_selected(self, grid: GeoGrid, r: int) -> None:
        if r < 0 or grid is not self._grid():
            self.sketch.set_highlight(None)
            return
        rows = grid.values()
        if r >= len(rows):
            return
        self.sketch.set_highlight(grid.kind, r)
        row = rows[r]
        m = row.get("frm") if row.get("frm") is not None else row.get("at")
        if m is not None:
            self.sketch.show_chainage(float(m))

    def _sketch_picked(self, kind: str, i: int) -> None:
        grid = {"interval": self.g_rocks, "structure": self.g_structs, "sample": self.g_samples}[kind]
        self.tabs.setCurrentWidget(grid)
        grid.table.setCurrentCell(i, 0)
        grid.table.selectRow(i)

    # --- assay fields --------------------------------------------------------------------------------
    def _fields_menu(self) -> None:
        m = QMenu(self)
        m.addAction(icon("plus"), tr("Add field ..."), self.add_field)
        rm = m.addMenu(icon("delete-outline"), tr("Remove field"))
        for f in self.assay_fields:
            rm.addAction(f, lambda f=f: self.remove_field(f))
        rm.setEnabled(bool(self.assay_fields))
        m.exec(self.b_fields.mapToGlobal(self.b_fields.rect().bottomLeft()))

    def add_field(self, name: str | None = None) -> bool:
        if name is None:
            name, ok = QInputDialog.getText(self, tr("Add assay field"), tr("Field name (e.g. gold, cu_pct):"))
            if not ok:
                return False
        try:
            name = self.store.add_field(name)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Add assay field"), str(e))
            return False
        self.assay_fields = self.store.assay_fields()
        self.g_samples.set_columns(sample_columns(self.assay_fields))
        self.changed.emit()
        return True

    def remove_field(self, name: str) -> bool:
        if QMessageBox.question(self, tr("Remove assay field"),
                                tr("Remove the field {name} from all pit samples? Its values are deleted.",
                                   name=name)) != QMessageBox.Yes:
            return False
        try:
            self.store.remove_field(name)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Remove assay field"), str(e))
            return False
        self.assay_fields = self.store.assay_fields()
        self.g_samples.set_columns(sample_columns(self.assay_fields))
        for g in self.geology.values():
            for s in g.samples:
                s.values.pop(name, None)
        self.changed.emit()
        return True

    # --- saving ------------------------------------------------------------------------------------
    def save(self) -> bool:
        if self.pid is None:
            return True
        errors, warnings = self._check()
        if errors:
            QMessageBox.warning(self, tr("Pit Geology"), tr("Not saved - please correct:") + "\n\n" +
                                "\n".join(errors[:15]) + ("\n..." if len(errors) > 15 else ""))
            return False
        geo = sorted_geology(self.current_geology())
        try:
            self.store.save(self.pid, geo)
            self.geology = self.store.geology()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Pit Geology"), tr("Could not save:\n{e}", e=e))
            return False
        pid = self.pid
        self._set_dirty(False)
        self._fill_tree(pid)                   # the counts; the rows in chainage order
        n = len(geo.intervals) + len(geo.structures) + len(geo.samples)
        self.status.setText(tr("Saved {n} records", n=n) + (tr(" ({w} warnings)", w=len(warnings)) if warnings else ""))
        self.changed.emit()
        return True

    def _leave(self) -> bool:
        if not self.dirty:
            return True
        ans = QMessageBox.question(self, tr("Pit Geology"), tr("Save the geology of this pit?"),
                                   QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if ans == QMessageBox.Save:
            return self.save()
        if ans == QMessageBox.Discard:
            self._set_dirty(False)
            return True
        return False

    def reload(self) -> None:
        """The survey changed elsewhere (Pit Manager): read again; the pit being edited keeps its rows."""
        if self.dirty:
            return
        try:
            self._load()
        except Exception:  # noqa: BLE001
            return
        self._fill_tree(self.pid)

    def closeEvent(self, e) -> None:
        if not self._leave():
            e.ignore()
            return
        super().closeEvent(e)

    # --- Excel ------------------------------------------------------------------------------------------
    def _folder(self) -> str:
        """Where the Excel file dialogs start: the workspace, else the project folder (home for MySQL)."""
        if self.workspace_folder:
            return str(self.workspace_folder)
        t = self.target
        return str(t.path.parent if not getattr(t, "is_mysql", False) and getattr(t, "path", None) else Path.home())

    def import_excel(self) -> bool:
        """Pit geology from an Excel workbook (the template filled in, or an export edited)."""
        from ..io import pit_geology_excel as pe
        from .excel_import import ExcelImportDialog

        if not self._leave():
            return False
        dlg = ExcelImportDialog(
            self, title=tr("Import pit geology from Excel"),
            intro=tr("Map many pits at once: save the template, fill in its sheets rocks, structures and samples in "
                     "Excel (each row names its pit), then choose the file here (a workbook from Export to Excel "
                     "works too). Chainages in metres from the pit's start or as a station + metres (P1-03+4.5). "
                     "A pit already mapped gets the file's rocks, structures or samples - each sheet that has rows "
                     "for it - or it is skipped. New assay columns become assay fields."),
            existing=tr("Pits already mapped"), folder=self._folder(),
            template_name="pit_geology_template.xlsx",
            save_template=lambda path: pe.save_template(self.store, path),
            read=lambda path: pe.read_import(self.store, path),
            apply=lambda plan, replace: pe.apply_import(self.store, plan, replace),
            result_text=pe.result_text)
        if dlg.exec() != QDialog.Accepted:
            return False
        self._set_dirty(False)
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Pit Geology"), tr("Cannot read the pits:\n{e}", e=e))
            return False
        self._fill_tree(self.pid)
        self.status.setText(dlg.result_message())
        self.changed.emit()
        return True

    def export_excel(self, path: str | None = None) -> bool:
        """Every pit's geology to an Excel workbook (what is saved: unsaved changes are offered
        for saving first)."""
        from ..io import pit_geology_excel as pe

        title = tr("Export to Excel")
        if self.dirty:
            ans = QMessageBox.question(self, title, tr("The geology of this pit has changes that are not saved. Save "
                                                       "them first?\n\n(Unsaved changes are not in the export.)"))
            if ans == QMessageBox.Yes and not self.save():
                return False
        if path is None:
            start = str(Path(self._folder()) / "pit_geology.xlsx")
            path, _ = QFileDialog.getSaveFileName(self, tr("Export pit geology to Excel"), start,
                                                  tr("Excel workbook (*.xlsx)"))
            if not path:
                return False
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            counts = pe.export_excel(self.store, path)
        except PermissionError:
            QMessageBox.warning(self, title, tr("{path}\n\nThe file cannot be written - is it open in Excel? Close it "
                                                "and try again.", path=path))
            return False
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, title, tr("The export failed:\n{error}", error=e))
            return False
        self.status.setText(tr("Exported to {file}: {counts}", file=Path(path).name,
                               counts=", ".join(f"{k} {v:,}" for k, v in counts.items())))
        return True

    # --- output --------------------------------------------------------------------------------------
    def render_page(self, p: QPainter, rect: QRectF, dpi: float) -> None:
        """The sketch on paper: a title, then the strips for the chainage range on screen."""
        d = self._sketch_data()
        if d is None:
            return
        style = LogStyle(**{**PAPER.__dict__, "scale": dpi / 96.0})
        from .drillhole_log import _font

        s = style.scale
        f = _font(style, 14, True)
        p.setFont(f)
        p.setPen(QColor(style.text))
        line_h = 24 * s
        p.drawText(QRectF(rect.left(), rect.top(), rect.width(), line_h), Qt.AlignLeft | Qt.AlignVCenter,
                   tr("Pit geology - {name}", name=d.title))
        m0, m1 = self.sketch.visible_range()
        p.setFont(_font(style, 10))
        p.setPen(QColor(style.muted))
        p.drawText(QRectF(rect.left(), rect.top() + line_h, rect.width(), 18 * s), Qt.AlignLeft | Qt.AlignVCenter,
                   tr("Chainage {a:,.1f} - {b:,.1f} m", a=m0, b=m1) + " · " + self.origin.text())
        body = QRectF(rect.left(), rect.top() + line_h + 26 * s, rect.width(), rect.height() - line_h - 26 * s)
        from .pit_sketch import sketch_scale

        _x, _y, aw, _ah, fit = sketch_scale(body, d, style)
        ppm = min(fit, aw / max(m1 - m0, 1e-6))
        paint_sketch(p, body, d, style, m0, ppm)

    def page_image(self, dpi: int = PRINT_DPI) -> QImage:
        w, h = int(round(297 / 25.4 * dpi)), int(round(210 / 25.4 * dpi))
        img = QImage(w, h, QImage.Format_RGB32)
        img.fill(QColor("white"))
        dpm = int(round(dpi / 0.0254))
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        p = QPainter(img)
        margin = 10 / 25.4 * dpi
        self.render_page(p, QRectF(margin, margin, w - 2 * margin, h - 2 * margin), dpi)
        p.end()
        return img

    def save_image_dialog(self) -> None:
        p = self.data.pit(self.pid) if self.data is not None and self.pid is not None else None
        if p is None:
            return
        start = str(Path(self.workspace_folder or ".") / f"pit_geology_{p.code}.png")
        path, _ = QFileDialog.getSaveFileName(self, tr("Save wall sketch"), start, tr("PNG image (*.png)"))
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not self.page_image().save(path, "PNG"):
            QMessageBox.warning(self, tr("Save wall sketch"), tr("Could not write {path}", path=path))
            return
        self.status.setText(tr("Saved: {path}", path=path))

    def print_dialog(self) -> None:
        p = self.data.pit(self.pid) if self.data is not None and self.pid is not None else None
        if p is None:
            return
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter

        printer = QPrinter(QPrinter.HighResolution)
        printer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Landscape,
                                          printer.pageLayout().margins()))
        printer.setDocName(tr("Pit geology - {name}", name=p.name))
        dlg = QPrintDialog(printer, self)
        if dlg.exec() != QDialog.Accepted:
            return
        painter = QPainter(printer)
        try:
            rect = printer.pageRect(QPrinter.DevicePixel)
            self.render_page(painter, QRectF(0, 0, rect.width(), rect.height()), printer.resolution())
        finally:
            painter.end()
        self.status.setText(tr("Sent to the printer"))
