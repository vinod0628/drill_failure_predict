"""Drilling Records: the database of drilling success and failure (design spec 3, 7.2).

One window for everything the risk models learn from and the driller records live:

* left: the holes, with a search box, area / result filters and a summary (holes, failure rate);
* right, **Hole record**: the collar in a clear form (identification, location, plan, status, site), then
  one tab per table - Survey, Geology, Runs, Hole size, Bit, Occurrences, Failure. Runs has *Next run*
  (From = the last To, the last values copied), *Record event* and *Record failure* (the one-tap forms
  of spec 7.2). *Save* runs the entry checks of spec 7.3 first: errors are marked on their cells and
  stop the save, warnings are shown;
* **Areas / Formation cards / Rigs**: the reference tables (site sheets, property cards, rig limits);
* **Overview**: failures by area and code, occurrences, and the fill rate of the run fields per crew;
* CSV import / export of all tables (spec 3.4), and the holes of the workspace's Drillhole Manager
  copied in as records of the current area.

Everything is written to the project database (SQLite or MySQL, risk_* tables).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QDate, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..io.database import Target
from ..risk import codes
from ..risk.schema import (AREA, BIT, COLLAR, FAILURE, FORMATION, GEOLOGY, HOLE_SIZE, HOLE_TABLES, MEASURE,
                           OCCURRENCE, RIG, ROCK, ROCK_CODE, RUN, SURVEY, Col)
from ..risk.store import Issue, RiskStore, apply_reference, default_reference, validate

REF_TABLES = (AREA, FORMATION, RIG, ROCK, ROCK_CODE, MEASURE)
REF_ICONS = {"area": "map-outline", "formation": "card-text-outline", "rig": "tower-fire", "rock": "diamond-stone",
             "rock_code": "format-list-bulleted-type", "measure": "book-open-variant"}
from . import theme
from .icons import icon
from .risk_widgets import Banner, ChipLabel, RecordTable, StatCard, chip, fmt_value, is_blank, pct

log = logging.getLogger(__name__)
TAB_TABLES = (SURVEY, GEOLOGY, RUN, HOLE_SIZE, BIT, OCCURRENCE, FAILURE)
COLLAR_GROUPS = (
    (tr("Identification"), ("hole_id", "area_id", "rig_model", "crew")),
    (tr("Collar location"), ("x", "y", "z")),
    (tr("Design (plan)"), ("plan_azimuth", "plan_dip", "planned_depth")),
    (tr("Status"), ("result", "final_depth", "start_date", "end_date")),
    (tr("At the rig"), ("casing_available", "fluids_available")),
)


# --- small forms --------------------------------------------------------------------------------
class MeasurePicker(QDialog):
    """Pick measure codes from the library (filtered by failure codes, if given)."""

    def __init__(self, parent=None, selected: str = "", fail_codes: list[str] | None = None, stage: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Measures"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        chosen = [s.strip() for s in str(selected or "").split(";") if s.strip()]
        v = QVBoxLayout(self)
        self.only = QCheckBox(tr("Only measures for {codes}", codes=", ".join(fail_codes))) if fail_codes else None
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("Search measures ..."))
        self.search.textChanged.connect(self._filter)
        v.addWidget(self.search)
        if self.only is not None:
            self.only.setChecked(True)
            self.only.toggled.connect(self._filter)
            v.addWidget(self.only)
        self.list = QListWidget()
        self.fail_codes = set(fail_codes or [])
        for code, m in codes.MEASURES.items():
            it = QListWidgetItem(f"{code}  ·  {tr(m.title)}   [{'/'.join(m.failures)}]  "
                                 f"{tr('prevention') if m.stage == 'P' else tr('response')}")
            it.setData(Qt.UserRole, code)
            it.setToolTip(tr(m.text))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if code in chosen else Qt.Unchecked)
            self.list.addItem(it)
        v.addWidget(self.list, 1)
        v.addWidget(QLabel(tr("Tick the measures in the order they were tried; the last one ticked is taken as the "
                              "one that solved the problem (when solved).")))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.order = list(chosen)
        self.list.itemChanged.connect(self._changed)
        self.resize(640, 520)
        self._filter()

    def _changed(self, it: QListWidgetItem) -> None:
        code = it.data(Qt.UserRole)
        if it.checkState() == Qt.Checked and code not in self.order:
            self.order.append(code)
        elif it.checkState() != Qt.Checked and code in self.order:
            self.order.remove(code)

    def _filter(self) -> None:
        q = self.search.text().strip().lower()
        only = self.only is not None and self.only.isChecked()
        for i in range(self.list.count()):
            it = self.list.item(i)
            m = codes.MEASURES[it.data(Qt.UserRole)]
            show = (not q or q in it.text().lower() or q in tr(m.text).lower()) and \
                (not only or bool(self.fail_codes & set(m.failures)) or it.checkState() == Qt.Checked)
            it.setHidden(not show)

    def value(self) -> str:
        return ";".join(self.order)


def _measure_row(parent, edit: QLineEdit, fail_codes_fn) -> QWidget:
    w = QWidget(parent)
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit, 1)
    b = QToolButton(w)
    b.setIcon(icon("format-list-checks"))
    b.setToolTip(tr("Pick from the measure library"))

    def pick():
        dlg = MeasurePicker(parent, edit.text(), fail_codes_fn())
        if dlg.exec():
            edit.setText(dlg.value())
    b.clicked.connect(pick)
    h.addWidget(b)
    return w


class EventDialog(QDialog):
    """Record an occurrence (cavity, fault zone, other problem) with its measures and result."""

    def __init__(self, parent, depth: float, kind: str = "cavity"):
        super().__init__(parent)
        self.setWindowTitle(tr("Record an event"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        f = QFormLayout(self)
        self.type = QComboBox()
        for k, t in codes.OCC_TYPES.items():
            self.type.addItem(tr(t), k)
        self.type.setCurrentIndex(max(0, self.type.findData(kind)))
        self.d_from, self.d_to = QDoubleSpinBox(), QDoubleSpinBox()
        for s, v in ((self.d_from, depth), (self.d_to, depth + 1.0)):
            s.setRange(0, 5000)
            s.setDecimals(2)
            s.setSuffix(" m")
            s.setValue(v)
        self.measures = QLineEdit()
        self.solved = QComboBox()
        self.solved.addItem(tr("Not yet / unknown"), None)
        self.solved.addItem(tr("Yes - solved"), "Y")
        self.solved.addItem(tr("No - not solved"), "N")
        self.notes = QLineEdit()
        f.addRow(tr("Type"), self.type)
        f.addRow(tr("From"), self.d_from)
        f.addRow(tr("To"), self.d_to)
        f.addRow(tr("Measures"), _measure_row(self, self.measures, self._codes))
        f.addRow(tr("Solved"), self.solved)
        f.addRow(tr("Notes"), self.notes)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.resize(460, 260)

    def _codes(self) -> list[str]:
        k = self.type.currentData()
        return {"cavity": ["CV", "LC"], "loss": ["LC"], "fault": ["CL", "SP", "LC"], "broken": ["CL", "CB"],
                "clay": ["CL", "SP"], "water": ["WI"]}.get(k, list(codes.FAIL_CODES))

    def _ok(self) -> None:
        if self.d_to.value() <= self.d_from.value():
            QMessageBox.warning(self, self.windowTitle(), tr("To must be greater than From."))
            return
        self.accept()

    def row(self) -> dict:
        return {"depth_from": self.d_from.value(), "depth_to": self.d_to.value(), "type": self.type.currentData(),
                "measures": self.measures.text().strip() or None, "solved": self.solved.currentData(),
                "notes": self.notes.text().strip() or None}


class FailureDialog(QDialog):
    """Record the failure of an abandoned hole (spec 7.2: FAILURE form)."""

    def __init__(self, parent, depth: float, current: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Record the failure"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        cur = current or {}
        f = QFormLayout(self)
        self.depth = QDoubleSpinBox()
        self.depth.setRange(0, 5000)
        self.depth.setDecimals(1)
        self.depth.setSuffix(" m")
        self.depth.setValue(float(cur.get("fail_depth") or depth))
        self.code = QComboBox()
        for k, (name, fam) in codes.FAIL_CODES.items():
            self.code.addItem(f"{k} - {tr(name)}  ({tr(codes.FAMILIES[fam])})", k)
        if cur.get("fail_code"):
            self.code.setCurrentIndex(max(0, self.code.findData(cur["fail_code"])))
        self.sev = QComboBox()
        for k, t in codes.SEVERITY.items():
            self.sev.addItem(tr(t), k)
        self.sev.setCurrentIndex(max(0, self.sev.findData(int(float(cur.get("severity") or 3)))))
        self.measures = QLineEdit(str(cur.get("measures") or ""))
        self.notes = QLineEdit(str(cur.get("notes") or ""))
        f.addRow(tr("Failure depth"), self.depth)
        f.addRow(tr("Failure code"), self.code)
        f.addRow(tr("Severity"), self.sev)
        f.addRow(tr("Measures tried"), _measure_row(self, self.measures, lambda: [self.code.currentData()]))
        f.addRow(tr("Notes"), self.notes)
        note = QLabel(tr("The hole's result becomes Failed and its final depth the failure depth."))
        note.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.resize(520, 260)

    def row(self) -> dict:
        return {"fail_depth": self.depth.value(), "fail_code": self.code.currentData(),
                "severity": self.sev.currentData(), "measures": self.measures.text().strip() or None,
                "notes": self.notes.text().strip() or None}


class NewHoleDialog(QDialog):
    def __init__(self, parent, areas: list[str], holes: list[str], default_area: str | None):
        super().__init__(parent)
        self.setWindowTitle(tr("New hole"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.holes = set(holes)
        f = QFormLayout(self)
        self.hid = QLineEdit()
        self.hid.setPlaceholderText(tr("e.g. DH-2026-004"))
        self.area = QComboBox()
        self.area.setEditable(True)
        self.area.addItems(areas)
        if default_area:
            self.area.setCurrentText(default_area)
        self.state = QComboBox()
        for k in ("planned", "in_progress", "success", "failed"):
            self.state.addItem(tr(codes.RESULTS[k]), k)
        self.copy = QComboBox()
        self.copy.addItem(tr("(nothing)"), None)
        for h in holes:
            self.copy.addItem(h, h)
        self.copy.setToolTip(tr("Copy the rig, design and site settings of another hole"))
        f.addRow(tr("Hole ID"), self.hid)
        f.addRow(tr("Area"), self.area)
        f.addRow(tr("State"), self.state)
        f.addRow(tr("Copy settings from"), self.copy)
        self.msg = QLabel("")
        self.msg.setStyleSheet(f"color: {theme.DANGER};")
        f.addRow(self.msg)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _ok(self) -> None:
        h = self.hid.text().strip()
        if not h:
            self.msg.setText(tr("Enter a hole ID."))
            return
        if h in self.holes:
            self.msg.setText(tr("{hole} is already in the database.", hole=h))
            return
        self.accept()


# --- the window ---------------------------------------------------------------------------------
class DrillingRecordsDialog(QDialog):
    changed = Signal()            # records written (the risk layer / prediction read them again)

    def __init__(self, target: Target, folder: Path | None, parent=None, current_area: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Drilling Records - success and failure database"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.target = target
        self.folder = folder
        self.store = RiskStore(target)
        self.current_area = current_area
        self.hole_id: str | None = None
        self.dirty = False
        self._loading = False
        self.collars = pd.DataFrame(columns=COLLAR.names)
        self.refs: dict[str, pd.DataFrame] = {}
        self._build()
        self.resize(1500, 900)
        QTimer.singleShot(0, self.reload)

    # --- layout ----------------------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel(tr("Drilling Records"))
        title.setStyleSheet("font-size: 15pt; font-weight: 700;")
        top.addWidget(title)
        self.db_label = QLabel(self.target.label)
        self.db_label.setStyleSheet(f"color: {theme.MUTED};")
        top.addWidget(self.db_label)
        top.addStretch(1)
        for ic, text, tip, slot in (
                ("plus-circle-outline", tr("New hole"), tr("Add a hole (planned, in progress or finished)"), self.new_hole),
                ("database-import-outline", tr("Import CSV ..."), tr("Import the tables from a folder of CSV files "
                                                                     "(collar.csv, survey.csv, run.csv, ...)"),
                 self.import_csv),
                ("database-export-outline", tr("Export CSV ..."), tr("Export every table as CSV (UTF-8)"),
                 self.export_csv),
                ("map-marker-plus-outline", tr("Holes without an area ..."),
                 tr("Give the drillholes entered in the Drillhole Manager without an area (and result) their area, "
                    "so that they count as records of that area"),
                 self.import_workspace_holes)):
            b = QPushButton(icon(ic), text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            top.addWidget(b)
        root.addLayout(top)
        self.banner = Banner(self)
        root.addWidget(self.banner)
        split = QSplitter(Qt.Horizontal, self)
        root.addWidget(split, 1)
        split.addWidget(self._left())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._hole_page(), icon("file-document-edit-outline"), tr("Hole record"))
        self.ref_tables = {}
        for t in REF_TABLES:
            self.tabs.addTab(self._ref_page(t), icon(REF_ICONS[t.name]), tr(t.title))
        self.overview = QTextBrowser()
        self.overview.setOpenLinks(False)
        self.tabs.addTab(self.overview, icon("chart-box-outline"), tr("Overview"))
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 1080])
        QShortcut(QKeySequence.Save, self, activated=self.save)

    def _left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        cards = QHBoxLayout()
        self.card_holes = StatCard(tr("holes"))
        self.card_fail = StatCard(tr("failed"))
        cards.addWidget(self.card_holes)
        cards.addWidget(self.card_fail)
        v.addLayout(cards)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("Search hole ID ..."))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._fill_list)
        v.addWidget(self.search)
        f = QHBoxLayout()
        self.f_area = QComboBox()
        self.f_area.currentIndexChanged.connect(self._fill_list)
        self.f_result = QComboBox()
        self.f_result.addItem(tr("All results"), None)
        for k, t in codes.RESULTS.items():
            self.f_result.addItem(tr(t), k)
        self.f_result.currentIndexChanged.connect(self._fill_list)
        f.addWidget(self.f_area, 1)
        f.addWidget(self.f_result, 1)
        v.addLayout(f)
        self.list = QTreeWidget()
        self.list.setHeaderLabels([tr("Hole"), tr("Area"), tr("Result"), tr("Depth (m)")])
        self.list.setRootIsDecorated(False)
        self.list.setAlternatingRowColors(True)
        self.list.setSortingEnabled(True)
        for col in (0, 2, 3):
            self.list.header().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.list.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.list.header().setMinimumSectionSize(60)
        w.setMinimumWidth(380)
        self.list.currentItemChanged.connect(self._hole_chosen)
        v.addWidget(self.list, 1)
        self.list_count = QLabel("")
        self.list_count.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(self.list_count)
        hint = QLabel(tr("Holes in progress and planned holes are listed first."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt;")
        v.addWidget(hint)
        return w

    def _hole_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        head = QHBoxLayout()
        self.h_title = QLabel(tr("No hole selected"))
        self.h_title.setStyleSheet("font-size: 14pt; font-weight: 700;")
        self.h_chip = ChipLabel()
        self.h_state = QLabel("")
        self.h_state.setStyleSheet(f"color: {theme.MUTED};")
        head.addWidget(self.h_title)
        head.addWidget(self.h_chip)
        head.addSpacing(12)
        head.addWidget(self.h_state)
        head.addStretch(1)
        self.b_delete = QPushButton(icon("delete-outline"), tr("Delete hole"))
        self.b_delete.clicked.connect(self.delete_hole)
        head.addWidget(self.b_delete)
        v.addLayout(head)
        # collar form, in groups
        self.fields: dict[str, QWidget] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        for gi, (gtitle, names) in enumerate(COLLAR_GROUPS):
            box = QGroupBox(gtitle)
            fl = QFormLayout(box)
            fl.setLabelAlignment(Qt.AlignRight)
            for n in names:
                c = COLLAR.col(n)
                w = self._field(c)
                w.setMinimumWidth(170 if c.kind in ("code", "text", "date") else 120)
                self.fields[n] = w
                label = QLabel(tr(c.label) + (" *" if c.required else ""))
                if c.tip:
                    label.setToolTip(tr(c.tip))
                    w.setToolTip(tr(c.tip))
                fl.addRow(label, w)
            grid.addWidget(box, gi // 3, gi % 3)          # two rows of three groups
        notes_box = QGroupBox(tr("Notes"))
        nl = QVBoxLayout(notes_box)
        self.f_notes = QPlainTextEdit()
        self.f_notes.setMaximumHeight(70)
        self.f_notes.textChanged.connect(self._mark_dirty)
        nl.addWidget(self.f_notes)
        grid.addWidget(notes_box, 1, 2)
        v.addLayout(grid)
        # per-hole tables
        self.sub = QTabWidget()
        self.tables: dict[str, RecordTable] = {}
        for t in TAB_TABLES:
            rt = RecordTable(t, help_text=tr(t.help) if t.help else "")
            rt.edited.connect(self._mark_dirty)
            self.tables[t.name] = rt
            page_w = rt
            if t is RUN:
                page_w = self._runs_page(rt)
            elif t is OCCURRENCE:
                page_w = self._occ_page(rt)
            elif t is FAILURE:
                page_w = self._failure_page(rt)
            self.sub.addTab(page_w, tr(t.title))
        v.addWidget(self.sub, 1)
        # checks + save
        self.issues = QListWidget()
        self.issues.setMaximumHeight(110)
        self.issues.itemActivated.connect(self._goto_issue)
        self.issues.itemClicked.connect(self._goto_issue)
        self.issues.hide()
        v.addWidget(self.issues)
        bar = QHBoxLayout()
        self.b_check = QPushButton(icon("check-decagram-outline"), tr("Check"))
        self.b_check.setToolTip(tr("Run the entry checks without saving: continuity of runs, ranges, codes, "
                                   "depths below the hole's depth"))
        self.b_check.clicked.connect(lambda: self.check(show_ok=True))
        self.b_revert = QPushButton(icon("restore"), tr("Revert"))
        self.b_revert.clicked.connect(lambda: self.load_hole(self.hole_id, force=True))
        self.b_save = QPushButton(icon("content-save-outline"), tr("Save hole"))
        self.b_save.setToolTip(tr("Save the hole: collar and all its tables at once (Ctrl+S)"))
        self.b_save.setDefault(True)
        self.b_save.clicked.connect(self.save)
        bar.addStretch(1)
        bar.addWidget(self.b_check)
        bar.addWidget(self.b_revert)
        bar.addWidget(self.b_save)
        v.addLayout(bar)
        self._enable_hole(False)
        return page

    def _runs_page(self, rt: RecordTable) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        b1 = QPushButton(icon("skip-next-outline"), tr("Next run"))
        b1.setToolTip(tr("Add the next run: From = the last run's To; RPM, flow, fluid ... copied from the last run"))
        b1.clicked.connect(self.next_run)
        b2 = QPushButton(icon("alert-octagon-outline"), tr("Record event ..."))
        b2.setToolTip(tr("Record a cavity, fault zone or other problem met at the bit, with its measures"))
        b2.clicked.connect(self.record_event)
        b3 = QPushButton(icon("close-octagon-outline"), tr("Record failure ..."))
        b3.setToolTip(tr("The hole is abandoned: failure code, depth, severity, measures tried"))
        b3.clicked.connect(self.record_failure)
        for b in (b1, b2, b3):
            bar.addWidget(b)
        bar.addStretch(1)
        self.fill_label = QLabel("")
        self.fill_label.setStyleSheet(f"color: {theme.MUTED};")
        bar.addWidget(self.fill_label)
        v.addLayout(bar)
        v.addWidget(rt, 1)
        return w

    def _occ_page(self, rt: RecordTable) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        b = QPushButton(icon("alert-octagon-outline"), tr("Record event ..."))
        b.clicked.connect(self.record_event)
        bm = QPushButton(icon("format-list-checks"), tr("Measures of the selected row ..."))
        bm.clicked.connect(lambda: self._pick_measures(rt))
        bar.addWidget(b)
        bar.addWidget(bm)
        bar.addStretch(1)
        v.addLayout(bar)
        v.addWidget(rt, 1)
        return w

    def _failure_page(self, rt: RecordTable) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        b = QPushButton(icon("close-octagon-outline"), tr("Record failure ..."))
        b.clicked.connect(self.record_failure)
        bar.addWidget(b)
        bar.addStretch(1)
        v.addLayout(bar)
        v.addWidget(rt, 1)
        return w

    def _ref_page(self, t) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        rt = RecordTable(t, hide=(), help_text=tr(t.help) if t.help else "")
        self.ref_tables[t.name] = rt
        v.addWidget(rt, 1)
        bar = QHBoxLayout()
        if t.name in default_reference():
            r = QPushButton(icon("restore"), tr("Restore defaults ..."))
            r.setToolTip(tr("Put back the predefined rows (your other rows are kept, or replaced - you choose)"))
            r.clicked.connect(lambda _=False, tt=t: self.restore_defaults(tt))
            bar.addWidget(r)
        bar.addStretch(1)
        b = QPushButton(icon("content-save-outline"), tr("Save {table}", table=tr(t.title).lower()))
        b.clicked.connect(lambda _=False, tt=t: self.save_reference(tt))
        bar.addWidget(b)
        v.addLayout(bar)
        return w

    def restore_defaults(self, t) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(tr("Restore defaults"))
        box.setText(tr("Restore the predefined {table}?\n\nAdd missing: the predefined rows that are not in the table "
                       "are added; your rows stay as they are.\nReplace all: the table becomes the predefined rows.",
                       table=tr(t.title).lower()))
        add = box.addButton(tr("Add missing"), QMessageBox.AcceptRole)
        rep = box.addButton(tr("Replace all"), QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() not in (add, rep):
            return
        rt = self.ref_tables[t.name]
        dflt = pd.DataFrame(default_reference()[t.name]).reindex(columns=t.names)
        if box.clickedButton() is add:
            cur = rt.frame()
            key = t.key[0]
            dflt = pd.concat([cur, dflt[~dflt[key].astype(str).isin(set(cur[key].astype(str)))]], ignore_index=True)
        rt.set_frame(dflt)
        self.save_reference(t)

    def _field(self, c: Col) -> QWidget:
        if c.kind in ("code", "yn"):
            w = QComboBox()
            w.setEditable(c.name in ("area_id", "rig_model"))
            if c.choices:
                w.addItem("", None)
                for k, t in c.choices.items():
                    w.addItem(tr(t), k)
            w.currentIndexChanged.connect(self._mark_dirty)
            w.editTextChanged.connect(self._mark_dirty)
            if c.name == "result":
                w.currentIndexChanged.connect(self._result_changed)
            return w
        if c.kind == "date":
            w = QDateEdit()
            w.setCalendarPopup(True)
            w.setDisplayFormat("yyyy-MM-dd")
            w.setMinimumDate(QDate(1900, 1, 1))
            w.setSpecialValueText(" ")
            w.dateChanged.connect(self._mark_dirty)
            return w
        w = QLineEdit()
        if c.kind in ("real", "int"):
            val = QDoubleValidator(w)
            val.setNotation(QDoubleValidator.StandardNotation)
            w.setValidator(val)
            w.setAlignment(Qt.AlignRight)
            if c.unit:
                w.setPlaceholderText(c.unit)
        w.textChanged.connect(self._mark_dirty)
        return w

    # --- loading -----------------------------------------------------------------------------------
    def reload(self) -> None:
        try:
            self.store.prepare()
            data = self.store.load_all()
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.banner.show_message(tr("The project database cannot be read: {e}", e=e), "error")
            return
        self.collars = data.collar
        self.refs = {t.name: data[t.name] for t in REF_TABLES}
        self.failures = data["failure"]
        for t in REF_TABLES:
            self.ref_tables[t.name].set_frame(data[t.name])
        self._choices()
        self._fill_list()
        self._overview(data)
        if not len(self.collars):
            self.banner.show_message(tr("No drilling records yet. Import a folder of CSV files (one per table) or add a "
                                        "hole with New hole."), "info",
                                     tr("Import CSV ..."), self.import_csv)
        else:
            self.banner.hide()
        if self.hole_id and self.hole_id in set(self.collars["hole_id"]):
            self.load_hole(self.hole_id, force=True)
        elif self.list.topLevelItemCount():
            self.list.setCurrentItem(self.list.topLevelItem(0))

    def _choices(self) -> None:
        areas = {str(a): str(n or a) for a, n in zip(self.refs["area"]["area_id"], self.refs["area"]["name"])}
        rigs = {str(r): str(r) for r in self.refs["rig"]["rig_model"]}
        forms = {str(f): str(f) for f in self.refs["formation"]["formation"]}
        for w, ch in ((self.fields["area_id"], areas), (self.fields["rig_model"], rigs)):
            cur = w.currentData() or w.currentText()
            self._loading = True
            w.clear()
            w.addItem("", None)
            for k, t in ch.items():
                w.addItem(f"{k} - {t}" if t != k else k, k)
            i = w.findData(cur)
            w.setCurrentIndex(max(0, i))
            self._loading = False
        self.tables["geology"].set_choices("formation", forms)
        rocks = {str(k): f"{k} - {n}" for k, n in zip(self.refs["rock"]["rock_class"], self.refs["rock"]["name"])}
        for name, col in (("rock_code", "rock_class"), ("formation", "main_rock"), ("formation", "second_rock")):
            self.ref_tables[name].set_choices(col, rocks)
        self.f_area.blockSignals(True)
        cur = self.f_area.currentData()
        self.f_area.clear()
        self.f_area.addItem(tr("All areas"), None)
        for k, t in areas.items():
            self.f_area.addItem(f"{k} - {t}" if t != k else k, k)
        self.f_area.setCurrentIndex(max(0, self.f_area.findData(cur)))
        self.f_area.blockSignals(False)

    def _fill_list(self) -> None:
        c = self.collars
        q = self.search.text().strip().lower()
        area, result = self.f_area.currentData(), self.f_result.currentData()
        m = np.ones(len(c), bool)
        if q:
            m &= c["hole_id"].str.lower().str.contains(q, regex=False).to_numpy()
        if area:
            m &= (c["area_id"] == area).to_numpy()
        if result:
            m &= (c["result"] == result).to_numpy()
        sub = c[m].copy()
        prio = {"in_progress": 0, "planned": 1, "failed": 2, "success": 3}
        sub["_p"] = sub["result"].map(prio).fillna(4)
        sub = sub.sort_values(["_p", "hole_id"])
        self.list.blockSignals(True)
        self.list.setSortingEnabled(False)
        self.list.clear()
        bold = QFont()
        bold.setBold(True)
        for r in sub.itertuples(index=False):
            it = QTreeWidgetItem([r.hole_id, str(r.area_id or ""), tr(codes.RESULTS.get(r.result, r.result or "")),
                                  fmt_value(COLLAR.col("final_depth"), r.final_depth)])
            it.setData(0, Qt.UserRole, r.hole_id)
            col = QColor(codes.RESULT_COLORS.get(r.result, "#888"))
            it.setForeground(2, col)
            it.setFont(2, bold)
            if r.result in ("in_progress", "planned"):
                it.setFont(0, bold)
            self.list.addTopLevelItem(it)
        self.list.blockSignals(False)
        self.list_count.setText(tr("{n} of {total} holes", n=len(sub), total=len(c)))
        n = len(c)
        failed = int((c["result"] == "failed").sum())
        fin = int(c["result"].isin(["success", "failed"]).sum())
        self.card_holes.set(f"{n:,}", None, tr("{a} areas · {p} in progress / planned", a=c["area_id"].nunique(),
                                               p=int(c["result"].isin(["in_progress", "planned"]).sum())))
        self.card_fail.set(pct(failed / fin if fin else None), codes.RESULT_COLORS["failed"],
                           tr("{f} of {n} finished holes", f=failed, n=fin))
        if self.hole_id:
            for i in range(self.list.topLevelItemCount()):
                if self.list.topLevelItem(i).data(0, Qt.UserRole) == self.hole_id:
                    self.list.blockSignals(True)
                    self.list.setCurrentItem(self.list.topLevelItem(i))
                    self.list.blockSignals(False)

    def _hole_chosen(self, cur, _prev) -> None:
        if cur is None:
            return
        hid = cur.data(0, Qt.UserRole)
        if hid != self.hole_id:
            if not self.load_hole(hid):
                self._fill_list()                         # stay on the hole being edited

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, self.windowTitle(),
                                 tr("{hole} has unsaved changes. Save them first?", hole=self.hole_id or tr("The hole")),
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def load_hole(self, hid: str | None, force: bool = False, records: dict | None = None) -> bool:
        if not force and not self._confirm_discard():
            return False
        self._loading = True
        try:
            recs = records if records is not None else (self.store.hole(hid) if hid else None)
            self.hole_id = hid
            col = recs["collar"].iloc[0].to_dict() if recs is not None and len(recs["collar"]) else {}
            for n, w in self.fields.items():
                self._set_field(w, COLLAR.col(n), col.get(n))
            self.f_notes.setPlainText(str(col.get("notes") or ""))
            for t in TAB_TABLES:
                self.tables[t.name].set_frame(recs[t.name] if recs is not None else None)
                self.tables[t.name].set_issues([])
            self._tab_titles()
            self.issues.clear()
            self.issues.hide()
            self._header(col)
            self._fill_rate()
        finally:
            self._loading = False
        self.dirty = False
        self._enable_hole(hid is not None)
        self._state_text()
        return True

    def _set_field(self, w, c: Col, v) -> None:
        if isinstance(w, QComboBox):
            key = None if is_blank(v) else str(v)
            i = w.findData(key)
            if i < 0 and key is not None:
                w.addItem(key, key)
                i = w.count() - 1
            w.setCurrentIndex(max(0, i))
        elif isinstance(w, QDateEdit):
            d = QDate.fromString(str(v or ""), "yyyy-MM-dd")
            w.setDate(d if d.isValid() else w.minimumDate())
        else:
            w.setText(fmt_value(c, v))

    def _get_field(self, w, c: Col):
        if isinstance(w, QComboBox):
            d = w.currentData()
            if d is None and w.isEditable() and w.currentText().strip():
                return w.currentText().strip().split(" - ")[0]
            return d
        if isinstance(w, QDateEdit):
            return None if w.date() == w.minimumDate() else w.date().toString("yyyy-MM-dd")
        t = w.text().strip()
        if not t:
            return None
        if c.kind in ("real", "int"):
            try:
                return float(t.replace(",", "."))
            except ValueError:
                return t
        return t

    def _header(self, col: dict) -> None:
        if not col:
            self.h_title.setText(tr("No hole selected"))
            self.h_chip.set("", "#000")
            return
        self.h_title.setText(str(col.get("hole_id")))
        res = col.get("result") or ""
        self.h_chip.set(tr(codes.RESULTS.get(res, res)), codes.RESULT_COLORS.get(res, "#888"))

    def _tab_titles(self) -> None:
        for i, t in enumerate(TAB_TABLES):
            n = self.tables[t.name].model_.rowCount()
            self.sub.setTabText(i, f"{tr(t.title)} ({n})" if n else tr(t.title))

    def _enable_hole(self, on: bool) -> None:
        for w in (self.b_save, self.b_revert, self.b_check, self.b_delete, self.sub):
            w.setEnabled(on)
        for w in self.fields.values():
            w.setEnabled(on)

    def _mark_dirty(self, *_):
        if self._loading or self.hole_id is None:
            return
        self.dirty = True
        self._state_text()
        self._tab_titles()

    def _state_text(self) -> None:
        if self.hole_id is None:
            self.h_state.setText("")
        elif self.dirty:
            self.h_state.setText("● " + tr("Unsaved changes"))
            self.h_state.setStyleSheet("color: #e3b341;")
        else:
            self.h_state.setText("✓ " + tr("Saved"))
            self.h_state.setStyleSheet(f"color: {theme.MUTED};")

    def _result_changed(self, *_):
        if self._loading:
            return
        res = self.fields["result"].currentData()
        if res:
            self.h_chip.set(tr(codes.RESULTS[res]), codes.RESULT_COLORS.get(res, "#888"))
        if res == "failed" and not len(self.tables["failure"].frame()):
            self.banner.show_message(tr("A failed hole needs its failure: code, depth, severity."), "warn",
                                     tr("Record failure ..."), self.record_failure)

    def _fill_rate(self) -> None:
        df = self.tables["run"].frame()
        if not len(df):
            self.fill_label.setText("")
            return
        parts = []
        for c in ("rpm", "rop", "flow_in", "pump_press", "return_pct", "rod_drop"):
            parts.append(f"{tr(RUN.col(c).label)} {100 * df[c].notna().mean():.0f}%")
        self.fill_label.setText(tr("Filled: ") + " · ".join(parts))

    # --- collecting / checking / saving -------------------------------------------------------------
    def _collar_row(self) -> dict:
        row = {n: self._get_field(w, COLLAR.col(n)) for n, w in self.fields.items()}
        row["notes"] = self.f_notes.toPlainText().strip() or None
        return row

    def check(self, show_ok: bool = False) -> bool:
        """Run the entry checks; mark the cells; True when there is no error."""
        collar = self._collar_row()
        all_issues: list[tuple[str, Issue]] = []
        _bad, iss = validate(COLLAR, pd.DataFrame([collar]))
        all_issues += [("collar", i) for i in iss]
        rig = self.refs.get("rig")
        max_rpm = None
        if rig is not None and collar.get("rig_model") in set(rig["rig_model"]):
            max_rpm = rig.loc[rig["rig_model"] == collar["rig_model"], "max_rpm"].iloc[0]
        ctx = {"final_depth": collar.get("final_depth"), "max_rpm": max_rpm}
        for t in TAB_TABLES:
            df = self.tables[t.name].frame()
            df["hole_id"] = collar.get("hole_id")
            _bad, iss = validate(t, df, ctx)
            self.tables[t.name].set_issues(iss)
            all_issues += [(t.name, i) for i in iss]
        if collar.get("result") == "failed" and not len(self.tables["failure"].frame()):
            all_issues.append(("failure", Issue("error", tr("Failure"), None, None,
                                                tr("The hole is Failed: record its failure (code, depth, severity)"))))
        fr = self.tables["failure"].frame()
        if len(fr) and collar.get("result") not in ("failed",):
            all_issues.append(("collar", Issue("warning", tr("Collar"), None, "result",
                                               tr("A failure is recorded but the result is not Failed"))))
        if collar.get("area_id") and collar["area_id"] not in set(self.refs["area"]["area_id"]):
            all_issues.append(("collar", Issue("warning", tr("Collar"), None, "area_id",
                                               tr("Area {a} has no site sheet yet (Areas tab)", a=collar["area_id"]))))
        geo = self.tables["geology"].frame()
        known = set(self.refs["formation"]["formation"])
        missing = sorted({f for f in geo["formation"].dropna() if f not in known})
        if missing:
            all_issues.append(("geology", Issue("warning", tr("Geology log"), None, "formation",
                                                tr("No formation card for {names}: the models use the rock class only",
                                                   names=", ".join(missing)))))
        self._show_issues(all_issues)
        errors = [i for _t, i in all_issues if i.level == "error"]
        if show_ok and not all_issues:
            self.banner.show_message(tr("All entry checks passed."), "ok")
        return not errors

    def _show_issues(self, issues) -> None:
        self.issues.clear()
        for tname, i in issues:
            it = QListWidgetItem(icon("alert-circle-outline" if i.level == "error" else "alert-outline"), i.text())
            it.setForeground(QColor("#ff8080" if i.level == "error" else "#e3b341"))
            it.setData(Qt.UserRole, (tname, i.row, i.column))
            self.issues.addItem(it)
        self.issues.setVisible(bool(issues))

    def _goto_issue(self, it: QListWidgetItem) -> None:
        tname, row, col = it.data(Qt.UserRole)
        if tname == "collar":
            w = self.fields.get(col)
            if w is not None:
                w.setFocus()
            return
        names = [t.name for t in TAB_TABLES]
        if tname in names:
            self.sub.setCurrentIndex(names.index(tname))
            if row is not None:
                self.tables[tname].select_cell(row, col)

    def save(self) -> bool:
        if self.hole_id is None:
            return False
        if not self.check():
            self.banner.show_message(tr("Not saved: fix the errors marked in red (click an error to go to it)."), "error")
            return False
        collar = self._collar_row()
        frames = {t.name: self.tables[t.name].frame() for t in TAB_TABLES}
        try:
            self.store.save_hole(collar, frames, old_id=self.hole_id)
        except Exception as e:  # noqa: BLE001 - shown
            QMessageBox.critical(self, self.windowTitle(), tr("The hole could not be saved:\n{e}", e=e))
            return False
        self.hole_id = collar["hole_id"]
        self.dirty = False
        self._state_text()
        warns = self.issues.count()
        self.banner.show_message(tr("{hole} saved.", hole=self.hole_id) +
                                 (" " + tr("{n} warnings - see the list below.", n=warns) if warns else ""),
                                 "warn" if warns else "ok")
        self.collars = self.store.read(COLLAR)
        self._fill_list()
        self._header(collar)
        self.changed.emit()
        return True

    def save_reference(self, t) -> None:
        rt = self.ref_tables[t.name]
        df = rt.frame()
        bad, iss = validate(t, df)
        key = t.key[0]
        dup = df[key].dropna().duplicated()
        for i in np.nonzero(dup.to_numpy())[0]:
            iss.append(Issue("error", tr(t.title), int(dup.index[i]), key, tr("{v} is listed twice", v=df[key].iloc[dup.index[i]])))
            bad.add(int(dup.index[i]))
        rt.set_issues(iss)
        if bad:
            self.banner.show_message(tr("{table} not saved: {n} rows have errors (marked in red).",
                                        table=tr(t.title), n=len(bad)), "error")
            return
        try:
            self.store.save_reference(t, df)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, self.windowTitle(), str(e))
            return
        self.refs[t.name] = df
        apply_reference(self.refs)                  # rock properties, rock codes and measures in use at once
        self._choices()
        msg = tr("{table} saved ({n} rows).", table=tr(t.title), n=len(df))
        if t.name in ("rock", "rock_code", "formation"):
            msg += " " + tr("Retrain the models (Risk prediction > Models) so they use the new properties.")
        self.banner.show_message(msg, "ok")
        self.changed.emit()

    # --- driller actions -----------------------------------------------------------------------------
    def _bit_depth(self) -> float:
        runs = self.tables["run"].frame()
        if len(runs) and runs["depth_to"].notna().any():
            return float(runs["depth_to"].max())
        v = self._get_field(self.fields["final_depth"], COLLAR.col("final_depth"))
        return float(v) if isinstance(v, float) else 0.0

    def next_run(self) -> None:
        rt = self.tables["run"]
        df = rt.model_.df
        values = rt.next_row_defaults()
        if len(df):
            last = df.iloc[-1].to_dict()
            length = 3.0
            try:
                length = float(last["depth_to"]) - float(last["depth_from"])
            except (TypeError, ValueError):
                pass
            for c in ("rpm", "flow_in", "fluid_type", "wob"):
                values[c] = last.get(c)
            if values.get("depth_from") is not None:
                values["depth_to"] = round(float(values["depth_from"]) + max(length, 1.5), 2)
        else:
            values = {"depth_from": 0.0, "depth_to": 3.0}
        rt.add_row(values)
        self.sub.setCurrentIndex([t.name for t in TAB_TABLES].index("run"))
        rt.select_cell(rt.model_.rowCount() - 1, "rop")

    def record_event(self) -> None:
        dlg = EventDialog(self, self._bit_depth())
        if dlg.exec():
            rt = self.tables["occurrence"]
            rt.model_.insert_row(None, dlg.row())
            self._mark_dirty()
            self.sub.setCurrentIndex([t.name for t in TAB_TABLES].index("occurrence"))

    def record_failure(self) -> None:
        cur = self.tables["failure"].frame()
        dlg = FailureDialog(self, self._bit_depth(), cur.iloc[0].to_dict() if len(cur) else None)
        if dlg.exec():
            row = dlg.row()
            self.tables["failure"].set_frame(pd.DataFrame([row]))
            self._set_field(self.fields["result"], COLLAR.col("result"), "failed")
            self._set_field(self.fields["final_depth"], COLLAR.col("final_depth"), row["fail_depth"])
            self._mark_dirty()
            self.sub.setCurrentIndex([t.name for t in TAB_TABLES].index("failure"))
            self.banner.show_message(tr("Failure recorded - press Save hole to write it."), "warn")

    def _pick_measures(self, rt: RecordTable) -> None:
        r = rt.view.currentIndex().row()
        if r < 0:
            QMessageBox.information(self, tr("Measures"), tr("Select a row first."))
            return
        cur = rt.model_.df.at[r, "measures"]
        dlg = MeasurePicker(self, cur or "")
        if dlg.exec():
            cols = [c.name for c in rt.model_.cols]
            rt.model_.setData(rt.model_.index(r, cols.index("measures")), dlg.value())

    # --- holes ---------------------------------------------------------------------------------------
    def new_hole(self) -> None:
        if not self._confirm_discard():
            return
        dlg = NewHoleDialog(self, list(self.refs["area"]["area_id"].astype(str)), list(self.collars["hole_id"]),
                            self.current_area)
        if not dlg.exec():
            return
        hid = dlg.hid.text().strip()
        collar = {"hole_id": hid, "area_id": dlg.area.currentText().split(" - ")[0], "result": dlg.state.currentData(),
                  "final_depth": 0.0}
        src = dlg.copy.currentData()
        if src:
            s = self.collars[self.collars["hole_id"] == src].iloc[0].to_dict()
            for k in ("rig_model", "plan_azimuth", "plan_dip", "planned_depth", "casing_available", "fluids_available",
                      "crew", "z"):
                collar[k] = s.get(k)
        recs = {"collar": pd.DataFrame([collar])}
        for t in TAB_TABLES:
            recs[t.name] = pd.DataFrame(columns=t.names)
        self.load_hole(hid, force=True, records=recs)
        self.dirty = True
        self._state_text()
        self.fields["x"].setFocus()
        self.banner.show_message(tr("New hole {hole}: fill in the collar and press Save hole.", hole=hid), "info")

    def delete_hole(self) -> None:
        if not self.hole_id:
            return
        if QMessageBox.question(self, self.windowTitle(),
                                tr("Delete {hole} and all its records (survey, geology, runs, events, failure)?",
                                   hole=self.hole_id)) != QMessageBox.Yes:
            return
        self.store.delete_hole(self.hole_id)
        self.hole_id = None
        self.dirty = False
        self.reload()
        self.changed.emit()

    # --- exchange ------------------------------------------------------------------------------------
    def import_csv(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Folder with the CSV files"),
                                                  str(self.folder or Path.home()))
        if not folder:
            return
        files = sorted(p.name for p in Path(folder).glob("*.csv"))
        if not files:
            QMessageBox.information(self, tr("Import CSV"), tr("No CSV files in this folder."))
            return
        box = QMessageBox(self)
        box.setWindowTitle(tr("Import CSV"))
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
        try:
            counts, issues = self.store.import_csv(folder, replace=box.clickedButton() is replace)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, tr("Import CSV"), tr("The import failed:\n{e}", e=e))
            return
        errs = [i for i in issues if i.level == "error"]
        text = tr("Imported: ") + ", ".join(f"{k} {v}" for k, v in counts.items())
        if errs:
            text += "\n\n" + tr("{n} rows skipped because of errors, e.g.:", n=len(errs)) + "\n" + \
                "\n".join(i.text() for i in errs[:12])
        QMessageBox.information(self, tr("Import CSV"), text)
        self.hole_id = None
        self.dirty = False
        self.reload()
        self.changed.emit()

    def export_csv(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Export to folder"), str(self.folder or Path.home()))
        if not folder:
            return
        files = self.store.export_csv(folder)
        self.banner.show_message(tr("{n} CSV files written to {folder}.", n=len(files), folder=folder), "ok")

    def import_workspace_holes(self) -> None:
        """The drillholes of the project database that have no area yet (entered in the Drillhole Manager): give them
        an area and, where they have none, a result. Their collar, survey and geology are already the records' (the
        drillhole tables are shared); the plan is taken from the first survey station where it is empty."""
        c = self.collars
        ids = sorted(c.loc[c["area_id"].isna() | (c["area_id"].astype(str).str.strip() == ""), "hole_id"].astype(str))
        if not ids:
            QMessageBox.information(self, self.windowTitle(), tr("Every drillhole of the database has an area."))
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Holes without an area"))
        v = QVBoxLayout(dlg)
        lab = QLabel(tr("{n} drillholes have no area: {holes}. Choose the area to give them, and the result for "
                        "those that have no state yet.", n=len(ids),
                        holes=", ".join(ids[:12]) + (" ..." if len(ids) > 12 else "")))
        lab.setWordWrap(True)
        v.addWidget(lab)
        f = QFormLayout()
        area = QComboBox()
        area.setEditable(True)
        area.addItems(list(self.refs["area"]["area_id"].astype(str)))
        if self.current_area:
            area.setCurrentText(self.current_area)
        res = QComboBox()
        for k in ("success", "failed", "in_progress", "planned"):
            res.addItem(tr(codes.RESULTS[k]), k)
        f.addRow(tr("Area"), area)
        f.addRow(tr("Result"), res)
        v.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if not dlg.exec() or not area.currentText().strip():
            return
        n = 0
        for hid in ids:
            recs = self.store.hole(hid)
            col = recs["collar"].iloc[0].to_dict()
            col["area_id"] = area.currentText().strip()
            if is_blank(col.get("result")):
                col["result"] = res.currentData()
            sv = recs["survey"]
            if is_blank(col.get("plan_azimuth")) and len(sv):
                col["plan_azimuth"] = float(sv["azimuth"].iloc[0])
            if is_blank(col.get("plan_dip")) and len(sv):
                col["plan_dip"] = float(sv["dip"].iloc[0])
            if is_blank(col.get("planned_depth")) and not is_blank(col.get("final_depth")):
                col["planned_depth"] = col["final_depth"]
            col.pop("state", None)
            self.store.save_hole(col, {})
            n += 1
        self.banner.show_message(tr("{n} holes given the area {a}.", n=n, a=area.currentText()), "ok")
        self.reload()
        self.changed.emit()

    # --- overview ------------------------------------------------------------------------------------
    def _overview(self, data) -> None:
        c = data.collar
        if not len(c):
            self.overview.setHtml(f"<p style='color:{theme.MUTED}'>{tr('No records yet.')}</p>")
            return
        f = data["failure"].copy()
        f["family"] = f["fail_code"].map(codes.family_of)
        rows = []
        for a, g in c.groupby("area_id"):
            fin = g[g["result"].isin(["success", "failed"])]
            nf = int((fin["result"] == "failed").sum())
            fc = f[f["hole_id"].isin(g["hole_id"])]["fail_code"].value_counts()
            top = ", ".join(f"{k} {v}" for k, v in fc.head(4).items())
            area_name = self.refs["area"].set_index("area_id")["name"].get(a, "") if len(self.refs["area"]) else ""
            rows.append(f"<tr><td><b>{a}</b></td><td>{area_name or ''}</td><td align=right>{len(g)}</td>"
                        f"<td align=right>{len(fin)}</td><td align=right>{nf}</td>"
                        f"<td align=right>{pct(nf / len(fin) if len(fin) else None, 1)}</td><td>{top}</td></tr>")
        fam = f["family"].value_counts()
        bars = "".join(
            f"<tr><td>{tr(codes.FAMILIES[k])}</td><td><div style='background:{codes.FAMILY_COLORS[k]};"
            f"width:{int(300 * v / max(fam.max(), 1))}px'>&nbsp;</div></td><td align=right>{v}</td></tr>"
            for k, v in fam.items())
        occ = data["occurrence"]["type"].value_counts()
        occ_rows = "".join(f"<tr><td>{tr(codes.OCC_TYPES.get(k, k))}</td><td align=right>{v}</td></tr>"
                           for k, v in occ.items())
        runs = data["run"]
        crew = ""
        if len(runs):
            rc = runs.merge(c[["hole_id", "crew"]], on="hole_id", how="left")
            lines = []
            for cr, g in rc.groupby(rc["crew"].fillna("-")):
                vals = " ".join(f"<td align=right>{100 * g[col].notna().mean():.0f}%</td>"
                                for col in ("rpm", "rop", "flow_in", "pump_press", "return_pct", "rod_drop", "torque"))
                lines.append(f"<tr><td>{cr}</td><td align=right>{len(g):,}</td>{vals}</tr>")
            crew = ("<h3>" + tr("Fill rate of the run fields per crew (spec 7.3)") + "</h3><table cellpadding=4>"
                    "<tr><th>" + tr("Crew") + "</th><th>" + tr("Runs") + "</th><th>RPM</th><th>ROP</th><th>"
                    + tr("Flow") + "</th><th>" + tr("Pressure") + "</th><th>" + tr("Return %") + "</th><th>"
                    + tr("Rod drop") + "</th><th>" + tr("Torque") + "</th></tr>" + "".join(lines[:30]) + "</table>")
        self.overview.setHtml(
            f"<h2>{tr('Drilling records')}</h2>"
            f"<p>{tr('{n} holes in {a} areas; {r:,} runs; {o} occurrences; {f} failures.', n=len(c), a=c['area_id'].nunique(), r=len(runs), o=len(data['occurrence']), f=len(f))}</p>"
            f"<h3>{tr('By area')}</h3><table cellpadding=4><tr><th>{tr('Area')}</th><th>{tr('Name')}</th>"
            f"<th>{tr('Holes')}</th><th>{tr('Finished')}</th><th>{tr('Failed')}</th><th>{tr('Failure rate')}</th>"
            f"<th>{tr('Top failure codes')}</th></tr>{''.join(rows)}</table>"
            f"<h3>{tr('Failures by family')}</h3><table cellpadding=3>{bars}</table>"
            f"<h3>{tr('Occurrences')}</h3><table cellpadding=3>{occ_rows}</table>{crew}")

    def closeEvent(self, e) -> None:
        if not self._confirm_discard():
            e.ignore()
            return
        super().closeEvent(e)
