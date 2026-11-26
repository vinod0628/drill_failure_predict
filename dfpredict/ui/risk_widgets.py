"""Widgets shared by the drilling-record and risk-prediction windows.

* ``RecordTable``: an editable table of one schema table (risk/schema.py) - typed editors (numbers,
  drop-down code lists showing "code - text", Yes / No, dates), units in the headers, tooltips, paste
  from Excel, add / insert / duplicate / delete rows, and the entry checks shown on the cells.
* ``chip`` / ``StatCard`` / ``Banner``: small coloured labels and summary tiles.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QDate, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDoubleValidator, QGuiApplication, QIntValidator, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..risk.schema import Col, Table
from . import theme
from .icons import icon

ERROR_BG = QColor(150, 40, 40, 150)
WARN_BG = QColor(150, 110, 20, 120)


def is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip()) or \
        (v is pd.NA)


def fmt_value(c: Col, v) -> str:
    if is_blank(v):
        return ""
    if c.kind in ("real",):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return str(v)
        return f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) < 1e5 else f"{x:.2f}"
    if c.kind == "int":
        try:
            return str(int(float(v)))
        except (TypeError, ValueError):
            return str(v)
    if c.kind in ("code", "yn") and c.choices:
        key = str(v)
        if key in c.choices:
            text = tr(c.choices[key])
            return text if c.kind == "yn" or text.startswith(key) else f"{key} - {text}"
    return str(v)


def _is_code(v) -> bool:
    """A short code worth showing with its text (LC, HQ, 2, M-OP-01); plain words (clastic, success) are not."""
    s = str(v)
    return s.isupper() or s.isdigit() or "-" in s


def header_text(c: Col) -> str:
    return f"{tr(c.label)} ({c.unit})" if c.unit else tr(c.label)


# ------------------------------------------------------------------------------------------------
class FrameModel(QAbstractTableModel):
    """A pandas DataFrame shown / edited by the schema's columns (the hole id column hidden)."""

    edited = Signal()

    def __init__(self, table: Table, df: pd.DataFrame | None = None, hide=("hole_id",)):
        super().__init__()
        self.table = table
        self.cols = [c for c in table.cols if c.name not in hide]
        self.df = pd.DataFrame(columns=table.names)
        self.cell_issues: dict[tuple[int, str], tuple[str, str]] = {}     # (row, col) -> (level, text)
        self.dynamic_choices: dict[str, dict] = {}                        # column -> choices (areas, rigs ...)
        self.set_frame(df)

    def set_frame(self, df: pd.DataFrame | None) -> None:
        self.beginResetModel()
        self.df = (df.copy() if df is not None else pd.DataFrame(columns=self.table.names)).reindex(
            columns=self.table.names).reset_index(drop=True)
        self.df = self.df.astype(object).where(pd.notna(self.df), None)
        self.cell_issues = {}
        self.endResetModel()

    def frame(self) -> pd.DataFrame:
        df = self.df.copy()
        for c in self.table.cols:
            if c.kind in ("real", "int"):
                df[c.name] = pd.to_numeric(df[c.name], errors="coerce")
        blank = df.drop(columns=[c for c in ("hole_id",) if c in df]).map(is_blank).all(axis=1)
        return df[~blank].reset_index(drop=True)

    def choices(self, c: Col) -> dict | None:
        return self.dynamic_choices.get(c.name) or c.choices

    # --- Qt model ---
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.cols)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            c = self.cols[section]
            if role == Qt.DisplayRole:
                return header_text(c) + (" *" if c.required else "")
            if role == Qt.ToolTipRole:
                tip = tr(c.tip) if c.tip else tr(c.label)
                if c.lo is not None or c.hi is not None:
                    tip += "\n" + tr("Allowed: {lo} - {hi} {unit}", lo=c.lo, hi=c.hi, unit=c.unit)
                if c.required:
                    tip += "\n" + tr("Required")
                return tip
        elif role == Qt.DisplayRole:
            return str(section + 1)
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        c = self.cols[index.column()]
        v = self.df.at[index.row(), c.name]
        if role == Qt.DisplayRole:
            if c.kind in ("code", "yn"):
                ch = self.choices(c)
                if ch and not is_blank(v) and str(v) in ch:
                    t = tr(ch[str(v)])
                    return t if c.kind == "yn" or str(t).startswith(str(v)) or t == str(v) or not _is_code(v) \
                        else f"{v} - {t}"
            return fmt_value(c, v)
        if role == Qt.EditRole:
            return "" if is_blank(v) else v
        if role == Qt.TextAlignmentRole and c.kind in ("real", "int"):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.BackgroundRole:
            iss = self.cell_issues.get((index.row(), c.name))
            if iss:
                return QBrush(ERROR_BG if iss[0] == "error" else WARN_BG)
        if role == Qt.ToolTipRole:
            iss = self.cell_issues.get((index.row(), c.name))
            if iss:
                return iss[1]
            return tr(c.tip) if c.tip else None
        return None

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        c = self.cols[index.column()]
        self.df.at[index.row(), c.name] = self._parse(c, value)
        self.cell_issues.pop((index.row(), c.name), None)
        self.dataChanged.emit(index, index)
        self.edited.emit()
        return True

    @staticmethod
    def _parse(c: Col, value):
        if is_blank(value):
            return None
        if c.kind in ("real", "int"):
            try:
                return float(str(value).replace(",", "."))
            except ValueError:
                return str(value)
        return str(value).strip()

    # --- rows ---
    def insert_row(self, at: int | None = None, values: dict | None = None) -> int:
        at = len(self.df) if at is None else max(0, min(at, len(self.df)))
        self.beginInsertRows(QModelIndex(), at, at)
        row = {n: None for n in self.table.names}
        row.update(values or {})
        top = self.df.iloc[:at]
        bottom = self.df.iloc[at:]
        self.df = pd.concat([top, pd.DataFrame([row]).astype(object), bottom], ignore_index=True)
        self.df = self.df.astype(object).where(pd.notna(self.df), None)
        self.cell_issues = {}
        self.endInsertRows()
        self.edited.emit()
        return at

    def remove_rows(self, rows: list[int]) -> None:
        if not rows:
            return
        self.beginResetModel()
        self.df = self.df.drop(index=[r for r in rows if r < len(self.df)]).reset_index(drop=True)
        self.cell_issues = {}
        self.endResetModel()
        self.edited.emit()

    def set_issues(self, issues) -> None:
        """Show entry-check issues (risk.store.Issue) on their cells."""
        self.cell_issues = {}
        for i in issues:
            if i.row is not None and i.column:
                key = (i.row, i.column)
                if key not in self.cell_issues or i.level == "error":
                    self.cell_issues[key] = (i.level, i.message)
        self.layoutChanged.emit()


class _Delegate(QStyledItemDelegate):
    def __init__(self, view: "RecordTable"):
        super().__init__(view)
        self.view = view

    def createEditor(self, parent, option, index):
        model: FrameModel = self.view.model_
        c = model.cols[index.column()]
        ch = model.choices(c)
        if c.kind in ("code", "yn") and ch is not None:
            box = QComboBox(parent)
            box.addItem("", None)
            for k, t in ch.items():
                text = tr(t)
                box.addItem(text if c.kind == "yn" or text.startswith(str(k)) or text == str(k) else f"{k} - {text}", k)
            return box
        if c.kind == "date":
            ed = QDateEdit(parent)
            ed.setCalendarPopup(True)
            ed.setDisplayFormat("yyyy-MM-dd")
            return ed
        ed = QLineEdit(parent)
        if c.kind == "real":
            v = QDoubleValidator(ed)
            v.setNotation(QDoubleValidator.StandardNotation)
            ed.setValidator(v)
        elif c.kind == "int":
            ed.setValidator(QIntValidator(ed))
        if c.unit:
            ed.setPlaceholderText(c.unit)
        return ed

    def setEditorData(self, editor, index):
        v = index.data(Qt.EditRole)
        if isinstance(editor, QComboBox):
            i = editor.findData(None if is_blank(v) else str(v))
            if i < 0 and not is_blank(v):
                editor.addItem(str(v), str(v))
                i = editor.count() - 1
            editor.setCurrentIndex(max(0, i))
        elif isinstance(editor, QDateEdit):
            d = QDate.fromString(str(v or ""), "yyyy-MM-dd")
            editor.setDate(d if d.isValid() else QDate.currentDate())
        else:
            editor.setText("" if is_blank(v) else fmt_value(self.view.model_.cols[index.column()], v))

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentData())
        elif isinstance(editor, QDateEdit):
            model.setData(index, editor.date().toString("yyyy-MM-dd"))
        else:
            model.setData(index, editor.text())


class RecordTable(QWidget):
    """A schema table with a small toolbar (add / insert / duplicate / delete / paste)."""

    edited = Signal()

    def __init__(self, table: Table, parent=None, hide=("hole_id",), toolbar: bool = True, help_text: str = ""):
        super().__init__(parent)
        self.table = table
        self.model_ = FrameModel(table, hide=hide)
        self.model_.edited.connect(self.edited)
        self.view = QTableView(self)
        self.view.setModel(self.model_)
        self.view.setItemDelegate(_Delegate(self))
        self.view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.view.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed |
                                  QAbstractItemView.AnyKeyPressed | QAbstractItemView.SelectedClicked)
        self.view.setAlternatingRowColors(True)
        self.view.verticalHeader().setDefaultSectionSize(24)
        h = self.view.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Interactive)
        h.setStretchLastSection(True)
        fm = self.view.fontMetrics()
        for i, c in enumerate(self.model_.cols):
            head = fm.horizontalAdvance(header_text(c) + " *") + 26
            base = c.width or (150 if c.kind in ("code",) else (220 if c.kind == "note" else 80))
            self.view.setColumnWidth(i, max(base, min(head, 260)))
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._menu)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        if toolbar:
            bar = QHBoxLayout()
            bar.setSpacing(4)
            self.b_add = self._button("table-row-plus-after", tr("Add row"), tr("Add a row at the end (Ctrl+N)"),
                                      lambda: self.add_row())
            self.b_ins = self._button("table-row-plus-before", "", tr("Insert a row above the selected one"),
                                      self.insert_row)
            self.b_dup = self._button("content-copy", "", tr("Duplicate the selected row"), self.duplicate_row)
            self.b_del = self._button("table-row-remove", "", tr("Delete the selected rows (Del)"), self.delete_rows)
            self.b_paste = self._button("content-paste", tr("Paste"), tr("Paste rows copied from Excel (Ctrl+V): "
                                                                         "the columns in the order shown"), self.paste)
            for b in (self.b_add, self.b_ins, self.b_dup, self.b_del, self.b_paste):
                bar.addWidget(b)
            bar.addStretch(1)
            self.count = QLabel("")
            self.count.setStyleSheet(f"color: {theme.MUTED};")
            bar.addWidget(self.count)
            lay.addLayout(bar)
            self.model_.modelReset.connect(self._count)
            self.model_.rowsInserted.connect(self._count)
        if help_text:
            hl = QLabel(help_text)
            hl.setWordWrap(True)
            hl.setStyleSheet(f"color: {theme.MUTED};")
            lay.addWidget(hl)
        lay.addWidget(self.view, 1)
        self._count()

    def _button(self, ic, text, tip, slot) -> QToolButton:
        b = QToolButton(self)
        b.setIcon(icon(ic))
        if text:
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        return b

    def _count(self, *_):
        if hasattr(self, "count"):
            self.count.setText(tr("{n} rows", n=self.model_.rowCount()))

    # --- data ---
    def set_frame(self, df: pd.DataFrame | None) -> None:
        self.model_.set_frame(df)
        self._count()

    def frame(self) -> pd.DataFrame:
        return self.model_.frame()

    def set_choices(self, column: str, choices: dict) -> None:
        self.model_.dynamic_choices[column] = choices

    def set_issues(self, issues) -> None:
        self.model_.set_issues(issues)

    def select_cell(self, row: int, column: str | None) -> None:
        cols = [c.name for c in self.model_.cols]
        ci = cols.index(column) if column in cols else 0
        idx = self.model_.index(row, ci)
        self.view.setCurrentIndex(idx)
        self.view.scrollTo(idx)
        self.view.setFocus()

    # --- rows ---
    def add_row(self, values: dict | None = None) -> int:
        r = self.model_.insert_row(None, values or self.next_row_defaults())
        self._count()
        self.select_cell(r, self.model_.cols[0].name)
        return r

    def next_row_defaults(self) -> dict:
        """A new interval row starts where the last one ended (spec 7.3: continuity)."""
        df = self.model_.df
        if self.table.interval and len(df):
            to = df["depth_to"].dropna()
            if len(to):
                return {"depth_from": float(to.iloc[-1])}
        return {}

    def insert_row(self) -> None:
        r = self.view.currentIndex().row()
        self.model_.insert_row(r if r >= 0 else 0)
        self._count()

    def duplicate_row(self) -> None:
        r = self.view.currentIndex().row()
        if r < 0:
            return
        vals = self.model_.df.iloc[r].to_dict()
        self.model_.insert_row(r + 1, vals)
        self._count()

    def delete_rows(self) -> None:
        rows = sorted({i.row() for i in self.view.selectionModel().selectedIndexes()})
        self.model_.remove_rows(rows)
        self._count()

    def paste(self) -> None:
        text = QGuiApplication.clipboard().text()
        if not text.strip():
            return
        start_r = max(0, self.view.currentIndex().row())
        start_c = max(0, self.view.currentIndex().column())
        lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
        cols = self.model_.cols
        for k, ln in enumerate(lines):
            cells = ln.split("\t") if "\t" in ln else ln.split(",")
            r = start_r + k
            if r >= self.model_.rowCount():
                self.model_.insert_row(None)
            for j, cell in enumerate(cells):
                ci = start_c + j
                if ci < len(cols):
                    c = cols[ci]
                    v = cell.strip()
                    ch = self.model_.choices(c)
                    if ch and v and v not in ch:                  # a text instead of a code
                        v = next((k2 for k2, t in ch.items() if v.lower() in (str(t).lower(), tr(t).lower())), v)
                    self.model_.setData(self.model_.index(r, ci), v)
        self._count()

    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Paste):
            self.paste()
        elif e.key() == Qt.Key_Delete and not self.view.state() == QAbstractItemView.EditingState:
            for i in self.view.selectionModel().selectedIndexes():
                self.model_.setData(i, None)
        elif e.matches(QKeySequence.New):
            self.add_row()
        else:
            super().keyPressEvent(e)

    def _menu(self, pos) -> None:
        m = QMenu(self)
        m.addAction(icon("table-row-plus-after"), tr("Add row"), lambda: self.add_row())
        m.addAction(icon("table-row-plus-before"), tr("Insert row above"), self.insert_row)
        m.addAction(icon("content-copy"), tr("Duplicate row"), self.duplicate_row)
        m.addAction(icon("content-paste"), tr("Paste from Excel"), self.paste)
        m.addSeparator()
        m.addAction(icon("table-row-remove"), tr("Delete rows"), self.delete_rows)
        m.exec(self.view.viewport().mapToGlobal(pos))


# ------------------------------------------------------------------------------------------------
def chip(text: str, color: str, fg: str = "#101214") -> str:
    """HTML for a small coloured label."""
    return (f"<span style='background:{color}; color:{fg}; border-radius:4px; padding:1px 6px; "
            f"font-weight:600;'>&nbsp;{text}&nbsp;</span>")


class ChipLabel(QLabel):
    def __init__(self, text: str = "", color: str = "#555", parent=None):
        super().__init__(parent)
        self.set(text, color)

    def set(self, text: str, color: str, fg: str = "#101214") -> None:
        self.setText(text)
        self.setStyleSheet(f"background:{color}; color:{fg}; border-radius:5px; padding:2px 8px; font-weight:600;")
        self.setVisible(bool(text))


class StatCard(QFrame):
    """A summary tile: a big value, a caption and an optional detail line."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.setObjectName("statCard")
        self.setStyleSheet(f"QFrame#statCard {{ background: {theme.PANEL_ALT}; border: 1px solid {theme.BORDER}; "
                           "border-radius: 8px; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(1)
        self.value = QLabel("-")
        self.value.setStyleSheet("font-size: 20pt; font-weight: 700;")
        self.caption = QLabel(caption)
        self.caption.setStyleSheet(f"color: {theme.MUTED};")
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt;")
        v.addWidget(self.value)
        v.addWidget(self.caption)
        v.addWidget(self.detail)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setMinimumHeight(92)

    def set(self, value: str, color: str | None = None, detail: str = "") -> None:
        self.value.setText(value)
        self.value.setStyleSheet(f"font-size: 20pt; font-weight: 700; color: {color or theme.TEXT};")
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))


class Banner(QFrame):
    """A one-line hint / warning bar with an icon and an optional action button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 6, 10, 6)
        self.icon = QLabel()
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.button = QPushButton()
        self.button.hide()
        h.addWidget(self.icon)
        h.addWidget(self.text, 1)
        h.addWidget(self.button)
        self.hide()

    def show_message(self, text: str, kind: str = "info", action: str = "", slot=None) -> None:
        dark = {"info": "#1f3b5c", "warn": "#5c4a1f", "error": "#5c1f24", "ok": "#1f4d2e"}
        light = {"info": "#dbeafe", "warn": "#fdf0c8", "error": "#fde2e2", "ok": "#dcf5e3"}
        icons = {"info": "information-outline", "warn": "alert-outline", "error": "alert-circle-outline",
                 "ok": "check-circle-outline"}
        kind = kind if kind in icons else "info"
        bg = (light if getattr(theme, "CURRENT", "dark") == "light" else dark)[kind]
        ic = icons[kind]
        self.setStyleSheet(f"background: {bg}; color: {theme.TEXT}; border-radius: 6px;")
        self.icon.setPixmap(icon(ic).pixmap(18, 18))
        self.text.setText(text)
        try:
            self.button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        if action and slot is not None:
            self.button.setText(action)
            self.button.clicked.connect(slot)
            self.button.show()
        else:
            self.button.hide()
        self.show()


def pct(v: float | None, digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{100 * v:.{digits}f}%"


# --- background jobs ------------------------------------------------------------------------------
from PySide6.QtCore import QObject, QThread, Slot  # noqa: E402


class _Job(QObject):
    """Runs ``fn(progress)`` in a worker thread; emits the result / error / progress."""

    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self):
        import logging
        import traceback

        try:
            r = self.fn(lambda p, t="": self.progress.emit(int(p), str(t)))
        except Exception as e:  # noqa: BLE001 - shown to the user
            logging.getLogger(__name__).error("job failed: %s", traceback.format_exc())
            self.failed.emit(str(e) or type(e).__name__)
            return
        self.done.emit(r)


class _Relay(QObject):
    """Lives in the GUI thread: the job's signals reach the callbacks through its slots, queued into the
    GUI thread (a plain Python callable connected to a signal would run in the worker thread)."""

    def __init__(self, parent, on_done, on_fail, on_progress):
        super().__init__(parent)
        self._done, self._fail, self._progress = on_done, on_fail, on_progress

    @Slot(object)
    def done(self, r):
        self._done(r)

    @Slot(str)
    def failed(self, msg):
        if self._fail is not None:
            self._fail(msg)

    @Slot(int, str)
    def progress(self, p, t):
        if self._progress is not None:
            self._progress(p, t)


def run_job(owner, fn, on_done, on_fail=None, on_progress=None):
    """Run ``fn(progress)`` in a worker thread; the callbacks run in the GUI thread."""
    from PySide6.QtCore import Qt

    th = QThread(owner)
    job = _Job(fn)
    relay = _Relay(owner, on_done, on_fail, on_progress)      # made here, in the GUI thread
    job.moveToThread(th)
    th.started.connect(job.run)
    job.done.connect(relay.done, Qt.QueuedConnection)
    job.failed.connect(relay.failed, Qt.QueuedConnection)
    job.progress.connect(relay.progress, Qt.QueuedConnection)
    for sig in (job.done, job.failed):
        sig.connect(th.quit)
    th.finished.connect(job.deleteLater)
    th.finished.connect(th.deleteLater)
    th.finished.connect(relay.deleteLater)
    jobs = owner.__dict__.setdefault("_risk_jobs", [])
    entry = (th, job, relay)
    jobs.append(entry)
    th.finished.connect(lambda: jobs.remove(entry) if entry in jobs else None)
    th.start()
    return th
