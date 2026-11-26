"""Section line points editor: shown when the line is finished in the view.

Lists the clicked points with their two in-plane coordinates (X / Y for a line
drawn in plan, X / Z in the XZ view, Y / Z in the YZ view). Coordinates can be
typed in, points added (after the selected one) or removed, and the order
changed. The line in the view follows every edit. "Draw section" confirms and
opens the cross-section.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..i18n import tr
from ..scene.section import AXIS_NAMES, END_MARK, START_MARK, duplicate_rows, remove_duplicate_points
from . import theme
from .icons import icon

MIN_POINTS = 2


class _CoordDelegate(QStyledItemDelegate):
    """Cell editor for a coordinate: a plain line edit accepting numbers.

    (A QDoubleSpinBox without buttons is laid out with a 1 px wide text field inside a
    table cell under the app style sheet, so the typed value was invisible.)"""

    def createEditor(self, parent, option, index):
        ed = QLineEdit(parent)
        v = QDoubleValidator(-1e9, 1e9, 3, ed)
        v.setNotation(QDoubleValidator.StandardNotation)
        v.setLocale(QLocale.c())                     # '.' as the decimal point
        ed.setValidator(v)
        ed.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ed.setFrame(False)
        # Compact: the app style sheet's padding / min-height are for forms, not table cells.
        ed.setStyleSheet(f"QLineEdit {{ padding: 0px 4px; min-height: 0px; border: 1px solid {theme.ACCENT};"
                         " border-radius: 0px; }")
        return ed

    def setEditorData(self, editor, index):
        text = str(index.data(Qt.EditRole) or "").replace(",", "")
        editor.setText(text)
        editor.selectAll()

    def setModelData(self, editor, model, index):
        text = editor.text().strip().replace(",", "")
        try:
            value = float(text)
        except ValueError:
            return                                   # keep the old value
        model.setData(index, f"{value:.3f}", Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class SectionPointsDialog(QDialog):
    """Edit the 2D (in-plane) coordinates of the section line's points."""

    points_changed = Signal(object)    # (k, 2) array: preview in the view while editing

    def __init__(self, parent, points_2d: np.ndarray, axes: tuple[int, int], legend: bool = True,
                 has_drillholes: bool = True):
        super().__init__(parent)
        self.axes = axes
        self.setWindowTitle(tr("Section line points"))
        self.setWindowIcon(icon("vector-polyline"))
        self.setMinimumSize(520, 380)
        self._updating = False
        self.removed_duplicates = 0

        v = QVBoxLayout(self)
        hint = QLabel(tr("Check or edit the points of the section line, then Draw section.\n"
                         "Double-click a coordinate to type a new value. A is the first point, A' the last."))
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.table = QTableWidget(0, 5, self)
        n1, n2 = tr(AXIS_NAMES[axes[0]]), tr(AXIS_NAMES[axes[1]])
        self.table.setHorizontalHeaderLabels([tr("Point"), n1, n2, tr("Segment m"), tr("Distance m")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.AnyKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        delegate = _CoordDelegate(self.table)
        self.table.setItemDelegateForColumn(1, delegate)
        self.table.setItemDelegateForColumn(2, delegate)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.b_add = self._btn("plus", tr("Add point"), tr("Insert a point after the selected one\n"
                               "(halfway to the next point, or beyond the last)"), self.add_point)
        self.b_remove = self._btn("minus", tr("Remove"), tr("Remove the selected point"), self.remove_point)
        self.b_up = self._btn("arrow-up", "", tr("Move the selected point up"), lambda: self.move_point(-1))
        self.b_down = self._btn("arrow-down", "", tr("Move the selected point down"), lambda: self.move_point(1))
        self.b_reverse = self._btn("swap-horizontal", tr("Reverse"), tr("Swap A and A' (reverse the point order)"),
                                   self.reverse)
        for b in (self.b_add, self.b_remove, self.b_up, self.b_down, self.b_reverse):
            row.addWidget(b)
        row.addStretch(1)
        self.total = QLabel("")
        row.addWidget(self.total)
        v.addLayout(row)
        self.dup_note = QLabel("")
        self.dup_note.setStyleSheet(f"color: {theme.WARN if hasattr(theme, 'WARN') else '#e0a030'};")
        self.dup_note.setVisible(False)
        v.addWidget(self.dup_note)

        self.legend_box = QCheckBox(tr("Show the drillhole rock legend on the exported map"))
        self.legend_box.setToolTip(tr("The colour key of the drillhole intervals (e.g. rock codes) in the\n"
                                      "exported section page. It is always shown in the section window."))
        self.legend_box.setChecked(legend)
        self.legend_box.setEnabled(has_drillholes)
        v.addWidget(self.legend_box)

        buttons = QDialogButtonBox(self)
        self.b_draw = buttons.addButton(tr("Draw section"), QDialogButtonBox.AcceptRole)
        self.b_draw.setIcon(icon("box-cutter"))
        self.b_draw.setDefault(True)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

        self.set_points(points_2d)
        if self.table.rowCount():
            self.table.selectRow(self.table.rowCount() - 1)

    @property
    def show_legend(self) -> bool:
        return self.legend_box.isChecked()

    def _btn(self, name, text, tip, slot) -> QPushButton:
        b = QPushButton(icon(name), text, self)
        b.setToolTip(tip)
        b.setAutoDefault(False)
        b.clicked.connect(slot)
        return b

    # --- data -------------------------------------------------------------------------
    def points(self) -> np.ndarray:
        out = []
        for r in range(self.table.rowCount()):
            out.append([float(self.table.item(r, c).text().replace(",", "")) for c in (1, 2)])
        return np.array(out, dtype=np.float64).reshape(-1, 2)

    def set_points(self, pts) -> None:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        self._updating = True
        try:
            self.table.setRowCount(len(pts))
            for r, (a, b) in enumerate(pts):
                for c, val in ((1, a), (2, b)):
                    self.table.setItem(r, c, QTableWidgetItem(f"{val:.3f}"))
                    self.table.item(r, c).setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                for c in (0, 3, 4):
                    it = QTableWidgetItem("")
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    it.setTextAlignment((Qt.AlignCenter if c == 0 else Qt.AlignRight) | Qt.AlignVCenter)
                    self.table.setItem(r, c, it)
        finally:
            self._updating = False
        self._refresh_derived()

    def _refresh_derived(self) -> None:
        pts = self.points()
        n = len(pts)
        seg = np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)] if n else np.zeros(0)
        dist = np.cumsum(seg)
        self._updating = True
        try:
            for r in range(n):
                name = START_MARK if r == 0 else (END_MARK if r == n - 1 else str(r))
                self.table.item(r, 0).setText(name)
                self.table.item(r, 3).setText("" if r == 0 else f"{seg[r]:,.2f}")
                self.table.item(r, 4).setText(f"{dist[r]:,.2f}")
        finally:
            self._updating = False
        self.total.setText(tr("{n} points    length {length:,.2f} m", n=n, length=dist[-1] if n else 0.0))
        dups = duplicate_rows(pts)
        if dups:
            names = ", ".join(self.table.item(r, 0).text() for r in dups)
            self.dup_note.setText(tr("Duplicate point ({names}: same as the point before) - "
                                     "removed when the section is drawn", names=names))
        self.dup_note.setVisible(bool(dups))
        self._update_buttons()
        self.points_changed.emit(pts)

    def _on_item_changed(self, item) -> None:
        if self._updating or item.column() not in (1, 2):
            return
        try:
            float(item.text().replace(",", ""))
        except ValueError:
            self._updating = True
            item.setText("0.000")
            self._updating = False
        self._refresh_derived()

    def _selected(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else self.table.rowCount() - 1

    def _update_buttons(self) -> None:
        n = self.table.rowCount()
        r = self._selected()
        self.b_remove.setEnabled(n > MIN_POINTS)
        self.b_up.setEnabled(n > 1 and r > 0)
        self.b_down.setEnabled(n > 1 and 0 <= r < n - 1)
        self.b_reverse.setEnabled(n > 1)
        self.b_draw.setEnabled(self.valid())

    def valid(self) -> bool:
        pts, _removed = remove_duplicate_points(self.points())
        return len(pts) >= MIN_POINTS

    # --- editing ----------------------------------------------------------------------
    def add_point(self) -> None:
        pts = self.points()
        r = self._selected()
        if not len(pts):
            return
        if r < len(pts) - 1:
            new = (pts[r] + pts[r + 1]) / 2.0
        elif len(pts) > 1:
            d = pts[-1] - pts[-2]
            new = pts[-1] + d * 0.5
        else:
            new = pts[-1] + 10.0
        pts = np.insert(pts, r + 1, new, axis=0)
        self.set_points(pts)
        self.table.selectRow(r + 1)

    def remove_point(self) -> None:
        pts = self.points()
        if len(pts) <= MIN_POINTS:
            return
        r = self._selected()
        self.set_points(np.delete(pts, r, axis=0))
        self.table.selectRow(min(r, len(pts) - 2))

    def move_point(self, step: int) -> None:
        pts = self.points()
        r = self._selected()
        k = r + step
        if not (0 <= r < len(pts) and 0 <= k < len(pts)):
            return
        pts[[r, k]] = pts[[k, r]]
        self.set_points(pts)
        self.table.selectRow(k)

    def reverse(self) -> None:
        r = self._selected()
        pts = self.points()[::-1]
        self.set_points(pts)
        self.table.selectRow(len(pts) - 1 - r)

    def accept(self) -> None:
        if self.table.state() == QAbstractItemView.EditingState:   # commit a value being typed
            self.table.setFocus()
        if not self.valid():
            return
        pts, removed = remove_duplicate_points(self.points())
        if removed:                       # duplicates go before the section is drawn
            self.set_points(pts)
        self.removed_duplicates = removed
        super().accept()
