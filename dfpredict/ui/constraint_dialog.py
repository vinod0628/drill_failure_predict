"""Block model constraint editor: a few conditions combined with and / or.

Each row is one condition, named by its letter:
  a  [DTM        ] [ore1 (solid)] [inside]
  b  [Block value] [sg          ] [>] [2.0]
Combine: all conditions (and) / any condition (or) / custom expression of the letters.
The number of blocks kept is shown as the constraint is edited.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import N_, tr
from ..scene.constraints import OP_TEXT, OPS, combine, letter, parse_logic
from ..scene.layers import SurfaceLayer
from ..scene.solid import dtm_kind
from .icons import icon

SOLID_MODES = ((N_("inside"), "inside"), (N_("outside"), "outside"))       # (shown with tr(), stored value)
SURFACE_MODES = ((N_("above"), "above"), (N_("below"), "below"))
DTM_KIND_TEXT = {"solid": N_("solid"), "surface": N_("surface")}


class _Row(QWidget):
    """One condition."""

    def __init__(self, dialog: "ConstraintDialog", cond: dict):
        super().__init__()
        self.dialog = dialog
        self.layer = dialog.layer
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.tag = QLabel("a")
        self.tag.setFixedWidth(16)
        self.tag.setStyleSheet("font-weight: 600;")
        h.addWidget(self.tag)
        self.kind = QComboBox()
        self.kind.addItem("DTM", "dtm")
        self.kind.addItem(tr("Block value"), "value")
        h.addWidget(self.kind)
        # DTM fields
        self.dtm = QComboBox()
        self.mode = QComboBox()
        # value fields
        self.attr = QComboBox()
        self.op = QComboBox()
        for o in OPS:
            self.op.addItem(OP_TEXT[o], o)
        self.number = QDoubleSpinBox()
        self.number.setDecimals(4)
        self.number.setRange(-1e12, 1e12)
        self.number.setKeyboardTracking(False)
        self.code = QComboBox()
        for w in (self.dtm, self.mode, self.attr, self.op, self.number, self.code):
            h.addWidget(w)
        h.addStretch(1)
        rm = QToolButton()
        rm.setIcon(icon("close"))
        rm.setToolTip(tr("Remove this condition"))
        rm.clicked.connect(lambda: dialog.remove_row(self))
        h.addWidget(rm)

        for name, kind in dialog.dtms:
            self.dtm.addItem(f"{name} ({tr(DTM_KIND_TEXT[kind]) if kind in DTM_KIND_TEXT else kind})", name)
        bm = self.layer.data
        for n in bm.attribute_names:
            self.attr.addItem(n, n)
        self.set_condition(cond)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.dtm.currentIndexChanged.connect(self._dtm_changed)
        self.attr.currentIndexChanged.connect(self._attr_changed)
        for w in (self.mode, self.op, self.code):
            w.currentIndexChanged.connect(dialog.changed)
        self.number.valueChanged.connect(dialog.changed)

    # --- fields ---------------------------------------------------------------------------
    def set_condition(self, cond: dict) -> None:
        kind = cond.get("kind", "dtm" if self.dialog.dtms else "value")
        self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
        if kind == "dtm":
            i = self.dtm.findData(cond.get("layer"))
            if i < 0 and cond.get("layer"):                         # saved, but not loaded now
                self.dtm.addItem(tr("{layer} (not loaded)", layer=cond["layer"]), cond["layer"])
                i = self.dtm.count() - 1
            self.dtm.setCurrentIndex(max(0, i))
            self._fill_modes(cond.get("mode"))
        else:
            self.attr.setCurrentIndex(max(0, self.attr.findData(cond.get("name"))))
            self._fill_value(cond.get("op", ">"), cond.get("value"))
        self._show_fields()

    def _dtm_kind(self) -> str | None:
        name = self.dtm.currentData()
        return dict(self.dialog.dtms).get(name)

    def _fill_modes(self, current=None) -> None:
        kind = self._dtm_kind()
        modes = SURFACE_MODES if kind == "surface" else SOLID_MODES
        self.mode.blockSignals(True)
        self.mode.clear()
        for text, value in modes:
            self.mode.addItem(tr(text), value)
        i = self.mode.findData(current)
        self.mode.setCurrentIndex(i if i >= 0 else (1 if kind == "surface" else 0))   # default below / inside
        self.mode.blockSignals(False)

    def _fill_value(self, op=">", value=None) -> None:
        bm = self.layer.data
        name = self.attr.currentData()
        cat = bool(name) and bm.is_categorical(name)
        self.op.blockSignals(True)
        self.op.clear()
        for o in (("=", "!=") if cat else OPS):
            self.op.addItem(OP_TEXT[o], o)
        self.op.setCurrentIndex(max(0, self.op.findData(op)))
        self.op.blockSignals(False)
        self.code.blockSignals(True)
        self.code.clear()
        if cat:
            self.code.addItems(list(bm.categories.get(name, [])))
            if value is not None and self.code.findText(str(value)) >= 0:
                self.code.setCurrentText(str(value))
        self.code.blockSignals(False)
        if not cat and name:
            rng = self.layer.value_range(name) or (0.0, 1.0)
            self.number.blockSignals(True)
            try:
                self.number.setValue(float(value) if value is not None else round(rng[0], 4))
            except (TypeError, ValueError):
                self.number.setValue(rng[0])
            self.number.setSingleStep(max((rng[1] - rng[0]) / 50.0, 1e-4))
            self.number.blockSignals(False)

    def _show_fields(self) -> None:
        dtm = self.kind.currentData() == "dtm"
        name = self.attr.currentData()
        cat = bool(name) and self.layer.data.is_categorical(name)
        self.dtm.setVisible(dtm)
        self.mode.setVisible(dtm)
        self.attr.setVisible(not dtm)
        self.op.setVisible(not dtm)
        self.number.setVisible(not dtm and not cat)
        self.code.setVisible(not dtm and cat)

    def _kind_changed(self) -> None:
        if self.kind.currentData() == "dtm":
            self._fill_modes()
        else:
            self._fill_value()
        self._show_fields()
        self.dialog.changed()

    def _dtm_changed(self) -> None:
        self._fill_modes(self.mode.currentData())
        self.dialog.changed()

    def _attr_changed(self) -> None:
        self._fill_value()
        self._show_fields()
        self.dialog.changed()

    def condition(self) -> dict | None:
        if self.kind.currentData() == "dtm":
            if self.dtm.currentData() is None:
                return None
            return {"kind": "dtm", "layer": self.dtm.currentData(), "mode": self.mode.currentData()}
        name = self.attr.currentData()
        if name is None:
            return None
        if self.layer.data.is_categorical(name):
            return {"kind": "value", "name": name, "op": self.op.currentData(), "value": self.code.currentText()}
        return {"kind": "value", "name": name, "op": self.op.currentData(), "value": float(self.number.value())}


class ConstraintDialog(QDialog):
    """Edit a block model layer's constraint. ``applied`` callback: after Apply / OK."""

    def __init__(self, layer, on_apply, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.on_apply = on_apply
        self.setWindowTitle(tr("Block constraints - {layer}", layer=layer.name))
        self.setMinimumWidth(560)
        self.dtms = [(lay.name, dtm_kind(lay.data)) for lay in (layer.scene.layers if layer.scene else [])
                     if isinstance(lay, SurfaceLayer)]
        self.rows: list[_Row] = []

        v = QVBoxLayout(self)
        hint = QLabel(tr("Show only the blocks that meet these conditions (the block centre is tested)."))
        hint.setWordWrap(True)
        v.addWidget(hint)
        box = QFrame()
        box.setFrameShape(QFrame.StyledPanel)
        self.rows_layout = QVBoxLayout(box)
        self.rows_layout.setContentsMargins(8, 8, 8, 8)
        self.rows_layout.setSpacing(6)
        self.empty = QLabel(tr("No conditions: all blocks are shown."))
        self.empty.setStyleSheet("color: #8a939d;")
        self.rows_layout.addWidget(self.empty)
        add = QPushButton(icon("plus"), tr("Add condition"))
        add.setAutoDefault(False)
        add.clicked.connect(lambda: self.add_row({}))
        row = QHBoxLayout()
        row.addWidget(add)
        row.addStretch(1)
        self.rows_layout.addLayout(row)
        v.addWidget(box)

        comb = QGridLayout()
        comb.addWidget(QLabel(tr("Combine:")), 0, 0)
        self.r_and = QRadioButton(tr("all conditions (and)"))
        self.r_or = QRadioButton(tr("any condition (or)"))
        self.r_custom = QRadioButton(tr("custom:"))
        self.expr = QLineEdit()
        self.expr.setPlaceholderText(tr("e.g.  a and (b or not c)"))
        self.group = QButtonGroup(self)
        for i, r in enumerate((self.r_and, self.r_or, self.r_custom)):
            self.group.addButton(r, i)
        comb.addWidget(self.r_and, 0, 1)
        comb.addWidget(self.r_or, 0, 2)
        comb.addWidget(self.r_custom, 1, 1)
        comb.addWidget(self.expr, 1, 2)
        self.expr_msg = QLabel("")
        self.expr_msg.setStyleSheet("color: #e0a030;")
        comb.addWidget(self.expr_msg, 2, 1, 1, 2)
        comb.setColumnStretch(2, 1)
        v.addLayout(comb)

        bottom = QHBoxLayout()
        self.count = QLabel("")
        self.count.setStyleSheet("font-weight: 600;")
        bottom.addWidget(self.count, 1)
        clear = QPushButton(tr("Clear all"))
        clear.setAutoDefault(False)
        clear.clicked.connect(self.clear_all)
        bottom.addWidget(clear)
        v.addLayout(bottom)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        v.addWidget(buttons)

        logic = layer.constraint_logic or "and"
        {"and": self.r_and, "or": self.r_or}.get(logic, self.r_custom).setChecked(True)
        if logic not in ("and", "or"):
            self.expr.setText(logic)
        for cond in layer.constraints:
            self.add_row(cond, update=False)
        self.group.idToggled.connect(lambda *_: self.changed())
        self.expr.textChanged.connect(lambda *_: self.changed())
        self.changed()

    # --- rows -----------------------------------------------------------------------------
    def add_row(self, cond: dict, update: bool = True) -> None:
        r = _Row(self, cond)
        self.rows.append(r)
        self.rows_layout.insertWidget(len(self.rows), r)
        if update:
            self.changed()

    def remove_row(self, r: _Row) -> None:
        self.rows.remove(r)
        r.setParent(None)
        r.deleteLater()
        self.changed()

    def clear_all(self) -> None:
        for r in list(self.rows):
            self.rows.remove(r)
            r.setParent(None)
            r.deleteLater()
        self.r_and.setChecked(True)
        self.expr.clear()
        self.changed()

    # --- state ------------------------------------------------------------------------------
    def conditions(self) -> list[dict]:
        return [c for c in (r.condition() for r in self.rows) if c is not None]

    def logic(self) -> str:
        if self.r_or.isChecked():
            return "or"
        if self.r_custom.isChecked() and self.expr.text().strip():
            return self.expr.text().strip()
        return "and"

    def _logic_error(self) -> str:
        if not self.r_custom.isChecked():
            return ""
        if not self.expr.text().strip():
            return tr("Type an expression of the letters, e.g. a and (b or c).")
        try:
            parse_logic(self.expr.text(), len(self.rows))
        except ValueError as e:
            return tr("Expression: {error}", error=e)
        return ""

    def changed(self) -> None:
        for i, r in enumerate(self.rows):
            r.tag.setText(letter(i))
        self.empty.setVisible(not self.rows)
        self.expr.setEnabled(self.r_custom.isChecked())
        err = self._logic_error()
        self.expr_msg.setText(err)
        self.ok_button.setEnabled(not err)
        self._update_count(err)

    def _update_count(self, err: str) -> None:
        bm = self.layer.data
        conds = self.conditions()
        if not conds:
            self.count.setText(tr("Blocks kept: all {n:,}", n=bm.n_blocks))
            return
        if err:
            self.count.setText(tr("Blocks kept: -"))
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            import numpy as np

            masks = []
            for c in conds:
                m = self.layer.condition_mask(c)
                masks.append(m if m is not None else np.ones(bm.n_blocks, dtype=bool))
            n = int(combine(masks, self.logic()).sum())
        finally:
            QApplication.restoreOverrideCursor()
        self.count.setText(tr("Blocks kept: {n:,} of {total:,}", n=n, total=bm.n_blocks))

    def apply(self) -> bool:
        if self._logic_error():
            return False
        self.on_apply(copy.deepcopy(self.conditions()), self.logic())
        return True

    def accept(self) -> None:
        if self.apply():
            super().accept()
