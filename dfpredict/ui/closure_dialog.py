"""Closure check window: when a traverse does not close, the reading mistake that best explains it,
shown with a bar chart (misclosure now and after each correction) and a plan (the traverse as
surveyed and as corrected). Works offline; nothing changes until "Use this correction"."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..ai import BAD, INFO, OK
from ..ai.closure_doctor import diagnose
from ..i18n import tr
from . import theme
from .ai_widgets import AMBER, BLUE, GREEN, RED, BigNumber, FindingList, MisclosureBars, PlanSketch, heading
from .icons import icon

CONFIDENCE = {"very likely": (GREEN, "Very likely"), "likely": (AMBER, "Likely"), "possible": (theme.MUTED, "Possible")}


def confidence_word(key: str) -> str:
    return {"very likely": tr("Very likely"), "likely": tr("Likely"), "possible": tr("Possible")}.get(key, key)


class ClosureDialog(QDialog):
    def __init__(self, manager):
        """``manager``: the Survey window (its current pit, edited setups and settings)."""
        super().__init__(manager)
        self.sm = manager
        self.setWindowTitle(tr("Closure check"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)          # modal: no minimise (no strip)
        self.resize(1080, 760)
        pit = next((p for p in manager.pits if p.id == manager.pid), None)
        self.pit_name = f"{pit.name} ({pit.code})" if pit else ""

        top = heading(tr("Closure check - {pit}", pit=self.pit_name),
                      tr("When the traverse does not close, this looks for the one reading mistake that best explains "
                         "it (a reading 180° off, two figures swapped, heights swapped, the wrong backsight ...). It "
                         "is worked out on this computer; nothing changes until you click Use this correction."))
        cards = QHBoxLayout()
        self.c_miss = BigNumber(tr("Misses by"), RED)
        self.c_allowed = BigNumber(tr("Allowed"), AMBER)
        self.c_acc = BigNumber(tr("Accuracy"), BLUE)
        for c in (self.c_miss, self.c_allowed, self.c_acc):
            cards.addWidget(c)

        self.verdict = FindingList()
        self.answer = QFrame()
        self.answer.setStyleSheet(f"QFrame {{ background: {theme.PANEL_ALT}; border-radius: 6px; }} "
                                  f"QLabel {{ background: transparent; border: none; }}")
        ah = QHBoxLayout(self.answer)
        self.a_badge = QLabel()
        self.a_badge.setFixedWidth(110)
        self.a_badge.setAlignment(Qt.AlignCenter)
        self.a_text = QLabel()
        self.a_text.setWordWrap(True)
        self.a_text.setStyleSheet("font-size: 11pt;")
        ah.addWidget(self.a_badge)
        ah.addWidget(self.a_text, 1)

        self.bars = MisclosureBars()
        self.plan = PlanSketch()
        charts = QSplitter()
        lw = QWidget()
        lv = QVBoxLayout(lw)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self._caption(tr("How far the traverse misses, as surveyed and after each possible correction "
                                      "(shorter is better; green passes)")))
        lv.addWidget(self.bars, 1)
        rw = QWidget()
        rv = QVBoxLayout(rw)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self._caption(tr("Plan: as surveyed (red dashed) and with the selected correction (green)")))
        rv.addWidget(self.plan, 1)
        charts.addWidget(lw)
        charts.addWidget(rw)
        charts.setSizes([520, 520])

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([tr("How likely"), tr("Setup"), tr("Where"), tr("What may be wrong"),
                                              tr("As entered"), tr("Should be"), tr("Misses by after")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.setMaximumHeight(200)

        self.b_use = QPushButton(icon("check-bold", GREEN), tr("Use this correction"))
        self.b_use.setToolTip(tr("Put the corrected value into the setup (you still check it and click Save pit)"))
        self.b_use.clicked.connect(self.use)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        self.foot = QLabel()
        self.foot.setStyleSheet(f"color: {theme.MUTED};")
        bottom.addWidget(self.foot, 1)
        bottom.addWidget(self.b_use)
        bottom.addWidget(close)

        root = QVBoxLayout(self)
        root.addWidget(top)
        root.addLayout(cards)
        root.addWidget(self.verdict)
        root.addWidget(self.answer)
        root.addWidget(charts, 1)
        root.addWidget(self.table)
        root.addLayout(bottom)
        self.run()

    @staticmethod
    def _caption(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet(f"color: {theme.MUTED};")
        return lab

    # --- the check -------------------------------------------------------------------------------
    def _inputs(self):
        import dataclasses

        sm = self.sm
        setups = dict(sm.setups)
        setups[sm.pid] = list(sm.rows)
        known = [s for s in sm.stations if s.kind == "known"]
        typed = sm._typed_entrance()
        if typed is not None:
            name, xyz = typed
            known = [dataclasses.replace(s, x=xyz[0], y=xyz[1], z=xyz[2]) if s.name == name else s for s in known]
        return sm.pits, known, setups, sm.settings

    def run(self) -> None:
        pits, known, setups, settings = self._inputs()
        self.known = {s.name for s in known}
        self.diag = d = diagnose(pits, known, setups, settings, self.sm.pid)
        self.foot.setText(tr("{n} possible mistakes tried", n=d.tried))
        if d.status == "no_closure":
            for c in (self.c_miss, self.c_allowed, self.c_acc):
                c.set("-", "")
            self.verdict.set_findings([(INFO, d.message)])
            self.answer.setVisible(False)
            self._fill([])
            return
        ratio = f"1:{d.ratio:,.0f}" if d.ratio < 1e9 else tr("perfect")
        self.c_miss.set(f"{d.misclosure:.3f} m", tr("closing on {onto}; height {dz:+.3f} m", onto=d.onto, dz=d.dz),
                        GREEN if d.status == "passes" else RED)
        self.c_allowed.set(f"{d.allowed:.3f} m", tr("over {length:,.0f} m of traverse", length=d.length))
        self.c_acc.set(ratio, tr("needs 1:{limit:,.0f}", limit=d.limit), GREEN if d.status == "passes" else RED)
        if d.status == "passes":
            self.verdict.set_findings([(OK, d.message)])
            self.answer.setVisible(False)
            self._fill([])
            return
        self.verdict.set_findings([(BAD, tr("The traverse does not close well enough: it misses {onto} by {m:.3f} m "
                                            "where {a:.3f} m is allowed.", onto=d.onto, m=d.misclosure,
                                            a=d.allowed))])
        self.answer.setVisible(True)
        if d.fixes:
            best = d.fixes[0]
            color = CONFIDENCE[best.confidence][0]
            self.a_badge.setText(confidence_word(best.confidence))
            self.a_badge.setStyleSheet(f"background: {color}; color: #111; border-radius: 6px; font-weight: 700; "
                                       f"padding: 10px;")
            self.a_text.setText(d.message)
        else:
            self.a_badge.setText(tr("Not found"))
            self.a_badge.setStyleSheet(f"background: {theme.MUTED}; color: #111; border-radius: 6px; "
                                       f"font-weight: 700; padding: 10px;")
            self.a_text.setText(d.message)
        self._fill(d.fixes)

    def _where(self, fix) -> str:
        if fix.setup < 0:
            return tr("Survey settings")
        r = fix.row
        return tr("at {at} → {fs}", at=r.at, fs=r.fs or "?")

    def _fill(self, fixes) -> None:
        t = self.table
        t.setRowCount(len(fixes))
        for i, f in enumerate(fixes):
            color, _w = CONFIDENCE[f.confidence]
            cells = ["● " + confidence_word(f.confidence), str(f.setup + 1) if f.setup >= 0 else "-", self._where(f),
                     f.what, f.old, f.new,
                     f"{f.error:.3f} m  " + (tr("(passes)") if f.passes else tr("(better)"))]
            for k, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if k == 0:
                    it.setForeground(QColor(color))
                if k == 6:
                    it.setForeground(QColor(GREEN if f.passes else AMBER))
                t.setItem(i, k, it)
        self.b_use.setEnabled(bool(fixes))
        self.table.setVisible(bool(fixes))
        if fixes:
            t.selectRow(0)
        else:
            self._draw(None)

    def _current(self):
        r = self.table.currentRow()
        return self.diag.fixes[r] if 0 <= r < len(self.diag.fixes) else None

    def _selected(self) -> None:
        self._draw(self._current())

    def _draw(self, fix) -> None:
        d = self.diag
        rows = [(tr("As surveyed"), max(math.hypot(d.misclosure, d.dz), 1e-4), False)]
        for i, f in enumerate(d.fixes):
            label = (tr("Setup {n}: {kind}", n=f.setup + 1, kind=f.new) if f.setup >= 0 else tr("Settings: {kind}",
                                                                                             kind=f.new))
            rows.append((label[:34], max(f.error, 1e-4), f.passes))
        sel = d.fixes.index(fix) + 1 if fix is not None else 0
        self.bars.set_rows(rows if len(rows) > 1 or d.status != "no_closure" else [], max(d.allowed, 1e-4), sel)
        p = self.plan
        p.clear()
        before = [xyz for _n, xyz in d.path] + ([d.miss_xyz] if d.miss_xyz is not None else [])
        names = [n if n not in self.known else "" for n, _x in d.path]
        if len(before) >= 2:
            p.line(before, RED, 2.0, Qt.DashLine)
        if d.miss_xyz is not None:
            p.mark(d.miss_xyz, RED, tr("lands here: {m:.2f} m off", m=d.misclosure))
        if fix is not None and len(fix.path) >= 2:
            after = [xyz for _n, xyz in fix.path] + ([d.onto_xyz] if d.onto_xyz is not None else [])
            p.line(after, GREEN, 3.0, Qt.SolidLine, [n if n not in self.known else "" for n, _x in fix.path] + [""])
            if fix.setup >= 0:
                at = dict(fix.path).get(fix.row.at)
                if at is not None:
                    p.mark(at, AMBER, tr("check here"), "ring")
        elif len(before) >= 2:
            p.line(before, RED, 2.0, Qt.DashLine, names + [""])
        for name, xyz in d.path:
            if name in self.known:
                p.mark(xyz, BLUE, name, "triangle")
        p.legend = [(RED, tr("as surveyed"), "dash"), (GREEN, tr("corrected"), "line"),
                    (BLUE, tr("known station"), "fill")]
        p.update()

    # --- apply -----------------------------------------------------------------------------------
    def use(self) -> None:
        fix = self._current()
        if fix is None:
            return
        sm = self.sm
        if fix.setup >= 0:
            sm.rows[fix.setup] = fix.row
            sm._set_dirty(True)
            sm._recompute()
            sm.table.selectRow(fix.setup)
            sm.status.setText(tr("Correction put into setup {n} - check it, then click Save pit", n=fix.setup + 1))
        else:
            if QMessageBox.question(self, tr("Closure check"),
                                    tr("Change the survey settings? This affects every pit:\n\n{what}\n{old} → {new}",
                                       what=fix.what, old=fix.old, new=fix.new)) != QMessageBox.Yes:
                return
            sm.store.set_settings({k: v for k, v in fix.settings.items()
                                   if k in ("closure_ratio", "declination", "convergence")})
            sm.store.recompute()
            sm.settings = sm.store.settings()
            sm.stations = sm.store.stations()
            sm._recompute()
            sm.changed.emit()
        self.run()

