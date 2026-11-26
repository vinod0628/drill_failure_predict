"""Reserve advisor window: how long the reserve lasts, and a check of every mined cut - in plain
words, with big numbers and simple charts. Works offline; changes nothing."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..ai import BAD, OK, WARN
from ..ai import reserve_advisor as ra
from ..i18n import tr
from . import theme
from .ai_widgets import (AMBER, BLUE, GREEN, LEVEL_COLOR, RED, BigNumber, FindingList, ForecastChart, MonthBars,
                         heading, level_word, tonnes_text)
from .icons import icon


def _scroll(w: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setFrameShape(QScrollArea.NoFrame)
    s.setWidget(w)
    return s


class ReserveAdvisorDialog(QDialog):
    def __init__(self, mgr, parent=None):
        super().__init__(parent or mgr.win)
        self.mgr = mgr
        self.setWindowTitle(tr("Reserve advisor"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
        self.resize(1080, 760)
        self.checks = []

        top = heading(tr("Reserve advisor"),
                      tr("Advice in plain words from your production records and the block model. It is worked out on "
                         "this computer (no internet needed) and changes nothing - you decide what to do."))
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet(f"color: {AMBER}; font-size: 10pt;")

        # --- tab 1: life of the reserve
        life = QWidget()
        lv = QVBoxLayout(life)
        self.q1 = QLabel(tr("How long will the reserve last?"))
        self.q1.setStyleSheet("font-size: 12pt; font-weight: 600;")
        lv.addWidget(self.q1)
        cards = QHBoxLayout()
        self.c_rate = BigNumber(tr("Typical month"), "#e0533f")
        self.c_left = BigNumber(tr("Ore left"), GREEN)
        self.c_life = BigNumber(tr("Lasts about"), AMBER)
        self.c_grade = BigNumber(tr("Grade of recent cuts"), BLUE)
        for c in (self.c_rate, self.c_left, self.c_life, self.c_grade):
            cards.addWidget(c)
        lv.addLayout(cards)
        charts = QHBoxLayout()
        self.chart = ForecastChart()
        self.bars = MonthBars()
        cl = QVBoxLayout()
        cl.addWidget(self._caption(tr("Ore mined so far and the forecast to the end of the reserve")))
        cl.addWidget(self.chart, 1)
        cr = QVBoxLayout()
        cr.addWidget(self._caption(tr("Ore mined each month (red: the months the typical rate comes from)")))
        cr.addWidget(self.bars, 1)
        charts.addLayout(cl, 3)
        charts.addLayout(cr, 2)
        lv.addLayout(charts, 1)
        lv.addWidget(self._caption(tr("What this means")))
        self.life_findings = FindingList()
        lv.addWidget(self.life_findings)

        # --- tab 2: check of the cuts
        cuts = QWidget()
        cv = QVBoxLayout(cuts)
        q2 = QLabel(tr("Are the recorded cuts right?"))
        q2.setStyleSheet("font-size: 12pt; font-weight: 600;")
        cv.addWidget(q2)
        cv.addWidget(self._caption(tr("Each cut is compared with the reserve and with the other cuts. A cut marked "
                                      "Check or Problem is worth a second look - the reason is written beside it.")))
        sc = QHBoxLayout()
        self.c_ok = BigNumber(level_word(OK), GREEN)
        self.c_warn = BigNumber(level_word(WARN), AMBER)
        self.c_bad = BigNumber(level_word(BAD), RED)
        for c in (self.c_ok, self.c_warn, self.c_bad):
            sc.addWidget(c)
        cv.addLayout(sc)
        self.only = QCheckBox(tr("Only show the cuts to check"))
        self.only.toggled.connect(self._fill_table)
        cv.addWidget(self.only)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([tr("Result"), tr("Date"), tr("Name"), tr("Ore (t)"), tr("Grade"),
                                              tr("Ore in the cut"), tr("What we noticed")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._show_cut)
        self.table.itemDoubleClicked.connect(lambda _it: self._open_cut())
        cv.addWidget(self.table, 2)
        self.cut_findings = FindingList()
        cv.addWidget(self.cut_findings, 1)
        self.b_open = QPushButton(icon("pencil-outline"), tr("Open this cut"))
        self.b_open.setToolTip(tr("Open the selected cut to look at it or correct it"))
        self.b_open.clicked.connect(self._open_cut)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.b_open)
        cv.addLayout(row)

        self.tabs = QTabWidget()
        self.tabs.addTab(_scroll(life), icon("chart-line"), tr("Life of the reserve"))
        self.tabs.addTab(cuts, icon("clipboard-check-outline"), tr("Check of the cuts"))

        b_refresh = QPushButton(icon("refresh"), tr("Refresh"))
        b_refresh.setToolTip(tr("Work everything out again (after new cuts or another 'as of' date)"))
        b_refresh.clicked.connect(self.refresh)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        self.foot = QLabel()
        self.foot.setStyleSheet(f"color: {theme.MUTED};")
        bottom.addWidget(self.foot, 1)
        bottom.addWidget(b_refresh)
        bottom.addWidget(close)

        root = QVBoxLayout(self)
        root.addWidget(top)
        root.addWidget(self.notice)
        root.addWidget(self.tabs, 1)
        root.addLayout(bottom)
        self.refresh()

    @staticmethod
    def _caption(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet(f"color: {theme.MUTED};")
        return lab

    # --- data ---------------------------------------------------------------------------------
    def refresh(self) -> None:
        r = self.mgr.reserve
        if not r.ready:
            self.notice.setText(tr("The reserve is not ready: {error}. Set it up in the Reserve window first.",
                                   error=r.error or tr("no block model")))
            self.tabs.setEnabled(False)
            return
        self.notice.setText("")
        self.tabs.setEnabled(True)
        from ..scene.reserve import monthly

        unit = r.setup.get("unit", "g/t")
        as_of = self.mgr.as_of or ra.today()
        s = r.dep.summary(self.mgr.as_of_day)
        f = ra.forecast(monthly(r.log, r.dep), s["total"].tonnes, s["remaining"].tonnes, s["remaining"].metal, as_of,
                        unit)
        self._fill_life(f, unit)
        self.checks = ra.review_cuts(r.dep, r.log.sorted_shapes(), unit, self.mgr.as_of_day)
        self._fill_table()
        self.foot.setText(tr("As of {date} · {n} cuts · grade in {unit}", date=as_of, n=len(r.log.shapes), unit=unit))

    def _fill_life(self, f, unit: str) -> None:
        if not f.ok:
            for c in (self.c_rate, self.c_life, self.c_grade):
                c.set("-", "")
            self.c_left.set(tonnes_text(f.remaining), tr("of {t} in total", t=tonnes_text(f.total)))
            self.chart.set_forecast(f, {})
            self.bars.set_forecast(f)
            self.life_findings.set_findings([(WARN, f.message)])
            return
        self.c_rate.set(tonnes_text(f.rate), tr("slow {lo} · fast {hi}", lo=tonnes_text(f.rate_low),
                                                hi=tonnes_text(f.rate_high)))
        self.c_left.set(tonnes_text(f.remaining), tr("{p:.0%} of the reserve; {g:,.2f} {unit} on average",
                                                     p=f.remaining / f.total if f.total else 0, g=f.grade_remaining,
                                                     unit=unit))
        if math.isfinite(f.months_left):
            self.c_life.set(ra._duration(f.months_left), tr("until {m}", m=ra.month_text(f.end_month)))
        else:
            self.c_life.set("?", tr("no ore mined recently"))
        if f.grade_recent is not None and f.grade_remaining > 0:
            gap = (f.grade_recent - f.grade_remaining) / f.grade_remaining
            color = GREEN if abs(gap) <= ra.GRADE_GAP else AMBER
            word = tr("richer") if gap > 0 else tr("poorer")
            self.c_grade.set(f"{f.grade_recent:,.2f} {unit}", tr("{p:.0%} {word} than the ore left", p=abs(gap),
                                                                   word=word), color)
        else:
            self.c_grade.set("-", "")
        lines = {"expected": ra.projection(f, f.rate), "slow": ra.projection(f, f.rate_low),
                 "fast": ra.projection(f, f.rate_high)}
        self.chart.set_forecast(f, lines)
        self.bars.set_forecast(f)
        self.life_findings.set_findings(f.findings)

    def _fill_table(self, *_a) -> None:
        unit = self.mgr.reserve.setup.get("unit", "g/t")
        n = {OK: 0, WARN: 0, BAD: 0}
        for c in self.checks:
            n[c.level] += 1
        self.c_ok.set(str(n[OK]), tr("cuts look normal"))
        self.c_warn.set(str(n[WARN]), tr("cuts worth a look"))
        self.c_bad.set(str(n[BAD]), tr("cuts probably wrong"))
        rows = [c for c in self.checks if not self.only.isChecked() or c.level != OK]
        t = self.table
        t.setRowCount(len(rows))
        for i, c in enumerate(rows):
            first = next((txt for lv, txt in c.reasons if lv in (BAD, WARN)), c.reasons[0][1])
            cells = ["● " + level_word(c.level), c.date, c.name, f"{c.tonnes:,.0f}",
                     f"{c.grade:,.2f} {unit}" if c.tonnes > 0 else "-", f"{c.fill:.0%}", first]
            for k, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if k == 0:
                    it.setForeground(QColor(LEVEL_COLOR[c.level]))
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if k in (3, 4, 5):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c.after_as_of and k:
                    it.setForeground(QColor(theme.MUTED))
                it.setData(Qt.UserRole, c.sid)
                it.setToolTip("\n".join(txt for _lv, txt in c.reasons))
                t.setItem(i, k, it)
        if rows:
            t.selectRow(0)
        else:
            self.cut_findings.set_findings([(OK, tr("No cut needs a second look."))] if self.checks else
                                           [(WARN, tr("No cuts recorded yet."))])
        self.b_open.setEnabled(bool(rows))

    def _selected(self):
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        if it is None:
            return None
        sid = it.data(Qt.UserRole)
        return next((c for c in self.checks if c.sid == sid), None)

    def _show_cut(self) -> None:
        c = self._selected()
        if c is not None:
            self.cut_findings.set_findings(c.reasons)

    def _open_cut(self) -> None:
        c = self._selected()
        if c is not None:
            self.mgr.edit_shape(c.sid)
