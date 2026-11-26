"""Risk Prediction: the failure risk of an ongoing (or planned) hole ahead of the bit, the measures to
take, and the plan check of a new design (design spec 5, 6, 7.1, 8, 9).

* **Ongoing / planned hole**: pick the hole, press *Predict*. The window shows
  - the headline: the chance of reaching the target depth, the summary sentence, the worst level ahead;
  - *Danger in a depth range*: the chance of failure (all / per family), of meeting a cavity, of crossing
    a fault - for the next 10 m, the next 50 m, to the target, or any range typed in;
  - the risk log: formations and contacts along the predicted path (±u), cavity / fault-zone probability,
    the risk index of each family against the alarm levels, the bit, the warning zones;
  - the risk register (spec 8.3) and, for the selected zone, the detail panel (spec 8.2): why the zone is
    dangerous (reason codes) and what to do (measures with their evidence);
  - messages: contact ties, model mismatch, distance to model control, outside experience, rules R1 - R8.
* **Plan check**: a design typed in (collar X, Y, Z, azimuth, dip, depth, rig) is checked the same way
  without being saved; *Safer options* tries other azimuths / dips.
* The 3D view shows the hole, its predicted path and cone, and the risk sleeves; the zone selected here is
  highlighted there, and a sleeve clicked with Identify selects its zone here.
* Export: register as CSV, report as HTML / PDF, warnings saved in the database with the model version.
"""

from __future__ import annotations

import base64
import html
import io
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..risk import codes
from ..risk.predict import Prediction, register_frame, warning_rows
from . import theme
from .icons import icon
from .risk_widgets import Banner, ChipLabel, StatCard, pct, run_job

log = logging.getLogger(__name__)
LEVEL_FG = {"green": "#0d1117", "yellow": "#0d1117", "orange": "#0d1117", "red": "#ffffff"}


def _level_chip(level: str) -> str:
    return (f"<span style='background:{codes.LEVEL_COLORS[level]}; color:{LEVEL_FG[level]}; border-radius:4px;"
            f" padding:1px 6px; font-weight:700'>&nbsp;{html.escape(tr(codes.LEVEL_NAMES[level]).upper())}&nbsp;</span>")


def _cjk_fonts() -> None:
    """Put an installed font with Chinese glyphs first in matplotlib's sans-serif list (once)."""
    import matplotlib
    from matplotlib import font_manager

    if getattr(_cjk_fonts, "done", False):
        return
    _cjk_fonts.done = True
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "DengXian", "SimSun", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Micro Hei"):
        if name in have:
            fam = list(matplotlib.rcParams["font.sans-serif"])
            matplotlib.rcParams["font.sans-serif"] = [name] + [f for f in fam if f != name]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False      # the minus sign of a CJK font may be missing
            break


def reach_color(p: float) -> str:
    return codes.LEVEL_COLORS["green"] if p >= 0.85 else (codes.LEVEL_COLORS["yellow"] if p >= 0.7 else
                                                          (codes.LEVEL_COLORS["orange"] if p >= 0.5 else
                                                           codes.LEVEL_COLORS["red"]))


# ------------------------------------------------------------------------------------------------
class RiskLog(QWidget):
    """The risk log chart (matplotlib): depth down, tracks side by side."""

    zone_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        _cjk_fonts()                          # Chinese titles / legends (matplotlib's own font has no CJK glyphs)
        self.fig = Figure(figsize=(6, 7), dpi=100, facecolor=theme.PANEL)
        self.canvas = FigureCanvasQTAgg(self.fig)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.canvas)
        self.pred: Prediction | None = None
        self.selected = -1
        self.canvas.mpl_connect("button_press_event", self._click)
        self._empty()

    def _style(self, ax) -> None:
        ax.set_facecolor(theme.BG)
        for s in ax.spines.values():
            s.set_color(theme.BORDER)
        ax.tick_params(colors=theme.MUTED, labelsize=8)
        ax.xaxis.label.set_color(theme.TEXT)
        ax.yaxis.label.set_color(theme.TEXT)
        ax.title.set_color(theme.TEXT)

    def _empty(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style(ax)
        ax.text(0.5, 0.5, tr("Pick a hole and press Predict"), ha="center", va="center", color=theme.MUTED,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw_idle()

    def show_prediction(self, p: Prediction | None, selected: int = -1) -> None:
        self.pred, self.selected = p, selected
        if p is None:
            self._empty()
            return
        from matplotlib.patches import Rectangle

        self.fig.clear()
        gs = self.fig.add_gridspec(1, 4, width_ratios=[0.9, 1.3, 2.2, 0.35], wspace=0.08, left=0.10, right=0.98,
                                   top=0.88, bottom=0.07)
        axs = [self.fig.add_subplot(gs[0, i]) for i in range(4)]
        for a in axs[1:]:
            a.sharey(axs[0])
        for a in axs:
            self._style(a)
        target = p.planned_depth
        top_d, bot_d = 0.0, target + 5
        md = p.md
        m = md <= bot_d
        # 1. units along the predicted path
        a0 = axs[0]
        names = p.unit
        colors = {}
        pal = ["#e6b85c", "#5a5f6b", "#4ea1f3", "#9ccc65", "#c792ea", "#f78c6c", "#89ddff", "#d4a373", "#a5d6a7",
               "#ffcb6b"]
        start = 0
        for k in range(1, len(md) + 1):
            if k == len(md) or names[k] != names[start]:
                u = names[start]
                colors.setdefault(u, pal[len(colors) % len(pal)])
                y0, y1 = md[start] - 0.5, md[k - 1] + 0.5
                a0.add_patch(Rectangle((0, y0), 1, y1 - y0, color=colors[u], alpha=0.85, lw=0))
                if y1 - y0 > 8 and y0 < bot_d:
                    a0.text(0.5, (y0 + min(y1, bot_d)) / 2, u, ha="center", va="center", fontsize=8,
                            color="#101214", fontweight="bold")
                start = k
        for d, up, lo, u in p.contacts:
            if d <= bot_d:
                a0.axhspan(d - u, d + u, color="white", alpha=0.10, lw=0)
                a0.axhline(d, color="white", lw=0.6, alpha=0.6)
        for d, name, share in p.crossings:
            if d <= bot_d:
                a0.plot([0.05, 0.95], [d, d], color="#ff4d4d", lw=2.0)
                a0.text(0.97, d, name, color="#ff6b6b", fontsize=7, va="bottom", ha="right")
        a0.set_xlim(0, 1)
        a0.set_xticks([])
        a0.set_title(tr("Formation"), fontsize=9)
        a0.set_ylabel(tr("Depth along the hole (m)"))
        # 2. Model H: cavity / fault-zone probability
        a1 = axs[1]
        a1.plot(p.p_cavity[m], md[m], color="#3d8bfd", lw=1.3, label=tr("cavity"))
        a1.plot(p.p_fault[m], md[m], color="#ff6b6b", lw=1.1, label=tr("fault zone"))
        a1.set_xlim(0, max(0.2, float(np.nanmax(np.r_[p.p_cavity[m], p.p_fault[m]])) * 1.15))
        a1.set_title(tr("Hazard (Model H)"), fontsize=9)
        a1.legend(loc="lower right", fontsize=7, facecolor=theme.PANEL, edgecolor=theme.BORDER, labelcolor=theme.TEXT)
        a1.tick_params(labelleft=False)
        # 3. risk index per family vs the levels
        a2 = axs[2]
        for lo, hi, lv in ((0.0, 0.15, "green"), (0.15, 0.35, "yellow"), (0.35, 0.60, "orange"), (0.60, 1.0, "red")):
            a2.axvspan(lo, hi, color=codes.LEVEL_COLORS[lv], alpha=0.10, lw=0)
        for fam, idx in p.index.items():
            a2.plot(idx[m], md[m], color=codes.FAMILY_COLORS[fam], lw=1.4, label=tr(codes.FAMILIES[fam]))
        a2.set_xlim(0, 1)
        a2.set_title(tr("Risk index by family"), fontsize=9)
        a2.legend(loc="lower right", fontsize=7, facecolor=theme.PANEL, edgecolor=theme.BORDER, labelcolor=theme.TEXT)
        a2.tick_params(labelleft=False)
        # 4. level strip
        a3 = axs[3]
        for k in np.nonzero(m)[0]:
            a3.add_patch(Rectangle((0, md[k] - 0.5), 1, 1, color=codes.LEVEL_COLORS[p.level[k]], lw=0))
        a3.set_xlim(0, 1)
        a3.set_xticks([])
        a3.set_title(tr("Level"), fontsize=9)
        a3.tick_params(labelleft=False)
        # zones, bit, target
        for zi, z in enumerate(p.zones):
            sel = zi == selected
            for a in axs[1:3]:
                a.axhspan(z.band_from, z.band_to, color=codes.LEVEL_COLORS[z.level], alpha=0.10 if not sel else 0.22,
                          lw=0)
                a.axhspan(z.from_, z.to, color=codes.LEVEL_COLORS[z.level], alpha=0.18 if not sel else 0.40, lw=0)
            if sel:
                for a in axs:
                    a.axhline(z.from_, color="white", lw=1.0, ls="--")
                    a.axhline(z.to, color="white", lw=1.0, ls="--")
        if p.drilled_depth > 0:
            for a in axs:
                a.axhline(p.drilled_depth, color="#ffffff", lw=1.6)
            a2.text(0.99, p.drilled_depth, " " + tr("bit {d:.0f} m", d=p.drilled_depth), color="white", fontsize=8,
                    ha="right", va="bottom")
        for a in axs:
            a.axhline(target, color="#58a6ff", lw=1.2, ls=":")
        a2.text(0.99, target, tr("target {d:.0f} m", d=target), color="#58a6ff", fontsize=8, ha="right", va="top")
        axs[0].set_ylim(bot_d, top_d)
        self.fig.suptitle(f"{p.hole_id} · {tr('chance of reaching target')} {100 * p.p_reach:.0f}%", color=theme.TEXT,
                          fontsize=10, y=0.985)
        self.canvas.draw_idle()

    def _click(self, ev) -> None:
        if self.pred is None or ev.ydata is None:
            return
        d = float(ev.ydata)
        for zi, z in enumerate(self.pred.zones):
            if z.band_from <= d <= z.band_to:
                self.zone_clicked.emit(zi)
                return

    def png_base64(self) -> str:
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=110, facecolor=self.fig.get_facecolor())
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------------------------------------
class PredictionDialog(QDialog):
    show_in_3d = Signal(object)              # Prediction | None
    zone_selected = Signal(str, int)         # hole id, zone index
    models_requested = Signal()
    records_requested = Signal()

    def __init__(self, engine, parent=None, prepare=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Risk Prediction - ahead of the bit"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.engine = engine
        self.prepare = prepare                # callable(): pass the scene's model surfaces to the engine
        self.pred: Prediction | None = None
        self.busy = False
        self._build()
        self.resize(1560, 960)
        self.refresh_holes()

    # --- layout ---------------------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        self.mode = QTabWidget()
        self.mode.setMaximumHeight(150)
        self.mode.addTab(self._hole_page(), icon("progress-wrench"), tr("Ongoing / planned hole"))
        self.mode.addTab(self._plan_page(), icon("compass-outline"), tr("Plan check (new design)"))
        root.addWidget(self.mode)
        st = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {theme.MUTED};")
        st.addWidget(self.progress)
        st.addWidget(self.status, 1)
        self.model_label = QLabel("")
        self.model_label.setStyleSheet(f"color: {theme.MUTED};")
        st.addWidget(self.model_label)
        b = QPushButton(icon("brain"), tr("Models ..."))
        b.setToolTip(tr("Train the models, see their validation, choose the version"))
        b.clicked.connect(self.models_requested)
        st.addWidget(b)
        root.addLayout(st)
        self.banner = Banner(self)
        root.addWidget(self.banner)
        # results
        main = QSplitter(Qt.Vertical)
        top = QSplitter(Qt.Horizontal)
        top.addWidget(self._summary_panel())
        self.chart = RiskLog()
        self.chart.zone_clicked.connect(self.select_zone)
        top.addWidget(self.chart)
        top.setSizes([520, 1000])
        main.addWidget(top)
        bottom = QSplitter(Qt.Horizontal)
        reg = QWidget()
        rv = QVBoxLayout(reg)
        rv.setContentsMargins(0, 0, 0, 0)
        rh = QHBoxLayout()
        lbl = QLabel(tr("Risk register"))
        lbl.setStyleSheet("font-weight: 700; font-size: 11pt;")
        rh.addWidget(lbl)
        self.only_ahead = QCheckBox(tr("Only ahead of the bit"))
        self.only_ahead.toggled.connect(self._fill_register)
        rh.addWidget(self.only_ahead)
        rh.addStretch(1)
        for ic, text, slot in (("file-delimited-outline", tr("CSV"), self.export_csv),
                               ("file-document-outline", tr("Report ..."), self.export_report),
                               ("database-arrow-down-outline", tr("Save warnings"), self.save_warnings)):
            bb = QPushButton(icon(ic), text)
            bb.clicked.connect(slot)
            rh.addWidget(bb)
        self.b_save_warn = bb
        bb.setToolTip(tr("Save the warning zones in the database with the model version (spec 7.4)"))
        rv.addLayout(rh)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([tr("From (m)"), tr("To (m)"), tr("±u (m)"), tr("Failure"), tr("P"),
                                              tr("Level"), tr("Reasons"), tr("Measures"), tr("Status")])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._row_selected)
        rv.addWidget(self.table, 1)
        bottom.addWidget(reg)
        self.detail = QTextBrowser()
        self.detail.setOpenExternalLinks(False)
        bottom.addWidget(self.detail)
        bottom.setSizes([850, 700])
        main.addWidget(bottom)
        main.setSizes([560, 360])
        root.addWidget(main, 1)

    def _hole_page(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        f = QFormLayout()
        self.hole = QComboBox()
        self.hole.setMinimumWidth(320)
        self.hole.setToolTip(tr("Holes in progress and planned holes first"))
        f.addRow(tr("Hole"), self.hole)
        self.paths = QSpinBox()
        self.paths.setRange(10, 300)
        self.paths.setValue(40)
        self.paths.setSuffix(" " + tr("trial paths"))
        self.paths.setToolTip(tr("Trial paths of the trajectory and model-error ensemble: more = smoother, slower"))
        f.addRow(tr("Uncertainty"), self.paths)
        h.addLayout(f)
        col = QVBoxLayout()
        self.b_predict = QPushButton(icon("play-circle-outline"), tr("Predict"))
        self.b_predict.setDefault(True)
        self.b_predict.setMinimumHeight(36)
        self.b_predict.setStyleSheet("font-weight: 700; font-size: 11pt;")
        self.b_predict.clicked.connect(self.run_hole)
        col.addWidget(self.b_predict)
        self.show3d = QCheckBox(tr("Show in the 3D view"))
        self.show3d.setChecked(True)
        self.show3d.toggled.connect(lambda on: self.show_in_3d.emit(self.pred if on else None))
        col.addWidget(self.show3d)
        h.addLayout(col)
        self.hole_info = QLabel("")
        self.hole_info.setWordWrap(True)
        self.hole_info.setStyleSheet(f"color: {theme.MUTED};")
        h.addWidget(self.hole_info, 1)
        b = QPushButton(icon("database-edit-outline"), tr("Records ..."))
        b.setToolTip(tr("Open the Drilling Records to enter runs, surveys, events"))
        b.clicked.connect(self.records_requested)
        h.addWidget(b)
        self.hole.currentIndexChanged.connect(self._hole_info)
        return w

    def _plan_page(self) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        self.p_name = QLineEdit("PLAN-1")
        self.p_area = QComboBox()
        self.p_rig = QComboBox()
        self.p_x, self.p_y, self.p_z = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for s in (self.p_x, self.p_y, self.p_z):
            s.setRange(-1e7, 1e7)
            s.setDecimals(2)
            s.setSuffix(" m")
        self.p_az, self.p_dip, self.p_depth = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        self.p_az.setRange(0, 360)
        self.p_az.setSuffix(" °")
        self.p_az.setWrapping(True)
        self.p_dip.setRange(-90, -20)
        self.p_dip.setValue(-70)
        self.p_dip.setSuffix(" °")
        self.p_dip.setToolTip(tr("Negative downward: -90 = vertical down"))
        self.p_depth.setRange(10, 5000)
        self.p_depth.setValue(300)
        self.p_depth.setSuffix(" m")
        items = [(tr("Name"), self.p_name), (tr("Area"), self.p_area), (tr("Rig"), self.p_rig),
                 (tr("Collar X"), self.p_x), (tr("Collar Y"), self.p_y), (tr("Collar Z"), self.p_z),
                 (tr("Azimuth"), self.p_az), (tr("Dip"), self.p_dip), (tr("Planned depth"), self.p_depth)]
        for i, (lab, wd) in enumerate(items):
            r, c = i % 3, (i // 3) * 2
            g.addWidget(QLabel(lab), r, c)
            g.addWidget(wd, r, c + 1)
        col = QVBoxLayout()
        b = QPushButton(icon("play-circle-outline"), tr("Check plan"))
        b.setStyleSheet("font-weight: 700;")
        b.setMinimumHeight(32)
        b.clicked.connect(self.run_plan)
        col.addWidget(b)
        b2 = QPushButton(icon("compass-rose"), tr("Safer options ..."))
        b2.setToolTip(tr("Try other azimuths (±15°, ±30°) and dips (±10°): chance of reaching the target for each"))
        b2.clicked.connect(self.run_options)
        col.addWidget(b2)
        b3 = QPushButton(icon("content-copy"), tr("From the selected hole"))
        b3.setToolTip(tr("Fill the design from the hole picked on the first tab"))
        b3.clicked.connect(self._design_from_hole)
        col.addWidget(b3)
        g.addLayout(col, 0, 6, 3, 1)
        g.setColumnStretch(7, 1)
        return w

    def _summary_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        cards = QHBoxLayout()
        self.c_reach = StatCard(tr("chance of reaching the target depth"))
        self.c_worst = StatCard(tr("worst level ahead"))
        cards.addWidget(self.c_reach, 3)
        cards.addWidget(self.c_worst, 2)
        v.addLayout(cards)
        self.summary = QLabel(tr("No prediction yet."))
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size: 11pt; padding: 4px;")
        v.addWidget(self.summary)
        box = QGroupBox(tr("Danger in a depth range"))
        bv = QVBoxLayout(box)
        rr = QHBoxLayout()
        self.r_from, self.r_to = QDoubleSpinBox(), QDoubleSpinBox()
        for s in (self.r_from, self.r_to):
            s.setRange(0, 5000)
            s.setDecimals(0)
            s.setSuffix(" m")
            s.valueChanged.connect(self._fill_range)
        rr.addWidget(QLabel(tr("From")))
        rr.addWidget(self.r_from)
        rr.addWidget(QLabel(tr("to")))
        rr.addWidget(self.r_to)
        for text, fn in ((tr("Next 10 m"), lambda: self._quick(10)), (tr("Next 50 m"), lambda: self._quick(50)),
                         (tr("To target"), lambda: self._quick(None))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            rr.addWidget(b)
        bv.addLayout(rr)
        self.range_view = QTextBrowser()
        self.range_view.setMaximumHeight(190)
        bv.addWidget(self.range_view)
        v.addWidget(box)
        mb = QGroupBox(tr("Messages (model, contacts, rules)"))
        mv = QVBoxLayout(mb)
        self.messages = QListWidget()
        self.messages.setWordWrap(True)
        mv.addWidget(self.messages)
        v.addWidget(mb, 1)
        return w

    # --- holes ------------------------------------------------------------------------------------
    def refresh_holes(self) -> None:
        try:
            c = self.engine.live_holes()
            data = self.engine.data
        except Exception as e:  # noqa: BLE001
            self.banner.show_message(tr("The drilling records cannot be read: {e}", e=e), "error")
            return
        cur = self.hole.currentData()
        self.hole.blockSignals(True)
        self.hole.clear()
        for r in c.itertuples(index=False):
            label = f"{r.hole_id}   ·   {tr(codes.RESULTS.get(r.result, r.result or ''))}   ·   {r.area_id}"
            if r.result == "in_progress":
                label += f"   ·   {tr('bit at {d:.0f} m of {t:.0f} m', d=r.final_depth or 0, t=r.planned_depth or 0)}"
            self.hole.addItem(label, r.hole_id)
            self.hole.setItemData(self.hole.count() - 1, QColor(codes.RESULT_COLORS.get(r.result, "#888")),
                                  Qt.ForegroundRole)
        i = self.hole.findData(cur)
        self.hole.setCurrentIndex(max(0, i))
        self.hole.blockSignals(False)
        self._hole_info()
        areas = list(data["area"]["area_id"].astype(str))
        rigs = list(data["rig"]["rig_model"].astype(str))
        for box, items, default in ((self.p_area, areas, self.engine.area_id), (self.p_rig, rigs, None)):
            cur = box.currentText()
            box.clear()
            box.addItems(items)
            box.setCurrentText(cur or default or (items[0] if items else ""))
        ms = self.engine.models
        if ms is None:
            self.model_label.setText(tr("No trained models"))
            self.banner.show_message(tr("No trained models in this workspace yet. Train them from the drilling records "
                                        "first (a few minutes)."), "warn", tr("Models ..."), self.models_requested.emit)
        else:
            self.model_label.setText(tr("Models {v} · {n} holes", v=ms.version, n=ms.trained_on.get("holes", "?")))

    def _hole_info(self) -> None:
        hid = self.hole.currentData()
        if not hid:
            self.hole_info.setText("")
            return
        c = self.engine.data.hole("collar", hid)
        if not len(c):
            return
        r = c.iloc[0]
        nsv = len(self.engine.data.hole("survey", hid))
        nrun = len(self.engine.data.hole("run", hid))
        self.hole_info.setText(tr("Design: azimuth {az}°, dip {dip}°, target {t} m · rig {rig} · {s} survey stations, "
                                  "{n} runs entered", az=_n(r["plan_azimuth"]), dip=_n(r["plan_dip"]),
                                  t=_n(r["planned_depth"]), rig=r["rig_model"] or "-", s=nsv, n=nrun))

    def _design_from_hole(self) -> None:
        hid = self.hole.currentData()
        if not hid:
            return
        r = self.engine.data.hole("collar", hid).iloc[0]
        self.p_name.setText(f"{hid}-ALT")
        self.p_area.setCurrentText(str(r["area_id"] or ""))
        self.p_rig.setCurrentText(str(r["rig_model"] or ""))
        for s, v in ((self.p_x, r["x"]), (self.p_y, r["y"]), (self.p_z, r["z"]), (self.p_az, r["plan_azimuth"]),
                     (self.p_dip, r["plan_dip"]), (self.p_depth, r["planned_depth"])):
            if pd.notna(v):
                s.setValue(float(v))

    # --- running ------------------------------------------------------------------------------------
    def _start(self, fn, done, text: str) -> None:
        if self.busy:
            return
        if self.engine.models is None:
            self.banner.show_message(tr("Train the models first."), "warn", tr("Models ..."), self.models_requested.emit)
            return
        if self.prepare is not None:
            self.prepare()
        self.busy = True
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText(text)
        self.banner.hide()
        for b in (self.b_predict,):
            b.setEnabled(False)

        def ok(r):
            self._stop()
            done(r)

        def bad(msg):
            self._stop()
            self.banner.show_message(tr("The prediction failed: {e}", e=msg), "error")

        def prog(p, t):
            self.progress.setValue(p)
            if t:
                self.status.setText(t)
        run_job(self, fn, ok, bad, prog)

    def _stop(self) -> None:
        self.busy = False
        self.progress.hide()
        self.status.setText("")
        self.b_predict.setEnabled(True)

    def run_hole(self) -> None:
        hid = self.hole.currentData()
        if not hid:
            return
        n = self.paths.value()
        self._start(lambda prog: self.engine.predict(hid, n_paths=n, progress=prog), self.set_prediction,
                    tr("Predicting {hole} ...", hole=hid))

    def _design(self) -> dict:
        return {"hole_id": self.p_name.text().strip() or "PLAN", "area_id": self.p_area.currentText(),
                "rig_model": self.p_rig.currentText(), "x": self.p_x.value(), "y": self.p_y.value(),
                "z": self.p_z.value(), "plan_azimuth": self.p_az.value(), "plan_dip": self.p_dip.value(),
                "planned_depth": self.p_depth.value(), "final_depth": 0.0, "result": "planned",
                "casing_available": "HW;NW;BW", "fluids_available": "water;polymer;bentonite"}

    def run_plan(self) -> None:
        d = self._design()
        if d["x"] == 0 and d["y"] == 0:
            self.banner.show_message(tr("Enter the collar position of the design (X, Y, Z)."), "warn")
            return
        n = self.paths.value()
        self._start(lambda prog: self.engine.predict(design=d, n_paths=n, progress=prog), self.set_prediction,
                    tr("Checking the plan {name} ...", name=d["hole_id"]))

    def run_options(self) -> None:
        d = self._design()
        if d["x"] == 0 and d["y"] == 0:
            self.banner.show_message(tr("Enter the collar position of the design (X, Y, Z)."), "warn")
            return
        self._start(lambda prog: self.engine.predictor.plan_options(d, progress=prog), self._show_options,
                    tr("Trying other azimuths and dips ..."))

    def _show_options(self, df: pd.DataFrame) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Safer options for {name}", name=self.p_name.text()))
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(tr("Chance of reaching the target for other azimuths and dips (best first). Double-click a "
                              "row to use it. A hint, not proof (spec 5.5): past crews chose settings in response to "
                              "what they met.")))
        t = QTableWidget(len(df), 6)
        t.setHorizontalHeaderLabels([tr("Azimuth"), tr("Dip"), tr("Reach target"), tr("Worst level"), tr("Zones"),
                                     tr("Main risk")])
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        for i, r in df.iterrows():
            vals = [f"{r['azimuth']:.0f}°", f"{r['dip']:.0f}°", pct(r["p_reach"]), tr(codes.LEVEL_NAMES[r["worst"]]),
                    str(r["zones"]), r["main_risk"]]
            for j, val in enumerate(vals):
                it = QTableWidgetItem(val + ("   ← " + tr("design") if j == 0 and r["is_design"] else ""))
                if j == 2:
                    it.setForeground(QColor(reach_color(r["p_reach"])))
                if j == 3:
                    it.setBackground(QColor(codes.LEVEL_COLORS[r["worst"]]))
                    it.setForeground(QColor(LEVEL_FG[r["worst"]]))
                if r["is_design"]:
                    f = QFont()
                    f.setBold(True)
                    it.setFont(f)
                t.setItem(i, j, it)
        t.horizontalHeader().setStretchLastSection(True)
        t.resizeColumnsToContents()

        def use(row, _col):
            self.p_az.setValue(float(df.loc[row, "azimuth"]))
            self.p_dip.setValue(float(df.loc[row, "dip"]))
            dlg.accept()
            self.run_plan()
        t.cellDoubleClicked.connect(use)
        v.addWidget(t)
        dlg.resize(820, 480)
        dlg.exec()

    # --- showing --------------------------------------------------------------------------------------
    def set_prediction(self, p: Prediction) -> None:
        self.pred = p
        col = reach_color(p.p_reach)
        self.c_reach.set(pct(p.p_reach), col, tr("{hole} · target {d:.0f} m", hole=p.hole_id, d=p.planned_depth))
        ahead = [z for z in p.zones if z.status != "Drilled"]
        worst = max([z.level for z in ahead] or ["green"], key=codes.level_rank)
        self.c_worst.set(tr(codes.LEVEL_NAMES[worst]).upper(), codes.LEVEL_COLORS[worst],
                         tr(codes.LEVEL_ACTION[worst]) + " · " + tr("{n} zones", n=len(ahead)))
        self.summary.setText(p.summary)
        self.messages.clear()
        for r in [r for r in p.rules if r.current]:
            it = QListWidgetItem(icon("alarm-light-outline", codes.LEVEL_COLORS[r.level]),
                                 f"{r.rule} · {tr(r.condition)}  -  " + tr(codes.REASON_TITLES[r.reason]))
            it.setForeground(QColor(codes.LEVEL_COLORS[r.level]))
            self.messages.addItem(it)
        for c, t in p.messages:
            it = QListWidgetItem(icon("information-outline"), f"{c}  {t}" if c else t)
            self.messages.addItem(it)
        if p.ensemble.drilled_depth > 0:
            self.messages.addItem(QListWidgetItem(
                icon("map-marker-path"),
                tr("Path ahead: build {b:+.1f}°/30 m, turn {t:+.1f}°/30 m from the last {n} survey stations; spread at "
                   "the target ±{s:.0f} m.", b=p.ensemble.build_rate, t=p.ensemble.turn_rate, n=min(4, p.ensemble.station_count),
                   s=p.ensemble.end_offset())))
        self.messages.addItem(QListWidgetItem(icon("layers-outline"), tr("Geological model: {m}", m=p.geomodel)))
        self.r_from.blockSignals(True)
        self.r_to.blockSignals(True)
        self.r_from.setValue(p.drilled_depth)
        self.r_to.setValue(min(p.planned_depth, p.drilled_depth + 50))
        self.r_from.blockSignals(False)
        self.r_to.blockSignals(False)
        self._fill_range()
        self._fill_register()
        self.chart.show_prediction(p)
        first = next((i for i, z in enumerate(p.zones) if z.status != "Drilled"), 0 if p.zones else -1)
        if first >= 0:
            self.select_zone(first)
        else:
            self.detail.setHtml(f"<h3>{tr('No warning zone')}</h3><p>{html.escape(p.summary)}</p>")
        if self.show3d.isChecked():
            self.show_in_3d.emit(p)

    def _quick(self, length: float | None) -> None:
        if self.pred is None:
            return
        a = self.pred.drilled_depth
        b = self.pred.planned_depth if length is None else min(self.pred.planned_depth, a + length)
        self.r_from.setValue(a)
        self.r_to.setValue(b)

    def _fill_range(self) -> None:
        p = self.pred
        if p is None:
            return
        a, b = self.r_from.value(), self.r_to.value()
        if b <= a:
            self.range_view.setHtml(f"<p style='color:{theme.MUTED}'>{tr('Choose a range: To after From.')}</p>")
            return
        rp = p.range_probabilities(a, b)

        def bar(v, color):
            # Qt rich text draws table cells, not CSS boxes: the bar is a coloured cell of the right width
            w = max(2, int(200 * min(1.0, max(v, 0.0) ** 0.5)))       # square root: small chances stay visible
            return (f"<td width=204><table cellspacing=0 cellpadding=0><tr><td bgcolor='{color}' width={w} "
                    f"height=11></td><td width={204 - w}></td></tr></table></td>"
                    f"<td align=right><b>{100 * v:.1f}%</b></td>")
        rows = [f"<tr><td>{tr('Failure (any)')}</td>{bar(rp.get('failure', 0), '#f85149')}</tr>"]
        for fam in codes.FAMILIES:
            rows.append(f"<tr><td>&nbsp;&nbsp;{tr(codes.FAMILIES[fam])}</td>{bar(rp.get(fam, 0), codes.FAMILY_COLORS[fam])}</tr>")
        rows.append(f"<tr><td>{tr('Meeting a cavity')}</td>{bar(rp.get('cavity', 0), '#3d8bfd')}</tr>")
        rows.append(f"<tr><td>{tr('Crossing a model fault')}</td>{bar(rp.get('fault_crossed', 0), '#ff6b6b')}</tr>")
        rows.append(f"<tr><td>{tr('Unmapped fault zone')}</td>{bar(rp.get('fault_zone', 0), '#c06060')}</tr>")
        live = [r for r in p.rules if r.current] if a <= p.drilled_depth + 10 else []
        alarm = ""
        if live:
            worst = max((r.level for r in live), key=codes.level_rank)
            alarm = (f"<p>{_level_chip(worst)} <b>{tr('Live alarm at the bit')}:</b> "
                     + ", ".join(f"{r.rule} {html.escape(tr(codes.REASON_TITLES[r.reason]))}" for r in live)
                     + f" - {html.escape(tr(codes.LEVEL_ACTION[worst]))}</p>")
        self.range_view.setHtml(f"<p>{tr('From {a:.0f} m to {b:.0f} m', a=a, b=b)}</p>{alarm}"
                                f"<table cellspacing=3>{''.join(rows)}</table>")

    def _fill_register(self) -> None:
        p = self.pred
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if p is None:
            return
        for zi, z in enumerate(p.zones):
            if self.only_ahead.isChecked() and z.status == "Drilled":
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [z.from_, z.to, z.u, ", ".join(z.codes), z.p, tr(codes.LEVEL_NAMES[z.level]),
                    ", ".join(c for c, _t in z.reasons if c), ", ".join(s.code for s in z.prevent), tr(z.status)]
            for j, v in enumerate(vals):
                it = QTableWidgetItem()
                if isinstance(v, float):
                    it.setData(Qt.DisplayRole, round(v, 2 if j == 4 else 0))
                else:
                    it.setText(str(v))
                it.setData(Qt.UserRole, zi)
                if j == 5:
                    it.setBackground(QColor(codes.LEVEL_COLORS[z.level]))
                    it.setForeground(QColor(LEVEL_FG[z.level]))
                    it.setData(Qt.UserRole + 1, codes.level_rank(z.level))
                if j == 8 and z.status != "Ahead":
                    it.setForeground(QColor(theme.MUTED))
                self.table.setItem(r, j, it)
        self.table.setSortingEnabled(True)

    def _row_selected(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        zi = items[0].data(Qt.UserRole)
        self._show_zone(zi)

    def select_zone(self, zi: int) -> None:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.UserRole) == zi:
                self.table.blockSignals(True)
                self.table.selectRow(r)
                self.table.blockSignals(False)
                break
        self._show_zone(zi)

    def _show_zone(self, zi: int) -> None:
        p = self.pred
        if p is None or not (0 <= zi < len(p.zones)):
            return
        self.detail.setHtml(zone_html(p, zi))
        self.chart.show_prediction(p, zi)
        self.zone_selected.emit(p.hole_id, zi)

    # --- export -------------------------------------------------------------------------------------
    def export_csv(self) -> None:
        if self.pred is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Risk register"), f"{self.pred.hole_id}_risk_register.csv",
                                              "CSV (*.csv)")
        if path:
            register_frame(self.pred).to_csv(path, index=False, encoding="utf-8-sig")
            self.banner.show_message(tr("Risk register written to {p}.", p=path), "ok")

    def export_report(self) -> None:
        if self.pred is None:
            return
        path, flt = QFileDialog.getSaveFileName(self, tr("Risk report"), f"{self.pred.hole_id}_risk_report.html",
                                                "HTML (*.html);;PDF (*.pdf)")
        if not path:
            return
        doc = report_html(self.pred, self.chart.png_base64(), self.r_from.value(), self.r_to.value())
        if path.lower().endswith(".pdf"):
            try:
                from PySide6.QtGui import QPageSize, QTextDocument
                from PySide6.QtPrintSupport import QPrinter

                pr = QPrinter(QPrinter.HighResolution)
                pr.setOutputFormat(QPrinter.PdfFormat)
                pr.setOutputFileName(path)
                pr.setPageSize(QPageSize(QPageSize.A4))
                td = QTextDocument()
                td.setHtml(doc)
                td.print_(pr)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, tr("Risk report"), tr("PDF could not be written ({e}); save as HTML and print "
                                                               "it from the browser.", e=e))
                return
        else:
            Path(path).write_text(doc, encoding="utf-8")
        self.banner.show_message(tr("Report written to {p}.", p=path), "ok")

    def save_warnings(self) -> None:
        if self.pred is None or not self.pred.zones:
            return
        if self.mode.currentIndex() == 1:
            QMessageBox.information(self, tr("Save warnings"), tr("A plan check is not saved (the design is not a hole "
                                                                  "in the records)."))
            return
        try:
            self.engine.store.save_warnings(warning_rows(self.pred))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, tr("Save warnings"), str(e))
            return
        self.banner.show_message(tr("{n} warnings of {hole} saved with model {v}.", n=len(self.pred.zones),
                                    hole=self.pred.hole_id, v=self.pred.model_version), "ok")


def _n(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if math.isnan(x) else f"{x:g}"


def zone_html(p: Prediction, zi: int) -> str:
    """The detail panel of a zone (spec 8.2), from the reason and measure libraries only."""
    z = p.zones[zi]
    color = codes.LEVEL_COLORS[z.level]
    names = ", ".join(tr(codes.FAIL_CODES[c][0]) for c in z.codes)
    out = [f"<div style='border-left:6px solid {color}; padding-left:8px'>",
           f"<h3 style='margin:0'>{_level_chip(z.level)} &nbsp;{html.escape(names)} ({', '.join(z.codes)})</h3>",
           f"<p style='color:{theme.MUTED}'>{html.escape(p.hole_id)} · {z.from_:.0f}-{z.to:.0f} m "
           f"(±{z.u:.0f} m: {z.band_from:.0f}-{z.band_to:.0f} m) · P = {z.p:.2f} · "
           f"{tr('chance of failure here')} {100 * z.prob10:.1f}%"
           f"{(' · <b>' + tr('level set by a live alarm rule') + '</b>') if any(r.current for r in z.rules) else ''} · "
           f"{tr('Model')} {html.escape(p.model_version)} · "
           f"{tr(z.status)}</p>",
           f"<p>{tr('Formation')}: <b>{html.escape(z.formation)}</b> · {tr('cavity probability')} "
           f"{100 * z.p_cavity:.0f}% · {tr('fault-zone probability')} {100 * z.p_fault:.0f}%</p>",
           f"<h4>{tr('Why this zone is dangerous')}</h4><ol>"]
    for c, t in z.reasons:
        out.append(f"<li><b>{c}</b> {html.escape(t)}</li>" if c else f"<li>{html.escape(t)}</li>")
    out.append("</ol>")
    if z.rules:
        out.append(f"<h4>{tr('Alarm rules triggered')}</h4><ul>")
        for r in z.rules:
            out.append(f"<li>{_level_chip(r.level)} <b>{r.rule}</b> {html.escape(tr(r.condition))} "
                       f"({r.depth_from:.1f}-{r.depth_to:.1f} m{', ' + tr('current run') if r.current else ''})</li>")
        out.append("</ul>")
    out.append(f"<h4>{tr('What to do')}</h4><p style='color:{theme.MUTED}'>"
               f"{tr('Precautions start at the top of the band: {d:.0f} m.', d=z.band_from)}</p><ul>")
    for s in z.prevent:
        out.append(f"<li><b>{s.code}</b>: {html.escape(s.text)} <span style='color:{theme.MUTED}'>- "
                   f"{html.escape(s.evidence)}</span></li>")
    out.append("</ul>")
    if z.respond:
        out.append(f"<h4>{tr('If the event starts')}</h4><ul>")
        for s in z.respond:
            out.append(f"<li><b>{s.code}</b>: {html.escape(s.text)} <span style='color:{theme.MUTED}'>- "
                       f"{html.escape(s.evidence)}</span></li>")
        out.append("</ul>")
    out.append("</div>")
    return "".join(out)


def report_html(p: Prediction, png_b64: str, a: float, b: float) -> str:
    """A printable report: headline, danger in range, register, chart and every zone's detail."""
    rp = p.range_probabilities(a, b)
    reg = register_frame(p)
    zones = "".join(zone_html(p, i).replace(theme.MUTED, "#666") + "<hr>" for i in range(len(p.zones)))
    msgs = "".join(f"<li>{html.escape((c + ' ') if c else '')}{html.escape(t)}</li>" for c, t in p.messages)
    rows = "".join(f"<tr><td>{tr(codes.FAMILIES[f])}</td><td align=right>{100 * rp.get(f, 0):.1f}%</td></tr>"
                   for f in codes.FAMILIES)
    return f"""<html><head><meta charset='utf-8'><title>{html.escape(p.hole_id)} risk report</title>
<style>body {{ font-family: Segoe UI, Arial; font-size: 10pt; color: #222; }} table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #bbb; padding: 3px 6px; }} th {{ background: #eee; }}</style></head><body>
<h1>{tr('Drilling risk report')} - {html.escape(p.hole_id)}</h1>
<p><b>{html.escape(p.summary)}</b></p>
<p>{tr('Chance of reaching the target depth')}: <b>{100 * p.p_reach:.0f}%</b> · {tr('target')} {p.planned_depth:.0f} m ·
{tr('bit')} {p.drilled_depth:.0f} m · {tr('Model')} {html.escape(p.model_version)} · {html.escape(p.geomodel)}</p>
<h2>{tr('Danger from {a:.0f} m to {b:.0f} m', a=a, b=b)}</h2>
<table><tr><td>{tr('Failure (any)')}</td><td align=right>{100 * rp.get('failure', 0):.1f}%</td></tr>{rows}
<tr><td>{tr('Meeting a cavity')}</td><td align=right>{100 * rp.get('cavity', 0):.1f}%</td></tr>
<tr><td>{tr('Crossing a model fault')}</td><td align=right>{100 * rp.get('fault_crossed', 0):.0f}%</td></tr></table>
<h2>{tr('Risk register')}</h2>{reg.to_html(index=False, border=0)}
<h2>{tr('Risk log')}</h2><img src='data:image/png;base64,{png_b64}' width='640'>
<h2>{tr('Messages')}</h2><ul>{msgs}</ul>
<h2>{tr('Warning zones')}</h2>{zones}
</body></html>"""
