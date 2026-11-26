"""Route designer window: a new drive from a station to a target, designed automatically within a
gradient limit and away from old workings, other drives and (optionally) the ore body. Results as
big numbers, plain sentences, a plan, a long section and the station list. Works offline; the
design can be shown, exported and saved to the database as a designed pit (listed under Designed
Pits in the Layers list) - the survey records are not changed."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..ai import BAD, INFO
from ..ai.route_planner import (Obstacles, RouteRequest, best_start, plan_route, ratio_text, write_csv,
                                write_str)
from ..i18n import tr
from . import theme
from .ai_widgets import (AMBER, BLUE, GREEN, RED, BigNumber, FindingList, PlanSketch, ProfileChart, colored,
                         heading, hull, tonnes_text)
from .icons import icon

ROCK_DENSITY = 2.7          # t/m³, for the waste rock of the drive
DETAIL = {"fast": 10.0, "normal": None, "fine": 2.5}


def _spin(lo, hi, value, decimals=1, step=1.0, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    return s


class RouteDialog(QDialog):
    def __init__(self, manager):
        """``manager``: the Pit Manager window (stations, pits and their profiles); its parent the main window
        (mined shapes, ore solids, the 3D view)."""
        super().__init__(manager)
        self.sm = manager
        self.win = manager.parent()
        self.route = None
        self.setWindowTitle(tr("Route designer"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
        self.resize(1240, 800)
        park = getattr(self.win, "parking", None)
        if park is not None:
            park.watch(self, "survey")              # minimising hides it with the Pit Manager

        top = heading(tr("Route designer"),
                      tr("Designs a new drive (decline, ramp or access) from a station to a target: as short as "
                         "possible, never steeper than the limit, and away from old workings, other drives and - if "
                         "you want - the ore. It is worked out on this computer; the survey records are not changed."))

        # --- settings -------------------------------------------------------------------------------
        box = QGroupBox(tr("What to design"))
        f = QFormLayout(box)
        f.setLabelAlignment(Qt.AlignRight)
        coords = self._stations()
        self.c_start = QComboBox()
        self.c_target = QComboBox()
        for name in sorted(coords, key=str):
            self.c_start.addItem(name)
            self.c_target.addItem(name)
        start = self._default_start(coords)
        if start:
            self.c_start.setCurrentText(start)
        self.c_start.setToolTip(tr("The station where the new drive starts (a station with coordinates)"))
        self.r_s_station = QRadioButton(tr("a station"))
        self.r_s_best = QRadioButton(tr("best point on the drives"))
        self.r_s_best.setToolTip(tr("Try every point along the existing drives (every few metres) and start where the "
                                    "route is safe, straight if possible, and shortest"))
        sgrp = QButtonGroup(self)
        sgrp.addButton(self.r_s_station)
        sgrp.addButton(self.r_s_best)
        self.r_s_station.setChecked(True)
        srow = QHBoxLayout()
        srow.addWidget(self.r_s_station)
        srow.addWidget(self.r_s_best)
        srow.addStretch(1)
        f.addRow(tr("Start at"), srow)
        # a station - or, for the best point, where to look: all pits or the pits ticked
        self.c_scope = QComboBox()
        self.c_scope.addItem(tr("on any pit"), False)
        self.c_scope.addItem(tr("on the pits ticked below"), True)
        self.c_scope.setToolTip(tr("Where the start may be: anywhere on the existing drives, or only on the pits you "
                                   "tick (all drives stay obstacles to keep away from)"))
        self.pit_list = QListWidget()
        self.pit_list.setMaximumHeight(118)
        self.pit_list.setToolTip(tr("Ticking a pit ticks its branches too; untick any you do not want"))
        self._fill_pit_list()
        self.pit_list.itemChanged.connect(self._pit_ticked)
        best_page = QWidget()
        bp = QVBoxLayout(best_page)
        bp.setContentsMargins(0, 0, 0, 0)
        bp.addWidget(self.c_scope)
        bp.addWidget(self.pit_list)
        self.pit_list.setVisible(False)
        self.c_scope.currentIndexChanged.connect(lambda _i: self.pit_list.setVisible(bool(self.c_scope.currentData())))
        self.start_pages = QStackedWidget()
        self.start_pages.addWidget(self.c_start)
        self.start_pages.addWidget(best_page)
        self.start_pages.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.r_s_best.toggled.connect(lambda on: self.start_pages.setCurrentIndex(1 if on else 0))
        f.addRow("", self.start_pages)
        self.c_dir = QComboBox()
        self.c_dir.addItem(tr("Automatic"), False)
        self.c_dir.addItem(tr("Set (bearing)"), True)
        self.c_dir.setToolTip(tr("The direction the new drive leaves the start: automatic (the search chooses), or a "
                                 "bearing you set - degrees from north, clockwise (90 = east)"))
        self.s_bearing = _spin(0.0, 359.9, 0.0, 1, 5.0, "°")
        self.s_bearing.setWrapping(True)
        self.s_bearing.setEnabled(False)
        self.c_dir.currentIndexChanged.connect(lambda _i: self.s_bearing.setEnabled(bool(self.c_dir.currentData())))
        drow = QHBoxLayout()
        drow.addWidget(self.c_dir, 1)
        drow.addWidget(self.s_bearing, 1)
        f.addRow(tr("Start direction"), drow)
        self.r_station = QRadioButton(tr("a station"))
        self.r_point = QRadioButton(tr("a point (X, Y, Z)"))
        grp = QButtonGroup(self)
        grp.addButton(self.r_station)
        grp.addButton(self.r_point)
        self.r_point.setChecked(True)
        tgt = QHBoxLayout()
        tgt.addWidget(self.r_point)
        tgt.addWidget(self.r_station)
        f.addRow(tr("Go to"), tgt)
        f.addRow("", self.c_target)
        s0 = coords.get(self.c_start.currentText(), np.zeros(3))
        self.tx = _spin(-1e8, 1e8, float(s0[0]) + 200.0, 2, 10.0)
        self.ty = _spin(-1e8, 1e8, float(s0[1]), 2, 10.0)
        self.tz = _spin(-1e5, 1e5, float(s0[2]) - 20.0, 2, 1.0)
        for w, lab in ((self.tx, tr("Target X (East)")), (self.ty, tr("Target Y (North)")),
                       (self.tz, tr("Target Z (Elevation)"))):
            f.addRow(lab, w)
        self.r_point.toggled.connect(self._target_mode)
        self.s_grad = _spin(3.0, 30.0, 7.0, 1, 0.5)
        self.s_grad.setPrefix(tr("1 in "))
        self.l_grad = QLabel()
        self.s_grad.valueChanged.connect(lambda v: self.l_grad.setText(f"({100.0 / v:.1f} %)"))
        self.l_grad.setText(f"({100.0 / self.s_grad.value():.1f} %)")
        gr = QHBoxLayout()
        gr.addWidget(self.s_grad, 1)
        gr.addWidget(self.l_grad)
        self.s_grad.setToolTip(tr("Steepest slope allowed for the drive: 1 in 7 means 1 m up or down for every 7 m "
                                  "along (14.3 %)"))
        f.addRow(tr("Steepest gradient"), gr)
        self.s_void = _spin(0.0, 500.0, 10.0, 1, 1.0, " m")
        self.s_void.setToolTip(tr("Rock to leave between the new drive and old workings (the mined shapes), wall to "
                                  "wall"))
        f.addRow(tr("Keep away from old workings"), self.s_void)
        self.s_drive = _spin(0.0, 500.0, 10.0, 1, 1.0, " m")
        self.s_drive.setToolTip(tr("Rock to leave between the new drive and the existing drives, wall to wall"))
        f.addRow(tr("Keep away from other drives"), self.s_drive)
        self.c_ore = QComboBox()
        self.c_ore.addItem(tr("(not used)"), "")
        for lay in self._solids():
            self.c_ore.addItem(lay.name, lay.name)
        self.c_ore.setToolTip(tr("The ore body (a closed solid). The route can stay outside it - e.g. so that a "
                                 "decline does not sterilise ore"))
        f.addRow(tr("Ore body"), self.c_ore)
        self.k_ore = QCheckBox(tr("stay outside the ore by"))
        self.s_ore = _spin(0.0, 500.0, 15.0, 1, 1.0, " m")
        ore_row = QHBoxLayout()
        ore_row.addWidget(self.k_ore)
        ore_row.addWidget(self.s_ore, 1)
        f.addRow("", ore_row)
        w, h = self._profile()
        self.s_w = _spin(1.0, 30.0, w, 1, 0.5, " m")
        self.s_h = _spin(1.0, 30.0, h, 1, 0.5, " m")
        size = QHBoxLayout()
        size.addWidget(self.s_w)
        size.addWidget(QLabel("×"))
        size.addWidget(self.s_h)
        f.addRow(tr("Drive size (W × H)"), size)
        from ..io.survey_store import STATION_LEVELS

        self.c_level = QComboBox()
        for key in ("floor", "middle", "back"):
            self.c_level.addItem(tr(STATION_LEVELS[key]), key)
        self.c_level.setCurrentIndex(max(self.c_level.findData(self._station_level()), 0))
        self.c_level.setToolTip(tr("Where the route's points (the survey stations) are in the drive's profile - as "
                                   "for the existing pits (Pit Manager: the pit's profile). Used to draw the drive in the "
                                   "3D view"))
        self.c_level.currentIndexChanged.connect(lambda _i: self.b_show.isChecked() and self._show_3d(True))
        f.addRow(tr("Stations in"), self.c_level)
        self.s_radius = _spin(5.0, 200.0, 15.0, 0, 5.0, " m")
        self.s_radius.setToolTip(tr("Smallest radius of a bend: trucks need room to turn, and the legs of a loop "
                                    "need rock between them"))
        f.addRow(tr("Smallest bend radius"), self.s_radius)
        self.s_cost = _spin(0.0, 1e7, 0.0, 0, 100.0)
        self.s_cost.setToolTip(tr("Development cost per metre of drive (any currency); 0 leaves the cost out"))
        f.addRow(tr("Cost per metre"), self.s_cost)
        self.c_detail = QComboBox()
        for key, text in (("normal", tr("Normal")), ("fast", tr("Fast (rough)")), ("fine", tr("Fine (slower)"))):
            self.c_detail.addItem(text, key)
        self.c_detail.setToolTip(tr("How fine the search grid is: Fine follows tight spaces better but takes longer"))
        f.addRow(tr("Search detail"), self.c_detail)
        self.b_run = QPushButton(icon("map-marker-path", "#111"), tr("Design the route"))
        self.b_run.setStyleSheet(f"QPushButton {{ background: {GREEN}; color: #111; font-weight: 700; "
                                 f"padding: 8px; border-radius: 4px; }}")
        self.b_run.clicked.connect(self.run)
        self.busy = QLabel()
        self.busy.setStyleSheet(f"color: {theme.MUTED};")
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(box)
        lv.addWidget(self.b_run)
        lv.addWidget(self.busy)
        lv.addStretch(1)
        self._target_mode(True)

        # --- results ------------------------------------------------------------------------------------
        cards = QHBoxLayout()
        self.c_len = BigNumber(tr("Drive length"), GREEN)
        self.c_dz = BigNumber(tr("Height change"), BLUE)
        self.c_grad = BigNumber(tr("Steepest part"), AMBER)
        self.c_clear = BigNumber(tr("Closest to old workings"), RED)
        self.c_cost = BigNumber(tr("Estimated cost"), "#c49bff")
        for c in (self.c_len, self.c_dz, self.c_grad, self.c_clear, self.c_cost):
            cards.addWidget(c)
        self.findings = FindingList()
        self.plan = PlanSketch(min_h=320)
        self.profile = ProfileChart()
        self.stations = QTableWidget(0, 6)
        self.stations.setHorizontalHeaderLabels([tr("Station"), "X", "Y", "Z", tr("Leg length (m)"),
                                                 tr("Gradient")])
        self.stations.verticalHeader().setVisible(False)
        self.stations.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stations.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.plan, icon("map-outline"), tr("Plan (from above)"))
        self.tabs.addTab(self.profile, icon("chart-line"), tr("Long section"))
        self.tabs.addTab(self.stations, icon("table"), tr("Stations"))
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.direct = QLabel()
        self.direct.setWordWrap(True)
        self.direct.setStyleSheet("font-size: 10pt;")
        hint = QLabel(tr("Map: wheel zooms, drag moves it, double-click shows everything, a click puts the target "
                         "there."))
        hint.setStyleSheet(f"color: {theme.MUTED};")
        from PySide6.QtWidgets import QScrollArea

        self.find_box = QScrollArea()
        self.find_box.setWidgetResizable(True)
        self.find_box.setFrameShape(QScrollArea.NoFrame)
        self.find_box.setWidget(self.findings)
        self.find_box.setMinimumHeight(118)
        self.find_box.setMaximumHeight(190)
        rv.addLayout(cards)
        rv.addWidget(self.find_box)
        rv.addWidget(self.direct)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(hint)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([380, 860])

        self.b_show = QPushButton(icon("cube-outline"), tr("Show in the 3D view"))
        self.b_show.setCheckable(True)
        self.b_show.setToolTip(tr("Draw the designed drive in the main view (see-through, green)"))
        self.b_show.toggled.connect(self._show_clicked)
        self.b_save = QPushButton(icon("database-plus-outline"), tr("Save to database ..."))
        self.b_save.setToolTip(tr("Keep the design in the project database as a designed pit: it is listed under "
                                  "Designed Pits in the Layers list and drawn in the 3D view (the survey records are "
                                  "not changed)"))
        self.b_save.clicked.connect(self.save_design)
        self.b_str = QPushButton(icon("file-export-outline"), tr("Export Surpac string ..."))
        self.b_str.setToolTip(tr("Save the centre line of the design as a Surpac .str file (for Surpac design)"))
        self.b_str.clicked.connect(self.export_str)
        self.b_csv = QPushButton(icon("table-arrow-down"), tr("Export stations (CSV) ..."))
        self.b_csv.clicked.connect(self.export_csv)
        self.b_png = QPushButton(icon("image-outline"), tr("Export map PNG ..."))
        self.b_png.setToolTip(tr("Save the map as shown (same area and zoom) as a 300 dpi PNG on white"))
        self.b_png.clicked.connect(self.export_png)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        for b in (self.b_show, self.b_save, self.b_str, self.b_csv, self.b_png):
            bottom.addWidget(b)
        bottom.addStretch(1)
        bottom.addWidget(close)

        root = QVBoxLayout(self)
        root.addWidget(top)
        root.addWidget(split, 1)
        root.addLayout(bottom)
        self.stale = False
        self._results(False)
        # the map follows the inputs at once
        for w in (self.tx, self.ty, self.tz, self.s_grad):
            w.valueChanged.connect(self._inputs_changed)
        self.c_target.currentIndexChanged.connect(self._inputs_changed)
        self.c_start.currentIndexChanged.connect(self._inputs_changed)
        self.r_s_best.toggled.connect(self._inputs_changed)
        self.r_s_best.toggled.connect(lambda _on: self._draw_map(fit=False))
        self.c_scope.currentIndexChanged.connect(self._inputs_changed)
        self.c_scope.currentIndexChanged.connect(lambda _i: self._draw_map(fit=False))
        self.c_dir.currentIndexChanged.connect(self._inputs_changed)
        self.s_bearing.valueChanged.connect(self._inputs_changed)
        self.r_point.toggled.connect(self._inputs_changed)
        self.c_ore.currentIndexChanged.connect(lambda _i: self._draw_map(fit=False))
        self.plan.clicked.connect(self._map_clicked)
        self._draw_map(fit=True)

    # --- data from the windows ---------------------------------------------------------------------
    def _stations(self) -> dict:
        net = self.sm.net
        return {n: np.asarray(xyz, float) for n, xyz in (net.coords.items() if net is not None else [])}

    def _default_start(self, coords) -> str | None:
        rows = [r for r in self.sm.rows if r.fs in coords] if self.sm.pid is not None else []
        return rows[-1].fs if rows else (next(iter(sorted(coords, key=str)), None))

    def _station_level(self) -> str:
        """Where the stations are in the profile of the selected pit (floor, middle or back)."""
        p = next((q for q in self.sm.pits if q.id == self.sm.pid), None)
        return getattr(p, "station_level", None) or "middle"

    def _profile(self) -> tuple[float, float]:
        p = next((q for q in self.sm.pits if q.id == self.sm.pid), None)
        return (p.profile_width, p.profile_height) if p else (5.0, 5.0)

    def _solids(self) -> list:
        from ..scene.reserve import Reserve

        scene = getattr(self.win, "scene", None)
        return Reserve.solid_layers(scene.layers) if scene is not None else []

    def _voids(self) -> list:
        mgr = getattr(self.win, "reserve", None)
        log = getattr(getattr(mgr, "reserve", None), "log", None)
        return list(log.shapes) if log is not None else []

    def _paths(self) -> list:
        """Every pit's centre lines: (points, station names, pit)."""
        from ..scene.survey import pit_paths

        coords = self._stations()
        setups = dict(self.sm.setups)
        if self.sm.pid is not None:
            setups[self.sm.pid] = self.sm.rows
        out = []
        for pit in self.sm.pits:
            for path in pit_paths({pit.id: setups.get(pit.id, [])}).get(pit.id, []):
                names = [n for n in path if n in coords]
                if len(names) >= 2:
                    out.append((np.array([coords[n] for n in names]), names, pit))
        return out

    def _drives(self) -> list:
        return [(pts, 0.5 * math.hypot(pit.profile_width, pit.profile_height), pit.name)
                for pts, _names, pit in self._paths()]

    def _target_mode(self, point: bool) -> None:
        for w in (self.tx, self.ty, self.tz):
            w.setEnabled(point)
        self.c_target.setEnabled(not point)

    # --- design ------------------------------------------------------------------------------------------
    def _fill_pit_list(self) -> None:
        """The pits as a tree (branches indented), the Pit Manager's selected pit and its branches ticked."""
        pits = list(self.sm.pits)
        ids = {p.id for p in pits}
        kids = {}
        for p in pits:                                   # a branch whose parent is missing: at the top
            kids.setdefault(p.parent_id if p.parent_id in ids else None, []).append(p)
        chosen = self._family(self.sm.pid) if self.sm.pid is not None else set()

        def add(parent, depth):
            for p in kids.get(parent, []):
                it = QListWidgetItem("    " * depth + (f"{p.name}  ({p.code})" if p.code else p.name))
                it.setData(Qt.UserRole, p.id)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if p.id in chosen else Qt.Unchecked)
                self.pit_list.addItem(it)
                add(p.id, depth + 1)

        add(None, 0)

    def _family(self, pid) -> set:
        out, todo = set(), [pid]
        while todo:
            q = todo.pop()
            if q not in out:
                out.add(q)
                todo += [p.id for p in self.sm.pits if p.parent_id == q]
        return out

    def _pit_ticked(self, item) -> None:
        """A pit ticked / unticked: its branches follow."""
        fam = self._family(item.data(Qt.UserRole)) - {item.data(Qt.UserRole)}
        self.pit_list.blockSignals(True)
        for i in range(self.pit_list.count()):
            it = self.pit_list.item(i)
            if it.data(Qt.UserRole) in fam:
                it.setCheckState(item.checkState())
        self.pit_list.blockSignals(False)
        self._inputs_changed()
        self._draw_map(fit=False)

    def _search_pits(self) -> set | None:
        """The pits the best start may be on (None: any)."""
        if not self.c_scope.currentData():
            return None
        return {self.pit_list.item(i).data(Qt.UserRole) for i in range(self.pit_list.count())
                if self.pit_list.item(i).checkState() == Qt.Checked}

    def _bearing(self) -> float | None:
        """The start direction set (degrees from north), or None for automatic."""
        return self.s_bearing.value() if self.c_dir.currentData() else None

    def _best_mode(self) -> bool:
        return self.r_s_best.isChecked()

    def request(self) -> RouteRequest | None:
        coords = self._stations()
        start = coords.get(self.c_start.currentText())
        if self.r_station.isChecked():
            target = coords.get(self.c_target.currentText())
        else:
            target = np.array([self.tx.value(), self.ty.value(), self.tz.value()])
        if self._best_mode():
            start = target                      # chosen along the drives by best_start
        if start is None:
            return None
        if self.r_station.isChecked():
            target = coords.get(self.c_target.currentText())
            tname = self.c_target.currentText()
        else:
            target = np.array([self.tx.value(), self.ty.value(), self.tz.value()])
            tname = tr("target")
        if target is None:
            return None
        return RouteRequest(tuple(start), tuple(target), 1.0 / self.s_grad.value(), self.s_void.value(),
                            self.s_drive.value(), self.s_ore.value() if self.k_ore.isChecked() else None,
                            self.s_w.value(), self.s_h.value(), DETAIL[self.c_detail.currentData()],
                            "" if self._best_mode() else self.c_start.currentText(), tname,
                            min_radius=self.s_radius.value(), start_bearing=self._bearing())

    def obstacles(self, req: RouteRequest) -> Obstacles:
        from ..scene.solid import closed_mesh

        drives = [(pts, hs, name) for pts, hs, name in self._drives()]
        ore = None
        name = self.c_ore.currentData()
        if name:
            lay = next((s for s in self._solids() if s.name == name), None)
            ore = closed_mesh(lay.data) if lay is not None else None
        return Obstacles(self._voids(), drives, ore)

    def run(self) -> None:
        if self._best_mode() and self._search_pits() == set():
            QMessageBox.information(self, tr("Route designer"), tr("Tick at least one pit to look for the start on."))
            return
        req = self.request()
        if req is None:
            QMessageBox.information(self, tr("Route designer"), tr("Choose a start station with coordinates (and a "
                                                                   "target)."))
            return
        self.busy.setText(tr("Designing the route ... (this can take a few seconds)"))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            obs = self.obstacles(req)
            self.obs = obs
            paths = self._paths()
            self._run_paths = paths
            if self._best_mode():
                lines = [(pts, names, pit.name) for pts, names, pit in paths]
                pits = self._search_pits()
                only = None if pits is None else {k for k, (_p, _n, pit) in enumerate(paths) if pit.id in pits}
                self.route = best_start(req, obs, lines, only)
            else:
                self.route = plan_route(req, obs)
        except Exception as e:  # noqa: BLE001
            self.route = None
            QApplication.restoreOverrideCursor()
            self.busy.setText("")
            QMessageBox.warning(self, tr("Route designer"), tr("The route could not be designed:\n{e}", e=e))
            return
        QApplication.restoreOverrideCursor()
        if getattr(self.route, "start_point", None) is not None:
            self.busy.setText(tr("{n:,} start points along the drives compared", n=self.route.compared))
        elif self.route.explored:
            self.busy.setText(tr("Search grid {c:.1f} m, {n:,} positions looked at", c=self.route.cell,
                                 n=self.route.explored))
        else:
            self.busy.setText("")
        self._fill()

    def _results(self, on: bool) -> None:
        for w in (self.c_len, self.c_dz, self.c_grad, self.c_clear, self.c_cost):
            w.setVisible(on)
        self.find_box.setVisible(on or (self.route is not None and not self.route.ok))
        self.direct.setVisible(not on or self.stale)
        for b in (self.b_show, self.b_save, self.b_str, self.b_csv):
            b.setEnabled(on)

    def _fill(self) -> None:
        r = self.route
        if r is None or not r.ok:
            self._results(False)
            self.findings.set_findings([(BAD, r.message if r else tr("No route"))] +
                                       ([(INFO, tr("For this height difference the drive must be at least {n:,.0f} m "
                                                   "long (the direct line is {d:,.0f} m).", n=r.needed, d=r.direct))]
                                        if r and r.needed > r.direct else []))
            self.b_show.setChecked(False)
            return
        req = r.request
        self.stale = False
        self._results(True)
        self.c_len.set(f"{r.length:,.0f} m", tr("{n} straight legs · direct line {d:,.0f} m", n=len(r.legs),
                                                 d=r.direct))
        self.c_dz.set(f"{r.climb:+,.1f} m", tr("up") if r.climb > 0 else (tr("down") if r.climb < 0 else tr("level")))
        ok = r.steepest <= req.gradient * 1.001
        self.c_grad.set(ratio_text(r.steepest).split(" (")[0], tr("limit {g}", g=ratio_text(req.gradient)),
                        GREEN if ok else RED)
        if r.void_clear is None:
            self.c_clear.set(tr("none near"), tr("no old workings close to the route"), GREEN)
        else:
            good = r.void_clear >= req.void_clearance - 0.5
            self.c_clear.set(f"{max(r.void_clear, 0):,.0f} m", tr("asked: {m:.0f} m", m=req.void_clearance),
                             GREEN if good else RED)
        waste = r.length * req.width * req.height * ROCK_DENSITY
        if self.s_cost.value() > 0:
            self.c_cost.set(f"{r.length * self.s_cost.value():,.0f}",
                            tr("{m:,.0f} m × {c:,.0f} per metre; waste rock {t}", m=r.length, c=self.s_cost.value(),
                               t=tonnes_text(waste)))
        else:
            self.c_cost.set(tonnes_text(waste), tr("waste rock to break (no cost per metre given)"))
        self.findings.set_findings(r.findings)
        self._draw_plan()
        self.profile.set_route(r.legs, req.gradient)
        t = self.stations
        t.setRowCount(len(r.points))
        for k, p in enumerate(r.points):
            lg = r.legs[k - 1] if k else None
            vals = [f"R-{k + 1:02d}", f"{p[0]:,.2f}", f"{p[1]:,.2f}", f"{p[2]:,.2f}",
                    "" if lg is None else f"{lg.length:,.1f}", "" if lg is None else ratio_text(lg.gradient)]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(k, c, it)
        if self.b_show.isChecked():
            self._show_3d(True)

    # --- the map ----------------------------------------------------------------------------------------
    def _ends(self):
        """(start, target) as now chosen (None when not available; the best start once designed)."""
        coords = self._stations()
        start = coords.get(self.c_start.currentText())
        if self._best_mode():
            r = self.route
            start = (r.points[0] if r is not None and r.ok and getattr(r, "start_point", None) is not None
                     and not self.stale else None)
        if self.r_station.isChecked():
            target = coords.get(self.c_target.currentText())
        else:
            target = np.array([self.tx.value(), self.ty.value(), self.tz.value()])
        return start, target

    def _draw_map(self, fit: bool = False) -> None:
        """The pits with their stations, old workings, the ore body and (when designed) the route."""
        p = self.plan
        p.clear()
        name = self.c_ore.currentData()
        lay = next((s for s in self._solids() if s.name == name), None) if name else None
        if lay is not None:
            from ..scene.solid import closed_mesh

            mesh = closed_mesh(lay.data)
            if mesh is not None:
                v = np.asarray(mesh[0], float)[np.unique(np.asarray(mesh[1]).ravel())]
                p.polygon(hull(v), "#5a4a1a", "#c9a227")
        for sh in self._voids():
            if sh.plane == "xy" and not sh.is_solid:
                poly = sh.outline()
            else:
                lo3, hi3 = sh.bounds()
                poly = np.array([[lo3[0], lo3[1]], [hi3[0], lo3[1]], [hi3[0], hi3[1]], [lo3[0], hi3[1]]])
            p.polygon(poly, "#6a2a2a", RED)
        seen = set()
        search = self._search_pits() if self._best_mode() else None
        for pts, names, pit in self._paths():
            if search is not None and pit.id in search:       # where the best start is looked for
                p.line(pts, "#5b8fd6", 6.0)
            p.line(pts, "#8a94a3", 3.0)
            keep = [k for k, n in enumerate(names) if n not in seen]
            seen.update(names)
            p.station_marks(pts[keep], [names[k] for k in keep], "#c8cdd4")
        r = self.route
        if r is not None and r.ok:
            pts = r.points
            every = max(1, int(math.ceil(len(pts) / 10)))
            p.line(pts, GREEN, 4.0, Qt.SolidLine, [str(k + 1) if k % every == 0 and 0 < k < len(pts) - 1 else ""
                                                   for k in range(len(pts))])
        p.legend = [(GREEN, tr("new route"), "line"), ("#8a94a3", tr("existing drives"), "line")] + \
            ([("#5b8fd6", tr("start looked for here"), "line")] if search is not None else []) + [
                    ("#c8cdd4", tr("stations"), "dot"), ("#6a2a2a", tr("old workings"), "fill"),
                    ("#5a4a1a", tr("ore body"), "fill")]
        self._update_pins()
        if fit:
            p.fit()

    def _update_pins(self) -> None:
        """Start (A) and target (B) on the map, the straight line between them, and what it means."""
        start, target = self._ends()
        pins = []
        a_name = self.route.request.start_name if self._best_mode() and start is not None else \
            self.c_start.currentText()
        if start is not None:
            pins.append((start, BLUE, "A  " + a_name, "dot"))
        self.plan.set_arrow(start, self._bearing())
        if target is not None:
            tname = self.c_target.currentText() if self.r_station.isChecked() else tr("target")
            pins.append((target, RED, "B  " + tname, "target"))
        self.plan.set_pins(pins)
        self.plan.set_guide(start, target)
        if start is None and target is not None and self._best_mode():
            self.direct.setText(tr("The start point will be chosen along the existing drives when you click Design "
                                   "the route."))
            return
        if start is None or target is None:
            self.direct.setText("")
            return
        d = math.hypot(*(target[:2] - start[:2]))
        h = float(target[2] - start[2])
        limit = 1.0 / self.s_grad.value()
        g = abs(h) / d if d > 1e-6 else math.inf
        head = tr("Straight line A → B: {d:,.0f} m, height {h:+,.1f} m", d=d, h=h)
        if math.isfinite(g):
            head += " = " + ratio_text(g)
        if g <= limit * 1.001:
            text = head + " - " + colored(tr("within the limit of {limit}: a straight drive is possible if nothing "
                                             "is in the way.", limit=ratio_text(limit)), GREEN)
        else:
            text = head + " - " + colored(tr("too steep for {limit}: the drive must be at least {n:,.0f} m long, so "
                                             "it will wind.", limit=ratio_text(limit), n=abs(h) / limit), AMBER)
        self.direct.setText(text)

    def _inputs_changed(self, *_a) -> None:
        self._update_pins()
        if self.route is not None and not self.stale:
            self.stale = True
            self.direct.setVisible(True)
            self.busy.setText(tr("The start or the target changed - click Design the route to update it."))

    def _map_clicked(self, x: float, y: float) -> None:
        """A click on the map puts the target there (X, Y; the elevation stays)."""
        if not self.r_point.isChecked():
            self.r_point.setChecked(True)
        self.tx.setValue(x)
        self.ty.setValue(y)

    def _draw_plan(self) -> None:
        """The map with the route, zoomed to the route."""
        self._draw_map(fit=False)
        pts = self.route.points
        lo, hi = pts.min(axis=0)[:2], pts.max(axis=0)[:2]
        pad = max(0.35 * max(hi[0] - lo[0], hi[1] - lo[1], 50.0), 40.0)
        self.plan.focus = (lo - pad, hi + pad)       # kept while the window is resized; "show everything"
        self.plan.fit()                              # (button / double-click) clears it

    def export_png(self) -> None:
        import datetime as _dt

        if not self._allowed(tr("Export map")):
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Export map"), str(Path(self.sm._export_folder()) /
                                                                         "route_plan.png"), tr("PNG image (*.png)"))
        if not path:
            return
        path = Path(path).with_suffix(".png")
        r = self.route
        a = r.request.start_name if r is not None and r.ok and r.request.start_name else self.c_start.currentText()
        title = tr("Route design - from {a} to {b}   {date}", a=a,
                   b=self.c_target.currentText() if self.r_station.isChecked() else tr("target"),
                   date=_dt.date.today().isoformat())
        r = self.route
        if r is not None and r.ok:
            title += "   " + tr("{m:,.0f} m, steepest {g}", m=r.length, g=ratio_text(r.steepest))
        try:
            w, h = self.plan.save_png(path, title)
        except OSError as e:
            QMessageBox.warning(self, tr("Export map"), str(e))
            return
        self.busy.setText(tr("Exported {name} ({w} x {h} px, 300 dpi)", name=path.name, w=w, h=h))

    # --- show / export ----------------------------------------------------------------------------------
    def _show_clicked(self, on: bool) -> None:
        """Show in the 3D view: draw the drive, frame it in the main view and step aside (with the Pit
        Manager) - everything is kept; the Pit Manager button brings both back."""
        self._show_3d(on)
        if not on or self.route is None or not self.route.ok:
            return
        win = self.win
        if hasattr(win, "show_in_view") and getattr(win.scene, "origin", None) is not None:
            P = np.asarray(self.route.points, float)
            win.show_in_view(P.min(axis=0), P.max(axis=0))
        park = getattr(win, "parking", None)
        if park is not None:
            park.park("survey")

    def _show_3d(self, on: bool) -> None:
        scene = getattr(self.win, "scene", None)
        if scene is None:
            return
        if not on or self.route is None or not self.route.ok or scene.origin is None:
            scene.clear_preview_solid()
            return
        import pyvista as pv

        from ..scene.survey import drive_mesh

        req = self.route.request
        V, quads = drive_mesh(self.route.points, req.width, req.height, self.c_level.currentData() or "middle")
        faces = np.hstack([np.full((len(quads), 1), 4), quads]).ravel()
        mesh = pv.PolyData(V - np.asarray(scene.origin, float), faces=faces)
        scene.set_preview_solid(mesh, color=(0.3, 0.9, 0.45))

    # --- save to the database ---------------------------------------------------------------------
    def _start_pit(self):
        """The surveyed pit the design starts on (None: not known)."""
        r = self.route
        sp = getattr(r, "start_point", None)
        paths = getattr(self, "_run_paths", None) or []
        if sp is not None:
            return paths[sp.line][2].id if 0 <= sp.line < len(paths) else None
        name = r.request.start_name
        st = next((s for s in self.sm.store.stations() if s.name == name), None) if name else None
        if st is not None and st.pit_id is not None:
            return st.pit_id
        return next((pit.id for _p, names, pit in paths if name in names), None)

    def _target_text(self) -> str:
        if self.r_station.isChecked():
            return self.c_target.currentText()
        t = self.route.request.target
        return f"X {t[0]:.2f}  Y {t[1]:.2f}  Z {t[2]:.2f}"

    def save_design(self) -> None:
        """The design as a designed pit in the project database (named by the user)."""
        from ..io.survey_store import DesignPit

        r = self.route
        if r is None or not r.ok:
            return
        title = tr("Save to database")
        if self.stale and QMessageBox.question(
                self, title, tr("The start or the target changed after this route was designed.\n"
                                "Save the route as designed anyway?")) != QMessageBox.Yes:
            return
        store = self.sm.store
        try:
            default = store.next_design_name("D")
        except Exception:  # noqa: BLE001 - no table yet: made when saving
            default = "D1"
        name, ok = QInputDialog.getText(self, title, tr("Name of the designed pit:"), text=default)
        if not ok or not name.strip():
            return
        req = r.request
        ore = self.c_ore.currentData() or ""
        d = DesignPit(0, name.strip(), [tuple(float(v) for v in p) for p in np.asarray(r.points, float)],
                      req.width, req.height, self.c_level.currentData() or "middle", self._start_pit(),
                      req.start_name or self.c_start.currentText(), self._target_text(), req.gradient,
                      float(r.steepest), float(r.length), float(r.climb),
                      {"void_clearance": req.void_clearance, "drive_clearance": req.drive_clearance,
                       "ore_body": ore, "ore_standoff": req.ore_standoff, "min_radius": req.min_radius,
                       "start_bearing": req.start_bearing, "best_start": self._best_mode(),
                       "cost_per_m": self.s_cost.value(), "detail": self.c_detail.currentData()})
        try:
            store.add_design(d)
        except Exception as e:  # noqa: BLE001 - a name in use, the database not reachable
            QMessageBox.warning(self, title, str(e))
            return
        if self.b_show.isChecked():          # the saved drive is drawn by the survey layer now
            self.b_show.setChecked(False)
        self.sm.changed.emit()               # the survey layer is read again: Designed Pits
        self.sm.reload_designs()             # and the Pit Manager lists it
        self.busy.setText(tr("Saved to the database as {name} (Layers: Designed Pits)", name=d.name))

    def _allowed(self, title: str) -> bool:
        return self.sm._export_allowed(title)

    def export_str(self) -> None:
        if self.route is None or not self._allowed(tr("Export route")):
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Export route"), str(Path(self.sm._export_folder()) /
                                                                            "route_design.str"),
                                              tr("Surpac string (*.str)"))
        if path:
            try:
                write_str(Path(path).with_suffix(".str"), self.route.points, "R")
            except OSError as e:
                QMessageBox.warning(self, tr("Export route"), str(e))
                return
            self.busy.setText(tr("Saved {name}", name=Path(path).with_suffix(".str").name))

    def export_csv(self) -> None:
        if self.route is None or not self._allowed(tr("Export route")):
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Export route"), str(Path(self.sm._export_folder()) /
                                                                            "route_design.csv"),
                                              tr("CSV file (*.csv)"))
        if path:
            try:
                write_csv(Path(path).with_suffix(".csv"), self.route, "R")
            except OSError as e:
                QMessageBox.warning(self, tr("Export route"), str(e))
                return
            self.busy.setText(tr("Saved {name}", name=Path(path).with_suffix(".csv").name))

    def closeEvent(self, e) -> None:
        if self.b_show.isChecked():
            self._show_3d(False)
        super().closeEvent(e)

