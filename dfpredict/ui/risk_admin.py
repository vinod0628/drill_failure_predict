"""Model administration windows of the drilling risk system.

* ``ModelsDialog`` - train Models H / A / B from the drilling records, see what they were trained on and how
  they score on held-out areas (spec 5.9, the gate of spec 10 phase 3), the tuned alarm thresholds, and the
  saved versions (spec 7.4: a version on every model; choose which one is used).
* ``GeoModelDialog`` - the workspace's geological model (after 3D MPM Studio's layer roles): the role of
  every surface layer, the contacts in stratigraphic order with the names of the units above and below,
  whether each unit has a formation card, and the error band settings (spec 5.6).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..risk import codes, models
from ..risk.geomodel import DEFAULTS, load_config, save_config, unit_pair
from ..scene.geo_roles import ROLES
from . import theme
from .icons import icon
from .risk_widgets import Banner, run_job

GATE_AUC = 0.75
GATE_RECALL = 0.70


def _f(v, digits=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{digits}f}"


def _model_kind(ms, v, fam) -> str:
    if ms.A.get(fam) is None:
        return tr("base rate + rules (too few failures)")
    auc = (v.get("auc_family") or {}).get(fam)
    if auc is not None and auc < models.WEAK_AUC:
        return tr("own model (weak: warns on its top 2 % only, rules carry it)")
    return tr("own model")


class ModelsDialog(QDialog):
    trained = Signal()

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Risk Models - training and validation"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.engine = engine
        v = QVBoxLayout(self)
        self.banner = Banner(self)
        v.addWidget(self.banner)
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel(tr("Saved versions (double-click to use)")))
        self.versions = QListWidget()
        self.versions.itemDoubleClicked.connect(self._use_version)
        lv.addWidget(self.versions, 1)
        split.addWidget(left)
        self.report = QTextBrowser()
        split.addWidget(self.report)
        split.setSizes([260, 760])
        v.addWidget(split, 1)
        bar = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.hide()
        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {theme.MUTED};")
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.status, 2)
        self.b_train = QPushButton(icon("brain"), tr("Train now"))
        self.b_train.setToolTip(tr("Train Models H, A and B on every finished hole of the records, with leave-one-area-"
                                   "out validation (a few minutes)"))
        self.b_train.clicked.connect(self.train)
        bar.addWidget(self.b_train)
        b = QPushButton(tr("Close"))
        b.clicked.connect(self.accept)
        bar.addWidget(b)
        v.addLayout(bar)
        self.resize(1080, 760)
        self.refresh()

    def refresh(self) -> None:
        self.versions.clear()
        cur = self.engine.models
        if self.engine.folder is not None:
            for ver, name in models.list_versions(self.engine.folder):
                it = QListWidgetItem(icon("check-circle" if cur is not None and cur.version == ver else "circle-outline"),
                                     ver)
                it.setData(Qt.UserRole, name)
                self.versions.addItem(it)
        self.report.setHtml(self._report(cur))

    def _report(self, ms) -> str:
        if ms is None:
            msg = tr("Press Train now. The models learn from every finished hole (success or failed) of the drilling "
                     "records.")
            return f"<h2>{tr('No trained models')}</h2><p>{msg}</p>"
        t = ms.trained_on
        v = ms.validation or {}

        def gate(val, limit):
            if val is None:
                return f"<span style='color:{theme.MUTED}'>-</span>"
            ok = val >= limit
            return f"<b style='color:{'#3fb950' if ok else '#e3b341'}'>{val:.2f} {'✓' if ok else '⚠'}</b>"
        fails = t.get("failures", {})
        rows = "".join(
            f"<tr><td>{tr(codes.FAMILIES[f])}</td><td align=right>{fails.get(f, 0)}</td>"
            f"<td>{_model_kind(ms, v, f)}</td>"
            f"<td align=right>{gate((v.get('auc_family') or {}).get(f), GATE_AUC)}</td>"
            f"<td align=right>{gate((v.get('recall_orange') or {}).get(f), GATE_RECALL)}</td>"
            f"<td align=right>{_f((v.get('lead_m') or {}).get(f), 0)}</td>"
            f"<td align=right>{' / '.join(f'{x:.2g}' for x in ms.thresholds.get(f, ()))}</td></tr>"
            for f in codes.FAMILIES)
        areas = "".join(f"<tr><td>{a}</td><td align=right>{gate(x, GATE_AUC)}</td></tr>"
                        for a, x in (v.get("auc_hole_by_area") or {}).items())
        return f"""
<h2>{tr('Models')} {ms.version}</h2>
<p>{tr('Created')} {ms.created} · {tr('trained on')} <b>{t.get('holes', 0):,}</b> {tr('holes')},
<b>{t.get('metres', 0):,}</b> {tr('metres')}, {tr('areas')} {', '.join(t.get('areas', []))} ·
{t.get('cavity_metres', 0):,} {tr('metres in cavities')}, {t.get('fault_metres', 0):,} {tr('metres in fault zones')}</p>
<h3>{tr('Validation: leave one area out (spec 5.9)')}</h3>
<p>{tr('Each area is predicted by models trained on the other areas only - the test that matters, because the '
       'workspace is a new area.')}</p>
<table cellpadding=4>
<tr><td>{tr('Model H - cavity, metre-level ROC-AUC')}</td><td>{gate(v.get('auc_H_cavity'), GATE_AUC)}</td></tr>
<tr><td>{tr('Model H - fault zone, metre-level ROC-AUC')}</td><td>{gate(v.get('auc_H_fault'), 0.6)}</td></tr>
<tr><td>{tr('Hole level - fails before target depth (concordance)')}</td><td>{gate(v.get('auc_hole'), 0.7)}</td></tr>
<tr><td>{tr('Spread between areas')}</td><td>{' - '.join(_f(x) for x in (v.get('auc_hole_spread') or []))}</td></tr>
<tr><td>{tr('False alarms (Orange) per 100 m')}</td><td>{_f(v.get('false_alarms_per_100m'))}</td></tr></table>
<h3>{tr('By failure family')}</h3>
<table cellpadding=4><tr><th>{tr('Family')}</th><th>{tr('Failures')}</th><th>{tr('Model')}</th>
<th>{tr('ROC-AUC (gate 0.75)')}</th><th>{tr('Recall at Orange (gate 0.70)')}</th><th>{tr('Lead (m)')}</th>
<th>{tr('Thresholds Y / O / R')}</th></tr>{rows}</table>
<p style='color:{theme.MUTED}'>{tr('Thresholds are tuned on the held-out areas so that 80% of failures are warned at '
                                    'Orange at least 3 m ahead, with Orange on at most 10% of the metres (spec 5.8). '
                                    'Families with fewer than 30 failures rely mostly on the rules and Model H (spec 5.9).')}</p>
<h3>{tr('Hole-level score by held-out area')}</h3><table cellpadding=3>{areas}</table>"""

    def _use_version(self, it) -> None:
        name = it.data(Qt.UserRole)
        (models.models_folder(self.engine.folder) / "current.txt").write_text(name, encoding="utf-8")
        self.engine.models_changed()
        self.refresh()
        self.trained.emit()

    def train(self) -> None:
        if self.engine.folder is None:
            return
        try:
            n = int(self.engine.data.collar["result"].isin(["success", "failed"]).sum())
        except Exception as e:  # noqa: BLE001
            self.banner.show_message(str(e), "error")
            return
        if n < 30:
            self.banner.show_message(tr("Only {n} finished holes in the records: enter or import more history first "
                                        "(the spec plans for 1,000+).", n=n), "warn")
            return
        self.b_train.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.show()

        def ok(ms):
            self.b_train.setEnabled(True)
            self.progress.hide()
            self.status.setText("")
            self.banner.show_message(tr("Models {v} trained and in use.", v=ms.version), "ok")
            self.refresh()
            self.trained.emit()

        def bad(msg):
            self.b_train.setEnabled(True)
            self.progress.hide()
            self.banner.show_message(tr("Training failed: {e}", e=msg), "error")

        def prog(p, t):
            self.progress.setValue(p)
            self.status.setText(t)
        self.engine.records_changed()
        run_job(self, lambda progress: self.engine.train(progress), ok, bad, prog)


# ------------------------------------------------------------------------------------------------
class GeoModelDialog(QDialog):
    """Roles of the surface layers, stratigraphic order and unit names, error band settings."""

    applied = Signal()

    def __init__(self, layers, folder: Path | None, cards: dict, parent=None, column_at=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Geological Model - roles, units and uncertainty"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        from ..scene.layers import StringLayer, SurfaceLayer

        self.layers = [l for l in layers if isinstance(l, (SurfaceLayer, StringLayer))]
        self.folder = folder
        self.cards = cards
        self.column_at = column_at
        self.cfg = {**DEFAULTS, **load_config(folder)}
        v = QVBoxLayout(self)
        intro = QLabel(tr("The geological model is read from the surface layers by their role: horizons / unconformities "
                          "are the contacts between units, faults give the fault crossings, formation solids give the "
                          "formation directly. The model is approximate: the error band ±u says how far a contact or "
                          "fault may be from where the model shows it."))
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(intro)
        split = QSplitter(Qt.Horizontal)
        # layers and roles
        g1 = QGroupBox(tr("Surface layers and their role"))
        l1 = QVBoxLayout(g1)
        self.t_layers = QTableWidget(len(self.layers), 2)
        self.t_layers.setHorizontalHeaderLabels([tr("Layer"), tr("Role")])
        self.t_layers.verticalHeader().hide()
        self.t_layers.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.role_boxes = []
        for i, lay in enumerate(self.layers):
            it = QTableWidgetItem(lay.name)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.t_layers.setItem(i, 0, it)
            box = QComboBox()
            for k, t in ROLES.items():
                box.addItem(tr(t), k)
            box.setCurrentIndex(max(0, box.findData(getattr(lay, "role", ""))))
            box.currentIndexChanged.connect(self._fill_units)
            self.t_layers.setCellWidget(i, 1, box)
            self.role_boxes.append(box)
        l1.addWidget(self.t_layers)
        b_guess = QPushButton(icon("auto-fix"), tr("Guess roles from names"))
        b_guess.clicked.connect(self._guess)
        l1.addWidget(b_guess)
        split.addWidget(g1)
        # units
        g2 = QGroupBox(tr("Units, top to bottom"))
        l2 = QVBoxLayout(g2)
        self.t_units = QTableWidget(0, 3)
        self.t_units.setHorizontalHeaderLabels([tr("Unit / formation"), tr("Base contact"), tr("Formation card")])
        self.t_units.verticalHeader().hide()
        self.t_units.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.t_units.itemChanged.connect(self._unit_edited)
        l2.addWidget(self.t_units)
        hint = QLabel(tr("Name each unit as in the formation cards (Drilling Records > Formation cards): the card is what "
                         "the models read. Contacts are ordered by their mean elevation."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        l2.addWidget(hint)
        split.addWidget(g2)
        v.addWidget(split, 1)
        # error band
        g3 = QGroupBox(tr("Error band ±u (spec 5.6)"))
        f = QFormLayout(g3)
        self.s = {}
        for key, label, lo, hi, suffix, tip in (
                ("u_near", tr("±u near model control"), 0.5, 200, " m", tr("Error near a hole the model was built from")),
                ("u_far", tr("±u far from control"), 1, 500, " m",
                 tr("Error where the model rests on surface mapping and geophysics only")),
                ("d_far", tr("Distance counted as 'far'"), 10, 5000, " m", tr("From the nearest control hole")),
                ("fault_zone", tr("Fault damage zone half width"), 0, 200, " m",
                 tr("A hole within this distance of a fault surface is in the fault zone")),
                ("thickness_factor", tr("Widening per variable unit above"), 0, 1, "",
                 tr("u grows by this share of (max - min thickness) of every unit above")),
        ):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2 if key == "thickness_factor" else 1)
            sp.setSingleStep(0.05 if key == "thickness_factor" else 1)
            sp.setValue(float(self.cfg.get(key, DEFAULTS.get(key, 0))))
            if suffix:
                sp.setSuffix(suffix)
            sp.setToolTip(tip)
            f.addRow(label, sp)
            self.s[key] = sp
        self.area = QComboBox()
        self.area.setEditable(True)
        f.addRow(tr("Area of this workspace"), self.area)
        v.addWidget(g3)
        self.preview = QLabel("")
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(self.preview)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("Apply"))
        bb.accepted.connect(self._apply)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.resize(1080, 760)
        self._fill_units()

    def set_areas(self, areas: list[str], current: str | None) -> None:
        self.area.addItems(areas)
        self.area.setCurrentText(str(self.cfg.get("area_id") or current or ""))

    def _guess(self) -> None:
        from ..scene.geo_roles import guess_role

        for lay, box in zip(self.layers, self.role_boxes):
            r = guess_role(lay.name)
            if r:
                box.setCurrentIndex(max(0, box.findData(r)))

    def _contacts(self) -> list:
        out = []
        for lay, box in zip(self.layers, self.role_boxes):
            if box.currentData() in ("horizon", "unconformity") and getattr(lay, "data", None) is not None:
                try:
                    z = float(np.nanmedian(lay.data.vertices[:, 2]))
                except Exception:  # noqa: BLE001
                    z = 0.0
                out.append((z, lay.name))
        return [n for _z, n in sorted(out, reverse=True)]

    def _fill_units(self) -> None:
        contacts = self._contacts()
        names = list(self.cfg.get("units") or [])
        n = len(contacts) + 1
        if len(names) != n:
            names = [None] * n
            for i, h in enumerate(contacts):
                pair = unit_pair(h)
                if pair:
                    names[i] = names[i] or pair[0]
                    names[i + 1] = pair[1]
            names = [u or f"Unit {i + 1}" for i, u in enumerate(names)]
        self.t_units.blockSignals(True)
        self.t_units.setRowCount(n)
        for i in range(n):
            self.t_units.setItem(i, 0, QTableWidgetItem(names[i]))
            base = QTableWidgetItem(contacts[i] if i < len(contacts) else tr("(lowest unit)"))
            base.setFlags(base.flags() & ~Qt.ItemIsEditable)
            self.t_units.setItem(i, 1, base)
            has = names[i] in self.cards
            st = QTableWidgetItem("✓ " + tr("card") if has else "⚠ " + tr("no card - rock class defaults"))
            st.setForeground(QColor("#3fb950" if has else "#e3b341"))
            st.setFlags(st.flags() & ~Qt.ItemIsEditable)
            self.t_units.setItem(i, 2, st)
        self.t_units.blockSignals(False)
        nf = sum(1 for b in self.role_boxes if b.currentData() == "fault")
        txt = tr("{c} contacts -> {u} units; {f} faults.", c=len(contacts), u=n, f=nf)
        if self.column_at is not None:
            try:
                col = self.column_at()
                if col:
                    txt += "  " + tr("Column at the workspace centre: ") + " / ".join(
                        f"{u} ({(t - b):.0f} m)" if math.isfinite(t) and math.isfinite(b) else u for u, t, b in col)
            except Exception:  # noqa: BLE001
                pass
        self.preview.setText(txt)

    def _unit_edited(self, it) -> None:
        if it.column() != 0:
            return
        has = it.text().strip() in self.cards
        st = self.t_units.item(it.row(), 2)
        if st is not None:
            st.setText("✓ " + tr("card") if has else "⚠ " + tr("no card - rock class defaults"))
            st.setForeground(QColor("#3fb950" if has else "#e3b341"))

    def _apply(self) -> None:
        for lay, box in zip(self.layers, self.role_boxes):
            lay.role = box.currentData() or ""
        units = [self.t_units.item(i, 0).text().strip() or f"Unit {i + 1}" for i in range(self.t_units.rowCount())]
        cfg = dict(self.cfg)
        cfg.update({k: float(sp.value()) for k, sp in self.s.items()})
        cfg["units"] = units
        cfg["order"] = self._contacts()
        if self.area.currentText().strip():
            cfg["area_id"] = self.area.currentText().strip()
        if self.folder is not None:
            save_config(self.folder, cfg)
        self.applied.emit()
        self.accept()
