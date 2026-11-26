"""The 3D slice panel (View > 3D slice): the main 3D view shows only the objects between two planes.

The plane (XY, XZ, YZ) sets the axis; From / To are set on a range bar (drag an end, or drag the middle to move the
whole slab) or typed, always inside the extent of the visible layers along that axis. The view is cut directly - no
other window: see scene/slicing.py. "View From / To section" turn the camera square to that face of the slab.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..scene.slicing import AXIS_NAMES, PLANE_TEXT, PLANES
from . import theme
from .icons import icon

DEFAULT_SHARE = 0.2          # a new slab: the middle fifth of the extent
MIN_GAP = 0.001              # the slab is never thinner than this share of the extent


class RangeSlider(QWidget):
    """A bar with two handles: ``lo`` and ``hi`` as shares (0..1) of the extent."""

    changed = Signal(float, float)      # while dragging
    released = Signal()

    R = 8                                # handle radius (px)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lo, self.hi = 0.4, 0.6
        self._drag = None                # "lo", "hi", "both"
        self._grab = 0.0
        self.setMinimumHeight(2 * self.R + 12)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tr("Drag an end to move From or To, drag the coloured part to move the whole slab "
                           "(a thin slab: drag it to move it, set its thickness below)"))

    def set_range(self, lo: float, hi: float) -> None:
        self.lo, self.hi = float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1))
        self.update()

    # --- geometry ---------------------------------------------------------------
    def _x(self, f: float) -> float:
        return self.R + 2 + f * max(self.width() - 2 * self.R - 4, 1)

    def _f(self, x: float) -> float:
        return float(np.clip((x - self.R - 2) / max(self.width() - 2 * self.R - 4, 1), 0.0, 1.0))

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cy = self.height() / 2.0
        x0, x1 = self._x(0.0), self._x(1.0)
        xl, xh = self._x(self.lo), self._x(self.hi)
        enabled = self.isEnabled()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.BORDER))
        p.drawRoundedRect(QRectF(x0, cy - 3, x1 - x0, 6), 3, 3)
        p.setBrush(QColor(theme.ACCENT if enabled else theme.DISABLED))
        p.drawRoundedRect(QRectF(xl, cy - 4, max(xh - xl, 1.0), 8), 4, 4)
        for x in (xl, xh):
            p.setPen(QPen(QColor(theme.ACCENT if enabled else theme.DISABLED), 2))
            p.setBrush(QColor(theme.PANEL))
            p.drawEllipse(QPointF(x, cy), self.R - 1, self.R - 1)
        p.end()

    # --- mouse ------------------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        x = e.position().x()
        xl, xh = self._x(self.lo), self._x(self.hi)
        if xh - xl < 2 * self.R + 4 and abs(x - 0.5 * (xl + xh)) <= self.R + 4:   # a thin slab: move it
            self._drag, self._grab = "both", self._f(x) - self.lo
        elif abs(x - xl) <= self.R + 2 and abs(x - xl) <= abs(x - xh):
            self._drag = "lo"
        elif abs(x - xh) <= self.R + 2:
            self._drag = "hi"
        elif xl < x < xh:
            self._drag, self._grab = "both", self._f(x) - self.lo
        else:                                   # outside: the nearer end jumps there
            self._drag = "lo" if x < xl else "hi"
            self._move(x)

    def mouseMoveEvent(self, e):
        if self._drag:
            self._move(e.position().x())

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = None
            self.released.emit()

    def _move(self, x: float) -> None:
        f = self._f(x)
        if self._drag == "lo":
            self.lo = min(f, self.hi - MIN_GAP)
        elif self._drag == "hi":
            self.hi = max(f, self.lo + MIN_GAP)
        else:
            w = self.hi - self.lo
            self.lo = float(np.clip(f - self._grab, 0.0, 1.0 - w))
            self.hi = self.lo + w
        self.lo, self.hi = max(self.lo, 0.0), min(self.hi, 1.0)
        self.update()
        self.changed.emit(self.lo, self.hi)


class SlicePanel(QWidget):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.slicer = scene.slicer
        self.plane = "XY"
        self.extent: dict[str, tuple[float, float]] = {}     # plane -> (min, max) world, of the visible layers
        self.ranges: dict[str, tuple[float, float]] = {}     # plane -> (from, to) last set
        self._busy = False
        self._timer = QTimer(self)                          # block models and section faces after a pause
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._update_full)

        lay = QVBoxLayout(self)
        self.on = QCheckBox(tr("Slice the 3D view"))
        self.on.setChecked(True)
        self.on.setToolTip(tr("Off: everything is shown again (the From / To values are kept)"))
        self.on.toggled.connect(lambda _on: self.apply())
        lay.addWidget(self.on)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Plane")))
        self.plane_group = QButtonGroup(self)
        self.plane_buttons = {}
        for name in PLANES:
            b = QRadioButton(name)
            b.setToolTip(tr(PLANE_TEXT[name]))
            self.plane_group.addButton(b)
            self.plane_buttons[name] = b
            row.addWidget(b)
            b.toggled.connect(lambda on, n=name: on and self.set_plane(n))
        row.addStretch(1)
        lay.addLayout(row)

        self.extent_label = QLabel()
        self.extent_label.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(self.extent_label)
        self.bar = RangeSlider()
        self.bar.changed.connect(self._bar_moved)
        self.bar.released.connect(self._update_full)
        lay.addWidget(self.bar)

        form = QFormLayout()
        self.lo_spin, self.hi_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for s, tip in ((self.lo_spin, tr("Slice from: the lower plane of the slab (world coordinate)")),
                       (self.hi_spin, tr("Slice to: the upper plane of the slab (world coordinate)"))):
            s.setDecimals(1)
            s.setSuffix(" m")
            s.setKeyboardTracking(False)
            s.setToolTip(tip)
            s.valueChanged.connect(self._spin_changed)
        form.addRow(tr("From"), self.lo_spin)
        form.addRow(tr("To"), self.hi_spin)
        self.thick_spin = QDoubleSpinBox()
        self.thick_spin.setDecimals(1)
        self.thick_spin.setSuffix(" m")
        self.thick_spin.setKeyboardTracking(False)
        self.thick_spin.setToolTip(tr("Thickness of the slab (To - From); a new thickness keeps the slab's centre"))
        self.thick_spin.valueChanged.connect(self._thick_changed)
        form.addRow(tr("Thickness"), self.thick_spin)
        lay.addLayout(form)

        row = QHBoxLayout()
        self.prev_btn = QPushButton(icon("skip-previous-outline"), tr("Previous"))
        self.prev_btn.setToolTip(tr("Move the slab back by its thickness"))
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn = QPushButton(icon("skip-next-outline"), tr("Next"))
        self.next_btn.setToolTip(tr("Move the slab on by its thickness"))
        self.next_btn.clicked.connect(lambda: self.step(1))
        self.full_btn = QPushButton(icon("arrow-expand-horizontal"), tr("Full range"))
        self.full_btn.setToolTip(tr("From / To at the ends of the extent: nothing is cut"))
        self.full_btn.clicked.connect(self.full_range)
        for b in (self.prev_btn, self.next_btn, self.full_btn):
            row.addWidget(b)
        lay.addLayout(row)

        self.faces = QCheckBox(tr("Section faces on solids"))
        self.faces.setChecked(True)
        self.faces.setToolTip(tr("Fill the cut of every solid at From and To in its colour; open surfaces "
                                 "(topography, faults) show their trace line"))
        self.faces.toggled.connect(lambda _on: self.apply())
        lay.addWidget(self.faces)

        row = QHBoxLayout()
        self.view_lo = QPushButton(icon("eye-outline"), tr("View From section"))
        self.view_lo.setToolTip(tr("Look square at the slab's From face, from outside the slab"))
        self.view_lo.clicked.connect(lambda: self.slicer.view_face("from"))
        self.view_hi = QPushButton(icon("eye-outline"), tr("View To section"))
        self.view_hi.setToolTip(tr("Look square at the slab's To face, from outside the slab"))
        self.view_hi.clicked.connect(lambda: self.slicer.view_face("to"))
        row.addWidget(self.view_lo)
        row.addWidget(self.view_hi)
        lay.addLayout(row)
        self.zoom_btn = QPushButton(icon("magnify-scan"), tr("Zoom to the slab"))
        self.zoom_btn.setToolTip(tr("Fit the view to the objects in the slab, keeping the view direction"))
        self.zoom_btn.clicked.connect(self.slicer.zoom_to)
        lay.addWidget(self.zoom_btn)

        note = QLabel(tr("Only the objects between From and To are drawn in the 3D view. Block models show the "
                         "whole blocks whose centres are in the slab; labels are hidden while slicing."))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(note)
        lay.addStretch(1)
        self.plane_buttons[self.plane].setChecked(True)

    # --- extent -----------------------------------------------------------------
    def refresh_extent(self) -> None:
        b = self.scene.world_bounds()
        self.extent = {}
        if b is not None:
            for name, a in PLANES.items():
                lo, hi = float(b[0][a]), float(b[1][a])
                if hi - lo < 1e-6:
                    lo, hi = lo - 0.5, hi + 0.5
                self.extent[name] = (lo, hi)
        for name, (lo, hi) in list(self.ranges.items()):     # keep the slab inside the (new) extent
            if name in self.extent:
                e0, e1 = self.extent[name]
                lo, hi = float(np.clip(lo, e0, e1)), float(np.clip(hi, e0, e1))
                if hi <= lo:
                    del self.ranges[name]
                else:
                    self.ranges[name] = (lo, hi)

    def current(self) -> tuple[float, float] | None:
        if self.plane not in self.extent:
            return None
        if self.plane not in self.ranges:
            e0, e1 = self.extent[self.plane]
            m, h = 0.5 * (e0 + e1), 0.5 * DEFAULT_SHARE * (e1 - e0)
            self.ranges[self.plane] = (m - h, m + h)
        return self.ranges[self.plane]

    # --- controls -----------------------------------------------------------------
    def set_plane(self, name: str) -> None:
        self.plane = name
        self._show()
        self.apply()

    def set_range(self, lo: float, hi: float, live: bool = False) -> None:
        """From / To in world coordinates (clamped to the extent)."""
        if self.plane not in self.extent:
            return
        e0, e1 = self.extent[self.plane]
        lo, hi = sorted((float(np.clip(lo, e0, e1)), float(np.clip(hi, e0, e1))))
        gap = MIN_GAP * (e1 - e0)
        if hi - lo < gap:
            hi = min(lo + gap, e1)
            lo = hi - gap
        self.ranges[self.plane] = (lo, hi)
        self._show()
        self.apply(live=live)

    def step(self, direction: int) -> None:
        cur = self.current()
        if cur is None:
            return
        lo, hi = cur
        e0, e1 = self.extent[self.plane]
        w = hi - lo
        lo = float(np.clip(lo + direction * w, e0, e1 - w))
        self.set_range(lo, lo + w)

    def full_range(self) -> None:
        if self.plane in self.extent:
            self.set_range(*self.extent[self.plane])

    def _bar_moved(self, flo: float, fhi: float) -> None:
        if self.plane not in self.extent:
            return
        e0, e1 = self.extent[self.plane]
        self.set_range(e0 + flo * (e1 - e0), e0 + fhi * (e1 - e0), live=True)

    def _spin_changed(self, _v) -> None:
        if self._busy:
            return
        lo, hi = self.lo_spin.value(), self.hi_spin.value()
        if self.sender() is self.lo_spin and lo > hi:
            hi = lo
        elif self.sender() is self.hi_spin and hi < lo:
            lo = hi
        self.set_range(lo, hi)

    def _thick_changed(self, w: float) -> None:
        cur = self.current()
        if self._busy or cur is None:
            return
        m = 0.5 * (cur[0] + cur[1])
        e0, e1 = self.extent[self.plane]
        lo = float(np.clip(m - 0.5 * w, e0, max(e1 - w, e0)))
        self.set_range(lo, lo + w)

    def _show(self) -> None:
        """The controls show the current plane's extent and slab."""
        cur = self.current()
        ok = cur is not None
        for w in (self.bar, self.lo_spin, self.hi_spin, self.thick_spin, self.prev_btn, self.next_btn, self.full_btn,
                  self.view_lo, self.view_hi, self.zoom_btn):
            w.setEnabled(ok)
        if not ok:
            self.extent_label.setText(tr("No visible layers to slice"))
            return
        e0, e1 = self.extent[self.plane]
        lo, hi = cur
        axis = AXIS_NAMES[PLANES[self.plane]]
        self.extent_label.setText(tr("Extent ({axis}): {lo} .. {hi} m", axis=axis, lo=f"{e0:,.1f}", hi=f"{e1:,.1f}"))
        self._busy = True
        try:
            for s, v in ((self.lo_spin, lo), (self.hi_spin, hi)):
                s.setRange(e0, e1)
                s.setSingleStep(_nice_step(e1 - e0))
                s.setValue(v)
            self.thick_spin.setRange(MIN_GAP * (e1 - e0), e1 - e0)
            self.thick_spin.setSingleStep(_nice_step(e1 - e0))
            self.thick_spin.setValue(hi - lo)
        finally:
            self._busy = False
        self.bar.set_range((lo - e0) / (e1 - e0), (hi - e0) / (e1 - e0))

    # --- the view -------------------------------------------------------------------
    def apply(self, live: bool = False) -> None:
        """Cut the main view (``live``: only move the planes now, the rest after a short pause)."""
        cur = self.current()
        if not self._shown or not self.on.isChecked() or cur is None:
            self._timer.stop()
            self.slicer.clear()
            return
        self.slicer.set(PLANES[self.plane], cur[0], cur[1], faces=self.faces.isChecked(), update=not live)
        if live:
            self._timer.start()

    def _update_full(self) -> None:
        if self.slicer.active:
            self.slicer.update()

    _shown = False

    def activate(self) -> None:
        self._shown = True
        self.refresh_extent()
        self._show()
        self.apply()

    def deactivate(self) -> None:
        self._shown = False
        self._timer.stop()
        self.slicer.clear()

    def layers_changed(self) -> None:
        """Layers were added, removed, shown or hidden: a new extent; block models / section faces again."""
        if not self._shown:
            return
        self.refresh_extent()
        self._show()
        self.apply()


def _nice_step(span: float) -> float:
    s = max(span / 100.0, 1e-3)
    m = 10 ** np.floor(np.log10(s))
    return float(next(k * m for k in (1, 2, 5, 10) if k * m >= s))


class SliceDock(QDockWidget):
    """The panel in a dock; closing it turns the slice off."""

    closed = Signal()

    def closeEvent(self, e):
        super().closeEvent(e)
        self.closed.emit()
