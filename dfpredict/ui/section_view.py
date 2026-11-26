"""Cross-section window: the scene cut along a section line, drawn flat.

Horizontal: distance along the section line from its start A. Vertical: the
coordinate along the view direction the line was drawn in (elevation for a line
drawn in plan). Scale 1:1 (no vertical exaggeration), with a scale bar, a
coordinate grid and the corridor width for strings / drillholes. Export writes one
printable 300 dpi PNG page (A4 / A3, white background) with the plan view and its
section line on top and the section below.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import vtk
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from ..i18n import N_, apply_vtk_font, tr
from ..scene.legend import LegendOverlay
from ..scene.map_grid import MapGrid2D
from ..scene.scale_bar import ScaleBar
from ..scene.scene import NavigationStyle
from ..scene.section import AXIS_NAMES, SECTION_NAME, SectionLine, SectionResult, build_section
from .icons import icon
from .section_print import DPI, PAPERS_MM, export_page, scalar_bar, section_markers

DEFAULT_HALF_WIDTH = 10.0
SECTION_COLOR = (1.0, 0.55, 0.12)
VIEW_NAMES = {2: N_("Plan view (XY)"), 1: N_("Section view looking north (XZ)"),
              0: N_("Section view looking west (YZ)")}
DRAWN_IN = {2: N_("drawn in plan view"), 1: N_("drawn in section view looking north"),
            0: N_("drawn in section view looking west")}


class _GridHost:
    """What MapGrid2D needs from a scene: the plotter, 2D mode and the label offset."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.is_2d = True
        self.origin = np.zeros(3)


class SectionDialog(QDialog):
    closed = Signal()

    def __init__(self, window, scene):
        super().__init__(window, Qt.Window)
        self.main = window
        self.scene = scene
        self.line: SectionLine | None = None
        self.result = SectionResult()
        self._props: list = []
        self._legend: LegendOverlay | None = None
        self.background = (0.0, 0.0, 0.0)
        self.foreground = (1.0, 1.0, 1.0)
        self.export_legend = True        # drillhole code legend on the exported page
        self.setWindowTitle(tr("Cross-section"))
        self.setWindowIcon(icon("box-cutter", "#3d9bff"))
        self.resize(1200, 640)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget(self)
        bar.setObjectName("sectionBar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 6, 8, 6)
        h.setSpacing(8)
        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        h.addWidget(self.info, 1)
        h.addWidget(QLabel(tr("Corridor ±")))
        self.width_box = QDoubleSpinBox()
        self.width_box.setRange(0.0, 100000.0)
        self.width_box.setDecimals(1)
        self.width_box.setSingleStep(5.0)
        self.width_box.setValue(DEFAULT_HALF_WIDTH)
        self.width_box.setSuffix(" m")
        self.width_box.setKeyboardTracking(False)
        self.width_box.setToolTip(tr("Strings, points and drillholes within this distance of the\n"
                                     "section are projected onto it. Surfaces and blocks are cut exactly."))
        self.width_box.valueChanged.connect(lambda _v: self.rebuild())
        h.addWidget(self.width_box)
        self.grid_box = QCheckBox(tr("Grid"))
        self.grid_box.setChecked(True)
        self.grid_box.toggled.connect(self._toggle_grid)
        h.addWidget(self.grid_box)
        self.b_fit = self._button("fit-to-screen-outline", tr("Zoom to the whole section (Home)"), self.fit)
        self.paper = QComboBox()
        self.paper.addItems(list(PAPERS_MM))
        self.paper.setToolTip(tr("Page size of the exported image (portrait, {dpi} dpi)", dpi=DPI))
        self.b_export = self._button(
            "image-move", tr("Export a printable page ({dpi} dpi PNG, white background):\n"
                             "the plan view with the section line on top, the section below", dpi=DPI),
            self.export_dialog, tr("Export"))
        h.addWidget(self.b_fit)
        h.addWidget(self.paper)
        h.addWidget(self.b_export)
        v.addWidget(bar)

        self.plotter = QtInteractor(self, multi_samples=8, auto_update=False)
        v.addWidget(self.plotter.interactor, 1)
        self.pos_label = QLabel("")
        self.pos_label.setObjectName("sectionStatus")
        self.pos_label.setContentsMargins(10, 3, 10, 3)
        v.addWidget(self.pos_label)

        p = self.plotter
        p.set_background("black")
        p.enable_parallel_projection()
        self.style = NavigationStyle()
        self.style.mode = "pan"          # left / middle drag pan, right drag / wheel zoom
        self.style.SetDefaultRenderer(p.renderer)
        if p.iren is not None:
            p.iren.untrack_click_position(side="left")
            p.iren.style = self.style
            p.iren.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.grid_host = _GridHost(p)
        self.grid = MapGrid2D(self.grid_host, axis_names=(tr("Distance along section"), "", ""), corner=(0.0, 0.0))
        self.grid.set_enabled(True)
        self.scale_bar = ScaleBar(p.renderer)
        self.message = vtk.vtkTextActor()
        tp = self.message.GetTextProperty()
        apply_vtk_font(tp)
        tp.SetFontSize(16)
        tp.SetColor(0.75, 0.78, 0.82)
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToCentered()
        self.message.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.message.SetPosition(0.5, 0.5)
        self.message.SetVisibility(False)
        p.renderer.AddViewProp(self.message)
        home = QToolButton(self)
        home.setShortcut(Qt.Key_Home)
        home.clicked.connect(self.fit)
        home.hide()

    def set_background(self, rgb) -> None:
        """Background of the section view (same as the main view); overlays follow it."""
        from ..scene.scene import foreground_for, grid_line_for

        self.background = tuple(float(c) for c in rgb[:3])
        fg = self.foreground = foreground_for(self.background)
        self.plotter.set_background(self.background)
        self.grid.set_colors(fg, grid_line_for(self.background), self.background)
        light, dark = (1.0, 1.0, 1.0), (0.12, 0.13, 0.15)
        on_light = fg != (0.93, 0.95, 0.97)
        self.scale_bar.set_colors(light, dark, dark if on_light else light, fg, self.background)
        self.message.GetTextProperty().SetColor(*fg)
        if self.line is not None:
            self.rebuild()                      # markers, labels, legend, colour bars
        else:
            self.plotter.render()

    def _button(self, name, tip, slot, text=None) -> QToolButton:
        b = QToolButton(self)
        b.setIcon(icon(name))
        b.setToolTip(tip)
        if text:
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.clicked.connect(slot)
        return b

    # --- content ----------------------------------------------------------------------
    @property
    def half_width(self) -> float:
        return float(self.width_box.value())

    def show_section(self, line: SectionLine) -> None:
        self.line = line
        self.rebuild()
        self.fit()
        self.show()
        self.raise_()
        self.activateWindow()

    def rebuild(self) -> None:
        """Cut the scene again (same camera): after a change of corridor or layers."""
        if self.line is None:
            return
        p = self.plotter
        for prop in self._props:
            p.renderer.RemoveViewProp(prop)
        self._props = []
        if self.scene.origin is None:
            self.result = SectionResult()
        else:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.result = build_section(self.scene, self.line, self.half_width)
            finally:
                QApplication.restoreOverrideCursor()
        res, line = self.result, self.line
        for a in res.actors:
            self._add(a)
        ax = line.axis
        origin_v = float(self.scene.origin[ax]) if self.scene.origin is not None else 0.0
        self.grid_host.origin = np.array([0.0, origin_v, 0.0])
        self.grid.axis_names = (tr("Distance along section"), tr(AXIS_NAMES[ax]), "")
        self.grid.set_enabled(self.grid_box.isChecked())
        self._add_markers(res, line)
        for i, (title, lut) in enumerate(res.scalar_bars[:2]):
            self._add(scalar_bar(title, lut, i, self.foreground))
        if self._legend is not None:
            self._legend.remove()
        self._legend = (LegendOverlay(p.renderer, res.legends, text_color=self.foreground,
                                      background=self.background) if res.legends else None)
        self.message.SetInput(tr("Nothing visible is cut by this section")
                              + ("" if self.half_width
                                 else "\n" + tr("(corridor is 0 m: strings and drillholes are skipped)")))
        self.message.SetVisibility(res.empty)
        self._update_info()
        p.render()

    def _add(self, prop) -> None:
        self.plotter.renderer.AddViewProp(prop)
        self._props.append(prop)

    def _v_range(self) -> tuple[float, float]:
        res, line = self.result, self.line
        if res.bounds is not None:
            return res.bounds[2], res.bounds[3]
        b = self.scene._local_bounds()
        if b is not None:
            return float(b[0][line.axis]), float(b[1][line.axis])
        return -1.0, 1.0

    def _add_markers(self, res: SectionResult, line: SectionLine) -> None:
        """A / A' at the ends, thin lines at the bends, drillhole names."""
        for prop in section_markers(res, line, self._v_range(), SECTION_COLOR, self.foreground,
                                    background=self.background):
            self._add(prop)

    def _update_info(self) -> None:
        line, res = self.line, self.result
        parts = [f"{SECTION_NAME}  {line.length:,.1f} m"]
        brg = line.bearing()
        if brg is not None:
            parts.append(tr("bearing {v:.1f}°", v=brg))
        if len(line.panels) > 1:
            parts.append(tr("{n} segments", n=len(line.panels)))
        parts.append(tr(DRAWN_IN[line.axis]))
        cut = ", ".join(res.counts) if res.counts else tr("nothing")
        self.info.setText("    ".join(parts) + "    |    " + tr("cut: {what}", what=cut))
        self.setWindowTitle(tr("Cross-section  {name}  {length:,.1f} m", name=SECTION_NAME, length=line.length))

    def _toggle_grid(self, on: bool) -> None:
        self.grid.set_enabled(on)
        self.plotter.render()

    def fit(self) -> None:
        if self.line is None:
            return
        v0, v1 = self._v_range()
        u0, u1 = 0.0, self.line.length
        if self.result.bounds is not None:
            u0, u1 = min(u0, self.result.bounds[0]), max(u1, self.result.bounds[1])
        cu, cv = (u0 + u1) / 2.0, (v0 + v1) / 2.0
        w, h = self.plotter.renderer.GetSize()
        aspect = (w / h) if h > 0 else 1.6
        half_v = max((v1 - v0) / 2.0, (u1 - u0) / 2.0 / aspect, 1.0) * 1.12
        cam = self.plotter.renderer.GetActiveCamera()
        cam.SetFocalPoint(cu, cv, 0.0)
        cam.SetPosition(cu, cv, 10000.0)
        cam.SetViewUp(0.0, 1.0, 0.0)
        cam.SetParallelScale(half_v)
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def layers_changed(self) -> None:
        if self.isVisible() and self.line is not None:
            self.rebuild()

    # --- status ---------------------------------------------------------------------
    def _on_mouse_move(self, iren, _event):
        if self.line is None:
            return
        x, y = iren.GetEventPosition()
        r = self.plotter.renderer
        r.SetDisplayPoint(x, y, 0.0)
        r.DisplayToWorld()
        wp = r.GetWorldPoint()
        u, v = wp[0] / (wp[3] or 1.0), wp[1] / (wp[3] or 1.0)
        ax = self.line.axis
        name = "XYZ"[ax]
        vw = v + (float(self.scene.origin[ax]) if self.scene.origin is not None else 0.0)
        txt = tr("Distance {u:,.2f} m    {name} {v:,.2f}", u=u, name=name, v=vw)
        world = self._world_at(u)
        if world is not None:
            i1, i2 = self.line.in_plane_axes
            txt += "    " + tr("on the line: {a} {va:,.2f}  {b} {vb:,.2f}", a="XYZ"[i1], va=world[i1],
                                   b="XYZ"[i2], vb=world[i2])
        self.pos_label.setText(txt)

    def _world_at(self, u: float) -> np.ndarray | None:
        """World position of chainage ``u`` on the section line (in-plane coordinates)."""
        line = self.line
        if not line.panels or u < 0 or u > line.length or self.scene.origin is None:
            return None
        for panel in line.panels:
            if u <= panel.u0 + panel.length + 1e-9:
                return self.scene.to_world(panel.a + panel.t * (u - panel.u0))
        return None

    # --- export ---------------------------------------------------------------------
    def export_dialog(self) -> None:
        if self.line is None:
            return
        folder = ""
        ws = getattr(self.main, "workspace", None)
        if ws is not None:
            folder = str(ws.folder)
        name = f"section_{datetime.now():%Y%m%d_%H%M%S}.png"
        path, _ = QFileDialog.getSaveFileName(self, tr("Export cross-section"), str(Path(folder) / name),
                                              tr("PNG image (*.png)"))
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self.export_png(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Export cross-section"), tr("Could not export the image:\n{e}", e=e))
            return
        self.info.setText(tr("Exported ({paper}, {dpi} dpi): {path}", paper=self.paper.currentText(), dpi=DPI,
                             path=path))

    def export_png(self, path: str, paper: str | None = None):
        """Printable page: plan view with the section line on top, the section below."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return export_page(path, self.scene, self.line, self.result, self.plotter.renderer,
                               self._v_range(), self.half_width, paper or self.paper.currentText(),
                               self.grid_box.isChecked(), self.export_legend)
        finally:
            QApplication.restoreOverrideCursor()

    # --- window -----------------------------------------------------------------------
    def closeEvent(self, e):
        self.closed.emit()
        super().closeEvent(e)

    def shutdown(self) -> None:
        self.plotter.close()
