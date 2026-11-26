"""Export map: the main view (2D or 3D) as a printable page, saved as a 300 dpi PNG.

The view is drawn again off-screen on white at the page's pixel size (300 dpi), so it is
sharp on paper (same approach as the cross-section export, ``section_print``):

- 2D views (plan XY, sections XZ / YZ): at a standard scale (fitted to what the window
  shows, or chosen), with grid lines, the scale bar (bottom left) and the coordinates
  written outside the map frame; a north arrow in plan view.
- 3D view: the view as on the screen (same direction, perspective or not) - no scale.

Legends (block model colours, drillhole codes, survey pits) go in a column right of the
map, never inside it. A footer gives the view, scale, paper and date.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import N_, tr
from ..scene.scene import VIEW_3D, VIEW_XY, VIEW_XZ, VIEW_YZ
from ..scene.section import AXIS_NAMES, clone_actor
from .icons import icon
from .section_print import (
    DPI,
    INK,
    PAPER_MM_PER_PX,
    PAPERS_MM,
    S,
    MapImage,
    _copy_camera,
    _decorate,
    _draw_axis_labels,
    _draw_legends,
    _plotter,
    _snapshot,
    mm_to_px,
    nice_ratio,
    text3d,
    to_qimage,
)

VIEW_TITLES = {VIEW_3D: N_("3D view"), VIEW_XY: N_("Plan view (XY)"),           # shown as tr(VIEW_TITLES[v])
               VIEW_XZ: N_("Section view looking north (XZ)"), VIEW_YZ: N_("Section view looking west (YZ)")}
SCALES = (500, 1000, 1250, 1500, 2000, 2500, 5000, 7500, 10000, 20000, 25000, 50000)
LEGEND_W_MM = 42


# --- rendering ---------------------------------------------------------------------------
def _add_layers(scene, renderer) -> None:
    """The visible layers' actors, copied for the export (line widths / points scaled to 300 dpi)."""
    from ..scene.layers import DrillholeLayer

    for layer in scene.layers:
        if not layer.visible:
            continue
        for actor in layer.actors:
            if not isinstance(actor, vtk.vtkActor) or not actor.GetVisibility():
                continue
            data = actor.GetMapper().GetInput() if actor.GetMapper() is not None else None
            if isinstance(data, vtk.vtkPolyData):
                renderer.AddActor(clone_actor(actor, data, lit=True, scale=S))
        if isinstance(layer, DrillholeLayer) and layer.show_labels:
            for hid in layer._trace_holes:
                try:
                    c = layer.data.trace(hid).collar - scene.origin
                except KeyError:
                    continue
                renderer.AddActor(text3d(hid, c, INK, 11 * S, (4 * S, 4 * S)))


def fit_ratio(scene, size) -> float:
    """The standard scale at which everything the window shows fits in a map of ``size`` pixels."""
    r = scene.plotter.renderer
    cam = r.GetActiveCamera()
    sw, sh = r.GetSize()
    mpp_screen = 2.0 * cam.GetParallelScale() / max(sh, 1)
    mpp = max(sw * mpp_screen / size[0], sh * mpp_screen / size[1])
    return nice_ratio(mpp * 1000.0 / PAPER_MM_PER_PX)


def render_view(scene, size, grid: bool = True, ratio: float | None = None) -> MapImage:
    """The main view drawn again on white at ``size`` pixels. 2D: at scale 1:``ratio`` (fitted
    to the window when None) with grid lines and the scale bar; 3D: the camera as on screen."""
    p = _plotter(size)
    _add_layers(scene, p.renderer)
    src = scene.plotter.renderer.GetActiveCamera()
    if scene.view == VIEW_3D:
        p.camera_position = [src.GetPosition(), src.GetFocalPoint(), src.GetViewUp()]
        cam = p.renderer.GetActiveCamera()
        sw, sh = scene.plotter.renderer.GetSize()
        a_src = sw / sh if sh else 1.0
        a_dst = size[0] / size[1]
        grow = max(1.0, a_src / a_dst)           # a taller page: keep the width that the window shows
        if src.GetParallelProjection():
            cam.ParallelProjectionOn()
            cam.SetParallelScale(src.GetParallelScale() * grow)
        else:
            cam.ParallelProjectionOff()
            half = math.radians(src.GetViewAngle()) / 2.0
            cam.SetViewAngle(math.degrees(2.0 * math.atan(math.tan(half) * grow)))
        p.renderer.ResetCameraClippingRange()
        img = _snapshot(p)
        p.close()
        return MapImage(img, None)
    _copy_camera(scene.plotter.renderer, p, size)
    ratio = float(ratio) if ratio else fit_ratio(scene, size)
    cam = p.renderer.GetActiveCamera()
    cam.SetParallelScale(size[1] / 2.0 * ratio * PAPER_MM_PER_PX / 1000.0)    # exactly 1:ratio on paper
    p.renderer.ResetCameraClippingRange()
    g, bar = _decorate(p, scene.origin, tuple(tr(n) for n in AXIS_NAMES), grid)
    img = _snapshot(p)
    layout = g.layout
    p.close()
    del bar
    return MapImage(img, layout, ratio=ratio)


# --- legends -------------------------------------------------------------------------------
def map_legends(scene) -> list:
    """Legends of the visible layers, for the column beside the map: ("codes", title,
    [(text, rgb)]) or ("bar", title, colours, vmin, vmax)."""
    from ..scene.layers import PALETTE, BlockModelLayer, DrillholeLayer
    from ..scene.survey import SurveyLayer

    out = []
    for layer in scene.layers:
        if not layer.visible:
            continue
        if isinstance(layer, BlockModelLayer) and layer.color_by:
            bm = layer.data
            name = layer.color_by
            if bm.is_categorical(name):
                fixed = bm.category_rgb.get(name) or []
                cats = bm.categories[name]
                out.append(("codes", f"{layer.name}: {name}",
                            [(c, fixed[i] if i < len(fixed) else PALETTE[i % len(PALETTE)])
                             for i, c in enumerate(cats)][:24]))
            else:
                lo, hi = layer.clim or layer.value_range(name) or (0.0, 1.0)
                lut = pv.LookupTable(cmap=layer.cmap, scalar_range=(lo, hi))
                rgb = [0.0, 0.0, 0.0]
                colours = []
                for v in np.linspace(lo, hi, 128):
                    lut.GetColor(float(v), rgb)
                    colours.append(tuple(rgb))
                out.append(("bar", f"{layer.name}: {name}", colours, float(lo), float(hi)))
        elif isinstance(layer, DrillholeLayer) and layer.field_name:
            entries = [(f"{code}  {desc}" if desc else str(code), rgb) for code, desc, rgb in layer.legend()]
            if entries:
                out.append(("codes", f"{layer.name}: {layer.field_name}", entries[:24]))
        elif isinstance(layer, SurveyLayer) and layer.data is not None and layer.data.pits:
            entries = [(f"{p.name} ({p.code})", layer.pit_color(p.id)) for p in layer.data.pits
                       if p.parent_id is None and layer.pit_shown(p.id)]
            if entries:
                out.append(("codes", layer.name, entries[:24]))
    return out


# --- the page --------------------------------------------------------------------------------
def page_pixels(paper: str, landscape: bool) -> tuple[int, int]:
    w, h = PAPERS_MM[paper]
    if landscape:
        w, h = h, w
    return mm_to_px(w), mm_to_px(h)


def page_layout(paper: str, landscape: bool, legend: bool, labels: bool) -> dict:
    """Pixel boxes (300 dpi): title, map (with gutters for the coordinates when ``labels``),
    legend column, footer."""
    W, H = page_pixels(paper, landscape)
    m, title, foot, gap = mm_to_px(10), mm_to_px(9), mm_to_px(6), mm_to_px(4)
    left = mm_to_px(12) if labels else 0
    below = mm_to_px(11) if labels else 0
    right = mm_to_px(4) if labels else 0
    lw = mm_to_px(LEGEND_W_MM) if legend else 0
    x = m + left
    y = m + title
    mw = W - 2 * m - left - right - (lw + gap if legend else 0)
    mh = H - 2 * m - title - below - foot
    boxes = {"page": (W, H), "title": QRect(m, m, W - 2 * m, title), "map": QRect(x, y, mw, mh),
             "footer": QRect(m, H - m - foot, W - 2 * m, foot)}
    if legend:
        boxes["legend"] = QRect(x + mw + right + gap, y, lw, mh)
    return boxes


def _north_arrow(p: QPainter, box: QRect, ink: QColor) -> None:
    """North arrow at the top right inside the map (plan view: north is up)."""
    s = mm_to_px(1)
    x, y = box.right() - 9 * s, box.top() + 5 * s
    p.setPen(QPen(ink, max(2, mm_to_px(0.2))))
    p.setBrush(QColor("white"))
    p.drawRect(QRect(int(x - 5 * s), int(y - 2 * s), int(10 * s), int(16 * s)))
    p.setBrush(ink)
    p.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x - 2.6 * s, y + 7 * s), QPointF(x + 2.6 * s, y + 7 * s)]))
    p.setBrush(Qt.NoBrush)
    f = QFont("Segoe UI")
    f.setPixelSize(mm_to_px(3.0))
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRect(int(x - 5 * s), int(y + 7 * s), int(10 * s), int(7 * s)), Qt.AlignCenter, "N")


def compose_page(mp: MapImage, title: str, footer: str, paper: str, landscape: bool, legends: list,
                 north: bool) -> QImage:
    L = page_layout(paper, landscape, bool(legends), mp.grid is not None)
    W, H = L["page"]
    out = QImage(W, H, QImage.Format_RGB32)
    out.fill(QColor("white"))
    dpm = int(round(DPI / 0.0254))
    out.setDotsPerMeterX(dpm)
    out.setDotsPerMeterY(dpm)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    ink = QColor.fromRgbF(*INK)
    f = QFont("Segoe UI")
    f.setPixelSize(mm_to_px(4.2))
    f.setBold(True)
    p.setFont(f)
    p.setPen(ink)
    p.drawText(L["title"], Qt.AlignLeft | Qt.AlignVCenter, QFontMetrics(f).elidedText(title, Qt.ElideRight,
                                                                                      L["title"].width()))
    box = L["map"]
    img = to_qimage(mp.image)
    p.drawImage(box, img)
    p.setPen(QPen(ink, max(2, mm_to_px(0.25))))
    p.drawRect(box)
    _draw_axis_labels(p, box, mp.grid, (img.width(), img.height()), ink)
    if north:
        _north_arrow(p, box, ink)
    if legends:
        _draw_legends(p, L["legend"], legends, ink)
    f.setBold(False)
    f.setPixelSize(mm_to_px(2.6))
    p.setFont(f)
    p.setPen(QColor("#555555"))
    p.drawText(L["footer"], Qt.AlignRight | Qt.AlignVCenter, footer)
    p.end()
    return out


def make_page(scene, title: str, paper: str = "A4", landscape: bool = True, grid: bool = True,
              legend: bool = True, ratio: float | None = None, workspace: str = "") -> tuple[QImage, float | None]:
    """The whole page for the main view. Returns (page, the scale denominator or None in 3D)."""
    legends = map_legends(scene) if legend else []
    two_d = scene.view != VIEW_3D
    L = page_layout(paper, landscape, bool(legends), two_d)
    mb = L["map"]
    mp = render_view(scene, (mb.width(), mb.height()), grid and two_d, ratio if two_d else None)
    scale = tr("scale 1:{r:,.0f} (printed at 100 %)", r=mp.ratio) if mp.ratio else tr("not to scale")
    page = (tr("{paper} landscape  {dpi} dpi", paper=paper, dpi=DPI) if landscape
            else tr("{paper} portrait  {dpi} dpi", paper=paper, dpi=DPI))
    view = VIEW_TITLES.get(scene.view, "")
    footer = "      ".join(x for x in (tr(view) if view else "", scale, page,
                                       workspace, f"GeoViewer  {datetime.now():%Y-%m-%d %H:%M}") if x)
    page = compose_page(mp, title, footer, paper, landscape, legends, scene.view == VIEW_XY)
    return page, mp.ratio


# --- the window ------------------------------------------------------------------------------
class MapExportDialog(QDialog):
    """Title, paper, orientation, scale (2D), grid and legend; a preview of the page; Export PNG."""

    def __init__(self, main, parent=None):
        super().__init__(parent or main, Qt.Window)
        self.main = main
        self.scene = main.scene
        self.setWindowTitle(tr("Export map"))
        self.setWindowIcon(icon("image-move", "#3d9bff"))
        self.resize(1080, 760)
        self.page: QImage | None = None
        self.ratio: float | None = None

        self.title = QLineEdit()
        self.paper = QComboBox()
        self.paper.addItems(["A4", "A3"])
        self.orient = QComboBox()
        self.orient.addItem(tr("Landscape"), True)
        self.orient.addItem(tr("Portrait"), False)
        self.scale = QComboBox()
        self.grid = QCheckBox(tr("Grid lines and coordinates"))
        self.grid.setChecked(True)
        self.legend = QCheckBox(tr("Legend (beside the map)"))
        self.legend.setChecked(True)
        self.title.setToolTip(tr("Written at the top of the page"))
        self.paper.setToolTip(tr("Paper size"))
        self.orient.setToolTip(tr("Landscape: wide page; portrait: tall page"))
        self.scale.setToolTip(tr("Fit view: the standard scale at which what the window shows fits the page.\n"
                                 "Or a fixed scale - true when printed at 100 % (no 'fit to page')."))
        self.grid.setToolTip(tr("Grid lines in the map and their coordinates outside the frame (2D views)"))
        self.legend.setToolTip(tr("Block model colours, drillhole codes and survey pits, in a column right of "
                                  "the map"))
        form = QFormLayout()
        form.addRow(tr("Title"), self.title)
        form.addRow(tr("Paper"), self.paper)
        form.addRow(tr("Orientation"), self.orient)
        self.scale_label = QLabel(tr("Scale"))
        form.addRow(self.scale_label, self.scale)
        form.addRow("", self.grid)
        form.addRow("", self.legend)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #9aa3ad;")
        form.addRow(self.note)

        buttons = QVBoxLayout()
        export = QPushButton(icon("image-move"), tr("Export PNG…"))
        export.setToolTip(tr("Save the page as a {dpi} dpi PNG image (true to scale when printed at 100 %)",
                             dpi=DPI))
        export.clicked.connect(self.save_image_dialog)
        export.setDefault(True)
        buttons.addWidget(export)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        buttons.addStretch(1)
        buttons.addWidget(close)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addLayout(form)
        lv.addLayout(buttons, 1)
        left.setFixedWidth(330)
        self.preview = QLabel("")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(400, 300)
        self.preview.setStyleSheet("background: #2b2f35; border: 1px solid #363b43;")
        self.preview.setToolTip(tr("Preview of the page"))
        self.info = QLabel("")
        self.info.setStyleSheet("color: #9aa3ad;")
        right = QVBoxLayout()
        right.addWidget(self.preview, 1)
        right.addWidget(self.info)
        h = QHBoxLayout(self)
        h.addWidget(left)
        h.addLayout(right, 1)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self.update_page)
        for w in (self.paper, self.orient, self.scale):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.grid, self.legend):
            w.toggled.connect(self._changed)
        self.title.textEdited.connect(self._changed)

    def start(self) -> None:
        """Fill in for the current view and draw the preview."""
        view = self.scene.view
        two_d = view != VIEW_3D
        ws = getattr(self.main, "workspace", None)
        self.workspace_name = Path(ws.folder).name if ws is not None else ""
        vt = VIEW_TITLES.get(view, "")
        self.title.setText(" - ".join(x for x in (self.workspace_name, tr(vt) if vt else "") if x))
        self.scale.blockSignals(True)
        self.scale.clear()
        if two_d:
            self.scale.addItem(tr("Fit view"), None)
            for n in SCALES:
                self.scale.addItem(f"1:{n:,}", float(n))
        self.scale.blockSignals(False)
        for w in (self.scale_label, self.scale, self.grid):
            w.setVisible(two_d)
        self.note.setText(tr("2D view: printed at a true scale; the coordinates are written outside the frame.")
                          if two_d else tr("3D view: printed as seen in the window (perspective has no scale). "
                                           "Choose a 2D view (XY, XZ, YZ) for a map to scale."))
        self.show()
        self.raise_()
        QTimer.singleShot(50, self.update_page)

    def _changed(self, *_a) -> None:
        self.page = None
        self._timer.start()

    def _options(self) -> dict:
        return {"title": self.title.text().strip(), "paper": self.paper.currentText(),
                "landscape": bool(self.orient.currentData()), "grid": self.grid.isChecked(),
                "legend": self.legend.isChecked(), "ratio": self.scale.currentData()}

    def update_page(self) -> QImage | None:
        if not any(lay.visible and lay.actors for lay in self.scene.layers):
            self.page = None
            self.preview.setText(tr("Nothing to export: show some layers first."))
            return None
        o = self._options()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.page, self.ratio = make_page(self.scene, o["title"], o["paper"], o["landscape"], o["grid"],
                                              o["legend"], o["ratio"], self.workspace_name)
        except Exception as e:  # noqa: BLE001
            self.page = None
            self.preview.setText(tr("The page could not be drawn:\n{e}", e=e))
            return None
        finally:
            QApplication.restoreOverrideCursor()
        self._show_preview()
        page = (tr("{paper} landscape, {dpi} dpi", paper=o["paper"], dpi=DPI) if o["landscape"]
                else tr("{paper} portrait, {dpi} dpi", paper=o["paper"], dpi=DPI))
        self.info.setText((tr("Scale 1:{r:,.0f} when printed at 100 %", r=self.ratio) if self.ratio
                           else tr("3D view")) + "    " + page)
        return self.page

    def _show_preview(self) -> None:
        if self.page is None:
            return
        pm = QPixmap.fromImage(self.page).scaled(self.preview.size() - QRect(0, 0, 8, 8).size(),
                                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pm)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._show_preview()

    def _ready(self) -> QImage | None:
        return self.page if self.page is not None else self.update_page()

    def _default_path(self, ext: str) -> str:
        ws = getattr(self.main, "workspace", None)
        folder = str(ws.folder) if ws is not None else ""
        return str(Path(folder) / f"map_{datetime.now():%Y%m%d_%H%M%S}.{ext}")

    def save_image_dialog(self) -> None:
        page = self._ready()
        if page is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Export map"), self._default_path("png"),
                                              tr("PNG image (*.png)"))
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not page.save(path, "PNG"):
            QMessageBox.warning(self, tr("Export map"), tr("Could not write {path}", path=path))
            return
        self.info.setText(tr("Saved: {path}", path=path))
