"""Printable export of a cross-section: one 300 dpi page, plan on top, section below.

Both views are rendered again off-screen on a white background at the page's
pixel size, so the image is sharp when printed. Line widths, point sizes,
fonts and grid spacing are scaled up from screen pixels (96 dpi) to 300 dpi,
which keeps them readable on paper. The camera keeps the area shown on screen
(more is shown where the page is wider or taller than the window).

Inside each map: data, grid lines, the scale bar (bottom left, on a white box)
and, optionally, the drillhole code legend. The coordinate labels are written
outside the map frame by the page painter: eastings / distances below, northings
/ elevations on the left, each with its axis name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pyvista as pv
import vtk
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen

from ..i18n import apply_vtk_font, tr
from ..scene.map_grid import MapGrid2D, format_value
from ..scene.scale_bar import ScaleBar
from ..scene.section import (
    AXIS_NAMES,
    DEPTH_POINTS,
    END_MARK,
    SECTION_NAME,
    START_MARK,
    SectionLine,
    SectionResult,
    clone_actor,
)

DPI = 300
SCREEN_DPI = 96
S = DPI / SCREEN_DPI                     # screen pixels -> print pixels
PAPERS_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}
INK = (0.08, 0.08, 0.1)
GRID_LINE = (0.55, 0.57, 0.6)
SECTION_COLOR = (0.95, 0.42, 0.0)        # a little darker than on screen: reads on white


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def page_size(paper: str = "A4") -> tuple[int, int]:
    w, h = PAPERS_MM[paper]
    return mm_to_px(w), mm_to_px(h)


@dataclass
class MapImage:
    """A rendered map and its grid (label values and their pixel positions)."""

    image: np.ndarray
    grid: dict | None
    legends: list = field(default_factory=list)   # drawn beside the map: see map_legends()
    ratio: float | None = None                     # map scale 1:ratio on the printed page


class _Host:
    def __init__(self, plotter, origin):
        self.plotter = plotter
        self.is_2d = True
        self.origin = np.asarray(origin, dtype=np.float64)


def _plotter(size) -> pv.Plotter:
    # No multisampling: at print size an 8x MSAA framebuffer needs ~0.5 GB of graphics memory,
    # and when that fails the image comes back black or half drawn. At 300 dpi the pixels
    # are small enough that FXAA smoothing is plenty.
    p = pv.Plotter(off_screen=True, window_size=list(size), lighting="light kit")
    p.ren_win.SetMultiSamples(0)
    p.set_background("white")
    p.enable_parallel_projection()
    try:
        p.enable_anti_aliasing("fxaa")
    except Exception:  # noqa: BLE001 - smoothing is optional
        pass
    return p


def _snapshot(p: pv.Plotter) -> np.ndarray:
    """Render and read back the image; render once more if the first comes back blank."""
    img = p.screenshot(return_img=True)
    if _blank(img):
        p.render()
        img = p.screenshot(return_img=True)
        if _blank(img):
            raise RuntimeError(tr("the graphics card returned an empty image for the print "
                                  "(not enough graphics memory?) - try A4, or close other 3D programs"))
    return img


def _blank(img: np.ndarray) -> bool:
    """A print view always has a white background somewhere: all black / one colour = failed."""
    flat = img[::17, ::17, :3].reshape(-1, 3)
    return bool((flat == flat[0]).all() and not (flat[0] == 255).all()) or not (flat == 255).all(axis=1).any()


def _overlay(p: pv.Plotter) -> vtk.vtkRenderer:
    """A top layer sharing the camera: what is drawn there is never hidden by the data."""
    rw = p.ren_win
    r = vtk.vtkRenderer()
    rw.SetNumberOfLayers(max(2, rw.GetNumberOfLayers()))
    r.SetLayer(1)
    r.InteractiveOff()
    r.SetActiveCamera(p.renderer.GetActiveCamera())
    rw.AddRenderer(r)
    return r


def _copy_camera(src_renderer, dst: pv.Plotter, size) -> None:
    """Same view centre and direction; the parallel scale grows so that everything
    visible in the source viewport is visible in ``size``."""
    sc = src_renderer.GetActiveCamera()
    # Through pyvista, so it counts as set: otherwise the first render resets it to an iso view.
    dst.camera_position = [sc.GetPosition(), sc.GetFocalPoint(), sc.GetViewUp()]
    dc = dst.renderer.GetActiveCamera()
    dc.ParallelProjectionOn()
    sw, sh = src_renderer.GetSize()
    a_src = sw / sh if sh else 1.0
    a_dst = size[0] / size[1]
    dc.SetParallelScale(sc.GetParallelScale() * max(1.0, a_src / a_dst))
    dst.renderer.ResetCameraClippingRange()


def _decorate(p: pv.Plotter, origin, axis_names, grid: bool):
    """Grid lines (labels go outside the frame) and the scale bar at the bottom left."""
    g = MapGrid2D(_Host(p, origin), axis_names=axis_names, corner=(0.0, 0.0), scale=S,
                  line_color=GRID_LINE, label_color=INK, show_labels=False, show_lines=grid)
    g.set_enabled(True)
    bar = ScaleBar(p.renderer, position="bottom-left", margin=(14, 14), scale=S, light=(1.0, 1.0, 1.0),
                   dark=INK, outline=INK, text_color=INK, text_background=(1.0, 1.0, 1.0),
                   text_background_opacity=0.0, box=(1.0, 1.0, 1.0), box_padding=6, box_outline=INK,
                   font_size=10, target_fraction=0.3)    # longer than on screen: more precise on paper
    return g, bar     # keep references until rendered


def text3d(text, pos, color, size, offset, background=(1.0, 1.0, 1.0), opacity=0.0):
    t = vtk.vtkBillboardTextActor3D()
    t.SetInput(text)
    t.SetPosition(*pos)
    t.SetDisplayOffset(int(offset[0]), int(offset[1]))
    tp = t.GetTextProperty()
    apply_vtk_font(tp)
    tp.SetFontSize(int(round(size)))
    tp.SetColor(*color)
    tp.SetBold(size >= 14)
    tp.SetBackgroundColor(*background)
    tp.SetBackgroundOpacity(opacity)
    t.PickableOff()
    return t


def scalar_bar(title, lut, index: int, color, scale: float = 1.0) -> vtk.vtkScalarBarActor:
    sb = vtk.vtkScalarBarActor()
    sb.SetLookupTable(lut)
    sb.SetTitle(title)
    sb.SetNumberOfLabels(6)
    sb.SetLabelFormat("%.2f")
    sb.SetPosition(0.93 - 0.08 * index, 0.28)
    sb.SetWidth(0.05)
    sb.SetHeight(0.45)
    sb.UnconstrainedFontSizeOn()
    for tp in (sb.GetTitleTextProperty(), sb.GetLabelTextProperty()):
        apply_vtk_font(tp)
        tp.SetColor(*color)
        tp.SetFontSize(int(round(12 * scale)))
        tp.ShadowOff()
        tp.ItalicOff()
    sb.GetLabelTextProperty().BoldOff()
    return sb


def section_markers(res: SectionResult, line: SectionLine, v_range, color, label_color,
                    scale: float = 1.0, background=(0.0, 0.0, 0.0), opacity=0.55) -> list:
    """Thin lines at the ends and bends of the line, A / 1 / 2 .. / A' above them, hole names."""
    v0, v1 = v_range
    pad = max(v1 - v0, line.length, 1.0) * 5
    ch = line.chainages
    pts, cells = [], []
    for i, u in enumerate(ch):
        pts += [(u, v0 - pad, -1.0), (u, v1 + pad, -1.0)]
        cells += [2, 2 * i, 2 * i + 1]
    mesh = pv.PolyData(np.array(pts, dtype=np.float32), lines=np.array(cells))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    a = vtk.vtkActor()
    a.SetMapper(mapper)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetOpacity(0.5)
    a.GetProperty().SetLineWidth(max(1.0, scale))
    a.GetProperty().LightingOff()
    a.PickableOff()
    out = [a]
    names = [START_MARK] + [f"{i}" for i in range(1, len(ch) - 1)] + [END_MARK]
    for u, text in zip(ch, names):
        out.append(text3d(text, (u, v1, DEPTH_POINTS + 2), color, 16 * scale, (6 * scale, 18 * scale),
                          background, opacity))
    for u, v, depth, text in res.labels:
        out.append(text3d(text, (u, v, depth), label_color, 11 * scale, (4 * scale, 4 * scale),
                          background, opacity))
    return out


# --- the two views --------------------------------------------------------------------
def _section_line_actors(line: SectionLine, depth: float) -> list:
    pts = np.asarray(line.points, dtype=np.float64).copy()
    pts[:, line.axis] = depth
    poly = pv.PolyData(pts.astype(np.float32), lines=np.array([len(pts), *range(len(pts))]))
    out = []
    for mesh, width, size in ((poly, 3.0 * S, 1.0), (pv.PolyData(pts.astype(np.float32)), 1.0, 9.0 * S)):
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(mesh)
        m.ScalarVisibilityOff()
        a = vtk.vtkActor()
        a.SetMapper(m)
        pr = a.GetProperty()
        pr.SetColor(*SECTION_COLOR)
        pr.LightingOff()
        pr.SetLineWidth(width)
        pr.SetPointSize(size)
        pr.SetRenderPointsAsSpheres(True)
        out.append(a)
    out.append(text3d(START_MARK, pts[0], SECTION_COLOR, 18 * S, (8 * S, 6 * S)))
    out.append(text3d(END_MARK, pts[-1], SECTION_COLOR, 18 * S, (8 * S, 6 * S)))
    return out


def render_plan(scene, line: SectionLine | None, size, grid: bool = True) -> MapImage:
    """The main 2D view re-drawn on white, with the section line and a scale bar."""
    from ..scene.layers import DrillholeLayer

    p = _plotter(size)
    r = p.renderer
    for layer in scene.layers:
        if not layer.visible:
            continue
        for actor in layer.actors:
            if not isinstance(actor, vtk.vtkActor) or not actor.GetVisibility():
                continue
            data = actor.GetMapper().GetInput() if actor.GetMapper() is not None else None
            if isinstance(data, vtk.vtkPolyData):
                r.AddActor(clone_actor(actor, data, lit=True, scale=S))
        if isinstance(layer, DrillholeLayer) and layer.show_labels:
            for hid in layer._trace_holes:
                try:
                    c = layer.data.trace(hid).collar - scene.origin
                except KeyError:
                    continue
                r.AddActor(text3d(hid, c, INK, 11 * S, (4 * S, 4 * S)))
    _copy_camera(scene.plotter.renderer, p, size)
    if line is not None and line.valid:
        top = _overlay(p)
        focal = p.renderer.GetActiveCamera().GetFocalPoint()
        for a in _section_line_actors(line, focal[line.axis]):   # on the focal plane: never clipped
            top.AddViewProp(a)
    g, bar = _decorate(p, scene.origin, tuple(tr(n) for n in AXIS_NAMES), grid)
    img = _snapshot(p)
    layout = g.layout
    p.close()
    del bar
    return MapImage(img, layout)


def render_section(view_renderer, res: SectionResult, line: SectionLine, origin_v: float,
                   v_range, size, grid: bool = True, legend: bool = True) -> MapImage:
    """The section re-drawn on white (same cut geometry and colours as on screen)."""
    p = _plotter(size)
    r = p.renderer
    for a in res.actors:
        r.AddActor(clone_actor(a, a.GetMapper().GetInput(), scale=S))
    for prop in section_markers(res, line, v_range, SECTION_COLOR, INK, S, (1.0, 1.0, 1.0), 0.0):
        r.AddViewProp(prop)
    ratio = fit_section_camera(p, res, line, v_range, size)
    g, bar = _decorate(p, (0.0, origin_v, 0.0), (tr("Distance along section"), tr(AXIS_NAMES[line.axis]), ""),
                       grid)
    img = _snapshot(p)
    layout = g.layout
    p.close()
    del bar
    return MapImage(img, layout, map_legends(res, legend), ratio)


# Standard map scales (x 10^k): the fitted scale is rounded to the next one that still fits.
NICE_SCALES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0)
PAPER_MM_PER_PX = 25.4 / DPI


def nice_ratio(ratio: float) -> float:
    """Smallest standard scale denominator >= ``ratio`` (1:1,000, 1:1,250, 1:1,500, 1:2,000 ...)."""
    if not (ratio > 0 and np.isfinite(ratio)):
        return 1000.0
    mag = 10.0 ** np.floor(np.log10(ratio))
    for m in NICE_SCALES + (10.0,):
        if m * mag >= ratio * (1 - 1e-9):
            return float(m * mag)
    return float(10.0 * mag)


def section_extent(res: SectionResult, line: SectionLine, v_range) -> tuple[float, float, float, float]:
    """(u0, u1, v0, v1): the whole line from A to A' and everything cut."""
    u0, u1 = 0.0, line.length
    v0, v1 = v_range
    if res.bounds is not None:
        u0, u1 = min(u0, res.bounds[0]), max(u1, res.bounds[1])
        v0, v1 = min(v0, res.bounds[2]), max(v1, res.bounds[3])
    if v1 - v0 < 1e-6:
        v0, v1 = v0 - 1.0, v1 + 1.0
    return u0, u1, v0, v1


def fit_section_camera(p: pv.Plotter, res: SectionResult, line: SectionLine, v_range, size) -> float:
    """Camera showing the whole section as large as the map allows, at a standard scale.
    Room is kept at the top for the A / 1 / A' marks and at the bottom for the scale bar.
    Returns the scale denominator (1:ratio on the page printed at 100 %)."""
    w, h = size
    side, top, bottom = mm_to_px(5), mm_to_px(13), mm_to_px(14)
    aw, ah = max(w - 2 * side, 10), max(h - top - bottom, 10)
    u0, u1, v0, v1 = section_extent(res, line, v_range)
    mpp = max((u1 - u0) / aw, (v1 - v0) / ah)              # metres per pixel that fits
    ratio = nice_ratio(mpp * 1000.0 / PAPER_MM_PER_PX)
    mpp = ratio * PAPER_MM_PER_PX / 1000.0
    cu = (u0 + u1) / 2.0
    cv = (v0 + v1) / 2.0 + (top - bottom) / 2.0 * mpp     # centre the data in the free area
    p.camera_position = [(cu, cv, 10000.0), (cu, cv, 0.0), (0.0, 1.0, 0.0)]
    cam = p.renderer.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetParallelScale(h / 2.0 * mpp)
    p.renderer.ResetCameraClippingRange()
    return ratio


def map_legends(res: SectionResult, codes: bool = True) -> list:
    """Legends drawn beside the section map on the page (not inside it):
    ("codes", title, [(text, rgb)]) for drillhole codes, ("bar", title, colours, vmin, vmax)
    for a colour map (block model attribute, numeric drillhole field)."""
    out = []
    if codes:
        out += [("codes", title, entries) for title, entries in res.legends]
    for title, lut in res.scalar_bars[:3]:
        vmin, vmax = (float(v) for v in lut.GetRange())
        rgb = [0.0, 0.0, 0.0]
        colours = []
        for v in np.linspace(vmin, vmax, 128):
            lut.GetColor(float(v), rgb)
            colours.append(tuple(rgb))
        out.append(("bar", title, colours, vmin, vmax))
    return out


# --- the page -------------------------------------------------------------------------
def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


LEGEND_W_MM = 38       # column right of the section map for its legends


def page_layout(paper: str = "A4", section_legend: bool = False) -> dict:
    """Pixel boxes of the page parts (300 dpi, portrait). Each map has a gutter on the
    left and below for its coordinate labels and axis names; with ``section_legend`` the
    section map leaves a column on its right for the legends."""
    W, H = page_size(paper)
    m = mm_to_px(10)
    title = mm_to_px(8)
    gap = mm_to_px(4)
    foot = mm_to_px(6)
    left = mm_to_px(12)          # northing / elevation labels (written upwards) + axis name
    below = mm_to_px(11)         # easting / distance labels + axis name
    right = mm_to_px(4)          # room for the last label running past the frame
    mw = W - 2 * m - left - right
    avail = H - 2 * m - 2 * (title + below) - gap - foot
    plan_h = int(avail * 0.55)
    sec_h = avail - plan_h
    x = m + left
    y = m
    boxes = {"page": (W, H), "margin": m}
    boxes["plan_title"] = QRect(m, y, W - 2 * m, title); y += title
    boxes["plan"] = QRect(x, y, mw, plan_h); y += plan_h + below + gap
    boxes["section_title"] = QRect(m, y, W - 2 * m, title); y += title
    if section_legend:
        lw = mm_to_px(LEGEND_W_MM)
        boxes["section"] = QRect(x, y, mw - lw - gap, sec_h)
        boxes["section_legend"] = QRect(x + mw - lw + right, y, lw, sec_h)
    else:
        boxes["section"] = QRect(x, y, mw, sec_h)
    y += sec_h + below
    boxes["footer"] = QRect(m, y, W - 2 * m, foot)
    return boxes


def _draw_axis_labels(p: QPainter, box: QRect, grid: dict | None, img_size, ink: QColor) -> None:
    """Coordinate labels outside ``box``: below for the horizontal axis, left for the vertical."""
    if not grid:
        return
    iw, ih = img_size
    sx, sy = box.width() / iw, box.height() / ih
    tick = mm_to_px(1.2)
    gap = mm_to_px(0.8)
    f = QFont("Segoe UI")
    f.setPixelSize(mm_to_px(2.6))
    p.setFont(f)
    fm = QFontMetrics(f)
    p.setPen(QPen(ink, max(2, mm_to_px(0.2))))
    step = grid["step"]
    # below: horizontal axis
    last_right = -1e9
    for val, px in zip(grid["u_values"], grid["u_pixels"]):
        if px < 0 or px > iw:
            continue
        x = box.left() + px * sx
        p.drawLine(QPointF(x, box.bottom()), QPointF(x, box.bottom() + tick))
        text = format_value(val, step)
        tw = fm.horizontalAdvance(text)
        if x - tw / 2 < last_right + gap:          # would overlap the previous label
            continue
        p.drawText(QRect(int(x - tw / 2 - 2), box.bottom() + tick + gap, tw + 4, fm.height()),
                   Qt.AlignCenter, text)
        last_right = x + tw / 2
    # left: vertical axis - labels written upwards (rotated 90 degrees), centred on their tick,
    # so long coordinates (e.g. northings) need little room
    last_top = 1e12
    for val, py in sorted(zip(grid["v_values"], grid["v_pixels"]), key=lambda vp: vp[1]):  # bottom up
        if py < 0 or py > ih:
            continue
        y = box.top() + (ih - py) * sy            # VTK pixels count from the bottom
        p.drawLine(QPointF(box.left() - tick, y), QPointF(box.left(), y))
        text = format_value(val, step)
        tw = fm.horizontalAdvance(text)
        if y + tw / 2 > last_top - gap:            # would overlap the label below
            continue
        p.save()
        p.translate(box.left() - tick - gap, y)
        p.rotate(-90)
        p.drawText(QRect(int(-tw / 2 - 2), -fm.height(), tw + 4, fm.height()),
                   Qt.AlignHCenter | Qt.AlignBottom, text)
        p.restore()
        last_top = y - tw / 2
    # axis names
    fb = QFont(f)
    fb.setBold(True)
    p.setFont(fb)
    step_txt = format_value(step, step)
    u_title = tr("{name}    grid {step} m", name=grid["u_name"], step=step_txt)
    p.drawText(QRect(box.left(), box.bottom() + tick + gap + fm.height(), box.width(), fm.height() + gap),
               Qt.AlignHCenter | Qt.AlignTop, u_title)
    p.save()
    p.translate(box.left() - tick - gap - fm.height() - gap, box.center().y())
    p.rotate(-90)
    p.drawText(QRect(-box.height() // 2, -fm.height(), box.height(), fm.height() + gap),
               Qt.AlignCenter, grid["v_name"])
    p.restore()


def _draw_legends(p: QPainter, box: QRect, legends: list, ink: QColor) -> None:
    """Drillhole code keys and colour bars, top to bottom in ``box`` (beside the section)."""
    f = QFont("Segoe UI")
    f.setPixelSize(mm_to_px(2.6))
    fb = QFont(f)
    fb.setBold(True)
    fm = QFontMetrics(f)
    row = int(fm.height() * 1.35)
    sw = mm_to_px(3.0)
    gap = mm_to_px(1.5)
    pen = QPen(ink, max(2, mm_to_px(0.15)))
    y = box.top()
    bottom = box.bottom()
    for item in legends:
        if y + 3 * row > bottom:
            break
        kind, title = item[0], item[1]
        p.setFont(fb)
        p.setPen(ink)
        p.drawText(QRect(box.left(), y, box.width(), row), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(fb).elidedText(title, Qt.ElideRight, box.width()))
        y += row
        p.setFont(f)
        if kind == "codes":
            for text, rgb in item[2]:
                if y + row > bottom:
                    break
                cy = y + row // 2
                p.setPen(pen)
                p.setBrush(QColor.fromRgbF(*rgb))
                p.drawRect(QRect(box.left(), cy - sw // 2, sw, sw))
                p.setBrush(Qt.NoBrush)
                tx = box.left() + sw + gap
                p.drawText(QRect(tx, y, box.right() - tx, row), Qt.AlignLeft | Qt.AlignVCenter,
                           fm.elidedText(text, Qt.ElideRight, box.right() - tx))
                y += row
        else:  # colour bar: vertical, minimum at the bottom
            colours, vmin, vmax = item[2], item[3], item[4]
            bar_h = min(mm_to_px(45), bottom - y - row)
            if bar_h < row * 2:
                break
            bw = mm_to_px(5)
            top = y + row // 3
            n = len(colours)
            for k, rgb in enumerate(colours):
                y1 = top + bar_h - (k + 1) * bar_h / n
                p.fillRect(QRectF(box.left(), y1, bw, bar_h / n + 1), QColor.fromRgbF(*rgb))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(box.left(), top, bw, bar_h))
            for k in range(6):
                v = vmin + (vmax - vmin) * k / 5
                ty = top + bar_h - bar_h * k / 5
                p.drawLine(QPointF(box.left() + bw, ty), QPointF(box.left() + bw + gap * 0.7, ty))
                p.drawText(QRect(box.left() + bw + gap, int(ty - row / 2), box.width() - bw - gap, row),
                           Qt.AlignLeft | Qt.AlignVCenter, f"{v:,.2f}")
            y = top + bar_h + row
        y += row // 2


def compose_page(plan, section, plan_title: str, section_title: str, footer: str,
                 paper: str = "A4") -> QImage:
    """``plan`` / ``section``: MapImage (or a bare QImage, without labels)."""
    legends = section.legends if isinstance(section, MapImage) else []
    L = page_layout(paper, bool(legends))
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
    f = QFont("Segoe UI")
    f.setPixelSize(mm_to_px(3.6))
    f.setBold(True)
    ink = QColor.fromRgbF(*INK)
    for title, mp, tkey, ikey in ((plan_title, plan, "plan_title", "plan"),
                                  (section_title, section, "section_title", "section")):
        img = to_qimage(mp.image) if isinstance(mp, MapImage) else mp
        grid = mp.grid if isinstance(mp, MapImage) else None
        p.setFont(f)
        p.setPen(ink)
        p.drawText(L[tkey], Qt.AlignLeft | Qt.AlignVCenter, title)
        box = L[ikey]
        p.drawImage(box, img)
        p.setPen(QPen(ink, max(2, mm_to_px(0.25))))
        p.drawRect(box)
        _draw_axis_labels(p, box, grid, (img.width(), img.height()), ink)
    if legends:
        _draw_legends(p, L["section_legend"], legends, ink)
    f.setBold(False)
    f.setPixelSize(mm_to_px(2.6))
    p.setFont(f)
    p.setPen(QColor("#555555"))
    p.drawText(L["footer"], Qt.AlignRight | Qt.AlignVCenter, footer)
    p.end()
    return out


def export_page(path: str, scene, line: SectionLine, res: SectionResult, section_renderer,
                v_range, half_width: float, paper: str = "A4", grid: bool = True,
                legend: bool = True) -> QImage:
    has_legends = bool(map_legends(res, legend))
    L = page_layout(paper, has_legends)
    pb, sb = L["plan"], L["section"]
    plan = render_plan(scene, line, (pb.width(), pb.height()), grid)
    origin_v = float(scene.origin[line.axis]) if scene.origin is not None else 0.0
    section = render_section(section_renderer, res, line, origin_v, v_range,
                             (sb.width(), sb.height()), grid, legend)
    view = {2: tr("Plan view (XY)"), 1: tr("Section view looking north (XZ)"),
            0: tr("Section view looking west (YZ)")}
    brg = line.bearing()
    plan_title = tr("{view} - section line {name}", view=view[line.axis], name=SECTION_NAME)
    parts = [tr("Cross-section {name}", name=SECTION_NAME), tr("length {v:,.1f} m", v=line.length)]
    if brg is not None:
        parts.append(tr("bearing {v:.1f}°", v=brg))
    parts.append(tr("corridor ±{v:g} m", v=half_width))
    parts.append(tr("scale 1:{r:,.0f} (H = V)", r=section.ratio) if section.ratio else tr("scale H = V"))
    sec_title = "    ".join(parts)
    a, b = scene.to_world(line.points[0]), scene.to_world(line.points[-1])
    i1, i2 = line.in_plane_axes
    xyz = "XYZ"
    footer = (f"{START_MARK}: {xyz[i1]} {a[i1]:,.2f}  {xyz[i2]} {a[i2]:,.2f}      "
              f"{END_MARK}: {xyz[i1]} {b[i1]:,.2f}  {xyz[i2]} {b[i2]:,.2f}      "
              f"{paper} {DPI} dpi      GeoViewer  {datetime.now():%Y-%m-%d %H:%M}")
    img = compose_page(plan, section, plan_title, sec_title, footer, paper)
    if not img.save(path, "PNG"):
        raise OSError(tr("cannot write {path}", path=path))
    return img
