"""Drillhole log window: opened by Identify on a drillhole.

Header: hole ID, collar X / Y / Z, length and collar direction.
Left:   strip log of the interval field (lithology) from 0 m at the top to the end of
        the hole, coloured as in the 3D view, with codes / descriptions and a depth axis;
        right of it up to two grade graphs (a field chosen above each), same depth scale.
Right:  the hole path projected on the XZ plane (top) and the XY plane centred on the
        collar (bottom), at the same scale and with a shared X axis.

Everything is drawn by ``paint_log`` with a QPainter, so the same drawing is used on
screen (dark), when printing and for the 300 dpi image (white).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPageLayout, QPageSize, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..io.drillholes import display_dip
from ..scene.layers import HoleLog
from ..scene.scale_bar import nice_length
from ..scene.scene_math import nice_step
from .icons import icon

PRINT_DPI = 300
MAX_GRAPHS = 2                       # grade graphs beside the strip log
GRAPH_COLORS = ("#ff5a4f", "#4fb4ff")  # first and second graph


@dataclass
class LogStyle:
    background: str
    panel: str
    text: str
    muted: str
    grid: str
    frame: str
    trace: str
    accent: str
    scale: float = 1.0          # font / line size factor (screen pixels = 1)


SCREEN = LogStyle("#1b1e22", "#14171a", "#e6e9ec", "#9aa0a6", "#3a4048", "#4a515b", "#7d848c", "#ffd21a")
PAPER = LogStyle("#ffffff", "#ffffff", "#141418", "#55595f", "#c9cdd2", "#141418", "#8a9096", "#e07800")


def _fmt(v: float, step: float) -> str:
    decimals = 0 if step >= 1 else min(3, int(math.ceil(-math.log10(step))))
    s = f"{v:,.{decimals}f}"
    return "0" if s in ("-0", "-0.0") else s


def _font(style: LogStyle, size: float, bold: bool = False) -> QFont:
    f = QFont("Segoe UI")
    f.setPixelSize(max(6, int(round(size * style.scale))))
    f.setBold(bold)
    return f


def _interval_at(log: HoleLog, depth: float):
    for iv in log.intervals:
        if iv[0] <= depth < iv[1] or (depth == iv[1] == log.intervals[-1][1]):
            return iv
    return None


def graph_values(rows) -> list:
    """(from, to, value) of a grade to plot: below detection (-1, Surpac's code) as 0, other
    negative codes (-2 not analysed, ...) left out (a gap in the line)."""
    out = []
    for a, b, v in rows:
        if v == -1:
            v = 0.0
        if v >= 0 and b > a:
            out.append((a, b, v))
    return out


def grade_at(log: HoleLog, name: str, depth: float):
    """The value of grade ``name`` at ``depth`` as stored (None: no sample there)."""
    for a, b, v in log.grades.get(name, []):
        if a <= depth < b or depth == b:
            return v
    return None


def grade_text(v) -> str:
    if v is None:
        return "–"
    if v == -1:
        return tr("below detection")
    if v < 0:
        return tr("not analysed") if v == -2 else f"{v:g}"
    return f"{v:,.3g}" if abs(v) < 100 else f"{v:,.1f}"


def depth_line(log: HoleLog, depth: float | None) -> str:
    """'Depth 31.90 m  –  QV1 Quartz vein  (29.00–37.00 m)', or 'Collar' without a marked depth."""
    if depth is None:
        return tr("Collar")
    text = tr("Depth {d:,.2f} m", d=depth)
    iv = _interval_at(log, depth)
    if iv is not None:
        name = f"{iv[2]} {iv[3]}".strip()
        text += f"   –   {name}  ({iv[0]:,.2f}–{iv[1]:,.2f} m)"
    return text


def header_lines(log: HoleLog, pick_depth: float | None) -> list[str]:
    """The four header lines: hole (length), depth and interval, X / Y / Z and azimuth / dip
    at the marked point (the collar when nothing is marked, e.g. on paper)."""
    at = pick_depth if pick_depth is not None and 0.0 <= pick_depth <= log.length else None
    d = at if at is not None else 0.0
    if len(log.depths) > 1:
        xyz = np.array([np.interp(d, log.depths, log.path[:, k]) for k in range(3)])
    else:
        xyz = np.asarray(log.collar, dtype=np.float64)
    lines = [tr("Drillhole {hole}  ({length:,.2f} m)", hole=log.hole_id, length=log.length),
             depth_line(log, at),
             f"X {xyz[0]:,.3f}     Y {xyz[1]:,.3f}     Z {xyz[2]:,.3f}"]
    ad = log.azimuth_dip(d)
    if ad is not None:
        lines.append(tr("azimuth {az:.1f}°     dip {dip:.1f}°", az=ad[0], dip=display_dip(ad[1])))  # 0 = down
    return lines


# --- the drawing ---------------------------------------------------------------------------
def paint_log(p: QPainter, rect: QRectF, log: HoleLog, style: LogStyle,
              pick_depth: float | None = None, strip_window: tuple[float, float] | None = None,
              draw_header: bool = True, grades: list | None = None, graph_titles: bool = True) -> dict:
    """Draw the whole log into ``rect``. ``grades``: the names of the grades drawn as graphs
    beside the strip log (at most two; their names above them when ``graph_titles`` - on
    screen a combobox is there instead). Returns where the strip log, the graphs and the
    projected paths were drawn ({"strip": ..., "panes": [...]}) so clicks can be turned into
    depths."""
    grades = [g for g in (grades or []) if g in log.grades][:MAX_GRAPHS]
    s = style.scale
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.fillRect(rect, QColor(style.background))
    pad = 10 * s
    r = rect.adjusted(pad, pad, -pad, -pad)

    if not draw_header:      # on screen the header is in the window's top bar
        top = r.top()
        return _paint_body(p, r, top, log, style, pick_depth, strip_window, grades, graph_titles)
    # header: hole and length / depth and interval / X Y Z / azimuth and dip at the marked point
    lines = header_lines(log, pick_depth)
    fh = _font(style, 20, True)
    fs = _font(style, 12)
    h1 = QFontMetricsF(fh).height()
    h2 = QFontMetricsF(fs).height()
    p.setPen(QColor(style.text))
    p.setFont(fh)
    p.drawText(QRectF(r.left(), r.top(), r.width(), h1), Qt.AlignLeft | Qt.AlignVCenter, lines[0])
    p.setFont(fs)
    y = r.top() + h1 + 2 * s
    for k, text in enumerate(lines[1:]):
        p.setPen(QColor(style.text if k < 2 else style.muted))
        p.drawText(QRectF(r.left(), y, r.width(), h2), Qt.AlignLeft | Qt.AlignVCenter, text)
        y += h2 + 2 * s
    top = y + 10 * s
    p.setPen(QPen(QColor(style.grid), max(1.0, s)))
    p.drawLine(QPointF(r.left(), top - 6 * s), QPointF(r.right(), top - 6 * s))
    return _paint_body(p, r, top, log, style, pick_depth, strip_window, grades, graph_titles)


GRAPH_W = 86          # width of one grade graph (screen pixels)
GRAPH_GAP = 10


def _paint_body(p: QPainter, r: QRectF, top: float, log: HoleLog, style: LogStyle, pick_depth,
                strip_window, grades=(), graph_titles=True) -> dict:
    """Strip log (and its grade graphs) on the left, XZ / XY projections on the right, below
    ``top``."""
    s = style.scale
    body = QRectF(r.left(), top, r.width(), r.bottom() - top)
    if grades:
        # as wide as its content: depth axis, rock column, the longest rock label, the graphs -
        # so the graphs follow the labels, and the projections get the rest
        fm = QFontMetricsF(_font(style, 11))
        label_w = max((fm.horizontalAdvance(f"{c}  {d}" if d else c) for _a, _b, c, d, _rgb in log.intervals),
                      default=0.0)
        label_w = min(max(label_w, 50 * s), 170 * s)
        graphs_w = len(grades) * (GRAPH_W + GRAPH_GAP) * s
        left_w = (44 + 40 + 8) * s + label_w + 14 * s + graphs_w      # axis, column, labels, graphs
        left_w = min(left_w, 0.62 * body.width())
        gap = 34 * s                                                   # graphs to the projections
    else:
        left_w = min(max(0.34 * body.width(), 230 * s), 400 * s)
        gap = 14 * s
    strip = _paint_strip(p, QRectF(body.left(), body.top(), left_w, body.height()), log, style, pick_depth,
                         strip_window, grades, graph_titles)
    panes = _paint_projections(p, QRectF(body.left() + left_w + gap, body.top(), body.width() - left_w - gap,
                                         body.height()), log, style, pick_depth)
    p.restore()
    return {"strip": strip, "panes": panes}


def _paint_strip(p: QPainter, rect: QRectF, log: HoleLog, style: LogStyle, pick_depth,
                 window: tuple[float, float] | None = None, grades=(), graph_titles=True) -> dict:
    """Strip log of the depth ``window`` (d0, d1); the whole hole when None. The grade
    graphs (``grades``) at its right, from the top to the bottom of the strip."""
    s = style.scale
    ft = _font(style, 12, True)
    f = _font(style, 11)
    fm = QFontMetricsF(f)
    p.setFont(ft)
    p.setPen(QColor(style.text))
    title_h = QFontMetricsF(ft).height() + 6 * s
    if grades:          # the row holds the graphs' comboboxes: tall enough that they stay above the
        title_h = max(title_h, 32 * s)      # grade labels (the log and the graphs start lower)
    title = f"{log.field_name or tr('Intervals')}" + (f"  ({log.table})" if log.table else "")
    p.drawText(QRectF(rect.left(), rect.top(), rect.width(), title_h), Qt.AlignLeft | Qt.AlignVCenter, title)
    axis_w = 44 * s
    col_w = 40 * s
    top = rect.top() + title_h + fm.height() * 1.6     # room for the "m" above the depth axis
    zoomed = window is not None
    bottom = rect.bottom() - fm.height() * 0.7      # room for the last depth label
    h = max(bottom - top, 10.0)
    L = max(log.length, 1e-6)
    d0, d1 = window if zoomed else (0.0, L)
    span = max(d1 - d0, 1e-6)
    x_col = rect.left() + axis_w
    column = QRectF(x_col, top, col_w, h)

    def y_of(d):
        return top + (d - d0) / span * h

    # depth axis
    step = nice_step(span, max(4, int(h / (40 * s))))
    p.setFont(f)
    p.setPen(QPen(QColor(style.muted), max(1.0, 0.8 * s)))
    d = math.ceil(d0 / step - 1e-9) * step
    while d <= d1 + 1e-9:
        y = y_of(d)
        p.drawLine(QPointF(x_col - 5 * s, y), QPointF(x_col, y))
        p.drawText(QRectF(rect.left(), y - fm.height() / 2, axis_w - 8 * s, fm.height()),
                   Qt.AlignRight | Qt.AlignVCenter, _fmt(d, step))
        d += step
    p.drawText(QRectF(rect.left(), top - fm.height() * 1.5, axis_w - 8 * s, fm.height()),
               Qt.AlignRight | Qt.AlignVCenter, "m")

    # intervals (clipped to the window)
    graph_w, graph_gap = GRAPH_W * s, GRAPH_GAP * s
    graphs_left = rect.right() - len(grades) * (graph_w + graph_gap) + graph_gap
    text_x = x_col + col_w + 8 * s
    text_w = (graphs_left - graph_gap if grades else rect.right()) - text_x
    p.setPen(QPen(QColor(style.frame), max(1.0, 0.6 * s)))
    p.setBrush(Qt.NoBrush)
    p.drawRect(column)
    for a, b, code, desc, rgb in log.intervals:
        a, b = max(a, 0.0, d0), min(b, L, d1)
        if b <= a:
            continue
        box = QRectF(x_col, y_of(a), col_w, y_of(b) - y_of(a))
        p.fillRect(box, QColor.fromRgbF(*rgb))
        p.setPen(QPen(QColor(style.frame), max(1.0, 0.6 * s)))
        if a > d0:
            p.drawLine(QPointF(x_col, box.top()), QPointF(x_col + col_w, box.top()))
        if box.height() >= fm.height() * 0.95 and text_w > 10 * s:
            label = f"{code}  {desc}" if desc else code
            p.setPen(QColor(style.text))
            p.drawText(QRectF(text_x, box.center().y() - fm.height() / 2, text_w, fm.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, fm.elidedText(label, Qt.ElideRight, text_w))
    p.setPen(QPen(QColor(style.frame), max(1.0, 0.8 * s)))
    p.drawRect(column)
    if zoomed:   # on the row of the "m" unit, so the log keeps its size when zooming
        p.setPen(QColor(style.muted))
        note = tr("showing {d0:,.2f}–{d1:,.2f} m   (double-click: whole hole)", d0=d0, d1=d1)
        w = (graphs_left - graph_gap if grades else rect.right()) - x_col
        p.drawText(QRectF(x_col, top - fm.height() * 1.5, w, fm.height()), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(note, Qt.ElideRight, w))
    # grade graphs: a line through the middle of each sample, its value across the graph;
    # the field name (on screen: a combobox) in the title row, the grade ticks in the row of
    # the "m" unit, the graph as high as the strip log
    graphs, titles = [], []
    for k, name in enumerate(grades):
        box = QRectF(graphs_left + k * (graph_w + graph_gap), top, graph_w, h)
        graphs.append(box)
        titles.append(QRectF(box.left(), rect.top(), box.width(), title_h))
        if graph_titles:
            p.setFont(ft)
            p.setPen(QColor(GRAPH_COLORS[k % len(GRAPH_COLORS)]))
            p.drawText(QRectF(box.left(), rect.top(), box.width(), title_h), Qt.AlignLeft | Qt.AlignVCenter,
                       QFontMetricsF(ft).elidedText(name, Qt.ElideRight, box.width()))
            p.setFont(f)
        _paint_graph(p, box, graph_values(log.grades.get(name, [])), y_of, style,
                     QColor(GRAPH_COLORS[k % len(GRAPH_COLORS)]), f)
    # picked depth (across the graphs too)
    if pick_depth is not None and d0 <= pick_depth <= d1:
        y = y_of(pick_depth)
        pen = QPen(QColor(style.accent), max(1.5, 1.6 * s))
        p.setPen(pen)
        p.drawLine(QPointF(x_col - 3 * s, y), QPointF(x_col + col_w + 3 * s, y))
        for box in graphs:
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        tri = QPolygonF([QPointF(x_col - 3 * s, y), QPointF(x_col - 11 * s, y - 5 * s),
                         QPointF(x_col - 11 * s, y + 5 * s)])
        p.setBrush(QColor(style.accent))
        p.drawPolygon(tri)
        p.setBrush(Qt.NoBrush)
    # area: the whole strip (zoom / pan); column: the diagram itself (picking)
    return {"area": QRectF(rect.left(), top - fm.height(), rect.width(), h + 2 * fm.height()),
            "column": column, "top": top, "h": h, "d0": d0, "d1": d1, "length": L,
            "graphs": graphs, "graph_titles": titles}


def grade_ticks(vmax: float) -> tuple[float, list[float]]:
    """(right end of the grade axis, tick values): a few round values from 0, the axis
    ending on the last tick at or above the hole's highest value."""
    if vmax <= 0:
        return 1.0, [0.0]
    best = None
    for k in (2, 3, 4):                     # round steps; at most 4 ticks, the axis as short as possible
        step = nice_step(vmax, k)
        n = int(math.ceil(vmax / step - 1e-9))
        if n + 1 <= 4 and (best is None or n * step < best[0] * best[1] - 1e-12):
            best = (n, step)
    n, step = best if best is not None else (int(math.ceil(vmax / nice_step(vmax, 2) - 1e-9)), nice_step(vmax, 2))
    return n * step, [i * step for i in range(n + 1)]


def _paint_graph(p: QPainter, box: QRectF, rows, y_of, style: LogStyle, color: QColor, font: QFont) -> None:
    """One grade against depth: a line through the middle of each sample (broken where there
    is no sample). Axes: depth down the left edge (no ticks - the strip log has them), grade
    along the top with a few round ticks, labelled above it."""
    s = style.scale
    vmax = max((v for _a, _b, v in rows), default=0.0)
    end, ticks = grade_ticks(vmax)
    pad = 3 * s
    w = box.width() - 2 * pad
    fm = QFontMetricsF(font)
    axis = QPen(QColor(style.muted), max(1.0, 0.8 * s))
    p.setPen(axis)
    p.drawLine(QPointF(box.left(), box.top()), QPointF(box.left(), box.bottom()))       # depth axis
    p.drawLine(QPointF(box.left(), box.top()), QPointF(box.right(), box.top()))         # grade axis
    p.setFont(font)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    xs = [box.left() + pad + t / end * w for t in ticks]
    # labels at the start and the end only (never overlapping): the first starts at the axis,
    # the last ends at the graph's right edge; the ticks between are not labelled
    labels = []                                          # (text, left, width)
    for i in sorted({0, len(ticks) - 1}):
        text = _fmt(ticks[i], step)
        tw = fm.horizontalAdvance(text)
        labels.append((text, box.left() if i == 0 else box.right() - tw, tw))
    for x in xs:
        p.setPen(axis)
        p.drawLine(QPointF(x, box.top()), QPointF(x, box.top() - 4 * s))
    p.setPen(QColor(style.muted))
    for text, x0, tw in labels:
        p.drawText(QRectF(x0, box.top() - 5 * s - fm.height(), tw + 1, fm.height()),
                   Qt.AlignLeft | Qt.AlignBottom, text)
    if vmax <= 0:
        return
    p.save()
    p.setClipRect(box)
    p.setPen(QPen(color, max(1.2, 1.4 * s), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    line: list = []
    prev_to = None
    for a, b, v in rows:
        if prev_to is not None and a > prev_to + 1e-6 and line:      # unsampled stretch: break
            p.drawPolyline(QPolygonF(line))
            line = []
        line.append(QPointF(box.left() + pad + v / end * w, y_of((a + b) / 2.0)))   # clipped to the box
        prev_to = b
    if line:
        if len(line) == 1:
            p.drawPoint(line[0])
        p.drawPolyline(QPolygonF(line))
    p.restore()


def _paint_projections(p: QPainter, rect: QRectF, log: HoleLog, style: LogStyle, pick_depth) -> list:
    s = style.scale
    f = _font(style, 11)
    ft = _font(style, 12, True)
    fm = QFontMetricsF(f)
    gap = 12 * s
    pane_h = (rect.height() - gap) / 2
    m_left, m_bottom, m_top, m_right = 62 * s, fm.height() * 2.4, QFontMetricsF(ft).height() + 6 * s, 10 * s
    plot_w = rect.width() - m_left - m_right
    plot_h = pane_h - m_top - m_bottom
    if plot_w < 20 or plot_h < 20:
        return []
    P, c = log.path, log.collar
    # same scale in both panes: metres per pixel that fits X, Z (top) and Y (bottom)
    half_x = max(np.abs(P[:, 0] - c[0]).max(), 1e-6)
    half_y = max(np.abs(P[:, 1] - c[1]).max(), 1e-6)
    zmin, zmax = float(P[:, 2].min()), float(P[:, 2].max())
    floor = 0.08 * log.length            # a vertical hole still gets a sensible width
    half_x, half_y = max(half_x, floor), max(half_y, floor)
    mpp = max(2 * half_x / plot_w, 2 * half_y / plot_h, max(zmax - zmin, floor) / plot_h) * 1.15
    cx, cy, cz = c[0], c[1], (zmin + zmax) / 2
    panes = [
        (tr("Section X–Z"), QRectF(rect.left() + m_left, rect.top() + m_top, plot_w, plot_h),
         P[:, 0], P[:, 2], cx, cz, tr("X (East)"), tr("Z (Elevation)"), False),
        (tr("Plan X–Y  (centred on the collar)"),
         QRectF(rect.left() + m_left, rect.top() + pane_h + gap + m_top, plot_w, plot_h),
         P[:, 0], P[:, 1], cx, cy, tr("X (East)"), tr("Y (North)"), True),
    ]
    seg_rgb = []
    mids = (log.depths[:-1] + log.depths[1:]) / 2
    for dmid in mids:
        iv = _interval_at(log, float(dmid))
        seg_rgb.append(iv[4] if iv is not None else None)
    pick_xyz = None
    if pick_depth is not None and len(log.depths) > 1:
        pick_xyz = np.array([np.interp(pick_depth, log.depths, P[:, k]) for k in range(3)])
    out = []
    for title, box, U, V, cu, cv, uname, vname, plan in panes:
        pts = _paint_pane(p, box, title, U, V, cu, cv, mpp, uname, vname, seg_rgb, log, style, pick_xyz,
                          plan=plan)
        out.append({"box": box, "pts": pts})
    return out


def _paint_pane(p, box: QRectF, title, U, V, cu, cv, mpp, uname, vname, seg_rgb, log, style, pick_xyz,
                plan: bool) -> np.ndarray:
    s = style.scale
    f = _font(style, 11)
    ft = _font(style, 12, True)
    fm = QFontMetricsF(f)

    def to_px(u, v):
        return QPointF(box.center().x() + (u - cu) / mpp, box.center().y() - (v - cv) / mpp)

    p.fillRect(box, QColor(style.panel))
    p.setFont(ft)
    p.setPen(QColor(style.text))
    p.drawText(QRectF(box.left(), box.top() - QFontMetricsF(ft).height() - 4 * s, box.width(),
                      QFontMetricsF(ft).height()), Qt.AlignLeft | Qt.AlignVCenter, title)
    # grid + coordinates outside the frame
    step = nice_step(mpp * min(box.width(), box.height() * 1.6), 6)
    p.setFont(f)
    grid_pen = QPen(QColor(style.grid), max(1.0, 0.6 * s))
    u0, u1 = cu - box.width() / 2 * mpp, cu + box.width() / 2 * mpp
    v0, v1 = cv - box.height() / 2 * mpp, cv + box.height() / 2 * mpp
    u = math.ceil(u0 / step) * step
    while u <= u1:
        x = to_px(u, cv).x()
        p.setPen(grid_pen)
        p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
        p.setPen(QColor(style.muted))
        p.drawText(QRectF(x - 60 * s, box.bottom() + 3 * s, 120 * s, fm.height()), Qt.AlignHCenter | Qt.AlignTop,
                   _fmt(u, step))
        u += step
    v = math.ceil(v0 / step) * step
    while v <= v1:
        y = to_px(cu, v).y()
        p.setPen(grid_pen)
        p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        p.setPen(QColor(style.muted))
        p.drawText(QRectF(box.left() - 62 * s, y - fm.height() / 2, 58 * s, fm.height()),
                   Qt.AlignRight | Qt.AlignVCenter, _fmt(v, step))
        v += step
    p.setPen(QColor(style.muted))
    p.drawText(QRectF(box.left(), box.bottom() + fm.height() + 4 * s, box.width(), fm.height()),
               Qt.AlignHCenter | Qt.AlignTop, tr("{axis}    grid {step} m", axis=uname, step=_fmt(step, step)))
    p.save()
    p.translate(box.left() - 62 * s + 2 * s, box.center().y())
    p.rotate(-90)
    p.drawText(QRectF(-box.height() / 2, -2 * s, box.height(), fm.height()), Qt.AlignHCenter | Qt.AlignTop, vname)
    p.restore()

    p.save()
    p.setClipRect(box)
    pts = [to_px(a, b) for a, b in zip(U, V)]
    # the whole path thin and grey, then the intervals in their colours
    p.setPen(QPen(QColor(style.trace), max(1.0, 1.2 * s)))
    p.drawPolyline(QPolygonF(pts))
    k = 0
    while k < len(seg_rgb):                      # one polyline per run of the same colour
        rgb = seg_rgb[k]
        j = k
        while j + 1 < len(seg_rgb) and seg_rgb[j + 1] == rgb:
            j += 1
        if rgb is not None:
            pen = QPen(QColor.fromRgbF(*rgb), max(2.0, 4.0 * s))
            pen.setCapStyle(Qt.FlatCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(QPolygonF(pts[k:j + 2]))
        k = j + 1
    # collar, end of hole, picked depth
    p.setPen(QPen(QColor(style.text), max(1.0, s)))
    p.setBrush(QColor(style.background))
    p.drawEllipse(pts[0], 5 * s, 5 * s)
    p.setBrush(QColor(style.text))
    p.drawRect(QRectF(pts[-1].x() - 3.5 * s, pts[-1].y() - 3.5 * s, 7 * s, 7 * s))
    p.setBrush(Qt.NoBrush)
    p.setFont(f)
    p.drawText(QRectF(pts[0].x() + 8 * s, pts[0].y() - fm.height(), 200 * s, fm.height()),
               Qt.AlignLeft | Qt.AlignVCenter, log.hole_id)
    avoid = [QRectF(pts[0].x() - 6 * s, pts[0].y() - fm.height(), 14 * s + fm.horizontalAdvance(log.hole_id),
                    fm.height() + 6 * s)]
    # length of the hole at its end (no "EOH" prefix)
    p.setPen(QColor(style.muted))
    near = math.hypot(pts[-1].x() - pts[0].x(), pts[-1].y() - pts[0].y()) < fm.height() * 1.5
    dy = fm.height() * 0.9 if near else 0.0      # e.g. a vertical hole in plan: below the hole ID
    end_text = f"{log.length:,.1f} m"
    p.drawText(QRectF(pts[-1].x() + 8 * s, pts[-1].y() - fm.height() / 2 + dy, 200 * s, fm.height()),
               Qt.AlignLeft | Qt.AlignVCenter, end_text)
    avoid.append(QRectF(pts[-1].x() - 5 * s, pts[-1].y() - fm.height() / 2 + dy - 2 * s,
                        13 * s + fm.horizontalAdvance(end_text), fm.height() + 4 * s))
    if pick_xyz is not None:
        q = to_px(pick_xyz[0], pick_xyz[2] if not plan else pick_xyz[1])
        p.setPen(QPen(QColor(style.accent), max(1.5, 1.6 * s)))
        p.setBrush(QColor(style.accent))
        p.drawEllipse(q, 4.5 * s, 4.5 * s)
        p.setBrush(Qt.NoBrush)
    p.restore()
    path_px = np.array([[q.x(), q.y()] for q in pts])
    if plan:                                     # the north arrow's corner
        avoid.append(QRectF(box.right() - 30 * s, box.top(), 30 * s, 30 * s + fm.height()))
    _paint_scale_bar(p, box, mpp, style, path_px, avoid)   # a corner clear of the hole and the labels
    if plan:   # north arrow
        x, y = box.right() - 16 * s, box.top() + 10 * s
        p.setPen(QPen(QColor(style.text), max(1.0, s)))
        p.setBrush(QColor(style.text))
        p.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x - 5 * s, y + 14 * s), QPointF(x + 5 * s, y + 14 * s)]))
        p.setBrush(Qt.NoBrush)
        p.drawText(QRectF(x - 10 * s, y + 15 * s, 20 * s, fm.height()), Qt.AlignHCenter | Qt.AlignTop, "N")
    p.setPen(QPen(QColor(style.frame), max(1.0, 0.8 * s)))
    p.drawRect(box)
    return path_px


def depth_at(layout: dict | None, log: HoleLog | None, x: float, y: float, reach: float = 20.0):
    """Depth along the hole under widget position (x, y): on the strip log, or the nearest
    point of the projected path in the XZ / XY panes (within ``reach`` pixels). None if
    the position is not on the hole."""
    if not layout or log is None:
        return None
    st = layout.get("strip")
    on_graph = st and any(g.contains(QPointF(x, y)) for g in st.get("graphs", []))
    if st and (on_graph or st["column"].adjusted(-3, -3, 3, 3).contains(QPointF(x, y))):   # on the diagrams
        d = st["d0"] + (y - st["top"]) / st["h"] * (st["d1"] - st["d0"])
        return float(np.clip(d, max(st["d0"], 0.0), min(st["d1"], log.length)))
    for pane in layout.get("panes", []):
        if not pane["box"].contains(QPointF(x, y)) or len(pane["pts"]) < 2:
            continue
        a, b = pane["pts"][:-1], pane["pts"][1:]
        ab = b - a
        L2 = (ab ** 2).sum(axis=1)
        t = np.clip(((np.array([x, y]) - a) * ab).sum(axis=1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        near = a + ab * t[:, None]
        d = np.hypot(near[:, 0] - x, near[:, 1] - y)
        k = int(np.argmin(d))
        if d[k] <= reach:
            return float(log.depths[k] + t[k] * (log.depths[k + 1] - log.depths[k]))
    return None


SCALE_BAR_CORNERS = ("bottom-left", "bottom-right", "top-left", "top-right")   # tried in this order


def _scale_bar_geometry(box: QRectF, mpp: float, style: LogStyle, corner: str, frac: float) -> dict:
    """Where the scale bar's parts go when it sits in ``corner`` of the pane, about ``frac``
    of the pane wide."""
    s = style.scale
    fm = QFontMetricsF(_font(style, 10))
    length = nice_length(mpp * box.width() * frac)
    px = length / mpp
    div, unit = (1000.0, "km") if length >= 1000 else (1.0, "m")
    labels = ["0", f"{length / 2 / div:,.6g}", f"{length / div:,.6g} {unit}"]
    bar_h, tick, pad, margin = 6 * s, 3 * s, 6 * s, 12 * s
    w0, w2 = fm.horizontalAdvance(labels[0]), fm.horizontalAdvance(labels[2])
    frame_w = w0 / 2 + px + w2 / 2 + 2 * pad
    frame_h = pad + fm.height() + 2 * s + tick + bar_h + pad
    left = box.left() + margin if corner.endswith("left") else box.right() - margin - frame_w
    top = box.bottom() - margin - frame_h if corner.startswith("bottom") else box.top() + margin
    if corner == "top-right":                      # below the north arrow of the plan
        top += 30 * s
    frame = QRectF(left, top, frame_w, frame_h)
    x0 = left + pad + w0 / 2
    label_y = top + pad
    y_bar = label_y + fm.height() + 2 * s + tick
    return {"frame": frame, "x0": x0, "px": px, "y_bar": y_bar, "label_y": label_y, "labels": labels,
            "bar_h": bar_h, "tick": tick}


def _dense(path_px: np.ndarray, step: float) -> np.ndarray:
    """Points every ``step`` pixels along a polyline (so a long straight leg crossing a box is seen)."""
    if len(path_px) < 2:
        return np.asarray(path_px, dtype=float).reshape(-1, 2)
    out = [path_px[:1]]
    for a, b in zip(path_px[:-1], path_px[1:]):
        n = max(int(math.ceil(math.hypot(*(b - a)) / step)), 1)
        t = np.arange(1, n + 1)[:, None] / n
        out.append(a + (b - a) * t)
    return np.vstack(out)


def scale_bar_place(box: QRectF, mpp: float, style: LogStyle, path_px=None, avoid=()) -> dict:
    """The scale bar's place: the first corner (bottom left, bottom right, top left, top
    right) where it covers neither the hole path nor the rectangles ``avoid`` (labels, the
    north arrow); a shorter bar if none is free; else where it covers the least."""
    s = style.scale
    pts = _dense(np.asarray(path_px, dtype=float), 3.0) if path_px is not None and len(path_px) else \
        np.zeros((0, 2))
    best, best_hits = None, None
    for frac in (0.3, 0.2):
        for corner in SCALE_BAR_CORNERS:
            g = _scale_bar_geometry(box, mpp, style, corner, frac)
            fr = g["frame"].adjusted(-4 * s, -4 * s, 4 * s, 4 * s)         # room for the line width
            if not box.contains(g["frame"]):
                continue
            inside = ((pts[:, 0] >= fr.left()) & (pts[:, 0] <= fr.right()) &
                      (pts[:, 1] >= fr.top()) & (pts[:, 1] <= fr.bottom())) if len(pts) else np.zeros(0, bool)
            hits = int(inside.sum()) + 1000 * sum(1 for r in avoid if r.intersects(fr))
            if hits == 0:
                g["corner"] = corner
                return g
            if best_hits is None or hits < best_hits:
                best, best_hits = dict(g, corner=corner), hits
    return best if best is not None else dict(_scale_bar_geometry(box, mpp, style, "bottom-left", 0.2),
                                              corner="bottom-left")


def _paint_scale_bar(p: QPainter, box: QRectF, mpp: float, style: LogStyle, path_px=None, avoid=()) -> QRectF:
    """Scale bar in a corner of a pane that is clear of the hole (see ``scale_bar_place``):
    one dark and one white half, ticks up, 0 / middle / end labels, on a framed box (same
    look as the cross-section print). Returns its frame."""
    s = style.scale
    f = _font(style, 10)
    fm = QFontMetricsF(f)
    g = scale_bar_place(box, mpp, style, path_px, avoid)
    x0, px, y_bar, label_y = g["x0"], g["px"], g["y_bar"], g["label_y"]
    bar_h, tick, labels = g["bar_h"], g["tick"], g["labels"]
    p.setPen(QPen(QColor(style.frame), max(1.0, 0.6 * s)))
    p.setBrush(QColor(style.background))
    p.drawRect(g["frame"])
    paper = QColor(style.background).lightness() > 128
    dark = QColor(20, 20, 24) if paper else QColor("#5f666f")   # dark half visible on a dark screen
    light = QColor(255, 255, 255)
    p.setPen(Qt.NoPen)
    p.setBrush(dark)
    p.drawRect(QRectF(x0, y_bar, px / 2, bar_h))
    p.setBrush(light)
    p.drawRect(QRectF(x0 + px / 2, y_bar, px / 2, bar_h))
    p.setBrush(Qt.NoBrush)
    outline = dark if paper else QColor(style.text)
    p.setPen(QPen(outline, max(1.0, 0.7 * s)))
    p.drawRect(QRectF(x0, y_bar, px, bar_h))
    p.setFont(f)
    for frac, text in zip((0.0, 0.5, 1.0), labels):
        x = x0 + px * frac
        p.drawLine(QPointF(x, y_bar), QPointF(x, y_bar - tick))
        p.setPen(QColor(style.text))
        w = fm.horizontalAdvance(text)
        p.drawText(QRectF(x - w / 2 - 2, label_y, w + 4, fm.height()), Qt.AlignCenter, text)
        p.setPen(QPen(outline, max(1.0, 0.7 * s)))
    return g["frame"]


# --- window ----------------------------------------------------------------------------------
class HoleLogView(QWidget):
    """Click (or drag) on the strip log or near the path in either pane: the marker moves
    to that depth in all three views."""

    depth_picked = Signal(float)
    grades_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.log: HoleLog | None = None
        self.pick_depth: float | None = None
        self.layout_info: dict | None = None
        self.window: tuple[float, float] | None = None     # strip log depth window (None: whole)
        self._pan: tuple[float, tuple[float, float]] | None = None   # (start y, window at start)
        self.grades: list[str] = []                         # the fields drawn as graphs (<= 2)
        self._wanted: list[str] = []                        # the user's choice, kept from hole to hole
        self.combos: list[QComboBox] = []
        for k in range(MAX_GRAPHS):
            c = QComboBox(self)
            c.setToolTip(tr("Field drawn in this graph (0 at the left, the hole's highest value at the right)"))
            c.setStyleSheet(f"QComboBox {{ color: {GRAPH_COLORS[k]}; font-weight: 600; }}")
            c.currentTextChanged.connect(lambda text, k=k: self._choose(k, text))
            c.hide()
            self.combos.append(c)
        self.setMinimumSize(640, 420)
        self.setMouseTracking(True)

    def set_log(self, log: HoleLog, pick_depth: float | None) -> None:
        self.log, self.pick_depth = log, pick_depth
        self.window = None
        names = list(log.grades)
        n = min(len(names), MAX_GRAPHS)                     # one field: one graph; none: no graph
        chosen = [g for g in self._wanted if g in names][:n]
        for g in names:                                     # fill up with the fields not shown yet
            if len(chosen) >= n:
                break
            if g not in chosen:
                chosen.append(g)
        self.grades = chosen
        for k, c in enumerate(self.combos):
            c.blockSignals(True)
            c.clear()
            if k < n:
                c.addItems(names)
                c.setCurrentText(chosen[k])
            c.blockSignals(False)
            c.setVisible(k < n)
        self.update()

    def _choose(self, k: int, text: str) -> None:
        if not text or k >= len(self.grades):
            return
        self.grades[k] = text
        self._wanted = list(self.grades)
        self.update()
        self.grades_changed.emit(list(self.grades))

    def paintEvent(self, _e):
        p = QPainter(self)
        if self.log is None:
            p.fillRect(self.rect(), QColor(SCREEN.background))
        else:
            self.layout_info = paint_log(p, QRectF(self.rect()), self.log, SCREEN, self.pick_depth,
                                         self.window, draw_header=False, grades=self.grades, graph_titles=False)
        p.end()
        self._place_combos()

    def _place_combos(self) -> None:
        """Each combobox above its graph."""
        boxes = ((self.layout_info or {}).get("strip") or {}).get("graph_titles", [])
        for k, c in enumerate(self.combos):
            if k < len(boxes) and k < len(self.grades):
                b = boxes[k]
                h = min(c.sizeHint().height(), int(b.height()) + 4)
                geo = (int(b.left()), int(b.center().y() - h / 2), int(b.width()), h)
                if c.geometry().getRect() != geo:
                    c.setGeometry(*geo)

    def _depth(self, e):
        pos = e.position()
        return depth_at(self.layout_info, self.log, pos.x(), pos.y())

    def pick(self, depth: float) -> None:
        self.pick_depth = depth
        self.update()
        self.depth_picked.emit(depth)

    # --- strip log zoom / pan ----------------------------------------------------------
    MIN_SPAN = 0.5          # metres: the deepest zoom

    def _strip(self):
        return (self.layout_info or {}).get("strip")

    def _in_strip(self, pos) -> bool:
        st = self._strip()
        return bool(st) and st["area"].contains(pos)

    def _set_window(self, d0: float, d1: float) -> None:
        L = self.log.length
        span = min(max(d1 - d0, self.MIN_SPAN), L)
        d0 = min(max(d0, 0.0), L - span)
        self.window = None if span >= L - 1e-9 else (d0, d0 + span)
        self.update()

    def wheelEvent(self, e):
        st = self._strip()
        pos = e.position()
        if self.log is None or not st or not st["area"].contains(pos):
            return super().wheelEvent(e)
        steps = e.angleDelta().y() / 120.0
        if not steps:
            return
        d0, d1 = st["d0"], st["d1"]
        frac = min(max((pos.y() - st["top"]) / st["h"], 0.0), 1.0)
        anchor = d0 + frac * (d1 - d0)               # this depth stays under the cursor
        span = (d1 - d0) * (0.8 ** steps)
        self._set_window(anchor - frac * span, anchor + (1 - frac) * span)
        e.accept()

    def _start_pan(self, pos) -> None:
        st = self._strip()
        self._pan = (pos.y(), (st["d0"], st["d1"]))
        self.setCursor(Qt.ClosedHandCursor)

    def mouseDoubleClickEvent(self, e):
        if self._in_strip(e.position()):
            self.window = None                       # whole hole again
            self.update()
            return
        super().mouseDoubleClickEvent(e)

    def mousePressEvent(self, e):
        pos = e.position()
        if e.button() == Qt.LeftButton:
            d = self._depth(e)
            if d is not None:                        # on the diagram or a path: pick
                self.pick(d)
                return
            if self._in_strip(pos):                  # beside the diagram: pan
                self._start_pan(pos)
                return
        if e.button() == Qt.MiddleButton and self._in_strip(pos):
            self._start_pan(pos)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._pan is not None and e.buttons() & (Qt.LeftButton | Qt.MiddleButton):
            y0, (d0, d1) = self._pan
            st = self._strip()
            shift = -(pos.y() - y0) / st["h"] * (d1 - d0)
            self._set_window(d0 + shift, d1 + shift)
            return
        d = self._depth(e)
        if d is not None:
            self.setCursor(Qt.PointingHandCursor)
        elif self._in_strip(pos):
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        if d is not None and e.buttons() & Qt.LeftButton:
            self.pick(d)

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.setCursor(Qt.OpenHandCursor if self._in_strip(e.position()) else Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(e)


class DrillholeLogDialog(QDialog):
    """Non-modal window showing one drillhole; Identify on another hole replaces it."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(tr("Drillhole"))
        self.setWindowIcon(icon("format-list-bulleted-type", "#3d9bff"))
        self.resize(1100, 760)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget(self)
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 8, 8, 4)
        # header (hole and length / depth and interval / X Y Z with azimuth and dip) on the left
        head = QVBoxLayout()
        head.setSpacing(1)
        self.head_title = QLabel("")
        self.head_title.setStyleSheet(f"font-size: 17px; font-weight: 600; color: {SCREEN.text};")
        self.head_depth = QLabel("")
        self.head_depth.setStyleSheet(f"color: {SCREEN.text};")
        self.head_point = QLabel("")
        self.head_point.setStyleSheet(f"color: {SCREEN.muted};")
        for lab in (self.head_title, self.head_depth, self.head_point):
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            head.addWidget(lab)
        h.addLayout(head, 1)
        right = QVBoxLayout()
        right.setSpacing(2)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        right.addLayout(buttons)
        self.info = QLabel("")               # short messages: "Saved: ...", "Sent to the printer"
        self.info.setStyleSheet(f"color: {SCREEN.muted}; font-size: 11px;")
        self.info.setAlignment(Qt.AlignRight | Qt.AlignTop)
        right.addWidget(self.info)
        right.addStretch(1)
        h.addLayout(right)
        for name, text, tip, slot in (
            ("printer-outline", tr("Print…"), tr("Print the log and both projections on one page"),
             self.print_dialog),
            ("image-move", tr("Save image…"),
             tr("Save the page as a {dpi} dpi PNG (white background)", dpi=PRINT_DPI),
             self.save_image_dialog),
        ):
            b = QToolButton(self)
            b.setIcon(icon(name))
            b.setText(text)
            b.setToolTip(tip)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        v.addWidget(bar)
        self.view = HoleLogView(self)
        self.view.depth_picked.connect(self._show_depth)
        self.view.grades_changed.connect(lambda _g: self._show_depth(self.view.pick_depth))
        v.addWidget(self.view, 1)
        self.workspace_folder: str = ""

    def show_hole(self, log: HoleLog, pick_depth: float | None = None) -> None:
        self.view.set_log(log, pick_depth)
        self.setWindowTitle(tr("Drillhole {hole}", hole=log.hole_id))
        self._show_depth(pick_depth)
        self.show()
        self.raise_()

    def _show_depth(self, depth: float | None) -> None:
        """Header in the top bar: hole (length) / depth and interval / X Y Z, azimuth and dip
        of the marked point."""
        log = self.view.log
        if log is None:
            return
        lines = header_lines(log, depth)
        self.head_title.setText(lines[0])
        values = "" if depth is None else "".join(f"     {g} {grade_text(grade_at(log, g, depth))}"
                                                  for g in self.view.grades)
        self.head_depth.setText(lines[1] + values if depth is not None
                                else tr("Click the log, a graph or the path to mark a depth"))
        self.head_point.setText("          ".join(lines[2:]))
        self.info.setText("")

    # --- output --------------------------------------------------------------------------
    def render_page(self, p: QPainter, rect: QRectF, dpi: float) -> None:
        style = LogStyle(**{**PAPER.__dict__, "scale": dpi / 96.0})
        paint_log(p, rect, self.view.log, style, None, grades=self.view.grades)   # no depth marker on paper

    def page_image(self, dpi: int = PRINT_DPI) -> QImage:
        """A4 landscape page on white."""
        w, h = int(round(297 / 25.4 * dpi)), int(round(210 / 25.4 * dpi))
        img = QImage(w, h, QImage.Format_RGB32)
        img.fill(QColor("white"))
        dpm = int(round(dpi / 0.0254))
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        p = QPainter(img)
        margin = 10 / 25.4 * dpi
        self.render_page(p, QRectF(margin, margin, w - 2 * margin, h - 2 * margin), dpi)
        p.end()
        return img

    def save_image_dialog(self) -> None:
        if self.view.log is None:
            return
        start = str(Path(self.workspace_folder or ".") / f"drillhole_{self.view.log.hole_id}.png")
        path, _ = QFileDialog.getSaveFileName(self, tr("Save drillhole log"), start, tr("PNG image (*.png)"))
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not self.page_image().save(path, "PNG"):
            QMessageBox.warning(self, tr("Save drillhole log"), tr("Could not write {path}", path=path))
            return
        self.info.setText(tr("Saved: {path}", path=path))

    def print_dialog(self) -> None:
        if self.view.log is None:
            return
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter

        printer = QPrinter(QPrinter.HighResolution)
        printer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Landscape,
                                          printer.pageLayout().margins()))
        printer.setDocName(tr("Drillhole {hole}", hole=self.view.log.hole_id))
        dlg = QPrintDialog(printer, self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.print_to(printer)
        self.info.setText(tr("Sent to the printer"))

    def print_to(self, printer) -> None:
        from PySide6.QtPrintSupport import QPrinter

        p = QPainter(printer)
        try:
            rect = printer.pageRect(QPrinter.DevicePixel)
            self.render_page(p, QRectF(0, 0, rect.width(), rect.height()), printer.resolution())
        finally:
            p.end()
