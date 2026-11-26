"""The wall sketch of a pit (坑道素描图): the drive unfolded along its chainage.

The usual "two walls and the back" layout: the back (roof) in the middle as seen from above (as on
the plan), the left wall folded out above it and the right wall below it, each wall's floor edge on
the outside. Chainage runs to the right, at the same scale across and along (a true-scale
sketch, with a scale bar). It shows the rocks (filled by rock colour, codes written in), the
structures (their trace on the wall, worked out from the dip direction / dip and the drive's
bearing, with the reading 135∠45), the samples (bars at their height, numbered) and the survey
stations on the chainage axis.

One QPainter routine ``paint_sketch`` draws it dark on screen and white on paper (the same look as
the drillhole log: ui/drillhole_log.LogStyle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QToolTip, QWidget

from ..i18n import tr
from ..io.pit_geology_store import (DEFAULT_SAMPLE_HEIGHT, INTERVAL_WALLS, SAMPLE_KINDS, STRUCTURE_COLORS,
                                    STRUCTURE_KINDS, WALLS, PitGeology, orientation_text)
from ..scene.scene_math import nice_step
from .drillhole_log import SCREEN, LogStyle, _font, _paint_scale_bar

SAMPLE_COLOR = "#1de9b6"
STRIPS = ("left", "back", "right")      # top to bottom


@dataclass
class SketchData:
    """What the sketch shows of one pit."""
    title: str
    width: float                        # m, the profile
    height: float
    length: float | None                # m along the drive to the last station (None: no survey yet)
    stations: dict                      # name -> chainage
    geo: PitGeology
    rock_colors: dict = field(default_factory=dict)     # code -> "#rrggbb"
    rock_desc: dict = field(default_factory=dict)       # code -> description
    bearing: object = None              # callable chainage -> drive bearing (None: not known)

    @property
    def extent(self) -> float:
        """The chainage the sketch covers: the survey and every record (at least 10 m)."""
        return max(self.length or 0.0, self.geo.extent(), 10.0)


def _strip_heights(d: SketchData) -> dict:
    return {"left": d.height, "back": d.width, "right": d.height}


def _readable_on(color: QColor) -> QColor:
    return QColor("#101014") if color.lightness() > 140 else QColor("#ffffff")


def _paper(style: LogStyle) -> bool:
    return QColor(style.background).lightness() > 128


def _kind_color(kind: str, style: LogStyle) -> QColor:
    c = QColor(STRUCTURE_COLORS.get(kind, "#b0b6bd"))
    if _paper(style) and c.lightness() > 200:            # white / pale on paper: draw it dark
        c = QColor("#141418")
    return c


def sketch_scale(rect: QRectF, d: SketchData, style: LogStyle, legend: bool = True) -> tuple:
    """(left, top, available width, available height) of the strips' area within ``rect`` and the
    scale that fits the strips' height in it (px per m)."""
    s = style.scale
    label_w = 86 * s
    legend_w = 170 * s if legend else 0.0
    top = 40 * s
    bottom = 70 * s
    aw = max(rect.width() - label_w - legend_w - 8 * s, 10.0)
    ah = max(rect.height() - top - bottom, 10.0)
    total = sum(_strip_heights(d).values()) or 1.0
    return rect.left() + label_w, rect.top() + top, aw, ah, ah / total


def paint_sketch(p: QPainter, rect: QRectF, d: SketchData, style: LogStyle, m0: float, ppm: float,
                 highlight=None, legend: bool = True) -> dict:
    """Draw the sketch in ``rect`` from chainage ``m0`` at ``ppm`` pixels per metre (both ways).
    ``highlight``: (kind, index) of a record drawn marked. Returns the layout: strips, the x of a
    chainage and the boxes of the records (for picking and tool tips)."""
    s = style.scale
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.fillRect(rect, QColor(style.background))
    x0, y0, aw, ah, fit = sketch_scale(rect, d, style, legend)
    ppm = min(ppm, fit)
    m1 = m0 + aw / ppm
    hs = _strip_heights(d)
    strips, y = {}, y0
    for k in STRIPS:
        strips[k] = QRectF(x0, y, aw, hs[k] * ppm)
        y += hs[k] * ppm
    body = QRectF(x0, y0, aw, y - y0)

    def xm(m):
        return x0 + (m - m0) * ppm

    def y_on(wall, h):
        """Screen y of a height above the floor on a wall (left: floor at the top edge)."""
        r = strips[wall]
        h = min(max(h, 0.0), d.height)
        return r.top() + h * ppm if wall == "left" else r.bottom() - h * ppm

    boxes = []
    f_small, f_norm, f_bold = _font(style, 9), _font(style, 10), _font(style, 10, True)
    fm_small = QFontMetricsF(f_small)

    # strips' background and the part past the last station
    for k in STRIPS:
        p.fillRect(strips[k], QColor(style.panel))
    if d.length is not None and d.length < m1:
        lx = max(xm(d.length), x0)
        past = QRectF(lx, body.top(), body.right() - lx, body.height())
        p.fillRect(past, QColor(style.grid).darker(130) if not _paper(style) else QColor("#eef0f2"))
    p.setClipRect(body)

    # rocks
    for i, r in enumerate(d.geo.intervals):
        if r.frm is None or r.to is None or r.to <= r.frm or r.to < m0 or r.frm > m1:
            continue
        c = QColor(d.rock_colors.get(r.rock) or "#8a9096")
        for w in r.walls():
            if w not in strips:
                continue
            band = QRectF(xm(r.frm), strips[w].top(), (r.to - r.frm) * ppm, strips[w].height())
            p.fillRect(band, c)
            p.setPen(QPen(QColor(style.frame), max(1.0, 0.8 * s)))
            p.drawLine(QPointF(band.left(), band.top()), QPointF(band.left(), band.bottom()))
            p.drawLine(QPointF(band.right(), band.top()), QPointF(band.right(), band.bottom()))
            p.setFont(f_bold)
            tw = QFontMetricsF(f_bold).horizontalAdvance(r.rock)
            if band.width() > tw + 6 * s and band.height() > 14 * s:
                p.setPen(_readable_on(c))
                p.drawText(band, Qt.AlignCenter, r.rock)
            boxes.append((band, "interval", i))
            if highlight == ("interval", i):
                p.setPen(QPen(QColor(style.accent), 2.5 * s))
                p.setBrush(Qt.NoBrush)
                p.drawRect(band.adjusted(1, 1, -1, -1))

    # station lines through the strips
    p.setPen(QPen(QColor(style.grid), max(1.0, 0.6 * s), Qt.DashLine))
    for name, c in d.stations.items():
        if m0 <= c <= m1:
            p.drawLine(QPointF(xm(c), body.top()), QPointF(xm(c), body.bottom()))

    # samples
    rib = max(0.25 * ppm, 4 * s)
    label_right: dict = {}                  # per strip row: where the last sample number ended
    for i, smp in enumerate(d.geo.samples):
        if smp.frm is None or smp.to is None or smp.to <= smp.frm or smp.to < m0 or smp.frm > m1:
            continue
        if smp.wall in ("left", "right"):
            h = DEFAULT_SAMPLE_HEIGHT if smp.height is None else smp.height
            yc = y_on(smp.wall, h)
        elif smp.wall == "back":
            yc = strips["back"].center().y()
        else:                                                      # face: at the end, across the back
            yc = strips["back"].center().y()
        bar = QRectF(xm(smp.frm), yc - rib / 2, max((smp.to - smp.frm) * ppm, 2 * s), rib)
        p.setPen(QPen(QColor("#0b3d33") if not _paper(style) else QColor("#0a5c4c"), max(1.0, 0.8 * s)))
        p.setBrush(QColor(SAMPLE_COLOR))
        p.drawRect(bar)
        p.setFont(f_small)
        p.setPen(QColor(style.text))
        tw = fm_small.horizontalAdvance(smp.name)
        label = QRectF(bar.center().x() - tw / 2 - 2, bar.bottom() + 1 * s, tw + 4, fm_small.height())
        row = round(yc)
        if label.left() > label_right.get(row, -1e9) + 2 * s:      # numbers that would overlap are left out
            p.drawText(label, Qt.AlignCenter, smp.name)
            label_right[row] = label.right()
        boxes.append((bar.adjusted(-2, -2, 2, 2), "sample", i))
        if highlight == ("sample", i):
            p.setPen(QPen(QColor(style.accent), 2.5 * s))
            p.setBrush(Qt.NoBrush)
            p.drawRect(bar.adjusted(-2, -2, 2, 2))

    # structures: the trace through the point, the reading beside it
    from ..scene.pit_geology import trace_direction

    for i, st in enumerate(d.geo.structures):
        if st.at is None or not (m0 - 5 <= st.at <= m1 + 5):
            continue
        col = _kind_color(st.kind, style)
        x = xm(st.at)
        if st.wall in ("left", "right"):
            h = d.height / 2 if st.height is None else st.height
            y = y_on(st.wall, h)
            strip = strips[st.wall]
        else:
            strip = strips["back"]
            y = strip.center().y()
        pt = QPointF(x, y)
        pen = QPen(col, (3.0 if highlight == ("structure", i) else 2.0) * s)
        if st.kind == "fault":
            pen.setWidthF(pen.widthF() * 1.4)
        p.setPen(pen)
        drawn = False
        if st.wall != "face" and st.dip is not None and st.dip_direction is not None and d.bearing is not None:
            tdir = trace_direction(st.dip_direction, st.dip, d.bearing(st.at), st.wall)
            if tdir is not None:
                along, other = tdir
                sy = 1.0 if st.wall == "left" else -1.0     # left wall: up is down the screen; back: left is up
                L = max(strip.height(), 30 * s)
                a = QPointF(x - along * L, y - sy * other * L)
                b = QPointF(x + along * L, y + sy * other * L)
                p.save()
                p.setClipRect(strip)
                p.drawLine(a, b)
                p.restore()
                drawn = True
        if not drawn:                                          # a symbol: no reading or on the face
            r = 5 * s
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygonF([QPointF(x, y - r), QPointF(x + r, y), QPointF(x, y + r), QPointF(x - r, y)]))
        p.setBrush(col)
        p.setPen(Qt.NoPen)
        p.drawEllipse(pt, 2.6 * s, 2.6 * s)
        text = orientation_text(st.dip_direction, st.dip) or tr(STRUCTURE_KINDS.get(st.kind, st.kind))
        if st.wall == "face":
            text = tr("face") + " " + text
        p.setFont(f_small)
        tw = fm_small.horizontalAdvance(text)
        tr_box = QRectF(x + 4 * s, y - fm_small.height() - 2 * s, tw + 4, fm_small.height())
        if tr_box.top() < strip.top():                          # at the strip's edge: below the point
            tr_box.moveTop(y + 2 * s)
        p.setPen(QColor(style.text))
        p.drawText(tr_box, Qt.AlignLeft | Qt.AlignVCenter, text)
        boxes.append((QRectF(x - 7 * s, y - 7 * s, 14 * s, 14 * s), "structure", i))
        if highlight == ("structure", i):
            p.setPen(QPen(QColor(style.accent), 2 * s))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(pt, 8 * s, 8 * s)
    p.setClipping(False)

    # frames, the fold lines and the strip names
    p.setPen(QPen(QColor(style.frame), max(1.0, 0.9 * s)))
    p.setBrush(Qt.NoBrush)
    p.drawRect(body)
    for k in ("back", "right"):
        p.drawLine(QPointF(x0, strips[k].top()), QPointF(body.right(), strips[k].top()))
    p.setFont(f_norm)
    p.setPen(QColor(style.text))
    names = {"left": tr("Left wall"), "back": tr("Back"), "right": tr("Right wall")}
    for k in STRIPS:
        r = strips[k]
        p.drawText(QRectF(rect.left(), r.top(), x0 - rect.left() - 6 * s, r.height()),
                   Qt.AlignRight | Qt.AlignVCenter | Qt.TextWordWrap, names[k])
    p.setFont(f_small)
    p.setPen(QColor(style.muted))
    for k, txt in (("left", tr("floor")), ("right", tr("floor"))):
        r = strips[k]
        yy = r.top() if k == "left" else r.bottom() - fm_small.height()
        p.drawText(QRectF(rect.left(), yy, x0 - rect.left() - 6 * s, fm_small.height()), Qt.AlignRight, txt)

    # chainage axis on top: ticks and the stations
    widest = fm_small.horizontalAdvance(f"{max(abs(m0), abs(m1)):,.0f}0") + 14 * s
    step = nice_step(widest / ppm, 1)                       # labels never overlap
    p.setPen(QPen(QColor(style.frame), max(1.0, 0.8 * s)))
    p.setFont(f_small)
    k = math.ceil(m0 / step)
    while k * step <= m1 + 1e-9:
        m = k * step
        x = xm(m)
        p.setPen(QPen(QColor(style.frame), max(1.0, 0.8 * s)))
        p.drawLine(QPointF(x, body.top()), QPointF(x, body.top() - 5 * s))
        p.drawLine(QPointF(x, body.bottom()), QPointF(x, body.bottom() + 5 * s))
        lab = f"{m:,.{0 if step >= 1 else 1}f}"
        tw = fm_small.horizontalAdvance(lab)
        p.setPen(QColor(style.text))
        p.drawText(QRectF(x - tw / 2 - 2, body.bottom() + 6 * s, tw + 4, fm_small.height()), Qt.AlignCenter, lab)
        k += 1
    last_right = -1e9
    for name, c in sorted(d.stations.items(), key=lambda kv: kv[1]):
        if not (m0 <= c <= m1):
            continue
        x = xm(c)
        tri = QPolygonF([QPointF(x - 4 * s, body.top() - 12 * s), QPointF(x + 4 * s, body.top() - 12 * s),
                         QPointF(x, body.top() - 5 * s)])
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(style.accent) if not _paper(style) else QColor("#1f4e79"))
        p.drawPolygon(tri)
        tw = fm_small.horizontalAdvance(name)
        if x - tw / 2 < last_right + 4 * s:                  # no room for the name: the marker only
            continue
        last_right = x + tw / 2
        p.setPen(QColor(style.text))
        p.drawText(QRectF(x - tw / 2 - 2, body.top() - 13 * s - fm_small.height(), tw + 4, fm_small.height()),
                   Qt.AlignCenter, name)
    p.setPen(QColor(style.muted))
    unit = tr("Chainage (m)")
    p.drawText(QRectF(rect.left(), body.bottom() + 6 * s, x0 - rect.left() - 6 * s, fm_small.height()),
               Qt.AlignRight, unit)

    # scale bar below the strips, at the bottom left
    bar_box = QRectF(x0 - 12 * s, body.bottom() + 6 * s + fm_small.height(), min(aw, (m1 - m0) * ppm) + 12 * s,
                     min(rect.bottom() - body.bottom() - 6 * s - fm_small.height(), 64 * s))
    if bar_box.height() > 40 * s:
        _paint_scale_bar(p, bar_box, 1.0 / ppm, style)

    if legend:
        _paint_legend(p, QRectF(body.right() + 14 * s, body.top(), rect.right() - body.right() - 14 * s,
                                rect.bottom() - body.top()), d, style)
    p.restore()
    return {"strips": strips, "body": body, "m0": m0, "ppm": ppm, "m1": m1, "boxes": boxes}


def _paint_legend(p: QPainter, r: QRectF, d: SketchData, style: LogStyle) -> None:
    """Rocks, structure types and samples used in the pit, in a column."""
    s = style.scale
    f, fb = _font(style, 9), _font(style, 9, True)
    fm = QFontMetricsF(f)
    lh = fm.height() + 3 * s
    y = r.top()
    rocks = []
    for iv in d.geo.intervals:
        if iv.rock and iv.rock not in rocks:
            rocks.append(iv.rock)
    kinds = []
    for st in d.geo.structures:
        if st.kind not in kinds:
            kinds.append(st.kind)

    def head(text):
        nonlocal y
        p.setFont(fb)
        p.setPen(QColor(style.text))
        p.drawText(QRectF(r.left(), y, r.width(), lh), Qt.AlignLeft | Qt.AlignVCenter, text)
        y += lh

    def line(text):
        p.setFont(f)
        p.setPen(QColor(style.text))
        p.drawText(QRectF(r.left() + 22 * s, y, r.width() - 22 * s, lh), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(text, Qt.ElideRight, r.width() - 22 * s))

    if rocks:
        head(tr("Rocks"))
        for code in rocks:
            p.setPen(QPen(QColor(style.frame), max(1.0, 0.6 * s)))
            p.setBrush(QColor(d.rock_colors.get(code) or "#8a9096"))
            p.drawRect(QRectF(r.left(), y + 3 * s, 16 * s, lh - 6 * s))
            desc = d.rock_desc.get(code, "")
            line(f"{code} ({desc})" if desc else code)
            y += lh
        y += 4 * s
    if kinds:
        head(tr("Structures"))
        for k in kinds:
            p.setPen(QPen(_kind_color(k, style), 2.2 * s))
            p.drawLine(QPointF(r.left(), y + lh / 2), QPointF(r.left() + 16 * s, y + lh / 2))
            line(tr(STRUCTURE_KINDS.get(k, k)))
            y += lh
        line_txt = tr("Reading: dip direction∠dip")
        p.setFont(f)
        p.setPen(QColor(style.muted))
        p.drawText(QRectF(r.left(), y, r.width(), lh), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(line_txt, Qt.ElideRight, r.width()))
        y += lh + 4 * s
    if d.geo.samples:
        head(tr("Samples"))
        p.setPen(QPen(QColor("#0a5c4c"), max(1.0, 0.6 * s)))
        p.setBrush(QColor(SAMPLE_COLOR))
        p.drawRect(QRectF(r.left(), y + lh / 2 - 3 * s, 16 * s, 6 * s))
        line(tr("Sample (numbered)"))
        y += lh + 4 * s
    p.setFont(f)
    p.setPen(QColor(style.muted))
    note = tr("Left / right as seen looking toward the face. The walls are folded out; the back is seen "
              "from above, as on the plan.")
    p.drawText(QRectF(r.left(), y, r.width(), r.bottom() - y), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, note)


def record_tip(d: SketchData, kind: str, i: int) -> str:
    """The tool tip of a record on the sketch."""
    g = d.geo
    if kind == "interval" and i < len(g.intervals):
        r = g.intervals[i]
        desc = d.rock_desc.get(r.rock, "")
        lines = [f"{r.rock}" + (f" ({desc})" if desc else ""), f"{r.frm:.2f} - {r.to:.2f} m, "
                 + tr(INTERVAL_WALLS.get(r.wall, r.wall)).lower()]
        lines += [v for v in (r.description, r.alteration, r.mineralization) if v]
        return "\n".join(lines)
    if kind == "structure" and i < len(g.structures):
        st = g.structures[i]
        lines = [tr(STRUCTURE_KINDS.get(st.kind, st.kind)) + (f"  {orientation_text(st.dip_direction, st.dip)}"
                                                               if st.dip is not None else ""),
                 f"{st.at:.2f} m, " + tr(WALLS.get(st.wall, st.wall)).lower()]
        lines += [v for v in (st.fill, st.description) if v]
        return "\n".join(lines)
    if kind == "sample" and i < len(g.samples):
        sm = g.samples[i]
        lines = [f"{sm.name}  ({tr(SAMPLE_KINDS.get(sm.kind, sm.kind)).lower()})", f"{sm.frm:.2f} - {sm.to:.2f} m, "
                 + tr(WALLS.get(sm.wall, sm.wall)).lower()]
        lines += [f"{k} = {v:g}" for k, v in sm.values.items() if v is not None]
        return "\n".join(lines)
    return ""


class WallSketch(QWidget):
    """The sketch on screen: wheel zooms along the drive (around the cursor), drag pans,
    double-click shows the whole pit; click a record to pick it, hover for its details."""

    picked = Signal(str, int)            # kind ("interval" / "structure" / "sample"), row

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.data: SketchData | None = None
        self.m0 = 0.0
        self.ppm = None                  # None: the largest that fits the height
        self.highlight = None
        self.layout_info = None
        self._drag = None
        self.style = SCREEN

    def set_data(self, d: SketchData | None, keep_view: bool = True) -> None:
        if not keep_view or self.data is None or d is None or d.title != self.data.title:
            self.m0, self.ppm = 0.0, None
        self.data = d
        self.update()

    def set_highlight(self, kind: str | None, i: int | None = None) -> None:
        self.highlight = (kind, i) if kind is not None else None
        self.update()

    def show_chainage(self, m: float) -> None:
        """Pan so chainage ``m`` is in view (a third from the left) when it is not."""
        lay = self.layout_info
        if lay is None or lay["m0"] <= m <= lay["m1"]:
            return
        self.m0 = max(m - (lay["m1"] - lay["m0"]) / 3.0, min(0.0, m))
        self.update()

    def _fit_ppm(self) -> float:
        _x, _y, aw, _ah, fit = sketch_scale(QRectF(self.rect()), self.data, self.style)
        return fit, aw

    def fit_all(self) -> None:
        if self.data is None:
            return
        fit, aw = self._fit_ppm()
        self.m0 = 0.0
        self.ppm = min(fit, aw / self.data.extent)
        self.update()

    def visible_range(self) -> tuple:
        lay = self.layout_info
        return (lay["m0"], lay["m1"]) if lay else (0.0, self.data.extent if self.data else 10.0)

    def paintEvent(self, _e):
        p = QPainter(self)
        if self.data is None:
            p.fillRect(self.rect(), QColor(self.style.background))
            p.setPen(QColor(self.style.muted))
            p.drawText(self.rect(), Qt.AlignCenter, tr("Select a pit"))
            p.end()
            return
        fit, _aw = self._fit_ppm()
        ppm = fit if self.ppm is None else min(self.ppm, fit)
        self.layout_info = paint_sketch(p, QRectF(self.rect()), self.data, self.style, self.m0, ppm,
                                        self.highlight)
        p.end()

    def wheelEvent(self, e):
        if self.data is None or self.layout_info is None:
            return
        lay = self.layout_info
        fit, aw = self._fit_ppm()
        x = e.position().x()
        m_at = lay["m0"] + (x - lay["body"].left()) / lay["ppm"]
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        lo = aw / max(self.data.extent * 4, 10.0)
        self.ppm = min(max(lay["ppm"] * f, lo), fit)
        self.m0 = m_at - (x - lay["body"].left()) / self.ppm
        self.update()

    def _hit(self, pos):
        if self.layout_info is None:
            return None
        for box, kind, i in reversed(self.layout_info["boxes"]):
            if box.contains(QPointF(pos)):
                return kind, i
        return None

    def mousePressEvent(self, e):
        if e.button() in (Qt.LeftButton, Qt.MiddleButton) and self.layout_info is not None:
            self._drag = (e.position().x(), self.layout_info["m0"], e.position().x())

    def mouseMoveEvent(self, e):
        if self._drag is not None and self.layout_info is not None:
            x0, m0, _ = self._drag
            self._drag = (x0, m0, e.position().x())
            self.m0 = m0 - (e.position().x() - x0) / self.layout_info["ppm"]
            self.update()
            return
        hit = self._hit(e.position().toPoint())
        if hit is not None and self.data is not None:
            QToolTip.showText(e.globalPosition().toPoint(), record_tip(self.data, *hit), self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, e):
        drag, self._drag = self._drag, None
        if drag is not None and abs(drag[2] - drag[0]) < 3:          # a click, not a pan
            hit = self._hit(e.position().toPoint())
            if hit is not None:
                self.picked.emit(*hit)

    def mouseDoubleClickEvent(self, _e):
        self.fit_all()
