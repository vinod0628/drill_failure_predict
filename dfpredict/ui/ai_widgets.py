"""Building blocks of the decision-support windows, made to be read at a glance by anyone:
big numbers, traffic-light findings in plain sentences, and simple charts (drawn with Qt, so they
work offline in the release without a plotting library)."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from ..ai import BAD, INFO, OK, WARN
from ..i18n import tr, trc
from . import theme

GREEN, AMBER, RED, BLUE = "#4cc76a", "#f0b429", "#ff6b6b", "#3d9bff"
LEVEL_COLOR = {OK: GREEN, WARN: AMBER, BAD: RED, INFO: BLUE}
LEVEL_SYMBOL = {OK: "✔", WARN: "!", BAD: "✖", INFO: "i"}


def level_word(level: str) -> str:
    return {OK: tr("OK"), WARN: trc("finding", "Check"), BAD: tr("Problem"), INFO: trc("finding", "Note")}[level]


def badge(level: str, size: int = 22) -> QLabel:
    lab = QLabel(LEVEL_SYMBOL[level])
    lab.setFixedSize(size, size)
    lab.setAlignment(Qt.AlignCenter)
    lab.setStyleSheet(f"background: {LEVEL_COLOR[level]}; color: #111; border-radius: {size // 2}px; "
                      f"font-weight: 700; font-size: {max(8, size // 2)}pt;")
    return lab


class FindingList(QWidget):
    """Findings as rows: a coloured symbol and a sentence."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSpacing(6)

    def set_findings(self, findings) -> None:
        while self.box.count():
            w = self.box.takeAt(0).widget()
            if w is not None:
                w.hide()                        # gone at once (deleteLater waits for the event loop)
                w.deleteLater()
        for level, text in findings:
            row = QFrame()
            row.setStyleSheet(f"QFrame {{ background: {theme.PANEL_ALT}; border-left: 4px solid "
                              f"{LEVEL_COLOR[level]}; border-radius: 3px; }}")
            h = QHBoxLayout(row)
            h.setContentsMargins(8, 6, 8, 6)
            h.addWidget(badge(level), 0, Qt.AlignTop)
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet("border: none; background: transparent; font-size: 10pt;")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            h.addWidget(lab, 1)
            self.box.addWidget(row)
        self.box.addStretch(1)

    def texts(self) -> list[str]:
        out = []
        for i in range(self.box.count()):
            w = self.box.itemAt(i).widget()
            if w is not None:
                out += [lab.text() for lab in w.findChildren(QLabel) if len(lab.text()) > 1]
        return out


class BigNumber(QFrame):
    """A card: a small title, a big value and a line under it."""

    def __init__(self, title: str, color: str = BLUE, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background: {theme.PANEL_ALT}; border-radius: 6px; "
                           f"border-top: 3px solid {color}; }} QLabel {{ border: none; background: transparent; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(2)
        self.title = QLabel(title.upper())
        self.title.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt; font-weight: 600;")
        self.value = QLabel("-")
        self.value.setStyleSheet(f"color: {color}; font-size: 17pt; font-weight: 700;")
        self.sub = QLabel("")
        self.sub.setStyleSheet(f"color: {theme.MUTED}; font-size: 9pt;")
        self.sub.setWordWrap(True)
        for w in (self.title, self.value, self.sub):
            v.addWidget(w)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, value: str, sub: str = "", color: str | None = None) -> None:
        self.value.setText(value)
        self.sub.setText(sub)
        if color:
            self.value.setStyleSheet(f"color: {color}; font-size: 17pt; font-weight: 700;")


def heading(text: str, sub: str = "") -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 4)
    v.setSpacing(2)
    t = QLabel(text)
    t.setStyleSheet("font-size: 14pt; font-weight: 700;")
    v.addWidget(t)
    if sub:
        s = QLabel(sub)
        s.setWordWrap(True)
        s.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(s)
    return w


# --- chart helpers ------------------------------------------------------------------------------------------
def nice(v: float) -> float:
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    return next(m * e for m in (1, 2, 2.5, 5, 10) if m * e >= v)


def tonnes_text(t: float) -> str:
    """Tonnes always in t (no kt / Mt), with thousands separators."""
    return tr("{v:,.0f} t", v=t)


class _Chart(QWidget):
    PAD = (58, 16, 18, 40)            # left, top, right, bottom

    def __init__(self, parent=None, min_h: int = 240):
        super().__init__(parent)
        self.setMinimumHeight(min_h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_rect(self) -> QRectF:
        l, t, r, b = self.PAD
        return QRectF(l, t, max(10, self.width() - l - r), max(10, self.height() - t - b))

    def _bg(self, p: QPainter) -> None:
        p.fillRect(self.rect(), QColor(theme.PANEL))

    @staticmethod
    def _text(p, x, y, text, color=None, align=Qt.AlignLeft | Qt.AlignVCenter, size=8.5, bold=False, w=200, h=16):
        f = QFont(p.font())
        f.setPointSizeF(size)
        f.setBold(bold)
        p.setFont(f)
        p.setPen(QColor(color or theme.MUTED))
        if align & Qt.AlignRight:
            r = QRectF(x - w, y - h / 2, w, h)
        elif align & Qt.AlignHCenter:
            r = QRectF(x - w / 2, y - h / 2, w, h)
        else:
            r = QRectF(x, y - h / 2, w, h)
        p.drawText(r, int(align), text)


class ForecastChart(_Chart):
    """Ore mined so far (filled), the forecast to the end of the reserve (dashed) with the slow - fast
    range (shaded), the total reserve line and today."""
    PAD = (96, 16, 18, 40)

    def __init__(self, parent=None):
        super().__init__(parent, 260)
        self.f = None
        self.lines = {}

    def set_forecast(self, f, lines: dict) -> None:
        """``lines``: name -> (month offsets, cumulative tonnes) for "expected", "slow", "fast"."""
        self.f, self.lines = f, lines
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._bg(p)
        f = self.f
        if f is None or not f.ok:
            self._text(p, self.width() / 2, self.height() / 2, tr("No forecast yet"), align=Qt.AlignCenter, w=400)
            return
        R = self.plot_rect()
        n_hist = len(f.months)
        x_end = max([n_hist - 1] + [xs[-1] for xs, _ys in self.lines.values() if xs])
        x_end = max(x_end, n_hist + 2)
        x_end += max(1, int(round(0.06 * x_end)))          # room after the end (labels, the "runs out" mark)
        y_max = f.total * 1.08 if f.total > 0 else 1.0

        def X(i):
            return R.left() + (i / max(x_end, 1)) * R.width()

        def Y(v):
            return R.bottom() - (v / y_max) * R.height()

        # grid + y labels
        step = nice(y_max / 5)
        p.setPen(QPen(QColor(theme.BORDER), 1))
        v = 0.0
        while v <= y_max:
            p.drawLine(QPointF(R.left(), Y(v)), QPointF(R.right(), Y(v)))
            self._text(p, R.left() - 6, Y(v), tonnes_text(v), align=Qt.AlignRight | Qt.AlignVCenter, w=92)
            v += step
        # x labels: years (or every few months for a short span)
        from ..ai.reserve_advisor import _mi, _month, month_text

        m0 = _mi(f.months[0])
        every = 12 if x_end > 30 else (3 if x_end > 9 else 1)
        px_year = R.width() / max(x_end / 12.0, 1e-9)
        year_step = next((n for n in (1, 2, 5, 10, 20, 50) if n * px_year >= 44), 50)   # labels never touch
        for i in range(0, int(x_end) + 1):
            m = m0 + i
            if (every == 12 and m % 12 == 0 and (m // 12) % year_step == 0) or (every < 12 and i % every == 0):
                p.setPen(QPen(QColor(theme.BORDER), 1, Qt.DotLine))
                p.drawLine(QPointF(X(i), R.top()), QPointF(X(i), R.bottom()))
                label = str(m // 12) if every == 12 else month_text(_month(m))
                self._text(p, X(i), R.bottom() + 12, label, align=Qt.AlignCenter, w=80)
        # slow - fast band
        slow, fast = self.lines.get("slow"), self.lines.get("fast")
        if slow and fast and slow[0] and fast[0]:
            poly = [QPointF(X(x), Y(y)) for x, y in zip(*fast)]
            poly += [QPointF(X(x), Y(y)) for x, y in reversed(list(zip(*slow)))]
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(240, 180, 41, 50))
            p.drawPolygon(QPolygonF(poly))
        # mined so far (filled area)
        path = QPainterPath(QPointF(X(0), Y(0)))
        for i, c in enumerate(f.cumulative):
            path.lineTo(QPointF(X(i), Y(c)))
        path.lineTo(QPointF(X(n_hist - 1), Y(0)))
        path.closeSubpath()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(224, 83, 63, 120))
        p.drawPath(path)
        p.setPen(QPen(QColor("#e0533f"), 2.5))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF([QPointF(X(i), Y(c)) for i, c in enumerate(f.cumulative)]))
        # expected forecast
        exp = self.lines.get("expected")
        if exp and exp[0]:
            p.setPen(QPen(QColor(AMBER), 2.5, Qt.DashLine))
            p.drawPolyline(QPolygonF([QPointF(X(x), Y(y)) for x, y in zip(*exp)]))
            if exp[1][-1] < f.total - 1e-6 and f.end_month:     # beyond the chart: say when, at the right
                self._text(p, R.right() - 4, Y(exp[1][-1]) - 16, tr("runs out ≈ {m} (beyond the chart)",
                                                                      m=month_text(f.end_month)), AMBER,
                           Qt.AlignRight | Qt.AlignVCenter, 9, True, 360)
            if exp[1][-1] >= f.total - 1e-6:
                xe = X(exp[0][-1])
                p.setPen(QPen(QColor(AMBER), 1, Qt.DotLine))
                p.drawLine(QPointF(xe, Y(f.total)), QPointF(xe, R.bottom()))
                label = tr("runs out ≈ {m}", m=month_text(f.end_month))
                f9 = QFont(p.font())
                f9.setPointSizeF(9)
                f9.setBold(True)
                width = QFontMetrics(f9).horizontalAdvance(label) + 12
                left = xe + width > R.right()             # right of the marker, away from the forecast line,
                self._text(p, xe - 6 if left else xe + 6, Y(f.total) + (-12 if left else 14), label, AMBER,
                           (Qt.AlignRight if left else Qt.AlignLeft) | Qt.AlignVCenter, 9, True, 260)
        # total reserve line
        p.setPen(QPen(QColor(GREEN), 1.5, Qt.DashLine))
        p.drawLine(QPointF(R.left(), Y(f.total)), QPointF(R.right(), Y(f.total)))
        self._text(p, R.left() + 6, Y(f.total) - 10, tr("Total reserve {t}", t=tonnes_text(f.total)), GREEN, w=300)
        # today
        xt = X(n_hist - 1)
        p.setPen(QPen(QColor(theme.TEXT), 1))
        p.drawLine(QPointF(xt, R.top()), QPointF(xt, R.bottom()))
        self._text(p, xt + 4, R.bottom() - 10, tr("today"), theme.TEXT, w=80)
        # legend
        y = self.height() - 12
        x = R.left()
        for color, label, style in (("#e0533f", tr("mined so far"), Qt.SolidLine),
                                    (AMBER, tr("forecast at the typical rate"), Qt.DashLine),
                                    ("#7a6a3a", tr("slow - fast months"), Qt.SolidLine)):
            p.setPen(QPen(QColor(color), 3, style))
            p.drawLine(QPointF(x, y), QPointF(x + 22, y))
            self._text(p, x + 27, y, label, w=220)
            x += 40 + p.fontMetrics().horizontalAdvance(label)


class MonthBars(_Chart):
    """Production per month (the months the typical rate comes from highlighted) and the typical month."""
    PAD = (90, 16, 18, 40)

    def __init__(self, parent=None):
        super().__init__(parent, 170)
        self.f = None

    def set_forecast(self, f) -> None:
        self.f = f
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._bg(p)
        f = self.f
        if f is None or not f.ok:
            return
        months, tonnes = f.months[-18:], f.tonnes[-18:]
        R = self.plot_rect()
        top = nice(max(max(tonnes), f.rate_high, 1.0) * 1.15)
        n = len(months)
        bw = R.width() / max(n, 1)
        for k in (0.0, 0.5, 1.0):
            y = R.bottom() - k * R.height()
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
            self._text(p, R.left() - 6, y, tonnes_text(top * k), align=Qt.AlignRight | Qt.AlignVCenter, w=86)
        for i, (m, t) in enumerate(zip(months, tonnes)):
            h = t / top * R.height()
            used = m in f.used_months
            color = QColor("#e0533f") if used else QColor(theme.MUTED)
            color.setAlpha(230 if used else 110)
            p.fillRect(QRectF(R.left() + i * bw + bw * 0.15, R.bottom() - h, bw * 0.7, h), color)
            if n <= 12 or i % 2 == 0:
                from ..scene.production import MONTH_ABBR

                mon = int(m[5:7])
                self._text(p, R.left() + i * bw + bw / 2, R.bottom() + 10, tr(MONTH_ABBR[mon - 1]),
                           align=Qt.AlignCenter, size=7.5, w=bw * 2)
                if i == 0 or mon == 1:
                    self._text(p, R.left() + i * bw + bw / 2, R.bottom() + 23, m[:4], align=Qt.AlignCenter,
                               size=7.5, w=bw * 2)
        y = R.bottom() - f.rate / top * R.height()
        p.setPen(QPen(QColor(AMBER), 2, Qt.DashLine))
        p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
        self._text(p, R.right() - 4, y - 10, tr("typical month {t}", t=tonnes_text(f.rate)), AMBER,
                   Qt.AlignRight | Qt.AlignVCenter, 9, True, 300)


class MisclosureBars(_Chart):
    """The misclosure now and after each candidate correction (log scale), against the allowed misclosure."""
    PAD = (170, 14, 70, 34)

    def __init__(self, parent=None):
        super().__init__(parent, 180)
        self.rows = []                # [(label, metres, passes)]
        self.allowed = 0.0
        self.selected = 1

    def set_rows(self, rows, allowed: float, selected: int = 1) -> None:
        self.rows, self.allowed, self.selected = rows, allowed, selected
        self.setMinimumHeight(60 + 30 * len(rows))
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._bg(p)
        if not self.rows:
            return
        R = self.plot_rect()
        lo, hi = -3.0, max(1.0, math.ceil(math.log10(max(v for _l, v, _p in self.rows) + 1e-9)))

        def X(v):
            lv = math.log10(max(v, 10 ** lo))
            return R.left() + (lv - lo) / (hi - lo) * R.width()

        for e in range(int(lo), int(hi) + 1):
            x = X(10 ** e)
            p.setPen(QPen(QColor(theme.BORDER), 1, Qt.DotLine))
            p.drawLine(QPointF(x, R.top()), QPointF(x, R.bottom()))
            label = {-3: "1 mm", -2: "1 cm", -1: "10 cm", 0: "1 m", 1: "10 m", 2: "100 m", 3: "1 km"}.get(e, f"1e{e} m")
            self._text(p, x, R.bottom() + 12, label, align=Qt.AlignCenter, w=60)
        rh = R.height() / len(self.rows)
        for i, (label, v, ok) in enumerate(self.rows):
            y = R.top() + i * rh
            if i == self.selected:
                p.fillRect(QRectF(0, y, self.width(), rh), QColor(61, 155, 255, 40))
            color = QColor(GREEN if ok else RED)
            p.fillRect(QRectF(R.left(), y + rh * 0.2, X(v) - R.left(), rh * 0.6), color)
            self._text(p, R.left() - 8, y + rh / 2, label, theme.TEXT, Qt.AlignRight | Qt.AlignVCenter, 9,
                       i == 0, 165)
            self._text(p, X(v) + 5, y + rh / 2, f"{v:.3f} m", theme.TEXT, w=70)
        xa = X(self.allowed)
        p.setPen(QPen(QColor(AMBER), 2, Qt.DashLine))
        p.drawLine(QPointF(xa, R.top() - 4), QPointF(xa, R.bottom()))
        self._text(p, xa, R.top() + 2, tr("allowed {m:.3f} m", m=self.allowed), AMBER, Qt.AlignCenter, 8.5, True, 160)


SCREEN = {"bg": None, "text": None, "muted": None, "grid": "#3a4048", "grid_text": "#8a939d"}
PAPER = {"bg": "#ffffff", "text": "#1b1d21", "muted": "#5a6068", "grid": "#d5d9de", "grid_text": "#7a8088"}


class PlanSketch(QWidget):
    """An interactive plan (X east, Y north): polylines, polygons, stations and marked points, with a
    scale bar, a north arrow and a legend. Wheel zooms (around the cursor), drag pans, double-click
    fits; + / - / fit buttons on the map. A click without dragging emits ``clicked(x, y)``.
    ``save_png`` writes the same view at 300 dpi on white."""

    clicked = Signal(float, float)

    def __init__(self, parent=None, min_h: int = 260):
        super().__init__(parent)
        self.setMinimumHeight(min_h)
        self.setMouseTracking(True)
        self.items = []             # ("line", pts, color, width, style, labels) / ("poly", pts, fill, edge)
        self.stations = []          # (pts (N, 2), names, color)
        self.marks = []             # (pt, color, text, shape) - part of the drawing
        self.pins = []              # (pt, color, text, shape) - moved without refitting (start / target)
        self.legend = []            # (color, text, kind)
        self.focus = None           # (lo, hi) plan area to fit; None: everything
        self.guide = None           # (a, b, color): a dashed line (e.g. the straight line start - target)
        self.arrow = None           # (point, bearing): the direction set for leaving the start
        self._c = np.zeros(2)       # view centre (world X, Y)
        self._s = 1.0               # pixels per metre
        self._auto = True           # fit on resize until the user zooms or pans
        self._press = None
        self._moved = False
        self._mouse = None
        bar = QHBoxLayout(self)
        bar.setContentsMargins(8, 8, 8, 8)
        self.buttons = []
        for text, tip, slot in (("+", tr("Zoom in"), lambda: self.zoom_by(1.5)),
                                ("−", tr("Zoom out"), lambda: self.zoom_by(1 / 1.5)),
                                ("⤢", tr("Show everything (double-click)"), self.fit_all)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setFixedSize(30, 28)
            b.setStyleSheet(f"QToolButton {{ background: {theme.PANEL_ALT}; color: {theme.TEXT}; font-size: 13pt; "
                            f"font-weight: 700; border: 1px solid {theme.BORDER}; border-radius: 4px; }} "
                            f"QToolButton:hover {{ background: {theme.HOVER}; }}")
            b.clicked.connect(slot)
            bar.addWidget(b, 0, Qt.AlignTop | Qt.AlignLeft)
            self.buttons.append(b)
        bar.addStretch(1)

    # --- content ------------------------------------------------------------------------------
    def clear(self) -> None:
        self.items, self.stations, self.marks, self.legend, self.focus = [], [], [], [], None
        self.update()

    def line(self, pts, color, width=2.0, style=Qt.SolidLine, labels=None) -> None:
        self.items.append(("line", np.asarray(pts, float)[:, :2], color, width, style, labels))

    def polygon(self, pts, fill, edge) -> None:
        self.items.append(("poly", np.asarray(pts, float)[:, :2], fill, edge, None, None))

    def station_marks(self, pts, names, color="#c8cdd4") -> None:
        if len(pts):
            self.stations.append((np.asarray(pts, float)[:, :2], list(names), color))

    def mark(self, pt, color, text="", shape="dot") -> None:
        self.marks.append((np.asarray(pt, float)[:2], color, text, shape))

    def set_guide(self, a=None, b=None, color="#f0b429") -> None:
        self.guide = None if a is None or b is None else (np.asarray(a, float)[:2], np.asarray(b, float)[:2], color)
        self.update()

    def set_arrow(self, pt=None, bearing: float | None = None) -> None:
        """An arrow from ``pt`` towards ``bearing`` (degrees from north, clockwise); None: no arrow."""
        self.arrow = None if pt is None or bearing is None else (np.asarray(pt, float)[:2], float(bearing))
        self.update()

    def set_pins(self, pins) -> None:
        self.pins = [(np.asarray(pt, float)[:2], color, text, shape) for pt, color, text, shape in pins]
        self.update()

    # --- view ---------------------------------------------------------------------------------
    def _extent(self):
        if self.focus is not None:
            return np.asarray(self.focus[0], float), np.asarray(self.focus[1], float)
        pts = ([it[1] for it in self.items] + [st[0] for st in self.stations] +
               [m[0][None] for m in self.marks + self.pins])
        if not pts:
            return None
        allp = np.vstack(pts)
        return allp.min(axis=0), allp.max(axis=0)

    def fit(self) -> None:
        """Show everything (or the focus area)."""
        ext = self._extent()
        if ext is None:
            return
        lo, hi = ext
        span = np.maximum(hi - lo, 1.0) * 1.15
        self._c = (lo + hi) / 2
        self._s = min(max(self.width() - 20, 50) / span[0], max(self.height() - 70, 50) / span[1])
        self._auto = True
        self.update()

    def fit_all(self) -> None:
        """Show everything (forget the focus area)."""
        self.focus = None
        self.fit()

    def zoom_by(self, factor: float, anchor=None) -> None:
        """Zoom by ``factor``, keeping the world point under ``anchor`` (pixels) in place."""
        before = self.to_world(anchor) if anchor is not None else None
        self._s = float(np.clip(self._s * factor, 1e-4, 1e4))
        if before is not None:
            self._c = self._c + (before - self.to_world(anchor))
        self._auto = False
        self.update()

    def to_world(self, pos) -> np.ndarray:
        return np.array([self._c[0] + (pos.x() - self.width() / 2) / self._s,
                         self._c[1] - (pos.y() - self.height() / 2) / self._s])

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._auto:
            self.fit()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if self._auto:
            self.fit()

    # --- mouse ----------------------------------------------------------------------------------
    def wheelEvent(self, e) -> None:
        steps = e.angleDelta().y() / 120.0
        if steps:
            self.zoom_by(1.25 ** steps, e.position())

    def mousePressEvent(self, e) -> None:
        if e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._press = (e.position(), self._c.copy(), e.button())
            self._moved = False

    def mouseMoveEvent(self, e) -> None:
        self._mouse = self.to_world(e.position())
        if self._press is not None:
            d = e.position() - self._press[0]
            if abs(d.x()) + abs(d.y()) > 4:
                self._moved = True
            if self._moved:
                self._c = self._press[1] - np.array([d.x(), -d.y()]) / self._s
                self._auto = False
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self._press is not None and not self._moved and self._press[2] == Qt.LeftButton:
            w = self.to_world(e.position())
            self.clicked.emit(float(w[0]), float(w[1]))
        self._press = None

    def mouseDoubleClickEvent(self, _e) -> None:
        self.fit_all()

    def leaveEvent(self, _e) -> None:
        self._mouse = None
        self.update()

    # --- drawing --------------------------------------------------------------------------------
    def paintEvent(self, _e):
        p = QPainter(self)
        self._paint(p, self.width(), self.height(), SCREEN, 1.0, self._s, self._c)

    def _paint(self, p: QPainter, W: float, H: float, pal: dict, k: float, s: float, c, top: float = 0.0,
               bottom: float = 0.0) -> None:
        """Draw the plan in a W x H area; ``k`` scales line widths, symbols and text (1 on screen)."""
        bg = QColor(pal["bg"] or theme.PANEL)
        text_c = QColor(pal["text"] or theme.TEXT)
        muted = QColor(pal["muted"] or theme.MUTED)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(QRectF(0, 0, W, H), bg)
        mid = top + (H - top - bottom) / 2         # bottom: a band kept free for the legend (prints)

        def T(q):
            return QPointF(W / 2 + (q[0] - c[0]) * s, mid - (q[1] - c[1]) * s)

        def font(size, bold=False):
            f = QFont(p.font())
            f.setPointSizeF(size * k)
            f.setBold(bold)
            p.setFont(f)

        p.setClipRect(QRectF(0, top, W, H - top - bottom))
        self._grid(p, W, H, pal, k, s, c, top, bottom, mid)
        for kind, pts, a, b, style, labels in self.items:
            if kind == "poly":
                p.setPen(QPen(QColor(b), 1.2 * k))
                p.setBrush(QColor(a))
                p.drawPolygon(QPolygonF([T(q) for q in pts]))
        for kind, pts, a, b, style, labels in self.items:
            if kind == "line":
                p.setPen(QPen(QColor(a), b * k, style, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush)
                p.drawPolyline(QPolygonF([T(q) for q in pts]))
        if self.arrow is not None:
            q, bearing = self.arrow
            o = T(q)
            b = math.radians(bearing)
            d = QPointF(math.sin(b), -math.cos(b))
            tip = o + d * (48 * k)
            side = QPointF(-d.y(), d.x())
            p.setPen(QPen(QColor("#3d9bff"), 3 * k, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(o, tip)
            p.setBrush(QColor("#3d9bff"))
            p.drawPolygon(QPolygonF([tip + d * (6 * k), tip - d * (8 * k) + side * (6 * k),
                                     tip - d * (8 * k) - side * (6 * k)]))
        if self.guide is not None:
            a, b, color = self.guide
            p.setPen(QPen(QColor(color), 2 * k, Qt.DashLine))
            p.drawLine(T(a), T(b))
        for pts, _names, color in self.stations:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color))
            for q in pts:
                p.drawEllipse(T(q), 2.6 * k, 2.6 * k)
        taken = []                                              # label boxes already drawn (no overlaps)

        def label(pt, text, color, size=8.0, bold=False, dx=5.0, dy=-5.0):
            font(size, bold)
            fm = p.fontMetrics()
            r = QRectF(pt.x() + dx * k, pt.y() + dy * k - fm.ascent(), fm.horizontalAdvance(text), fm.height())
            if any(r.intersects(o) for o in taken):
                return
            taken.append(r.adjusted(-2 * k, -1 * k, 2 * k, 1 * k))
            p.setPen(QColor(color))
            p.drawText(QPointF(r.left(), pt.y() + dy * k), text)

        # marked points first, so their names always show; then route labels, then station names
        for q, color, text, shape in self.pins + self.marks:
            pt = T(q)
            p.setPen(QPen(QColor("#111"), 1 * k))
            p.setBrush(QColor(color))
            if shape == "ring":
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(color), 3 * k))
                p.drawEllipse(pt, 11 * k, 11 * k)
            elif shape == "triangle":
                p.drawPolygon(QPolygonF([pt + QPointF(0, -8 * k), pt + QPointF(7 * k, 5 * k),
                                         pt + QPointF(-7 * k, 5 * k)]))
            elif shape == "target":
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(color), 2.5 * k))
                p.drawEllipse(pt, 9 * k, 9 * k)
                p.drawLine(pt + QPointF(-14 * k, 0), pt + QPointF(14 * k, 0))
                p.drawLine(pt + QPointF(0, -14 * k), pt + QPointF(0, 14 * k))
            else:
                p.drawEllipse(pt, 6 * k, 6 * k)
            if text:
                label(pt, text, color, 9.0, True, 12.0, 4.0)
        for kind, pts, a, b, style, labels in self.items:
            if kind == "line" and labels:
                for q, text in zip(pts, labels):
                    if text:
                        label(T(q), text, text_c)
        for pts, names, _color in self.stations:
            for q, name in zip(pts, names):
                if name:
                    label(T(q), name, muted, 7.5)
        p.setClipping(False)
        # bottom strip: scale bar, legend, cursor coordinates
        strip = 34 * k
        sc = QColor(bg)
        sc.setAlpha(225)
        p.fillRect(QRectF(0, H - strip, W, strip), sc)
        length = nice((W * 0.18) / s)
        x0, y0 = 14 * k, H - strip / 2
        p.setPen(QPen(text_c, 2 * k))
        p.drawLine(QPointF(x0, y0), QPointF(x0 + length * s, y0))
        for x in (x0, x0 + length * s):
            p.drawLine(QPointF(x, y0 - 4 * k), QPointF(x, y0 + 4 * k))
        font(8)
        p.drawText(QPointF(x0 + length * s + 6 * k, y0 + 4 * k), f"{length:g} m")
        x = x0 + length * s + 60 * k
        for color, text, kind in self.legend:
            if kind == "fill":
                p.fillRect(QRectF(x, y0 - 6 * k, 14 * k, 12 * k), QColor(color))
            elif kind == "dot":
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(color))
                p.drawEllipse(QPointF(x + 7 * k, y0), 4 * k, 4 * k)
            else:
                p.setPen(QPen(QColor(color), 3 * k, Qt.DashLine if kind == "dash" else Qt.SolidLine))
                p.drawLine(QPointF(x, y0), QPointF(x + 18 * k, y0))
            p.setPen(muted)
            p.drawText(QPointF(x + 22 * k, y0 + 4 * k), text)
            x += 34 * k + p.fontMetrics().horizontalAdvance(text)
        if self._mouse is not None and k == 1.0:
            p.setPen(muted)
            p.drawText(QRectF(W - 260, H - strip, 250, strip), int(Qt.AlignRight | Qt.AlignVCenter),
                       f"X {self._mouse[0]:,.1f}   Y {self._mouse[1]:,.1f}")
        nx, ny = W - 20 * k, top + 30 * k
        p.setPen(QPen(text_c, 1 * k))
        p.setBrush(text_c)
        p.drawPolygon(QPolygonF([QPointF(nx, ny - 14 * k), QPointF(nx + 6 * k, ny + 2 * k),
                                 QPointF(nx - 6 * k, ny + 2 * k)]))
        font(8, True)
        p.drawText(QPointF(nx - 4 * k, ny + 16 * k), "N")

    def _grid(self, p: QPainter, W, H, pal, k, s, c, top, bottom, mid) -> None:
        """Map grid: easting / northing lines at a round spacing (about every 110 px), labelled along the
        bottom (E) and the left side (N) of the map area."""
        step = nice(110 * k / s)
        strip = 34 * k                                   # the scale bar / legend strip at the bottom
        y_lo = H - max(bottom, strip)                    # bottom of the map area (above the strip)
        x0w, x1w = c[0] - (W / 2) / s, c[0] + (W / 2) / s
        y0w, y1w = c[1] - (y_lo - mid) / s, c[1] + (mid - top) / s
        p.setPen(QPen(QColor(pal["grid"]), max(1.0, 0.8 * k)))
        xs = [i * step for i in range(math.ceil(x0w / step), math.floor(x1w / step) + 1)]
        ys = [j * step for j in range(math.ceil(y0w / step), math.floor(y1w / step) + 1)]
        for x in xs:
            px = W / 2 + (x - c[0]) * s
            p.drawLine(QPointF(px, top), QPointF(px, H - bottom))
        for y in ys:
            py = mid - (y - c[1]) * s
            p.drawLine(QPointF(0, py), QPointF(W, py))
        f = QFont(p.font())
        f.setPointSizeF(7 * k)
        p.setFont(f)
        p.setPen(QColor(pal["grid_text"]))
        fm = p.fontMetrics()
        dec = 0 if step >= 1 else 2
        for x in xs:
            px = W / 2 + (x - c[0]) * s
            t = f"E {x:,.{dec}f}"
            if px < 70 * k or px + fm.horizontalAdvance(t) > W - 4 * k:
                continue                                 # keep the corners free for the N labels / arrow
            p.drawText(QPointF(px + 3 * k, y_lo - 4 * k), t)
        for y in ys:
            py = mid - (y - c[1]) * s
            if py < top + 48 * k or py > y_lo - 18 * k:
                continue                                 # below the zoom buttons, above the E labels
            p.drawText(QPointF(4 * k, py - 3 * k), f"N {y:,.{dec}f}")

    def save_png(self, path, title: str = "", long_side: int = 3300) -> tuple[int, int]:
        """The view as shown, on white at 300 dpi (long side ``long_side`` px), a title band on top."""
        from PySide6.QtGui import QImage

        w0, h0 = max(self.width(), 100), max(self.height(), 100)
        k = long_side / max(w0, h0)
        head = 30.0 * k if title else 0.0
        band = 34.0 * k                             # the legend band below the map (it covers nothing)
        W, H = int(round(w0 * k)), int(round(h0 * k + head + band))
        img = QImage(W, H, QImage.Format_RGB32)
        img.fill(QColor("#ffffff"))
        p = QPainter(img)
        self._paint(p, W, H, PAPER, k, self._s * k, self._c, head, band)
        if title:
            f = QFont(p.font())
            f.setPointSizeF(10 * k)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("#1b1d21"))
            p.drawText(QRectF(14 * k, 0, W - 28 * k, head), int(Qt.AlignLeft | Qt.AlignVCenter), title)
            p.setPen(QPen(QColor("#c8cdd4"), 1 * k))
            p.drawLine(QPointF(0, head), QPointF(W, head))
        p.end()
        dpm = int(round(300 / 0.0254))
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        if not img.save(str(path)):
            raise OSError(tr("Could not write {path}", path=path))
        return W, H


class ProfileChart(_Chart):
    """Long section of a route: distance along it (plan) against elevation, each leg coloured by its
    gradient against the limit, with the gradient written on it."""

    def __init__(self, parent=None):
        super().__init__(parent, 200)
        self.legs, self.limit = [], 1 / 7

    def set_route(self, legs, limit: float) -> None:
        self.legs, self.limit = legs, limit
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._bg(p)
        if not self.legs:
            return
        from ..ai.route_planner import ratio_text

        ch = [0.0]
        for lg in self.legs:
            ch.append(ch[-1] + lg.plan)
        zs = [self.legs[0].a[2]] + [lg.b[2] for lg in self.legs]
        R = self.plot_rect()
        zlo, zhi = min(zs), max(zs)
        pad = max((zhi - zlo) * 0.15, 5.0)
        zlo, zhi = zlo - pad, zhi + pad
        L = max(ch[-1], 1.0)

        def X(u):
            return R.left() + u / L * R.width()

        def Y(z):
            return R.bottom() - (z - zlo) / (zhi - zlo) * R.height()

        step = nice((zhi - zlo) / 4)
        z = math.ceil(zlo / step) * step
        while z <= zhi:
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawLine(QPointF(R.left(), Y(z)), QPointF(R.right(), Y(z)))
            self._text(p, R.left() - 6, Y(z), f"{z:,.0f} m", align=Qt.AlignRight | Qt.AlignVCenter, w=56)
            z += step
        xs = nice(L / 6)
        u = 0.0
        while u <= L + 1e-9:
            self._text(p, X(u), R.bottom() + 12, f"{u:,.0f} m", align=Qt.AlignCenter, w=70)
            u += xs
        for i, lg in enumerate(self.legs):
            g = abs(lg.gradient)
            color = GREEN if g <= self.limit * 0.9 else (AMBER if g <= self.limit * 1.001 else RED)
            a, b = QPointF(X(ch[i]), Y(zs[i])), QPointF(X(ch[i + 1]), Y(zs[i + 1]))
            p.setPen(QPen(QColor(color), 4, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(a, b)
            mid = (a + b) / 2
            if b.x() - a.x() > 60:
                self._text(p, mid.x(), mid.y() - 12, ratio_text(lg.gradient), color, Qt.AlignCenter, 8, True, 140)
        for i, label in ((0, "A"), (len(zs) - 1, "B")):
            pt = QPointF(X(ch[i]), Y(zs[i]))
            p.setPen(QPen(QColor("#111"), 1))
            p.setBrush(QColor(BLUE if i == 0 else RED))
            p.drawEllipse(pt, 6, 6)
            self._text(p, pt.x() + 8, pt.y() - 10, label, theme.TEXT, size=9, bold=True, w=30)
        self._text(p, R.right(), R.bottom() + 26, tr("distance along the route (plan)"), align=Qt.AlignRight |
                   Qt.AlignVCenter, w=300)
        y = 10
        x = R.left() + 4
        for color, label in ((GREEN, tr("gentle")), (AMBER, tr("near the limit")), (RED, tr("too steep"))):
            p.fillRect(QRectF(x, y - 4, 14, 8), QColor(color))
            self._text(p, x + 18, y, label, w=150)
            x += 40 + p.fontMetrics().horizontalAdvance(label)


def hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of plan points (monotone chain)."""
    P = np.unique(np.round(np.asarray(pts, float)[:, :2], 3), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1])


def colored(text: str, color: str) -> str:
    return f"<span style='color:{color}; font-weight:600'>{text}</span>"


