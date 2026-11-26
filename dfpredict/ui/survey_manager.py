"""Pit Manager: main pits and their branches, and each pit's traverse entered setup by setup,
with a live plan map of the centre line and the computed coordinates; and the designed pits
(drives designed with the route designer and saved to the database): their name, profile, note
and centre line points can be changed here, or the designed pit deleted.

Every change is computed at once (nothing is written until Save pit). Angles are typed as
"120 30 00" or "120.5" and shown as 120°30′00″.
"""

from __future__ import annotations

import copy
import dataclasses
import math

from pathlib import Path

from PySide6.QtCore import QDate, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..i18n import N_, tr, trc, trn
from ..io.survey_store import (STATION_LEVELS, SetupRow, SurveyStore, compute, format_dms, next_station_name,
                               parse_angle)
from ..scene.survey import STATUS_COLORS, pit_paths
from . import theme
from .icons import icon

DESIGN_ROLE = Qt.UserRole + 1           # tree items of designed pits: the design id
GROUP_ROLE = Qt.UserRole + 2            # the "Designed Pits" group item
ANGLE_HINT = N_("Type 120 30 00 (degrees minutes seconds) or 120.5; shown as 120°30′00″.")
BEARING_TYPE_TEXT = {"magnetic": N_("Magnetic (compass)"), "true": N_("True (gyro)"), "grid": N_("Grid")}


def bearing_type_text(kind: str) -> str:
    """The shown name of a bearing kind ("Grid" here is a grid bearing, not the map grid)."""
    return trc("bearing", "Grid") if kind == "grid" else tr(BEARING_TYPE_TEXT.get(kind, kind))
STATUS_WORD = {"known": N_("known"), "entered": N_("given"), "closed": N_("closed"), "open": N_("not checked"),
               "failed": N_("closure failed"), "unresolved": N_("no coordinates")}
# the plan map on paper (the PNG export): dark lines and text on white
PRINT_MARKER = 0.45                  # station symbols on the PNG: under half the screen size
PRINT_LINE = "#5a6068"                # the PNG: every pit's centre line (and known stations) in grey
PRINT_COLORS = {"BG": "#ffffff", "TEXT": "#1b1d21", "MUTED": "#7a828c", "BORDER": "#e1e4e8", "ACCENT": "#1764c4"}
EXPORT_FORMATS = {"shp": (N_("Centre lines - ESRI shapefile (.shp)"), N_("Shapefile (*.shp)")),
                  "dtm": (N_("Drive solids - Surpac DTM (.dtm + .str)"), N_("Surpac DTM (*.dtm)"))}


def _angle_hint() -> str:
    return tr(ANGLE_HINT)


def _method(row) -> str:
    """The row's method as chosen in the form (a compass setup whose bearing is not typed yet
    is still a compass setup), else as stored."""
    return getattr(row, "ui_method", None) or row.method


def _f(text):
    """A number from a field ("" -> None); ValueError for anything else."""
    t = str(text or "").strip().replace(",", "")
    return float(t) if t else None


def _fnum(text):
    """A number from a field; None for empty or not a number."""
    try:
        return _f(text)
    except ValueError:
        return None


def _wide(size, edit, sample):
    """``size`` widened to show ``sample`` in the edit's (styled) font with its frame and padding."""
    size.setWidth(max(size.width(), edit.fontMetrics().horizontalAdvance(sample) + 22))
    return size


class _AngleEdit(QLineEdit):
    """An angle field: type 120 30 00 or 120.5; shown as 120°30′00″; red when it cannot be read."""

    changed = Signal()

    def __init__(self, placeholder=None):
        super().__init__()
        self.setPlaceholderText(tr("e.g. 120 30 00") if placeholder is None else placeholder)
        self.editingFinished.connect(self._tidy)
        self.textEdited.connect(lambda *_: self.changed.emit())
        self._ok = True
        self._hint = tr("Type an angle as 120 30 00 (degrees minutes seconds) or 120.5")

    def setToolTip(self, text):                 # the hover hint; an error is shown above it
        self._hint = text
        super().setToolTip(text)

    def minimumSizeHint(self):                  # wide enough for any angle: never clipped
        return _wide(super().minimumSizeHint(), self, "-359°59′59.9″")

    def sizeHint(self):
        return _wide(super().sizeHint(), self, "-359°59′59.9″")

    def value(self):
        try:
            v = parse_angle(self.text())
            self._mark(True)
            return v
        except ValueError as e:
            self._mark(False, str(e))
            return None

    def set_value(self, v):
        self.setText(format_dms(v) if v is not None else "")
        self._mark(True)

    def _tidy(self):
        try:
            v = parse_angle(self.text())
        except ValueError as e:
            self._mark(False, str(e))
            return
        self.setText(format_dms(v) if v is not None else "")
        self._mark(True)
        self.changed.emit()

    def _mark(self, ok, why=""):
        self._ok = ok
        self.setStyleSheet("" if ok else f"border: 1px solid {theme.DANGER};")
        QLineEdit.setToolTip(self, self._hint if ok else f"⚠ {why}\n\n{self._hint}")


class _NumEdit(QLineEdit):
    changed = Signal()

    def __init__(self, placeholder=""):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.textEdited.connect(lambda *_: self.changed.emit())
        self._hint = ""

    def setToolTip(self, text):
        self._hint = text
        super().setToolTip(text)

    def minimumSizeHint(self):
        return _wide(super().minimumSizeHint(), self, "-10,000.000")

    def sizeHint(self):
        return _wide(super().sizeHint(), self, "-10,000.000")

    def value(self):
        try:
            v = _f(self.text())
            self.setStyleSheet("")
            QLineEdit.setToolTip(self, self._hint)
            return v
        except ValueError:
            self.setStyleSheet(f"border: 1px solid {theme.DANGER};")
            QLineEdit.setToolTip(self, "⚠ " + tr("Type a number") + f"\n\n{self._hint}")
            return None

    def set_value(self, v, decimals=3):
        self.setText("" if v is None else f"{v:.{decimals}f}".rstrip("0").rstrip(".") if decimals else str(v))


# --- the plan map ----------------------------------------------------------------------------------
class PlanMap(QWidget):
    """Plan view of the pits' centre lines: the edited pit in colour (its legs with arrows,
    stations and names), the other pits faded; north arrow and scale bar. Wheel zooms,
    dragging pans, double-click fits."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 280)
        self.coords: dict = {}
        self.status: dict = {}
        self.lines: list = []          # (pit id, [names])
        self.current = None
        self.selected = None           # (from, to) of the selected setup's leg
        self.known: set = set()
        self._view = None              # (cx, cy, metres per pixel)
        self._drag = None
        self.palette_override: dict = {}   # theme colour name -> colour (the PNG export: print colours)
        self.marker = 1.0                  # station symbol size factor (smaller on the PNG export)
        self.uniform = False               # the PNG export: every pit alike in grey (no highlight,
                                           # no arrows), every station marked and named
        self.designs: list = []            # designed pits: (id, [(x, y, z)]) - dashed green
        self.current_design = None         # the designed pit being edited (its points numbered)

    def _col(self, name) -> QColor:
        return QColor(self.palette_override.get(name) or getattr(theme, name))

    def save_png(self, path, title: str = "", dpi: int = 300, long_side: int = 3300) -> tuple:
        """The map as shown (same area, same zoom), on white, as a sharp PNG: drawn off-screen
        at the on-screen layout and scaled up so the long side is about ``long_side`` pixels
        (11 inches at 300 dpi). Returns the image size (w, h)."""
        if self._view is None:
            raise ValueError(tr("The plan map is empty - nothing to export."))
        head = 30 if title else 0
        off = PlanMap()
        off.coords, off.status, off.lines, off.current = self.coords, self.status, self.lines, self.current
        off.designs, off.current_design = self.designs, self.current_design
        off.known, off.selected = self.known, None                 # no selected-leg highlight on paper
        off.palette_override = PRINT_COLORS
        off.marker = PRINT_MARKER
        off.uniform = True
        off.setFont(self.font())
        off.resize(self.width(), self.height())
        off._view = self._view
        k = long_side / max(self.width(), self.height() + head)
        img = QImage(round(self.width() * k), round((self.height() + head) * k), QImage.Format_ARGB32)
        img.fill(QColor("#ffffff"))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.scale(k, k)
        if title:
            f = QFont(self.font())
            f.setPointSizeF(10.5)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(PRINT_COLORS["TEXT"]))
            p.drawText(QRectF(10, 0, self.width() - 20, head), Qt.AlignVCenter | Qt.AlignLeft, title)
        off.render(p, QPoint(0, head))
        p.end()
        dpm = round(dpi / 0.0254)                                  # set after drawing: fonts keep their size
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        if not img.save(str(path), "PNG"):
            raise OSError(tr("The image could not be written:\n{path}", path=path))
        return img.width(), img.height()

    def set_data(self, coords, status, lines, current, selected, known, refit=False, designs=None,
                 current_design=None):
        self.coords, self.status, self.lines, self.current = coords, status, lines, current
        self.selected, self.known = selected, known
        self.designs, self.current_design = list(designs or []), current_design
        n_own = sum(1 for pid, names in lines if pid == current for n in names if n in coords)
        grew, self._n_own = n_own > getattr(self, "_n_own", 0), n_own
        if refit or self._view is None or (grew and not self._shows_current()):   # a zoom the user chose is kept
            self.fit()
        self.update()

    def _shows_current(self) -> bool:
        """Every station of the edited pit inside the view (else it is fitted again, e.g. when
        a new setup runs off the edge)."""
        if self._view is None:
            return False
        for pid, names in self.lines:
            if pid != self.current:
                continue
            for n in names:
                if n in self.coords:
                    q = self._to_px(*self.coords[n][:2])
                    if not (8 <= q.x() <= self.width() - 8 and 8 <= q.y() <= self.height() - 8):
                        return False
        return True

    def fit(self):
        own = [self.coords[n] for pid, names in self.lines if pid == self.current for n in names if n in self.coords]
        own += [q for did, pts in self.designs if did == self.current_design for q in pts]
        pts = own if len(own) >= 2 else [c for c in self.coords.values()] + [q for _d, pts in self.designs
                                                                              for q in pts]
        if not pts:
            self._view = None
            return
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys), 20.0) * 1.3
        self._view = (cx, cy, span / max(min(self.width(), self.height()) - 40, 50))
        self.update()

    def _to_px(self, x, y):
        cx, cy, mpp = self._view
        return QPointF(self.width() / 2 + (x - cx) / mpp, self.height() / 2 - (y - cy) / mpp)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self._col("BG"))
        fg, muted = self._col("TEXT"), self._col("MUTED")
        if self._view is None:
            p.setPen(muted)
            p.drawText(self.rect(), Qt.AlignCenter, tr("The pit's centre line appears here\nas the survey is entered"))
            return
        cx, cy, mpp = self._view
        # grid
        step = _nice(mpp * 90)
        p.setPen(QPen(self._col("BORDER"), 1))
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        x0 = math.floor((cx - self.width() / 2 * mpp) / step) * step
        y0 = math.floor((cy - self.height() / 2 * mpp) / step) * step
        x = x0
        while x < cx + self.width() / 2 * mpp:
            q = self._to_px(x, cy)
            p.setPen(QPen(self._col("BORDER"), 1))
            p.drawLine(QPointF(q.x(), 0), QPointF(q.x(), self.height()))
            p.setPen(muted)
            p.drawText(QPointF(q.x() + 2, self.height() - 4), f"{x:,.0f}")
            x += step
        y = y0
        while y < cy + self.height() / 2 * mpp:
            q = self._to_px(cx, y)
            p.setPen(QPen(self._col("BORDER"), 1))
            p.drawLine(QPointF(0, q.y()), QPointF(self.width(), q.y()))
            p.setPen(muted)
            p.drawText(QPointF(2, q.y() - 2), f"{y:,.0f}")
            y += step
        # other pits, then the edited pit
        for pid, names in sorted(self.lines, key=lambda t: t[0] == self.current):
            pts = [self.coords[n] for n in names if n in self.coords]
            if len(pts) < 2:
                continue
            mine = pid == self.current and not self.uniform
            if self.uniform:
                p.setPen(QPen(QColor(PRINT_LINE), 1.4))
            else:
                p.setPen(QPen(self._col("ACCENT") if mine else self._col("MUTED"), 2.5 if mine else 1.2))
            for a, b in zip(names, names[1:]):
                if a in self.coords and b in self.coords:
                    pa, pb = self._to_px(*self.coords[a][:2]), self._to_px(*self.coords[b][:2])
                    sel = mine and self.selected == (a, b)
                    if sel:
                        p.setPen(QPen(QColor("#f0a24a"), 4.0))
                    p.drawLine(pa, pb)
                    if mine:
                        _arrow(p, pa, pb, QColor("#f0a24a") if sel else self._col("ACCENT"))
                    if sel:
                        p.setPen(QPen(self._col("ACCENT"), 2.5))
        # designed pits: dashed; the one being edited bold, its points numbered
        for did, pts in sorted(self.designs, key=lambda t: t[0] == self.current_design):
            if len(pts) < 2:
                continue
            mine = did == self.current_design and not self.uniform
            col = QColor(PRINT_LINE) if self.uniform else QColor("#4ce67a" if mine else "#3f8f58")
            p.setPen(QPen(col, 3.0 if mine else 1.4, Qt.DashLine))
            q = [self._to_px(x[0], x[1]) for x in pts]
            for a, b in zip(q, q[1:]):
                p.drawLine(a, b)
            if mine:
                p.setPen(QPen(self._col("TEXT"), 1.0))
                p.setBrush(QColor("#4ce67a"))
                for k, c in enumerate(q, start=1):
                    p.drawEllipse(c, 4.0, 4.0)
                    p.drawText(c + QPointF(6, -6), str(k))
                p.setBrush(Qt.NoBrush)
        # stations of the edited pit (and known ones); on the PNG every station
        f.setPointSizeF(7.5 if self.uniform else 8.5)
        p.setFont(f)
        own = {n for pid, names in self.lines if pid == self.current or self.uniform for n in names}
        fm = p.fontMetrics()
        s = self.marker
        r = 7 * s                                   # symbol radius (the triangle's reach)
        # names that matter most first (the selected leg, known stations), so they keep their label
        first = set(self.selected or ()) | self.known
        shown = [(n, self._to_px(c[0], c[1])) for n, c in sorted(self.coords.items(), key=lambda kv: kv[0] not in first)
                 if n in own or n in self.known]
        for n, q in shown:
            col = (QColor(PRINT_LINE if n in self.known else "#ffffff") if self.uniform else
                   QColor.fromRgbF(*STATUS_COLORS.get(self.status.get(n, ""), (0.6, 0.6, 0.6))))
            p.setPen(QPen(fg, min(1.0, s + 0.2)))
            p.setBrush(col)
            if n in self.known:
                p.drawPolygon(QPolygonF([q + QPointF(0, -7 * s), q + QPointF(6 * s, 5 * s), q + QPointF(-6 * s, 5 * s)]))
            else:
                p.drawEllipse(q, 4.5 * s, 4.5 * s)
        p.setBrush(Qt.NoBrush)
        # the names: beside the symbol (right, left, above, below - the first place free of other
        # names and symbols); where stations crowd, further out with a thin leader line to the
        # station; a name with no free place at all is left out
        symbols = [QRectF(q.x() - r, q.y() - r, 2 * r, 2 * r) for _n, q in shown]
        placed: list[QRectF] = []
        leaders: list = []
        area = QRectF(self.rect())
        a, gap = fm.ascent(), 3 + 4 * s
        dirs = [(math.cos(math.radians(d)), -math.sin(math.radians(d))) for d in (30, -30, 0, 150, 210, 180, 90, 270,
                                                                                  60, 120, 240, 300)]
        for i, (n, q) in enumerate(shown):
            if not area.contains(q):                     # off the map: no name for an unseen station
                continue
            w, h = fm.horizontalAdvance(n), fm.height()
            near = [(gap, -gap + 2), (gap, gap - 2 + a), (gap + 1, a / 2 - 1),
                    (-gap - w, -gap + 2), (-gap - w, gap - 2 + a), (-gap - 1 - w, a / 2 - 1),
                    (-w / 2, -r - 2), (-w / 2, r + 2 + a)]
            far = []
            for R in (3.2 * gap, 5 * gap, 7.5 * gap):
                for ux, uy in dirs:                  # the box on that side of the point at distance R
                    bx = ux * R - (w if ux < -0.3 else w / 2 if ux <= 0.3 else 0)
                    by = uy * R + (a if uy > 0.3 else a / 2 if uy >= -0.3 else 0)
                    far.append((bx, by, (ux, uy)))
            for ox, oy, lead in [(x, y, None) for x, y in near] + far:
                box = QRectF(q.x() + ox, q.y() + oy - a, w, h)
                if not area.contains(box) or any(box.intersects(b) for b in placed) or \
                        any(j != i and box.intersects(b) for j, b in enumerate(symbols)):
                    continue
                placed.append(box.adjusted(-2, -1, 2, 1))
                if lead is not None:                 # leader: from the symbol to the box's nearest point
                    end = QPointF(min(max(q.x(), box.left()), box.right()), min(max(q.y(), box.top()), box.bottom()))
                    leaders.append((q + QPointF(lead[0] * r, lead[1] * r), end))
                p.setPen(fg)
                p.drawText(QPointF(q.x() + ox, q.y() + oy), n)
                break
        p.setPen(QPen(self._col("MUTED"), 0.8))
        for p0, p1 in leaders:
            p.drawLine(p0, p1)
        # north arrow and scale bar
        nx, ny = self.width() - 26, 18
        p.setPen(QPen(fg, 1.5))
        p.drawLine(QPointF(nx, ny + 30), QPointF(nx, ny + 6))
        path = QPainterPath()
        path.addPolygon(QPolygonF([QPointF(nx, ny), QPointF(nx - 5, ny + 10), QPointF(nx + 5, ny + 10)]))
        p.fillPath(path, fg)
        p.drawText(QPointF(nx - 4, ny + 44), "N")
        bar = _nice(mpp * 100)
        px = bar / mpp
        p.drawLine(QPointF(14, self.height() - 22), QPointF(14 + px, self.height() - 22))
        for xx in (14, 14 + px):
            p.drawLine(QPointF(xx, self.height() - 26), QPointF(xx, self.height() - 18))
        p.drawText(QPointF(14, self.height() - 28), f"{bar:g} m")

    def wheelEvent(self, e):
        if self._view is None:
            return
        cx, cy, mpp = self._view
        k = 0.8 if e.angleDelta().y() > 0 else 1.25
        self._view = (cx, cy, mpp * k)
        self.update()

    def mousePressEvent(self, e):
        self._drag = (e.position(), self._view)

    def mouseMoveEvent(self, e):
        if self._drag and self._view:
            start, (cx, cy, mpp) = self._drag
            d = e.position() - start
            self._view = (cx - d.x() * mpp, cy + d.y() * mpp, mpp)
            self.update()

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def mouseDoubleClickEvent(self, _e):
        self.fit()


def _nice(v):
    """A round length near ``v`` (1, 2, 5 x 10^n)."""
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    return next(m * e for m in (1, 2, 5, 10) if m * e >= v)


def _arrow(p, a, b, color):
    d = b - a
    n = math.hypot(d.x(), d.y())
    if n < 18:
        return
    u = QPointF(d.x() / n, d.y() / n)
    m = a + d * 0.55
    w = QPointF(-u.y(), u.x())
    path = QPainterPath()
    path.addPolygon(QPolygonF([m + u * 7, m - u * 5 + w * 4.5, m - u * 5 - w * 4.5]))
    p.fillPath(path, color)


# --- dialogs for new pits ------------------------------------------------------------------------------
class _NewPitDialog(QDialog):
    def __init__(self, parent, title, branch_of=None, stations=()):
        super().__init__(parent)
        self.setWindowTitle(title)
        f = QFormLayout(self)
        self.name = QLineEdit()
        self.name.setPlaceholderText(tr("e.g. North decline") if branch_of is None else tr("e.g. Level 2 access"))
        f.addRow(tr("Name"), self.name)
        self.name.setToolTip(tr("Name of the pit as you call it underground; a code (P1, P1A ...) is given "
                                "automatically"))
        self.branch = branch_of is not None
        if self.branch:
            f.addRow(QLabel(tr("Branch of <b>{name}</b>", name=branch_of)))
            self.start = QComboBox()
            self.start.addItems(list(stations))
            f.addRow(tr("Starts at station"), self.start)
            self.start.setToolTip(tr("The station of the parent pit where this branch leaves it"))
            hint = tr("The branch starts at a station of its parent with coordinates (computed).")
        else:
            self.start = QLineEdit()
            self.start.setPlaceholderText(tr("e.g. E1"))
            f.addRow(tr("Entrance station"), self.start)
            self.x, self.y, self.z = _NumEdit(tr("Easting")), _NumEdit(tr("Northing")), _NumEdit(tr("Elevation"))
            f.addRow(tr("X (East)"), self.x)
            f.addRow(tr("Y (North)"), self.y)
            f.addRow(tr("Z (Elevation)"), self.z)
            self.start.setToolTip(tr("Name of the entrance station (its exact coordinates are known), e.g. E1"))
            self.x.setToolTip(tr("Easting of the entrance station (m)"))
            self.y.setToolTip(tr("Northing of the entrance station (m)"))
            self.z.setToolTip(tr("Elevation of the entrance station mark (m)"))
            hint = tr("The entrance's exact coordinates - the survey starts from them.")
        h = QLabel(hint)
        h.setWordWrap(True)
        h.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow(h)
        b = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b.accepted.connect(self.accept)
        b.rejected.connect(self.reject)
        f.addRow(b)

    def values(self):
        if self.branch:
            return self.name.text().strip(), self.start.currentText(), None
        return self.name.text().strip(), self.start.text().strip(), (self.x.value(), self.y.value(), self.z.value())


class _SettingsDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle(tr("Survey settings"))
        f = QFormLayout(self)
        self.ratio = QDoubleSpinBox()
        self.ratio.setRange(500, 1e7)
        self.ratio.setDecimals(0)
        self.ratio.setPrefix("1 : ")
        self.ratio.setValue(settings.get("closure_ratio", 10000))
        self.decl = _AngleEdit(tr("e.g. -7 30 00 (west negative)"))
        self.decl.set_value(settings.get("declination", 0.0))
        self.conv = _AngleEdit(tr("e.g. 1 15 00"))
        self.conv.set_value(settings.get("convergence", 0.0))
        f.addRow(tr("Closure tolerance"), self.ratio)
        f.addRow(tr("Magnetic declination"), self.decl)
        f.addRow(tr("Grid convergence"), self.conv)
        self.ratio.setToolTip(tr("Least closure ratio a closed traverse must reach, e.g. 1 : 10,000 = 1 m "
                                 "misclosure per 10 km of traverse"))
        self.decl.setToolTip(tr("Magnetic declination: angle from true north to magnetic north, east +, west −.\n"
                                "true bearing = magnetic bearing + declination.") + " " + _angle_hint())
        self.conv.setToolTip(tr("Grid convergence: angle from true north to grid north.\n"
                                "grid bearing = true bearing − convergence.") + " " + _angle_hint())
        h = QLabel(tr("Compass bearings: true = magnetic + declination (east +, west −). "
                      "Grid = true − convergence. The tolerance is the least closure ratio a traverse must reach."))
        h.setWordWrap(True)
        h.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow(h)
        b = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b.accepted.connect(self.accept)
        b.rejected.connect(self.reject)
        f.addRow(b)

    def values(self):
        return {"closure_ratio": self.ratio.value(), "declination": self.decl.value() or 0.0,
                "convergence": self.conv.value() or 0.0}


# --- the manager ---------------------------------------------------------------------------------------
class _ExportPitsDialog(QDialog):
    """Which pits (all, or the selected pit and its branches) and which format."""

    def __init__(self, parent, n_all: int, pit=None, n_family: int = 0, fmt: str = "shp"):
        super().__init__(parent)
        self.setWindowTitle(tr("Export pits"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        v = QVBoxLayout(self)
        g1 = QGroupBox(tr("Pits"))
        l1 = QVBoxLayout(g1)
        self.r_all = QRadioButton(tr("All pits  ({n})", n=n_all))
        self.r_one = QRadioButton(tr("{name} ({code}) and its branches  ({n})", name=pit.name, code=pit.code,
                                     n=n_family) if pit else
                                  tr("The selected pit and its branches  (select a pit first)"))
        self.r_one.setEnabled(pit is not None)
        (self.r_one if pit is not None and pit.parent_id is not None else self.r_all).setChecked(True)
        l1.addWidget(self.r_all)
        l1.addWidget(self.r_one)
        v.addWidget(g1)
        g2 = QGroupBox(tr("Export"))
        l2 = QVBoxLayout(g2)
        self.fmt = {}
        for key, (text, _flt) in EXPORT_FORMATS.items():
            r = QRadioButton(tr(text))
            r.setChecked(key == fmt)
            self.fmt[key] = r
            l2.addWidget(r)
        note = QLabel(tr("Shapefile: one 3D line per pit, with its name, code, parent and profile.\n"
                         "Surpac DTM: each pit is an object, its drive a closed solid (profile as in the view)."))
        note.setStyleSheet(f"color: {theme.MUTED}; font-size: 9pt;")
        l2.addWidget(note)
        v.addWidget(g2)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("Export ..."))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def values(self):
        """(selected pit and branches?, format key)."""
        return self.r_one.isChecked(), next(k for k, r in self.fmt.items() if r.isChecked())


class SurveyManager(QDialog):
    changed = Signal()            # the database was written: read the survey layer again
    geology_requested = Signal(object)   # pit id: open Pit Geology at that pit

    SUMMARY = ["#", N_("At"), N_("Direction from"), N_("Foresight"), N_("Bearing (grid)"), N_("HD (m)"), "ΔZ (m)",
               N_("Check")]             # headers: tr() where shown

    def __init__(self, store: SurveyStore, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Pit Manager"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)        # can be maximised
        self.resize(1320, 800)
        self.store = store
        self.pits = []
        self.stations = []
        self.setups = {}
        self.settings = {}
        self.pid = None
        self.designs = []                   # designed pits (survey_store.DesignPit)
        self.did = None                     # the designed pit shown (then self.pid is None)
        self.rows: list[SetupRow] = []
        self.cur = -1
        self.dirty = False
        self._filling = False
        self.net = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._recompute)

        # left: pits
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._pit_clicked)
        b_main = QPushButton(icon("plus"), tr("New main pit"))
        b_main.clicked.connect(self.new_main_pit)
        b_branch = QPushButton(icon("source-branch"), tr("New branch"))
        b_branch.setToolTip(tr("A branch of the selected pit, starting at one of its stations"))
        b_branch.clicked.connect(self.new_branch)
        b_del = QPushButton(icon("delete-outline"), tr("Delete"))
        b_del.setToolTip(tr("Delete the selected pit, its branches, setups and stations - or the selected "
                            "designed pit"))
        b_del.clicked.connect(self.delete_pit)
        b_set = QPushButton(icon("cog-outline"), tr("Survey settings ..."))
        b_set.setToolTip(tr("Closure tolerance, magnetic declination and grid convergence"))
        b_main.setToolTip(tr("A new main pit, starting at an entrance station with known X, Y, Z"))
        b_set.clicked.connect(self.edit_settings)
        b_exp = QPushButton(icon("export"), tr("Export pits ..."))
        b_exp.setToolTip(tr("Save all pits, or the selected pit with its branches:\n"
                            "centre lines as a shapefile (.shp), or the drives as Surpac solids (.dtm + .str)"))
        b_exp.clicked.connect(self.export_pits)
        b_imp = QPushButton(icon("database-import-outline"), tr("Import from Excel ..."))
        b_imp.setToolTip(tr("Add or update pits and their setups from an Excel workbook (a template to fill in "
                            "is provided)"))
        b_imp.clicked.connect(self.import_excel)
        b_route = QPushButton(icon("map-marker-path", "#f0b429"), tr("Design a route ..."))
        b_route.setToolTip(tr("Design a new drive from a station to a target: within a gradient limit and away from "
                              "old workings and other drives (worked out on this computer)"))
        b_route.clicked.connect(self.design_route)
        b_geo = QPushButton(icon("layers-outline"), tr("Pit geology ..."))
        b_geo.setToolTip(tr("The rocks, structures and samples mapped in the selected pit (Ctrl+J)"))
        b_geo.clicked.connect(lambda: self.geology_requested.emit(self.pid))
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel(tr("Pits")))
        lv.addWidget(self.tree, 1)
        r = QHBoxLayout()
        r.addWidget(b_main)
        r.addWidget(b_branch)
        lv.addLayout(r)
        lv.addWidget(b_del)
        lv.addWidget(b_imp)
        lv.addWidget(b_exp)
        lv.addWidget(b_route)
        lv.addWidget(b_geo)
        lv.addWidget(b_set)

        # middle: pit header, setups list, setup form, messages
        self.title = QLabel(tr("Select or add a pit"))
        self.title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        self.p_name = QLineEdit()
        self.p_code = QLineEdit()
        self.p_code.setMaximumWidth(80)
        self.p_code.setToolTip(tr("Short code used to number the stations, e.g. P1 -> P1-01, P1-02"))
        self.p_w = QDoubleSpinBox()
        self.p_h = QDoubleSpinBox()
        for s in (self.p_w, self.p_h):
            s.setRange(0.5, 50)
            s.setDecimals(1)
            s.setSuffix(" m")
        self.p_level = QComboBox()
        for k, v in STATION_LEVELS.items():
            self.p_level.addItem(tr(v), k)
        self.p_adjust = QCheckBox(tr("Spread a passing closure (Bowditch)"))
        for w in (self.p_name, self.p_code):
            w.textEdited.connect(self._pit_edited)
        for w in (self.p_w, self.p_h):
            w.valueChanged.connect(self._pit_edited)
        self.p_level.currentIndexChanged.connect(self._pit_edited)
        self.p_adjust.toggled.connect(self._pit_edited)
        head = QGridLayout()
        head.addWidget(QLabel(tr("Name")), 0, 0)
        head.addWidget(self.p_name, 0, 1)
        head.addWidget(QLabel(tr("Code")), 0, 2)
        head.addWidget(self.p_code, 0, 3)
        head.addWidget(QLabel(tr("Profile")), 1, 0)
        prof = QHBoxLayout()
        prof.addWidget(self.p_w)
        prof.addWidget(QLabel(tr("wide x")))
        prof.addWidget(self.p_h)
        prof.addWidget(QLabel(tr("high, stations in the")))
        prof.addWidget(self.p_level)
        prof.addStretch(1)
        head.addLayout(prof, 1, 1, 1, 3)
        head.addWidget(self.p_adjust, 2, 1, 1, 3)
        self.l_entrance = QLabel(tr("Entrance"))
        self.p_x, self.p_y, self.p_z = _NumEdit(tr("Easting")), _NumEdit(tr("Northing")), _NumEdit(tr("Elevation"))
        self.entrance_row = QWidget()
        er = QHBoxLayout(self.entrance_row)
        er.setContentsMargins(0, 0, 0, 0)
        for lab, w in (("X", self.p_x), ("Y", self.p_y), ("Z", self.p_z)):
            er.addWidget(QLabel(lab))
            er.addWidget(w, 1)
            w.changed.connect(self._entrance_edited)
        head.addWidget(self.l_entrance, 3, 0)
        head.addWidget(self.entrance_row, 3, 1, 1, 3)

        self.table = QTableWidget(0, len(self.SUMMARY))
        self.table.setHorizontalHeaderLabels([tr(h) for h in self.SUMMARY])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.currentCellChanged.connect(lambda r, *_: self._select(r))
        b_add = QPushButton(icon("plus"), tr("Add setup"))
        b_add.setToolTip(tr("The next setup: at the last foresight, backsight on the station before, "
                            "next station number"))
        b_add.clicked.connect(self.add_setup)
        b_rm = QPushButton(icon("delete-outline"), tr("Delete setup"))
        b_rm.setToolTip(tr("Delete the selected setup"))
        b_rm.clicked.connect(self.delete_setup)
        b_up = QPushButton(icon("arrow-up"), "")
        b_up.setToolTip(tr("Move up"))
        b_up.clicked.connect(lambda: self.move_setup(-1))
        b_dn = QPushButton(icon("arrow-down"), "")
        b_dn.setToolTip(tr("Move down"))
        b_dn.clicked.connect(lambda: self.move_setup(1))
        trow = QHBoxLayout()
        for b in (b_add, b_rm, b_up, b_dn):
            trow.addWidget(b)
        trow.addStretch(1)
        self.b_doctor = QPushButton(icon("stethoscope", "#f0b429"), tr("Closure check"))
        self.b_doctor.setToolTip(tr("The traverse does not close? Find the reading mistake that best explains it"))
        self.b_doctor.clicked.connect(self.closure_check)
        trow.addWidget(self.b_doctor)

        self.form = self._setup_form()
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setTextInteractionFlags(Qt.TextSelectableByMouse)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.addWidget(self.title)
        mv.addLayout(head)
        mv.addWidget(self.table, 2)
        mv.addLayout(trow)
        mv.addWidget(self.form, 3)
        mv.addWidget(self.msg)

        # right: map + coordinates
        self.map = PlanMap()
        self.coords = QTableWidget(0, 5)
        self.coords.setHorizontalHeaderLabels([tr("Station"), tr("X (East)"), tr("Y (North)"), "Z", tr("Status")])
        self.coords.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.coords.verticalHeader().setVisible(False)
        self.coords.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.coords.horizontalHeader().setStretchLastSection(True)
        right = QSplitter(Qt.Vertical)
        mapbox = QWidget()
        mb = QVBoxLayout(mapbox)
        mb.setContentsMargins(0, 0, 0, 0)
        hint = QLabel(tr("Plan view - wheel zooms, drag pans, double-click fits"))
        hint.setStyleSheet(f"color: {theme.MUTED};")
        b_png = QPushButton(icon("image-outline"), tr("Export PNG ..."))
        b_png.setToolTip(tr("Save the plan map (same area and zoom) as a 300 dpi PNG on white: every pit in grey, "
                            "every station marked and named"))
        b_png.clicked.connect(self.export_map_png)
        hrow = QHBoxLayout()
        hrow.addWidget(hint, 1)
        hrow.addWidget(b_png)
        mb.addLayout(hrow)
        mb.addWidget(self.map, 1)
        right.addWidget(mapbox)
        right.addWidget(self.coords)
        right.setSizes([480, 240])

        self.mid_stack = QStackedWidget()           # a surveyed pit | a designed pit
        self.mid_stack.addWidget(mid)
        self.mid_stack.addWidget(self._design_page())
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.mid_stack)
        split.addWidget(right)
        split.setSizes([220, 620, 480])

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {theme.MUTED};")
        self.b_save = QPushButton(icon("content-save-outline"), tr("Save pit"))
        self.b_save.clicked.connect(self.save)
        self.b_discard = QPushButton(icon("undo"), tr("Discard changes"))
        self.b_discard.clicked.connect(self._discard)
        close = QPushButton(tr("Close"))
        close.setToolTip(tr("Close the window (asks to save changes first)"))
        self.b_save.setToolTip(tr("Write this pit (name, profile, setups) to the database; the drives in the view "
                                  "follow"))
        self.b_discard.setToolTip(tr("Forget the changes to this pit since it was last saved"))
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.b_discard)
        bottom.addWidget(self.b_save)
        bottom.addWidget(close)
        v = QVBoxLayout(self)
        v.addWidget(split, 1)
        v.addLayout(bottom)
        self._hints()
        self._layout_form()                         # only the chosen method's fields, also before a pit is shown
        self._set_dirty(False)

    # --- the setup form -------------------------------------------------------------------------
    def _setup_form(self) -> QGroupBox:
        g = QGroupBox(tr("Setup"))
        grid = QGridLayout(g)
        self.f_at = QLineEdit()
        self.f_at.setToolTip(tr("Station the instrument stands on"))
        self.f_hi = _NumEdit(tr("e.g. 1.500"))
        self.f_date = QDateEdit(QDate.currentDate())
        self.f_date.setCalendarPopup(True)
        self.f_date.setDisplayFormat("yyyy-MM-dd")
        self.f_surveyor = QLineEdit()
        self.l_at = QLabel(tr("Instrument at"))
        grid.addWidget(self.l_at, 0, 0)
        grid.addWidget(self.f_at, 0, 1)
        grid.addWidget(QLabel(tr("HI (m)")), 0, 2)
        grid.addWidget(self.f_hi, 0, 3)
        grid.addWidget(QLabel(tr("Date")), 0, 4)
        grid.addWidget(self.f_date, 0, 5)
        grid.addWidget(QLabel(tr("Surveyor")), 0, 6)
        grid.addWidget(self.f_surveyor, 0, 7)

        self.r_angles = QRadioButton(tr("Backsight + angles (total station)"))
        self.r_bearing = QRadioButton(tr("Bearing of the leg (compass)"))
        self.r_coords = QRadioButton(tr("Coordinates (X, Y, Z)"))
        self._method = QButtonGroup(self)
        self._method.addButton(self.r_angles)
        self._method.addButton(self.r_bearing)
        self._method.addButton(self.r_coords)
        self.r_angles.setChecked(True)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel(tr("Direction from:")))
        mrow.addWidget(self.r_angles)
        mrow.addWidget(self.r_bearing)
        mrow.addWidget(self.r_coords)
        mrow.addStretch(1)
        grid.addLayout(mrow, 1, 0, 1, 8)

        # backsight (angles method)
        self.g_bs = QGroupBox(tr("Backsight"))
        bg = QGridLayout(self.g_bs)
        self.bs_kind = QComboBox()
        self.bs_kind.addItem(tr("Station"), "station")
        self.bs_kind.addItem(tr("Known bearing (reference mark)"), "bearing")
        self.f_bs = QLineEdit()
        self.f_bs.setPlaceholderText(tr("station name, e.g. E1"))
        self.f_bs_brg = _AngleEdit(tr("bearing to the mark"))
        self.f_bs_type = QComboBox()
        for k in BEARING_TYPE_TEXT:
            self.f_bs_type.addItem(bearing_type_text(k), k)
        self.f_bs_type.setCurrentIndex(2)
        self.f_bs_ha = _AngleEdit(tr("0 00 00 if zeroed"))
        self.f_bs_za = _AngleEdit(tr("optional"))
        self.f_bs_sd = _NumEdit(tr("optional"))
        self.f_bs_ht = _NumEdit(tr("optional"))
        self.l_bs_target = QLabel(tr("Station"))
        bg.addWidget(QLabel(tr("Sight on")), 0, 0)
        bg.addWidget(self.bs_kind, 0, 1, 1, 3)
        bg.addWidget(self.l_bs_target, 1, 0)
        bg.addWidget(self.f_bs, 1, 1, 1, 3)
        bg.addWidget(self.f_bs_brg, 1, 1)
        bg.addWidget(self.f_bs_type, 1, 2, 1, 2)
        for c, (lab, w) in enumerate((("HA", self.f_bs_ha), ("ZA", self.f_bs_za), (tr("SD (m)"), self.f_bs_sd),
                                      (tr("HT (m)"), self.f_bs_ht))):
            bg.addWidget(QLabel(lab), 2 + c // 2, (c % 2) * 2)
            bg.addWidget(w, 2 + c // 2, (c % 2) * 2 + 1)
        for col in (1, 3):                      # the fields share the width; labels keep theirs
            bg.setColumnStretch(col, 1)

        # foresight
        self.g_fs = QGroupBox(tr("Foresight (new station)"))
        fg = QGridLayout(self.g_fs)
        self.f_fs = QLineEdit()
        self.f_fs.setToolTip(tr("Numbered automatically - type another name if you like"))
        self.f_fs_ha = _AngleEdit(tr("horizontal angle"))
        self.f_fs_brg = _AngleEdit(tr("bearing read on the compass"))
        self.f_fs_type = QComboBox()
        for k in BEARING_TYPE_TEXT:
            self.f_fs_type.addItem(bearing_type_text(k), k)
        self.f_fs_za = _AngleEdit(tr("90 = level, over 90 = down"))
        self.f_fs_sd = _NumEdit(tr("slope distance"))
        self.f_fs_ht = _NumEdit(tr("prism height"))
        fg.addWidget(QLabel(tr("Station")), 0, 0)
        fg.addWidget(self.f_fs, 0, 1)
        self.l_fs_ha = QLabel("HA")
        fg.addWidget(self.l_fs_ha, 1, 0)
        fg.addWidget(self.f_fs_ha, 1, 1)
        self.l_fs_brg = QLabel(tr("Bearing"))
        fg.addWidget(self.l_fs_brg, 1, 0)
        fg.addWidget(self.f_fs_brg, 1, 1)
        fg.addWidget(self.f_fs_type, 1, 2, 1, 2)
        self.l_fs_za, self.l_fs_sd, self.l_fs_ht = QLabel("ZA"), QLabel(tr("SD (m)")), QLabel(tr("HT (m)"))
        fg.addWidget(self.l_fs_za, 2, 0)
        fg.addWidget(self.f_fs_za, 2, 1)
        fg.addWidget(self.l_fs_sd, 2, 2)
        fg.addWidget(self.f_fs_sd, 2, 3)
        fg.addWidget(self.l_fs_ht, 3, 0)
        fg.addWidget(self.f_fs_ht, 3, 1)
        # method "coords": the station's coordinates, already calculated, typed in
        self.f_fs_x, self.f_fs_y, self.f_fs_z = (_NumEdit(tr("easting")), _NumEdit(tr("northing")),
                                                 _NumEdit(tr("elevation")))
        self.l_fs_x, self.l_fs_y, self.l_fs_z = QLabel(tr("X (East)")), QLabel(tr("Y (North)")), QLabel("Z")
        for k, (lab, w) in enumerate(((self.l_fs_x, self.f_fs_x), (self.l_fs_y, self.f_fs_y),
                                      (self.l_fs_z, self.f_fs_z))):
            fg.addWidget(lab, 4 + k, 0)                 # one under the other (user request)
            fg.addWidget(w, 4 + k, 1)
        for col in (1, 3):
            fg.setColumnStretch(col, 1)
        sights = QHBoxLayout()                  # the foresight takes the whole row when there is no backsight
        sights.addWidget(self.g_bs, 1)
        sights.addWidget(self.g_fs, 1)
        grid.addLayout(sights, 2, 0, 1, 8)

        # face right
        self.f_two = QCheckBox(tr("Two faces (face right readings)"))
        grid.addWidget(self.f_two, 3, 0, 1, 4)
        self.g_fr = QGroupBox(tr("Face right"))
        rg = QGridLayout(self.g_fr)
        self.f_bs_ha_r, self.f_bs_za_r = _AngleEdit(), _AngleEdit()
        self.f_fs_ha_r, self.f_fs_za_r = _AngleEdit(), _AngleEdit()
        for c, (lab, w) in enumerate(((tr("BS HA"), self.f_bs_ha_r), (tr("BS ZA"), self.f_bs_za_r),
                                      (tr("FS HA"), self.f_fs_ha_r), (tr("FS ZA"), self.f_fs_za_r))):
            rg.addWidget(QLabel(lab), 0, c * 2)
            rg.addWidget(w, 0, c * 2 + 1)
        grid.addWidget(self.g_fr, 4, 0, 1, 8)

        # signals
        for w in (self.f_at, self.f_bs, self.f_surveyor):
            w.textEdited.connect(self._form_edited)
        self.f_fs.textEdited.connect(self._fs_renamed)
        for w in (self.f_hi, self.f_bs_brg, self.f_bs_ha, self.f_bs_za, self.f_bs_sd, self.f_bs_ht, self.f_fs_ha,
                  self.f_fs_brg, self.f_fs_za, self.f_fs_sd, self.f_fs_ht, self.f_bs_ha_r, self.f_bs_za_r,
                  self.f_fs_ha_r, self.f_fs_za_r, self.f_fs_x, self.f_fs_y, self.f_fs_z):
            w.changed.connect(self._form_edited)
        for w in (self.bs_kind, self.f_bs_type, self.f_fs_type):
            w.currentIndexChanged.connect(self._form_edited)
        self._method.buttonToggled.connect(lambda _b, on: on and self._form_edited())
        self.f_two.toggled.connect(self._form_edited)
        self.f_date.dateChanged.connect(self._form_edited)
        g.setEnabled(False)
        return g

    def _hints(self) -> None:
        """A hover hint on every field: what it is and how to type it."""
        for w, text in (
            (self.p_name, tr("Name of the pit")),
            (self.p_code, tr("Short code used to number the stations, e.g. P1 -> P1-01, P1-02 ...")),
            (self.p_w, tr("Width of the drive, wall to wall (m)")),
            (self.p_h, tr("Height of the drive, floor to backs (m)")),
            (self.p_level, tr("Where the survey stations are in the drive: in the backs (roof), the floor or the "
                              "middle.\nThe drive is drawn from the stations with this profile.")),
            (self.p_x, tr("Easting of the entrance station (m). Changing it moves every station of this main pit "
                          "and its branches.")),
            (self.p_y, tr("Northing of the entrance station (m). Changing it moves every station of this main pit "
                          "and its branches.")),
            (self.p_z, tr("Elevation of the entrance station mark (m). Changing it moves every station of this main "
                          "pit and its branches.")),
            (self.p_adjust, tr("When the traverse closes within the tolerance, spread the misclosure over the legs "
                               "(Bowditch rule)")),
            (self.tree, tr("Main pits and their branches; select one to see and edit its survey")),
            (self.table, tr("The setups of this pit in order; select one to edit it below")),
            (self.coords, tr("Computed coordinates of this pit's stations")),
            (self.f_at, tr("Station the instrument stands on (normally the previous setup's foresight)")),
            (self.f_hi, tr("HI - instrument height: from the station mark to the instrument's horizontal axis (m)")),
            (self.f_date, tr("Date of the survey")),
            (self.f_surveyor, tr("Who did the survey")),
            (self.r_angles, tr("Total station: orient on a backsight, then read HA, ZA and SD to the foresight")),
            (self.r_bearing, tr("Compass: read the bearing of the leg directly - no backsight needed.\n"
                                "Magnetic bearings are turned into grid with the declination in Survey settings.")),
            (self.r_coords, tr("Coordinates: type the new station's X, Y, Z when they were already calculated "
                               "(e.g. by another program or an earlier survey).\nThey are used as they are, like a "
                               "known station: no readings, never adjusted; a later traverse can close on them.")),
            (self.f_fs_x, tr("X (easting) of the station, already calculated (m)")),
            (self.f_fs_y, tr("Y (northing) of the station, already calculated (m)")),
            (self.f_fs_z, tr("Z (elevation) of the station mark, already calculated (m)")),
            (self.bs_kind, tr("What the instrument is oriented on: a station with coordinates, or a reference "
                              "mark of known bearing")),
            (self.f_bs, tr("Backsight station - usually the station before; its coordinates orient the instrument")),
            (self.f_bs_brg, tr("Bearing from the instrument to the reference mark.") + " " + _angle_hint()),
            (self.f_bs_type, tr("Kind of bearing: grid, true (gyro) or magnetic (compass).\n"
                                "True and magnetic are turned into grid with Survey settings.")),
            (self.f_bs_ha, tr("HA on the backsight - what the horizontal circle showed when sighting the backsight:\n"
                              "  • circle set to zero on the backsight (the usual way): 0 00 00\n"
                              "  • circle not set: the reading shown, e.g. 47 12 30\n"
                              "  • circle set to the backsight's bearing: leave it empty\n"
                              "Only the angle turned (HA foresight − HA backsight) is used.") + " " + _angle_hint()),
            (self.f_bs_za, tr("ZA on the backsight - zenith angle (optional, for the check).") + " " + _angle_hint()),
            (self.f_bs_sd, tr("SD to the backsight (optional): compared with the stations' coordinates as a check "
                              "(m)")),
            (self.f_bs_ht, tr("HT - target (prism) height at the backsight (m)")),
            (self.f_fs, tr("Name of the new station - numbered automatically, type another name if you like.\n"
                           "A station that already has coordinates closes the traverse (closure check).")),
            (self.f_fs_ha, tr("HA on the foresight - the horizontal circle reading.\n"
                              "Bearing = backsight bearing + (HA foresight − HA backsight).") + " " + _angle_hint()),
            (self.f_fs_brg, tr("Bearing of the leg read on the compass (or a true / grid bearing).") + " "
             + _angle_hint()),
            (self.f_fs_type, tr("Kind of bearing: magnetic (compass), true (gyro) or grid.\n"
                                "Magnetic: true = magnetic + declination; grid = true − convergence.")),
            (self.f_fs_za, tr("ZA - zenith angle: 90° = level, under 90° = up, over 90° = down.") + " "
             + _angle_hint()),
            (self.f_fs_sd, tr("SD - slope distance to the prism (m). HD = SD × sin ZA")),
            (self.f_fs_ht, tr("HT - prism height at the foresight (m). ΔZ = SD × cos ZA + HI − HT")),
            (self.f_two, tr("Also enter the face right readings; both faces are averaged (flagged when they "
                            "differ by more than 30″)")),
            (self.f_bs_ha_r, tr("Face right HA on the backsight (about face left + 180°).") + " " + _angle_hint()),
            (self.f_bs_za_r, tr("Face right ZA on the backsight (about 360° − face left).") + " " + _angle_hint()),
            (self.f_fs_ha_r, tr("Face right HA on the foresight (about face left + 180°).") + " " + _angle_hint()),
            (self.f_fs_za_r, tr("Face right ZA on the foresight (about 360° − face left).") + " " + _angle_hint()),
            (self.map, tr("Plan view of the centre lines: this pit in colour, the selected leg in orange.\n"
                          "Wheel zooms, drag pans, double-click fits.")),
        ):
            w.setToolTip(text)

    def _layout_form(self) -> None:
        """Show only the fields the chosen method needs."""
        angles = self.r_angles.isChecked()
        coords = self.r_coords.isChecked()
        self.g_bs.setVisible(angles)
        by_station = self.bs_kind.currentData() == "station"
        self.f_bs.setVisible(by_station)
        self.l_bs_target.setText(tr("Station") if by_station else tr("Bearing"))
        self.f_bs_brg.setVisible(not by_station)
        self.f_bs_type.setVisible(not by_station)
        for w in (self.l_fs_ha, self.f_fs_ha):
            w.setVisible(angles)
        for w in (self.l_fs_brg, self.f_fs_brg, self.f_fs_type):
            w.setVisible(not angles and not coords)
        for w in (self.l_fs_za, self.f_fs_za, self.l_fs_sd, self.f_fs_sd, self.l_fs_ht, self.f_fs_ht):
            w.setVisible(not coords)
        for w in (self.l_fs_x, self.f_fs_x, self.l_fs_y, self.f_fs_y, self.l_fs_z, self.f_fs_z):
            w.setVisible(coords)
        self.f_hi.setEnabled(not coords)            # no instrument readings for typed coordinates
        self.l_at.setText(tr("From station") if coords else tr("Instrument at"))
        self.g_fs.setTitle(tr("New station (coordinates already calculated)") if coords
                           else tr("Foresight (new station)"))
        self.f_two.setVisible(angles)
        self.g_fr.setVisible(angles and self.f_two.isChecked())

    # --- loading ------------------------------------------------------------------------------------
    def start(self) -> bool:
        try:
            self.store.prepare()
            self._reload()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.parent(), tr("Pit Manager"), tr("The project database cannot be used:\n{e}",
                                                                     e=e))
            return False
        self.status.setText(tr("Database: {name}", name=self.store.target.short))
        return True

    def _reload(self, select=None) -> None:
        """Read the pits (and designed pits) again; ``select``: a pit id, or ("d", design id)."""
        self.pits = self.store.pits()
        self.stations = self.store.stations()
        self.setups = self.store.setups()
        self.settings = self.store.settings()
        self.designs = self.store.designs()
        self.tree.blockSignals(True)
        self.tree.clear()
        items = {}
        for p in sorted(self.pits, key=lambda q: (q.parent_id is not None, q.id)):
            it = QTreeWidgetItem([f"{p.name}  ({p.code})"])
            it.setData(0, Qt.UserRole, p.id)
            if p.parent_id is None:
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
            items[p.id] = it
        for p in self.pits:
            parent = items.get(p.parent_id)
            (parent.addChild if parent is not None else self.tree.addTopLevelItem)(items[p.id])
        d_items = {}
        if self.designs:                            # after the pits: the designed pits
            g = QTreeWidgetItem([tr("Designed Pits")])
            g.setData(0, GROUP_ROLE, "designs")
            g.setIcon(0, icon("pencil-ruler"))
            g.setToolTip(0, tr("Drives designed with the route designer and saved to the database (not surveyed)"))
            f = g.font(0)
            f.setBold(True)
            g.setFont(0, f)
            self.tree.addTopLevelItem(g)
            for x in self.designs:
                it = QTreeWidgetItem([x.name])
                it.setData(0, DESIGN_ROLE, x.id)
                it.setIcon(0, icon("vector-polyline", "#4ce67a"))
                it.setToolTip(0, tr("Designed pit {name}: from {a} to {b}, {m:,.0f} m", name=x.name,
                                    a=x.start_name or "-", b=x.target_name or "-", m=x.length))
                g.addChild(it)
                d_items[x.id] = it
        self.tree.expandAll()
        self.tree.blockSignals(False)
        if isinstance(select, tuple) and select[1] in d_items:
            self.tree.setCurrentItem(d_items[select[1]])
            return
        if select is None and self.did in d_items:
            self.tree.setCurrentItem(d_items[self.did])
            return
        target = select if select in items else (self.pid if self.pid in items else
                                                 (self.pits[0].id if self.pits else None))
        if target is not None:
            self.tree.setCurrentItem(items[target])
        else:
            self._show_pit(None)

    def reload_designs(self) -> None:
        """The designed pits changed elsewhere (saved by the route designer, renamed or deleted in the layers
        list): the tree follows; the pit being edited keeps its changes."""
        keep = ("d", self.did) if self.did is not None else None
        self._reload(select=keep)
        if self.did is not None and not self.dirty:
            if any(x.id == self.did for x in self.designs):
                self._show_design(self.did)
            else:
                self._show_pit(self.pits[0].id if self.pits else None)

    @staticmethod
    def _key(item):
        if item is None:
            return None
        if item.data(0, GROUP_ROLE) is not None:
            return ("g", None)
        if item.data(0, DESIGN_ROLE) is not None:
            return ("d", item.data(0, DESIGN_ROLE))
        return ("p", item.data(0, Qt.UserRole))

    def _shown_key(self):
        return ("d", self.did) if self.did is not None else ("p", self.pid)

    def _pit_clicked(self, item, _prev) -> None:
        key = self._key(item) or ("p", None)
        if key == self._shown_key() or key[0] == "g":      # the group: what is shown stays
            return
        if not self._leave():
            self.tree.blockSignals(True)
            for it in self._all_items():
                if self._key(it) == self._shown_key():
                    self.tree.setCurrentItem(it)
            self.tree.blockSignals(False)
            return
        if key[0] == "d":
            self._show_design(key[1])
        else:
            self._show_pit(key[1])

    def _all_items(self):
        out, todo = [], [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while todo:
            it = todo.pop()
            out.append(it)
            todo += [it.child(i) for i in range(it.childCount())]
        return out

    def _show_pit(self, pid) -> None:
        self.pid = pid
        self.did = None
        self.mid_stack.setCurrentIndex(0)
        self.b_save.setText(tr("Save pit"))
        p = next((q for q in self.pits if q.id == pid), None)
        self._filling = True
        if p is None:
            self.title.setText(tr("Add a main pit to start - its entrance's coordinates are needed"))
            self.rows = []
        else:
            parent = next((q for q in self.pits if q.id == p.parent_id), None)
            self.title.setText(tr("{name}   - branch of {parent} at {start}", name=p.name, parent=parent.name,
                                  start=p.start_station) if parent else
                               tr("{name}   - entrance {start}", name=p.name, start=p.start_station))
            self.p_name.setText(p.name)
            self.p_code.setText(p.code)
            self.p_w.setValue(p.profile_width)
            self.p_h.setValue(p.profile_height)
            self.p_level.setCurrentIndex(max(self.p_level.findData(p.station_level), 0))
            self.p_adjust.setChecked(p.adjust)
            self.rows = copy.deepcopy(self.setups.get(pid, []))
        main = p is not None and p.parent_id is None
        self.l_entrance.setVisible(main)
        self.entrance_row.setVisible(main)
        if main:
            st = self._entrance(p)
            self.l_entrance.setText(tr("Entrance {name}", name=p.start_station))
            for w, v in zip((self.p_x, self.p_y, self.p_z), (st.x, st.y, st.z) if st else (None,) * 3):
                w.set_value(v)
        self._filling = False
        self.cur = -1
        self._set_dirty(False)
        self._recompute(refit=True)
        if self.rows:
            self._goto(0)
        else:
            self._select(-1)

    # --- editing --------------------------------------------------------------------------------------
    def _select(self, r) -> None:
        self.cur = r if 0 <= r < len(self.rows) else -1
        self.form.setEnabled(self.cur >= 0)
        if self.cur < 0:
            self.form.setTitle(tr("Setup"))
            return
        row = self.rows[self.cur]
        self._filling = True
        self.form.setTitle(tr("Setup {n} - instrument at {at}", n=self.cur + 1, at=row.at))
        self.f_at.setText(row.at)
        self.f_hi.set_value(row.hi)
        d = QDate.fromString(row.date or "", "yyyy-MM-dd")
        self.f_date.setDate(d if d.isValid() else QDate.currentDate())
        self.f_surveyor.setText(row.surveyor)
        {"bearing": self.r_bearing, "coords": self.r_coords}.get(_method(row), self.r_angles).setChecked(True)
        self.f_fs_x.set_value(row.fs_x)
        self.f_fs_y.set_value(row.fs_y)
        self.f_fs_z.set_value(row.fs_z)
        self.bs_kind.setCurrentIndex(1 if (row.bs_station is None and row.bs_bearing is not None) else 0)
        self.f_bs.setText(row.bs_station or "")
        self.f_bs_brg.set_value(row.bs_bearing)
        self.f_bs_type.setCurrentIndex(max(self.f_bs_type.findData(row.bs_bearing_type or "grid"), 0))
        self.f_bs_ha.set_value(row.bs_ha)
        self.f_bs_za.set_value(row.bs_za)
        self.f_bs_sd.set_value(row.bs_sd)
        self.f_bs_ht.set_value(row.bs_ht)
        self.f_fs.setText(row.fs or "")
        self.f_fs_ha.set_value(row.fs_ha)
        self.f_fs_brg.set_value(row.fs_bearing)
        self.f_fs_type.setCurrentIndex(max(self.f_fs_type.findData(row.fs_bearing_type or "magnetic"), 0))
        self.f_fs_za.set_value(row.fs_za)
        self.f_fs_sd.set_value(row.fs_sd)
        self.f_fs_ht.set_value(row.fs_ht)
        two = any(v is not None for v in (row.bs_ha_r, row.bs_za_r, row.fs_ha_r, row.fs_za_r))
        self.f_two.setChecked(two)
        self.f_bs_ha_r.set_value(row.bs_ha_r)
        self.f_bs_za_r.set_value(row.bs_za_r)
        self.f_fs_ha_r.set_value(row.fs_ha_r)
        self.f_fs_za_r.set_value(row.fs_za_r)
        self._layout_form()
        self._filling = False
        self._redraw_map()

    def _goto(self, r) -> None:
        """Select setup ``r`` in the table and show it in the form - also when the table's
        current row does not change (another pit, or a deleted setup, at the same row)."""
        if 0 <= r < len(self.rows):
            self.table.selectRow(r)
        self._select(r)

    def _form_edited(self, *_a) -> None:
        if self._filling or self.cur < 0:
            return
        self._layout_form()
        r = self.rows[self.cur]
        r.at = self.f_at.text().strip()
        r.hi = self.f_hi.value()
        r.date = self.f_date.date().toString("yyyy-MM-dd")
        r.surveyor = self.f_surveyor.text().strip()
        angles = self.r_angles.isChecked()
        coords = self.r_coords.isChecked()
        r.ui_method = "angles" if angles else ("coords" if coords else "bearing")
        if coords:                                   # typed coordinates: no backsight, no readings
            r.fs_x, r.fs_y, r.fs_z = self.f_fs_x.value(), self.f_fs_y.value(), self.f_fs_z.value()
            r.bs_station = r.bs_bearing = r.fs_bearing = None
            r.bs_ha = r.bs_za = r.bs_sd = r.bs_ht = r.fs_ha = None
            r.bs_ha_r = r.bs_za_r = r.fs_ha_r = r.fs_za_r = None
            r.fs_za = r.fs_sd = None
            self.form.setTitle(tr("Setup {n} - instrument at {at}", n=self.cur + 1, at=r.at))
            self._set_dirty(True)
            self._timer.start()
            return
        r.fs_x = r.fs_y = r.fs_z = None
        if angles:
            by_station = self.bs_kind.currentData() == "station"
            r.bs_station = self.f_bs.text().strip() or None if by_station else None
            r.bs_bearing = None if by_station else self.f_bs_brg.value()
            r.bs_bearing_type = self.f_bs_type.currentData()
            r.bs_ha, r.bs_za, r.bs_sd, r.bs_ht = (self.f_bs_ha.value(), self.f_bs_za.value(), self.f_bs_sd.value(),
                                                  self.f_bs_ht.value())
            r.fs_ha, r.fs_bearing = self.f_fs_ha.value(), None
            two = self.f_two.isChecked()
            r.bs_ha_r = self.f_bs_ha_r.value() if two else None
            r.bs_za_r = self.f_bs_za_r.value() if two else None
            r.fs_ha_r = self.f_fs_ha_r.value() if two else None
            r.fs_za_r = self.f_fs_za_r.value() if two else None
        else:
            r.fs_bearing = self.f_fs_brg.value()
            r.fs_bearing_type = self.f_fs_type.currentData()
        r.fs_za, r.fs_sd, r.fs_ht = self.f_fs_za.value(), self.f_fs_sd.value(), self.f_fs_ht.value()
        self.form.setTitle(tr("Setup {n} - instrument at {at}", n=self.cur + 1, at=r.at))
        self._set_dirty(True)
        self._timer.start()

    def _fs_renamed(self, text) -> None:
        """A foresight renamed: the later setups that stood on it / sighted it follow."""
        if self._filling or self.cur < 0:
            return
        old, new = self.rows[self.cur].fs, text.strip()
        self.rows[self.cur].fs = new or None
        if old and new and old != new:
            for r in self.rows[self.cur + 1:]:
                if r.at == old:
                    r.at = new
                if r.bs_station == old:
                    r.bs_station = new
        self._set_dirty(True)
        self._timer.start()

    def _station_names(self) -> set:
        names = {s.name for s in self.stations}
        names |= {r.fs for rows in self.setups.values() for r in rows if r.fs}
        names |= {r.fs for r in self.rows if r.fs}
        return names

    def add_setup(self) -> None:
        p = next((q for q in self.pits if q.id == self.pid), None)
        if p is None:
            return
        code = self.p_code.text().strip() or p.code
        if self.rows:
            last = self.rows[-1]
            at = last.fs or last.at
            row = SetupRow(at, last.hi, bs_station=last.at, bs_ht=last.fs_ht, fs_ht=last.fs_ht,
                           date=last.date, surveyor=last.surveyor)
            if _method(last) == "bearing":                      # keep surveying the same way
                row = SetupRow(at, last.hi, fs_bearing_type=last.fs_bearing_type, fs_ht=last.fs_ht,
                               date=last.date, surveyor=last.surveyor)
                row.ui_method = "bearing"
            elif _method(last) == "coords":                     # more stations with known coordinates
                row = SetupRow(at, date=last.date, surveyor=last.surveyor)
                row.ui_method = "coords"
        else:
            back = self._parent_backsight(p)
            row = SetupRow(p.start_station or "", 1.5, bs_station=back, bs_bearing=None if back else 0.0,
                           date=QDate.currentDate().toString("yyyy-MM-dd"))
        row.fs = next_station_name(code, self._station_names())
        self.rows.append(row)
        self._set_dirty(True)
        self._recompute()
        self._goto(len(self.rows) - 1)

    def _parent_backsight(self, p):
        """For a branch's first setup: the station before the junction on the parent's line."""
        if p.parent_id is None:
            return None
        for r in self.setups.get(p.parent_id, []):
            if r.fs == p.start_station:
                return r.at
        return None

    def delete_setup(self) -> None:
        if self.cur < 0:
            return
        del self.rows[self.cur]
        self._set_dirty(True)
        self._recompute()
        self._goto(min(self.cur, len(self.rows) - 1))

    def move_setup(self, step) -> None:
        i = self.cur
        j = i + step
        if i < 0 or not 0 <= j < len(self.rows):
            return
        self.rows[i], self.rows[j] = self.rows[j], self.rows[i]
        self._set_dirty(True)
        self._recompute()
        self._goto(j)

    def _entrance(self, p):
        """The known entrance station of main pit ``p`` (None if it has none)."""
        return next((s for s in self.stations if s.name == p.start_station and s.kind == "known"), None)

    def _typed_entrance(self):
        """(station name, (x, y, z)) typed for the shown main pit's entrance, when all three are
        numbers; else None."""
        p = next((q for q in self.pits if q.id == self.pid), None)
        if p is None or p.parent_id is not None or not p.start_station:
            return None
        xyz = (self.p_x.value(), self.p_y.value(), self.p_z.value())
        return None if None in xyz else (p.start_station, xyz)

    def _entrance_edited(self) -> None:
        if self._filling:
            return
        self._set_dirty(True)
        self._timer.start()                      # the whole main pit and its branches move with it

    def _pit_edited(self, *_a) -> None:
        if not self._filling:
            self._set_dirty(True)

    def _set_dirty(self, on) -> None:
        self.dirty = on and (self.pid is not None or self.did is not None)
        self.b_save.setEnabled(self.dirty)
        self.b_discard.setEnabled(self.dirty)

    # --- computing and showing -------------------------------------------------------------------
    def _recompute(self, refit=False) -> None:
        setups = dict(self.setups)
        if self.pid is not None:
            setups[self.pid] = self.rows
        known = [s for s in self.stations if s.kind == "known"]
        typed = self._typed_entrance()
        if typed is not None:
            name, xyz = typed
            known = [dataclasses.replace(s, x=xyz[0], y=xyz[1], z=xyz[2]) if s.name == name else s for s in known]
        try:
            self.net = compute(self.pits, known, setups, self.settings)
        except Exception as e:  # noqa: BLE001 - never break the window on odd input
            self.net = None
            self.msg.setText(f"⚠ {e}")
            return
        self._fill_table()
        self._fill_coords()
        self._fill_messages()
        self._redraw_map(refit)

    def _fill_table(self) -> None:
        net = self.net
        legs = {lg.setup_index: lg for lg in net.legs if lg.pit_id == self.pid}
        bad = {i for pid, i, _m in net.messages if pid == self.pid}
        t = self.table
        keep = self.cur
        t.blockSignals(True)
        t.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            lg = legs.get(i)
            how = (tr("coordinates") if _method(r) == "coords" else
                   tr("compass {brg}", brg=format_dms(r.fs_bearing) if r.fs_bearing is not None else "?")
                   if _method(r) == "bearing" else
                   tr("BS {name}", name=r.bs_station) if r.bs_station else
                   tr("bearing {brg}", brg=format_dms(r.bs_bearing)) if r.bs_bearing is not None else "-")
            cells = [str(i + 1), r.at, how, r.fs or "-", format_dms(lg.bearing) if lg else "-",
                     f"{lg.hd:,.3f}" if lg else "-", f"{lg.dz:+,.3f}" if lg else "-",
                     "⚠" if i in bad else (tr("closes") if lg and lg.closing else ("✓" if lg else "-"))]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c in (4, 5, 6):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 7 and text == "⚠":
                    it.setForeground(QColor(theme.DANGER))
                t.setItem(i, c, it)
        t.blockSignals(False)
        if 0 <= keep < len(self.rows):
            t.selectRow(keep)

    def _fill_coords(self) -> None:
        net = self.net
        names = []
        p = next((q for q in self.pits if q.id == self.pid), None)
        if p is not None and p.start_station:
            names.append(p.start_station)
        names += [r.fs for r in self.rows if r.fs and r.fs not in names]
        t = self.coords
        t.setRowCount(len(names))
        for i, n in enumerate(names):
            c = net.coords.get(n)
            st = net.status.get(n, "unresolved" if c is None else "")
            word = tr(STATUS_WORD[st]) if st in STATUS_WORD else st
            vals = [n] + ([f"{v:,.3f}" for v in c] if c else ["-", "-", "-"]) + [word]
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if 1 <= k <= 3:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if k == 4:
                    it.setForeground(QColor.fromRgbF(*STATUS_COLORS.get(st, (0.6, 0.6, 0.6))))
                t.setItem(i, k, it)

    def _fill_messages(self) -> None:
        net = self.net
        res = net.pits.get(self.pid)
        out = []
        if res is not None and res.closure is not None:
            de, dn, dz, lin, ratio, passed, onto = res.closure
            r = tr("perfect") if math.isinf(ratio) else f"1:{ratio:,.0f}"
            limit = self.settings['closure_ratio']
            if passed:
                verdict = (tr("passes 1:{limit:,.0f}, spread over the legs.", limit=limit) if res.adjusted else
                           tr("passes 1:{limit:,.0f}.", limit=limit))
            else:
                verdict = (f"<span style='color:{theme.DANGER}'>"
                           + tr("worse than 1:{limit:,.0f}: check heights, station names and readings", limit=limit)
                           + "</span>")
            out.append(tr("<b>{mark} Closes on {onto}</b>: misclosure {lin:.3f} m (ΔE {de:+.3f}, ΔN {dn:+.3f}, "
                          "ΔZ {dz:+.3f}) over {length:,.1f} m = {ratio} - {verdict}", mark='✓' if passed else '✖',
                          onto=onto, lin=lin, de=de, dn=dn, dz=dz, length=res.length, ratio=r, verdict=verdict))
        elif self.rows and all(_method(r) == "coords" for r in self.rows):
            out.append(tr("Every station of this pit has coordinates typed in (already calculated): "
                          "nothing to check."))
        elif self.rows:
            out.append(tr("Open traverse (not checked yet): close it on a known station, or back on the entrance, "
                          "to check it."))
        for pid, i, m in net.messages:
            if pid == self.pid:
                out.append("<span style='color:#f0a24a'>⚠ " + tr("Setup {n}: {msg}", n=i + 1, msg=m) + "</span>")
        self.msg.setText("<br>".join(out))

    def _redraw_map(self, refit=False) -> None:
        if self.net is None:
            return
        setups = dict(self.setups)
        if self.pid is not None:
            setups[self.pid] = self.rows
        lines = [(pid, names) for pid, ls in pit_paths(setups).items() for names in ls]
        sel = None
        if 0 <= self.cur < len(self.rows) and self.rows[self.cur].fs:
            sel = (self.rows[self.cur].at, self.rows[self.cur].fs)
        known = {s.name for s in self.stations if s.kind == "known"}
        designs = [(x.id, x.points) for x in self.designs]
        if self.did is not None:
            try:
                edited = self._d_points()
            except ValueError:
                edited = None
            if edited:
                designs = [(i, edited if i == self.did else pts) for i, pts in designs]
        self.map.set_data(self.net.coords, self.net.status, lines, self.pid, sel, known, refit, designs, self.did)

    # --- pits ----------------------------------------------------------------------------------------
    def new_main_pit(self) -> None:
        if not self._leave():
            return
        dlg = _NewPitDialog(self, tr("New main pit"))
        while dlg.exec():
            name, start, xyz = dlg.values()
            try:
                if not start:
                    raise ValueError(tr("Type the entrance station's name"))
                if xyz is None or None in xyz:
                    raise ValueError(tr("Type the entrance's X, Y and Z"))
                pid = self.store.add_pit(name, start_station=start, known=xyz)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, tr("New main pit"), str(e))
                continue
            self._after_pit_change(pid, tr("Added main pit {name} - now add its first setup", name=name))
            return

    def new_branch(self) -> None:
        if self.pid is None:
            QMessageBox.information(self, tr("New branch"), tr("Select the pit the branch starts from."))
            return
        if not self._leave():
            return
        parent = next(q for q in self.pits if q.id == self.pid)
        net = self.net
        names = [parent.start_station] + [r.fs for r in self.setups.get(parent.id, []) if r.fs]
        names = [n for n in names if n and net is not None and n in net.coords]
        if not names:
            QMessageBox.information(self, tr("New branch"), tr("{name} has no stations with coordinates yet.",
                                                               name=parent.name))
            return
        dlg = _NewPitDialog(self, tr("New branch"), branch_of=parent.name, stations=names)
        while dlg.exec():
            name, start, _ = dlg.values()
            try:
                pid = self.store.add_pit(name, parent_id=parent.id, start_station=start)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, tr("New branch"), str(e))
                continue
            self._after_pit_change(pid, tr("Added branch {name} at {start} - now add its first setup", name=name,
                                           start=start))
            return

    def delete_pit(self) -> None:
        if self.did is not None:
            self.delete_design()
            return
        p = next((q for q in self.pits if q.id == self.pid), None)
        if p is None:
            return
        n_br = sum(1 for q in self.pits if q.parent_id == p.id)
        ask = (tr("Delete {name} and its {n} branch(es) with all setups and stations?\n\nThis cannot be undone.",
                  name=p.name, n=n_br) if n_br else
               tr("Delete {name} with all setups and stations?\n\nThis cannot be undone.", name=p.name))
        if QMessageBox.question(self, tr("Delete pit"), ask) != QMessageBox.Yes:
            return
        self.store.delete_pit(p.id)
        self.store.recompute()
        self.dirty = False
        self.pid = None
        self._after_pit_change(None, tr("Deleted {name}", name=p.name))

    def _after_pit_change(self, select, text) -> None:
        self.dirty = False
        self._reload(select=select)
        self.status.setText(text)
        self.changed.emit()
        if select is not None and not self.rows:
            self.add_setup()

    def edit_settings(self) -> None:
        dlg = _SettingsDialog(self, self.settings)
        if dlg.exec():
            self.store.set_settings(dlg.values())
            self.store.recompute()
            self.settings = self.store.settings()
            self.stations = self.store.stations()
            self._recompute()
            self.status.setText(tr("Survey settings saved"))
            self.changed.emit()

    # --- saving ------------------------------------------------------------------------------------
    # --- export -----------------------------------------------------------------------------------
    def _export_allowed(self, title: str) -> bool:
        """Export needs a license with data export (Ultimate); asks the main window, which
        offers the license dialog."""
        from .. import licensing

        win = self.parent()
        if win is not None and hasattr(win, "_licensed_for"):
            return win._licensed_for(licensing.FEAT_EXPORT, title)
        st = licensing.refresh()
        if st.ok and st.license.has(licensing.FEAT_EXPORT):
            return True
        QMessageBox.information(self, title, tr("Your license does not include data export."))
        return False

    def _export_folder(self) -> str:
        last = getattr(self, "_last_export_dir", None)
        if last:
            return last
        win = self.parent()
        ws = getattr(win, "workspace", None)
        if ws is not None and getattr(ws, "folder", None):
            return str(ws.folder)
        path = getattr(self.store.target, "path", None)
        return str(Path(path).parent) if path else str(Path.home())

    def _saved_survey(self):
        """The survey as stored in the database (what the view draws)."""
        from ..scene.survey import SurveyData

        return SurveyData(self.store.pits(), {s.name: s for s in self.store.stations()},
                          pit_paths(self.store.setups()), self.store.target.label)

    def export_pits(self) -> None:
        """All pits, or the selected pit and its branches: centre lines to a shapefile, or the
        drives to a Surpac DTM. Uses the saved survey (unsaved changes are offered for saving)."""
        from ..io import pit_export

        if not self._export_allowed(tr("Export pits")):
            return
        if self.dirty:
            if not self._leave():
                return
            self._show_pit(self.pid)               # the saved pit (after Discard the form still had the edits)
        try:
            data = self._saved_survey()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Export pits"), tr("The survey cannot be read:\n{e}", e=e))
            return
        if not data.pits:
            QMessageBox.information(self, tr("Export pits"), tr("There are no pits to export yet."))
            return
        pit = data.pit(self.pid)
        family = len(pit_export.export_pits(data, pit.id)) if pit else 0
        dlg = _ExportPitsDialog(self, len(data.pits), pit, family, getattr(self, "_last_export_fmt", "shp"))
        if dlg.exec() != QDialog.Accepted:
            return
        one, fmt = dlg.values()
        self._last_export_fmt = fmt
        stem = f"{pit.code or pit.name}_and_branches" if one else "all_pits"
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        ext = ".shp" if fmt == "shp" else ".dtm"
        path, _ = QFileDialog.getSaveFileName(self, tr("Export pits"), str(Path(self._export_folder()) / (stem + ext)),
                                              tr(EXPORT_FORMATS[fmt][1]))
        if not path:
            return
        path = Path(path).with_suffix(ext)
        self._last_export_dir = str(path.parent)
        writer = pit_export.write_centerlines_shp if fmt == "shp" else pit_export.write_drives_dtm
        try:
            res = writer(path, data, pit.id if one else None)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Export pits"), tr("Nothing was exported:\n{e}", e=e))
            return
        if fmt == "shp":
            head = trn("{n} pit ({parts} lines) written to", "{n} pits ({parts} lines) written to", res.pits,
                       parts=res.parts)
        else:
            head = trn("{n} pit ({parts} solids) written to", "{n} pits ({parts} solids) written to", res.pits,
                       parts=res.parts)
        text = (head + "\n" + "\n".join(f"    {f.name}" for f in res.files) + "\n"
                + tr("in {folder}", folder=path.parent))
        if res.skipped:
            text += "\n\n" + tr("Left out (no coordinates yet): {names}", names=", ".join(res.skipped))
        self.status.setText(tr("Exported {name}", name=path.name))
        QMessageBox.information(self, tr("Export pits"), text)

    def import_excel(self) -> None:
        """Pits and setups from an Excel workbook (the template filled in)."""
        from ..io import survey_import as si
        from .excel_import import ExcelImportDialog

        if not self._leave():
            return
        dlg = ExcelImportDialog(
            self, title=tr("Import survey from Excel"),
            intro=tr("Add pits and their setups at once: save the template, fill in its sheets pits and setups "
                     "in Excel, then choose the file here. The traverse is computed during the check, so "
                     "problems show before anything is imported. A pit already in the database (same name) "
                     "is replaced - its setups too, when the setups sheet has rows for it - or skipped."),
            existing=tr("Pits already in the database"), folder=self._export_folder(),
            template_name="survey_template.xlsx",
            save_template=si.save_template,
            read=lambda path: si.read_import(self.store, path),
            apply=lambda plan, replace: si.apply_import(self.store, plan, replace),
            result_text=si.result_text)
        if dlg.exec() != QDialog.Accepted:
            return
        self._last_export_dir = dlg.folder
        self.dirty = False
        select, self.pid = self.pid, None             # show the (maybe replaced) pit again
        self._reload(select=select)
        self.status.setText(dlg.result_message())
        self.changed.emit()

    # --- decision support (AI, offline) ---------------------------------------------------------
    def closure_check(self) -> None:
        """Find the single reading mistake that best explains the pit's misclosure."""
        from .closure_dialog import ClosureDialog

        if self.pid is None:
            QMessageBox.information(self, tr("Closure check"), tr("Select a pit first."))
            return
        ClosureDialog(self).exec()

    def design_route(self) -> None:
        """Design a new drive (non-modal, next to this window)."""
        from .route_dialog import RouteDialog

        if self.net is None or not self.net.coords:
            QMessageBox.information(self, tr("Route designer"), tr("There are no stations with coordinates yet: add "
                                                                   "a main pit with its entrance first."))
            return
        dlg = getattr(self, "route_dialog", None)
        if dlg is None or not dlg.isVisible():
            self.route_dialog = dlg = RouteDialog(self)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def export_map_png(self) -> None:
        """The plan map as shown, as a 300 dpi PNG on white."""
        if not self._export_allowed(tr("Export map")):
            return
        if self.map._view is None:
            QMessageBox.information(self, tr("Export map"), tr("The plan map is empty - nothing to export."))
            return
        p = next((q for q in self.pits if q.id == self.pid), None)
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in (f"plan_{p.code or p.name}" if p else "plan"))
        path, _ = QFileDialog.getSaveFileName(self, tr("Export plan map"),
                                              str(Path(self._export_folder()) / (stem + ".png")),
                                              tr("PNG image (*.png)"))
        if not path:
            return
        path = Path(path).with_suffix(".png")
        self._last_export_dir = str(path.parent)
        title = (tr("Survey plan - {name} ({code})", name=p.name, code=p.code) if p else tr("Survey plan")) + \
            f"   {QDate.currentDate().toString('yyyy-MM-dd')}" + ("   " + tr("(unsaved changes)") if self.dirty else "")
        try:
            w, h = self.map.save_png(path, title)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Export map"), str(e))
            return
        self.status.setText(tr("Exported {name} ({w} x {h} px, 300 dpi)", name=path.name, w=w, h=h))

    def save(self) -> bool:
        if self.did is not None:
            return self._save_design()
        if self.pid is None:
            return True
        try:
            p = next(q for q in self.pits if q.id == self.pid)
            if p.parent_id is None:
                typed = self._typed_entrance()
                if typed is None:
                    raise ValueError(tr("Type the entrance's X, Y and Z"))
                st = self._entrance(p)
                if st is None or (st.x, st.y, st.z) != typed[1]:
                    self.store.set_known(typed[0], *typed[1], pit_id=p.id)
            self.store.update_pit(self.pid, name=self.p_name.text(), code=self.p_code.text(),
                                  adjust=self.p_adjust.isChecked(),
                                  profile=(self.p_w.value(), self.p_h.value(), self.p_level.currentData()))
            rows = [r for r in self.rows if r.at]
            self.store.save_setups(self.pid, rows)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Save pit"), tr("The pit was not saved:\n{e}", e=e))
            return False
        self.dirty = False
        self._reload(select=self.pid)
        self._show_pit(self.pid)
        res = self.net.pits.get(self.pid) if self.net else None
        self.status.setText(tr("Saved") if res is None or res.closure is None else
                            tr("Saved - traverse closed") if res.closure[5] else tr("Saved - closure failed"))
        self.changed.emit()
        return True

    def _leave(self) -> bool:
        if not self.dirty:
            return True
        ans = QMessageBox.question(self, tr("Pit Manager"), tr("Save the changes to this pit?"),
                                   QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if ans == QMessageBox.Save:
            return self.save()
        if ans == QMessageBox.Discard:
            self.dirty = False
            return True
        return False

    def _discard(self) -> None:
        if self.did is not None:
            self._show_design(self.did)
        else:
            self._show_pit(self.pid)

    # --- designed pits ------------------------------------------------------------------------------
    def _design_page(self) -> QWidget:
        """The middle of the window for a designed pit: name, profile, note and its centre line points."""
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        self.d_title = QLabel()
        self.d_title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        self.d_name = QLineEdit()
        self.d_note = QLineEdit()
        self.d_note.setPlaceholderText(tr("e.g. option B, approved, to be checked ..."))
        self.d_w, self.d_h = QDoubleSpinBox(), QDoubleSpinBox()
        for s in (self.d_w, self.d_h):
            s.setRange(0.5, 50)
            s.setDecimals(1)
            s.setSuffix(" m")
            s.valueChanged.connect(self._design_edited)
        self.d_level = QComboBox()
        for k, val in STATION_LEVELS.items():
            self.d_level.addItem(tr(val), k)
        self.d_level.setToolTip(tr("Where the points are in the drive's profile"))
        self.d_level.currentIndexChanged.connect(self._design_edited)
        for w in (self.d_name, self.d_note):
            w.textEdited.connect(self._design_edited)
        head = QGridLayout()
        head.addWidget(QLabel(tr("Name")), 0, 0)
        head.addWidget(self.d_name, 0, 1)
        head.addWidget(QLabel(tr("Profile")), 1, 0)
        prof = QHBoxLayout()
        prof.addWidget(self.d_w)
        prof.addWidget(QLabel(tr("wide x")))
        prof.addWidget(self.d_h)
        prof.addWidget(QLabel(tr("high, stations in the")))
        prof.addWidget(self.d_level)
        prof.addStretch(1)
        head.addLayout(prof, 1, 1)
        head.addWidget(QLabel(tr("Note")), 2, 0)
        head.addWidget(self.d_note, 2, 1)
        self.d_info = QLabel()
        self.d_info.setWordWrap(True)
        self.d_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.d_table = QTableWidget(0, 6)
        self.d_table.setHorizontalHeaderLabels(["#", tr("X (East)"), tr("Y (North)"), "Z", tr("Leg length (m)"),
                                                tr("Gradient")])
        self.d_table.verticalHeader().setVisible(False)
        self.d_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.d_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.d_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.d_table.setToolTip(tr("The centre line of the designed drive, point by point: double-click X, Y or Z "
                                   "to change it"))
        self.d_table.itemChanged.connect(self._design_point_changed)
        b_add = QPushButton(icon("plus"), tr("Insert point"))
        b_add.setToolTip(tr("A new point halfway along the leg after the selected point (at the end: the last leg "
                            "carried on)"))
        b_add.clicked.connect(self.insert_design_point)
        b_rm = QPushButton(icon("delete-outline"), tr("Delete point"))
        b_rm.setToolTip(tr("Delete the selected point (2 are needed at least)"))
        b_rm.clicked.connect(self.delete_design_point)
        row = QHBoxLayout()
        row.addWidget(b_add)
        row.addWidget(b_rm)
        row.addStretch(1)
        hint = QLabel(tr("A designed pit is a plan, not a survey: it is kept apart from the surveyed pits and "
                         "nothing is computed from it. Save writes the changes to the database; the 3D view "
                         "follows."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(self.d_title)
        v.addLayout(head)
        v.addWidget(self.d_info)
        v.addWidget(self.d_table, 1)
        v.addLayout(row)
        v.addWidget(hint)
        return page

    def _design(self):
        return next((x for x in self.designs if x.id == self.did), None)

    def _show_design(self, did) -> None:
        x = next((q for q in self.designs if q.id == did), None)
        if x is None:
            self._show_pit(None)
            return
        self.pid, self.did, self.rows, self.cur = None, did, [], -1
        self.mid_stack.setCurrentIndex(1)
        self.b_save.setText(tr("Save designed pit"))
        self._filling = True
        self.d_title.setText(tr("Designed pit {name}", name=x.name))
        self.d_name.setText(x.name)
        self.d_note.setText(x.note)
        self.d_w.setValue(x.profile_width)
        self.d_h.setValue(x.profile_height)
        self.d_level.setCurrentIndex(max(self.d_level.findData(x.station_level), 0))
        t = self.d_table
        t.blockSignals(True)
        t.setRowCount(len(x.points))
        for k, p in enumerate(x.points):
            for c, val in enumerate((p[0], p[1], p[2]), start=1):
                it = QTableWidgetItem(f"{val:.3f}")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(k, c, it)
        t.blockSignals(False)
        self._filling = False
        self._design_numbers()
        self._set_dirty(False)
        self._recompute(refit=True)

    def _d_points(self) -> list:
        """The points as typed in the table; ValueError naming the first one that is not a number."""
        out = []
        for k in range(self.d_table.rowCount()):
            vals = []
            for c in (1, 2, 3):
                it = self.d_table.item(k, c)
                v = _fnum(it.text() if it is not None else "")
                if v is None:
                    raise ValueError(tr("Point {n}: X, Y and Z must be numbers", n=k + 1))
                vals.append(v)
            out.append(tuple(vals))
        return out

    def _design_numbers(self) -> None:
        """Point numbers, leg lengths and gradients, and the summary above the table, from the points as typed."""
        from ..ai.route_planner import ratio_text
        from ..io.survey_store import design_measures

        t = self.d_table
        t.blockSignals(True)
        good = []
        for k in range(t.rowCount()):
            vals = [_fnum(t.item(k, c).text()) if t.item(k, c) is not None else None for c in (1, 2, 3)]
            good.append(None if None in vals else tuple(vals))
            for c, text in ((0, str(k + 1)), (4, ""), (5, "")):
                it = QTableWidgetItem(text)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(k, c, it)
            for c in (1, 2, 3):
                it = t.item(k, c)
                if it is not None:
                    it.setForeground(QColor(theme.TEXT) if _fnum(it.text()) is not None else QColor("#ff6b6b"))
            if k and good[k] is not None and good[k - 1] is not None:
                a, b = good[k - 1], good[k]
                plan = math.hypot(b[0] - a[0], b[1] - a[1])
                t.item(k, 4).setText(f"{math.sqrt(plan ** 2 + (b[2] - a[2]) ** 2):,.1f}")
                t.item(k, 5).setText(ratio_text((b[2] - a[2]) / plan) if plan > 1e-9 else tr("vertical"))
        t.blockSignals(False)
        x = self._design()
        if x is None:
            return
        pit = next((p for p in self.pits if p.id == x.start_pit_id), None)
        lines = [tr("From {a}{pit} to {b}", a=x.start_name or "-", pit=f" ({pit.name})" if pit else "",
                    b=x.target_name or "-")]
        if None in good or len(good) < 2:
            lines.append("<span style='color:#ff6b6b'>⚠ " + tr("Every point needs X, Y and Z (2 points at least)")
                         + "</span>")
        else:
            length, climb, steepest = design_measures(good)
            text = tr("{m:,.1f} m long, height change {h:+,.1f} m, steepest {g}", m=length, h=climb,
                      g=ratio_text(steepest))
            if x.gradient and steepest > x.gradient * 1.001:
                text += " - <span style='color:#f0a24a'>" + tr("steeper than the {g} it was designed for",
                                                                g=ratio_text(x.gradient)) + "</span>"
            lines.append(text)
        if x.created_at:
            lines.append(tr("Saved from the route designer {at}", at=x.created_at.replace("T", " ")))
        self.d_info.setText("<br>".join(lines))

    def _design_edited(self, *_a) -> None:
        if not self._filling and self.did is not None:
            self._set_dirty(True)

    def _design_point_changed(self, item) -> None:
        if self._filling or item.column() not in (1, 2, 3):
            return
        self._set_dirty(True)
        self._design_numbers()
        self._redraw_map()

    def insert_design_point(self) -> None:
        try:
            pts = self._d_points()
        except ValueError as e:
            QMessageBox.information(self, tr("Insert point"), str(e))
            return
        if not pts:
            return
        k = self.d_table.currentRow()
        k = len(pts) - 1 if k < 0 else k
        if k < len(pts) - 1:
            new = tuple((a + b) / 2.0 for a, b in zip(pts[k], pts[k + 1]))
        else:
            prev = pts[k - 1] if k else (pts[k][0] - 10.0, pts[k][1], pts[k][2])
            new = tuple(2 * b - a for a, b in zip(prev, pts[k]))
        self._set_design_points(pts[:k + 1] + [new] + pts[k + 1:], k + 1)

    def delete_design_point(self) -> None:
        k = self.d_table.currentRow()
        if k < 0:
            return
        if self.d_table.rowCount() <= 2:
            QMessageBox.information(self, tr("Delete point"), tr("A designed pit needs at least 2 points"))
            return
        self.d_table.removeRow(k)
        self._set_dirty(True)
        self._design_numbers()
        self._redraw_map()
        self.d_table.selectRow(min(k, self.d_table.rowCount() - 1))

    def _set_design_points(self, pts, select=None) -> None:
        t = self.d_table
        t.blockSignals(True)
        t.setRowCount(len(pts))
        for k, p in enumerate(pts):
            for c, val in enumerate(p, start=1):
                it = QTableWidgetItem(f"{val:.3f}")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(k, c, it)
        t.blockSignals(False)
        self._set_dirty(True)
        self._design_numbers()
        self._redraw_map()
        if select is not None:
            t.selectRow(select)

    def _save_design(self) -> bool:
        did = self.did
        try:
            self.store.update_design(did, self.d_name.text(), self.d_note.text(),
                                     (self.d_w.value(), self.d_h.value(), self.d_level.currentData()),
                                     self._d_points())
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Save designed pit"), tr("The designed pit was not saved:\n{e}", e=e))
            return False
        self.dirty = False
        self._reload(select=("d", did))
        self._show_design(did)
        self.status.setText(tr("Saved"))
        self.changed.emit()
        return True

    def delete_design(self) -> None:
        x = self._design()
        if x is None:
            return
        if QMessageBox.question(self, tr("Delete designed pit"),
                                tr("Delete the designed pit {name} from the database?\n"
                                   "The surveyed pits are not changed.", name=x.name)) != QMessageBox.Yes:
            return
        try:
            self.store.delete_design(x.id)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Delete designed pit"), str(e))
            return
        self.dirty = False
        self.did = None
        self._reload(select=self.pits[0].id if self.pits else None)
        if self.did is None and self.mid_stack.currentIndex() == 1:
            self._show_pit(self.pits[0].id if self.pits else None)
        self.status.setText(tr("Deleted {name}", name=x.name))
        self.changed.emit()

    def closeEvent(self, e) -> None:
        if not self._leave():
            e.ignore()
            return
        super().closeEvent(e)

