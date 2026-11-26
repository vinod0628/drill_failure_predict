"""Icons: Material Design icons via qtawesome, plus small text icons (XY, XZ, ...)."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap

from . import theme

ICON_COLOR = theme.TEXT
ICON_ACTIVE = "#ffffff"
ICON_DISABLED = "#5f666f"


@lru_cache(maxsize=None)
def icon(name: str, color: str = ICON_COLOR) -> QIcon:
    import qtawesome as qta

    return qta.icon(
        f"mdi6.{name}", color=color, color_active=ICON_ACTIVE, color_disabled=ICON_DISABLED
    )


@lru_cache(maxsize=None)
def text_icon(text: str, color: str = ICON_COLOR, size: int = 64) -> QIcon:
    """A crisp icon showing short text inside a rounded frame (e.g. 'XY')."""
    ic = QIcon()
    for mode, col in ((QIcon.Normal, color), (QIcon.Disabled, ICON_DISABLED)):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen_col = QColor(col)
        p.setPen(pen_col)
        r = QRectF(size * 0.08, size * 0.16, size * 0.84, size * 0.68)
        pen = p.pen()
        pen.setWidthF(size * 0.06)
        p.setPen(pen)
        p.drawRoundedRect(r, size * 0.12, size * 0.12)
        f = QFont("Segoe UI")
        f.setBold(True)
        f.setPixelSize(int(size * 0.38))
        p.setFont(f)
        p.drawText(r, Qt.AlignCenter, text)
        p.end()
        ic.addPixmap(pm, mode)
    return ic


APP_ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def render_app_icon(size: int):
    """The GeoViewer icon at ``size`` pixels (QImage): a blue cube on a dark rounded square -
    the same design as the .exe icon (packaging/make_icon.py writes geoviewer.ico from this)."""
    from PySide6.QtGui import QImage

    import qtawesome as qta

    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#1f2329"))
    r = size * 0.06
    p.drawRoundedRect(QRectF(r, r, size - 2 * r, size - 2 * r), size * 0.2, size * 0.2)
    m = int(size * 0.16)
    qta.icon("mdi6.cube-scan", color="#3d9bff").paint(p, m, m, size - 2 * m, size - 2 * m)
    p.end()
    return img


@lru_cache(maxsize=None)
def app_icon() -> QIcon:
    """The application icon (window title bar, taskbar, Alt+Tab) at every usual size."""
    ic = QIcon()
    for s in APP_ICON_SIZES:
        ic.addPixmap(QPixmap.fromImage(render_app_icon(s)))
    return ic
