"""A colour map chooser that shows each colour map as its gradient, not its name ("jet").

The closed box paints the chosen gradient across its whole width, so it fits any sidebar
width; the list shows every colour map as a gradient bar (its name is the tooltip)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QComboBox, QStyle, QStyleOptionComboBox, QStylePainter

from . import theme

SWATCH = QSize(150, 14)          # gradient bars in the list


@lru_cache(maxsize=None)
def gradient_image(name: str, width: int = 256) -> QImage:
    """A width x 1 image of a matplotlib colour map (grey if it is unknown)."""
    try:
        import matplotlib

        cmap = matplotlib.colormaps[name]
        rgba = (np.asarray(cmap(np.linspace(0.0, 1.0, width)))[:, :4] * 255 + 0.5).astype(np.uint8)
    except Exception:  # noqa: BLE001 - unknown name: a plain grey bar
        rgba = np.full((width, 4), 128, dtype=np.uint8)
        rgba[:, 3] = 255
    rgba = np.ascontiguousarray(rgba.reshape(1, width, 4))
    return QImage(rgba.data, width, 1, width * 4, QImage.Format_RGBA8888).copy()


def gradient_pixmap(name: str, size: QSize = SWATCH) -> QPixmap:
    pm = QPixmap(size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.drawImage(QRect(0, 0, size.width(), size.height()), gradient_image(name))
    p.setPen(QColor(theme.BORDER))
    p.drawRect(0, 0, size.width() - 1, size.height() - 1)
    p.end()
    return pm


class ColormapCombo(QComboBox):
    def __init__(self, names, current: str | None = None, parent=None):
        super().__init__(parent)
        self.setIconSize(SWATCH)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(0)
        for n in names:
            self.addItem(QIcon(gradient_pixmap(n)), "", n)
            self.setItemData(self.count() - 1, n, Qt.ToolTipRole)
        self.view().setMinimumWidth(SWATCH.width() + 28)
        i = self.findData(current)
        self.setCurrentIndex(i if i >= 0 else 0)
        self.currentIndexChanged.connect(lambda _i: self.setToolTip(self.currentData() or ""))
        self.setToolTip(self.currentData() or "")

    def minimumSizeHint(self) -> QSize:          # narrow enough for the 240 px sidebar
        return QSize(80, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        return QSize(160, super().sizeHint().height())

    def paintEvent(self, _e) -> None:
        p = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = ""
        opt.currentIcon = QIcon()
        p.drawComplexControl(QStyle.CC_ComboBox, opt)
        r = self.style().subControlRect(QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self)
        r = r.adjusted(3, 5, -3, -5)
        if r.width() > 4 and r.height() > 2:
            p.drawImage(r, gradient_image(self.currentData() or ""))
            p.setPen(QColor(theme.BORDER))
            p.drawRect(r.adjusted(0, 0, -1, -1))
