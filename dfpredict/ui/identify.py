"""Identify tool: click an object in the view to see what it is.

A click (press and release without dragging) picks; dragging still rotates /
pans, so the view can be navigated while the tool is on. The picked object is
highlighted in the view and its information is shown in a floating panel
next to the cursor.
"""

from __future__ import annotations

import html

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import N_, tr
from ..scene.picking import pick
from . import theme
from .icons import icon

CLICK_TOLERANCE = 4  # pixels the mouse may move between press and release for a click

_FLAGS = Qt.Tool | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint | Qt.WindowDoesNotAcceptFocus


class IdentifyPanel(QWidget):
    """Floating result panel (a tool window, like the toolbar, so it shows over the GL view)."""

    closed = Signal()

    def __init__(self, host: QWidget, anchor: QWidget, icon_name: str = "information-outline"):
        super().__init__(host, _FLAGS)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.anchor = anchor
        self._offset: QPoint | None = None      # top-left relative to the anchor
        self._drag: QPoint | None = None
        self._rows: list[tuple[str, str | None]] = []
        self._title = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QFrame(self)
        frame.setObjectName("identifyPanel")
        outer.addWidget(frame)
        v = QVBoxLayout(frame)
        v.setContentsMargins(10, 6, 6, 8)
        v.setSpacing(4)

        self.header = QWidget(frame)
        self.header.setCursor(Qt.SizeAllCursor)
        h = QHBoxLayout(self.header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        ic = QLabel(self.header)
        ic.setPixmap(icon(icon_name, theme.ACCENT).pixmap(QSize(18, 18)))
        h.addWidget(ic)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.title_lbl = QLabel(self.header)
        self.title_lbl.setObjectName("identifyTitle")
        self.sub_lbl = QLabel(self.header)
        self.sub_lbl.setObjectName("identifySub")
        self.sub_lbl.setWordWrap(True)
        self.title_lbl.setWordWrap(True)
        titles.addWidget(self.title_lbl)
        titles.addWidget(self.sub_lbl)
        h.addLayout(titles, 1)
        b_copy = self._button("content-copy", tr("Copy to clipboard"))
        b_copy.clicked.connect(self.copy)
        b_close = self._button("close", tr("Close (Esc)"))
        b_close.clicked.connect(self.close_panel)
        h.addWidget(b_copy, 0, Qt.AlignTop)
        h.addWidget(b_close, 0, Qt.AlignTop)
        v.addWidget(self.header)

        self.body = QLabel(frame)
        self.body.setObjectName("identifyBody")
        self.body.setTextFormat(Qt.RichText)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll = QScrollArea(frame)
        self.scroll.setObjectName("identifyScroll")
        self.scroll.setWidget(self.body)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        v.addWidget(self.scroll, 1)

    def _button(self, name: str, tip: str) -> QToolButton:
        b = QToolButton(self.header)
        b.setObjectName("headerButton")
        b.setIcon(icon(name))
        b.setIconSize(QSize(16, 16))
        b.setToolTip(tip)
        b.setFocusPolicy(Qt.NoFocus)
        return b

    # --- content -----------------------------------------------------------------
    def show_result(self, layer_name: str, kind: str, title: str, rows, global_pos: QPoint | None) -> None:
        self._rows = rows
        self._title = f"{layer_name} - {title}"
        self.title_lbl.setText(html.escape(layer_name))
        self.sub_lbl.setText(html.escape(f"{kind}  ·  {title}"))
        parts = ["<table cellspacing='0' cellpadding='1' width='100%'>"]
        for key, val in rows:
            if val is None:
                parts.append(f"<tr><td colspan='2' style='padding-top:6px; color:{theme.ACCENT};"
                             f" font-weight:600;'>{html.escape(key)}</td></tr>")
            else:
                parts.append(f"<tr><td style='color:{theme.MUTED}; padding-right:12px;'>{html.escape(key)}</td>"
                             f"<td style='color:{theme.TEXT};'>{html.escape(val)}</td></tr>")
        parts.append("</table>")
        self.body.setText("".join(parts))
        self._fit_size()
        if global_pos is not None:
            local = self.anchor.mapFromGlobal(global_pos)
            self._offset = QPoint(local.x() + 18, local.y() + 14)
        elif self._offset is None:
            self._offset = QPoint(16, 16)
        self.reposition()
        self.show()
        self.raise_()

    def _fit_size(self) -> None:
        self.body.adjustSize()
        hint = self.body.sizeHint()
        max_h = max(160, int(self.anchor.height() * 0.6))
        header_w = max(self.title_lbl.sizeHint().width(), self.sub_lbl.sizeHint().width()) + 100
        w = min(max(hint.width() + 40, header_w, 260), 460)
        body_h = min(hint.height() + 4, max_h)
        self.scroll.setFixedHeight(body_h)
        self.setFixedWidth(w)
        self.adjustSize()

    def text(self) -> str:
        lines = [self._title]
        for key, val in self._rows:
            lines.append(f"[{key}]" if val is None else f"{key}\t{val}")
        return "\n".join(lines)

    def copy(self) -> None:
        QGuiApplication.clipboard().setText(self.text())

    def close_panel(self) -> None:
        self.hide()
        self.closed.emit()

    # --- placement -----------------------------------------------------------------
    def reposition(self) -> None:
        if self._offset is None:
            return
        aw, ah = self.anchor.width(), self.anchor.height()
        x = min(max(8, self._offset.x()), max(8, aw - self.width() - 8))
        y = min(max(8, self._offset.y()), max(8, ah - self.height() - 8))
        if self._offset.x() + self.width() > aw - 8 and self._offset.x() - self.width() - 36 > 8:
            x = self._offset.x() - self.width() - 36  # flip to the left of the cursor
        self.move(self.anchor.mapToGlobal(QPoint(x, y)))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.header.geometry().contains(
                self.header.parentWidget().mapFrom(self, e.position().toPoint())):
            self._drag = e.globalPosition().toPoint() - self.pos()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            g = e.globalPosition().toPoint() - self._drag
            self.move(g)
            self._offset = self.anchor.mapFromGlobal(g)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag = None
        super().mouseReleaseEvent(e)


class ClickTool(QObject):
    """A view tool driven by clicks (press + release without dragging); dragging still
    navigates. Subclasses implement ``clicked`` and ``clear``."""

    status = Signal(str)
    HINT = ""
    PANEL_ICON = "information-outline"

    def __init__(self, window, scene, plotter):
        super().__init__(window)
        self.scene = scene
        self.plotter = plotter
        self.view_widget = plotter.interactor
        self.active = False
        self.panel = IdentifyPanel(window, self.view_widget, self.PANEL_ICON)
        self.panel.closed.connect(self.clear)
        self._press: tuple[int, int] | None = None
        iren = plotter.iren.interactor if plotter.iren is not None else None
        if iren is not None:
            iren.AddObserver("LeftButtonPressEvent", self._on_press, 10.0)
            iren.AddObserver("LeftButtonReleaseEvent", self._on_release, 10.0)
        window.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Move, QEvent.Resize, QEvent.WindowStateChange) and self.panel.isVisible():
            QTimer.singleShot(0, self.panel.reposition)
        return False

    def set_active(self, on: bool) -> None:
        self.active = on
        self.view_widget.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        if on:
            self.status.emit(tr(self.HINT))
        else:
            self.clear()

    def clear(self) -> None:
        self.panel.hide()

    @property
    def showing(self) -> bool:
        return self.panel.isVisible()

    def clicked(self, x: int, y: int, global_pos: QPoint) -> None:
        raise NotImplementedError

    # --- mouse ---------------------------------------------------------------------------
    def _on_press(self, iren, _ev):
        self._press = iren.GetEventPosition() if self.active else None

    def _on_release(self, iren, _ev):
        if not self.active or self._press is None:
            return
        x, y = iren.GetEventPosition()
        px, py = self._press
        self._press = None
        if abs(x - px) <= CLICK_TOLERANCE and abs(y - py) <= CLICK_TOLERANCE:
            pos = QCursor.pos()
            QTimer.singleShot(0, lambda: self.clicked(x, y, pos))


class IdentifyTool(ClickTool):
    """Turns clicks in the view into picks while active."""

    identified = Signal(object, object)    # (layer, IdentifyResult)

    HINT = N_("Identify: click an object in the view (drag still navigates, Esc to stop)")

    def __init__(self, window, scene, plotter):
        super().__init__(window, scene, plotter)
        self.hit_layer = None

    def clear(self) -> None:
        self.panel.hide()
        self.hit_layer = None
        self.scene.clear_highlight()

    @property
    def showing(self) -> bool:
        """A result is shown: the panel, or just the highlight (drillholes, whose details
        are in the log window)."""
        return self.panel.isVisible() or self.hit_layer is not None

    def clicked(self, x, y, global_pos):
        self.identify_at(x, y, global_pos)

    def identify_at(self, x: int, y: int, global_pos: QPoint | None = None) -> bool:
        """Pick at display position (x, y) (VTK pixels, origin bottom-left)."""
        QApplication.setOverrideCursor(Qt.WaitCursor)  # ~0.1 s per million drawn faces
        try:
            hit = pick(self.plotter.renderer, self.scene.layers, x, y)
        finally:
            QApplication.restoreOverrideCursor()
        if hit is None:
            self.clear()
            self.status.emit(tr("Identify: nothing here"))
            return False
        try:
            res = hit.layer.identify(hit)
        except Exception as e:  # noqa: BLE001 - never let a pick crash the app
            self.clear()
            self.status.emit(tr("Identify failed: {error}", error=e))
            return False
        self.hit_layer = hit.layer
        marker = res.marker if res.marker is not None else hit.local
        self.scene.set_highlight(res.highlight, marker, fill=res.fill)
        if res.extra.get("hole_id"):
            self.panel.hide()            # a drillhole is shown in its own log window instead
        else:
            self.panel.show_result(hit.layer.name, tr(hit.layer.kind), res.title, res.rows, global_pos)
        self.status.emit(tr("Identify: {name} - {title}", name=hit.layer.name, title=res.title))
        self.last_hit = hit                    # e.g. the risk layer finds the zone of the picked sleeve
        self.identified.emit(hit.layer, res)
        return True

    def layers_changed(self) -> None:
        """Drop the result when its layer was removed, replaced or hidden."""
        lay = self.hit_layer
        if lay is not None and (lay not in self.scene.layers or not lay.visible):
            self.clear()
