"""Floating, draggable toolbar over the 3D view, plus a round button that brings it back.

The 3D view is a native OpenGL window, and ordinary child widgets drawn over
it can disappear or lose transparency on Windows. Both widgets here are
therefore small frameless, translucent tool windows owned by the main window.
They follow the view when the main window moves, resizes or changes state.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QAction, QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from ..i18n import tr
from . import theme
from .icons import icon

_FLAGS = (
    Qt.Tool
    | Qt.FramelessWindowHint
    | Qt.WindowDoesNotAcceptFocus
    | Qt.NoDropShadowWindowHint
)
MARGIN = 12
SLIDE = 48
DURATION = 220


def _setup_overlay(w: QWidget) -> None:
    w.setAttribute(Qt.WA_TranslucentBackground)
    w.setAttribute(Qt.WA_ShowWithoutActivating)
    w.setAttribute(Qt.WA_AlwaysShowToolTips)


def _fade_slide(w: QWidget, start: QPoint, end: QPoint, o0: float, o1: float) -> QParallelAnimationGroup:
    group = QParallelAnimationGroup(w)
    a = QPropertyAnimation(w, b"pos", group)
    a.setStartValue(start)
    a.setEndValue(end)
    a.setDuration(DURATION)
    a.setEasingCurve(QEasingCurve.OutCubic)
    b = QPropertyAnimation(w, b"windowOpacity", group)
    b.setStartValue(o0)
    b.setEndValue(o1)
    b.setDuration(DURATION)
    b.setEasingCurve(QEasingCurve.OutCubic)
    group.addAnimation(a)
    group.addAnimation(b)
    return group


class FloatingToolbar(QWidget):
    hidden_by_user = Signal()

    def __init__(self, host: QWidget, anchor: QWidget):
        super().__init__(host, _FLAGS)
        _setup_overlay(self)
        self.anchor = anchor
        self._from_right = MARGIN   # distance from the view's right edge
        self._top = MARGIN          # distance from the view's top edge
        self._drag_offset: QPoint | None = None
        self._anim = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self)
        self.frame.setObjectName("floatingBar")
        outer.addWidget(self.frame)
        self.lay = QVBoxLayout(self.frame)
        self.lay.setContentsMargins(5, 4, 5, 6)
        self.lay.setSpacing(2)
        self.lay.setAlignment(Qt.AlignHCenter)

        self.grip = QLabel(self.frame)
        self.grip.setObjectName("barGrip")
        self.grip.setPixmap(icon("drag-horizontal-variant", theme.MUTED).pixmap(QSize(20, 20)))
        self.grip.setAlignment(Qt.AlignCenter)
        self.grip.setCursor(Qt.SizeAllCursor)
        self.grip.setToolTip(tr("Drag to move the toolbar"))
        self.lay.addWidget(self.grip)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(2)
        self.grid.setVerticalSpacing(2)
        self.lay.addLayout(self.grid)
        self._items: list[tuple[str, QWidget]] = []
        self._columns = 0
        self._one_column_height: int | None = None

    # --- content ---------------------------------------------------------------
    def add_action(self, action: QAction) -> QToolButton:
        b = QToolButton(self.frame)
        b.setDefaultAction(action)
        b.setIconSize(QSize(22, 22))
        b.setFocusPolicy(Qt.NoFocus)
        b.setAutoRaise(True)
        self._items.append(("button", b))
        self._columns = 0
        self._one_column_height = None
        return b

    def add_separator(self) -> None:
        s = QFrame(self.frame)
        s.setObjectName("barSeparator")
        self._items.append(("separator", s))
        self._columns = 0
        self._one_column_height = None

    def _relayout(self, columns: int) -> None:
        """Lay the buttons out in 1 column, or 2 when the view is too short for one."""
        if columns == self._columns:
            return
        self._columns = columns
        while self.grid.count():
            self.grid.takeAt(0)
        row = col = 0
        for kind, w in self._items:
            if kind == "separator":
                if col:
                    row, col = row + 1, 0
                w.setFixedWidth(28 if columns == 1 else 64)
                self.grid.addWidget(w, row, 0, 1, columns, Qt.AlignHCenter)
                row += 1
            else:
                self.grid.addWidget(w, row, col, Qt.AlignHCenter)
                col += 1
                if col == columns:
                    row, col = row + 1, 0
        self._shrink_to_contents()

    def _shrink_to_contents(self) -> None:
        # A top-level window keeps its old size unless the layouts are re-activated first.
        self.grid.invalidate()
        self.lay.activate()
        self.layout().activate()
        self.resize(self.sizeHint())

    def _fit_columns(self, available: int) -> None:
        if self._one_column_height is None:
            self._relayout(1)
            self._one_column_height = self.sizeHint().height()
        self._relayout(1 if self._one_column_height <= available else 2)

    # --- placement -------------------------------------------------------------
    def target_pos(self) -> QPoint:
        self._fit_columns(self.anchor.height() - 2 * MARGIN)
        if self.size() != self.sizeHint():
            self._shrink_to_contents()
        aw, ah = self.anchor.width(), self.anchor.height()
        w, h = self.width(), self.height()
        x = aw - w - self._from_right
        y = self._top
        x = max(MARGIN, min(x, aw - w - MARGIN))
        y = max(MARGIN, min(y, max(MARGIN, ah - h - MARGIN)))
        return self.anchor.mapToGlobal(QPoint(x, y))

    def reposition(self) -> None:
        if self._anim is None:
            self.move(self.target_pos())

    # --- dragging ----------------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.grip.geometry().contains(
            self.frame.mapFrom(self, e.position().toPoint())
        ):
            self._drag_offset = e.globalPosition().toPoint() - self.pos()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag_offset is not None:
            g = e.globalPosition().toPoint() - self._drag_offset
            local = self.anchor.mapFromGlobal(g)
            self._from_right = self.anchor.width() - local.x() - self.width()
            self._top = local.y()
            self.move(self.target_pos())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag_offset = None
        super().mouseReleaseEvent(e)

    # --- animation ---------------------------------------------------------------
    def show_animated(self) -> None:
        if self._anim is not None:
            self._anim.stop()
            self._anim = None
        end = self.target_pos()
        start = end + QPoint(SLIDE, 0)
        self.setWindowOpacity(0.0)
        self.move(start)
        self.show()
        self.raise_()
        self._run(_fade_slide(self, start, end, 0.0, 1.0), on_done=None)

    def hide_animated(self) -> None:
        if not self.isVisible() or self._anim is not None:
            return
        start = self.pos()
        self._run(_fade_slide(self, start, start + QPoint(SLIDE, 0), self.windowOpacity(), 0.0),
                  on_done=self._after_hide)

    def _after_hide(self):
        self.hide()
        self.setWindowOpacity(1.0)
        self.hidden_by_user.emit()

    def _run(self, anim, on_done):
        self._anim = anim

        def finished():
            self._anim = None
            if on_done:
                on_done()

        anim.finished.connect(finished)
        anim.start()


class FloatingButton(QWidget):
    """Round button on the right edge of the view that brings the toolbar back."""

    clicked = Signal()
    SIZE = 46

    def __init__(self, host: QWidget, anchor: QWidget, tooltip: str):
        super().__init__(host, _FLAGS)
        _setup_overlay(self)
        self.anchor = anchor
        self.setFixedSize(self.SIZE + 8, self.SIZE + 8)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip)
        self._hover = False
        self._anim = None
        self._icon = icon("tools", "#ffffff")

    def target_pos(self) -> QPoint:
        x = self.anchor.width() - self.width() - MARGIN
        y = (self.anchor.height() - self.height()) // 2
        return self.anchor.mapToGlobal(QPoint(x, y))

    def reposition(self) -> None:
        if self._anim is None:
            self.move(self.target_pos())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRect(4, 4, self.SIZE, self.SIZE)
        col = QColor(theme.ACCENT)
        if not self._hover:
            col.setAlpha(215)
        p.setPen(QPen(QColor(255, 255, 255, 60), 1))
        p.setBrush(col)
        p.drawEllipse(r)
        s = 24
        self._icon.paint(p, QRect(r.center().x() - s // 2 + 1, r.center().y() - s // 2 + 1, s, s))
        p.end()

    def enterEvent(self, e):
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)

    def show_animated(self) -> None:
        end = self.target_pos()
        start = end + QPoint(SLIDE // 2, 0)
        self.setWindowOpacity(0.0)
        self.move(start)
        self.show()
        self.raise_()
        self._anim = _fade_slide(self, start, end, 0.0, 1.0)
        self._anim.finished.connect(lambda: setattr(self, "_anim", None))
        self._anim.start()

    def hide_animated(self) -> None:
        if not self.isVisible():
            return
        start = self.pos()
        self._anim = _fade_slide(self, start, start + QPoint(SLIDE // 2, 0), self.windowOpacity(), 0.0)

        def done():
            self._anim = None
            self.hide()
            self.setWindowOpacity(1.0)

        self._anim.finished.connect(done)
        self._anim.start()


class OverlayController(QObject):
    """Keeps the toolbar and round button placed over the view and toggles between them."""

    def __init__(self, host: QWidget, toolbar: FloatingToolbar, button: FloatingButton):
        super().__init__(host)
        self.host = host
        self.toolbar = toolbar
        self.button = button
        self.toolbar_shown = True
        host.installEventFilter(self)
        toolbar.anchor.installEventFilter(self)
        toolbar.hidden_by_user.connect(self.button.show_animated)
        button.clicked.connect(self.show_toolbar)

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.Move, QEvent.Resize, QEvent.WindowStateChange, QEvent.Show):
            self.reposition()
            if t == QEvent.Show:
                self._sync_visibility()
        return False

    def reposition(self):
        self.toolbar.reposition()
        self.button.reposition()

    def _sync_visibility(self):
        if self.toolbar_shown:
            self.toolbar.show()
            self.button.hide()
        else:
            self.toolbar.hide()
            self.button.show()

    def show_toolbar(self):
        if self.toolbar_shown and self.toolbar.isVisible():
            return
        self.toolbar_shown = True
        self.button.hide_animated()
        self.toolbar.show_animated()

    def hide_toolbar(self):
        if not self.toolbar_shown:
            return
        self.toolbar_shown = False
        self.toolbar.hide_animated()

    def toggle(self):
        if self.toolbar_shown:
            self.hide_toolbar()
        else:
            self.show_toolbar()
