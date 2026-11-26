"""Tool windows (Survey, Route designer, Reserve, ...) that step aside instead of minimising.

A minimised owned window on Windows becomes a small title-bar strip over the main window. Here a
tool window that is minimised - or that steps aside so the user can look at the 3D view (Show in the
3D view, Draw mined outline) - is hidden together with the windows of its group, keeping everything
it shows. The toolbar button of that tool brings the whole group back as it was."""

from __future__ import annotations

import shiboken6
from PySide6.QtCore import QEvent, QObject, Qt, QTimer

PARKED = "gv_parked"


class WindowParking(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._windows: list = []                 # [(window, group)] in the order they were watched

    def _alive(self) -> list:
        self._windows = [(w, g) for w, g in self._windows if shiboken6.isValid(w)]
        return self._windows

    def watch(self, window, group: str) -> None:
        """Minimising ``window`` hides its group; :meth:`restore` brings the group back."""
        if any(w is window for w, _g in self._alive()):
            return
        self._windows.append((window, group))
        window.setProperty(PARKED, False)
        window.installEventFilter(self)

    def group_of(self, window) -> str | None:
        return next((g for w, g in self._alive() if w is window), None)

    def eventFilter(self, obj, ev) -> bool:
        t = ev.type()
        if t == QEvent.WindowStateChange and obj.isWindow() and obj.windowState() & Qt.WindowMinimized:
            QTimer.singleShot(0, lambda o=obj: self._minimized(o))
        elif t in (QEvent.Show, QEvent.Close):
            obj.setProperty(PARKED, False)
        return False

    def _minimized(self, window) -> None:
        if not shiboken6.isValid(window):
            return
        window.setWindowState(window.windowState() & ~Qt.WindowMinimized)
        group = self.group_of(window)
        if group is not None:
            self.park(group, window)

    def park(self, group: str, also=None) -> None:
        """Hide the visible windows of ``group`` (their contents are kept); ``also``: the window that was
        just minimised (no longer counted as visible on some systems)."""
        # all first: hiding a window can hide the windows it owns (the route designer with the Survey)
        wins = [w for w, g in self._alive() if g == group and (w.isVisible() or w is also)]
        for w in wins:
            w.setProperty(PARKED, True)
        for w in wins:
            w.hide()
            w.setProperty(PARKED, True)

    def parked(self, group: str) -> list:
        return [w for w, g in self._alive() if g == group and w.property(PARKED)]

    def restore(self, group: str) -> bool:
        """Show the hidden windows of ``group`` again, as they were; True when there were any."""
        wins = self.parked(group)
        for w in wins:
            w.setProperty(PARKED, False)
            w.setWindowState(w.windowState() & ~Qt.WindowMinimized)
            w.show()
            w.raise_()
        if wins:
            wins[-1].activateWindow()
        return bool(wins)
