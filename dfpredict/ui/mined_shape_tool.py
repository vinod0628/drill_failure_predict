"""Mined outline tool: draw the outline of a mined-out area in a 2D view.

Same clicks as the cross-section tool: left clicks add corners (dragging still pans /
zooms), a right click or Enter closes the outline, Backspace removes the last corner,
Esc cancels. The closed outline is handed on (``outline_done``) for the From / To and
the date to be entered.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal

from ..i18n import N_, tr, trn
from ..scene.section import remove_duplicate_points
from .section_tool import SectionTool

PLANE_OF_AXIS = {2: "xy", 1: "xz", 0: "yz"}


def polygon_area(p2: np.ndarray) -> float:
    x, y = p2[:, 0], p2[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0)


class MinedShapeTool(SectionTool):
    outline_done = Signal(str, object)     # plane ("xy" / "xz" / "yz"), outline (K, 2) world coordinates

    HINT = N_("Mined outline: click the corners of the mined area, then right-click or Enter "
              "to close it (Backspace removes the last corner, Esc cancels)")     # shown as tr(HINT)

    def clear(self) -> None:
        was = self.drawing
        self.points, self.done, self.axis = [], False, None
        self._preview = None
        self.scene.clear_outline()
        if was:
            self.drawing_changed.emit(False)

    def add_point(self, x: float, y: float) -> None:
        super().add_point(x, y)
        n = len(self.points)
        if n:
            if n >= 3:
                msg = trn("Mined outline: {n} corner, {area:,.0f} m² - click the next corner, right-click or Enter "
                          "to close",
                          "Mined outline: {n} corners, {area:,.0f} m² - click the next corner, right-click or Enter "
                          "to close", n, area=polygon_area(self.points_2d()))
            else:
                msg = trn("Mined outline: {n} corner - click the next corner, right-click or Enter to close",
                          "Mined outline: {n} corners - click the next corner, right-click or Enter to close", n)
            self.status.emit(msg)

    def undo(self) -> None:
        if not self.drawing:
            return
        self.points.pop()
        if not self.points:
            self.clear()
            self.status.emit(tr(self.HINT))
            return
        self._redraw()
        self.status.emit(tr("Mined outline: {n} corner(s)", n=len(self.points)))

    def finish(self) -> bool:
        if not self.drawing:
            return False
        pts, _removed = remove_duplicate_points(self.points_2d())
        if len(pts) > 3 and np.linalg.norm(pts[0] - pts[-1]) < 0.05:
            pts = pts[:-1]                    # clicked the first corner again to close
        if len(pts) < 3 or polygon_area(pts) < 0.01:
            self.status.emit(tr("Mined outline: click at least three corners, then right-click or Enter"))
            return False
        plane = PLANE_OF_AXIS[self.axis]
        self.clear()
        self.outline_done.emit(plane, pts)
        return True

    def _redraw(self) -> None:
        local = [self.scene.to_local(p) for p in self.points]
        prev = None
        if self.drawing and self._preview is not None:
            prev = self.scene.view_plane_point(*self._preview)
        self.scene.set_outline(local, prev, closed=not self.drawing)
