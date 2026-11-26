"""Cross-section tool: draw the section line in a 2D view.

Left clicks add points to the line (dragging still pans / zooms); a thin line
follows the cursor from the last point. A right click or Enter finishes the
line: the points editor opens (coordinates can be typed in, points added or
removed) and its "Draw section" button opens the section. Backspace removes the
last point, Esc cancels. The finished line stays drawn until a new one is started. Only available in the
2D views (plan XY and the XZ / YZ sections).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal

from ..i18n import N_, tr, trn
from ..scene.section import DUPLICATE_TOL, SectionLine, remove_duplicate_points
from .identify import CLICK_TOLERANCE


class SectionTool(QObject):
    status = Signal(str)
    finished = Signal(object)          # SectionLine (scene-local), after the points were confirmed
    edit_requested = Signal()          # right click / Enter: the window shows the points editor
    drawing_changed = Signal(bool)     # True while a line is being drawn

    HINT = N_("Cross-section: click points along the section line, then right-click or Enter "
              "to show the section (Backspace removes the last point, Esc cancels)")     # shown as tr(HINT)

    def __init__(self, window, scene, plotter):
        super().__init__(window)
        self.scene = scene
        self.plotter = plotter
        self.view_widget = plotter.interactor
        self.active = False
        self.points: list[np.ndarray] = []   # world coordinates
        self.axis: int | None = None         # view axis the line was drawn in
        self.done = False                    # finished line (kept on screen)
        self._press = {}
        self._preview: tuple[int, int] | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(15)   # at most one preview render per frame
        self._preview_timer.timeout.connect(self._draw_preview)
        iren = plotter.iren.interactor if plotter.iren is not None else None
        if iren is not None:
            for button in ("Left", "Right"):
                iren.AddObserver(f"{button}ButtonPressEvent", self._on_press, 10.0)
                iren.AddObserver(f"{button}ButtonReleaseEvent", self._on_release, 10.0)
            iren.AddObserver("MouseMoveEvent", self._on_move, 10.0)

    # --- state ----------------------------------------------------------------------
    @property
    def drawing(self) -> bool:
        return bool(self.points) and not self.done

    @property
    def showing(self) -> bool:
        return bool(self.points)

    def set_active(self, on: bool) -> None:
        self.active = on
        self.view_widget.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        if on:
            self.status.emit(tr(self.HINT))
        else:
            self.clear()

    def clear(self) -> None:
        was = self.drawing
        self.points, self.done, self.axis = [], False, None
        self._preview = None
        self.scene.clear_section_line()
        if was:
            self.drawing_changed.emit(False)

    def view_changed(self) -> None:
        """A different 2D view: a line drawn in another plane no longer applies."""
        if self.points and self.axis != self.scene.view_axis:
            self.clear()

    def line(self) -> SectionLine | None:
        if len(self.points) < 2 or self.axis is None:
            return None
        sl = SectionLine(np.array([self.scene.to_local(p) for p in self.points]), self.axis)
        return sl if sl.valid else None

    # --- editing --------------------------------------------------------------------
    def add_point(self, x: float, y: float) -> None:
        axis = self.scene.view_axis
        if axis is None or self.scene.origin is None:
            return
        if self.done or (self.points and axis != self.axis):
            self.points, self.done = [], False
        started = not self.points
        self.axis = axis
        new = self.scene.to_world(self.scene.view_plane_point(x, y))
        if self.points and self._same_spot(new, self.points[-1]):
            # a second click on the same spot (e.g. a double-click) adds no node
            self.status.emit(trn("Cross-section: same place as the last point - ignored ({n} point)",
                                 "Cross-section: same place as the last point - ignored ({n} points)",
                                 len(self.points)))
            return
        self.points.append(new)
        self._preview = (x, y)
        self._redraw()
        if started:
            self.drawing_changed.emit(True)
        n = len(self.points)
        if n > 1:
            self.status.emit(tr("Cross-section: {n} points, {length:,.1f} m - click the next point, "
                                "right-click or Enter to finish", n=n, length=self._length()))
        else:
            self.status.emit(tr("Cross-section: {n} point - click the next point, right-click or Enter to finish",
                                n=n))

    def _same_spot(self, a, b) -> bool:
        """Within the click tolerance on screen (in the view plane)."""
        p, q = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
        p[self.axis] = q[self.axis] = 0.0
        r = self.plotter.renderer
        h = max(r.GetSize()[1], 1)
        world_per_px = 2.0 * r.GetActiveCamera().GetParallelScale() / h
        return float(np.linalg.norm(p - q)) < max(CLICK_TOLERANCE * world_per_px, DUPLICATE_TOL)

    def remove_duplicates(self) -> int:
        """Drop nodes repeating the node before them; returns how many were removed."""
        if len(self.points) < 2 or self.axis is None:
            return 0
        kept, removed = remove_duplicate_points(self.points_2d())
        if removed:
            self.points = self._from_2d(kept)
            self._redraw()
        return removed

    def undo(self) -> None:
        if not self.drawing:
            return
        self.points.pop()
        if not self.points:
            self.clear()
            self.status.emit(tr(self.HINT))
            return
        self._redraw()
        self.status.emit(tr("Cross-section: {n} point(s)", n=len(self.points)))

    def finish(self) -> bool:
        """Right click / Enter: ask for the points editor (``edit_requested``)."""
        if not self.drawing:
            return False
        if self.line() is None:
            self.status.emit(tr("Cross-section: click at least two points, then right-click or Enter"))
            return False
        self._preview = None
        self._redraw()
        self.edit_requested.emit()
        return True

    # --- the points editor works in the two in-plane coordinates ------------------------
    @property
    def in_plane_axes(self) -> tuple[int, int]:
        return tuple(i for i in range(3) if i != self.axis)

    def points_2d(self) -> np.ndarray:
        i1, i2 = self.in_plane_axes
        return np.array([[p[i1], p[i2]] for p in self.points], dtype=np.float64).reshape(-1, 2)

    def _from_2d(self, pts2) -> list[np.ndarray]:
        i1, i2 = self.in_plane_axes
        depth = float(self.points[0][self.axis]) if self.points else 0.0
        out = []
        for a, b in np.asarray(pts2, dtype=np.float64).reshape(-1, 2):
            w = np.zeros(3)
            w[i1], w[i2], w[self.axis] = a, b, depth
            out.append(w)
        return out

    def preview_2d(self, pts2) -> None:
        """Show edited points in the view without changing the line yet."""
        local = [self.scene.to_local(p) for p in self._from_2d(pts2)]
        self.scene.set_section_line(local, None)

    def edit_cancelled(self) -> None:
        """Back to drawing with the clicked points."""
        self._redraw()
        self.status.emit(tr("Cross-section: {n} points - click more points, right-click or Enter to finish",
                            n=len(self.points)))

    def apply_2d(self, pts2) -> SectionLine | None:
        """Use the confirmed points: the line is finished and ``finished`` is emitted.
        Repeated nodes are removed first."""
        pts2, _removed = remove_duplicate_points(pts2)
        self.points = self._from_2d(pts2)
        line = self.line()
        if line is None:
            return None
        self.done = True
        self._preview = None
        self._redraw()
        self.drawing_changed.emit(False)
        self.status.emit(tr("Cross-section: {length:,.1f} m - click to draw a new line", length=line.length))
        self.finished.emit(line)
        return line

    def _length(self) -> float:
        p = np.array(self.points)
        p[:, self.axis] = 0.0
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def _redraw(self) -> None:
        local = [self.scene.to_local(p) for p in self.points]
        prev = None
        if self.drawing and self._preview is not None:
            prev = self.scene.view_plane_point(*self._preview)
        self.scene.set_section_line(local, prev)

    def _draw_preview(self) -> None:
        if self.drawing:
            self._redraw()

    # --- mouse ----------------------------------------------------------------------
    def _on_press(self, iren, event):
        self._press[event[:4]] = iren.GetEventPosition() if self.active else None

    def _on_release(self, iren, event):
        button = event[:4]       # "Left" / "Righ"
        press = self._press.pop(button, None)
        if not self.active or press is None or self.scene.view_axis is None:
            return
        x, y = iren.GetEventPosition()
        if abs(x - press[0]) > CLICK_TOLERANCE or abs(y - press[1]) > CLICK_TOLERANCE:
            return
        if button == "Left":
            QTimer.singleShot(0, lambda: self.add_point(x, y))
        else:
            QTimer.singleShot(0, self.finish)

    def _on_move(self, iren, _event):
        if self.active and self.drawing:
            self._preview = iren.GetEventPosition()
            if not self._preview_timer.isActive():
                self._preview_timer.start()
