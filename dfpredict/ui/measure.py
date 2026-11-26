"""Measure tool: click two points to get the distance and angles between them.

Points are taken on the data under the cursor (surfaces, block faces, strings,
drillholes) and snap to a nearby vertex (string point, triangle or block corner).
In the 2D views a click on empty space uses the view plane through the centre
of the view. A third click starts a new measurement.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from ..i18n import N_, tr
from ..scene.picking import _projector, pick
from ..scene.scene import VIEW_3D
from .identify import ClickTool

SNAP_PX = 8.0  # a vertex this close to the cursor (pixels) is used instead of the picked spot


def measure(a, b) -> dict:
    """Distance and angles from world point a to world point b.

    bearing: azimuth of the plan direction, clockwise from north (+Y), 0..360 degrees.
    inclination: angle above (+) or below (-) the horizontal, -90..90 degrees.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = b - a
    plan = float(math.hypot(d[0], d[1]))
    dist = float(np.linalg.norm(d))
    return {
        "distance": dist,
        "plan": plan,
        "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]),
        "bearing": (math.degrees(math.atan2(d[0], d[1])) % 360.0) if plan > 1e-9 else None,
        "inclination": math.degrees(math.atan2(d[2], plan)) if dist > 1e-9 else None,
        "gradient": (100.0 * d[2] / plan) if plan > 1e-9 else None,
    }


def _deg(v, signed=False) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}°" if signed else f"{v:.2f}°"


def measure_rows(a, b) -> tuple[dict, list[tuple[str, str | None]]]:
    m = measure(a, b)
    inc = m["inclination"]
    inc_txt = _deg(inc, signed=True)
    if inc is not None and abs(inc) > 0.005:
        inc_txt += "  " + (tr("(up)") if inc > 0 else tr("(down)"))
    rows = [
        (tr("Distance"), f"{m['distance']:,.3f} m"),
        (tr("Plan distance"), f"{m['plan']:,.3f} m"),
        ("ΔX", f"{m['dx']:+,.3f} m"),
        ("ΔY", f"{m['dy']:+,.3f} m"),
        ("ΔZ", f"{m['dz']:+,.3f} m"),
        (tr("Angles"), None),
        (tr("Bearing"), _deg(m["bearing"])),
        (tr("Inclination"), inc_txt),
        (tr("Gradient"), "-" if m["gradient"] is None else f"{m['gradient']:+,.2f} %"),
        (tr("Point A"), None),
        ("X", f"{a[0]:,.3f}"), ("Y", f"{a[1]:,.3f}"), ("Z", f"{a[2]:,.3f}"),
        (tr("Point B"), None),
        ("X", f"{b[0]:,.3f}"), ("Y", f"{b[1]:,.3f}"), ("Z", f"{b[2]:,.3f}"),
    ]
    return m, rows


class MeasureTool(ClickTool):
    HINT = N_("Measure: click the first point (drag still navigates, Esc to stop)")
    PANEL_ICON = "ruler"

    def __init__(self, window, scene, plotter):
        super().__init__(window, scene, plotter)
        self.points: list[np.ndarray] = []   # world coordinates

    def clear(self) -> None:
        self.panel.hide()
        self.points = []
        self.scene.clear_measure()

    @property
    def showing(self) -> bool:
        return bool(self.points) or self.panel.isVisible()

    # --- picking a point ----------------------------------------------------------------
    def point_at(self, x: int, y: int) -> tuple[np.ndarray | None, str]:
        """Scene-local point under display position (x, y) and what it lies on."""
        r = self.plotter.renderer
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            hit = pick(r, self.scene.layers, x, y)
        finally:
            QApplication.restoreOverrideCursor()
        if hit is not None:
            p = np.asarray(hit.local, dtype=np.float64)
            try:
                corners = np.asarray(hit.cell_points(), dtype=np.float64).reshape(-1, 3)
            except Exception:  # noqa: BLE001
                corners = np.zeros((0, 3))
            if len(corners):
                scr = _projector(r)(corners)
                dist = np.hypot(scr[:, 0] - x, scr[:, 1] - y)
                k = int(np.argmin(dist))
                if dist[k] <= SNAP_PX:
                    return corners[k], tr("{name} (vertex)", name=hit.layer.name)
            return p, hit.layer.name
        if self.scene.view != VIEW_3D:
            return self._on_view_plane(x, y), tr("view plane")
        return None, ""

    def _on_view_plane(self, x: int, y: int) -> np.ndarray:
        return self.scene.view_plane_point(x, y)

    # --- clicks -------------------------------------------------------------------------
    def clicked(self, x: int, y: int, global_pos: QPoint | None = None) -> bool:
        if self.scene.origin is None:
            self.status.emit(tr("Measure: open some data first"))
            return False
        local, where = self.point_at(x, y)
        if local is None:
            self.status.emit(tr("Measure: no data here - click on an object (or use a 2D view)"))
            return False
        world = self.scene.to_world(local)
        if len(self.points) >= 2:
            self.points = []
            self.panel.hide()
        self.points.append(world)
        self._show(global_pos, where)
        return True

    def _show(self, global_pos, where: str) -> None:
        local = [self.scene.to_local(p) for p in self.points]
        if len(self.points) == 1:
            self.scene.set_measure(local)
            self.status.emit(tr("Measure: first point on {where} - click the second point", where=where))
            return
        a, b = self.points
        m, rows = measure_rows(a, b)
        label = f"{m['distance']:,.2f} m"
        if m["bearing"] is not None or m["inclination"] is not None:
            label += "\n" + tr("Brg {brg}   Inc {inc}", brg=_deg(m["bearing"]), inc=_deg(m["inclination"], signed=True))
        self.scene.set_measure(local, label)
        self.panel.show_result(tr("Measurement"), "A → B", f"{m['distance']:,.3f} m", rows, global_pos)
        self.status.emit(tr("Measure: {d:,.3f} m, bearing {brg}, inclination {inc} - click to measure again",
                            d=m["distance"], brg=_deg(m["bearing"]), inc=_deg(m["inclination"], signed=True)))

    def layers_changed(self) -> None:
        if self.scene.origin is None:
            self.clear()
