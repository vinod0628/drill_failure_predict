"""Screen-fixed map grid for the 2D views (XY plan, XZ / YZ sections).

The grid frame is tied to the viewport, not to the data. Before every render
the visible world rectangle is computed from the camera, grid lines are placed
at round world coordinates inside it, and the labels on the bottom and left
edges show those coordinates. Panning moves the lines and changes the labels;
zooming also changes the interval. The grid is a 2D overlay drawn on top of
the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import vtk

from ..i18n import N_, apply_vtk_font, tr
from .scene_math import nice_step

LINE_COLOR = (0.62, 0.66, 0.71)
LINE_OPACITY = 0.45
LABEL_COLOR = (0.82, 0.85, 0.88)
MIN_PIXELS = 110          # minimum spacing between grid lines on screen
AXIS_NAMES = (N_("X (East)"), N_("Y (North)"), N_("Z (Elevation)"))   # shown as tr(AXIS_NAMES[i])


@dataclass
class GridLayout:
    step: float
    u_values: np.ndarray   # world coordinates of vertical lines
    u_pixels: np.ndarray   # their x position in pixels
    v_values: np.ndarray   # world coordinates of horizontal lines
    v_pixels: np.ndarray   # their y position in pixels


def layout_grid(center_u: float, center_v: float, world_per_pixel: float, width: int, height: int,
                sign_u: float = 1.0, sign_v: float = 1.0, min_pixels: float = MIN_PIXELS) -> GridLayout:
    """Grid lines at round world values covering a ``width`` x ``height`` pixel viewport.

    ``center_u/v`` is the world coordinate at the viewport centre; ``sign_u/v``
    is +1 when the world axis increases to the right / upwards.
    """
    step = nice_step(world_per_pixel * min_pixels, 1)
    hw, hh = world_per_pixel * width / 2.0, world_per_pixel * height / 2.0

    def axis(center, half, sign, size):
        lo, hi = center - half, center + half
        first = math.ceil(lo / step - 1e-9)
        last = math.floor(hi / step + 1e-9)
        vals = np.arange(first, last + 1, dtype=np.float64) * step
        pix = size / 2.0 + sign * (vals - center) / world_per_pixel
        return vals, pix

    uv, up = axis(center_u, hw, sign_u, width)
    vv, vp = axis(center_v, hh, sign_v, height)
    return GridLayout(step, uv, up, vv, vp)


def format_value(v: float, step: float) -> str:
    decimals = 0 if step >= 1 else min(6, int(math.ceil(-math.log10(step))))
    s = f"{v:,.{decimals}f}"
    return "0" if s in ("-0", "-0.0") else s


class MapGrid2D:
    """Viewport-fixed grid overlay with coordinate labels; active only in 2D views."""

    def __init__(self, scene, axis_names=None, corner=(0.12, 0.16), scale: float = 1.0,
                 line_color=LINE_COLOR, label_color=LABEL_COLOR, label_background=(0.0, 0.0, 0.0),
                 label_background_opacity: float = 0.55, show_labels: bool = True, show_lines: bool = True):
        """``scene`` needs ``plotter``, ``is_2d`` and ``origin`` (added to the labels).
        ``corner``: fraction of the width / height kept free of labels at the bottom left.
        ``scale``: size factor for fonts, spacing and offsets (e.g. ~3 for a 300 dpi print).
        ``show_labels`` False: lines only; the label positions are kept in ``layout`` (after a
        render) so they can be drawn elsewhere, e.g. outside the map frame of a print."""
        self.scene = scene
        self.axis_names = tuple(axis_names) if axis_names is not None else tuple(tr(n) for n in AXIS_NAMES)
        self.corner = corner
        self.s = float(scale)
        self.label_color = label_color
        self.label_background = label_background
        self.label_background_opacity = label_background_opacity
        self.show_labels = show_labels
        self.show_lines = show_lines
        self.layout: dict | None = None     # last grid: step, values and pixel positions, axes
        self.top_offset = 0                 # pixels kept free at the top left (the view caption)
        self.renderer = scene.plotter.renderer
        self.enabled = False
        self._key = None

        self._lines = vtk.vtkPolyData()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self._lines)
        self.line_actor = vtk.vtkActor2D()
        self.line_actor.SetMapper(mapper)
        prop = self.line_actor.GetProperty()
        prop.SetColor(*line_color)
        prop.SetOpacity(LINE_OPACITY)
        prop.SetLineWidth(max(1.0, 0.6 * self.s))
        self.line_actor.SetVisibility(False)
        self.line_actor.PickableOff()
        self.renderer.AddViewProp(self.line_actor)

        self.labels: list[vtk.vtkTextActor] = []
        self.titles = [self._text_actor(bold=True), self._text_actor(bold=True)]
        self.renderer.AddObserver("StartEvent", self._on_render)

    # ------------------------------------------------------------------------
    def _text_actor(self, bold=False) -> vtk.vtkTextActor:
        t = vtk.vtkTextActor()
        tp = t.GetTextProperty()
        tp.SetFontSize(int(round(12 * self.s)))
        tp.SetColor(*self.label_color)
        tp.SetBold(bold)
        apply_vtk_font(tp)
        tp.SetBackgroundColor(*self.label_background)
        tp.SetBackgroundOpacity(self.label_background_opacity)
        tp.SetFrame(False)
        t.SetVisibility(False)
        t.PickableOff()
        self.renderer.AddViewProp(t)
        return t

    def _label(self, i: int) -> vtk.vtkTextActor:
        while len(self.labels) <= i:
            self.labels.append(self._text_actor())
        return self.labels[i]

    def set_colors(self, label_color, line_color, label_background) -> None:
        """Colours for a new background (labels, their box, grid lines)."""
        self.label_color = tuple(label_color)
        self.label_background = tuple(label_background)
        self.line_actor.GetProperty().SetColor(*line_color)
        for t in self.labels + self.titles:
            tp = t.GetTextProperty()
            tp.SetColor(*self.label_color)
            tp.SetBackgroundColor(*self.label_background)
        self._key = None

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self._key = None
        if not on:
            self._hide()

    def _hide(self):
        self.line_actor.SetVisibility(False)
        for t in self.labels + self.titles:
            t.SetVisibility(False)

    # ------------------------------------------------------------------------
    def _axes(self, cam):
        """Indices and signs of the world axes pointing right / up on screen."""
        d = np.array(cam.GetDirectionOfProjection())
        up = np.array(cam.GetViewUp())
        right = np.cross(d, up)
        iu, iv = int(np.argmax(np.abs(right))), int(np.argmax(np.abs(up)))
        return iu, float(np.sign(right[iu]) or 1.0), iv, float(np.sign(up[iv]) or 1.0)

    def _on_render(self, *_):
        sc = self.scene
        if not self.enabled or not sc.is_2d or sc.origin is None:
            if self.line_actor.GetVisibility():
                self._hide()
            return
        cam = self.renderer.GetActiveCamera()
        w, h = self.renderer.GetSize()
        if w < 2 or h < 2 or not cam.GetParallelProjection():
            self._hide()
            return
        key = (cam.GetFocalPoint(), cam.GetParallelScale(), cam.GetViewUp(),
               cam.GetDirectionOfProjection(), w, h, tuple(sc.origin))
        if key == self._key:
            return
        self._key = key
        self._update(cam, w, h)

    def _update(self, cam, w: int, h: int) -> None:
        iu, su, iv, sv = self._axes(cam)
        wpp = 2.0 * cam.GetParallelScale() / h
        focal = np.array(cam.GetFocalPoint()) + self.scene.origin
        g = layout_grid(focal[iu], focal[iv], wpp, w, h, su, sv, MIN_PIXELS * self.s)

        # Lines (display coordinates).
        nu, nv = len(g.u_pixels), len(g.v_pixels)
        pts = np.zeros((2 * (nu + nv), 3))
        pts[0:2 * nu:2, 0] = g.u_pixels
        pts[1:2 * nu:2, 0] = g.u_pixels
        pts[1:2 * nu:2, 1] = h
        pts[2 * nu::2, 1] = g.v_pixels
        pts[2 * nu + 1::2, 1] = g.v_pixels
        pts[2 * nu + 1::2, 0] = w
        vpts = vtk.vtkPoints()
        vpts.SetNumberOfPoints(len(pts))
        for i, p in enumerate(pts):
            vpts.SetPoint(i, p)
        cells = vtk.vtkCellArray()
        for i in range(nu + nv):
            cells.InsertNextCell(2)
            cells.InsertCellPoint(2 * i)
            cells.InsertCellPoint(2 * i + 1)
        self._lines.SetPoints(vpts)
        self._lines.SetLines(cells)
        self._lines.Modified()
        self.line_actor.SetVisibility(self.show_lines)
        self.layout = {"step": g.step, "u_values": g.u_values, "u_pixels": g.u_pixels,
                       "v_values": g.v_values, "v_pixels": g.v_pixels, "iu": iu, "iv": iv,
                       "u_name": self.axis_names[iu], "v_name": self.axis_names[iv], "size": (w, h)}
        if not self.show_labels:
            for t in self.labels + self.titles:
                t.SetVisibility(False)
            return

        # Labels: keep clear of the orientation axes in the bottom-left corner.
        corner_w, corner_h = self.corner[0] * w, self.corner[1] * h
        s = self.s
        k = 0
        for val, px in zip(g.u_values, g.u_pixels):
            if px < corner_w or px > w - 40 * s:
                continue
            t = self._label(k)
            k += 1
            t.SetInput(format_value(val, g.step))
            tp = t.GetTextProperty()
            tp.SetJustificationToCentered()
            tp.SetVerticalJustificationToBottom()
            t.SetDisplayPosition(int(px), int(4 * s))
            t.SetVisibility(True)
        for val, py in zip(g.v_values, g.v_pixels):
            if py < corner_h or py > h - 20 * s - self.top_offset:
                continue
            t = self._label(k)
            k += 1
            t.SetInput(format_value(val, g.step))
            tp = t.GetTextProperty()
            tp.SetJustificationToLeft()
            tp.SetVerticalJustificationToCentered()
            t.SetDisplayPosition(int(4 * s), int(py))
            t.SetVisibility(True)
        for t in self.labels[k:]:
            t.SetVisibility(False)

        # Axis titles with the current interval.
        step_txt = format_value(g.step, g.step)
        tx, ty = self.titles
        tx.SetInput(tr("{axis}    grid {step} m", axis=self.axis_names[iu], step=step_txt))
        txp = tx.GetTextProperty()
        txp.SetJustificationToRight()
        txp.SetVerticalJustificationToBottom()
        tx.SetDisplayPosition(int(w - 8 * s), int(24 * s))
        tx.SetVisibility(True)
        ty.SetInput(self.axis_names[iv])
        typ = ty.GetTextProperty()
        typ.SetJustificationToLeft()
        typ.SetVerticalJustificationToTop()
        ty.SetDisplayPosition(int(4 * s), int(h - 6 * s - self.top_offset))
        ty.SetVisibility(True)
