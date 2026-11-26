"""Screen-fixed scale bar for parallel-projection views.

Before every render the length of a round distance (1, 2 or 5 x 10^k metres)
that fits in about a fifth of the viewport width is worked out from the
camera, and the bar is drawn as two halves, dark then white, with labels at
0, the middle and the end. Zooming changes the distance; panning does not move
the bar.
"""

from __future__ import annotations

import math

import numpy as np
import vtk

from ..i18n import apply_vtk_font

LIGHT, DARK = (0.95, 0.96, 0.97), (0.12, 0.13, 0.15)
OUTLINE = (0.95, 0.96, 0.97)
BAR_HEIGHT = 7
TARGET_FRACTION = 0.2      # of the viewport width
MIN_PIXELS = 60


def nice_length(max_length: float) -> float:
    """The largest 1, 2, 4 or 5 x 10^k not above ``max_length`` (... 100, 200, 400, 500, 1000 ...)."""
    if not (max_length > 0 and math.isfinite(max_length)):
        return 1.0
    mag = 10.0 ** math.floor(math.log10(max_length))
    for m in (5.0, 4.0, 2.0, 1.0):
        if m * mag <= max_length * (1 + 1e-12):
            return m * mag
    return mag


def scale_layout(world_per_pixel: float, width: int, min_pixels: float = MIN_PIXELS,
                 fraction: float = TARGET_FRACTION) -> tuple[float, float, int]:
    """(bar length in metres, its length in pixels, number of blocks)."""
    target = max(width * fraction, min_pixels) * world_per_pixel
    length = nice_length(target)
    blocks = 2                 # one dark and one white half
    return length, length / world_per_pixel, blocks


class ScaleBar:
    """Scale bar overlay drawn by ``renderer`` (bottom left by default)."""

    def __init__(self, renderer: vtk.vtkRenderer, position: str = "bottom-center", margin=(16, 30),
                 scale: float = 1.0, light=LIGHT, dark=DARK, outline=OUTLINE, text_color=LIGHT,
                 text_background=(0.0, 0.0, 0.0), text_background_opacity: float = 0.55,
                 box=None, box_padding: float = 6.0, box_outline=None, font_size: float = 12.0,
                 target_fraction: float = TARGET_FRACTION):
        """``scale``: size factor for the bar, fonts and margins (e.g. ~3 for a 300 dpi print).
        ``box``: colour of a rectangle drawn behind the whole bar and its labels (None: no box);
        ``box_padding``: its margin around them, in screen pixels; ``box_outline``: colour of a thin
        frame around that box (None: no frame). ``font_size``: label size in screen points."""
        self.renderer = renderer
        self.position = position
        self.s = float(scale)
        self.margin = (margin[0] * self.s, margin[1] * self.s)
        self.colors = (light, dark, outline)
        self.text_color = text_color
        self.text_background = text_background
        self.text_background_opacity = text_background_opacity
        self.box_padding = box_padding * self.s
        self.font_size = font_size
        self.target_fraction = target_fraction   # longest bar, as a fraction of the view width
        self.enabled = True
        self.length = 0.0
        self.pixels = 0.0
        self._key = None

        self.box_actor = None
        self._box = vtk.vtkPolyData()
        if box is not None:                      # added first: drawn behind the bar
            bm = vtk.vtkPolyDataMapper2D()
            bm.SetInputData(self._box)
            bm.ScalarVisibilityOff()
            self.box_actor = vtk.vtkActor2D()
            self.box_actor.SetMapper(bm)
            self.box_actor.GetProperty().SetColor(*box)
            self.box_actor.PickableOff()
            renderer.AddViewProp(self.box_actor)
        self.frame_actor = None
        self._frame = vtk.vtkPolyData()
        if box is not None and box_outline is not None:
            fm = vtk.vtkPolyDataMapper2D()
            fm.SetInputData(self._frame)
            fm.ScalarVisibilityOff()
            self.frame_actor = vtk.vtkActor2D()
            self.frame_actor.SetMapper(fm)
            self.frame_actor.GetProperty().SetColor(*box_outline)
            self.frame_actor.GetProperty().SetLineWidth(max(1.0, 0.6 * self.s))
            self.frame_actor.PickableOff()
            renderer.AddViewProp(self.frame_actor)

        self._poly = vtk.vtkPolyData()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self._poly)
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarModeToUseCellData()
        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetLineWidth(max(1.0, self.s))
        self.actor.PickableOff()
        renderer.AddViewProp(self.actor)

        self.labels = [self._text() for _ in range(3)]
        renderer.AddObserver("StartEvent", self._on_render)

    def _text(self) -> vtk.vtkTextActor:
        t = vtk.vtkTextActor()
        tp = t.GetTextProperty()
        tp.SetFontSize(int(round(self.font_size * self.s)))
        tp.SetColor(*self.text_color)
        apply_vtk_font(tp)
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToBottom()
        tp.SetBackgroundColor(*self.text_background)
        tp.SetBackgroundOpacity(self.text_background_opacity)
        t.PickableOff()
        self.renderer.AddViewProp(t)
        return t

    def set_colors(self, light, dark, outline, text_color, text_background) -> None:
        """Colours for a new background."""
        self.colors = (light, dark, outline)
        self.text_color = tuple(text_color)
        self.text_background = tuple(text_background)
        for t in self.labels:
            tp = t.GetTextProperty()
            tp.SetColor(*self.text_color)
            tp.SetBackgroundColor(*self.text_background)
        self._key = None

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self._key = None
        if not on:
            self._show(False)

    def _show(self, on: bool) -> None:
        self.actor.SetVisibility(on)
        if self.box_actor is not None:
            self.box_actor.SetVisibility(on)
        if self.frame_actor is not None:
            self.frame_actor.SetVisibility(on)
        for t in self.labels:
            t.SetVisibility(on)

    def _on_render(self, *_):
        cam = self.renderer.GetActiveCamera()
        w, h = self.renderer.GetSize()
        if not self.enabled or w < 2 or h < 2 or not cam.GetParallelProjection():
            self._show(False)
            return
        key = (cam.GetParallelScale(), w, h)
        if key == self._key:
            return
        self._key = key
        self._update(2.0 * cam.GetParallelScale() / h, w)

    def _update(self, wpp: float, w: int) -> None:
        length, px, blocks = scale_layout(wpp, w, MIN_PIXELS * self.s, self.target_fraction)
        light, dark, outline = self.colors
        self.length, self.pixels = length, px
        mx, my = self.margin
        x0 = (w - px) / 2.0 if self.position == "bottom-center" else float(mx)
        y0, y1 = float(my), float(my + BAR_HEIGHT * self.s)
        xs = x0 + np.linspace(0.0, px, blocks + 1)

        pts = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        for i in range(blocks):
            k = pts.GetNumberOfPoints()
            for x, y in ((xs[i], y0), (xs[i + 1], y0), (xs[i + 1], y1), (xs[i], y1)):
                pts.InsertNextPoint(x, y, 0.0)
            polys.InsertNextCell(4, [k, k + 1, k + 2, k + 3])
        k = pts.GetNumberOfPoints()
        for x, y in ((x0, y0), (x0 + px, y0), (x0 + px, y1), (x0, y1)):
            pts.InsertNextPoint(x, y, 0.0)
        lines.InsertNextCell(5, [k, k + 1, k + 2, k + 3, k])
        for tick in (0, blocks // 2 if blocks % 2 == 0 else None, blocks):
            if tick is None:
                continue
            k = pts.GetNumberOfPoints()
            pts.InsertNextPoint(xs[tick], y0, 0.0)               # ticks point up, towards
            pts.InsertNextPoint(xs[tick], y1 + 4 * self.s, 0.0)  # their labels
            lines.InsertNextCell(2, [k, k + 1])
        # Cell colours in VTK's cell order for poly data: verts, lines, then polys.
        for _ in range(lines.GetNumberOfCells()):
            colors.InsertNextTuple3(*[int(c * 255) for c in outline])
        for i in range(blocks):
            colors.InsertNextTuple3(*[int(c * 255) for c in (dark if i % 2 == 0 else light)])
        self._poly.SetPoints(pts)
        self._poly.SetPolys(polys)
        self._poly.SetLines(lines)
        self._poly.GetCellData().SetScalars(colors)
        self._poly.Modified()

        div, unit = (1000.0, "km") if length >= 1000 else (1.0, "m")
        mid = "" if blocks % 2 else f"{length / 2.0 / div:,.6g}"   # 5 blocks: no middle tick
        texts = [("0", xs[0]), (mid, x0 + px / 2.0), (f"{length / div:,.6g} {unit}", xs[-1])]
        for t, (txt, x) in zip(self.labels, texts):
            t.SetInput(txt)
            t.SetDisplayPosition(int(round(x)), int(y1 + 6 * self.s))
            t.SetVisibility(bool(txt))
        self.actor.SetVisibility(True)
        if self.box_actor is not None:
            self._update_box(x0, px, y0)

    def _update_box(self, x0: float, px: float, y_bottom: float) -> None:
        """Rectangle around the bar and its visible labels, plus padding."""
        lo_x, hi_x, lo_y, hi_y = x0, x0 + px, y_bottom, y_bottom
        # Text extents from the text renderer (the actors' own bounding boxes are empty
        # until the text has been drawn once).
        tr = vtk.vtkTextRenderer.GetInstance()
        rw = self.renderer.GetRenderWindow()
        dpi = rw.GetDPI() if rw is not None else 72
        bb = [0, 0, 0, 0]
        for t in self.labels:
            if t.GetVisibility() and t.GetInput():
                if not tr.GetBoundingBox(t.GetTextProperty(), t.GetInput(), bb, dpi):
                    continue
                x, y = t.GetPosition()
                lo_x, hi_x = min(lo_x, x + bb[0]), max(hi_x, x + bb[1])
                lo_y, hi_y = min(lo_y, y + bb[2]), max(hi_y, y + bb[3])
        p = self.box_padding
        pts = vtk.vtkPoints()
        for x, y in ((lo_x - p, lo_y - p), (hi_x + p, lo_y - p), (hi_x + p, hi_y + p), (lo_x - p, hi_y + p)):
            pts.InsertNextPoint(x, y, 0.0)
        polys = vtk.vtkCellArray()
        polys.InsertNextCell(4, [0, 1, 2, 3])
        self._box.SetPoints(pts)
        self._box.SetPolys(polys)
        self._box.Modified()
        self.box_actor.SetVisibility(True)
        if self.frame_actor is not None:
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(5, [0, 1, 2, 3, 0])
            self._frame.SetPoints(pts)
            self._frame.SetLines(lines)
            self._frame.Modified()
            self.frame_actor.SetVisibility(True)
