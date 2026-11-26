"""Screen-fixed colour key (e.g. drillhole lithology codes) at the top left of a view.

A title line per key followed by one row per entry: a colour square and its
text. Laid out in pixels before every render, so it keeps its size when the
view is zoomed or resized; ``scale`` enlarges it for prints.
"""

from __future__ import annotations

import vtk

from ..i18n import apply_vtk_font


class LegendOverlay:
    def __init__(self, renderer: vtk.vtkRenderer, legends, scale: float = 1.0, text_color=(1.0, 1.0, 1.0),
                 background=(0.0, 0.0, 0.0), opacity: float = 0.7, origin=(64, 30)):
        """``legends``: [(title, [(text, rgb), ...]), ...]; ``origin``: pixels from the left and
        from the top edge (screen units, multiplied by ``scale``)."""
        self.renderer = renderer
        self.s = float(scale)
        self.origin = origin
        self.text_color = text_color
        self._key = None
        self._rows = []        # (text actor, rgb or None for a title)
        for title, entries in legends:
            self._rows.append((self._text(title, bold=True), None))
            for text, rgb in entries:
                self._rows.append((self._text(text), tuple(rgb)))

        self._box = vtk.vtkPolyData()
        self.box_actor = self._actor2d(self._box, background, opacity)
        self._swatches = vtk.vtkPolyData()
        self.swatch_actor = self._actor2d(self._swatches, None, 1.0)
        self.swatch_actor.GetMapper().SetColorModeToDirectScalars()
        self.swatch_actor.GetMapper().SetScalarModeToUseCellData()
        self._frame = vtk.vtkPolyData()
        self.frame_actor = self._actor2d(self._frame, text_color, 0.6)
        # Background and frame below the text and swatches.
        self._props = [self.box_actor, self.frame_actor, self.swatch_actor] + [t for t, _ in self._rows]
        for p in self._props:
            renderer.AddViewProp(p)
        self._tag = renderer.AddObserver("StartEvent", self._on_render)

    def _actor2d(self, poly, color, opacity) -> vtk.vtkActor2D:
        m = vtk.vtkPolyDataMapper2D()
        m.SetInputData(poly)
        a = vtk.vtkActor2D()
        a.SetMapper(m)
        if color is not None:
            m.ScalarVisibilityOff()
            a.GetProperty().SetColor(*color)
        a.GetProperty().SetOpacity(opacity)
        a.GetProperty().SetLineWidth(max(1.0, self.s))
        a.PickableOff()
        return a

    def _text(self, text: str, bold: bool = False) -> vtk.vtkTextActor:
        t = vtk.vtkTextActor()
        t.SetInput(text)
        tp = t.GetTextProperty()
        tp.SetFontSize(int(round(12 * self.s)))
        apply_vtk_font(tp)
        tp.SetBold(bold)
        tp.SetColor(*self.text_color)
        tp.SetJustificationToLeft()
        tp.SetVerticalJustificationToCentered()
        t.PickableOff()
        return t

    def remove(self) -> None:
        for p in self._props:
            self.renderer.RemoveViewProp(p)
        self.renderer.RemoveObserver(self._tag)

    @property
    def empty(self) -> bool:
        return not self._rows

    # ------------------------------------------------------------------------------------
    def _on_render(self, *_):
        if not self._rows:
            return
        w, h = self.renderer.GetSize()
        if (w, h) == self._key:
            return
        self._key = (w, h)
        s = self.s
        row_h = 18 * s
        sw = 11 * s
        pad = 6 * s
        x0 = self.origin[0] * s
        top = h - self.origin[1] * s
        bbox = [0.0] * 4
        text_w = 0.0
        for t, _ in self._rows:
            t.GetBoundingBox(self.renderer, bbox)
            text_w = max(text_w, bbox[1] - bbox[0])
        width = pad + sw + pad + text_w + pad
        height = pad * 2 + row_h * len(self._rows)

        def quad(pts, polys, x1, y1, x2, y2):
            k = pts.GetNumberOfPoints()
            for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                pts.InsertNextPoint(x, y, 0.0)
            polys.InsertNextCell(4, [k, k + 1, k + 2, k + 3])
            return k

        # background and frame
        pts, polys = vtk.vtkPoints(), vtk.vtkCellArray()
        quad(pts, polys, x0, top - height, x0 + width, top)
        self._box.SetPoints(pts)
        self._box.SetPolys(polys)
        self._box.Modified()
        fpts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        for x, y in ((x0, top - height), (x0 + width, top - height), (x0 + width, top), (x0, top)):
            fpts.InsertNextPoint(x, y, 0.0)
        lines.InsertNextCell(5, [0, 1, 2, 3, 0])
        self._frame.SetPoints(fpts)
        self._frame.SetLines(lines)
        self._frame.Modified()

        # rows
        spts, spolys = vtk.vtkPoints(), vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        for i, (t, rgb) in enumerate(self._rows):
            yc = top - pad - row_h * (i + 0.5)
            if rgb is None:                      # title: starts at the swatch column
                t.SetDisplayPosition(int(x0 + pad), int(yc))
                continue
            quad(spts, spolys, x0 + pad, yc - sw / 2, x0 + pad + sw, yc + sw / 2)
            colors.InsertNextTuple3(*[int(round(c * 255)) for c in rgb])
            t.SetDisplayPosition(int(x0 + pad + sw + pad), int(yc))
        self._swatches.SetPoints(spts)
        self._swatches.SetPolys(spolys)
        self._swatches.GetCellData().SetScalars(colors)
        self._swatches.Modified()
