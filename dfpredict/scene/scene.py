"""Scene: the VTK renderer, the coordinate datum, layers, camera and navigation."""

from __future__ import annotations

import math

import numpy as np
import vtk

from .slicing import Slicer
from ..i18n import N_, apply_vtk_font, tr
from .layers import Layer
from .map_grid import MapGrid2D
from .scene_math import nice_grid, nice_step  # noqa: F401 - re-exported

VIEW_3D, VIEW_XY, VIEW_XZ, VIEW_YZ = "3d", "xy", "xz", "yz"

LIGHT_TEXT, DARK_TEXT = (0.93, 0.95, 0.97), (0.10, 0.11, 0.13)


def foreground_for(rgb) -> tuple[float, float, float]:
    """Text / line colour readable on a background: light on dark, dark on light."""
    r, g, b = (float(c) for c in rgb[:3])
    return LIGHT_TEXT if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else DARK_TEXT


def grid_line_for(rgb) -> tuple[float, float, float]:
    """Grid line colour on a background: a grey a little towards the text colour."""
    fg = foreground_for(rgb)
    return tuple(0.55 * f + 0.45 * float(c) for f, c in zip(fg, rgb[:3]))


class NavigationStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera with a switchable left-button action.

    * mode "rotate": left drag rotates (Shift+left pans), right drag / wheel zoom
    * mode "pan":    left drag pans the scene in the view plane; no mouse action rotates.
                     Used for the pan tool in 3D and always in the 2D XY / XZ / YZ views.
    * in every mode and view: hold the middle button and move to pan.
    Single-key VTK shortcuts (q/e quit, w/s wireframe, ...) are disabled.
    """

    def __init__(self):
        super().__init__()
        self.mode = "rotate"
        self._panning = False
        self._middle_panning = False
        self.AddObserver("LeftButtonPressEvent", self._left_down)
        self.AddObserver("LeftButtonReleaseEvent", self._left_up)
        self.AddObserver("MiddleButtonPressEvent", self._middle_down)
        self.AddObserver("MiddleButtonReleaseEvent", self._middle_up)
        # Right drag zooms. Handled here (not by the VTK default) because the default grabs
        # the interactor's focus, which hides the release from other observers (tools).
        self._zooming = False
        self.AddObserver("RightButtonPressEvent", self._right_down)
        self.AddObserver("RightButtonReleaseEvent", self._right_up)
        self.AddObserver("CharEvent", lambda o, e: None)
        self.AddObserver("KeyPressEvent", lambda o, e: None)

    def _middle_down(self, obj, event):
        x, y = self.GetInteractor().GetEventPosition()
        self.FindPokedRenderer(x, y)
        if self.GetCurrentRenderer() is None:
            return
        self._middle_panning = True
        self.StartPan()

    def _middle_up(self, obj, event):
        if self._middle_panning:
            self._middle_panning = False
            self.EndPan()

    def _right_down(self, obj, event):
        x, y = self.GetInteractor().GetEventPosition()
        self.FindPokedRenderer(x, y)
        if self.GetCurrentRenderer() is None:
            return
        self._zooming = True
        self.StartDolly()

    def _right_up(self, obj, event):
        if self._zooming:
            self._zooming = False
            self.EndDolly()

    def _left_down(self, obj, event):
        iren = self.GetInteractor()
        x, y = iren.GetEventPosition()
        self.FindPokedRenderer(x, y)
        if self.mode == "rotate" and not iren.GetShiftKey():
            self.StartRotate()
        else:
            self._panning = True
            self.StartPan()

    def _left_up(self, obj, event):
        if self._panning:
            self._panning = False
            self.EndPan()
        else:
            self.EndRotate()


class Scene:
    def __init__(self, plotter):
        self.plotter = plotter
        self.origin: np.ndarray | None = None
        self.layers: list[Layer] = []
        self.view = VIEW_3D
        self.grid_visible = False
        self.perspective = False   # 3D projection; the 2D views are always orthographic
        self.slab = None  # (point, normal, half_width) in world coords; set by the slice tool
        self.slicer = Slicer(self)  # 3D slice: objects shown only between two planes (slicing.py)
        self.background = (0.0, 0.0, 0.0)
        self.foreground = foreground_for(self.background)   # overlay text / labels

        p = plotter
        p.set_background("black")
        p.enable_parallel_projection()
        p.enable_depth_peeling(number_of_peels=6)
        self.style = NavigationStyle()
        self.style.SetDefaultRenderer(p.renderer)
        if p.iren is not None:  # None only for off-screen plotters (tests, exports)
            self.axes_actor = p.add_axes(color="white", line_width=2, viewport=(0.0, 0.0, 0.12, 0.16))
            # PyVista watches left double-clicks to toggle chart interaction, and on every
            # double-click it re-installs its own (rotating) style, replacing ours. We have
            # no charts, so stop that, and register our style through PyVista so any
            # internal style refresh keeps it.
            p.iren.untrack_click_position(side="left")
            p.iren.style = self.style
            # Last line of defence: re-attach our style if anything still swaps it.
            p.iren.interactor.AddObserver("LeftButtonPressEvent", self._ensure_style, 100.0)
            p.iren.interactor.AddObserver("MiddleButtonPressEvent", self._ensure_style, 100.0)
            p.iren.interactor.AddObserver("RightButtonPressEvent", self._ensure_style, 100.0)
        # Map grid: screen-fixed overlay in 2D views, box grid around the data in 3D.
        self.map_grid = MapGrid2D(self)
        # View caption in the top-left corner: "3D VIEW", "XY PLANE", ...
        self.caption_visible = True
        self.caption = vtk.vtkTextActor()
        tp = self.caption.GetTextProperty()
        apply_vtk_font(tp)
        tp.BoldOn()
        tp.SetColor(*self.foreground)
        tp.SetBackgroundOpacity(0.0)          # no box: the text itself is see-through
        tp.SetFrame(False)
        self.caption_size = 20
        self.caption_transparency = 40        # percent
        tp.SetJustificationToLeft()
        tp.SetVerticalJustificationToTop()
        self.caption.PickableOff()
        p.renderer.AddViewProp(self.caption)
        p.renderer.AddObserver("StartEvent", self._place_caption)
        self._update_caption()
        # Identify highlight: drawn by a top-layer renderer sharing the main camera, so
        # it stays visible through surfaces and blocks and never takes part in picking.
        self._overlay: vtk.vtkRenderer | None = None
        self._highlight: list = []
        self._measure: list = []
        self._section: list = []
        self._outline: list = []          # mined outline being drawn
        self._preview_solid: list = []    # mined shape being entered (see-through)

    CAPTIONS = {VIEW_3D: N_("3D VIEW"), VIEW_XY: N_("XY PLANE"), VIEW_XZ: N_("XZ PLANE"),   # shown: tr()
                VIEW_YZ: N_("YZ PLANE")}
    CAPTION_MARGIN = 10        # pixels from the top-left corner

    def set_background(self, rgb) -> None:
        """Background of the view; texts and lines drawn over it switch to a readable colour."""
        self.background = tuple(float(c) for c in rgb[:3])
        fg = self.foreground = foreground_for(self.background)
        self.plotter.set_background(self.background)
        self.caption.GetTextProperty().SetColor(*fg)
        self.map_grid.set_colors(fg, grid_line_for(self.background), self.background)
        axes = getattr(self, "axes_actor", None)
        if axes is not None:
            for cap in (axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(),
                        axes.GetZAxisCaptionActor2D()):
                cap.GetCaptionTextProperty().SetColor(*fg)
        for bar in list(getattr(self.plotter, "scalar_bars", {}).values()):
            bar.GetTitleTextProperty().SetColor(*fg)
            bar.GetLabelTextProperty().SetColor(*fg)
        for layer in self.layers:
            if hasattr(layer, "show_labels"):          # drillhole names: drawn in the text colour
                layer.rebuild()
        self.render()

    def set_caption_visible(self, on: bool) -> None:
        self.caption_visible = bool(on)
        self._update_caption()
        self.render()

    def set_caption_style(self, size: int, transparency: int) -> None:
        """Font size in points and transparency in percent (0 = solid)."""
        self.caption_size = int(size)
        self.caption_transparency = int(transparency)
        self._update_caption()
        self.render()

    def _update_caption(self) -> None:
        self.caption.SetInput(tr(self.CAPTIONS[self.view]) if self.view in self.CAPTIONS else "")
        self.caption.SetVisibility(self.caption_visible)
        tp = self.caption.GetTextProperty()
        tp.SetFontSize(self.caption_size)
        tp.SetOpacity(max(0.1, 1.0 - self.caption_transparency / 100.0))
        # the 2D grid's vertical axis title moves below the caption while it is shown
        self.map_grid.top_offset = int(self.caption_size * 1.5 + 8) if self.caption_visible else 0
        self.map_grid._key = None

    def _place_caption(self, *_):
        w, h = self.plotter.renderer.GetSize()
        self.caption.SetDisplayPosition(self.CAPTION_MARGIN, max(h - self.CAPTION_MARGIN, 0))

    def _ensure_style(self, iren, _event) -> None:
        if iren.GetInteractorStyle() is not self.style:
            iren.SetInteractorStyle(self.style)

    # --- layers -------------------------------------------------------------
    def add_layer(self, layer: Layer) -> None:
        if self.origin is None:
            b = layer.world_bounds()
            if b is not None:
                centre = (b[0] + b[1]) / 2.0
                self.origin = np.round(centre / 100.0) * 100.0
            else:
                self.origin = np.zeros(3)
        layer.build(self)
        self.layers.append(layer)
        if len(self.layers) == 1:
            self.set_view(VIEW_3D)
        self.refresh_grid()
        self.render()

    def remove_layer(self, layer: Layer) -> None:
        layer.remove_actors()
        if layer in self.layers:
            self.layers.remove(layer)
        self.refresh_grid()
        self.render()

    def replace_layer(self, old: Layer, new: Layer) -> None:
        """Swap a reloaded layer in, keeping its place in the layer order."""
        i = self.layers.index(old)
        old.remove_actors()
        new.build(self)
        self.layers[i] = new
        self.refresh_grid()
        self.render()

    def clear(self) -> None:
        """Remove every layer and forget the coordinate datum (a new workspace may be elsewhere)."""
        for layer in self.layers:
            layer.remove_actors()
        self.layers = []
        self.origin = None
        self.slab = None
        self.slicer.forget()
        self.clear_highlight(render=False)
        self.clear_measure(render=False)
        self.clear_section_line(render=False)
        self.clear_outline(render=False)
        self.clear_preview_solid(render=False)
        self.refresh_grid()
        self.render()

    def render(self) -> None:
        self.plotter.render()

    # --- coordinates ---------------------------------------------------------
    def to_world(self, local) -> np.ndarray:
        return np.asarray(local, dtype=np.float64) + (self.origin if self.origin is not None else 0.0)

    def to_local(self, world) -> np.ndarray:
        return np.asarray(world, dtype=np.float64) - (self.origin if self.origin is not None else 0.0)

    def world_bounds(self, visible_only: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        los, his = [], []
        for layer in self.layers:
            if visible_only and not layer.visible:
                continue
            b = layer.world_bounds()
            if b is not None:
                los.append(b[0])
                his.append(b[1])
        if not los:
            return None
        return np.min(los, axis=0), np.max(his, axis=0)

    def slab_mask(self, world_points: np.ndarray) -> np.ndarray:
        """Points inside the active slice slab (all True when no slice is active)."""
        if self.slab is None:
            return np.ones(len(world_points), dtype=bool)
        point, normal, half = self.slab
        d = (world_points - point) @ normal
        return np.abs(d) <= half

    # --- camera ---------------------------------------------------------------
    def _local_bounds(self):
        b = self.world_bounds()
        if b is None:
            return None
        return self.to_local(b[0]), self.to_local(b[1])

    def fit(self) -> None:
        lb = self._local_bounds()
        if lb is None:
            self.plotter.reset_camera()
        else:
            lo, hi = lb
            self.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        self.render()

    def set_view(self, view: str) -> None:
        """3D oblique view, or 2D plan (XY), section looking north (XZ) / west (YZ)."""
        self.view = view
        cam = self.plotter.camera
        lb = self._local_bounds()
        if lb is None:
            lo, hi = np.zeros(3) - 1, np.zeros(3) + 1
        else:
            lo, hi = lb
        c = (lo + hi) / 2.0
        d = float(np.linalg.norm(hi - lo)) or 10.0
        if view == VIEW_XY:
            pos, up = c + [0, 0, d], (0, 1, 0)
        elif view == VIEW_XZ:
            pos, up = c - [0, d, 0], (0, 0, 1)
        elif view == VIEW_YZ:
            pos, up = c + [d, 0, 0], (0, 0, 1)
        else:
            az, el = math.radians(210.0), math.radians(35.0)  # from the south-west, looking down
            v = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
            pos, up = c + d * v, (0, 0, 1)
        cam.focal_point = tuple(c)
        cam.position = tuple(pos)
        cam.up = up
        self.style.mode = "pan" if view != VIEW_3D else self.style.mode
        self._update_caption()
        self._apply_projection(keep_size=False)   # fit() below sizes the view
        self.refresh_grid()
        self.fit()  # renders, which also lays out the 2D map grid

    @property
    def is_2d(self) -> bool:
        return self.view != VIEW_3D

    @property
    def view_axis(self) -> int | None:
        """World axis the 2D view looks along (0 X, 1 Y, 2 Z); None in 3D."""
        return {VIEW_XY: 2, VIEW_XZ: 1, VIEW_YZ: 0}.get(self.view)

    def view_plane_point(self, x: float, y: float) -> np.ndarray:
        """Scene-local point under display position (x, y) on the plane through the focal point."""
        r = self.plotter.renderer
        fp = r.GetActiveCamera().GetFocalPoint()
        r.SetWorldPoint(*fp, 1.0)
        r.WorldToDisplay()
        z = r.GetDisplayPoint()[2]
        r.SetDisplayPoint(x, y, z)
        r.DisplayToWorld()
        w = r.GetWorldPoint()
        return np.array(w[:3]) / (w[3] or 1.0)

    def set_navigation(self, mode: str) -> None:
        """'rotate' or 'pan'. Rotation is ignored in 2D views."""
        self.style.mode = "pan" if self.is_2d else mode

    def zoom(self, factor: float) -> None:
        cam = self.plotter.renderer.GetActiveCamera()
        if cam.GetParallelProjection():
            cam.Zoom(factor)
        else:
            cam.Dolly(factor)      # move closer: changing the lens angle would distort the view
            self.plotter.renderer.ResetCameraClippingRange()
        self.render()

    # --- projection ---------------------------------------------------------------
    def set_perspective(self, on: bool) -> None:
        """Perspective (True) or orthographic (False) projection for the 3D view.
        The 2D views stay orthographic; the choice applies again when back in 3D."""
        self.perspective = bool(on)
        self._apply_projection(keep_size=True)
        self.render()

    def _apply_projection(self, keep_size: bool) -> None:
        """Switch the camera to the projection the current view needs. With ``keep_size``
        the area around the focal point keeps its size on screen."""
        cam = self.plotter.renderer.GetActiveCamera()
        want_parallel = not (self.perspective and self.view == VIEW_3D)
        if bool(cam.GetParallelProjection()) == want_parallel:
            return
        fp = np.array(cam.GetFocalPoint())
        pos = np.array(cam.GetPosition())
        dist = float(np.linalg.norm(pos - fp)) or 1.0
        half_angle = math.radians(cam.GetViewAngle()) / 2.0
        if want_parallel:                         # perspective -> orthographic
            if keep_size:
                cam.SetParallelScale(dist * math.tan(half_angle))
            cam.ParallelProjectionOn()
        else:                                     # orthographic -> perspective
            if keep_size:
                d = cam.GetParallelScale() / math.tan(half_angle)
                cam.SetPosition(*(fp + (pos - fp) / dist * d))
            cam.ParallelProjectionOff()
        self.plotter.renderer.ResetCameraClippingRange()

    def camera_state(self) -> dict:
        """View mode and camera in world coordinates (independent of the datum)."""
        cam = self.plotter.camera
        o = self.origin if self.origin is not None else np.zeros(3)
        return {
            "mode": self.view,
            "grid": self.grid_visible,
            "focal_point": [float(v) for v in np.array(cam.GetFocalPoint()) + o],
            "position": [float(v) for v in np.array(cam.GetPosition()) + o],
            "view_up": [float(v) for v in cam.GetViewUp()],
            "parallel_scale": float(cam.GetParallelScale()),
            "perspective": bool(self.perspective),
        }

    def apply_camera_state(self, state: dict) -> None:
        """Restore a camera saved by camera_state (the view mode is set by the caller)."""
        try:
            o = self.origin if self.origin is not None else np.zeros(3)
            fp = np.array(state["focal_point"], dtype=np.float64) - o
            pos = np.array(state["position"], dtype=np.float64) - o
            up = np.array(state["view_up"], dtype=np.float64)
            scale = float(state["parallel_scale"])
        except (KeyError, TypeError, ValueError):
            return
        if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(pos)) and scale > 0) or np.allclose(fp, pos):
            return
        cam = self.plotter.camera
        cam.SetFocalPoint(*fp)
        cam.SetPosition(*pos)
        cam.SetViewUp(*up)
        cam.SetParallelScale(scale)
        if "perspective" in state:        # saved position and scale already match it
            self.perspective = bool(state["perspective"])
            cam.SetParallelProjection(not (self.perspective and self.view == VIEW_3D))
        self.plotter.renderer.ResetCameraClippingRange()
        self.render()

    # --- identify highlight -------------------------------------------------------
    def _overlay_renderer(self) -> vtk.vtkRenderer:
        if self._overlay is None:
            rw = self.plotter.ren_win
            r = vtk.vtkRenderer()
            r.SetLayer(rw.GetNumberOfLayers())
            rw.SetNumberOfLayers(rw.GetNumberOfLayers() + 1)
            r.InteractiveOff()
            r.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
            rw.AddRenderer(r)
            self._overlay = r
        return self._overlay

    def set_highlight(self, geometry, point_local=None, color=(1.0, 0.86, 0.1), fill=None) -> None:
        """Show picked geometry (scene-local lines / points), an optional see-through
        surface ``fill`` and a marker at the picked point."""
        import pyvista as pv

        self.clear_highlight(render=False)
        items = []
        if fill is not None and fill.n_points:
            items.append((fill, dict(point_size=1.0, opacity=0.35, lit=True)))
        if geometry is not None and geometry.n_points:
            items.append((geometry, dict(line_width=3.5, point_size=10.0)))
        if point_local is not None:
            items.append((pv.PolyData(np.asarray([point_local], dtype=np.float32)),
                          dict(point_size=13.0, spheres=True)))
        self._highlight = self._overlay_actors(items, color)
        self.render()

    def clear_highlight(self, render: bool = True) -> None:
        self._remove_overlay(self._highlight)
        self._highlight = []
        if render:
            self.render()

    # --- measurement ------------------------------------------------------------------
    def set_measure(self, points_local, label: str | None = None, color=(0.25, 0.9, 1.0)) -> None:
        """Measured points (scene-local) joined by a line, with a label at the line middle."""
        import pyvista as pv

        self.clear_measure(render=False)
        pts = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
        if not len(pts):
            self.render()
            return
        items = [(pv.PolyData(pts.astype(np.float32)), dict(point_size=12.0, spheres=True))]
        if len(pts) > 1:
            line = pv.PolyData(pts.astype(np.float32), lines=np.array([len(pts), *range(len(pts))]))
            items.insert(0, (line, dict(line_width=2.5, point_size=1.0)))
        self._measure = self._overlay_actors(items, color)
        if label and len(pts) > 1:
            txt = vtk.vtkBillboardTextActor3D()
            txt.SetInput(label)
            txt.SetPosition(*((pts[0] + pts[-1]) / 2.0))
            txt.SetDisplayOffset(8, 8)
            tp = txt.GetTextProperty()
            tp.SetFontSize(15)
            tp.SetColor(1.0, 1.0, 1.0)
            tp.SetBackgroundColor(0.08, 0.1, 0.13)
            tp.SetBackgroundOpacity(0.85)
            tp.SetFrame(True)
            tp.SetFrameColor(*color)
            tp.BoldOn()
            txt.PickableOff()
            self._overlay_renderer().AddActor(txt)
            self._measure.append(txt)
        self.render()

    def clear_measure(self, render: bool = True) -> None:
        self._remove_overlay(self._measure)
        self._measure = []
        if render:
            self.render()

    # --- section line -----------------------------------------------------------------
    SECTION_COLOR = (1.0, 0.55, 0.12)

    def set_section_line(self, points_local, preview_local=None) -> None:
        """The cross-section line being drawn (or last drawn), with A / A' at its ends.
        ``preview_local``: the cursor position, joined to the last point by a thin line."""
        import pyvista as pv

        self.clear_section_line(render=False)
        pts = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
        if not len(pts):
            self.render()
            return
        color = self.SECTION_COLOR
        items = [(pv.PolyData(pts.astype(np.float32)), dict(point_size=9.0, spheres=True))]
        if len(pts) > 1:
            line = pv.PolyData(pts.astype(np.float32), lines=np.array([len(pts), *range(len(pts))]))
            items.insert(0, (line, dict(line_width=3.0, point_size=1.0)))
        if preview_local is not None:
            seg = np.array([pts[-1], preview_local], dtype=np.float32)
            items.append((pv.PolyData(seg, lines=np.array([2, 0, 1])),
                          dict(line_width=1.5, point_size=1.0, opacity=0.7)))
        self._section = self._overlay_actors(items, color)
        from .section import END_MARK, START_MARK

        ends = [(START_MARK, pts[0])] + ([(END_MARK, pts[-1])] if len(pts) > 1 and preview_local is None
                                         else [])
        for text, pos in ends:
            txt = vtk.vtkBillboardTextActor3D()
            txt.SetInput(text)
            txt.SetPosition(*pos)
            txt.SetDisplayOffset(8, 6)
            tp = txt.GetTextProperty()
            tp.SetFontSize(16)
            tp.BoldOn()
            tp.SetColor(*color)
            tp.SetBackgroundColor(0.0, 0.0, 0.0)
            tp.SetBackgroundOpacity(0.6)
            txt.PickableOff()
            self._overlay_renderer().AddActor(txt)
            self._section.append(txt)
        self.render()

    def clear_section_line(self, render: bool = True) -> None:
        self._remove_overlay(self._section)
        self._section = []
        if render:
            self.render()

    OUTLINE_COLOR = (0.25, 1.0, 0.45)

    def set_outline(self, points_local, preview_local=None, closed: bool = False) -> None:
        """A mined outline being drawn: its points, the edges, and (while drawing) thin lines
        from the last point to the cursor and back to the first point."""
        import pyvista as pv

        self.clear_outline(render=False)
        pts = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
        if not len(pts):
            self.render()
            return
        items = [(pv.PolyData(pts.astype(np.float32)), dict(point_size=9.0, spheres=True))]
        if len(pts) > 1:
            ids = list(range(len(pts))) + ([0] if closed and len(pts) > 2 else [])
            line = pv.PolyData(pts.astype(np.float32), lines=np.array([len(ids), *ids]))
            items.insert(0, (line, dict(line_width=3.0, point_size=1.0)))
        if preview_local is not None:
            seg = np.array([pts[-1], preview_local, pts[0]], dtype=np.float32)
            n = 3 if len(pts) > 1 else 2
            items.append((pv.PolyData(seg[:n], lines=np.array([n, *range(n)])),
                          dict(line_width=1.5, point_size=1.0, opacity=0.7)))
        self._outline = self._overlay_actors(items, self.OUTLINE_COLOR)
        self.render()

    def clear_outline(self, render: bool = True) -> None:
        self._remove_overlay(self._outline)
        self._outline = []
        if render:
            self.render()

    def set_preview_solid(self, mesh_local, color=(1.0, 0.86, 0.1)) -> None:
        """A see-through solid (the mined shape being entered) with its edges drawn on top."""
        self.clear_preview_solid(render=False)
        if mesh_local is None or mesh_local.n_cells == 0:
            self.render()
            return
        edges = mesh_local.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                 feature_edges=True, manifold_edges=False,
                                                 feature_angle=30)
        surf = self._overlay_actors([(mesh_local, dict(point_size=1.0, opacity=0.28, lit=True))], color)
        lines = self._overlay_actors([(edges, dict(line_width=2.0, point_size=1.0))], color)
        self._preview_solid = surf + lines
        self.render()

    def clear_preview_solid(self, render: bool = True) -> None:
        self._remove_overlay(self._preview_solid)
        self._preview_solid = []
        if render:
            self.render()

    # --- overlay helpers ------------------------------------------------------------
    def _remove_overlay(self, actors) -> None:
        if self._overlay is not None:
            for a in actors:
                self._overlay.RemoveActor(a)

    def _overlay_actors(self, items, color) -> list:
        r = self._overlay_renderer()
        actors = []
        for mesh, opts in items:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.PickableOff()
            prop = actor.GetProperty()
            prop.SetColor(*color)
            if opts.get("lit"):
                prop.SetAmbient(0.45)
                prop.SetDiffuse(0.55)
            else:
                prop.LightingOff()
            prop.SetOpacity(opts.get("opacity", 1.0))
            prop.SetLineWidth(opts.get("line_width", 1.0))
            prop.SetPointSize(opts["point_size"])
            if opts.get("spheres"):
                prop.SetRenderPointsAsSpheres(True)
            r.AddActor(actor)
            actors.append(actor)
        return actors

    # --- map grid ---------------------------------------------------------------
    def set_grid_visible(self, visible: bool) -> None:
        self.grid_visible = visible
        self.refresh_grid()
        self.render()

    def refresh_grid(self) -> None:
        """2D views: screen-fixed map grid (recomputed on every pan / zoom).
        3D view: labelled box grid around the visible data."""
        p = self.plotter
        p.remove_bounds_axes()
        self.map_grid.set_enabled(self.grid_visible and self.is_2d)
        if not self.grid_visible or self.is_2d:
            return
        b = self.world_bounds()
        if b is None:
            return
        lo, hi, n = nice_grid(*b)
        llo, lhi = self.to_local(lo), self.to_local(hi)
        p.show_grid(
            bounds=(llo[0], lhi[0], llo[1], lhi[1], llo[2], lhi[2]),
            axes_ranges=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]],
            color="#9aa0a6", font_size=10, fmt="%.0f", grid="back",
            location="outer",
            ticks="outside", xtitle=tr("X (East)"), ytitle=tr("Y (North)"), ztitle="Z",
            n_xlabels=n[0], n_ylabels=n[1], n_zlabels=n[2],
            render=False,
        )
