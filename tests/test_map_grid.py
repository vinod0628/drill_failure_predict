"""2D map grid: fixed to the viewport, labels follow pan / zoom."""

import os

import numpy as np
import pytest

from dfpredict.scene.map_grid import format_value, layout_grid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --- pure layout ---------------------------------------------------------------
def test_layout_covers_viewport_with_round_values():
    g = layout_grid(center_u=1650.0, center_v=7380.0, world_per_pixel=1.0, width=1000, height=600)
    assert g.step == 200  # 110 px * 1 m/px -> next round interval
    assert np.all(g.u_values % g.step == 0) and np.all(g.v_values % g.step == 0)
    assert g.u_pixels.min() >= 0 and g.u_pixels.max() <= 1000
    assert g.v_pixels.min() >= 0 and g.v_pixels.max() <= 600
    # Every line of the visible range is present (no gaps at the edges).
    assert g.u_values[0] - g.step < 1650 - 500 and g.u_values[-1] + g.step > 1650 + 500


def test_layout_pan_shifts_lines_not_frame():
    a = layout_grid(1000.0, 5000.0, 0.5, 800, 600)
    b = layout_grid(1037.0, 5000.0, 0.5, 800, 600)  # panned 37 m east
    assert a.step == b.step
    # The line at a given world value moves 74 px left on screen.
    common = np.intersect1d(a.u_values, b.u_values)
    pa = a.u_pixels[np.searchsorted(a.u_values, common)]
    pb = b.u_pixels[np.searchsorted(b.u_values, common)]
    np.testing.assert_allclose(pa - pb, 74.0)


def test_layout_zoom_changes_interval():
    coarse = layout_grid(0.0, 0.0, 2.0, 800, 600)
    fine = layout_grid(0.0, 0.0, 0.1, 800, 600)
    assert coarse.step > fine.step
    for g in (coarse, fine):
        spacing_px = g.step / (2.0 if g is coarse else 0.1)
        assert spacing_px >= 110


def test_layout_reversed_axis():
    g = layout_grid(0.0, 0.0, 1.0, 1000, 600, sign_u=-1.0)
    # World values increase to the left: larger value -> smaller pixel x.
    assert np.all(np.diff(g.u_pixels) < 0)


def test_format_value():
    assert format_value(7400.0, 100.0) == "7,400"
    assert format_value(12.5, 2.5) == "12.5" or format_value(12.5, 2.5) == "12"
    assert format_value(0.25, 0.05) == "0.25"
    assert format_value(-0.0, 1.0) == "0"


# --- in a real (off-screen) renderer ----------------------------------------------
@pytest.fixture(scope="module")
def scene():
    import pyvista as pv

    from dfpredict.scene.scene import Scene

    try:
        plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no OpenGL available: {e}")
    sc = Scene(plotter)
    sc.origin = np.array([1600.0, 7400.0, 200.0])
    plotter.add_mesh(pv.Box(bounds=(-400, 400, -300, 300, -100, 100)), render=False)
    plotter.reset_camera()
    sc.set_view("xy")
    sc.set_grid_visible(True)
    yield sc
    plotter.close()


def _render(sc):
    # An off-screen PyVista plotter ignores plotter.render() until shown; draw directly.
    sc.plotter.ren_win.Render()


def _labels(sc):
    return [t.GetInput() for t in sc.map_grid.labels if t.GetVisibility()]


def _pan(sc, dx_world, dy_world):
    """Move the camera in the view plane, as a mouse pan does."""
    cam = sc.plotter.camera
    shift = np.array([dx_world, dy_world, 0.0])
    cam.focal_point = tuple(np.array(cam.focal_point) + shift)
    cam.position = tuple(np.array(cam.position) + shift)
    _render(sc)


def _line_points(mg):
    return np.array([mg._lines.GetPoint(i) for i in range(mg._lines.GetNumberOfPoints())])


def test_grid_is_screen_fixed_while_panning(scene):
    sc = scene
    _render(sc)
    mg = sc.map_grid
    assert mg.line_actor.GetVisibility()
    assert sc.plotter.renderer.cube_axes_actor is None  # no world-attached box grid in 2D
    before = _labels(sc)
    _pan(sc, 137.0, -61.0)
    after = _labels(sc)
    assert before and after and before != after  # coordinates changed
    w, h = sc.plotter.renderer.GetSize()
    pts = _line_points(mg)
    # Every line still spans the full viewport (frame fixed to the screen).
    vertical = pts[0::2, 0] == pts[1::2, 0]
    assert vertical.any() and (~vertical).any()
    assert np.all(pts[0::2][vertical, 1] == 0) and np.all(pts[1::2][vertical, 1] == h)
    assert np.all(pts[0::2][~vertical, 0] == 0) and np.all(pts[1::2][~vertical, 0] == w)


def test_labels_show_world_coordinates(scene):
    sc = scene
    _pan(sc, 0.0, 0.0)
    cam = sc.plotter.camera
    centre_x = cam.focal_point[0] + sc.origin[0]
    xs = [float(t.GetInput().replace(",", "")) for t in sc.map_grid.labels
          if t.GetVisibility() and t.GetTextProperty().GetJustification() == 1]  # centred = X labels
    assert min(xs) < centre_x < max(xs)  # labels bracket the world X at the view centre


def test_grid_interval_follows_zoom(scene):
    sc = scene
    _render(sc)
    step0 = sc.map_grid.titles[0].GetInput()
    sc.zoom(4.0)
    _render(sc)
    step1 = sc.map_grid.titles[0].GetInput()
    sc.zoom(0.25)
    _render(sc)
    assert step0 != step1


def test_grid_off_in_3d(scene):
    sc = scene
    sc.set_view("3d")
    _render(sc)
    assert not sc.map_grid.line_actor.GetVisibility()
    assert not any(t.GetVisibility() for t in sc.map_grid.labels)
    sc.set_view("xz")
    _render(sc)
    assert sc.map_grid.line_actor.GetVisibility()
    assert sc.map_grid.titles[1].GetInput().startswith("Z")
    sc.set_view("xy")
