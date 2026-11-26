"""Mouse navigation: pan mode and 2D views must pan, never rotate.

Regression test for: after a double-click PyVista re-installed its own
rotating interactor style, so every later drag rotated (also in pan mode and
in the 2D XY / XZ / YZ views).
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def scene():
    pytest.importorskip("pyvistaqt")
    import pyvista as pv
    from PySide6.QtWidgets import QApplication

    from dfpredict.scene.scene import Scene

    app = QApplication.instance() or QApplication([])
    try:
        from pyvistaqt import QtInteractor

        plotter = QtInteractor(None, auto_update=False)
        plotter.window_size = (800, 600)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no OpenGL window available: {e}")
    sc = Scene(plotter)
    plotter.add_mesh(pv.Box(bounds=(0, 300, 0, 200, 0, 100)), render=False)
    plotter.reset_camera()
    plotter.render()
    yield sc
    plotter.close()
    del app


def _click(iren, x, y):
    iren.SetEventPosition(x, y)
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")


def _drag(scene, dx=120, dy=40, button="Left"):
    """Drag across the view with a mouse button ("Left" or "Middle");
    return (view direction change in deg, focal point shift)."""
    iren = scene.plotter.iren.interactor
    cam = scene.plotter.camera
    f0, p0 = np.array(cam.focal_point), np.array(cam.position)
    d0 = (f0 - p0) / np.linalg.norm(f0 - p0)
    x, y = 400, 300
    iren.SetShiftKey(0)
    iren.SetControlKey(0)
    iren.SetEventPosition(x, y)
    iren.InvokeEvent(f"{button}ButtonPressEvent")
    for i in range(1, 11):
        iren.SetEventPosition(x + dx * i // 10, y + dy * i // 10)
        iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent(f"{button}ButtonReleaseEvent")
    f1, p1 = np.array(cam.focal_point), np.array(cam.position)
    d1 = (f1 - p1) / np.linalg.norm(f1 - p1)
    return float(np.degrees(np.arccos(np.clip(d0 @ d1, -1, 1)))), float(np.linalg.norm(f1 - f0))


def test_rotate_mode_rotates(scene):
    scene.set_view("3d")
    scene.set_navigation("rotate")
    turned, _ = _drag(scene)
    assert turned > 1.0


def test_pan_mode_pans_even_after_double_click(scene):
    scene.set_view("3d")
    iren = scene.plotter.iren.interactor
    _click(iren, 400, 300)
    _click(iren, 400, 300)  # double-click
    scene.set_navigation("pan")
    for _ in range(3):
        turned, moved = _drag(scene)
        assert turned < 1e-6
        assert moved > 0.0
    assert iren.GetInteractorStyle() is scene.style


@pytest.mark.parametrize("view, direction", [("xy", (0, 0, -1)), ("xz", (0, 1, 0)), ("yz", (-1, 0, 0))])
def test_2d_views_only_pan(scene, view, direction):
    scene.set_navigation("rotate")       # the user had rotate selected in 3D
    scene.set_view(view)
    iren = scene.plotter.iren.interactor
    _click(iren, 400, 300)
    _click(iren, 400, 300)
    for _ in range(2):
        turned, moved = _drag(scene)
        assert turned < 1e-6, f"{view}: view rotated by {turned:.2f} deg"
        assert moved > 0.0
    cam = scene.plotter.camera
    d = np.array(cam.focal_point) - np.array(cam.position)
    np.testing.assert_allclose(d / np.linalg.norm(d), direction, atol=1e-9)
    scene.set_view("3d")


@pytest.mark.parametrize("view, mode", [("3d", "rotate"), ("3d", "pan"), ("xy", "rotate"), ("xz", "pan")])
def test_middle_drag_pans(scene, view, mode):
    """Holding the middle button and moving pans in every view and navigation mode."""
    scene.set_view(view)
    scene.set_navigation(mode)
    for _ in range(2):
        turned, moved = _drag(scene, button="Middle")
        assert turned < 1e-6, f"{view}/{mode}: middle drag rotated by {turned:.2f} deg"
        assert moved > 0.0
    # the left button keeps its own action afterwards
    turned, _ = _drag(scene)
    assert (turned > 1.0) == (view == "3d" and mode == "rotate")
    scene.set_view("3d")


def test_right_drag_zooms_and_release_reaches_tools(scene):
    """Right drag zooms (no rotation), and the release still reaches other observers
    (the cross-section tool finishes its line on a right click)."""
    scene.set_view("xy")
    iren = scene.plotter.iren.interactor
    seen = []
    tag = iren.AddObserver("RightButtonReleaseEvent", lambda o, e: seen.append(e), 10.0)
    cam = scene.plotter.camera
    s0 = cam.GetParallelScale()
    iren.SetEventPosition(400, 300)
    iren.InvokeEvent("RightButtonPressEvent")
    for i in range(1, 11):
        iren.SetEventPosition(400, 300 + 8 * i)
        iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("RightButtonReleaseEvent")
    iren.RemoveObserver(tag)
    assert cam.GetParallelScale() != pytest.approx(s0)
    assert seen == ["RightButtonReleaseEvent"]
    scene.set_view("3d")


def test_3d_view_reset_after_rotation(scene):
    """Home in 3D: the default direction comes back however the view was rotated."""
    scene.set_view("3d")
    cam = scene.plotter.camera
    d0, u0 = np.array(cam.GetDirectionOfProjection()), np.array(cam.GetViewUp())
    scene.set_navigation("rotate")
    turned, _ = _drag(scene, 200, 90)
    assert turned > 5.0
    scene.set_view("3d")
    np.testing.assert_allclose(cam.GetDirectionOfProjection(), d0, atol=1e-9)
    np.testing.assert_allclose(cam.GetViewUp(), u0, atol=1e-9)


def _screen_pos(scene, world):
    r = scene.plotter.renderer
    r.SetWorldPoint(*world, 1.0)
    r.WorldToDisplay()
    return np.array(r.GetDisplayPoint()[:2])


def test_perspective_toggle_keeps_view_and_2d_stays_orthographic(scene):
    scene.set_view("3d")
    cam = scene.plotter.renderer.GetActiveCamera()
    fp = np.array(cam.GetFocalPoint())
    # a point beside the focal point, in the plane through it facing the camera
    right = np.cross(np.array(cam.GetDirectionOfProjection()), np.array(cam.GetViewUp()))
    side = fp + right / np.linalg.norm(right) * 50.0
    before = _screen_pos(scene, side)
    scene.set_perspective(True)
    assert not cam.GetParallelProjection()
    np.testing.assert_allclose(cam.GetFocalPoint(), fp, atol=1e-6)        # same centre
    np.testing.assert_allclose(_screen_pos(scene, side), before, atol=1.5)  # same size on screen
    assert scene.camera_state()["perspective"] is True
    s0 = cam.GetViewAngle()
    scene.zoom(1.25)                                     # perspective zoom moves the camera
    assert cam.GetViewAngle() == pytest.approx(s0)
    scene.set_view("xy")
    assert cam.GetParallelProjection()                   # 2D is always orthographic
    scene.set_view("3d")
    assert not cam.GetParallelProjection()               # the 3D choice comes back
    scene.set_perspective(False)
    assert cam.GetParallelProjection()
    np.testing.assert_allclose(_screen_pos(scene, side), _screen_pos(scene, side), atol=1e-9)


def test_perspective_saved_in_camera_state(scene):
    scene.set_view("3d")
    scene.set_perspective(True)
    state = scene.camera_state()
    scene.set_perspective(False)
    scene.apply_camera_state(state)
    assert scene.perspective is True
    assert not scene.plotter.renderer.GetActiveCamera().GetParallelProjection()
    scene.set_perspective(False)


def test_view_caption(scene):
    """Top-left caption: 3D VIEW / XY PLANE / XZ PLANE / YZ PLANE; can be hidden."""
    for view, text in (("3d", "3D VIEW"), ("xy", "XY PLANE"), ("xz", "XZ PLANE"), ("yz", "YZ PLANE")):
        scene.set_view(view)
        assert scene.caption.GetInput().strip() == text
    tp = scene.caption.GetTextProperty()
    assert tp.GetBackgroundOpacity() == 0.0          # no box behind the caption
    scene.set_caption_style(28, 30)
    assert tp.GetFontSize() == 28 and tp.GetOpacity() == pytest.approx(0.7)   # the text is see-through
    assert scene.map_grid.top_offset > 28            # the 2D grid title keeps clear of it
    scene.set_caption_visible(False)
    assert not scene.caption.GetVisibility() and scene.map_grid.top_offset == 0
    scene.set_caption_visible(True)
    assert scene.caption.GetVisibility()
    scene.set_view("3d")


def test_caption_setting_is_saved(tmp_path):
    from PySide6.QtCore import QSettings

    from dfpredict.ui.settings_dialog import AppSettings

    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    assert AppSettings.load(qs).show_view_caption is True          # on by default
    assert (AppSettings.load(qs).caption_size, AppSettings.load(qs).caption_transparency) == (20, 40)
    s = AppSettings.load(qs)
    s.show_view_caption, s.caption_size, s.caption_transparency = False, 32, 60
    s.save(qs)
    qs.sync()
    r = AppSettings.load(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    assert (r.show_view_caption, r.caption_size, r.caption_transparency) == (False, 32, 60)


def test_background_colour_and_readable_overlays(scene):
    from dfpredict.scene.scene import DARK_TEXT, LIGHT_TEXT, foreground_for

    assert foreground_for((0, 0, 0)) == LIGHT_TEXT and foreground_for((1, 1, 1)) == DARK_TEXT
    assert foreground_for((0.42, 0.44, 0.47)) == LIGHT_TEXT       # mid-dark grey: light text
    scene.set_background((1.0, 1.0, 1.0))
    assert scene.plotter.renderer.GetBackground() == pytest.approx((1.0, 1.0, 1.0))
    assert scene.caption.GetTextProperty().GetColor() == pytest.approx(DARK_TEXT)
    assert scene.map_grid.label_color == pytest.approx(DARK_TEXT)
    scene.set_background((0.0, 0.0, 0.0))
    assert scene.caption.GetTextProperty().GetColor() == pytest.approx(LIGHT_TEXT)


def test_background_setting_is_saved(tmp_path):
    from PySide6.QtCore import QSettings

    from dfpredict.ui.settings_dialog import AppSettings

    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    s = AppSettings.load(qs)
    assert s.background == "#000000" and s.background_rgb == (0.0, 0.0, 0.0)
    s.background = "#ffffff"
    s.save(qs)
    qs.sync()
    r = AppSettings.load(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    assert r.background == "#ffffff" and r.background_rgb == (1.0, 1.0, 1.0)


def test_3d_view_and_home_start_in_rotate_mode(tmp_path, monkeypatch):
    """Back from a 2D view to 3D, and Home in 3D, the left mouse rotates again (the Rotate
    button on) - also when Pan was chosen before."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    import dfpredict.ui.main_window as mw
    from dfpredict.scene.scene import VIEW_3D, VIEW_XY

    app = QApplication.instance() or QApplication([])  # noqa: F841
    ini = str(tmp_path / "settings.ini")                       # never the user's own settings
    monkeypatch.setattr(mw, "QSettings", lambda *a: QSettings(ini, QSettings.IniFormat))
    win = mw.MainWindow()
    win.quiet_errors = True
    try:
        win.set_view(VIEW_XY)
        assert win.a_pan.isChecked() and not win.a_rotate.isEnabled() and win.scene.style.mode == "pan"
        win.set_view(VIEW_3D)
        assert win.a_rotate.isChecked() and win.a_rotate.isEnabled() and win.scene.style.mode == "rotate"
        win.a_pan.trigger()                                    # the user picks Pan in 3D ...
        assert win.a_pan.isChecked() and win.scene.style.mode == "pan"
        win.home()                                             # ... Home: rotate again
        assert win.a_rotate.isChecked() and win.scene.style.mode == "rotate"
        win.a_pan.trigger()
        win.set_view(VIEW_3D)                                  # 3D chosen again (already in 3D)
        assert win.a_rotate.isChecked() and win.scene.style.mode == "rotate"
        win.set_view(VIEW_XY)
        win.home()                                             # Home in 2D: still panning
        assert win.a_pan.isChecked() and win.scene.style.mode == "pan"
    finally:
        win.close()
