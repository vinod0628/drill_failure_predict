"""Export map: the main view as a 300 dpi page (2D at a standard scale, 3D as seen)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def scene():
    import pyvista as pv

    from dfpredict.scene.scene import Scene

    try:
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no OpenGL available: {e}")
    sc = Scene(plotter)
    yield sc
    plotter.close()


def _with_survey(sc):
    from test_survey_display import survey_layer

    sc.clear()
    lay = survey_layer()
    sc.add_layer(lay)
    return lay


def test_plan_page_is_to_scale_with_legend_and_footer(scene):
    from PySide6.QtWidgets import QApplication

    import math

    from dfpredict.ui.map_export import fit_ratio, make_page, page_layout, page_pixels
    from dfpredict.ui.section_print import NICE_SCALES

    app = QApplication.instance() or QApplication([])  # noqa: F841
    lay = _with_survey(scene)
    scene.set_view("xy")
    scene.fit()
    page, ratio = make_page(scene, "Test - plan", "A4", True, True, True, None, "ws")
    assert (page.width(), page.height()) == page_pixels("A4", True) == (3508, 2480)       # A4 landscape, 300 dpi
    L = page_layout("A4", True, True, True)
    assert ratio == fit_ratio(scene, (L["map"].width(), L["map"].height()))
    mantissa = ratio / 10 ** math.floor(math.log10(ratio))
    assert any(abs(mantissa - m) < 1e-9 for m in NICE_SCALES + (10.0,))          # a standard scale
    page2, ratio2 = make_page(scene, "Test", "A3", False, False, False, 2500.0)
    assert ratio2 == 2500.0 and (page2.width(), page2.height()) == page_pixels("A3", False)
    # the legend column lists the main pits in their colours
    from dfpredict.ui.map_export import map_legends
    legends = map_legends(scene)
    assert ("codes", lay.name, [("Decline (P1)", lay.pit_color(1)), ("North (P2)", lay.pit_color(4))]) in legends


def test_3d_page_has_no_scale(scene):
    from PySide6.QtWidgets import QApplication

    from dfpredict.ui.map_export import make_page

    app = QApplication.instance() or QApplication([])  # noqa: F841
    _with_survey(scene)
    scene.set_view("3d")
    page, ratio = make_page(scene, "Test - 3D", "A4", False, True, False, 5000.0)
    assert ratio is None and not page.isNull() and page.height() > page.width()
    scene.set_view("xy")
