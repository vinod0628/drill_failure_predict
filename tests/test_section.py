"""Cross-sections: section line geometry, cutting, scale bar, points editor, print page."""

import os

import numpy as np
import pytest
import pyvista as pv

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.scene.scale_bar import nice_length, scale_layout  # noqa: E402
from dfpredict.scene.section import (  # noqa: E402
    SectionLine,
    block_rectangles,
    clip_segments,
    cut_lines,
    cut_points,
    cut_polygons,
)


def test_section_line_chainage_and_bearing():
    line = SectionLine(np.array([[0, 0, 5], [100, 0, 9], [100, 50, 1]]), axis=2)
    assert line.length == pytest.approx(150.0)
    np.testing.assert_allclose(line.chainages, [0, 100, 150])
    assert line.bearing() == pytest.approx(np.degrees(np.arctan2(100, 50)))
    p0, p1 = line.panels
    np.testing.assert_allclose(p0.t, [1, 0, 0])
    assert abs(p0.n @ [0, 0, 1]) < 1e-12          # panels are vertical for a plan line
    assert p1.u0 == pytest.approx(100.0)


def test_section_line_drops_repeated_points():
    line = SectionLine(np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0]]), axis=2)
    assert len(line.panels) == 1 and line.valid
    assert not SectionLine(np.array([[1, 1, 0], [1, 1, 0]]), axis=2).valid


def test_clip_segments_to_box():
    planes = np.array([[1, 0, 0, 10], [-1, 0, 0, 0]], dtype=float)   # 0 <= x <= 10
    p0 = np.array([[-5.0, 0, 0], [2, 0, 0], [20, 0, 0]])
    p1 = np.array([[5.0, 0, 0], [8, 0, 0], [30, 0, 0]])
    keep, t0, t1 = clip_segments(p0, p1, planes)
    assert keep.tolist() == [True, True, False]
    assert t0[0] == pytest.approx(0.5) and t1[0] == pytest.approx(1.0)
    assert t0[1] == pytest.approx(0.0) and t1[1] == pytest.approx(1.0)


def test_cut_closed_box_gives_its_outline():
    box = pv.Box(bounds=(10, 30, -5, 5, -20, 0)).triangulate()
    box.cell_data["rgb"] = np.tile([200, 10, 10], (box.n_cells, 1)).astype(np.uint8)
    line = SectionLine(np.array([[0, 0, 0], [50, 0, 0]]), axis=2)    # along x through the box
    out = cut_polygons(box, line.panels[0], 2)
    assert out is not None and "rgb" in out.cell_data
    b = out.bounds
    assert b[0] == pytest.approx(10) and b[1] == pytest.approx(30)   # u: where x crosses the box
    assert b[2] == pytest.approx(-20) and b[3] == pytest.approx(0)   # v: elevation


def test_cut_lines_keeps_only_the_corridor_and_projects():
    pts = np.array([[5, -3, 1], [5, 3, 1], [20, 30, 0], [25, 30, 0]], dtype=float)
    mesh = pv.PolyData(pts, lines=np.array([2, 0, 1, 2, 2, 3]))
    mesh.cell_data["string"] = np.array([7, 8])
    line = SectionLine(np.array([[0, 0, 0], [50, 0, 0]]), axis=2)
    out = cut_lines(mesh, line.panels[0], 2, half_width=2.0)
    assert out.n_cells == 1 and out.cell_data["string"].tolist() == [7]
    np.testing.assert_allclose(np.asarray(out.points)[:, :2], [[5, 1], [5, 1]], atol=1e-6)


def test_cut_points_within_half_width():
    mesh = pv.PolyData(np.array([[5, 1, 2], [5, 9, 2], [60, 0, 2]], dtype=float))
    mesh.point_data["record"] = np.array([1, 2, 3])
    line = SectionLine(np.array([[0, 0, 0], [50, 0, 0]]), axis=2)
    out = cut_points(mesh, line.panels[0], 2, half_width=2.0)
    assert out.point_data["record"].tolist() == [1]


def test_block_rectangles_oblique_section():
    cen = np.array([[5, 5, 0], [15, 5, 0], [5, 15, 0], [15, 15, 0]], dtype=float)
    sizes = np.full((4, 3), 10.0)
    # Diagonal from (0,0) to (20,20): passes through the two blocks on the diagonal only
    # (it touches the other two at a single corner).
    line = SectionLine(np.array([[0, 0, 0], [20, 20, 0]]), axis=2)
    idx, rect = block_rectangles(cen, sizes, line.panels[0], 2)
    assert sorted(idx.tolist()) == [0, 3]
    b = rect.bounds
    assert b[0] == pytest.approx(0) and b[1] == pytest.approx(np.hypot(20, 20))
    assert b[2] == pytest.approx(-5) and b[3] == pytest.approx(5)


def test_block_rectangles_xz_view():
    """A line drawn in the XZ view sweeps along Y; v is then the Y coordinate."""
    cen = np.array([[5, 100, 5]], dtype=float)
    sizes = np.array([[10, 4, 10]], dtype=float)
    line = SectionLine(np.array([[0, 0, 5], [20, 0, 5]]), axis=1)   # horizontal line at z=5
    idx, rect = block_rectangles(cen, sizes, line.panels[0], 1)
    assert idx.tolist() == [0]
    b = rect.bounds
    assert (b[0], b[1]) == pytest.approx((0, 10))
    assert (b[2], b[3]) == pytest.approx((98, 102))


@pytest.mark.parametrize("wpp, width, length", [(1.0, 1000, 200), (0.37, 800, 50), (12.0, 1200, 2000)])
def test_scale_bar_round_lengths(wpp, width, length):
    got, px, blocks = scale_layout(wpp, width)
    assert got == pytest.approx(length)
    assert px == pytest.approx(length / wpp)
    assert px <= width * 0.2 + 1e-9 or px <= 60 + 1e-9
    assert blocks == 2


def test_nice_length():
    assert nice_length(7.3) == 5
    assert nice_length(19.9) == 10
    assert nice_length(200) == 200
    assert nice_length(0.0) == 1.0


# --- Qt parts ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_points_editor_add_remove_reverse(qapp):
    from dfpredict.ui.section_points import SectionPointsDialog

    seen = []
    dlg = SectionPointsDialog(None, np.array([[0.0, 0.0], [100.0, 0.0]]), (0, 1))
    dlg.points_changed.connect(seen.append)
    assert dlg.table.horizontalHeaderItem(1).text().startswith("X")
    dlg.table.selectRow(0)
    dlg.add_point()                                   # halfway between A and B
    np.testing.assert_allclose(dlg.points(), [[0, 0], [50, 0], [100, 0]])
    dlg.table.item(1, 2).setText("30.000")            # typed coordinate
    assert dlg.points()[1].tolist() == [50.0, 30.0]
    assert seen and np.allclose(seen[-1][1], [50, 30])
    dlg.reverse()
    assert dlg.points()[0].tolist() == [100.0, 0.0]
    dlg.table.selectRow(1)
    dlg.remove_point()
    assert len(dlg.points()) == 2
    assert not dlg.b_remove.isEnabled()               # at least two points stay
    dlg.table.item(1, 1).setText("100.000")           # both points equal: cannot draw
    assert not dlg.b_draw.isEnabled()


def test_print_page_is_300_dpi_white(qapp):
    from PySide6.QtGui import QColor, QImage

    from dfpredict.ui.section_print import compose_page, page_size

    assert page_size("A4") == (2480, 3508)
    plan = QImage(400, 300, QImage.Format_RGB32)
    plan.fill(QColor("white"))
    sec = QImage(400, 200, QImage.Format_RGB32)
    sec.fill(QColor("white"))
    img = compose_page(plan, sec, "Plan", "Section", "footer", "A4")
    assert (img.width(), img.height()) == (2480, 3508)
    assert round(img.dotsPerMeterX() * 0.0254) == 300
    assert img.pixelColor(5, 5) == QColor("white")


def test_points_editor_legend_option(qapp):
    from dfpredict.ui.section_points import SectionPointsDialog

    dlg = SectionPointsDialog(None, np.array([[0.0, 0.0], [10.0, 0.0]]), (0, 1), legend=False)
    assert dlg.show_legend is False
    dlg.legend_box.setChecked(True)
    assert dlg.show_legend is True
    off = SectionPointsDialog(None, np.array([[0.0, 0.0], [10.0, 0.0]]), (0, 1), has_drillholes=False)
    assert not off.legend_box.isEnabled()


def test_blank_render_detection():
    from dfpredict.ui.section_print import _blank

    black = np.zeros((100, 100, 3), dtype=np.uint8)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    map_img = white.copy()
    map_img[20:60, 20:60] = (200, 120, 40)
    assert _blank(black)
    assert not _blank(map_img)
    assert not _blank(white)          # an empty area on white is a valid print


def test_interval_colours_survive_the_cut():
    """The section copy colours drillhole intervals by their rgb array, not by the
    first cell array (regression: intervals came out grey)."""
    import vtk

    from dfpredict.scene.section import clone_actor

    pts = np.array([[5, 0, 0], [5, 0, -10]], dtype=float)
    mesh = pv.PolyData(pts, lines=np.array([2, 0, 1]))
    mesh.cell_data["gv_row"] = np.array([3])
    mesh.cell_data["rgb"] = np.array([[10, 200, 30]], dtype=np.uint8)
    mesh.cell_data.active_scalars_name = "rgb"
    src = vtk.vtkActor()
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(mesh)
    m.SetColorModeToDirectScalars()
    m.SetScalarModeToUseCellData()
    src.SetMapper(m)
    line = SectionLine(np.array([[0, 0, 0], [20, 0, 0]]), axis=2)
    out = cut_lines(mesh, line.panels[0], 2, half_width=2.0)
    assert list(out.cell_data.keys())[0] == "gv_row"        # rgb is not the first array
    a = clone_actor(src, out)
    assert a.GetMapper().GetInput().GetCellData().GetScalars().GetName() == "rgb"


def test_section_ends_are_a_and_a_prime(qapp):
    from dfpredict.ui.section_points import SectionPointsDialog
    from dfpredict.ui.section_print import section_markers
    from dfpredict.scene.section import SECTION_NAME, SectionResult

    assert SECTION_NAME == "A - A'"
    dlg = SectionPointsDialog(None, np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]), (0, 1))
    assert [dlg.table.item(r, 0).text() for r in range(3)] == ["A", "1", "A'"]
    line = SectionLine(np.array([[0, 0, 0], [5, 0, 0], [10, 5, 0]]), axis=2)
    texts = [p.GetInput() for p in section_markers(SectionResult(), line, (0, 1), (1, 0, 0), (0, 0, 0))
             if hasattr(p, "GetInput") and isinstance(p.GetInput(), str)]
    assert texts == ["A", "1", "A'"]


def test_section_legends_go_beside_the_map(qapp):
    import vtk

    from dfpredict.scene.section import SectionResult
    from dfpredict.ui.section_print import map_legends, page_layout

    lut = vtk.vtkLookupTable()
    lut.SetRange(0.0, 15.0)
    lut.Build()
    res = SectionResult(legends=[("rock (geology)", [("SH  Shale", (0.3, 0.7, 0.95))])],
                        scalar_bars=[("gold", lut)])
    kinds = [item[0] for item in map_legends(res, codes=True)]
    assert kinds == ["codes", "bar"]
    bar = map_legends(res, codes=False)
    assert [b[0] for b in bar] == ["bar"] and bar[0][3:] == (0.0, 15.0)
    assert map_legends(SectionResult(), codes=True) == []
    with_col, without = page_layout("A4", True), page_layout("A4", False)
    assert with_col["section"].right() < with_col["section_legend"].left()      # outside the frame
    assert with_col["section_legend"].right() <= with_col["page"][0]
    assert without["section"].width() > with_col["section"].width()
    assert "section_legend" not in without


@pytest.mark.parametrize("raw, nice", [(780, 800), (1001, 1250), (2100, 2500), (7836, 8000), (9000, 10000),
                                       (1000, 1000), (12400, 12500)])
def test_nice_print_scale(raw, nice):
    from dfpredict.ui.section_print import nice_ratio

    assert nice_ratio(raw) == pytest.approx(nice)


def test_section_print_fit_uses_the_frame(qapp):
    """The exported section is fitted to its frame (not the screen zoom), at a standard scale,
    and the whole line from A to A' plus the cut data is inside."""
    import pyvista as pv

    from dfpredict.scene.section import SectionResult
    from dfpredict.ui.section_print import PAPER_MM_PER_PX, fit_section_camera, section_extent

    line = SectionLine(np.array([[0, 0, 0], [900, 0, 0]]), axis=2)
    res = SectionResult(bounds=(0.0, 900.0, -100.0, 50.0))
    size = (2000, 1100)
    p = pv.Plotter(off_screen=True, window_size=list(size))
    try:
        ratio = fit_section_camera(p, res, line, (-100.0, 50.0), size)
        cam = p.renderer.GetActiveCamera()
        mpp = 2 * cam.GetParallelScale() / size[1]
        assert ratio == pytest.approx(mpp * 1000 / PAPER_MM_PER_PX)
        half_w = size[0] / 2 * mpp
        cu = cam.GetFocalPoint()[0]
        u0, u1, v0, v1 = section_extent(res, line, (-100.0, 50.0))
        assert cu - half_w <= u0 and cu + half_w >= u1          # A and A' are on the map
        # the width is the limit here: less than ~25 % to spare (no needless shrinking)
        assert (u1 - u0) / (2 * half_w) > 0.75
    finally:
        p.close()


@pytest.mark.parametrize("max_len, length", [(380, 200), (420, 400), (499, 400), (510, 500), (999, 500),
                                             (1000, 1000), (150, 100)])
def test_scale_bar_lengths_include_400(max_len, length):
    assert nice_length(max_len) == length


def test_print_scale_bar_is_long():
    """On paper the bar may take up to 30 % of the map width (longer than on screen)."""
    from dfpredict.ui.section_print import PAPER_MM_PER_PX, page_layout

    w = page_layout("A4", True)["section"].width()
    mpp = 7500 * PAPER_MM_PER_PX / 1000            # 1:7,500
    length, px, blocks = scale_layout(mpp, w, 60 * 3.125, 0.3)
    assert length == 200 and 0.15 < px / w <= 0.3 and blocks == 2


def test_points_editor_shows_typed_value(qapp):
    """Regression: the cell editor's text field was 1 px wide under the app style sheet, so a
    value being typed was invisible. It is a plain line edit now, accepting numbers only."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLineEdit

    from dfpredict.ui.section_points import SectionPointsDialog
    from dfpredict.ui.theme import apply_dark_theme

    apply_dark_theme(qapp)
    dlg = SectionPointsDialog(None, np.array([[956.572, 90.0], [2454.322, 210.0]]), (0, 2))
    dlg.show()
    qapp.processEvents()
    idx = dlg.table.model().index(1, 2)
    dlg.table.setCurrentIndex(idx)
    dlg.table.edit(idx)
    qapp.processEvents()
    ed = QApplication.focusWidget()
    assert isinstance(ed, QLineEdit) and ed.width() > 40
    QTest.keyClicks(ed, "12x3.5")                 # the letter is refused
    assert ed.text() == "123.5"
    QTest.keyClick(ed, Qt.Key_Return)
    qapp.processEvents()
    assert dlg.points()[1].tolist() == [2454.322, 123.5]
    assert dlg.isVisible()                        # Enter commits the cell, not the dialog
    dlg.close()


def test_remove_duplicate_points_under_5_cm():
    from dfpredict.scene.section import DUPLICATE_TOL, duplicate_rows, remove_duplicate_points

    assert DUPLICATE_TOL == 0.05
    pts = np.array([[0, 0], [0.04, 0], [100, 0], [100, 0.03], [100, 0.1], [100, 50], [100, 50]], dtype=float)
    assert duplicate_rows(pts) == [1, 3, 6]
    kept, removed = remove_duplicate_points(pts)
    assert removed == 3
    np.testing.assert_allclose(kept, [[0, 0], [100, 0], [100, 0.1], [100, 50]])
    # exactly 0.05 m is a real segment, just under is a duplicate
    assert remove_duplicate_points(np.array([[0.0, 0.0], [0.05, 0.0]]))[1] == 0
    assert remove_duplicate_points(np.array([[0.0, 0.0], [0.0499, 0.0]]))[1] == 1
    # 3D points and the section line use the same rule
    line = SectionLine(np.array([[0, 0, 0], [0.01, 0.01, 0], [50, 0, 0], [50, 0.049, 0]]), axis=2)
    assert len(line.points) == 2 and line.length == pytest.approx(50.0)


def test_points_editor_removes_duplicates_on_draw(qapp):
    from dfpredict.ui.section_points import SectionPointsDialog

    dlg = SectionPointsDialog(None, np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]]), (0, 1))
    dlg.table.item(1, 1).setText("0.020")           # point 1 now 2 cm from A
    dlg.table.item(1, 2).setText("0.010")
    assert dlg.dup_note.isVisibleTo(dlg) and "1" in dlg.dup_note.text()
    assert len(dlg.points()) == 3                    # not removed while editing
    dlg.accept()
    assert dlg.result() == 1                         # QDialog.Accepted
    assert dlg.removed_duplicates == 1
    np.testing.assert_allclose(dlg.points(), [[0, 0], [200, 0]])


def test_points_editor_refuses_a_line_of_duplicates(qapp):
    from dfpredict.ui.section_points import SectionPointsDialog

    dlg = SectionPointsDialog(None, np.array([[10.0, 10.0], [10.03, 10.0]]), (0, 1))
    assert not dlg.b_draw.isEnabled()                # only one real point
