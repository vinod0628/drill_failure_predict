import os

import numpy as np
import pandas as pd
import pytest

from dfpredict.io import read_drillholes_access, read_drillholes_csv
from dfpredict.io.drillholes import DrillholeDB, _direction












def test_min_curvature_on_circular_arc():
    """A hole turning 90 degrees over a quarter circle ends where the arc ends."""
    r = 100.0
    md = np.linspace(0, np.pi / 2 * r, 10)
    dip = -90.0 + np.degrees(md / r)  # from vertical down to horizontal (east)
    collar = pd.DataFrame({"hole_id": ["A"], "x": [0.0], "y": [0.0], "z": [0.0], "max_depth": [md[-1]]})
    survey = pd.DataFrame({"hole_id": "A", "depth": md, "azimuth": 90.0, "dip": dip})
    db = DrillholeDB(collar, survey, {})
    tr = db.trace("A")
    np.testing.assert_allclose(tr.positions_at([md[-1]])[0], [r, 0, -r], atol=1e-6)
    # Mid-segment point lies on the circle too.
    q = (md[3] + md[4]) / 2
    p = tr.positions_at([q])[0]
    np.testing.assert_allclose(np.hypot(p[0] - r, p[2]), r, atol=1e-6)


def test_direction_convention():
    np.testing.assert_allclose(_direction(np.array([0.0]), np.array([0.0]))[0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(_direction(np.array([90.0]), np.array([0.0]))[0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(_direction(np.array([0.0]), np.array([-90.0]))[0], [0, 0, -1], atol=1e-12)





























# --- drillholes in the project database (geodatabase.sqlite) ----------------------------------------










def test_unreachable_mysql_gives_a_message_not_a_layer(tmp_path):
    from dfpredict.io import database
    from dfpredict.scene.loaders import load_database_drillholes
    from dfpredict.workspace import Workspace

    database.configure(database.DatabaseConfig("mysql", host="127.0.0.1", port=1, user="nobody"))
    layer, errors = load_database_drillholes(tmp_path)
    assert layer is None and "Cannot reach the MySQL server" in errors[0]
    ws = Workspace(tmp_path)
    layers, errors, _ = ws.load()
    assert not errors and "Cannot reach the MySQL server" in ws.database_error
    lay, survey = layers                                      # listed, greyed out, empty
    assert lay.offline and lay.source == "database" and not lay.visible and lay.data.collars.empty
    assert type(survey).__name__ == "SurveyLayer" and survey.offline and survey.data.pits == []










def test_grade_axis_ticks_are_round_and_end_on_the_highest_value():
    from dfpredict.ui.drillhole_log import grade_ticks

    assert grade_ticks(135.0) == (150.0, [0.0, 50.0, 100.0, 150.0])           # not 0-200: little waste
    assert grade_ticks(473.8) == (500.0, [0.0, 250.0, 500.0])
    assert grade_ticks(8.3) == (10.0, [0.0, 5.0, 10.0])
    end, ticks = grade_ticks(0.87)
    assert ticks[0] == 0.0 and ticks[-1] == end >= 0.87 and 2 <= len(ticks) <= 5
    assert all(abs(b - a - (ticks[1] - ticks[0])) < 1e-12 for a, b in zip(ticks, ticks[1:]))   # even steps
    assert grade_ticks(0.0) == (1.0, [0.0])                                    # nothing above 0


def test_scale_bar_goes_to_a_corner_clear_of_the_hole():
    """The scale bar of a projection pane: bottom left unless the hole path (or a label / the
    north arrow) is there, then the next free corner; a shorter bar when no corner is free."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QApplication

    from dfpredict.ui.drillhole_log import SCREEN, scale_bar_place

    app = QApplication.instance() or QApplication([])  # noqa: F841
    box = QRectF(100, 50, 600, 400)
    mpp = 0.5
    up_right = np.array([[650.0, 80.0], [500.0, 250.0]])                       # nowhere near the bottom left
    assert scale_bar_place(box, mpp, SCREEN, up_right)["corner"] == "bottom-left"
    to_bottom_left = np.array([[650.0, 80.0], [110.0, 445.0]])                 # a long leg into that corner
    g = scale_bar_place(box, mpp, SCREEN, to_bottom_left)
    assert g["corner"] == "bottom-right" and box.contains(g["frame"])
    # bottom left and right taken by the path, top right by the north arrow: top left
    v = np.array([[110.0, 445.0], [400.0, 300.0], [690.0, 445.0]])
    arrow = [QRectF(box.right() - 30, box.top(), 30, 50)]
    assert scale_bar_place(box, mpp, SCREEN, v, arrow)["corner"] == "top-left"
    # the hole everywhere: still placed, inside the pane
    everywhere = np.array([[x, y] for y in np.linspace(55, 445, 40) for x in (105, 695)])
    g = scale_bar_place(box, mpp, SCREEN, everywhere)
    assert g["corner"] in ("bottom-left", "bottom-right", "top-left", "top-right") and box.contains(g["frame"])




