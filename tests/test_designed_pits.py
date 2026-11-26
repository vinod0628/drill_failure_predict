"""Designed pits: route designer results kept in the project database (survey_designs +
survey_design_points), listed under "Designed Pits" below the pits in the layers list and drawn
by the survey layer."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.io.database import Database, project_target  # noqa: E402
from dfpredict.io.survey_store import DesignPit, SurveyStore  # noqa: E402
from dfpredict.scene.survey import DESIGN_COLOR, SurveyData  # noqa: E402

from test_survey_display import _drawn, scene, survey_layer  # noqa: E402,F401


def _design(name="D1", did=0):
    return DesignPit(did, name, [(1000.0, 5040.0, 250.0), (1060.0, 5040.0, 245.0), (1060.0, 5100.0, 240.0)],
                     5.0, 4.0, "floor", 1, "A2", "X 1060.00  Y 5100.00  Z 240.00", 1 / 7, 1 / 12, 120.0, -10.0,
                     {"void_clearance": 10.0, "best_start": True})


def test_designs_are_stored_renamed_and_deleted(tmp_path):
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        notes = store.prepare(db)
    assert "made table survey_designs" in notes and "made table survey_design_points" in notes
    assert store.designs() == [] and store.next_design_name("D") == "D1"
    did = store.add_design(_design())
    back = store.designs()
    assert [d.id for d in back] == [did]
    d = back[0]
    assert d.name == "D1" and d.points == [(1000.0, 5040.0, 250.0), (1060.0, 5040.0, 245.0), (1060.0, 5100.0, 240.0)]
    assert (d.profile_width, d.profile_height, d.station_level, d.start_pit_id) == (5.0, 4.0, "floor", 1)
    assert d.gradient == pytest.approx(1 / 7) and d.length == 120.0 and d.climb == -10.0
    assert d.params == {"void_clearance": 10.0, "best_start": True} and d.created_at
    assert store.next_design_name("D") == "D2"
    with pytest.raises(ValueError, match="already a designed pit"):
        store.add_design(_design())
    store.add_design(_design("D2"))
    with pytest.raises(ValueError, match="already a designed pit"):
        store.rename_design(did, "D2")
    store.rename_design(did, "Decline extension")
    assert [d.name for d in store.designs()] == ["Decline extension", "D2"]
    store.delete_design(did)
    assert [d.name for d in store.designs()] == ["D2"]
    assert all(d.id != did for d in store.designs())


def test_a_database_without_the_design_tables_gets_them_when_saving(tmp_path):
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        db.execute("CREATE TABLE survey_pits (id INTEGER NOT NULL PRIMARY KEY, name VARCHAR(100) NOT NULL)")
        db.con.commit()
    assert store.designs() == []                      # an older database: nothing, no error
    store.add_design(_design())
    assert [d.name for d in store.designs()] == ["D1"]


def _with_designs():
    lay = survey_layer()
    d = lay.data
    lay.data = SurveyData(d.pits, d.stations, d.paths, d.source, designs=[_design("D1", 1), _design("D2", 2)])
    lay.data.designs[1].points = [(p[0] + 200.0, p[1], p[2]) for p in lay.data.designs[1].points]
    return lay


def test_layers_list_shows_designed_pits_under_the_pits():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from dfpredict.ui.layer_tree import DESIGN_ROLE, GROUP_ROLE, LayerTree
    from dfpredict.ui.properties_panel import PropertiesPanel

    app = QApplication.instance() or QApplication([])  # noqa: F841
    tree = LayerTree()
    plain = survey_layer()
    tree.add_layer(plain)
    top = tree.tree.topLevelItem(0)
    assert [top.child(i).text(0) for i in range(top.childCount())] == ["Decline  (P1)", "North  (P2)"]  # none yet
    lay = _with_designs()
    tree.replace_layer(plain, lay)
    kids = [top.child(i) for i in range(top.childCount())]
    assert [k.text(0) for k in kids] == ["Decline  (P1)", "North  (P2)", "Designed Pits"]   # after the pits
    group = kids[-1]
    assert group.data(0, GROUP_ROLE) == "designs" and group.checkState(0) == Qt.Checked
    assert [group.child(i).text(0) for i in range(group.childCount())] == ["D1", "D2"]
    d1 = group.child(0)
    assert d1.data(0, DESIGN_ROLE) == 1
    got = []
    tree.design_visibility_changed.connect(lambda layer, did, on: got.append(("one", did, on)))
    tree.designs_visibility_changed.connect(lambda layer, on: got.append(("all", on)))
    tree.visibility_changed.connect(lambda layer, on: got.append(("layer", on)))
    d1.setCheckState(0, Qt.Unchecked)
    group.setCheckState(0, Qt.Unchecked)
    assert got == [("one", 1, False), ("all", False)]            # never the whole layer
    # selecting a designed pit shows it in the properties
    props = PropertiesPanel(lambda: None)
    tree.design_selected.connect(props.set_design)
    tree.tree.setCurrentItem(d1)
    assert props.design == 1
    from PySide6.QtWidgets import QLabel

    texts = {w.text() for w in props.findChildren(QLabel)}
    assert "D1" in texts and "120.0 m" in texts and "1 in 12.0 (8.3 %)" in texts
    # a reload keeps the designed pit selected
    again = _with_designs()
    tree.replace_layer(lay, again)
    assert tree.current_design() == 1 and tree.tree.currentItem().data(0, Qt.UserRole) is again


def test_designed_pits_are_drawn_and_hidden(scene):  # noqa: F811
    lay = _with_designs()
    scene.clear()
    scene.add_layer(lay)

    def designs_drawn():
        lay.scene.plotter.ren_win.Render()
        out = set()
        for a in lay.actors:
            m = getattr(a, "mapper", None)
            ds = getattr(m, "dataset", None) if m is not None else None
            if ds is not None and "design" in ds.cell_data and not ds.n_lines:
                out |= {int(x) for x in ds.cell_data["design"]}
        return out

    assert designs_drawn() == {1, 2} and _drawn(lay)[0] == {1, 2, 3, 4}
    lay.set_design_style(2, visible=False)
    assert designs_drawn() == {1}
    lay.set_show_designs(False)
    assert designs_drawn() == set() and _drawn(lay)[0] == {1, 2, 3, 4}   # the surveyed pits stay
    lay.set_show_designs(True)
    lay.set_design_style(2, visible=True, color=(1.0, 0.0, 0.0))
    assert lay.design_color(2) == (1.0, 0.0, 0.0) and lay.design_color(1) == DESIGN_COLOR
    # settings saved with the workspace
    again = _with_designs()
    again.apply_settings(lay.settings())
    assert again.design_color(2) == (1.0, 0.0, 0.0) and again.show_designs
    # volume: the closed tube, about width x height x length
    assert lay.design_volume(1) == pytest.approx(5.0 * 4.0 * np.hypot(60, 5) + 5.0 * 4.0 * np.hypot(60, 5), rel=0.05)
    lo, hi = lay.design_bounds(1)
    assert lo[0] < 1000.0 and hi[0] > 1060.0 and lay.design_bounds() is not None
    scene.clear()


def test_a_designed_pit_is_changed_in_the_database(tmp_path):
    from dfpredict.io.survey_store import design_measures

    store = SurveyStore(project_target(tmp_path))
    did = store.add_design(_design())
    pts = [(0.0, 0.0, 0.0), (30.0, 40.0, -5.0), (30.0, 80.0, -5.0)]
    store.update_design(did, "D1 option B", "to be checked", (6.0, 5.5, "back"), pts)
    d = store.designs()[0]
    assert (d.name, d.note, d.profile_width, d.profile_height, d.station_level) == ("D1 option B", "to be checked",
                                                                                     6.0, 5.5, "back")
    assert d.points == pts and d.climb == -5.0
    assert d.length == pytest.approx(np.sqrt(50 ** 2 + 25) + 40.0) and d.steepest == pytest.approx(0.1)
    assert design_measures(pts) == pytest.approx((d.length, -5.0, 0.1))
    assert d.params == {"void_clearance": 10.0, "best_start": True}       # kept
    with pytest.raises(ValueError, match="at least 2 points"):
        store.update_design(did, "D1", points=[(0.0, 0.0, 0.0)])


def test_pit_manager_lists_edits_and_deletes_designed_pits(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from dfpredict.ui.survey_manager import DESIGN_ROLE, GROUP_ROLE, SurveyManager

    app = QApplication.instance() or QApplication([])  # noqa: F841
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[2]}"))
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        store.prepare(db)
    pid = store.add_pit("Main decline", start_station="E", known=(1000.0, 5000.0, 250.0))
    did = store.add_design(_design())
    dlg = SurveyManager(store)
    assert dlg.start() and dlg.windowTitle() == "Pit Manager"
    tops = [dlg.tree.topLevelItem(i) for i in range(dlg.tree.topLevelItemCount())]
    assert [t.text(0) for t in tops] == ["Main decline  (P1)", "Designed Pits"]       # after the pits
    group = tops[-1]
    assert group.data(0, GROUP_ROLE) == "designs" and group.child(0).text(0) == "D1"
    assert dlg.pid == pid and dlg.mid_stack.currentIndex() == 0
    changed = []
    dlg.changed.connect(lambda: changed.append(1))
    dlg.tree.setCurrentItem(group.child(0))                    # the designed pit: its own page
    assert dlg.did == did and dlg.pid is None and dlg.mid_stack.currentIndex() == 1
    assert dlg.d_name.text() == "D1" and dlg.d_table.rowCount() == 3 and not dlg.dirty
    assert dlg.map.current_design == did and dlg.map.designs[0][0] == did
    # change a point, insert one, rename: saved to the database
    dlg.d_table.item(2, 3).setText("235")
    assert dlg.dirty and dlg.b_save.isEnabled()
    dlg.d_table.selectRow(0)
    dlg.insert_design_point()                                  # halfway along the first leg
    assert dlg.d_table.rowCount() == 4 and dlg.d_table.item(1, 1).text() == "1030.000"
    dlg.d_name.setText("D1 option B")
    dlg.d_name.textEdited.emit("D1 option B")
    assert dlg.save()
    d = store.designs()[0]
    assert d.name == "D1 option B" and len(d.points) == 4 and d.points[-1][2] == 235.0 and changed
    assert dlg.did == did and not dlg.dirty and group is not dlg.tree.topLevelItem(1)   # the tree was read again
    assert dlg.tree.topLevelItem(1).child(0).text(0) == "D1 option B"
    # going back to the pit and deleting the designed pit
    dlg.tree.setCurrentItem(dlg.tree.topLevelItem(0))
    assert dlg.pid == pid and dlg.did is None and dlg.mid_stack.currentIndex() == 0
    dlg.tree.setCurrentItem(dlg.tree.topLevelItem(1).child(0))
    dlg.delete_pit()                                           # the Delete button: the designed pit only
    assert store.designs() == [] and [p.id for p in store.pits()] == [pid]
    assert dlg.tree.topLevelItemCount() == 1 and dlg.pid == pid and dlg.mid_stack.currentIndex() == 0
    dlg.dirty = False
    dlg.close()


def test_clicking_visible_in_the_properties_does_not_crash_and_the_group_has_its_own_page():
    """A mouse click on a pit's Visible rebuilt the panel under the check box being clicked (a crash).
    The "Designed Pits" group has its own page: Visible there hides only the designed pits."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QCheckBox

    from dfpredict.ui.layer_tree import GROUP_ROLE, LayerTree
    from dfpredict.ui.properties_panel import PropertiesPanel

    app = QApplication.instance() or QApplication([])
    lay = _with_designs()
    tree = LayerTree()
    tree.add_layer(lay)
    props = PropertiesPanel(lambda: None)
    props.resize(300, 600)
    props.show()
    tree.pit_selected.connect(props.set_pit)
    tree.designs_selected.connect(props.set_designs)
    tree.design_selected.connect(props.set_design)
    tree.layer_selected.connect(props.set_layer)
    top = tree.tree.topLevelItem(0)

    def boxes():
        app.processEvents()
        return [b for b in props.findChildren(QCheckBox) if b.isVisible()]

    def click(b):
        QTest.mouseClick(b, Qt.LeftButton, Qt.NoModifier, QPoint(8, b.height() // 2))
        app.processEvents()

    tree.tree.setCurrentItem(top.child(0))                       # main pit 1
    click(boxes()[0])
    assert lay.pit_style(1)["visible"] is False
    click(boxes()[0])
    assert lay.pit_style(1)["visible"] is True
    group = next(top.child(i) for i in range(top.childCount()) if top.child(i).data(0, GROUP_ROLE))
    tree.tree.setCurrentItem(group)
    assert props.designs_group
    click(boxes()[0])                                            # Visible of Designed Pits
    assert lay.show_designs is False and lay.visible is True
    assert all(lay.pit_shown(p.id) for p in lay.data.pits)       # the surveyed pits are not affected
    click(boxes()[0])
    click(boxes()[1])                                            # Centre lines: all designed pits
    assert not any(lay.design_style(x.id)["centerline"] for x in lay.data.designs)
    tree.tree.setCurrentItem(group.child(1))
    click(boxes()[1])                                            # one designed pit's Centre line
    assert lay.design_style(2)["centerline"] and not lay.design_style(1)["centerline"]
    props.close()
