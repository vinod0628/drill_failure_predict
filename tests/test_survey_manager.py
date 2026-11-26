"""The Survey window: a pit's setups typed in the form, auto-numbered stations, the live
computation, compass setups, saving, and the plan map."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.io.database import Database, project_target  # noqa: E402
from dfpredict.io.survey_store import SurveyStore  # noqa: E402


@pytest.fixture
def manager(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from dfpredict.ui.survey_manager import SurveyManager

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[2]}"))
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        store.prepare(db)
    dlg = SurveyManager(store)
    assert dlg.start()
    yield dlg, store, app
    dlg.dirty = False
    dlg.close()


def _type(widget, text):
    widget.setText(text)
    widget.editingFinished.emit() if hasattr(widget, "changed") else widget.textEdited.emit(text)
    if hasattr(widget, "changed"):
        widget.changed.emit()


def test_typing_a_traverse_numbers_stations_computes_and_saves(manager):
    dlg, store, app = manager
    pid = store.add_pit("North decline", start_station="E", known=(1000.0, 5000.0, 250.0))
    dlg._after_pit_change(pid, "added")                 # what New main pit does after its dialog
    assert len(dlg.rows) == 1                            # the first setup is added at once ...
    r0 = dlg.rows[0]
    assert r0.at == "E" and r0.fs == "P1-01"             # ... at the entrance, the station numbered
    # orient on a known bearing, then the foresight: 120 30 00 is shown as 120°30′00″
    dlg.bs_kind.setCurrentIndex(1)
    _type(dlg.f_bs_brg, "45")
    _type(dlg.f_fs_ha, "120 30 00")
    assert dlg.f_fs_ha.text() == "120°30′00″"
    _type(dlg.f_fs_za, "95")
    _type(dlg.f_fs_sd, "50")
    _type(dlg.f_fs_ht, "1.4")
    _type(dlg.f_hi, "1.5")
    dlg._recompute()
    assert dlg.net.coords["P1-01"] == pytest.approx((1042.917, 4974.720, 245.742), abs=1e-3)
    assert dlg.table.item(0, 4).text() == "120°30′00″"
    # the next setup: at P1-01, backsight on E, foresight P1-02
    dlg.add_setup()
    r1 = dlg.rows[1]
    assert (r1.at, r1.bs_station, r1.fs) == ("P1-01", "E", "P1-02") and dlg.cur == 1
    assert dlg.f_bs.text() == "E" and dlg.f_fs.text() == "P1-02"
    # measured by compass instead: the backsight fields go away
    dlg.r_bearing.setChecked(True)
    assert dlg.g_bs.isHidden() and not dlg.f_fs_brg.isHidden() and dlg.f_fs_ha.isHidden()
    assert dlg.f_fs_type.currentData() == "magnetic"
    _type(dlg.f_fs_brg, "160 15 00")
    dlg.f_fs_type.setCurrentIndex(dlg.f_fs_type.findData("grid"))
    _type(dlg.f_fs_za, "92")
    _type(dlg.f_fs_sd, "40")
    _type(dlg.f_fs_ht, "1.45")
    _type(dlg.f_hi, "1.45")
    dlg._recompute()
    assert dlg.net.coords["P1-02"] == pytest.approx((1056.426, 4937.096, 244.346), abs=1e-3)
    # renaming the foresight; the station name is the user's choice
    dlg.f_fs.setText("S2")
    dlg.f_fs.textEdited.emit("S2")
    dlg._recompute()
    assert "S2" in dlg.net.coords and dlg.rows[1].fs == "S2"
    dlg.map.resize(400, 300)
    dlg.map.grab()                                       # the plan map paints
    assert dlg.dirty and dlg.save()
    st = {s.name: s for s in store.stations()}
    assert (st["S2"].x, st["S2"].y) == pytest.approx((1056.426, 4937.096), abs=1e-3)
    back = store.setups()[pid]
    assert back[1].method == "bearing" and back[1].bs_station is None and back[1].fs_bearing == pytest.approx(160.25)
    # shown again after saving: the compass setup keeps its form
    dlg.table.selectRow(1)
    assert dlg.r_bearing.isChecked() and dlg.f_fs_brg.text() == "160°15′00″"


def test_branch_starts_on_a_parent_station_and_backsights_the_one_before(manager):
    dlg, store, app = manager
    pid = store.add_pit("Main", start_station="E", known=(1000.0, 5000.0, 250.0))
    from dfpredict.io.survey_store import SetupRow
    store.save_setups(pid, [SetupRow("E", 1.5, bs_bearing=45.0, fs="P1-01", fs_ha=90.0, fs_za=90.0, fs_sd=20.0,
                                     fs_ht=1.5),
                            SetupRow("P1-01", 1.5, bs_station="E", bs_ha=0.0, fs="P1-02", fs_ha=180.0, fs_za=90.0,
                                     fs_sd=20.0, fs_ht=1.5)])
    bid = store.add_pit("Level 2", parent_id=pid, start_station="P1-01")
    dlg._after_pit_change(bid, "added")
    pit = next(p for p in dlg.pits if p.id == bid)
    assert pit.code == "P1A"
    r = dlg.rows[0]
    assert (r.at, r.bs_station, r.fs) == ("P1-01", "E", "P1A-01")
    # the tree shows the branch under its main pit
    top = dlg.tree.topLevelItem(0)
    assert top.childCount() == 1 and "Level 2" in top.child(0).text(0)
    # a typing error is shown (red, the reason above the hint), not raised
    _type(dlg.f_fs_ha, "12 75 00")
    assert "border" in dlg.f_fs_ha.styleSheet() and dlg.f_fs_ha.toolTip().startswith("⚠")
    assert "horizontal circle" in dlg.f_fs_ha.toolTip()
    _type(dlg.f_fs_ha, "12 15 00")
    assert dlg.f_fs_ha.toolTip().startswith("HA on the foresight")


def test_the_form_follows_the_selected_pit(manager):
    """Another pit whose first setup is on the same table row: the form shows that pit's setup."""
    dlg, store, app = manager
    from dfpredict.io.survey_store import SetupRow
    pid = store.add_pit("Main", start_station="E", known=(1000.0, 5000.0, 250.0))
    store.save_setups(pid, [SetupRow("E", 1.5, bs_bearing=45.0, fs="P1-01", fs_ha=90.0, fs_za=90.0, fs_sd=20.0,
                                     fs_ht=1.5)])
    bid = store.add_pit("XC", parent_id=pid, start_station="P1-01")
    store.save_setups(bid, [SetupRow("P1-01", 1.5, fs="P1A-01", fs_bearing=300.0, fs_za=90.0, fs_sd=10.0,
                                     fs_ht=1.5)])
    dlg._reload(select=pid)
    assert dlg.f_at.text() == "E" and dlg.r_angles.isChecked()
    item = next(it for it in dlg._all_items() if it.data(0, 0x0100) == bid)
    dlg.tree.setCurrentItem(item)
    assert dlg.f_at.text() == "P1-01" and dlg.f_fs.text() == "P1A-01" and dlg.r_bearing.isChecked()
    # every field has a hover hint
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QLineEdit
    fields = dlg.findChildren(QLineEdit) + dlg.findChildren(QComboBox) + dlg.findChildren(QAbstractButton)
    missing = [w for w in fields if w.isVisibleTo(dlg) and not w.toolTip()
               and not w.objectName().startswith("qt_")             # Qt's own parts (spin box text, table corner)
               and not isinstance(w.parent(), QComboBox)]
    assert not missing, [type(w).__name__ + ":" + (w.text() if hasattr(w, "text") else "") for w in missing]


def test_entrance_coordinates_can_be_changed(manager):
    """A main pit's entrance X, Y, Z are edited in the pit header: the pit and its branches move
    with it, at once on the map and in the database on Save pit. Branches have no entrance row."""
    dlg, store, app = manager
    from dfpredict.io.survey_store import SetupRow
    pid = store.add_pit("Main", start_station="E", known=(1000.0, 5000.0, 250.0))
    store.save_setups(pid, [SetupRow("E", 1.5, bs_bearing=45.0, fs="P1-01", fs_ha=90.0, fs_za=90.0, fs_sd=20.0,
                                     fs_ht=1.5)])
    bid = store.add_pit("XC", parent_id=pid, start_station="P1-01")
    store.save_setups(bid, [SetupRow("P1-01", 1.5, fs="P1A-01", fs_bearing=0.0, fs_bearing_type="grid",
                                     fs_za=90.0, fs_sd=10.0, fs_ht=1.5)])
    dlg._reload(select=pid)
    assert not dlg.entrance_row.isHidden() and dlg.l_entrance.text() == "Entrance E"
    assert dlg.p_x.text() == "1000" and dlg.p_z.text() == "250"
    _type(dlg.p_x, "1010.5")
    _type(dlg.p_z, "249")
    dlg._recompute()                                          # live, before saving
    assert dlg.net.coords["P1-01"] == pytest.approx((1030.5, 5000.0, 249.0))
    assert dlg.dirty and dlg.save()
    st = {s.name: s for s in store.stations()}
    assert (st["E"].x, st["E"].z) == (1010.5, 249.0) and st["E"].status == "known"
    assert (st["P1A-01"].x, st["P1A-01"].y, st["P1A-01"].z) == pytest.approx((1030.5, 5010.0, 249.0))
    item = next(it for it in dlg._all_items() if it.data(0, 0x0100) == bid)
    dlg.tree.setCurrentItem(item)
    assert dlg.entrance_row.isHidden()                        # a branch starts on a computed station


def test_station_coordinates_typed_in_directly(manager):
    """The third method: the new station's X, Y, Z (already calculated) typed in - no backsight,
    no readings; the next setup carries on the same way; saved and shown again."""
    dlg, store, app = manager
    pid = store.add_pit("North decline", start_station="E", known=(1000.0, 5000.0, 250.0))
    dlg._after_pit_change(pid, "added")
    dlg.r_coords.setChecked(True)
    assert dlg.g_bs.isHidden() and dlg.f_fs_ha.isHidden() and dlg.f_fs_sd.isHidden() and dlg.f_fs_brg.isHidden()
    assert not dlg.f_fs_x.isHidden() and not dlg.f_fs_z.isHidden() and not dlg.f_hi.isEnabled()
    _type(dlg.f_fs_x, "1030")
    _type(dlg.f_fs_y, "5040")
    _type(dlg.f_fs_z, "240")
    dlg._recompute()
    assert dlg.net.coords["P1-01"] == (1030.0, 5040.0, 240.0)
    assert dlg.table.item(0, 2).text() == "coordinates" and dlg.table.item(0, 5).text() == "50.000"
    assert dlg.coords.item(1, 4).text() == "given"
    dlg.add_setup()
    assert dlg.r_coords.isChecked() and dlg.rows[1].at == "P1-01" and dlg.rows[1].fs == "P1-02"
    _type(dlg.f_fs_x, "1030")
    _type(dlg.f_fs_y, "5100")
    _type(dlg.f_fs_z, "238.5")
    dlg._recompute()
    assert dlg.net.coords["P1-02"] == (1030.0, 5100.0, 238.5)
    assert "nothing to check" in dlg.msg.text()
    assert dlg.save()
    back = store.setups()[pid]
    assert [r.method for r in back] == ["coords", "coords"] and back[1].fs_z == 238.5
    dlg.table.selectRow(0)
    assert dlg.r_coords.isChecked() and dlg.f_fs_x.text() == "1030" and dlg.f_fs_y.text() == "5040"
    # back to a total station setup: the typed coordinates are dropped
    dlg.table.selectRow(1)
    dlg.r_angles.setChecked(True)
    assert dlg.rows[1].fs_x is None and not dlg.g_bs.isHidden() and dlg.f_hi.isEnabled()
