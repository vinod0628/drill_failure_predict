"""Pit geology: the records kept per pit (rocks, structures, samples), their checks, the chainage
along a pit (0 m at the entrance of a main pit, at the junction station of a branch), where the
records sit on the drive's walls, the drawing in the view and the Pit Geology window."""

import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.io.database import Database, project_target  # noqa: E402
from dfpredict.io.pit_geology_store import (GeoInterval, GeoSample, GeoStructure, PitGeology,  # noqa: E402
                                            PitGeologyStore, check_geology, next_sample_name, orientation_text,
                                            plane_normal)
from dfpredict.io.survey_store import Pit, SetupRow, Station, SurveyStore, compute  # noqa: E402
from dfpredict.scene.pit_geology import (axis_names, parse_chainage, pit_axis, station_offset_text,  # noqa: E402
                                         trace_direction)
from dfpredict.scene.survey import SurveyData, SurveyLayer, pit_paths  # noqa: E402


def _leg(at, fs, brg, sd=20.0, za=90.0):
    return SetupRow(at, 1.5, fs=fs, fs_bearing=brg, fs_bearing_type="grid", fs_za=za, fs_sd=sd, fs_ht=1.5)


def survey_data() -> SurveyData:
    """Main pit 1 from entrance E north (E -> A1 20 m -> A2 20 m), branch 2 east from A1 (-> B1 15 m).
    Profile 4 m wide x 3 m high, stations in the floor."""
    pits = [Pit(1, "Decline", start_station="E", code="P1", profile_width=4.0, profile_height=3.0,
                station_level="floor"),
            Pit(2, "XC1", 1, "A1", code="P1A", profile_width=4.0, profile_height=3.0, station_level="floor")]
    setups = {1: [_leg("E", "A1", 0.0), _leg("A1", "A2", 0.0)], 2: [_leg("A1", "B1", 90.0, sd=15.0)]}
    e = Station("E", 1, "known", 1000.0, 5000.0, 250.0)
    net = compute(pits, [e], setups)
    stations = {"E": e}
    for n, pid in {"A1": 1, "A2": 1, "B1": 2}.items():
        stations[n] = Station(n, pid, "station", *net.coords[n], status="open")
    return SurveyData(pits, stations, pit_paths(setups), "test")


# --- chainage ----------------------------------------------------------------------------------------
def test_chainage_starts_at_the_entrance_and_at_a_branch_junction():
    d = survey_data()
    main, branch = pit_axis(d, 1), pit_axis(d, 2)
    assert main.station_chainage() == pytest.approx({"E": 0.0, "A1": 20.0, "A2": 40.0})
    assert branch.station_chainage() == pytest.approx({"A1": 0.0, "B1": 15.0})     # 0 m at the junction
    assert main.length == pytest.approx(40.0) and branch.length == pytest.approx(15.0)
    # a station and the metres taped past (or before) it; the longest matching name wins
    st = main.station_chainage()
    assert parse_chainage("A1+4.5", st) == pytest.approx(24.5)
    assert parse_chainage("a1 + 4,5", st) == pytest.approx(24.5)
    assert parse_chainage("A2-2", st) == pytest.approx(38.0)
    assert parse_chainage("A1", st) == pytest.approx(20.0)
    assert parse_chainage("12.25", st) == 12.25 and parse_chainage("", st) is None
    assert parse_chainage("P1-10+2", {"P1-1": 5.0, "P1-10": 50.0}) == pytest.approx(52.0)
    for bad in ("X9+3", "A1+", "A1*3", "A1++2"):
        with pytest.raises(ValueError):
            parse_chainage(bad, st)
    assert station_offset_text(24.5, st) == "A1+4.50"


def test_axis_follows_the_line_from_the_start_station():
    assert axis_names([["A", "B"], ["S", "A"]], "S") == ["S", "A", "B"]
    assert axis_names([["S", "A"], ["X", "Y"]], "S") == ["S", "A"]          # a separate line is not joined
    assert axis_names([], "S") == []


def test_points_on_the_walls_left_and_right_looking_toward_the_face():
    ax = pit_axis(survey_data(), 1)                    # heading north from E (1000, 5000, 250), floor stations
    left = ax.wall_point(10.0, "left", 1.0)
    right = ax.wall_point(10.0, "right", 2.0)
    back = ax.wall_point(10.0, "back")
    assert left == pytest.approx((998.0, 5010.0, 251.0))    # west of the centre line when heading north
    assert right == pytest.approx((1002.0, 5010.0, 252.0))
    assert back == pytest.approx((1000.0, 5010.0, 253.0))   # the back: the profile height above the floor
    assert ax.wall_point(10.0, "left")[2] == pytest.approx(251.5)      # no height: half way up
    assert ax.bearing_at(10.0) == pytest.approx(0.0)
    # past the last station the drive goes on along the last leg
    assert ax.wall_point(45.0, "back") == pytest.approx((1000.0, 5045.0, 253.0))
    assert ax.outward(10.0, "left") == pytest.approx((-1.0, 0.0, 0.0))
    # the branch heads east: its left wall is on the north side
    br = pit_axis(survey_data(), 2)
    assert br.wall_point(5.0, "left", 0.0)[:2] == pytest.approx((1005.0, 5022.0))


def test_structure_orientation_and_its_trace_on_the_walls():
    assert orientation_text(135, 45) == "135∠45" and orientation_text(None, 45) == ""
    assert plane_normal(90, 0) == pytest.approx((0.0, 0.0, 1.0))
    nx, ny, nz = plane_normal(90, 45)                       # dips east: the normal leans east
    assert nx == pytest.approx(math.sqrt(0.5)) and ny == pytest.approx(0) and nz == pytest.approx(math.sqrt(0.5))
    # drive heading north: a plane striking north (dipping east) shows a level trace on the walls ...
    along, up = trace_direction(90, 45, 0.0, "left")
    assert abs(up) == pytest.approx(0.0, abs=1e-9) and abs(along) == pytest.approx(1.0)
    # ... and a trace along the drive on the back
    along, left = trace_direction(90, 45, 0.0, "back")
    assert abs(left) == pytest.approx(0.0, abs=1e-9)
    # a plane dipping 45° forward (north): going forward on a wall, its trace goes down
    along, up = trace_direction(0, 45, 0.0, "right")
    assert along * up < 0 and abs(up / along) == pytest.approx(1.0)
    # parallel to the wall: no trace
    assert trace_direction(90, 90, 0.0, "left") is None


# --- checks ----------------------------------------------------------------------------------------
def test_checks_find_the_mistakes_before_saving():
    geo = PitGeology(
        [GeoInterval(0, 10, "all", "SH"), GeoInterval(8, 12, "left", "ST"), GeoInterval(12, 11, "all", "ST"),
         GeoInterval(20, 25, "back", "")],
        [GeoStructure(5, "left", 1, "fault", 135, 95), GeoStructure(6, "back", None, "joint", None, 30),
         GeoStructure(-1, "right", None, "vein", 10, 10)],
        [GeoSample("S1", 0, 1), GeoSample("S1", 1, 2), GeoSample("", 2, 1.5)])
    errors, warnings = check_geology(geo, length=22.0)
    text = "\n".join(errors)
    assert "Rocks rows 1 and 2 overlap on the left wall (8.00 - 10.00 m)" in text
    assert "Rocks row 3: To must be more than From" in text
    assert "Rocks row 4: choose the rock" in text
    assert "Structures row 1: the dip must be 0° to 90°" in text
    assert "Structures row 2: type both the dip direction and the dip (or neither)" in text
    assert "Structures row 3: Chainage must be 0 m or more" in text
    assert "sample number S1 is used twice (rows 1 and 2)" in text
    assert "Samples row 3: type the sample number" in text and "Samples row 3: To must be more than From" in text
    assert any("Rocks row 4: To 25.00 m is beyond the last station (22.00 m)" in w for w in warnings)
    good = PitGeology([GeoInterval(0, 10, "left", "SH"), GeoInterval(0, 10, "right", "ST")],
                      [GeoStructure(5, "left", 1, "fault", 135, 60)], [GeoSample("S1", 0, 1)])
    assert check_geology(good, 40.0) == ([], [])          # different walls do not overlap
    assert next_sample_name("PS-009") == "PS-010" and next_sample_name("A12b") == "A13b"
    assert next_sample_name("") == ""


# --- storage -------------------------------------------------------------------------------------
@pytest.fixture
def stores(tmp_path):
    target = project_target(tmp_path)
    survey = SurveyStore(target)
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        db.execute("CREATE TABLE drillhole_assay (hole_id TEXT, depth_from REAL, depth_to REAL, gold REAL, "
                   "silver REAL)")
        db.con.commit()
        survey.prepare(db)
        notes = PitGeologyStore(target).prepare(db)
    assert "made table survey_geo_samples" in notes
    pid = survey.add_pit("Decline", start_station="E", known=(1000.0, 5000.0, 250.0))
    survey.save_setups(pid, [_leg("E", "P1-01", 0.0), _leg("P1-01", "P1-02", 0.0)])
    survey.recompute()
    return survey, PitGeologyStore(target), pid


def test_geology_is_saved_per_pit_and_deleted_with_the_pit(stores):
    survey, store, pid = stores
    assert store.assay_fields() == ["gold", "silver"]           # the drillholes' assay fields
    geo = PitGeology([GeoInterval(0, 12.5, "all", "SH", "grey shale", "", "", "2026-11-25", "Bat")],
                     [GeoStructure(8, "left", 1.5, "fault", 135, 60, 0.3, "clay")],
                     [GeoSample("PS-001", 0, 1, "left", 1.0, "channel", 10, 3, 4.2, {"gold": 1.25})])
    store.save(pid, geo)
    again = store.geology()[pid]
    assert again.intervals[0].rock == "SH" and again.intervals[0].geologist == "Bat"
    assert again.structures[0].dip_direction == 135 and again.structures[0].fill == "clay"
    assert again.samples[0].values == {"gold": 1.25, "silver": None}
    # saving again replaces the pit's records
    store.save(pid, PitGeology([GeoInterval(0, 5, "left", "ST")]))
    again = store.geology(pid)[pid]
    assert len(again.intervals) == 1 and not again.structures and not again.samples
    # an assay field added / removed
    assert store.add_field("cu_pct") == "cu_pct" and store.assay_fields()[-1] == "cu_pct"
    with pytest.raises(ValueError):
        store.add_field("from_m")
    with pytest.raises(ValueError):
        store.add_field("2x")
    store.remove_field("cu_pct")
    assert "cu_pct" not in store.assay_fields()
    survey.delete_pit(pid)
    assert store.geology() == {}


def test_the_layer_draws_the_geology_and_identify_describes_it(stores):
    import pyvista as pv

    from dfpredict.scene.loaders import load_database_survey
    from dfpredict.scene.scene import Scene

    survey, store, pid = stores
    store.save(pid, PitGeology([GeoInterval(0, 10, "all", "SH"), GeoInterval(10, 30, "right", "QV")],
                               [GeoStructure(5, "left", 1.5, "fault", 135, 60)],
                               [GeoSample("PS-1", 10, 11, "right", 1.0, values={"gold": 2.5})]))
    lay, errors = load_database_survey(store.target.path.parent)
    assert not errors and lay.data.geology[pid].intervals[1].rock == "QV"
    try:
        plotter = pv.Plotter(off_screen=True, window_size=(400, 300))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no OpenGL available: {e}")
    sc = Scene(plotter)
    try:
        sc.add_layer(lay)
        kinds = [k for k, _p, _i in lay._geo_items]
        assert kinds == ["interval", "interval", "structure", "sample"]
        meshes = [a.mapper.dataset for a in lay.actors if getattr(a, "_geoviewer_solid", False)]
        rocks = next(m for m in meshes if m.n_cells and m.get_cell(0).n_points == 4 and len(set(m["geo"])) == 2)
        assert rocks.n_cells == 3 + 2                    # all walls (3 bands) + the right wall past station P1-01
        # Identify of the sample
        class Hit:
            mesh = pv.PolyData(np.zeros((1, 3)))
            local = np.zeros(3)

            @staticmethod
            def cell_value(name):
                return 3
        Hit.mesh.cell_data["geo"] = np.array([3])
        got = lay.identify(Hit)
        rows = dict((k, v) for k, v in got.rows)
        assert got.title == "Sample PS-1 in Decline" and rows["gold"] == "2.5" and rows["From"].startswith("10.00 m")
        # switched off: not drawn
        lay.show_geology = lay.show_structures = lay.show_samples = False
        lay.rebuild()
        assert lay._geo_items == []
        settings = lay.settings()
        assert settings["show_geology"] is False and settings["structure_size"] == 1.0
    finally:
        plotter.close()


# --- the window ----------------------------------------------------------------------------------------
@pytest.fixture
def window(stores, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from dfpredict.ui.pit_geology import PitGeologyWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    survey, store, pid = stores
    w = PitGeologyWindow(store.target)
    assert w.start(pid)
    yield w, store, pid, warned, app
    w.dirty = False
    w.close()


def _cell(grid, key):
    return next(j for j, c in enumerate(grid.cols) if c.key == key)


def test_window_records_a_pit_and_saves_it(window):
    w, store, pid, warned, _app = window
    assert w.pid == pid and "0 m = entrance station E" in w.origin.text()
    assert w.stations_m == pytest.approx({"E": 0.0, "P1-01": 20.0, "P1-02": 40.0})
    g = w.g_rocks
    g.add_row()
    assert g.values()[0]["frm"] == 0.0 and g.values()[0]["wall"] == "all"
    g.set_cell(0, _cell(g, "to"), g.parse(g.cols[_cell(g, "to")], "P1-01+2.5"))     # typed from a station
    g.set_cell(0, _cell(g, "rock"), "SH")
    g.add_row()                                                                       # carries on: From = last To
    assert g.values()[1]["frm"] == pytest.approx(22.5)
    g.set_cell(1, _cell(g, "to"), 30.0)
    g.set_cell(1, _cell(g, "rock"), "ST")
    s = w.g_structs
    s.add_row()
    s.set_cell(0, _cell(s, "at"), 8.0)
    s.set_cell(0, _cell(s, "kind"), "fault")
    s.set_cell(0, _cell(s, "dip_direction"), s.parse(s.cols[_cell(s, "dip_direction")], "135 30 00"))
    s.set_cell(0, _cell(s, "dip"), 60.0)
    assert s.table.item(0, _cell(s, "dip_direction")).text() == "135.5"
    sm = w.g_samples
    assert [c.key for c in sm.cols if c.key.startswith("a:")] == ["a:gold", "a:silver"]
    sm.add_row()
    sm.set_cell(0, _cell(sm, "name"), "PS-009")
    sm.set_cell(0, _cell(sm, "a:gold"), 3.5)
    sm.add_row()                                                                       # the next number, same length
    assert sm.values()[1]["name"] == "PS-010" and sm.values()[1]["frm"] == 1.0 and sm.values()[1]["to"] == 2.0
    assert w.dirty and w.tabs.tabText(0) == "Rocks (2)"
    assert w.save() and not warned
    geo = store.geology(pid)[pid]
    assert [(r.frm, r.to, r.rock) for r in geo.intervals] == [(0.0, 22.5, "SH"), (22.5, 30.0, "ST")]
    assert geo.structures[0].dip_direction == pytest.approx(135.5)
    assert geo.samples[0].values["gold"] == 3.5 and geo.samples[1].name == "PS-010"
    # the sketch shows it; a click on a record selects its row
    w.sketch.resize(900, 300)
    img = w.sketch.grab()
    assert not img.isNull() and w.sketch.layout_info["boxes"]
    w._sketch_picked("sample", 1)
    assert w.tabs.currentWidget() is sm and sm.table.currentRow() == 1
    # the page for printing / the image
    page = w.page_image(dpi=60)
    assert page.width() > 600


def test_window_refuses_to_save_mistakes(window):
    w, store, pid, warned, _app = window
    g = w.g_rocks
    g.add_row()
    g.set_cell(0, _cell(g, "to"), 10.0)                 # no rock chosen
    assert not w.save() and "choose the rock" in warned[-1]
    assert store.geology(pid) == {}
    with pytest.raises(ValueError):
        g.parse(g.cols[_cell(g, "frm")], "Q9+2")       # no such station


# --- Excel -------------------------------------------------------------------------------------------
def _fill(path, sheets: dict) -> None:
    """Write rows into a workbook's sheets: name -> [row values], below the header."""
    import openpyxl

    wb = openpyxl.load_workbook(path)
    for name, rows in sheets.items():
        ws = wb[name]
        for r, values in enumerate(rows, start=2):
            for c, v in enumerate(values, start=1):
                ws.cell(r, c).value = v                    # (ws.cell(r, c, None) would not clear it)
    wb.save(path)


def test_excel_template_import_checks_and_export_round_trip(stores, tmp_path):
    import openpyxl

    from dfpredict.io import pit_geology_excel as pe

    survey, store, pid = stores
    tpl = tmp_path / "tpl.xlsx"
    pe.save_template(store, tpl)
    wb = openpyxl.load_workbook(tpl)
    assert {"rocks", "structures", "samples", "stations"} <= set(wb.sheetnames)
    heads = [c.value for c in wb["samples"][1]]
    assert heads[:3] == ["pit", "sample", "from_m"] and {"gold", "silver"} <= set(heads)
    st = [[c.value for c in row] for row in wb["stations"].iter_rows(min_row=2)]
    assert st == [["Decline", "P1", "E", 0.0], ["Decline", "P1", "P1-01", 20.0], ["Decline", "P1", "P1-02", 40.0]]

    # filled in: by name and by code, metres and station + metres, Chinese names of walls / types,
    # a new assay column (copper); one mistake per kind first
    rocks = [["Decline", 0, "P1-01+2.5", "两壁一顶", "SH", "grey shale"],
             ["P1", "P1-01+2.5", 30, "left", "ST"],
             ["P1", 25, 35, "Left wall", "GR"]]                        # overlaps row 3 on the left wall
    structs = [["Decline", "P1-01-2", "左壁", 1.5, "断层", 135, 60, 0.3, "clay"],
               ["Decline", 12, "back", None, "joint", "200 30 00", 95]]  # dip above 90
    heads_s = heads + ["copper"]
    samples = [dict(zip(heads_s, ["Decline", "PS-001", 0, 1.5, "left", 1.0, "channel", 10, 3, 4.2])),
               dict(zip(heads_s, ["Decline", "PS-002", 1.5, 3, "left"])),
               dict(zip(heads_s, ["Nowhere", "PS-003", 0, 1, "left"]))]     # no such pit
    samples[0].update(gold=1.25, copper=0.4)
    samples[1].update(gold="<0.01")                                     # not a number
    wb2 = openpyxl.load_workbook(tpl)
    wb2["samples"].cell(1, len(heads) + 1, "copper")
    wb2.save(tpl)
    _fill(tpl, {"rocks": rocks, "structures": structs,
                "samples": [[s.get(h) for h in heads_s] for s in samples]})
    plan = pe.read_import(store, tpl)
    errors, _w = plan.problems(True)
    text = "\n".join(errors)
    assert "Rocks rows 3 and 4 overlap" in text                        # Excel rows
    assert "Structures row 3" in text and "dip must be 0° to 90°" in text
    assert "pit Nowhere is not in the database" in text
    assert "row 3: column gold: not a number" in text
    assert plan.new_fields == ["copper"]
    with pytest.raises(ValueError):
        pe.apply_import(store, plan, True)
    assert store.geology() == {}

    # corrected
    rocks[2][1] = 30
    structs[1][6] = 85
    samples[1]["gold"] = 0.005
    del samples[2]
    _fill(tpl, {"rocks": rocks, "structures": structs,
                "samples": [[s.get(h) for h in heads_s] for s in samples] + [[None] * len(heads_s)]})
    plan = pe.read_import(store, tpl)
    assert plan.problems(True)[0] == []
    assert "Records to import: rocks 3, structures 2, samples 2" in plan.summary(True)
    counts = pe.apply_import(store, plan, True)
    assert counts["added"] == 1 and counts["samples"] == 2
    geo = store.geology()[pid]
    assert [(r.frm, r.to, r.wall, r.rock) for r in geo.intervals] == [
        (0, 22.5, "all", "SH"), (22.5, 30, "left", "ST"), (30, 35, "left", "GR")]
    assert (geo.structures[1].at, geo.structures[1].wall, geo.structures[1].kind) == (18, "left", "fault")
    assert geo.structures[0].dip_direction == pytest.approx(200.5)
    assert geo.samples[0].values == {"gold": 1.25, "silver": None, "copper": 0.4}
    assert "copper" in store.assay_fields()

    # the export reads back as it is; a second import of only rocks keeps the structures and samples
    out = tmp_path / "out.xlsx"
    counts = pe.export_excel(store, out)
    assert counts == {"rocks": 3, "structures": 2, "samples": 2, "stations": 3}
    again = pe.read_import(store, out)
    assert again.problems(True) == ([], []) and not again.pits[0].new
    assert again.problems(False)[0] == [] and again.todo(False) == []           # skipped
    wb3 = openpyxl.load_workbook(out)
    for name in ("structures", "samples"):
        ws = wb3[name]
        ws.delete_rows(2, ws.max_row)
    wb3["rocks"].delete_rows(3, 2)
    wb3["rocks"].cell(2, 3, 35)                                                   # one rock 0 - 35 m
    wb3.save(out)
    pe.apply_import(store, pe.read_import(store, out), True)
    geo = store.geology()[pid]
    assert len(geo.intervals) == 1 and len(geo.structures) == 2 and len(geo.samples) == 2

    # a sample number of another pit is refused; the database changing after the check is caught
    other = survey.add_pit("Crosscut", start_station="X", known=(1100.0, 5000.0, 250.0))
    _fill(out, {"samples": [["Crosscut", "PS-001", 0, 1, "right"]]})
    plan = pe.read_import(store, out)
    assert any("sample PS-001 is already in pit Decline" in e for e in plan.problems(True)[0])
    _fill(out, {"samples": [["Crosscut", "PS-101", 0, 1, "right"]]})
    plan = pe.read_import(store, out)
    store.save(other, PitGeology([GeoInterval(0, 1, "all", "SH")]))
    with pytest.raises(ValueError, match="changed since"):
        pe.apply_import(store, plan, True)


def test_window_exports_and_imports_excel(window, tmp_path, monkeypatch):
    from dfpredict.io import pit_geology_excel as pe

    w, store, pid, warned, _app = window
    store.save(pid, PitGeology([GeoInterval(0, 10, "all", "SH")], [], [GeoSample("PS-1", 0, 1, "left")]))
    w.reload()
    path = str(tmp_path / "pg")
    assert w.export_excel(path) and (tmp_path / "pg.xlsx").exists() and not warned
    assert "rocks 1" in w.status.text()
    plan = pe.read_import(store, path + ".xlsx")
    assert plan.problems(True) == ([], [])


def _wait(app, cond, timeout=60.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        app.processEvents()
        time.sleep(0.02)
    return cond()


def test_pit_geology_from_the_main_window_shows_in_the_view(demo, tmp_path, monkeypatch):
    import shutil

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QMessageBox

    import dfpredict.ui.main_window as mw
    from dfpredict.scene.survey import SurveyLayer

    app = QApplication.instance() or QApplication([])
    (tmp_path / "s.ini").write_text("[database]\nbackend=sqlite\n[workspace]\nrestore_last=false\n",
                                    encoding="utf-8")
    monkeypatch.setattr(mw, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[1:]}"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    ws = tmp_path / "ws"
    ws.mkdir()
    shutil.copy2(demo / "geodatabase.sqlite", ws / "geodatabase.sqlite")
    win = mw.MainWindow()
    win.quiet_errors = True
    try:
        win.open_workspace(ws, confirm=False)

        def survey():
            return next((lay for lay in win.scene.layers if isinstance(lay, SurveyLayer)), None)
        assert _wait(app, lambda: not win.loads_pending and not win._threads and survey() is not None)
        win.open_pit_geology(2)                          # a branch: 0 m at its junction station
        dlg = win.pit_geology
        assert dlg is not None and dlg.pid == 2 and dlg.stations_m[dlg.axis.names[0]] == 0.0
        assert "junction station P1-17" in dlg.origin.text()
        g = dlg.g_rocks
        g.add_row()
        g.set_cell(0, _cell(g, "to"), 15.0)
        g.set_cell(0, _cell(g, "rock"), "QV1")
        assert dlg.save()
        assert _wait(app, lambda: survey() is not None and 2 in survey().data.geology and not win._threads)
        assert survey().data.geology[2].intervals[0].rock == "QV1"
        assert any(k == "interval" for k, _p, _i in survey()._geo_items)
        win.open_pit_geology(1)                          # open already: shows the other pit
        assert dlg.pid == 1
    finally:
        if win.pit_geology is not None:
            win.pit_geology.dirty = False
        win.close()
