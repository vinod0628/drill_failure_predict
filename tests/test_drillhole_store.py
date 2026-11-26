"""Drillhole Manager storage: tables, holes, rock colours, assay fields, checks."""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.io.database import Database, project_target
from dfpredict.io.drillhole_store import ASSAY, COLLAR, GEOLOGY, STYLES, SURVEY, DrillholeStore, check_hole


INFO_FIELDS = ["organization", "drilling_manager", "geology_surveyor", "start_date", "end_date", "state"]
DRILLING_FIELDS = ["area_id", "rig_model", "plan_azimuth", "plan_dip", "planned_depth", "casing_available",
                   "fluids_available", "crew", "notes"]           # the drilling records (drilling_schema.py)
GEOLOGY_DRILLING = ["formation", "weathering", "broken", "bedding_angle"]


def _imported(folder):
    """Tables as a user imported them from Excel: their own column names, Surpac-style styles."""
    with Database.sqlite(folder / "geodatabase.sqlite") as db:
        db.write_frame(COLLAR, pd.DataFrame({"BHID": ["H1", "H2"], "EAST": [100.0, 200.0], "NORTH": [10.0, 20.0],
                                             "RL": [50.0, 60.0], "EOH": [30.0, 40.0], "section": ["7120", "7160"]}))
        db.write_frame(SURVEY, pd.DataFrame({"BHID": ["H1", "H2"], "depth": [0.0, 0.0], "azimuth": [0.0, 90.0],
                                             "dip": [0.0, 30.0]}))
        db.write_frame(GEOLOGY, pd.DataFrame({"BHID": ["H1", "H1", "H2"], "from": [0.0, 10.0, 0.0],
                                              "to": [10.0, 30.0, 40.0], "rock": ["SH", "QV1", "ST"]}))
        db.write_frame(STYLES, pd.DataFrame({
            "code": ["", "SH", "QV1", "7120"], "field_name": ["gold", "rock", "rock", "section"],
            "table_name": ["sample", "geology", "geology", "collar"],
            "graphics_colour": ["red", "r=0.0 g=0.6 b=1.0", "blue", "cyan"], "line_style": ["|", "|", "|", "|"]}))
        db.write_frame("drillhole_translation", pd.DataFrame({
            "code": ["SH", "QV1"], "description": ["Shale", "Quartz vein"], "field_name": ["rock", "rock"],
            "table_name": ["geology", "geology"]}))
    return DrillholeStore(project_target(folder))


def test_prepare_adds_what_is_missing_and_keeps_everything(tmp_path):
    store = _imported(tmp_path)
    notes = store.prepare()
    assert f"made table {ASSAY}" in notes
    assert f"added field rock to {STYLES}" in notes and f"added field color to {STYLES}" in notes
    assert f"filled rock / color of 2 {STYLES} rows from the Surpac colours" in notes
    with Database.sqlite(tmp_path / "geodatabase.sqlite", create=False) as db:
        # the Surpac columns and the non-rock rows stay; rock / color added and filled
        assert db.columns(STYLES) == ["code", "field_name", "table_name", "graphics_colour", "line_style",
                                      "rock", "color"]
        assert db.scalar(f"SELECT COUNT(*) FROM {STYLES}") == 4
        assert db.columns(ASSAY) == ["hole_id", "depth_from", "depth_to", "gold", "silver"]
        idx = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {f"idx_{t}_hole" for t in (COLLAR, SURVEY, GEOLOGY, ASSAY)} <= idx
    assert store.styles() == {"SH": "#0099ff", "QV1": "#0000ff"}          # geology rock rows only
    assert store.prepare() == []                                          # nothing to do the second time
    # an empty database gets all tables
    empty = DrillholeStore(project_target(tmp_path / "new"))
    (tmp_path / "new").mkdir()
    made = [n for n in empty.prepare() if n.startswith("made table")]
    assert len(made) == 6 + 9 and empty.hole_ids() == []               # + the drilling record tables


def test_init_never_removes_tables_fields_or_rows(tmp_path):
    """A database that is set up and holds data: the init only adds missing tables / fields."""
    from dfpredict.project_db import init_project_database

    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        db.write_frame(COLLAR, pd.DataFrame({"BHID": ["A"], "EAST": [1.0], "NORTH": [2.0], "RL": [3.0],
                                             "EOH": [50.0], "contractor": ["X Drilling"]}))
        db.write_frame(GEOLOGY, pd.DataFrame({"BHID": ["A"], "from": [0.0], "to": [5.0]}))      # no rock field
        db.write_frame(ASSAY, pd.DataFrame({"hole_id": ["A"], "depth_from": [0.0], "depth_to": [1.0],
                                            "cu_pct": [0.8]}))                                # no gold / silver
        db.write_frame("drillhole_notes", pd.DataFrame({"BHID": ["A"], "text": ["keep me"]}))
        db.write_frame("my_own_table", pd.DataFrame({"a": [1, 2, 3]}))
    target = project_target(tmp_path)
    notes = init_project_database(target, tmp_path)
    assert f"added field rock to {GEOLOGY}" in notes and f"added field note to {GEOLOGY}" in notes
    added = [n for n in notes if n.startswith("added field")]
    assert not any(f" {f} to {COLLAR}" in n for n in added for f in ("hole_id", "x", "y", "z", "max_depth"))  # BHID ...
    assert not any(ASSAY in n for n in added)
    assert "made the production tables" in notes
    with Database.sqlite(tmp_path / "geodatabase.sqlite", create=False) as db:
        assert db.columns(COLLAR) == ["BHID", "EAST", "NORTH", "RL", "EOH", "contractor"] + INFO_FIELDS +             DRILLING_FIELDS
        assert db.query(f"SELECT * FROM {COLLAR}") == [("A", 1.0, 2.0, 3.0, 50.0, "X Drilling") + (None,) * 15]
        assert db.columns(ASSAY) == ["hole_id", "depth_from", "depth_to", "cu_pct"]      # gold not added back
        assert db.query(f"SELECT * FROM {ASSAY}") == [("A", 0.0, 1.0, 0.8)]
        assert db.query(f"SELECT * FROM {GEOLOGY}") == [("A", 0.0, 5.0) + (None,) * 6]   # rock, note, ... added
        assert db.query("SELECT * FROM drillhole_notes") == [("A", "keep me")]
        assert db.query("SELECT COUNT(*) FROM my_own_table") == [(3,)]
    assert init_project_database(target, tmp_path) == []                                # all there now
    # a production table missing a field gets it, rows kept
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db, db.transaction():
        db.execute("INSERT INTO prod_mined_shapes (id, mined_on, name, view_plane, depth_from, depth_to, outline) "
                   "VALUES (1, '2026-01-01', 'a', 'xy', 0, 1, '[[0,0],[1,0],[1,1]]')")
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        db.execute("ALTER TABLE prod_mined_shapes DROP COLUMN note")
    assert init_project_database(target, tmp_path) == ["added field note to prod_mined_shapes"]
    with Database.sqlite(tmp_path / "geodatabase.sqlite", create=False) as db:
        assert db.query("SELECT id, name, note FROM prod_mined_shapes") == [(1, "a", None)]


def test_holes_keep_the_imported_column_names(tmp_path):
    store = _imported(tmp_path)
    store.prepare()
    assert store.hole_ids() == ["H1", "H2"]
    h = store.load_hole("H1")
    assert h.collar == {"EAST": 100.0, "NORTH": 10.0, "RL": 50.0, "EOH": 30.0, "section": "7120",
                        **dict.fromkeys(INFO_FIELDS + DRILLING_FIELDS)}    # information fields added, empty
    assert h.tables[GEOLOGY].to_dict("list") == {"from": [0.0, 10.0], "to": [10.0, 30.0], "rock": ["SH", "QV1"],
                                                 "note": [None, None],            # note added by prepare
                                                 **{c: [None, None] for c in GEOLOGY_DRILLING}}
    assert list(h.tables[ASSAY].columns) == ["depth_from", "depth_to", "gold", "silver"]
    # edit: longer hole, a new geology split, assays, rename to H1A
    h.collar["EOH"] = "35"
    h.tables[GEOLOGY] = pd.DataFrame({"from": ["10", "0", "30"], "to": ["30", "10", "35"],
                                      "rock": ["QV1", "SH", "ST"],            # unsorted: saved by depth
                                      "note": ["vuggy quartz, pyrite 2 %", "", None]})
    h.tables[ASSAY] = pd.DataFrame({"depth_from": [0, 2], "depth_to": [2, 4], "gold": ["1.5", None],
                                    "silver": [None, "12"]})
    assert store.save_hole("H1", h, new_id="H1A") == "H1A"
    assert store.hole_ids() == ["H1A", "H2"]
    back = store.load_hole("H1A")
    assert back.collar["EOH"] == 35.0
    assert back.tables[GEOLOGY]["rock"].tolist() == ["SH", "QV1", "ST"]
    assert back.tables[GEOLOGY]["note"].tolist() == [None, "vuggy quartz, pyrite 2 %", None]   # empty -> NULL
    assert back.tables[SURVEY]["dip"].tolist() == [0.0]                       # renamed in every table
    a = back.tables[ASSAY]
    assert a["gold"].tolist()[0] == 1.5 and pd.isna(a["gold"].tolist()[1]) and a["silver"].tolist()[1] == 12.0
    with pytest.raises(ValueError, match="already"):
        store.save_hole("H1A", back, new_id="H2")
    # add and remove
    store.add_hole("H10", {"EAST": "1", "NORTH": "2", "RL": "3", "EOH": "50"})
    assert store.hole_ids() == ["H1A", "H2", "H10"]                            # natural order
    with pytest.raises(ValueError, match="already"):
        store.add_hole("H2", {})
    assert store.remove_hole("H1A") == 1 + 1 + 3 + 2                           # collar, survey, geology, assay
    assert store.hole_ids() == ["H2", "H10"]


def test_rock_colours_and_codes(tmp_path):
    store = _imported(tmp_path)
    store.prepare()
    store.set_style("ST", "#AA5500")
    store.set_style("SH", "#123456")
    store.set_style("UM", "#00ff00")                                           # a code not used yet
    store.remove_style("QV1")
    assert store.styles() == {"SH": "#123456", "ST": "#aa5500", "UM": "#00ff00"}
    assert store.rock_codes() == [("QV1", "Quartz vein", 1), ("SH", "Shale", 1), ("ST", "", 1), ("UM", "", 0)]
    # the layer draws rock with these colours
    from dfpredict.io.drillholes import read_drillholes_sqlite
    from dfpredict.scene.layers import DrillholeLayer

    db = read_drillholes_sqlite(project_target(tmp_path))
    assert db.rock_colors["SH"] == pytest.approx((0x12 / 255, 0x34 / 255, 0x56 / 255))
    lay = DrillholeLayer(name="d", path=None, data=db)
    assert lay.table == "geology" and lay.field_name == "rock"
    assert lay.code_color("ST") == pytest.approx((0xAA / 255, 0x55 / 255, 0.0))
    lay.category_colors["SH"] = (1.0, 1.0, 1.0)                               # e.g. saved in the workspace
    assert lay.code_color("SH") == pytest.approx((0x12 / 255, 0x34 / 255, 0x56 / 255))   # database wins
    assert len(lay.code_color("QV1")) == 3                                     # no colour: automatic


def test_assay_fields(tmp_path):
    store = _imported(tmp_path)
    store.prepare()
    store.add_field("copper")
    assert store.assay_fields() == ["gold", "silver", "copper"]
    store.rename_field("silver", "ag_ppm")
    store.remove_field("gold")
    assert store.assay_fields() == ["ag_ppm", "copper"]
    for bad in ("1abc", "a b", "depth_from", ""):
        with pytest.raises(ValueError):
            store.add_field(bad)
    with pytest.raises(ValueError, match="already"):
        store.add_field("COPPER")
    with pytest.raises(ValueError, match="not an assay field"):
        store.remove_field("hole_id")


def test_checks_before_saving(tmp_path):
    store = _imported(tmp_path)
    store.prepare()
    ci = store.info(COLLAR)
    tables = {
        SURVEY: (store.info(SURVEY), pd.DataFrame({"depth": ["0", "0", "50"], "azimuth": ["10", "400", "10"],
                                                   "dip": ["0", "30", "200"]})),
        GEOLOGY: (store.info(GEOLOGY), pd.DataFrame({"from": ["0", "8", "20", None], "to": ["10", "15", "18", "5"],
                                                     "rock": ["SH", "ST", "QV1", "SH"]})),
        ASSAY: (store.info(ASSAY), pd.DataFrame({"depth_from": ["0"], "depth_to": ["45"], "gold": ["abc"],
                                                 "silver": [None]})),
    }
    errors, warnings = check_hole(ci, {"EAST": "1", "NORTH": "", "RL": "3", "EOH": "40"}, tables)
    assert "Collar: NORTH must be a number" in errors
    assert "Survey row 2: depth 0 is in the table twice" in errors
    assert "Survey row 2: azimuth must be 0 to 360" in errors
    assert "Survey row 3: dip must be 0 (straight down) to 180 (straight up)" in errors
    assert "Geology row 3: From must be 0 or more and less than To (20 - 18)" in errors
    assert "Geology row 4: From and To must be numbers" in errors
    assert "Assay row 1: gold must be a number (or empty)" in errors
    assert "Geology: rows 1 and 2 overlap (0-10 and 8-15)" in warnings
    assert "Survey row 3: depth 50 is past the end of the hole (40 m)" in warnings
    assert "Assay row 1: 45 m is past the end of the hole (40 m)" in warnings
    ok, _w = check_hole(ci, {"EAST": "1", "NORTH": "2", "RL": "3", "EOH": "40"},
                        {GEOLOGY: (store.info(GEOLOGY), pd.DataFrame({"from": ["0"], "to": ["10"], "rock": ["SH"]}))})
    assert ok == []


def test_manager_window_edits_and_saves_a_hole(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from dfpredict.ui.drillhole_manager import DrillholeManager

    app = QApplication.instance() or QApplication([])
    store = _imported(tmp_path)
    dlg = DrillholeManager(store)
    changed = []
    dlg.changed.connect(lambda: changed.append(1))
    assert dlg.start() and changed                                             # tables were prepared
    assert [dlg.holes.item(i).text() for i in range(dlg.holes.count())] == ["H1", "H2"]
    assert dlg.hid == "H1" and dlg.g_geology.table.rowCount() == 2
    assert dlg.t_rocks.rowCount() == 3                                         # QV1, SH, ST
    # edit a rock code and its note, add an assay row, then save
    assert dlg.g_geology.columns == ["from", "to", "rock", "note"] + GEOLOGY_DRILLING
    dlg.g_geology.table.item(1, 2).setText("ST")
    dlg.g_geology.table.item(1, 3).setText("sheared, chlorite")
    assert dlg.dirty and dlg.b_save.isEnabled()
    dlg.hole_tabs.setCurrentWidget(dlg.g_assay)
    dlg.g_assay.add_row()
    t = dlg.g_assay.table
    for c, v in enumerate(["0", "2", "3.25", ""]):
        t.item(0, c).setText(v)
    n = len(changed)
    assert dlg.save_hole() and not dlg.dirty and len(changed) == n + 1
    h = store.load_hole("H1")
    assert h.tables[GEOLOGY]["rock"].tolist() == ["SH", "ST"]
    assert h.tables[GEOLOGY]["note"].tolist() == [None, "sheared, chlorite"]
    assert h.tables[ASSAY]["gold"].tolist() == [3.25]
    # an invalid row stops the save (the warning box is answered automatically)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    dlg.g_geology.table.item(0, 1).setText("-5")
    assert not dlg.save_hole() and dlg.dirty
    dlg._show_hole("H1")                                                       # discard
    assert not dlg.dirty
    dlg.close()
    app.processEvents()


def test_table_updates_counts_every_change(tmp_path):
    """Triggers made by the set-up count each insert / update / delete, whoever makes it; Refresh
    compares the versions (a changed value is seen although the row count stays the same)."""
    from dfpredict.io.drillholes import drillhole_signature
    from dfpredict.project_db import init_project_database
    from dfpredict.scene.production import MinedShape, ProductionLog, production_signature

    store = _imported(tmp_path)
    target = project_target(tmp_path)
    notes = init_project_database(target, tmp_path)
    assert "made table table_updates" in notes and any(n.startswith("added change tracking") for n in notes)
    with Database.sqlite(tmp_path / "geodatabase.sqlite", create=False) as db:
        names = {r[0] for r in db.query("SELECT table_name FROM table_updates")}
        assert {COLLAR, SURVEY, GEOLOGY, ASSAY, STYLES, "prod_mined_shapes", "prod_settings"} <= names
    dh0, prod0 = drillhole_signature(target), production_signature(target)
    assert dh0[0] == "tracked" and prod0[0] == "tracked"
    # an outside tool changes one value (same number of rows)
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db, db.transaction():
        db.execute(f"UPDATE {GEOLOGY} SET rock = 'ST' WHERE BHID = 'H1' AND \"from\" = 10")
    dh1 = drillhole_signature(target)
    assert dh1 != dh0 and production_signature(target) == prod0          # only the drillholes changed
    with Database.sqlite(tmp_path / "geodatabase.sqlite", create=False) as db:
        assert db.scalar(f"SELECT version FROM table_updates WHERE table_name = '{GEOLOGY}'") == 1
    # GeoViewer's own writes are counted the same way
    store.set_style("ST", "#123456")
    assert drillhole_signature(target) != dh1
    ProductionLog.load(tmp_path).put(MinedShape(1, "2026-01-01", "a", "xy", [[0, 0], [1, 0], [1, 1]], 0, 1))
    assert production_signature(target) != prod0
    # a table replaced by an import tool (its triggers gone): the layout part changes, the set-up re-adds them
    before = drillhole_signature(target)
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        db.write_frame(SURVEY, pd.DataFrame({"BHID": ["H1"], "depth": [0.0], "azimuth": [5.0], "dip": [0.0]}))
    assert drillhole_signature(target) != before
    assert init_project_database(target, tmp_path) == ["added change tracking (3 triggers)"]
    assert init_project_database(target, tmp_path) == []


def test_notes_are_not_loaded_for_the_display_but_shown_on_identify(tmp_path):
    from dfpredict.io.drillholes import read_drillholes_sqlite
    from dfpredict.scene.layers import DrillholeLayer

    store = _imported(tmp_path)
    store.prepare()
    h = store.load_hole("H1")
    h.tables[GEOLOGY] = pd.DataFrame({"from": ["0", "10"], "to": ["10", "30"], "rock": ["SH", "QV1"],
                                      "note": ["grey shale, laminated", None]})
    store.save_hole("H1", h)
    db = read_drillholes_sqlite(project_target(tmp_path))
    assert "note" not in db.intervals["geology"].columns                 # not read with the drillholes
    assert db.lazy_fields == {"geology": ["note"]}
    assert db.interval_notes("geology", "H1", 0.0, 10.0) == {"note": "grey shale, laminated"}
    assert db.interval_notes("geology", "H1", 10.0, 30.0) == {"note": None}
    lay = DrillholeLayer(name="d", path=None, data=db)
    g = db.intervals["geology"]
    row = g[(g.hole_id == "H1") & (g.depth_from == 0.0)].iloc[0]
    assert ("note", "grey shale, laminated") in lay._interval_rows("geology", row)   # Identify shows it


def test_rock_cells_show_code_and_description_and_save_the_code(tmp_path):
    from PySide6.QtWidgets import QApplication

    from dfpredict.ui.drillhole_manager import DrillholeManager, rock_code, rock_label

    app = QApplication.instance() or QApplication([])
    assert rock_label("SH", "Shale") == "SH (Shale)" and rock_label("ST", "") == "ST"
    assert rock_code("SH (Shale)") == "SH" and rock_code("ST") == "ST"
    store = _imported(tmp_path)
    store.prepare()
    store.set_description("ST", "Schist")
    dlg = DrillholeManager(store)
    assert dlg.start()
    g = dlg.g_geology
    assert [g.table.item(r, 2).text() for r in range(2)] == ["SH (Shale)", "QV1 (Quartz vein)"]
    assert [c for c, _col, _d in g.rock_codes] == ["QV1", "SH", "ST"]           # registered rocks
    g.table.item(1, 2).setText("ST")                                          # picked from the list
    assert g.table.item(1, 2).text() == "ST (Schist)"
    assert g.frame()["rock"].tolist() == ["SH", "ST"]                          # only the code is saved
    dlg.dirty = False
    dlg.close()
    app.processEvents()


def test_export_to_excel(tmp_path):
    import openpyxl

    store = _imported(tmp_path)
    store.prepare()
    store.add_hole("H10", {"EAST": "5", "NORTH": "6", "RL": "7", "EOH": "20"})
    h = store.load_hole("H1")
    h.tables[ASSAY] = pd.DataFrame({"depth_from": ["2", "0"], "depth_to": ["4", "2"], "gold": ["1.5", "0.2"],
                                    "silver": [None, "3"]})
    store.save_hole("H1", h)
    counts = store.export_excel(tmp_path / "holes.xlsx")
    assert counts == {"collar": 3, "geology": 3, "survey": 2, "assay": 2}
    book = openpyxl.load_workbook(tmp_path / "holes.xlsx")
    assert book.sheetnames == ["collar", "geology", "survey", "assay"]
    assert book["collar"].freeze_panes == "A2"
    collar = pd.read_excel(tmp_path / "holes.xlsx", sheet_name="collar")
    assert collar["BHID"].tolist() == ["H1", "H2", "H10"]                      # natural order
    assay = pd.read_excel(tmp_path / "holes.xlsx", sheet_name="assay")
    assert assay[["depth_from", "gold"]].values.tolist() == [[0.0, 0.2], [2.0, 1.5]]   # by depth


def test_collar_information_fields(tmp_path):
    from PySide6.QtWidgets import QApplication

    from dfpredict.io.drillhole_store import HOLE_STATES, state_code, state_text
    from dfpredict.ui.drillhole_manager import DrillholeManager

    store = _imported(tmp_path)
    store.prepare()
    ci = store.info(COLLAR)
    ok = {"EAST": "1", "NORTH": "2", "RL": "3", "EOH": "40"}
    assert list(HOLE_STATES.items()) == [(5, "Planned"), (1, "In work"), (2, "Completed with ore hit"),
                                         (3, "Completed without ore hit"), (4, "Failed")]
    assert state_code(6) == 3 and state_code("Completed") == 3          # "Completed" of an earlier version
    assert [state_code(v) for v in (2, "2", 2.0, "Failed", "in work", "", None, 7, "Done")] == \
        [2, 2, 2, 4, 1, None, None, None, None]
    assert state_text(3) == "Completed without ore hit" and state_text(None) == ""
    with Database.sqlite(tmp_path / "geodatabase.sqlite", create=False) as db:
        assert "SMALLINT" in db.column_types(COLLAR)["state"]
    errors, _w = check_hole(ci, {**ok, "start_date": "2026-03-10", "end_date": "2026-03-02", "state": "Done"}, {})
    assert "Collar: the end date is before the start date" in errors
    assert any(e.startswith("Collar: state must be one of: Planned, In work") for e in errors)
    errors, _w = check_hole(ci, {**ok, "start_date": "10/03/2026"}, {})
    assert errors == ["Collar: start date must be a date like 2026-03-31"]
    assert check_hole(ci, {**ok, "start_date": "2026-03-01", "end_date": "", "state": "1"}, {})[0] == []
    # the Collar tab: text boxes, date pickers (may be empty), the state list
    app = QApplication.instance() or QApplication([])
    dlg = DrillholeManager(store)
    assert dlg.start()
    e = dlg.collar_edits
    assert e["start_date"].value() == "" and e["state"].currentText() == ""
    e["organization"].setText("Mine Exploration Ltd")
    e["drilling_manager"].setText("J. Kim")
    e["geology_surveyor"].setText("S. Park")
    e["start_date"].setDate(e["start_date"].date().fromString("2026-03-01", "yyyy-MM-dd"))
    e["state"].setCurrentText("Completed with ore hit")       # shown as text
    assert e["state"].currentData() == 2
    assert dlg.dirty and dlg.save_hole()
    c = store.load_hole("H1").collar
    assert (c["organization"], c["drilling_manager"], c["geology_surveyor"]) == ("Mine Exploration Ltd", "J. Kim",
                                                                                 "S. Park")
    assert (c["start_date"], c["end_date"], c["state"]) == ("2026-03-01", None, 2)      # stored as its code
    dlg.close()
    app.processEvents()
    sheet = pd.read_excel(_export(store, tmp_path), sheet_name="collar")
    assert sheet.loc[sheet.BHID == "H1", "state"].tolist() == ["Completed with ore hit"]   # text in Excel


def _export(store, tmp_path):
    store.export_excel(tmp_path / "x.xlsx")
    return tmp_path / "x.xlsx"
