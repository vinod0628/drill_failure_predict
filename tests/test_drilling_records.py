"""The drilling records shared with the Drilling Failure Prediction system (io/drilling_schema.py):
the tables and fields made, hole state and drilling result, the exchange format, writes that keep
the other program's columns, the move of an earlier risk_* database, the entry checks, the CSV
drilling history, the area of the drillhole layer and the Drillhole Manager's drilling tabs."""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.io import drilling_schema as ds
from dfpredict.io.database import Database, project_target
from dfpredict.io.drillhole_store import COLLAR, GEOLOGY, SURVEY, DrillholeStore, check_hole


def _store(folder) -> DrillholeStore:
    folder.mkdir(exist_ok=True)
    store = DrillholeStore(project_target(folder))
    store.prepare()
    return store


def _db(folder, create=False):
    return Database.sqlite(folder / "geodatabase.sqlite", create=create)


def test_prepare_makes_the_drilling_tables_and_fields_once(tmp_path):
    store = DrillholeStore(project_target(tmp_path))
    notes = store.prepare()
    for t in ds.NEW_TABLES:
        assert f"made table {t.name}" in notes
    assert "filled drillhole_measure with 37 predefined rows" in notes
    assert "filled drillhole_rig with 2 predefined rows" in notes
    with _db(tmp_path) as db:
        assert [c.name for c in ds.COLLAR_ADDED] == db.columns(COLLAR)[-len(ds.COLLAR_ADDED):]
        assert {"formation", "weathering", "broken", "bedding_angle"} <= set(db.columns(GEOLOGY))
        idx = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'index'")}
        trig = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert {f"idx_{t.name}_hole" for t in ds.RECORD_TABLES} <= idx
    assert store.prepare() == []                                   # nothing to do the second time
    from dfpredict.project_db import init_project_database

    init_project_database(project_target(tmp_path), tmp_path)
    with _db(tmp_path) as db:
        trig = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert "gv_drillhole_run_ai" in trig                            # Refresh sees changes of the records
    assert store.measures()[0]["code"] == "M-OP-01" and len(store.measures()) == 37


def test_state_and_result_and_the_exchange_format():
    assert list(ds.HOLE_STATES) == [ds.PLANNED, ds.IN_WORK, ds.ORE_HIT, ds.NO_ORE_HIT, ds.FAILED]
    assert [ds.result_of_state(s) for s in (1, 2, 3, 4, 5, 6, None, 9)] == [
        "in_progress", "success", "success", "failed", "planned", "success", None, None]   # 6: an earlier version's
    assert ds.state_of_result("success") == ds.NO_ORE_HIT                                  # ore hit not known
    assert ds.state_of_result("success", current=ds.ORE_HIT) == ds.ORE_HIT     # ore hit is kept
    assert ds.state_of_result("failed", current=ds.ORE_HIT) == ds.FAILED
    assert ds.state_of_result("nonsense") is None
    stored = pd.DataFrame({"hole_id": ["A", "B"], "max_depth": [120.0, 0.0], "state": [2, 5], "plan_dip": [20.0, 0.0]})
    ex = ds.to_exchange(ds.COLLAR, stored)
    assert ex["final_depth"].tolist() == [120.0, 0.0] and ex["result"].tolist() == ["success", "planned"]
    assert ex["plan_dip"].tolist() == [-70.0, -90.0]                           # negative downward
    back = ds.from_exchange(ds.COLLAR, ex.drop(columns="state"), {"A": 2})
    assert back["state"].tolist() == [2, 5] and back["plan_dip"].tolist() == [20.0, 0.0]
    assert "result" not in back and "max_depth" in back
    geo = ds.to_exchange(ds.GEOLOGY, pd.DataFrame({"hole_id": ["A"], "note": ["core"]}))
    assert list(geo.columns) == ["hole_id", "notes"]


def test_writes_keep_the_columns_of_the_other_program(tmp_path):
    store = _store(tmp_path)
    store.add_hole("H1", {"x": "1", "y": "2", "z": "3", "max_depth": "50", "organization": "Survey team"})
    with _db(tmp_path) as db:
        db.execute(f"ALTER TABLE {GEOLOGY} ADD COLUMN alteration TEXT")
        db.executemany(f"INSERT INTO {GEOLOGY} (hole_id, depth_from, depth_to, rock, alteration) VALUES (?, ?, ?, ?, ?)",
                       [("H1", 0, 20, "SH", "sericite"), ("H1", 20, 50, "DO", "none")])
        db.con.commit()
        with db.transaction():
            ds.upsert_collar(db, {"hole_id": "H1", "area_id": "N1", "state": ds.IN_WORK, "planned_depth": 300})
            ds.write_hole_rows(db, ds.GEOLOGY, "H1", pd.DataFrame(
                {"depth_from": [0.0, 20.0, 50.0], "depth_to": [20.0, 50.0, 60.0], "rock": ["SH", "DO", "DO"],
                 "formation": ["F1", "F2", "F2"]}))
    h = store.load_hole("H1")
    assert h.collar["organization"] == "Survey team" and h.collar["area_id"] == "N1" and h.collar["state"] == 1
    geo = h.tables[GEOLOGY]
    assert geo["alteration"].tolist() == ["sericite", "none", None]           # same depths: kept
    assert geo["formation"].tolist() == ["F1", "F2", "F2"]


def _legacy(folder):
    """A database of an earlier dfpredict: GeoViewer's tables plus its own risk_* tables."""
    store = _store(folder)
    store.add_hole("DH1", {"x": "10", "y": "20", "z": "300", "max_depth": "139", "state": "2",
                           "organization": "Benchmark"})
    with _db(folder) as db:
        db.executemany(f"INSERT INTO {SURVEY} (hole_id, depth, azimuth, dip) VALUES (?, ?, ?, ?)",
                       [("DH1", 0, 180, 20.0)])
        db.executemany(f"INSERT INTO {GEOLOGY} (hole_id, depth_from, depth_to, rock) VALUES (?, ?, ?, ?)",
                       [("DH1", 0, 50, "U1"), ("DH1", 50, 139, "U2")])
        db.con.commit()
        db.write_frame("risk_collar", pd.DataFrame({
            "hole_id": ["DH1", "A01-001"], "area_id": ["SB3", "A01"], "x": [10.0, 500.0], "y": [20.0, 600.0],
            "z": [300.0, 610.0], "rig_model": ["XY-44", "XY-44"], "plan_azimuth": [180.0, 90.0],
            "plan_dip": [-70.0, -60.0], "planned_depth": [139.0, 400.0], "final_depth": [94.5, 400.0],
            "result": ["failed", "success"], "notes": [None, "history"]}))
        db.write_frame("risk_survey", pd.DataFrame({"hole_id": ["DH1", "A01-001", "A01-001"], "depth": [0.0, 0.0, 30.0],
                                                    "azimuth": [180.0, 90.0, 91.0], "dip": [-70.0, -60.0, -61.0]}))
        db.write_frame("risk_geology", pd.DataFrame({
            "hole_id": ["DH1", "A01-001"], "depth_from": [0.0, 0.0], "depth_to": [50.0, 400.0], "rock": ["SS", "LS"],
            "formation": ["U1", "Lime"], "weathering": [1, 0], "broken": ["N", "Y"], "bedding_angle": [70.0, 45.0],
            "notes": [None, "karst"]}))
        db.write_frame("risk_run", pd.DataFrame({"hole_id": ["DH1", "DH1"], "depth_from": [0.0, 3.0],
                                                 "depth_to": [3.0, 6.0], "rpm": [800.0, 820.0]}))
        db.write_frame("risk_failure", pd.DataFrame({"hole_id": ["DH1"], "fail_depth": [94.5], "fail_code": ["SP"],
                                                     "severity": [5]}))
        db.write_frame("risk_area", pd.DataFrame({"area_id": ["SB3", "A01"], "name": ["Here", "Past"]}))
        db.write_frame("risk_measure", pd.DataFrame({"code": ["M-OP-01"], "title": ["Edited"], "text": ["My text"],
                                                     "failures": ["CV"], "stage": ["P"], "expert_order": [0]}))
        db.write_frame("risk_rock", pd.DataFrame({"rock_class": ["dolomite"], "name": ["Dolomite"]}))
    return DrillholeStore(project_target(folder))


def test_an_earlier_risk_database_is_moved_into_the_shared_tables(tmp_path):
    store = _legacy(tmp_path)
    notes = store.prepare()
    assert any("moved the drilling records of 2 holes" in n for n in notes)
    assert any("the state of 1 holes was set" in n for n in notes)
    dh1 = store.load_hole("DH1")
    c = dh1.collar
    assert c["organization"] == "Benchmark" and c["x"] == 10.0                   # GeoViewer's values kept
    assert c["max_depth"] == 94.5                                     # failed there: the recorded outcome wins
    assert c["state"] == ds.FAILED and c["area_id"] == "SB3" and c["plan_dip"] == pytest.approx(20.0)
    assert dh1.tables[GEOLOGY]["rock"].tolist() == ["U1", "U2"]                 # its own log stays ...
    assert dh1.tables[GEOLOGY]["formation"].tolist() == ["U1", None]            # ... filled where it matches
    assert len(dh1.tables[SURVEY]) == 1 and len(dh1.tables[ds.RUN.name]) == 2
    assert dh1.tables[ds.FAILURE.name]["fail_code"].tolist() == ["SP"]
    a = store.load_hole("A01-001")
    assert a.collar["state"] == ds.NO_ORE_HIT and a.collar["notes"] == "history"
    assert a.tables[SURVEY]["dip"].tolist() == [30.0, 29.0]                     # 0 = down
    assert a.tables[GEOLOGY]["note"].tolist() == ["karst"]
    meas = {m["code"]: m for m in store.measures()}
    assert meas["M-OP-01"]["text"] == "My text" and len(meas) == 37             # the edited measure replaces the default
    with _db(tmp_path) as db:
        tables = set(db.tables())
    assert {"risk_collar_old", "risk_run_old", "risk_measure_old"} <= tables
    assert "risk_collar" not in tables and "risk_rock" in tables                # the predictor's own data stays
    assert store.prepare() == []                                                # once
    assert store.choices()["main_rock"] == {"dolomite": "Dolomite"}            # rock classes of risk_rock


def test_checks_of_the_drilling_records(tmp_path):
    store = _store(tmp_path)
    store.add_hole("P1", {"x": "1", "y": "2", "z": "3"})
    ci = store.info(COLLAR)
    run = store.info(ds.RUN.name)
    collar = {"x": "1", "y": "2", "z": "3", "max_depth": "0", "state": "5", "plan_dip": "200"}
    errors, _w = check_hole(ci, collar, {})
    assert errors == ["Collar: Planned dip = 200.0 is outside 0 - 180 °"]       # 0 m is fine for a planned hole
    errors, _w = check_hole(ci, {**collar, "state": "6", "plan_dip": ""}, {})
    assert errors == ["Collar: the hole length must be more than 0"]
    runs = pd.DataFrame({"depth_from": ["0", "3", "5"], "depth_to": ["3", "6", "9"], "rpm": ["800", "1500", ""]})
    errors, warnings = check_hole(ci, {**collar, "state": "4", "max_depth": "8", "plan_dip": ""},
                                  {ds.RUN.name: (run, runs), ds.FAILURE.name: (store.info(ds.FAILURE.name),
                                                                               pd.DataFrame())},
                                  context={"max_rpm": 1300})
    assert any("RPM 1500.0 is above the rig maximum 1300" in e for e in errors)
    assert any("row 3: Run From 5.0 m must equal the previous run's To 6.0 m (overlap)" in e for e in errors)
    assert any("below the hole's depth" in w for w in warnings)
    assert any("record its failure" in w for w in warnings)
    assert any("For the drilling history fill in: Area" in w for w in warnings)


def test_drilling_history_csv_round_trip(tmp_path):
    store = _legacy(tmp_path / "a")
    store.prepare()
    out = tmp_path / "csv"
    counts = store.export_drilling_csv(out, ["A01"])
    assert counts["collar"] == 1 and counts["survey"] == 2 and counts["area"] == 2
    collar = pd.read_csv(out / "collar.csv")
    assert collar.loc[0, "result"] == "success" and collar.loc[0, "final_depth"] == 400.0
    assert pd.read_csv(out / "survey.csv")["dip"].tolist() == [-60.0, -61.0]     # the exchange format
    # a bad row is skipped and reported
    runs = pd.DataFrame({"hole_id": ["A01-001"] * 2, "depth_from": [0, 3], "depth_to": [3, 2], "rpm": [900, 900]})
    runs.to_csv(out / "run.csv", index=False)
    other = _store(tmp_path / "b")
    other.add_hole("KEEP", {"x": "1", "y": "1", "z": "1", "max_depth": "5", "area_id": "A01"})
    counts, issues = other.import_drilling_csv(out)
    assert counts["collar"] == 1 and counts["run"] == 1
    assert [i.message for i in issues if i.level == "error"] == ["To (2) must be greater than From (3)"]
    h = other.load_hole("A01-001")
    assert h.collar["state"] == ds.NO_ORE_HIT and h.tables[SURVEY]["dip"].tolist() == [30.0, 29.0]
    assert "KEEP" in other.hole_ids()                                          # merge keeps other holes
    counts, _ = other.import_drilling_csv(out, replace_areas=True)
    assert counts["removed"] == 1 and "KEEP" not in other.hole_ids()            # same area, not in the files


def test_drillhole_layer_shows_the_area_being_drilled(tmp_path):
    from dfpredict.io import database
    from dfpredict.scene.loaders import load_database_drillholes

    store = _legacy(tmp_path)
    store.prepare()
    store.add_hole("NEW1", {"x": "11", "y": "21", "z": "300", "state": "5", "area_id": "SB3"})
    database.configure(database.DatabaseConfig(backend="sqlite"))
    layer, errors = load_database_drillholes(tmp_path)
    assert errors == []
    assert len(layer.all_data.collars) == 3 and layer.shown_areas() == ["SB3"]
    assert sorted(layer.data.hole_ids) == ["DH1", "NEW1"]
    assert set(layer.data.surveys["hole_id"]) == {"DH1"} and "run" in layer.data.intervals
    layer.apply_settings({"area": layer.AREA_ALL})
    assert len(layer.data.collars) == 3
    layer.apply_settings({"area": "A01"})
    assert layer.data.hole_ids == ["A01-001"] and layer.settings()["area"] == "A01"


def test_manager_edits_the_drilling_records(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from dfpredict.ui.drillhole_manager import DrillholeManager

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    store = _legacy(tmp_path)
    dlg = DrillholeManager(store)
    assert dlg.start()
    assert dlg.area_filter.count() == 3                                          # all, A01, SB3
    dlg.area_filter.setCurrentIndex(dlg.area_filter.findData("A01"))
    assert dlg.count.text() == "1 of 2 holes"
    dlg._show_hole("DH1")
    titles = [dlg.hole_tabs.tabText(i) for i in range(dlg.hole_tabs.count())]
    assert titles[4:] == ["Runs (2)", "Hole size", "Bit", "Occurrences", "Failure (1)"]
    labels = [dlg.drill_form.itemAt(i, dlg.drill_form.ItemRole.LabelRole).widget().text()
              for i in range(dlg.drill_form.rowCount())]
    assert labels[:3] == ["Area", "Rig model", "Planned azimuth (°)"]
    run = dlg.drill_grids[ds.RUN.name]
    run.table.item(1, run.columns.index("rpm")).setText("820")
    run.next_row()                                                               # From = 6, rpm copied
    rows = run.frame()
    assert rows.iloc[-1]["depth_from"] == "6" and rows.iloc[-1]["rpm"] == "820"
    run.table.item(2, run.columns.index("depth_to")).setText("9")
    fail = dlg.drill_grids[ds.FAILURE.name]
    assert fail.frame()["fail_code"].tolist() == ["SP"]                         # the code, shown as its text
    assert fail.table.item(0, fail.columns.index("fail_code")).text() == "SP - Stuck rods / stuck pipe"
    assert dlg.save_hole()
    assert store.load_hole("DH1").tables[ds.RUN.name]["depth_to"].tolist() == [3.0, 6.0, 9.0]
    # the reference tables
    g = dlg.ref_grids[ds.AREA.name]
    g.add_row()
    g.table.item(2, g.columns.index("area_id")).setText("N2")
    assert dlg.save_reference(ds.AREA.name)
    assert store.reference(ds.AREA.name)["area_id"].tolist() == ["A01", "N2", "SB3"]
    assert "N2" in store.choices()["area_id"]
    dlg.dirty = False
    dlg.close()
