"""Tests of the drilling risk system: database (predefined data, entry checks), rock mechanics, geological
model, trajectory, rules, generator, end-to-end prediction on the demo workspace, and the windows."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dfpredict.io.database import Target
from dfpredict.risk import codes, rockmech
from dfpredict.risk.schema import COLLAR, RUN, SURVEY
from dfpredict.risk.store import RiskStore, default_reference, validate

RISK_DEMO = Path(__file__).resolve().parents[1] / "demo_data" / "sb3_workspace"


@pytest.fixture()
def store(tmp_path) -> RiskStore:
    s = RiskStore(Target("sqlite", path=tmp_path / "geodatabase.sqlite"))
    s.prepare()
    return s


# --- database ---------------------------------------------------------------------------------------
def test_new_database_has_predefined_data(store):
    d = store.load_all()
    assert len(d["rock"]) == len(rockmech.DEFAULTS) >= 40
    assert len(d["measure"]) == len(codes.DEFAULT_MEASURES)
    assert len(d["rock_code"]) == len(codes.DEFAULT_ROCK_CODES)
    assert len(d["rig"]) >= 1
    # preparing again neither duplicates nor replaces user edits
    rock = d["rock"].copy()
    rock.loc[rock["rock_class"] == "dolomite", "ucs"] = 180.0
    store.save_reference("rock", rock)
    store.prepare()
    d2 = store.load_all()
    assert len(d2["rock"]) == len(rock)
    assert float(d2["rock"].set_index("rock_class").loc["dolomite", "ucs"]) == 180.0


def test_project_database_setup_makes_risk_tables(tmp_path):
    from dfpredict.project_db import init_project_database

    t = Target("sqlite", path=tmp_path / "geodatabase.sqlite")
    notes = init_project_database(t, tmp_path)
    assert any("risk_rock" in n for n in notes)
    assert len(RiskStore(t).read("measure")) == len(codes.DEFAULT_MEASURES)


def test_edited_rock_properties_are_used(store):
    from dfpredict.risk.features import props_for

    rock = store.read("rock")
    rock.loc[rock["rock_class"] == "schist", ["ucs", "cai"]] = [260.0, 5.2]
    store.save_reference("rock", rock)
    store.load_all()                                   # applies the database's properties
    p = props_for("SCH", None, {})
    assert p.ucs == 260.0 and p.cai == 5.2 and p.hard == 5 and p.abr == 5
    store.save_reference("rock", pd.DataFrame(default_reference()["rock"]))
    store.load_all()
    assert props_for("SCH", None, {}).ucs == rockmech.DEFAULTS["schist"].ucs_typ


def test_save_hole_round_trip_and_rename(store):
    collar = {"hole_id": "H1", "area_id": "A", "x": 1.0, "y": 2.0, "z": 3.0, "planned_depth": 100.0,
              "final_depth": 30.0, "result": "in_progress"}
    sv = pd.DataFrame({"depth": [0.0, 30.0], "azimuth": [10.0, 11.0], "dip": [-70.0, -69.0]})
    store.save_hole(collar, {"survey": sv})
    h = store.hole("H1")
    assert len(h["survey"]) == 2 and h["collar"].iloc[0]["result"] == "in_progress"
    store.save_hole(dict(collar, hole_id="H2"), {}, old_id="H1")
    assert len(store.hole("H2")["survey"]) == 2 and not len(store.hole("H1")["collar"])


def test_entry_checks():
    runs = pd.DataFrame({"hole_id": ["H"] * 3, "depth_from": [0.0, 3.0, 7.0], "depth_to": [3.0, 6.0, 9.0],
                         "rpm": [700, 5000, 700]})
    bad, issues = validate(RUN, runs, {"max_rpm": 1300})
    msgs = " ".join(i.message for i in issues)
    assert "must equal the previous run" in msgs and "above the rig maximum" in msgs
    assert bad == {1, 2}
    _bad, iss = validate(SURVEY, pd.DataFrame({"hole_id": ["H"], "depth": [10.0], "azimuth": [400.0], "dip": [-60.0]}))
    assert any(i.column == "azimuth" for i in iss)
    _bad, iss = validate(COLLAR, pd.DataFrame([{"hole_id": "H", "result": "nope"}]))
    assert any("required" in i.message for i in iss) and any("unknown code" in i.message for i in iss)


def test_csv_round_trip(store, tmp_path):
    collar = {"hole_id": "H1", "area_id": "A", "x": 1, "y": 2, "z": 3, "planned_depth": 50, "final_depth": 50,
              "result": "success"}
    store.save_hole(collar, {"geology": pd.DataFrame({"depth_from": [0.0], "depth_to": [50.0], "rock": ["DO"]})})
    out = tmp_path / "csv"
    store.export_csv(out)
    s2 = RiskStore(Target("sqlite", path=tmp_path / "other.sqlite"))
    s2.prepare()
    counts, issues = s2.import_csv(out)
    assert counts["collar"] == 1 and not [i for i in issues if i.level == "error"]
    assert s2.hole("H1")["geology"].iloc[0]["rock"] == "DO"


# --- the drillhole tables shared with GeoViewer (io/drilling_schema.py) -----------------------------------
def test_schema_matches_the_shared_drillhole_tables():
    """Every column of the records the risk system keeps in a shared table is a column of the shared contract
    (exchange format), so the programs cannot drift apart."""
    from dfpredict.io import drilling_schema as ds
    from dfpredict.risk.schema import ALL_TABLES

    for t in ALL_TABLES:
        if not t.shared:
            assert t.sql_name.startswith("risk_") and t.name in ("rock", "rock_code", "warning")
            continue
        sch = ds.table(t.sql_name)
        assert t.sql_name == sch.name
        missing = set(t.names) - set(ds.exchange_names(sch))
        assert not missing, (t.name, missing)


def test_records_are_geoviewers_drillholes(store, tmp_path):
    """A hole saved in the Drilling Records is a drillhole of GeoViewer's Drillhole Manager (dip 0 = down,
    max_depth, state), and what only GeoViewer writes survives a save."""
    from dfpredict.io import drilling_schema as ds
    from dfpredict.io.drillhole_store import DrillholeStore

    gv = DrillholeStore(store.target)
    gv.add_hole("H1", {"x": "1", "y": "2", "z": "3", "max_depth": "50", "state": "2", "organization": "Survey team"})
    with store.target.connect() as db:
        db.executemany("INSERT INTO drillhole_assay (hole_id, depth_from, depth_to, gold) VALUES (?, ?, ?, ?)",
                       [("H1", 0, 1, 2.5)])
        db.executemany("INSERT INTO drillhole_geology (hole_id, depth_from, depth_to, rock, note) VALUES (?, ?, ?, ?, ?)",
                       [("H1", 0, 50, "DO", "vuggy")])
        db.con.commit()
    rec = store.hole("H1")
    c = rec["collar"].iloc[0].to_dict()
    assert c["result"] == "success" and c["final_depth"] == 50.0 and c["state"] == 2
    c.update(area_id="A", plan_dip=-70.0, planned_depth=60.0)
    geo = rec["geology"].assign(formation="F1")
    assert geo["notes"].tolist() == ["vuggy"]
    store.save_hole({k: v for k, v in c.items() if k != "state"}, {
        "geology": geo, "survey": pd.DataFrame({"depth": [0.0], "azimuth": [90.0], "dip": [-70.0]}),
        "run": pd.DataFrame({"depth_from": [0.0], "depth_to": [3.0], "rpm": [800.0]})})
    h = gv.load_hole("H1")
    assert h.collar["organization"] == "Survey team" and h.collar["state"] == ds.ORE_HIT   # kept: ore hit
    assert h.collar["plan_dip"] == 20.0 and h.collar["area_id"] == "A"                  # 0 = down
    assert h.tables["drillhole_survey"]["dip"].tolist() == [20.0]
    assert h.tables["drillhole_geology"][["rock", "note", "formation"]].values.tolist() == [["DO", "vuggy", "F1"]]
    assert len(h.tables["drillhole_assay"]) == 1 and len(h.tables["drillhole_run"]) == 1
    store.save_hole({**{k: v for k, v in c.items() if k != "state"}, "result": "failed", "final_depth": 30.0}, {})
    assert gv.load_hole("H1").collar["state"] == ds.FAILED and gv.load_hole("H1").collar["max_depth"] == 30.0
    assert store.summary()["result"] == {"failed": 1}
    store.delete_hole("H1")
    assert gv.hole_ids() == []                                                          # every table


def test_records_of_an_earlier_version_are_moved_in(tmp_path):
    """A database with the risk_* tables of an earlier version: the same records after the move."""
    from dfpredict.io.database import Database

    path = tmp_path / "geodatabase.sqlite"
    collar = pd.DataFrame({"hole_id": ["A1", "A2"], "area_id": ["A", "A"], "x": [1.0, 2.0], "y": [1.0, 2.0],
                           "z": [5.0, 5.0], "plan_azimuth": [90.0, 0.0], "plan_dip": [-60.0, -90.0],
                           "planned_depth": [100.0, 80.0], "final_depth": [100.0, 41.5],
                           "result": ["success", "failed"]})
    with Database.sqlite(path) as db:
        db.write_frame("risk_collar", collar)
        db.write_frame("risk_survey", pd.DataFrame({"hole_id": ["A1"], "depth": [0.0], "azimuth": [90.0],
                                                    "dip": [-60.0]}))
        db.write_frame("risk_failure", pd.DataFrame({"hole_id": ["A2"], "fail_depth": [41.5], "fail_code": ["CL"],
                                                     "severity": [4]}))
    s = RiskStore(Target("sqlite", path=path))
    assert any("moved the drilling records of 2 holes" in n for n in s.prepare())
    d = s.load_all()
    c = d["collar"].set_index("hole_id")
    assert c.loc["A2", "result"] == "failed" and c.loc["A2", "final_depth"] == 41.5
    assert c.loc["A1", "plan_dip"] == -60.0 and d["survey"]["dip"].tolist() == [-60.0]
    assert d["failure"]["severity"].tolist() == ["4"]
    with Database.sqlite(path, create=False) as db:
        assert "risk_collar_old" in db.tables() and "risk_collar" not in db.tables()


def test_model_control_is_the_workspace_area_only(tmp_path):
    from dfpredict.risk.engine import RiskEngine

    s = RiskStore(Target("sqlite", path=tmp_path / "geodatabase.sqlite"))
    s.prepare()
    for hid, area, x, res in (("H1", "SB", 5000.0, "success"), ("H2", "SB", 5100.0, "planned"),
                              ("P1", "OLD", 100.0, "success")):
        s.save_hole({"hole_id": hid, "area_id": area, "x": x, "y": 1.0, "z": 1.0, "planned_depth": 50.0,
                     "final_depth": 0.0 if res == "planned" else 50.0, "result": res}, {})
    eng = RiskEngine(s.target, tmp_path)
    assert len(eng.data.collar) == 3 and eng.area_id == "SB"     # the area of the planned hole
    assert eng.control_xy().tolist() == [[5000.0, 1.0]]          # drilled holes of the area, not the history


# --- rock mechanics -------------------------------------------------------------------------------------
def test_rock_property_table_is_consistent():
    for k, p in rockmech.DEFAULTS.items():
        assert p.ucs[0] <= p.ucs[1] <= p.ucs[2], k
        assert p.cai[0] <= p.cai[1] <= p.cai[2], k
        assert 0 <= p.poisson <= 0.5 and 1.0 <= p.density <= 5.0, k
    assert rockmech.strength_class(20) == 1 and rockmech.strength_class(300) == 5
    assert rockmech.abrasivity_class(0.5) == 1 and rockmech.abrasivity_class(4.5) == 5
    assert codes.rock_class_of("BIF") == "ironstone" and codes.rock_class_of("Mica schist") == "schist"


# --- geological model, trajectory, rules ------------------------------------------------------------------
@pytest.fixture(scope="module")
def sb3_model():
    if not RISK_DEMO.exists():
        pytest.skip("demo workspace not built (tools/make_demo_workspace.py)")
    from dfpredict.risk.geomodel import build_model, load_config, surfaces_from_files

    pr = [(p, "horizon") for p in sorted((RISK_DEMO / "model").glob("horizon_*.dtm"))]
    pr += [(p, "fault") for p in sorted((RISK_DEMO / "faults").glob("F*.dtm"))]
    return build_model(surfaces_from_files(pr), load_config(RISK_DEMO))


def test_geomodel_units_and_faults(sb3_model):
    m = sb3_model
    assert m.units == ["U1", "U2", "U3", "U4", "U5", "U6"] and len(m.faults) == 4
    z = np.arange(310, 0, -5.0)
    pts = np.column_stack([np.full_like(z, 5500.0), np.full_like(z, 8450.0), z])
    idx = m.unit_index(pts)
    assert idx[0] == 0 and idx[-1] == 5 and np.all(np.diff(idx) >= 0)       # top to bottom, in order
    assert m.uncertainty(pts).min() >= m.config["u_near"] - 1e-9


def test_predicted_path_follows_the_survey_trend():
    from dfpredict.risk.trajectory import predict_paths

    sv = pd.DataFrame({"depth": [0, 30, 60, 90], "azimuth": [100, 101, 102, 103], "dip": [-70, -69, -68, -67]})
    ens = predict_paths("H", (0, 0, 0), 100, -70, 300, sv, 90, n_paths=20, seed=1)
    assert ens.build_rate > 0.5 and ens.turn_rate > 0.5
    az, dip = ens.azimuth_dip(len(ens.md) - 1)
    assert dip > -67 and az > 103                     # keeps flattening and turning right
    assert ens.radius[-1] > ens.radius[100] >= 0
    plan = predict_paths("P", (0, 0, 0), 45, -60, 200, None, 0, n_paths=5)
    assert abs(plan.azimuth_dip(0)[1] + 60) < 1e-6


def test_rules():
    from dfpredict.risk.rules import evaluate

    runs = pd.DataFrame({"depth_from": [0, 3, 6, 9, 12], "depth_to": [3, 6, 9, 12, 15],
                         "rpm": [800] * 5, "rop": [3, 3, 3, 3, 12], "flow_in": [40] * 5,
                         "pump_press": [2.0, 2.0, 2.0, 1.3, 1.3], "return_pct": [95, 95, 95, 60, 0],
                         "rod_drop": [0, 0, 0, 0, 0.6]})
    hits = {(h.rule, h.current) for h in evaluate(runs)}
    assert ("R1", False) in hits and ("R2", True) in hits and ("R7", True) in hits and ("R6", False) in hits


def test_generator_failure_rate():
    from dfpredict.risk.demo import generate_history

    f = generate_history(n_per_area=12, seed=3)
    rate = (f["collar"]["result"] == "failed").mean()
    assert 0.05 < rate < 0.35
    assert set(f["failure"]["fail_code"]) <= set(codes.FAIL_CODES)
    assert f["run"]["depth_from"].notna().all()


# --- end to end on the demo workspace ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    if not (RISK_DEMO / "drillrisk" / "models").exists():
        pytest.skip("demo workspace / models not built (tools/make_demo_workspace.py)")
    from dfpredict.risk.engine import RiskEngine
    from dfpredict.risk.geomodel import surfaces_from_files

    ws = tmp_path_factory.mktemp("ws")
    for name in ("geodatabase.sqlite", "drillrisk"):
        src = RISK_DEMO / name
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, ws / name)
    eng = RiskEngine(Target("sqlite", path=ws / "geodatabase.sqlite"), ws)
    pr = [(p, "horizon") for p in sorted((RISK_DEMO / "model").glob("horizon_*.dtm"))]
    pr += [(p, "fault") for p in sorted((RISK_DEMO / "faults").glob("F*.dtm"))]
    eng.set_surfaces(surfaces_from_files(pr))
    return eng


def test_predict_ongoing_hole(engine):
    p = engine.predict("DH-2026-001", n_paths=16)
    assert 0.0 < p.p_reach < 1.0
    assert p.drilled_depth > 100 and p.planned_depth == 320
    assert any(c == "R-M02" for c, _t in p.messages)                  # tied to the logged U1/U2 contact
    assert any(name == "F4" for _d, name, _s in p.crossings)
    ahead = [z for z in p.zones if z.status != "Drilled"]
    assert ahead, "the dolomite / F4 interval ahead should be a warning zone"
    z = max(ahead, key=lambda z: z.p)
    assert z.reasons and z.prevent and all(s.code in codes.MEASURES for s in z.prevent)
    rp = p.range_probabilities(p.drilled_depth, p.planned_depth)
    assert 0 <= rp["failure"] <= 1 and rp["fault_crossed"] > 0.5


def test_live_rule_on_pressure_drop(engine):
    p = engine.predict("DH-2026-002", n_paths=12)
    assert any(r.rule == "R1" and r.current for r in p.rules)
    at_bit = [z for z in p.zones if z.status == "At bit"]
    assert at_bit and codes.level_rank(max(at_bit, key=lambda z: z.p).level) >= 2


def test_plan_check_and_options(engine):
    design = {"hole_id": "PLAN-T", "area_id": "SB3", "rig_model": "XY-44", "x": 5450.0, "y": 8420.0, "z": 316.0,
              "plan_azimuth": 300.0, "plan_dip": -70.0, "planned_depth": 300.0, "final_depth": 0.0, "result": "planned"}
    p = engine.predict(design=design, n_paths=10)
    assert p.drilled_depth == 0 and 0 < p.p_reach <= 1
    opts = engine.predictor.plan_options(design, d_az=(-30, 0, 30), d_dip=(0,), n_paths=6)
    assert len(opts) == 3 and opts["is_design"].sum() == 1


# --- windows ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_records_dialog(qapp, engine):
    from dfpredict.ui.history_dialog import DrillingRecordsDialog

    dlg = DrillingRecordsDialog(engine.target, engine.folder)
    dlg.reload()
    assert dlg.list.topLevelItemCount() > 1000
    assert dlg.list.topLevelItem(0).data(0, 0x0100) in ("DH-2026-001", "DH-2026-002")   # live holes first
    dlg.load_hole("DH-2026-001", force=True)
    n_runs = dlg.tables["run"].model_.rowCount()
    dlg.next_run()
    assert dlg.tables["run"].model_.rowCount() == n_runs + 1 and dlg.dirty
    assert dlg.check()                                  # the copied run continues the last one
    dlg.load_hole("DH-2026-001", force=True)
    assert dlg.ref_tables["rock"].model_.rowCount() == len(rockmech.ROCKS)
    dlg.close()


def test_prediction_and_admin_dialogs(qapp, engine):
    from dfpredict.ui.prediction_dialog import PredictionDialog, report_html
    from dfpredict.ui.risk_admin import ModelsDialog

    dlg = PredictionDialog(engine)
    p = engine.predict("DH-2026-001", n_paths=10)
    dlg.set_prediction(p)
    assert dlg.table.rowCount() == len(p.zones)
    if p.zones:
        dlg.select_zone(0)
        assert "Why this zone is dangerous" in dlg.detail.toHtml()
    assert "Risk register" in report_html(p, dlg.chart.png_base64(), 0, 100)
    md = ModelsDialog(engine)
    assert "Validation" in md.report.toHtml()
    dlg.close()
    md.close()
