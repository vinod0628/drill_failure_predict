"""The project database on a real MySQL server.

Skipped unless GEOVIEWER_TEST_MYSQL names a server and an empty database the test may
fill and clear:  host:port:user:password:database   (e.g. 127.0.0.1:3306:geo:pw:geoviewer_test)
"""

import os

import numpy as np
import pandas as pd
import pytest

from dfpredict.io import database
from dfpredict.io.database import DatabaseConfig

SPEC = os.environ.get("GEOVIEWER_TEST_MYSQL", "")
pytestmark = pytest.mark.skipif(not SPEC, reason="GEOVIEWER_TEST_MYSQL not set (no MySQL server to test)")


@pytest.fixture()
def mysql():
    host, port, user, password, name = SPEC.split(":", 4)
    cfg = DatabaseConfig("mysql", host, int(port), name, user, password)
    database.create_mysql_database(cfg)
    with database.Database.mysql(cfg) as db:                     # start empty
        for t in db.tables():
            if t.lower().startswith(("drillhole_", "prod_", "survey_", "gvnew", "gvold")):
                db.execute(f"DROP TABLE {db.q(t)}")
        db.con.commit()
    database.configure(cfg)
    yield cfg


def test_check_connection(mysql):
    text = database.check_connection(mysql)
    assert text.startswith("Connected to MySQL 8") and "no GeoViewer tables yet" in text




def test_production_on_mysql_and_import_from_sqlite(mysql, tmp_path):
    from dfpredict.scene.production import MinedShape, ProductionLog

    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    # records kept in geodatabase.sqlite while the setting was SQLite ...
    database.configure(DatabaseConfig(backend="sqlite"))
    old = ProductionLog(tmp_path)
    old.reserve = {"solid": "ore1", "grade": "gold"}
    old.put(MinedShape(1, "2026-11-01", "Cut 1", "xy", sq, 200.0, 205.5, "first"))
    # ... are copied to MySQL the first time the folder opens with MySQL
    database.configure(mysql)
    log = ProductionLog.load(tmp_path)
    assert log.db.is_mysql and log.imported_from == str(tmp_path / "geodatabase.sqlite")
    assert [(s.id, s.name, s.note) for s in log.shapes] == [(1, "Cut 1", "first")]
    log.put(MinedShape(log.next_id(), "2026-11-02", "Cut 2 100% 'ore'", "yz", sq, 1.0, 4.0))
    log.put(MinedShape(1, "2026-11-01", "Cut 1b", "xy", sq[:3], 200.0, 206.0))
    back = ProductionLog.load(tmp_path)
    assert back.imported_from is None
    assert [(s.id, s.name, len(s.points)) for s in back.shapes] == [(1, "Cut 1b", 3), (2, "Cut 2 100% 'ore'", 4)]
    assert back.reserve == {"solid": "ore1", "grade": "gold"}
    back.remove(2)
    assert [s.id for s in ProductionLog.load(tmp_path).shapes] == [1]
    with back.db.connect() as con:
        assert con.query("SELECT id FROM prod_mined_shapes") == [(1,)]
        assert con.scalar("SELECT JSON_LENGTH(outline) FROM prod_mined_shapes WHERE id = 1") == 3


def test_init_makes_every_table_and_keeps_data(mysql, tmp_path):
    import pandas as pd

    from dfpredict.project_db import init_project_database

    target = database.project_target(tmp_path)
    notes = init_project_database(target, tmp_path)
    assert "made table drillhole_collar" in notes and "made the production tables" in notes
    with target.connect() as db:
        names = {t.lower() for t in db.tables()}
        assert {"drillhole_collar", "drillhole_survey", "drillhole_geology", "drillhole_assay", "drillhole_styles",
                "drillhole_translation", "prod_settings", "prod_mined_shapes"} <= names
        db.execute("INSERT INTO drillhole_collar (hole_id, x, y, z, max_depth) VALUES ('A1', 1, 2, 3, 40)")
        db.con.commit()
        db.execute("ALTER TABLE drillhole_assay DROP COLUMN gold")       # the user removed a field
        db.write_frame("drillhole_extra", pd.DataFrame({"hole_id": ["A1"], "note": ["keep"]}))
    # nothing removed; only the new table gets its change tracking
    assert init_project_database(target, tmp_path) == ["added change tracking (3 triggers)"]
    assert init_project_database(target, tmp_path) == []                  # nothing missing, nothing changed
    with target.connect() as db:
        assert db.query("SELECT hole_id, max_depth FROM drillhole_collar") == [("A1", 40.0)]
        assert db.columns("drillhole_assay") == ["hole_id", "depth_from", "depth_to", "silver"]
        assert db.query("SELECT * FROM drillhole_extra") == [("A1", "keep")]


def test_a_missing_mysql_database_is_an_error_not_made(mysql, tmp_path):
    from dataclasses import replace

    from dfpredict.project_db import init_project_database

    cfg = replace(mysql, database=mysql.database + "_typo")
    with pytest.raises(database.DatabaseError, match="has no database 'geoviewer_test_typo'"):
        init_project_database(database.Target("mysql", config=cfg), None)
    with database.Database.mysql(cfg, database=False) as db:
        assert cfg.database not in {r[0] for r in db.query("SHOW DATABASES")}


def test_survey_on_mysql(mysql, tmp_path):
    """Pits, a branch, total station and compass setups stored on MySQL and computed; the survey
    layer is read from it."""
    from dfpredict.io.survey_store import SetupRow, SurveyStore
    from dfpredict.scene.loaders import load_database_survey, survey_signature

    store = SurveyStore(database.project_target(tmp_path))
    store.prepare()
    store.prepare()                                              # again: nothing to add, nothing lost
    pid = store.add_pit("Main", start_station="E", known=(1000.0, 5000.0, 250.0))
    store.set_settings({"declination": -7.5, "convergence": 1.25})
    store.save_setups(pid, [
        SetupRow("E", 1.5, bs_bearing=45.0, fs="P1-01", fs_ha=120.5, fs_za=95.0, fs_sd=50.0, fs_ht=1.4),
        SetupRow("P1-01", 1.45, fs="P1-02", fs_bearing=169.0, fs_bearing_type="magnetic", fs_za=92.0, fs_sd=40.0,
                 fs_ht=1.45)])
    bid = store.add_pit("Level 2", parent_id=pid, start_station="P1-01")
    store.save_setups(bid, [SetupRow("P1-01", 1.5, bs_station="E", bs_ha=0.0, fs="P1A-01", fs_ha=90.0, fs_za=90.0,
                                     fs_sd=10.0, fs_ht=1.5)])
    st = {s.name: s for s in store.stations()}
    assert (st["P1-02"].x, st["P1-02"].y, st["P1-02"].z) == pytest.approx((1056.426, 4937.096, 244.346), abs=1e-3)
    assert st["P1A-01"].x is not None and [p.code for p in store.pits()] == ["P1", "P1A"]
    assert store.setups()[pid][1].method == "bearing"
    layer, errors = load_database_survey(tmp_path)
    assert not errors and len(layer.data.pits) == 2 and layer.db_signature == survey_signature(layer.db_target)
