"""Reserve management: mined shapes, production.json and the block model depletion."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.io.block_model import BlockModel
from dfpredict.scene.production import (Depletion, MinedShape, ProductionLog, day_of, point_in_polygon,
                                        prism_mesh, reserve_blocks)
from dfpredict.scene.reserve import Reserve, ensure_depletion_attrs, monthly, uses_depletion


def grid_model(nx=10, ny=10, nz=10, size=5.0, sg=2.5, grade=2.0) -> BlockModel:
    """Blocks of ``size`` m filling [0, n*size] on each axis."""
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    c = (np.stack([i, j, k], -1).reshape(-1, 3) + 0.5) * size
    n = len(c)
    return BlockModel(None, c.astype(np.float64), np.full((n, 3), size, np.float32),
                      {"sg": np.full(n, sg, np.float32), "au": np.full(n, grade, np.float32)})


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def depletion(bm, n=None):
    """The depletion as the app makes it (2 x 2 x 2 points per block), or with ``n`` per axis."""
    idx, dens, grade = reserve_blocks(bm, None, "sg", 2.7, "au")
    return Depletion(bm, idx, dens, grade) if n is None else Depletion(bm, idx, dens, grade, n=n)


# --- geometry ---------------------------------------------------------------------------------
def test_point_in_polygon_concave():
    L = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], float)
    pts = np.array([[2, 2], [8, 2], [8, 8], [2, 8], [11, 1]], float)
    assert point_in_polygon(pts[:, 0], pts[:, 1], L).tolist() == [True, True, False, True, False]


def test_prism_volume_matches_area_times_thickness():
    L = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]]
    sh = MinedShape(1, "2026-01-05", "L", "xy", L, 100.0, 106.0)
    assert sh.area() == pytest.approx(64.0)
    m = prism_mesh(sh)
    assert m.volume == pytest.approx(64.0 * 6.0, rel=1e-6)
    assert m.bounds[4] == pytest.approx(100.0) and m.bounds[5] == pytest.approx(106.0)
    # a section-view shape sweeps along Y
    sx = MinedShape(2, "2026-01-05", "S", "xz", square(0, 0, 10, 5), 20.0, 23.0)
    mx = prism_mesh(sx)
    assert mx.bounds[2] == pytest.approx(20.0) and mx.bounds[3] == pytest.approx(23.0)
    assert mx.volume == pytest.approx(150.0)


# --- production records (prod_* tables of geodatabase.sqlite) ---------------------------------------
PROD = {"prod_settings", "prod_mined_shapes"}


def test_log_round_trip_and_backup(tmp_path):
    from dfpredict.io.database import Database

    log = ProductionLog(tmp_path)
    log.reserve = {"grade": "au", "solid": "ore1", "density_value": 2.7}
    log.shapes = [MinedShape(1, "2026-02-01", "Stope 1", "xy", square(0, 0, 10, 10), 5.0, 2.0, "note")]
    assert not log.exists
    log.save()
    path = tmp_path / "geodatabase.sqlite"
    assert log.exists and log.where == str(path)
    with Database.sqlite(path, create=False) as db:
        assert set(db.tables()) == PROD
        assert db.query("SELECT id, mined_on, depth_from, depth_to FROM prod_mined_shapes") == \
            [(1, "2026-02-01", 5.0, 2.0)]
        assert db.scalar("SELECT outline FROM prod_mined_shapes WHERE id = 1") == \
            "[[0.0,0.0],[10.0,0.0],[10.0,10.0],[0.0,10.0]]"         # one JSON text per shape
    log.put(MinedShape(log.next_id(), "2026-02-03", "Stope 2", "yz", square(0, 0, 5, 5), 0, 3))
    assert not (tmp_path / "geodatabase.sqlite.bak").exists()      # new database: nothing to back up
    back = ProductionLog.load(tmp_path)
    assert back.exists and [s.id for s in back.shapes] == [1, 2]
    assert back.shapes[0].lo == 2.0 and back.shapes[0].hi == 5.0 and back.shapes[0].note == "note"
    assert back.shapes[0].points == square(0, 0, 10, 10) and back.shapes[1].plane == "yz"
    assert back.reserve == {"grade": "au", "solid": "ore1", "density_value": 2.7}
    # edit, delete
    back.put(MinedShape(2, "2026-02-04", "Stope 2b", "yz", square(0, 0, 6, 6), 1, 3))
    assert (tmp_path / "geodatabase.sqlite.bak").exists()          # copied before this session's first change
    back.remove(1)
    again = ProductionLog.load(tmp_path)
    assert [(s.id, s.name, s.date) for s in again.shapes] == [(2, "Stope 2b", "2026-02-04")]
    with Database.sqlite(path, create=False) as db:
        assert db.query("SELECT id FROM prod_mined_shapes") == [(2,)]
    assert again.next_id() == 3


def test_outline_text_is_compact_json_to_the_millimetre():
    from dfpredict.scene.production import outline_points, outline_text

    t = outline_text([[1855.37762193, 7477.68913229], [1828.3565654, 7460.80097]])
    assert t == "[[1855.378,7477.689],[1828.357,7460.801]]"
    assert outline_points(t) == [[1855.378, 7477.689], [1828.357, 7460.801]]
    assert outline_points("") == [] and outline_points("not json") == []


def test_version_2_corner_rows_are_upgraded_to_json(tmp_path):
    """geodatabase.sqlite of the previous version (one row per outline corner) is upgraded on open."""
    import sqlite3

    from dfpredict.io.database import Database

    con = sqlite3.connect(tmp_path / "geodatabase.sqlite")
    con.executescript("""
        CREATE TABLE prod_settings (setting VARCHAR(100) NOT NULL PRIMARY KEY, value TEXT);
        CREATE TABLE prod_mined_shapes (id INTEGER NOT NULL PRIMARY KEY, mined_on VARCHAR(10) NOT NULL,
            name VARCHAR(200) NOT NULL, view_plane VARCHAR(2) NOT NULL, depth_from DOUBLE NOT NULL,
            depth_to DOUBLE NOT NULL, note TEXT, created_at VARCHAR(19), updated_at VARCHAR(19));
        CREATE TABLE prod_mined_shape_points (shape_id INTEGER NOT NULL, seq INTEGER NOT NULL, u DOUBLE NOT NULL,
            v DOUBLE NOT NULL, PRIMARY KEY (shape_id, seq));
        INSERT INTO prod_settings VALUES ('format', 'GeoViewer production'), ('version', '2'),
                                         ('reserve.solid', '"ore1"');
        INSERT INTO prod_mined_shapes VALUES (1, '2026-11-22', 'Cut 1', 'xy', 215.0, 224.0, '', NULL, NULL),
                                             (2, '2026-11-23', 'Cut 2', 'xy', 205.0, 215.0, 'n', NULL, NULL);
        INSERT INTO prod_mined_shape_points VALUES (1, 0, 0, 0), (1, 1, 5, 0), (1, 2, 5, 5),
                                                   (2, 1, 10, 0), (2, 0, 0, 0), (2, 2, 10, 10), (2, 3, 0, 10);
        CREATE TABLE drillhole_collar (hole_id TEXT, x REAL, y REAL, z REAL);
    """)
    con.close()
    log = ProductionLog.load(tmp_path)
    assert [(s.id, s.points) for s in log.shapes] == [
        (1, [[0, 0], [5, 0], [5, 5]]), (2, [[0, 0], [10, 0], [10, 10], [0, 10]])]   # corners in seq order
    with Database.sqlite(tmp_path / "geodatabase.sqlite", create=False) as db:
        # nothing is dropped: the corner rows stay, the outlines are copied into the shapes' rows
        assert set(db.tables()) == {"prod_settings", "prod_mined_shapes", "prod_mined_shape_points",
                                    "drillhole_collar"}
        assert db.scalar("SELECT COUNT(*) FROM prod_mined_shape_points") == 7
        assert db.scalar("SELECT outline FROM prod_mined_shapes WHERE id = 1") == "[[0.0,0.0],[5.0,0.0],[5.0,5.0]]"
        assert db.scalar("SELECT value FROM prod_settings WHERE setting = 'version'") == "3"
    log.put(MinedShape(3, "2026-11-24", "Cut 3", "xz", [[1, 2], [3, 4], [5, 6]], 0, 2))
    again = ProductionLog.load(tmp_path)
    assert [s.id for s in again.shapes] == [1, 2, 3] and again.shapes[1].note == "n"


def test_failed_write_changes_nothing(tmp_path, monkeypatch):
    log = ProductionLog(tmp_path)
    log.put(MinedShape(1, "2026-02-01", "a", "xy", square(0, 0, 10, 10), 0, 2))

    def boom(db, sh):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ProductionLog, "_write_shape", staticmethod(boom))
    with pytest.raises(RuntimeError):
        log.put(MinedShape(2, "2026-02-02", "b", "xy", square(0, 0, 5, 5), 0, 2))
    assert [s.id for s in log.shapes] == [1]                       # not added in memory either
    monkeypatch.undo()
    assert [s.id for s in ProductionLog.load(tmp_path).shapes] == [1]


def _json(shapes):
    return json.dumps({"format": "GeoViewer production", "version": 1,
                       "reserve": {"grade": "gold", "solid": "ore1"}, "shapes": shapes})


CUT1 = {"id": 1, "date": "2026-11-22", "name": "Cut 1", "plane": "xy", "points": square(0, 0, 4, 4),
        "from": 195.0, "to": 199.3, "note": ""}


def test_production_json_is_imported_and_a_newer_tmp_wins(tmp_path):
    (tmp_path / "production.json").write_text(_json([]))
    (tmp_path / "production.json.tmp").write_text(_json([CUT1]))   # a save that was never swapped in
    t = (tmp_path / "production.json").stat().st_mtime_ns
    os.utime(tmp_path / "production.json.tmp", ns=(t + 10**9, t + 10**9))
    log = ProductionLog.load(tmp_path)
    assert log.imported_from == str(tmp_path / "production.json.tmp")
    assert [(s.name, s.from_, s.to) for s in log.shapes] == [("Cut 1", 195.0, 199.3)]
    assert log.reserve["solid"] == "ore1" and (tmp_path / "geodatabase.sqlite").exists()
    assert (tmp_path / "production.json").exists()                 # the old files are left alone
    again = ProductionLog.load(tmp_path)                           # read from the database now
    assert again.imported_from is None and len(again.shapes) == 1


def test_production_sqlite_of_the_first_version_is_imported(tmp_path):
    """production.sqlite (tables without the prod_ prefix) wins over an older production.json."""
    import sqlite3

    (tmp_path / "production.json").write_text(_json([]))
    con = sqlite3.connect(tmp_path / "production.sqlite")
    con.executescript("""
        CREATE TABLE production_settings (setting TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE mined_shapes (id INTEGER PRIMARY KEY, mined_on TEXT, name TEXT, view_plane TEXT,
                                   depth_from REAL, depth_to REAL, note TEXT, created_at TEXT, updated_at TEXT);
        CREATE TABLE mined_shape_points (shape_id INTEGER, seq INTEGER, u REAL, v REAL);
        INSERT INTO production_settings VALUES ('format', 'GeoViewer production'), ('version', '1'),
                                               ('reserve.solid', '"ore1"');
        INSERT INTO mined_shapes VALUES (7, '2026-11-22', 'Cut 7', 'xy', 215.0, 224.0, '', NULL, NULL);
        INSERT INTO mined_shape_points VALUES (7, 0, 0, 0), (7, 1, 5, 0), (7, 2, 5, 5);
    """)
    con.close()
    log = ProductionLog.load(tmp_path)
    assert log.imported_from == str(tmp_path / "production.sqlite")
    assert [(s.id, s.name, len(s.points)) for s in log.shapes] == [(7, "Cut 7", 3)]
    assert log.reserve == {"solid": "ore1"} and log.next_id() == 8


def test_foreign_tables_are_not_used(tmp_path):
    from dfpredict.io.database import Database

    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db, db.transaction():
        db.execute("CREATE TABLE prod_settings (setting TEXT, value TEXT)")
        db.execute("INSERT INTO prod_settings VALUES ('format', 'something else')")
    r = Reserve()
    r.open(tmp_path)
    assert r.log is None and "does not hold GeoViewer production records" in r.open_error
    assert not r.update([]) and "Could not read" in r.error


def test_unreachable_mysql_is_reported(tmp_path):
    from dfpredict.io import database

    database.configure(database.DatabaseConfig("mysql", host="127.0.0.1", port=1, user="nobody"))
    r = Reserve()
    r.open(tmp_path)
    assert r.log is None and "Cannot reach the MySQL server 127.0.0.1:1" in r.open_error
    assert "MySQL geoviewer@127.0.0.1" in r.open_error


def test_settings_keep_the_database_choice_and_encrypt_the_password(tmp_path):
    from PySide6.QtCore import QSettings

    from dfpredict.io.database import DatabaseConfig
    from dfpredict.ui.settings_dialog import AppSettings

    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    s = AppSettings(database=DatabaseConfig("mysql", "db.mine.local", 3307, "mine", "geo", "s3cret"))
    s.save(qs)
    qs.sync()
    assert "s3cret" not in (tmp_path / "s.ini").read_text()
    back = AppSettings.load(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    assert back.database == s.database
    fresh = AppSettings.load(QSettings(str(tmp_path / "none.ini"), QSettings.IniFormat)).database
    # SQLite (the workspace's geodatabase.sqlite) by default in this application; MySQL defaults kept, no password
    assert (fresh.backend, fresh.host, fresh.port, fresh.database, fresh.user, fresh.password) == (
        "sqlite", "localhost", 3306, "geoviewer", "root", "")


# --- depletion ---------------------------------------------------------------------------------
def test_partial_blocks_and_totals():
    bm = grid_model()
    dep = depletion(bm)
    t_block = 125.0 * 2.5
    assert dep.summary()["total"].tonnes == pytest.approx(1000 * t_block)
    # 10 x 10 m outline, 2.5 m thick: four blocks, each half mined
    sh = MinedShape(1, "2026-03-01", "a", "xy", square(0, 0, 10, 10), 0.0, 2.5)
    dep.apply([sh])
    f = dep.fraction()
    assert sorted(np.unique(f).tolist()) == [0.0, 0.5]
    assert (f == 0.5).sum() == 4
    s = dep.summary()
    assert s["produced"].tonnes == pytest.approx(2 * t_block)
    assert s["produced"].grade == pytest.approx(2.0)
    assert s["produced"].tonnes + s["remaining"].tonnes == pytest.approx(s["total"].tonnes)
    attrs = dep.block_attributes()
    assert np.bincount(attrs["status"], minlength=3).tolist() == [996, 4, 0]


def test_overlapping_shapes_are_not_counted_twice_and_as_of():
    bm = grid_model()
    dep = depletion(bm)
    t_block = 125.0 * 2.5
    a = MinedShape(1, "2026-03-01", "a", "xy", square(0, 0, 10, 10), 0.0, 5.0)    # 4 whole blocks
    b = MinedShape(2, "2026-04-10", "b", "xy", square(5, 0, 15, 10), 0.0, 5.0)    # 2 of them again + 2 new
    dep.apply([b, a])                       # order of the list does not matter, dates do
    ta, na = dep.shape_totals(1)
    tb, nb = dep.shape_totals(2)
    assert ta.tonnes == pytest.approx(4 * t_block) and na == 4
    assert tb.tonnes == pytest.approx(2 * t_block) and nb == 2
    assert dep.summary()["produced"].tonnes == pytest.approx(6 * t_block)
    assert dep.summary(day_of("2026-03-31"))["produced"].tonnes == pytest.approx(4 * t_block)
    assert dep.summary(day_of("2026-02-28"))["produced"].tonnes == 0.0
    st = dep.block_attributes(day_of("2026-03-31"))["status"]
    assert (st == 2).sum() == 4
    # previewing b again (editing it): its own points are free, a's are "already mined"
    p = dep.preview(b)
    assert p["new"].tonnes == pytest.approx(2 * t_block)
    assert p["already"].tonnes == pytest.approx(2 * t_block)
    assert p["blocks"] == 4 and p["fill"] == pytest.approx(1.0)


def test_preview_fill_and_profile_ore_range():
    bm = grid_model(nz=4)                   # ore from z 0 to 20
    dep = depletion(bm)
    sh = MinedShape(9, "2026-05-01", "p", "xy", square(10, 10, 20, 20), 15.0, 25.0)   # half above the ore
    p = dep.preview(sh)
    assert p["fill"] == pytest.approx(0.5)
    assert p["new"].tonnes == pytest.approx(100 * 5 * 2.5)
    edges, ore, old, (lo, hi) = dep.profile(sh)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(20.0)
    assert ore.sum() == pytest.approx(100 * 20 * 2.5)
    assert old.sum() == 0.0
    # nothing under an outline beside the model
    assert dep.profile(MinedShape(10, "2026-05-01", "x", "xy", square(80, 80, 90, 90), 0, 5)) is None


def test_monthly_rows():
    bm = grid_model()
    dep = depletion(bm)
    log = ProductionLog(Path("."))
    log.shapes = [MinedShape(1, "2026-03-01", "a", "xy", square(0, 0, 10, 10), 0, 5),
                  MinedShape(2, "2026-03-20", "b", "xy", square(10, 0, 20, 10), 0, 5),
                  MinedShape(3, "2026-04-02", "c", "xy", square(20, 0, 30, 10), 0, 5)]
    dep.apply(log.shapes)
    rows = monthly(log, dep)
    assert [r["month"] for r in rows] == ["2026-03", "2026-04"]
    assert [r["shapes"] for r in rows] == [2, 1]
    assert rows[-1]["cumulative"] == pytest.approx(dep.summary()["produced"].tonnes)
    assert rows[-1]["remaining"] == pytest.approx(dep.summary()["remaining"].tonnes)


def test_placeholder_attributes_and_uses_depletion():
    bm = grid_model(2, 2, 2)
    assert ensure_depletion_attrs(bm)
    assert not ensure_depletion_attrs(bm)
    assert not bm.valid_mask("status").any() and not bm.valid_mask("mined_fraction").any()
    assert bm.categories["status"] == ["Unmined", "Partly mined", "Mined"]

    class L:
        color_by = "au"
        constraints = [{"kind": "value", "name": "status", "op": "!=", "value": "Mined"}]

    assert uses_depletion(L())
    L.constraints = []
    assert not uses_depletion(L())


# --- the demo reserve (ore1) ---------------------------------------------------------------------
def test_demo_reserve_setup_and_totals(demo, tmp_path):
    from dfpredict.scene.loaders import load_paths

    layers, errors = load_paths([demo / "block_model.csv", demo / "ore1.dtm", demo / "topo1.dtm"])
    assert not errors
    r = Reserve()
    r.open(tmp_path)
    assert r.auto_setup(layers)
    s = r.setup
    assert (s["block_model"], s["solid"], s["density"], s["grade"]) == ("block_model", "ore1", "sg", "gold")
    assert r.update(layers) and not r.error
    tot = r.dep.summary()["total"]
    assert tot.tonnes == pytest.approx(7.485e6, rel=2e-3)
    assert tot.grade == pytest.approx(1.34, abs=0.01)
    bm = r.inject()
    assert set(np.unique(bm.attributes["status"]).tolist()) <= {-1, 0}
    # a solid that is not loaded: not ready, and the choice is kept
    r.set_setup(solid="missing")
    assert not r.update(layers) and "missing" in r.error
    assert not r.auto_setup(layers) and r.setup["solid"] == "missing"


def test_mined_shapes_layer_not_saved_in_workspace():
    from dfpredict.scene.layers import MinedShapesLayer
    from dfpredict.workspace import saved_layers

    lay = MinedShapesLayer(name="Mined shapes", path=None)
    assert saved_layers([lay]) == []


def test_mined_blocks_setting_is_saved_and_old_status_colouring_dropped(demo, tmp_path):
    """Colour by status / mined_fraction is gone (an old workspace falls back to no colouring);
    the "Mined blocks" choice is kept with the layer."""
    import shutil

    from dfpredict.scene.layers import BlockModelLayer
    from dfpredict.scene.loaders import load_paths
    from dfpredict.workspace import layer_spec, load_spec

    shutil.copy2(demo / "block_model.csv", tmp_path / "block_model.csv")
    (bm_layer,), _ = load_paths([tmp_path / "block_model.csv"])
    assert bm_layer.mined_display == "gray"                     # the default
    bm_layer.mined_display = "hidden"
    spec = layer_spec(bm_layer, tmp_path)
    spec["settings"]["color_by"] = "status"                     # as saved by an earlier version
    new, errs = load_spec(spec, tmp_path)
    assert not errs and isinstance(new, BlockModelLayer)
    assert new.mined_display == "hidden" and new.color_by != "status"
    new.apply_settings({"mined_display": "sideways"})           # unknown values are not taken
    assert new.mined_display == "hidden"


def test_mined_blocks_gray_transparent_hidden(tmp_path):
    """Mined blocks (>= half mined) keep the colour of their grade: mixed with gray, see-through,
    or not drawn; unmined blocks are drawn as before (with the colour bar)."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyvista as pv

    from dfpredict.scene.layers import MINED_GRAY, MINED_OPACITY, BlockModelLayer

    class _Scene:                                              # what a layer needs to draw
        origin = np.zeros(3)
        foreground = "white"
        layers = []

        def __init__(self):
            self.plotter = pv.Plotter(off_screen=True)

        def slab_mask(self, c):
            return np.ones(len(c), dtype=bool)

    bm = grid_model(4, 4, 2)                                   # 32 blocks, au = 2.0 everywhere
    bm.attributes["au"] = np.linspace(0, 3, bm.n_blocks).astype(np.float32)
    ensure_depletion_attrs(bm)
    f = np.zeros(bm.n_blocks, np.float32)
    f[:8] = 1.0                                                # 8 fully mined
    f[8:10] = 0.5                                              # 2 half mined: count as mined
    f[10:12] = 0.4                                             # 2 partly: shown as unmined
    bm.attributes["mined_fraction"][:] = f
    lay = BlockModelLayer(name="bm", path=None, data=bm, color_by="au")
    lay.scene = _Scene()
    mined = lay.mined_mask()
    assert mined.sum() == 10 and not mined[10:].any()
    own = lay.block_rgb(np.arange(10))
    assert (own == lay.block_rgb(np.arange(10))).all() and own.std() > 0     # the grade colours
    # gray: the block's own colour mixed with gray
    lay.rebuild()
    assert len(lay.actors) == 2 and lay._mined_actor is lay.actors[-1]
    data = lay._mined_actor.mapper.dataset
    blocks = data.cell_data["block"]
    expect = (lay.block_rgb(blocks) * (1 - MINED_GRAY) + 128 * MINED_GRAY).round()
    assert np.abs(data.cell_data["rgb"].astype(float) - expect).max() <= 1
    assert MINED_GRAY == 1.0 and (data.cell_data["rgb"] == 128).all()        # plain gray
    assert set(blocks) <= set(np.flatnonzero(mined))
    # transparent: own colour, see-through; both sets closed
    lay.mined_display = "transparent"
    lay.rebuild()
    data = lay._mined_actor.mapper.dataset
    assert (data.cell_data["rgb"] == lay.block_rgb(data.cell_data["block"])).all()
    assert lay._mined_actor.prop.opacity == pytest.approx(MINED_OPACITY)
    assert set(data.cell_data["block"]) == set(np.flatnonzero(mined))          # every mined block has faces
    # show: drawn like every other block (one mesh, the grade colours)
    lay.mined_display = "show"
    lay.rebuild()
    assert len(lay.actors) == 1 and lay._mined_actor is None and lay.n_shown == bm.n_blocks
    # hidden: not drawn at all
    lay.mined_display = "hidden"
    lay.rebuild()
    assert lay._mined_actor is None and lay.n_shown == bm.n_blocks - 10
    drawn = lay.actors[0].mapper.dataset.cell_data["block"]
    assert not mined[drawn].any()
    lay.scene.plotter.close()


# --- performance helpers: same answers as the brute-force versions -----------------------------------
def test_block_index_matches_a_full_scan():
    from dfpredict.scene.production import BlockIndex

    rng = np.random.default_rng(7)
    n = 20000
    size = rng.choice([2.5, 5.0, 10.0], size=(n, 1)) * np.ones((1, 3))          # sub-blocks of three sizes
    c = rng.uniform([0, 0, 0], [800, 600, 300], size=(n, 3))
    lo, hi = c - size / 2, c + size / 2
    idx = BlockIndex(lo, hi)
    for _ in range(300):
        a = rng.uniform([-50, -50, -20], [850, 650, 320])
        b = a + rng.uniform([0, 0, 0], [60, 60, 40])
        brute = np.flatnonzero(np.all((hi >= a) & (lo <= b), axis=1))
        np.testing.assert_array_equal(idx.query(a, b), brute)
    assert len(BlockIndex(np.zeros((0, 3)), np.zeros((0, 3))).query([0, 0, 0], [1, 1, 1])) == 0


def test_all_shape_totals_match_one_shape_at_a_time():
    bm = grid_model(20, 20, 6)
    dep = depletion(bm)
    shapes = [MinedShape(i + 1, f"2026-03-{i + 1:02d}", f"c{i}", "xy",
                         square(3 * i, 2 * i, 3 * i + 17, 2 * i + 13), 1.0 + i, 9.0 + i) for i in range(12)]
    dep.apply(shapes)
    fast = dep.all_shape_totals()
    for sh in shapes:
        hit = dep.point_shape == sh.id                         # the per-shape scan it replaces
        f = hit.mean(axis=1)
        t = dep.tonnes * f
        tot, blocks = fast.get(sh.id, (None, 0))
        if not (f > 0).any():
            assert tot is None
            continue
        assert tot.tonnes == pytest.approx(t.sum()) and tot.metal == pytest.approx((t * dep.grade).sum())
        assert blocks == int((f > 0).sum())
    assert sum(t.tonnes for t, _b in fast.values()) == pytest.approx(dep.summary()["produced"].tonnes)


def test_backup_holds_only_the_production_tables(tmp_path):
    from dfpredict.io.database import Database

    log = ProductionLog(tmp_path)
    log.put(MinedShape(1, "2026-02-01", "a", "xy", square(0, 0, 10, 10), 0, 2))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:             # big imported drillhole table
        db.write_frame("drillhole_sample", pd.DataFrame({"hole_id": ["H"] * 50000, "gold": np.arange(50000.0)}))
        names = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert "idx_prod_mined_shapes_mined_on" in names                          # shapes indexed by date
    again = ProductionLog.load(tmp_path)
    again.put(MinedShape(2, "2026-02-02", "b", "xy", square(0, 0, 5, 5), 0, 2))
    bak = tmp_path / "geodatabase.sqlite.bak"
    with Database.sqlite(bak, create=False) as db:
        assert set(db.tables()) == {"prod_settings", "prod_mined_shapes"}
        assert db.query("SELECT id FROM prod_mined_shapes") == [(1,)]        # as before this session's change
    assert bak.stat().st_size < (tmp_path / "geodatabase.sqlite").stat().st_size / 5


def test_metal_is_shown_in_kg_or_tonnes_never_ounces():
    from dfpredict.scene.production import metal_text

    assert metal_text(10_008_900.0, "g/t") == "10,008.9 kg"           # grams of metal -> kg
    assert metal_text(250_000.0, "%") == "2,500 t"                    # tonnes x % -> tonnes of metal
    assert "oz" not in metal_text(1e6, "g/t")


# --- mined solids (a closed .dtm instead of an outline + From / To) ------------------------------
def _box_solid(x0, y0, z0, x1, y1, z1):
    """A closed box (8 vertices, 12 triangles, outward)."""
    import pyvista as pv

    m = pv.Box(bounds=(x0, x1, y0, y1, z0, z1)).triangulate().clean()
    return np.asarray(m.points, dtype=np.float64), m.faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def test_mined_solid_depletes_like_the_same_outline_cut():
    bm = grid_model()
    v, t = _box_solid(0, 0, 0, 10, 10, 5)
    solid = MinedShape.from_solid(1, "2026-03-01", "stope", v, t, "stope.dtm")
    assert solid.is_solid and solid.plane == "3d" and (solid.from_, solid.to) == (0.0, 5.0)
    assert solid.volume() == pytest.approx(500.0)
    lo, hi = solid.bounds()
    assert lo.tolist() == [0, 0, 0] and hi.tolist() == [10, 10, 5]
    prism = MinedShape(2, "2026-03-01", "cut", "xy", square(0, 0, 10, 10), 0.0, 5.0)
    a, b = depletion(bm), depletion(bm)
    a.apply([solid])
    b.apply([prism])
    assert a.summary()["produced"].tonnes == pytest.approx(b.summary()["produced"].tonnes)
    assert a.summary()["produced"].tonnes == pytest.approx(4 * 125.0 * 2.5)          # the 4 blocks inside
    # overlapping a drawn cut: counted once
    later = MinedShape(3, "2026-04-01", "cut", "xy", square(5, 0, 15, 10), 0.0, 5.0)
    a.apply([solid, later])
    assert a.shape_totals(3)[0].tonnes == pytest.approx(2 * 125.0 * 2.5)


def test_mined_solid_is_stored_with_the_records(tmp_path):
    from dfpredict.scene.production import solid_blob, solid_from_blob

    v, t = _box_solid(1000.5, 7000.25, 200.125, 1010.5, 7010.25, 205.125)    # mine coordinates kept exactly
    vb, tb = solid_from_blob(solid_blob(v, t))
    assert np.array_equal(vb, v) and np.array_equal(tb, t)
    assert solid_from_blob(b"not a solid") is None and solid_from_blob(None) is None
    log = ProductionLog(tmp_path)
    log.put(MinedShape(1, "2026-02-01", "cut", "xy", square(0, 0, 10, 10), 0, 2))
    log.put(MinedShape.from_solid(2, "2026-02-05", "Stope 7", v, t, "stope7.dtm", "survey"))
    back = ProductionLog.load(tmp_path)
    s = back.get(2)
    assert s.is_solid and s.source == "stope7.dtm" and s.note == "survey" and s.name == "Stope 7"
    assert np.array_equal(s.vertices, v) and s.volume() == pytest.approx(500.0)
    assert not back.get(1).is_solid and back.get(1).points == square(0, 0, 10, 10)


def test_a_surface_dtm_is_not_a_mined_solid(demo):
    from dfpredict.ui.mined_shape_dialog import load_mined_solid

    with pytest.raises(ValueError, match="no closed solid"):
        load_mined_solid(demo / "topo1.dtm")                     # a surface
    v, t = load_mined_solid(demo / "ug1.dtm")                    # a closed solid
    assert len(t) > 0 and MinedShape.from_solid(1, "2026-01-01", "x", v, t).volume() > 0


def test_mined_solid_dialog_figures(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from dfpredict.ui.mined_shape_dialog import MinedSolidDialog

    app = QApplication.instance() or QApplication([])
    bm = grid_model()

    class _R:                                                   # the reserve, as the dialog uses it
        dep = depletion(bm)
        log = ProductionLog(tmp_path)
        error = ""
        setup = {"unit": "g/t"}

    class _S:                                                   # the scene preview
        origin = None
        def set_preview_solid(self, *a): pass
        def clear_preview_solid(self): pass

    _R.dep.apply([])
    v, t = _box_solid(0, 0, 0, 10, 10, 5)
    dlg = MinedSolidDialog(None, _S(), _R(), MinedShape.from_solid(1, "2026-01-01", "Stope", v, t, "s.dtm"))
    assert dlg.f_new.text().startswith(f"{4 * 125 * 2.5:,.0f} t")
    assert dlg.f_fill.text() == "100%" and dlg.warn.text() == ""
    got = []
    dlg.saved.connect(got.append)
    dlg._save()
    assert got and got[0].is_solid and got[0].name == "Stope"
    app.processEvents()


def test_reserve_setup_follows_a_renamed_layer():
    """Renaming the block model or the reserve solid in the layer list keeps the reserve working:
    the setup names the layer by its new name; other renames change nothing."""
    from types import SimpleNamespace

    r = Reserve()
    assert r.renamed("block_model", "BM 2026") == {}                   # no records open: nothing
    r.log = SimpleNamespace(reserve={"block_model": "block_model", "solid": "ore1", "grade": "au"})
    assert r.renamed("block_model", "BM 2026") == {"block_model": "BM 2026"}
    assert r.renamed("ore1", "Ore zone") == {"solid": "Ore zone"}
    assert r.renamed("topo1", "Topography") == {}
    r.set_setup(**r.renamed("ore1", "Ore zone"))
    assert r.setup["solid"] == "Ore zone" and r.setup["block_model"] == "block_model"
