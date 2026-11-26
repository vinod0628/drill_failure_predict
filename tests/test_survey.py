"""Underground survey: angles, traverse computation, closure / Bowditch, branches, storage."""

import math

import pytest

from dfpredict.io.database import Database, project_target
from dfpredict.io.survey_store import (Pit, SetupRow, Station, SurveyStore, compute, format_dms, grid_bearing,
                                       parse_angle)

E = Station("E", 1, "known", 1000.0, 5000.0, 250.0)


def example_rows():
    """The worked example: E oriented on a known bearing of 45°, then S1, S2."""
    return [SetupRow("E", 1.5, bs_bearing=45.0, fs="S1", fs_ha=120.5, fs_za=95.0, fs_sd=50.0, fs_ht=1.4),
            SetupRow("S1", 1.45, bs_station="E", bs_ha=300.5, fs="S2", fs_ha=160.25, fs_za=92.0, fs_sd=40.0,
                     fs_ht=1.45)]


def test_angles_typed_in_every_usual_way():
    for text in ("120 30 00", "120°30′00″", "120°30'00\"", "120-30-00", "120:30:00", "120.5", "120d30m"):
        assert parse_angle(text) == pytest.approx(120.5), text
    assert parse_angle("") is None and parse_angle(None) is None
    assert parse_angle("0 0 36") == pytest.approx(0.01)
    with pytest.raises(ValueError):
        parse_angle("120 75 00")
    assert format_dms(120.5) == "120°30′00″" and format_dms(45.0) == "45°00′00″"
    assert format_dms(160.250278) == "160°15′01″"


def test_the_worked_example():
    net = compute([Pit(1, "Main")], [E], {1: example_rows()})
    s1, s2 = net.coords["S1"], net.coords["S2"]
    assert s1 == pytest.approx((1042.917, 4974.720, 245.742), abs=1e-3)
    assert s2 == pytest.approx((1056.426, 4937.096, 244.346), abs=1e-3)
    assert net.status["S1"] == "open" and net.status["E"] == "known"
    leg1 = net.legs[0]
    assert leg1.bearing == pytest.approx(120.5) and leg1.hd == pytest.approx(49.810, abs=1e-3)
    assert not net.messages


def test_circle_set_to_zero_gives_the_same_bearings():
    """Readings with the circle at 0° on the backsight: bearing = BS bearing + (HA_FS - HA_BS)."""
    rows = example_rows()
    rows[0].bs_ha, rows[0].fs_ha = 0.0, 75.5                 # 120°30' - 45°
    rows[1].bs_ha, rows[1].fs_ha = 0.0, 219.75               # 160°15' - 300°30' + 360
    net = compute([Pit(1, "Main")], [E], {1: rows})
    assert net.coords["S2"] == pytest.approx((1056.426, 4937.096, 244.346), abs=1e-3)


def test_two_faces_are_averaged_and_bad_faces_flagged():
    rows = example_rows()
    rows[0].fs_ha_r, rows[0].fs_za_r = 300.5 + 4 / 3600, 265.0      # right face: HA + 180, ZA = 360 - 95
    net = compute([Pit(1, "Main")], [E], {1: rows})
    assert net.legs[0].bearing == pytest.approx(120.5 + 2 / 3600)   # the mean
    rows[0].fs_ha_r = 300.5 + 90 / 3600                             # 90" apart: flagged
    net = compute([Pit(1, "Main")], [E], {1: rows})
    assert any("faces differ by 90″" in m for _p, _i, m in net.messages)


def test_backsight_check_catches_a_wrong_station():
    rows = example_rows()
    rows[1].bs_za, rows[1].bs_sd, rows[1].bs_ht = 85.057, 49.996, 1.5       # right: matches leg 1
    assert not compute([Pit(1, "Main")], [E], {1: rows}).messages
    rows[1].bs_sd = 45.0                                                    # wrong
    msgs = compute([Pit(1, "Main")], [E], {1: rows}).messages
    assert any("backsight check at S1" in m for _p, _i, m in msgs)


def test_true_and_magnetic_bearings_become_grid_bearings():
    s = {"convergence": 1.5, "declination": -8.0}
    assert grid_bearing(45.0, "grid", s) == 45.0
    assert grid_bearing(45.0, "true", s) == pytest.approx(43.5)            # grid = true - convergence
    assert grid_bearing(45.0, "magnetic", s) == pytest.approx(35.5)        # true = magnetic + declination
    rows = example_rows()
    rows[0].bs_bearing, rows[0].bs_bearing_type = 46.5, "true"
    net = compute([Pit(1, "Main")], [E], {1: rows}, s)
    assert net.coords["S1"] == pytest.approx((1042.917, 4974.720, 245.742), abs=1e-3)


def _square_loop(error=0.0):
    """A 4 x 100 m loop E -> A -> B -> C -> E (level), the last distance off by ``error``."""
    def row(at, bs, brg_bs, fs, brg_fs, sd):
        return SetupRow(at, 1.5, bs_station=bs, bs_ha=brg_bs, fs=fs, fs_ha=brg_fs, fs_za=90.0, fs_sd=sd, fs_ht=1.5)
    return [SetupRow("E", 1.5, bs_bearing=0.0, fs="A", fs_ha=90.0, fs_za=90.0, fs_sd=100.0, fs_ht=1.5),
            row("A", "E", 270.0, "B", 180.0, 100.0), row("B", "A", 0.0, "C", 270.0, 100.0),
            row("C", "B", 90.0, "E", 0.0, 100.0 + error)]


def test_closed_loop_misclosure_ratio_and_bowditch():
    net = compute([Pit(1, "Main", adjust=True)], [E], {1: _square_loop(error=0.02)})
    de, dn, dz, lin, ratio, passed, onto = net.pits[1].closure
    assert onto == "E" and lin == pytest.approx(0.02, abs=1e-9) and ratio == pytest.approx(400.02 / 0.02, rel=1e-6)
    assert passed and net.pits[1].adjusted
    # Bowditch: the error is spread by distance - C (300 of the 400.02 m) moves back 300/400.02 of it
    assert net.coords["C"][1] == pytest.approx(4900.0 - 0.02 * 300 / 400.02, abs=1e-9)
    assert net.coords["A"][0] == pytest.approx(1100.0, abs=1e-9)          # no east-west error to spread
    assert net.status["A"] == net.status["C"] == "closed"
    # a loop that does not meet the tolerance is not adjusted and is marked failed
    bad = compute([Pit(1, "Main")], [E], {1: _square_loop(error=0.5)})
    assert not bad.pits[1].closure[5] and not bad.pits[1].adjusted and bad.status["B"] == "failed"
    assert bad.coords["C"][1] == pytest.approx(4900.0)                    # left as measured


def test_branches_start_on_their_parent_in_any_order():
    pits = [Pit(2, "Branch", parent_id=1, start_station="S1"), Pit(1, "Main")]
    branch = [SetupRow("S1", 1.5, bs_station="E", bs_ha=300.5, fs="B1", fs_ha=30.5, fs_za=90.0, fs_sd=20.0,
                       fs_ht=1.5)]
    net = compute(pits, [E], {2: branch, 1: example_rows()})
    s1 = net.coords["S1"]
    b1 = net.coords["B1"]
    assert math.hypot(b1[0] - s1[0], b1[1] - s1[1]) == pytest.approx(20.0)
    assert (math.degrees(math.atan2(b1[0] - s1[0], b1[1] - s1[1])) % 360) == pytest.approx(30.5)
    # a branch whose parent station is not reached is reported, not guessed
    net = compute(pits, [E], {2: [SetupRow("Q9", 1.5, bs_station="E", fs="B1", fs_ha=1, fs_za=90, fs_sd=5)]})
    assert any("station Q9 has no coordinates yet" in m for _p, _i, m in net.messages)


def test_storage_round_trip_and_stored_coordinates(tmp_path):
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        notes = store.prepare(db)
    assert "made table survey_pits" in notes and "made table survey_shots" in notes
    main = store.add_pit("Main pit 1", start_station="E", known=(1000.0, 5000.0, 250.0))
    branch = store.add_pit("Branch 1A", parent_id=main, start_station="S1")
    rows = example_rows()
    rows[0].fs_ha_r, rows[0].fs_za_r = 300.5, 265.0
    store.save_setups(main, rows)
    store.save_setups(branch, [SetupRow("S1", 1.5, bs_station="E", bs_ha=300.5, fs="B1", fs_ha=30.5, fs_za=90.0,
                                        fs_sd=20.0, fs_ht=1.5)])
    st = {s.name: s for s in store.stations()}
    assert (st["S2"].x, st["S2"].y, st["S2"].z) == pytest.approx((1056.426, 4937.096, 244.346), abs=1e-3)
    assert st["S2"].pit_id == main and st["B1"].pit_id == branch and st["E"].kind == "known"
    back = store.setups()
    assert [r.at for r in back[main]] == ["E", "S1"] and back[main][0].fs_ha_r == 300.5
    assert back[main][1].bs_station == "E" and back[branch][0].fs == "B1"
    assert [(p.name, p.parent_id) for p in store.pits()] == [("Main pit 1", None), ("Branch 1A", main)]
    with pytest.raises(ValueError, match="already a pit"):
        store.add_pit("Main pit 1", start_station="X", known=(0, 0, 0))
    store.set_settings({"closure_ratio": 5000})
    assert store.settings()["closure_ratio"] == 5000.0
    # deleting the main pit takes its branch along
    assert sorted(store.delete_pit(main)) == sorted([main, branch])
    assert store.pits() == [] and store.stations() == []


def test_pit_codes_and_station_numbers():
    from dfpredict.io.survey_store import next_code, next_station_name

    pits = [Pit(1, "Main 1", code="P1"), Pit(2, "B", parent_id=1, code="P1A")]
    assert next_code(pits, None) == "P2"
    assert next_code(pits, 1) == "P1B"                        # second branch of P1
    assert next_code(pits, 2) == "P1A1"                       # a branch of the branch
    assert next_station_name("P1", ["P1-01", "P1-02", "E"]) == "P1-03"
    assert next_station_name("P1A", []) == "P1A-01"


def test_store_gives_codes(tmp_path):
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        store.prepare(db)
    a = store.add_pit("Main pit 1", start_station="E1", known=(0, 0, 0))
    b = store.add_pit("Main pit 2", start_station="E2", known=(10, 0, 0))
    c = store.add_pit("Branch", parent_id=a, start_station="P1-03")
    codes = {p.id: p.code for p in store.pits()}
    assert codes == {a: "P1", b: "P2", c: "P1A"}
    store.update_pit(a, code="NORTH")
    assert next(p for p in store.pits() if p.id == a).code == "NORTH"


def test_compass_bearing_with_declination_needs_no_backsight(tmp_path):
    """Method "bearing": S1 -> S2 measured by compass; magnetic is turned into grid with the
    declination (true = magnetic + declination) and the convergence (grid = true - convergence)."""
    settings = {"declination": -7.5, "convergence": 1.25}
    rows = example_rows()
    # the compass read 168°30' for S1 -> S2: true 161°00', grid 159°45' ... make it the example's 160°15'
    rows[1] = SetupRow("S1", 1.45, fs="S2", fs_bearing=160.25 + 1.25 + 7.5, fs_bearing_type="magnetic",
                       fs_za=92.0, fs_sd=40.0, fs_ht=1.45)
    assert rows[1].method == "bearing" and example_rows()[1].method == "angles"
    net = compute([Pit(1, "Main")], [E], {1: rows}, settings)
    assert net.legs[1].bearing == pytest.approx(160.25)
    assert net.coords["S2"] == pytest.approx((1056.426, 4937.096, 244.346), abs=1e-3)
    # a whole compass traverse from the entrance: no backsight anywhere
    compass = [SetupRow("E", 1.5, fs="C1", fs_bearing=90.0, fs_bearing_type="grid", fs_za=90.0, fs_sd=10.0, fs_ht=1.5),
               SetupRow("C1", 1.5, fs="C2", fs_bearing=180.0, fs_bearing_type="true", fs_za=90.0, fs_sd=10.0,
                        fs_ht=1.5)]
    net = compute([Pit(1, "Main")], [E], {1: compass}, {"convergence": 0.0})
    assert net.coords["C2"] == pytest.approx((1010.0, 4990.0, 250.0), abs=1e-9)
    # stored and read back as a bearing setup
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        store.prepare(db)
    pid = store.add_pit("Main", start_station="E", known=(1000.0, 5000.0, 250.0))
    store.set_settings(settings)
    store.save_setups(pid, rows)
    back = store.setups()[pid][1]
    assert back.method == "bearing" and back.fs_bearing == pytest.approx(169.0) and back.fs_bearing_type == "magnetic"
    assert back.bs_station is None
    s2 = {s.name: s for s in store.stations()}["S2"]
    assert (s2.x, s2.y, s2.z) == pytest.approx((1056.426, 4937.096, 244.346), abs=1e-3)


def test_typed_coordinates_are_fixed_points(tmp_path):
    """Method "coords": a station's X, Y, Z already calculated elsewhere are typed in and used as
    they are. The leg to it is drawn (bearing / HD / dZ from the coordinates), a later traverse
    continues from it, and a closure checks - and adjusts - only the legs measured after it."""
    rows = [SetupRow("E", fs="G1", fs_x=1030.0, fs_y=5040.0, fs_z=240.0),          # given: 50 m at 36.87°
            SetupRow("G1", 1.5, bs_station="E", bs_ha=0.0, fs="S1", fs_ha=90.0 - 216.869898 + 360.0, fs_za=90.0,
                     fs_sd=100.0, fs_ht=1.5)]                                       # due east 100 m
    assert rows[0].method == "coords" and rows[1].method == "angles"
    net = compute([Pit(1, "Main")], [E], {1: rows})
    assert net.coords["G1"] == (1030.0, 5040.0, 240.0) and net.status["G1"] == "entered"
    lg = net.legs[0]
    assert lg.entered and lg.bearing == pytest.approx(36.869898) and lg.hd == pytest.approx(50.0)
    assert lg.dz == pytest.approx(-10.0)
    assert net.coords["S1"] == pytest.approx((1130.0, 5040.0, 240.0), abs=1e-6)
    assert not net.messages
    # a traverse closing on the given station: the misclosure covers only the measured legs, and
    # Bowditch moves only them - never the given coordinates
    loop = [SetupRow("E", fs="G1", fs_x=1000.0, fs_y=5100.0, fs_z=250.0),
            SetupRow("G1", 1.5, bs_station="E", bs_ha=180.0, fs="A", fs_ha=90.0, fs_za=90.0, fs_sd=100.0,
                     fs_ht=1.5),
            SetupRow("A", 1.5, bs_station="G1", bs_ha=270.0, fs="B", fs_ha=180.0, fs_za=90.0, fs_sd=100.0, fs_ht=1.5),
            SetupRow("B", 1.5, bs_station="A", bs_ha=0.0, fs="C", fs_ha=270.0, fs_za=90.0, fs_sd=100.0, fs_ht=1.5),
            SetupRow("C", 1.5, bs_station="B", bs_ha=90.0, fs="G1", fs_ha=0.0, fs_za=90.0, fs_sd=100.02, fs_ht=1.5)]
    net = compute([Pit(1, "Main", adjust=True)], [E], {1: loop})
    de, dn, dz, lin, ratio, passed, onto = net.pits[1].closure
    assert onto == "G1" and lin == pytest.approx(0.02, abs=1e-9)
    assert net.pits[1].length == pytest.approx(400.02)             # not the 100 m to the given station
    assert passed and net.pits[1].adjusted
    assert net.coords["G1"] == (1000.0, 5100.0, 250.0) and net.status["G1"] == "entered"
    assert net.status["A"] == "closed"
    # missing values and a clash with a known station are reported
    net = compute([Pit(1, "Main")], [E], {1: [SetupRow("E", fs="G1", fs_x=1.0, fs_y=2.0)]})
    assert "G1" not in net.coords and any("X, Y and Z are needed" in m for _p, _i, m in net.messages)
    net = compute([Pit(1, "Main")], [E], {1: [SetupRow("S0", fs="E", fs_x=1.0, fs_y=2.0, fs_z=3.0)]})
    assert net.coords["E"] == (1000.0, 5000.0, 250.0)
    assert any("E is a known station" in m for _p, _i, m in net.messages)
    # stored and read back
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        store.prepare(db)
    pid = store.add_pit("Main", start_station="E", known=(1000.0, 5000.0, 250.0))
    store.save_setups(pid, rows)
    back = store.setups()[pid]
    assert back[0].method == "coords" and (back[0].fs_x, back[0].fs_y, back[0].fs_z) == (1030.0, 5040.0, 240.0)
    assert back[0].fs == "G1" and back[0].bs_station is None and back[0].fs_sd is None
    assert back[1].method == "angles" and back[1].fs_x is None
    st = {s.name: s for s in store.stations()}
    assert (st["G1"].x, st["G1"].y, st["G1"].z, st["G1"].status) == (1030.0, 5040.0, 240.0, "entered")
    assert st["S1"].x == pytest.approx(1130.0, abs=1e-6)


def test_old_setups_table_gets_the_coordinate_fields(tmp_path):
    """A database made before typed coordinates existed: the set-up adds fs_x / fs_y / fs_z."""
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        db.execute("CREATE TABLE survey_setups (id INTEGER NOT NULL PRIMARY KEY, pit_id INTEGER NOT NULL, "
                   "seq INTEGER NOT NULL, at_station VARCHAR(64) NOT NULL)")
        db.con.commit()
        notes = SurveyStore(project_target(tmp_path)).prepare(db)
        assert {"added field fs_x to survey_setups", "added field fs_z to survey_setups"} <= set(notes)


def test_typed_coordinates_work_in_main_pits_and_branches_alike():
    """Any station can be given by its coordinates: the first station of a main pit (from the
    entrance), a station in the middle of a traverse, and a branch's first and later stations -
    in any order of the pits."""
    pits = [Pit(2, "Branch", parent_id=1, start_station="S1"), Pit(1, "Main")]
    main = [SetupRow("E", fs="S1", fs_x=1010.0, fs_y=5010.0, fs_z=249.0),            # first station: typed
            SetupRow("S1", 1.5, bs_station="E", bs_ha=45.0, fs="S2", fs_ha=270.0, fs_za=90.0, fs_sd=10.0, fs_ht=1.5),
            SetupRow("S2", fs="S3", fs_x=1030.0, fs_y=5000.0, fs_z=248.0)]           # mid-traverse: typed
    branch = [SetupRow("S1", fs="B1", fs_x=990.0, fs_y=5030.0, fs_z=249.5),           # branch's first: typed
              SetupRow("B1", 1.5, bs_station="S1", bs_ha=0.0, fs="B2", fs_ha=225.0, fs_za=90.0, fs_sd=5.0,
                       fs_ht=1.5),
              SetupRow("B2", fs="B3", fs_x=985.0, fs_y=5040.0, fs_z=250.0)]           # branch, later: typed
    net = compute(pits, [E], {2: branch, 1: main})
    assert not net.messages
    for name, xyz in (("S1", (1010.0, 5010.0, 249.0)), ("S3", (1030.0, 5000.0, 248.0)),
                      ("B1", (990.0, 5030.0, 249.5)), ("B3", (985.0, 5040.0, 250.0))):
        assert net.coords[name] == xyz and net.status[name] == "entered", name
    assert net.coords["S2"] == pytest.approx((1020.0, 5010.0, 249.0), abs=1e-9)      # due east of S1
    assert net.coords["B2"] == pytest.approx((990.0, 5035.0, 249.5), abs=1e-9)       # carries on from B1
    drawn = {(lg.pit_id, lg.frm, lg.to) for lg in net.legs}
    assert {(1, "E", "S1"), (1, "S2", "S3"), (2, "S1", "B1"), (2, "B2", "B3")} <= drawn
