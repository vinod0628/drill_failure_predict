"""Decision support (geoviewer/ai): reserve forecast, cut check, best levels, the closure doctor, the route
planner, and their windows. All offline."""

import dataclasses
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.ai import BAD, INFO, OK, WARN  # noqa: E402
from dfpredict.ai import reserve_advisor as ra  # noqa: E402
from dfpredict.ai.closure_doctor import diagnose  # noqa: E402
from dfpredict.ai.route_planner import Obstacles, RouteRequest, _layout, plan_route, write_csv, write_str  # noqa: E402
from dfpredict.io.block_model import BlockModel  # noqa: E402
from dfpredict.io.survey_store import Pit, SetupRow, Station, bearing_of  # noqa: E402
from dfpredict.scene.production import Depletion, MinedShape, day_of, reserve_blocks  # noqa: E402


# --- helpers ----------------------------------------------------------------------------------------
def grid_model(n=10, size=5.0, grade=None):
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    c = (np.stack([i, j, k], -1).reshape(-1, 3) + 0.5) * size
    g = np.full(len(c), 2.0, np.float32) if grade is None else grade(c).astype(np.float32)
    return BlockModel(None, c.astype(np.float64), np.full((len(c), 3), size, np.float32),
                      {"sg": np.full(len(c), 2.5, np.float32), "au": g})


def depletion(bm):
    idx, dens, grade = reserve_blocks(bm, None, "sg", 2.7, "au")
    return Depletion(bm, idx, dens, grade)


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def readings(pts, names, hi=1.55, ht=1.35):
    """Consistent field readings of a closed traverse through ``pts`` (the first is the known entrance)."""
    rows, n = [], len(pts)
    seq = list(range(n)) + [0]
    for k in range(n):
        a, f = seq[k], seq[k + 1]
        A, F = pts[a], pts[f]
        dx, dy, dz = F[0] - A[0], F[1] - A[1], F[2] - A[2]
        hd = math.hypot(dx, dy)
        line = dz - hi + ht
        sd, za = math.hypot(hd, line), math.degrees(math.atan2(hd, line))
        brg_f = bearing_of(dx, dy)
        if k == 0:
            rows.append(SetupRow(names[a], hi, bs_bearing=37.0, bs_ha=0.0, fs=names[f], fs_ha=(brg_f - 37.0) % 360,
                                 fs_za=za, fs_sd=round(sd, 3), fs_ht=ht))
        else:
            B = pts[seq[k - 1]]
            brg_b = bearing_of(B[0] - A[0], B[1] - A[1])
            rows.append(SetupRow(names[a], hi, bs_station=names[seq[k - 1]], bs_ha=0.0, fs=names[f],
                                 fs_ha=(brg_f - brg_b) % 360, fs_za=za, fs_sd=round(sd, 3), fs_ht=ht))
    return rows


PTS = [(1000.0, 5000.0, 250.0), (1062.3, 4981.7, 244.1), (1118.9, 5023.4, 239.6), (1105.2, 5097.8, 236.0),
       (1041.6, 5110.3, 241.7), (987.4, 5068.9, 246.2)]
NAMES = ["E", "S1", "S2", "S3", "S4", "S5"]
E = Station("E", 1, "known", *PTS[0])


# --- reserve: forecast ----------------------------------------------------------------------------------
def test_forecast_uses_the_recent_full_months():
    months = [f"2026-0{m}" for m in range(1, 8)]
    tonnes = [10000, 12000, 11000, 9000, 13000, 12000, 5000]          # July is not over yet
    rows = [{"month": m, "tonnes": t, "metal": 2.0 * t} for m, t in zip(months, tonnes)]
    f = ra.forecast(rows, 1_000_000.0, 900_000.0, 1_800_000.0, "2026-07-15")
    assert f.ok and f.used_months == months[:6]
    assert f.rate == pytest.approx(11500.0)                              # median of the six full months
    assert f.rate_low <= f.rate <= f.rate_high
    assert f.months_left == pytest.approx(900_000 / 11500.0)
    assert f.end_month == "2033-02"                                      # 78.3 months -> 79 after July 2026
    assert f.trend == pytest.approx((34000 / 3 - 11000) / 11000)          # steady
    levels = [lv for lv, _t in f.findings]
    assert levels[0] == INFO and OK in levels                            # rate sentence; steady, grade in line
    assert "11,500 t a month" in f.findings[0][1]
    xs, ys = ra.projection(f, f.rate)
    assert ys[-1] == pytest.approx(f.total) and all(b >= a for a, b in zip(ys, ys[1:]))


def test_forecast_warns_about_high_grading_and_falling_production():
    rows = [{"month": f"2026-{m:02d}", "tonnes": t, "metal": t * g} for m, t, g in
            zip(range(1, 8), [20000, 20000, 20000, 9000, 9000, 9000, 0], [2, 2, 2, 4, 4, 4, 0])]
    f = ra.forecast(rows, 500_000.0, 380_000.0, 380_000.0 * 2.0, "2026-07-02")
    text = " ".join(t for _lv, t in f.findings)
    assert "Production fell 55%" in text and "high-grading" in text
    assert ra.forecast([], 10.0, 10.0, 10.0, "2026-07-01").ok is False


# --- reserve: cut check and best levels -----------------------------------------------------------------
def test_cut_check_explains_each_cut():
    bm = grid_model()
    dep = depletion(bm)
    good = MinedShape(1, "2026-01-05", "Good", "xy", square(5, 5, 25, 25), 10.0, 20.0)
    outside = MinedShape(2, "2026-01-06", "Outside", "xy", square(100, 100, 120, 120), 10.0, 20.0)
    again = MinedShape(3, "2026-02-01", "Again", "xy", square(5, 5, 25, 25), 10.0, 20.0)
    thick = MinedShape(4, "2026-02-02", "Thick", "xy", square(30, 30, 45, 45), 5.0, 45.0)
    shapes = [good, outside, again, thick]
    dep.apply(shapes)
    checks = {c.name: c for c in ra.review_cuts(dep, shapes, "g/t", day_of("2026-03-01"))}
    assert checks["Good"].level == OK and checks["Good"].tonnes > 0 and checks["Good"].fill == pytest.approx(1.0)
    assert checks["Outside"].level == BAD and "no new ore" in checks["Outside"].reasons[0][1]
    assert checks["Again"].level == BAD and checks["Again"].already > 0
    assert any("thickness" in t for _lv, t in checks["Thick"].reasons) and checks["Thick"].level == WARN
    order = [c.level for c in ra.review_cuts(dep, shapes, "g/t")]
    assert order == sorted(order, key={BAD: 0, WARN: 1, OK: 2}.get)      # problems first


def test_best_levels_follow_the_rich_ore_not_mined_yet():
    bm = grid_model(grade=lambda c: np.where((c[:, 2] > 30) & (c[:, 2] < 40), 8.0, 1.0))
    dep = depletion(bm)
    sh = MinedShape(9, "2026-03-01", "New", "xy", square(0, 0, 50, 50), 0.0, 10.0)
    best = ra.suggest_levels(dep, sh, 10.0, "metal")
    assert best.ok and best.from_ == pytest.approx(30.0, abs=1.3) and best.to - best.from_ == pytest.approx(10.0)
    assert best.grade == pytest.approx(8.0, rel=0.05) and best.alternatives
    # once that band is mined, the suggestion moves to ore not mined yet
    dep.apply([MinedShape(1, "2026-01-01", "Rich", "xy", square(0, 0, 50, 50), 30.0, 40.0)])
    nxt = ra.suggest_levels(dep, sh, 10.0, "metal")
    assert nxt.ok and (nxt.to <= 30.0 + 1e-6 or nxt.from_ >= 40.0 - 1e-6)
    assert not ra.suggest_levels(dep, MinedShape(9, "2026-03-01", "Far", "xy", square(200, 200, 210, 210), 0, 5),
                                 5.0).ok


# --- survey: closure doctor -------------------------------------------------------------------------------
def _plant(i, **change):
    rows = readings(PTS, NAMES)
    rows[i] = dataclasses.replace(rows[i], **change)
    return rows


@pytest.mark.parametrize("setup, change, kind", [
    (2, "fs_face", "fs_face"),
    (1, "ha_swap", "ha_swap"),
    (3, "hi_ht", "hi_ht"),
    (1, "za_vertical", "za_vertical"),
    (4, "sd_digits", "sd_digits"),
    (3, "bs_station", "bs_station"),
])
def test_the_planted_mistake_is_found_first(setup, change, kind):
    r = readings(PTS, NAMES)[setup]
    sd = f"{r.fs_sd:.3f}"
    edits = {"fs_face": dict(fs_ha=(r.fs_ha + 180) % 360), "ha_swap": dict(bs_ha=r.fs_ha, fs_ha=0.0),
             "hi_ht": dict(hi=r.fs_ht, fs_ht=r.hi), "za_vertical": dict(fs_za=90 - r.fs_za),
             "sd_digits": dict(fs_sd=float(sd[1] + sd[0] + sd[2:])), "bs_station": dict(bs_station="S1")}[change]
    d = diagnose([Pit(1, "Main")], [E], {1: _plant(setup, **edits)}, {}, 1)
    assert d.status == "found"
    best = d.fixes[0]
    if kind != "fs_face":                      # a backsight 180° off explains it as well as a foresight: a tie
        assert best.kind == kind
    assert best.setup == setup and best.passes and best.error < 0.01
    assert best.confidence in ("very likely", "likely")
    assert f"setup {setup + 1}" in d.message


def test_closure_doctor_without_a_mistake_or_a_closure():
    ok = diagnose([Pit(1, "Main")], [E], {1: readings(PTS, NAMES)}, {}, 1)
    assert ok.status == "passes" and not ok.fixes
    open_rows = readings(PTS, NAMES)[:3]
    assert diagnose([Pit(1, "Main")], [E], {1: open_rows}, {}, 1).status == "no_closure"
    # two mistakes at once: no single correction closes it, and it says so
    rows = _plant(1, fs_sd=readings(PTS, NAMES)[1].fs_sd + 0.37)
    rows[3] = dataclasses.replace(rows[3], fs_sd=rows[3].fs_sd - 0.52)
    d = diagnose([Pit(1, "Main")], [E], {1: rows}, {}, 1)
    assert d.status in ("not_found", "found") and not any(f.passes for f in d.fixes)


# --- pit: route planner ------------------------------------------------------------------------------------
def test_route_in_open_ground_is_one_straight_leg():
    r = plan_route(RouteRequest((0.3, 0.7, 0.11), (400.0, 150.0, -30.37)), Obstacles())
    assert r.ok and len(r.legs) == 1
    assert np.allclose(r.points[0], (0.3, 0.7, 0.11)) and np.allclose(r.points[-1], (400.0, 150.0, -30.37))
    assert r.steepest <= 1 / 7 * 1.001 and r.length == pytest.approx(math.dist((0.3, 0.7, 0.11),
                                                                                (400.0, 150.0, -30.37)))


def test_a_route_leaving_the_end_of_a_wide_drive_goes_straight_when_nothing_is_in_the_way():
    # the user's case: start at the end of a 10 x 10 m drive that runs north-east, target 104 m to the
    # north-north-west and level - the free zone at the junction must cover the whole band the drive blocks
    s = (1177.97, 7444.01, -329.39)
    t = (1121.10, 7531.25, -329.00)
    drive = (np.array([[1209.4, 7470.2, -329.39], s]), 0.5 * math.hypot(10, 10), "Level -330 SW")
    r = plan_route(RouteRequest(s, t, 1 / 7, 10.0, 10.0, None, 10.0, 10.0, min_radius=15.0),
                   Obstacles(drives=[drive]))
    assert r.ok and len(r.legs) == 1 and r.length == pytest.approx(math.dist(s, t))


def test_straight_legs_are_checked_with_the_true_clearances_not_the_grid_margin():
    # level target 100 m away at 60° to the drive the route leaves: a straight drive keeps more than the
    # 10 m clearance, so it must be one leg (the grid's safety margin used to bend it)
    drive = (np.array([[-200.0, 0.0, 0.0], [200.0, 0.0, 0.0]]), 0.5 * math.hypot(10, 10), "D")
    a = math.radians(60)
    t = (100 * math.cos(a), 100 * math.sin(a), 0.0)
    r = plan_route(RouteRequest((0.0, 0.0, 0.0), t, 1 / 7, 10.0, 10.0, None, 10.0, 10.0, min_radius=15.0),
                   Obstacles(drives=[drive]))
    assert r.ok and len(r.legs) == 1 and r.drive_clear >= 10.0
    # ... but at 45° a straight drive would come within 10 m of it just past the junction: it bends away
    a = math.radians(45)
    t = (100 * math.cos(a), 100 * math.sin(a), 0.0)
    r = plan_route(RouteRequest((0.0, 0.0, 0.0), t, 1 / 7, 10.0, 10.0, None, 10.0, 10.0, min_radius=15.0),
                   Obstacles(drives=[drive]))
    assert r.ok and len(r.legs) > 1 and r.drive_clear >= 9.5
    # old workings: a stope whose wall is 11 m (wall to wall) from the straight line leaves it straight
    void = MinedShape(1, "2026-01-01", "Stope", "xy", square(150, 14.5, 250, 140), -40.0, 10.0)
    r = plan_route(RouteRequest((0.0, 0.0, 0.0), (400.0, 0.0, 0.0), void_clearance=10.0), Obstacles(voids=[void]))
    assert r.ok and len(r.legs) == 1 and r.void_clear >= 10.0


def test_a_target_a_few_mm_above_the_start_keeps_a_normal_grid():
    # the user's case: target Z typed as -329.39, start at -329.3919 (1.9 mm lower). The levels used to
    # be spaced 1.9 mm apart, the grid grew to 190 m cells and no route was found
    s, t = (1296.934, 7513.958, -329.3919), (1249.24, 7600.71, -329.39)
    req = RouteRequest(s, t, 1 / 7, 10.0, 10.0, None, 10.0, 10.0, min_radius=15.0)
    assert _layout(req).cell < 6.0
    r = plan_route(req, Obstacles())
    assert r.ok and len(r.legs) == 1 and r.length == pytest.approx(math.dist(s, t))


def test_a_steep_target_makes_the_route_wind_within_the_gradient():
    r = plan_route(RouteRequest((0.0, 0.0, 0.0), (150.0, 50.0, -90.2)), Obstacles())
    assert r.ok and r.needed > r.direct and r.plan_length >= r.needed - 1e-6
    assert r.steepest <= 1 / 7 * 1.001 and len(r.legs) > 1
    assert any("makes a loop" in t for _lv, t in r.findings)


def test_the_route_keeps_away_from_old_workings_and_drives():
    void = MinedShape(1, "2026-01-01", "Stope", "xy", square(150, -60, 250, 140), -40.0, 10.0)
    r = plan_route(RouteRequest((0.0, 0.0, 0.0), (400.0, 40.0, -20.0), void_clearance=10.0),
                   Obstacles(voids=[void]))
    assert r.ok and r.void_clear is not None and r.void_clear >= 9.5
    assert r.plan_length > r.direct                           # it goes around
    drive = (np.array([[100.0, -40.0, 0.0], [100.0, 40.0, 0.0]]), 2.5, "P9")
    r2 = plan_route(RouteRequest((0.0, 0.0, 0.0), (300.0, 0.0, 0.0)), Obstacles(drives=[drive]))
    assert r2.ok and r2.drive_clear >= 9.0


def test_no_route_through_a_closed_wall_and_the_export(tmp_path):
    wall = MinedShape(1, "2026-01-01", "Wall", "xy", square(80, -500, 120, 500), -300.0, 300.0)
    r = plan_route(RouteRequest((0.0, 0.0, 0.0), (200.0, 0.0, 0.0)), Obstacles(voids=[wall]))
    assert not r.ok and "No route" in r.message
    ok = plan_route(RouteRequest((0.0, 0.0, 0.0), (120.0, 30.0, -5.0)), Obstacles())
    write_str(tmp_path / "route.str", ok.points, "R")
    from dfpredict.io.surpac_str import read_str

    s = read_str(tmp_path / "route.str")
    xyz = s.records_xyz[~np.isnan(s.records_xyz[:, 0])]
    assert np.allclose(xyz, ok.points, atol=1e-3)
    write_csv(tmp_path / "route.csv", ok, "R")
    assert (tmp_path / "route.csv").read_text(encoding="utf-8-sig").startswith("station,x,y,z")


# --- the windows ---------------------------------------------------------------------------------------
@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_reserve_advisor_window(qapp):
    from dfpredict.ui.reserve_advisor_dialog import ReserveAdvisorDialog

    bm = grid_model()
    dep = depletion(bm)
    shapes = [MinedShape(i + 1, f"2026-0{m}-10", f"Cut {i + 1}", "xy", square(5, 5, 45, 45), 5.0 * i, 5.0 * i + 5)
              for i, m in enumerate([1, 2, 3, 4, 5, 6])]
    dep.apply(shapes)
    log = SimpleNamespace(shapes=shapes, sorted_shapes=lambda: shapes)
    opened = []
    mgr = SimpleNamespace(reserve=SimpleNamespace(ready=True, error="", log=log, dep=dep, setup={"unit": "g/t"}),
                          as_of="2026-06-20", as_of_day=day_of("2026-06-20"), edit_shape=opened.append, win=None)
    dlg = ReserveAdvisorDialog(mgr)
    assert dlg.tabs.isEnabled() and dlg.c_rate.value.text().endswith("t")
    assert any("t a month" in t for t in dlg.life_findings.texts())
    assert dlg.table.rowCount() == 6 and dlg.c_ok.value.text() == "6"
    dlg._open_cut()
    assert opened == [dlg.checks[0].sid]
    dlg.close()
    mgr.reserve.ready, mgr.reserve.error = False, "Choose the grade attribute"
    dlg = ReserveAdvisorDialog(mgr)
    assert not dlg.tabs.isEnabled() and "Choose the grade attribute" in dlg.notice.text()
    dlg.close()


@pytest.fixture
def survey(tmp_path, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from dfpredict.io.database import Database, project_target
    from dfpredict.io.survey_store import SurveyStore
    from dfpredict.ui.survey_manager import SurveyManager

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[2]}"))
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        store.prepare(db)
    pid = store.add_pit("Loop", start_station="E", known=PTS[0])
    store.save_setups(pid, readings(PTS, NAMES))
    sm = SurveyManager(store)
    assert sm.start()
    yield sm
    sm.dirty = False
    sm.close()


def test_closure_window_finds_and_applies_the_correction(survey):
    from dfpredict.ui.closure_dialog import ClosureDialog

    sm = survey
    good = sm.rows[2].fs_sd
    sd = f"{good:.3f}"
    sm.rows[2] = dataclasses.replace(sm.rows[2], fs_sd=float(sd[1] + sd[0] + sd[2:]))
    sm._recompute()
    dlg = ClosureDialog(sm)
    assert dlg.diag.status == "found" and "slope distance" in dlg.a_text.text()
    assert dlg.table.rowCount() >= 1 and dlg.b_use.isEnabled()
    dlg.use()
    assert sm.rows[2].fs_sd == pytest.approx(good) and sm.dirty
    assert dlg.diag.status == "passes"                       # checked again after the correction
    dlg.close()


def test_route_window_designs_and_explains(survey):
    from dfpredict.ui.route_dialog import RouteDialog

    sm = survey
    dlg = RouteDialog(sm)
    s = dlg._stations()[dlg.c_start.currentText()]
    dlg.tx.setValue(float(s[0]) + 250.0)
    dlg.ty.setValue(float(s[1]) - 300.0)
    dlg.tz.setValue(float(s[2]) - 30.0)
    dlg.s_cost.setValue(1000.0)
    dlg.run()
    assert dlg.route is not None and dlg.route.ok
    assert dlg.c_len.value.text().endswith(" m") and dlg.c_grad.value.text().startswith("1 in")
    assert dlg.stations.rowCount() == len(dlg.route.points)
    assert dlg.findings.texts()[0].startswith("A route was found")
    dlg.close()


def test_a_winding_route_has_gradual_bends_and_room_between_its_legs():
    from dfpredict.ai.route_planner import _bend

    r = plan_route(RouteRequest((0.0, 0.0, 0.0), (120.0, 90.0, -61.0), drive_clearance=10.0, width=5.0, height=5.0),
                   Obstacles())
    assert r.ok and r.plan_length >= r.needed - 1e-6 and r.steepest <= 1 / 7 * 1.001
    bends = [_bend(r.points[k - 1], r.points[k], r.points[k + 1]) for k in range(1, len(r.points) - 1)]
    assert max(bends) <= 51.0                                      # no hairpins, no knots
    assert r.self_clear is None or r.self_clear >= 9.5             # the loop's legs keep the clearance
    assert r.length <= 1.15 * r.needed                             # still short: length comes first
    assert np.allclose(r.points[-1], (120.0, 90.0, -61.0))          # ends exactly on the target


def test_route_window_map_follows_the_inputs_and_exports(survey, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from dfpredict.ui.route_dialog import RouteDialog

    dlg = RouteDialog(survey)
    assert dlg.plan.stations and dlg.plan.items                    # the pits and stations from the start
    s = dlg._stations()[dlg.c_start.currentText()]
    dlg.tx.setValue(float(s[0]) + 100.0)
    pin = dict((text.split()[0], pt) for pt, _c, text, _s in dlg.plan.pins)
    assert pin["B"][0] == pytest.approx(float(s[0]) + 100.0)       # the target moves at once
    assert "Straight line A → B" in dlg.direct.text()
    dlg._map_clicked(float(s[0]) + 40.0, float(s[1]) - 60.0)       # a click on the map puts the target there
    assert dlg.tx.value() == pytest.approx(float(s[0]) + 40.0) and dlg.ty.value() == pytest.approx(float(s[1]) - 60)
    before = dlg.plan._s
    dlg.plan.zoom_by(2.0)
    assert dlg.plan._s == pytest.approx(before * 2.0)
    monkeypatch.setattr(dlg, "_allowed", lambda _t: True)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "plan.png"), ""))
    dlg.export_png()
    from PySide6.QtGui import QImage

    img = QImage(str(tmp_path / "plan.png"))
    assert not img.isNull() and max(img.width(), img.height()) == 3300
    assert round(img.dotsPerMeterX() * 0.0254) == 300
    dlg.close()


def test_a_start_direction_set_by_the_user_is_followed():
    from dfpredict.ai.route_planner import bearing_of

    drive = (np.array([[-200.0, 0.0, 0.0], [200.0, 0.0, 0.0]]), 0.5 * math.hypot(10, 10), "D")
    for b in (0.0, 30.0):
        r = plan_route(RouteRequest((0.0, 0.0, 0.0), (60.0, 150.0, -5.0), 1 / 7, 10.0, 10.0, None, 10.0, 10.0,
                                    min_radius=15.0, start_bearing=b), Obstacles(drives=[drive]))
        assert r.ok and bearing_of(r.points[0], r.points[1]) == pytest.approx(b, abs=0.01)
        assert r.steepest <= 1 / 7 * 1.001 and r.drive_clear >= 9.5
        assert any("as asked" in t for _lv, t in r.findings)
    # along the existing drive (30° off it): the new drive would run beside it - refused, with the reason
    r = plan_route(RouteRequest((0.0, 0.0, 0.0), (-50.0, 200.0, 0.0), 1 / 7, 10.0, 10.0, None, 10.0, 10.0,
                                min_radius=15.0, start_bearing=60.0), Obstacles(drives=[drive]))
    assert not r.ok and "start direction" in r.message


def test_the_best_start_point_along_the_drives():
    from dfpredict.ai.route_planner import best_start

    hs = 0.5 * math.hypot(5, 5)
    main = np.array([[-200.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    xc = np.array([[100.0, 0.0, 0.0], [100.0, 60.0, 0.0]])
    lines = [(main, ["M-01", "M-02", "M-03", "M-04"], "Main"), (xc, ["M-03", "XC-01"], "XC")]
    obs = Obstacles(drives=[(pts, hs, name) for pts, _n, name in lines])
    t = (50.0, 80.0, 0.0)
    r = best_start(RouteRequest(t, t, 1 / 7, 10.0, 10.0, None, 5.0, 5.0), obs, lines)
    # the end of the crosscut is nearest: a straight 54 m drive from there (not 80 m from the main drive)
    assert r.ok and len(r.legs) == 1 and r.start_point.drive == "XC"
    assert r.length == pytest.approx(math.hypot(50, 20), abs=0.5)
    assert r.findings[0][1].startswith("Best start point: at station XC-01")
    # heading north is asked: the target due north of a point of the main drive - straight from there
    r = best_start(RouteRequest(t, t, 1 / 7, 10.0, 10.0, None, 5.0, 5.0, start_bearing=0.0), obs, lines)
    assert r.ok and len(r.legs) == 1 and r.start_point.drive == "Main"
    assert r.points[0][0] == pytest.approx(50.0, abs=0.01) and r.length == pytest.approx(80.0, abs=0.01)
    # a target far below: no straight drive within the gradient from anywhere - the shortest safe winding route
    r = best_start(RouteRequest((50.0, 80.0, -40.0), (50.0, 80.0, -40.0), 1 / 7, 10.0, 10.0, None, 5.0, 5.0),
                   obs, lines)
    assert r.ok and r.steepest <= 1 / 7 * 1.001 and r.plan_length >= 40 * 7 - 1


def test_the_best_start_point_only_on_the_pits_chosen():
    from dfpredict.ai.route_planner import best_start

    hs = 0.5 * math.hypot(5, 5)
    main = np.array([[-200.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    xc = np.array([[100.0, 0.0, 0.0], [100.0, 60.0, 0.0]])
    lines = [(main, ["M-01", "M-02", "M-03", "M-04"], "Main"), (xc, ["M-03", "XC-01"], "XC")]
    obs = Obstacles(drives=[(pts, hs, name) for pts, _n, name in lines])
    t = (50.0, 80.0, 0.0)
    r = best_start(RouteRequest(t, t, 1 / 7, 10.0, 10.0, None, 5.0, 5.0), obs, lines, only={0})
    assert r.ok and r.start_point.drive == "Main" and len(r.legs) == 1      # the crosscut is not searched ...
    assert r.length == pytest.approx(80.0, abs=0.5)
    assert r.drive_clear is None or r.drive_clear >= 9.5                   # ... but still kept clear of
