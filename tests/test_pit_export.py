"""Pit export: centre lines to a shapefile, drives to a Surpac DTM (all pits, or a pit and its
branches), and the Survey window's Export pits / Export PNG buttons."""

import os
import struct

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.io import pit_export as pe  # noqa: E402
from dfpredict.io.surpac_dtm import read_dtm  # noqa: E402
from dfpredict.io.survey_store import Pit, Station  # noqa: E402
from dfpredict.scene.survey import SurveyData  # noqa: E402


@pytest.fixture
def survey():
    """Main pit M1 (3 legs, with a bend) -> branch B1 -> its branch B1a; a second main pit M2;
    and B2, a branch of M1 with no coordinates yet."""
    pits = [Pit(1, "Main decline", None, "E1", code="P1", profile_width=5, profile_height=4),
            Pit(2, "Level 1", 1, "P1-02", code="P1A", profile_width=3, profile_height=3),
            Pit(3, "Crosscut", 2, "P1A-01", code="P1A1", profile_width=3, profile_height=3),
            Pit(4, "Second", None, "E2", code="P2", profile_width=4, profile_height=4, station_level="middle"),
            Pit(5, "Planned", 1, "P1-01", code="P1B")]
    xyz = {"E1": (0, 0, 0), "P1-01": (100, 0, -10), "P1-02": (150, 50, -15), "P1-03": (150, 120, -20),
           "P1A-01": (250, 50, -15), "P1A-02": (350, 50, -15), "P1A1-01": (250, 0, -15),
           "E2": (1000, 0, 0), "P2-01": (1000, 80, -5)}
    stations = {n: Station(n, None, "station", *v) for n, v in xyz.items()}
    stations["NOXYZ"] = Station("NOXYZ", 5)
    paths = {1: [["E1", "P1-01", "P1-02", "P1-03"]], 2: [["P1-02", "P1A-01", "P1A-02"]], 3: [["P1A-01", "P1A1-01"]],
             4: [["E2", "P2-01"]], 5: [["P1-01", "NOXYZ"]]}
    return SurveyData(pits, stations, paths)


def test_selection_all_pits_or_a_pit_and_its_branches(survey):
    assert [p.code for p in pe.export_pits(survey)] == ["P1", "P1A", "P1A1", "P1B", "P2"]
    assert [p.code for p in pe.export_pits(survey, 1)] == ["P1", "P1A", "P1A1", "P1B"]
    assert [p.code for p in pe.export_pits(survey, 2)] == ["P1A", "P1A1"]     # a branch with its own branch
    assert [p.code for p in pe.export_pits(survey, 4)] == ["P2"]


def _read_shp(path):
    b = path.read_bytes()
    code, words = struct.unpack(">i20xi", b[:28])
    version, shape = struct.unpack("<2i", b[28:36])
    assert (code, version, shape, words * 2) == (9994, 1000, 13, len(b))
    out, pos = [], 100
    while pos < len(b):
        _num, n = struct.unpack(">2i", b[pos:pos + 8])
        c = b[pos + 8:pos + 8 + 2 * n]
        _t, *_box, nparts, npts = struct.unpack("<i4d2i", c[:44])
        o = 44 + 4 * nparts
        xy = np.frombuffer(c[o:o + 16 * npts], "<f8").reshape(-1, 2)
        z = np.frombuffer(c[o + 16 * npts + 16:o + 24 * npts + 16], "<f8")
        m = np.frombuffer(c[o + 24 * npts + 32:o + 32 * npts + 32], "<f8")
        out.append((np.column_stack([xy, z]), m))
        pos += 8 + 2 * n
    shx = path.with_suffix(".shx").read_bytes()
    assert struct.unpack(">i", shx[24:28])[0] * 2 == len(shx) == 100 + 8 * len(out)
    return out


def _read_dbf(path):
    b = path.read_bytes()
    n, head, rec = struct.unpack("<IHH", b[4:12])
    assert len(b) == head + n * rec + 1 and b[-1] == 0x1A
    fields, pos = [], 32
    while b[pos] != 0x0D:
        fields.append((b[pos:pos + 11].rstrip(b"\0").decode(), b[pos + 16]))
        pos += 32
    rows = []
    for i in range(n):
        r, o = b[head + i * rec + 1:head + (i + 1) * rec], 0
        row = {}
        for name, size in fields:
            row[name] = r[o:o + size].decode("utf-8").strip()
            o += size
        rows.append(row)
    return rows


def test_centre_lines_to_shapefile(survey, tmp_path):
    res = pe.write_centerlines_shp(tmp_path / "pits.shp", survey)
    assert res.pits == 4 and res.parts == 4 and res.skipped == ["Planned"]       # no coordinates: left out
    assert all(f.exists() for f in res.files) and {f.suffix for f in res.files} == {".shp", ".shx", ".dbf", ".cpg"}
    recs = _read_shp(tmp_path / "pits.shp")
    pts, m = recs[0]
    np.testing.assert_allclose(pts, [[0, 0, 0], [100, 0, -10], [150, 50, -15], [150, 120, -20]])
    assert m[0] == 0 and m[-1] == pytest.approx(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    rows = _read_dbf(tmp_path / "pits.dbf")
    assert [r["CODE"] for r in rows] == ["P1", "P1A", "P1A1", "P2"]
    assert rows[2]["PARENT"] == "P1A" and rows[2]["MAIN_PIT"] == "P1" and rows[0]["PARENT"] == ""
    assert float(rows[0]["LENGTH_2D"]) == pytest.approx(100 + np.hypot(50, 50) + 70, abs=0.01)
    assert float(rows[1]["WIDTH"]) == 3 and rows[1]["STN_LEVEL"] == "back"


def test_one_pit_and_branches_to_shapefile(survey, tmp_path):
    res = pe.write_centerlines_shp(tmp_path / "p1a", survey, 2)                 # the suffix is added
    assert res.pits == 2 and (tmp_path / "p1a.shp").exists()
    assert [r["CODE"] for r in _read_dbf(tmp_path / "p1a.dbf")] == ["P1A", "P1A1"]


def test_drives_to_surpac_dtm_are_closed_solids(survey, tmp_path):
    res = pe.write_drives_dtm(tmp_path / "drives.dtm", survey)
    assert res.pits == 4 and res.skipped == ["Planned"]
    dtm = read_dtm(tmp_path / "drives.dtm")                    # the app's own reader takes it back
    assert dtm.str_name == "drives.str" and [o.id for o in dtm.objects] == [1, 2, 3, 4]
    # branches start at the parent's wall: P1A1 at P1A's (level) wall, 1.5 m from its station
    lengths = {1: None, 2: None, 3: 50.0 - 1.5, 4: 80.0}       # P1A leaves the inclined P1: see below
    areas = {1: 20, 2: 9, 3: 9, 4: 16}
    for obj, tri in dtm.iter_trisolations():
        assert tri.closed and tri.options["direction"] == "solid"
        V, T = dtm.vertices, tri.triangles
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        vol = np.einsum("ij,ij->i", a, np.cross(b - a, c - a)).sum() / 6
        assert vol < 0                                          # wound like Surpac's own solids
        if lengths[obj.id] is not None:                         # a straight drive: area x length
            assert -vol == pytest.approx(areas[obj.id] * lengths[obj.id], rel=1e-5)   # .str: 3 decimals
        # every edge has its neighbour, and the neighbour points back
        nb = tri.neighbours
        assert (nb > 0).all()
        for i, row in enumerate(nb):
            for n in row:
                assert i + 1 in nb[n - 1]
    # no checksum on the header (Surpac refuses a wrong one; the format allows none)
    assert (tmp_path / "drives.dtm").read_text().splitlines()[0] == "drives.str"
    assert dtm.checksum == "" and dtm.options == {}
    # the string file: each station's profile a closed ring (4 corners + the first again)
    sf = dtm.string_file
    assert sf.n_segments > 0 and all(sf.segment_is_closed(i) and sf.seg_end[i] - sf.seg_start[i] == 5 for i in range(sf.n_segments))
    # ... in Y, X, Z order, with the pit's code and name
    first = (tmp_path / "drives.str").read_text().splitlines()[2]
    sn, y, x, _z, d1, d2 = [s.strip() for s in first.split(",")]
    assert sn == "1" and (d1, d2) == ("P1", "Main decline")
    assert abs(float(x)) < 10 and abs(float(y)) < 10           # near E1 (0, 0)


def test_a_branch_starts_at_its_parents_wall(survey, tmp_path):
    """P1A1 leaves P1A partway (P1A is 3 m wide, running east at y = 50) heading south: its
    solid starts on P1A's south wall (y = 48.5), not at the station inside P1A."""
    pe.write_drives_dtm(tmp_path / "b.dtm", survey, 3)
    dtm = read_dtm(tmp_path / "b.dtm")
    V, T = dtm.vertices, dtm.objects[0].trisolations[0].triangles
    used = V[np.unique(T)]
    assert used[:, 1].max() == pytest.approx(48.5)               # nothing inside P1A
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    vol = -np.einsum("ij,ij->i", a, np.cross(b - a, c - a)).sum() / 6
    assert vol == pytest.approx(3 * 3 * (50 - 1.5))              # 48.5 m of drive left, still closed


def _tee(branches=2):
    """A decline D (5 wide) ending at station J, where a level runs east (L1) and, with
    ``branches`` = 2, west (L2): a T junction."""
    pits = [Pit(1, "Decline", None, "E", code="D", profile_width=5, profile_height=5),
            Pit(2, "Level east", 1, "J", code="L1", profile_width=4, profile_height=4),
            Pit(3, "Level west", 1, "J", code="L2", profile_width=4, profile_height=4)][:1 + branches]
    xyz = {"E": (0, -100, 0), "J": (0, 0, 0), "L1-01": (100, 0, 0), "L2-01": (-100, 0, 0)}
    paths = {1: [["E", "J"]], 2: [["J", "L1-01"]], 3: [["J", "L2-01"]]}
    return SurveyData(pits, {n: Station(n, None, "station", *v) for n, v in xyz.items()},
                      {k: v for k, v in paths.items() if k <= 1 + branches})


def test_a_line_ending_at_a_tee_ends_on_the_level_wall():
    from dfpredict.scene.survey import pit_drives

    d = _tee()                       # decline 5 m high, level 4 m high, both with stations in the back
    (_P, V, _Q), = pit_drives(d, d.pit(1))
    end = V[-4:]                                                 # corners: bottom L, bottom R, top R, top L
    assert end[2:, 1] == pytest.approx([-2, -2])                 # its upper part stops at the level's wall
    assert end[:2, 1] == pytest.approx([0, 0]) and end[:2, 2] == pytest.approx([-5, -5])
    # ... its lowest metre is below the level's floor: that floor wedge runs on to the station (no step)
    assert V[:-4, 1].max() < -99                                 # its start is untouched
    (_P, V1, _Q), = pit_drives(d, d.pit(2))
    assert V1[:4, 0] == pytest.approx([0] * 4)                   # the level halves meet at the station


def test_an_inclined_decline_meets_the_level_without_a_gap():
    """The decline drops 14 m over 100 m into a level of the same 5 x 5 profile (stations in the
    back). At the level's wall the decline is 0.35 m higher than the level: its floor corners
    stop on the wall (they are inside the level there), its roof corners - above the level's
    roof - run on to the station, where the decline's roof comes down to the level's roof. So
    the roof wedge between the wall and the station is kept: no vertical gap."""
    from dfpredict.scene.survey import pit_drives

    d = _tee()
    for p in d.pits:
        p.profile_width = p.profile_height = 5
    d.stations["E"].z = 14.0
    (_P, V, _Q), = pit_drives(d, d.pit(1))
    end = V[-4:]
    assert end[:2, 1] == pytest.approx([-2.5, -2.5])             # floor corners on the wall ...
    assert end[:2, 2] == pytest.approx([-5 + 0.14 * 2.5] * 2)    # ... on the decline's grade
    assert end[2:, 1] == pytest.approx([0, 0], abs=0.01) and end[2:, 2] == pytest.approx([0, 0], abs=0.002)  # roof to the station


def test_a_single_branch_carrying_on_from_the_end_stays_joined():
    from dfpredict.scene.survey import drive_mesh, pit_drives

    d = _tee(branches=1)                                         # an L corner: no cut, no gap
    (P, V, _Q), = pit_drives(d, d.pit(1))
    np.testing.assert_allclose(V, drive_mesh(P, 5, 5)[0])
    (P, V, _Q), = pit_drives(d, d.pit(2))
    np.testing.assert_allclose(V, drive_mesh(P, 4, 4)[0])


def test_an_angled_branch_gets_a_slanted_end_on_the_wall():
    from dfpredict.scene.survey import drive_mesh

    parent = np.array([[0, 0, 0], [100, 0, 0]], float)
    Vp, _ = drive_mesh(parent, 4, 3)                              # walls at y = -2 and y = +2
    Vp = Vp.reshape(-1, 4, 3)
    walls = [Vp[:, 0, :2], Vp[:, 1, :2]]
    ang = np.radians(60)
    branch = np.array([[50, 0, 0], [50 + 100 * np.cos(ang), 100 * np.sin(ang), 10]], float)   # rises 10 m
    V, _ = drive_mesh(branch, 3, 3, start_walls=walls)
    first = V[:4]
    assert first[:, 1] == pytest.approx([2, 2, 2, 2])            # all four corners on the north wall
    assert first[0, 0] != pytest.approx(first[1, 0])             # a slanted end, not square
    # each corner moved along the branch, so it keeps the branch's grade
    plan = np.hypot(first[:, 0] - 50, first[:, 1])
    assert first[0, 2] - (-3) == pytest.approx(0.1 * plan[0], abs=0.6)
    np.testing.assert_allclose(V[4:], drive_mesh(branch, 3, 3)[0][4:])   # the far end unchanged


def test_drive_levels_put_the_station_in_the_back_floor_or_middle(survey, tmp_path):
    pe.write_drives_dtm(tmp_path / "one.dtm", survey, 4)       # P2: stations in the middle, 4 m high
    V = read_dtm(tmp_path / "one.dtm").vertices
    V = V[~np.isnan(V).any(axis=1)]
    assert V[:, 2].max() == pytest.approx(2.0) and V[:, 2].min() == pytest.approx(-5 - 2.0)


def test_nothing_to_export(survey, tmp_path):
    with pytest.raises(ValueError):
        pe.write_drives_dtm(tmp_path / "x.dtm", survey, 5)     # only a pit without coordinates


# --- the Survey window ---------------------------------------------------------------------------
@pytest.fixture
def manager(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from dfpredict.io.database import Database, project_target
    from dfpredict.io.survey_store import SurveyStore
    from dfpredict.ui.survey_manager import SurveyManager

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[2]}"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        store.prepare(db)
    dlg = SurveyManager(store)
    assert dlg.start()
    yield dlg, store, app
    dlg.dirty = False
    dlg.close()


def _straight_pit(dlg, store):
    from dfpredict.io.survey_store import SetupRow

    pid = store.add_pit("Decline", start_station="E", known=(1000.0, 5000.0, 250.0))
    store.save_setups(pid, [SetupRow(at="E", hi=0, bs_bearing=90, fs="P1-01", fs_ha=0, fs_za=90, fs_sd=50, fs_ht=0),
                            SetupRow(at="P1-01", hi=0, bs_station="E", fs="P1-02", fs_ha=180, fs_za=90, fs_sd=40,
                                     fs_ht=0)])
    dlg._reload(select=pid)
    dlg._show_pit(pid)
    return pid


def test_survey_window_exports_pits_and_the_plan_png(manager, tmp_path, monkeypatch):
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QDialog, QFileDialog

    from dfpredict.ui import survey_manager as sm

    dlg, store, app = manager
    _straight_pit(dlg, store)
    for fmt, name in (("shp", "lines.shp"), ("dtm", "solids.dtm")):
        monkeypatch.setattr(sm._ExportPitsDialog, "exec", lambda self: QDialog.Accepted)
        monkeypatch.setattr(sm._ExportPitsDialog, "values", lambda self, f=fmt: (False, f))
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, n=name, **k: (str(tmp_path / n), ""))
        dlg.export_pits()
    assert (tmp_path / "lines.shp").exists() and (tmp_path / "lines.dbf").exists()
    dtm = read_dtm(tmp_path / "solids.dtm")
    assert len(dtm.objects) == 1 and dtm.objects[0].trisolations[0].closed
    # the plan map as a 300 dpi PNG on white
    dlg.map.resize(600, 450)
    dlg.map.fit()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "plan.png"), ""))
    dlg.export_map_png()
    img = QImage(str(tmp_path / "plan.png"))
    assert not img.isNull() and max(img.width(), img.height()) >= 3000
    assert round(img.dotsPerMeterX() * 0.0254) == 300
    assert img.pixelColor(img.width() // 2, 3).name() == "#ffffff"            # white paper


def test_export_needs_the_export_feature(manager, monkeypatch):
    from datetime import date

    from PySide6.QtWidgets import QFileDialog

    from dfpredict import licensing

    dlg, store, app = manager
    _straight_pit(dlg, store)
    lic = licensing.License(name="T", issued=date(2026, 1, 1), expires=None, edition=1,
                            features=licensing.FEAT_DATABASE)                  # Professional: no export
    st = licensing.Status(True, lic, "", None)
    monkeypatch.setattr(licensing, "refresh", lambda: st)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: pytest.fail("not licensed"))
    dlg.export_pits()
    dlg.export_map_png()


def test_pit_volume_of_a_pit_and_with_its_branches():
    """Properties show the volume of the drives as drawn: the decline ends on the level's wall, so the
    junction is not counted twice; 'with branches' adds the whole family."""
    from dfpredict.scene.survey import SurveyLayer

    lay = SurveyLayer(name="Pits", path=None, data=_tee(), visible=False)
    assert lay.pit_volume(2) == pytest.approx(4 * 4 * 100, rel=1e-6)
    # the decline (5 m) stops on the wall of the 4 m high level; its lowest metre runs on under the level's floor
    assert 5 * 5 * (100 - 2) < lay.pit_volume(1) < 5 * 5 * 100
    assert lay.pit_volume(1, branches=True) == pytest.approx(lay.pit_volume(1) + 2 * 1600, rel=1e-6)
    assert lay.pit_family(1) == {1, 2, 3} and lay.pit_volume(3, branches=True) == lay.pit_volume(3)
