import numpy as np
import pytest

from dfpredict.io import read_block_csv


@pytest.fixture
def bm_path(demo):
    p = demo / "block_model.csv"
    if not p.exists():
        pytest.skip("block_model.csv not present")
    return p


def test_read_block_csv(bm_path, tmp_path):
    bm = read_block_csv(bm_path, cache_dir=tmp_path)
    assert bm.n_blocks == 60635
    assert set(bm.attribute_names) == {"classification_no", "gold", "sg"}
    sizes = {tuple(s) for s in bm.block_sizes()}
    assert sizes == {(5, 5, 5), (10, 10, 10), (20, 20, 20)}
    assert bm.valid_mask("gold").sum() == 9053
    np.testing.assert_allclose(bm.centroids[0], [1280.0, 7110.0, 0.0])


def test_grid_alignment(bm_path, tmp_path):
    bm = read_block_csv(bm_path, cache_dir=tmp_path)
    g = bm.grid()
    np.testing.assert_allclose(g.cell, [5, 5, 5])
    np.testing.assert_allclose(g.origin, [1270, 7100, -10])  # matches the .mdl header
    # Reconstruct centroids from the integer grid.
    rec = g.origin + (g.ijk_min + g.span / 2.0) * g.cell
    np.testing.assert_allclose(rec, bm.centroids)
    # Sub-blocks never overlap: total volume equals the sum of cell counts.
    occ = np.zeros(g.shape, dtype=np.uint8)
    for (i, j, k), (si, sj, sk) in zip(g.ijk_min, g.span):
        occ[i:i + si, j:j + sj, k:k + sk] += 1
    assert occ.max() == 1


def test_cache_roundtrip(bm_path, tmp_path):
    a = read_block_csv(bm_path, cache_dir=tmp_path)
    caches = list(tmp_path.glob("*.gvbm"))
    assert caches and not list(tmp_path.glob("*.npz"))       # encrypted, never plain arrays
    assert b"gold" not in caches[0].read_bytes()
    b = read_block_csv(bm_path, cache_dir=tmp_path)
    np.testing.assert_array_equal(a.centroids, b.centroids)
    np.testing.assert_array_equal(a.sizes, b.sizes)
    for n in a.attribute_names:
        np.testing.assert_array_equal(a.attributes[n], b.attributes[n])


def test_text_attributes(tmp_path):
    p = tmp_path / "bm.csv"
    p.write_text(
        "xc,yc,zc,xinc,yinc,zinc,rock,au\n"
        "5,5,5,10,10,10,OX,1.5\n"
        "15,5,5,10,10,10,FR,-99\n"
        "25,5,5,10,10,10,OX,2.0\n"
    )
    bm = read_block_csv(p, use_cache=False)
    assert bm.is_categorical("rock")
    labels = [bm.categories["rock"][c] for c in bm.attributes["rock"]]
    assert labels == ["OX", "FR", "OX"]
    assert bm.valid_mask("au").tolist() == [True, False, True]


def test_points_inside_closed_mesh():
    """Ray-parity point-in-solid: a sphere (volume) and a cube (exact), points near edges."""
    import pyvista as pv

    from dfpredict.scene.solid import points_inside

    s = pv.Sphere(radius=10.0, theta_resolution=60, phi_resolution=60).triangulate()
    tri = np.asarray(s.faces).reshape(-1, 4)[:, 1:]
    g = np.mgrid[-12:12:1.0, -12:12:1.0, -12:12:1.0].reshape(3, -1).T + 0.5
    m = points_inside(g, np.asarray(s.points), tri)
    r = np.linalg.norm(g, axis=1)
    far = np.abs(r - 10.0) > 0.4
    assert ((r < 10.0) == m)[far].all()
    assert m.sum() == pytest.approx(s.volume, rel=0.03)
    # a cube, with points exactly on the planes through its edges and vertices (rays hit edges)
    box = pv.Box(bounds=(0, 10, 0, 10, 0, 10)).triangulate()
    btri = np.asarray(box.faces).reshape(-1, 4)[:, 1:]
    pts = np.array([[5, 5, 5], [0.0001, 5, 5], [10.0001, 5, 5], [5, 5, 11], [5, 5, -1],
                    [10, 10, 5], [0, 0, 5], [5, 0, 9.999]], dtype=float)
    got = points_inside(pts, np.asarray(box.points), btri)
    assert got.tolist() == [True, True, False, False, False, got[5], got[6], got[7]]  # edge cases: no crash
    assert points_inside(np.zeros((0, 3)), np.asarray(box.points), btri).shape == (0,)


def test_surface_heights_and_watertight():
    from dfpredict.scene.solid import surface_heights, watertight

    # a tilted plane z = 0.5 x + 10 over [0, 100] x [0, 50], two triangles
    V = np.array([[0, 0, 10], [100, 0, 60], [100, 50, 60], [0, 50, 10]], dtype=float)
    T = np.array([[0, 1, 2], [0, 2, 3]])
    pts = np.array([[20, 10, 0], [80, 40, 0], [50, 25, 0], [150, 25, 0]], dtype=float)
    z = surface_heights(pts, V, T)
    np.testing.assert_allclose(z[:3], 0.5 * pts[:3, 0] + 10, atol=1e-3)
    assert np.isnan(z[3])                                   # off the surface
    assert not watertight(T)
    import pyvista as pv
    box = pv.Box().triangulate()
    assert watertight(np.asarray(box.faces).reshape(-1, 4)[:, 1:])


def test_properties_panel_fits_a_240px_sidebar(demo):
    """No widget may force the properties wider than the sidebar (long paths, wide combos)."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pathlib import Path

    import pyvista as pv
    from PySide6.QtWidgets import QApplication

    from dfpredict.io.surpac_dtm import read_dtm
    from dfpredict.scene.layers import BlockModelLayer, SurfaceLayer
    from dfpredict.scene.scene import Scene
    from dfpredict.ui.properties_panel import PropertiesPanel

    app = QApplication.instance() or QApplication([])
    sc = Scene(pv.Plotter(off_screen=True))
    long_dir = Path("E:/a_rather_long_folder_name/another_long_folder_name/project_1")
    for n in ("ore1", "topo1"):
        sc.add_layer(SurfaceLayer(name=n, path=long_dir / f"{n}.dtm", data=read_dtm(demo / f"{n}.dtm")))
    bm = BlockModelLayer(name="block_model", path=long_dir / "block_model.csv",
                         data=read_block_csv(demo / "block_model.csv"))
    sc.add_layer(bm)
    bm.constrain_solid, bm.value_min = "ore1", 4.0
    panel = PropertiesPanel(on_changed=lambda: None)
    panel.resize(240, 800)
    panel.show()
    for lay in (bm, sc.layers[0]):
        panel.set_layer(lay)
        app.processEvents()
        body = panel.scroll.widget()
        assert body.minimumSizeHint().width() <= panel.scroll.viewport().width() - 12   # room for a scroll bar
    assert all(k != "File" for k, _v in bm.info())


# --- constraints -------------------------------------------------------------------------
def _scene_with(demo, dtms=("ore1", "topo1")):
    import pyvista as pv

    from dfpredict.io.surpac_dtm import read_dtm
    from dfpredict.scene.layers import BlockModelLayer, SurfaceLayer
    from dfpredict.scene.scene import Scene

    sc = Scene(pv.Plotter(off_screen=True))
    for n in dtms:
        sc.add_layer(SurfaceLayer(name=n, path=None, data=read_dtm(demo / f"{n}.dtm")))
    lay = BlockModelLayer(name="bm", path=None, data=read_block_csv(demo / "block_model.csv"))
    sc.add_layer(lay)
    lay.hide_null = False
    lay.rebuild()
    return sc, lay


def test_logic_parser_and_combine():
    from dfpredict.scene.constraints import combine, parse_logic

    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    c = np.array([0, 0, 0, 1], bool)
    assert combine([a, b], "and").tolist() == [True, False, False, False]
    assert combine([a, b], "or").tolist() == [True, True, True, False]
    assert combine([a, b, c], "a and (b or c)").tolist() == [True, False, False, False]
    assert combine([a, b, c], "(a or c) and not b").tolist() == [False, True, False, True]
    assert combine([a, b, c], "A AND NOT (B OR C)").tolist() == [False, True, False, False]
    for bad in ("a and", "a or (b", "a b", "d", "a & b", "__import__", "not", ""):
        with pytest.raises(ValueError):
            parse_logic(bad, 3)


def test_constraint_dtm_and_value_conditions(demo):
    sc, lay = _scene_with(demo)
    bm = lay.data
    total = bm.n_blocks
    lay.constraints = [{"kind": "dtm", "layer": "ore1", "mode": "inside"}]
    lay.rebuild()
    inside = lay.n_shown
    assert inside == int(bm.valid_mask("gold").sum())          # the estimate's own boundary
    lay.constraints[0]["mode"] = "outside"
    lay.rebuild()
    assert lay.n_shown == total - inside
    # surface: above + below = all (every block is under topo1's extent)
    lay.constraints = [{"kind": "dtm", "layer": "topo1", "mode": "below"}]
    lay.rebuild()
    below = lay.n_shown
    lay.constraints[0]["mode"] = "above"
    lay.rebuild()
    assert below + lay.n_shown == total
    # block value, and combined with the solid
    sg = bm.attributes["sg"]
    lay.constraints = [{"kind": "value", "name": "sg", "op": ">", "value": 2.0}]
    lay.rebuild()
    assert lay.n_shown == int(((sg > 2.0) & bm.valid_mask("sg")).sum())
    lay.constraints = [{"kind": "dtm", "layer": "ore1", "mode": "inside"},
                       {"kind": "value", "name": "gold", "op": ">=", "value": 4.0}]
    lay.constraint_logic = "and"
    lay.rebuild()
    gold = bm.attributes["gold"]
    expect = int((lay.condition_mask(lay.constraints[0]) & (gold >= 4.0) & bm.valid_mask("gold")).sum())
    assert lay.n_shown == expect and 0 < expect < inside
    lay.constraint_logic = "or"
    lay.rebuild()
    assert lay.n_shown == inside                               # gold >= 4 lies inside ore1 anyway
    lay.constraint_logic = "a and not b"
    lay.rebuild()
    assert lay.n_shown == inside - expect
    assert lay.constraint_summary() == "a: inside ore1\nb: gold ≥ 4\n→ a and not b"


def test_constraint_on_a_text_attribute(demo):
    sc, lay = _scene_with(demo, dtms=())
    bm = lay.data
    cats = [n for n in bm.attribute_names if bm.is_categorical(n)]
    if not cats:
        pytest.skip("no text attribute in the demo block model")
    name = cats[0]
    code = bm.categories[name][0]
    lay.constraints = [{"kind": "value", "name": name, "op": "=", "value": code}]
    lay.rebuild()
    assert lay.n_shown == int((bm.attributes[name] == 0).sum())


def test_constraint_settings_saved_and_old_workspaces_converted(demo):
    from dfpredict.scene.layers import BlockModelLayer

    sc, lay = _scene_with(demo)
    lay.constraints = [{"kind": "dtm", "layer": "ore1", "mode": "inside"},
                       {"kind": "value", "name": "sg", "op": "<=", "value": 2.5}]
    lay.constraint_logic = "a or b"
    s = lay.settings()
    other = BlockModelLayer(name="x", path=None, data=lay.data)
    other.apply_settings(s)
    assert other.constraints == lay.constraints and other.constraint_logic == "a or b"
    old = BlockModelLayer(name="y", path=None, data=lay.data)
    old.apply_settings({"constrain_solid": "topo1", "constrain_mode": "below"})
    assert old.constraints == [{"kind": "dtm", "layer": "topo1", "mode": "below"}]
    bad = BlockModelLayer(name="z", path=None, data=lay.data)
    bad.apply_settings({"constraints": [{"kind": "nonsense"}]})
    assert bad.constraints == []                                # invalid saved data ignored


def test_constraint_follows_a_removed_dtm(demo):
    sc, lay = _scene_with(demo)
    lay.constraints = [{"kind": "dtm", "layer": "ore1", "mode": "inside"}]
    lay.rebuild()
    ore = next(l for l in sc.layers if l.name == "ore1")
    sc.remove_layer(ore)
    assert lay.constraints_changed() and lay.missing_dtms() == ["ore1"]
    lay.rebuild()
    assert lay.n_shown == lay.data.n_blocks                     # not loaded: not applied


def test_copies_of_a_layer_in_an_old_workspace_are_opened_once(demo, tmp_path):
    """Workspaces of earlier versions may list a layer twice ("Duplicate view", e.g. "Pits (2)"
    of the database survey layer, which could not be removed): only the first is opened, and
    the workspace is saved without the copy."""
    import json
    import shutil

    from dfpredict.workspace import WORKSPACE_FILE, Workspace

    for f in ("ore1_bm.gvbm", "ore1.dtm", "ore1.str"):
        shutil.copy2(demo / f, tmp_path / f)          # keep the time: new files wait a moment
    ws = Workspace(tmp_path)
    layers, errors, _ = ws.load()
    ws.save(layers, None)
    data = json.loads((tmp_path / WORKSPACE_FILE).read_text(encoding="utf-8"))
    by_name = {spec["name"]: spec for spec in data["layers"]}
    survey = next(spec for spec in data["layers"] if spec["type"] == "SurveyLayer")
    data["layers"] += [dict(by_name["ore1_bm"], name="ore1_bm (2)"), dict(survey, name="Pits (2)")]
    (tmp_path / WORKSPACE_FILE).write_text(json.dumps(data), encoding="utf-8")
    layers2, errors2, _ = Workspace(tmp_path).load()
    names = [lay.name for lay in layers2]
    assert "ore1_bm (2)" not in names and "Pits (2)" not in names and not errors2
    assert names.count("ore1_bm") == 1 and sum(type(x).__name__ == "SurveyLayer" for x in layers2) == 1
    Workspace(tmp_path).save(layers2, None)
    saved = json.loads((tmp_path / WORKSPACE_FILE).read_text(encoding="utf-8"))
    assert "Pits (2)" not in [spec["name"] for spec in saved["layers"]]


def test_constraint_dialog(demo):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from dfpredict.ui.constraint_dialog import ConstraintDialog

    app = QApplication.instance() or QApplication([])
    sc, lay = _scene_with(demo)
    applied = []
    dlg = ConstraintDialog(lay, lambda c, l: applied.append((c, l)))
    assert dlg.count.text() == f"Blocks kept: all {lay.data.n_blocks:,}"
    dlg.add_row({"kind": "dtm", "layer": "ore1", "mode": "inside"})
    dlg.add_row({"kind": "value", "name": "gold", "op": ">=", "value": 4.0})
    assert [r.tag.text() for r in dlg.rows] == ["a", "b"]
    assert dlg.rows[0].dtm.currentText() == "ore1 (solid)"
    dlg.rows[1].kind.setCurrentIndex(0)                         # to DTM, then pick the surface
    dlg.rows[1].dtm.setCurrentIndex(dlg.rows[1].dtm.findData("topo1"))
    assert [dlg.rows[1].mode.itemText(i) for i in range(dlg.rows[1].mode.count())] == ["above", "below"]
    assert dlg.rows[1].mode.currentData() == "below"
    dlg.rows[1].kind.setCurrentIndex(1)                         # back to a block value
    dlg.rows[1].attr.setCurrentIndex(dlg.rows[1].attr.findData("gold"))
    dlg.rows[1].op.setCurrentIndex(dlg.rows[1].op.findData(">="))
    dlg.rows[1].number.setValue(4.0)
    dlg.r_custom.setChecked(True)
    dlg.expr.setText("a and (b")
    assert dlg.expr_msg.text() and not dlg.ok_button.isEnabled()
    dlg.expr.setText("a and not b")
    assert dlg.expr_msg.text() == "" and dlg.ok_button.isEnabled()
    assert dlg.count.text().startswith("Blocks kept: ") and " of " in dlg.count.text()
    assert dlg.apply()
    conds, logic = applied[-1]
    assert logic == "a and not b" and conds[0] == {"kind": "dtm", "layer": "ore1", "mode": "inside"}
    assert conds[1] == {"kind": "value", "name": "gold", "op": ">=", "value": 4.0}
    dlg.remove_row(dlg.rows[0])
    assert [r.tag.text() for r in dlg.rows] == ["a"]


def test_value_condition_on_text_codes_synthetic():
    from dfpredict.io.block_model import BlockModel
    from dfpredict.scene.constraints import value_mask

    n = 6
    bm = BlockModel(None, np.zeros((n, 3)), np.ones((n, 3), np.float32),
                    {"rock": np.array([0, 1, 1, 2, -1, 0], np.int32), "au": np.arange(n, dtype=np.float32)},
                    {"rock": ["OX", "TR", "FR"]}, -99.0)
    assert value_mask(bm, {"kind": "value", "name": "rock", "op": "=", "value": "TR"}).tolist() == \
        [False, True, True, False, False, False]
    assert value_mask(bm, {"kind": "value", "name": "rock", "op": "!=", "value": "TR"}).tolist() == \
        [True, False, False, True, False, True]                 # the null block meets neither
    assert value_mask(bm, {"kind": "value", "name": "au", "op": "<", "value": 2}).tolist() == \
        [True, True, False, False, False, False]
