"""Identify: GPU pick + exact resolution on real (off-screen) renders of the demo data."""

from pathlib import Path

import numpy as np
import pytest

from dfpredict.scene.loaders import load_paths
from dfpredict.scene.picking import _projector, pick


@pytest.fixture(scope="module")
def scene():
    import pyvista as pv

    from dfpredict.scene.scene import Scene

    try:
        plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no OpenGL available: {e}")
    sc = Scene(plotter)
    yield sc
    plotter.close()


def _show_only(sc, layer, view="xy"):
    sc.clear()
    sc.add_layer(layer)
    sc.set_view(view)
    sc.plotter.ren_win.Render()


def _screen(sc, local_pt):
    return _projector(sc.plotter.renderer)(np.asarray([local_pt]))[0]


def _rows(res):
    return {k: v for k, v in res.rows if v is not None}


def _load(demo, name):
    layers, errors = load_paths([demo / name])
    assert not errors
    return layers[0]


def test_pick_block_shows_attributes(scene, demo):
    sc = scene
    bm = _load(demo, "block_model.csv")
    _show_only(sc, bm)
    # A top face seen from above: the block with the highest top among the shown blocks.
    mask = bm.visible_mask()
    i = int(np.flatnonzero(mask)[np.argmax((bm.data.centroids[:, 2] + bm.data.sizes[:, 2] / 2)[mask])])
    c = bm.data.centroids[i].copy()
    c[2] += bm.data.sizes[i, 2] / 2
    x, y, _ = _screen(sc, sc.to_local(c))
    hit = pick(sc.plotter.renderer, sc.layers, int(round(x)), int(round(y)))
    assert hit is not None and hit.layer is bm and hit.kind == "polygon"
    res = bm.identify(hit)
    rows = _rows(res)
    assert res.title.startswith("Block")
    for name in bm.data.attribute_names:
        assert name in rows
    # The picked spot lies on the block's top face, inside its footprint.
    j = int(hit.cell_value("block"))
    w = sc.to_world(hit.local)
    half = bm.data.sizes[j] / 2 + 1e-3
    assert np.all(np.abs(w[:2] - bm.data.centroids[j, :2]) <= half[:2])
    assert res.highlight is not None and res.highlight.n_lines == 12


def test_pick_surface_identifies_whole_object(scene, demo):
    sc = scene
    surf = _load(demo, "topo1.dtm")
    _show_only(sc, surf)
    x, y = (np.array(sc.plotter.renderer.GetSize()) / 2).astype(int)
    hit = pick(sc.plotter.renderer, sc.layers, int(x), int(y))
    assert hit is not None and hit.layer is surf
    res = surf.identify(hit)
    rows = _rows(res)
    assert rows["Kind"] == "Surface" and "Triangle" not in rows and "Dip" not in rows
    assert {"Object", "Triangles", "Surface area", "Plan area", "X range"} <= set(rows)
    obj = int(hit.cell_value("object"))
    n = sum(t.n_triangles for o, t in surf.data.iter_trisolations() if o.id == obj)
    assert rows["Triangles"] == f"{n:,}"
    assert res.fill is not None and res.fill.n_cells == n      # the whole object is highlighted
    # The picked Z lies on the triangle's plane: between its corner elevations.
    tri = sc.to_world(hit.cell_points())
    z = sc.to_world(hit.local)[2]
    assert tri[:, 2].min() - 1e-3 <= z <= tri[:, 2].max() + 1e-3


def test_pick_solid_reports_volume(scene, demo):
    sc = scene
    solid = _load(demo, "ore1.dtm")
    _show_only(sc, solid)
    x, y = (np.array(sc.plotter.renderer.GetSize()) / 2).astype(int)
    hit = None
    for dx in range(0, 200, 10):  # find any part of a solid near the centre
        hit = pick(sc.plotter.renderer, sc.layers, int(x) + dx, int(y))
        if hit is not None:
            break
    assert hit is not None
    rows = _rows(solid.identify(hit))
    assert rows["Kind"] == "Solid" and float(rows["Volume"].split()[0].replace(",", "")) > 0


def test_pick_point_identifies_single_point(scene, demo):
    sc = scene
    st = _load(demo, "layout1.str")
    st.show_points = True
    _show_only(sc, st)
    sf = st.data
    s = int(np.argmax(sf.seg_end - sf.seg_start))
    rec = int(sf.seg_start[s]) + 1
    x, y, _ = _screen(sc, sc.to_local(sf.records_xyz[rec]))
    hit = pick(sc.plotter.renderer, sc.layers, int(round(x)), int(round(y)))
    assert hit is not None and hit.layer is st
    res = st.identify(hit)
    if res.title.startswith("Point"):
        rows = _rows(res)
        assert rows["String"] == str(int(sf.seg_string[s]))
        assert res.marker is not None and res.highlight is None


def test_pick_string_near_line(scene, demo):
    sc = scene
    st = _load(demo, "layout1.str")
    _show_only(sc, st)
    sf = st.data
    s = int(np.argmax(sf.seg_end - sf.seg_start))
    a, b = sf.records_xyz[sf.seg_start[s]], sf.records_xyz[sf.seg_start[s] + 1]
    mid = sc.to_local((a + b) / 2)
    x, y, _ = _screen(sc, mid)
    hit = pick(sc.plotter.renderer, sc.layers, int(round(x)) + 3, int(round(y)) + 2)  # not exactly on it
    assert hit is not None and hit.layer is st and hit.kind == "line"
    res = st.identify(hit)
    rows = _rows(res)
    assert rows["String"] == str(int(sf.seg_string[s]))
    assert "Length 3D" in rows and "Description" in rows




def _hit(kind, tag):
    from dfpredict.scene.picking import PickHit

    return PickHit(tag, None, None, 0, kind, np.array([0]), np.array([1.0]), np.zeros(3), 0.0)


def test_choose_prefers_visible_nearest_line():
    from dfpredict.scene.picking import _choose

    surface = (0.0, 0.10, _hit("polygon", "surface"))
    line_behind = (0.5, 0.30, _hit("line", "behind"))     # hidden by the surface at the cursor
    line_on_top = (2.0, 0.05, _hit("line", "on top"))
    line_closer = (0.8, 0.06, _hit("line", "closest"))
    assert _choose([surface, line_behind]).layer == "surface"
    assert _choose([surface, line_behind, line_on_top]).layer == "on top"
    assert _choose([surface, line_on_top, line_closer]).layer == "closest"
    far_line = (5.0, 0.0, _hit("line", "far"))
    assert _choose([surface, far_line]).layer == "surface"
    front = (0.0, 0.05, _hit("polygon", "front"))
    assert _choose([surface, front]).layer == "front"
    assert _choose([]) is None


def test_primitive_ids_map_back_to_cells():
    import pyvista as pv

    from dfpredict.scene.picking import _CellLookup

    pts = np.random.default_rng(0).random((12, 3))
    # triangle (1 primitive), quad (2), pentagon (3); then a polyline of 4 points (3 segments)
    poly = pv.PolyData(pts, faces=[3, 0, 1, 2, 4, 3, 4, 5, 6, 5, 7, 8, 9, 10, 11],
                       lines=[4, 0, 3, 6, 9])
    lk = _CellLookup(poly)
    # Lines come first in VTK cell order and in OpenGL primitive order.
    assert [lk.cell_of_primitive(i) for i in range(3)] == [0, 0, 0]
    assert [lk.cell_of_primitive(i) for i in range(3, 9)] == [1, 2, 2, 3, 3, 3]
    assert lk.cell_of_primitive(9) is None
    assert lk.get(3)[0] == "polygon" and len(lk.get(3)[1]) == 5


def test_pick_empty_space_returns_none(scene, demo):
    sc = scene
    _show_only(sc, _load(demo, "layout1.str"))
    assert pick(sc.plotter.renderer, sc.layers, 3, 3) is None


def test_highlight_does_not_take_part_in_picking(scene, demo):
    sc = scene
    surf = _load(demo, "topo1.dtm")
    _show_only(sc, surf)
    x, y = (np.array(sc.plotter.renderer.GetSize()) / 2).astype(int)
    hit = pick(sc.plotter.renderer, sc.layers, int(x), int(y))
    res = surf.identify(hit)
    sc.set_highlight(res.highlight, hit.local)
    sc.plotter.ren_win.Render()
    again = pick(sc.plotter.renderer, sc.layers, int(x), int(y))
    assert again is not None and again.layer is surf and again.cell_id == hit.cell_id
    sc.clear_highlight()






def test_database_and_mined_shapes_layers_cannot_be_removed():
    from dfpredict.io.database import Target
    from dfpredict.scene.layers import MinedShapesLayer, StringLayer
    from dfpredict.scene.loaders import offline_database_layer

    dh = offline_database_layer(Target("sqlite", path=Path("x.sqlite")), "test")
    assert not dh.removable and not MinedShapesLayer(name="Mined shapes", path=None).removable
    assert StringLayer(name="s", path=None).removable
