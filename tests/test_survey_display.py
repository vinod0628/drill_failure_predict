"""The survey layer in the main window: pits listed as a tree in the layers list, and each
pit's display settings (shown / hidden, colour or the parent's colour, centre lines and
stations for main pits), saved with the workspace."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict.io.survey_store import Pit, SetupRow, Station, compute  # noqa: E402
from dfpredict.scene.survey import FAMILY_COLORS, SurveyData, SurveyLayer, pit_paths  # noqa: E402

E = Station("E", 1, "known", 1000.0, 5000.0, 250.0)
F = Station("F", 4, "known", 2000.0, 5000.0, 250.0)


def survey_layer() -> SurveyLayer:
    """Main pit 1 (E -> A1 -> A2) with branch 2 at A1 (-> B1), which has branch 3 at B1
    (-> C1); main pit 4 from F (-> D1)."""
    pits = [Pit(1, "Decline", code="P1"), Pit(2, "Level", 1, "A1", code="P1A"), Pit(3, "XC", 2, "B1", code="P1A1"),
            Pit(4, "North", code="P2", start_station="F")]
    pits[0].start_station = "E"

    def leg(at, fs, brg):
        return SetupRow(at, 1.5, fs=fs, fs_bearing=brg, fs_bearing_type="grid", fs_za=90.0, fs_sd=20.0, fs_ht=1.5)
    setups = {1: [leg("E", "A1", 0.0), leg("A1", "A2", 0.0)], 2: [leg("A1", "B1", 90.0)], 3: [leg("B1", "C1", 0.0)],
              4: [leg("F", "D1", 180.0)]}
    net = compute(pits, [E, F], setups)
    owner = {"A1": 1, "A2": 1, "B1": 2, "C1": 3, "D1": 4}
    stations = {"E": E, "F": F}
    for n, pid in owner.items():
        stations[n] = Station(n, pid, "station", *net.coords[n], status="open")
    return SurveyLayer(name="Survey (database)", path=None,
                       data=SurveyData(pits, stations, pit_paths(setups), "test"))


def test_colours_are_inherited_unless_a_pit_has_its_own():
    lay = survey_layer()
    assert lay.pit_color(1) == FAMILY_COLORS[0] and lay.pit_color(4) == FAMILY_COLORS[1]
    assert lay.pit_color(2) == lay.pit_color(3) == FAMILY_COLORS[0]          # branches: the main pit's
    lay.set_pit_style(1, color=(1.0, 0.0, 0.0))
    assert lay.pit_color(2) == lay.pit_color(3) == (1.0, 0.0, 0.0)           # follow the parent
    lay.set_pit_style(2, color=(0.0, 1.0, 0.0))
    assert lay.pit_color(2) == lay.pit_color(3) == (0.0, 1.0, 0.0)           # own colour, passed down
    assert lay.default_color(2) == (1.0, 0.0, 0.0)                            # what "inherit" would give
    lay.set_pit_style(2, color=None)                                          # inherit again
    assert lay.pit_color(3) == (1.0, 0.0, 0.0)


def test_hiding_a_pit_hides_its_branches_and_settings_are_saved():
    lay = survey_layer()
    lay.set_pit_style(2, visible=False)
    assert lay.pit_shown(1) and not lay.pit_shown(2) and not lay.pit_shown(3) and lay.pit_shown(4)
    lay.set_pit_style(1, centerline=False, stations=False, color=(0.2, 0.4, 0.6))
    saved = json.loads(json.dumps(lay.settings()))                            # into the workspace file
    again = survey_layer()
    again.apply_settings(saved)
    assert not again.pit_shown(3) and again.pit_color(3) == pytest.approx((0.2, 0.4, 0.6))
    assert again.pit_style(1)["centerline"] is False and again.pit_style(1)["stations"] is False
    again.apply_settings({"pit_styles": "nonsense"})                          # a broken file is ignored
    assert again.pit_style(1)["centerline"] is False


@pytest.fixture(scope="module")
def scene():
    import pyvista as pv

    from dfpredict.scene.scene import Scene

    try:
        plotter = pv.Plotter(off_screen=True, window_size=(600, 500))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no OpenGL available: {e}")
    sc = Scene(plotter)
    yield sc
    plotter.close()


def _drawn(lay):
    """(pits with drives, pits with centre lines, station names drawn)."""
    import pyvista as pv

    lay.scene.plotter.ren_win.Render()                 # the meshes are brought up to date by a render
    drives, lines = set(), set()
    for a in lay.actors:
        mapper = a.GetMapper()
        if mapper is None or not hasattr(mapper, "GetInput"):          # the labels
            continue
        m = a.mapper.dataset                                         # what is drawn (pyvista)
        if "pit" in m.cell_data:
            ids = {int(x) for x in m.cell_data["pit"]}
            (lines if m.n_lines else drives).update(ids)
    return drives, lines, list(getattr(lay, "_station_names", []))


def test_hidden_pits_centre_lines_and_stations_are_not_drawn(scene):
    lay = survey_layer()
    scene.clear()
    scene.add_layer(lay)
    drives, lines, names = _drawn(lay)
    assert drives == lines == {1, 2, 3, 4} and {"A1", "B1", "C1", "D1"} <= set(names)
    lay.set_pit_style(2, visible=False)                       # the level and its crosscut go
    drives, lines, names = _drawn(lay)
    assert drives == lines == {1, 4} and "B1" not in names and "C1" not in names and "A1" in names
    lay.set_pit_style(2, visible=True)
    lay.set_pit_style(1, centerline=False, stations=False)    # main pit 1 and all its branches
    drives, lines, names = _drawn(lay)
    assert drives == {1, 2, 3, 4} and lines == {4} and not {"E", "A1", "B1", "C1"} & set(names)
    assert {"F", "D1"} <= set(names)
    scene.clear()


def test_layers_list_shows_the_pit_tree_and_properties_follow():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QCheckBox, QLabel

    from dfpredict.ui.layer_tree import PIT_ROLE, LayerTree
    from dfpredict.ui.properties_panel import PropertiesPanel

    app = QApplication.instance() or QApplication([])  # noqa: F841
    lay = survey_layer()
    tree = LayerTree()
    tree.add_layer(lay)
    top = tree.tree.topLevelItem(0)
    mains = [top.child(i) for i in range(top.childCount())]
    assert [m.text(0) for m in mains] == ["Decline  (P1)", "North  (P2)"]
    level = mains[0].child(0)
    assert level.text(0) == "Level  (P1A)" and level.child(0).text(0) == "XC  (P1A1)"     # nested
    assert level.isExpanded() and top.isExpanded()
    # unticking a pit asks for it to be hidden
    got = []
    tree.pit_visibility_changed.connect(lambda layer, pid, on: got.append((pid, on)))
    level.setCheckState(0, Qt.Unchecked)
    assert got == [(2, False)]
    # selecting a pit shows its properties: a branch has "Inherit parent pit's color", a main
    # pit has centre lines and stations instead
    props = PropertiesPanel(lambda: None)
    styles = []
    props.on_pit_style = lambda layer, pid, **kw: (styles.append((pid, kw)), layer.set_pit_style(pid, **kw))
    tree.pit_selected.connect(props.set_pit)
    tree.tree.setCurrentItem(level)
    boxes = {b.text() for b in props.findChildren(QCheckBox)}
    assert "Inherit parent pit's color" in boxes
    inherit = next(b for b in props.findChildren(QCheckBox) if b.text() == "Inherit parent pit's color")
    assert inherit.isChecked()
    inherit.setChecked(False)                                  # own colour: starts as the inherited one
    assert styles[-1] == (2, {"color": FAMILY_COLORS[0]}) and lay.pit_style(2)["color"] == FAMILY_COLORS[0]
    tree.tree.setCurrentItem(mains[0])
    texts = {w.text() for w in props.findChildren(QLabel)}
    assert "Centre lines" in texts and "Stations" in texts
    assert "Inherit parent pit's color" not in {b.text() for b in props.findChildren(QCheckBox)}
    # a reload of the layer keeps the tree and the selected pit
    again = survey_layer()
    again.apply_settings(lay.settings())
    tree.replace_layer(lay, again)
    assert tree.current_pit() == 1 and tree.tree.currentItem().data(0, Qt.UserRole) is again
    assert tree.tree.currentItem().data(0, PIT_ROLE) == 1


def _reference_drive_mesh(points, width, height, level="back"):
    """The first (point by point) drive_mesh: the vectorised one must give the same mesh."""
    import numpy as np
    P = np.asarray(points, dtype=np.float64)
    k = len(P)
    seg = np.diff(P[:, :2], axis=0)
    lens = np.linalg.norm(seg, axis=1)
    dirs = np.zeros_like(seg)
    good = lens > 1e-9
    dirs[good] = seg[good] / lens[good, None]
    for i in range(len(dirs)):
        if not good[i]:
            dirs[i] = dirs[i - 1] if i > 0 else np.array([1.0, 0.0])
    right = np.zeros((k, 2))
    scale = np.ones(k)
    for i in range(k):
        a = dirs[max(i - 1, 0)]
        b = dirs[min(i, len(dirs) - 1)]
        d = a + b
        n = np.linalg.norm(d)
        d = d / n if n > 1e-9 else b
        right[i] = (d[1], -d[0])
        scale[i] = 1.0 / max(float(np.dot(right[i], (b[1], -b[0]))), 0.35)
    off = {"back": -height, "middle": -height / 2.0, "floor": 0.0}.get(level, -height)
    w2 = (width / 2.0) * scale[:, None]
    r3 = np.hstack([right * w2, np.zeros((k, 1))])
    bot = P + np.array([0.0, 0.0, off])
    top = bot + np.array([0.0, 0.0, height])
    V = np.stack([bot - r3, bot + r3, top + r3, top - r3], axis=1).reshape(-1, 3)
    quads = []
    for i in range(k - 1):
        for c in range(4):
            a, b = i * 4 + c, i * 4 + (c + 1) % 4
            quads.append((a, b, b + 4, a + 4))
    quads.append((3, 2, 1, 0))
    e = (k - 1) * 4
    quads.append((e, e + 1, e + 2, e + 3))
    return V, np.asarray(quads, dtype=np.int64)


def test_drive_mesh_with_array_operations_matches_the_point_by_point_one():
    import numpy as np

    from dfpredict.scene.survey import drive_mesh

    rng = np.random.default_rng(7)
    paths = [np.array([[0, 0, 0], [10, 0, -1.0]]),                                   # one leg
             np.array([[0, 0, 0], [0, 0, -5], [10, 0, -6], [10, 0, -9.0]]),           # vertical first and last
             np.array([[0, 0, 0], [10, 0, 0], [10, 0, -4], [10, 0, -8], [10, 10, -9.0]]),   # vertical middle
             np.array([[0, 0, 0], [10, 0, 0], [0, 0, 0], [0, 10, 0.0]]),              # U-turn
             np.array([[0, 0, 0], [0, 0, -3], [0, 0, -6.0]])]                        # all vertical
    for _ in range(40):
        k = int(rng.integers(2, 60))
        steps = rng.normal(0, 10, (k - 1, 3))
        steps[rng.random(k - 1) < 0.1, :2] = 0.0                                     # some vertical legs
        start = np.array([1000.0, 5000.0, 250.0])
        paths.append(np.vstack([start, start + np.cumsum(steps, axis=0)]))
    for P in paths:
        for level in ("back", "floor", "middle"):
            V, Q = drive_mesh(P, 5.0, 4.0, level)
            V0, Q0 = _reference_drive_mesh(P, 5.0, 4.0, level)
            assert Q.dtype == np.int64 and np.array_equal(Q, Q0)
            assert np.allclose(V, V0, atol=1e-9)


def test_layer_list_every_layer_can_be_dragged_and_empty_database_layers_are_greyed():
    """No fixed places: any layer can be dragged anywhere (never into another layer); the
    database layers with nothing stored yet are greyed out with a hint."""
    import pandas as pd
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from dfpredict.io.drillholes import DrillholeDB
    from dfpredict.scene.layers import DrillholeLayer, Layer, MinedShapesLayer
    from dfpredict.scene.survey import SurveyData
    from dfpredict.ui.layer_tree import LayerTree

    app = QApplication.instance() or QApplication([])  # noqa: F841
    tree = LayerTree()
    orders = []
    tree.order_changed.connect(lambda: orders.append([x.name for x in tree.layers_top_down()]))
    a, b = Layer("ore1", None), Layer("topo1", None)
    pits = SurveyLayer(name="Pits", path=None, data=SurveyData([], {}, {}, ""))              # nothing yet
    db = DrillholeDB(pd.DataFrame({"hole_id": ["H1"], "x": [0.0], "y": [0.0], "z": [0.0], "max_depth": [10.0]}),
                     pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"]), {})
    holes = DrillholeLayer(name="Drillholes (database)", path=None, data=db)
    holes.source = "database"
    mined = MinedShapesLayer(name="Mined shapes", path=None)                                  # no shapes yet
    for lay in (a, pits, b, holes, mined):
        tree.add_layer(lay)
    names = [x.name for x in tree.layers_top_down()]
    assert names == ["Mined shapes", "Drillholes (database)", "topo1", "Pits", "ore1"]       # newest on top
    item = {x.name: tree.tree.topLevelItem(i) for i, x in enumerate(tree.layers_top_down())}
    assert not item["Pits"].flags() & Qt.ItemIsEnabled and not item["Mined shapes"].flags() & Qt.ItemIsEnabled
    assert item["Drillholes (database)"].flags() & Qt.ItemIsEnabled                         # has a hole
    assert "No pits yet" in item["Pits"].toolTip(0)
    assert all(item[n].flags() & Qt.ItemIsDragEnabled for n in item)                         # all movable
    assert not any(item[n].flags() & Qt.ItemIsDropEnabled for n in item)                     # never nested
    # a drag of ore1 to the very top stays there
    t = tree.tree
    t.takeTopLevelItem(t.indexOfTopLevelItem(item["ore1"]))
    t.insertTopLevelItem(0, item["ore1"])
    t.moved.emit()
    assert orders[-1] == ["ore1", "Mined shapes", "Drillholes (database)", "topo1", "Pits"]
    # the first mined shape: no longer greyed out
    mined.shapes = [object()]
    tree.refresh_state(mined)
    assert item["Mined shapes"].flags() & Qt.ItemIsEnabled


def test_dragging_a_layer_never_copies_the_layer_object():
    """Starting a drag must not copy the items' data: the layer kept on each item is attached to
    the view and cannot be pickled (Qt's default drag data pickled it and the app crashed)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QWidget

    from dfpredict.scene.layers import Layer
    from dfpredict.ui.layer_tree import LAYER_MIME, LayerTree

    app = QApplication.instance() or QApplication([])  # noqa: F841
    tree = LayerTree()
    drawn = Layer("ore1", None)
    drawn.scene = QWidget()                              # like a layer in the view: not picklable
    tree.add_layer(drawn)
    tree.add_layer(Layer("topo1", None))
    survey = survey_layer()
    tree.add_layer(survey)
    t = tree.tree
    item = next(t.topLevelItem(i) for i in range(t.topLevelItemCount()) if t.topLevelItem(i).text(0) == "ore1")
    md = t.mimeData([item])                              # what a drag start does
    assert md.formats() == [LAYER_MIME]
    assert t.model().canDropMimeData(md, Qt.MoveAction, 0, 0, t.rootIndex())
    pits = tree._pit_items(next(t.topLevelItem(i) for i in range(t.topLevelItemCount())
                                if tree._layer(t.topLevelItem(i)) is survey))
    assert pits and not any(p.flags() & (Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled) for p in pits)
