"""Workspace save / open, folder scanning, change detection and reload."""

import json
import os
import shutil
import time

import pytest

from dfpredict.workspace import (
    WORKSPACE_FILE,
    Workspace,
    changed_layers,
    list_data_files,
    loadable_groups,
    reload_layers,
)

FILES = ["pit1.dtm", "pit1.str", "gold_dtm.ssi", "layout1.str", "ore1_bm.gvbm"]   # no Surpac .accdb in the tests


@pytest.fixture()
def ws_dir(demo, tmp_path):
    for f in FILES:
        shutil.copy2(demo / f, tmp_path / f)
    old = time.time() - 60  # well past the "still being written" window
    for f in FILES:
        os.utime(tmp_path / f, (old, old))
    return tmp_path


def _age(path, seconds=60):
    t = time.time() - seconds
    os.utime(path, (t, t))


def _names(layers):
    return sorted(lay.name for lay in layers)


def test_scan_skips_paired_and_duplicate_files(ws_dir):
    files = list_data_files(ws_dir)
    groups = loadable_groups(files, files)
    chosen = sorted(p.name for g in groups for p in g)
    # pit1.str belongs to pit1.dtm
    assert chosen == ["layout1.str", "ore1_bm.gvbm", "pit1.dtm"]


def test_open_folder_without_workspace_file(ws_dir):
    layers, errors, view = Workspace(ws_dir).load()
    assert not errors and view is None
    # the project database's drillhole layer is always listed (made empty by the database set-up)
    assert _names(layers) == ["Drillholes (database)", "Survey (database)", "layout1", "ore1_bm", "pit1"]
    pit = next(lay for lay in layers if lay.name == "pit1")
    assert {p.name for p in pit.sources} == {"pit1.dtm", "pit1.str", "gold_dtm.ssi"}


def test_block_model_csv_is_imported_not_loaded(ws_dir, demo):
    """A block model CSV never becomes a layer: the workspace lists it for the import (.gvbm),
    from the folder scan, from Refresh and from a CSV layer of an older workspace file."""
    shutil.copy2(demo / "block_model.csv", ws_dir / "block_model.csv")
    _age(ws_dir / "block_model.csv")
    ws = Workspace(ws_dir)
    layers, errors, _ = ws.load()
    assert not errors
    assert "block_model" not in _names(layers)
    assert [p.name for p in ws.block_csvs] == ["block_model.csv"]
    ws.save(layers, None)

    # an older workspace file with the CSV as a layer: not loaded, listed for the import
    data = json.loads((ws_dir / WORKSPACE_FILE).read_text(encoding="utf-8"))
    data["layers"].append({"name": "block_model", "type": "BlockModelLayer", "part": "",
                           "files": ["block_model.csv"], "settings": {}})
    (ws_dir / WORKSPACE_FILE).write_text(json.dumps(data), encoding="utf-8")
    ws2 = Workspace(ws_dir)
    layers2, errors, _ = ws2.load()
    assert not errors and "block_model" not in _names(layers2)
    assert [p.name for p in ws2.block_csvs] == ["block_model.csv"]

    # a CSV that appears later: Refresh lists it; with its .gvbm next to it, it was imported already
    ws2.block_csvs = []
    shutil.copy2(demo / "block_model.csv", ws_dir / "more.csv")
    _age(ws_dir / "more.csv")
    (ws_dir / "other.csv").write_bytes((demo / "block_model.csv").read_bytes())
    shutil.copy2(demo / "ore1_bm.gvbm", ws_dir / "other.gvbm")
    _age(ws_dir / "other.csv")
    _age(ws_dir / "other.gvbm")
    groups, _waiting = ws2.find_new_files()
    assert [p.name for p in ws2.block_csvs] == ["more.csv"]
    assert sorted(p.name for g in groups for p in g) == ["other.gvbm"]


def test_save_and_reopen_keeps_settings_and_view(ws_dir):
    ws = Workspace(ws_dir)
    layers, _, _ = ws.load()
    bm = next(lay for lay in layers if lay.name == "ore1_bm")
    bm.opacity, bm.cmap, bm.value_min, bm.visible = 0.4, "viridis", 2.5, False
    pit = next(lay for lay in layers if lay.name == "pit1")
    pit.color, pit.use_styles = (0.1, 0.2, 0.3), False
    view = {"mode": "xz", "grid": True, "focal_point": [1, 2, 3], "position": [1, -50, 3],
            "view_up": [0, 0, 1], "parallel_scale": 123.0}
    ws.save(layers, view)

    data = json.loads((ws_dir / WORKSPACE_FILE).read_text(encoding="utf-8"))
    assert all(not os.path.isabs(f) for spec in data["layers"] for f in spec["files"])  # relative paths

    layers2, errors, view2 = Workspace(ws_dir).load()
    assert not errors
    assert [lay.name for lay in layers2] == [lay.name for lay in layers]  # same order
    bm2 = next(lay for lay in layers2 if lay.name == "ore1_bm")
    assert (bm2.opacity, bm2.cmap, bm2.value_min, bm2.visible) == (0.4, "viridis", 2.5, False)
    pit2 = next(lay for lay in layers2 if lay.name == "pit1")
    assert pit2.color == (0.1, 0.2, 0.3) and pit2.use_styles is False
    assert view2 == view


def test_workspace_survives_being_moved(ws_dir, tmp_path_factory):
    ws = Workspace(ws_dir)
    layers, _, _ = ws.load()
    ws.save(layers, None)
    moved = tmp_path_factory.mktemp("moved") / "ws"
    shutil.copytree(ws_dir, moved)
    layers2, errors, _ = Workspace(moved).load()
    assert not errors and len(layers2) == len(layers)
    assert all(str(moved) in str(lay.inputs[0]) for lay in layers2 if lay.source != "database")


def test_missing_file_is_reported_others_load(ws_dir):
    ws = Workspace(ws_dir)
    layers, _, _ = ws.load()
    ws.save(layers, None)
    (ws_dir / "layout1.str").unlink()
    layers2, errors, _ = Workspace(ws_dir).load()
    assert len(layers2) == len(layers) - 1
    assert any("layout1" in e and "not found" in e for e in errors)


def test_only_changed_layers_reload_with_their_settings(ws_dir):
    layers, _, _ = Workspace(ws_dir).load()
    assert changed_layers(layers).changed == []
    pit = next(lay for lay in layers if lay.name == "pit1")
    pit.opacity = 0.3
    # The paired .str of the DTM changes -> only the DTM layer is reloaded.
    with open(ws_dir / "pit1.str", "a", encoding="latin-1") as f:
        f.write("\n")
    _age(ws_dir / "pit1.str", 30)
    rep = changed_layers(layers)
    assert [lay.name for lay in rep.changed] == ["pit1"]
    pairs, errors = reload_layers(rep.changed)
    assert not errors and len(pairs) == 1
    old, new = pairs[0]
    assert old is pit and new is not pit and new.opacity == 0.3
    assert changed_layers([new]).changed == []


def test_file_being_written_waits(ws_dir):
    layers, _, _ = Workspace(ws_dir).load()
    with open(ws_dir / "layout1.str", "a", encoding="latin-1") as f:
        f.write("\n")  # modified just now
    rep = changed_layers(layers)
    assert rep.changed == [] and [lay.name for lay in rep.waiting] == ["layout1"]


def test_failed_reload_not_retried_until_file_changes_again(ws_dir):
    layers, _, _ = Workspace(ws_dir).load()
    lay = next(x for x in layers if x.name == "layout1")
    (ws_dir / "layout1.str").write_text("garbage\n", encoding="latin-1")
    _age(ws_dir / "layout1.str", 30)
    pairs, errors = reload_layers(changed_layers([lay]).changed)
    assert pairs == [] and errors
    assert changed_layers([lay]).changed == []          # same broken file: not retried
    shutil.copy2(ws_dir / "pit1.str", ws_dir / "layout1.str")
    _age(ws_dir / "layout1.str", 20)
    assert changed_layers([lay]).changed == [lay]       # fixed file: reloaded


def test_new_files_found_once(ws_dir, demo):
    ws = Workspace(ws_dir)
    layers, _, _ = ws.load()
    ws.save(layers, None)
    assert ws.find_new_files() == ([], 0)
    sub = ws_dir / "new"
    sub.mkdir()
    shutil.copy(demo / "rings12.str", sub / "rings12.str")  # new modification time
    groups, waiting = ws.find_new_files()
    assert groups == [] and waiting == 1                # just written: next time
    _age(sub / "rings12.str")
    groups, waiting = ws.find_new_files()
    assert [p.name for g in groups for p in g] == ["rings12.str"] and waiting == 0
    assert ws.find_new_files() == ([], 0)               # reported only once


def test_files_added_while_closed_load_on_open(ws_dir, demo):
    ws = Workspace(ws_dir)
    layers, _, _ = ws.load()
    ws.save(layers, None)
    shutil.copy2(demo / "rings12.str", ws_dir / "rings12.str")
    _age(ws_dir / "rings12.str")
    layers2, _, _ = Workspace(ws_dir).load()
    assert "rings12" in _names(layers2)


def test_removed_layer_does_not_come_back(ws_dir):
    ws = Workspace(ws_dir)
    layers, _, _ = ws.load()
    layers = [lay for lay in layers if lay.name != "layout1"]  # removed by the user
    ws.save(layers, None)
    layers2, _, _ = Workspace(ws_dir).load()
    assert "layout1" not in _names(layers2)


def test_fingerprint_ignores_camera(ws_dir):
    layers, _, _ = Workspace(ws_dir).load()
    a = Workspace.fingerprint(layers, {"mode": "3d", "grid": False, "position": [0, 0, 1]})
    b = Workspace.fingerprint(layers, {"mode": "3d", "grid": False, "position": [5, 5, 5]})
    assert a == b
    layers[0].opacity = 0.5
    assert Workspace.fingerprint(layers, {"mode": "3d", "grid": False}) != a


def test_fingerprint_with_camera_sees_camera_moves(ws_dir):
    layers, _, _ = Workspace(ws_dir).load()
    v = {"mode": "3d", "grid": False, "position": [1000.0, 2000.0, 300.0]}
    a = Workspace.fingerprint(layers, v, camera=True)
    assert Workspace.fingerprint(layers, {**v, "position": [1000.0, 2000.0, 300.0 + 1e-9]}, camera=True) == a
    assert Workspace.fingerprint(layers, {**v, "position": [1001.0, 2000.0, 300.0]}, camera=True) != a


def test_layer_with_missing_file_is_kept_and_restored(ws_dir, tmp_path_factory):
    ws = Workspace(ws_dir)
    layers, _, _ = ws.load()
    lay = next(x for x in layers if x.name == "layout1")
    lay.opacity = 0.25
    ws.save(layers, None)
    keep = tmp_path_factory.mktemp("keep") / "layout1.str"
    shutil.move(str(ws_dir / "layout1.str"), keep)     # e.g. a drive not connected

    ws2 = Workspace(ws_dir)
    layers2, errors, _ = ws2.load()
    assert "layout1" not in _names(layers2) and errors
    assert [s["name"] for s in ws2.pending_specs] == ["layout1"]
    ws2.save(layers2, None)                             # auto-save must not drop it
    data = json.loads((ws_dir / WORKSPACE_FILE).read_text(encoding="utf-8"))
    assert "layout1" in [s["name"] for s in data["layers"]]
    assert ws2.load_pending() == []                     # still missing

    shutil.move(str(keep), ws_dir / "layout1.str")
    _age(ws_dir / "layout1.str")
    restored = ws2.load_pending()
    assert [lay.name for _s, lay in restored] == ["layout1"] and restored[0][1].opacity == 0.25


def test_unreadable_workspace_file_is_backed_up(ws_dir):
    (ws_dir / WORKSPACE_FILE).write_text("{ not json", encoding="utf-8")
    ws = Workspace(ws_dir)
    layers, errors, _ = ws.load()
    assert layers and errors                            # folder contents instead
    ws.save(layers, None)
    assert (ws_dir / (WORKSPACE_FILE[:-5] + ".json.bak")).read_text(encoding="utf-8") == "{ not json"
    assert Workspace(ws_dir).read()["format"]


def test_app_made_layers_keep_their_last_state(tmp_path):
    """"Mined shapes" is made by the Reserve tools at each start (not a file layer): it comes
    back hidden if it was hidden, with its opacity / colour / name; a session without the
    layer keeps what was saved; hiding it is a change that the autosave writes."""
    from dfpredict.scene.layers import MinedShapesLayer
    from dfpredict.workspace import Workspace, restore_app_layer

    ws = Workspace(tmp_path)
    ws.load()
    mined = MinedShapesLayer(name="Mined shapes", path=None, color=(1.0, 0.6, 0.2))
    before = Workspace.fingerprint([mined], None)
    mined.visible, mined.opacity, mined.name = False, 0.3, "Stopes"
    assert Workspace.fingerprint([mined], None) != before           # hiding it is a change to save
    ws.save([mined], None)

    again = Workspace(tmp_path)
    again.load()
    fresh = MinedShapesLayer(name="Mined shapes", path=None, color=(1.0, 0.6, 0.2))
    restore_app_layer(again, fresh)
    assert fresh.visible is False and fresh.opacity == pytest.approx(0.3) and fresh.name == "Stopes"
    assert fresh.color == pytest.approx((1.0, 0.6, 0.2))

    again.save([], None)                                             # a session without the layer
    third = Workspace(tmp_path)
    third.load()
    other = MinedShapesLayer(name="Mined shapes", path=None)
    restore_app_layer(third, other)
    assert other.visible is False and other.name == "Stopes"         # still remembered
    restore_app_layer(None, other)                                   # no workspace: nothing happens


def test_workspace_is_saved_when_the_app_closes_not_after_each_change(demo, tmp_path, monkeypatch):
    """Changes (e.g. a layer hidden) are not written right away; closing the app saves them."""
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QMessageBox

    import dfpredict.ui.main_window as mw
    from dfpredict.workspace import WORKSPACE_FILE

    app = QApplication.instance() or QApplication([])
    (tmp_path / "s.ini").write_text("[database]\nbackend=sqlite\n[workspace]\nrestore_last=false\n",
                                    encoding="utf-8")       # SQLite: no MySQL server is tried
    monkeypatch.setattr(mw, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    for f in ("pit1.str", "pit1.dtm"):
        shutil.copy2(demo / f, ws_dir / f)
    win = mw.MainWindow()
    try:
        win.open_workspace(ws_dir, confirm=False)
        t0 = time.time()
        while (win.loads_pending or win._threads or win.workspace is None or win.workspace.saved_state is None) \
                and time.time() - t0 < 60:
            app.processEvents()
            time.sleep(0.02)
        for _ in range(20):
            app.processEvents()
        f = ws_dir / WORKSPACE_FILE
        before = f.read_text(encoding="utf-8") if f.exists() else None          # nothing changed yet
        lay = next(x for x in win.scene.layers if x.name == "pit1")
        win._set_layer_visible(lay, False)              # a change ...
        for _ in range(3):
            app.processEvents()
            time.sleep(0.3)
        assert (f.read_text(encoding="utf-8") if f.exists() else None) == before   # ... not written yet
    finally:
        win.close()                                     # closing the app saves
    saved = json.loads((ws_dir / WORKSPACE_FILE).read_text(encoding="utf-8"))
    spec = next(s for s in saved["layers"] if s["name"] == lay.name)
    assert spec["settings"]["visible"] is False
