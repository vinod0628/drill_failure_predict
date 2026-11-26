"""Block model management: the encrypted .gvbm file, block models inside a solid, Surpac ASCII import,
attributes - and the manager in the main window. (Grade estimation is GeoViewer's, not part of this
application.)"""

from __future__ import annotations

import os
import shutil
import time

import numpy as np
import pytest

from dfpredict.blockmodel import attributes as battr
from dfpredict.blockmodel import store
from dfpredict.blockmodel.build import blocks_in_solid, mesh_volume, read_surpac_ascii
from dfpredict.io.block_model import read_block_csv


@pytest.fixture(scope="module")
def ore(demo):
    from dfpredict.io import read_dtm
    from dfpredict.scene.solid import closed_mesh

    return closed_mesh(read_dtm(demo / "ore1.dtm"))




# --- the file ---------------------------------------------------------------------------------
def test_gvbm_round_trip_is_exact_and_unreadable(demo, tmp_path):
    bm = read_block_csv(demo / "block_model.csv", use_cache=False)
    bm.info = {"name": "demo", "descriptions": {"gold": "Au g/t"}}
    p = store.save_block_model(bm, tmp_path / "demo.gvbm")
    raw = p.read_bytes()
    assert raw[:4] == store.MAGIC and b"gold" not in raw and b"classification" not in raw
    assert len(raw) < (demo / "block_model.csv").stat().st_size / 5          # compressed
    back = store.read_block_model(p)
    np.testing.assert_array_equal(back.centroids, bm.centroids)
    np.testing.assert_array_equal(back.sizes, bm.sizes)
    for n in bm.attributes:
        np.testing.assert_array_equal(back.attributes[n], bm.attributes[n])
    assert back.info["descriptions"]["gold"] == "Au g/t" and back.info["name"] == "demo"
    assert not list(tmp_path.glob("*.tmp"))


def test_gvbm_rejects_changed_or_cut_files(demo, tmp_path):
    bm = read_block_csv(demo / "block_model.csv", use_cache=False)
    p = store.save_block_model(bm, tmp_path / "a.gvbm")
    raw = bytearray(p.read_bytes())
    for pos in (10, 60, len(raw) // 2, len(raw) - 3):       # header, first piece, middle, tag
        bad = bytearray(raw)
        bad[pos] ^= 0x01
        q = tmp_path / f"bad{pos}.gvbm"
        q.write_bytes(bytes(bad))
        with pytest.raises(store.BlockFileError):
            store.read_block_model(q)
    q = tmp_path / "cut.gvbm"
    q.write_bytes(bytes(raw[:-40]))
    with pytest.raises(store.BlockFileError):
        store.read_block_model(q)
    q = tmp_path / "text.gvbm"
    q.write_text("X,Y,Z\n")
    with pytest.raises(store.BlockFileError):
        store.read_block_model(q)


def test_large_model_spans_several_pieces(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CHUNK", 1 << 12)              # many small pieces
    n = 20000
    rng = np.random.default_rng(3)
    ijk = rng.integers(0, 200, (n, 3))
    from dfpredict.io.block_model import BlockModel

    bm = BlockModel(None, ijk * 5.0 + 2.5, np.full((n, 3), 5.0, dtype=np.float32),
                    {"au": rng.random(n).astype(np.float32)})
    p = store.save_block_model(bm, tmp_path / "big.gvbm")
    back = store.read_block_model(p)
    np.testing.assert_array_equal(back.attributes["au"], bm.attributes["au"])
    np.testing.assert_allclose(back.centroids, bm.centroids)


# --- building -----------------------------------------------------------------------------------
@pytest.mark.parametrize("sub", [1, 2, 4])
def test_blocks_inside_solid(ore, sub):
    from dfpredict.scene.solid import points_inside

    bm = blocks_in_solid(*ore, (10, 10, 5), sub=sub, attributes=[("gold", None, "Au"), ("sg", 2.7, "")])
    vol = float(np.prod(bm.sizes.astype(np.float64), axis=1).sum())
    solid = mesh_volume(*ore)
    assert abs(vol - solid) / solid < (0.02 if sub == 1 else 0.01)
    if sub == 1:
        assert points_inside(bm.centroids, *ore).all()
    assert (bm.attributes["sg"] == np.float32(2.7)).all() and (bm.attributes["gold"] == -99).all()
    g = bm.grid()                                    # aligned; sub-blocks never overlap
    occ = np.zeros(g.shape, dtype=np.uint8)
    for (i, j, k), (a, b, c) in zip(g.ijk_min, g.span):
        occ[i:i + a, j:j + b, k:k + c] += 1
    assert occ.max() == 1
    assert len(bm.block_sizes()) == (1 if sub == 1 else 2)


def test_origin_is_kept_on_whole_blocks(ore):
    bm = blocks_in_solid(*ore, (10, 10, 10), origin=(3.0, 7.0, 1.0))
    lo = bm.centroids - bm.sizes / 2
    np.testing.assert_allclose(np.mod(lo - [3.0, 7.0, 1.0], 10.0) % 10.0, 0, atol=1e-6)


def test_surpac_ascii_import_with_header_lines_and_spaces(tmp_path, demo):
    p = tmp_path / "export.txt"
    p.write_text("Block model export from Surpac\nOrigin 0 0 0\n\n"
                 "X Y Z size(X) size(Y) size(Z) au rock\n"
                 "5 5 5 10 10 10 1.5 OX\n15 5 5 10 10 10 -99 FR\n", encoding="latin-1")
    bm = read_surpac_ascii(p)
    assert bm.n_blocks == 2 and bm.attributes["au"][0] == np.float32(1.5)
    assert bm.is_categorical("rock")
    csv = read_surpac_ascii(demo / "block_model.csv")
    assert csv.n_blocks == 60635 and set(csv.attribute_names) == {"classification_no", "gold", "sg"}
    assert csv.info["history"]


def test_attribute_operations(ore):
    bm = blocks_in_solid(*ore, (20, 20, 10), attributes=[("gold", None, "")])
    battr.add_attribute(bm, "cu", 0.5, "copper %")
    with pytest.raises(ValueError):
        battr.add_attribute(bm, "CU")                      # same name, other case
    with pytest.raises(ValueError):
        battr.add_attribute(bm, "status")                  # the reserve's attribute
    with pytest.raises(ValueError):
        battr.add_attribute(bm, "2nd")
    battr.rename_attribute(bm, "cu", "copper")
    assert "cu" not in bm.attributes and battr.descriptions(bm)["copper"] == "copper %"
    m = np.zeros(bm.n_blocks, dtype=bool)
    m[:10] = True
    assert battr.fill_attribute(bm, "copper", 2.0, m) == 10
    assert battr.attribute_stats(bm, "copper")["max"] == 2.0
    battr.fill_attribute(bm, "copper", None)
    assert battr.attribute_stats(bm, "copper")["n"] == 0
    battr.remove_attribute(bm, "copper")
    assert list(bm.attributes) == ["gold"]
    assert len(bm.info["history"]) >= 4


# --- samples ----------------------------------------------------------------------------------






# --- search, variogram, estimation ---------------------------------------------------------------












# --- the manager in the main window ----------------------------------------------------------
def _wait(app, cond, timeout=60):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def test_block_model_manager_workflow(demo, tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    import dfpredict.ui.main_window as mw
    from dfpredict.scene.layers import BlockModelLayer

    app = QApplication.instance() or QApplication([])
    (tmp_path / "s.ini").write_text("[database]\nbackend=sqlite\n[workspace]\nrestore_last=false\n",
                                    encoding="utf-8")
    monkeypatch.setattr(mw, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[1:]}"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    ws = tmp_path / "ws"
    ws.mkdir()
    for f in ("ore1.dtm", "ore1.str", "geodatabase.sqlite"):
        shutil.copy2(demo / f, ws / f)
    out = ws / "ore1_bm.gvbm"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win = mw.MainWindow()
    win.quiet_errors = True
    try:
        win.open_workspace(ws, confirm=False)
        assert _wait(app, lambda: not win.loads_pending and not win._threads and win.workspace is not None
                     and any(lay.name == "ore1" for lay in win.scene.layers))
        win.open_block_manager()
        mgr = win.block_manager
        assert mgr is not None and mgr.bm is None
        from dfpredict.ui.block_manager import NewFromSolidDialog

        dlg = NewFromSolidDialog(win.scene, ws, mgr)
        assert dlg.solid.currentText() == "ore1"
        dlg.sx.setValue(20)
        dlg.sy.setValue(20)
        dlg.sz.setValue(10)
        dlg._create()
        assert dlg.result_model is not None and out.exists()
        mgr._adopt(dlg.result_model, dlg.out_path)
        assert mgr.bm.n_blocks > 100 and {"gold", "sg"} <= set(mgr.bm.attributes)

        # saved: the layer shows the model; the attribute table lists its attributes (read only)
        mgr._changed("gold")
        assert mgr.save()
        assert _wait(app, lambda: any(isinstance(lay, BlockModelLayer) and lay.path == out
                                      and "gold" in lay.data.attributes for lay in win.scene.layers), 30)
        assert mgr.table.rowCount() == len(mgr.bm.attributes)
        assert not hasattr(mgr, "b_est") and not hasattr(mgr, "b_add")      # no editing / estimation here
        back = store.read_block_model(out)
        assert back.n_blocks == mgr.bm.n_blocks

        # Close (button / X / Esc): asks about unsaved changes; Cancel keeps the manager open
        assert mgr.isVisible()
        mgr.dirty = True
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)
        mgr.close()
        assert mgr.isVisible()
        mgr.reject()
        assert mgr.isVisible()
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
        mgr.close()
        assert not mgr.isVisible()
        win.open_block_manager()                        # opens again
        assert win.block_manager.isVisible()
        mgr.dirty = False
        win.block_manager.reject()                      # Esc, nothing unsaved: closes at once
        assert not win.block_manager.isVisible()
    finally:
        if getattr(win, "block_manager", None) is not None:
            win.block_manager.dirty = False
        win.close()
        win.deleteLater()
        app.processEvents()



def test_opened_block_csv_is_imported_in_the_manager(demo, tmp_path, monkeypatch):
    """Opening a block model CSV imports it into a .gvbm in the Block Model Manager (never shown
    as CSV), then offers to delete the CSV; the .gvbm shows as the layer."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QFile, QSettings
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    import dfpredict.ui.main_window as mw
    from dfpredict.scene.layers import BlockModelLayer
    from dfpredict.ui.block_manager import ImportAsciiDialog

    app = QApplication.instance() or QApplication([])
    (tmp_path / "s.ini").write_text("[database]\nbackend=sqlite\n[workspace]\nrestore_last=false\n",
                                    encoding="utf-8")
    monkeypatch.setattr(mw, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[1:]}"))
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[1]) or QMessageBox.Yes)
    monkeypatch.setattr(QFile, "moveToTrash", lambda *a: (False, ""))   # not into the user's Recycle Bin
    ws = tmp_path / "ws"
    ws.mkdir()
    csv = ws / "block_model.csv"
    shutil.copy2(demo / "block_model.csv", csv)
    out = ws / "block_model.gvbm"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    offered = []

    def run_import(dlg):
        offered.append(dlg.source)
        dlg._import()
        return dlg.result()

    monkeypatch.setattr(ImportAsciiDialog, "exec", run_import)
    win = mw.MainWindow()
    try:
        win.open_paths([str(csv)])                   # no workspace yet: the CSV's folder becomes one
        assert _wait(app, lambda: not win._import_queue and not win._importing and out.exists(), 60)
        assert offered == [csv]
        assert win.block_manager is not None and win.block_manager.path == out
        assert asked == ["Remove the original file?"] and not csv.exists()     # deleted when confirmed
        assert _wait(app, lambda: any(isinstance(lay, BlockModelLayer) and lay.path == out
                                      for lay in win.scene.layers), 30)
        assert not any(isinstance(lay, BlockModelLayer) and str(lay.path).endswith(".csv")
                       for lay in win.scene.layers)
        assert store.read_block_model(out).n_blocks == read_surpac_ascii(demo / "block_model.csv").n_blocks
    finally:
        if getattr(win, "block_manager", None) is not None:
            win.block_manager.dirty = False
        win.close()
        win.deleteLater()
        app.processEvents()

# --- worker processes (the window keeps responding) -----------------------------------------




def test_points_inside_in_workers(ore, monkeypatch):
    from dfpredict.blockmodel import parallel
    from dfpredict.scene.solid import points_inside

    monkeypatch.setattr(parallel, "SMALL_INSIDE", 0)
    pts = np.random.default_rng(0).uniform([1400, 7100, -20], [2000, 7650, 260], (50000, 3))
    try:
        np.testing.assert_array_equal(parallel.points_inside(pts, *ore), points_inside(pts, *ore))
    finally:
        parallel.shutdown()


def test_progress_dialog_is_only_touched_from_the_gui_thread(monkeypatch):
    """The job reports from its own thread; the dialog must be updated in the GUI thread only
    (touching widgets from another thread crashes or hangs the window)."""
    import threading

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QProgressDialog

    from dfpredict.ui.block_manager import run_with_progress

    app = QApplication.instance() or QApplication([])
    main = threading.get_ident()
    threads = set()
    for name in ("setValue", "setLabelText", "setRange"):
        orig = getattr(QProgressDialog, name)
        monkeypatch.setattr(QProgressDialog, name,
                            lambda self, *a, _o=orig: (threads.add(threading.get_ident()), _o(self, *a))[1])

    def work(progress, _cancel):
        assert threading.get_ident() != main                    # the job runs in its own thread
        for i in range(20):
            progress(i / 20, f"step {i}")
            time.sleep(0.02)
        progress(None, "busy")
        time.sleep(0.15)
        return 42

    assert run_with_progress(None, "test", work) == 42
    assert threads == {main}
    app.processEvents()
