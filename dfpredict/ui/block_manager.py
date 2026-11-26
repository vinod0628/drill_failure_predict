"""Block Model Manager: the block models of the workspace (.gvbm, encrypted binary files).

* New from solid: blocks inside an ore solid (block size X / Y / Z, optional sub-blocking at
  the boundary) with float attributes and their default values.
* Import Surpac ASCII: a Surpac block model text / CSV export becomes a .gvbm file.
* Attributes: listed with their statistics (read only here - attribute editing and grade estimation
  are GeoViewer's).

Changes are kept in memory until Save; then the block model layer in the view is read again.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..blockmodel import BLOCK_MODEL_EXT
from ..blockmodel import attributes as battr
from ..blockmodel.build import (attribute_name_ok, blocks_in_solid, csv_preview, describe, grid_for_solid,
                                mesh_volume, read_surpac_ascii)
from ..blockmodel.store import BlockFileError, add_history, read_block_model, save_block_model
from ..i18n import tr
from ..scene.layers import DEPLETION_ATTRS, fmt_num
from . import theme
from .icons import icon

log = logging.getLogger(__name__)


# --- background work with a progress dialog --------------------------------------------------
class _Job(QObject):
    progress = Signal(float, str)
    done = Signal(object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancelled = False
        self._last = 0.0

    def _report(self, f, text="") -> None:
        import time

        now = time.monotonic()
        if now - self._last < 0.1 and f is not None and 0 < f < 1:
            return                                     # at most ~10 updates a second
        self._last = now
        self.progress.emit(-1.0 if f is None else float(f), str(text))

    @Slot()
    def run(self):
        from ..blockmodel.parallel import Cancelled

        try:
            res = self.fn(self._report, lambda: self.cancelled)
        except Cancelled as e:
            res = e
        except Exception as e:  # noqa: BLE001 - shown to the user
            log.warning("block model job failed: %s", traceback.format_exc())
            res = e
        self.done.emit(res)


class _Relay(QObject):
    """Receives a job's signals in the thread that made it (the GUI thread)."""

    def __init__(self, on_progress, on_done):
        super().__init__()
        self._p, self._d = on_progress, on_done

    @Slot(float, str)
    def on_progress(self, f, text):
        self._p(f, text)

    @Slot(object)
    def on_done(self, res):
        self._d(res)


def run_with_progress(parent, title: str, fn, cancellable: bool = True):
    """Run ``fn(progress, cancelled)`` in a thread while a progress dialog shows; returns its
    result, or the exception it raised (``parallel.Cancelled`` when cancelled).

    ``progress(fraction, text)``: fraction 0..1, or None while the length is unknown (a busy
    bar). The heavy work itself goes to worker processes (blockmodel.parallel), so this thread
    mostly waits and the window keeps responding."""
    from PySide6.QtCore import QEventLoop

    dlg = QProgressDialog(title, tr("Cancel") if cancellable else "", 0, 1000, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.setMinimumWidth(460)
    if not cancellable:
        dlg.setCancelButton(None)
    job = _Job(fn)
    thread = QThread()
    job.moveToThread(thread)
    out = {}
    loop = QEventLoop()

    def finished(res):
        out["r"] = res
        loop.quit()

    def prog(f, text):
        if job.cancelled:
            return
        if f < 0:
            if dlg.maximum() != 0:
                dlg.setRange(0, 0)                   # busy bar
        else:
            if dlg.maximum() == 0:
                dlg.setRange(0, 1000)
            dlg.setValue(int(max(0.0, min(1.0, f)) * 1000))
        if text:
            dlg.setLabelText(text)

    def cancel():
        job.cancelled = True
        dlg.show()                                   # QProgressDialog hides itself on Cancel
        dlg.setLabelText(tr("Cancelling ..."))
        dlg.setRange(0, 0)

    # The job lives in the worker thread. Plain functions connected to its signals would run
    # in that thread too and touch the dialog from there (crash / hang): a relay object of
    # this (GUI) thread receives them instead, so they are queued to the GUI thread.
    relay = _Relay(prog, finished)
    job.progress.connect(relay.on_progress)
    job.done.connect(relay.on_done)
    dlg.canceled.connect(cancel)
    thread.started.connect(job.run)
    dlg.setValue(0)
    dlg.show()
    thread.start()
    QApplication.setOverrideCursor(Qt.BusyCursor)
    try:
        loop.exec()
    finally:
        QApplication.restoreOverrideCursor()
        thread.quit()
        thread.wait()
        dlg.close()
        dlg.deleteLater()
    return out.get("r")


def _num_edit(value: float, lo: float = -1e9, hi: float = 1e9, decimals: int = 3, step: float = 1.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    s.setMinimumWidth(90)
    return s


def solid_layers(scene) -> list:
    """The DTM layers of the scene that are solids (closed), by name."""
    from ..scene.layers import SurfaceLayer
    from ..scene.solid import closed_mesh

    out = []
    for lay in scene.layers if scene is not None else []:
        if isinstance(lay, SurfaceLayer) and lay.data is not None:
            try:
                if closed_mesh(lay.data) is not None:
                    out.append(lay)
            except Exception:  # noqa: BLE001
                continue
    return out


def solid_mesh(layer):
    from ..scene.solid import closed_mesh

    return closed_mesh(layer.data)


def _muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color: {theme.MUTED};")
    return lab


# --- new block model inside a solid -----------------------------------------------------------
class NewFromSolidDialog(QDialog):
    def __init__(self, scene, folder: Path | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("New block model from solid"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.folder = folder
        self.solids = solid_layers(scene)
        self.extra_mesh = None          # a .dtm chosen with Browse
        self.result_model = None
        self.out_path: Path | None = None

        self.solid = QComboBox()
        for lay in self.solids:
            self.solid.addItem(lay.name, lay)
        browse = QPushButton(icon("folder-open-outline"), tr("Browse ..."))
        browse.clicked.connect(self._browse)
        srow = QHBoxLayout()
        srow.addWidget(self.solid, 1)
        srow.addWidget(browse)

        self.sx, self.sy, self.sz = (_num_edit(v, 0.01, 10000, 3) for v in (10.0, 10.0, 5.0))
        size_row = QHBoxLayout()
        for lab, w in (("X", self.sx), ("Y", self.sy), ("Z", self.sz)):
            size_row.addWidget(QLabel(lab))
            size_row.addWidget(w)
        self.auto_origin = QCheckBox(tr("Automatic (whole blocks below the solid)"))
        self.auto_origin.setChecked(True)
        self.ox, self.oy, self.oz = (_num_edit(0.0, -1e8, 1e8, 3) for _ in range(3))
        orow = QHBoxLayout()
        for lab, w in (("X", self.ox), ("Y", self.oy), ("Z", self.oz)):
            orow.addWidget(QLabel(lab))
            orow.addWidget(w)
            w.setEnabled(False)
        self.auto_origin.toggled.connect(lambda on: [w.setEnabled(not on) for w in (self.ox, self.oy, self.oz)])
        self.sub = QComboBox()
        for n, text in ((1, tr("None")), (2, tr("2 x 2 x 2 at the boundary")), (4, tr("4 x 4 x 4 at the boundary")),
                        (8, tr("8 x 8 x 8 at the boundary"))):
            self.sub.addItem(text, n)
        self.name = QLineEdit()
        self.null = _num_edit(-99.0, -1e9, 1e9, 3)

        self.attrs = QTableWidget(0, 3)
        self.attrs.setHorizontalHeaderLabels([tr("Attribute"), tr("Default value"), tr("Description")])
        self.attrs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.attrs.verticalHeader().setVisible(False)
        self.attrs.setMinimumHeight(140)
        for nm, dv, ds in (("gold", "", tr("grade, g/t")), ("sg", "2.7", tr("specific gravity, t/m³"))):
            self._add_attr_row(nm, dv, ds)
        add = QPushButton(icon("table-row-plus-after"), tr("Add attribute"))
        add.clicked.connect(lambda: self._add_attr_row("", "", ""))
        rem = QPushButton(icon("table-row-remove"), tr("Remove attribute"))
        rem.clicked.connect(lambda: self.attrs.removeRow(self.attrs.currentRow()) if self.attrs.currentRow() >= 0 else None)
        arow = QHBoxLayout()
        arow.addWidget(add)
        arow.addWidget(rem)
        arow.addStretch(1)

        self.preview = _muted("")
        form = QFormLayout()
        form.addRow(tr("Ore solid"), srow)
        form.addRow(tr("Block size (m)"), size_row)
        form.addRow(tr("Origin"), self.auto_origin)
        form.addRow("", orow)
        form.addRow(tr("Sub-blocking"), self.sub)
        form.addRow(tr("Null value"), self.null)
        form.addRow(tr("Name"), self.name)
        box = QGroupBox(tr("Attributes (numbers); an empty default is the null value"))
        bl = QVBoxLayout(box)
        bl.addWidget(self.attrs)
        bl.addLayout(arow)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("Create"))
        bb.accepted.connect(self._create)
        bb.rejected.connect(self.reject)
        v = QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(box, 1)
        v.addWidget(self.preview)
        v.addWidget(bb)
        for w in (self.sx, self.sy, self.sz, self.ox, self.oy, self.oz):
            w.valueChanged.connect(self._update_preview)
        self.auto_origin.toggled.connect(self._update_preview)
        self.sub.currentIndexChanged.connect(self._update_preview)
        self.solid.currentIndexChanged.connect(self._solid_changed)
        self.resize(560, 600)
        self._solid_changed()

    def _add_attr_row(self, name, default, desc):
        r = self.attrs.rowCount()
        self.attrs.insertRow(r)
        for c, t in enumerate((name, default, desc)):
            self.attrs.setItem(r, c, QTableWidgetItem(t))

    def _browse(self):
        start = str(self.folder or Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, tr("Ore solid"), start, tr("Surpac DTM (*.dtm)"))
        if not path:
            return
        from ..io import read_dtm
        from ..scene.solid import closed_mesh

        try:
            mesh = closed_mesh(read_dtm(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return
        if mesh is None:
            QMessageBox.warning(self, self.windowTitle(), tr("{name} is not a closed solid", name=Path(path).name))
            return
        self.extra_mesh = (Path(path).stem, mesh)
        self.solid.addItem(Path(path).stem, "file")
        self.solid.setCurrentIndex(self.solid.count() - 1)

    def mesh(self):
        data = self.solid.currentData()
        if data is None:
            return None, None
        if data == "file":
            return self.extra_mesh
        return data.name, solid_mesh(data)

    def _solid_changed(self):
        name, _mesh = self.mesh()
        if name and not self.name.text().strip() or self.name.text().endswith("_bm"):
            self.name.setText(f"{name}_bm" if name else "")
        self._update_preview()

    def _origin(self):
        return None if self.auto_origin.isChecked() else (self.ox.value(), self.oy.value(), self.oz.value())

    def _update_preview(self):
        name, mesh = self.mesh()
        if mesh is None:
            self.preview.setText(tr("Open a solid DTM (e.g. the ore wireframe) in the workspace, or Browse for one."))
            return
        size = (self.sx.value(), self.sy.value(), self.sz.value())
        try:
            origin, n = grid_for_solid(*mesh, size, self._origin())
        except Exception as e:  # noqa: BLE001
            self.preview.setText(str(e))
            return
        if self.auto_origin.isChecked():
            for w, val in zip((self.ox, self.oy, self.oz), origin):
                w.blockSignals(True)
                w.setValue(float(val))
                w.blockSignals(False)
        vol = mesh_volume(*mesh)
        self.preview.setText(tr("Grid {nx} x {ny} x {nz} blocks from origin {ox:.2f}, {oy:.2f}, {oz:.2f}. "
                                "Solid volume {vol} m³: about {est} blocks inside.",
                                nx=int(n[0]), ny=int(n[1]), nz=int(n[2]), ox=origin[0], oy=origin[1], oz=origin[2],
                                vol=fmt_num(round(vol)), est=fmt_num(int(vol / np.prod(size)))))

    def attributes(self):
        out, seen = [], set()
        for r in range(self.attrs.rowCount()):
            nm = (self.attrs.item(r, 0).text() if self.attrs.item(r, 0) else "").strip()
            if not nm:
                continue
            dv = (self.attrs.item(r, 1).text() if self.attrs.item(r, 1) else "").strip()
            ds = (self.attrs.item(r, 2).text() if self.attrs.item(r, 2) else "").strip()
            from ..io.block_model import BlockModel

            why = attribute_name_ok(BlockModel(None, np.zeros((0, 3)), np.zeros((0, 3)), {}), nm)
            if not why and nm.lower() in seen:
                why = tr("There is already an attribute {name}", name=nm)
            if why:
                raise ValueError(f"{nm}: {why}")
            try:
                default = float(dv) if dv else None
            except ValueError:
                raise ValueError(tr("{name}: the default value must be a number", name=nm)) from None
            seen.add(nm.lower())
            out.append((nm, default, ds))
        return out

    def _create(self):
        name, mesh = self.mesh()
        if mesh is None:
            QMessageBox.information(self, self.windowTitle(), tr("Choose the ore solid first."))
            return
        try:
            attrs = self.attributes()
        except ValueError as e:
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return
        title = (self.name.text().strip() or f"{name}_bm")
        start = str((self.folder or Path.cwd()) / f"{title}{BLOCK_MODEL_EXT}")
        path, _ = QFileDialog.getSaveFileName(self, tr("Save block model"), start,
                                              tr("GeoViewer block model (*.gvbm)"))
        if not path:
            return
        path = Path(path).with_suffix(BLOCK_MODEL_EXT)
        size = (self.sx.value(), self.sy.value(), self.sz.value())
        sub = int(self.sub.currentData())
        origin = self._origin()
        null = self.null.value()

        def work(progress, _cancel):
            bm = blocks_in_solid(*mesh, size, origin, sub, attrs, null, title,
                                 progress=lambda f: progress(f * 0.9, tr("Finding the blocks inside {name} ...", name=name)))
            add_history(bm, f"Created inside solid {name}: blocks {size[0]:g} x {size[1]:g} x {size[2]:g} m"
                            + (f", sub-blocking {sub}" if sub > 1 else "") + f", {bm.n_blocks:,} blocks")
            bm.info["source_solid"] = name
            progress(0.95, tr("Saving ..."))
            save_block_model(bm, path)
            bm.path = path
            return bm

        res = run_with_progress(self, tr("New block model from solid"), work, cancellable=False)
        if isinstance(res, Exception):
            QMessageBox.warning(self, self.windowTitle(), str(res))
            return
        self.result_model, self.out_path = res, path
        self.accept()


# --- Surpac ASCII import ----------------------------------------------------------------------
class ImportAsciiDialog(QDialog):
    def __init__(self, source: Path, folder: Path | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Import Surpac block model (ASCII)"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.source = source
        self.folder = folder
        self.result_model = None
        self.out_path: Path | None = None
        cols, rows = csv_preview(source)
        table = QTableWidget(len(rows), len(cols))
        table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(rows):
            for c, val in enumerate(row[:len(cols)]):
                table.setItem(r, c, QTableWidgetItem(val))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setMaximumHeight(170)
        self.name = QLineEdit(source.stem)
        self.null = _num_edit(-99.0, -1e9, 1e9, 3)
        form = QFormLayout()
        form.addRow(tr("File"), QLabel(source.name))
        form.addRow(tr("Name"), self.name)
        form.addRow(tr("Null value"), self.null)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("Import"))
        bb.accepted.connect(self._import)
        bb.rejected.connect(self.reject)
        v = QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(QLabel(tr("First rows")))
        v.addWidget(table)
        v.addWidget(_muted(tr("The blocks are saved in a GeoViewer block model file (.gvbm): compressed and "
                              "encrypted, readable only by GeoViewer. After the import, keep the ASCII file "
                              "somewhere safe or delete it: it holds the same data as plain text.")))
        v.addWidget(bb)
        self.resize(640, 420)

    def _import(self):
        title = self.name.text().strip() or self.source.stem
        start = str((self.folder or self.source.parent) / f"{title}{BLOCK_MODEL_EXT}")
        path, _ = QFileDialog.getSaveFileName(self, tr("Save block model"), start,
                                              tr("GeoViewer block model (*.gvbm)"))
        if not path:
            return
        path = Path(path).with_suffix(BLOCK_MODEL_EXT)
        null = self.null.value()

        def work(progress, _cancel):
            progress(0.1, tr("Reading {name} ...", name=self.source.name))
            bm = read_surpac_ascii(self.source, null)
            bm.info["name"] = title
            progress(0.7, tr("Saving ..."))
            save_block_model(bm, path)
            bm.path = path
            return bm

        res = run_with_progress(self, self.windowTitle(), work, cancellable=False)
        if isinstance(res, Exception):
            QMessageBox.warning(self, self.windowTitle(), str(res))
            return
        self.result_model, self.out_path = res, path
        self.accept()


def offer_to_remove_source(parent, source: Path, out: Path) -> bool:
    """After an import: the ASCII / CSV file holds the same block values as plain text - offer
    to delete it (to the Recycle Bin when possible). True when it was removed."""
    if not source.exists():
        return False
    r = QMessageBox.question(
        parent, tr("Remove the original file?"),
        tr("{name} was imported as {out}.\n\nThe original file holds the same block values as plain text. "
           "Delete it now? It is moved to the Recycle Bin when possible.", name=source.name, out=out.name),
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if r != QMessageBox.Yes:
        return False
    from PySide6.QtCore import QFile

    from ..io.block_model import drop_csv_cache

    drop_csv_cache(source)              # cache of earlier versions (keyed on the file: before it goes)
    ok, _where = QFile.moveToTrash(str(source))
    if not ok:                                       # e.g. a network drive has no Recycle Bin
        try:
            source.unlink()
        except OSError as e:
            QMessageBox.warning(parent, tr("Remove the original file?"),
                                tr("Could not delete {name}: {error}", name=source.name, error=e))
            return False
    log.info("removed the imported block model file %s", source)
    return True


# --- attribute dialogs ------------------------------------------------------------------------




# --- the manager window -----------------------------------------------------------------------
class BlockModelManager(QDialog):
    saved = Signal(object)            # Path of a block model file written

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle(tr("Block Model Manager"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
        self.bm = None
        self.path: Path | None = None
        self.dirty = False

        # left: the block models of the workspace
        self.files = QListWidget()
        self.files.currentItemChanged.connect(self._file_chosen)
        b_new = QPushButton(icon("cube-scan"), tr("New from solid ..."))
        b_new.setToolTip(tr("Make a block model of the blocks inside an ore solid"))
        b_new.clicked.connect(self.new_from_solid)
        b_imp = QPushButton(icon("file-import-outline"), tr("Import Surpac ASCII ..."))
        b_imp.setToolTip(tr("Import a Surpac block model text / CSV export into a GeoViewer block model"))
        b_imp.clicked.connect(lambda: self.import_ascii())
        b_open = QPushButton(icon("folder-open-outline"), tr("Open ..."))
        b_open.clicked.connect(self.open_file)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel(tr("Block models in the workspace")))
        lv.addWidget(self.files, 1)
        for b in (b_new, b_imp, b_open):
            lv.addWidget(b)

        # right: the open block model
        self.title = QLabel("")
        self.title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([tr("Attribute"), tr("Description"), tr("Blocks with value"),
                                              tr("Min"), tr("Mean"), tr("Max")])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history = QListWidget()
        self.history.setMaximumHeight(120)
        self.b_save = QPushButton(icon("content-save-outline"), tr("Save"))
        self.b_save.clicked.connect(self.save)
        self.b_save_as = QPushButton(icon("content-save-move-outline"), tr("Save as ..."))
        self.b_save_as.clicked.connect(self.save_as)
        self.state = QLabel("")
        self.state.setStyleSheet(f"color: {theme.MUTED};")
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        brow = QHBoxLayout()
        brow.addWidget(self.state, 1)
        brow.addWidget(self.b_save)
        brow.addWidget(self.b_save_as)
        brow.addWidget(close)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.title)
        rv.addWidget(self.info)
        rv.addWidget(QLabel(tr("Attributes")))
        rv.addWidget(self.table, 1)
        rv.addWidget(QLabel(tr("History")))
        rv.addWidget(self.history)
        self.empty = _muted(tr("Choose a block model on the left, or make one: New from solid (blocks inside an "
                               "ore wireframe) or Import Surpac ASCII (a block model text export from Surpac).\n\n"
                               "Block models are kept in GeoViewer block model files (.gvbm): compressed and "
                               "encrypted, so the block values cannot be read without GeoViewer."))
        rv.addWidget(self.empty)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 700])
        v = QVBoxLayout(self)
        v.addWidget(split, 1)
        v.addLayout(brow)
        self.resize(1000, 640)
        self.refresh_files()
        self._show()

    # --- files -----------------------------------------------------------------------------
    @property
    def folder(self) -> Path | None:
        ws = getattr(self.main, "workspace", None)
        return ws.folder if ws is not None else None

    @property
    def scene(self):
        return getattr(self.main, "scene", None)

    def refresh_files(self, select: Path | None = None) -> None:
        paths = []
        if self.folder is not None and self.folder.is_dir():
            from ..workspace import list_data_files

            paths = [p for p in list_data_files(self.folder) if p.suffix.lower() == BLOCK_MODEL_EXT]
        if self.path is not None and self.path not in paths:
            paths.append(self.path)
        if select is not None and select not in paths:
            paths.append(select)
        self.files.blockSignals(True)
        self.files.clear()
        for p in paths:
            rel = p.relative_to(self.folder) if self.folder and p.is_relative_to(self.folder) else p
            it = QListWidgetItem(icon("cube-outline"), str(rel.with_suffix("")))
            it.setData(Qt.UserRole, str(p))
            self.files.addItem(it)
            if p == (select or self.path):
                self.files.setCurrentItem(it)
        self.files.blockSignals(False)

    def _confirm_discard(self) -> bool:
        if not self.dirty or self.bm is None:
            return True
        r = QMessageBox.question(self, self.windowTitle(),
                                 tr("{name} has unsaved changes. Save them?", name=self.bm.info.get("name", "")),
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def _file_chosen(self, cur, _prev) -> None:
        if cur is None:
            return
        p = Path(cur.data(Qt.UserRole))
        if p == self.path:
            return
        if not self._confirm_discard():
            self.refresh_files()
            return
        self.load(p)

    def load(self, path: Path) -> bool:
        res = run_with_progress(self, tr("Opening {name}", name=path.name),
                                lambda _p, _c: read_block_model(path), cancellable=False)
        if isinstance(res, Exception):
            QMessageBox.warning(self, self.windowTitle(), str(res))
            return False
        self.bm, self.path, self.dirty = res, path, False
        self.refresh_files(path)
        self._show()
        return True

    def open_file(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, tr("Open block model"), str(self.folder or Path.cwd()),
                                              tr("GeoViewer block model (*.gvbm)"))
        if path:
            self.load(Path(path))

    def new_from_solid(self) -> None:
        if not self._confirm_discard():
            return
        dlg = NewFromSolidDialog(self.scene, self.folder, self)
        if dlg.exec() and dlg.result_model is not None:
            self._adopt(dlg.result_model, dlg.out_path)

    def import_ascii(self, source: str | Path | None = None) -> bool:
        """Import a Surpac block model ASCII / CSV export into a .gvbm (``source``: the file, e.g.
        a block model CSV the user opened; None: choose one). Afterwards the original file can be
        deleted. True when imported."""
        if not self._confirm_discard():
            return False
        if source is None:
            source, _ = QFileDialog.getOpenFileName(
                self, tr("Import Surpac block model (ASCII)"), str(self.folder or Path.cwd()),
                tr("Block model text export (*.csv *.txt *.asc *.dat);;All files (*.*)"))
            if not source:
                return False
        source = Path(source)
        try:
            dlg = ImportAsciiDialog(source, self.folder, self)
        except Exception as e:  # noqa: BLE001 - not a block model export
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return False
        if not (dlg.exec() and dlg.result_model is not None):
            return False
        self._adopt(dlg.result_model, dlg.out_path)
        offer_to_remove_source(self, source, dlg.out_path)
        return True

    def _adopt(self, bm, path: Path) -> None:
        self.bm, self.path, self.dirty = bm, path, False
        self.refresh_files(path)
        self._show()
        self.saved.emit(path)

    # --- display ---------------------------------------------------------------------------
    def selected_attribute(self) -> str | None:
        r = self.table.currentRow()
        if r < 0 or self.table.item(r, 0) is None:
            return None
        return self.table.item(r, 0).data(Qt.UserRole)

    def _show(self, select: str | None = None) -> None:
        bm = self.bm
        has = bm is not None
        for w in (self.table, self.history, self.b_save_as):
            w.setEnabled(has)
        self.b_save.setEnabled(has and self.dirty)
        self.empty.setVisible(not has)
        if not has:
            self.title.setText("")
            self.info.setText("")
            self.table.setRowCount(0)
            self.history.clear()
            self.state.setText("")
            return
        d = describe(bm)
        self.title.setText(bm.info.get("name") or (self.path.stem if self.path else ""))
        sizes = ", ".join(" x ".join(fmt_num(float(v), 3) for v in s) for s in d["sizes"][:4])
        if len(d["sizes"]) > 4:
            sizes += " ..."
        lo, hi = d["lo"], d["hi"]
        lines = [
            tr("File: {file}", file=str(self.path) if self.path else "-"),
            tr("Blocks: {n} ({kind})", n=fmt_num(d["n_blocks"]),
               kind=tr("sub-blocked") if d["sub_blocked"] else tr("regular")),
            tr("Block sizes (m): {sizes}", sizes=sizes),
            tr("Extent: X {x0} - {x1}, Y {y0} - {y1}, Z {z0} - {z1}", x0=fmt_num(lo[0], 2), x1=fmt_num(hi[0], 2),
               y0=fmt_num(lo[1], 2), y1=fmt_num(hi[1], 2), z0=fmt_num(lo[2], 2), z1=fmt_num(hi[2], 2)),
            tr("Volume: {v} m³    Null value: {null}", v=fmt_num(round(d["volume"])), null=fmt_num(bm.null_value)),
        ]
        if bm.info.get("source_solid"):
            lines.append(tr("Made inside solid: {name}", name=bm.info["source_solid"]))
        self.info.setText("\n".join(lines))
        desc = battr.descriptions(bm)
        names = [n for n in bm.attributes if n not in DEPLETION_ATTRS]
        self.table.setRowCount(len(names))
        for r, n in enumerate(names):
            st = battr.attribute_stats(bm, n)
            cells = [n, desc.get(n, ""), fmt_num(st.get("n", 0))]
            if st.get("text"):
                cells += [tr("text"), tr("{n} codes", n=st["codes"]), ""]
            elif st.get("n"):
                cells += [fmt_num(st["min"], 4), fmt_num(st["mean"], 4), fmt_num(st["max"], 4)]
            else:
                cells += ["-", "-", "-"]
            for c, t in enumerate(cells):
                it = QTableWidgetItem(t)
                if c == 0:
                    it.setData(Qt.UserRole, n)
                if c >= 2:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, it)
            if n == select:
                self.table.selectRow(r)
        if select is None and names and self.table.currentRow() < 0:
            self.table.selectRow(0)
        self.history.clear()
        for h in reversed(bm.info.get("history", [])):
            self.history.addItem(f"{str(h.get('when', '')).replace('T', ' ')}   {h.get('what', '')}")
        self.state.setText(tr("Unsaved changes") if self.dirty else tr("Saved"))

    def _changed(self, select: str | None = None) -> None:
        self.dirty = True
        self._show(select)







    # --- saving --------------------------------------------------------------------------------
    def save(self) -> bool:
        if self.bm is None:
            return False
        if self.path is None:
            return self.save_as()
        return self._write(self.path)

    def save_as(self) -> bool:
        if self.bm is None:
            return False
        start = str(self.path or (self.folder or Path.cwd()) / f"{self.bm.info.get('name', 'block_model')}{BLOCK_MODEL_EXT}")
        path, _ = QFileDialog.getSaveFileName(self, tr("Save block model"), start, tr("GeoViewer block model (*.gvbm)"))
        if not path:
            return False
        return self._write(Path(path).with_suffix(BLOCK_MODEL_EXT))

    def _write(self, path: Path) -> bool:
        bm = self.bm
        res = run_with_progress(self, tr("Saving {name}", name=path.name),
                                lambda _p, _c: save_block_model(bm, path, skip=DEPLETION_ATTRS), cancellable=False)
        if isinstance(res, Exception):
            QMessageBox.warning(self, self.windowTitle(), tr("Could not save {name}: {error}", name=path.name, error=res))
            return False
        self.path = path
        bm.path = path
        self.dirty = False
        self.refresh_files(path)
        self._show(self.selected_attribute())
        self.saved.emit(path)
        return True

    @staticmethod
    def _shutdown_workers() -> None:
        from ..blockmodel import parallel

        parallel.shutdown()               # the worker processes are not needed any more

    def reject(self) -> None:
        """Close button, Esc and the window's X (QDialog's closeEvent calls this): ask about
        unsaved changes, then hide. Must not call close(): that runs inside a close already,
        Qt ignores the nested close and the dialog would stay open."""
        if not self._confirm_discard():
            return
        self._shutdown_workers()
        super().reject()

    def closeEvent(self, e) -> None:
        if self.isVisible():
            super().closeEvent(e)         # -> reject(); still visible afterwards = Cancel, ignored
            return
        if not self._confirm_discard():   # closed while hidden / parked (e.g. another workspace opens)
            e.ignore()
            return
        self._shutdown_workers()
        e.accept()


__all__ = ["BlockModelManager", "run_with_progress", "solid_layers", "solid_mesh", "BlockFileError"]
