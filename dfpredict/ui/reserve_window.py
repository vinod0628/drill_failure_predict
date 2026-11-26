"""Reserve management window: total / produced / remaining, the mined shapes, by month.

``ReserveManager`` ties the reserve (scene/reserve.py) to the main window: it loads
production records (prod_* tables of the project database), recomputes the depletion when layers or shapes change,
puts ``mined_fraction`` / ``status`` into the block model (memory only), keeps the
"Mined shapes" layer up to date and runs the outline tool and the mined-shape dialog.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from PySide6.QtCore import QDate, QObject, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDateEdit, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
                               QStackedWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout)

from ..scene.production import day_of, month_label, month_of, today
from ..scene.reserve import Reserve, clear_depletion_attrs, ensure_depletion_attrs, monthly, uses_depletion
from ..i18n import N_, tr, trc
from . import theme
from .icons import icon

log = logging.getLogger(__name__)

WHOLE_MODEL = N_("(whole block model)")
FIXED = N_("Fixed value")
PLANE_SHORT = {"xy": N_("Plan XY"), "xz": N_("Section XZ"), "yz": N_("Section YZ"), "3d": N_("Solid (.dtm)")}


def _metal_short(metal: float, unit: str) -> str:
    """Metal as a number for the tables (the unit is in the column title)."""
    return f"{metal / 100.0:,.1f}" if unit == "%" else f"{metal / 1000.0:,.1f}"


# --- the manager -------------------------------------------------------------------------------
class ReserveManager(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.reserve = Reserve()
        self.folder: Path | None = None
        self.layer = None                  # MinedShapesLayer
        self.show_shapes = True
        self.as_of: str | None = None      # None: today
        self.window: ReserveWindow | None = None
        self.shape_dialog = None
        self._syncing = False
        self._state = None
        self.db_sig = None                             # fingerprint of the records shown (see remember_records)

    # --- which folder's records ---------------------------------------------------
    def _folder(self) -> Path | None:
        ws = self.win.workspace
        if ws is not None:
            return Path(ws.folder)
        bms = Reserve.block_layers(self.win.scene.layers)
        if bms and bms[0].path is not None:
            return Path(bms[0].path).parent
        return None

    @property
    def as_of_day(self) -> int:
        return day_of(self.as_of or today())

    # --- keeping up with the layers ---------------------------------------------------------
    def prepare(self, layers) -> None:
        """Before block models are shown: mined attributes exist (null until computed)."""
        for lay in Reserve.block_layers(layers):
            ensure_depletion_attrs(lay.data)

    def reset(self) -> None:
        """Another workspace: forget everything (the scene was cleared)."""
        self.reserve = Reserve()
        self.folder, self.layer, self._state = None, None, None
        self.show_shapes = True
        if self.shape_dialog is not None:
            self.shape_dialog.close()
        if self.window is not None:
            self.window.refresh()

    def database_changed(self) -> None:
        """Another database (or it became reachable / unreachable): read the production
        records again and update the figures, the mined shapes and the block colours."""
        if self.shape_dialog is not None:
            self.shape_dialog.close()
        self.folder, self._state = None, None          # the next sync opens the records again
        self.reserve = Reserve()
        self.sync(force=True)

    def sync(self, force: bool = False) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._sync(force)
        except Exception:  # noqa: BLE001 - never break the main window
            log.exception("reserve update failed")
        finally:
            self._syncing = False

    def _sync(self, force: bool) -> None:
        layers = self.win.scene.layers
        if self.layer is not None and self.layer not in layers:
            self.layer, self.show_shapes = None, False       # removed by the user
        folder = self._folder()
        if folder is None:
            return
        if folder != self.folder or self.reserve.log is None:
            self.folder = folder
            self.reserve.open(folder)
            self._state = None
            self.remember_records()
            lg = self.reserve.log
            if lg is not None and lg.imported_from is not None:
                self.win.status_msg.setText(tr("Production records moved from {name} into the project database "
                                               "({where})", name=Path(lg.imported_from).name, where=lg.where))
        r = self.reserve
        if r.log is not None and r.auto_setup(layers) and r.log.exists:
            self._write(r.log.save_setup, quiet=True)
        r.update(layers)
        shapes = tuple((s.id, s.date, s.from_, s.to, s.plane, len(s.points), s.source,
                        len(s.triangles) if s.is_solid else 0) for s in (r.log.shapes if r.log else []))
        state = (r._key, shapes, self.as_of_day, r.ready)
        if state != self._state or force:
            self._state = state
            self._apply()
        if self.window is not None and self.window.isVisible():
            self.window.refresh()

    def _apply(self) -> None:
        r = self.reserve
        bm = r.inject(self.as_of_day) if r.ready else None
        for lay in Reserve.block_layers(self.win.scene.layers):
            if bm is None:
                clear_depletion_attrs(lay.data)        # no records: no mined colours from before
            if (bm is not None and lay.data is bm or bm is None) and uses_depletion(lay):
                lay.rebuild()
        self._update_layer()
        if self.win.props.layer is not None:
            self.win.props.refresh()
        self.win.scene.render()

    def _update_layer(self) -> None:
        from ..scene.layers import MinedShapesLayer

        r = self.reserve
        shapes = [s for s in r.log.sorted_shapes() if s.day <= self.as_of_day] if r.log else []
        if self.layer is None:
            # listed also with no shapes yet (greyed out); not before other data set the origin
            if not self.show_shapes or self.win.scene.origin is None or r.log is None:
                return
            from ..workspace import restore_app_layer

            self.layer = MinedShapesLayer(name="Mined shapes", path=None, color=(1.0, 0.6, 0.2))
            restore_app_layer(self.win.workspace, self.layer)    # hidden / opacity / colour as last time
            self.layer.shapes = shapes
            self._fill_stats()
            self.win._add_layers([self.layer])
            return
        self.layer.shapes = shapes
        self._fill_stats()
        self.layer.rebuild()
        self.win.layer_tree.refresh_state(self.layer)       # greyed out while there are no shapes

    def _fill_stats(self) -> None:
        r = self.reserve
        self.layer.unit = r.setup.get("unit", "g/t")
        self.layer.stats = {s.id: r.dep.shape_totals(s.id) for s in self.layer.shapes} if r.ready else {}

    def set_show_shapes(self, on: bool) -> None:
        self.show_shapes = on
        if self.layer is not None:
            self.layer.set_visible(on)
            self.win.layer_tree.update_check(self.layer)
            self.win.scene.render()
        elif on:
            self._update_layer()
            self.win.scene.render()

    # --- setup changes -------------------------------------------------------------------
    def layer_renamed(self, old: str, new: str) -> None:
        """The block model or the reserve solid was renamed: keep using it (saved setup too)."""
        kw = self.reserve.renamed(old, new)
        if kw:
            self.set_setup(**kw)

    def set_setup(self, **kw) -> None:
        if self.reserve.log is None:
            return
        self.reserve.set_setup(**kw)
        if self.reserve.log.exists:
            self._write(self.reserve.log.save_setup)
        self.sync(force=True)

    def set_as_of(self, date: str | None) -> None:
        self.as_of = None if not date or date == today() else date
        self.sync()

    # --- the window --------------------------------------------------------------------------
    def show_window(self) -> None:
        self.sync()
        park = getattr(self.win, "parking", None)
        if self.window is not None and park is not None and park.restore("reserve"):
            self.window.refresh()                   # as it was left (with the advisor, if open)
            return
        if self.window is None:
            self.window = ReserveWindow(self)
            if park is not None:
                park.watch(self.window, "reserve")
        self.window.refresh()
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    # --- drawing and entering shapes -------------------------------------------------------------
    def can_draw(self) -> bool:
        self.sync()
        if self.folder is None:
            QMessageBox.information(self.win, tr("Mined outline"), tr("Open a workspace (or the block model) first."))
            return False
        if not self.reserve.ready:
            QMessageBox.information(self.win, tr("Mined outline"),
                                    tr("The reserve is not ready: {error}.\n\nSet it up in the Reserve window first.",
                                       error=self.reserve.error))
            self.show_window()
            return False
        return True

    def outline_done(self, plane: str, points) -> None:
        from .mined_shape_dialog import new_shape

        if not self.reserve.ready:
            self.can_draw()
            return
        self._open_dialog(new_shape(self.reserve.log, plane, points), editing=False)

    def edit_shape(self, sid: int) -> None:
        import copy

        sh = self.reserve.log.get(sid) if self.reserve.log else None
        if sh is None or not self.reserve.ready:
            return
        self._open_dialog(copy.deepcopy(sh), editing=True)

    def add_solid(self) -> None:
        """Record a mined solid from a Surpac .dtm (e.g. a surveyed stope)."""
        from PySide6.QtWidgets import QFileDialog

        from .mined_shape_dialog import load_mined_solid

        if not self.reserve.ready:
            self.can_draw()
            return
        start = str(self.folder) if self.folder is not None else ""
        path, _ = QFileDialog.getOpenFileName(self.win, tr("Load mined solid"), start, tr("Surpac DTM (*.dtm)"))
        if not path:
            return
        try:
            vertices, triangles = load_mined_solid(path)
        except Exception as e:  # noqa: BLE001 - not a solid, unreadable file
            QMessageBox.warning(self.win, tr("Load mined solid"), f"{Path(path).name}\n\n{e}")
            return
        from ..scene.production import MinedShape

        sid = self.reserve.log.next_id()
        shape = MinedShape.from_solid(sid, today(), Path(path).stem, vertices, triangles, Path(path).name)
        self._open_dialog(shape, editing=False)

    def _open_dialog(self, shape, editing: bool) -> None:
        from .mined_shape_dialog import MinedShapeDialog, MinedSolidDialog

        if self.shape_dialog is not None:
            self.shape_dialog.close()
        kind = MinedSolidDialog if shape.is_solid else MinedShapeDialog
        dlg = kind(self.win, self.win.scene, self.reserve, shape, editing)
        dlg.saved.connect(self.save_shape)
        dlg.deleted.connect(self.delete_shape)
        dlg.destroyed.connect(lambda *_: setattr(self, "shape_dialog", None))
        self.shape_dialog = dlg
        dlg.show()

    def remember_records(self) -> None:
        """The fingerprint of the production records as now shown (Refresh compares with it)."""
        from ..io.database import project_target
        from ..scene.production import production_signature

        self.db_sig = production_signature(project_target(self.folder)) if self.folder is not None else None

    def _write(self, fn, *args, quiet: bool = False) -> bool:
        """Run a database write; on failure tell the user (nothing is half written)."""
        try:
            fn(*args)
            self.remember_records()                    # our own change: nothing to re-read
            return True
        except Exception as e:  # noqa: BLE001 - sqlite3 errors, locked or read-only file
            log.warning("production database write failed: %s", e)
            if not quiet:
                QMessageBox.warning(self.win, tr("Reserve"),
                                    tr("Could not save to the production database\n{where}\n\n{error}\n\n"
                                       "Nothing was changed. Check that the folder is not read-only and that no "
                                       "other program has the file open, then try again.",
                                       where=self.reserve.log.where, error=e))
            return False

    def _write_failed(self) -> None:
        if self.shape_dialog is not None:
            self.shape_dialog.written = False        # keep the dialog open so nothing typed is lost

    def save_shape(self, sh) -> None:
        if not self._write(self.reserve.log.put, sh):
            self._write_failed()
            return
        if sh.day > self.as_of_day:          # show the new cut even when dated after "as of"
            self.as_of = None if sh.date <= today() else sh.date
        self.sync(force=True)
        if self.reserve.ready:
            tot, _b = self.reserve.dep.shape_totals(sh.id)
            unit = self.reserve.setup.get("unit", "g/t")
            self.win.status_msg.setText(tr("Saved {name} ({date}): {t:,.0f} t @ {g:,.2f} {unit} - draw the next "
                                           "outline or press Esc", name=sh.name, date=sh.date, t=tot.tonnes,
                                           g=tot.grade, unit=unit))
        if self.window is not None:                 # also when hidden while drawing: shown selected on return
            self.window.select(sh.id)

    def delete_shape(self, sid: int) -> None:
        if not self._write(self.reserve.log.remove, sid):
            self._write_failed()
            return
        self.sync(force=True)

    def zoom_to(self, sid: int) -> None:
        sh = self.reserve.log.get(sid) if self.reserve.log else None
        if sh is None or self.win.scene.origin is None:
            return
        lo, hi = sh.bounds()
        pad = max(float((hi - lo).max()) * 0.6, 10.0)
        lo, hi = self.win.scene.to_local(lo - pad), self.win.scene.to_local(hi + pad)
        self.win.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        self.win.scene.refresh_grid()
        self.win.scene.render()


# --- the window ---------------------------------------------------------------------------------
class _Card(QFrame):
    def __init__(self, title: str, accent: str):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet(f"QFrame#card {{ background: {theme.PANEL_ALT}; border: 1px solid {theme.BORDER};"
                           f" border-left: 4px solid {accent}; border-radius: 6px; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(1)
        t = QLabel(tr(title).upper())
        t.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt; font-weight: 600; letter-spacing: 1px;"
                        " border: none;")
        self.big = QLabel("-")
        self.big.setStyleSheet("font-size: 15pt; font-weight: 600; border: none;")
        self.line1 = QLabel("")
        self.line2 = QLabel("")
        for w in (self.line1, self.line2):
            w.setStyleSheet(f"color: {theme.TEXT}; border: none;")
        for w in (t, self.big, self.line1, self.line2):
            v.addWidget(w)

    def set(self, big: str, line1: str = "", line2: str = "") -> None:
        self.big.setText(big)
        self.line1.setText(line1)
        self.line2.setText(line2)


def _item(text: str, align=Qt.AlignRight | Qt.AlignVCenter, data=None) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment(align)
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if data is not None:
        it.setData(Qt.UserRole, data)
    return it


class ReserveWindow(QDialog):
    HOW_TO = N_("<p style='font-size:11pt; font-weight:600'>No mined shapes yet</p>"
              "<p>To record what was mined:</p>"
              "<p>1. Click <b>Draw mined outline</b> (the view changes to the plan view).<br>"
              "2. Click the corners of the mined area, then <b>right-click</b> (or press Enter).<br>"
              "3. Set the date and the levels <b>From</b> / <b>To</b> - the chart shows where the ore is.<br>"
              "4. Click <b>Save</b>. The produced and remaining figures update at once.</p>")
    NO_DATABASE = N_("<p style='font-size:11pt; font-weight:600'>The production records cannot be read</p>"
                   "<p>The database chosen in <b>Settings &gt; Database</b> cannot be reached (see the "
                   "message above).<br>Nothing is lost: the records are shown again when the database "
                   "can be reached.</p><p>Check the MySQL server name, user and password with <b>Test "
                   "connection</b>, or choose SQLite.</p>")
    CUT_COLUMNS = [N_("Date"), N_("Name"), N_("Drawn in"), N_("From"), N_("To"), N_("Thickness (m)"), N_("Tonnes"),
                   N_("Grade"), N_("Metal"), N_("Blocks")]
    MONTH_COLUMNS = [N_("Month"), N_("Cuts"), N_("Tonnes"), N_("Grade"), N_("Metal"), N_("Cumulative t"),
                     N_("Remaining t"), N_("Remaining metal")]

    def __init__(self, mgr: ReserveManager):
        super().__init__(mgr.win)
        self.mgr = mgr
        self.setWindowTitle(tr("Reserve"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
        self._filling = False

        # --- setup (collapsible) -------------------------------------------------------
        self.b_setup = QToolButton()
        self.b_setup.setText(trc("reserve", "Setup"))
        self.b_setup.setIcon(icon("cog-outline"))
        self.b_setup.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.b_setup.setCheckable(True)
        self.b_setup.toggled.connect(lambda on: self.setup_box.setVisible(on))
        self.status = QLabel()
        self.status.setWordWrap(True)

        self.setup_box = QFrame()
        sf = QFormLayout(self.setup_box)
        sf.setLabelAlignment(Qt.AlignRight)
        self.c_bm = QComboBox()
        self.c_solid = QComboBox()
        self.c_dens = QComboBox()
        self.s_dens = QDoubleSpinBox()
        self.s_dens.setRange(0.1, 25.0)
        self.s_dens.setDecimals(2)
        self.s_dens.setSuffix(" t/m³")
        self.c_grade = QComboBox()
        self.c_unit = QComboBox()
        self.c_unit.addItems(["g/t", "%"])
        dens_row = QHBoxLayout()
        dens_row.addWidget(self.c_dens, 1)
        dens_row.addWidget(self.s_dens)
        grade_row = QHBoxLayout()
        grade_row.addWidget(self.c_grade, 1)
        grade_row.addWidget(self.c_unit)
        sf.addRow(tr("Block model"), self.c_bm)
        sf.addRow(tr("Reserve solid"), self.c_solid)
        sf.addRow(tr("Density"), dens_row)
        sf.addRow(tr("Grade"), grade_row)
        self.c_bm.setToolTip(tr("The block model holding the grade and density"))
        self.c_solid.setToolTip(tr("Blocks whose centre is inside this solid make up the total reserve"))
        self.c_dens.setToolTip(tr("Density attribute of the block model, or a fixed value for all blocks"))
        self.c_unit.setToolTip(tr("g/t: metal shown in kg;  %: metal shown in tonnes"))
        self.c_bm.activated.connect(lambda _i: self._setup("block_model", self.c_bm.currentText()))
        self.c_solid.activated.connect(
            lambda _i: self._setup("solid", "" if self.c_solid.currentData() == WHOLE_MODEL
                                   else self.c_solid.currentData()))
        self.c_dens.activated.connect(
            lambda _i: self._setup("density", "" if self.c_dens.currentData() == FIXED else self.c_dens.currentData()))
        self.s_dens.editingFinished.connect(lambda: self._setup("density_value", round(self.s_dens.value(), 3)))
        self.c_grade.activated.connect(lambda _i: self._setup("grade", self.c_grade.currentText()))
        self.c_unit.activated.connect(lambda _i: self._setup("unit", self.c_unit.currentText()))

        # --- as of + cards ---------------------------------------------------------------
        self.d_asof = QDateEdit()
        self.d_asof.setCalendarPopup(True)
        self.d_asof.setDisplayFormat("yyyy-MM-dd")
        self.d_asof.setToolTip(tr("Figures, block status and mined shapes up to and including this day"))
        self.d_asof.dateChanged.connect(self._as_of_changed)
        b_today = QPushButton(tr("Today"))
        b_today.clicked.connect(lambda: self.d_asof.setDate(QDate.currentDate()))
        top = QHBoxLayout()
        top.addWidget(QLabel(tr("As of")))
        top.addWidget(self.d_asof)
        top.addWidget(b_today)
        top.addStretch(1)
        self.b_ai = QPushButton(icon("creation", "#f0b429"), tr("Advisor"))
        self.b_ai.setToolTip(tr("How long the reserve lasts and a check of every cut, explained in plain words "
                                "(worked out on this computer, changes nothing)"))
        self.b_ai.clicked.connect(self._advisor)
        top.addWidget(self.b_ai)
        top.addWidget(self.b_setup)

        self.card_total = _Card(N_("Total reserve"), "#8a94a3")
        self.card_prod = _Card(N_("Produced"), "#e0533f")
        self.card_rem = _Card(N_("Remaining"), "#4cc76a")
        cards = QHBoxLayout()
        for c in (self.card_total, self.card_prod, self.card_rem):
            cards.addWidget(c, 1)

        # --- tables ----------------------------------------------------------------------------
        self.t_cuts = self._table(self.CUT_COLUMNS, stretch=1)
        self.t_cuts.itemDoubleClicked.connect(lambda _it: self._edit())
        self.t_cuts.itemSelectionChanged.connect(self._update_buttons)
        self.t_month = self._table(self.MONTH_COLUMNS)
        self.tabs = QTabWidget()
        # no shapes yet: how to start, in place of the empty table
        self.how_to = QLabel(tr(self.HOW_TO))
        self.how_to.setAlignment(Qt.AlignCenter)
        self.how_to.setWordWrap(True)
        self.how_to.setStyleSheet(f"color: {theme.MUTED}; background: {theme.PANEL};")
        self.cuts_page = QStackedWidget()
        self.cuts_page.addWidget(self.how_to)
        self.cuts_page.addWidget(self.t_cuts)
        self.tabs.addTab(self.cuts_page, icon("vector-polygon"), tr("Mined shapes"))
        self.tabs.addTab(self.t_month, icon("calendar-month"), tr("By month"))
        self.tabs.currentChanged.connect(lambda _i: self._update_buttons())

        self.b_draw = QPushButton(icon("vector-polygon", "#4cc76a"), tr("Draw mined outline"))
        self.b_draw.setToolTip(tr("Draw the outline of the mined area in the plan (or a section) view, "
                                  "then enter the date and From / To"))
        self.b_draw.clicked.connect(self._draw)
        self.b_solid = QPushButton(icon("cube-scan", "#4cc76a"), tr("Load mined solid ..."))
        self.b_solid.setToolTip(tr("Record a mined solid from a Surpac .dtm (a closed solid, e.g. a surveyed "
                                   "stope) instead of drawing an outline"))
        self.b_solid.clicked.connect(lambda: self.mgr.add_solid())
        self.b_edit = QPushButton(icon("pencil-outline"), tr("Edit"))
        self.b_edit.clicked.connect(self._edit)
        self.b_del = QPushButton(icon("delete-outline"), tr("Delete"))
        self.b_del.clicked.connect(self._delete)
        self.b_zoom = QPushButton(icon("magnify-scan"), tr("Zoom to"))
        self.b_zoom.clicked.connect(self._zoom)
        self.b_csv = QPushButton(icon("table-arrow-down"), tr("Export CSV"))
        self.b_csv.setToolTip(tr("Save the table shown (mined shapes or by month) as a CSV file"))
        self.b_csv.clicked.connect(self._export)
        btns = QHBoxLayout()
        for b in (self.b_draw, self.b_solid, self.b_edit, self.b_del, self.b_zoom):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(self.b_csv)

        # --- show in the view -------------------------------------------------------------------
        self.b_shapes = QPushButton(icon("cube-outline"), tr("Mined shapes in the view"))
        self.b_shapes.setCheckable(True)
        self.b_shapes.toggled.connect(lambda on: None if self._filling else self.mgr.set_show_shapes(on))
        view_row = QHBoxLayout()
        view_row.addWidget(self.b_shapes)
        mined_hint = QLabel(tr("Mined blocks: gray / transparent / hidden - in the block model's properties"))
        mined_hint.setStyleSheet(f"color: {theme.MUTED};")
        view_row.addSpacing(12)
        view_row.addWidget(mined_hint)
        view_row.addStretch(1)

        self.foot = QLabel()
        self.foot.setStyleSheet(f"color: {theme.MUTED};")
        self.foot.setWordWrap(True)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.setup_box)
        root.addWidget(self.status)
        root.addLayout(cards)
        root.addLayout(view_row)
        root.addWidget(self.tabs, 1)
        root.addLayout(btns)
        root.addWidget(self.foot)
        self.resize(860, 620)
        self.b_setup.setChecked(False)
        self.setup_box.setVisible(False)

    def keyPressEvent(self, e):
        # Enter confirms the value being typed; it does not press a button (Save, ...).
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            e.accept()
            return
        super().keyPressEvent(e)

    def _advisor(self) -> None:
        """The reserve advisor (forecast and cut check), next to this window."""
        from .reserve_advisor_dialog import ReserveAdvisorDialog

        dlg = getattr(self, "advisor", None)
        if dlg is None or not dlg.isVisible():
            self.advisor = dlg = ReserveAdvisorDialog(self.mgr, self)
            park = getattr(self.mgr.win, "parking", None)
            if park is not None:
                park.watch(dlg, "reserve")
        else:
            dlg.refresh()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    @staticmethod
    def _table(cols, stretch: int | None = None) -> QTableWidget:
        """A read-only table; column ``stretch`` takes the spare width (else the last one)."""
        t = QTableWidget(0, len(cols))
        for c, name in enumerate(cols):
            t.setHorizontalHeaderItem(c, QTableWidgetItem())
            _set_header(t, c, tr(name), name)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setAlternatingRowColors(True)
        h = t.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        if stretch is None:
            h.setStretchLastSection(True)
        else:
            h.setSectionResizeMode(stretch, QHeaderView.Stretch)
        h.setMinimumSectionSize(60)
        return t

    # --- filling ------------------------------------------------------------------------------
    def refresh(self) -> None:
        self._filling = True
        try:
            self._fill()
        finally:
            self._filling = False
        self._update_buttons()

    def _fill(self) -> None:
        mgr, r = self.mgr, self.mgr.reserve
        layers = mgr.win.scene.layers
        s = r.setup
        # setup combos
        bms = [b.name for b in Reserve.block_layers(layers)]
        self._combo(self.c_bm, bms or ([s["block_model"]] if s["block_model"] else []), s["block_model"])
        solids = [WHOLE_MODEL] + [lay.name for lay in Reserve.solid_layers(layers)]
        if s["solid"] and s["solid"] not in solids:
            solids.append(s["solid"])
        self._combo(self.c_solid, solids, s["solid"] or WHOLE_MODEL, special=(WHOLE_MODEL,))
        nums = Reserve.number_names(r.bm_layer.data) if r.bm_layer is not None else []
        self._combo(self.c_dens, [FIXED] + nums, s["density"] or FIXED, special=(FIXED,))
        self.s_dens.setValue(float(s["density_value"]))
        self.s_dens.setEnabled(not s["density"])
        self._combo(self.c_grade, nums or ([s["grade"]] if s["grade"] else []), s["grade"])
        self.c_unit.setCurrentText(s.get("unit", "g/t"))
        self.d_asof.blockSignals(True)
        self.d_asof.setDate(QDate.fromString(mgr.as_of or today(), "yyyy-MM-dd"))
        self.d_asof.blockSignals(False)

        unit = s.get("unit", "g/t")
        ready = r.ready and not r.error
        if not r.ready or r.error:
            self.status.setText(f"⚠ {r.error or tr('Not set up')}" + ("" if r.log is None else tr(" - open Setup")))
            self.status.setStyleSheet("color: #ffb347;")
            self.b_setup.setChecked(r.log is not None)
        else:
            where = tr("inside {solid}", solid=s["solid"]) if s["solid"] else tr("in the whole block model")
            dens = s["density"] or f"{float(s['density_value']):g} t/m³"
            self.status.setText(tr("{n:,} reserve blocks {where} · density {dens} · grade {grade}",
                                   n=len(r.dep.idx), where=where, dens=dens, grade=s["grade"]))
            self.status.setStyleSheet(f"color: {theme.MUTED};")
        self.status.setVisible(True)
        unreadable = r.log is None and bool(getattr(r, "open_error", ""))
        if unreadable:
            self.foot.setText(tr("The production records cannot be read - check Settings > Database."))
        elif mgr.folder is not None:
            self.foot.setText(tr("Mined shapes are saved in {where}. The block model file is not changed.",
                                 where=r.log.where if r.log else "-"))
        else:
            self.foot.setText(tr("Open a workspace to keep a production record."))
        self.how_to.setText(tr(self.NO_DATABASE if unreadable else self.HOW_TO))

        # cards
        if ready:
            sm = r.dep.summary(mgr.as_of_day)
            tot, prod, rem = sm["total"], sm["produced"], sm["remaining"]
            pct = prod.tonnes / tot.tonnes if tot.tonnes else 0.0

            def metal(m):
                from ..scene.production import metal_text

                return metal_text(m, unit)

            self.card_total.set(f"{tot.tonnes:,.0f} t", f"{tot.grade:,.2f} {unit}", metal(tot.metal))
            pg = (tr("{g:,.2f} {unit}  ·  {pct:.1%} of total", g=prod.grade, unit=unit, pct=pct)
                  if prod.tonnes > 0 else tr("nothing mined yet"))
            self.card_prod.set(f"{prod.tonnes:,.0f} t", pg,
                               metal(prod.metal))
            self.card_rem.set(f"{rem.tonnes:,.0f} t", tr("{g:,.2f} {unit}  ·  {pct:.1%} of total", g=rem.grade,
                                                         unit=unit, pct=1 - pct),
                              metal(rem.metal))
        else:
            for c in (self.card_total, self.card_prod, self.card_rem):
                c.set("-")

        # shapes table
        shapes = r.log.sorted_shapes() if r.log else []
        colors = {}
        if mgr.layer is not None:
            colors = mgr.layer.month_colors()
        keep = self._selected_id()
        t = self.t_cuts
        t.setRowCount(len(shapes))
        self.cuts_page.setCurrentIndex(1 if shapes else 0)
        as_of = mgr.as_of_day
        for row, sh in enumerate(shapes):
            tot, blocks = r.dep.shape_totals(sh.id) if ready else (None, 0)
            later = sh.day > as_of
            cells = [_item(sh.date, Qt.AlignLeft | Qt.AlignVCenter, sh.id),
                     _item(sh.name, Qt.AlignLeft | Qt.AlignVCenter),
                     _item(tr(PLANE_SHORT[sh.plane]) if sh.plane in PLANE_SHORT else sh.plane,
                           Qt.AlignLeft | Qt.AlignVCenter),
                     _item(f"{sh.from_:,.1f}"), _item(f"{sh.to:,.1f}"), _item(f"{sh.depth:,.1f}"),
                     _item(f"{tot.tonnes:,.0f}" if tot else "-"),
                     _item(f"{tot.grade:,.2f}" if tot else "-"),
                     _item(_metal_short(tot.metal, unit) if tot else "-"),
                     _item(f"{blocks:,}" if tot else "-")]
            rgb = colors.get(month_of(sh.date))
            if rgb is not None:
                cells[0].setIcon(icon("square-rounded", QColor.fromRgbF(*rgb).name()))
            if later:
                for c in cells:
                    c.setForeground(QColor(theme.MUTED))
                    c.setToolTip(tr("After the 'as of' date - not counted"))
            for col, c in enumerate(cells):
                t.setItem(row, col, c)
        if keep is not None:
            self.select(keep)

        # monthly table
        rows = monthly(r.log, r.dep) if ready and r.log else []
        t = self.t_month
        t.setRowCount(len(rows))
        for i, m in enumerate(rows):
            cells = [_item(month_label(m["month"]), Qt.AlignLeft | Qt.AlignVCenter), _item(f"{m['shapes']}"),
                     _item(f"{m['tonnes']:,.0f}"), _item(f"{m['grade']:,.2f}"),
                     _item(_metal_short(m["metal"], unit)), _item(f"{m['cumulative']:,.0f}"),
                     _item(f"{m['remaining']:,.0f}"), _item(_metal_short(m["remaining_metal"], unit))]
            for col, c in enumerate(cells):
                t.setItem(i, col, c)
        mu = "t" if unit == "%" else "kg"
        _set_header(self.t_cuts, 8, tr("Metal ({unit})", unit=mu), f"Metal ({mu})")
        _set_header(self.t_cuts, 7, tr("Grade ({unit})", unit=unit), f"Grade ({unit})")
        _set_header(self.t_month, 3, tr("Grade ({unit})", unit=unit), f"Grade ({unit})")
        _set_header(self.t_month, 4, tr("Metal ({unit})", unit=mu), f"Metal ({mu})")
        _set_header(self.t_month, 7, tr("Remaining metal ({unit})", unit=mu), f"Remaining metal ({mu})")

        # view controls
        self.b_shapes.setChecked(mgr.layer is not None and mgr.layer.visible and mgr.show_shapes)
        self.b_shapes.setEnabled(bool(shapes))
        self.b_draw.setEnabled(r.log is not None)
        self.b_solid.setEnabled(r.log is not None)

    @staticmethod
    def _combo(c: QComboBox, items: list[str], current: str, special=()) -> None:
        """``items`` are the values (item data); those in ``special`` are shown translated."""
        c.blockSignals(True)
        c.clear()
        for it in items:
            c.addItem(tr(it) if it in special else it, it)
        if current in items:
            c.setCurrentIndex(items.index(current))
        c.blockSignals(False)

    def _selected_id(self) -> int | None:
        rows = self.t_cuts.selectionModel().selectedRows()
        if not rows:
            return None
        it = self.t_cuts.item(rows[0].row(), 0)
        return it.data(Qt.UserRole) if it is not None else None

    def select(self, sid: int) -> None:
        for row in range(self.t_cuts.rowCount()):
            it = self.t_cuts.item(row, 0)
            if it is not None and it.data(Qt.UserRole) == sid:
                self.tabs.setCurrentIndex(0)
                self.t_cuts.selectRow(row)
                self.t_cuts.scrollToItem(it)
                return

    def _update_buttons(self) -> None:
        on = self._selected_id() is not None and self.tabs.currentIndex() == 0
        for b in (self.b_edit, self.b_del, self.b_zoom):
            b.setEnabled(on)

    # --- actions ---------------------------------------------------------------------------
    def _setup(self, key, value) -> None:
        if not self._filling:
            self.mgr.set_setup(**{key: value})

    def _as_of_changed(self, d: QDate) -> None:
        if not self._filling:
            self.mgr.set_as_of(d.toString("yyyy-MM-dd"))

    def _draw(self) -> None:
        win = self.mgr.win
        win.start_mined_outline()
        park = getattr(win, "parking", None)
        if park is not None and win.a_mined.isChecked():
            park.park("reserve")                    # out of the way of the view; the Reserve button brings it back

    def _edit(self) -> None:
        sid = self._selected_id()
        if sid is not None:
            self.mgr.edit_shape(sid)

    def _delete(self) -> None:
        sid = self._selected_id()
        sh = self.mgr.reserve.log.get(sid) if sid is not None else None
        if sh is None:
            return
        ans = QMessageBox.question(self, tr("Delete mined shape"),
                                   tr("Delete {name} ({date})?\n\nIts tonnes go back to the remaining reserve.",
                                      name=sh.name, date=sh.date))
        if ans == QMessageBox.Yes:
            self.mgr.delete_shape(sid)

    def _zoom(self) -> None:
        sid = self._selected_id()
        if sid is not None:
            self.mgr.zoom_to(sid)

    def _export(self) -> None:
        t = self.t_cuts if self.tabs.currentIndex() == 0 else self.t_month
        name = "production_shapes.csv" if t is self.t_cuts else "production_monthly.csv"
        start = str((self.mgr.folder or Path.cwd()) / name)
        path, _ = QFileDialog.getSaveFileName(self, tr("Export CSV"), start, tr("CSV files (*.csv)"))
        if not path:
            return
        try:
            write_table_csv(t, path)
        except OSError as e:
            QMessageBox.warning(self, tr("Export CSV"), tr("Could not write {path}:\n{error}", path=path, error=e))
            return
        self.mgr.win.status_msg.setText(tr("Saved {path}", path=path))


def _set_header(t: QTableWidget, col: int, shown: str, english: str) -> None:
    """A column title: ``shown`` in the current language, ``english`` kept for the CSV export."""
    it = t.horizontalHeaderItem(col)
    it.setText(shown)
    it.setData(Qt.UserRole, english)


def _header_name(t: QTableWidget, col: int) -> str:
    it = t.horizontalHeaderItem(col)
    english = it.data(Qt.UserRole)
    return english if isinstance(english, str) and english else it.text()


def write_table_csv(t: QTableWidget, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([_header_name(t, c) for c in range(t.columnCount())])
        for r in range(t.rowCount()):
            w.writerow([_plain(t.item(r, c).text() if t.item(r, c) else "") for c in range(t.columnCount())])


def _plain(text: str) -> str:
    """Numbers without the thousands separators; other text as shown."""
    bare = text.replace(",", "")
    try:
        float(bare)
        return bare
    except ValueError:
        return text

