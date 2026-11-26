"""Main window: layers + properties on the left, 3D view on the right, floating toolbar."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import vtk
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .. import __version__
from ..i18n import tr
from ..io import database
from ..scene.layers import Layer
from ..scene.loaders import FILE_FILTER, load_paths
from ..scene.scene import VIEW_3D, VIEW_XY, VIEW_XZ, VIEW_YZ, Scene
from ..workspace import (WORKSPACE_FILE, Workspace, _from_text, _key, _to_text, changed_layers, layer_spec, load_spec,
                         reload_layers, split_block_csvs)
from .floating_toolbar import FloatingButton, FloatingToolbar, OverlayController
from .icons import app_icon, icon, text_icon
from .identify import IdentifyTool
from .layer_tree import LayerTree
from .measure import MeasureTool
from .section_points import SectionPointsDialog
from .section_tool import SectionTool
from .mined_shape_tool import MinedShapeTool
from .drillhole_manager import DrillholeManager
from .survey_manager import SurveyManager
from .window_parking import WindowParking
from .reserve_window import ReserveManager
from .properties_panel import PropertiesPanel
from .settings_dialog import AppSettings, SettingsDialog

APP_NAME = "DrillFailPredict"
APP_TITLE = "Drilling Failure Prediction & Prevention"
AUTOSAVE_POLL_MS = 2000      # retry of a save asked while files still load
STATUS_WARNING_MS = 30000    # how long a status bar warning stays
log = logging.getLogger(__name__)


def default_workspace_folder() -> Path:
    """Used when opened files lie in a folder where the workspace file cannot be written."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / APP_NAME / "workspace"


class _Worker(QObject):
    """Runs one function in a worker thread and hands back (job, result or exception)."""

    finished = Signal(object, object)

    def __init__(self, job: dict, fn):
        super().__init__()
        self.job = job
        self.fn = fn

    @Slot()
    def run(self):
        try:
            result = self.fn()
        except Exception as e:  # noqa: BLE001 - delivered to the GUI thread and reported there
            log.exception("background job %s failed", self.job.get("kind"))
            result = e
        self.finished.emit(self.job, result)


def _notes_text(notes: list[str]) -> str:
    """What the database set-up did, short: "made 6 tables; moved the production records ..."."""
    tables = [n for n in notes if n.startswith("made table ")]
    if len(tables) <= 1:
        return "; ".join(notes)
    rest = [n for n in notes if n not in tables]
    return "; ".join([tr("made {n} tables", n=len(tables))] + rest)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(app_icon())                     # title bar, taskbar, Alt+Tab
        self.resize(1500, 900)
        self.setAcceptDrops(True)
        self.settings = QSettings(APP_NAME, APP_NAME)
        self.app_settings = AppSettings.load(self.settings)
        database.configure(self.app_settings.database)       # SQLite file or MySQL server
        self._threads: list[tuple[QThread, _Worker]] = []
        self._was_maximized = False
        self._fake_fullscreen = False          # Windows full screen done by the app (see toggle_fullscreen)
        self.loads_pending = 0            # background jobs not yet finished
        self.load_errors: list[str] = []  # all load errors this session
        self.quiet_errors = False         # True: collect errors without a message box (smoke test)
        self.workspace: Workspace | None = None
        self._generation = 0              # bumped when the scene is cleared; stale results are dropped
        self._refresh_running = False
        self._open_after_load: list[str] = []   # files to open once the workspace being opened is shown
        self._import_queue: list[Path] = []     # block model CSVs waiting for their import (.gvbm)
        self._importing = False
        self._autosave_failed = False

        # --- layout -----------------------------------------------------------
        central = QWidget(self)
        central.setObjectName("central")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        self.hsplit = QSplitter(Qt.Horizontal, central)
        root.addWidget(self.hsplit)

        self.left = QWidget()
        self.left.setObjectName("leftPanel")
        lv = QVBoxLayout(self.left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.vsplit = QSplitter(Qt.Vertical, self.left)
        self.layer_tree = LayerTree()
        self.props = PropertiesPanel(on_changed=self._after_property_change)
        self.vsplit.addWidget(self.layer_tree)
        self.vsplit.addWidget(self.props)
        self.vsplit.setSizes([380, 520])
        lv.addWidget(self.vsplit)

        self.view_frame = QFrame()
        vf = QVBoxLayout(self.view_frame)
        vf.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.view_frame, multi_samples=8, auto_update=False)
        vf.addWidget(self.plotter.interactor)

        self.hsplit.addWidget(self.left)
        self.hsplit.addWidget(self.view_frame)
        self.hsplit.setStretchFactor(1, 1)
        self.hsplit.setSizes([300, 1200])
        self.setCentralWidget(central)

        self.status_msg = QLabel(tr("Ready"))
        self.status_refresh = QLabel("")
        self.status_xyz = QLabel("")
        self.status_xyz.setMinimumWidth(320)
        self.status_warn = QLabel("")           # e.g. a -w folder that is not a workspace
        self.status_warn.setStyleSheet("color: #f0a030;")
        self.status_warn.hide()
        self._warn_timer = QTimer(self, singleShot=True, interval=STATUS_WARNING_MS)
        self._warn_timer.timeout.connect(self.clear_status_warning)
        self.statusBar().addWidget(self.status_warn)
        self.statusBar().addWidget(self.status_msg, 1)
        self.status_db = QLabel("")             # which database, connected or not
        self.statusBar().addPermanentWidget(self.status_db)
        self.statusBar().addPermanentWidget(self.status_refresh)
        self.statusBar().addPermanentWidget(self.status_xyz)
        self.statusBar().setSizeGripEnabled(False)

        # --- scene ---------------------------------------------------------------
        self.scene = Scene(self.plotter)
        self._picker = vtk.vtkWorldPointPicker()
        self._mouse_obs_tag = self.plotter.iren.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.identify = IdentifyTool(self, self.scene, self.plotter)
        self.identify.status.connect(self.status_msg.setText)
        self.identify.identified.connect(self._on_identified)
        self.hole_dialog = None                    # DrillholeLogDialog, created on first use
        self.measure = MeasureTool(self, self.scene, self.plotter)
        self.measure.status.connect(self.status_msg.setText)
        self.section = SectionTool(self, self.scene, self.plotter)
        self.section.status.connect(self.status_msg.setText)
        self.section.edit_requested.connect(self._edit_section_points)
        self.section.finished.connect(self._show_section)
        self.section.drawing_changed.connect(self._section_drawing)
        self.section_dialog = None                 # SectionDialog, created on first use
        self._section_legend = True                # rock legend on the exported section page
        self._section_timer = QTimer(self)         # re-cut the open section after layer changes
        self._section_timer.setSingleShot(True)
        self._section_timer.setInterval(300)
        self._section_timer.timeout.connect(self._recut_section)
        # reserve: mined outlines (drawn like a section line, but closed) and the Reserve window
        self.mining = MinedShapeTool(self, self.scene, self.plotter)
        self.mining.status.connect(self.status_msg.setText)
        self.mining.outline_done.connect(self._outline_done)
        self.mining.drawing_changed.connect(self._section_drawing)
        self.reserve = ReserveManager(self)
        self.drillhole_manager: DrillholeManager | None = None
        self.survey_manager: SurveyManager | None = None
        self.pit_geology = None                  # the Pit Geology window (ui/pit_geology.py)
        # drilling risk (risk/*, ui/history_dialog.py, ui/prediction_dialog.py, ui/risk_admin.py)
        self.slice_dock = None                   # the 3D slice panel, made on first use
        self._pending_roles: dict[str, str] = {}  # file key -> geological role for layers being loaded
        self.risk_engine = None                  # risk.engine.RiskEngine of the open workspace
        self.records_dialog = None
        self.prediction_dialog = None
        self.risk_layer = None                   # scene.risk_layer.RiskLayer
        # tool windows hide (keeping what they show) instead of minimising; their toolbar button brings them back
        self.parking = WindowParking(self)
        self.db_error = ""                       # why the project database cannot be used ("" = connected)
        self._db_warned = ""                     # the error last shown in a warning (not repeated by Refresh)

        # --- panels wiring -----------------------------------------------------
        self.layer_tree.layer_selected.connect(self.props.set_layer)
        self.layer_tree.layer_selected.connect(self._on_layer_selected)
        self.layer_tree.visibility_changed.connect(self._set_layer_visible)
        self.layer_tree.remove_requested.connect(self.remove_layer)
        self.layer_tree.zoom_requested.connect(self.zoom_to_layer)
        self.layer_tree.rename_requested.connect(self.rename_layer)
        self.layer_tree.open_requested.connect(self.open_dialog)
        self.layer_tree.order_changed.connect(self._layer_order_changed)
        self.layer_tree.pit_selected.connect(self.props.set_pit)
        self.layer_tree.pit_selected.connect(lambda layer, _pid: self._on_layer_selected(layer))
        self.layer_tree.pit_visibility_changed.connect(lambda layer, pid, on: self.set_pit_style(layer, pid,
                                                                                                visible=on))
        self.layer_tree.pit_zoom_requested.connect(self.zoom_to_pit)
        self.layer_tree.pit_geology_requested.connect(lambda _layer, pid: self.open_pit_geology(pid))
        self.props.on_pit_style = self.set_pit_style
        self.layer_tree.design_selected.connect(self.props.set_design)
        self.layer_tree.design_selected.connect(lambda layer, _did: self._on_layer_selected(layer))
        self.layer_tree.design_visibility_changed.connect(
            lambda layer, did, on: self.set_design_style(layer, did, visible=on))
        self.layer_tree.designs_visibility_changed.connect(self.set_designs_shown)
        self.layer_tree.design_zoom_requested.connect(self.zoom_to_design)
        self.layer_tree.design_rename_requested.connect(self.rename_design)
        self.layer_tree.design_delete_requested.connect(self.delete_design)
        self.props.on_design_style = self.set_design_style
        self.layer_tree.designs_selected.connect(self.props.set_designs)
        self.layer_tree.designs_selected.connect(self._on_layer_selected)
        self.props.on_designs_shown = self.set_designs_shown

        # --- actions, toolbar, shortcuts -----------------------------------------
        self._build_actions()
        self.toolbar = FloatingToolbar(self, self.plotter.interactor)
        self.fab = FloatingButton(self, self.plotter.interactor, tr("Show toolbar (Ctrl+T)"))
        self._fill_toolbar()
        self.overlay = OverlayController(self, self.toolbar, self.fab)

        esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        esc.setContext(Qt.ApplicationShortcut)
        esc.activated.connect(self.on_escape)
        # While a section line / mined outline is drawn: Enter finishes it, Backspace removes
        # the last point.
        self._section_keys = []
        for key, slot in ((Qt.Key_Return, self._finish_drawing), (Qt.Key_Enter, self._finish_drawing),
                          (Qt.Key_Backspace, self._undo_drawing)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(slot)
            sc.setEnabled(False)
            self._section_keys.append(sc)
        delete = QShortcut(QKeySequence(Qt.Key_Delete), self.layer_tree.tree)
        delete.activated.connect(lambda: self.layer_tree.current_layer()
                                 and self.remove_layer(self.layer_tree.current_layer()))

        # --- automatic refresh -------------------------------------------------------
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(lambda: self.refresh(manual=False))
        self._apply_app_settings()
        cfg = self.app_settings.database
        self.status_db.setText("● " + (tr("MySQL {db}: connecting ...", db=cfg.database) if cfg.is_mysql
                                       else "SQLite"))
        self.status_db.setStyleSheet("color: #9aa3ad;")
        QTimer.singleShot(0, self._startup_connect)     # after a last workspace was asked to open

        # --- workspace saving: when the app closes (and when another workspace is opened, or
        # Ctrl+S) - not after every change (user request, 2026-11-23) ---------------------------
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_poll = QTimer(self)          # no longer started: no periodic saving
        self._update_title()

    # ------------------------------------------------------------------------
    def _act(self, name, tip, slot, icon_obj, shortcut=None, checkable=False, enabled=True):
        a = QAction(icon_obj, name, self)
        a.setToolTip(f"{tip}  ({shortcut})" if shortcut else tip)
        a.setCheckable(checkable)
        a.setEnabled(enabled)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.ApplicationShortcut)
        if slot is not None:
            a.triggered.connect(slot)
        self.addAction(a)  # shortcuts work without a menu bar
        return a

    def _build_actions(self):
        A = self._act
        self.a_open = A(tr("Open files"), tr("Open files"), self.open_dialog, icon("file-plus-outline"), "Ctrl+O")
        self.a_open_ws = A(tr("Open workspace"), tr("Open a workspace folder"), self.open_workspace_dialog,
                           icon("folder-open-outline"), "Ctrl+Shift+O")
        # Not on the toolbar: the workspace is saved when the app closes. Ctrl+S saves now.
        self.a_save_ws = A(tr("Save workspace"), tr("Save the workspace now"),
                           lambda: self._autosave(force=True), icon("content-save-outline"), "Ctrl+S")
        self.a_save_ws_as = A(tr("Save workspace as"), tr("Save the workspace in another folder"),
                              self.save_workspace_as, icon("content-save-move-outline"), "Ctrl+Shift+S")
        self.a_refresh = A(tr("Refresh"), tr("Reload layers whose files changed"), lambda: self.refresh(manual=True),
                           icon("refresh"), "F5")
        self.a_export_map = A(tr("Export map"), tr("Export map: the current view (2D to scale, or 3D) as a page - "
                                                   "a 300 dpi PNG image"),
                              self.open_map_export, icon("image-move"), "Ctrl+Shift+P")
        self.a_license = A(tr("License"), tr("License: machine code and license key"), self.open_license_dialog,
                           icon("certificate-outline"))
        self.a_settings = A(tr("Settings"), tr("Settings (auto refresh, start-up)"), self.open_settings,
                            icon("cog-outline"), "Ctrl+,")

        self.a_rotate = A(tr("Rotate"), tr("Rotate mode: left-drag rotates"), lambda: self._set_nav("rotate"),
                          icon("rotate-3d-variant"), "Ctrl+R", checkable=True)
        self.a_pan = A(tr("Pan"), tr("Pan mode: left-drag pans"), lambda: self._set_nav("pan"),
                       icon("pan"), "Ctrl+P", checkable=True)
        nav = QActionGroup(self)
        nav.addAction(self.a_rotate)
        nav.addAction(self.a_pan)
        self.a_rotate.setChecked(True)

        self.a_zoom_in = A(tr("Zoom in"), tr("Zoom in"), lambda: self.scene.zoom(1.25),
                           icon("magnify-plus-outline"), "Ctrl+=")
        self.a_zoom_out = A(tr("Zoom out"), tr("Zoom out"), lambda: self.scene.zoom(0.8),
                            icon("magnify-minus-outline"), "Ctrl+-")
        self.a_fit = A(tr("Home"),
                       tr("Home: zoom to all visible data\n(in 3D also back to the default view direction)"),
                       self.home, icon("fit-to-screen-outline"), "Home")

        self.a_view3d = A("3D", tr("3D view"), lambda: self.set_view(VIEW_3D), text_icon("3D"), "Ctrl+1", True)
        self.a_xy = A("XY", tr("2D plan view (XY)"), lambda: self.set_view(VIEW_XY), text_icon("XY"), "Ctrl+2", True)
        self.a_xz = A("XZ", tr("2D section looking north (XZ)"), lambda: self.set_view(VIEW_XZ),
                      text_icon("XZ"), "Ctrl+3", True)
        self.a_yz = A("YZ", tr("2D section looking west (YZ)"), lambda: self.set_view(VIEW_YZ),
                      text_icon("YZ"), "Ctrl+4", True)
        views = QActionGroup(self)
        for a in (self.a_view3d, self.a_xy, self.a_xz, self.a_yz):
            views.addAction(a)
        self.a_view3d.setChecked(True)
        self._view_actions = {VIEW_3D: self.a_view3d, VIEW_XY: self.a_xy, VIEW_XZ: self.a_xz, VIEW_YZ: self.a_yz}

        self.a_persp = A(tr("Perspective"),
                         tr("Perspective / orthographic projection of the 3D view\n"
                            "(on: perspective, off: orthographic; 2D views are always orthographic)"),
                         None, icon("perspective-less"), "Ctrl+5", True)
        self.a_persp.toggled.connect(self._toggle_perspective)
        self.a_grid = A(tr("Grid"), tr("Show / hide map grid"), None, icon("grid"), "Ctrl+G", True)
        self.a_grid.toggled.connect(self._toggle_grid)

        self.a_identify = A(tr("Identify"), tr("Identify: click an object to see its information"),
                            None, icon("information-outline"), "Ctrl+I", checkable=True)
        self.a_identify.toggled.connect(lambda on: self._tool_toggled(self.identify, on))
        self.a_identify.toggled.connect(lambda on: on and self.parking.restore("identify"))
        self.a_measure = A(tr("Measure"), tr("Measure: click two points for distance, bearing and inclination"),
                           None, icon("ruler"), "Ctrl+M", checkable=True)
        self.a_measure.toggled.connect(lambda on: self._tool_toggled(self.measure, on))
        self.a_section = A(tr("Cross-section"), tr("Cross-section (2D views): click points along the section line, "
                                                   "right-click or Enter to finish"),
                           None, icon("box-cutter"), "Ctrl+K",
                           checkable=True, enabled=False)
        self.a_section.toggled.connect(lambda on: self._tool_toggled(self.section, on))
        self.a_section.toggled.connect(lambda on: on and self.parking.restore("section"))
        self.a_mined = A(tr("Mined outline"), tr("Mined outline (2D views): click the corners of the mined area, "
                                                 "right-click or Enter to close it, then enter the date and From / To"),
                         None, icon("vector-polygon"), "Ctrl+L", checkable=True, enabled=False)
        self.a_mined.toggled.connect(self._mined_toggled)
        self.a_reserve = A(tr("Reserve"), tr("Reserve: total, produced and remaining ore; the mined shapes"),
                           lambda: self.reserve.show_window(), icon("pickaxe"), "Ctrl+E")
        self.a_drillholes = A(tr("Drillholes"), tr("Drillhole Manager: add / remove holes, edit collar, survey, "
                                                   "geology and assay, rock colours"), self.open_drillhole_manager,
                              icon("database-edit-outline"), "Ctrl+D")
        self.a_blocks = A(tr("Block models"), tr("Block Model Manager: the block models of the workspace - new block model inside a solid, Surpac ASCII import, attribute statistics"),
                          self.open_block_manager, icon("cube-outline"), "Ctrl+B")
        self.a_survey = A(tr("Pit Manager"), tr("Pit Manager: main pits and branches, traverse setups (total "
                                                "station or compass), closure check, designed pits; the drives are "
                                                "drawn from it"),
                          self.open_survey_manager,
                          icon("map-marker-path"), "Ctrl+U")
        self.a_pit_geology = A(tr("Pit geology"), tr("Pit Geology: the rocks, structures (dip direction / dip) and "
                                                     "samples mapped in the pits, by chainage from the pit's start; "
                                                     "the wall sketch"),
                               lambda: self.open_pit_geology(), icon("layers-outline"), "Ctrl+J")

        # --- drilling risk ---
        self.a_records = A(tr("Drilling records"), tr("Drilling Records: the success / failure database - collars, "
                                                      "surveys, geology, runs, events, failures, formation cards, rock "
                                                      "properties"), self.open_drilling_records,
                           icon("database-cog-outline"), "Ctrl+H")
        self.a_predict = A(tr("Risk prediction"), tr("Risk Prediction: failure probability ahead of the bit of an ongoing "
                                                     "or planned hole, warning zones and measures"),
                           lambda: self.open_risk_prediction(), icon("alert-decagram-outline"), "Ctrl+Shift+R")
        self.a_plan = A(tr("Plan check"), tr("Plan check: the safety of a designed hole from its collar X, Y, Z, "
                                             "azimuth, dip and depth"), lambda: self.open_risk_prediction(plan=True),
                        icon("compass-outline"))
        self.a_geomodel = A(tr("Geological model"), tr("Geological Model: roles of the surface layers (horizons, faults, "
                                                       "formations), unit names, error band ±u"), self.open_geomodel,
                            icon("layers-triple-outline"), "Ctrl+Shift+G")
        self.a_import_geo = A(tr("Import geological model"), tr("Import geological model surfaces / solids (Surpac DTM, "
                                                                "DXF, GOCAD) and give them a role"),
                              lambda: self.import_with_role(""), icon("layers-plus"))
        self.a_import_fault = A(tr("Import fault surfaces"), tr("Import fault surfaces (Surpac DTM, DXF, GOCAD)"),
                                lambda: self.import_with_role("fault"), icon("vector-line"))
        self.a_slice = A(tr("3D slice"), tr("3D slice: the 3D view shows only the objects between two planes"), None,
                         icon("box-shadow"), "Ctrl+Shift+L", checkable=True)
        self.a_slice.toggled.connect(self._toggle_slice)

        self.a_fullscreen = A(tr("Full screen"), tr("Full screen view (Esc to exit)"), self.toggle_fullscreen,
                              icon("fullscreen"), "F11", True)
        self.a_hide_bar = A(tr("Hide toolbar"), tr("Hide toolbar"), lambda: self.overlay.hide_toolbar(),
                            icon("chevron-double-right"))
        self.a_toggle_bar = A(tr("Toggle toolbar"), tr("Show / hide toolbar"), lambda: self.overlay.toggle(),
                              icon("tools"), "Ctrl+T")

    def _fill_toolbar(self):
        t = self.toolbar
        for a in (self.a_open_ws, self.a_open, self.a_refresh, self.a_export_map):
            t.add_action(a)
        t.add_separator()
        for a in (self.a_rotate, self.a_pan, self.a_zoom_in, self.a_zoom_out, self.a_fit):
            t.add_action(a)
        t.add_separator()
        for a in (self.a_view3d, self.a_xy, self.a_xz, self.a_yz, self.a_persp, self.a_grid):
            t.add_action(a)
        t.add_separator()
        for a in (self.a_identify, self.a_measure, self.a_section):
            t.add_action(a)
        t.add_separator()
        # Reserve (Ctrl+E) and Mined outline (Ctrl+L) stay available by shortcut; not shown on the toolbar
        for a in (self.a_drillholes, self.a_blocks, self.a_survey, self.a_pit_geology):
            t.add_action(a)
        t.add_separator()
        for a in (self.a_records, self.a_predict, self.a_plan, self.a_geomodel, self.a_import_geo, self.a_import_fault,
                  self.a_slice):
            t.add_action(a)
        t.add_separator()
        t.add_action(self.a_fullscreen)
        t.add_action(self.a_settings)
        t.add_action(self.a_hide_bar)

    # --- navigation / views -----------------------------------------------------
    def _set_nav(self, mode: str):
        self.scene.set_navigation(mode)

    def set_view(self, view: str):
        self.scene.set_view(view)
        self._view_actions[view].setChecked(True)
        two_d = view != VIEW_3D
        self.a_rotate.setEnabled(not two_d)
        # 2D views pan (they cannot be rotated); the 3D view always starts in rotate mode
        (self.a_pan if two_d else self.a_rotate).setChecked(True)
        self.scene.set_navigation("pan" if two_d else "rotate")
        self.a_section.setEnabled(two_d)
        self.a_mined.setEnabled(two_d and not self.db_error)
        self.a_persp.setEnabled(not two_d)
        if not two_d and self.a_section.isChecked():
            self.a_section.setChecked(False)
        if not two_d and self.a_mined.isChecked():
            self.a_mined.setChecked(False)
        self.section.view_changed()
        self.mining.view_changed()
        self.status_msg.setText({VIEW_3D: tr("3D view"), VIEW_XY: tr("Plan view (XY)"),
                                 VIEW_XZ: tr("Section view looking north (XZ)"),
                                 VIEW_YZ: tr("Section view looking west (YZ)")}[view])
        self._schedule_autosave()

    def home(self):
        """3D: back to the default view (from the south-west, looking down) showing all data,
        however the view was rotated. 2D: zoom to all data (the direction is fixed there)."""
        if self.scene.view == VIEW_3D:
            self.scene.set_view(VIEW_3D)
            self.a_rotate.setChecked(True)               # back to rotate mode, as when 3D is chosen
            self.scene.set_navigation("rotate")
            self.status_msg.setText(tr("3D view reset"))
        else:
            self.scene.fit()
        self._schedule_autosave()

    def _toggle_perspective(self, on: bool):
        if on == self.scene.perspective:
            return
        self.scene.set_perspective(on)
        self.a_persp.setIcon(icon("perspective-more" if on else "perspective-less"))
        self.status_msg.setText(tr("3D view: perspective") if on else tr("3D view: orthographic"))
        self._schedule_autosave()

    def _sync_perspective_action(self) -> None:
        """Show the scene's projection on the button (after a workspace sets it)."""
        on = self.scene.perspective
        self.a_persp.blockSignals(True)
        self.a_persp.setChecked(on)
        self.a_persp.blockSignals(False)
        self.a_persp.setIcon(icon("perspective-more" if on else "perspective-less"))

    def _toggle_grid(self, on: bool):
        self.scene.set_grid_visible(on)
        self._schedule_autosave()

    # --- full screen -------------------------------------------------------------
    # Windows: a window that exactly covers the monitor gets the exclusive full-screen mode
    # (OpenGL / driver "fullscreen optimisation"), in which the app's other top-level windows
    # - the floating toolbar, its round button, the Identify / Measure panels - are not drawn,
    # at the latest from the second time on. So on Windows full screen is done here: no title
    # bar or frame, the monitor minus one pixel row at the bottom. Elsewhere: showFullScreen().
    def is_full_screen(self) -> bool:
        return self._fake_fullscreen or self.isFullScreen()

    def toggle_fullscreen(self, on: bool | None = None):
        if on is None:
            on = not self.is_full_screen()
        if on and not self.is_full_screen():
            self._was_maximized = self.isMaximized()
            self._toolbar_before_fs = self.overlay.toolbar_shown
            self.left.hide()
            self.statusBar().hide()
            if not (os.name == "nt" and self._enter_windowed_fullscreen()):
                self.showFullScreen()
        elif not on and self.is_full_screen():
            self.left.show()
            self.statusBar().show()
            if self._fake_fullscreen:
                self._leave_windowed_fullscreen()
            elif self._was_maximized:
                self.showMaximized()
            else:
                self.showNormal()
        fs = self.is_full_screen()
        self.a_fullscreen.setChecked(fs)
        self.a_fullscreen.setIcon(icon("fullscreen-exit" if fs else "fullscreen"))
        QTimer.singleShot(50, self._overlays_after_fullscreen)

    def _overlays_after_fullscreen(self) -> None:
        """Full screen: the toolbar makes way for the round tools button (a click on it brings
        the toolbar back). Leaving full screen: the toolbar returns if it was shown before."""
        ov = self.overlay
        ov.reposition()
        if self.is_full_screen():
            if ov.toolbar_shown and self.toolbar.isVisible():
                ov.hide_toolbar()                    # fades out, then the round button fades in
            else:
                ov.toolbar_shown = False
                self.toolbar.hide()
                self.fab.reposition()
                self.fab.show()
                self.fab.raise_()
        elif getattr(self, "_toolbar_before_fs", False) and not ov.toolbar_shown:
            ov.show_toolbar()
        else:
            w = self.toolbar if ov.toolbar_shown else self.fab
            w.show()
            w.raise_()

    @staticmethod
    def _user32():
        import ctypes

        u = ctypes.windll.user32
        u.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        u.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        u.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
        u.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        u.MonitorFromWindow.restype = ctypes.c_void_p          # a 64-bit handle (not int)
        u.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        u.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        return u

    @staticmethod
    def _monitor_rects(u, mon):
        """(whole display, work area) of a display handle, as wintypes.RECT."""
        import ctypes
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        u.GetMonitorInfoW(ctypes.c_void_p(mon), ctypes.byref(mi))
        return mi.rcMonitor, mi.rcWork

    def _enter_windowed_fullscreen(self) -> bool:
        """Full screen on the display the window is on now (with several displays: only that
        one), noted before a maximized window is restored - its normal place may be on another
        display, e.g. after it was dragged there maximized."""
        try:
            import ctypes

            u = self._user32()
            hwnd = ctypes.c_void_p(int(self.winId()))
            mon = u.MonitorFromWindow(hwnd, 2)             # the display the window is on (nearest)
            self._fs_monitor = mon
            if self.isMaximized():
                self.showNormal()               # a maximized window would snap back to the work area
                QApplication.processEvents()
            self._fs_saved_geometry = self.geometry()
            GWL_STYLE = -16
            style = u.GetWindowLongPtrW(hwnd, GWL_STYLE)
            self._fs_saved_style = style
            frame = 0x00C00000 | 0x00040000 | 0x00080000 | 0x00020000 | 0x00010000  # caption, sizing frame, menus
            u.SetWindowLongPtrW(hwnd, GWL_STYLE, style & ~frame)

            r, _work = self._monitor_rects(u, mon)        # that display only
            SWP = 0x0004 | 0x0020 | 0x0040        # no z-order change, frame changed, show
            u.SetWindowPos(hwnd, None, r.left, r.top, r.right - r.left, r.bottom - r.top - 1, SWP)
            self._fake_fullscreen = True
            return True
        except Exception:  # noqa: BLE001 - fall back to Qt's full screen
            log.exception("windowed full screen failed")
            return False

    def _leave_windowed_fullscreen(self) -> None:
        self._fake_fullscreen = False
        try:
            import ctypes

            u = self._user32()
            hwnd = ctypes.c_void_p(int(self.winId()))
            u.SetWindowLongPtrW(hwnd, -16, self._fs_saved_style)
            g = self._fs_saved_geometry
            u.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0020)  # frame changed only
            self.setGeometry(g)
            mon = getattr(self, "_fs_monitor", None)
            if mon and u.MonitorFromWindow(hwnd, 2) != mon:
                # the normal place is on another display: bring the window back onto the display
                # it was full screen on, so it is maximized / shown there again
                from ctypes import wintypes

                _r, w = self._monitor_rects(u, mon)
                cur = wintypes.RECT()
                u.GetWindowRect(hwnd, ctypes.byref(cur))             # physical pixels, as the displays
                ww = min(cur.right - cur.left, w.right - w.left)
                wh = min(cur.bottom - cur.top, w.bottom - w.top)
                u.SetWindowPos(hwnd, None, w.left + (w.right - w.left - ww) // 2, w.top + (w.bottom - w.top - wh) // 2,
                               ww, wh, 0x0004)
        except Exception:  # noqa: BLE001
            log.exception("leaving windowed full screen failed")
        if self._was_maximized:
            self.showMaximized()
        else:
            self.showNormal()

    def _tools(self):
        return ((self.identify, self.a_identify), (self.measure, self.a_measure),
                (self.section, self.a_section), (self.mining, self.a_mined))

    def _tool_toggled(self, tool, on: bool) -> None:
        """Identify, Measure and Cross-section all use clicks in the view: one at a time."""
        if on:
            for other, action in self._tools():
                if other is not tool and action.isChecked():
                    action.setChecked(False)
        tool.set_active(on)
        if not on and not any(action.isChecked() for _t, action in self._tools()):
            self.status_msg.setText(tr("Ready"))

    # --- cross-section -------------------------------------------------------------
    def _section_drawing(self, _on: bool = False) -> None:
        on = self.section.drawing or self.mining.drawing
        for sc in self._section_keys:
            sc.setEnabled(on)

    def _drawing_tool(self):
        return self.mining if self.mining.drawing else self.section

    def _finish_drawing(self) -> None:
        self._drawing_tool().finish()

    def _undo_drawing(self) -> None:
        self._drawing_tool().undo()

    # --- reserve: mined outlines ------------------------------------------------------
    def start_mined_outline(self) -> None:
        """From the Reserve window: switch to the plan view when in 3D and start drawing."""
        if not self.reserve.can_draw():
            return
        if self.scene.view == VIEW_3D:
            self.set_view(VIEW_XY)
        self.a_mined.setChecked(True)
        self.status_msg.setText(tr(MinedShapeTool.HINT))
        self.activateWindow()
        self.plotter.interactor.setFocus()

    def _mined_toggled(self, on: bool) -> None:
        if on and not self.reserve.can_draw():
            self.a_mined.blockSignals(True)
            self.a_mined.setChecked(False)
            self.a_mined.blockSignals(False)
            return
        self._tool_toggled(self.mining, on)

    def _outline_done(self, plane: str, points) -> None:
        self.reserve.outline_done(plane, points)

    def _edit_section_points(self) -> None:
        from ..scene.layers import DrillholeLayer

        tool = self.section
        removed = tool.remove_duplicates()
        if removed:
            self.status_msg.setText(tr("Cross-section: {n} duplicate point(s) removed", n=removed))
        if tool.line() is None:          # nothing left but one point
            tool.edit_cancelled()
            return
        has_dh = any(isinstance(lay, DrillholeLayer) and lay.visible for lay in self.scene.layers)
        dlg = SectionPointsDialog(self, tool.points_2d(), tool.in_plane_axes, self._section_legend, has_dh)
        dlg.points_changed.connect(tool.preview_2d)
        if dlg.exec():
            self._section_legend = dlg.show_legend
            tool.apply_2d(dlg.points())
        else:
            tool.edit_cancelled()

    def _show_section(self, line) -> None:
        from .section_view import SectionDialog

        if self.section_dialog is None:
            self.section_dialog = SectionDialog(self, self.scene)
            self.section_dialog.set_background(self.app_settings.background_rgb)
            self.section_dialog.closed.connect(self._section_closed)
            self.parking.watch(self.section_dialog, "section")
        self.section_dialog.export_legend = self._section_legend
        self.section_dialog.show_section(line)

    def _section_closed(self) -> None:
        if not self.section.drawing:      # the finished line belongs to the closed section
            self.section.clear()

    def _recut_section(self) -> None:
        if self.section_dialog is not None:
            self.section_dialog.layers_changed()

    def _on_identified(self, layer, res) -> None:
        """Identify on a drillhole also opens the drillhole log window; on a risk sleeve, selects its zone in the
        Risk Prediction window."""
        from ..scene.risk_layer import RiskLayer

        if isinstance(layer, RiskLayer):
            hit = getattr(self.identify, "last_hit", None)
            found = layer.zone_at(hit) if hit is not None else None
            dlg = self.prediction_dialog
            if found and found[1] >= 0 and dlg is not None and dlg.pred is not None and dlg.pred.hole_id == found[0]:
                dlg.select_zone(found[1])
            return
        hid = res.extra.get("hole_id") if res is not None else None
        if not hid or not hasattr(layer, "hole_log"):
            return
        from .drillhole_log import DrillholeLogDialog

        try:
            log = layer.hole_log(hid)
        except Exception as e:  # noqa: BLE001 - the identify panel is still shown
            log_ = tr("Drillhole log for {hole} failed: {error}", hole=hid, error=e)
            self.status_msg.setText(log_)
            return
        if self.hole_dialog is None:
            self.hole_dialog = DrillholeLogDialog(self)
            self.parking.watch(self.hole_dialog, "identify")
        self.hole_dialog.workspace_folder = str(self.workspace.folder) if self.workspace else ""
        self.hole_dialog.show_hole(log, res.extra.get("depth"))

    def _layers_changed(self) -> None:
        self.identify.layers_changed()
        # block models constrained by a solid that was reloaded or removed: constrain again
        from ..scene.layers import BlockModelLayer

        for lay in self.scene.layers:
            if isinstance(lay, BlockModelLayer) and lay.constraints_changed():
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    lay.rebuild()
                finally:
                    QApplication.restoreOverrideCursor()
                if self.props.layer is lay:
                    self.props.refresh()
                self.scene.render()
        if self.section_dialog is not None and self.section_dialog.isVisible():
            self._section_timer.start()
        if self.slice_dock is not None:
            self.slice_dock.widget().layers_changed()
        self.reserve.sync()

    def on_escape(self):
        """Esc steps back: clear the tool result, then stop the tool, then leave full screen."""
        if self.identify.showing:
            self.identify.clear()
        elif self.measure.showing:
            self.measure.clear()
            self.status_msg.setText(tr(MeasureTool.HINT))
        elif self.section.drawing:
            self.section.clear()
            self.status_msg.setText(tr(SectionTool.HINT))
        elif self.mining.drawing:
            self.mining.clear()
            self.status_msg.setText(tr(MinedShapeTool.HINT))
        elif self.a_identify.isChecked():
            self.a_identify.setChecked(False)
        elif self.a_measure.isChecked():
            self.a_measure.setChecked(False)
        elif self.a_section.isChecked():
            self.a_section.setChecked(False)
        elif self.a_mined.isChecked():
            self.a_mined.setChecked(False)
        elif self.is_full_screen():
            self.toggle_fullscreen(False)

    # --- background jobs -------------------------------------------------------------
    def _start_job(self, kind: str, fn, **extra) -> None:
        job = {"kind": kind, "generation": self._generation, **extra}
        self.loads_pending += 1
        if kind != "refresh":
            QApplication.setOverrideCursor(Qt.BusyCursor)
        thread = QThread(self)
        worker = _Worker(job, fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_job_done)
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda: self._threads.remove((thread, worker)))
        self._threads.append((thread, worker))
        thread.start()

    @Slot(object, object)
    def _on_job_done(self, job: dict, result):
        kind = job["kind"]
        self.loads_pending = max(0, self.loads_pending - 1)
        if kind != "refresh":
            QApplication.restoreOverrideCursor()
        else:
            self._refresh_running = False
        if job["generation"] != self._generation:
            return  # the scene was cleared (another workspace opened) while this ran
        if isinstance(result, Exception):
            self._report_errors([f"{kind}: {result}"], APP_NAME, popup=kind != "refresh" or job["manual"])
            return
        if kind == "files":
            layers, errors = result
            self._add_layers(layers)
            self._finish_load(tr("Loaded {n} layer(s)", n=len(layers)), errors, tr("Some files could not be opened"))
        elif kind == "workspace":
            self._workspace_loaded(*result)
        elif kind == "refresh":
            self._refresh_done(job, result)
        elif kind == "connect":
            self._connect_done(job, result)

    def _add_layers(self, layers: list[Layer], errors: list[str] | None = None) -> None:
        from ..scene.geo_roles import ROLE_COLORS

        for layer in layers:                      # a role chosen in an Import button (fault, horizon, ...)
            for p in layer.inputs:
                role = self._pending_roles.pop(_key(Path(p)), None)
                if role is not None and hasattr(layer, "role"):
                    layer.role = role
                    if role in ROLE_COLORS:
                        layer.color = ROLE_COLORS[role]
                        if hasattr(layer, "use_styles"):
                            layer.use_styles = False
        first = not self.scene.layers
        self.reserve.prepare(layers)     # mined status attributes exist before block models are drawn
        for layer in layers:
            try:
                self.scene.add_layer(layer)
                self.layer_tree.add_layer(layer)
            except Exception as e:  # noqa: BLE001
                log.exception("adding layer %s failed", layer.name)
                if errors is not None:
                    errors.append(f"{layer.name}: {e}")
                else:
                    self._report_errors([f"{layer.name}: {e}"], tr("Some layers could not be shown"))
        if first and self.scene.layers:
            self.set_view(self.scene.view)
        self._layers_changed()           # e.g. a block model's constraining solid just arrived
        self.scene.render()
        self.props.refresh()
        self._schedule_autosave()

    def _finish_load(self, message: str, errors: list[str], title: str) -> None:
        self.status_msg.setText(message + (tr(", {n} error(s)", n=len(errors)) if errors else ""))
        self._report_errors(errors, title)

    def _report_errors(self, errors: list[str], title: str, popup: bool = True) -> None:
        if not errors:
            return
        self.load_errors.extend(errors)
        for e in errors:
            log.warning("%s", e)
        if popup and not self.quiet_errors:
            QMessageBox.warning(self, title, "\n\n".join(errors[:30]) + ("\n\n..." if len(errors) > 30 else ""))

    # --- files ----------------------------------------------------------------------
    def open_dialog(self):
        start = self.settings.value("last_dir", str(self.workspace.folder if self.workspace else Path.cwd()))
        paths, _ = QFileDialog.getOpenFileNames(self, tr("Open files"), start, tr(FILE_FILTER))
        if paths:
            self.settings.setValue("last_dir", str(Path(paths[0]).parent))
            self.open_paths(paths)

    def open_paths(self, paths: list[str]):
        paths = list(paths)
        if not paths:
            return
        if self.workspace is None and not self.quiet_errors and not self._adopt_workspace(paths):
            return  # the folder's saved workspace is being opened; the files follow it
        others, blocks = split_block_csvs([Path(p) for p in paths])
        if blocks:                   # block model CSVs are imported into .gvbm, never shown as CSV
            self.import_block_csvs(blocks)
        paths = [str(p) for p in others]
        if not paths:
            return
        names = ", ".join(Path(p).name for p in paths[:3]) + (" ..." if len(paths) > 3 else "")
        self.status_msg.setText(tr("Loading {names}", names=names))
        self._start_job("files", lambda: load_paths(paths))

    def _adopt_workspace(self, paths: list[str]) -> bool:
        """Files opened while no workspace is open start one, so they are saved and come
        back at the next start: the folder of the first file, or (when that folder cannot
        be written) the default workspace folder. A folder that already has a workspace file
        opens that workspace and the files are added to it. False: files were queued."""
        candidates = [Path(paths[0]).resolve().parent, default_workspace_folder()]
        for folder in candidates:
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            ws = Workspace(folder)
            if ws.file.exists():
                if self.open_workspace(folder, confirm=False):
                    self._open_after_load = list(paths)
                    return False
                continue
            if not ws.writable():
                continue
            ws.mark_existing_files_known()  # other files in the folder are not added by Refresh
            ws.saved_state = ""              # ready: the first change (these files) is saved
            self._set_workspace(ws)
            return True
        return True  # nowhere to save: show the files anyway

    # --- workspace -------------------------------------------------------------------
    def _update_title(self):
        if self.workspace is None:
            self.setWindowTitle(APP_TITLE)
        else:
            self.setWindowTitle(f"{self.workspace.name} - {APP_TITLE}   [{self.workspace.folder}]")

    def _view_state(self) -> dict:
        return self.scene.camera_state()

    def _set_workspace(self, ws: Workspace) -> None:
        self.workspace = ws
        database.set_project_folder(ws.folder)                # where geodatabase.sqlite is kept
        self._autosave_failed = False
        if not self.quiet_errors:
            self.settings.setValue("workspace/last", str(ws.folder))
            self.settings.sync()
        self._update_title()

    # --- automatic saving ---------------------------------------------------------
    def _schedule_autosave(self) -> None:
        """A change that the workspace file holds. Nothing is written now: the workspace is
        saved when the app closes, when another workspace is opened, or with Ctrl+S."""
        return

    def _autosave(self, force: bool = False) -> bool:
        """Write the workspace file when layers, display settings or the view changed.
        Not while the workspace is still opening (it would be saved half-loaded) and, unless
        forced, not while files load in the background."""
        ws = self.workspace
        if ws is None or ws.saved_state is None or self.quiet_errors:
            return False
        if self.loads_pending and not force:
            self._autosave_timer.start(AUTOSAVE_POLL_MS)
            return False
        view = self._view_state()
        if Workspace.fingerprint(self.scene.layers, view, camera=True) == ws.saved_state:
            return True
        try:
            ws.save(self.scene.layers, view)
        except OSError as e:
            if not self._autosave_failed:  # tell once, not every few seconds
                self._autosave_failed = True
                log.warning("workspace not saved: %s", e)
                self.status_msg.setText(tr("Workspace could not be saved: {error}", error=e))
            return False
        self._autosave_failed = False
        return True

    def _start_dir(self) -> str:
        if self.workspace is not None:
            return str(self.workspace.folder.parent)
        return str(self.settings.value("last_dir", str(Path.cwd())))

    def open_workspace_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, tr("Open workspace folder"), self._start_dir())
        if folder:
            self.open_workspace(folder, confirm=False)

    def open_workspace(self, folder: str | Path, confirm: bool = True) -> bool:
        """Show a workspace folder, replacing everything currently shown (the current
        workspace is saved first). ``confirm`` is kept for callers; nothing is asked now."""
        p = Path(folder)
        if p.is_file() and p.suffix.lower() == ".json":
            p = p.parent
        if not p.is_dir():
            self._report_errors([tr("Workspace folder not found:\n{path}", path=p)], tr("Open workspace"))
            return False
        self.clear_status_warning()
        self.clear_all()
        ws = Workspace(p)
        self._set_workspace(ws)
        self.status_msg.setText(tr("Opening workspace {folder} ...", folder=ws.folder))
        self._start_job("workspace", ws.load)
        return True

    def clear_all(self) -> None:
        self._autosave(force=True)
        self._autosave_timer.stop()
        self._open_after_load = []
        self._import_queue = []
        self._generation += 1
        self.identify.clear()
        self.measure.clear()
        self.section.clear()
        self.mining.clear()
        if self.drillhole_manager is not None:
            self.drillhole_manager.close()
            self.drillhole_manager = None
        if self.survey_manager is not None:
            self.survey_manager.close()
            self.survey_manager = None
        if self.pit_geology is not None:
            self.pit_geology.close()
            self.pit_geology = None
        if getattr(self, "block_manager", None) is not None:
            self.block_manager.close()             # asks to save unsaved block model changes
            self.block_manager = None
        if getattr(self, "map_export", None) is not None:
            self.map_export.close()
        if self.hole_dialog is not None:
            self.hole_dialog.hide()
        for dlg in (self.records_dialog, self.prediction_dialog):
            if dlg is not None:
                dlg.close()
        self.records_dialog = self.prediction_dialog = None
        self.risk_engine = None
        self.risk_layer = None
        self.a_slice.setChecked(False)               # the slice is of this workspace
        self._section_timer.stop()
        if self.section_dialog is not None:
            self.section_dialog.line = None
            self.section_dialog.hide()
        self.layer_tree.clear()
        self.props.set_layer(None)
        self.scene.clear()
        self.workspace = None
        database.set_project_folder(None)
        self.reserve.reset()
        self._update_title()

    def _workspace_loaded(self, layers: list[Layer], errors: list[str], view: dict | None) -> None:
        self._add_layers(layers, errors)
        if view and self.scene.layers:
            mode = view.get("mode") if view.get("mode") in self._view_actions else VIEW_3D
            self.set_view(mode)
            self.a_grid.setChecked(bool(view.get("grid")))
            self.scene.apply_camera_state(view)
            self._sync_perspective_action()
        self._autosave_timer.stop()
        ws = self.workspace
        made = ""
        if ws is not None:
            ws.saved_state = Workspace.fingerprint(self.scene.layers, self._view_state(), camera=True)
            name = ws.name
            if ws.database_notes:                  # e.g. a new MySQL database got its tables
                made = tr(" - database: {notes}", notes=_notes_text(ws.database_notes))
        else:
            name = "?"
        # Nothing is active after opening a workspace, so no legend shows until a layer is clicked.
        self.layer_tree.tree.setCurrentItem(None)
        self.layer_tree.tree.clearSelection()
        self._finish_load(tr("Workspace {name}: {n} layer(s){made}", name=name, n=len(layers), made=made), errors,
                          tr("Some workspace layers could not be opened"))
        if ws is not None:
            self._set_database_state(ws.database_error)
        self._take_block_csvs(ws)
        queued, self._open_after_load = self._open_after_load, []
        shown = {_key(p) for lay in self.scene.layers for p in lay.inputs}
        queued = [p for p in queued if _key(Path(p)) not in shown]
        if queued:
            self.open_paths(queued)

    def save_workspace_as(self) -> bool:
        folder = QFileDialog.getExistingDirectory(self, tr("Save workspace in folder"), self._start_dir())
        if not folder:
            return False
        self._autosave(force=True)  # the current workspace keeps its own, up-to-date file
        ws = Workspace(folder)
        if ws.file.exists() and (self.workspace is None or ws.folder != self.workspace.folder):
            r = QMessageBox.question(self, tr("Replace workspace?"),
                                     tr("{file} already exists.\nReplace it with the current workspace?", file=ws.file))
            if r != QMessageBox.Yes:
                return False
        # Data files already in the folder but not shown now are not added by Refresh later.
        ws.mark_existing_files_known()
        if self.workspace is not None and self.workspace.folder == ws.folder:
            ws.known |= self.workspace.known
        if self.workspace is not None:  # layers not opened yet move along, paths re-based
            old = self.workspace.folder
            ws.pending_specs = [{**spec, "files": [_to_text(_from_text(f, old), ws.folder)
                                                   for f in spec.get("files", [])]}
                                for spec in self.workspace.pending_specs]
        if not self._save_to(ws):
            return False
        self._set_workspace(ws)
        return True

    def _save_to(self, ws: Workspace) -> bool:
        try:
            path = ws.save(self.scene.layers, self._view_state())
        except OSError as e:
            self._report_errors([tr("Could not save the workspace:\n{error}", error=e)], tr("Save workspace"))
            return False
        self.status_msg.setText(tr("Workspace saved: {path}", path=path))
        return True

    def handle_open_request(self, workspace: str | None, files: list[str]) -> None:
        """A later ``GeoViewer -w FOLDER [files]`` handed its workspace to this window: come to
        the front, close the current workspace and open that one (files are added after it).
        While a modal dialog is open the request waits until it is closed."""
        self.bring_to_front()
        if QApplication.activeModalWidget() is not None:
            QTimer.singleShot(500, lambda: self.handle_open_request(workspace, files))
            return
        if workspace and self.open_command_line_workspace(workspace):
            if files:
                self._open_after_load = list(files)
        elif files:
            self.open_paths(files)

    def open_command_line_workspace(self, folder: str | Path) -> bool:
        """``-w FOLDER``: open it only when it is a GeoViewer workspace folder (it holds
        geoviewer_workspace.json, or FOLDER names that file). Otherwise nothing changes (a running
        app keeps its workspace) and a warning shows in the status bar - no message box."""
        p = Path(folder)
        if p.is_file() and p.name.lower() == WORKSPACE_FILE:
            p = p.parent
        if not p.exists():
            warning = tr("Workspace not found: {path}", path=p)
        elif not (p / WORKSPACE_FILE).is_file():
            warning = tr("Not a workspace folder (no {file}): {path}", file=WORKSPACE_FILE, path=p)
        else:
            return self.open_workspace(p, confirm=False)
        self.load_errors.append(warning)            # the smoke test still fails on it
        log.warning("%s", warning)
        self.show_status_warning(warning)
        return False

    def show_status_warning(self, text: str) -> None:
        self.status_warn.setText("⚠ " + text)
        self.status_warn.setToolTip(text)
        self.status_warn.show()
        self._warn_timer.start()

    def clear_status_warning(self) -> None:
        self._warn_timer.stop()
        self.status_warn.clear()
        self.status_warn.hide()

    def bring_to_front(self) -> None:
        if self.isMinimized():
            self.setWindowState((self.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive)
        self.show()
        self.raise_()
        self.activateWindow()

    def restore_last_workspace(self) -> bool:
        if not self.app_settings.restore_last_workspace:
            return False
        last = self.settings.value("workspace/last", "")
        if last and Path(str(last)).is_dir():
            return self.open_workspace(str(last), confirm=False)
        return False

    # --- refresh ----------------------------------------------------------------------
    def refresh(self, manual: bool = True) -> None:
        """Reload layers whose files changed and add data files new in the workspace folder."""
        if self._refresh_running or (self.loads_pending and not manual):
            if manual:
                self.status_msg.setText(tr("Refresh: busy, try again in a moment"))
            return
        if not self.scene.layers and self.workspace is None:
            if manual:
                self.status_msg.setText(tr("Refresh: nothing to refresh"))
            return
        layers = list(self.scene.layers)
        ws = self.workspace

        def work():
            from ..scene.production import production_signature

            if ws is not None and (ws.database_error or not ws.database_ready):
                ws.init_database()      # try again (server back, ...) / first time (folder adopted for files)
            prod_sig = (production_signature(database.project_target(ws.folder))
                        if ws is not None and not ws.database_error else None)
            rep = changed_layers(layers)
            pairs, errors = reload_layers(rep.changed)
            new_layers, waiting_new, restored = [], 0, []
            if ws is not None:
                restored = ws.load_pending()
                groups, waiting_new = ws.find_new_files()
                for g in groups:
                    lays, errs = load_paths(g)
                    new_layers += lays
                    errors += errs
                lays, errs = ws.new_database_layers()      # drillhole tables imported meanwhile
                new_layers += lays
                errors += errs
            return pairs, new_layers, errors, rep, waiting_new, restored, ws, prod_sig

        self._refresh_running = True
        if manual:
            self.status_msg.setText(tr("Refreshing ..."))
        self._start_job("refresh", work, manual=manual)

    def _refresh_done(self, job: dict, result) -> None:
        pairs, new_layers, errors, rep, waiting_new, restored, ws, prod_sig = result
        db_changed = ws is not None and ws is self.workspace and ws.database_error != self.db_error
        records_changed = (not db_changed and prod_sig is not None and ws is self.workspace
                           and self.reserve.folder is not None and prod_sig != self.reserve.db_sig)
        if restored and ws is self.workspace:  # layers whose files were missing when opened
            for spec, _layer in restored:
                if spec in ws.pending_specs:
                    ws.pending_specs.remove(spec)
            new_layers = [lay for _s, lay in restored] + new_layers
        updated = []
        self.reserve.prepare([new for _old, new in pairs])   # mined status attributes on reloaded block models
        for old, new in pairs:
            if old not in self.scene.layers:  # removed by the user meanwhile
                continue
            try:
                self.scene.replace_layer(old, new)
            except Exception as e:  # noqa: BLE001
                log.exception("replacing layer %s failed", old.name)
                errors.append(f"{old.name}: {e}")
                continue
            self.layer_tree.replace_layer(old, new)
            if self.props.layer is old:
                self.props.set_layer(new)
            updated.append(new.name)
        if new_layers:
            self._add_layers(new_layers, errors)
        self._layers_changed()
        if updated or new_layers:
            self.scene.refresh_grid()
            self.scene.render()

        parts = []
        if updated:
            names = ", ".join(updated[:4]) + (" ..." if len(updated) > 4 else "")
            parts.append(tr("{n} updated ({names})", n=len(updated), names=names))
        if new_layers:
            parts.append(tr("{n} new ({names})", n=len(new_layers),
                            names=", ".join(lay.name for lay in new_layers[:4])))
        waiting = len(rep.waiting) + waiting_new
        if waiting:
            parts.append(tr("{n} file(s) still being written", n=waiting))
        if rep.missing:
            parts.append(tr("{n} file(s) missing ({names})", n=len(rep.missing),
                            names=", ".join(p.name for p in rep.missing[:3])))
        if errors:
            parts.append(tr("{n} error(s)", n=len(errors)))
        if parts:
            self.status_msg.setText(tr("Refresh: {parts}", parts="; ".join(parts)))
        elif job["manual"]:
            self.status_msg.setText(tr("Refresh: no changes"))
        self._report_errors(errors, tr("Refresh"), popup=job["manual"])
        if db_changed:                           # the database became reachable (or not) meanwhile
            self.reserve.database_changed()
            self._set_database_state(ws.database_error)
        elif records_changed:                    # mined shapes changed elsewhere (another computer, SQL tool)
            self.reserve.database_changed()
            text = self.status_msg.text()
            self.status_msg.setText(tr("Refresh: production records updated") if text == tr("Refresh: no changes")
                                    else text + tr("; production records updated"))
        if waiting and job["manual"]:
            QTimer.singleShot(2000, lambda: self.refresh(manual=False))
        if ws is self.workspace:
            self._take_block_csvs(ws)
        self._schedule_autosave()

    # --- drillhole manager -------------------------------------------------------------
    def _licensed_for(self, feature: int, title: str) -> bool:  # noqa: ARG002 - kept for the GeoViewer code paths
        """GeoViewer asks for a license here; this application has no license check."""
        return True

    def open_license_dialog(self) -> None:
        from .license_dialog import LicenseDialog

        LicenseDialog(self).exec()

    def open_drillhole_manager(self) -> None:
        """The Drillhole Manager for the project database of the open workspace."""
        from .. import licensing

        if not self._licensed_for(licensing.FEAT_DATABASE, tr("Drillhole Manager")):
            return
        if self.workspace is None:
            QMessageBox.information(self, tr("Drillhole Manager"),
                                    tr("Open a workspace first: the drillholes are kept in its project database."))
            return
        if self.drillhole_manager is not None and self.parking.restore("drillholes"):
            return
        dlg = self.drillhole_manager
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        from ..io.drillhole_store import DrillholeStore

        dlg = DrillholeManager(DrillholeStore(database.project_target(self.workspace.folder)), self)
        dlg.changed.connect(self._drillholes_changed)
        if not dlg.start():
            dlg.deleteLater()
            return
        self.drillhole_manager = dlg
        self.parking.watch(dlg, "drillholes")
        dlg.show()

    def open_block_manager(self) -> None:
        """The Block Model Manager for the block models (.gvbm) of the open workspace."""
        from .. import licensing

        if not self._licensed_for(licensing.FEAT_DATABASE, tr("Block Model Manager")):
            return
        if self.workspace is None:
            QMessageBox.information(self, tr("Block Model Manager"),
                                    tr("Open a workspace first: its block models are kept in its folder."))
            return
        dlg = getattr(self, "block_manager", None)
        if dlg is not None and self.parking.restore("blocks"):
            return
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        from .block_manager import BlockModelManager

        if dlg is not None:
            dlg.deleteLater()
        dlg = self.block_manager = BlockModelManager(self, self)
        dlg.saved.connect(self._block_model_saved)
        self.parking.watch(dlg, "blocks")
        dlg.show()

    def _block_model_saved(self, path) -> None:
        """A block model file was written by the manager: show it (new) or read it again."""
        path = Path(path)
        shown = any(_key(p) == _key(path) for lay in self.scene.layers for p in lay.inputs)
        ws = self.workspace
        inside = ws is not None and _key(path).startswith(_key(ws.folder) + os.sep)
        if shown or (inside and _key(path) not in ws.known):
            # Refresh reloads the changed layer / adds the new file (a file just written counts as
            # "still being written" and is picked up by the follow-up refresh a moment later)
            if not hasattr(self, "_dh_timer"):
                self._dh_timer = QTimer(self)
                self._dh_timer.setSingleShot(True)
                self._dh_timer.timeout.connect(self._reload_drillholes)
            self._dh_timer.start(300)
            return
        if ws is not None:
            ws.known.add(_key(path))
        self.open_paths([str(path)])

    # --- block model CSVs: imported into GeoViewer block models (.gvbm) ------------------------
    def _take_block_csvs(self, ws: Workspace | None) -> None:
        if ws is not None and ws.block_csvs:
            found, ws.block_csvs = ws.block_csvs, []
            self.import_block_csvs(found)

    def import_block_csvs(self, paths: list[Path]) -> None:
        """Block model CSVs are never shown as they are: each is imported into a GeoViewer
        block model (.gvbm) in the Block Model Manager, one after the other, and then offered
        for deletion. The .gvbm then shows as the layer."""
        queue = self._import_queue
        if self.workspace is not None:            # Refresh does not offer them again
            self.workspace.known |= {_key(Path(p)) for p in paths}
        for p in map(Path, paths):
            if _key(p) not in {_key(q) for q in queue}:
                queue.append(p)
        if self.quiet_errors:                     # smoke test / scripted: no dialogs, CSVs not shown
            for p in queue:
                log.info("block model CSV not imported (no dialogs): %s", p)
            queue.clear()
            return
        if queue and not self._importing:
            QTimer.singleShot(0, self._next_block_import)

    def _next_block_import(self) -> None:
        queue = self._import_queue
        if self._importing or not queue:
            return
        if QApplication.activeModalWidget() is not None:      # e.g. an error message still open
            QTimer.singleShot(300, self._next_block_import)
            return
        path = queue.pop(0)
        self._importing = True
        try:
            if path.exists() and not self._import_block_csv(path):
                self.show_status_warning(tr("{name} not opened: a block model CSV opens once it is imported as a "
                                            "GeoViewer block model (.gvbm)", name=path.name))
        finally:
            self._importing = False
        if queue:
            QTimer.singleShot(0, self._next_block_import)

    def _import_block_csv(self, path: Path) -> bool:
        from .block_manager import ImportAsciiDialog, offer_to_remove_source

        self.status_msg.setText(tr("{name} is a block model CSV: import it as a GeoViewer block model (.gvbm)",
                                   name=path.name))
        if self.workspace is not None:
            self.open_block_manager()                         # the import runs in the manager
            mgr = getattr(self, "block_manager", None)
            if mgr is not None and mgr.isVisible():
                mgr.raise_()
                return mgr.import_ascii(path)
        # no Block Model Manager here (license, no workspace): the import dialog alone
        try:
            dlg = ImportAsciiDialog(path, self.workspace.folder if self.workspace else None, self)
        except Exception as e:  # noqa: BLE001 - not readable as a block model export
            self._report_errors([f"{path.name}: {e}"], tr("Import Surpac block model (ASCII)"))
            return False
        if not (dlg.exec() and dlg.result_model is not None):
            return False
        self._block_model_saved(dlg.out_path)
        offer_to_remove_source(self, path, dlg.out_path)
        return True

    def open_map_export(self) -> None:
        """The export window for the current view (2D or 3D): a page saved as PNG."""
        from .. import licensing

        if not self._licensed_for(licensing.FEAT_EXPORT, tr("Export map")):
            return
        from .map_export import MapExportDialog

        if not any(lay.visible for lay in self.scene.layers):
            QMessageBox.information(self, tr("Export map"),
                                    tr("Nothing to export: open a workspace or show some layers."))
            return
        dlg = getattr(self, "map_export", None)
        if dlg is None:
            dlg = self.map_export = MapExportDialog(self)
            self.parking.watch(dlg, "map_export")
        dlg.start()

    def open_survey_manager(self) -> None:
        """The Survey Manager for the project database of the open workspace."""
        from .. import licensing

        if not self._licensed_for(licensing.FEAT_DATABASE, tr("Pit Manager")):
            return
        if self.workspace is None:
            QMessageBox.information(self, tr("Pit Manager"),
                                    tr("Open a workspace first: the survey is kept in its project database."))
            return
        if self.survey_manager is not None and self.parking.restore("survey"):
            return                                  # as it was left (with the route designer, if open)
        dlg = self.survey_manager
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        from ..io.survey_store import SurveyStore

        dlg = SurveyManager(SurveyStore(database.project_target(self.workspace.folder)), self)
        dlg.changed.connect(self._drillholes_changed)
        dlg.changed.connect(lambda: self.pit_geology is not None and self.pit_geology.reload())
        dlg.geology_requested.connect(self.open_pit_geology)
        if not dlg.start():
            dlg.deleteLater()
            return
        self.survey_manager = dlg
        self.parking.watch(dlg, "survey")
        dlg.show()

    def open_pit_geology(self, pid=None) -> None:
        """The Pit Geology window for the project database of the open workspace (``pid``: show that pit)."""
        from .. import licensing

        if not self._licensed_for(licensing.FEAT_DATABASE, tr("Pit Geology")):
            return
        if self.workspace is None:
            QMessageBox.information(self, tr("Pit Geology"),
                                    tr("Open a workspace first: the pits are kept in its project database."))
            return
        dlg = self.pit_geology
        if dlg is not None:
            if not self.parking.restore("pit_geology") and not dlg.isVisible():
                if not dlg.start(pid):                  # closed before: read the pits again
                    return
                dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            if pid is not None:
                dlg.select_pit(pid)
            return
        from .pit_geology import PitGeologyWindow

        dlg = PitGeologyWindow(database.project_target(self.workspace.folder), self,
                               workspace_folder=str(self.workspace.folder))
        dlg.changed.connect(self._drillholes_changed)
        if not dlg.start(pid):
            dlg.deleteLater()
            return
        self.pit_geology = dlg
        self.parking.watch(dlg, "pit_geology")
        dlg.show()

    def _drillholes_changed(self) -> None:
        """The manager wrote to the database: read the drillhole layer again (soon, once)."""
        for lay in self.scene.layers:
            if lay.source == "database":
                lay.db_signature = None              # changed values keep the row counts: force a reload
                lay.db_file_sig = None
        if self.workspace is not None:                         # e.g. the first hole / pit: show the layer
            kinds = {type(lay).__name__ for lay in self.scene.layers if lay.source == "database"}
            if "DrillholeLayer" not in kinds:
                self.workspace.database_drillholes_seen = False
            if "SurveyLayer" not in kinds:
                self.workspace.database_survey_seen = False
        if not hasattr(self, "_dh_timer"):
            self._dh_timer = QTimer(self)
            self._dh_timer.setSingleShot(True)
            self._dh_timer.timeout.connect(self._reload_drillholes)
        self._dh_timer.start(300)

    def _reload_drillholes(self) -> None:
        if self._refresh_running:
            self._dh_timer.start(500)                # try again when the running refresh is done
            return
        self.refresh(manual=True)

    # --- settings ------------------------------------------------------------------
    def open_settings(self) -> None:
        folder = str(self.workspace.folder) if self.workspace is not None else None
        dlg = SettingsDialog(self.app_settings, self, workspace_folder=folder)
        if dlg.exec():
            old_db = self.app_settings.database
            self.app_settings = dlg.result_settings()
            self.app_settings.save(self.settings)
            self.settings.sync()
            self._apply_app_settings()
            if self.app_settings.database != old_db:
                self._database_changed()

    def _database_changed(self) -> None:
        """Another database in the settings: connect to it and show its drillholes and
        production records at once (the other layers stay as they are)."""
        database.configure(self.app_settings.database)
        self._db_warned = ""
        self.connect_database(manual=True)

    # --- database connection --------------------------------------------------------------
    def _startup_connect(self) -> None:
        """At start: connect to the chosen database (a workspace being opened does it itself)."""
        if self.workspace is None and not self.loads_pending:
            self.connect_database(manual=False)

    def connect_database(self, manual: bool = True) -> None:
        """Connect in the background and set up missing tables / fields; then put the
        drillhole layer and the production records of that database in place."""
        ws = self.workspace
        olds = [lay for lay in self.scene.layers if lay.source == "database"]
        cfg = self.app_settings.database
        if ws is None and not cfg.is_mysql:
            self._set_database_state("")                 # SQLite: connected when a workspace opens
            return
        self.status_msg.setText(tr("Connecting to {db} ...", db=cfg.describe()))
        specs = [layer_spec(lay, None) for lay in olds]

        def work():
            from ..project_db import init_project_database
            from ..scene.loaders import offline_database_layer

            if ws is not None:
                ws.init_database()
                error, notes, folder = ws.database_error, ws.database_notes, ws.folder
            else:
                folder = None
                try:
                    notes, error = init_project_database(database.project_target(Path.cwd()), None), ""
                except Exception as e:  # noqa: BLE001
                    notes, error = [], str(e) or type(e).__name__
            pairs, added, errors = [], [], []
            if ws is not None:
                target = database.project_target(folder)
                for old, spec in zip(olds, specs):
                    new, errs = (load_spec(spec, folder) if not error else (None, []))
                    errors += errs
                    pairs.append((old, new or offline_database_layer(target, error or "; ".join(errs), spec)))
                kinds = {type(lay).__name__ for lay in olds}
                ws.database_drillholes_seen = "DrillholeLayer" in kinds
                ws.database_survey_seen = "SurveyLayer" in kinds
                if len(kinds) < 2:
                    added, errs = ws.new_database_layers()
                    errors += errs
            return error, notes, pairs, added, errors

        self._start_job("connect", work, manual=manual)

    def _connect_done(self, job: dict, result) -> None:
        error, notes, pairs, added, errors = result
        for old, new in pairs:
            if old in self.scene.layers:
                try:
                    self.scene.replace_layer(old, new)
                    self.layer_tree.replace_layer(old, new)
                except Exception as e:  # noqa: BLE001
                    log.exception("replacing %s failed", old.name)
                    errors.append(f"{old.name}: {e}")
        if added:
            self._add_layers(added)
        self.reserve.database_changed()                  # production records of this database
        if self.section_dialog is not None and self.section_dialog.isVisible():
            self.section_dialog.layers_changed()
        self._layers_changed()
        cfg = self.app_settings.database
        if not error:
            made = f" - {_notes_text(notes)}" if notes else ""
            self.status_msg.setText(tr("Connected to {db}{made}", db=cfg.describe(), made=made))
        if errors:
            self._report_errors(errors, tr("Database"), popup=job.get("manual", True))
        self._set_database_state(error)

    def _set_database_state(self, error: str) -> None:
        """Connected or not: switch the database tools on / off and say so (a warning once
        per problem; Refresh retrying does not repeat it)."""
        self.db_error = error
        ok = not error
        cfg = self.app_settings.database
        for a in (self.a_drillholes, self.a_survey, self.a_pit_geology, self.a_reserve):
            a.setEnabled(ok)
        self.a_mined.setEnabled(ok and self.scene.view != VIEW_3D)
        if not ok:
            if self.a_mined.isChecked():
                self.a_mined.setChecked(False)
            if self.reserve.window is not None:
                self.reserve.window.hide()
            if self.drillhole_manager is not None:
                self.drillhole_manager.close()
                self.drillhole_manager = None
            if self.survey_manager is not None:
                self.survey_manager.close()
                self.survey_manager = None
            if self.pit_geology is not None:
                self.pit_geology.close()
                self.pit_geology = None
        name = "MySQL " + cfg.database if cfg.is_mysql else "SQLite"
        self.status_db.setText(f"● {name}" if ok else "● " + tr("{name}: not connected", name=name))
        self.status_db.setStyleSheet(f"color: {'#5fd38d' if ok else '#ff6b6b'};")
        self.status_db.setToolTip(cfg.describe() + ("" if ok else f"\n{error}"))
        if ok:
            self._db_warned = ""
            return
        if error == self._db_warned:
            return
        self._db_warned = error
        log.warning("database not connected: %s", error)
        if self.quiet_errors:
            return
        box = QMessageBox(QMessageBox.Warning, tr("Database"),
                          tr("GeoViewer cannot connect to the database\n{db}\n\n{error}\n\n"
                             "Drillholes and production (Reserve, Mined outline, Drillholes) are switched off "
                             "until the database can be reached. Check the server and Settings > Database, "
                             "then press F5 (or Connect in Settings) to try again.", db=cfg.describe(), error=error),
                          parent=self)
        settings = box.addButton(tr("Database settings ..."), QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is settings:
            SettingsDialog._last_page = 2
            self.open_settings()

    def _apply_app_settings(self) -> None:
        s = self.app_settings
        from . import theme

        if theme.CURRENT != s.theme:                     # the theme chosen in Settings: at once
            theme.apply_theme(QApplication.instance(), s.theme)
        self.scene.set_background(s.background_rgb)
        if self.section_dialog is not None:
            self.section_dialog.set_background(s.background_rgb)
        self.scene.set_caption_style(s.caption_size, s.caption_transparency)
        self.scene.set_caption_visible(s.show_view_caption)
        if s.auto_refresh:
            self._auto_timer.start(s.refresh_interval * 1000)
            self.status_refresh.setText(tr("Auto refresh: {s} s", s=s.refresh_interval))
            self.a_refresh.setToolTip(tr("Reload layers whose files changed  (F5)\nAuto refresh every {s} s",
                                         s=s.refresh_interval))
        else:
            self._auto_timer.stop()
            self.status_refresh.setText("")
            self.a_refresh.setToolTip(tr("Reload layers whose files changed  (F5)\nAuto refresh is off (Settings)"))

    # --- layers -------------------------------------------------------------------
    # --- geological model import / 3D slice (after 3D MPM Studio) -------------------------------------
    def import_with_role(self, role: str) -> None:
        """Open surfaces / solids (Surpac, DXF, GOCAD) and give them a geological role (horizon, fault, ...)."""
        from PySide6.QtWidgets import QInputDialog

        from ..scene.geo_roles import ROLES

        title = tr("Import fault surfaces") if role == "fault" else tr("Import geological model")
        start = self.settings.value("last_dir", str(self.workspace.folder if self.workspace else Path.cwd()))
        filt = tr("Surfaces, solids, strings (*.dtm *.str *.dxf *.ts *.pl *.so *.gp);;All files (*.*)")
        paths, _ = QFileDialog.getOpenFileNames(self, title, start, filt)
        if not paths:
            return
        if not role:
            keys = list(ROLES)
            texts = [tr(ROLES[k]) for k in keys]
            text, ok = QInputDialog.getItem(self, title, tr("Role of the imported layers:"), texts,
                                            keys.index("horizon"), False)
            if not ok:
                return
            role = keys[texts.index(text)]
        self.settings.setValue("last_dir", str(Path(paths[0]).parent))
        for p in paths:
            self._pending_roles[_key(Path(p))] = role
        self.open_paths(paths)

    def _toggle_slice(self, on: bool) -> None:
        from .slice_panel import SliceDock, SlicePanel

        if self.slice_dock is None:
            if not on:
                return
            d = self.slice_dock = SliceDock(tr("3D slice"), self)
            d.setObjectName("dock_slice")
            d.setWidget(SlicePanel(self.scene, d))
            d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                          | QDockWidget.DockWidgetClosable)
            self.addDockWidget(Qt.RightDockWidgetArea, d)
            d.closed.connect(lambda: self.a_slice.setChecked(False))
        panel = self.slice_dock.widget()
        if on:
            self.slice_dock.show()
            self.slice_dock.raise_()
            panel.activate()
        else:
            self.slice_dock.hide()
            panel.deactivate()

    # --- drilling risk -------------------------------------------------------------------------------
    def _risk_target(self):
        folder = self.workspace.folder if self.workspace else None
        if folder is None:
            return None, None
        return database.project_target(folder), folder

    def _need_workspace(self, title: str) -> bool:
        if self.workspace is None:
            QMessageBox.information(self, title, tr("Open a workspace first (the drilling records live in its project "
                                                    "database)."))
            return False
        if self.db_error:
            QMessageBox.warning(self, title, tr("The project database is not available:\n{e}", e=self.db_error))
            return False
        return True

    def _engine(self):
        from ..risk.engine import RiskEngine

        target, folder = self._risk_target()
        if target is None:
            return None
        if self.risk_engine is None or self.risk_engine.folder != folder or self.risk_engine.target != target:
            self.risk_engine = RiskEngine(target, folder)
        self._risk_surfaces()
        return self.risk_engine

    def _risk_surfaces(self) -> None:
        from ..risk.geomodel import surfaces_from_layers

        if self.risk_engine is not None:
            self.risk_engine.set_surfaces(surfaces_from_layers(self.scene.layers))

    def open_drilling_records(self) -> None:
        if not self._need_workspace(tr("Drilling records")):
            return
        from .history_dialog import DrillingRecordsDialog

        eng = self._engine()
        if self.records_dialog is None:
            self.records_dialog = DrillingRecordsDialog(eng.target, eng.folder, self, current_area=eng.area_id)
            self.records_dialog.changed.connect(self._records_changed)
        self.records_dialog.show()
        self.records_dialog.raise_()
        self.records_dialog.activateWindow()

    def _records_changed(self) -> None:
        if self.risk_engine is not None:
            self.risk_engine.records_changed()
        if self.prediction_dialog is not None:
            self.prediction_dialog.refresh_holes()

    def open_risk_prediction(self, plan: bool = False) -> None:
        if not self._need_workspace(tr("Risk prediction")):
            return
        from .prediction_dialog import PredictionDialog

        eng = self._engine()
        if self.prediction_dialog is None:
            dlg = self.prediction_dialog = PredictionDialog(eng, self, prepare=self._risk_surfaces)
            dlg.show_in_3d.connect(self._show_risk)
            dlg.zone_selected.connect(self._risk_zone_selected)
            dlg.models_requested.connect(self.open_risk_models)
            dlg.records_requested.connect(self.open_drilling_records)
        else:
            self.prediction_dialog.refresh_holes()
        self.prediction_dialog.mode.setCurrentIndex(1 if plan else 0)
        self.prediction_dialog.show()
        self.prediction_dialog.raise_()
        self.prediction_dialog.activateWindow()

    def open_risk_models(self) -> None:
        if not self._need_workspace(tr("Risk models")):
            return
        from .risk_admin import ModelsDialog

        eng = self._engine()
        dlg = ModelsDialog(eng, self)
        dlg.trained.connect(lambda: self.prediction_dialog is not None and self.prediction_dialog.refresh_holes())
        dlg.exec()

    def open_geomodel(self) -> None:
        if not self._need_workspace(tr("Geological model")):
            return
        from .risk_admin import GeoModelDialog

        eng = self._engine()
        try:
            data = eng.data
            cards = {r["formation"]: r for r in data["formation"].to_dict("records")}
            areas = list(data["area"]["area_id"].astype(str))
        except Exception:  # noqa: BLE001 - no records yet
            cards, areas = {}, []

        def column():
            m = eng.geomodel
            if not m.horizons:
                return []
            h = m.horizons[0]
            ny, nx = h.z.shape
            return m.column_at(h.x0 + nx * h.d / 2, h.y0 + ny * h.d / 2)
        dlg = GeoModelDialog(self.scene.layers, eng.folder, cards, self, column_at=column)
        dlg.set_areas(areas, eng.area_id)
        if dlg.exec():
            eng.geomodel_changed()
            self._risk_surfaces()
            self.props.refresh()
            self._schedule_autosave()
            self.status_msg.setText(tr("Geological model updated - predict again to use it."))

    def _show_risk(self, pred) -> None:
        from ..scene.risk_layer import RiskLayer

        if pred is None:
            if self.risk_layer is not None and self.risk_layer in self.scene.layers:
                self.risk_layer.set_visible(False)
                self.scene.render()
            return
        events = []
        try:
            occ = self.risk_engine.data.hole("occurrence", pred.hole_id)
            events = [(float(a), t, 1, t) for a, t in zip(occ["depth_from"], occ["type"])]
        except Exception:  # noqa: BLE001
            pass
        if self.risk_layer is None or self.risk_layer not in self.scene.layers:
            from ..workspace import restore_app_layer

            self.risk_layer = RiskLayer(name=tr("Drilling risk"), path=None)
            restore_app_layer(self.workspace, self.risk_layer)
            self.risk_layer.predictions = {pred.hole_id: pred}
            self.risk_layer.events = {pred.hole_id: events}
            self.risk_layer.visible = True
            self._add_layers([self.risk_layer])
        else:
            self.risk_layer.predictions = {pred.hole_id: pred}
            self.risk_layer.events = {pred.hole_id: events}
            self.risk_layer.selected = None
            self.risk_layer.visible = True
            self.risk_layer.rebuild()
            self.layer_tree.refresh() if hasattr(self.layer_tree, "refresh") else None
        lo, hi = self.risk_layer.world_bounds()
        self.show_in_view(lo, hi, margin=0.6)
        self.scene.render()

    def _risk_zone_selected(self, hole_id: str, zi: int) -> None:
        lay = self.risk_layer
        if lay is None or lay not in self.scene.layers or hole_id not in lay.predictions:
            return
        lay.selected = (hole_id, zi)
        lay.rebuild()
        self.scene.render()

    def remove_layer(self, layer: Layer):
        if not layer.removable:                      # database drillholes, mined shapes: hide them instead
            self.status_msg.setText(tr("{name} stays in the list - untick it to hide it", name=layer.name))
            return
        self.scene.remove_layer(layer)
        self.layer_tree.remove_layer(layer)
        self._layers_changed()
        self._schedule_autosave()

    def rename_layer(self, layer: Layer) -> None:
        from PySide6.QtWidgets import QInputDialog

        from ..scene.layers import BlockModelLayer

        text, ok = QInputDialog.getText(self, tr("Rename layer"), tr("Name:"), text=layer.name)
        name = text.strip()
        if not ok or not name or name == layer.name:
            return
        if any(lay.name == name for lay in self.scene.layers if lay is not layer):
            self._report_errors([tr("A layer called {name} exists already.", name=name)], tr("Rename layer"))
            return
        old, layer.name = layer.name, name
        for lay in self.scene.layers:              # constraints that use this DTM follow the name
            if isinstance(lay, BlockModelLayer):
                for cond in lay.constraints:
                    if cond.get("kind") == "dtm" and cond.get("layer") == old:
                        cond["layer"] = name
                if old in lay._dtm_cache:
                    lay._dtm_cache[name] = lay._dtm_cache.pop(old)
        self.reserve.layer_renamed(old, name)      # the reserve's block model / ore solid follow too
        self.layer_tree.update_name(layer)
        self.props.refresh()
        self._schedule_autosave()

    def show_in_view(self, lo_world, hi_world, margin: float = 0.25) -> None:
        """Bring the main window forward and frame the box lo-hi (world coordinates) with some room around."""
        import numpy as np

        lo, hi = np.asarray(self.scene.to_local(lo_world), float), np.asarray(self.scene.to_local(hi_world), float)
        pad = margin * max(float(np.max(hi - lo)), 20.0)
        lo, hi = lo - pad, hi + pad
        self.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        self.scene.render()
        self.raise_()
        self.activateWindow()

    def zoom_to_layer(self, layer: Layer | None):
        if layer is None:
            return
        b = layer.world_bounds()
        if b is None:
            return
        lo, hi = self.scene.to_local(b[0]), self.scene.to_local(b[1])
        self.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        self.scene.render()

    def zoom_to_pit(self, layer, pid) -> None:
        """Zoom to a pit of the survey layer and its branches."""
        b = layer.pit_bounds(pid) if hasattr(layer, "pit_bounds") else None
        if b is None:
            return
        lo, hi = self.scene.to_local(b[0]), self.scene.to_local(b[1])
        self.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        self.scene.render()

    def zoom_to_design(self, layer, did) -> None:
        """Zoom to a designed pit (None: to all of them)."""
        b = layer.design_bounds(did) if hasattr(layer, "design_bounds") else None
        if b is None:
            return
        lo, hi = self.scene.to_local(b[0]), self.scene.to_local(b[1])
        self.plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        self.scene.render()

    def set_design_style(self, layer, did, **kw) -> None:
        """A designed pit shown / hidden or given a colour (layers list or properties)."""
        layer.set_design_style(did, **kw)
        self.layer_tree.update_pits(layer)
        if self.props.layer is layer and "visible" in kw:
            self.props.refresh()
        self._after_property_change()

    def set_designs_shown(self, layer, on: bool) -> None:
        """The "Designed Pits" group switched on / off."""
        layer.set_show_designs(on)
        self.layer_tree.update_pits(layer)
        if self.props.layer is layer and (self.props.designs_group or self.props.design is not None):
            self.props.refresh()
        self._after_property_change()

    def _design_store(self, layer):
        from ..io.survey_store import SurveyStore

        target = getattr(layer, "db_target", None)
        if target is None and self.workspace is not None:
            target = database.project_target(self.workspace.folder)
        return SurveyStore(target) if target is not None else None

    def rename_design(self, layer, did) -> None:
        from PySide6.QtWidgets import QInputDialog

        x = layer.data.design(did) if layer.data is not None else None
        store = self._design_store(layer)
        if x is None or store is None:
            return
        text, ok = QInputDialog.getText(self, tr("Rename designed pit"), tr("Name:"), text=x.name)
        if not ok or not text.strip() or text.strip() == x.name:
            return
        try:
            store.rename_design(did, text)
        except Exception as e:  # noqa: BLE001 - a name in use, the database not reachable
            QMessageBox.warning(self, tr("Rename designed pit"), str(e))
            return
        self._drillholes_changed()
        if self.survey_manager is not None:
            self.survey_manager.reload_designs()

    def delete_design(self, layer, did) -> None:
        x = layer.data.design(did) if layer.data is not None else None
        store = self._design_store(layer)
        if x is None or store is None:
            return
        if QMessageBox.question(self, tr("Delete designed pit"),
                                tr("Delete the designed pit {name} from the database?\n"
                                   "The surveyed pits are not changed.", name=x.name)) != QMessageBox.Yes:
            return
        try:
            store.delete_design(did)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Delete designed pit"), str(e))
            return
        layer.pit_styles.pop(f"d{did}", None)
        self._drillholes_changed()
        if self.survey_manager is not None:
            self.survey_manager.reload_designs()

    def set_pit_style(self, layer, pid, **kw) -> None:
        """A pit's display setting changed (layers list or properties): draw it, keep the
        list's check boxes / colour swatches and the properties in step, save the workspace."""
        layer.set_pit_style(pid, **kw)
        self.layer_tree.update_pits(layer)
        if self.props.layer is layer and "visible" in kw:
            self.props.refresh()
        self._after_property_change()

    def _on_layer_selected(self, active: Layer | None):
        """Only the layer active in the TOC shows its legend (block model colour bar)."""
        for layer in self.scene.layers:
            layer.set_legend_active(layer is active)
        self.scene.render()

    def _layer_order_changed(self) -> None:
        """The list was reordered (a drag, or a layer added): the layers are kept in the same
        order (bottom of the list first), which is the order saved in the workspace."""
        listed = list(reversed(self.layer_tree.layers_top_down()))
        rest = [lay for lay in self.scene.layers if lay not in listed]
        self.scene.layers[:] = rest + [lay for lay in listed if lay in self.scene.layers]
        self._schedule_autosave()

    def _set_layer_visible(self, layer: Layer, visible: bool):
        layer.set_visible(visible)
        self.scene.refresh_grid()
        self.scene.render()
        self._layers_changed()
        if self.props.layer is layer:
            self.props.refresh()
        self._schedule_autosave()

    def _after_property_change(self):
        self._layers_changed()
        self.scene.refresh_grid()
        self.scene.render()
        self._schedule_autosave()

    # --- status bar coordinates ------------------------------------------------------
    def _on_mouse_move(self, obj, _event):
        if self.scene.origin is None:
            return
        x, y = obj.GetEventPosition()
        self._picker.Pick(x, y, 0, self.plotter.renderer)
        w = self.scene.to_world(self._picker.GetPickPosition())
        self.status_xyz.setText(f"X {w[0]:,.2f}    Y {w[1]:,.2f}    Z {w[2]:,.2f}")

    # --- drag & drop ---------------------------------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        folders = [p for p in paths if Path(p).is_dir()]
        if folders:  # a dropped folder opens as a workspace
            self.open_workspace(folders[0])
        elif paths:
            self.open_paths(paths)

    def closeEvent(self, e):
        bmm = getattr(self, "block_manager", None)
        if bmm is not None and bmm.dirty and not self.quiet_errors and not bmm._confirm_discard():
            e.ignore()                              # unsaved block model changes: the user chose Cancel
            return
        self._auto_timer.stop()
        self._autosave_poll.stop()
        self._autosave_timer.stop()
        for thread, _ in list(self._threads):
            thread.quit()
            thread.wait(2000)
        self._autosave(force=True)
        if "dfpredict.blockmodel.parallel" in sys.modules:       # estimation worker processes, if started
            sys.modules["dfpredict.blockmodel.parallel"].shutdown()
        self.identify.panel.close()
        self.measure.panel.close()
        if self.hole_dialog is not None:
            self.hole_dialog.close()
        if self.reserve.window is not None:
            self.reserve.window.close()
        if self.drillhole_manager is not None:
            self.drillhole_manager.close()
        if self.survey_manager is not None:
            self.survey_manager.close()
        if self.pit_geology is not None:
            self.pit_geology.close()
        if getattr(self, "map_export", None) is not None:
            self.map_export.close()
        self._section_timer.stop()
        if self.section_dialog is not None:
            self.section_dialog.hide()
            self.section_dialog.shutdown()
        self.toolbar.close()
        self.fab.close()
        self.plotter.close()
        super().closeEvent(e)


__all__ = ["MainWindow", "APP_NAME", "APP_TITLE", "__version__"]
