"""Visible GUI walkthrough of the drilling-risk functions - run it with run_gui_test.bat.

The real main window opens on a COPY of the demo workspace (demo_data/sb3_workspace is never changed) and
the mouse cursor moves slowly over every step, so you can follow it:

 1. the main window with the geological model (horizons, faults by role) and the drillholes;
 2. Drilling Records: search, pick DH-2026-001, the collar form, the Survey / Geology / Runs tabs, Next run,
    the entry checks, Revert; the reference tabs (formation cards, rock properties, measure library, overview);
 3. Risk Prediction: DH-2026-001 (tied contact, red zone ahead), the register rows and their details, the danger in
    the next 10 m / 50 m / to target; DH-2026-002 (live alarm rule R1);
 4. Plan check: a design typed in, Check plan, Safer options;
 5. the Models window, the Geological Model window, the 3D slice;
 6. the 3D view: the risk layer, Identify on a risk sleeve.

Not part of the normal test run (it needs a screen and takes a few minutes): it runs only when DFP_GUI_TEST=1.
DFP_GUI_SPEED = seconds per step (default 1.2; larger = slower).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("DFP_GUI_TEST") != "1",
                                reason="visible GUI walkthrough: run_gui_test.bat (sets DFP_GUI_TEST=1)")

DEMO = Path(__file__).resolve().parents[1] / "demo_data" / "sb3_workspace"
SPEED = float(os.environ.get("DFP_GUI_SPEED", "1.2"))


# --- slow, visible interaction ---------------------------------------------------------------------
class Driver:
    """Moves the real mouse cursor smoothly, clicks, types, waits - all slowly enough to watch."""

    def __init__(self, app):
        self.app = app

    def wait(self, seconds: float | None = None) -> None:
        end = time.monotonic() + (SPEED if seconds is None else seconds)
        while time.monotonic() < end:
            self.app.processEvents()
            time.sleep(0.015)

    def until(self, cond, timeout: float = 180.0, what: str = "") -> None:
        end = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > end:
                raise AssertionError(f"timed out waiting for {what or cond}")
            self.app.processEvents()
            time.sleep(0.03)

    def say(self, text: str) -> None:
        print(f"  -> {text}", flush=True)

    def move_to(self, widget, pos=None) -> None:
        """Glide the cursor to a widget (its centre, or ``pos`` in its coordinates)."""
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QCursor

        if widget is None or not widget.isVisible():
            return
        target = widget.mapToGlobal(pos if pos is not None else widget.rect().center())
        start = QCursor.pos()
        steps = max(12, int(40 * SPEED))
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)                        # ease in / out
            QCursor.setPos(QPoint(int(start.x() + (target.x() - start.x()) * t),
                                  int(start.y() + (target.y() - start.y()) * t)))
            self.app.processEvents()
            time.sleep(0.6 * SPEED / steps)
        self.wait(0.25 * SPEED)

    def click(self, widget, pos=None, double: bool = False) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        self.move_to(widget, pos)
        p = pos if pos is not None else widget.rect().center()
        if double:
            QTest.mouseDClick(widget, Qt.LeftButton, Qt.NoModifier, p)
        else:
            QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, p)
        self.wait(0.6 * SPEED)

    def type(self, widget, text: str, clear: bool = True) -> None:
        from PySide6.QtTest import QTest

        self.click(widget)
        if clear and hasattr(widget, "selectAll"):
            widget.selectAll()
        elif clear and hasattr(widget, "lineEdit") and widget.lineEdit() is not None:
            widget.lineEdit().selectAll()
        QTest.keyClicks(widget, text, delay=int(60 * SPEED))
        self.wait(0.5 * SPEED)

    def tab(self, tabs, index: int) -> None:
        bar = tabs.tabBar()
        self.click(bar, bar.tabRect(index).center())

    def table_row(self, view, row: int, col: int = 0) -> None:
        rect = view.visualRect(view.model().index(row, col))
        self.click(view.viewport(), rect.center())

    def tree_row(self, tree, item) -> None:
        self.click(tree.viewport(), tree.visualItemRect(item).center())

    def close(self, dialog, button=None) -> None:
        """Pause so the dialog can be looked at, then close it (with its button when given)."""
        self.wait(1.5 * SPEED)
        if button is not None:
            self.click(button)
        else:
            dialog.close()
        self.wait(0.8 * SPEED)

    def later(self, seconds: float, fn) -> None:
        """Run ``fn`` while a modal dialog's own loop is running (e.g. to look at it, then close it)."""
        from PySide6.QtCore import QTimer

        QTimer.singleShot(int(seconds * 1000), fn)


def toolbar_button(win, action):
    from PySide6.QtWidgets import QToolButton

    for b in win.toolbar.findChildren(QToolButton):
        if b.defaultAction() is action and b.isVisible():
            return b
    return None


def press(d: Driver, win, action) -> None:
    """Click a toolbar button (or trigger its action when the button is hidden)."""
    b = toolbar_button(win, action)
    if b is not None:
        d.click(b)
    else:
        action.trigger()
        d.wait()


def button_with_text(parent, text: str):
    from PySide6.QtWidgets import QPushButton

    for b in parent.findChildren(QPushButton):
        if b.isVisible() and b.text().replace("&", "").startswith(text):
            return b
    return None


# --- the walkthrough --------------------------------------------------------------------------------------
@pytest.fixture()
def workspace(tmp_path) -> Path:
    if not (DEMO / "drillrisk" / "models").exists():
        pytest.skip("demo workspace not built: tools/make_demo_workspace.py")
    ws = tmp_path / "sb3_workspace"
    shutil.copytree(DEMO, ws)
    return ws


def test_gui_walkthrough(workspace):
    from PySide6.QtWidgets import QApplication, QDialog, QPushButton

    app = QApplication.instance() or QApplication([])
    from dfpredict.io import database
    from dfpredict.ui.main_window import MainWindow
    from dfpredict.ui.theme import apply_dark_theme

    apply_dark_theme(app)
    d = Driver(app)
    print(f"\nGUI walkthrough (DFP_GUI_SPEED = {SPEED} s per step)")

    # 1. main window + demo workspace -------------------------------------------------------------
    d.say("open the main window and the demo workspace (copy)")
    win = MainWindow()
    database.configure(database.DatabaseConfig(backend="sqlite"))
    win.app_settings.database = database.DatabaseConfig(backend="sqlite")
    win.showMaximized()
    d.wait(1.0)
    assert win.open_workspace(workspace, confirm=False)
    d.until(lambda: not win._threads and win.loads_pending == 0, 180, "the workspace to load")
    d.wait(2 * SPEED)
    names = {lay.name: getattr(lay, "role", "") for lay in win.scene.layers}
    assert names.get("F4") == "fault" and names.get("horizon_H2_U2-U3") == "horizon"
    d.say(f"{len(names)} layers loaded: horizons and faults with their roles")
    for row in range(min(4, win.layer_tree.tree.topLevelItemCount())):     # walk over the layer list
        d.tree_row(win.layer_tree.tree, win.layer_tree.tree.topLevelItem(row))

    # 2. Drilling Records ------------------------------------------------------------------------------
    d.say("Drilling Records")
    press(d, win, win.a_records)
    rec = win.records_dialog
    d.until(lambda: rec is not None and rec.list.topLevelItemCount() > 1000, 60, "the records to load")
    d.wait()
    d.type(rec.search, "DH-2026")
    assert rec.list.topLevelItemCount() == 3
    d.tree_row(rec.list, rec.list.topLevelItem(0))
    d.until(lambda: rec.hole_id == "DH-2026-001", 20, "DH-2026-001 to load")
    d.say("collar form of DH-2026-001")
    for name in ("hole_id", "area_id", "x", "plan_azimuth", "plan_dip", "planned_depth", "result", "final_depth"):
        d.move_to(rec.fields[name])
    for i, title in enumerate(("Survey", "Geology log", "Runs")):
        d.say(f"tab: {title}")
        d.tab(rec.sub, i)
    n_runs = rec.tables["run"].model_.rowCount()
    d.say("Next run: From = the last To, values copied")
    d.click(button_with_text(rec, "Next run") or rec.b_check)
    assert rec.tables["run"].model_.rowCount() == n_runs + 1 and rec.dirty
    d.say("Check: the entry checks")
    d.click(rec.b_check)
    d.wait()
    d.say("Revert: the new run is dropped")
    d.click(rec.b_revert)
    assert not rec.dirty and rec.tables["run"].model_.rowCount() == n_runs
    for idx, title in ((2, "Formation cards"), (4, "Rock properties"), (6, "Measure library"), (7, "Overview")):
        d.say(f"tab: {title}")
        d.tab(rec.tabs, idx)
        if title == "Rock properties":
            assert rec.ref_tables["rock"].model_.rowCount() >= 40
            d.table_row(rec.ref_tables["rock"].view, 12, 4)               # dolomite, UCS
    d.tab(rec.tabs, 0)
    d.close(rec)
    assert not rec.isVisible()

    # 3. Risk Prediction ------------------------------------------------------------------------------
    d.say("Risk Prediction: DH-2026-001")
    press(d, win, win.a_predict)
    dlg = win.prediction_dialog
    d.wait()
    i = dlg.hole.findData("DH-2026-001")
    d.click(dlg.hole)
    dlg.hole.setCurrentIndex(i)
    dlg.hole.hidePopup()
    d.wait()
    d.click(dlg.b_predict)
    d.until(lambda: not dlg.busy and dlg.pred is not None and dlg.pred.hole_id == "DH-2026-001", 300,
            "the prediction of DH-2026-001")
    p = dlg.pred
    assert 0 < p.p_reach < 1 and any(c == "R-M02" for c, _t in p.messages)
    d.say(dlg.summary.text())
    d.wait(2 * SPEED)
    for r in range(dlg.table.rowCount()):
        d.say(f"register row {r + 1}: zone details")
        d.table_row(dlg.table, r, 0)
        d.wait(SPEED)
    d.move_to(dlg.chart)
    for text in ("Next 10 m", "Next 50 m", "To target"):
        d.say(f"danger in a depth range: {text}")
        d.click(button_with_text(dlg, text))
    d.say("DH-2026-002: pump pressure dropped in the last runs (rule R1)")
    i = dlg.hole.findData("DH-2026-002")
    d.click(dlg.hole)
    dlg.hole.setCurrentIndex(i)
    dlg.hole.hidePopup()
    d.click(dlg.b_predict)
    d.until(lambda: not dlg.busy and dlg.pred is not None and dlg.pred.hole_id == "DH-2026-002", 300,
            "the prediction of DH-2026-002")
    assert any(r.rule == "R1" and r.current for r in dlg.pred.rules)
    d.move_to(dlg.messages)
    d.wait(2 * SPEED)

    # 4. Plan check ------------------------------------------------------------------------------------
    d.say("Plan check: a new design")
    d.tab(dlg.mode, 1)
    for widget, value in ((dlg.p_name, "PLAN-GUI"), (dlg.p_x, "5450"), (dlg.p_y, "8420"), (dlg.p_z, "316"),
                          (dlg.p_az, "300"), (dlg.p_dip, "-70"), (dlg.p_depth, "300")):
        d.type(widget, value)
    d.click(button_with_text(dlg, "Check plan"))
    d.until(lambda: not dlg.busy and dlg.pred is not None and dlg.pred.hole_id == "PLAN-GUI", 300, "the plan check")
    assert dlg.pred.drilled_depth == 0
    d.say(dlg.summary.text())
    d.wait(2 * SPEED)

    def look_at_options():
        box = app.activeModalWidget()
        if box is None:
            d.later(1.0, look_at_options)
            return
        d.say("safer options: azimuth / dip alternatives")
        d.wait(3 * SPEED)
        box.close()
    dlg.paths.setValue(10)
    d.say("Safer options (a few predictions)")
    b = button_with_text(dlg, "Safer options")
    d.move_to(b)
    shown = []
    orig = dlg._show_options

    def show_and_close(df):
        shown.append(len(df))
        d.later(0.5, look_at_options)
        orig(df)
    dlg._show_options = show_and_close
    d.click(b)
    d.until(lambda: shown and not dlg.busy and app.activeModalWidget() is None, 600, "the safer options")
    assert shown[0] == 15
    d.tab(dlg.mode, 0)

    # 5. Models, Geological Model, 3D slice --------------------------------------------------------------
    def look_and_close(label: str, button_text: str):
        def run():
            box = app.activeModalWidget()
            if box is None:
                d.later(0.5, run)
                return
            d.say(label)
            d.wait(3 * SPEED)
            btn = button_with_text(box, button_text)
            if btn is not None:
                d.click(btn)
            else:
                box.reject()
        return run

    d.later(1.0, look_and_close("Models: training data, leave-one-area-out validation, versions", "Close"))
    d.say("Models window")
    d.click(button_with_text(dlg, "Models"))
    d.wait()
    d.close(dlg)
    d.later(1.0, look_and_close("Geological Model: roles, units top to bottom, error band", "Cancel"))
    d.say("Geological Model window")
    press(d, win, win.a_geomodel)
    d.wait()
    d.say("3D slice on / off")
    press(d, win, win.a_slice)
    d.wait(2 * SPEED)
    press(d, win, win.a_slice)
    assert not win.a_slice.isChecked()

    # 6. the 3D view with the risk layer ------------------------------------------------------------------
    d.say("3D view: the risk layer of the last prediction (DH-2026-002 / plan)")
    assert win.risk_layer is not None and win.risk_layer in win.scene.layers
    d.move_to(win.plotter.interactor)
    d.wait(3 * SPEED)
    d.say("done - closing")
    d.wait(1.5 * SPEED)
    for w in app.topLevelWidgets():
        if isinstance(w, QDialog) and w.isVisible():
            w.close()
    win.close()
    d.wait(1.0)
    del QPushButton
