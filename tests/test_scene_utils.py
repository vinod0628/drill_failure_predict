import numpy as np

from dfpredict.scene.scene import nice_grid, nice_step


def test_nice_step():
    assert nice_step(930) == 200
    assert nice_step(720) == 200 or nice_step(720) == 125  # ~6 lines
    assert nice_step(60) == 10
    assert nice_step(0) == 1.0


def test_nice_grid_round_values():
    lo, hi, n = nice_grid(np.array([1180.93, 7021.09, 156.0]), np.array([2110.93, 7741.09, 243.0]))
    step_x = (hi[0] - lo[0]) / (n[0] - 1)
    assert lo[0] % step_x == 0 and hi[0] % step_x == 0
    assert lo[0] <= 1180.93 and hi[0] >= 2110.93
    assert lo[1] <= 7021.09 and hi[1] >= 7741.09


def test_freeze_watchdog_logs_where_the_gui_hangs(tmp_path):
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from dfpredict.app import FreezeWatchdog

    app = QApplication.instance() or QApplication([])
    log = tmp_path / "geoviewer.log"
    dog = FreezeWatchdog(app, log, limit=3.0)

    def stuck_in_the_gui():
        time.sleep(6.5)            # the GUI thread is blocked

    QTimer.singleShot(200, stuck_in_the_gui)
    end = time.monotonic() + 9.0
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.05)
    dog.stop()
    text = log.read_text(encoding="utf-8")
    assert "GUI not responding" in text
    assert "stuck_in_the_gui" in text            # the stack shows the blocking function
    assert "responsive again" in text
