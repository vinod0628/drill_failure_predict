"""A second ``GeoViewer -w FOLDER`` hands the workspace to the running app (single_instance.py)."""

import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CLIENT = """
import sys
from PySide6.QtCore import QCoreApplication
from dfpredict.single_instance import make_request, send_to_running
app = QCoreApplication([])
ok = send_to_running(make_request(sys.argv[2] or None, sys.argv[3:]), name=sys.argv[1])
sys.exit(0 if ok else 1)
"""


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _run_client(app, name: str, workspace: str, files=(), timeout: float = 30.0) -> int:
    proc = subprocess.Popen([sys.executable, "-c", CLIENT, name, workspace, *files], cwd=ROOT)
    end = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    if proc.poll() is None:
        proc.kill()
        pytest.fail("client did not finish")
    for _ in range(20):            # the request signal comes with the last events
        app.processEvents()
    return proc.returncode


def test_request_goes_to_running_app(app, tmp_path):
    from dfpredict.single_instance import InstanceServer

    name = f"GeoViewerTest-{uuid.uuid4().hex}"
    server = InstanceServer(name=name)
    assert server.listen()
    got = []
    server.requested.connect(lambda ws, files: got.append((ws, files)))
    try:
        f = tmp_path / "a.str"
        assert _run_client(app, name, str(tmp_path), [str(f)]) == 0
    finally:
        server.close()
    assert got == [(str(tmp_path.resolve()), [str(f.resolve())])]


def test_no_running_app_starts_normally(app):
    from dfpredict.single_instance import make_request, send_to_running

    assert not send_to_running(make_request("x", []), name=f"GeoViewerTest-{uuid.uuid4().hex}")


def test_handle_request_switches_workspace(app, tmp_path, monkeypatch):
    """The running window closes its workspace and opens the requested one."""
    from dfpredict.ui.main_window import MainWindow

    win = MainWindow()
    win.quiet_errors = True
    opened = []
    monkeypatch.setattr(win, "open_workspace", lambda folder, confirm=True: opened.append(folder) or True)
    (tmp_path / "geoviewer_workspace.json").write_text("{}")
    try:
        win.handle_open_request(str(tmp_path), ["b.dtm"])
        assert opened == [tmp_path]
        assert win._open_after_load == ["b.dtm"]
    finally:
        win.close()


@pytest.mark.parametrize("case", ["missing", "not_workspace"])
def test_wrong_workspace_warns_in_status_bar(app, tmp_path, monkeypatch, case):
    """A -w folder that does not exist or has no geoviewer_workspace.json: no message box,
    the current workspace stays, a warning shows in the status bar."""
    from PySide6.QtWidgets import QMessageBox

    from dfpredict.ui.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail("message box shown"))
    win = MainWindow()
    opened = []
    monkeypatch.setattr(win, "open_workspace", lambda folder, confirm=True: opened.append(folder) or True)
    folder = tmp_path / "w2"
    if case == "not_workspace":
        folder.mkdir()
        (folder / "a.str").write_text("x")
    try:
        win.handle_open_request(str(folder), [])
        assert opened == []
        assert not win.status_warn.isHidden()
        assert str(folder) in win.status_warn.text()
        folder.mkdir(exist_ok=True)              # now a real workspace: named by its file
        (folder / "geoviewer_workspace.json").write_text("{}")
        win.handle_open_request(str(folder / "geoviewer_workspace.json"), [])
        assert opened == [folder]
    finally:
        win.close()
