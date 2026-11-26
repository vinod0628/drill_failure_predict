"""Application entry point: ``python -m dfpredict [-w FOLDER] [files...]``.

Options:
  -w, --workspace FOLDER   open this workspace folder (instead of the last one). When
                           GeoViewer is already running, that window closes its workspace and
                           opens this one; no second window starts (see single_instance.py).
  --smoke-test             open the window, load the given workspace / files, then quit by
                           itself. Exit code 0 = loaded without errors, 2 = load errors,
                           3 = timeout. Used by the release build to check the packaged .exe.

Without -w or files, the last workspace is opened again (unless switched off in Settings).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback

log = logging.getLogger(__name__)
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="DrillFailPredict", add_help=False)
    ap.add_argument("-w", "--workspace", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("files", nargs="*")
    args, _unknown = ap.parse_known_args(argv)  # ignore Qt's own options (-style, ...)
    return args


def log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "DrillFailPredict" / "logs"


def _setup_logging(smoke: bool) -> Path:
    """Send stdout/stderr to a log file when there is no console (windowed .exe),
    and log unhandled exceptions instead of letting them vanish or abort the app."""
    path = log_dir() / "dfpredict.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 2_000_000:
            path.replace(path.with_suffix(".old.log"))
        if sys.stderr is None or sys.stdout is None:
            f = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives for the process
            if sys.stdout is None:
                sys.stdout = f
            if sys.stderr is None:
                sys.stderr = f
    except OSError:
        pass

    def excepthook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n--- unhandled exception ---\n" + text)
        except OSError:
            pass
        if sys.stderr is not None:
            sys.stderr.write(text)
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        if smoke:
            if app is not None:
                app.exit(4)
            return
        if app is not None:
            from .i18n import tr

            QMessageBox.critical(None, tr("GeoViewer - unexpected error"),
                                 tr("{error}\n\nDetails were written to:\n{path}", error=value, path=path))

    sys.excepthook = excepthook
    return path


def _arm_smoke_test(app, win, timeout_ms: int = 60000) -> None:
    import time

    from PySide6.QtCore import QTimer

    start = time.monotonic()
    state = {"done_at": None}

    def poll():
        busy = bool(win._threads) or win.loads_pending > 0
        if not busy and state["done_at"] is None:
            state["done_at"] = time.monotonic()
        if state["done_at"] is not None and time.monotonic() - state["done_at"] > 1.0:
            app.exit(2 if win.load_errors else 0)
        elif (time.monotonic() - start) * 1000 > timeout_ms:
            app.exit(3)

    timer = QTimer(app)
    timer.timeout.connect(poll)
    timer.start(250)
    app._smoke_timer = timer  # keep a reference


class FreezeWatchdog:
    """Writes the stack of every thread to the log when the GUI stops responding.

    A timer in the GUI thread ticks every second; a background thread checks the last
    tick and, after ``limit`` seconds without one, dumps all stacks once (until the GUI
    responds again). Costs nothing while the app runs normally."""

    def __init__(self, app, log_path: Path, limit: float = 15.0):
        import threading
        import time

        from PySide6.QtCore import QTimer

        self.log_path = log_path
        self.limit = limit
        self._time = time
        self._last = time.monotonic()
        self._dumped = False
        self._timer = QTimer(app)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="freeze-watchdog", daemon=True)
        self._thread.start()

    def _tick(self) -> None:
        if self._dumped:
            self._append(f"--- GUI responsive again after "
                         f"{self._time.monotonic() - self._last:.0f} s ---\n")
        self._last = self._time.monotonic()
        self._dumped = False

    def _append(self, text: str) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass

    def _watch(self) -> None:
        import faulthandler

        while not self._stop.wait(2.0):
            stalled = self._time.monotonic() - self._last
            if stalled > self.limit and not self._dumped:
                self._dumped = True
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- GUI not responding for {stalled:.0f} s "
                                f"({self._time.strftime('%Y-%m-%d %H:%M:%S')}); stacks: ---\n")
                        f.flush()
                        faulthandler.dump_traceback(file=f, all_threads=True)
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        self._timer.stop()


def saved_language() -> str:
    """The language chosen in Settings: "en" (default) or "zh_CN"."""
    from PySide6.QtCore import QSettings

    from .i18n import LANGUAGE_KEY

    return str(QSettings("DrillFailPredict", "DrillFailPredict").value(LANGUAGE_KEY, "") or "")


def saved_theme() -> str:
    """The theme chosen in Settings (dark / light; dark when none)."""
    from PySide6.QtCore import QSettings

    from .ui.settings_dialog import THEME_KEY

    return str(QSettings("DrillFailPredict", "DrillFailPredict").value(THEME_KEY, "dark") or "dark")


def apply_language(app, code: str | None = None):
    """Use the chosen language (English unless Chinese is chosen in Settings) for
    GeoViewer's texts, and Qt's own translation of its standard buttons and dialogs."""
    from PySide6.QtCore import QLibraryInfo, QTranslator

    from . import i18n

    lang = i18n.set_language(code or saved_language() or i18n.DEFAULT)     # English unless Chinese is chosen
    old = getattr(app, "_qt_translator", None)
    if old is not None:
        app.removeTranslator(old)
        app._qt_translator = None
    if lang.code != i18n.DEFAULT and lang.qt:
        import PySide6

        qt = QTranslator(app)
        for folder in (QLibraryInfo.path(QLibraryInfo.TranslationsPath),
                       str(Path(PySide6.__file__).resolve().parent / "translations")):
            if qt.load(f"qtbase_{lang.qt}", folder):
                app.installTranslator(qt)
                app._qt_translator = qt
                break
    log.info("language: %s", lang.code)
    return lang


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    args = parse_args(argv[1:])
    smoke = args.smoke_test
    log_path = _setup_logging(smoke)
    from PySide6.QtWidgets import QApplication

    if os.name == "nt":
        # its own taskbar button and icon on Windows (run from source, the taskbar would group
        # the window under python.exe and show Python's icon)
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DrillFailPredict.DrillFailPredict")
        except Exception:  # noqa: BLE001 - only the taskbar grouping
            pass
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("DrillFailPredict")
    app.setOrganizationName("DrillFailPredict")
    # GeoViewer already open: "-w FOLDER" (or files) goes to that window instead of a new one.
    # The smoke test always runs on its own and never takes requests.
    forward = not smoke and (args.workspace or args.files)
    if forward:
        from .single_instance import make_request, send_to_running

        if send_to_running(make_request(args.workspace, args.files)):
            return 0
    apply_language(app)                       # before any window is made: texts are read once
    from .ui.icons import app_icon

    app.setWindowIcon(app_icon())             # every window, the taskbar and Alt+Tab
    watchdog = FreezeWatchdog(app, log_path)  # noqa: F841 - lives as long as the app

    from .ui.main_window import MainWindow
    from .ui.theme import apply_theme

    apply_theme(app, saved_theme())               # dark (default) or light, chosen in Settings

    win = MainWindow()
    win.quiet_errors = smoke
    if smoke:                       # test the bundled SQLite, not a MySQL server the settings may name
        from .io import database

        database.configure(database.DatabaseConfig(backend="sqlite"))
    win.showMaximized()
    instance = None
    if not smoke:                   # later starts with -w hand their workspace to this window
        from .single_instance import InstanceServer

        instance = InstanceServer(app)
        instance.requested.connect(win.handle_open_request)
        instance.listen()
    if args.workspace:             # a wrong folder: a status bar warning, the app opens empty
        win.open_command_line_workspace(args.workspace)
    elif not args.files and not smoke:
        win.restore_last_workspace()
    if args.files:
        win.open_paths(args.files)
    if smoke:
        _arm_smoke_test(app, win)
    code = app.exec()
    if instance is not None:
        instance.close()
    watchdog.stop()
    if smoke and win.load_errors and sys.stderr is not None:
        sys.stderr.write("\n".join(win.load_errors) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
