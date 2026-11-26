"""One GeoViewer per user: ``GeoViewer.exe -w FOLDER`` while GeoViewer already runs hands
the workspace to the running window, which closes its workspace and opens the new one;
no second window is started.

The running app listens on a local socket (a named pipe on Windows, private to the user).
A new start with ``-w`` (or files) first tries to connect: when a running app takes the
request and answers, the new process quits. When nothing answers, it starts normally and
becomes the listening one. Starting without ``-w`` or files opens a new window as before.

Message: one line of UTF-8 JSON ``{"workspace": "<absolute folder>" | null, "files": [...]}``,
answered with ``ok\\n``.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger(__name__)

CONNECT_MS = 500       # a running app accepts at once; nothing there fails at once too
REPLY_MS = 5000        # the running app may be busy for a moment (e.g. loading)
ACK = b"ok\n"
MAX_MESSAGE = 1 << 20


def server_name() -> str:
    """The socket name: one per Windows user (named pipes are visible machine-wide)."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no user name: still one name for this account
        user = os.environ.get("USERNAME", "")
    return "DrillFailPredict-" + hashlib.sha1(user.lower().encode("utf-8")).hexdigest()[:16]


def make_request(workspace: str | None, files: list[str]) -> dict:
    """Paths made absolute here: the running app has another current folder."""
    return {"workspace": str(Path(workspace).resolve()) if workspace else None,
            "files": [str(Path(f).resolve()) for f in files]}


def _allow_foreground() -> None:
    """Windows only lets the app the user just started take the foreground; hand that
    right to the running GeoViewer so its window can come to the front."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.user32.AllowSetForegroundWindow(-1)  # ASFW_ANY
    except Exception:  # noqa: BLE001 - only the window activation
        pass


def send_to_running(request: dict, name: str | None = None) -> bool:
    """Hand the request to a running GeoViewer. True: it took it (this process should quit)."""
    sock = QLocalSocket()
    sock.connectToServer(name or server_name())
    if not sock.waitForConnected(CONNECT_MS):
        return False
    _allow_foreground()
    sock.write(json.dumps(request).encode("utf-8") + b"\n")
    sock.flush()
    sock.waitForBytesWritten(REPLY_MS)
    reply = b""
    while ACK not in reply and sock.waitForReadyRead(REPLY_MS):
        reply += bytes(sock.readAll())
    sock.disconnectFromServer()
    if ACK in reply:
        log.info("handed to the running GeoViewer: %s", request)
        return True
    log.warning("a running GeoViewer did not answer; starting a new one")
    return False


class InstanceServer(QObject):
    """Listens for requests from later starts; emits ``requested(workspace, files)``."""

    requested = Signal(object, list)

    def __init__(self, parent: QObject | None = None, name: str | None = None):
        super().__init__(parent)
        self.name = name or server_name()
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.UserAccessOption)
        self.server.newConnection.connect(self._on_connection)

    def listen(self) -> bool:
        if self.server.listen(self.name):
            return True
        if os.name != "nt":                       # a socket file left by a crashed app
            QLocalServer.removeServer(self.name)
            if self.server.listen(self.name):
                return True
        log.warning("single instance: cannot listen on %s: %s", self.name, self.server.errorString())
        return False

    def close(self) -> None:
        self.server.close()

    def _on_connection(self) -> None:
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            sock.setProperty("buf", b"")
            sock.readyRead.connect(lambda s=sock: self._on_ready(s))
            sock.disconnected.connect(sock.deleteLater)
            QTimer.singleShot(REPLY_MS * 2, sock, sock.abort)   # a client that never finishes

    def _on_ready(self, sock: QLocalSocket) -> None:
        buf = bytes(sock.property("buf") or b"") + bytes(sock.readAll())
        if b"\n" not in buf:
            if len(buf) > MAX_MESSAGE:
                sock.abort()
            else:
                sock.setProperty("buf", buf)
            return
        line = buf.split(b"\n", 1)[0]
        try:
            msg = json.loads(line.decode("utf-8"))
            workspace = msg.get("workspace") or None
            files = [str(f) for f in msg.get("files") or []]
        except (ValueError, AttributeError, TypeError) as e:
            log.warning("single instance: bad request %r: %s", line[:200], e)
            sock.abort()
            return
        sock.write(ACK)
        sock.flush()
        log.info("request from a new start: workspace=%s files=%s", workspace, files)
        self.requested.emit(workspace, files)
