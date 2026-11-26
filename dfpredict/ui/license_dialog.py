"""License dialog: shows this PC's machine code and the current license, and lets the user
paste or load a license key. Opened at startup when there is no valid license, and from
Help > License.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, licensing
from ..i18n import tr
from .icons import icon


class LicenseDialog(QDialog):
    """Modal when the app has no license (blocks start until a key is entered or the user
    quits); non-modal from the menu."""

    def __init__(self, parent=None, must_license: bool = False):
        super().__init__(parent)
        self.must_license = must_license
        self.setWindowTitle(tr("GeoViewer License"))
        self.setWindowIcon(icon("certificate-outline", "#3d9bff"))
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(10)

        title = QLabel("GeoViewer")
        title.setStyleSheet("font-size: 16pt; font-weight: 600;")
        v.addWidget(title)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        # machine code
        mc_box = QFrame()
        mc_box.setFrameShape(QFrame.StyledPanel)
        mv = QVBoxLayout(mc_box)
        mv.setSpacing(4)
        lab = QLabel(tr("This computer's machine code"))
        lab.setStyleSheet("color: #9aa3ad;")
        mv.addWidget(lab)
        row = QHBoxLayout()
        self.machine = QLineEdit(licensing.machine_code())
        self.machine.setReadOnly(True)
        self.machine.setStyleSheet("font-family: Consolas, monospace; font-size: 12pt;")
        copy = QPushButton(icon("content-copy"), tr("Copy"))
        copy.setToolTip(tr("Copy the machine code to send to the vendor for a license"))
        copy.clicked.connect(self._copy_machine)
        row.addWidget(self.machine, 1)
        row.addWidget(copy)
        mv.addLayout(row)
        hint = QLabel(tr("Send this code to the vendor to get a license key bound to this computer. "
                         "It is based on the hardware and does not change when Windows is reinstalled."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa3ad; font-size: 9pt;")
        mv.addWidget(hint)
        v.addWidget(mc_box)

        v.addWidget(QLabel(tr("Paste your license key:")))
        self.key_edit = QPlainTextEdit()
        self.key_edit.setPlaceholderText("GEOV-XXXXXXXXXXXXXXXXXXXX-...")
        self.key_edit.setFixedHeight(84)
        self.key_edit.setStyleSheet("font-family: Consolas, monospace;")
        v.addWidget(self.key_edit)

        btns = QHBoxLayout()
        load = QPushButton(icon("file-upload-outline"), tr("Load from file…"))
        load.clicked.connect(self._load_file)
        btns.addWidget(load)
        btns.addStretch(1)
        self.activate = QPushButton(icon("check-circle-outline"), tr("Activate"))
        self.activate.setDefault(True)
        self.activate.clicked.connect(self._activate)
        btns.addWidget(self.activate)
        self.quit_btn = QPushButton(tr("Quit") if must_license else tr("Close"))
        self.quit_btn.clicked.connect(self.reject)
        btns.addWidget(self.quit_btn)
        v.addLayout(btns)

        foot = QLabel(f"GeoViewer {__version__}")
        foot.setStyleSheet("color: #6c747e; font-size: 8pt;")
        foot.setAlignment(Qt.AlignRight)
        v.addWidget(foot)
        self._show_status()

    def _show_status(self) -> None:
        st = licensing.refresh()
        if st.ok:
            lic = st.license
            left = lic.days_left()
            when = (tr("Perpetual license") if left is None
                    else tr("{n} days left (until {date:%Y-%m-%d})", n=left, date=lic.expires))
            self.status.setText(tr("<b style='color:#5fd38d'>Licensed</b> to {name} — {edition}. {when}.",
                                   name=lic.name, edition=lic.edition_name, when=when))
            self.quit_btn.setText(tr("Close"))
        else:
            self.status.setText(tr("<b style='color:#ff6b6b'>Not licensed</b> — {reason}. "
                                   "Enter a key below to activate.", reason=st.reason))

    def _copy_machine(self) -> None:
        QGuiApplication.clipboard().setText(self.machine.text())
        self.status.setText(tr("Machine code copied to the clipboard."))

    def _load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Open license key"), "",
                                              tr("License key (*.key *.txt);;All files (*)"))
        if path:
            try:
                self.key_edit.setPlainText(open(path, encoding="utf-8").read().strip())
            except OSError as e:
                QMessageBox.warning(self, tr("License"), tr("Could not read the file:\n{error}", error=e))

    def _activate(self) -> None:
        key = self.key_edit.toPlainText().strip()
        if not key:
            QMessageBox.information(self, tr("License"), tr("Paste the license key first."))
            return
        try:
            st = licensing.install(key)
        except licensing.LicenseError as e:
            QMessageBox.warning(self, tr("License"), tr("That key is not valid:\n{error}", error=e))
            return
        if st.ok:
            QMessageBox.information(self, tr("License"), tr("Activated. Licensed to {name}.", name=st.license.name))
            self.accept()
        else:
            QMessageBox.warning(self, tr("License"), tr("The key was saved but is not usable:\n{reason}",
                                                        reason=st.reason))
            self._show_status()


def ensure_licensed(parent=None) -> bool:
    """Show the dialog until the app is licensed or the user quits. Returns True when licensed."""
    if licensing.current().ok:
        return True
    dlg = LicenseDialog(parent, must_license=True)
    dlg.exec()
    return licensing.refresh().ok
