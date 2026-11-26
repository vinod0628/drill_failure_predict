"""Import from Excel (Drillhole Manager, Survey): save the template, choose the filled-in
workbook, see the check, Import.

The window only drives what it is given: a function to write the template, one to read and
check a workbook (giving a plan with ``summary(replace)`` and ``problems(replace)``), and one to
write the plan. Nothing is written until Import; Import is possible only without errors.
"""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextBrowser,
    QVBoxLayout,
)

from ..i18n import tr
from . import theme
from .icons import icon

MAX_LISTED = 200                       # problems listed in the window (the count is always shown)
WARN = "#f0b429"
OK = "#5cc98b"


class ExcelImportDialog(QDialog):
    def __init__(self, parent, *, title: str, intro: str, existing: str, folder: str, template_name: str,
                 save_template, read, apply, result_text):
        """``existing``: what is already in the database ("Holes", "Pits") for the choice of
        replacing or skipping them; ``folder``: where the file dialogs start."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(780, 560)
        self._save_template, self._read, self._apply, self._result_text = save_template, read, apply, result_text
        self.folder = folder
        self.template_name = template_name
        self.plan = None
        self.counts = None                 # the counts after a successful import

        hint = QLabel(intro)
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        self.b_template = QPushButton(icon("file-excel-outline"), tr("Save template ..."))
        self.b_template.setToolTip(tr("Save an empty workbook with the columns to fill in and an instructions sheet"))
        self.b_template.clicked.connect(self.save_template)
        self.path = QLineEdit()
        self.path.setPlaceholderText(tr("The filled-in workbook (.xlsx)"))
        self.path.editingFinished.connect(self.check)
        b_browse = QPushButton(icon("folder-open-outline"), tr("Browse ..."))
        b_browse.clicked.connect(self.browse)
        self.b_check = QPushButton(icon("refresh"), tr("Check again"))
        self.b_check.setToolTip(tr("Read the workbook again (e.g. after correcting it in Excel)"))
        self.b_check.clicked.connect(self.check)
        self.r_replace = QRadioButton(tr("Replace them with the file's data"))
        self.r_skip = QRadioButton(tr("Leave them as they are (skip)"))
        self.r_replace.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.r_replace)
        grp.addButton(self.r_skip)
        self.r_replace.toggled.connect(lambda _on: self._show())

        grid = QGridLayout()
        grid.addWidget(QLabel(tr("1. Template")), 0, 0)
        trow = QHBoxLayout()
        trow.addWidget(self.b_template)
        note = QLabel(tr("fill it in Excel and save it"))
        note.setStyleSheet(f"color: {theme.MUTED};")
        trow.addWidget(note)
        trow.addStretch(1)
        grid.addLayout(trow, 0, 1)
        grid.addWidget(QLabel(tr("2. Workbook")), 1, 0)
        prow = QHBoxLayout()
        prow.addWidget(self.path, 1)
        prow.addWidget(b_browse)
        prow.addWidget(self.b_check)
        grid.addLayout(prow, 1, 1)
        grid.addWidget(QLabel(existing), 2, 0)
        mrow = QHBoxLayout()
        mrow.addWidget(self.r_replace)
        mrow.addWidget(self.r_skip)
        mrow.addStretch(1)
        grid.addLayout(mrow, 2, 1)
        grid.setColumnStretch(1, 1)

        self.report = QTextBrowser()
        self.report.setOpenLinks(False)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.b_import = self.buttons.button(QDialogButtonBox.Ok)
        self.b_import.setText(tr("Import"))
        self.b_import.setIcon(icon("database-import-outline"))
        self.buttons.accepted.connect(self.do_import)
        self.buttons.rejected.connect(self.reject)
        v = QVBoxLayout(self)
        v.addWidget(hint)
        v.addLayout(grid)
        v.addWidget(QLabel(tr("3. Check")))
        v.addWidget(self.report, 1)
        v.addWidget(self.buttons)
        self._show()

    @property
    def replace(self) -> bool:
        return self.r_replace.isChecked()

    # --- the template --------------------------------------------------------------------------
    def save_template(self) -> None:
        start = str(Path(self.folder) / self.template_name)
        path, _ = QFileDialog.getSaveFileName(self, tr("Save template"), start, tr("Excel workbook (*.xlsx)"))
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            self._save_template(path)
        except PermissionError:
            QMessageBox.warning(self, tr("Save template"), tr("{path}\n\nThe file cannot be written - is it open in "
                                                              "Excel? Close it and try again.", path=path))
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("Save template"), tr("The template was not saved:\n{error}", error=e))
            return
        self.folder = str(Path(path).parent)
        if not self.path.text().strip():
            self.path.setText(path)                       # the file to fill in is the one to import
            self.plan = None
            self._show()
        if QMessageBox.question(self, tr("Save template"), tr("Template saved:\n{path}\n\nOpen it in Excel now?",
                                                              path=path)) == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # --- the workbook ----------------------------------------------------------------------------
    def browse(self) -> None:
        start = self.path.text().strip() or self.folder
        path, _ = QFileDialog.getOpenFileName(self, tr("Choose the filled-in workbook"), start,
                                              tr("Excel workbook (*.xlsx *.xlsm)"))
        if path:
            self.path.setText(path)
            self.folder = str(Path(path).parent)
            self.check()

    def check(self) -> None:
        path = self.path.text().strip()
        self.plan = None
        self._error = None
        if path:
            try:
                self.plan = self._read(path)
            except PermissionError:
                self._error = tr("The file cannot be read - is it open in Excel with changes not saved? Save it "
                                 "(or close it) and click Check again.")
            except FileNotFoundError:
                self._error = tr("There is no file {path}", path=path)
            except Exception as e:  # noqa: BLE001 - not a workbook, damaged, ...
                self._error = tr("The workbook cannot be read:\n{error}", error=e)
        self._show()

    def _show(self) -> None:
        h = []
        err = getattr(self, "_error", None)
        ok = False
        if err:
            h.append(f"<p style='color:{theme.DANGER}'>{_esc(err)}</p>")
        elif self.plan is None:
            h.append(f"<p style='color:{theme.MUTED}'>" + _esc(tr("Choose the filled-in workbook - it is checked "
                                                                  "here before anything is imported.")) + "</p>")
        else:
            errors, warnings = self.plan.problems(self.replace)
            h.append("<p>" + "<br>".join(_esc(s) for s in self.plan.summary(self.replace)) + "</p>")
            if errors:
                h.append(f"<p style='color:{theme.DANGER}'><b>"
                         + _esc(tr("Errors: {n:,} - correct them in the workbook, save it and click Check again. "
                                   "Nothing is imported while there are errors.", n=len(errors))) + "</b></p>")
                h.append(_list(errors, theme.DANGER))
            if warnings:
                h.append(f"<p style='color:{WARN}'><b>" + _esc(tr("Warnings: {n:,} - please check:", n=len(warnings)))
                         + "</b></p>")
                h.append(_list(warnings, WARN))
            ok = not errors and bool(self.plan.todo(self.replace))
            if ok:
                h.append(f"<p style='color:{OK}'><b>" + _esc(tr("Ready - click Import.")) + "</b></p>")
            elif not errors:
                h.append(f"<p style='color:{theme.MUTED}'>" + _esc(tr("Nothing to import.")) + "</p>")
        self.report.setHtml("".join(h))
        self.b_import.setEnabled(ok)
        self.b_check.setEnabled(bool(self.path.text().strip()))

    # --- import ------------------------------------------------------------------------------------
    def do_import(self) -> None:
        if self.plan is None:
            return
        try:
            self.counts = self._apply(self.plan, self.replace)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, self.windowTitle(), tr("Nothing was imported:\n{error}", error=e))
            return
        self.accept()

    def result_message(self) -> str:
        return self._result_text(self.counts) if self.counts is not None else ""


def _esc(text) -> str:
    return html.escape(str(text)).replace("\n", "<br>")


def _list(items, color) -> str:
    more = len(items) - MAX_LISTED
    rows = "".join(f"<li>{_esc(t)}</li>" for t in items[:MAX_LISTED])
    if more > 0:
        rows += f"<li>{_esc(tr('... and {n} more', n=more))}</li>"
    return f"<ul style='color:{color}; margin-top:0'>{rows}</ul>"
