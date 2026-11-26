"""Application settings (stored in QSettings), shown in a dialog with one page per group:

* General  - start-up and automatic refresh
* Display  - view caption and background colour
* Database - where drillholes and production records are kept: SQLite (a file in the
  workspace folder) or a MySQL server
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import LANGUAGE_KEY, N_, available_languages, tr
from ..io.database import PROJECT_DB, DatabaseConfig, check_connection
from ..io.secret import protect, unprotect
from . import theme
from .icons import icon

DEFAULT_INTERVAL = 10  # seconds
DEFAULT_CAPTION_SIZE = 20          # points
DEFAULT_CAPTION_TRANSPARENCY = 40  # percent (0 = solid)
DEFAULT_BACKGROUND = "#000000"


def _bool(v, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    return bool(v)


THEME_KEY = "ui/theme"                       # dark / light

@dataclass
class AppSettings:
    auto_refresh: bool = False
    refresh_interval: int = DEFAULT_INTERVAL   # seconds
    restore_last_workspace: bool = True
    show_view_caption: bool = True             # "3D VIEW" / "XY PLANE" ... in the top-left corner
    caption_size: int = DEFAULT_CAPTION_SIZE
    caption_transparency: int = DEFAULT_CAPTION_TRANSPARENCY
    background: str = DEFAULT_BACKGROUND       # map views (main view, cross-section)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    language: str = "en"                       # "en" English or "zh_CN" Chinese; used at the next start
    theme: str = "dark"                        # dark / light (theme.THEMES); applied at once

    @property
    def view_background(self) -> str:
        """The views' background: the one chosen, or - left at the default - the theme's (white when light)."""
        if (self.background or DEFAULT_BACKGROUND).lower() == DEFAULT_BACKGROUND.lower():
            return theme.PALETTES.get(self.theme, theme.DARK)["VIEW_BACKGROUND"]
        return self.background

    @property
    def background_rgb(self) -> tuple[float, float, float]:
        c = QColor(self.view_background)
        if not c.isValid():
            c = QColor(DEFAULT_BACKGROUND)
        return c.redF(), c.greenF(), c.blueF()

    @classmethod
    def load(cls, qs: QSettings) -> "AppSettings":
        def _int(key, default, lo, hi):
            try:
                return max(lo, min(hi, int(qs.value(key, default))))
            except (TypeError, ValueError):
                return default

        def _str(key, default=""):
            v = qs.value(key, default)
            return str(v) if v is not None else default

        backend = _str("database/backend", "sqlite").lower()          # SQLite (workspace file) unless MySQL is chosen
        db = DatabaseConfig(
            backend=backend if backend in ("sqlite", "mysql") else "mysql",
            host=_str("database/mysql_host", "localhost"),
            port=_int("database/mysql_port", 3306, 1, 65535),
            database=_str("database/mysql_database", "geoviewer"),
            user=_str("database/mysql_user", "root"),
            password=unprotect(_str("database/mysql_password", "")),
        )
        return cls(
            auto_refresh=_bool(qs.value("refresh/auto"), False),
            refresh_interval=_int("refresh/interval", DEFAULT_INTERVAL, 1, 3600),
            restore_last_workspace=_bool(qs.value("workspace/restore_last"), True),
            show_view_caption=_bool(qs.value("view/caption"), True),
            caption_size=_int("view/caption_size", DEFAULT_CAPTION_SIZE, 8, 72),
            caption_transparency=_int("view/caption_transparency", DEFAULT_CAPTION_TRANSPARENCY, 0, 90),
            background=_str("view/background", DEFAULT_BACKGROUND) or DEFAULT_BACKGROUND,
            database=db,
            language=_str(LANGUAGE_KEY, "en") or "en",
            theme=_str(THEME_KEY, "dark") if _str(THEME_KEY, "dark") in theme.THEMES else "dark",
        )

    def save(self, qs: QSettings) -> None:
        qs.setValue("refresh/auto", self.auto_refresh)
        qs.setValue("refresh/interval", self.refresh_interval)
        qs.setValue("workspace/restore_last", self.restore_last_workspace)
        qs.setValue("view/caption", self.show_view_caption)
        qs.setValue("view/caption_size", self.caption_size)
        qs.setValue("view/caption_transparency", self.caption_transparency)
        qs.setValue("view/background", self.background)
        qs.setValue(LANGUAGE_KEY, self.language)
        qs.setValue(THEME_KEY, self.theme)
        d = self.database
        qs.setValue("database/backend", d.backend)
        qs.setValue("database/mysql_host", d.host)
        qs.setValue("database/mysql_port", d.port)
        qs.setValue("database/mysql_database", d.database)
        qs.setValue("database/mysql_user", d.user)
        qs.setValue("database/mysql_password", protect(d.password))   # encrypted for this Windows user


# --- the dialog --------------------------------------------------------------------------------
def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color: {theme.MUTED};")
    return lab


def _form(box: QGroupBox) -> QFormLayout:
    f = QFormLayout(box)
    f.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    f.setHorizontalSpacing(14)
    f.setVerticalSpacing(8)
    return f


class SettingsDialog(QDialog):
    PAGES = ((N_("General"), "cog-outline"), (N_("Display"), "monitor"), (N_("Database"), "database-outline"))
    _last_page = 0

    def __init__(self, settings: AppSettings, parent=None, workspace_folder=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Settings"))
        self.setMinimumSize(760, 650)
        self._settings = settings
        self._folder = workspace_folder

        self.nav = QListWidget()
        self.nav.setObjectName("settingsNav")
        self.nav.setFixedWidth(170)
        self.nav.setIconSize(QSize(20, 20))
        self.nav.setStyleSheet(
            f"QListWidget#settingsNav {{ background: {theme.PANEL}; border: none; border-right: 1px solid "
            f"{theme.BORDER}; padding-top: 8px; outline: 0; }}"
            f"QListWidget#settingsNav::item {{ padding: 9px 12px; border-left: 3px solid transparent; }}"
            f"QListWidget#settingsNav::item:hover {{ background: {theme.HOVER}; }}"
            f"QListWidget#settingsNav::item:selected {{ background: {theme.ACCENT_DIM}; color: #ffffff; "
            f"border-left: 3px solid {theme.ACCENT}; }}")
        self.pages = QStackedWidget()
        for (name, ic), page in zip(self.PAGES, (self._general_page(), self._display_page(),
                                                 self._database_page())):
            QListWidgetItem(icon(ic), tr(name), self.nav)
            scroll = QScrollArea()                    # small screens: the page scrolls, nothing overlaps
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            self.pages.addWidget(scroll)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.currentRowChanged.connect(lambda i: setattr(SettingsDialog, "_last_page", i))
        self.nav.setCurrentRow(SettingsDialog._last_page)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.nav)
        body.addWidget(self.pages, 1)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 12)
        v.addLayout(body, 1)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {theme.BORDER};")
        v.addWidget(line)
        bl = QHBoxLayout()
        bl.setContentsMargins(16, 6, 16, 0)
        bl.addWidget(buttons)
        v.addLayout(bl)

    # --- page frame ------------------------------------------------------------------------
    @staticmethod
    def _page(title: str, intro: str) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(22, 16, 22, 8)
        v.setSpacing(12)
        t = QLabel(title)
        t.setStyleSheet("font-size: 15pt; font-weight: 600;")
        v.addWidget(t)
        v.addWidget(_hint(intro))
        return w, v

    # --- General -------------------------------------------------------------------------------
    def _general_page(self) -> QWidget:
        s = self._settings
        w, v = self._page(tr("General"), tr("What GeoViewer does when it starts, and how it keeps up with "
                                            "files that change on disk."))
        g = QGroupBox(tr("Language"))
        f = _form(g)
        self.language = QComboBox()
        for lang in available_languages():
            label = lang.name if not lang.english_name or lang.english_name == lang.name else                 f"{lang.name}  ({lang.english_name})"
            self.language.addItem(label, lang.code)
        self.language.setCurrentIndex(max(self.language.findData(s.language or "en"), 0))
        f.addRow(tr("Language"), self.language)
        f.addRow(_hint(tr("Used the next time GeoViewer starts.")))
        v.addWidget(g)

        g = QGroupBox(tr("Start-up"))
        f = _form(g)
        self.restore = QCheckBox(tr("Open the last workspace when GeoViewer starts"))
        self.restore.setChecked(s.restore_last_workspace)
        f.addRow(self.restore)
        v.addWidget(g)

        g = QGroupBox(tr("Refresh"))
        f = _form(g)
        self.auto = QCheckBox(tr("Reload changed files automatically"))
        self.auto.setChecked(s.auto_refresh)
        self.interval = QSpinBox()
        self.interval.setRange(1, 3600)
        self.interval.setSuffix(" s")
        self.interval.setValue(s.refresh_interval)
        self.interval.setEnabled(s.auto_refresh)
        self.interval.setMaximumWidth(120)
        self.auto.toggled.connect(self.interval.setEnabled)
        f.addRow(self.auto)
        f.addRow(tr("Check every"), self.interval)
        f.addRow(_hint(tr("F5 always refreshes by hand: layers whose files changed are reloaded, new files "
                          "in the workspace folder are added.")))
        v.addWidget(g)
        v.addStretch(1)
        return w

    # --- Display -------------------------------------------------------------------------------
    def _display_page(self) -> QWidget:
        s = self._settings
        w, v = self._page(tr("Display"), tr("How the map views look (main 3D / 2D view and the cross-section)."))
        g = QGroupBox(tr("Theme"))
        f = _form(g)
        self.theme = QComboBox()
        for key, text in theme.THEMES.items():
            self.theme.addItem(tr(text), key)
        self.theme.setCurrentIndex(max(self.theme.findData(s.theme), 0))
        f.addRow(tr("Theme"), self.theme)
        f.addRow(_hint(tr("Light suits screenshots for papers and reports. Applied at once; the views turn white "
                          "too unless a background colour was chosen. Some icons change at the next start.")))
        v.addWidget(g)
        g = QGroupBox(tr("View caption"))
        f = _form(g)
        self.caption = QCheckBox(tr("Show the view name in the top-left corner (3D VIEW, XY PLANE, ...)"))
        self.caption.setChecked(s.show_view_caption)
        f.addRow(self.caption)
        self.caption_size = QSpinBox()
        self.caption_size.setRange(8, 72)
        self.caption_size.setSuffix(" pt")
        self.caption_size.setValue(s.caption_size)
        self.caption_transparency = QSpinBox()
        self.caption_transparency.setRange(0, 90)
        self.caption_transparency.setSingleStep(10)
        self.caption_transparency.setSuffix(" %")
        self.caption_transparency.setValue(s.caption_transparency)
        self.caption_transparency.setToolTip(tr("0 % = solid text"))
        for sp in (self.caption_size, self.caption_transparency):
            sp.setMaximumWidth(120)
            sp.setEnabled(s.show_view_caption)
            self.caption.toggled.connect(sp.setEnabled)
        f.addRow(tr("Font size"), self.caption_size)
        f.addRow(tr("Transparency"), self.caption_transparency)
        v.addWidget(g)

        g = QGroupBox(tr("Background"))
        f = _form(g)
        self.background = s.background
        self.bg_button = QPushButton()
        self.bg_button.setObjectName("colorButton")
        self.bg_button.setMinimumWidth(140)
        self.bg_button.setToolTip(tr("Click to choose the background colour"))
        self.bg_button.clicked.connect(self._choose_background)
        bg_default = QToolButton()
        bg_default.setText(tr("Default"))
        bg_default.setToolTip(tr("Black"))
        bg_default.clicked.connect(lambda: self._set_background(DEFAULT_BACKGROUND))
        row = QHBoxLayout()
        row.addWidget(self.bg_button)
        row.addWidget(bg_default)
        row.addStretch(1)
        f.addRow(tr("Colour"), row)
        v.addWidget(g)
        self._set_background(self.background)
        v.addStretch(1)
        return w

    def _set_background(self, color: str) -> None:
        self.background = QColor(color).name()
        self.bg_button.setText(self.background.upper())
        fg = "#000000" if QColor(self.background).lightness() > 140 else "#ffffff"
        self.bg_button.setStyleSheet(f"QPushButton#colorButton {{ background: {self.background}; color: {fg}; "
                                     "min-width: 120px; padding: 5px 10px; }")

    def _choose_background(self) -> None:
        c = QColorDialog.getColor(QColor(self.background), self, tr("Background colour"))
        if c.isValid():
            self._set_background(c.name())

    # --- Database ------------------------------------------------------------------------------
    def _database_page(self) -> QWidget:
        d = self._settings.database
        w, v = self._page(tr("Database"), tr("Where the drillholes (drillhole_* tables) and the production "
                                             "records (prod_* tables) are kept. GeoViewer connects at start."))
        g = QGroupBox(tr("Use"))
        gv = QVBoxLayout(g)
        self.r_mysql = QRadioButton(tr("MySQL server  (default - shared by several computers)"))
        self.r_sqlite = QRadioButton(tr("SQLite file in the workspace folder  (one computer, nothing to install)"))
        self._backend = QButtonGroup(self)
        self._backend.addButton(self.r_sqlite)
        self._backend.addButton(self.r_mysql)
        (self.r_mysql if d.is_mysql else self.r_sqlite).setChecked(True)
        gv.addWidget(self.r_mysql)
        gv.addWidget(self.r_sqlite)
        where = (f"{self._folder}\\{PROJECT_DB}" if self._folder
                 else tr("{file} in the workspace folder", file=PROJECT_DB))
        self.sqlite_info = _hint(tr("File: {where}  (connected and set up automatically)", where=where))
        self.sqlite_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sqlite_info.setContentsMargins(24, 0, 0, 6)
        gv.addWidget(self.sqlite_info)
        v.addWidget(g)

        self.g_mysql = QGroupBox(tr("MySQL server"))
        gl = QVBoxLayout(self.g_mysql)          # wrapped labels go below the form (in a form they get clipped)
        gl.setSpacing(8)
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        f.setHorizontalSpacing(14)
        f.setVerticalSpacing(8)
        gl.addLayout(f)
        self.m_host = QLineEdit(d.host)
        self.m_host.setPlaceholderText(tr("server name or IP address, e.g. 192.168.1.20"))
        self.m_port = QSpinBox()
        self.m_port.setRange(1, 65535)
        self.m_port.setValue(d.port)
        self.m_port.setMaximumWidth(120)
        self.m_db = QLineEdit(d.database)
        self.m_db.setPlaceholderText(tr("e.g. geoviewer"))
        self.m_user = QLineEdit(d.user)
        self.m_pass = QLineEdit(d.password)
        self.m_pass.setEchoMode(QLineEdit.Password)
        eye = self.m_pass.addAction(icon("eye-outline"), QLineEdit.TrailingPosition)
        eye.setToolTip(tr("Show / hide the password"))
        eye.triggered.connect(lambda: self.m_pass.setEchoMode(
            QLineEdit.Normal if self.m_pass.echoMode() == QLineEdit.Password else QLineEdit.Password))
        f.addRow(tr("Server"), self.m_host)
        f.addRow(tr("Port"), self.m_port)
        f.addRow(tr("Database"), self.m_db)
        f.addRow(tr("User"), self.m_user)
        f.addRow(tr("Password"), self.m_pass)
        row = QHBoxLayout()
        self.b_test = QPushButton(icon("lan-connect"), tr("Connect"))
        self.b_test.setToolTip(tr("Connect with these settings; missing tables / fields are added"))
        self.b_test.clicked.connect(self.connect_database)
        row.addWidget(self.b_test)
        row.addStretch(1)
        gl.addLayout(row)
        self.m_status = QLabel("")
        self.m_status.setWordWrap(True)
        self.m_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        gl.addWidget(self.m_status)
        gl.addWidget(_hint(tr("The database must exist (make an empty one). When connected, missing tables and "
                              "fields are added automatically; existing data is never removed. The password is "
                              "stored encrypted for your Windows user.")))
        v.addWidget(self.g_mysql)
        v.addStretch(1)
        self.r_mysql.toggled.connect(self._backend_changed)
        self._backend_changed()
        return w

    def _backend_changed(self) -> None:
        mysql = self.r_mysql.isChecked()
        self.g_mysql.setEnabled(mysql)
        self.sqlite_info.setEnabled(not mysql)
        self.m_status.setText("")

    def database_config(self) -> DatabaseConfig:
        return DatabaseConfig(backend="mysql" if self.r_mysql.isChecked() else "sqlite",
                              host=self.m_host.text().strip(), port=self.m_port.value(),
                              database=self.m_db.text().strip(), user=self.m_user.text().strip(),
                              password=self.m_pass.text())

    def _show_status(self, ok: bool, text: str) -> None:
        self.m_status.setStyleSheet(f"color: {'#5fd38d' if ok else theme.DANGER};")
        self.m_status.setText(("✔ " if ok else "✖ ") + text)

    def connect_database(self) -> bool:
        """Connect with the settings shown and set up missing tables / fields (and a missing
        MySQL database). Shows the result; True when connected."""
        from pathlib import Path

        from ..io.database import Target
        from ..project_db import init_project_database

        cfg = self.database_config()
        if cfg.is_mysql:
            missing = [tr(n) for n, val in (("Server", cfg.host), ("Database", cfg.database), ("User", cfg.user))
                       if not val]
            if missing:
                self._show_status(False, tr("MySQL needs: {fields}.", fields=", ".join(missing)))
                return False
            target = Target("mysql", config=cfg)
        elif self._folder:
            target = Target("sqlite", path=Path(self._folder) / PROJECT_DB)
        else:
            self._show_status(True, tr("SQLite: {file} is used in the workspace folder; it is connected "
                                       "when a workspace is opened.", file=PROJECT_DB))
            return True
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            notes = init_project_database(target, self._folder)
            text = check_connection(cfg, brief=True) if cfg.is_mysql else tr("Connected to {target}.", target=target.label)
        except Exception as e:  # noqa: BLE001 - unreachable, wrong password, no rights, read-only folder
            QGuiApplication.restoreOverrideCursor()
            self._show_status(False, str(e) or type(e).__name__)
            return False
        QGuiApplication.restoreOverrideCursor()
        tables = [n for n in notes if n.startswith("made table ")]
        rest = [n for n in notes if n not in tables] + ([tr("made {n} tables", n=len(tables))] if len(tables) > 1
                                                        else tables)
        if cfg.is_mysql:
            text = text.rstrip(".。") + " - " + tr("database {name}.", name=cfg.database)   # brief: one sentence
        self._show_status(True, text + " " + (tr("Set up: {notes}.", notes="; ".join(rest)) if rest
                                              else tr("All tables are ready.")))
        return True

    def accept(self) -> None:
        cfg = self.database_config()
        if cfg.is_mysql and cfg != self._settings.database:
            missing = [tr(n) for n, val in (("Server", cfg.host), ("Database", cfg.database), ("User", cfg.user))
                       if not val]
            if missing:
                self.nav.setCurrentRow(2)
                QMessageBox.warning(self, tr("Settings"), tr("MySQL needs: {fields}.", fields=", ".join(missing)))
                return
            if not self.connect_database():
                self.nav.setCurrentRow(2)
                ans = QMessageBox.question(
                    self, tr("Settings"), tr("GeoViewer cannot connect to this MySQL database:\n\n"
                                             "{error}\n\nUse these settings anyway?", error=self.m_status.text()[2:]))
                if ans != QMessageBox.Yes:
                    return
        if (self.language.currentData() or "") != (self._settings.language or ""):
            QMessageBox.information(self, tr("Settings"), tr("The new language is used the next time GeoViewer "
                                                             "starts."))
        super().accept()

    def result_settings(self) -> AppSettings:
        return AppSettings(self.auto.isChecked(), self.interval.value(), self.restore.isChecked(),
                           self.caption.isChecked(), self.caption_size.value(),
                           self.caption_transparency.value(), self.background, self.database_config(),
                           self.language.currentData() or "en", self.theme.currentData() or "dark")
