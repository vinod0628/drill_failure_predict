"""Themes: Fusion style with a dark (default) or a light palette and a stylesheet. The light theme suits screenshots
for print (papers, reports). The module-level colours (BG, TEXT, MUTED ...) are those of the theme in use: widgets
read them when they are made."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from ..i18n import N_

THEMES = {"dark": N_("Dark"), "light": N_("Light")}   # shown: tr()
DARK = {"BG": "#1b1d21", "PANEL": "#22252a", "PANEL_ALT": "#282c32", "INPUT": "#2d3138", "BORDER": "#363b43",
        "TEXT": "#e3e6ea", "MUTED": "#9aa3ad", "ACCENT": "#3d9bff", "ACCENT_DIM": "#24476f", "HOVER": "#323740",
        "DANGER": "#ff6b6b", "SCROLL": "#454b54", "SCROLL_HOVER": "#5a616b", "BORDER_HOVER": "#4a515b",
        "DISABLED": "#5f666f", "FLOAT": "rgba(34, 37, 42, 235)", "IDENTIFY": "rgba(30, 33, 38, 242)",
        "ICON_ACTIVE": "#ffffff", "VIEW_BACKGROUND": "#000000"}
LIGHT = {"BG": "#eef0f3", "PANEL": "#ffffff", "PANEL_ALT": "#f6f7f9", "INPUT": "#ffffff", "BORDER": "#c6ccd4",
         "TEXT": "#1f2328", "MUTED": "#5c6672", "ACCENT": "#0a66c2", "ACCENT_DIM": "#cfe2f8", "HOVER": "#e6eaf0",
         "DANGER": "#c62828", "SCROLL": "#b9c0c9", "SCROLL_HOVER": "#9aa3ad", "BORDER_HOVER": "#9aa3ad",
         "DISABLED": "#a3aab3", "FLOAT": "rgba(255, 255, 255, 240)", "IDENTIFY": "rgba(255, 255, 255, 245)",
         "ICON_ACTIVE": "#000000", "VIEW_BACKGROUND": "#ffffff"}
PALETTES = {"dark": DARK, "light": LIGHT}
CURRENT = "dark"
globals().update(DARK)                                  # BG, PANEL, TEXT ... of the theme in use


def _stylesheet(BG, PANEL, PANEL_ALT, INPUT, BORDER, TEXT, MUTED, ACCENT, ACCENT_DIM, HOVER, DANGER, SCROLL,
                SCROLL_HOVER, BORDER_HOVER, DISABLED, FLOAT, IDENTIFY, **_unused) -> str:
    return f"""
* {{ font-family: "Segoe UI"; font-size: 9pt; color: {TEXT}; }}
QMainWindow, QWidget#central {{ background: {BG}; }}
QWidget#leftPanel {{ background: {PANEL}; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical {{ height: 4px; }}
QSplitter::handle:hover {{ background: {ACCENT_DIM}; }}

QLabel#panelTitle {{
    color: {MUTED}; font-size: 8pt; font-weight: 600; letter-spacing: 1px;
    padding: 2px 0px;
}}
QFrame#panelHeader {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}

QTreeWidget, QListWidget {{
    background: {PANEL}; border: none; outline: 0;
    show-decoration-selected: 1;
}}
QTreeWidget::item, QListWidget::item {{ height: 26px; padding-left: 2px; border-radius: 4px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {HOVER}; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: {ACCENT_DIM}; color: {TEXT}; }}
QTreeWidget::indicator, QCheckBox::indicator {{ width: 14px; height: 14px; }}

QScrollArea {{ background: {PANEL}; border: none; }}
QScrollArea > QWidget > QWidget {{ background: {PANEL}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {SCROLL}; border-radius: 4px; min-height: 24px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: {SCROLL_HOVER}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {SCROLL}; border-radius: 4px; min-width: 24px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 6px; margin-top: 14px; padding: 8px 6px 6px 6px;
    background: {PANEL_ALT};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {MUTED};
    font-size: 8pt; font-weight: 600;
}}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {INPUT}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 3px 6px; min-height: 18px; selection-background-color: {ACCENT_DIM};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {BORDER_HOVER}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {INPUT}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DIM};
}}

QPushButton {{
    background: {INPUT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 10px;
}}
QPushButton:hover {{ background: {HOVER}; border-color: {BORDER_HOVER}; }}
QPushButton:pressed {{ background: {ACCENT_DIM}; }}
QPushButton#colorButton {{ min-width: 46px; min-height: 18px; padding: 2px; }}

QToolButton#headerButton {{
    background: transparent; border: none; border-radius: 4px; padding: 3px;
}}
QToolButton#headerButton:hover {{ background: {HOVER}; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
}}

QToolTip {{
    background: {PANEL_ALT}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px;
    border-radius: 4px;
}}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 6px; }}
QMenu {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMessageBox {{ background: {PANEL}; }}

/* Floating toolbar */
QFrame#floatingBar {{
    background: {FLOAT}; border: 1px solid {BORDER}; border-radius: 12px;
}}
QFrame#floatingBar QToolButton {{
    background: transparent; border: none; border-radius: 8px; padding: 0px;
    min-width: 36px; max-width: 36px; min-height: 34px; max-height: 34px;
}}
QFrame#floatingBar QToolButton:hover {{ background: {HOVER}; }}
QFrame#floatingBar QToolButton:checked {{ background: {ACCENT_DIM}; }}
QFrame#floatingBar QToolButton:disabled {{ background: transparent; }}
QFrame#barSeparator {{ background: {BORDER}; min-height: 1px; max-height: 1px; margin: 3px 6px; }}
QLabel#barGrip {{ padding: 2px; }}

/* Identify result panel */
QFrame#identifyPanel {{
    background: {IDENTIFY}; border: 1px solid {ACCENT_DIM}; border-radius: 10px;
}}
QLabel#identifyTitle {{ color: {TEXT}; font-weight: 600; }}
QLabel#identifySub {{ color: {MUTED}; font-size: 8pt; }}
QScrollArea#identifyScroll, QScrollArea#identifyScroll > QWidget > QWidget {{ background: transparent; }}
QLabel#identifyBody {{ background: transparent; }}
"""


STYLESHEET = _stylesheet(**DARK)


def apply_theme(app: QApplication, name: str = "dark") -> str:
    """Use the dark or the light theme (unknown names: dark); returns the name used."""
    global CURRENT, STYLESHEET
    name = name if name in PALETTES else "dark"
    c = PALETTES[name]
    CURRENT = name
    globals().update(c)
    STYLESHEET = _stylesheet(**c)
    app.setStyle("Fusion")
    pal = QPalette()
    q = QColor
    pal.setColor(QPalette.Window, q(c["BG"]))
    pal.setColor(QPalette.WindowText, q(c["TEXT"]))
    pal.setColor(QPalette.Base, q(c["PANEL"]))
    pal.setColor(QPalette.AlternateBase, q(c["PANEL_ALT"]))
    pal.setColor(QPalette.ToolTipBase, q(c["PANEL_ALT"]))
    pal.setColor(QPalette.ToolTipText, q(c["TEXT"]))
    pal.setColor(QPalette.Text, q(c["TEXT"]))
    pal.setColor(QPalette.Button, q(c["INPUT"]))
    pal.setColor(QPalette.ButtonText, q(c["TEXT"]))
    pal.setColor(QPalette.BrightText, q(c["DANGER"]))
    pal.setColor(QPalette.Highlight, q(c["ACCENT_DIM"]))
    pal.setColor(QPalette.HighlightedText, q(c["TEXT"]))
    pal.setColor(QPalette.Link, q(c["ACCENT"]))
    pal.setColor(QPalette.PlaceholderText, q(c["MUTED"]))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        pal.setColor(QPalette.Disabled, role, q(c["DISABLED"]))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)
    return name


def apply_dark_theme(app: QApplication) -> None:
    apply_theme(app, "dark")


def apply_light_theme(app: QApplication) -> None:
    apply_theme(app, "light")
