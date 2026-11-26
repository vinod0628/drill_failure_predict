"""Settings > Display > Theme: dark (default) or light, saved, applied at once; the views follow the theme."""

from __future__ import annotations

import os


def test_theme_setting(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from dfpredict.ui import theme
    from dfpredict.ui.settings_dialog import AppSettings, SettingsDialog

    s = AppSettings()
    assert s.theme == "dark" and s.view_background.lower() == "#000000"
    dlg = SettingsDialog(s)
    assert [dlg.theme.itemData(i) for i in range(dlg.theme.count())] == ["dark", "light"]
    dlg.theme.setCurrentIndex(1)
    r = dlg.result_settings()
    assert r.theme == "light" and r.view_background.lower() == "#ffffff"      # views turn white too
    r.background = "#203040"                                                   # a chosen colour wins
    assert r.view_background == "#203040"
    ini = str(tmp_path / "s.ini")
    r.save(QSettings(ini, QSettings.IniFormat))
    assert AppSettings.load(QSettings(ini, QSettings.IniFormat)).theme == "light"
    try:
        theme.apply_theme(app, "light")
        assert theme.CURRENT == "light" and theme.TEXT == theme.LIGHT["TEXT"]
    finally:
        theme.apply_theme(app, "dark")
    assert theme.CURRENT == "dark"
    dlg.close()
