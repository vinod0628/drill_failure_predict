"""Translations: tr / trn, choosing a language, adding one, the catalogs' completeness, Qt's own
translation, the VTK font, and the windows in Chinese."""

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict import i18n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import i18n_extract  # noqa: E402


def test_english_is_the_key_and_the_fallback():
    assert i18n.language().code == "en"
    assert i18n.tr("Open files") == "Open files"
    assert i18n.tr("{n:,} holes", n=1234) == "1,234 holes"
    assert i18n.trn("{n} hole", "{n} holes", 1) == "1 hole"
    assert i18n.trn("{n} hole", "{n} holes", 3) == "3 holes"
    assert i18n.N_("Station") == "Station"
    assert i18n.set_language("xx_YY").code == "en"              # unknown: English


def test_chinese_translates_and_keeps_placeholders():
    lang = i18n.set_language("zh_CN")
    assert lang.code == "zh_CN" and lang.name == "简体中文" and lang.qt == "zh_CN"
    msgs = json.loads((i18n.LOCALE_DIR / "zh_CN.json").read_text(encoding="utf-8"))["messages"]
    key = next(k for k, v in msgs.items() if v and "{" not in k)
    assert i18n.tr(key) == msgs[key] != key
    assert i18n.tr("a text nobody translated") == "a text nobody translated"
    for code in ("zh_CN", "zh-CN", "zh", "ZH_cn", "zh_TW"):
        assert i18n.match_language(code) == "zh_CN", code
    assert i18n.match_language("de_DE") == "en" and i18n.match_language("") == "en"


def test_a_broken_translation_never_breaks_the_app(monkeypatch):
    monkeypatch.setattr(i18n, "_messages", {"{n} m": "{x} 米", "{n} hole": ["{n} 个", "{n} 个"]})
    assert i18n.tr("{n} m", n=5) == "5 m"                         # wrong placeholder: English
    assert i18n.trn("{n} hole", "{n} holes", 2) == "2 个"


def test_adding_a_language_is_one_file(tmp_path, monkeypatch):
    """A new JSON file in the locales folder is listed and used - nothing else to change."""
    monkeypatch.setattr(i18n, "LOCALE_DIR", tmp_path)
    monkeypatch.setattr(i18n_extract, "LOCALES", tmp_path)
    assert i18n_extract.main(["--new", "fr", "--name", "Français", "--english-name", "French"]) == 0
    data = json.loads((tmp_path / "fr.json").read_text(encoding="utf-8"))
    assert data["language"]["code"] == "fr" and len(data["messages"]) > 2
    assert set(data["messages"].values()) == {""}                  # all to translate
    done, todo = [k for k in sorted(data["messages"]) if "{" not in k][:2]
    data["messages"][done] = "Traduit"
    (tmp_path / "fr.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert [lang.code for lang in i18n.available_languages()] == ["en", "fr"]
    assert i18n.set_language("fr_FR").code == "fr"
    assert i18n.tr(done) == "Traduit"
    assert i18n.tr(todo) == todo                                    # untranslated: English


def test_every_text_in_the_code_is_translated_to_chinese():
    """The extraction tool's check: no text missing in zh_CN.json, none unused, and every
    translation keeps the English placeholders."""
    found = i18n_extract.extract()
    keys = set(found) - i18n_extract.plural_keys()
    assert len(keys) > 1000                                        # the whole interface is marked
    data = i18n_extract.load(i18n.LOCALE_DIR / "zh_CN.json")
    pr = i18n_extract.problems(keys, data)
    assert len(found) > 1000
    assert pr["missing"] == [], f"{len(pr['missing'])} untranslated, e.g. {pr['missing'][:5]}"
    assert pr["unused"] == [], pr["unused"][:5]
    assert pr["placeholders"] == [], pr["placeholders"][:5]


def test_qt_standard_buttons_follow_the_language():
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    from dfpredict.app import apply_language

    app = QApplication.instance() or QApplication([])
    try:
        assert apply_language(app, "zh_CN").code == "zh_CN"
        assert app._qt_translator is not None
        assert QCoreApplication.translate("QPlatformTheme", "Cancel") == "取消"
    finally:
        apply_language(app, "en")
    assert app._qt_translator is None and QCoreApplication.translate("QPlatformTheme", "Cancel") == "Cancel"


def test_vtk_text_gets_a_font_with_chinese_letters():
    import vtk

    tp = vtk.vtkTextProperty()
    i18n.apply_vtk_font(tp, bold=True)
    assert tp.GetFontFamily() == vtk.VTK_ARIAL and tp.GetBold()
    i18n.set_language("zh_CN")
    path = i18n.font_file()
    if path is None:
        pytest.skip("no Chinese font on this computer")
    i18n.apply_vtk_font(tp)
    assert tp.GetFontFamily() == vtk.VTK_FONT_FILE and tp.GetFontFile() == path
    # VTK can measure (so draw) Chinese text with it
    tr = vtk.vtkTextRenderer()
    bb = [0, 0, 0, 0]
    assert tr.GetBoundingBox(tp, "三维视图", bb, 96) and bb[1] - bb[0] > 20


def test_the_windows_open_in_chinese(tmp_path, monkeypatch):
    """Every window builds in Chinese (no text formatted wrongly), and shows Chinese."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton

    import dfpredict.ui.main_window as mw

    from dfpredict.io.database import Database, project_target
    from dfpredict.io.survey_store import SurveyStore

    app = QApplication.instance() or QApplication([])
    i18n.set_language("zh_CN")
    from dfpredict.ui.settings_dialog import AppSettings, SettingsDialog
    from dfpredict.ui.survey_manager import SurveyManager

    ini = str(tmp_path / "settings.ini")                       # never the user's own settings
    Path(ini).write_text("[database]\nbackend=sqlite\n[workspace]\nrestore_last=false\n", encoding="utf-8")
    monkeypatch.setattr(mw, "QSettings", lambda *a: QSettings(ini, QSettings.IniFormat))
    win = mw.MainWindow()
    try:
        assert any("一" <= ch <= "鿿" for a in win.actions() for ch in a.toolTip())
        dlg = SettingsDialog(AppSettings(), win)
        texts = [w.text() for cls in (QLabel, QPushButton) for w in dlg.findChildren(cls)]
        assert any(any("一" <= ch <= "鿿" for ch in t) for t in texts)
        dlg.close()
        store = SurveyStore(project_target(tmp_path))
        with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
            store.prepare(db)
        store.add_pit("主巷道", start_station="E", known=(1000.0, 5000.0, 250.0))
        sm = SurveyManager(store, win)
        assert sm.start()
        sm.add_setup()
        sm._recompute()
        assert sm.r_coords.text() != "Coordinates (X, Y, Z)"
        sm.dirty = False
        sm.close()
        app.processEvents()
    finally:
        win.quiet_errors = True
        win.close()


def test_the_language_setting_is_saved_and_listed(tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from dfpredict.ui.settings_dialog import AppSettings, SettingsDialog

    app = QApplication.instance() or QApplication([])  # noqa: F841
    ini = str(tmp_path / "s.ini")
    assert AppSettings.load(QSettings(ini, QSettings.IniFormat)).language == "en"   # English unless chosen
    dlg = SettingsDialog(AppSettings())
    codes = [dlg.language.itemData(i) for i in range(dlg.language.count())]
    assert codes == ["en", "zh_CN"]                                                  # only English and Chinese
    assert dlg.language.currentData() == "en"
    dlg.language.setCurrentIndex(codes.index("zh_CN"))
    s = dlg.result_settings()
    assert s.language == "zh_CN"
    qs = QSettings(ini, QSettings.IniFormat)
    s.save(qs)
    qs.sync()
    assert AppSettings.load(QSettings(ini, QSettings.IniFormat)).language == "zh_CN"
    assert QSettings(ini, QSettings.IniFormat).value(i18n.LANGUAGE_KEY) == "zh_CN"
    dlg.close()
