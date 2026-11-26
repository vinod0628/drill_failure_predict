"""Import from Excel: the templates, reading and checking a filled-in workbook, replacing or
skipping what is in the database, writing nothing on errors, and the import window."""

import os

import openpyxl
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfpredict import i18n  # noqa: E402
from dfpredict.io import drillhole_import as di  # noqa: E402
from dfpredict.io import survey_import as si  # noqa: E402
from dfpredict.io.database import Database, project_target  # noqa: E402
from dfpredict.io.drillhole_store import ASSAY, GEOLOGY, SURVEY, DrillholeStore  # noqa: E402
from dfpredict.io.survey_store import SetupRow, SurveyStore  # noqa: E402


def fill(path, out, rows: dict):
    """The template at ``path`` with ``rows`` (sheet -> [{column: value}]) typed in from row 2."""
    wb = openpyxl.load_workbook(path)
    for sheet, recs in rows.items():
        ws = wb[sheet]
        heads = [c.value for c in ws[1]]
        for r, rec in enumerate(recs, start=2):
            for name, value in rec.items():
                if name not in heads:                     # a column the template does not have
                    heads.append(name)
                    ws.cell(1, len(heads), name)
                ws.cell(r, heads.index(name) + 1, value)
    wb.save(out)
    return out


def values(df):
    """A frame's rows with blanks (None / NaN) as None."""
    return [[None if pd.isna(v) else v for v in row] for row in df.values.tolist()]


# --- drillholes ------------------------------------------------------------------------------------
@pytest.fixture
def holes(tmp_path):
    store = DrillholeStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite"):
        pass
    store.prepare()
    store.add_hole("H1", {"x": "100", "y": "200", "z": "50", "max_depth": "60"})
    h = store.load_hole("H1")
    h.tables[SURVEY] = pd.DataFrame({"depth": [0.0], "azimuth": [0.0], "dip": [0.0]})
    h.tables[GEOLOGY] = pd.DataFrame({"depth_from": [0.0], "depth_to": [60.0], "rock": ["GR"], "note": [None]})
    store.save_hole("H1", h)
    di.save_template(store, tmp_path / "template.xlsx")
    return store, tmp_path


def test_drillhole_template_has_the_database_columns(holes):
    store, tmp = holes
    store.add_field("copper")
    di.save_template(store, tmp / "t2.xlsx")
    wb = openpyxl.load_workbook(tmp / "t2.xlsx")
    assert wb.sheetnames == ["Instructions", "collar", "survey", "geology", "assay"]
    head = lambda s: [c.value for c in wb[s][1]]                       # noqa: E731
    assert head("collar")[:5] == ["hole_id", "x", "y", "z", "max_depth"] and "state" in head("collar")
    assert head("survey") == ["hole_id", "depth", "azimuth", "dip"]
    assert head("assay") == ["hole_id", "depth_from", "depth_to", "gold", "silver", "copper"]
    assert wb["collar"]["B1"].comment is not None and "Required" in wb["collar"]["B1"].comment.text
    lists = [dv.formula1 for dv in wb["collar"].data_validations.dataValidation if dv.type == "list"]
    assert lists == ['"Planned,In work,Completed with ore hit,Completed without ore hit,Failed"']
    assert wb["collar"]["A2"].number_format == "@"                    # hole ids stay text
    texts = [c.value for row in wb["Instructions"].iter_rows() for c in row if c.value]
    assert "Easting of the collar (m)." in texts
    # the instructions in the user's language; the column names stay as they are
    i18n.set_language("zh_CN")
    di.save_template(store, tmp / "t3.xlsx")
    wb = openpyxl.load_workbook(tmp / "t3.xlsx")
    assert wb.sheetnames[0] != "Instructions" and wb.sheetnames[1:] == ["collar", "survey", "geology", "assay"]
    assert [c.value for c in wb["collar"][1]][:2] == ["hole_id", "x"]


def test_drillholes_are_added_replaced_or_skipped(holes):
    store, tmp = holes
    book = fill(tmp / "template.xlsx", tmp / "f.xlsx", {
        "collar": [{"hole_id": "H1", "x": 101, "y": 200, "z": 50, "max_depth": 80},
                   {"hole_id": 205, "x": 300, "y": 400, "z": 60, "start_date": "2026-03-01", "state": "In work"}],
        "survey": [{"hole_id": "H1", "depth": 0, "azimuth": 45, "dip": 10},
                   {"hole_id": "H1", "depth": 50, "azimuth": 47, "dip": 12}],
        "assay": [{"hole_id": 205, "depth_from": 1.5, "depth_to": 3, "gold": 0.4, "copper": 0.9},
                  {"hole_id": 205, "depth_from": 0, "depth_to": 1.5, "gold": 2.5}],
    })
    plan = di.read_import(store, book)
    assert plan.problems(True) == ([], [])
    assert plan.new_fields == ["copper"]
    assert plan.summary(True)[0] == "Holes in the file: 2 (1 new, 1 already in the database)"
    # skip: only the new hole
    counts = di.apply_import(store, plan, replace=False)
    assert (counts["added"], counts["updated"], counts["skipped"]) == (1, 0, 1)
    assert store.hole_ids() == ["205", "H1"]                          # 205 typed as a number: kept as text
    assert store.load_hole("H1").collar["x"] == 100.0
    new = store.load_hole("205")
    assert new.collar["state"] == 1 and new.collar["start_date"] == "2026-03-01"
    assert values(new.tables[ASSAY][["depth_from", "gold", "copper"]]) == [[0.0, 2.5, None], [1.5, 0.4, 0.9]]
    # replace: H1's collar and survey from the file, its geology (no rows in the file) kept
    plan = di.read_import(store, book)
    assert plan.summary(True)[0] == "Holes in the file: 2 (0 new, 2 already in the database)"
    counts = di.apply_import(store, plan, replace=True)
    assert (counts["added"], counts["updated"], counts[SURVEY]) == (0, 2, 2)
    h1 = store.load_hole("H1")
    assert (h1.collar["x"], h1.collar["max_depth"]) == (101.0, 80.0)
    assert h1.tables[SURVEY]["azimuth"].tolist() == [45.0, 47.0]
    assert h1.tables[GEOLOGY]["rock"].tolist() == ["GR"]


def test_drillhole_errors_name_the_row_and_nothing_is_written(holes):
    store, tmp = holes
    book = fill(tmp / "template.xlsx", tmp / "bad.xlsx", {
        "collar": [{"hole_id": "N1", "x": "east", "y": 1, "z": 1}],
        "survey": [{"hole_id": "N1", "depth": 0, "azimuth": 400, "dip": 0}],
        "geology": [{"hole_id": "N1", "depth_from": 10, "depth_to": 5, "rock": "GR"},
                    {"depth_from": 1, "depth_to": 2},
                    {"hole_id": "GHOST", "depth_from": 0, "depth_to": 1}],
        "assay": [{"hole_id": "N1", "depth_from": 0, "depth_to": 1, "gold": "lots", "2bad": 1}],
    })
    plan = di.read_import(store, book)
    errors, _w = plan.problems(True)
    text = "\n".join(errors)
    assert "Hole N1: Collar: x must be a number" in text
    assert "Hole N1: Survey row 2: azimuth must be 0 to 360" in text
    assert "Hole N1: Geology row 2: From must be 0 or more and less than To (10 - 5)" in text
    assert "Sheet geology row 3: no hole ID" in text
    assert "Hole GHOST (Geology) is not in the database and has no row in the collar sheet" in text
    assert "Hole N1: Assay row 2: gold must be a number (or empty)" in text
    assert "Sheet assay, column 2bad: A field name starts with a letter" in text
    with pytest.raises(ValueError):
        di.apply_import(store, plan, True)
    assert store.hole_ids() == ["H1"] and "2bad" not in store.assay_fields()


def test_an_exported_workbook_imports_again(holes):
    store, tmp = holes
    store.export_excel(tmp / "export.xlsx")
    plan = di.read_import(store, tmp / "export.xlsx")
    assert plan.problems(True) == ([], [])
    before = store.load_hole("H1")
    di.apply_import(store, plan, True)
    after = store.load_hole("H1")
    assert after.collar == before.collar
    for t in (SURVEY, GEOLOGY):
        assert values(after.tables[t]) == values(before.tables[t])


def test_a_state_typed_in_another_language_is_understood(holes):
    """A template filled in while Chinese was on imports with English on, and the other way."""
    store, tmp = holes
    zh = i18n.every_translation("Completed with ore hit") - {"Completed with ore hit"}
    assert zh
    book = fill(tmp / "template.xlsx", tmp / "zh.xlsx",
                {"collar": [{"hole_id": "Z1", "x": 1, "y": 2, "z": 3, "state": zh.pop()}]})
    plan = di.read_import(store, book)
    assert plan.problems(True)[0] == []
    di.apply_import(store, plan, True)
    assert store.load_hole("Z1").collar["state"] == 2


# --- survey ------------------------------------------------------------------------------------------
@pytest.fixture
def survey(tmp_path):
    store = SurveyStore(project_target(tmp_path))
    with Database.sqlite(tmp_path / "geodatabase.sqlite") as db:
        store.prepare(db)
    si.save_template(tmp_path / "template.xlsx")
    return store, tmp_path


WORKED = [   # the worked example of test_survey: E oriented on 45°, then S1, S2
    {"pit": "Main", "at": "E", "hi": 1.5, "bs_bearing": "45", "fs": "S1", "fs_ha": "120 30 00", "fs_za": "95",
     "fs_sd": 50, "fs_ht": 1.4},
    {"pit": "Main", "at": "S1", "hi": 1.45, "bs_station": "E", "bs_ha": "300 30 00", "fs": "S2",
     "fs_ha": "160°15′00″", "fs_za": 92, "fs_sd": 40, "fs_ht": 1.45},
]


def test_survey_template(survey):
    _store, tmp = survey
    wb = openpyxl.load_workbook(tmp / "template.xlsx")
    assert wb.sheetnames == ["Instructions", "pits", "setups"]
    heads = [c.value for c in wb["setups"][1]]
    assert heads[:4] == ["pit", "at", "method", "fs"] and {"fs_x", "fs_y", "fs_z", "bs_ha_r"} <= set(heads)
    lists = {dv.formula1 for dv in wb["setups"].data_validations.dataValidation if dv.type == "list"}
    assert '"angles,bearing,coords"' in lists and '"grid,true,magnetic"' in lists
    col = heads.index("fs_ha") + 1
    assert wb["setups"].cell(2, col).number_format == "@"            # angles stay text: 120:30:00 is no time


def test_main_pit_and_branch_with_every_method(survey):
    store, tmp = survey
    book = fill(tmp / "template.xlsx", tmp / "f.xlsx", {
        "pits": [{"name": "Main", "start_station": "E", "entrance_x": 1000, "entrance_y": 5000, "entrance_z": 250},
                 {"name": "Branch", "parent": "P1", "start_station": "S1", "station_level": "floor",
                  "adjust": "no", "profile_width": 4}],
        "setups": WORKED + [
            {"pit": "P1A", "at": "S1", "method": "bearing", "fs": "B1", "fs_bearing": "10", "fs_za": 90, "fs_sd": 20},
            {"pit": "Branch", "at": "B1", "fs": "B2", "fs_x": 1050, "fs_y": 5000, "fs_z": 240}],
    })
    plan = si.read_import(store, book)
    assert plan.problems(True) == ([], [])
    assert plan.summary(True) == ["Pits in the file: 2 (2 new, 0 already in the database)", "Setups to import: 4",
                                  "Stations computed: 4; without coordinates: 0"]
    assert si.apply_import(store, plan, True) == {"added": 2, "updated": 0, "skipped": 0, "setups": 4}
    pits = {p.name: p for p in store.pits()}
    assert (pits["Main"].code, pits["Branch"].code, pits["Branch"].parent_id) == ("P1", "P1A", pits["Main"].id)
    assert (pits["Branch"].station_level, pits["Branch"].adjust, pits["Branch"].profile_width) == ("floor", False, 4.0)
    st = {s.name: s for s in store.stations()}
    assert (st["S1"].x, st["S1"].y, st["S1"].z) == pytest.approx((1042.917, 4974.720, 245.742), abs=1e-3)
    assert (st["S2"].x, st["S2"].y) == pytest.approx((1056.426, 4937.096), abs=1e-3)
    assert st["B2"].status == "entered" and st["E"].kind == "known"
    rows = store.setups()[pits["Branch"].id]
    assert [r.method for r in rows] == ["bearing", "coords"]


def test_survey_errors_and_warnings_before_anything_is_written(survey):
    store, tmp = survey
    book = fill(tmp / "template.xlsx", tmp / "bad.xlsx", {
        "pits": [{"name": "Main", "start_station": "E"},
                 {"name": "Loop A", "parent": "Loop B", "start_station": "X"},
                 {"name": "Loop B", "parent": "Loop A", "start_station": "Y"},
                 {"name": "Lost", "parent": "Nowhere", "start_station": "Z"}],
        "setups": [{"pit": "Main", "at": "E", "bs_bearing": "45", "fs": "S1", "fs_ha": "120 75 00", "fs_za": 90,
                    "fs_sd": 10},
                   {"pit": "Main", "at": "S1", "method": "coords", "fs": "S2", "fs_x": 1, "fs_y": 2},
                   {"pit": "Ghost", "at": "Q"}],
    })
    plan = si.read_import(store, book)
    text = "\n".join(plan.problems(True)[0])
    assert "Sheet pits row 2: a main pit needs the entrance's X, Y and Z" in text
    assert "its parents lead back to it (a loop of branches)" in text
    assert "Sheet pits row 5: parent pit Nowhere is neither in the file nor in the database" in text
    assert "Sheet setups row 2, fs_ha: minutes and seconds must be below 60: '120 75 00'" in text
    assert "Sheet setups row 3, fs_x / fs_y / fs_z: X, Y and Z of the foresight are needed (method coords)" in text
    assert "Sheet setups row 4: pit Ghost is neither in the pits sheet nor in the database" in text
    with pytest.raises(ValueError):
        si.apply_import(store, plan, True)
    assert store.pits() == []
    # a correct file whose traverse does not compute: warnings (from the computation), import allowed
    book = fill(tmp / "template.xlsx", tmp / "warn.xlsx", {
        "pits": [{"name": "Main", "start_station": "E", "entrance_x": 0, "entrance_y": 0, "entrance_z": 0}],
        "setups": [{"pit": "Main", "at": "K", "bs_bearing": "0", "fs": "S1", "fs_ha": 0, "fs_za": 90, "fs_sd": 5}]})
    plan = si.read_import(store, book)
    errors, warnings = plan.problems(True)
    assert errors == [] and warnings == ["Pit Main, setups row 2: setup at K not computed: station K has no "
                                         "coordinates yet"]


def test_a_pit_in_the_database_is_replaced_or_skipped(survey):
    store, tmp = survey
    pid = store.add_pit("Main", start_station="E", known=(1000.0, 5000.0, 250.0))
    store.update_pit(pid, profile=(6.0, 4.0, "middle"), note="old note")
    store.save_setups(pid, [SetupRow("E", 1.5, bs_bearing=0.0, fs="OLD", fs_ha=90.0, fs_za=90.0, fs_sd=10.0)])
    book = fill(tmp / "template.xlsx", tmp / "f.xlsx", {
        "pits": [{"name": "main", "entrance_z": 251, "entrance_x": 1000, "entrance_y": 5000}],   # found by name
        "setups": WORKED})
    plan = si.read_import(store, book)
    assert plan.problems(True) == ([], [])
    assert si.apply_import(store, plan, False)["skipped"] == 1
    assert [r.fs for r in store.setups()[pid]] == ["OLD"]
    plan = si.read_import(store, book)
    assert si.apply_import(store, plan, True) == {"added": 0, "updated": 1, "skipped": 0, "setups": 2}
    p = store.pits()[0]
    assert (p.name, p.profile_width, p.station_level, p.note) == ("Main", 6.0, "middle", "old note")  # empty: kept
    assert [r.fs for r in store.setups()[pid]] == ["S1", "S2"]
    st = {s.name: s for s in store.stations()}
    assert st["E"].z == 251.0 and st["S1"].z == pytest.approx(246.742, abs=1e-3)


# --- the window ------------------------------------------------------------------------------------
def test_the_import_windows_of_both_managers(holes, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    from dfpredict.ui.drillhole_manager import DrillholeManager
    from dfpredict.ui.excel_import import ExcelImportDialog
    from dfpredict.ui.survey_manager import SurveyManager

    app = QApplication.instance() or QApplication([])
    store, tmp = holes
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)   # don't open Excel
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[2]}"))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp / "from_window"), ""))
    seen = {}

    def run(dlg, path):
        dlg.save_template()                               # the template, saved from the window
        seen["template"] = (tmp / "from_window.xlsx").exists() and dlg.path.text().endswith("from_window.xlsx")
        assert not dlg.b_import.isEnabled()               # the empty template: nothing to import
        dlg.path.setText(str(path))
        dlg.check()
        seen["ready"] = dlg.b_import.isEnabled()
        seen["report"] = dlg.report.toPlainText()
        dlg.do_import()
        return dlg.result()

    monkeypatch.setattr(ExcelImportDialog, "exec", lambda self: run(self, seen["file"]))
    seen["file"] = fill(tmp / "template.xlsx", tmp / "f.xlsx",
                        {"collar": [{"hole_id": "W9", "x": 1, "y": 2, "z": 3}]})
    dm = DrillholeManager(store)
    assert dm.start()
    dm.import_excel()
    assert seen["template"] and seen["ready"] and "Ready - click Import." in seen["report"]
    assert dm.holes.count() == 2 and dm.status.text().startswith("Imported - holes added: 1, replaced: 0")
    dm.close()

    sstore = SurveyStore(project_target(tmp_path / "s"))
    (tmp_path / "s").mkdir()
    with Database.sqlite(tmp_path / "s" / "geodatabase.sqlite") as db:
        sstore.prepare(db)
    si.save_template(tmp_path / "s" / "t.xlsx")
    seen["file"] = fill(tmp_path / "s" / "t.xlsx", tmp_path / "s" / "f.xlsx", {
        "pits": [{"name": "Main", "start_station": "E", "entrance_x": 1000, "entrance_y": 5000, "entrance_z": 250}],
        "setups": WORKED})
    sm = SurveyManager(sstore)
    assert sm.start()
    sm.import_excel()
    assert seen["ready"] and sm.tree.topLevelItemCount() == 1
    assert [r.fs for r in sm.rows] == ["S1", "S2"]                    # the imported pit is shown
    assert sm.status.text() == "Imported - pits added: 1, replaced: 0, skipped: 0; setups: 2"
    sm.dirty = False
    sm.close()
    app.processEvents()
