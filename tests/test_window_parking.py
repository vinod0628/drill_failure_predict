"""Tool windows step aside (hidden, contents kept) instead of minimising; their toolbar button restores them."""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QVBoxLayout, QWidget  # noqa: E402

from dfpredict.ui.window_parking import WindowParking  # noqa: E402


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _dialog(parent, text):
    d = QDialog(parent)
    d.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
    d.edit = QLineEdit(text, d)
    QVBoxLayout(d).addWidget(d.edit)
    return d


def test_minimising_hides_the_group_and_its_button_brings_it_back(qapp):
    main = QWidget()
    park = WindowParking(main)
    survey = _dialog(main, "pit P1")
    route = _dialog(survey, "target B")
    reserve = _dialog(main, "cut 1")
    for w, g in ((survey, "survey"), (route, "survey"), (reserve, "reserve")):
        park.watch(w, g)
        w.show()
    qapp.processEvents()
    route.edit.setText("target C")                       # what the user entered is kept

    route.showMinimized()                                # minimise one window of the group ...
    for _ in range(5):
        qapp.processEvents()
    assert not route.isVisible() and not survey.isVisible()          # ... the group steps aside, no title strip
    assert reserve.isVisible()                                       # other tools are not touched
    assert not route.windowState() & Qt.WindowMinimized
    assert park.parked("survey") == [survey, route] and park.parked("reserve") == []

    assert park.restore("survey")                        # the Survey toolbar button
    qapp.processEvents()
    assert survey.isVisible() and route.isVisible() and route.edit.text() == "target C"
    assert park.parked("survey") == [] and not park.restore("survey")


def test_a_window_closed_by_the_user_is_not_brought_back(qapp):
    main = QWidget()
    park = WindowParking(main)
    reserve = _dialog(main, "cut 1")
    park.watch(reserve, "reserve")
    reserve.show()
    park.park("reserve")                                 # Draw mined outline: steps aside
    assert not reserve.isVisible() and park.parked("reserve") == [reserve]
    reserve.show()
    reserve.close()                                      # shown again, then closed with its Close button
    assert park.parked("reserve") == [] and not park.restore("reserve")
    reserve.deleteLater()
    qapp.processEvents()
