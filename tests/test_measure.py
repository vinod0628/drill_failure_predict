"""Measure tool: distance and angles between two points."""

import math

import pytest

from dfpredict.ui.measure import measure, measure_rows


def test_distance_and_angles():
    m = measure((0, 0, 0), (30, 40, 0))
    assert m["distance"] == pytest.approx(50) and m["plan"] == pytest.approx(50)
    assert m["bearing"] == pytest.approx(math.degrees(math.atan2(30, 40)))
    assert m["inclination"] == pytest.approx(0) and m["gradient"] == pytest.approx(0)


@pytest.mark.parametrize("b, bearing", [((0, 10, 0), 0), ((10, 0, 0), 90), ((0, -10, 0), 180),
                                        ((-10, 0, 0), 270), ((-10, 10, 0), 315)])
def test_bearing_clockwise_from_north(b, bearing):
    assert measure((0, 0, 0), b)["bearing"] == pytest.approx(bearing)


def test_inclination_and_gradient():
    m = measure((100, 200, 50), (110, 200, 40))   # 10 m east, 10 m down
    assert m["inclination"] == pytest.approx(-45) and m["gradient"] == pytest.approx(-100)
    v = measure((0, 0, 0), (0, 0, 25))             # vertical: no bearing
    assert v["bearing"] is None and v["inclination"] == pytest.approx(90) and v["gradient"] is None


def test_large_coordinates_keep_precision():
    m = measure((512345.123, 7012345.456, 1200.0), (512345.124, 7012345.456, 1200.0))
    assert m["distance"] == pytest.approx(0.001, abs=1e-6)


def test_rows():
    _, rows = measure_rows((0, 0, 0), (3, 4, -5))
    d = {k: v for k, v in rows if v is not None}
    assert d["Distance"].startswith("7.071") and d["Plan distance"] == "5.000 m"
    assert "(down)" in d["Inclination"]
