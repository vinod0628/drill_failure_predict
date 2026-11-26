import warnings

import pytest

from dfpredict.io import read_ssi
from dfpredict.io.colors import NAMED_COLORS, parse_color


def test_parse_color():
    assert parse_color("yellow") == (1.0, 1.0, 0.0)
    assert parse_color("r=1.0 g=0.6 b=0.2") == (1.0, 0.6, 0.2)
    assert parse_color("Forest  Green") == NAMED_COLORS["forest green"]
    with pytest.warns(UserWarning):
        assert parse_color("no such colour") == (0.7, 0.7, 0.7)


def test_ring_default_and_singles(demo):
    ss = read_ssi(demo / "ring.ssi")
    d = ss.sections["STRINGS"].default
    assert d.label == "yellow line"
    assert d.line_color == (1.0, 1.0, 0.0)
    assert d.face_color == (1.0, 1.0, 0.0)
    assert d.gloss == 5.0
    assert d.opacity == 1.0
    assert ss.string_style(2).line_color == (0.0, 0.0, 1.0)
    assert ss.string_style(999).label == "yellow line"  # falls back to default


def test_range_rules(demo):
    ss = read_ssi(demo / "gold_dtm.ssi")
    # "1, 32000, 10" covers 1, 11, 21, ...; "2, 32000, 10" covers 2, 12, ...
    assert ss.string_style(11) is ss.string_style(1)
    assert ss.string_style(12) is ss.string_style(2)
    assert ss.string_style(11) is not ss.string_style(12)


def test_numeric_colours(demo):
    ss = read_ssi(demo / "alter_lines_yellow_red.ssi")
    assert "TRIOBJECTS" in ss.sections


def test_all_demo_ssi_colours_known(demo):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # unknown colour names would warn
        for p in demo.glob("*.ssi"):
            read_ssi(p)
