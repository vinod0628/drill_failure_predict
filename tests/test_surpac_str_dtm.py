from collections import Counter

import numpy as np
import pytest

from dfpredict.io import SurpacFormatError, read_dtm, read_str


def test_str_coordinates_are_swapped_to_xyz(demo):
    sf = read_str(demo / "layout2.str")
    assert sf.header.name == "layout"
    assert sf.header.styles_file == "SSI_STYLES:styles.ssi"
    # File record: "3, 7349.719, 1482.153, -288.608" -> Y, X, Z
    np.testing.assert_allclose(sf.records_xyz[0], [1482.153, 7349.719, -288.608])
    assert sf.records_string[0] == 3


def test_str_x_range_matches_dxf(demo):
    """topo2.dxf (X, Y order) covers X 1181-2082 and Y 7022-7722; topo1.str must agree."""
    lo, hi = read_str(demo / "topo1.str").bounds()
    assert 1100 < lo[0] < hi[0] < 2200
    assert 6900 < lo[1] < hi[1] < 7800


def test_str_segments(demo):
    sf = read_str(demo / "pit1.str")
    # Separators are kept as records (NaN) but never inside a segment.
    assert np.isnan(sf.records_xyz[sf.records_string == 0]).all()
    for i in range(sf.n_segments):
        seg = sf.records_string[sf.seg_start[i]:sf.seg_end[i]]
        assert (seg == sf.seg_string[i]).all()
    assert set(sf.string_numbers) >= {1, 2, 40}
    assert sum(sf.seg_end - sf.seg_start) == sf.point_mask.sum()


def test_str_closed_segments(demo):
    sf = read_str(demo / "ore1.str")
    closed = [sf.segment_is_closed(i) for i in range(sf.n_segments)]
    assert all(closed)  # ore outlines are closed polygons


def _edge_counts(tris: np.ndarray) -> Counter:
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    return Counter(map(tuple, e))


@pytest.mark.parametrize("name", ["ore1", "layout2"])
def test_dtm_closed_solid_is_watertight(demo, name):
    """Every edge of a closed solid is shared by exactly two triangles.

    This only holds when the vertex numbering (records incl. separators) is right.
    """
    dtm = read_dtm(demo / f"{name}.dtm")
    for _, tri in dtm.iter_trisolations():
        assert tri.closed
        counts = _edge_counts(tri.triangles)
        assert set(counts.values()) == {2}


def test_dtm_neighbours_share_edges(demo):
    dtm = read_dtm(demo / "ore1.dtm")
    tri = dtm.objects[0].trisolations[0]
    index = {tid: k for k, tid in enumerate(tri.triangle_ids)}
    for k in range(0, tri.n_triangles, 37):
        a = set(tri.triangles[k])
        for nb in tri.neighbours[k]:
            if nb in index:
                assert len(a & set(tri.triangles[index[nb]])) == 2


def test_dtm_solid_volume(demo):
    dtm = read_dtm(demo / "ore1.dtm")
    pts, faces = dtm.compact_mesh()
    p = pts - pts.mean(axis=0)
    v = np.einsum("ij,ij->i", p[faces[:, 0]], np.cross(p[faces[:, 1]], p[faces[:, 2]])).sum() / 6
    assert abs(v) > 1e5  # a real, non-degenerate solid (cubic metres)


def test_dtm_open_surface_and_spot_points(demo):
    dtm = read_dtm(demo / "topo1.dtm")
    assert len(dtm.vertices) > dtm.string_file.n_records  # extra points appended
    tri = dtm.objects[0].trisolations[0]
    assert not tri.closed
    assert dtm.n_triangles == 8494


def test_dtm_multiple_objects(demo):
    dtm = read_dtm(demo / "ug1.dtm")
    assert len(dtm.objects) >= 3
    assert any(len(o.trisolations) > 1 for o in dtm.objects)
    for o in dtm.objects:
        pts, faces = dtm.compact_mesh(obj_id=o.id)
        assert len(faces) == o.n_triangles
        assert np.isfinite(pts).all()


def test_all_demo_dtms_load(demo):
    for p in sorted(demo.glob("*.dtm")):
        dtm = read_dtm(p)
        assert dtm.n_triangles > 0, p.name


def test_dtm_detects_edited_string_file(demo, tmp_path):
    src = (demo / "ore1.str").read_text().splitlines()
    (tmp_path / "ore1.str").write_text("\n".join(src[:100] + src[-1:]) + "\n")
    (tmp_path / "ore1.dtm").write_text((demo / "ore1.dtm").read_text())
    with pytest.raises(SurpacFormatError):
        read_dtm(tmp_path / "ore1.dtm")
