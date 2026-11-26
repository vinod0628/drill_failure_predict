import numpy as np

from dfpredict.io.block_model import BlockModel
from dfpredict.scene.block_faces import exposed_faces, face_quads


def _model(centroids, sizes):
    c = np.asarray(centroids, dtype=np.float64)
    s = np.asarray(sizes, dtype=np.float32)
    return BlockModel(None, c, s, {"v": np.zeros(len(c), np.float32)})


def test_single_block_has_six_faces():
    g = _model([[5, 5, 5]], [[10, 10, 10]]).grid()
    b, d = exposed_faces(g)
    assert sorted(d.tolist()) == [0, 1, 2, 3, 4, 5]


def test_solid_cube_shows_only_shell():
    n = 4
    c = [[i * 10 + 5, j * 10 + 5, k * 10 + 5] for i in range(n) for j in range(n) for k in range(n)]
    g = _model(c, [[10, 10, 10]] * len(c)).grid()
    b, d = exposed_faces(g)
    assert len(b) == 6 * n * n


def test_mask_exposes_neighbours():
    c = [[5, 5, 5], [15, 5, 5], [25, 5, 5]]
    g = _model(c, [[10, 10, 10]] * 3).grid()
    assert len(exposed_faces(g)[0]) == 14
    # Hide the middle block: the outer two become separate cubes.
    assert len(exposed_faces(g, np.array([True, False, True]))[0]) == 12


def test_subblocks_partial_cover():
    """A 20 m block next to one 10 m sub-block: its +X face is only partly covered, so it stays."""
    g = _model([[10, 10, 10], [25, 5, 5]], [[20, 20, 20], [10, 10, 10]]).grid()
    b, d = exposed_faces(g)
    big_plus_x = (b == 0) & (d == 1)
    small_minus_x = (b == 1) & (d == 0)
    assert big_plus_x.any()
    assert not small_minus_x.any()


def test_quads_face_outward():
    g = _model([[5, 5, 5]], [[10, 10, 10]]).grid()
    b, d = exposed_faces(g)
    pts, cells = face_quads(g, b, d, np.zeros(3))
    q = pts.reshape(-1, 4, 3).astype(np.float64)
    normals = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    centre = q.mean(axis=1) - 5.0
    assert (np.einsum("ij,ij->i", normals, centre) > 0).all()


def test_demo_model_faces(demo, tmp_path):
    from dfpredict.io import read_block_csv

    p = demo / "block_model.csv"
    if not p.exists():
        return
    bm = read_block_csv(p, cache_dir=tmp_path)
    g = bm.grid()
    b, _ = exposed_faces(g)
    assert 0 < len(b) < 6 * bm.n_blocks
