"""Exposed-face extraction for (sub-blocked) block models.

Drawing every block as a cube costs 12 triangles per block, which is too much
for tens of millions of blocks. Only faces that touch empty space (or a block
that is filtered out) can ever be seen, so we keep just those.

Method: mark every shown block on a boolean occupancy grid at the finest
block resolution, then for each block face test the cells just outside it.
A face is hidden only when all of those cells are occupied. Work is grouped
by block span (a handful of sizes in a sub-blocked model), so everything
stays vectorised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.block_model import BlockGrid

# Max cells in the occupancy grid (1 byte each) before falling back to
# drawing all faces of the shown blocks.
MAX_GRID_CELLS = 400_000_000

# Unit-cube corners for each face, ordered counter-clockwise seen from outside.
# Face order: -X, +X, -Y, +Y, -Z, +Z
FACE_CORNERS = np.array(
    [
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    ],
    dtype=np.float64,
)


def _group_by_span(span: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(span (3,), row indices)] for each distinct block span."""
    key = (span[:, 0].astype(np.int64) << 40) | (span[:, 1].astype(np.int64) << 20) | span[:, 2]
    if len(key) and (key == key[0]).all():
        return [(span[0], np.arange(len(key)))]
    codes, uniques = pd.factorize(key, sort=False)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    groups = []
    for g in range(len(uniques)):
        rows = order[bounds[g]:bounds[g + 1]]
        groups.append((span[rows[0]], rows))
    return groups


def exposed_faces(grid: BlockGrid, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (block_index (F,), face_dir (F,)) of visible faces of the masked blocks.

    ``face_dir`` is 0..5 for -X, +X, -Y, +Y, -Z, +Z.
    """
    n = len(grid.ijk_min)
    sel = np.arange(n) if mask is None else np.flatnonzero(mask)
    if len(sel) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int8)

    nx, ny, nz = (int(v) + 2 for v in grid.shape)   # padded by one cell on every side
    if nx * ny * nz > MAX_GRID_CELLS:
        blocks = np.repeat(sel, 6)
        dirs = np.tile(np.arange(6, dtype=np.int8), len(sel))
        return blocks, dirs

    stride = np.array([ny * nz, nz, 1], dtype=np.int64)
    ijk = grid.ijk_min[sel].astype(np.int64) + 1
    base = ijk @ stride                             # flat index of each block's min cell
    groups = _group_by_span(grid.span[sel])

    # Occupancy of shown blocks.
    occ = np.zeros(nx * ny * nz, dtype=bool)
    for s, rows in groups:
        b0 = base[rows]
        for a in range(s[0]):
            for b in range(s[1]):
                for c in range(s[2]):
                    occ[b0 + (a * stride[0] + b * stride[1] + c)] = True

    out_b: list[np.ndarray] = []
    out_d: list[np.ndarray] = []
    for s, rows in groups:
        b0 = base[rows]
        for axis in range(3):
            u, v = [a for a in range(3) if a != axis]
            for side in (0, 1):
                plane = -stride[axis] if side == 0 else int(s[axis]) * stride[axis]
                covered = np.ones(len(rows), dtype=bool)
                for a in range(s[u]):
                    for b in range(s[v]):
                        covered &= occ[b0 + (plane + a * stride[u] + b * stride[v])]
                        if not covered.any():
                            break
                    if not covered.any():
                        break
                vis = rows[~covered]
                if len(vis):
                    out_b.append(sel[vis])
                    out_d.append(np.full(len(vis), axis * 2 + side, dtype=np.int8))
    if not out_b:
        return np.zeros(0, np.int64), np.zeros(0, np.int8)
    return np.concatenate(out_b), np.concatenate(out_d)


def face_quads(
    grid: BlockGrid, blocks: np.ndarray, dirs: np.ndarray, offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Quad geometry for the given faces.

    Returns (points (4F, 3) float32, faces (5F,) int64 in VTK cell layout).
    ``offset`` is subtracted from world coordinates (the scene datum).
    """
    lo = grid.origin - offset + grid.ijk_min[blocks] * grid.cell
    size = grid.span[blocks] * grid.cell
    corners = FACE_CORNERS[dirs]                              # (F, 4, 3)
    pts = lo[:, None, :] + corners * size[:, None, :]
    pts = pts.reshape(-1, 3).astype(np.float32)
    nf = len(blocks)
    cells = np.empty((nf, 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = np.arange(4 * nf, dtype=np.int64).reshape(nf, 4)
    return pts, cells.ravel()
