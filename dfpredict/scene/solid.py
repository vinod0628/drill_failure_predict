"""Point tests against Surpac DTMs: inside a solid, above / below a surface.

Both use vertical rays. Points are grouped by their (x, y) column - block centroids
share few columns - each triangle is tested only against the columns under its plan
footprint, and the crossings of a column's ray with the triangles are collected:

* solid (closed mesh): a point is inside when the surface crosses the ray an odd
  number of times above it;
* surface (open mesh): the surface height at a point is the top-most crossing of its
  column; a point is above / below the surface when it is higher / lower than that.
  Points outside the surface's plan extent have no height (NaN).

The query columns are moved by a tiny, irregular offset so that a ray never runs
exactly through a triangle edge or vertex (which would count a crossing twice).
"""

from __future__ import annotations

import numpy as np

_JITTER = np.array([3.1415926e-5, 2.7182818e-5])   # metres: breaks edge / vertex hits


def _crossings(Q: np.ndarray, T: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Columns of the points Q and every crossing of a column ray with the triangles T.

    Returns (col_of (N,), n_cols, cross_col (K,), cross_z (K,))."""
    q = 1e-3                                                 # columns: unique (x, y) at 1 mm
    kx = np.round((Q[:, 0] - lo[0]) / q).astype(np.int64)
    ky = np.round((Q[:, 1] - lo[1]) / q).astype(np.int64)
    key = kx * (int(np.round((hi[1] - lo[1]) / q)) + 3) + ky
    ukey, first, col_of = np.unique(key, return_index=True, return_inverse=True)
    cols = Q[first, :2] + _JITTER                            # (C, 2) ray positions
    order = np.argsort(cols[:, 0], kind="stable")
    cx = cols[order, 0]

    cross_col, cross_z = [], []
    tx, ty, tz = T[:, :, 0], T[:, :, 1], T[:, :, 2]
    xmin, xmax = tx.min(axis=1), tx.max(axis=1)
    ymin, ymax = ty.min(axis=1), ty.max(axis=1)
    area2 = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0])
    i0 = np.searchsorted(cx, xmin, side="left")
    i1 = np.searchsorted(cx, xmax, side="right")
    for t in np.flatnonzero((np.abs(area2) > 1e-12) & (i1 > i0)):   # vertical triangles never cross
        idx = order[i0[t]:i1[t]]
        yy = cols[idx, 1]
        idx = idx[(yy >= ymin[t]) & (yy <= ymax[t])]
        if not len(idx):
            continue
        px, py = cols[idx, 0], cols[idx, 1]
        (ax, bx, cxv), (ay, by, cyv) = tx[t], ty[t]
        w0 = ((bx - px) * (cyv - py) - (cxv - px) * (by - py)) / area2[t]
        w1 = ((cxv - px) * (ay - py) - (ax - px) * (cyv - py)) / area2[t]
        w2 = 1.0 - w0 - w1
        hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if hit.any():
            cross_col.append(idx[hit])
            cross_z.append(w0[hit] * tz[t, 0] + w1[hit] * tz[t, 1] + w2[hit] * tz[t, 2])
    if cross_col:
        return col_of, len(ukey), np.concatenate(cross_col), np.concatenate(cross_z)
    return col_of, len(ukey), np.zeros(0, dtype=np.int64), np.zeros(0)


def _prepare(points, vertices, triangles):
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)].reshape(-1, 3, 3)
    return P, T


def points_inside(points: np.ndarray, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Boolean mask: which ``points`` (N, 3) are inside the closed mesh (vertices, triangles)."""
    P, T = _prepare(points, vertices, triangles)
    inside = np.zeros(len(P), dtype=bool)
    if len(P) == 0 or len(T) == 0:
        return inside
    lo, hi = T.min(axis=(0, 1)), T.max(axis=(0, 1))
    cand = np.flatnonzero(np.all((P >= lo - 1e-9) & (P <= hi + 1e-9), axis=1))   # in the bounding box
    if not len(cand):
        return inside
    Q = P[cand]
    col_of, n_cols, cc, cz = _crossings(Q, T, lo, hi)
    if not len(cc):
        return inside
    # crossings above each point: sort by (column, z) and search the point's (column, z)
    zspan = max(hi[2] - lo[2], 1e-6) * 2.0 + 1.0
    ckey = cc + (cz - lo[2] + 0.25) / zspan / 2.0             # column + fraction in [0, 0.5)
    ckey.sort()
    end = np.searchsorted(ckey, np.arange(n_cols) + 0.75)     # end of each column's crossings
    pkey = col_of + (Q[:, 2] - lo[2] + 0.25) / zspan / 2.0
    above = end[col_of] - np.searchsorted(ckey, pkey, side="right")
    inside[cand] = (above % 2) == 1
    return inside


def surface_heights(points: np.ndarray, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Height (z) of the surface straight above / below each point - the top-most one where
    the surface overlaps itself in plan. NaN where the surface does not cover the point."""
    P, T = _prepare(points, vertices, triangles)
    z = np.full(len(P), np.nan)
    if len(P) == 0 or len(T) == 0:
        return z
    lo, hi = T.min(axis=(0, 1)), T.max(axis=(0, 1))
    cand = np.flatnonzero((P[:, 0] >= lo[0] - 1e-9) & (P[:, 0] <= hi[0] + 1e-9)
                          & (P[:, 1] >= lo[1] - 1e-9) & (P[:, 1] <= hi[1] + 1e-9))   # in plan extent
    if not len(cand):
        return z
    Q = P[cand]
    col_of, n_cols, cc, cz = _crossings(Q, T, lo, hi)
    if not len(cc):
        return z
    top = np.full(n_cols, -np.inf)
    np.maximum.at(top, cc, cz)
    top[np.isinf(top)] = np.nan
    z[cand] = top[col_of]
    return z


def watertight(triangles: np.ndarray) -> bool:
    """A closed surface: every edge is shared by exactly two triangles."""
    f = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return False
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool((counts == 2).all())


def closed_mesh(dtm) -> tuple[np.ndarray, np.ndarray] | None:
    """(vertices, triangles) of the closed trisolations of a Surpac DTM; None if it has none.
    Closed is judged from the geometry (watertight), not the file's "closed=yes" flag."""
    tris = [t.triangles for _obj, t in dtm.iter_trisolations() if watertight(t.triangles)]
    if not tris:
        return None
    return np.asarray(dtm.vertices, dtype=np.float64), np.concatenate(tris).astype(np.int64)


def all_triangles(dtm) -> tuple[np.ndarray, np.ndarray] | None:
    """(vertices, triangles) of every trisolation of a Surpac DTM; None if it has none."""
    tris = [t.triangles for _obj, t in dtm.iter_trisolations()]
    if not tris:
        return None
    return np.asarray(dtm.vertices, dtype=np.float64), np.concatenate(tris).astype(np.int64)


def dtm_kind(dtm) -> str:
    """"solid" when the DTM has closed trisolations (a volume), else "surface"."""
    return "solid" if closed_mesh(dtm) is not None else "surface"
