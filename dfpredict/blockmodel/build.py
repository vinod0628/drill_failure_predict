"""New block models: inside an ore solid, or imported from a Surpac block model ASCII export.

Inside a solid
    The solid's bounding box is covered by a regular grid of blocks (size X, Y, Z; the
    origin snapped to whole blocks unless given). A block is kept when its centre is inside
    the solid. With sub-blocking (``sub`` = 2, 4 or 8) every block is split into sub^3 cells
    first: a block whose cells are all inside stays whole, a block on the boundary keeps only
    its cells inside (sub-blocks), so the model follows the solid's shape more closely.

    The inside test uses vertical rays (as ``scene.solid``): for every grid column the
    crossings with the solid are paired into inside intervals, which fill the cells whose
    centres lie between them - no per-block point test, so large models build quickly.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..i18n import tr
from ..io.block_model import SURPAC_NULL, BlockModel, _detect_columns, _from_dataframe, _norm
from ..scene.solid import _crossings
from .store import add_history

MAX_CELLS = 400_000_000          # finest cells in the solid's box (memory / time guard)


def mesh_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Volume of a closed triangle mesh (divergence theorem; orientation-independent)."""
    T = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    c = T.reshape(-1, 3).mean(axis=0) if len(T) else np.zeros(3)
    T = T - c
    return float(abs(np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum()) / 6.0)


def _used_bounds(vertices, triangles):
    used = np.asarray(vertices, dtype=np.float64)[np.unique(np.asarray(triangles, dtype=np.int64))]
    return used.min(axis=0), used.max(axis=0)


def grid_for_solid(vertices, triangles, size, origin=None) -> tuple[np.ndarray, np.ndarray]:
    """(origin, number of blocks per axis) covering the solid. Without ``origin`` it is the
    box's minimum corner rounded down to a whole number of blocks."""
    size = np.asarray(size, dtype=np.float64)
    lo, hi = _used_bounds(vertices, triangles)
    if origin is None:
        origin = np.floor(lo / size) * size
    origin = np.asarray(origin, dtype=np.float64)
    origin = origin + np.floor((lo - origin) / size + 1e-9) * size   # a given origin: whole blocks below the box
    n =np.ceil((hi - origin) / size - 1e-9).astype(np.int64)
    return origin, np.maximum(n, 1)


def blocks_in_solid(vertices, triangles, size, origin=None, sub: int = 1, attributes=None,
                    null_value: float = SURPAC_NULL, name: str = "", progress=None) -> BlockModel:
    """A block model of the blocks inside a closed mesh. ``attributes``: [(name, default
    value, description), ...] (float attributes). ``progress(fraction)`` is called now and then."""
    size = np.asarray(size, dtype=np.float64)
    if np.any(size <= 0):
        raise ValueError(tr("Block sizes must be greater than 0"))
    sub = int(sub)
    if sub not in (1, 2, 4, 8):
        raise ValueError(tr("Sub-blocking must be 1, 2, 4 or 8"))
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    origin, n = grid_for_solid(V, F, size, origin)
    fine = size / sub
    nf = n * sub
    if int(np.prod(nf, dtype=np.float64)) > MAX_CELLS:
        raise ValueError(tr("Too many blocks: {n:,} cells in the solid's box. Use larger blocks or less "
                            "sub-blocking.", n=int(np.prod(nf, dtype=np.float64))))

    # crossings of every fine column's vertical ray with the solid
    ii, jj = np.meshgrid(np.arange(nf[0]), np.arange(nf[1]), indexing="ij")
    cols_xy = np.column_stack([origin[0] + (ii.ravel() + 0.5) * fine[0], origin[1] + (jj.ravel() + 0.5) * fine[1]])
    Tri = V[F]
    lo, hi = Tri.min(axis=(0, 1)), Tri.max(axis=(0, 1))
    inbox = ((cols_xy[:, 0] >= lo[0]) & (cols_xy[:, 0] <= hi[0]) & (cols_xy[:, 1] >= lo[1]) & (cols_xy[:, 1] <= hi[1]))
    col_ids = np.flatnonzero(inbox)
    if progress:
        progress(0.05)
    if not len(col_ids):
        raise ValueError(tr("No block centre is inside the solid"))
    Q = np.column_stack([cols_xy[col_ids], np.zeros(len(col_ids))])
    col_of, n_cols, cc, cz = _crossings(Q, Tri, lo, hi)
    if progress:
        progress(0.45)
    # col_of maps each query to its (deduplicated) column; columns here are unique already
    uniq_col = np.empty(n_cols, dtype=np.int64)
    uniq_col[col_of] = col_ids
    order = np.lexsort((cz, cc))
    cc, cz = cc[order], cz[order]
    starts = np.searchsorted(cc, np.arange(n_cols))
    counts = np.bincount(cc, minlength=n_cols)
    rank = np.arange(len(cc)) - starts[cc]
    ok = (counts[cc] % 2) == 0                      # an odd count: the ray grazed the mesh, skip it
    enter = ok & (rank % 2 == 0)
    c_in = cc[enter]
    z0 = cz[enter]
    z1 = cz[np.flatnonzero(enter) + 1]
    k0 = np.ceil((z0 - origin[2]) / fine[2] - 0.5).astype(np.int64)
    k1 = np.floor((z1 - origin[2]) / fine[2] - 0.5).astype(np.int64)
    k0 = np.clip(k0, 0, nf[2])
    k1 = np.clip(k1, -1, nf[2] - 1)
    keep = k1 >= k0
    col_global = uniq_col[c_in[keep]]
    k0, k1 = k0[keep], k1[keep]
    if not len(col_global):
        raise ValueError(tr("No block centre is inside the solid"))

    # fine cells inside, one parent row (x index) at a time
    fi = col_global // nf[1]
    fj = col_global % nf[1]
    ijk_parts, span_parts = [], []
    rows = np.unique(fi // sub)
    for r_i, row in enumerate(rows):
        sel = np.flatnonzero(fi // sub == row)
        diff = np.zeros((sub, nf[1], nf[2] + 1), dtype=np.int32)
        np.add.at(diff, (fi[sel] - row * sub, fj[sel], k0[sel]), 1)
        np.add.at(diff, (fi[sel] - row * sub, fj[sel], k1[sel] + 1), -1)
        inside = np.cumsum(diff, axis=2)[:, :, :nf[2]] > 0          # (sub, nfy, nfz)
        if sub == 1:
            j, k = np.nonzero(inside[0])
            ijk_parts.append(np.column_stack([np.full(len(j), row), j, k]))
            continue
        cube = inside.reshape(sub, n[1], sub, n[2], sub)              # (a, J, b, K, c)
        full = cube.all(axis=(0, 2, 4))                               # (J, K)
        J, K = np.nonzero(full)
        if len(J):
            ijk_parts.append(np.column_stack([np.full(len(J), row * sub), J * sub, K * sub]))
            span_parts.append(np.full((len(J), 3), sub))
        partial = cube.any(axis=(0, 2, 4)) & ~full
        Jp, Kp = np.nonzero(partial)
        if len(Jp):
            blocks = cube[:, Jp, :, Kp, :]                            # (P, a, b, c)
            p, a, b, c = np.nonzero(blocks)
            ijk_parts.append(np.column_stack([row * sub + a, Jp[p] * sub + b, Kp[p] * sub + c]))
            span_parts.append(np.ones((len(p), 3), dtype=np.int64))
        if progress and r_i % 16 == 0:
            progress(0.5 + 0.45 * r_i / len(rows))
    ijk = np.concatenate(ijk_parts).astype(np.int64)
    span = np.concatenate(span_parts).astype(np.int64) if sub > 1 else np.ones_like(ijk)
    if not len(ijk):
        raise ValueError(tr("No block centre is inside the solid"))
    order = np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))
    ijk, span = ijk[order], span[order]
    centroids = origin + (ijk + span / 2.0) * fine
    sizes = (span * fine).astype(np.float32)
    attrs, desc = {}, {}
    for a in attributes or []:
        nm, default, description = (list(a) + [None, ""])[:3]
        attrs[nm] = np.full(len(ijk), null_value if default is None else float(default), dtype=np.float32)
        if description:
            desc[nm] = description
    bm = BlockModel(None, centroids, sizes, attrs, {}, null_value,
                    info={"name": name, "descriptions": desc,
                          "parent_size": [float(v) for v in size], "sub_blocking": sub,
                          "grid_origin": [float(v) for v in origin], "grid_blocks": [int(v) for v in n]})
    if progress:
        progress(1.0)
    return bm


# --- Surpac ASCII export --------------------------------------------------------------------
_SEPS = [",", "\t", ";", "|"]


def _header_line(lines: list[str]) -> tuple[int, str | None]:
    """(index, delimiter) of the column-name line: the first line holding centroid and block
    size names. Delimiter None = white space."""
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith(("#", "//", "!")):
            continue
        for sep in _SEPS + [None]:
            parts = [p.strip().strip('"') for p in (line.split(sep) if sep else line.split())]
            if len(parts) < 6:
                continue
            try:
                _detect_columns(parts)
            except ValueError:
                continue
            return i, sep
    raise ValueError(tr("No header with X, Y, Z and block size columns found (e.g. X,Y,Z,size(X),size(Y),size(Z),...)"))


def read_surpac_ascii(path, null_value: float = SURPAC_NULL) -> BlockModel:
    """Read a Surpac block model ASCII export (CSV / text: one row per block with the
    centroid, block size and attributes). Header lines before the column names (model
    description, origin, ...) are skipped. Nothing is cached: the data stays in memory only."""
    path = Path(path)
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        head = [f.readline() for _ in range(200)]
    idx, sep = _header_line(head)
    kw = {"sep": sep} if sep else {"sep": r"\s+"}
    try:
        if sep:
            df = pd.read_csv(path, skiprows=idx, engine="pyarrow", **kw)
        else:
            raise ValueError
    except Exception:  # noqa: BLE001 - pyarrow does not take every layout
        df = pd.read_csv(path, skiprows=idx, encoding="latin-1", skipinitialspace=True, **kw)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.lower().startswith("unnamed")]]
    cen, siz = _detect_columns(list(df.columns))
    df = df.dropna(subset=cen + siz)
    for c in cen + siz:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cen + siz)
    if df.empty:
        raise ValueError(tr("No blocks in {name}", name=path.name))
    bm = _from_dataframe(df.reset_index(drop=True), None, null_value)
    bm.info = {"name": path.stem, "descriptions": {}, "source_file": path.name}
    add_history(bm, f"Imported from Surpac ASCII {path.name} ({bm.n_blocks:,} blocks)")
    return bm


def describe(bm: BlockModel) -> dict:
    """Facts shown in the manager: blocks, extent, block sizes, volume."""
    lo, hi = bm.bounds()
    sizes = bm.block_sizes()
    vol = float(np.prod(bm.sizes.astype(np.float64), axis=1).sum())
    return {"n_blocks": bm.n_blocks, "lo": lo, "hi": hi, "sizes": sizes, "volume": vol,
            "sub_blocked": len(sizes) > 1}


def attribute_name_ok(bm: BlockModel, name: str, old: str | None = None) -> str:
    """'' when ``name`` can be an attribute name, else why not."""
    from ..scene.layers import DEPLETION_ATTRS

    n = (name or "").strip()
    if not n:
        return tr("Enter a name")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n):
        return tr("Use letters, digits and _ only, starting with a letter (e.g. gold, au_ok)")
    if len(n) > 40:
        return tr("At most 40 characters")
    low = n.lower()
    if low in DEPLETION_ATTRS:
        return tr("{name} is used by the reserve tools", name=n)
    if _norm(n) in {"x", "y", "z", "xc", "yc", "zc", "sizex", "sizey", "sizez", "xsize", "ysize", "zsize"}:
        return tr("{name} is a coordinate / block size name", name=n)
    if any(a.lower() == low for a in bm.attributes if a != old):
        return tr("There is already an attribute {name}", name=n)
    return ""


def csv_preview(path, rows: int = 5) -> tuple[list[str], list[list[str]]]:
    """Column names and the first rows of a Surpac ASCII export (for the import dialog)."""
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        head = [f.readline() for _ in range(200 + rows)]
    idx, sep = _header_line(head)
    text = "".join(head[idx:idx + rows + 1])
    if sep:
        data = list(csv.reader(io.StringIO(text), delimiter=sep))
    else:
        data = [ln.split() for ln in text.splitlines()]
    data = [[c.strip() for c in r] for r in data if r]
    return (data[0], data[1:]) if data else ([], [])
