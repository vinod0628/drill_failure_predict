"""Block model reader for CSV exports from Surpac (and similar packages).

The CSV is expected to hold one row per block (parent block or sub-block)
with the block centroid, block size and any number of attributes, e.g.::

    X,Y,Z,classification_no,size (X), size(Y), size(Z),gold,sg
    1280.0000,7110.0000,0.0000,1,20.000000,20.000000,20.000000,-99.00,2.23

Column names are matched loosely (case, spaces, brackets and underscores are
ignored). Numeric attributes are stored as float32 to keep memory down on
models with tens of millions of blocks. Text attributes are stored as int32
codes with a category list.

The first read of a CSV saves a binary cache so later loads skip CSV parsing; see
``dfpredict.io.cache``. The cache is written in the encrypted block model format
(``dfpredict.blockmodel.store``), never as plain arrays: block values are sensitive data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..i18n import tr
from .cache import cache_path_for

SURPAC_NULL = -99.0
_CACHE_KIND = "blockmodel-v2"            # v1 was a plain .npz (removed when seen)
_OLD_CACHE_KIND = "blockmodel-v1"

_CENTROID = {
    0: {"x", "xc", "xcentre", "xcenter", "xcentroid", "centroidx", "east", "easting", "xcen"},
    1: {"y", "yc", "ycentre", "ycenter", "ycentroid", "centroidy", "north", "northing", "ycen"},
    2: {"z", "zc", "zcentre", "zcenter", "zcentroid", "centroidz", "rl", "elev", "elevation", "zcen"},
}
_SIZE = {
    0: {"sizex", "xsize", "dimx", "xdim", "xinc", "dx", "blocksizex", "xlength", "xlen"},
    1: {"sizey", "ysize", "dimy", "ydim", "yinc", "dy", "blocksizey", "ylength", "ylen"},
    2: {"sizez", "zsize", "dimz", "zdim", "zinc", "dz", "blocksizez", "zlength", "zlen"},
}


def _norm(name: str) -> str:
    return re.sub(r"[\s_()\[\]\-.]", "", str(name)).lower()


@dataclass
class BlockGrid:
    """Integer description of a sub-blocked model on its finest grid."""
    origin: np.ndarray      # (3,) float64, minimum corner of the model
    cell: np.ndarray        # (3,) float64, finest block size per axis
    ijk_min: np.ndarray     # (N, 3) int32, minimum-corner cell index of each block
    span: np.ndarray        # (N, 3) int32, block size in finest cells
    shape: np.ndarray       # (3,) int64, grid size in finest cells


@dataclass
class BlockModel:
    path: Path | None
    centroids: np.ndarray                    # (N, 3) float64
    sizes: np.ndarray                        # (N, 3) float32
    attributes: dict[str, np.ndarray]        # name -> (N,) float32 or int32 codes
    categories: dict[str, list[str]] = field(default_factory=dict)  # text attrs: code -> label
    null_value: float = SURPAC_NULL
    # fixed colours of some text attributes (e.g. the computed mined "status"): name -> [rgb, ...]
    category_rgb: dict = field(default_factory=dict)
    # managed block models (.gvbm): name, attribute descriptions, history, ... (see blockmodel.store)
    info: dict = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.centroids)

    @property
    def attribute_names(self) -> list[str]:
        return list(self.attributes)

    def is_categorical(self, name: str) -> bool:
        return name in self.categories

    def valid_mask(self, name: str) -> np.ndarray:
        """True where the attribute has a value (not NaN and not the Surpac null)."""
        a = self.attributes[name]
        if self.is_categorical(name):
            return a >= 0
        return np.isfinite(a) & (a != np.float32(self.null_value))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = self.sizes.astype(np.float64) / 2.0
        return (self.centroids - half).min(axis=0), (self.centroids + half).max(axis=0)

    def block_sizes(self) -> np.ndarray:
        """Distinct block sizes, shape (K, 3)."""
        return np.unique(self.sizes, axis=0)

    def grid(self, tol: float = 1e-3) -> BlockGrid:
        """Map every block onto the finest grid.

        Requires every block size to be a whole multiple of the smallest
        size on that axis, which holds for regular and sub-blocked models.
        """
        cell = self.sizes.min(axis=0).astype(np.float64)
        lo = self.centroids - self.sizes.astype(np.float64) / 2.0
        origin = lo.min(axis=0)
        fmin = (lo - origin) / cell
        span_f = self.sizes.astype(np.float64) / cell
        ijk_min = np.rint(fmin)
        span = np.rint(span_f)
        if np.abs(fmin - ijk_min).max(initial=0) > tol or np.abs(span_f - span).max(initial=0) > tol:
            raise ValueError(tr("Blocks do not align to a common grid"))
        ijk_min = ijk_min.astype(np.int32)
        span = span.astype(np.int32)
        shape = (ijk_min + span).max(axis=0).astype(np.int64)
        return BlockGrid(origin=origin, cell=cell, ijk_min=ijk_min, span=span, shape=shape)


def _detect_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    normed = {c: _norm(c) for c in columns}
    cen, siz = [], []
    for axis in range(3):
        c = next((c for c in columns if normed[c] in _CENTROID[axis]), None)
        s = next((c for c in columns if normed[c] in _SIZE[axis]), None)
        if c is None:
            raise ValueError(tr("No centroid column for axis {axis} in {columns}", axis="XYZ"[axis], columns=columns))
        if s is None:
            raise ValueError(tr("No block size column for axis {axis} in {columns}", axis="XYZ"[axis], columns=columns))
        cen.append(c)
        siz.append(s)
    return cen, siz


def _from_dataframe(df: pd.DataFrame, path: Path | None, null_value: float) -> BlockModel:
    cen_cols, size_cols = _detect_columns(list(df.columns))
    centroids = df[cen_cols].to_numpy(dtype=np.float64)
    sizes = df[size_cols].to_numpy(dtype=np.float32)
    attrs: dict[str, np.ndarray] = {}
    cats: dict[str, list[str]] = {}
    for col in df.columns:
        if col in cen_cols or col in size_cols:
            continue
        name = str(col).strip()
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            attrs[name] = s.to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            cat = pd.Categorical(s.astype("string").str.strip())
            attrs[name] = np.asarray(cat.codes, dtype=np.int32)
            cats[name] = [str(c) for c in cat.categories]
    return BlockModel(path, centroids, sizes, attrs, cats, null_value)


def _save_cache(bm: BlockModel, cpath: Path) -> None:
    from ..blockmodel.store import save_block_model

    save_block_model(bm, cpath)


def _load_cache(cpath: Path, path: Path) -> BlockModel:
    from ..blockmodel.store import read_block_model

    bm = read_block_model(cpath)
    bm.path = path
    bm.info = {}
    return bm


def _drop_plain_cache(path: Path, cache_dir) -> None:
    """Remove the plain .npz cache earlier versions wrote for ``path``."""
    try:
        cache_path_for(path, _OLD_CACHE_KIND, cache_dir).unlink(missing_ok=True)
    except OSError:
        pass


def drop_csv_cache(path: str | Path, cache_dir: str | Path | None = None) -> None:
    """Remove the caches kept for a block model CSV (e.g. once the CSV is deleted)."""
    try:                                  # the cache name is keyed on the file: call before deleting it
        paths = (cache_path_for(path, _CACHE_KIND, cache_dir).with_suffix(".gvbm"),
                 cache_path_for(path, _OLD_CACHE_KIND, cache_dir))
    except OSError:
        return
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def read_block_csv(
    path: str | Path,
    null_value: float = SURPAC_NULL,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> BlockModel:
    path = Path(path)
    cpath = cache_path_for(path, _CACHE_KIND, cache_dir).with_suffix(".gvbm") if use_cache else None
    if use_cache:
        _drop_plain_cache(path, cache_dir)
    if cpath is not None and cpath.exists():
        try:
            return _load_cache(cpath, path)
        except Exception:
            cpath.unlink(missing_ok=True)
    try:
        df = pd.read_csv(path, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    bm = _from_dataframe(df, path, null_value)
    if cpath is not None:
        try:
            _save_cache(bm, cpath)
        except OSError:
            pass
    return bm
