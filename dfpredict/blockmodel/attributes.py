"""Add, rename, describe, remove and fill block model attributes (all in memory; the
manager saves the block model afterwards). New attributes are float numbers."""

from __future__ import annotations

import numpy as np

from ..i18n import tr
from ..io.block_model import BlockModel
from .build import attribute_name_ok
from .store import add_history


def _check(bm: BlockModel, name: str, old: str | None = None) -> str:
    why = attribute_name_ok(bm, name, old)
    if why:
        raise ValueError(why)
    return name.strip()


def descriptions(bm: BlockModel) -> dict:
    return bm.info.setdefault("descriptions", {})


def add_attribute(bm: BlockModel, name: str, default: float | None = None, description: str = "") -> str:
    name = _check(bm, name)
    value = bm.null_value if default is None else float(default)
    bm.attributes[name] = np.full(bm.n_blocks, value, dtype=np.float32)
    if description:
        descriptions(bm)[name] = description
    add_history(bm, f"Added attribute {name} (default {value:g})")
    return name


def rename_attribute(bm: BlockModel, old: str, new: str) -> str:
    if old not in bm.attributes:
        raise KeyError(old)
    new = _check(bm, new, old)
    if new == old:
        return new
    bm.attributes = {(new if k == old else k): v for k, v in bm.attributes.items()}
    if old in bm.categories:
        bm.categories[new] = bm.categories.pop(old)
    d = descriptions(bm)
    if old in d:
        d[new] = d.pop(old)
    add_history(bm, f"Renamed attribute {old} to {new}")
    return new


def set_description(bm: BlockModel, name: str, text: str) -> None:
    d = descriptions(bm)
    if text:
        d[name] = text
    else:
        d.pop(name, None)


def remove_attribute(bm: BlockModel, name: str) -> None:
    bm.attributes.pop(name)
    bm.categories.pop(name, None)
    descriptions(bm).pop(name, None)
    add_history(bm, f"Removed attribute {name}")


def fill_attribute(bm: BlockModel, name: str, value: float | None, mask: np.ndarray | None = None,
                   where: str = "") -> int:
    """Set ``name`` to ``value`` (None = the null value) on the blocks in ``mask`` (all when
    None). Returns how many blocks were set."""
    if bm.is_categorical(name):
        raise ValueError(tr("{name} is a text attribute: only number attributes can be filled", name=name))
    a = bm.attributes[name]
    v = bm.null_value if value is None else float(value)
    if mask is None:
        a[:] = v
        n = bm.n_blocks
    else:
        a[mask] = v
        n = int(np.count_nonzero(mask))
    add_history(bm, f"Set {name} = {v:g} on {n:,} blocks" + (f" ({where})" if where else ""))
    return n


def attribute_stats(bm: BlockModel, name: str) -> dict:
    """Blocks with a value, min, mean (volume weighted) and max of a number attribute."""
    a = bm.attributes[name]
    m = bm.valid_mask(name)
    n = int(m.sum())
    if bm.is_categorical(name):
        return {"n": n, "text": True, "codes": len(bm.categories.get(name, []))}
    if not n:
        return {"n": 0}
    vol = np.prod(bm.sizes[m].astype(np.float64), axis=1)
    v = a[m].astype(np.float64)
    return {"n": n, "min": float(v.min()), "mean": float(np.average(v, weights=vol)), "max": float(v.max())}
