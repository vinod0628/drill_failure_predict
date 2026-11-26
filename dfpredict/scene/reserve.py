"""The reserve of a workspace: setup (block model, reserve solid, density, grade), the
production log (prod_* tables of the project database) and the depletion computed from it. No Qt here."""

from __future__ import annotations

import numpy as np

from ..i18n import tr
from .production import STATUS_CATEGORIES, STATUS_COLORS, Depletion, ProductionLog, month_of, reserve_blocks

DEFAULT_SETUP = {"block_model": "", "solid": "", "density": "", "density_value": 2.7, "grade": "",
                 "unit": "g/t"}
GRADE_HINTS = ("au", "gold", "grade", "cu", "copper", "ag", "silver", "zn", "pb", "ni", "fe")
DENSITY_HINTS = ("sg", "density", "dens", "bd", "bulk_density")
_KINDS: dict = {}      # id(DtmFile) -> (DtmFile, "solid" / "surface")


class Reserve:
    """``update(layers)`` (re)computes after any change; ``inject(as_of)`` puts the computed
    ``mined_fraction`` / ``status`` attributes into the block model (memory only)."""

    def __init__(self):
        self.log: ProductionLog | None = None
        self.dep: Depletion | None = None
        self.error = tr("No workspace")
        self.open_error = ""
        self.bm_layer = None
        self._key = None
        self._inside_key = None
        self._inside = None

    # --- setup ---------------------------------------------------------------------
    @property
    def setup(self) -> dict:
        return {**DEFAULT_SETUP, **(self.log.reserve if self.log else {})}

    def open(self, folder) -> None:
        """Read the production database of ``folder``. When it cannot be read, ``log`` is None
        and ``open_error`` says why (nothing is written to an unreadable database)."""
        self.dep, self._key = None, None
        self.open_error = ""
        from ..io.database import project_target

        try:
            self.log = ProductionLog.load(folder)
        except Exception as e:  # noqa: BLE001 - damaged / locked / foreign file, server down
            self.log = None
            self.open_error = tr("Could not read the production records ({where}): {error}",
                                 where=project_target(folder).label, error=e)

    @staticmethod
    def block_layers(layers) -> list:
        from .layers import BlockModelLayer

        return [lay for lay in layers if isinstance(lay, BlockModelLayer) and lay.data is not None]

    @staticmethod
    def solid_layers(layers) -> list:
        from .layers import SurfaceLayer
        from .solid import dtm_kind

        out = []
        for lay in layers:
            if isinstance(lay, SurfaceLayer) and lay.data is not None:
                kind = _KINDS.get(id(lay.data))
                if kind is None or kind[0] is not lay.data:
                    try:
                        kind = (lay.data, dtm_kind(lay.data))
                    except Exception:  # noqa: BLE001 - an odd DTM is just not offered
                        kind = (lay.data, "surface")
                    if len(_KINDS) > 200:
                        _KINDS.clear()
                    _KINDS[id(lay.data)] = kind
                if kind[1] == "solid":
                    out.append(lay)
        return out

    @staticmethod
    def number_names(bm) -> list[str]:
        return [n for n in bm.attribute_names if not bm.is_categorical(n) and n != "mined_fraction"]

    def auto_setup(self, layers) -> bool:
        """Fill what is not chosen yet with sensible choices; True when the setup changed.
        A choice the user made is kept even while its layer is not loaded."""
        if self.log is None:
            return False
        saved = dict(self.log.reserve or {})
        s = {**DEFAULT_SETUP, **saved}
        bms = self.block_layers(layers)
        if not s["block_model"] and bms:
            s["block_model"] = bms[0].name
        bm_layer = next((b for b in bms if b.name == s["block_model"]), None)
        if bm_layer is not None:
            nums = self.number_names(bm_layer.data)
            if s["grade"] not in nums:
                pref = [n for n in nums if n.lower() in GRADE_HINTS]
                pref += [n for n in nums if n.lower() not in DENSITY_HINTS and "class" not in n.lower()]
                s["grade"] = (pref or nums or [""])[0]
            if "density" not in saved:
                s["density"] = next((n for n in nums if n.lower() in DENSITY_HINTS), "")
            elif s["density"] and s["density"] not in nums:
                s["density"] = ""
        if "solid" not in saved:
            solids = [lay.name for lay in self.solid_layers(layers)]
            ore = [n for n in solids if "ore" in n.lower() or "reserve" in n.lower()]
            s["solid"] = (ore or solids or [""])[0]
        if bm_layer is None and "solid" not in saved:
            s.pop("solid")                  # decide once the block model is there
        if s != saved:
            self.log.reserve = s
            return True
        return False

    def set_setup(self, **kw) -> None:
        self.log.reserve = {**self.setup, **kw}
        self._key = None

    def renamed(self, old: str, new: str) -> dict:
        """A layer was renamed in the layer list: the setup follows when it named that layer as
        the block model or the reserve solid. Returns what changed ({} when nothing)."""
        if self.log is None:
            return {}
        s = self.setup
        return {k: new for k in ("block_model", "solid") if s.get(k) == old}

    # --- compute ---------------------------------------------------------------------
    def _fail(self, msg: str) -> bool:
        self.error, self.dep = msg, None
        return False

    def update(self, layers) -> bool:
        """Recompute when needed. Returns True when the depletion is ready."""
        if self.log is None:
            return self._fail(self.open_error or tr("No workspace"))
        s = self.setup
        bm_layer = next((b for b in self.block_layers(layers) if b.name == s["block_model"]), None)
        self.bm_layer = bm_layer
        if bm_layer is None:
            return self._fail(tr("Open the block model first") if not s["block_model"]
                              else tr("Block model {name} is not loaded", name=s["block_model"]))
        bm = bm_layer.data
        if not s["grade"] or s["grade"] not in bm.attributes:
            return self._fail(tr("Choose the grade attribute"))
        solid = None
        if s["solid"]:
            solid = next((lay for lay in self.solid_layers(layers) if lay.name == s["solid"]), None)
            if solid is None:
                return self._fail(tr("Reserve solid {name} is not loaded", name=s["solid"]))
        dens_name = s["density"] if s["density"] in bm.attributes else None
        key = (id(bm), id(solid.data) if solid else None, dens_name, float(s["density_value"]), s["grade"])
        if key != self._key or self.dep is None:
            inside = None
            if solid is not None:
                ik = (id(bm), id(solid.data))
                if ik != self._inside_key:
                    from .solid import closed_mesh, points_inside

                    self._inside = points_inside(bm.centroids, *closed_mesh(solid.data))
                    self._inside_key = ik
                inside = self._inside
            idx, dens, grade = reserve_blocks(bm, inside, dens_name, float(s["density_value"]), s["grade"])
            self.dep = Depletion(bm, idx, dens, grade)
            self._key = key
        self.error = "" if len(self.dep.idx) else tr("No reserve blocks (no block with a grade inside the reserve "
                                                     "solid)")
        self.dep.apply(self.log.shapes)
        return True

    @property
    def ready(self) -> bool:
        return self.dep is not None

    def inject(self, as_of: int | None = None):
        """Put ``mined_fraction`` and ``status`` into the reserve block model (memory only);
        returns that BlockModel (None when not ready)."""
        if self.dep is None:
            return None
        bm = self.dep.bm
        bm.attributes.update(self.dep.block_attributes(as_of))
        bm.categories["status"] = list(STATUS_CATEGORIES)
        bm.category_rgb["status"] = list(STATUS_COLORS)
        return bm


def ensure_depletion_attrs(bm) -> bool:
    """Placeholder (null) mined attributes, so a saved colouring by them can be shown before
    the reserve is computed. True when added."""
    if "status" in bm.attributes and "mined_fraction" in bm.attributes:
        return False
    bm.attributes.setdefault("mined_fraction", np.full(bm.n_blocks, np.nan, dtype=np.float32))
    bm.attributes.setdefault("status", np.full(bm.n_blocks, -1, dtype=np.int32))
    bm.categories["status"] = list(STATUS_CATEGORIES)
    bm.category_rgb["status"] = list(STATUS_COLORS)
    return True


def clear_depletion_attrs(bm) -> None:
    """Mined attributes back to null (e.g. the production records cannot be read)."""
    if "status" in bm.attributes:
        bm.attributes["status"][:] = -1
    if "mined_fraction" in bm.attributes:
        bm.attributes["mined_fraction"][:] = np.nan


def uses_depletion(layer) -> bool:
    """A block model layer drawn from the mined attributes: mined blocks shown gray /
    transparent / hidden, or a constraint on them (or an old colouring by them)."""
    if getattr(layer, "mined_display", None) in ("gray", "transparent", "hidden"):
        return True
    if getattr(layer, "color_by", None) in ("mined_fraction", "status"):
        return True
    return any(c.get("kind") == "value" and c.get("name") in ("mined_fraction", "status")
               for c in getattr(layer, "constraints", None) or [])


def monthly(log: ProductionLog, dep: Depletion) -> list[dict]:
    """Production per month: shapes, tonnes, grade, metal, cumulative tonnes and remaining."""
    total = float(dep.tonnes.sum())
    total_metal = float((dep.tonnes * dep.grade).sum())
    rows: dict[str, dict] = {}
    for sh in log.sorted_shapes():
        tot, _blocks = dep.shape_totals(sh.id)
        m = month_of(sh.date)
        r = rows.setdefault(m, {"month": m, "shapes": 0, "tonnes": 0.0, "metal": 0.0})
        r["shapes"] += 1
        r["tonnes"] += tot.tonnes
        r["metal"] += tot.metal
    out, cum, cum_m = [], 0.0, 0.0
    for m in sorted(rows):
        r = rows[m]
        cum += r["tonnes"]
        cum_m += r["metal"]
        out.append({**r, "grade": r["metal"] / r["tonnes"] if r["tonnes"] > 0 else 0.0,
                    "cumulative": cum, "remaining": total - cum, "remaining_metal": total_metal - cum_m})
    return out
