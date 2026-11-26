"""Per-metre features (design spec 5.1, 5.3).

One row per metre of hole. The same function makes the rows for a past hole (from its geology log,
optionally with noise, spec 5.6.1) and for the metres ahead of the bit (from the workspace geological
model along a predicted path, spec 5.6.3): both are first turned into a :class:`HoleContext` - unit
intervals with their property cards, fault zones, hole sizes, bits and runs along the hole.

Coordinates and formation names are left out on purpose (spec 5.3, 5.10): what is learnt carries over
through rock properties and position relative to contacts. No Qt here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..i18n import N_
from . import codes

# --- feature groups (spec 5.3) -------------------------------------------------------------------
# rock: formation card + the mechanical properties of the rock class (rockmech.py, editable: risk_rock)
ROCK = ["carb", "dolo", "hard", "abr", "bedding", "clay", "karst", "clastic", "ucs", "cai", "e_mod", "brittle",
        "aniso", "soluble", "porosity", "density"]
POSITION = ["d_top", "d_base", "relpos", "thick", "thick_anom"]
CONTACT = ["up_carb", "up_hard", "up_clastic", "up_ucs_ratio", "ct_clastic_carb", "ct_hard_soft", "d_first_carb",
           "carb_drilled",
           "d_unconf"]
GEOMETRY = ["incidence"]
FAULTS = ["in_fault", "d_fault"]
HAZARD = ["h_cavity", "h_fault", "h_cavity10", "h_fault10"]
KARST = ["above_wt", "s_dip", "s_fold", "s_fpkm", "s_cexp", "s_unconf", "s_sink", "s_valley"]
HOLE = ["depth", "diam", "n_red", "incl", "dogleg", "depth_ratio"]
SETUP = ["bit", "rpm", "flow"]
RESPONSE = ["pen_rev", "pq", "d_pen", "d_pq", "rop_rel", "ret", "rod_drop"]

BASE = ROCK + POSITION + CONTACT + GEOMETRY + FAULTS + KARST + HOLE + SETUP + RESPONSE
# The base of the unit a failed hole stopped in is never logged: its position features would give the
# failure away (spec 5.9, leakage). Models A and B use only the distance below the top of the unit.
POSITION_TOP = ["d_top"]
MODEL_FEATURES = {
    "H": ROCK + POSITION + CONTACT + GEOMETRY + KARST,
    "A": ROCK + POSITION_TOP + CONTACT + GEOMETRY + FAULTS + HAZARD + KARST + HOLE + SETUP,
    "B": ROCK + POSITION_TOP + CONTACT + GEOMETRY + FAULTS + HAZARD + KARST + HOLE + SETUP + RESPONSE,
}
FEATURE_TEXT = {      # for the explanation of a warning (SHAP top features)
    "carb": N_("carbonate share"), "dolo": N_("dolomite"), "hard": N_("hardness"), "abr": N_("abrasiveness"),
    "bedding": N_("bedding style"), "clay": N_("swelling clay"), "karst": N_("known karst"), "clastic": N_("clastic rock"),
    "ucs": N_("rock strength (UCS)"), "cai": N_("abrasivity (CAI)"), "e_mod": N_("stiffness (Young's modulus)"),
    "brittle": N_("brittleness (UCS / tensile)"), "aniso": N_("foliation / anisotropy"), "soluble": N_("solubility"),
    "porosity": N_("porosity"), "density": N_("density"), "up_ucs_ratio": N_("strength contrast with the unit above"),
    "d_top": N_("distance below the top of the unit"), "d_base": N_("distance above the base of the unit"),
    "relpos": N_("position in the unit"), "thick": N_("unit thickness"), "thick_anom": N_("thickness anomaly"),
    "up_carb": N_("carbonate share of the unit above"), "up_hard": N_("hardness of the unit above"),
    "up_clastic": N_("clastic unit above"), "ct_clastic_carb": N_("clastic-over-carbonate contact"),
    "ct_hard_soft": N_("hard-over-soft contact"), "d_first_carb": N_("distance below the top of the first carbonate"),
    "carb_drilled": N_("carbonate drilled so far"), "d_unconf": N_("distance below the unconformity"),
    "incidence": N_("angle between hole and bedding"), "in_fault": N_("inside a fault zone"),
    "d_fault": N_("distance to the nearest fault"), "h_cavity": N_("cavity probability"), "h_fault": N_("fault-zone probability"),
    "h_cavity10": N_("cavity probability in the next 10 m"), "h_fault10": N_("fault-zone probability in the next 10 m"),
    "above_wt": N_("height above the water table"), "s_dip": N_("area bedding dip"), "s_fold": N_("fold tightness"),
    "s_fpkm": N_("faults per km"), "s_cexp": N_("carbonate exposed"), "s_unconf": N_("unconformity above carbonate"),
    "s_sink": N_("sinkholes mapped"), "s_valley": N_("carbonate height above valley floor"), "depth": N_("depth"),
    "diam": N_("hole diameter"), "n_red": N_("size reductions so far"), "incl": N_("inclination"), "dogleg": N_("dogleg so far"),
    "depth_ratio": N_("depth / rig rated depth"), "bit": N_("bit type"), "rpm": N_("RPM"), "flow": N_("pump flow"),
    "pen_rev": N_("penetration per revolution"), "pq": N_("pressure / flow"), "d_pen": N_("change in penetration per rev"),
    "d_pq": N_("change in pressure / flow"), "rop_rel": N_("ROP / normal ROP for the rock"), "ret": N_("return %"),
    "rod_drop": N_("rod drop"),
}
# Mismatch table for noise (spec 5.6.1): a rock class may be logged / modelled as another.
MISMATCH = {"dolomite": "limestone", "limestone": "dolomite", "sandstone": "quartzite", "quartzite": "sandstone",
            "siltstone": "shale", "shale": "siltstone", "marl": "limestone"}


# --- property cards -------------------------------------------------------------------------------
@dataclass
class Props:
    """Rock properties of a unit interval: the formation card (karst, clay, bedding, thickness, carbonate share,
    lab UCS / CAI) and the mechanical properties of the logged rock's class."""

    carb: float
    dolo: float
    hard: float               # strength class 1-5 (ISRM, from UCS)
    abr: float                # abrasivity class 1-5 (from CAI)
    bedding: float
    clay: float
    karst: float
    clastic: float
    ucs: float = np.nan       # MPa
    cai: float = np.nan
    e_mod: float = np.nan     # GPa
    brittle: float = np.nan   # UCS / tensile
    aniso: float = np.nan     # 0 - 2
    soluble: float = np.nan   # 0 - 2
    porosity: float = np.nan  # %
    density: float = np.nan   # g/cm3
    thick_min: float = np.nan
    thick_max: float = np.nan
    rock_class: str = ""
    formation: str = ""

    def vector(self) -> list[float]:
        return [self.carb, self.dolo, self.hard, self.abr, self.bedding, self.clay, self.karst, self.clastic,
                self.ucs, self.cai, self.e_mod, self.brittle, self.aniso, self.soluble, self.porosity, self.density]


def _f(v, default=np.nan) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def props_for(rock: str | None, formation: str | None, cards: dict, swap_class: str | None = None) -> Props:
    """Properties of one interval. The logged rock's class gives the mechanical properties (UCS, CAI,
    stiffness, brittleness, anisotropy, solubility, porosity, density - rockmech.py / risk_rock); the
    formation card gives the formation-level values (karst, clay, bedding, thickness range, carbonate
    share) and, where measured, the lab UCS / CAI of the formation's main rock."""
    from . import rockmech

    card = cards.get(formation or "") or {}
    main = card.get("main_rock")
    cls = swap_class or codes.rock_class_of(rock or main or "", main)
    rp = rockmech.props(cls)
    ucs, cai, carb = rp.ucs_typ, rp.cai_typ, rp.carbonate
    is_main = rock is None or codes.rock_class_of(rock, None) == main
    if card:
        carb = _f(card.get("carbonate_pct"), carb) if is_main else carb
        if is_main:
            ucs = _f(card.get("ucs_mpa"), ucs)
            cai = _f(card.get("cai_lab"), cai)
    hard = float(rockmech.strength_class(ucs))
    abr = float(rockmech.abrasivity_class(cai))
    if card and is_main and _f(card.get("ucs_mpa")) != _f(card.get("ucs_mpa")):       # no lab UCS: card classes
        hard = _f(card.get("hardness"), hard)
        abr = _f(card.get("abrasiveness"), abr)
    clay_default = float(rp.swelling)
    return Props(carb=carb, dolo=1.0 if cls in ("dolomite", "siliceous_dolomite") else 0.0, hard=hard, abr=abr,
                 bedding=codes.BEDDING_CODE.get(card.get("bedding") or "", np.nan),
                 clay=float(codes.CLAY_CODE.get(card.get("swelling_clay") or "", clay_default)),
                 karst=codes.KARST_CODE.get(card.get("karst") or "", float(rp.soluble) if rp.soluble else np.nan),
                 clastic=1.0 if rp.clastic else 0.0, ucs=ucs, cai=cai, e_mod=rp.e_mod,
                 brittle=ucs / max(rp.tensile * ucs / max(rp.ucs_typ, 0.1), 0.1), aniso=float(rp.anisotropy),
                 soluble=float(rp.soluble), porosity=rp.porosity, density=rp.density,
                 thick_min=_f(card.get("thick_min")), thick_max=_f(card.get("thick_max")), rock_class=cls,
                 formation=formation or "")


# --- the hole context ---------------------------------------------------------------------------
@dataclass
class HoleContext:
    """Everything along one hole that the features need (depths along the hole, m)."""

    md: np.ndarray                         # (n,) metre centres
    xyz: np.ndarray                        # (n, 3)
    dirs: np.ndarray                       # (n, 3) unit direction (E, N, Up)
    iv_from: np.ndarray                    # unit intervals, sorted, covering the hole
    iv_to: np.ndarray
    iv_props: list[Props]
    incidence: np.ndarray                  # (n,) degrees between hole and bedding
    fault_zones: list = field(default_factory=list)          # (from, to)
    sizes: list = field(default_factory=list)                # (from, to, size code)
    bits: list = field(default_factory=list)                 # (from, to, bit type)
    runs: pd.DataFrame | None = None                         # depth_from, depth_to, rpm, rop, ...
    area: dict = field(default_factory=dict)
    rig: dict = field(default_factory=dict)
    planned: dict = field(default_factory=dict)              # rpm / flow / bit / size for metres without runs
    live_from: float | None = None         # metres below this use the last runs' medians (ahead of the bit)
    open_end: bool = False                 # the last interval ends where the log ends (its base is unknown)


def _interval_index(md: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(starts, md, side="right") - 1, 0, max(len(starts) - 1, 0))


def build_features(ctx: HoleContext, rop_norms: dict | None = None) -> pd.DataFrame:
    """The feature table (columns ``BASE``; HAZARD columns are added by the models) of one hole."""
    md = ctx.md
    n = len(md)
    out: dict[str, np.ndarray] = {}
    if n == 0:
        return pd.DataFrame(columns=BASE)
    k = _interval_index(md, ctx.iv_from)
    P = np.array([p.vector() for p in ctx.iv_props], dtype=np.float64)        # (I, 8)
    for j, name in enumerate(ROCK):
        out[name] = P[k, j]
    thick = (ctx.iv_to - ctx.iv_from)
    out["d_top"] = md - ctx.iv_from[k]
    out["d_base"] = ctx.iv_to[k] - md
    out["thick"] = thick[k]
    out["relpos"] = np.clip(out["d_top"] / np.maximum(thick[k], 0.1), 0, 1)
    tmin = np.array([p.thick_min for p in ctx.iv_props])[k]
    tmax = np.array([p.thick_max for p in ctx.iv_props])[k]
    sin_inc = np.clip(np.sin(np.radians(np.nan_to_num(ctx.incidence, nan=60.0))), 0.2, 1.0)
    true_thick = thick[k] * sin_inc
    out["thick_anom"] = (true_thick - (tmin + tmax) / 2) / np.maximum(tmax - tmin, 5.0)
    if ctx.open_end:                        # the unit the log ends in: its base is not known
        last = k == len(ctx.iv_from) - 1
        for name in ("d_base", "thick", "relpos", "thick_anom"):
            out[name] = np.where(last, np.nan, out[name])
    # contact setting: the unit above (in the hole)
    up = np.where(k > 0, k - 1, -1)
    up_props = np.where(up[:, None] >= 0, P[np.maximum(up, 0)], np.nan)
    out["up_carb"], out["up_hard"], out["up_clastic"] = up_props[:, 0], up_props[:, 2], up_props[:, 7]
    out["up_ucs_ratio"] = np.log10(np.clip(up_props[:, 8], 0.1, None) / np.clip(out["ucs"], 0.1, None))
    out["ct_clastic_carb"] = ((out["up_clastic"] > 0.5) & (out["carb"] >= 50)).astype(float)
    out["ct_hard_soft"] = out["up_hard"] - out["hard"]
    carb_iv = P[:, 0] >= 50
    if carb_iv.any():
        first = ctx.iv_from[np.argmax(carb_iv)]
        out["d_first_carb"] = np.where(md >= first, md - first, np.nan)
    else:
        out["d_first_carb"] = np.full(n, np.nan)
    # carbonate drilled so far
    cum = np.zeros(n)
    for i, is_c in enumerate(carb_iv):
        if is_c:
            cum += np.clip(md - ctx.iv_from[i], 0, ctx.iv_to[i] - ctx.iv_from[i])
    out["carb_drilled"] = cum
    unconf = str(ctx.area.get("unconformity") or "").upper() == "Y"
    out["d_unconf"] = out["d_first_carb"] if unconf else np.full(n, np.nan)
    out["incidence"] = np.asarray(ctx.incidence, np.float64)
    # faults
    inf = np.zeros(n)
    dfault = np.full(n, 300.0)
    for a, b in ctx.fault_zones:
        inf = np.maximum(inf, ((md >= a) & (md <= b)).astype(float))
        dfault = np.minimum(dfault, np.maximum(0.0, np.maximum(a - md, md - b)))
    out["in_fault"], out["d_fault"] = inf, dfault
    # karst setting (site sheet)
    a = ctx.area
    wt = _f(a.get("water_table"))
    out["above_wt"] = ctx.xyz[:, 2] - wt if not math.isnan(wt) else np.full(n, np.nan)
    dmin, dmax = _f(a.get("dip_min")), _f(a.get("dip_max"))
    for name, v in (("s_dip", np.nanmean([dmin, dmax]) if not (math.isnan(dmin) and math.isnan(dmax)) else np.nan),
                    ("s_fold", _f(a.get("fold_tightness"))), ("s_fpkm", _f(a.get("faults_per_km"))),
                    ("s_cexp", _yn(a.get("carbonate_exposed"))), ("s_unconf", _yn(a.get("unconformity"))),
                    ("s_sink", _yn(a.get("sinkholes"))), ("s_valley", _f(a.get("carb_above_valley")))):
        out[name] = np.full(n, v, dtype=np.float64)
    # hole state
    out["depth"] = md.astype(np.float64)
    diam = np.full(n, np.nan)
    nred = np.zeros(n)
    size_at = np.array([""] * n, dtype=object)
    prev = None
    for f, t, s in sorted(ctx.sizes):
        m = (md >= f) & (md < t)
        diam[m] = codes.HOLE_SIZES.get(s, np.nan)
        size_at[m] = s
        if prev is not None and s != prev:
            nred[md >= f] += 1
        prev = s
    if ctx.planned.get("size") and np.isnan(diam).any():
        m = np.isnan(diam)
        diam[m] = codes.HOLE_SIZES.get(ctx.planned["size"], np.nan)
        size_at[m] = ctx.planned["size"]
    out["diam"], out["n_red"] = diam, nred
    out["incl"] = np.degrees(np.arcsin(np.clip(-ctx.dirs[:, 2], -1, 1)))
    out["dogleg"] = dogleg_so_far(md, ctx.dirs)
    ratio = np.full(n, np.nan)
    for s in codes.HOLE_SIZES:
        rd = _f(ctx.rig.get(f"depth_{s}"))
        if rd > 0:
            ratio[size_at == s] = md[size_at == s] / rd
    out["depth_ratio"] = ratio
    # drilling setup and response
    bit = np.full(n, np.nan)
    for f, t, b in ctx.bits:
        bit[(md >= f) & (md < t)] = codes.BIT_CODE.get(b, np.nan)
    if ctx.planned.get("bit") is not None:
        bit[np.isnan(bit)] = codes.BIT_CODE.get(ctx.planned["bit"], np.nan)
    out["bit"] = bit
    resp = _run_features(ctx, md, rop_norms, P, k)
    out.update(resp)
    df = pd.DataFrame(out)
    return df[BASE]


def _yn(v) -> float:
    s = str(v or "").upper()
    return 1.0 if s == "Y" else (0.0 if s == "N" else np.nan)


def dogleg_so_far(md: np.ndarray, dirs: np.ndarray, span: float = 30.0) -> np.ndarray:
    """Running maximum of the dogleg severity (degrees per 30 m)."""
    n = len(md)
    if n < 2:
        return np.zeros(n)
    j = np.searchsorted(md, md - span)
    cosb = np.clip(np.einsum("ij,ij->i", dirs, dirs[j]), -1, 1)
    dl = np.degrees(np.arccos(cosb)) * span / np.maximum(md - md[j], 1.0)
    dl[md - md[j] < span * 0.5] = 0.0
    return np.maximum.accumulate(dl)


def _run_features(ctx: HoleContext, md, rop_norms, P, k) -> dict:
    n = len(md)
    nan = np.full(n, np.nan)
    out = {c: nan.copy() for c in ("rpm", "flow") + tuple(RESPONSE)}
    runs = ctx.runs
    if runs is not None and len(runs):
        r = runs.sort_values("depth_from")
        f = r["depth_from"].to_numpy(float)
        t = r["depth_to"].to_numpy(float)
        g = lambda c: r[c].to_numpy(float) if c in r else np.full(len(r), np.nan)  # noqa: E731
        rpm, rop, flow, press = g("rpm"), g("rop"), g("flow_in"), g("pump_press")
        ret, rod = g("return_pct"), g("rod_drop")
        pen = rop / np.where(rpm > 0, rpm, np.nan) * 1000.0           # mm per revolution x 1000/60 scale
        pq = press / np.where(flow > 0, flow, np.nan) * 100.0
        prev_pen = pd.Series(pen).shift(1).rolling(3, min_periods=1).mean().to_numpy()
        prev_pq = pd.Series(pq).shift(1).rolling(3, min_periods=1).mean().to_numpy()
        d_pen = pen / prev_pen - 1.0
        d_pq = pq / prev_pq - 1.0
        idx = np.searchsorted(f, md, side="right") - 1
        inside = (idx >= 0) & (md < t[np.clip(idx, 0, len(t) - 1)] + 1e-9)
        if ctx.live_from is not None:
            inside &= md <= ctx.live_from + 1e-9
        ii = idx[inside]
        for name, arr in (("rpm", rpm), ("flow", flow), ("pen_rev", pen), ("pq", pq), ("d_pen", d_pen),
                          ("d_pq", d_pq), ("ret", ret), ("rod_drop", rod)):
            out[name][inside] = arr[ii]
        if rop_norms:
            cls = np.array([p.rock_class for p in ctx.iv_props], dtype=object)[k]
            bitc = r["bit_type"].to_numpy() if "bit_type" in r else None
            norm = np.array([rop_norms.get(c, np.nan) for c in cls], dtype=float)
            ropm = np.full(n, np.nan)
            ropm[inside] = rop[ii]
            out["rop_rel"] = ropm / norm
            del bitc
        # ahead of the bit: the medians of the last 3 runs (spec 5.3 drilling setup), response as last run
        ahead = ~inside & (md > (t.max() if len(t) else 0) - 1e-9)
        if ctx.live_from is not None:
            ahead = md > ctx.live_from
        if ahead.any() and len(r):
            last3 = slice(max(0, len(r) - 3), len(r))
            out["rpm"][ahead] = np.nanmedian(rpm[last3]) if np.isfinite(rpm[last3]).any() else np.nan
            out["flow"][ahead] = np.nanmedian(flow[last3]) if np.isfinite(flow[last3]).any() else np.nan
            # response features persist only a few metres ahead (Model B rescores the next 3-10 m)
            near = ahead & (md <= (ctx.live_from if ctx.live_from is not None else t.max()) + 10.0)
            for name, arr in (("pen_rev", pen), ("pq", pq), ("d_pen", d_pen), ("d_pq", d_pq), ("ret", ret),
                              ("rod_drop", rod)):
                out[name][near] = arr[-1]
            if rop_norms:
                out["rop_rel"][near] = out["rop_rel"][inside][-1] if inside.any() else np.nan
    for name in ("rpm", "flow"):
        v = ctx.planned.get(name)
        if v is not None:
            m = np.isnan(out[name])
            out[name][m] = float(v)
    return out


# --- contexts from records ---------------------------------------------------------------------
def intervals_from_log(geo: pd.DataFrame, end: float, cards: dict, rng: np.random.Generator | None = None,
                       noise: dict | None = None) -> tuple[np.ndarray, np.ndarray, list[Props], np.ndarray]:
    """Unit intervals from a geology log: consecutive rows of the same formation (or rock, without a
    formation) are one unit. With ``rng``: noise as in spec 5.6.1 - contacts shifted, thin units
    dropped now and then, rock classes swapped by the mismatch table. Also returns the logged bedding
    angle (to core axis) per interval."""
    g = geo.sort_values("depth_from")
    rows = []
    for r in g.itertuples(index=False):
        unit = r.formation if isinstance(r.formation, str) and r.formation else r.rock
        f, t = float(r.depth_from), float(r.depth_to)
        if rows and rows[-1]["unit"] == unit and abs(rows[-1]["to"] - f) < 0.5:
            rows[-1]["to"] = t
            continue
        rows.append({"unit": unit, "from": f, "to": t, "rock": r.rock, "formation": r.formation,
                     "bed": _f(getattr(r, "bedding_angle", np.nan))})
    if not rows:
        return np.array([0.0]), np.array([end]), [props_for(None, None, cards)], np.array([np.nan])
    rows[0]["from"] = 0.0
    rows[-1]["to"] = max(rows[-1]["to"], end)
    for a, b in zip(rows, rows[1:]):            # close gaps
        a["to"] = b["from"] = (a["to"] + b["from"]) / 2 if a["to"] < b["from"] else b["from"]
    swaps = {}
    if rng is not None:
        nz = noise or {}
        sd, p_drop, p_swap = nz.get("shift", 6.0), nz.get("drop", 0.08), nz.get("swap", 0.05)
        for a, b in zip(rows, rows[1:]):        # shift each contact
            lo, hi = a["from"] + 0.5, b["to"] - 0.5
            if hi > lo:
                c = float(np.clip(b["from"] + rng.normal(0, sd), lo, hi))
                a["to"] = b["from"] = c
        keep = [rows[0]]
        for r in rows[1:]:
            if r["to"] - r["from"] < 6.0 and rng.random() < p_drop:
                keep[-1]["to"] = r["to"]                    # a thin unit missing in the model
            else:
                keep.append(r)
        rows = keep
        for i, r in enumerate(rows):
            cls = codes.rock_class_of(r["rock"], (cards.get(r["formation"] or "") or {}).get("main_rock"))
            if cls in MISMATCH and rng.random() < p_swap:
                swaps[i] = MISMATCH[cls]
    props = [props_for(r["rock"], r["formation"], cards, swaps.get(i)) for i, r in enumerate(rows)]
    return (np.array([r["from"] for r in rows]), np.array([r["to"] for r in rows]), props,
            np.array([r["bed"] for r in rows]))


def rop_norms_from_runs(runs: pd.DataFrame, geology: pd.DataFrame, cards: dict) -> dict:
    """Median ROP per rock class over all runs (the 'normal ROP' of spec 5.3)."""
    if runs is None or not len(runs) or geology is None or not len(geology):
        return {}
    g = geology.sort_values(["hole_id", "depth_from"])
    out = {}
    mids = (runs["depth_from"] + runs["depth_to"]) / 2
    cls = pd.Series(index=runs.index, dtype=object)
    for hid, rr in runs.groupby("hole_id"):
        gg = g[g["hole_id"] == hid]
        if not len(gg):
            continue
        k = np.clip(np.searchsorted(gg["depth_from"].to_numpy(float), mids[rr.index].to_numpy(), "right") - 1, 0,
                    len(gg) - 1)
        rocks = gg["rock"].to_numpy()[k]
        forms = gg["formation"].to_numpy()[k]
        cls[rr.index] = [codes.rock_class_of(r, (cards.get(f or "") or {}).get("main_rock")) for r, f in
                         zip(rocks, forms)]
    df = pd.DataFrame({"cls": cls, "rop": runs["rop"]}).dropna()
    for c, s in df.groupby("cls")["rop"]:
        out[c] = float(s.median())
    return out
