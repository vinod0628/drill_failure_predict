"""Models H, A and B (design spec 5.1 - 5.5, 5.8 - 5.10): training, validation, saving, scoring.

* **Model H** - where the hole meets a hazard: P(cavity), P(fault zone) per metre (gradient-boosted
  trees, one output per occurrence type, trained on all holes, failed or not).
* **Model A** - ahead of the bit: the hazard h per metre of each failure family (discrete-time
  hazard: label 1 in the metre where the hole failed, 0 above; rows below a failure are not used).
  Model H's (out-of-area) probabilities are features.
* **Model B** - while drilling: the same with the drilling-response features, label 1 in the last L m
  before a failure (the precursor window).

Training rows: one per metre of every finished hole, from its logs, plus a noisy copy (spec 5.6.1).
Validation: leave one area out (spec 5.9). Thresholds are tuned per family on the held-out areas so
that 80 % of failures are warned (Orange) at least 3 m ahead. The models are saved as JSON (LightGBM
model text) with a version, in ``<workspace>/drillrisk/models``. No Qt here.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import codes
from .features import (BASE, MODEL_FEATURES, HoleContext, build_features, intervals_from_log, rop_norms_from_runs)
from .store import RiskData
from .trajectory import drilled_trace

log = logging.getLogger(__name__)

MIN_FAILURES = 12                 # a family needs this many failures for a model of its own
PRECURSOR_L = 6.0                 # m (spec 5.1: L = 3 - 10 m)
WINDOW = 10                       # m: risk score = hazard over the next 10 m (spec 5.8)
LEAD_MIN = 3.0                    # m: a warning must come at least this far ahead (spec 5.8)
TARGET_RECALL = 0.80
MAX_YELLOW, MAX_ORANGE, MAX_RED = 0.25, 0.10, 0.02     # most metres of a family at or above each level
WEAK_AUC = 0.65                                        # below: the family's model warns only on its top 2 %
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.06, "num_leaves": 15, "min_data_in_leaf": 80,
              "feature_fraction": 0.85, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 2.0,
              "verbose": -1, "num_threads": 0, "seed": 1}
ROUNDS = {"H": 220, "A": 160, "B": 140}
NEG_KEEP = {"H": 0.35, "A": 0.25, "B": 0.25}      # share of label-0 rows kept (weighted back up)
TRANSFER_CHECK = ["thick", "carb", "hard", "abr", "depth", "incidence", "above_wt", "s_dip", "s_fpkm", "s_valley"]


def lgb():
    import lightgbm

    return lightgbm


# --- the training table -------------------------------------------------------------------------
@dataclass
class TrainingTable:
    X: pd.DataFrame                    # BASE features
    hole: np.ndarray                   # hole id per row
    area: np.ndarray
    md: np.ndarray
    replicate: np.ndarray              # 0 = logs as recorded, 1.. = noisy copies
    y_haz: dict                        # "cavity" / "fault" -> (n,) 0/1
    y_fail: dict                       # family -> (n,) 0/1 (failure metre)
    y_pre: dict                        # family -> (n,) 0/1 (precursor window)
    has_runs: np.ndarray               # rows with run parameters (for Model B)
    rop_norms: dict
    severity: dict                     # family -> mean severity / 5


def hole_context(hid: str, data: RiskData, cards: dict, areas: dict, rigs: dict, end: float | None = None,
                 rng=None, rop_norms=None) -> HoleContext | None:
    """The context of a recorded hole from its logs (to its final depth, or ``end``)."""
    col = data.hole("collar", hid)
    if not len(col):
        return None
    c = col.iloc[0]
    depth = float(end if end is not None else (c["final_depth"] if pd.notna(c["final_depth"]) else 0.0))
    if depth < 1:
        return None
    sv = data.hole("survey", hid)
    trace = drilled_trace(hid, (c["x"], c["y"], c["z"]), sv, _num(c["plan_azimuth"], 0.0),
                          _num(c["plan_dip"], -90.0), depth)
    md = np.arange(0.5, depth, 1.0)
    if not len(md):
        return None
    xyz = trace.positions_at(md)
    dirs = trace.direction_at(md)
    geo = data.hole("geology", hid)
    f, t, props, bed = intervals_from_log(geo, depth, cards, rng=rng)
    k = np.clip(np.searchsorted(f, md, side="right") - 1, 0, len(f) - 1)
    inc = bed[k].astype(float)
    incl = np.degrees(np.arcsin(np.clip(-dirs[:, 2], -1, 1)))
    inc = np.where(np.isnan(inc), incl, inc)                    # flat bedding when no angle was logged
    if rng is not None:
        inc = np.clip(inc + rng.normal(0, 5, len(inc)), 0, 90)
    occ = data.hole("occurrence", hid)
    faults = [(float(a), float(b)) for a, b, ty in zip(occ["depth_from"], occ["depth_to"], occ["type"]) if ty == "fault"]
    sizes = [(float(a), float(b), s) for a, b, s in zip(*[data.hole("hole_size", hid)[x] for x in
                                                         ("depth_from", "depth_to", "size")])]
    bits = [(float(a), float(b), s) for a, b, s in zip(*[data.hole("bit", hid)[x] for x in
                                                        ("depth_from", "depth_to", "bit_type")])]
    runs = data.hole("run", hid)
    return HoleContext(md=md, xyz=xyz, dirs=dirs, iv_from=f, iv_to=t, iv_props=props, incidence=inc,
                       fault_zones=faults, sizes=sizes, bits=bits, runs=runs, area=areas.get(c["area_id"], {}),
                       rig=rigs.get(c["rig_model"], {}), open_end=True)


def _num(v, default=np.nan):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def reference_dicts(data: RiskData) -> tuple[dict, dict, dict]:
    cards = {r["formation"]: r for r in data["formation"].to_dict("records")}
    areas = {r["area_id"]: r for r in data["area"].to_dict("records")}
    rigs = {r["rig_model"]: r for r in data["rig"].to_dict("records")}
    return cards, areas, rigs


def training_holes(data: RiskData, exclude_area: str | None = None) -> list[str]:
    c = data.collar
    m = c["result"].isin(["success", "failed"]) & (c["final_depth"].fillna(0) > 1)
    if exclude_area:
        m &= c["area_id"] != exclude_area
    return c.loc[m, "hole_id"].tolist()


def build_training_table(data: RiskData, noise_copies: int = 1, seed: int = 3, progress=None,
                         holes: list[str] | None = None) -> TrainingTable:
    rng = np.random.default_rng(seed)
    cards, areas, rigs = reference_dicts(data)
    rop_norms = rop_norms_from_runs(data["run"], data["geology"], cards)
    holes = holes if holes is not None else training_holes(data)
    fails = data["failure"].set_index("hole_id") if len(data["failure"]) else pd.DataFrame()
    area_of = dict(zip(data.collar["hole_id"], data.collar["area_id"]))
    parts, meta = [], []
    for i, hid in enumerate(holes):
        for rep in range(1 + noise_copies):
            ctx = hole_context(hid, data, cards, areas, rigs, rng=rng if rep else None, rop_norms=rop_norms)
            if ctx is None:
                continue
            X = build_features(ctx, rop_norms)
            n = len(X)
            occ = data.hole("occurrence", hid)
            y = {"cavity": np.zeros(n, np.int8), "fault": np.zeros(n, np.int8)}
            for a, b, ty in zip(occ["depth_from"], occ["depth_to"], occ["type"]):
                if ty in y:
                    y[ty][(ctx.md + 0.5 > a) & (ctx.md - 0.5 < b)] = 1
            yf = {f: np.zeros(n, np.int8) for f in codes.FAMILIES}
            yp = {f: np.zeros(n, np.int8) for f in codes.FAMILIES}
            if hid in fails.index:
                fr = fails.loc[hid]
                fr = fr.iloc[0] if isinstance(fr, pd.DataFrame) else fr
                fam = codes.family_of(fr["fail_code"])
                fd = float(fr["fail_depth"])
                k = int(min(n - 1, max(0, math.floor(fd))))
                yf[fam][k] = 1
                yp[fam][(ctx.md >= fd - PRECURSOR_L) & (ctx.md <= fd + 0.5)] = 1
                keep = np.arange(n) <= k
                X = X[keep]
                y = {a: b[keep] for a, b in y.items()}
                yf = {a: b[keep] for a, b in yf.items()}
                yp = {a: b[keep] for a, b in yp.items()}
                n = int(keep.sum())
            parts.append((X, y, yf, yp))
            meta.append((hid, area_of.get(hid, ""), ctx.md[:n], rep))
        if progress is not None and i % 20 == 0:
            progress(i, len(holes))
    if not parts:
        raise ValueError("No finished holes with logs to learn from")
    X = pd.concat([p[0] for p in parts], ignore_index=True)
    cat = lambda key, sub: np.concatenate([p[key][sub] for p in parts])  # noqa: E731
    sev = {}
    if len(data["failure"]):
        fdf = data["failure"].copy()
        fdf["fam"] = fdf["fail_code"].map(codes.family_of)
        for fam, g in fdf.groupby("fam"):
            sev[fam] = float(pd.to_numeric(g["severity"], errors="coerce").mean() / 5.0)
    return TrainingTable(
        X=X, hole=np.concatenate([np.full(len(m[2]), m[0], dtype=object) for m in meta]),
        area=np.concatenate([np.full(len(m[2]), m[1], dtype=object) for m in meta]),
        md=np.concatenate([m[2] for m in meta]), replicate=np.concatenate([np.full(len(m[2]), m[3]) for m in meta]),
        y_haz={k: cat(1, k) for k in ("cavity", "fault")}, y_fail={f: cat(2, f) for f in codes.FAMILIES},
        y_pre={f: cat(3, f) for f in codes.FAMILIES}, has_runs=X["pen_rev"].notna().to_numpy(),
        rop_norms=rop_norms, severity={f: sev.get(f, 0.6) for f in codes.FAMILIES})


# --- fitting ------------------------------------------------------------------------------------
def _fit(X: pd.DataFrame, y: np.ndarray, kind: str, rng, rounds: int | None = None):
    """A LightGBM booster on a subsample of the label-0 rows (weighted back), or None (no positives)."""
    y = np.asarray(y)
    if y.sum() < 3:
        return None
    keep = (y == 1) | (rng.random(len(y)) < NEG_KEEP[kind])
    w = np.where(y[keep] == 1, 1.0, 1.0 / NEG_KEEP[kind])
    ds = lgb().Dataset(X[keep], label=y[keep], weight=w, free_raw_data=True)
    params = dict(LGB_PARAMS)
    if kind != "H":
        params["min_data_in_leaf"] = 40
    return lgb().train(params, ds, num_boost_round=rounds or ROUNDS[kind])


def _with_hazard(X: pd.DataFrame, h_cav: np.ndarray, h_flt: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Add Model H's probabilities: this metre and the maximum over the next 10 m of the same hole."""
    X = X.copy()
    X["h_cavity"], X["h_fault"] = h_cav, h_flt
    X["h_cavity10"] = _forward_max(h_cav, groups)
    X["h_fault10"] = _forward_max(h_flt, groups)
    return X


def _forward_max(v: np.ndarray, groups: np.ndarray, w: int = WINDOW) -> np.ndarray:
    s = pd.Series(v[::-1])
    g = pd.Series(groups[::-1])
    out = s.groupby(g.values).transform(lambda x: x.rolling(w, min_periods=1).max())
    return out.to_numpy()[::-1]


def window_score(h: np.ndarray, groups: np.ndarray | None = None, w: int = WINDOW, centred: bool = False) -> np.ndarray:
    """1 - prod(1 - h) over the next ``w`` metres (centred: over i - w/2 .. i + w/2) of the same hole."""
    lh = np.log1p(-np.clip(h, 0, 0.999999))
    s = pd.Series(lh[::-1] if not centred else lh)
    if groups is None:
        groups = np.zeros(len(h))
    g = pd.Series((groups[::-1] if not centred else groups))
    if centred:
        r = s.groupby(g.values).transform(lambda x: x.rolling(w, min_periods=1, center=True).sum()).to_numpy()
    else:
        r = s.groupby(g.values).transform(lambda x: x.rolling(w, min_periods=1).sum()).to_numpy()[::-1]
    return 1.0 - np.exp(r)


@dataclass
class ModelSet:
    """The trained models and everything needed to use them."""

    version: str
    created: str
    H: dict = field(default_factory=dict)            # "cavity"/"fault" -> booster
    A: dict = field(default_factory=dict)            # family -> booster (None: base rate)
    B: dict = field(default_factory=dict)
    base_rate: dict = field(default_factory=dict)    # family -> failures per metre (fallback)
    severity: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)   # family -> (yellow, orange, red) on the window score
    rop_norms: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)       # feature -> (min, max) of the training data
    cavity_stats: list = field(default_factory=list) # rows for R-K02
    validation: dict = field(default_factory=dict)
    trained_on: dict = field(default_factory=dict)

    # --- scoring ---
    def hazard_probs(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cols = MODEL_FEATURES["H"]
        out = []
        for k in ("cavity", "fault"):
            b = self.H.get(k)
            out.append(b.predict(X[cols]) if b is not None else np.zeros(len(X)))
        return out[0], out[1]

    def with_hazard(self, X: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
        hc, hf = self.hazard_probs(X)
        return _with_hazard(X, hc, hf, groups)

    def family_hazards(self, XH: pd.DataFrame, live: bool = False) -> dict[str, np.ndarray]:
        """Per-metre hazard of each family (Model A; Model B where ``live`` and response features exist)."""
        out = {}
        for fam in codes.FAMILIES:
            b = self.A.get(fam)
            h = b.predict(XH[MODEL_FEATURES["A"]]) if b is not None else np.full(len(XH), self.base_rate.get(fam, 1e-4))
            out[fam] = np.clip(h, 0, 0.5)
        return out

    def precursor(self, XH: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for fam in codes.FAMILIES:
            b = self.B.get(fam)
            out[fam] = b.predict(XH[MODEL_FEATURES["B"]]) if b is not None else np.zeros(len(XH))
        return out

    def contributions(self, XH: pd.DataFrame, fam: str, model: str = "A") -> pd.DataFrame | None:
        """SHAP values (LightGBM's tree contributions) of a family model, per feature."""
        b = (self.A if model == "A" else self.B).get(fam)
        if b is None or not len(XH):
            return None
        cols = MODEL_FEATURES[model]
        c = b.predict(XH[cols], pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=cols)

    def hazard_contributions(self, X: pd.DataFrame, kind: str) -> pd.DataFrame | None:
        b = self.H.get(kind)
        if b is None or not len(X):
            return None
        cols = MODEL_FEATURES["H"]
        return pd.DataFrame(b.predict(X[cols], pred_contrib=True)[:, :-1], columns=cols)

    def level_index(self, score: np.ndarray, fam: str) -> np.ndarray:
        """The window score mapped to the 0-1 risk index whose level boundaries are the spec's starting
        levels (0.15 / 0.35 / 0.60), through this family's tuned thresholds."""
        y, o, r = self.thresholds.get(fam, codes.DEFAULT_THRESHOLDS)
        xs = [0.0, y, o, r, max(r * 2.0, r + 0.2), 1.0]
        ys = [0.0, 0.15, 0.35, 0.60, 0.85, 1.0]
        return np.interp(np.asarray(score, float), xs, ys)

    # --- files ---
    def to_json(self) -> dict:
        boost = lambda d: {k: (v.model_to_string() if v is not None else None) for k, v in d.items()}  # noqa: E731
        return {"format": "drill-risk-models", "version": self.version, "created": self.created,
                "H": boost(self.H), "A": boost(self.A), "B": boost(self.B), "base_rate": self.base_rate,
                "severity": self.severity, "thresholds": {k: list(v) for k, v in self.thresholds.items()},
                "rop_norms": self.rop_norms, "ranges": {k: list(v) for k, v in self.ranges.items()},
                "cavity_stats": self.cavity_stats, "validation": self.validation, "trained_on": self.trained_on}

    @classmethod
    def from_json(cls, d: dict) -> "ModelSet":
        load = lambda m: {k: (lgb().Booster(model_str=v) if v else None) for k, v in (m or {}).items()}  # noqa: E731
        return cls(version=d["version"], created=d.get("created", ""), H=load(d.get("H")), A=load(d.get("A")),
                   B=load(d.get("B")), base_rate=d.get("base_rate", {}), severity=d.get("severity", {}),
                   thresholds={k: tuple(v) for k, v in (d.get("thresholds") or {}).items()},
                   rop_norms=d.get("rop_norms", {}), ranges={k: tuple(v) for k, v in (d.get("ranges") or {}).items()},
                   cavity_stats=d.get("cavity_stats", []), validation=d.get("validation", {}),
                   trained_on=d.get("trained_on", {}))


# --- training -----------------------------------------------------------------------------------
def train(data: RiskData, validate: bool = True, noise_copies: int = 1, seed: int = 5, progress=None,
          version: str | None = None) -> ModelSet:
    """Train Models H, A and B on every finished hole; with ``validate``, leave-one-area-out scores and
    threshold tuning (else the starting thresholds)."""
    rng = np.random.default_rng(seed)
    say = progress or (lambda pct, text: None)
    say(2, "Building the training table (one row per metre) ...")
    tt = build_training_table(data, noise_copies=noise_copies, seed=seed,
                              progress=lambda i, n: say(2 + int(28 * i / max(n, 1)), f"Features: hole {i} of {n}"))
    X, groups, areas = tt.X, tt.hole, tt.area
    gkey = np.array([f"{h}#{r}" for h, r in zip(groups, tt.replicate)], dtype=object)
    area_list = sorted(set(areas))
    # Model H, out of area (leave one area out) for the stacked features of A / B
    say(32, "Model H: cavities and fault zones (out-of-area) ...")
    h_oof = {k: np.zeros(len(X)) for k in ("cavity", "fault")}
    folds = area_list if len(area_list) >= 3 else None
    for k in ("cavity", "fault"):
        if folds:
            for a in folds:
                tr, te = areas != a, areas == a
                b = _fit(X.loc[tr, MODEL_FEATURES["H"]], tt.y_haz[k][tr], "H", rng, rounds=150)
                if b is not None:
                    h_oof[k][te] = b.predict(X.loc[te, MODEL_FEATURES["H"]])
        else:
            b = _fit(X[MODEL_FEATURES["H"]], tt.y_haz[k], "H", rng)
            h_oof[k] = b.predict(X[MODEL_FEATURES["H"]]) if b is not None else 0.0
    XH = _with_hazard(X, h_oof["cavity"], h_oof["fault"], gkey)
    ms = ModelSet(version=version or _dt.datetime.now().strftime("v%Y%m%d-%H%M"),
                  created=_dt.datetime.now().isoformat(timespec="seconds"))
    say(45, "Model H: final fit ...")
    for k in ("cavity", "fault"):
        ms.H[k] = _fit(X[MODEL_FEATURES["H"]], tt.y_haz[k], "H", rng)
    clean = tt.replicate == 0
    metres = float(clean.sum())
    for fam in codes.FAMILIES:
        ms.base_rate[fam] = float(tt.y_fail[fam][clean].sum() / max(metres, 1.0))
    ms.severity = tt.severity
    ms.rop_norms = tt.rop_norms
    n_fail = {fam: int(tt.y_fail[fam][clean].sum()) for fam in codes.FAMILIES}
    say(50, "Model A: failure hazard per family ...")
    for fam in codes.FAMILIES:
        if n_fail[fam] >= MIN_FAILURES:
            ms.A[fam] = _fit(XH[MODEL_FEATURES["A"]], tt.y_fail[fam], "A", rng)
        else:
            ms.A[fam] = None
    say(58, "Model B: precursors while drilling ...")
    live = tt.has_runs
    for fam in codes.FAMILIES:
        if n_fail[fam] >= MIN_FAILURES and tt.y_pre[fam][live].sum() >= 10:
            ms.B[fam] = _fit(XH.loc[live, MODEL_FEATURES["B"]], tt.y_pre[fam][live], "B", rng)
        else:
            ms.B[fam] = None
    ms.ranges = {c: (float(np.nanmin(X[c])), float(np.nanmax(X[c]))) for c in TRANSFER_CHECK
                 if c in X and np.isfinite(X[c]).any()}
    ms.cavity_stats = cavity_statistics(X[clean], tt.y_haz["cavity"][clean], groups[clean], data)
    ms.trained_on = {"holes": int(len(set(groups))), "metres": int(metres), "areas": area_list,
                     "failures": n_fail, "cavity_metres": int(tt.y_haz["cavity"][clean].sum()),
                     "fault_metres": int(tt.y_haz["fault"][clean].sum())}
    ms.thresholds = {fam: codes.DEFAULT_THRESHOLDS for fam in codes.FAMILIES}
    if validate and folds:
        say(62, "Validation: leave one area out ...")
        ms.validation = loao_validate(tt, XH, h_oof, ms, rng, lambda p, t: say(62 + int(36 * p), t))
        ms.thresholds = ms.validation.pop("_thresholds", ms.thresholds)
    say(100, "Done")
    return ms


def loao_validate(tt: TrainingTable, XH: pd.DataFrame, h_oof: dict, ms: ModelSet, rng, say) -> dict:
    """Leave-one-area-out scores (spec 5.9) and the thresholds tuned on the held-out areas (spec 5.8)."""
    from sklearn.metrics import roc_auc_score

    clean = tt.replicate == 0
    areas = tt.area
    area_list = sorted(set(areas))
    hz = {fam: np.zeros(len(XH)) for fam in codes.FAMILIES}
    for i, a in enumerate(area_list):
        say(i / max(len(area_list), 1), f"Validation: area {a} held out ({i + 1} of {len(area_list)})")
        tr, te = (areas != a), (areas == a) & clean
        for fam in codes.FAMILIES:
            y = tt.y_fail[fam]
            if ms.A.get(fam) is not None and y[tr].sum() >= 5:
                b = _fit(XH.loc[tr, MODEL_FEATURES["A"]], y[tr], "A", rng, rounds=120)
                hz[fam][te] = b.predict(XH.loc[te, MODEL_FEATURES["A"]]) if b is not None else ms.base_rate[fam]
            else:
                hz[fam][te] = ms.base_rate.get(fam, 1e-4)
    out: dict = {"areas": area_list}
    idx = np.nonzero(clean)[0]
    g = tt.hole[idx]
    md = tt.md[idx]
    # Model H metre-level AUC (out of area)
    for k in ("cavity", "fault"):
        y = tt.y_haz[k][idx]
        out[f"auc_H_{k}"] = float(roc_auc_score(y, h_oof[k][idx])) if 0 < y.sum() < len(y) else None
    thresholds, recall, lead, fam_auc = {}, {}, {}, {}
    total_h = np.zeros(len(idx))
    for fam in codes.FAMILIES:
        h = hz[fam][idx]
        total_h += h
        y = tt.y_fail[fam][idx]
        sc = window_score(h, g) * ms.severity.get(fam, 0.6)
        fam_auc[fam] = float(roc_auc_score(y, h)) if 0 < y.sum() < len(y) and np.ptp(h) > 0 else None
        # best score 3 - 20 m ahead of each failure of this family
        best = []
        for k in np.nonzero(y)[0]:
            hk = g[k]
            m = (g == hk) & (md <= md[k] - LEAD_MIN) & (md >= md[k] - 20)
            best.append(float(sc[m].max()) if m.any() else 0.0)
        best = np.array(best)
        # Orange: the score that warns TARGET_RECALL of the failures >= 3 m ahead - but never on more than
        # MAX_ORANGE of all metres (a warning everywhere is no warning); Red / Yellow likewise
        q = lambda p: float(np.quantile(sc, p))  # noqa: E731
        if ms.A.get(fam) is None or float(np.ptp(sc)) < 1e-12:
            # no model of its own (too few failures): the base rate never raises an alarm - the rules do
            c = float(np.max(sc)) if len(sc) else 1e-4
            thresholds[fam] = (round(c * 3, 7), round(c * 6, 7), round(c * 12, 7))
            recall[fam], lead[fam] = None, None
            continue
        # a weak model (held-out AUC < 0.65) warns only on its very highest metres: the rules carry that family
        weak = fam_auc[fam] is None or fam_auc[fam] < WEAK_AUC
        max_orange, max_red = (0.02, 0.005) if weak else (MAX_ORANGE, MAX_RED)
        if len(best) >= 3 and not weak:
            orange = max(float(np.quantile(best, 1 - TARGET_RECALL)), q(1 - max_orange))
        else:
            orange = q(1 - max_orange)
        orange = float(np.clip(orange, 1e-5, 0.35))
        red = max(orange * 1.7, q(1 - max_red))
        yellow = min(orange * 0.6, max(orange * 0.43, q(1 - MAX_YELLOW)))
        thresholds[fam] = (round(yellow, 6), round(orange, 6), round(red, 6))
        recall[fam] = float((best >= orange).mean()) if len(best) else None
        # lead distance: from the first Orange metre (within 50 m above) to the failure
        leads = []
        for k in np.nonzero(y)[0]:
            m = (g == g[k]) & (md <= md[k]) & (md >= md[k] - 50) & (sc >= orange)
            if m.any():
                leads.append(float(md[k] - md[m].min()))
        lead[fam] = float(np.median(leads)) if leads else None
    # hole level: "fails before target depth" - the concordance of the cumulative hazard (a failed hole
    # against every hole still drilling at its failure depth: no bias from failed holes being shorter)
    fail_any0 = np.any([tt.y_fail[f][idx] == 1 for f in codes.FAMILIES], axis=0)
    out["auc_hole"] = concordance(g, md, total_h, fail_any0)
    per_area = {}
    ar = areas[idx]
    for a in area_list:
        m = ar == a
        per_area[a] = concordance(g[m], md[m], total_h[m], fail_any0[m])
    out["auc_hole_by_area"] = per_area
    vals = [v for v in per_area.values() if v is not None]
    out["auc_hole_spread"] = [float(min(vals)), float(max(vals))] if vals else None
    # false alarms per 100 m: Orange metres of any family with no failure within the next 20 m
    orange_any = np.zeros(len(idx), bool)
    for fam in codes.FAMILIES:
        orange_any |= window_score(hz[fam][idx], g) * ms.severity.get(fam, 0.6) >= thresholds[fam][1]
    fail_any = np.any([tt.y_fail[f][idx] == 1 for f in codes.FAMILIES], axis=0)
    near_fail = pd.Series(fail_any[::-1].astype(float)).groupby(g[::-1]).transform(
        lambda x: x.rolling(21, min_periods=1).max()).to_numpy()[::-1] > 0
    starts = orange_any & ~np.r_[False, orange_any[:-1]] & ~near_fail
    out["false_alarms_per_100m"] = float(starts.sum() / max(len(idx), 1) * 100.0)
    out["auc_family"] = fam_auc
    out["recall_orange"] = recall
    out["lead_m"] = lead
    out["_thresholds"] = thresholds
    return out


def concordance(holes: np.ndarray, md: np.ndarray, h: np.ndarray, failed_metre: np.ndarray) -> float | None:
    """Harrell's C for the discrete-time hazard model: for each hole that failed at depth D, the share of
    holes still drilling below D whose cumulative hazard down to D is lower (ties count half)."""
    df = pd.DataFrame({"hole": holes, "md": md, "h": h, "f": failed_metre})
    df["H"] = df.groupby("hole")["h"].cumsum()
    ends = df.groupby("hole")["md"].max()
    fails = df[df["f"]]
    if not len(fails) or len(ends) < 3:
        return None
    by_hole = {k: (g["md"].to_numpy(), g["H"].to_numpy()) for k, g in df.groupby("hole")}
    conc = total = 0.0
    for hid, D, Hi in zip(fails["hole"], fails["md"], fails["H"]):
        for hj, end in ends.items():
            if hj == hid or end <= D:
                continue
            m, Hc = by_hole[hj]
            Hj = Hc[min(len(m) - 1, int(np.searchsorted(m, D, side="right")) - 1)] if len(m) else 0.0
            total += 1
            conc += 1.0 if Hi > Hj else (0.5 if Hi == Hj else 0.0)
    return float(conc / total) if total else None


def cavity_statistics(X: pd.DataFrame, y_cav: np.ndarray, holes: np.ndarray, data: RiskData) -> list[dict]:
    """For R-K02: share of hole-unit intersections that met a cavity within d m below the top of the
    unit, by rock class (dolomite / limestone / other carbonate) and contact type."""
    df = pd.DataFrame({"hole": holes, "d_top": X["d_top"].to_numpy(), "carb": X["carb"].to_numpy(),
                       "dolo": X["dolo"].to_numpy(), "cc": X["ct_clastic_carb"].to_numpy(),
                       "thick": X["thick"].to_numpy(), "cav": y_cav})
    df = df[df["carb"] >= 50]
    if not len(df):
        return []
    df["unit"] = (df["d_top"] < 0.6).astype(int).groupby(df["hole"]).cumsum()
    rows = []
    for (cls, cc), g in df.assign(cls=np.where(df["dolo"] > 0.5, "dolomite", "limestone")).groupby(["cls", "cc"]):
        thick = g.groupby(["hole", "unit"])["thick"].first()
        for d in (10, 20, 30, 50):
            units = thick[thick >= d]
            hits = g[g["d_top"] <= d].groupby(["hole", "unit"])["cav"].max().reindex(units.index).fillna(0)
            if len(units) >= 5:
                rows.append({"rock_class": cls, "contact": "clastic-over-carbonate" if cc > 0.5 else "carbonate",
                             "d": d, "pct": round(100.0 * float(hits.mean()), 1), "n": int(len(units))})
    return rows


# --- files --------------------------------------------------------------------------------------
def models_folder(workspace: str | Path) -> Path:
    return Path(workspace) / "drillrisk" / "models"


def save(ms: ModelSet, workspace: str | Path) -> Path:
    folder = models_folder(workspace)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"models_{ms.version}.json"
    p.write_text(json.dumps(ms.to_json()), encoding="utf-8")
    (folder / "current.txt").write_text(p.name, encoding="utf-8")
    return p


def load_current(workspace: str | Path | None) -> ModelSet | None:
    if workspace is None:
        return None
    folder = models_folder(workspace)
    cur = folder / "current.txt"
    try:
        name = cur.read_text(encoding="utf-8").strip() if cur.exists() else None
        if name is None:
            files = sorted(folder.glob("models_*.json"))
            if not files:
                return None
            name = files[-1].name
        return ModelSet.from_json(json.loads((folder / name).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError) as e:
        log.warning("cannot load the risk models: %s", e)
        return None


def list_versions(workspace: str | Path) -> list[tuple[str, str]]:
    """(version, file name) of the saved models, newest first."""
    out = []
    for p in sorted(models_folder(workspace).glob("models_*.json"), reverse=True):
        out.append((p.stem.replace("models_", ""), p.name))
    return out
