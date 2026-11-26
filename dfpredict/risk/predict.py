"""Predicting the risk ahead of the bit for one hole (design spec 4, 5, 6, 7.1, 8).

``Predictor.predict`` runs the whole chain for an ongoing hole (or a planned one, or a design that is
not saved - the plan check):

1. the path ahead from the design and the surveys so far, as an ensemble of trial paths (trajectory.py);
2. the geological model along every path: units, contacts, faults, bedding - with the model error band
   ±u sampled per path, tied to the contacts logged so far (spec 5.6.4); logged geology above the bit;
3. per-metre features -> Model H (cavity / fault-zone probability) -> Model A (failure hazard per
   family), and Model B for the next 10 m when runs are entered; the area correction (spec 5.10.4);
4. the hazards averaged over the paths, the risk score (hazard over 10 m x severity weight), the alarm
   level per metre, rules R1 - R8 overriding the level at the bit;
5. warning zones (from - u to + u), reason codes from the models' SHAP values, measures ranked by evidence,
   the chance of reaching the target depth and the danger probabilities over any depth range.

``plan_options`` reruns the chain for other azimuths / dips (spec 5.5 planning what-if, 9). No Qt here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..i18n import tr
from . import codes
from .features import MODEL_FEATURES, HoleContext, Props, build_features, intervals_from_log, props_for
from .geomodel import GeoModel
from .measures import MeasureAdvisor, Suggestion, fill
from .models import ModelSet, hole_context, reference_dicts, window_score
from .rules import RuleHit, evaluate
from .store import RiskData
from .trajectory import PathEnsemble, predict_paths, survey_spread

log = logging.getLogger(__name__)
EXTEND = 60.0           # m beyond the target depth (the base of the last unit is known there)
# Model H gives the chance that a metre lies IN a cavity / fault zone; one occurrence spans several metres, so the
# chance of meeting one over a range uses the per-metre probability divided by the typical occurrence length.
CAVITY_LENGTH = 2.0     # m, typical cavity height along the hole
PQ_COLLAR = 24.0        # m, PQ collar casing assumed for a hole not started
FAULT_ZONE_LENGTH = 20.0   # m, typical fault damage zone crossed along the hole
FAMILY_CODES_SHOWN = {"karst": ["LC", "CV"], "instability": ["CL", "SP"], "trajectory": ["DV"], "bit": ["BG", "BW"]}
REASON_OF_FEATURE = {
    "carb": "R-K01", "dolo": "R-K01", "karst": "R-K01", "d_top": "R-K01", "d_first_carb": "R-K01",
    "carb_drilled": "R-K01", "d_unconf": "R-K01", "above_wt": "R-K01", "h_cavity": "R-K01", "h_cavity10": "R-K01",
    "s_cexp": "R-K02", "s_sink": "R-K02", "s_unconf": "R-K02", "s_valley": "R-K02", "ct_clastic_carb": "R-K02",
    "up_clastic": "R-K02", "up_carb": "R-K02",
    "in_fault": "R-F01", "d_fault": "R-F01", "h_fault": "R-F02", "h_fault10": "R-F02", "s_fpkm": "R-F02",
    "incidence": "R-B01", "bedding": "R-B01", "s_dip": "R-B01", "dogleg": "R-B01", "incl": "R-B01",
    "hard": "R-L01", "abr": "R-L01", "bit": "R-L01", "ct_hard_soft": "R-L01", "up_hard": "R-L01",
    "clay": "R-C01", "clastic": "R-C01",
    "depth": "R-H01", "depth_ratio": "R-H01", "n_red": "R-H01", "diam": "R-H01",
    "rpm": "R-L01", "flow": "R-K01",
    "pq": "R-D01", "d_pq": "R-D03", "pen_rev": "R-D02", "d_pen": "R-D02", "rop_rel": "R-D02", "ret": "R-D06",
    "rod_drop": "R-D07",
}


@dataclass
class Zone:
    from_: float
    to: float
    level: str
    p: float                          # risk index 0-1 (level boundaries 0.15 / 0.35 / 0.60)
    prob10: float                     # chance of failure within 10 m around the zone (raw)
    families: list
    codes: list
    u: float
    status: str                       # Drilled / At bit / Ahead
    reasons: list = field(default_factory=list)       # (code, text)
    prevent: list = field(default_factory=list)       # Suggestion
    respond: list = field(default_factory=list)
    rules: list = field(default_factory=list)         # RuleHit
    p_cavity: float = 0.0
    p_fault: float = 0.0
    formation: str = ""

    @property
    def band_from(self) -> float:
        return max(0.0, self.from_ - self.u)

    @property
    def band_to(self) -> float:
        return self.to + self.u

    @property
    def family_text(self) -> str:
        return " / ".join(tr(codes.FAMILIES[f]) for f in self.families)

    @property
    def title(self) -> str:
        names = ", ".join(tr(codes.FAIL_CODES[c][0]) for c in self.codes)
        return f"{tr(codes.LEVEL_NAMES[self.level]).upper()} - {names} ({', '.join(self.codes)})"


@dataclass
class Prediction:
    hole_id: str
    collar: dict
    ensemble: PathEnsemble
    md: np.ndarray                    # metre centres of the scored part (0 .. target + EXTEND)
    h: dict                           # family -> mean hazard per metre
    p_cavity: np.ndarray              # Model H, mean over the paths
    p_fault: np.ndarray
    index: dict                       # family -> risk index per metre
    level: np.ndarray                 # per metre (str)
    u: np.ndarray                     # ±u per metre
    unit: list                        # unit / formation on the predicted path, per metre
    unit_share: np.ndarray            # share of the paths in that unit
    zones: list
    rules: list
    messages: list                    # hole-level (code, text)
    p_reach: float
    planned_depth: float
    drilled_depth: float
    contacts: list                    # (depth, upper, lower, model depth band ±u) on the predicted path
    crossings: list                   # (depth, fault, share of paths)
    model_version: str
    geomodel: str
    area_factor: dict
    summary: str = ""
    outside: list = field(default_factory=list)

    @property
    def target_mask(self) -> np.ndarray:
        return self.md <= self.planned_depth

    def range_probabilities(self, a: float, b: float) -> dict:
        """Chances over the depth range a - b (m): failure (all and per family), cavity met (Model H),
        a model fault crossed (share of the paths), an unmapped fault zone met (Model H)."""
        m = (self.md >= a) & (self.md <= b)
        out = {"from": a, "to": b}
        if not m.any():
            return out
        tot = np.zeros(m.sum())
        for f, h in self.h.items():
            out[f] = float(1 - np.prod(1 - h[m]))
            tot += h[m]
        out["failure"] = float(1 - np.prod(1 - np.clip(tot, 0, 0.99)))
        out["cavity"] = float(1 - np.prod(1 - np.clip(self.p_cavity[m] / CAVITY_LENGTH, 0, 0.99)))
        out["fault_zone"] = float(1 - np.prod(1 - np.clip(self.p_fault[m] / FAULT_ZONE_LENGTH, 0, 0.99)))
        out["fault_crossed"] = float(max([s for d, _f, s in self.crossings if a <= d <= b] or [0.0]))
        return out


# ------------------------------------------------------------------------------------------------
class Predictor:
    """Everything needed to predict in one workspace: records, models, geological model."""

    def __init__(self, data: RiskData, models: ModelSet, geomodel: GeoModel, area_id: str | None = None):
        self.data = data
        self.models = models
        self.model = geomodel
        self.cards, self.areas, self.rigs = reference_dicts(data)
        self.model.cards = {**self.model.cards, **self.cards}
        past = data["survey"][data["survey"]["hole_id"].isin(
            data.collar.loc[data.collar["result"].isin(["success", "failed"]), "hole_id"])]
        self.spread = survey_spread(past)
        self.advisor = MeasureAdvisor(data, self.cards)
        self._area_factor: dict[str, dict] = {}
        self.area_id = area_id

    # --- one hole -------------------------------------------------------------------------------
    def predict(self, hole_id: str | None = None, design: dict | None = None, n_paths: int = 40, seed: int = 1,
                progress=None) -> Prediction:
        say = progress or (lambda pct, text: None)
        if design is not None:
            collar = dict(design)
            recs = {k: pd.DataFrame(columns=self.data[k].columns) for k in
                    ("survey", "geology", "run", "hole_size", "bit", "occurrence")}
        else:
            c = self.data.hole("collar", hole_id)
            if not len(c):
                raise KeyError(hole_id)
            collar = c.iloc[0].to_dict()
            recs = {k: self.data.hole(k, hole_id) for k in ("survey", "geology", "run", "hole_size", "bit", "occurrence")}
        hid = str(collar["hole_id"])
        planned = float(collar.get("planned_depth") or 0) or 300.0
        result = collar.get("result") or "planned"
        drilled = float(collar.get("final_depth") or 0) if result != "planned" else 0.0
        if not np.isfinite(drilled):
            drilled = 0.0
        area = self.areas.get(collar.get("area_id"), {})
        rig = self.rigs.get(collar.get("rig_model"), {})
        az0 = _num(collar.get("plan_azimuth"), 0.0)
        dip0 = _num(collar.get("plan_dip"), -90.0)
        say(5, tr("Predicting the path ahead of the bit ..."))
        ens = predict_paths(hid, (collar["x"], collar["y"], collar["z"]), az0, dip0, planned, recs["survey"],
                            drilled, model=self.model, n_paths=n_paths, spread=self.spread, extend=EXTEND, seed=seed)
        messages: list[tuple[str, str]] = []
        # --- ties to the logged contacts (spec 5.6.4) ---
        say(15, tr("Tying the model to the logged contacts ..."))
        dz_tie, tie_depth, u_factor = self._ties(ens, recs["geology"], drilled, messages)
        # --- the paths: mean path first, then the trial paths ---
        paths = np.concatenate([ens.mean_xyz[None], ens.samples])
        pdirs = np.concatenate([ens.mean_dirs[None], ens.sample_dirs])
        P = len(paths)
        md_full = ens.md
        end = float(md_full[-1])
        mds = np.arange(0.5, end, 1.0)
        rng = np.random.default_rng(seed + 17)
        e = np.r_[0.0, rng.normal(0, 1, P - 1)]                     # model error per path (in units of u)
        say(25, tr("Reading the geological model along {n} paths ...", n=P))
        xyz_all = np.stack([_interp_path(md_full, paths[j], mds) for j in range(P)])
        dir_all = np.stack([_interp_path(md_full, pdirs[j], mds) for j in range(P)])
        dir_all /= np.linalg.norm(dir_all, axis=2, keepdims=True)
        flat = xyz_all.reshape(-1, 3)
        idx0 = self.model.unit_index(flat).reshape(P, -1) if self.model.ok else np.zeros((P, len(mds)), int)
        u_all = self.model.uncertainty(flat, idx0.ravel()).reshape(P, -1) * u_factor
        if tie_depth is not None:                                    # surer near the tie
            near = np.clip(0.4 + (mds - tie_depth) / 150.0, 0.4, 1.0)
            u_all = u_all * np.where(mds >= tie_depth, near, 1.0)[None, :]
        dz = (e[:, None] * u_all).reshape(-1)
        if dz_tie is not None:
            dz_h = dz_tie[:, None] + dz[None, :]
        else:
            dz_h = dz
        idx = self.model.unit_index(flat, dz_h).reshape(P, -1) if self.model.ok else idx0
        nrm = self.model.bedding_normal(flat, idx.ravel()).reshape(P, -1, 3)
        inc = np.degrees(np.arcsin(np.clip(np.abs(np.einsum("pij,pij->pi", dir_all, nrm)), 0, 1)))
        hw = float(self.model.config.get("fault_zone", 12.0))
        crossings_all = self._crossings(xyz_all, mds)
        # --- features for every path ---
        say(40, tr("Building the features ..."))
        log_iv = intervals_from_log(recs["geology"], max(drilled, 1.0), self.cards) if drilled > 0 and len(
            recs["geology"]) else None
        occ = recs["occurrence"]
        drilled_faults = [(float(a), float(b)) for a, b, t in zip(occ["depth_from"], occ["depth_to"], occ["type"])
                          if t == "fault"]
        sizes = [(float(a), float(b), s) for a, b, s in zip(recs["hole_size"]["depth_from"],
                                                            recs["hole_size"]["depth_to"], recs["hole_size"]["size"])]
        bits = [(float(a), float(b), s) for a, b, s in zip(recs["bit"]["depth_from"], recs["bit"]["depth_to"],
                                                           recs["bit"]["bit_type"])]
        planned_setup = {"size": sizes[-1][2] if sizes else "HQ", "bit": bits[-1][2] if bits else "impregnated"}
        if not sizes:                         # not started: the usual programme - PQ collar casing, then HQ
            sizes = [(0.0, PQ_COLLAR, "PQ"), (PQ_COLLAR, end + 1, planned_setup["size"])]
        if sizes:
            sizes[-1] = (sizes[-1][0], end + 1, sizes[-1][2])
        if bits:
            bits[-1] = (bits[-1][0], end + 1, bits[-1][2])
        runs = recs["run"] if len(recs["run"]) else None
        frames = []
        units_p0 = None
        for j in range(P):
            f, t, props = self._intervals(idx[j], mds, log_iv, drilled)
            if j == 0:
                units_p0 = (f, t, props)
            fz = list(drilled_faults) + [(c - hw, c + hw) for c, _n in crossings_all[j] if c > drilled]
            ctx = HoleContext(md=mds, xyz=xyz_all[j], dirs=dir_all[j], iv_from=f, iv_to=t, iv_props=props,
                              incidence=inc[j], fault_zones=fz, sizes=sizes, bits=bits, runs=runs, area=area,
                              rig=rig, planned=planned_setup, live_from=drilled if runs is not None else None)
            frames.append(build_features(ctx, self.models.rop_norms))
        X = pd.concat(frames, ignore_index=True)
        groups = np.repeat(np.arange(P), len(mds))
        say(60, tr("Scoring: Models H, A and B ..."))
        XH = self.models.with_hazard(X, groups)
        hz = self.models.family_hazards(XH)
        corr = self.area_correction(collar.get("area_id"))
        for fam in hz:
            hz[fam] = np.clip(hz[fam] * corr.get(fam, 1.0), 0, 0.5)
        live = runs is not None and drilled > 0
        pre = self.models.precursor(XH) if live else None
        M = len(mds)
        h_mean = {f: v.reshape(P, M).mean(axis=0) for f, v in hz.items()}
        pc = XH["h_cavity"].to_numpy().reshape(P, M).mean(axis=0)
        pf = XH["h_fault"].to_numpy().reshape(P, M).mean(axis=0)
        u = u_all.mean(axis=0)
        # --- risk index per family and metre ---
        index = {}
        for fam, h in h_mean.items():
            sc = window_score(h, None, centred=True) * self.models.severity.get(fam, 0.6)
            index[fam] = self.models.level_index(sc, fam)
            if live and pre is not None:
                near = (mds > drilled) & (mds <= drilled + 10)
                pb = pre[fam].reshape(P, M).mean(axis=0) * self.models.severity.get(fam, 0.6)
                index[fam] = np.where(near, np.maximum(index[fam], self.models.level_index(pb / 3.0, fam)), index[fam])
            index[fam][mds < drilled - 0.5] *= 1.0                    # drilled part: shown as it was scored
        # --- rules (spec 5.7) ---
        say(70, tr("Checking the alarm rules ..."))
        rock_at = self._rock_class_at(recs["geology"])
        form_at = self._formation_at(recs["geology"])
        rule_hits = evaluate(recs["run"] if len(recs["run"]) else None, recs["survey"], collar, rock_at, form_at)
        for r in rule_hits:
            if r.current:
                m = (mds >= drilled - 3) & (mds <= drilled + 10)
                floor_ = {"yellow": 0.2, "orange": 0.45, "red": 0.75}[r.level]
                index[r.family] = np.where(m, np.maximum(index[r.family], floor_), index[r.family])
        lvl_idx = np.max(np.stack(list(index.values())), axis=0)
        level = np.array([codes.level_of(v) for v in lvl_idx], dtype=object)
        # --- unit on the predicted path ---
        names = [self.model.unit_name(int(i)) for i in idx[0]]
        if units_p0 is not None:
            k = np.clip(np.searchsorted(units_p0[0], mds, "right") - 1, 0, len(units_p0[0]) - 1)
            names = [units_p0[2][i].formation or units_p0[2][i].rock_class for i in k]
        share = (idx == idx[0][None, :]).mean(axis=0)
        # --- zones ---
        say(78, tr("Assembling the warning zones, reasons and measures ..."))
        zones = self._zones(mds, level, lvl_idx, index, h_mean, pc, pf, u, drilled, planned, X.iloc[:M], XH.iloc[:M],
                            names, crossings_all, rule_hits, collar, area, rig, sizes, planned_setup)
        # --- reaching the target ---
        tot = np.zeros((P, M))
        for f, v in hz.items():
            tot += v.reshape(P, M)
        ahead = (mds > drilled) & (mds <= planned)
        p_reach = float(np.mean(np.prod(1 - np.clip(tot[:, ahead], 0, 0.99), axis=1))) if ahead.any() else 1.0
        # --- contacts and faults on the predicted path ---
        contacts = []
        if self.model.ok:
            for c in self.model.contacts_along(xyz_all[0], mds, dz_tie[:, None] if dz_tie is not None else 0.0):
                k = min(M - 1, int(c.depth))
                contacts.append((c.depth, c.upper, c.lower, float(u[k])))
        cross = {}
        for j, cl in enumerate(crossings_all):
            for d, name in cl:
                cross.setdefault(name, []).append(d)
        crossings = sorted([(float(np.median(v)), name, len(v) / P) for name, v in cross.items()], key=lambda x: x[0])
        # --- hole-level messages ---
        self._hole_messages(messages, XH.iloc[:M], mds, drilled, u, corr, collar)
        outside = [m for m in messages if m[0] == "R-M04"]
        pred = Prediction(hid, collar, ens, mds, h_mean, pc, pf, index, level, u, names, share, zones, rule_hits,
                          messages, p_reach, planned, drilled, contacts, crossings, self.models.version,
                          self.model.describe(), corr, outside=outside)
        pred.summary = summary_text(pred)
        say(100, tr("Done"))
        return pred

    # --- pieces -----------------------------------------------------------------------------------
    def _ties(self, ens: PathEnsemble, geo: pd.DataFrame, drilled: float, messages: list):
        """Offsets between the contacts logged so far and the model's (spec 5.6.4). Returns the vertical
        shift per horizon (or None), the depth of the deepest tie and the factor widening ±u."""
        if not self.model.ok or not self.model.horizons or drilled <= 0 or not len(geo):
            return None, None, 1.0
        units = self.model.units
        g = geo.sort_values("depth_from")
        logged = []                                   # (depth, upper, lower) where the formation changes
        prev = None
        for r in g.itertuples(index=False):
            form = r.formation if isinstance(r.formation, str) and r.formation in units else None
            if form is None:
                continue
            if prev is not None and form != prev:
                logged.append((float(r.depth_from), prev, form))
            prev = form
        k_bit = max(2, int(np.searchsorted(ens.md, drilled, side="right")))
        mod = self.model.contacts_along(ens.mean_xyz[:k_bit + 30], ens.md[:k_bit + 30])
        dz = np.zeros(len(self.model.horizons))
        tie_depth, tied = None, False
        u_factor = 1.0
        for d_act, up, lo in logged:
            c = next((c for c in mod if c.upper == up and c.lower == lo), None)
            if c is None or up not in units or lo not in units:
                continue
            h = units.index(lo) - 1
            off = d_act - c.depth
            vz = abs(float(ens.mean_dirs[min(len(ens.md) - 1, int(d_act)), 2])) or 1.0
            shift = -off * vz
            dz[h] = shift
            for k in range(h + 1, len(dz)):           # fully for the next contact, less for deeper ones
                dz[k] = shift * (1.0 if k == h + 1 else 0.5)
            tie_depth, tied = d_act, True
            messages.append(("R-M02", fill(codes.REASONS["R-M02"], contact=f"{up}/{lo}", d_act=f"{d_act:.1f}",
                                           d_mod=f"{c.depth:.1f}", off=f"{off:+.1f}")))
        # mismatch at the bit
        last = g.iloc[-1]
        form = last["formation"] if isinstance(last["formation"], str) else None
        if form in units:
            p_bit = ens.mean_xyz[min(len(ens.md) - 1, k_bit - 1)][None]
            m_unit = self.model.unit_name(int(self.model.unit_index(p_bit, dz[:, None] if tied else 0.0)[0]))
            if m_unit != form:
                u_factor = 1.5
                u_here = float(self.model.uncertainty(p_bit)[0]) * u_factor
                messages.append(("R-M01", fill(codes.REASONS["R-M01"], lith=f"{last['rock']} ({form})",
                                               depth=f"{drilled:.0f}", formation=m_unit, u=f"{u_here:.0f}")))
        return (dz if tied else None), tie_depth, u_factor

    def _crossings(self, xyz_all: np.ndarray, mds: np.ndarray) -> list[list[tuple[float, str]]]:
        """Fault crossings per path (batched signed distances)."""
        P, M, _ = xyz_all.shape
        out = [[] for _ in range(P)]
        if not self.model.faults:
            return out
        flat = xyz_all.reshape(-1, 3)
        for f in self.model.faults:
            sd = f.signed_distance(flat).reshape(P, M)
            s = np.sign(sd)
            pj, kj = np.nonzero((s[:, :-1] * s[:, 1:] < 0) & (np.abs(sd[:, :-1]) < 3) & (np.abs(sd[:, 1:]) < 3))
            for j, k in zip(pj, kj):
                t = sd[j, k] / (sd[j, k] - sd[j, k + 1])
                out[j].append((float(mds[k] + t), f.name))
        return out

    def _intervals(self, idx: np.ndarray, mds: np.ndarray, log_iv, drilled: float):
        """Unit intervals along one path: logged above the bit, model below."""
        f, t, props = [], [], []
        start = 0
        n = len(idx)
        for k in range(1, n + 1):
            if k == n or idx[k] != idx[start]:
                a = float(mds[start] - 0.5)
                b = float(mds[k - 1] + 0.5)
                name = self.model.unit_name(int(idx[start]))
                f.append(a)
                t.append(b)
                props.append(self._unit_props(name))
                start = k
        f, t = np.array(f), np.array(t)
        if log_iv is None or drilled <= 0:
            return f, t, props
        lf, lt, lp, _bed = log_iv
        keep = t > drilled
        mf, mt = f[keep], t[keep]
        mp = [p for p, k in zip(props, keep) if k]
        if len(mf):
            mf[0] = drilled
        lt = lt.copy()
        lt[-1] = drilled
        if len(mp) and lp and mp[0].formation == lp[-1].formation:     # the unit at the bit goes on
            lt[-1] = mt[0]
            mf, mt, mp = mf[1:], mt[1:], mp[1:]
        return np.r_[lf, mf], np.r_[lt, mt], list(lp) + mp

    def _unit_props(self, name: str) -> Props:
        card = self.cards.get(name)
        if card is None:
            return props_for(name, None, {})
        return props_for(None, name, self.cards)

    def _rock_class_at(self, geo: pd.DataFrame):
        if not len(geo):
            return None
        g = geo.sort_values("depth_from")
        f = g["depth_from"].to_numpy(float)

        def at(d):
            r = g.iloc[int(np.clip(np.searchsorted(f, d, "right") - 1, 0, len(g) - 1))]
            return codes.rock_class_of(r["rock"], (self.cards.get(r["formation"] or "") or {}).get("main_rock"))
        return at

    def _formation_at(self, geo: pd.DataFrame):
        if not len(geo):
            return None
        g = geo.sort_values("depth_from")
        f = g["depth_from"].to_numpy(float)
        return lambda d: g.iloc[int(np.clip(np.searchsorted(f, d, "right") - 1, 0, len(g) - 1))]["formation"]

    def area_correction(self, area_id: str | None) -> dict:
        """Raise or lower each family's risk by the observed / expected failures of the area's finished
        holes, weighted n / (n + 5) after n holes (spec 5.10.4)."""
        if not area_id:
            return {}
        if area_id in self._area_factor:
            return self._area_factor[area_id]
        c = self.data.collar
        holes = c.loc[(c["area_id"] == area_id) & c["result"].isin(["success", "failed"]), "hole_id"].tolist()
        out = {}
        if holes:
            frames, groups = [], []
            for i, hid in enumerate(holes):
                ctx = hole_context(hid, self.data, self.cards, self.areas, self.rigs)
                if ctx is None:
                    continue
                frames.append(build_features(ctx, self.models.rop_norms))
                groups.append(np.full(len(ctx.md), i))
            if frames:
                X = pd.concat(frames, ignore_index=True)
                XH = self.models.with_hazard(X, np.concatenate(groups))
                hz = self.models.family_hazards(XH)
                fails = self.data["failure"][self.data["failure"]["hole_id"].isin(holes)]
                obs = fails["fail_code"].map(codes.family_of).value_counts().to_dict()
                n = len(holes)
                w = n / (n + 5.0)
                for fam, h in hz.items():
                    exp = float(h.sum())
                    ratio = (obs.get(fam, 0) + 0.5) / (exp + 0.5)
                    out[fam] = float(np.clip(ratio ** w, 1 / 3, 3.0))
                out["_holes"] = n
        self._area_factor[area_id] = out
        return out

    def _zones(self, mds, level, lvl_idx, index, h_mean, pc, pf, u, drilled, planned, X0, XH0, names,
               crossings_all, rule_hits, collar, area, rig, sizes, planned_setup) -> list[Zone]:
        zones = []
        on = np.array([codes.level_rank(v) >= 1 for v in level]) & (mds <= planned + 0.5)
        k = 0
        M = len(mds)
        while k < M:
            if not on[k]:
                k += 1
                continue
            j = k
            while j < M and on[j]:
                j += 1
            a, b = float(mds[k] - 0.5), float(mds[j - 1] + 0.5)
            sel = slice(k, j)
            fams = [f for f in codes.FAMILIES if index[f][sel].max() >= 0.15]
            if not fams:
                k = j
                continue
            peak = max(fams, key=lambda f: index[f][sel].max())
            fams.sort(key=lambda f: -index[f][sel].max())
            zl = max(level[sel], key=codes.level_rank)
            status = ("Drilled" if drilled > 0 and b <= drilled else
                      ("At bit" if drilled > 0 and a <= drilled + 10 else "Ahead"))
            zcodes = []
            for f in fams:
                zcodes += self._codes_for(f, X0.iloc[sel])
            p10 = float(1 - np.prod(1 - sum(h_mean[f][sel] for f in fams)))
            z = Zone(a, b, zl, float(lvl_idx[sel].max()), p10, fams, zcodes, float(np.mean(u[sel])), status,
                     p_cavity=float(1 - np.prod(1 - np.clip(pc[sel] / CAVITY_LENGTH, 0, 0.99))),
                     p_fault=float(pf[sel].max()), formation=max(set(names[k:j]), key=list(names[k:j]).count))
            z.names = list(names[k:j])                        # formation of each metre (for the reasons)
            z.rules = [r for r in rule_hits if r.depth_to >= a - 3 and r.depth_from <= b + 3]
            z.reasons = self._reasons(z, fams, sel, X0, XH0, crossings_all, collar, rig, sizes)
            ctx = self._measure_ctx(z, X0.iloc[sel], collar, rig, sizes, planned_setup, drilled)
            z.prevent, z.respond = self.advisor.suggest(z.codes, z.level, ctx)
            zones.append(z)
            k = j
        return zones

    @staticmethod
    def _codes_for(fam: str, X: pd.DataFrame) -> list[str]:
        if fam == "karst":
            out = ["LC", "CV"]
            if X["above_wt"].lt(0).any() and X["in_fault"].gt(0).any():
                out.append("WI")
            return out
        if fam == "instability":
            if X["bedding"].eq(3).any() or X["in_fault"].gt(0).any():
                return ["CL", "SP", "CB"]
            return ["CL", "SP"]
        if fam == "trajectory":
            return ["DV"]
        out = ["BG", "BW"]
        if X["dogleg"].gt(3).any():
            out.append("RF")
        return out

    def _reasons(self, z: Zone, fams: list, sel: slice, X0: pd.DataFrame, XH0: pd.DataFrame, crossings_all, collar,
                 rig, sizes) -> list[tuple[str, str]]:
        """The features that raised the risk most (SHAP) for each family of the zone, each mapped to its message
        (spec 5.8): up to 3 reasons, 4 when two families share the zone."""
        per_family = []
        for fam in fams:
            contrib = self.models.contributions(XH0.iloc[sel], fam, "A")
            order = []
            if contrib is not None:
                mean = contrib.mean(axis=0).sort_values(ascending=False)
                for feat, v in mean.items():
                    if v <= 0:
                        break
                    code = REASON_OF_FEATURE.get(feat)
                    if code and code not in order:
                        order.append(code)
            if fam == "karst" and float(X0.iloc[sel]["carb"].mean()) >= 50:
                order = ["R-K01"] + [c for c in order if c != "R-K01"]
            per_family.append(order)
        order = []
        for k in range(max([len(o) for o in per_family] or [0])):      # interleave: each family's best first
            for o in per_family:
                if k < len(o) and o[k] not in order:
                    order.append(o[k])
        fam = fams[0] if fams else "karst"
        x = X0.iloc[sel]
        if any(r.current for r in z.rules):
            for r in z.rules:
                if r.reason not in order:
                    order.insert(0, r.reason)
        out = []
        for code in order:
            text = self._reason_text(code, z, x, crossings_all, collar, rig, sizes)
            if text:
                out.append((code, text))
            if len(out) >= (4 if len(fams) > 1 else 3):
                break
        if not out:
            out.append(("", tr("Risk from the combination of rock, contacts and hole state ({family}).",
                               family=tr(codes.FAMILIES[fam]))))
        return out

    def _reason_text(self, code, z: Zone, x: pd.DataFrame, crossings_all, collar, rig, sizes) -> str | None:
        R = codes.REASONS
        card = self.cards.get(z.formation) or {}
        main = card.get("main_rock")
        cls = tr(codes.ROCK_CLASSES[main][0]).lower() if main in codes.ROCK_CLASSES else self._class_name(x)
        if code == "R-K01":
            if float(x["carb"].mean()) < 50:
                return None
            return fill(R[code], from_=f"{z.from_:.0f}", to=f"{z.to:.0f}", formation=z.formation,
                        d=f"{float(np.nanmean(x['d_top'])):.0f}", p=f"{100 * z.p_cavity:.0f}")
        if code == "R-K02":
            d = float(np.nanmean(x["d_top"]))
            contact = "clastic-over-carbonate" if float(x["ct_clastic_carb"].max()) > 0.5 else "carbonate"
            rc = "dolomite" if float(x["dolo"].mean()) > 0.5 else "limestone"
            rows = [r for r in self.models.cavity_stats if r["rock_class"] == rc and r["contact"] == contact]
            if not rows or float(x["carb"].mean()) < 50:
                return None
            r = min(rows, key=lambda r: abs(r["d"] - max(d, 10)))
            return fill(R[code], x=f"{r['pct']:.0f}", rock_class=tr(codes.ROCK_CLASSES[rc][0]).lower(), d=r["d"],
                        contact_type=contact)
        if code == "R-F01":
            cs = [(d, n) for d, n in crossings_all[0] if z.band_from - 15 <= d <= z.band_to + 15]
            if not cs:
                allc = [(d, n) for cl in crossings_all for d, n in cl if z.band_from - 15 <= d <= z.band_to + 15]
                if not allc:
                    return None
                cs = allc
            d, n = cs[0]
            return fill(R[code], name=n, depth=f"{d:.0f}", u=f"{z.u:.0f}")
        if code == "R-F02":
            return fill(R[code], p=f"{100 * z.p_fault:.0f}", from_=f"{z.from_:.0f}", to=f"{z.to:.0f}")
        if code == "R-B01":
            a = float(np.nanmean(x["incidence"]))
            if a >= 35:
                return None
            return fill(R[code], a=f"{a:.0f}")
        if code == "R-L01":
            if float(x["hard"].mean()) < 4 and float(x["abr"].mean()) < 4:
                return None
            return fill(R[code], rock_class=cls, from_=f"{z.from_:.0f}", to=f"{z.to:.0f}")
        if code == "R-C01":
            clay = x["clay"].to_numpy(dtype=float)
            c = float(np.nanmax(clay)) if len(clay) else 0.0
            if not c >= 1:
                return None
            names = getattr(z, "names", None) or []
            i = int(np.nanargmax(clay))
            form = names[i] if i < len(names) and names[i] else z.formation     # the unit that has the clay
            return fill(R[code], formation=form, clay=tr(list(codes.CLAY_LEVELS.values())[int(min(c, 2))]))
        if code == "R-H01":
            ratio = float(np.nanmax(x["depth_ratio"])) if x["depth_ratio"].notna().any() else None
            size = sizes[-1][2] if sizes else "HQ"
            if ratio is None:
                return None
            return fill(R[code], pct=f"{100 * ratio:.0f}", size=size, n=int(x["n_red"].max()))
        for r in z.rules:
            if r.reason == code:
                return fill(R[code], **{k: v for k, v in r.values.items()}, formation=z.formation, rock_class=cls)
        return None

    @staticmethod
    def _class_name(x: pd.DataFrame) -> str:
        h, carb, dolo = float(x["hard"].mean()), float(x["carb"].mean()), float(x["dolo"].mean())
        if dolo > 0.5:
            return tr("dolomite")
        if carb >= 50:
            return tr("limestone")
        if h >= 5:
            return tr("quartzite")
        return tr("rock")

    def _measure_ctx(self, z: Zone, x: pd.DataFrame, collar, rig, sizes, planned_setup, drilled) -> dict:
        dolo = float(x["dolo"].mean()) > 0.5
        carb = float(x["carb"].mean()) >= 50
        hard = float(x["hard"].mean())
        cls = "dolomite" if dolo else ("limestone" if carb else ("quartzite" if hard >= 5 else
                                                                  ("schist" if z.formation.upper().startswith(("U6", "PS"))
                                                                   else "sandstone")))
        card = self.cards.get(z.formation) or {}
        if card.get("main_rock"):
            cls = card["main_rock"]
        size = next((s for f, t, s in sizes if f <= z.from_ < t), planned_setup.get("size"))
        return {"from_": z.from_, "to": z.to, "band_from": z.band_from, "band_to": z.band_to, "rock_class": cls,
                "contact": "clastic-over-carbonate" if float(x["ct_clastic_carb"].max()) > 0.5 else "other",
                "depth": z.from_, "size": size, "bit": planned_setup.get("bit"), "formation": z.formation,
                "casing": collar.get("casing_available"), "fluids": collar.get("fluids_available"),
                "drilled": z.from_ <= drilled, "pull_limit": f"{rig.get('pull_kn', 0) * 0.8:.0f}" if rig.get("pull_kn")
                else None, "torque_limit": f"{rig.get('max_torque', 0) * 0.8:.0f} N·m" if rig.get("max_torque") else None}

    def _hole_messages(self, messages, XH0: pd.DataFrame, mds, drilled, u, corr, collar) -> None:
        ahead = mds > drilled
        if self.model.ok and ahead.any():
            xy = np.array([[float(collar["x"]), float(collar["y"]), float(collar["z"])]])
            dist = float(self.model.control_distance(xy)[0])
            if dist > 0.5 * float(self.model.config.get("d_far", 250.0)):
                messages.append(("R-M03", fill(codes.REASONS["R-M03"], dist=f"{dist:.0f}",
                                               u=f"{float(np.max(u[ahead])):.0f}")))
        # outside experience (spec 5.10.5)
        rng_ = self.models.ranges
        x = XH0[ahead] if ahead.any() else XH0
        for feat in ("thick", "carb", "hard", "abr", "depth", "above_wt", "s_dip", "s_fpkm", "s_valley"):
            if feat not in rng_ or feat not in x:
                continue
            lo, hi = rng_[feat]
            v = x[feat].dropna()
            if not len(v):
                continue
            worst = float(v.max()) if v.max() > hi else (float(v.min()) if v.min() < lo else None)
            if worst is not None:
                from .features import FEATURE_TEXT

                messages.append(("R-M04", fill(codes.REASONS["R-M04"], feature=tr(FEATURE_TEXT.get(feat, feat)),
                                               value=f"{worst:.1f}", min=f"{lo:.1f}", max=f"{hi:.1f}")))
        for fam, f in corr.items():
            if fam.startswith("_") or abs(f - 1.0) < 0.1:
                continue
            messages.append(("", tr("Area correction: {family} risk ×{f:.2f} (from {n} finished holes of this area).",
                                    family=tr(codes.FAMILIES[fam]), f=f, n=corr.get("_holes", 0))))

    # --- planning what-if (spec 5.5, 9) --------------------------------------------------------------
    def plan_options(self, design: dict, d_az=(-30, -15, 0, 15, 30), d_dip=(-10, 0, 10), n_paths: int = 12,
                     progress=None) -> pd.DataFrame:
        """The chance of reaching the target and the worst level for other azimuths / dips of a design."""
        rows = []
        combos = [(a, d) for d in d_dip for a in d_az]
        for i, (da, dd) in enumerate(combos):
            dsg = dict(design)
            dsg["plan_azimuth"] = (float(design["plan_azimuth"]) + da) % 360
            dsg["plan_dip"] = float(np.clip(float(design["plan_dip"]) + dd, -90, -30))
            dsg["result"] = "planned"
            dsg["final_depth"] = 0.0
            p = self.predict(design=dsg, n_paths=n_paths, seed=3)
            worst = max([z.level for z in p.zones if z.status == "Ahead"] or ["green"], key=codes.level_rank)
            main = max(p.zones, key=lambda z: z.p) if p.zones else None
            rows.append({"azimuth": round(dsg["plan_azimuth"], 1), "dip": round(dsg["plan_dip"], 1),
                         "p_reach": p.p_reach, "worst": worst, "zones": len(p.zones),
                         "main_risk": f"{main.from_:.0f}-{main.to:.0f} m {main.family_text}" if main else "",
                         "is_design": da == 0 and dd == 0})
            if progress is not None:
                progress(int(100 * (i + 1) / len(combos)), tr("Option {i} of {n}", i=i + 1, n=len(combos)))
        return pd.DataFrame(rows).sort_values("p_reach", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------
def _num(v, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def _interp_path(md: np.ndarray, xyz: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(q, md, xyz[:, a]) for a in range(3)])


def summary_text(p: Prediction) -> str:
    """The hole-level warning of spec 5.2, in a pre-defined sentence."""
    fail = 1.0 - p.p_reach
    ahead = [z for z in p.zones if z.status != "Drilled"]
    main = max(ahead, key=lambda z: z.p) if ahead else None
    state = tr("Planned hole") if p.drilled_depth <= 0 else tr("Hole")
    s = tr("{state} {hole}: {pct:.0f}% chance of failing before {depth:.0f} m.", state=state, hole=p.hole_id,
           pct=100 * fail, depth=p.planned_depth)
    if main is not None:
        s += " " + tr("Most of the risk lies at {a:.0f}-{b:.0f} m ({family}).", a=main.from_, b=main.to,
                      family=main.family_text.lower())
    else:
        s += " " + tr("No warning zone ahead.")
    return s


def warning_rows(p: Prediction) -> list[dict]:
    """The zones as rows of the risk_warning table (spec 7.4: saved with the model version)."""
    import datetime as _dt

    now = _dt.datetime.now().isoformat(timespec="seconds")
    return [{"hole_id": p.hole_id, "created": now, "depth_from": round(z.from_, 1), "depth_to": round(z.to, 1),
             "fail_codes": ",".join(z.codes), "p": round(z.p, 3), "level": z.level,
             "reasons": ";".join(c for c, _t in z.reasons if c), "measures": ";".join(s.code for s in z.prevent),
             "model_version": p.model_version} for z in p.zones]


def register_frame(p: Prediction) -> pd.DataFrame:
    """The risk register of spec 8.3."""
    rows = []
    for z in p.zones:
        rows.append({"From (m)": round(z.from_, 1), "To (m)": round(z.to, 1), "Band ±u (m)": round(z.u, 1),
                     "Failure": ", ".join(z.codes), "P": round(z.p, 2), "Level": tr(codes.LEVEL_NAMES[z.level]),
                     "Reasons": ", ".join(c for c, _t in z.reasons if c), "Measures": ", ".join(s.code for s in z.prevent),
                     "Status": tr(z.status)})
    return pd.DataFrame(rows)
