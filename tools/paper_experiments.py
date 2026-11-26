"""Experiments behind the papers and the thesis (all on the demo workspace).

    .venv\\Scripts\\python.exe tools\\paper_experiments.py [ablation] [decision] [figures]

* ablation - leave-one-area-out validation of the full model and of variants: without the rock-mechanical
  features, without noise augmentation, without the stacked Model H features, with a random split by hole
  instead of by area; global SHAP importance per family.  (~25 min)
* decision - ongoing / planned holes: predictions, sensitivity to the error band u and to the number of trial
  paths, the effect of the contact tie, an azimuth x dip map of the chance of reaching the target, the
  evidence of the suggested measures, the run parameters around the R1 alarm.  (~10 min)
* figures - charts for the manuscripts from the saved results.

Results: docs/publications/results/*.json; figures: docs/publications/figures/*.png.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfpredict.io.database import Target  # noqa: E402
from dfpredict.risk import codes, features, models  # noqa: E402
from dfpredict.risk.store import RiskStore  # noqa: E402

WS = ROOT / "demo_data" / "sb3_workspace"
OUT = ROOT / "docs" / "publications"
RES = OUT / "results"
FIG = OUT / "figures"
MECH = ["ucs", "cai", "e_mod", "brittle", "aniso", "soluble", "porosity", "density", "up_ucs_ratio"]


def _save(name, obj):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if isinstance(
        o, (np.floating, np.integer)) else str(o)), encoding="utf-8")


def _load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _summary(ms):
    v = ms.validation or {}
    return {"auc_H_cavity": v.get("auc_H_cavity"), "auc_H_fault": v.get("auc_H_fault"), "auc_hole": v.get("auc_hole"),
            "auc_hole_spread": v.get("auc_hole_spread"), "auc_hole_by_area": v.get("auc_hole_by_area"),
            "false_alarms_per_100m": v.get("false_alarms_per_100m"), "auc_family": v.get("auc_family"),
            "recall_orange": v.get("recall_orange"), "lead_m": v.get("lead_m"), "thresholds": ms.thresholds,
            "trained_on": ms.trained_on, "severity": ms.severity}


# ------------------------------------------------------------------------------------------------
def ablation():
    store = RiskStore(Target("sqlite", path=WS / "geodatabase.sqlite"))
    data = store.load_all()
    original = copy.deepcopy(features.MODEL_FEATURES)
    out = {}

    def run(name, **kw):
        t = time.time()
        print(f"[{name}] training ...", flush=True)
        ms = models.train(data, **kw)
        out[name] = _summary(ms)
        out[name]["seconds"] = round(time.time() - t)
        print(f"[{name}] done in {out[name]['seconds']} s: C={out[name]['auc_hole']}", flush=True)
        _save("ablation", out)
        return ms

    try:
        full = run("full")
        _save("shap_importance", shap_importance(full, data))
        for k in features.MODEL_FEATURES:                         # 2. without rock mechanics
            features.MODEL_FEATURES[k] = [f for f in original[k] if f not in MECH]
        run("no_rock_mechanics")
        features.MODEL_FEATURES.clear()
        features.MODEL_FEATURES.update(copy.deepcopy(original))
        run("no_noise", noise_copies=0)                            # 3. no noise augmentation
        for k in ("A", "B"):                                      # 4. no stacked hazard features
            features.MODEL_FEATURES[k] = [f for f in original[k] if f not in features.HAZARD]
        run("no_stacking")
        features.MODEL_FEATURES.clear()
        features.MODEL_FEATURES.update(copy.deepcopy(original))
        # 5. random split by hole instead of leave-one-area-out
        orig_build = models.build_training_table

        def random_folds(*a, **k):
            tt = orig_build(*a, **k)
            holes = np.array(sorted(set(tt.hole)))
            rng = np.random.default_rng(11)
            fold = dict(zip(holes, rng.integers(0, 9, len(holes))))
            tt.area = np.array([f"R{fold[h]}" for h in tt.hole], dtype=object)
            return tt
        models.build_training_table = random_folds
        try:
            run("random_split")
        finally:
            models.build_training_table = orig_build
    finally:
        features.MODEL_FEATURES.clear()
        features.MODEL_FEATURES.update(original)


def shap_importance(ms, data, n_rows: int = 40000):
    """Mean |SHAP| per feature of every model, on a random sample of the (clean) training rows."""
    tt = models.build_training_table(data, noise_copies=0)
    rng = np.random.default_rng(5)
    idx = rng.choice(len(tt.X), size=min(n_rows, len(tt.X)), replace=False)
    X = tt.X.iloc[idx].reset_index(drop=True)
    groups = tt.hole[idx]
    XH = ms.with_hazard(X, groups)
    out = {}
    for k in ("cavity", "fault"):
        c = ms.hazard_contributions(X, k)
        if c is not None:
            out[f"H_{k}"] = c.abs().mean().sort_values(ascending=False).head(15).round(5).to_dict()
    for fam in codes.FAMILIES:
        c = ms.contributions(XH, fam, "A")
        if c is not None:
            out[f"A_{fam}"] = c.abs().mean().sort_values(ascending=False).head(15).round(5).to_dict()
    return out


# ------------------------------------------------------------------------------------------------
def _engine():
    from dfpredict.risk.engine import RiskEngine
    from dfpredict.risk.geomodel import surfaces_from_files

    eng = RiskEngine(Target("sqlite", path=WS / "geodatabase.sqlite"), WS)
    pr = [(p, "horizon") for p in sorted((WS / "model").glob("horizon_*.dtm"))]
    pr += [(p, "fault") for p in sorted((WS / "faults").glob("F*.dtm"))]
    eng.set_surfaces(surfaces_from_files(pr))
    return eng


def _pred_dict(p):
    return {"hole": p.hole_id, "summary": p.summary, "p_reach": p.p_reach, "planned": p.planned_depth,
            "drilled": p.drilled_depth, "build": p.ensemble.build_rate, "turn": p.ensemble.turn_rate,
            "end_spread": p.ensemble.end_offset(), "contacts": p.contacts, "crossings": p.crossings,
            "messages": p.messages, "area_factor": p.area_factor,
            "rules": [{"rule": r.rule, "from": r.depth_from, "to": r.depth_to, "current": r.current, "values": r.values}
                      for r in p.rules],
            "zones": [{"from": z.from_, "to": z.to, "u": z.u, "level": z.level, "p": z.p, "prob10": z.prob10,
                       "codes": z.codes, "families": z.families, "status": z.status, "formation": z.formation,
                       "p_cavity": z.p_cavity, "p_fault": z.p_fault, "reasons": z.reasons,
                       "prevent": [{"code": s.code, "text": s.text, "evidence": s.evidence, "uses": s.uses,
                                    "solved": s.solved, "kind": s.kind} for s in z.prevent],
                       "respond": [{"code": s.code, "text": s.text, "evidence": s.evidence, "uses": s.uses,
                                    "solved": s.solved, "kind": s.kind} for s in z.respond]} for z in p.zones],
            "range_next50": p.range_probabilities(p.drilled_depth, min(p.planned_depth, p.drilled_depth + 50)),
            "range_target": p.range_probabilities(p.drilled_depth, p.planned_depth),
            "md": p.md.tolist(), "h": {k: v.tolist() for k, v in p.h.items()}, "p_cavity": p.p_cavity.tolist(),
            "p_fault": p.p_fault.tolist(), "u": p.u.tolist(), "level": list(p.level), "unit": list(p.unit),
            "index": {k: v.tolist() for k, v in p.index.items()},
            "mean_xyz": p.ensemble.mean_xyz.tolist(), "radius": p.ensemble.radius.tolist(), "ens_md": p.ensemble.md.tolist(),
            "samples": p.ensemble.samples[:25].tolist()}


def decision():
    eng = _engine()
    out = {"predictions": {}}
    for hid in ("DH-2026-001", "DH-2026-002", "DH-2026-003"):
        print("predict", hid, flush=True)
        out["predictions"][hid] = _pred_dict(eng.predict(hid, n_paths=60))
    _save("decision_predictions", out)
    # sensitivity: number of trial paths (repeat seeds)
    sens = {"paths": {}, "u_far": {}, "tie": {}}
    pr = eng.predictor
    for n in (10, 20, 40, 80):
        vals = [pr.predict("DH-2026-001", n_paths=n, seed=s).p_reach for s in range(6)]
        sens["paths"][n] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "values": vals}
        print("paths", n, sens["paths"][n]["mean"], sens["paths"][n]["sd"], flush=True)
    # error band
    base_cfg = dict(eng.geomodel.config)
    for uf in (10.0, 20.0, 30.0, 45.0, 60.0):
        eng.geomodel.config["u_far"] = uf
        pr._area_factor = pr._area_factor            # keep
        p = pr.predict("DH-2026-001", n_paths=40, seed=1)
        z = max(p.zones, key=lambda z: z.p) if p.zones else None
        sens["u_far"][uf] = {"p_reach": p.p_reach, "main_zone": [z.from_, z.to, z.u, z.level] if z else None,
                             "n_zones": len(p.zones), "mean_u_ahead": float(np.mean(p.u[p.md > p.drilled_depth]))}
        print("u_far", uf, p.p_reach, flush=True)
    eng.geomodel.config.update(base_cfg)
    # tie on / off
    p_tie = pr.predict("DH-2026-001", n_paths=40, seed=1)
    orig = pr._ties
    pr._ties = lambda ens, geo, drilled, messages: (None, None, 1.0)
    try:
        p_no = pr.predict("DH-2026-001", n_paths=40, seed=1)
    finally:
        pr._ties = orig
    for name, p in (("with_tie", p_tie), ("without_tie", p_no)):
        sens["tie"][name] = {"p_reach": p.p_reach, "contacts": p.contacts,
                             "zones": [[z.from_, z.to, z.level, z.u] for z in p.zones]}
    _save("decision_sensitivity", sens)
    # plan check map around DH-2026-003 and a designed hole near DH-2026-001
    maps = {}
    data = eng.data
    for name, base in (("DH-2026-003", data.hole("collar", "DH-2026-003").iloc[0].to_dict()),
                       ("PLAN-A", {"hole_id": "PLAN-A", "area_id": "SB3", "rig_model": "XY-44", "x": 5450.0,
                                   "y": 8420.0, "z": 316.0, "plan_azimuth": 300.0, "plan_dip": -70.0,
                                   "planned_depth": 300.0, "casing_available": "HW;NW;BW",
                                   "fluids_available": "water;polymer;bentonite"})):
        print("plan map", name, flush=True)
        df = pr.plan_options(dict(base, result="planned", final_depth=0.0), d_az=(-60, -30, 0, 30, 60),
                             d_dip=(-15, 0, 15), n_paths=12)
        maps[name] = {"base": {k: base.get(k) for k in ("x", "y", "z", "plan_azimuth", "plan_dip", "planned_depth")},
                      "options": df.to_dict("records")}
    _save("decision_planmaps", maps)
    # measure evidence statistics
    adv = pr.advisor
    occ = adv.occ
    ev = {}
    for code in [m for m in codes.MEASURES if m.startswith(("M-LC", "M-CV", "M-CL", "M-CS", "M-SP", "M-DV", "M-BG", "M-RF",
                                                             "M-BH", "M-WI"))]:
        tried = occ[occ["measures"].map(lambda ms, c=code: c in ms)]
        if len(tried):
            solved = sum(1 for ms, s in zip(tried["measures"], tried["solved"]) if s and ms and ms[-1] == code)
            ev[code] = {"uses": int(len(tried)), "solved_last": int(solved), "rate": solved / len(tried)}
    from dfpredict.risk.demo import MEASURE_SUCCESS
    _save("decision_measures", {"evidence": ev, "truth_success": MEASURE_SUCCESS,
                                "window_dolomite_HQ": adv.operating_window("dolomite", "HQ", "impregnated"),
                                "window_sandstone_HQ": adv.operating_window("sandstone", "HQ", "impregnated")})
    # runs around the R1 alarm of DH-2026-002
    runs = data.hole("run", "DH-2026-002")
    _save("decision_runs_dh002", runs.tail(12).to_dict("records"))


# ------------------------------------------------------------------------------------------------
def dataset_stats():
    store = RiskStore(Target("sqlite", path=WS / "geodatabase.sqlite"))
    d = store.load_all()
    c = d.collar
    fin = c[c["result"].isin(["success", "failed"])]
    f = d["failure"].copy()
    f["family"] = f["fail_code"].map(codes.family_of)
    by_area = []
    for a, g in fin.groupby("area_id"):
        ff = f[f["hole_id"].isin(g["hole_id"])]
        by_area.append({"area": a, "holes": len(g), "failed": int((g["result"] == "failed").sum()),
                        "metres": float(g["final_depth"].sum()), "mean_depth": float(g["final_depth"].mean()),
                        "families": ff["family"].value_counts().to_dict()})
    occ = d["occurrence"]["type"].value_counts().to_dict()
    _save("dataset", {"holes": len(c), "finished": len(fin), "failed": int((fin["result"] == "failed").sum()),
                      "metres": float(fin["final_depth"].sum()), "runs": len(d["run"]), "surveys": len(d["survey"]),
                      "geology": len(d["geology"]), "occurrences": occ, "fail_codes": f["fail_code"].value_counts().to_dict(),
                      "families": f["family"].value_counts().to_dict(), "severity_mean": float(pd.to_numeric(f["severity"], errors="coerce").mean()),
                      "by_area": by_area, "formations": len(d["formation"]), "rocks": len(d["rock"]),
                      "areas": d["area"].to_dict("records")})


def seeds(values=(6, 7)):
    """Repeat 'full' and 'no_rock_mechanics' with other random seeds (noise copies, subsampling, trees), so the
    ablation differences can be compared with the run-to-run spread."""
    store = RiskStore(Target("sqlite", path=WS / "geodatabase.sqlite"))
    data = store.load_all()
    original = copy.deepcopy(features.MODEL_FEATURES)
    out = {}
    try:
        for sd in values:
            for name in ("full", "no_rock_mechanics"):
                features.MODEL_FEATURES.clear()
                features.MODEL_FEATURES.update(copy.deepcopy(original))
                if name == "no_rock_mechanics":
                    for k in features.MODEL_FEATURES:
                        features.MODEL_FEATURES[k] = [f for f in original[k] if f not in MECH]
                t = time.time()
                print(f"[{name} seed {sd}] training ...", flush=True)
                ms = models.train(data, seed=sd)
                out[f"{name}_seed{sd}"] = _summary(ms)
                out[f"{name}_seed{sd}"]["seconds"] = round(time.time() - t)
                print(f"[{name} seed {sd}] C={out[f'{name}_seed{sd}']['auc_hole']}", flush=True)
                _save("ablation_seeds", out)
    finally:
        features.MODEL_FEATURES.clear()
        features.MODEL_FEATURES.update(original)


if __name__ == "__main__":
    what = set(sys.argv[1:]) or {"ablation", "decision"}
    if "stats" in what or "ablation" in what or "decision" in what:
        dataset_stats()
    if "decision" in what:
        decision()
    if "ablation" in what:
        ablation()
    if "seeds" in what:
        seeds()
