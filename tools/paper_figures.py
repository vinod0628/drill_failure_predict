"""Figures for the papers and the thesis (light theme): program screenshots and charts from the experiment results.

    .venv\\Scripts\\python.exe tools\\paper_figures.py [shots] [charts]

Reads docs/publications/results/*.json (tools/paper_experiments.py), writes docs/publications/figures/*.png.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

WS = ROOT / "demo_data" / "sb3_workspace"
RES = ROOT / "docs" / "publications" / "results"
FIG = ROOT / "docs" / "publications" / "figures"


def R(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------------------------------------------
def shots():
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from dfpredict import i18n
    from dfpredict.ui import theme

    i18n.set_language("en")
    theme.apply_theme(app, "light")
    app.setFont(QFont("Segoe UI", 9))
    from dfpredict.io.database import Target
    from dfpredict.risk.engine import RiskEngine
    from dfpredict.risk.geomodel import surfaces_from_files
    from dfpredict.ui.history_dialog import DrillingRecordsDialog
    from dfpredict.ui.prediction_dialog import PredictionDialog
    from dfpredict.ui.risk_admin import GeoModelDialog, ModelsDialog
    from dfpredict.scene.loaders import load_paths

    eng = RiskEngine(Target("sqlite", path=WS / "geodatabase.sqlite"), WS)
    pr = [(p, "horizon") for p in sorted((WS / "model").glob("horizon_*.dtm"))]
    pr += [(p, "fault") for p in sorted((WS / "faults").glob("F*.dtm"))]
    eng.set_surfaces(surfaces_from_files(pr))
    FIG.mkdir(parents=True, exist_ok=True)
    preds = {}
    dlg = PredictionDialog(eng)
    dlg.resize(1600, 980)
    dlg.show()
    for hid in ("DH-2026-001", "DH-2026-002", "DH-2026-003"):
        p = eng.predict(hid, n_paths=40)
        preds[hid] = p
        i = dlg.hole.findData(hid)
        dlg.hole.setCurrentIndex(i)
        dlg.set_prediction(p)
        app.processEvents()
        dlg.grab().save(str(FIG / f"shot_prediction_{hid}.png"))
        dlg.chart.fig.set_size_inches(7.5, 7.5)
        dlg.chart.show_prediction(p, max(range(len(p.zones)), key=lambda k: p.zones[k].p) if p.zones else -1)
        dlg.chart.fig.savefig(FIG / f"risklog_{hid}.png", dpi=170, facecolor=dlg.chart.fig.get_facecolor())
    dlg.mode.setCurrentIndex(1)
    dlg._design_from_hole()
    app.processEvents()
    dlg.mode.grab().save(str(FIG / "shot_plancheck.png"))
    dlg.close()
    rec = DrillingRecordsDialog(eng.target, eng.folder)
    rec.resize(1600, 950)
    rec.show()
    rec.reload()
    rec.load_hole("DH-2026-001", force=True)
    for i, name in ((0, "survey"), (1, "geology"), (2, "runs"), (5, "occurrences")):
        rec.sub.setCurrentIndex(i)
        app.processEvents()
        rec.grab().save(str(FIG / f"shot_records_{name}.png"))
    for i, name in ((1, "areas"), (2, "formations"), (4, "rock"), (6, "measures"), (7, "overview")):
        rec.tabs.setCurrentIndex(i)
        app.processEvents()
        rec.grab().save(str(FIG / f"shot_records_{name}.png"))
    rec.close()
    md = ModelsDialog(eng)
    md.resize(1080, 820)
    md.show()
    app.processEvents()
    md.grab().save(str(FIG / "shot_models.png"))
    md.close()
    layers, _ = load_paths(sorted((WS / "model").glob("horizon_*.dtm")) + sorted((WS / "faults").glob("F*.dtm")))
    for lay in layers:
        lay.role = "fault" if lay.name.startswith("F") else "horizon"
    gd = GeoModelDialog(layers, WS, {r["formation"]: r for r in eng.data["formation"].to_dict("records")})
    gd.set_areas(list(eng.data["area"]["area_id"].astype(str)), "SB3")
    gd.resize(1080, 760)
    gd.show()
    app.processEvents()
    gd.grab().save(str(FIG / "shot_geomodel.png"))
    gd.close()
    view3d(preds)


def view3d(preds):
    import numpy as np
    import pyvista as pv

    from dfpredict.scene.loaders import load_paths
    from dfpredict.scene.risk_layer import RiskLayer
    from dfpredict.scene.scene import Scene

    def render(name, holes, zoom, faults=("F4",), hzop=0.18, az=None):
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))
        scene = Scene(pl)
        files = sorted((WS / "model").glob("horizon_*.dtm")) + [WS / "faults" / f"{f}.dtm" for f in faults]
        layers, _ = load_paths(files)
        cols = [(0.93, 0.80, 0.45), (0.35, 0.35, 0.40), (0.30, 0.65, 0.95), (0.62, 0.78, 0.45), (0.80, 0.55, 0.85)]
        for i, lay in enumerate(layers):
            lay.use_styles = False
            if lay.name.startswith("F"):
                lay.color, lay.opacity = (0.9, 0.25, 0.25), 0.16
            else:
                lay.color, lay.opacity = cols[i % 5], hzop
            scene.add_layer(lay)
        rl = RiskLayer(name="Drilling risk", path=None)
        rl.predictions = {h: preds[h] for h in holes}
        if len(holes) == 1:
            p = preds[holes[0]]
            zi = max(range(len(p.zones)), key=lambda k: p.zones[k].p)
            rl.selected = (holes[0], zi)
        scene.add_layer(rl)
        scene.set_background((1, 1, 1))
        pl.render()
        if az:
            pl.camera.azimuth = az
        pl.camera.zoom(zoom)
        pl.screenshot(str(FIG / name))
        pl.close()
    render("view3d_dh001.png", ["DH-2026-001"], 3.4, hzop=0.10)
    render("view3d_all.png", ["DH-2026-001", "DH-2026-002", "DH-2026-003"], 2.2, faults=("F1", "F2", "F3", "F4"),
           hzop=0.12)


# ------------------------------------------------------------------------------------------------
def charts():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from dfpredict.risk import codes, rockmech

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                         "savefig.dpi": 200})
    fam_names = codes.FAMILIES
    fam_cols = codes.FAMILY_COLORS

    def save(f, name):
        f.tight_layout()
        f.savefig(FIG / name, dpi=200)
        plt.close(f)

    # --- system architecture ---
    f, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#445",
                                    lw=0.8))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h - 6.2, text, ha="center", va="top", fontsize=8.2, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#445", lw=0.9))
    box(2, 47, 29, 13, "Geological model", "horizons, faults, solids by role;\nerror band ±u; contact ties", "#f6efe0")
    box(35.5, 47, 29, 13, "Live drillhole", "design, surveys, geology log,\nruns, events (Drilling Records)", "#e7f0fa")
    box(69, 47, 29, 13, "Past holes (history)", "multi-area records: logs, runs,\nfailures, occurrences, measures", "#eaf6ea")
    box(2, 27, 96, 13, "Project database + feature engine",
        "minimum-curvature desurvey · 1-m rows · rock-mechanical descriptors and formation cards · position in unit ·\n"
        "contact setting · incidence angle · faults · site sheet · hole state · drilling setup and response", "#f2f2f2")
    box(2, 6, 30, 14, "Trajectory and model ensemble", "trial paths from surveys and design;\nmodel offsets ~ N(0, u)", "#f6efe0")
    box(35, 6, 30, 14, "Models H, A, B + rules R1-R8", "hazard probability (H), failure hazard\nper family (A), precursors (B)",
        "#e7f0fa")
    box(68, 6, 30, 14, "Risk and countermeasures", "levels, zones ±u, SHAP reasons,\nevidence-ranked measures, P(reach)",
        "#fbe9e7")
    for x in (16.5, 50, 83.5):
        arrow(x, 47, x, 40.5)
    arrow(17, 27, 17, 20.5)
    arrow(50, 27, 50, 20.5)
    arrow(32.5, 13, 34.5, 13)
    arrow(65.5, 13, 67.5, 13)
    ax.text(50, 1.5, "finished holes and measure outcomes return to the history; models are retrained", ha="center",
            fontsize=8.5, style="italic", color="#333")
    save(f, "fig_architecture.png")

    # --- dataset ---
    ds = R("dataset")
    if ds:
        ba = [a for a in ds["by_area"]]
        f, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        x = np.arange(len(ba))
        axs[0].bar(x, [a["holes"] for a in ba], color="#7f9cc0", label="finished holes")
        axs[0].bar(x, [a["failed"] for a in ba], color="#c0392b", label="failed")
        axs[0].set_xticks(x, [a["area"] for a in ba])
        axs[0].set_ylabel("holes")
        axs[0].legend(fontsize=8)
        axs[0].set_title("(a) Holes and failures per area")
        bottom = np.zeros(len(ba))
        for fam in fam_names:
            vals = np.array([a["families"].get(fam, 0) for a in ba])
            axs[1].bar(x, vals, bottom=bottom, color=fam_cols[fam], label=fam_names[fam])
            bottom += vals
        axs[1].set_xticks(x, [a["area"] for a in ba])
        axs[1].set_ylabel("failures")
        axs[1].legend(fontsize=7.5)
        axs[1].set_title("(b) Failures per family and area")
        save(f, "fig_dataset.png")

    # --- rocks ---
    f, ax = plt.subplots(figsize=(9, 5.4))
    groups = {"clastic": "#b8860b", "carbonate": "#1f77b4", "chemical": "#8e44ad", "metamorphic": "#16a085",
              "igneous": "#c0392b", "ore": "#7f8c8d", "fault": "#e67e22", "weathered": "#95a5a6"}
    for k, pr in rockmech.DEFAULTS.items():
        ax.errorbar(pr.cai[1], pr.ucs[1], xerr=[[pr.cai[1] - pr.cai[0]], [pr.cai[2] - pr.cai[1]]],
                    yerr=[[pr.ucs[1] - pr.ucs[0]], [pr.ucs[2] - pr.ucs[1]]], fmt="o", ms=4, color=groups[pr.group],
                    alpha=0.75, lw=0.5)
        ax.annotate(k, (pr.cai[1], pr.ucs[1]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    for v in (25, 50, 100, 250):
        ax.axhline(v, color="#999", lw=0.5, ls=":")
    for v in (1, 2, 3, 4):
        ax.axvline(v, color="#999", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Cerchar abrasivity index CAI")
    ax.set_ylabel("Uniaxial compressive strength UCS (MPa)")
    for g, c in groups.items():
        ax.plot([], [], "o", color=c, label=g)
    ax.legend(fontsize=7.5, ncol=4, loc="lower right")
    save(f, "fig_rocks.png")

    dp = R("decision_predictions")
    if dp:
        p1 = dp["predictions"]["DH-2026-001"]
        # trajectory ensemble
        f, axs = plt.subplots(1, 2, figsize=(10, 4.2))
        xyz = np.array(p1["mean_xyz"])
        md = np.array(p1["ens_md"])
        o = xyz[0]
        k = int(np.searchsorted(md, p1["drilled"]))
        for s in p1["samples"]:
            s = np.array(s)
            axs[0].plot(s[:, 0] - o[0], s[:, 1] - o[1], color="#9ab", lw=0.5)
            axs[1].plot(np.hypot(s[:, 0] - o[0], s[:, 1] - o[1]), s[:, 2], color="#9ab", lw=0.5)
        for a, (xx, yy) in ((axs[0], (xyz[:, 0] - o[0], xyz[:, 1] - o[1])),
                            (axs[1], (np.hypot(xyz[:, 0] - o[0], xyz[:, 1] - o[1]), xyz[:, 2]))):
            a.plot(xx[:k], yy[:k], color="#c0392b", lw=2.2, label="drilled (surveys)")
            a.plot(xx[k:], yy[k:], color="#1f4e79", lw=1.6, ls="--", label="predicted mean path")
        az, dpp = math.radians(300), math.radians(-72)
        dd = np.array([0, p1["planned"]])
        axs[0].plot(dd * math.cos(dpp) * math.sin(az), dd * math.cos(dpp) * math.cos(az), color="#27ae60", lw=1,
                    label="straight design")
        axs[1].plot(dd * math.cos(dpp), o[2] + dd * math.sin(dpp), color="#27ae60", lw=1)
        axs[0].set_title("(a) Plan view (m from collar)")
        axs[1].set_title("(b) Vertical section along the hole")
        axs[0].set_aspect("equal")
        axs[1].set_xlabel("horizontal distance (m)")
        axs[1].set_ylabel("elevation (m)")
        axs[0].legend(fontsize=7.5, loc="lower left")
        save(f, "fig_trajectory.png")
        # hazard + survival for three holes
        f, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
        for a, (hid, p) in zip(axs, dp["predictions"].items()):
            mdp = np.array(p["md"])
            m = mdp <= p["planned"]
            for fam, h in p["h"].items():
                a.plot(mdp[m], np.array(h)[m], color=fam_cols[fam], lw=1.1, label=fam_names[fam])
            if p["drilled"] > 0:
                a.axvline(p["drilled"], color="k", lw=0.8)
            a.set_title(f"{hid}: P(reach) = {p['p_reach']:.2f}", fontsize=9.5)
            a.set_xlabel("depth along hole (m)")
        axs[0].set_ylabel("hazard per metre")
        axs[0].legend(fontsize=7)
        save(f, "fig_hazard_three.png")
        # runs of DH-2026-002 around the R1 alarm
        runs = R("decision_runs_dh002")
        if runs:
            import pandas as pd

            r = pd.DataFrame(runs)
            f, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
            y = (r["depth_from"] + r["depth_to"]) / 2
            for a, col, lab, c in ((axs[0], "pump_press", "pump pressure (MPa)", "#1f4e79"),
                                   (axs[1], "return_pct", "return (%)", "#16a085"), (axs[2], "rop", "ROP (m/h)", "#c0392b")):
                a.plot(r[col].astype(float), y, marker="o", color=c, ms=3)
                a.set_xlabel(lab)
            axs[0].invert_yaxis()
            axs[0].set_ylabel("depth along hole (m)")
            for a in axs:
                a.axhspan(float(r["depth_from"].iloc[-2]), float(r["depth_to"].iloc[-1]), color="#f0883e", alpha=0.2)
            save(f, "fig_runs_dh002.png")
    sens = R("decision_sensitivity")
    if sens:
        f, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
        ns = sorted(int(k) for k in sens["paths"])
        axs[0].errorbar(ns, [sens["paths"][str(n)]["mean"] for n in ns], yerr=[sens["paths"][str(n)]["sd"] for n in ns],
                        marker="o", capsize=4, color="#1f4e79")
        axs[0].set_xscale("log", base=2)
        axs[0].set_xlabel("number of trial paths N")
        axs[0].set_ylabel("P(reach target), mean ± sd (6 seeds)")
        axs[0].set_title("(a) Monte Carlo convergence")
        us = sorted(float(k) for k in sens["u_far"])
        axs[1].plot(us, [sens["u_far"][str(u) if str(u) in sens["u_far"] else f"{u}"]["p_reach"] for u in us], marker="s",
                    color="#c0392b", label="P(reach target)")
        ax2 = axs[1].twinx()
        ax2.plot(us, [sens["u_far"][f"{u}"]["mean_u_ahead"] for u in us], marker="^", color="#7f8c8d",
                 label="mean u ahead of the bit")
        ax2.set_ylabel("mean u ahead (m)")
        axs[1].set_xlabel("u_far (m)")
        axs[1].set_title("(b) Sensitivity to the error band")
        axs[1].set_ylabel("P(reach target)")
        save(f, "fig_sensitivity.png")
    maps = R("decision_planmaps")
    if maps:
        f, axs = plt.subplots(1, len(maps), figsize=(10, 3.8))
        for a, (name, m) in zip(np.atleast_1d(axs), maps.items()):
            import pandas as pd

            df = pd.DataFrame(m["options"])
            if (df["azimuth"] > 180).any():
                df["azimuth"] = df["azimuth"].where(df["azimuth"] > 90, df["azimuth"] + 360)
            piv = df.pivot_table(index="dip", columns="azimuth", values="p_reach")
            im = a.imshow(piv.values, cmap="RdYlGn", vmin=max(0.0, piv.values.min() - 0.02), vmax=1.0, aspect="auto",
                          origin="lower")
            a.set_xticks(range(len(piv.columns)), [f"{c % 360:.0f}" for c in piv.columns])
            a.set_yticks(range(len(piv.index)), [f"{i:.0f}" for i in piv.index])
            for (i, j), val in np.ndenumerate(piv.values):
                a.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
            b = m["base"]
            a.set_title(f"{name}: design az {b['plan_azimuth']:.0f}°, dip {b['plan_dip']:.0f}°", fontsize=9.5)
            a.set_xlabel("azimuth (°)")
            a.set_ylabel("dip (°)")
            a.grid(False)
            f.colorbar(im, ax=a, fraction=0.046, label="P(reach target)")
        save(f, "fig_planmaps.png")
    mev = R("decision_measures")
    if mev:
        ev = mev["evidence"]
        truth = mev["truth_success"]
        ks = [k for k in ev if k in truth and ev[k]["uses"] >= 5]
        f, ax = plt.subplots(figsize=(7.5, 4))
        x = np.array([truth[k] for k in ks])
        y = np.array([ev[k]["rate"] for k in ks])
        s = np.array([ev[k]["uses"] for k in ks])
        ax.scatter(x, y, s=10 + s / 3, color="#1f4e79", alpha=0.7)
        for k, xi, yi in zip(ks, x, y):
            ax.annotate(k, (xi, yi), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=0.8)
        ax.set_xlabel("true success probability of the measure (generator)")
        ax.set_ylabel("solved rate learned from the records")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        save(f, "fig_measure_evidence.png")
    ab = R("ablation")
    if ab:
        names = [n for n in ("full", "no_rock_mechanics", "no_noise", "no_stacking", "random_split") if n in ab]
        labels = {"full": "full", "no_rock_mechanics": "no rock\nmechanics", "no_noise": "no noise\naugment.",
                  "no_stacking": "no H\nstacking", "random_split": "random hole\nsplit"}
        f, axs = plt.subplots(1, 3, figsize=(11, 3.6))
        x = np.arange(len(names))
        axs[0].bar(x, [ab[n]["auc_hole"] or 0 for n in names], color="#1f4e79")
        axs[0].set_title("(a) Hole-level concordance C")
        axs[0].set_ylim(0.5, 1.0)
        for i, fam in enumerate(("karst", "instability")):
            axs[1].bar(x + (i - 0.5) * 0.38, [(ab[n]["auc_family"] or {}).get(fam) or 0 for n in names], 0.38,
                       color=fam_cols[fam], label=fam_names[fam])
            axs[2].bar(x + (i - 0.5) * 0.38, [(ab[n]["recall_orange"] or {}).get(fam) or 0 for n in names], 0.38,
                       color=fam_cols[fam], label=fam_names[fam])
        axs[1].set_title("(b) Metre-level ROC-AUC")
        axs[1].set_ylim(0.5, 1.0)
        axs[2].set_title("(c) Recall at Orange (≥ 3 m ahead)")
        axs[2].set_ylim(0, 1.0)
        axs[1].legend(fontsize=7)
        for a in axs:
            a.set_xticks(x, [labels[n] for n in names], fontsize=7.5)
        save(f, "fig_ablation.png")
        full = ab["full"]
        f, ax = plt.subplots(figsize=(7.5, 3.4))
        areas = list((full.get("auc_hole_by_area") or {}).keys())
        vals = [full["auc_hole_by_area"][a] or 0 for a in areas]
        ax.bar(areas, vals, color="#7f9cc0")
        ax.axhline(full["auc_hole"], color="#c0392b", lw=1, ls="--", label=f"pooled C = {full['auc_hole']:.2f}")
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("hole-level concordance")
        ax.legend(fontsize=8)
        save(f, "fig_auc_by_area.png")
    sh = R("shap_importance")
    if sh:
        from dfpredict.risk.features import FEATURE_TEXT

        keys = [k for k in ("H_cavity", "A_karst", "A_instability", "A_trajectory") if k in sh]
        f, axs = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.6))
        for a, k in zip(np.atleast_1d(axs), keys):
            import textwrap
            items = sorted(sh[k].items(), key=lambda kv: -kv[1])[:10][::-1]
            a.barh(range(len(items)), [v for _, v in items], color="#1f4e79")
            a.set_yticks(range(len(items)), [textwrap.fill(str(FEATURE_TEXT.get(n, n)), 26) for n, _ in items])
            a.set_title(k.replace("_", " "), fontsize=9.5)
            a.tick_params(axis="y", labelsize=7)
            a.set_xlabel("mean |SHAP|")
        save(f, "fig_shap.png")


if __name__ == "__main__":
    what = set(sys.argv[1:]) or {"shots", "charts"}
    if "charts" in what:
        charts()
    if "shots" in what:
        shots()
