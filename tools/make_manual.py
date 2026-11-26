"""Build docs/Algorithm & Manual.pdf - the algorithm description and user manual of the drilling risk system.

    .venv\\Scripts\\python.exe tools\\make_manual.py [--no-shots]

Tables (database fields, rock properties, measures, reasons, rules, model settings, validation scores) are read
from the code and the demo workspace, so the manual matches the system it describes; rebuild it after changes.
Formulas are drawn with matplotlib (mathtext), the pages laid out and printed to PDF with Qt (QTextDocument).
Figures and screenshots go to docs/manual_assets/.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

DOCS = ROOT / "docs"
ASSETS = DOCS / "manual_assets"
OUT = DOCS / "Algorithm & Manual.pdf"
WS = ROOT / "demo_data" / "sb3_workspace"
E = html.escape

# ------------------------------------------------------------------------------------------------
# small HTML builders
# ------------------------------------------------------------------------------------------------
_eq_no = [0]
_fig_no = [0]
_tab_no = [0]


def h1(t):
    return f"<h1 style='page-break-before: always'>{t}</h1>"


def h2(t):
    return f"<h2>{t}</h2>"


def h3(t):
    return f"<h3>{t}</h3>"


def p(t):
    return f"<p>{t}</p>"


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def ol(items):
    return "<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def note(t):
    return (f"<table width='100%' cellpadding='6' cellspacing='0' style='background-color:#eef4fb'>"
            f"<tr><td><b>Note.</b> {t}</td></tr></table>")


def table(head, rows, widths=None, caption=None, small=False):
    _tab_no[0] += 1
    fs = "font-size:8pt;" if small else ""
    out = []
    if caption:
        out.append(f"<p class='cap'>Table {_tab_no[0]}. {caption}</p>")
    out.append(f"<table width='100%' border='1' cellspacing='0' cellpadding='3' style='border-color:#9aa5b1;{fs}'>")
    # Qt rich text does not pass a table's font size to its cells: set it on every cell
    def sp(c):
        if not small:
            return c
        c = str(c).replace("<b>", "<b style='font-size:7.5pt'>")      # bold text keeps its own size otherwise
        return f"<span style='font-size:7.5pt'>{c}</span>"
    out.append("<tr>" + "".join(
        f"<th bgcolor='#dfe6ee' align='left'{f' width={widths[i]}%' if widths else ''}>{sp(h)}</th>"
        for i, h in enumerate(head)) + "</tr>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td valign='top'>{sp(c)}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "".join(out)


def eq(tex: str, name: str, scale: float = 1.0) -> str:
    """A numbered display formula rendered with matplotlib mathtext."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _eq_no[0] += 1
    path = ASSETS / f"eq_{name}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
    plt.close(fig)
    from PySide6.QtGui import QImage

    w = QImage(str(path)).width()
    width = int(w / 220 * 96 * 0.92 * scale)
    return (f"<table width='100%' cellspacing='0' cellpadding='2'><tr><td width='92%' align='center'>"
            f"<img src='{path.as_posix()}' width='{width}'></td>"
            f"<td width='8%' align='right'>({_eq_no[0]})</td></tr></table>")


def fig(path: Path, caption: str, width: int = 620) -> str:
    _fig_no[0] += 1
    return (f"<p align='center'><img src='{path.as_posix()}' width='{width}'></p>"
            f"<p class='cap' align='center'>Figure {_fig_no[0]}. {caption}</p>")


def code(t):
    return f"<code>{E(t)}</code>"


# ------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------
def figures(engine, pred, ms) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from dfpredict.risk import codes, rockmech

    out = {}
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

    # architecture
    f, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, title, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#556",
                                    lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9, weight="bold")
        ax.text(x + w / 2, y + h - 6.3, text, ha="center", va="top", fontsize=7.4, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color="#556", lw=0.9))
    box(2, 46, 28, 12, "Geological model", "horizons, faults, solids\n(layer roles) + error band ±u", "#f6efe0")
    box(36, 46, 28, 12, "Live drillhole", "design, surveys, geology log,\nruns, events (Drilling Records)", "#e7f0fa")
    box(70, 46, 28, 12, "Past holes (history)", "1,000+ holes from many areas:\nlogs, runs, failures, events", "#eaf6ea")
    box(2, 27, 96, 12, "Database + feature engine (project database: drillhole_* tables shared with GeoViewer)",
        "minimum-curvature desurvey · 1 m rows · rock mechanics + formation cards · position in unit · contacts ·\n"
        "incidence angle · faults · site sheet · hole state · drilling setup / response", "#f2f2f2")
    box(2, 8, 30, 13, "Path + model ensemble", "trial paths from surveys and design;\nmodel offsets ~ N(0, u); ties", "#f6efe0")
    box(35, 8, 30, 13, "Models H, A, B + rules", "cavity / fault-zone P (H), hazard per\nfamily (A), precursors (B), R1 - R8",
        "#e7f0fa")
    box(68, 8, 30, 13, "Risk + measures", "levels, zones ±u, reasons (SHAP),\nmeasures ranked by evidence, P(reach)",
        "#fbe9e7")
    for x in (16, 50, 84):
        arrow(x, 46, x, 39.5)
    arrow(17, 27, 17, 21.5)
    arrow(50, 27, 50, 21.5)
    arrow(32.5, 14.5, 34.5, 14.5)
    arrow(65.5, 14.5, 67.5, 14.5)
    ax.text(50, 2, "every finished hole joins the history and the models are retrained", ha="center", fontsize=8,
            style="italic", color="#444")
    p_ = ASSETS / "fig_architecture.png"
    f.savefig(p_, dpi=170, bbox_inches="tight")
    plt.close(f)
    out["architecture"] = p_

    # trajectory ensemble of DH-2026-001
    ens = pred.ensemble
    f, axs = plt.subplots(1, 2, figsize=(10, 4.3))
    o = ens.mean_xyz[0]
    for s in ens.samples[:30]:
        axs[0].plot(s[:, 0] - o[0], s[:, 1] - o[1], color="#9ab", lw=0.5)
        axs[1].plot(np.hypot(s[:, 0] - o[0], s[:, 1] - o[1]), s[:, 2], color="#9ab", lw=0.5)
    k = int(np.searchsorted(ens.md, ens.drilled_depth))
    for a, xy, lab in ((axs[0], (ens.mean_xyz[:, 0] - o[0], ens.mean_xyz[:, 1] - o[1]), "plan"),
                       (axs[1], (np.hypot(ens.mean_xyz[:, 0] - o[0], ens.mean_xyz[:, 1] - o[1]), ens.mean_xyz[:, 2]),
                        "section")):
        a.plot(xy[0][:k], xy[1][:k], color="#c0392b", lw=2.2, label="drilled (surveys)")
        a.plot(xy[0][k:], xy[1][k:], color="#1f4e79", lw=1.6, ls="--", label="predicted (mean path)")
    az, dp = math.radians(300), math.radians(-72)
    dd = np.array([0, ens.planned_depth])
    axs[0].plot(dd * math.cos(dp) * math.sin(az), dd * math.cos(dp) * math.cos(az), color="#27ae60", lw=1,
                label="straight design")
    axs[1].plot(dd * math.cos(dp), o[2] + dd * math.sin(dp), color="#27ae60", lw=1, label="straight design")
    axs[0].set_title("Plan (m from collar)")
    axs[1].set_title("Vertical section along the hole")
    axs[0].set_aspect("equal")
    axs[1].set_xlabel("horizontal distance (m)")
    axs[1].set_ylabel("elevation (m)")
    axs[0].legend(fontsize=7, loc="lower left")
    p_ = ASSETS / "fig_trajectory.png"
    f.tight_layout()
    f.savefig(p_, dpi=160)
    plt.close(f)
    out["trajectory"] = p_

    # hazard and survival along DH-2026-001
    md = pred.md
    m = md <= pred.planned_depth
    f, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for fam, h in pred.h.items():
        axs[0].plot(md[m], h[m], color=codes.FAMILY_COLORS[fam], label=codes.FAMILIES[fam])
    axs[0].axvline(pred.drilled_depth, color="k", lw=1)
    axs[0].text(pred.drilled_depth, axs[0].get_ylim()[1] * 0.95, " bit", fontsize=8)
    axs[0].set_title("Hazard per metre h (Model A, mean of trial paths)")
    axs[0].set_xlabel("depth along the hole (m)")
    axs[0].legend(fontsize=7)
    tot = sum(pred.h.values())
    ahead = (md > pred.drilled_depth) & m
    surv = np.cumprod(1 - np.where(ahead, np.clip(tot, 0, 0.99), 0.0))
    axs[1].plot(md[m], surv[m], color="#1f4e79")
    axs[1].set_ylim(min(0.8, float(surv[m].min()) - 0.02), 1.005)
    axs[1].set_title("Survival from the bit: product of (1 - h)")
    axs[1].set_xlabel("depth along the hole (m)")
    axs[1].axhline(pred.p_reach, color="#c0392b", lw=0.8, ls=":")
    axs[1].text(md[m][0] + 5, pred.p_reach, f" P(reach target) ≈ {pred.p_reach:.2f}", va="bottom", fontsize=8)
    f.tight_layout()
    p_ = ASSETS / "fig_hazard.png"
    f.savefig(p_, dpi=160)
    plt.close(f)
    out["hazard"] = p_

    # level index mapping
    f, ax = plt.subplots(figsize=(6.2, 3.3))
    y, o_, r = ms.thresholds["karst"]
    xs = np.array([0, y, o_, r, max(2 * r, r + 0.2), 1.0])
    s = np.linspace(0, max(2.5 * r, 0.03), 400)
    ax.plot(s, np.interp(s, xs, [0, .15, .35, .6, .85, 1]), color="#1f4e79")
    for lv, a, b in (("green", 0, .15), ("yellow", .15, .35), ("orange", .35, .6), ("red", .6, 1)):
        ax.axhspan(a, b, color=codes.LEVEL_COLORS[lv], alpha=0.18)
    for v, lab in ((y, "yellow"), (o_, "orange"), (r, "red")):
        ax.axvline(v, color="#555", lw=0.6, ls=":")
        ax.text(v, 0.02, f" {lab}\n {v:.4f}", fontsize=7)
    ax.set_xlabel("risk score S (karst family, tuned thresholds)")
    ax.set_ylabel("risk index P")
    ax.set_ylim(0, 1)
    f.tight_layout()
    p_ = ASSETS / "fig_levelmap.png"
    f.savefig(p_, dpi=160)
    plt.close(f)
    out["levelmap"] = p_

    # rock properties: UCS vs CAI
    f, ax = plt.subplots(figsize=(9, 5.2))
    groups = {"clastic": "#d4a373", "carbonate": "#3d8bfd", "chemical": "#8e44ad", "metamorphic": "#16a085",
              "igneous": "#c0392b", "ore": "#7f8c8d", "fault": "#e67e22", "weathered": "#95a5a6"}
    for k_, pr in rockmech.DEFAULTS.items():
        ax.errorbar(pr.cai[1], pr.ucs[1], xerr=[[pr.cai[1] - pr.cai[0]], [pr.cai[2] - pr.cai[1]]],
                    yerr=[[pr.ucs[1] - pr.ucs[0]], [pr.ucs[2] - pr.ucs[1]]], fmt="o", ms=4,
                    color=groups[pr.group], alpha=0.75, lw=0.5, capsize=0)
        ax.annotate(k_, (pr.cai[1], pr.ucs[1]), fontsize=5.8, xytext=(2, 2), textcoords="offset points")
    for v in (25, 50, 100, 250):
        ax.axhline(v, color="#999", lw=0.5, ls=":")
    for v in (1, 2, 3, 4):
        ax.axvline(v, color="#999", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Cerchar abrasivity index CAI (typical, with range)")
    ax.set_ylabel("UCS (MPa, typical, with range)")
    for g, c in groups.items():
        ax.plot([], [], "o", color=c, label=g)
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    f.tight_layout()
    p_ = ASSETS / "fig_rocks.png"
    f.savefig(p_, dpi=170)
    plt.close(f)
    out["rocks"] = p_

    # validation by family
    v = ms.validation or {}
    fams = list(codes.FAMILIES)
    f, ax = plt.subplots(figsize=(7.5, 3.3))
    x = np.arange(len(fams))
    auc = [((v.get("auc_family") or {}).get(fm) or 0) for fm in fams]
    rec = [((v.get("recall_orange") or {}).get(fm) or 0) for fm in fams]
    ax.bar(x - 0.18, auc, 0.36, label="ROC-AUC (metre level, held-out areas)", color="#1f4e79")
    ax.bar(x + 0.18, rec, 0.36, label="recall at Orange, >= 3 m ahead", color="#e67e22")
    ax.axhline(0.75, color="#1f4e79", lw=0.7, ls="--")
    ax.axhline(0.70, color="#e67e22", lw=0.7, ls=":")
    ax.set_xticks(x, [codes.FAMILIES[fm] for fm in fams])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")
    f.tight_layout()
    p_ = ASSETS / "fig_validation.png"
    f.savefig(p_, dpi=160)
    plt.close(f)
    out["validation"] = p_
    return out


def screenshots(engine) -> dict:
    """The windows, rendered offscreen with the Windows fonts (English)."""
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from dfpredict.ui.history_dialog import DrillingRecordsDialog
    from dfpredict.ui.prediction_dialog import PredictionDialog
    from dfpredict.ui.risk_admin import ModelsDialog

    app = QApplication.instance()
    app.setFont(QFont("Segoe UI", 9))
    out = {}
    dlg = PredictionDialog(engine)
    dlg.resize(1600, 980)
    dlg.show()
    p_ = engine.predict("DH-2026-001", n_paths=30)
    dlg.set_prediction(p_)
    app.processEvents()
    out["predict"] = ASSETS / "shot_prediction.png"
    dlg.grab().save(str(out["predict"]))
    dlg.mode.setCurrentIndex(1)
    dlg._design_from_hole()
    app.processEvents()
    out["plan"] = ASSETS / "shot_plan.png"
    dlg.mode.grab().save(str(out["plan"]))
    dlg.close()
    rec = DrillingRecordsDialog(engine.target, engine.folder)
    rec.resize(1600, 950)
    rec.show()
    rec.reload()
    rec.load_hole("DH-2026-001", force=True)
    rec.sub.setCurrentIndex(2)
    app.processEvents()
    out["records"] = ASSETS / "shot_records.png"
    rec.grab().save(str(out["records"]))
    rec.tabs.setCurrentIndex(4)
    app.processEvents()
    out["rock"] = ASSETS / "shot_rock.png"
    rec.grab().save(str(out["rock"]))
    rec.close()
    md = ModelsDialog(engine)
    md.resize(1080, 760)
    md.show()
    app.processEvents()
    out["models"] = ASSETS / "shot_models.png"
    md.grab().save(str(out["models"]))
    md.close()
    # 3D view
    try:
        import numpy as np
        import pyvista as pv

        from dfpredict.scene.loaders import load_paths
        from dfpredict.scene.risk_layer import RiskLayer
        from dfpredict.scene.scene import Scene

        pl = pv.Plotter(off_screen=True, window_size=(1500, 1000))
        scene = Scene(pl)
        files = sorted((WS / "model").glob("horizon_*.dtm")) + sorted((WS / "faults").glob("F4.dtm"))
        layers, _ = load_paths(files)
        cols = [(0.93, 0.80, 0.45), (0.35, 0.35, 0.40), (0.30, 0.65, 0.95), (0.62, 0.78, 0.45), (0.80, 0.55, 0.85)]
        for i, lay in enumerate(layers):
            lay.use_styles = False
            lay.color, lay.opacity = ((0.9, 0.2, 0.2), 0.35) if lay.name == "F4" else (cols[i % 5], 0.12)
            scene.add_layer(lay)
        rl = RiskLayer(name="Drilling risk", path=None)
        rl.predictions = {"DH-2026-001": p_}
        zi = max(range(len(p_.zones)), key=lambda i: p_.zones[i].p)
        rl.selected = ("DH-2026-001", zi)
        scene.add_layer(rl)
        pl.set_background("#15171b")
        pl.render()
        pl.camera.zoom(6.0)
        out["view3d"] = ASSETS / "shot_3d.png"
        pl.screenshot(str(out["view3d"]))
        pl.close()
        del np
    except Exception as e:  # noqa: BLE001
        print("3D screenshot skipped:", e)
    return out


# ------------------------------------------------------------------------------------------------
# the document
# ------------------------------------------------------------------------------------------------
def build(no_shots: bool = False) -> Path:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    ASSETS.mkdir(parents=True, exist_ok=True)
    from dfpredict import __version__
    from dfpredict.io.database import Target
    from dfpredict.risk import codes, features, models, rockmech, rules
    from dfpredict.risk import geomodel as gm
    from dfpredict.risk import measures as meas
    from dfpredict.risk import predict as pr
    from dfpredict.risk import trajectory as trj
    from dfpredict.risk.engine import RiskEngine
    from dfpredict.risk.geomodel import surfaces_from_files
    from dfpredict.risk.schema import ALL_TABLES, HOLE_TABLES, REFERENCE_TABLES

    print("predicting the demo holes ...")
    engine = RiskEngine(Target("sqlite", path=WS / "geodatabase.sqlite"), WS)
    prs = [(q, "horizon") for q in sorted((WS / "model").glob("horizon_*.dtm"))]
    prs += [(q, "fault") for q in sorted((WS / "faults").glob("F*.dtm"))]
    engine.set_surfaces(surfaces_from_files(prs))
    ms = engine.models
    p1 = engine.predict("DH-2026-001", n_paths=40)
    p2 = engine.predict("DH-2026-002", n_paths=30)
    p3 = engine.predict("DH-2026-003", n_paths=30)
    print("figures ...")
    F = figures(engine, p1, ms)
    S = {} if no_shots else screenshots(engine)
    v = ms.validation or {}
    t = ms.trained_on

    parts: list[str] = []
    A = parts.append

    # --- title ---
    A(f"""<br><br><br><br><br><p align='center'><span style='font-size:26pt; font-weight:bold; color:#1f4e79'>Drilling Failure
Prediction &amp; Prevention</span></p>
<p align='center'><span style='font-size:18pt'>Algorithm &amp; Manual</span></p>
<p align='center' style='margin-top:40px'>Mathematical principles · data preparation · rock property calibration ·
model training and validation · prediction and measures for ongoing and planned drillholes · operation</p>
<p align='center' style='margin-top:60px'>Software version {__version__} · models {E(ms.version)} ·
built on the GeoViewer platform</p>
<p align='center' style='margin-top:200px; color:#555'>Implements <i>Drilling Failure Prediction &amp; Prevention System —
Design Spec</i> (docs folder). Section references “spec x.y” point to that document.</p>""")

    # --- contents ---
    A(h1("Contents"))
    A(ol(["Overview of the system", "Mathematical principles", "Data preparation and construction",
          "Rock property calibration", "Model training and validation",
          "Predicting failure probability and suggesting measures (ongoing and planned drillholes)",
          "Operation manual", "Appendices: code lists and libraries"]))

    # =========================================================================================
    A(h1("1. Overview of the system"))
    A(p("The system predicts <b>where</b> and <b>why</b> a drillhole is likely to fail and gives the driller a fixed set "
        "of countermeasures. It learns from past holes drilled in many areas (successful and failed) and applies what "
        "it learned to a new area, where an approximate geological model exists. The past areas share rock types with "
        "the new area, but not formations, and formation thickness changes strongly even inside one area. Lessons "
        "therefore transfer only through what the rock is (its mechanical and karst properties) and where it sits "
        "relative to its contacts - never through formation names or coordinates."))
    A(fig(F["architecture"], "Data flow: geological model, live hole and history feed the feature engine; the models, "
                             "rules and evidence produce levels, zones, reasons and measures.", 640))
    A(p("The system asks two linked questions (spec 1.1): first, will the hole meet a <b>hazard</b> here (a cavity, a "
        "fault zone, hard or broken rock)? Model H answers it. Second, <b>will it fail</b> here under these drilling "
        "conditions? Model A answers it ahead of the bit, Model B while drilling. Fixed rules R1 - R8 catch clear "
        "physical signs even when a model is wrong. All of them combine into one alarm level per metre."))
    A(table(["Failure family", "Failure codes", "Main causes in this terrain"],
            [[codes.FAMILIES[f], ", ".join(f"{c} {codes.FAIL_CODES[c][0]}" for c in codes.FAMILY_CODES[f]),
              {"karst": "karst cavities and open fractures in carbonate, fault damage zones, zones below unconformities",
               "instability": "fault gouge and breccia, weak and swelling rock, long open hole, thin fluid",
               "trajectory": "low angle between hole and bedding / foliation, strong-weak bed boundaries, faults",
               "bit": "strong fine-grained rock (glazing), abrasive rock (wear), doglegs and vibration"}[f]]
             for f in codes.FAMILIES], [18, 42, 40], "The four failure families and their codes (spec 2)."))
    A(p("The application is the GeoViewer desktop program (Python, PySide6, VTK/PyVista) extended with the drilling "
        "risk functions. All records live in the workspace's project database (SQLite file <i>geodatabase.sqlite</i> "
        "in the workspace folder, or a MySQL database), in the drillhole tables <i>drillhole_*</i> that GeoViewer and "
        "3D MPM Studio use too (one database for all three programs; the risk system's own data in <i>risk_rock</i>, "
        "<i>risk_rock_code</i> and <i>risk_warning</i>). The geological model is read "
        "from the workspace's surface layers by their role (after 3D MPM Studio)."))

    # =========================================================================================
    A(h1("2. Mathematical principles"))
    A(h2("2.1 Desurvey: the minimum-curvature method"))
    A(p("Survey stations give depth along the hole <i>s</i>, azimuth <i>a</i> (clockwise from north) and dip <i>d</i> "
        "(negative downward). The unit direction at a station is"))
    A(eq(r"\mathbf{t} = (\cos d\,\sin a,\ \cos d\,\cos a,\ \sin d)", "dir"))
    A(p("Between stations 1 and 2 the hole is a circular arc. With the dogleg angle β and the ratio factor RF:"))
    A(eq(r"\beta = \arccos(\mathbf{t}_1\cdot\mathbf{t}_2),\qquad RF = \frac{2}{\beta}\tan\frac{\beta}{2}\ (RF=1\ \mathrm{for}\ \beta\to 0)", "dogleg"))
    A(eq(r"\mathbf{x}_2 = \mathbf{x}_1 + \frac{\Delta s}{2}\,RF\,(\mathbf{t}_1+\mathbf{t}_2)", "mincurv"))
    A(p("Positions between stations use the same arc (spherical interpolation of the tangent). The dogleg severity in "
        "degrees per 30 m is DLS = β·30/Δs; it feeds the deviation features and rule R5."))

    A(h2("2.2 Path ahead of the bit"))
    A(p("For an ongoing hole the drilled part is the desurveyed path. Ahead of the bit the path continues from the last "
        "station. The build rate <i>b</i> (dip change) and turn rate <i>r</i> (azimuth change) are fitted to the last "
        "four stations by weighted least squares (weights rising linearly from 0.5 to 1 toward the latest station), "
        "in degrees per 30 m, and fade with distance ahead of the bit:"))
    A(eq(r"\Delta d_k = b\,e^{-\ell_k/L}\,\frac{\Delta s}{30} + \varepsilon_d,\qquad \Delta a_k = r\,e^{-\ell_k/L}\,\frac{\Delta s}{30} + \varepsilon_a,\qquad L = %d\ \mathrm{m}" % trj.FADE, "buildturn"))
    A(p(f"ℓ is the distance ahead of the bit. The random terms ε are independent per 1 m step with standard deviations "
        f"σ<sub>d</sub>·√(Δs/30) and σ<sub>a</sub>·√(Δs/30), where σ is the robust spread (1.4826 × median absolute "
        f"deviation) of the dip / azimuth change per 30 m in the surveys of all past holes (default "
        f"{trj.DEFAULT_SPREAD['dip']}° and {trj.DEFAULT_SPREAD['azimuth']}° per 30 m). Each trial path also draws its "
        f"own build and turn rate around the fitted ones (σ 0.6°/30 m with four or more stations, else 1.0°)."))
    A(p("Where the hole meets bedding or foliation at a low angle, the bit walks toward perpendicular to it (spec 2, "
        "DV precursors). Every 5 m the direction <b>t</b> is turned toward the projection of the bedding normal "
        "<b>n</b> (oriented along the hole) by"))
    A(eq(r"\delta = W\,\frac{5}{30}\cos\alpha\quad(\alpha < 60^\circ),\qquad W = %.1f^\circ/30\,\mathrm{m}" % trj.WALK, "walk"))
    A(p("where α is the incidence angle (section 2.3). The ensemble of N trial paths (default 40) plus the mean path "
        "(no random terms) runs to the planned depth + 60 m. The spread of the trial paths around the mean path is "
        "the uncertainty cone shown in the 3D view. A planned hole starts from the designed collar azimuth and dip."))
    A(fig(F["trajectory"], "Predicted path of DH-2026-001: drilled part (red), mean predicted path (dashed), 30 of the "
                           "trial paths (grey) and the straight design (green).", 640))

    A(h2("2.3 Geological model: units, contacts, bedding, faults"))
    A(p(f"Every horizon / unconformity surface is gridded once on a {gm.GRID:.0f} m plan grid (linear inside its "
        "triangles, nearest value outside its footprint); elevations are read by bilinear interpolation, z<sub>h</sub>(x, y). "
        "With the H contacts ordered top to bottom, the unit of a point (x, y, z) is the number of contacts above it:"))
    A(eq(r"k(x,y,z) = \sum_{h=1}^{H} \mathbf{1}\left[z_h(x,y) + \delta_h > z\right]", "unit"))
    A(p("δ<sub>h</sub> is the model offset of the trial (section 2.4). Formation solids (closed meshes) are converted "
        "to 5 × 5 × 2 m voxels and win over the horizons where they exist. The bedding normal comes from the gradient "
        "of the contact below the point's unit, and the incidence angle between the hole and the bedding is"))
    A(eq(r"\mathbf{n} = \frac{(-\partial z_h/\partial x,\ -\partial z_h/\partial y,\ 1)}{\left|\cdot\right|},\qquad \alpha = \arcsin\left|\mathbf{t}\cdot\mathbf{n}\right|", "incidence"))
    A(p("α = 90° means the hole crosses the bedding at right angles; below about 30° deviation and bit walk become "
        "likely (spec 2). Fault surfaces give the signed distance f(<b>x</b>) of every path point (VTK implicit "
        "distance). A crossing lies between consecutive points k, k+1 with opposite signs, both closer than 3 m, at"))
    A(eq(r"s_c = s_k + \frac{f_k}{f_k - f_{k+1}}\,(s_{k+1}-s_k)", "crossing"))
    A(p(f"and the fault damage zone is taken as s<sub>c</sub> ± w (w = {gm.DEFAULTS['fault_zone']:.0f} m by default, "
        "Geological Model window)."))

    A(h2("2.4 The error band ±u of the geological model"))
    A(p("The model is built from surface mapping, geophysics and few holes, so contacts and faults may lie away from "
        "where it shows them (spec 5.6). The error band at a point grows with the plan distance D to the nearest "
        "model control (the holes the model was built from) and below every unit whose thickness varies widely:"))
    A(eq(r"u_0 = u_{near} + (u_{far}-u_{near})\,\min\left(1,\frac{D}{D_{far}}\right),\qquad u = \sqrt{u_0^2 + \sum_{j\ \mathrm{above}} \left[c\,(T^{max}_j - T^{min}_j)\right]^2}", "uband"))
    A(p(f"with defaults u<sub>near</sub> = {gm.DEFAULTS['u_near']:.0f} m, u<sub>far</sub> = {gm.DEFAULTS['u_far']:.0f} m, "
        f"D<sub>far</sub> = {gm.DEFAULTS['d_far']:.0f} m and c = {gm.DEFAULTS['thickness_factor']}; T<sup>min</sup>, "
        "T<sup>max</sup> are the thickness range of the unit's formation card. Each trial path j draws one standard "
        "normal number e<sub>j</sub> and shifts all contacts by δ = e<sub>j</sub>·u (correlated shift), so the "
        "predictions are averaged over the model error as well as over the path."))
    A(p("<b>Tie to reality while drilling</b> (spec 5.6.4). When the logged geology of the hole crosses a contact "
        "between two model units, the along-hole offset Δs = s<sub>logged</sub> − s<sub>model</sub> becomes a vertical "
        "shift of the contacts below:"))
    A(eq(r"\delta_{h+1} = -\Delta s\,|t_z|,\qquad \delta_{h+2\ldots H} = -0.5\,\Delta s\,|t_z|,\qquad u \leftarrow u\cdot\mathrm{clip}\left(0.4 + \frac{s - s_{tie}}{150},\ 0.4,\ 1\right)", "tie"))
    A(p("If the logged unit at the bit differs from the model unit there, a mismatch flag (R-M01) widens u ahead by "
        "50 %."))

    A(h2("2.5 Failure as a hazard per metre"))
    A(p("A failed hole stops at its failure depth; a successful hole reaches its final depth without failing. A "
        "discrete-time hazard model handles both (spec 5.2): h<sub>f,i</sub> is the chance that the hole fails in metre "
        "i by family f, given that it has not failed above. The chance of reaching the target from the bit is"))
    A(eq(r"P_{reach} = \frac{1}{N}\sum_{j=1}^{N}\ \prod_{i\ \in\ (s_{bit},\,s_{target}]} \left(1 - \sum_f h_{f,i}^{(j)}\right)", "preach"))
    A(p("averaged over the trial paths j. The alarm score of a family at metre i is the chance of failure within the "
        "10 m around it, times the family's severity weight (mean severity of its past failures / 5):"))
    A(eq(r"S_{f,i} = w_f\left[1 - \prod_{k=i-5}^{i+4}\left(1 - \bar h_{f,k}\right)\right],\qquad w_f = \frac{\overline{\mathrm{severity}}_f}{5}", "score"))
    A(fig(F["hazard"], "DH-2026-001: hazard per metre of each family and the survival from the bit to the target.", 640))

    A(h2("2.6 Gradient-boosted trees, class weights and SHAP reasons"))
    A(p("Models H, A and B are gradient-boosted decision trees (LightGBM). The log-odds is a sum of M regression trees "
        "f<sub>m</sub>, fitted one after another to the gradient of the binary log-loss:"))
    A(eq(r"F(\mathbf{x}) = \sum_{m=1}^{M} \eta\, f_m(\mathbf{x}),\qquad p(\mathbf{x}) = \frac{1}{1+e^{-F(\mathbf{x})}}", "gbm"))
    A(p("Failures are rare (one positive metre per failed hole among hundreds of thousands). To keep training fast "
        "without biasing the probabilities, only a share q of the label-0 rows is kept and each kept row gets weight "
        "1/q; the weighted log-loss is then an unbiased estimate of the full one, so p stays a calibrated hazard."))
    A(eq(r"\mathcal{L} = \sum_{y_i=1} \ell(1,p_i) + \frac{1}{q}\sum_{y_i=0,\ \mathrm{kept}} \ell(0,p_i)", "weights"))
    A(p("The reason codes come from SHAP values: for trees they are computed exactly (TreeSHAP, LightGBM "
        "<i>pred_contrib</i>) and add up to the model output, F(<b>x</b>) = φ<sub>0</sub> + Σ<sub>j</sub> φ<sub>j</sub>. "
        "The features with the largest positive φ over the zone are mapped to pre-defined messages (section 6.1)."))

    A(h2("2.7 Stacking, levels and corrections"))
    A(p("Model H's cavity and fault-zone probabilities, for the metre and as the maximum over the next 10 m, are "
        "features of Models A and B. During training they are <b>out-of-area</b> predictions (each area predicted by "
        "a Model H trained on the other areas), so A and B learn how much to trust H in a new area."))
    A(p("The score S is mapped to the risk index P whose level boundaries are the spec's starting levels "
        "(0.15 / 0.35 / 0.60) through each family's tuned thresholds (y, o, r) by piecewise-linear interpolation:"))
    A(eq(r"P = \mathrm{interp}\left(S;\ [0,\ y,\ o,\ r,\ \max(2r, r+0.2),\ 1] \rightarrow [0,\ 0.15,\ 0.35,\ 0.60,\ 0.85,\ 1]\right)", "levelmap"))
    A(fig(F["levelmap"], "Mapping of the karst family's risk score to the risk index and the alarm levels.", 430))
    A(p("<b>Area correction</b> (spec 5.10.4). After n finished holes in the new area, the observed failures O<sub>f</sub> "
        "are compared with the expected Σh of Model A on those holes, and each family's hazard is multiplied by"))
    A(eq(r"c_f = \mathrm{clip}\left(\left(\frac{O_f+0.5}{E_f+0.5}\right)^{n/(n+5)},\ \frac{1}{3},\ 3\right)", "areacorr"))
    A(p("<b>Danger over a depth range [a, b]</b>. Failure: 1 − Π(1 − Σ<sub>f</sub>h<sub>f</sub>) over the metres of the "
        "range. Model H gives the chance that a metre lies <i>in</i> a cavity or fault zone; one occurrence spans "
        f"several metres, so the chance of meeting one uses p/L with the typical lengths L<sub>cavity</sub> = "
        f"{pr.CAVITY_LENGTH:.0f} m and L<sub>fault zone</sub> = {pr.FAULT_ZONE_LENGTH:.0f} m. A model fault is "
        "crossed with the share of trial paths that cross it in the range."))
    A(h2("2.8 Validation measures"))
    A(p("<b>Concordance (hole level).</b> For every failed hole i (failure at depth D<sub>i</sub>) and every hole j still "
        "drilling below D<sub>i</sub>, the pair is concordant when the cumulative hazard of i down to D<sub>i</sub> is "
        "larger than that of j down to the same depth (ties count ½):"))
    A(eq(r"C = \frac{\sum_{i\ \mathrm{failed}}\sum_{j:\ s_j^{end} > D_i} \mathbf{1}\left[H_i(D_i) > H_j(D_i)\right] + \frac{1}{2}\mathbf{1}\left[H_i(D_i) = H_j(D_i)\right]}{\#\ \mathrm{pairs}},\qquad H(D)=\sum_{s\leq D} h", "cindex"))
    A(p("It replaces a plain AUC of 'fails before target depth', which would be biased because failed holes are shorter. "
        "Metre-level ROC-AUC, recall at Orange, lead distance and false alarms are defined in section 5.3."))

    # =========================================================================================
    A(h1("3. Data preparation and construction"))
    A(p("This chapter explains what data the system needs, how to prepare it, where it is stored and how the per-metre "
        "training table is built from it. Good data preparation matters more than any model setting: the models can "
        "only learn what the records show."))
    A(h2("3.1 Where the data lives"))
    A(p("All records are tables of the workspace's project database (Settings → Database): the SQLite file "
        "<i>geodatabase.sqlite</i> in the workspace folder (default; nothing to install) or one MySQL database shared "
        "by several users. The tables are made automatically when a workspace is opened or the database connected; "
        "existing tables only get missing columns added - nothing is ever dropped."))
    A(table(["Table", "Content", "Spec"],
            [[tb.sql_name, tb.title, {"area": "5.10", "formation": "3.5", "rig": "3.1 I", "rock": "-",
                                            "rock_code": "3.4", "measure": "6.1", "collar": "3.1 A", "survey": "3.1 B",
                                            "geology": "3.1 C", "hole_size": "3.1 D", "bit": "3.1 E", "run": "3.1 F / 3.3",
                                            "failure": "3.1 G", "occurrence": "3.1 H", "warning": "7.4"}.get(tb.name, "")]
             for tb in ALL_TABLES], [22, 60, 18], "The drilling-record tables."))
    A(p("<b>Conventions</b> (spec 3.4): depths are measured <b>along the hole</b> from the collar (never true vertical "
        "depth); dip is negative downward (−90 = vertical down), azimuth clockwise from north; one projected coordinate "
        "system and one vertical datum for holes and model; <b>unknown values stay blank</b> - never 0; code lists are "
        "controlled (failure codes, occurrence types, bit types, hole sizes, fluids, rock codes, measures); free text "
        "goes into the notes fields."))
    A(p("<b>Shared with GeoViewer.</b> The drillhole tables are the same as GeoViewer's and 3D MPM Studio's "
        "(<i>io/drilling_schema.py</i>, an identical copy in each program), stored as they are there: dip in degrees "
        "from straight down (0 = down), <i>max_depth</i> for the final / current depth and the hole's <i>state</i> "
        "(Planned, In work, Completed with ore hit, Completed without ore hit, Failed) for the result. The Drilling Records "
        "and the CSV files show them as above (negative-downward dip, final depth, result); the conversion is done "
        "when reading and writing. Saving a hole here keeps what only GeoViewer records (organization, assays, core "
        "notes ...). A database of an earlier version (tables <i>risk_collar</i>, <i>risk_run</i> ...) is moved "
        "into the drillhole tables once; the old tables are kept as <i>risk_*_old</i>."))
    A(h2("3.2 Predefined reference data"))
    A(p("When a database is set up, the reference tables are filled with the predefined data (only while a table is "
        "empty, so edits are never replaced): the rock property table (46 rock classes, chapter 4), the rock-code "
        "mapping (66 codes), the measure library (37 measures, appendix) and two generic example rigs. In Drilling "
        "Records each of these tabs has <i>Restore defaults</i> (add the missing rows, or replace the table)."))

    A(h2("3.3 Reference tables to fill in once"))
    for tb in REFERENCE_TABLES:
        if tb.name in ("rock",):
            continue
        rows = [[f"<b>{c.name}</b>", E(c.label) + (" *" if c.required else ""), E(c.unit),
                 E(c.tip) if c.tip else ("; ".join(f"{k} = {E(v)}" for k, v in list(c.choices.items())[:6])
                                         if c.choices else "")]
                for c in tb.cols]
        A(h3(f"{tb.title} ({tb.sql_name})"))
        if tb.help:
            A(p(E(tb.help)))
        A(table(["Field", "Label (* required)", "Unit", "Meaning / allowed values"], rows, [16, 26, 9, 49],
                small=True))
    A(p("<b>Site sheet (areas).</b> A geologist fills one row per drilling location, old or new, before drilling starts: "
        "the bedding dip range, fold tightness, faults per km, whether carbonate is exposed, whether an unconformity "
        "lies above the carbonate, whether sinkholes are mapped, the height of the carbonate above the local valley "
        "floor and the regional water table (if known). These few fields let the models tell a strongly karstified "
        "area from a weakly karstified one with the same rocks (spec 5.10). Keep them to these fields: the models learn "
        "each field only from as many examples as there are areas."))
    A(p("<b>Formation cards.</b> One card per formation, reused in every area where it occurs; give the formation the "
        "same name everywhere, also in the geological model. The card describes the formation by what transfers: age, "
        "main and secondary rock (rock classes of chapter 4), carbonate share, bedding style, swelling clay, known karst, "
        "thickness range, equivalent formations in other areas, and - where measured - the laboratory UCS and CAI. "
        "A formation the models have never seen is predicted from its card."))
    A(p("<b>Rigs.</b> The rated depth per hole size, maximum RPM, pump capacity, rod pull limit and torque turn a rig "
        "model into 'how close to its limit' (feature depth / rated depth, entry check of RPM)."))

    A(h2("3.4 Hole records"))
    A(p("Every hole - past, ongoing or planned - has a collar row and rows in the per-hole tables. The same fields are "
        "used for past and live holes, so a finished hole becomes training data without conversion (spec 7)."))
    for tb in (ALL_TABLES[6],) + HOLE_TABLES:
        rows = [[f"<b>{c.name}</b>", E(c.label) + (" *" if c.required else ""), E(c.unit),
                 E(c.tip) if c.tip else ("; ".join(f"{k} = {E(str(v))}" for k, v in list(c.choices.items())[:8])
                                         if c.choices else "")]
                for c in tb.cols if c.name != "hole_id"]
        A(h3(f"{tb.title} ({tb.sql_name})"))
        if tb.help:
            A(p(E(tb.help)))
        A(table(["Field", "Label (* required)", "Unit", "Meaning / allowed values"], rows, [16, 26, 9, 49],
                small=True))
    A(h3("How to code the history"))
    A(ul([
        "<b>Result.</b> success = reached its final depth; failed = abandoned (or the planned target given up) because "
        "of a problem; in progress = being drilled (final depth = current bit depth); planned = not started.",
        "<b>Failure</b> (one row per failed hole): the depth where the hole was lost or abandoned, the failure code of "
        "the cause (map your own fail codes once to the ten codes of the appendix), the severity 1 - 5 and the measures "
        "tried, in the order they were tried.",
        "<b>Occurrences</b>: every cavity and every fault zone met, also in successful holes (they train Model H), and "
        "every other problem with the measures used and whether it was solved. The <b>last</b> measure of a solved "
        "occurrence is counted as the one that solved it; a measure of an unsolved occurrence or of a failure counts as "
        "not solved. A cavity met without problems is recorded with no measures.",
        "<b>Geology</b>: the logged rock code (mapped to a rock class by the rock-code table) and the formation name "
        "(its card). Log the bedding angle to the core axis where possible - it is the incidence angle of the past "
        "hole; without it the hole inclination is used (flat bedding). Mark broken zones.",
        "<b>Hole size and bit</b>: one row per interval; each size reduction normally marks a casing point. "
        "Do not move a size reduction made <i>because</i> of a failure before the failure depth (leakage, spec 5.9).",
        "<b>Runs</b>: one row per core run (1.5 - 6 m). Per-run values are needed for the live warnings (rules R1 - R4, "
        "Model B); per-shift or per-hole values support only the pre-drill risk. Record return %, rod drop, bit load "
        "and torque from now on (spec 3.3): rules R6 - R8 use them at once, the models once about 100 holes have them.",
        "<b>Free text measures</b> of old reports are mapped once to the measure codes while digitising."]))
    A(p("Example rows (CSV, spec 3.1):"))
    A("<pre style='font-size:8pt; background-color:#f4f6f8'>"
      "run.csv\nhole_id,depth_from,depth_to,rpm,rop,flow_in,pump_press,return_pct,rod_drop\n"
      "DH-0731,412.0,415.0,750,3.3,38,2.1,96,\nDH-0731,415.0,418.0,760,4.5,38,1.6,80,\n"
      "DH-0731,418.0,419.6,700,9.6,38,0.4,0,0.8\n\nfailure.csv\nhole_id,fail_depth,fail_code,severity,measures\n"
      "DH-0731,419.6,LC,4,M-LC-03;M-LC-06\n\noccurrence.csv\nhole_id,depth_from,depth_to,type,measures,solved\n"
      "DH-0731,418.5,419.6,cavity,M-LC-03;M-LC-06,N\nDH-0688,233.0,236.2,cavity,M-CV-02,Y\n"
      "DH-0688,301.5,309.0,fault,M-FL-01;M-CS-01,Y</pre>")
    A(h2("3.5 Ways to enter and import data"))
    A(ul([
        "<b>Drilling Records window</b> (Ctrl+H, chapter 7): hole by hole, with forms, drop-down code lists, "
        "<i>Next run</i>, <i>Record event</i>, <i>Record failure</i> and the entry checks.",
        "<b>CSV import</b> (Drilling Records → Import CSV): a folder with one UTF-8 file per table named "
        "<i>collar.csv, survey.csv, geology.csv, hole_size.csv, bit.csv, run.csv, occurrence.csv, failure.csv, "
        "area.csv, formation.csv, rig.csv, rock.csv, rock_code.csv, measure.csv</i>. Column names are matched ignoring "
        "case and spaces, with aliases (HOLE_ID / BHID, FROM / TO, ROCK_CODE / LITH, FLOW, PRESSURE, RETURN, STATUS "
        "...); code lists accept the code or its text (e.g. 'Success', 'Yes'). <i>Merge</i>: a hole in the files gets "
        "their collar values and its rows replaced in each file with rows for it; <i>Replace areas</i> also removes "
        "the holes of the areas in collar.csv that are in none of the files. Rows failing the entry checks are skipped "
        "and listed.",
        "<b>Holes without an area</b>: the drillholes entered in the Drillhole Manager are the same records (the "
        "drillhole tables are shared); this gives those without an area their area and, where missing, a result.",
        "<b>Demo history</b>: <i>tools\\make_demo_workspace.py</i> generates a process-based synthetic history "
        "(chapter 5.4) - for testing only."]))
    A(h2("3.6 Entry checks (spec 7.3)"))
    A(table(["Check", "Rule", "Effect"], [
        ["Required fields", "fields marked * must be filled", "error - not saved"],
        ["Physical ranges", "each number within its limits (e.g. return 0 - 120 %, dip −90 - 90°, UCS 0.01 - 600 MPa)",
         "error"],
        ["Codes", "code fields only take listed codes", "error"],
        ["From &lt; To", "every interval", "error"],
        ["Continuity", "a run's From must equal the previous run's To (no gap, no overlap)", "error"],
        ["Overlap", "geology / size / bit intervals must not overlap", "error"],
        ["Rig limits", "RPM not above the rig's maximum", "error"],
        ["Depth", "intervals / failure below the hole's final depth", "warning"],
        ["Consistency", "a loss noted with 100 % return; a failure but result not Failed; Failed without failure row",
         "warning / error"],
        ["Cards", "formation without a card; area without a site sheet", "warning"]],
        [18, 60, 22], "Entry checks run by Check and Save."))
    A(h2("3.7 Preparing the geological model"))
    A(ol([
        "Export the model as surfaces: one Surpac DTM (+STR), DXF or GOCAD TS file per object - horizons (contacts), "
        "unconformities, faults, topography, and formation solids if available.",
        "Open them in the workspace (or <i>Import geological model</i> / <i>Import fault surfaces</i>, which ask or "
        "set the role) and set each layer's <b>Role</b> (Properties panel or Geological Model window).",
        "In the Geological Model window, check the contacts are in stratigraphic order and name the units between them "
        "exactly as the formation cards (a status column shows whether each unit has a card). A horizon named like "
        "<i>horizon_H2_U2-U3</i> names the units above and below automatically.",
        "Set the error band: u near / far, the distance counted as far, the fault damage zone half width and the "
        "widening per variable unit, and the workspace's area (its site sheet). Model control = the collars of the "
        "workspace's drillholes and of the finished holes of the area.",
        "The settings are stored in <i>&lt;workspace&gt;\\drillrisk\\geomodel.json</i>; the roles in the workspace file."]))
    A(h2("3.8 Building the training table"))
    A(p("Every finished hole (success or failed) is resampled into 1 m rows from the collar to its final or failure "
        "depth (spec 5.1). For each row the features of section 3.9 are computed from the hole's own logs (past holes "
        "have no geological model) and the labels are set:"))
    A(ul([
        "<b>Model H</b>: cavity = 1 if the metre overlaps a cavity occurrence; fault = 1 if it overlaps a fault zone.",
        "<b>Model A</b>: 1 in the metre of the failure for its family, 0 in every metre above; rows below a failure are "
        "not used.",
        f"<b>Model B</b>: 1 in the last {models.PRECURSOR_L:.0f} m before the failure (the precursor window), "
        "for rows with run data."]))
    A(p("<b>Training with noise</b> (spec 5.6.1). The models must later run on the approximate model geology, so every "
        "past hole is also added a second time with noisy geology: each logged contact is shifted by N(0, 6 m) "
        "(kept inside its neighbours), units thinner than 6 m are dropped with probability 0.08 (formation missing in "
        "the model), the rock class is swapped with probability 0.05 by the mismatch table (dolomite ↔ limestone, "
        "sandstone ↔ quartzite, siltstone ↔ shale, marl → limestone), and the incidence angle gets N(0, 5°) noise."))
    A(p("<b>Leakage control</b> (spec 5.9). The geology log of a failed hole ends at the failure, so the base of the unit "
        "the hole stopped in is unknown: its distance-to-base, thickness, relative position and thickness anomaly are "
        "set to missing, and Models A and B use only the distance below the top of the unit. Size reductions and "
        "measures made because of a failure are not used for the metres above it."))
    A(h2("3.9 Features"))
    feats = features
    groups = [("Rock (card + mechanics)", feats.ROCK, "geology log + card + rock table", "model unit + card", "H A B"),
              ("Position in the unit", feats.POSITION, "geology log", "model along the path, ±u", "H (A, B: d_top)"),
              ("Contact setting", feats.CONTACT, "geology log", "model", "H A B"),
              ("Contact geometry", feats.GEOMETRY, "bedding angle in the log", "model surfaces", "H A B"),
              ("Faults", feats.FAULTS, "fault occurrences", "fault surfaces ± zone", "A B"),
              ("Hazard (stacked)", feats.HAZARD, "Model H (out of area)", "Model H", "A B"),
              ("Karst setting / site sheet", feats.KARST, "collar, area", "path, area", "H A B"),
              ("Hole state", feats.HOLE, "survey, hole size, rig", "plan / records", "A B"),
              ("Drilling setup", feats.SETUP, "bit, runs", "plan or last 3 runs", "A B"),
              ("Drilling response", feats.RESPONSE, "runs", "live runs only", "B")]
    A(table(["Group", "Features", "Past holes", "Ahead of the bit", "Models"],
            [[g, ", ".join(f"{f_} ({E(feats.FEATURE_TEXT.get(f_, ''))})" for f_ in fs), s1, s2, m_]
             for g, fs, s1, s2, m_ in groups], [14, 46, 14, 16, 10], "Features per metre (spec 5.3).", small=True))
    A(p("Coordinates and formation names are left out on purpose; what is learnt carries over to new areas through rock "
        "properties and position relative to contacts. The rock mechanics features come from the logged rock's class "
        "(chapter 4); the formation card adds the formation-level values and the lab UCS / CAI."))
    A(h2("3.10 How much data"))
    A(ul(["Past holes: the spec plans for 1,000+ holes with 10 - 20 % failed; the demo uses 1,304 finished holes.",
          f"A failure family gets a model of its own from {models.MIN_FAILURES} failures (spec: about 30); with fewer, "
          "the base rate and the rules carry it.",
          "Leave-one-area-out testing needs at least 5 - 8 areas.",
          "Run parameters per core run for the live warnings; the Overview tab shows the fill rate per crew."]))

    # =========================================================================================
    A(h1("4. Rock property calibration"))
    A(p("Most drilling failures are controlled by the mechanical behaviour of the rock: strong fine-grained rock glazes "
        "diamond bits, abrasive rock wears them, foliated rock and strong / weak bed boundaries deflect the hole, weak "
        "and swelling rock caves and sticks the rods, soluble rock hosts cavities. The system therefore describes every "
        "metre by the properties of its rock class, kept in the editable table <i>risk_rock</i>."))
    A(h2("4.1 The property table"))
    A(p("The predefined values are typical properties of fresh intact rock compiled from the standard references: "
        "Goodman (1989) <i>Introduction to Rock Mechanics</i>; Lama &amp; Vutukuri (1978) <i>Handbook on Mechanical "
        "Properties of Rocks</i>; Hoek (2007) <i>Practical Rock Engineering</i> (constant m<sub>i</sub>); ISRM suggested "
        "methods (1981) for the strength classes; ASTM D7625, Plinninger &amp; Restner (2008) and Käsling &amp; Thuro "
        "(2010) for the Cerchar abrasivity index; Thuro (1997) for the equivalent quartz content; Bruland (1998) and "
        "Dahl et al. (2012) for the NTNU drilling rate index. Ranges vary with grain size, cementation, weathering "
        "and fabric - replace them with your own laboratory values (section 4.4)."))
    rows = []
    for k, q in rockmech.DEFAULTS.items():
        rows.append([f"<b>{k}</b><br>{E(q.name)}", q.group, f"{q.ucs[0]:g}-<b>{q.ucs[1]:g}</b>-{q.ucs[2]:g}",
                     f"{q.tensile:g}", f"{q.e_mod:g}", f"{q.poisson:g}", f"{q.density:g}", f"{q.porosity:g}",
                     f"{q.cai[0]:g}-<b>{q.cai[1]:g}</b>-{q.cai[2]:g}", f"{q.quartz_eq:g}", f"{q.dri:g}", f"{q.mi:g}",
                     f"{q.carbonate:g}", f"{q.soluble}/{q.swelling}/{q.anisotropy}/{q.fractures}"])
    A(table(["Rock class", "Group", "UCS MPa", "σt MPa", "E GPa", "ν", "ρ g/cm³", "n %", "CAI", "Qeq %", "DRI", "mi",
             "Carb %", "S/Sw/A/F"], rows, [14, 9, 11, 5, 5, 5, 6, 5, 10, 5, 5, 4, 6, 10],
            "Predefined rock properties (min - typical - max for UCS and CAI). S/Sw/A/F = solubility / swelling / "
            "anisotropy / fracturing tendency, 0 none - 2 strong.", small=True))
    A(fig(F["rocks"], "Strength and abrasivity of the predefined rock classes (typical values and ranges); dotted lines "
                      "are the ISRM strength and CAI class boundaries.", 640))
    A(h2("4.2 Derived classes"))
    A(table(["Class", "Strength (ISRM, UCS)", "Abrasivity (CAI, ASTM D7625)"],
            [[str(c), rockmech.STRENGTH_NAMES[c], rockmech.ABRASIVITY_NAMES[c]] for c in range(1, 6)], [10, 45, 45],
            "Strength and abrasivity classes used as the features 'hard' and 'abr'."))
    A(h2("4.3 How the properties enter the features"))
    A(ol(["The logged rock code is mapped to a rock class (table <i>risk_rock_code</i>; codes not listed are matched by "
          "name, e.g. 'dolomite').",
          "The class gives UCS, CAI, Young's modulus, brittleness (UCS / tensile strength), anisotropy, solubility, "
          "porosity and density of the metre, and the strength / abrasivity classes.",
          "If the rock is the main rock of the formation card and the card has a lab UCS / CAI, those replace the "
          "class values. The card also gives carbonate share, karst, swelling clay, bedding style and thickness range.",
          "The strength contrast with the unit above, log10(UCS<sub>above</sub> / UCS), describes hard-over-soft "
          "contacts (deviation, rod failure)."]))
    A(h2("4.4 Calibrating with laboratory and drilling data"))
    A(ol([
        "<b>Map the rock codes.</b> Check that every rock code of your logs is in Drilling Records → Rock codes with the "
        "right class; add classes for rocks not listed (Rock properties → Add row) with their properties.",
        "<b>Laboratory tests per formation.</b> Uniaxial compressive strength (ISRM suggested method; at least 5 "
        "specimens per formation), or point-load index I<sub>s(50)</sub> converted with UCS ≈ k·I<sub>s(50)</sub>, "
        "k = 20 - 25 (calibrate k on a few UCS tests of the same rock); Cerchar abrasivity (ASTM D7625, 5 scratches "
        "per sample); density and porosity. Enter the medians in the formation card (UCS (lab), CAI (lab)).",
        "<b>Class-wide changes.</b> Where your rocks differ systematically from the defaults (e.g. strongly "
        "silicified dolomite everywhere), edit the class in Rock properties: the typical value is used, min / max "
        "document the range. Blank cells fall back to the default.",
        "<b>Check against drilling.</b> ROP of successful runs falls with UCS (approximately ROP ∝ UCS<sup>−0.5</sup> "
        "at the same RPM and bit); bit life in metres falls with CAI. Plot the median ROP per formation (Risk "
        "Prediction uses the median ROP per rock class as 'normal ROP') against the UCS you entered; a formation far "
        "off the trend usually has a wrong class or lab value.",
        "<b>Retrain</b> the models (Models → Train now) after any change of rock properties, rock codes or cards, and "
        "compare the validation scores with the previous version (the versions stay available)."]))
    A(note("The rock table is shared by all areas of one project database. Changing a property changes the features of "
           "every past hole of that rock class - which is the point of calibration, but it requires retraining."))

    # =========================================================================================
    A(h1("5. Model training and validation"))
    A(h2("5.1 The models"))
    lgb_p = ", ".join(f"{k} = {val}" for k, val in models.LGB_PARAMS.items() if k not in ("verbose", "num_threads", "seed"))
    A(table(["Model", "Question", "Label", "Features", "Trees", "Label-0 rows kept"], [
        ["H (cavity, fault zone)", "Will the hole meet a hazard here?", "metre inside a cavity / fault occurrence",
         f"{len(features.MODEL_FEATURES['H'])}: rock, position, contacts, geometry, karst setting", models.ROUNDS["H"],
         f"{models.NEG_KEEP['H']:.0%}"],
        ["A (per family)", "Will it fail here, ahead of the bit?", "failure metre (0 above it)",
         f"{len(features.MODEL_FEATURES['A'])}: H's features + faults, stacked H, hole state, setup", models.ROUNDS["A"],
         f"{models.NEG_KEEP['A']:.0%}"],
        ["B (per family)", "Is a failure coming in the next metres?", f"last {models.PRECURSOR_L:.0f} m before a failure",
         f"{len(features.MODEL_FEATURES['B'])}: A's features + drilling response", models.ROUNDS["B"],
         f"{models.NEG_KEEP['B']:.0%}"]], [14, 18, 18, 30, 8, 12], "Models H, A and B."))
    A(p(f"LightGBM settings: {E(lgb_p)} (min_data_in_leaf 40 for A and B). A family gets models A and B when it has at "
        f"least {models.MIN_FAILURES} failures in the records, otherwise its hazard is the base rate (failures per "
        "metre) and only the rules warn for it."))
    A(h2("5.2 Training procedure"))
    A(ol([
        "Load all records; apply the database's rock properties, rock codes and measures.",
        "Build the training table: every finished hole as logged, plus one noisy copy (section 3.8).",
        "Model H, out of area: for each area, fit H on the other areas and predict it - these probabilities become "
        "the stacked features of A and B. Then fit the final H on all holes.",
        "Model A per family on all rows; Model B per family on the rows with run data.",
        "Record the feature ranges (for the outside-experience flag R-M04), the cavity statistics by rock class and "
        "contact (reason R-K02), the median ROP per rock class and the severity weights.",
        "Validation leave-one-area-out (5.3) and tuning of the alarm thresholds.",
        "Save as <i>&lt;workspace&gt;\\drillrisk\\models\\models_vYYYYMMDD-HHMM.json</i> (LightGBM model text + settings + "
        "scores) and make it current. Every warning saved later carries this version (spec 7.4)."]))
    A(p("Start it with Risk Prediction → Models → Train now, or <i>.venv\\Scripts\\python.exe tools\\train_models.py "
        "&lt;workspace&gt;</i>. Retrain monthly or every 10 new finished holes, check the scores, and keep the older "
        "version current if the new one is worse (double-click a version in the Models window)."))
    A(h2("5.3 Validation: leave one area out"))
    A(p("The workspace is a new area, so the test that matters is how well each area is predicted by models trained on "
        "the other areas only (spec 5.9). Each hole stays whole in one fold. The measures:"))
    A(table(["Measure", "Definition", "Gate (spec 10)"], [
        ["Model H ROC-AUC", "metre-level AUC of P(cavity) / P(fault zone) on the held-out areas", "> 0.75 (cavity)"],
        ["Family ROC-AUC", "metre-level AUC of the family hazard h against the failure metre", "> 0.75"],
        ["Hole-level concordance", "C of section 2.8 for 'fails before target depth'", "-"],
        ["Recall at Orange", "share of failures with an Orange (or higher) score 3 - 20 m before the failure",
         "≥ 70 % (karst, instability)"],
        ["Lead distance", "median distance from the first Orange metre (within 50 m) to the failure", "median ≥ 3 m"],
        ["False alarms / 100 m", "starts of Orange stretches not followed by a failure within 20 m, per 100 m", "accepted by crews"],
        ["Spread between areas", "range of the hole-level score over the held-out areas", "-"]],
        [22, 60, 18], "Validation measures."))
    A(p(f"<b>Threshold tuning</b> (spec 5.8). For each family the Orange threshold is the score that warns "
        f"{models.TARGET_RECALL:.0%} of its failures at least {models.LEAD_MIN:.0f} m ahead - but never on more than "
        f"{models.MAX_ORANGE:.0%} of all metres (a warning everywhere is no warning). Red = max(1.7 × Orange, the "
        f"{1 - models.MAX_RED:.0%} quantile of the scores); Yellow = min(0.6 × Orange, max(0.43 × Orange, the "
        f"{1 - models.MAX_YELLOW:.0%} quantile)). A weak model (held-out AUC &lt; {models.WEAK_AUC}) warns only on its top "
        "2 % of metres (Red 0.5 %), and a family without a model never warns from its base rate: the rules carry it."))
    A(h2("5.4 Results on the demo data"))
    A(p(f"The demo history is generated by a process model (<i>dfpredict\\risk\\demo.py</i>): 8 past areas × 150 holes "
        "in Proterozoic - Palaeozoic basins, each with its own column (formations missing, thickness varying laterally), "
        "dips, faults and karst setting; holes drilled metre by metre with trajectories walking toward the bedding "
        "normal, cavities most likely in cavernous carbonate below clastic-over-carbonate contacts and above the water "
        "table, failures caused by the rock mechanics and the drilling programme, precursors in the runs. Plus the 104 "
        f"finished holes of the workspace. Current models {E(ms.version)}: trained on {t.get('holes', 0):,} holes, "
        f"{t.get('metres', 0):,} metres, failures {', '.join(f'{codes.FAMILIES[k]} {n}' for k, n in t.get('failures', {}).items())}."))
    fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    A(table(["Measure", "Value"], [
        ["Model H cavity ROC-AUC", fmt(v.get("auc_H_cavity"))], ["Model H fault-zone ROC-AUC", fmt(v.get("auc_H_fault"))],
        ["Hole-level concordance", fmt(v.get("auc_hole"))],
        ["Spread between areas", " - ".join(fmt(x) for x in (v.get("auc_hole_spread") or []))],
        ["False alarms (Orange) per 100 m", fmt(v.get("false_alarms_per_100m"))]] +
        [[f"{codes.FAMILIES[f]}: ROC-AUC / recall at Orange / lead (m)",
          f"{fmt((v.get('auc_family') or {}).get(f))} / {fmt((v.get('recall_orange') or {}).get(f))} / "
          f"{fmt((v.get('lead_m') or {}).get(f))}"] for f in codes.FAMILIES], [60, 40],
        "Leave-one-area-out scores of the current demo models."))
    A(fig(F["validation"], "Family scores on held-out areas against the gates of spec 10 (dashed: AUC 0.75, dotted: "
                           "recall 0.70).", 520))
    A(p("Karst-and-fluid and hole-instability pass the gates. Model H for fault zones is weak by design: the history's "
        "fault zones are random planes not predictable from rock features - in the workspace the fault surfaces of the "
        "geological model give the faults. Trajectory and bit-and-string failures are rare and rely mostly on the "
        "rules (spec 5.9: 'expect less for rare families'). These scores are for synthetic data; real scores come from "
        "real history."))
    if S.get("models"):
        A(fig(S["models"], "The Models window: training data, validation against the gates, thresholds, versions.", 560))

    # =========================================================================================
    A(h1("6. Predicting failure probability and suggesting measures"))
    A(h2("6.1 Ongoing drillhole"))
    A(p("Risk Prediction → pick the hole → Predict. The chain (dfpredict\\risk\\predict.py):"))
    A(ol([
        "<b>Path.</b> The trial paths from the design and the surveys entered so far (section 2.2), to the planned "
        "depth + 60 m.",
        "<b>Ties.</b> Contacts logged so far are compared with the model's along the drilled path; the offsets shift "
        "the contacts below and narrow u (R-M02); a mismatch at the bit widens u (R-M01) (section 2.4).",
        "<b>Geology along every path.</b> Above the bit the logged units; below it the model units with the trial's "
        "offset; incidence angle from the model bedding; fault crossings ± the damage zone.",
        "<b>Features</b> per metre and path (section 3.9); hole sizes and bits as recorded (the last ones continue); "
        "RPM and flow ahead = medians of the last 3 runs.",
        "<b>Model H → Model A</b> per family; the area correction c<sub>f</sub>; mean over the paths.",
        "<b>Model B</b> (when runs are entered) for the next 10 m: the index there is at least the level of the "
        "precursor probability.",
        "<b>Rules R1 - R8</b> on the runs and surveys; a rule triggered on the last run raises the level from 3 m above "
        "to 10 m below the bit to at least its own level (Yellow 0.20, Orange 0.45, Red 0.75).",
        "<b>Levels</b> per metre: the highest family index: Green below 0.15, Yellow 0.15 - 0.35, Orange 0.35 - 0.60, Red from 0.60.",
        "<b>Zones</b>: consecutive metres at Yellow or above, drawn from (top − u) to (bottom + u); precautions start at "
        "the top of the band. Status: Drilled / At bit (within 10 m of the bit) / Ahead.",
        "<b>Reasons</b> and <b>measures</b> per zone; <b>P(reach target)</b>; the danger over any depth range; "
        "hole-level messages."]))
    A(h3("Alarm rules"))
    A(table(["Rule", "Condition (limits in risk/rules.py)", "Family", "Code", "Level", "Message"],
            [[r, E(cond), codes.FAMILIES[fam], c, lv, rc] for r, (cond, fam, c, lv, rc) in codes.RULES.items()],
            [6, 44, 16, 7, 9, 18], "Fixed rules R1 - R8 (spec 5.7); R6 - R8 need the new fields.", small=True))
    A(h3("Reasons"))
    A(p("For every zone the SHAP values of each family's Model A over the zone's metres are averaged; the features with "
        "the largest positive contributions are mapped to reason codes (e.g. carbonate share, karst, distance below the "
        "top of the carbonate, stacked cavity probability → R-K01; clastic-over-carbonate contact, site sheet → R-K02; "
        "fault distance → R-F01; stacked fault-zone probability → R-F02; incidence angle, foliation → R-B01; strength, "
        "abrasivity, bit → R-L01; swelling clay → R-C01; depth / rated depth → R-H01; drilling response → R-D..). The "
        "families of a zone take turns, so a zone of karst and instability shows the best reasons of both (up to 4). "
        "A triggered rule's message comes first. Each reason is a pre-defined sentence filled with the zone's values "
        "(appendix)."))
    A(h3("Measures"))
    A(ol([
        "<b>Candidates</b> from the decision table (spec 6.2, appendix) for the zone's failure codes and level: "
        "Yellow - watch; Orange - prepare (plus Yellow's); Red - act (plus Orange's and Yellow's), and the response "
        "measures 'if the event starts'.",
        "<b>Filters</b>: casing / size-reduction measures only when a smaller size exists and casing is on site; "
        "re-planning only before the zone is drilled; fluid changes only when that fluid is on site.",
        f"<b>Evidence ranking</b>: cavity / loss / caving / casing measures by their solved rate among the {meas.SIMILAR} "
        "most similar past occurrences where they were tried (similarity: 2 × different rock class + different contact "
        "type + |Δdepth| / 200 m); operating-window measures (M-OP-01 - 03, M-OP-07, M-BH-03) filled with the 25th - "
        "75th percentiles of RPM, flow and ROP of successful runs in the same rock class (formation, hole size, bit "
        "type where enough data: ≥ 20 runs from ≥ 3 holes), excluding runs within 20 m above a failure; everything "
        "else by the expert order of the measure library.",
        f"<b>Order</b>: data-ranked measures with at least {meas.LIMITED} uses first (highest solved rate first), then "
        "operating windows, then expert order; the top 3 prevention and top 3 response measures are shown, each with "
        "its evidence tag ('Data: solved 7 of 9 similar cases', 'limited evidence', 'Data: settings from 142 "
        "successful holes', 'Expert rule')."]))
    A(h3("Hole-level messages"))
    A(ul(["R-M02 contact tie, R-M01 model mismatch, R-M03 far from model control (collar farther than half the "
          "'far' distance), R-M04 outside experience (a feature ahead outside the range of the training data: thickness, "
          "carbonate, hardness, abrasivity, depth, height above water table, site sheet values), the area correction, "
          "and the path summary (build / turn rate, spread at the target)."]))
    A(h2("6.2 Planned drillhole and plan check"))
    A(p("A planned hole (result 'planned') is predicted the same way with these differences: the path starts from the "
        f"designed collar azimuth and dip (no surveys, no ties), the usual programme is assumed for hole sizes (PQ collar "
        f"casing to {pr.PQ_COLLAR:.0f} m, then the planned size), all zones are 'Ahead', and Model B and the rules do "
        "not apply. The register printed before spudding is the pre-drill risk register of spec 7.1; agree casing "
        "points and materials from its Orange and Red zones."))
    A(p("<b>Plan check</b> evaluates a design that is not in the records: collar X, Y, Z, azimuth, dip, planned depth, "
        "rig and area. <b>Safer options</b> reruns it for azimuth ± 15° / ± 30° and dip ± 10° (15 designs) and lists the "
        "chance of reaching the target, the worst level and the main risk of each, best first; double-click a row to "
        "check it in full. The options are a hint, not proof: past crews chose settings in response to what they met "
        "(spec 5.5)."))
    if S.get("plan"):
        A(fig(S["plan"], "The Plan check tab: the design of a new hole.", 600))
    A(h2("6.3 Worked examples (demo workspace)"))
    def zone_rows(p_):
        return [[f"{z.from_:.0f} - {z.to:.0f}", f"±{z.u:.0f}", ", ".join(z.codes), f"{z.p:.2f}", z.level, z.status,
                 ", ".join(c for c, _ in z.reasons if c), ", ".join(s_.code for s_ in z.prevent)] for z in p_.zones]
    for p_, txt in ((p1, "In progress; the logged U1/U2 contact lies 4.5 m deeper than the model, so the contacts below "
                         "are shifted. Ahead: the black shale U2 over the cavernous dolomite U3 and fault F4."),
                    (p2, "In progress; the pump pressure fell more than 25 % at the same flow in the last runs and the "
                         "return dropped below 70 %: rules R1 and R6 raise the level at the bit."),
                    (p3, "Planned; the collar lies on fault F2, crossed at about 17 m: the plan check flags the fault zone "
                         "at the top of the hole and suggests moving the collar.")):
        A(h3(E(p_.hole_id)))
        A(p(E(txt) + " <b>" + E(p_.summary) + "</b>"))
        A(table(["Zone (m)", "Band", "Failure", "P", "Level", "Status", "Reasons", "Measures"], zone_rows(p_),
                [11, 6, 15, 6, 8, 8, 22, 24], small=True))
    z = max(p1.zones, key=lambda z_: z_.p)
    A(p("Detail of the main zone of DH-2026-001 (spec 8.2):"))
    A(ul([f"<b>{c}</b> {E(t_)}" for c, t_ in z.reasons] +
         [f"<b>{s_.code}</b>: {E(s_.text)} - <i>{E(s_.evidence)}</i>" for s_ in z.prevent]))
    rp = p1.range_probabilities(p1.drilled_depth, min(p1.planned_depth, p1.drilled_depth + 50))
    A(p(f"Danger in the next 50 m ({p1.drilled_depth:.0f} - {min(p1.planned_depth, p1.drilled_depth + 50):.0f} m): "
        f"failure {rp['failure']:.1%}, karst and fluid {rp['karst']:.1%}, hole instability {rp['instability']:.1%}, "
        f"meeting a cavity {rp['cavity']:.0%}, crossing a model fault {rp['fault_crossed']:.0%}."))
    if S.get("predict"):
        A(fig(S["predict"], "Risk Prediction of DH-2026-001: headline, danger in a range, messages, risk log, register "
                            "and the detail of the selected zone.", 640))
    if S.get("view3d"):
        A(fig(S["view3d"], "3D view: drilled part coloured by formation, risk sleeves by level, fault F4 crossing (red "
                           "disc), the predicted path ahead (dashed).", 560))

    # =========================================================================================
    A(h1("7. Operation manual"))
    A(h2("7.1 Installation and start"))
    A(ul(["<b>install_env.bat</b> - creates <i>.venv</i> (Python 3.10) offline from the wheel store E:\\python_packages "
          "with the exact versions of requirements-offline.txt, then runs the checks.",
          "<b>run_dfpredict.bat -w demo_data\\sb3_workspace</b> - starts the program on the demo workspace (without "
          "arguments it reopens the last workspace).",
          "<b>run_gui_test.bat [seconds per step]</b> - a visible walkthrough of all drilling-risk functions on a copy "
          "of the demo workspace.",
          "Settings → Database: SQLite (workspace file, default) or MySQL; Settings → General → Language: English / "
          "中文 (used at the next start); Settings → Display → Theme: dark or light (applied at once)."]))
    A(h2("7.2 Toolbar: drilling risk"))
    A(table(["Button", "Shortcut", "Use"], [
        ["Drilling records", "Ctrl+H", "the success / failure database and all reference data"],
        ["Risk prediction", "Ctrl+Shift+R", "failure probability, zones and measures of an ongoing or planned hole"],
        ["Plan check", "", "the same for a design typed in (not saved)"],
        ["Geological model", "Ctrl+Shift+G", "roles, unit names, error band, area"],
        ["Import geological model / fault surfaces", "", "open DTM / DXF / GOCAD files with a role"],
        ["3D slice", "Ctrl+Shift+L", "show only the objects between two planes"]], [30, 16, 54]))
    A(h2("7.3 Drilling Records"))
    A(ol(["Left: the holes (in progress and planned first), search box, area and result filters, summary cards.",
          "Hole record tab: the collar form (identification, location, design, status, at the rig, notes) and one tab per "
          "table. Edit cells directly (drop-down lists for codes; units in the headers; red cells = errors with the "
          "reason in the tooltip); paste rows from Excel.",
          "Runs: <i>Next run</i> adds the next core run with From = last To and the last RPM / flow / fluid copied; "
          "enter ROP, pressure, return, rod drop. <i>Record event</i> adds a cavity / fault zone / problem with measures "
          "and solved; <i>Record failure</i> sets the hole Failed with its failure.",
          "<i>Check</i> runs the entry checks; <i>Save hole</i> (Ctrl+S) writes collar and tables in one transaction; "
          "<i>Revert</i> drops unsaved changes.",
          "Reference tabs: Areas, Formation cards, Rig specifications, Rock properties, Rock codes, Measure library - "
          "each with Save and (where predefined) Restore defaults. Overview: failures by area and code, occurrences, "
          "fill rate of the run fields per crew.",
          "Import / export CSV; Holes without an area."]))
    if S.get("records"):
        A(fig(S["records"], "Drilling Records: the hole list, the collar form and the runs of DH-2026-001.", 640))
    if S.get("rock"):
        A(fig(S["rock"], "Drilling Records → Rock properties: the editable mechanical properties.", 600))
    A(h2("7.4 Risk Prediction window"))
    A(ul(["<b>Top</b>: hole (with state and bit depth), number of trial paths, Predict, Show in the 3D view, Records.",
          "<b>Headline</b>: chance of reaching the target depth (green ≥ 85 %, yellow ≥ 70 %, orange ≥ 50 %, else red), "
          "the worst level ahead and the summary sentence of spec 5.2.",
          "<b>Danger in a depth range</b>: From / To or Next 10 m, Next 50 m, To target - failure (any and per family), "
          "meeting a cavity, crossing a model fault, an unmapped fault zone; a live alarm at the bit is shown on top.",
          "<b>Messages</b>: live rules, contact ties, mismatch, model control, outside experience, area correction, path.",
          "<b>Risk log</b>: formation column with contacts and fault crossings, Model H cavity / fault-zone probability, "
          "risk index per family over the level bands, level strip, bit and target; zones with their ±u band; click a "
          "zone to select it.",
          "<b>Risk register</b> (spec 8.3): From, To, ±u, failure codes, P, level, reasons, measures, status; sort by any "
          "column; <i>Only ahead of the bit</i>.",
          "<b>Detail panel</b> (spec 8.2): level and failure names, band, P, chance of failure, formation, cavity / fault "
          "probability, why the zone is dangerous, rules triggered, what to do (from the top of the band), if the event "
          "starts.",
          "<b>Export</b>: CSV (register), Report (HTML or PDF with chart and all zones), Save warnings (database, with the "
          "model version). <b>Models ...</b>: training and versions."]))
    A(h2("7.5 3D view"))
    A(p("The Drilling risk layer shows the predicted hole: drilled part coloured by formation, the planned part as a "
        "dashed tube, the uncertainty cone, risk sleeves in 1 m segments (green / yellow / orange / red) with a paler "
        "±u band, fault crossings (red discs), recorded events (spheres) and the label with the chance of reaching the "
        "target. The zone selected in the register is wrapped in a white shell. With Identify, click a sleeve to see "
        "the zone and select it in the register. Properties: diameter, cone, band, events, labels, lowest level shown."))
    A(h2("7.6 Daily use for a hole being drilled"))
    A(ol(["Before spudding: register the hole (collar, design, rig, casing and fluids on site); Predict; print the "
          "pre-drill register (Report); agree casing points and materials on site from the Orange / Red zones.",
          "After every run: enter the run (Next run), any event, the quick geology; after each survey (every 30 - 50 m, "
          "10 - 20 m in trajectory-risk zones) enter the station; Predict again (the next 30 - 50 m are rescored).",
          "Follow the measures of the zone ahead from the top of its band; record what was done and whether it worked "
          "(Record event) - these outcomes turn expert-ranked measures into data-ranked ones.",
          "If the hole is abandoned: Record failure. When finished: set the result and the final depth; the hole joins "
          "the training data at the next retraining."]))
    A(h2("7.7 Troubleshooting"))
    A(table(["Symptom", "Cause / remedy"], [
        ["'No trained models'", "Models → Train now (needs finished holes in the records), or choose a saved version."],
        ["'The project database is not available'", "Settings → Database: SQLite, or check the MySQL server / account."],
        ["A unit shows 'no card'", "name the unit as a formation card (Geological Model window) or add the card."],
        ["Prediction looks wrong in one formation", "check its rock codes, card and lab UCS / CAI (chapter 4), retrain."],
        ["Many false alarms", "check the validation; record more outcomes; thresholds are tuned at each training."],
        ["Outside-experience messages (R-M04)", "the new area differs from the history: rely more on rules and expert "
                                                "measures; add history from similar areas."]], [35, 65]))

    # =========================================================================================
    A(h1("8. Appendices"))
    A(h2("A. Failure codes"))
    A(table(["Code", "Failure", "Family"], [[c, n, codes.FAMILIES[f]] for c, (n, f) in codes.FAIL_CODES.items()],
            [10, 55, 35]))
    A(h2("B. Measure library (spec 6.1)"))
    A(table(["Code", "Measure (values in braces filled by the system)", "Failures", "Stage"],
            [[m.code, E(m.text), ", ".join(m.failures), m.stage] for m in codes.DEFAULT_MEASURES.values()],
            [12, 64, 16, 8], small=True))
    A(h2("C. Decision table (spec 6.2)"))
    A(table(["Failure", "Yellow (watch)", "Orange (prepare)", "Red / event started"],
            [[c, ", ".join(d.get("yellow", [])), ", ".join(d.get("orange", [])), ", ".join(d.get("red", []))]
             for c, d in codes.DECISION.items()], [10, 25, 30, 35], small=True))
    A(h2("D. Reason codes (spec 5.8, 5.10)"))
    A(table(["Code", "Title", "Message"], [[c, E(codes.REASON_TITLES.get(c, "")), E(t_)] for c, t_ in codes.REASONS.items()],
            [10, 22, 68], small=True))
    A(h2("E. Files"))
    A(table(["Path", "Content"], [
        ["&lt;workspace&gt;\\geodatabase.sqlite", "project database (drillhole_* tables shared with GeoViewer, risk_* tables) when SQLite is used"],
        ["&lt;workspace&gt;\\geoviewer_workspace.json", "layers, roles, display settings"],
        ["&lt;workspace&gt;\\drillrisk\\geomodel.json", "unit names, contact order, error band, area"],
        ["&lt;workspace&gt;\\drillrisk\\models\\", "trained models (one JSON per version), current.txt"],
        ["tools\\make_demo_workspace.py", "builds the demo workspace and its history"],
        ["tools\\train_models.py", "trains the models of a workspace from the command line"],
        ["tools\\make_manual.py", "builds this document"]], [40, 60]))

    # --- print ---
    css = """
    body { font-family: 'Segoe UI'; font-size: 10pt; color: #1b1f24; }
    h1 { font-size: 18pt; color: #1f4e79; margin-top: 0px; margin-bottom: 10px; }
    h2 { font-size: 13pt; color: #1f4e79; margin-top: 16px; margin-bottom: 6px; }
    h3 { font-size: 11pt; color: #2c3e50; margin-top: 12px; margin-bottom: 4px; }
    p { margin-top: 3px; margin-bottom: 6px; line-height: 125%; }
    p.cap { font-size: 8.5pt; color: #444; margin-top: 2px; margin-bottom: 12px; }
    li { margin-bottom: 3px; }
    th { font-size: 8.5pt; }
    code { font-family: 'Consolas'; font-size: 9pt; }
    """
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import QFont, QPageLayout, QPageSize, QTextDocument
    from PySide6.QtPrintSupport import QPrinter

    doc = QTextDocument()
    doc.setDefaultFont(QFont("Segoe UI", 10))
    doc.setDefaultStyleSheet(css)
    doc.setHtml("<html><body>" + "".join(parts) + "</body></html>")
    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(str(OUT))
    printer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait, QMarginsF(16, 16, 16, 16),
                                      QPageLayout.Millimeter))
    printer.setDocName("Drilling Failure Prediction & Prevention - Algorithm & Manual")
    printer.setCreator("dfpredict tools/make_manual.py")
    doc.print_(printer)
    (ASSETS / "manual.html").write_text("<html><head><meta charset='utf-8'><style>" + css + "</style></head><body>" +
                                        "".join(parts) + "</body></html>", encoding="utf-8")
    del QSizeF
    print("written", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-shots", action="store_true", help="skip the window screenshots")
    a = ap.parse_args()
    build(no_shots=a.no_shots)
