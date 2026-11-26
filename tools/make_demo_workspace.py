"""Build the demo workspace of the drilling risk system.

    python tools/make_demo_workspace.py [--holes-per-area 150] [--no-train]

    python tools/make_demo_workspace.py --source <SB3_mixed/inputs folder>    (copy the inputs in first)

* ``demo_data/sb3_workspace``: the geological model of 3D MPM Studio's synthetic scenario SB3 (five
  stratigraphic contacts H1 - H5 between units U1 - U6, four faults F1 - F4, the formation block model,
  the ore solids), its 104 drillholes in the project database (geodatabase.sqlite), the workspace file
  with the layers' roles, and the drilling records:
    - the history: 8 past areas x N holes (Proterozoic - Palaeozoic basins, see risk/demo.py);
    - the formation cards of U1 - U6 (never drilled in the past areas), the site sheet of the area;
    - the records of the 104 finished holes of the workspace;
    - DH-2026-001 (in progress, heading for the dolomite U3 and fault F4), DH-2026-002 (in progress, pump
      pressure dropping in the last runs), DH-2026-003 (planned);
  and the trained models (drillrisk/models).
* ``demo_data/drilling_history_csv``: the history as CSV files (spec 3.4), for the CSV import.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfpredict.io.database import Target  # noqa: E402
from dfpredict.risk import demo, models  # noqa: E402
from dfpredict.risk.geomodel import build_model, save_config, surfaces_from_files  # noqa: E402
from dfpredict.risk.schema import ALL_TABLES  # noqa: E402
from dfpredict.risk.store import RiskStore, default_reference  # noqa: E402

OUT = ROOT / "demo_data" / "sb3_workspace"
CSV_OUT = ROOT / "demo_data" / "drilling_history_csv"
UNIT_COLORS = {"horizon_H1_U1-U2": (0.93, 0.80, 0.45), "horizon_H2_U2-U3": (0.35, 0.35, 0.40),
               "horizon_H3_U3-U4": (0.30, 0.65, 0.95), "horizon_H4_U4-U5": (0.62, 0.78, 0.45),
               "horizon_H5_U5-U6": (0.80, 0.55, 0.85)}


def copy_inputs(src: Path) -> None:
    """Copy the SB3 inputs (3D MPM Studio's synthetic scenario SB3_mixed/inputs) into the demo workspace."""
    OUT.mkdir(parents=True, exist_ok=True)
    for sub, pattern in (("model", "horizon_*"), ("model", "ore_solids.*"), ("faults", "F*"),
                         ("drillholes", "*.csv"), ("block_model", "original_bm.gvbm")):
        (OUT / sub).mkdir(exist_ok=True)
        for p in (src / sub).glob(pattern):
            shutil.copy2(p, OUT / sub / p.name)
    shutil.copy2(src / "geodatabase.sqlite", OUT / "geodatabase.sqlite")


def layer(name, files, color, role="", visible=True, opacity=1.0, typ="SurfaceLayer"):
    return {"name": name, "type": typ, "part": "", "files": files,
            "settings": {"visible": visible, "opacity": opacity, "color": list(color), "use_styles": False,
                         "smooth": True, "show_edges": False, "role": role}}


def write_workspace() -> None:
    layers = [{"name": "Drillholes (database)", "type": "DrillholeLayer", "part": "", "files": [],
               "settings": {"visible": True, "diameter": 3.0, "area": "SB3"},   # not the history of the past areas
               "source": "database"}]
    for name, col in UNIT_COLORS.items():
        layers.append(layer(name, [f"model/{name}.dtm"], col, "horizon", opacity=0.55))
    for f in ("F1", "F2", "F3", "F4"):
        layers.append(layer(f, [f"faults/{f}.dtm"], (0.90, 0.20, 0.20), "fault", opacity=0.6))
    layers.append(layer("ore_solids", ["model/ore_solids.dtm"], (0.95, 0.75, 0.15), "orebody", visible=False))
    layers.append({"name": "original_bm", "type": "BlockModelLayer", "part": "", "files": ["block_model/original_bm.gvbm"],
                   "settings": {"visible": False, "opacity": 1.0, "color": [0.45, 0.82, 0.4], "color_by": "lith",
                                "cmap": "turbo", "hide_null": True, "value_min": None, "value_max": None,
                                "clim": None, "show_edges": False, "constraints": [], "constraint_logic": "and",
                                "mined_display": "gray"}})
    ws = {"format": "GeoViewer workspace", "version": 1, "app_version": "0.1.0", "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "view": None, "layers": layers,
          # every data file of the folder is known: the drillhole CSVs are the database's holes (not a 2nd layer)
          "known_files": sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*")
                                if p.is_file() and p.suffix.lower() in (".csv", ".dtm", ".str", ".gvbm", ".ts", ".dxf"))}
    (OUT / "geoviewer_workspace.json").write_text(json.dumps(ws, indent=2), encoding="utf-8")


# Colours of the rock codes logged in the workspace's holes (drillhole_styles, in the families of the units' colours).
ROCK_COLORS = {"SS": "#e8d8a8", "SLT": "#c8b480", "SH": "#505050", "DO": "#a0c8e8", "LS": "#80b0d8", "ORE": "#ff2020",
               "TF": "#406040", "SC": "#9070a0", "QZ": "#f0f0e0", "GR": "#e08080"}


def reset_records(path: Path) -> None:
    """The demo's drilling records are generated again: the risk system's tables of an earlier build (and of
    earlier versions, risk_* / risk_*_old) are dropped; the drillhole tables are rewritten by replace_all."""
    from dfpredict.io.database import Database

    with Database.sqlite(path, create=False) as db:
        for t in db.tables():
            if t.lower().startswith("risk_"):
                db.execute(f"DROP TABLE {db.q(t)}")
        db.con.commit()


def rock_styles(store_target, used: set[str]) -> None:
    """Colours and descriptions of the rock codes used in the workspace's geology log (where missing)."""
    from dfpredict.io.drillhole_store import DrillholeStore
    from dfpredict.risk import codes, rockmech

    dh = DrillholeStore(store_target)
    have, desc = dh.styles(), {c: d for c, d, _n in dh.rock_codes()}
    for code in sorted(used):
        if code not in have and code in ROCK_COLORS:
            dh.set_style(code, ROCK_COLORS[code])
        cls = codes.DEFAULT_ROCK_CODES.get(code)
        if not desc.get(code) and cls in rockmech.DEFAULTS:
            dh.set_description(code, rockmech.DEFAULTS[cls].name)


def geomodel():
    pr = [(p, "horizon") for p in sorted((OUT / "model").glob("horizon_*.dtm"))]
    pr += [(p, "fault") for p in sorted((OUT / "faults").glob("F*.dtm"))]
    collars = pd.read_csv(OUT / "drillholes" / "collar.csv")
    surf = surfaces_from_files(pr)
    cards = demo.cards_dict(demo.cards_frame(demo.CATALOGUE + demo.WORKSPACE_CARDS))
    m = build_model(surf, {"units": ["U1", "U2", "U3", "U4", "U5", "U6"]}, cards=cards,
                    control_xy=collars[["x", "y"]].to_numpy(float))
    save_config(OUT, {"units": m.units, "u_near": 8.0, "u_far": 30.0, "d_far": 250.0, "fault_zone": 12.0,
                      "thickness_factor": 0.10, "voxel": [5.0, 5.0, 2.0],
                      "control_holes": "drillholes of the project database (collars)"})
    return m


def find_collar(m, az, dip, want_fault="F4", top_md=(160, 215)):
    """A collar whose straight design meets the top of the dolomite U3 and then fault F4 inside U3."""
    from dfpredict.io.drillholes import _direction

    d = np.arange(0, 420, 1.0)
    v = _direction(np.array([az]), np.array([dip]))[0]
    best = None
    for x in np.arange(5250, 5800, 20.0):
        for y in np.arange(8200, 8700, 20.0):
            xyz = np.array([x, y, 315.0]) + d[:, None] * v
            cs = m.contacts_along(xyz, d)
            u3 = [c.depth for c in cs if c.lower == "U3"]
            u4 = [c.depth for c in cs if c.lower == "U4"]
            if not u3 or not u4 or not (top_md[0] <= u3[0] <= top_md[1]):
                continue
            fc = [c.depth for c in m.fault_crossings(xyz, d) if c.fault == want_fault]
            if fc and u3[0] + 5 <= fc[0] <= u4[0] - 3:
                score = abs(fc[0] - (u3[0] + 15))
                if best is None or score < best[0]:
                    best = (score, x, y, u3[0], fc[0], u4[0])
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holes-per-area", type=int, default=150)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--source", default=None,
                    help="folder with the SB3 inputs to copy in (model/, faults/, drillholes/, block_model/, "
                         "geodatabase.sqlite); default: the files already in demo_data/sb3_workspace")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.source:
        print("copying the SB3 inputs from", a.source, "...")
        copy_inputs(Path(a.source))
    elif not (OUT / "model").exists():
        raise SystemExit("demo_data/sb3_workspace has no model/ folder: give --source <SB3 inputs folder>")
    write_workspace()
    print("building the geological model ...")
    m = geomodel()
    print(" ", m.describe())
    print(f"generating the history ({a.holes_per_area} holes per area) ...")
    hist = demo.generate_history(n_per_area=a.holes_per_area, seed=7,
                                 progress=lambda i, n: print(f"  {i}/{n}", end="\r"))
    print()
    # the workspace's own records
    holes = pd.read_csv(OUT / "drillholes" / "collar.csv")
    surveys = pd.read_csv(OUT / "drillholes" / "survey.csv")
    geology = pd.read_csv(OUT / "drillholes" / "geology.csv")
    print("records of the 104 finished workspace holes ...")
    ws = demo.workspace_records(m, holes, surveys, geology)
    # the ongoing holes
    best = find_collar(m, 300.0, -72.0)
    print("  DH-2026-001 collar:", best)
    _, x1, y1, u3_top, f4, _u4 = best
    live = [demo.ongoing_hole(m, "DH-2026-001", (float(x1), float(y1)), 300.0, -72.0, 320.0,
                              current=round(u3_top - 22.0), seed=21, contact_offset=4.0),
            demo.ongoing_hole(m, "DH-2026-002", (5720.0, 8640.0), 0.0, -65.0, 300.0, current=171.0, seed=22,
                              pressure_drop=True),
            demo.ongoing_hole(m, "DH-2026-003", (5340.0, 8560.0), 120.0, -60.0, 340.0, current=0, seed=23)]
    for rec in live:
        demo._collect(ws, rec)
    frames = {}
    defaults = default_reference()
    for t in ALL_TABLES:
        if t.name == "warning":
            frames[t.name] = pd.DataFrame(columns=t.names)
            continue
        if t.name in ("rock", "rock_code", "measure"):          # the predefined reference data
            frames[t.name] = pd.DataFrame(defaults[t.name]).reindex(columns=t.names)
            continue
        parts = [hist.get(t.name, pd.DataFrame())]
        if t.name in ws:
            parts.append(pd.DataFrame(ws[t.name]))
        if t.name == "area":
            parts.append(pd.DataFrame([demo.SB3_AREA]))
        if t.name == "formation":
            parts = [demo.cards_frame(demo.CATALOGUE + demo.WORKSPACE_CARDS)]
        frames[t.name] = pd.concat([p for p in parts if len(p)], ignore_index=True).reindex(columns=t.names)
    reset_records(OUT / "geodatabase.sqlite")
    target = Target("sqlite", path=OUT / "geodatabase.sqlite")
    store = RiskStore(target)
    print("project database:", store.prepare()[:3], "...")
    store.replace_all(frames)            # the shared drillhole tables (the 104 SB3 holes keep their assays)
    rock_styles(target, set(pd.DataFrame(ws["geology"])["rock"].dropna().astype(str)))
    s = store.summary()
    print("  holes by result:", s["result"], " failures:", sum(s["fail_code"].values()))
    # the history alone as CSV (spec 3.4)
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    for t in ALL_TABLES:
        if t.name in hist:
            hist[t.name].reindex(columns=t.names).to_csv(CSV_OUT / f"{t.name}.csv", index=False, encoding="utf-8")
    if not a.no_train:
        print("training Models H, A, B (leave-one-area-out validation) ...")
        data = store.load_all()
        ms = models.train(data, progress=lambda p, txt: print(f"  {p:3d}% {txt}", end="\r"))
        print()
        p = models.save(ms, OUT)
        print("  saved", p.name)
        print("  validation:", json.dumps({k: v for k, v in ms.validation.items() if k != "auc_hole_by_area"},
                                          default=lambda o: round(o, 3) if isinstance(o, float) else str(o)))
        print("  thresholds:", ms.thresholds)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
