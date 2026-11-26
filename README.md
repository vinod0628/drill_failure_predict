# Drilling Failure Prediction & Prevention

A Windows desktop system that predicts **where and why an ongoing drillhole is likely to fail** and tells the driller what to do about it. It learns from past holes, successful and failed, in distant drilling areas, then runs ahead of the bit on the current workspace's approximate geological model.

It implements `docs/Drilling Failure Prediction & Prevention System — Design Spec.pdf` on the GeoViewer platform. The stack is the same as GeoViewer / 3D MPM Studio: Python 3.10, PySide6, VTK/PyVista, NumPy/pandas and the project database (SQLite or MySQL), plus LightGBM for the models. The spec's suggested PostgreSQL / FastAPI / Next.js stack is not used. Workspaces are GeoViewer workspaces (`geoviewer_workspace.json`), and the geological-model management and display follow 3D MPM Studio.

---

## Quick start

```
install_env.bat                                   (once: offline install from E:\python_packages into .venv)
run_dfpredict.bat -w demo_data\sb3_workspace      (open the demo workspace)
```

Then use the **Drilling risk** group of the floating toolbar:

| Button | Shortcut | What it does |
|---|---|---|
| Drilling records | Ctrl+H | The success / failure database: holes, surveys, geology, runs, events, failures, plus the reference data (areas, formation cards, rigs, rock properties, rock codes, measure library) |
| Risk prediction | Ctrl+Shift+R | Failure probability ahead of the bit for an ongoing or planned hole: warning zones, reasons, measures |
| Plan check | | The safety of a designed hole from its collar X, Y, Z, azimuth, dip and depth, with safer options |
| Geological model | Ctrl+Shift+G | Roles of the surface layers, unit names, the error band ±u |
| Import geological model / fault surfaces | | Surpac DTM, DXF or GOCAD files opened with a role |
| 3D slice | Ctrl+Shift+L | Shows only what lies between two planes; solids show filled section faces |

Everything GeoViewer does is still there: layers, properties, sections, Identify, Measure, the Drillhole Manager, block models, map export and so on.

The Drillhole Manager edits the same holes as Drilling Records. It also has the drilling tabs: Runs, Hole size, Bit, Occurrences, Failure and Drilling reference. The drillhole layer's **Area** setting (Properties) shows the workspace's area. Automatic shows the areas with holes in work or planned, so the history of distant areas stays out of the 3D view.

---

## How it works

### Data: the drillhole tables shared with GeoViewer

The records are in the workspace's project database (Settings → Database: SQLite `geodatabase.sqlite` in the workspace folder, or MySQL). The tables are made automatically.

**One drillhole database.** The drillhole tables are the same as GeoViewer's and 3D MPM Studio's. Those programs record the same history in their Drillhole Manager, but have no prediction. The tables are described once in `dfpredict/io/drilling_schema.py`, an identical copy in each program.

| Table | Content (spec section) |
|---|---|
| `drillhole_collar` | Hole, area, collar XYZ, rig, design (azimuth, dip, target depth), depth reached (`max_depth`), state, casing and fluids on site (3.1 A) |
| `drillhole_survey`, `drillhole_geology`, `drillhole_hole_size`, `drillhole_bit` | Survey stations, geology log (rock, formation, broken zone, bedding angle), hole sizes, bits (B–E) |
| `drillhole_run` | Per-run RPM, ROP, flow, pump pressure, plus the new fields return %, rod drop, WOB, torque, fluid, recovery (F, 3.3) |
| `drillhole_failure`, `drillhole_occurrence` | Failure label (code, depth, severity, measures); cavities, fault zones and other problems with measures and solved Y/N (G, H) |
| `drillhole_area`, `drillhole_formation`, `drillhole_rig`, `drillhole_measure` | Site sheets (5.10), formation property cards (3.5) with optional lab UCS / CAI, rig limits (I), measure library (6.1) |
| `risk_rock`, `risk_rock_code` | The risk system's own data: rock mechanical properties and rock-code mapping, **predefined, editable** |
| `risk_warning` | Warnings issued, with the model version (7.4) |

**How the tables are stored**, the same in all three programs:
- **Dips** are degrees from straight down (0 = down).
- **`max_depth`** is the final or current depth.
- **`state`** is the hole's status: Planned, In work, Completed with ore hit, Completed without ore hit, Failed. A success whose ore hit is not known is stored as Completed without ore hit.

**How this program shows them.** Drilling Records and the CSV files keep the driller's format: dip negative downward, final depth, and the result (success / failed / in progress / planned). The conversion happens when reading and writing.

**Each program keeps the other's data.** Saving a hole here keeps what only GeoViewer records: organization, assays, core notes and so on. A "success" keeps GeoViewer's "with / without ore hit".

**Databases of earlier versions.** A database with the tables `risk_collar`, `risk_run`, … is moved into the drillhole tables when it is opened, once. The old tables are kept as `risk_*_old`. Where a hole was in both sets of tables, GeoViewer's values are kept and its empty fields are filled; a recorded failure sets the state and depth reached.

**Predefined data.** When a workspace's database is set up, the reference tables are filled:
- 46 rock types with their mechanical properties;
- 66 rock codes;
- the 37 measures of spec 6.1;
- example rigs.

This only happens while a table is empty, so your edits are never replaced. "Restore defaults" in each tab adds the missing rows or resets the table.

**Rock mechanical properties** (`risk/rockmech.py`, editable in Drilling Records → Rock properties) are typical values of fresh intact rock from the standard references:
- UCS (min / typical / max), tensile strength, Young's modulus, Poisson's ratio, density, porosity: Goodman 1989; Lama & Vutukuri 1978; Hoek 2007;
- Cerchar abrasivity index CAI and equivalent quartz: ASTM D7625; Plinninger;
- NTNU drilling rate index DRI;
- Hoek–Brown mᵢ;
- solubility, swelling, anisotropy (foliation) and fracturing tendency.

The rocks cover clastic, carbonate, evaporite, metamorphic (schist, slate, phyllite, quartzite, marble, gneiss, hornfels …), igneous, ore, fault and weathered rock, including Proterozoic basin rocks such as banded iron formation. From UCS and CAI the system derives the ISRM strength class and the abrasivity class. Formation cards can carry measured UCS / CAI, which take priority. **Retrain the models after changing the properties.**

### Geological model (after 3D MPM Studio)

The model is read from the surface layers by their **role**, set in Properties → Role or in the Geological Model window:
- **Horizons / unconformities** are the contacts. Ordered top to bottom, they split the ground into units, which you name like the formation cards. A name such as `horizon_H2_U2-U3` names the units automatically.
- **Faults** give the fault crossings and fault distances.
- **Formation solids** are converted to voxels.
- **Topography** gives the ground surface.

Surfaces are gridded once, so the trial paths are read quickly.

The model is approximate (built from ground and geophysical surveys and sparse holes). The error band ±u (spec 5.6) grows from *u near* to *u far* with distance from model control, and widens below units whose thickness varies widely. Every trial path samples its own model offset from ±u. When the bit meets a contact, the offset shifts the contacts below and narrows u (R-M02). A logged rock that disagrees with the model raises a mismatch flag and widens u ahead (R-M01).

### Predicting an ongoing hole

1. **Path ahead of the bit** (`risk/trajectory.py`). The drilled part is desurveyed by minimum curvature from the surveys. Ahead, the hole continues at the build and turn rate of the recent stations, fading with distance, plus the walk toward perpendicular to bedding where the incidence angle is low. An ensemble of trial paths (deviation spread taken from past holes' surveys) gives the uncertainty cone out to the **designed depth**.
2. **Features per metre** (`risk/features.py`, spec 5.3). These are rock properties from the formation card plus the rock's mechanics (UCS, CAI, stiffness, brittleness, anisotropy, solubility, porosity, density); position in the unit; contact setting (including the strength contrast at contacts); incidence angle; faults; the karst setting from the site sheet; hole state; drilling setup; and, while drilling, the drilling response. There are deliberately no coordinates and no formation names, so the learning transfers to new, distant areas.
3. **Models** (`risk/models.py`, spec 5.4–5.5), all gradient-boosted trees (LightGBM):
   - **Model H**: cavity and fault-zone probability per metre;
   - **Model A**: failure hazard per metre for each family (karst and fluid, hole instability, trajectory, bit and string) as a discrete-time hazard;
   - **Model B**: precursors in the next 3–10 m from the latest runs.

   They are trained on every finished hole, with a noisy copy of the logs (spec 5.6.1), and validated **leave-one-area-out** (spec 5.9). The validation leaves out the features of the unit base that would give a failure away. Families with too few failures, or a weak held-out score, rely on the rules.
4. **Risk** (spec 5.2, 5.8). The hazards are averaged over the trial paths and corrected for the area (observed vs expected failures of the area's finished holes, weighted n/(n+5)). The risk index is the hazard over 10 m times the severity weight, through thresholds tuned per family on held-out areas. Fixed rules R1–R8 override the level at the bit. The chance of reaching the target is ∏(1 − h).
5. **Output** (spec 6, 8):
   - warning zones from (top − u) to (bottom + u);
   - reason codes from the models' SHAP values (LightGBM tree contributions);
   - measures from the decision table, ranked by solved rate among the most similar past occurrences, with RPM / flow / ROP filled from successful runs in the same rock, size and bit, and an evidence tag on every measure;
   - the danger probabilities over any depth range;
   - the risk register (CSV), a report (HTML / PDF), and warnings saved with the model version.

**Plan check** runs the same chain on a design that is not saved: collar X, Y, Z, azimuth, dip, depth and rig. **Safer options** reruns it for azimuth ±15° / ±30° and dip ±10° (spec 5.5, 9).

### 3D view

The Drilling risk layer shows, for the predicted hole:
- the drilled part coloured by formation;
- the planned part as a dashed tube, with the uncertainty cone;
- risk sleeves in 1 m segments (green / yellow / orange / red) with a paler ±u band;
- fault crossings, events, and a label with the chance of reaching the target.

The zone selected in the register is highlighted. Identify on a sleeve selects its zone in the Risk Prediction window.

---

## Demo workspace

`demo_data/sb3_workspace` is built by `.venv\Scripts\python.exe tools\make_demo_workspace.py` (from the inputs already in the folder; `--source <folder>` copies them in again).

- **Geological model** from 3D MPM Studio's synthetic scenario SB3: five contacts between U1 cover sandstone, U2 black shale, U3 dolomite, U4 siltstone, U5 tuff and U6 schist; four faults F1–F4; the formation block model and the ore solids. Its 104 drillholes are in the project database.
- **History:** 8 past areas × 150 holes in Proterozoic–Palaeozoic basins (schist, slate, quartzite, dolomite, marble, limestone, shale, banded iron formation …). Each area has its own column, missing formations and thickness varying within the area. The holes are drilled by a process model (`risk/demo.py`) in which the rock mechanics, karst setting, faults and drilling programme cause the problems, with precursors in the runs. 16–18 % of holes failed. The history is also in `demo_data/drilling_history_csv` for the CSV import.
- **Workspace area SB3:** records of the 104 finished holes, plus three live holes:
  - **DH-2026-001**, in progress at 182 m of 320 m: heading for the dolomite U3 and fault F4. Its logged U1/U2 contact is 4.5 m deeper than the model, so the predictions are tied.
  - **DH-2026-002**, in progress: pump pressure fell by over 25 % in the last runs (rule R1).
  - **DH-2026-003**, planned.
- **Models** trained on all of it (`drillrisk/models`).

Retrain any workspace with `.venv\Scripts\python.exe tools\train_models.py <workspace>` or the Models button.

## Tests

```
.venv\Scripts\python.exe -m pytest -q tests\test_risk.py      (drilling risk: database, mechanics, model, rules, prediction, windows)
.venv\Scripts\python.exe -m pytest -q tests                   (plus GeoViewer's regression tests; their data is in tests\data\geoviewer_demo)
.venv\Scripts\python.exe tools\smoke_test.py
run_gui_test.bat [seconds per step]                          (visible GUI walkthrough, see below)
```

**GUI walkthrough** (`run_gui_test.bat`, `tests\test_gui_walkthrough.py`): the real main window opens on a copy of the demo workspace and the mouse moves slowly over every drilling-risk function: Drilling Records (search, collar form, tabs, Next run, entry checks, Revert, reference tabs), Risk Prediction (DH-2026-001, register rows, danger in a range, DH-2026-002 with rule R1), Plan check with Safer options, the Models and Geological Model windows, the 3D slice and the 3D risk layer. Dialogs stay open for a moment before they close. `run_gui_test.bat 2` runs it slower (2 s per step), `run_gui_test.bat 0.6` faster. Do not use the mouse or keyboard while it runs; it takes about 5 minutes at the normal pace. The normal test run skips it.

**Languages:** English and Chinese (Settings → Language). Every text of the interface is in `dfpredict\i18n\locales\zh_CN.json`; `tests\test_i18n.py` fails when a text is not translated.

## Project layout

```
dfpredict/
  risk/        codes (failure codes, families, reason / measure libraries, decision table, rules),
               schema + store (risk_* tables, predefined data, CSV, entry checks), rockmech (rock properties),
               geomodel, trajectory, features, models (H / A / B, validation), rules, measures, predict, engine, demo
  scene/       GeoViewer scene and layers + geo_roles, slicing (3D slice), risk_layer
  ui/          GeoViewer windows + history_dialog (Drilling Records), prediction_dialog (Risk Prediction / Plan check),
               risk_admin (Models, Geological Model), risk_widgets, slice_panel
  io/          GeoViewer readers + gocad (from 3D MPM Studio)
tools/  make_demo_workspace.py, train_models.py, smoke_test.py
tests/  test_risk.py + GeoViewer's tests
```
