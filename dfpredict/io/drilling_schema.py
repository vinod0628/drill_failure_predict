"""The drilling database: the drillhole tables that GeoViewer, 3D MPM Studio and the Drilling Failure
Prediction system (dfpredict) share, described once.

Each program keeps an identical copy of this file (geoviewer/io/drilling_schema.py,
mpm3d/io/drilling_schema.py, dfpredict/io/drilling_schema.py): it is the contract between them.
Change it in one program and copy it to the others. GeoViewer and 3D MPM Studio only record the
drilling history (Drillhole Manager); the prediction is dfpredict's alone.

All tables are in the project database (geodatabase.sqlite or MySQL), prefix drillhole_:

    drillhole_collar      one row per hole: GeoViewer's x, y, z, max_depth, organization, drilling_manager,
                          geology_surveyor, start_date, end_date, state, plus the drilling fields area_id,
                          rig_model, plan_azimuth, plan_dip, planned_depth, casing_available,
                          fluids_available, crew, notes
    drillhole_survey      survey stations                                       (unchanged)
    drillhole_geology     geology log: rock, note, plus formation, weathering, broken, bedding_angle
    drillhole_assay       grades                                                (unchanged, GeoViewer)
    drillhole_styles, drillhole_translation   rock colours, code descriptions   (unchanged)
    drillhole_hole_size   hole size by depth (each reduction usually marks casing)          per hole
    drillhole_bit         bit type by depth                                                 per hole
    drillhole_run         drilling parameters of each core run                              per hole
    drillhole_occurrence  cavities, fault zones and other problems, measures, solved Y/N    per hole
    drillhole_failure     the failure of a failed hole (one row)                            per hole
    drillhole_area        site sheet of each drilling area, old or new
    drillhole_formation   formation property cards
    drillhole_rig         rig specifications
    drillhole_measure     the measure library (predefined, editable)

Only the risk system's own data keeps the prefix risk_ (dfpredict: rock properties, rock codes and
the warnings it issued).

Conventions, the same in both programs:
* depths along the hole in metres from the collar; blank = unknown (never 0);
* dips as in drillhole_survey - degrees from straight down: 0 = down, 90 = horizontal. The planned
  dip of the collar (plan_dip) too;
* max_depth is the depth reached: the length of a finished hole, the bit depth of a hole being
  drilled, 0 or blank for a planned hole. planned_depth is the designed (target) depth;
* state is THE status of a hole (HOLE_STATES, stored as a small number): Planned, In work, Completed with
  ore hit, Completed without ore hit, Failed. The drilling result (success / failed / in progress / planned)
  follows from it (``result_of_state``); a success whose ore hit is not known is "without ore hit";
* each program writes only the columns it knows and keeps the others: a hole saved in one program
  keeps the fields of the other (``upsert_collar``, ``write_hole_rows``).

The exchange format - the CSV files of the drilling history (design spec 3.4) and dfpredict's
records in memory - writes dips negative downward (-90 = straight down), final_depth for
max_depth, result next to state and notes for the geology note: ``to_exchange`` and
``from_exchange`` convert.

Databases of earlier dfpredict versions (risk_collar, risk_survey, ...) are moved into these tables
by ``prepare``, once; the old tables are kept, renamed risk_<name>_old.
No Qt here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..i18n import N_, tr

PREFIX = "drillhole_"

# --- hole state and drilling result ---------------------------------------------------------------
IN_WORK, ORE_HIT, NO_ORE_HIT, FAILED, PLANNED = 1, 2, 3, 4, 5
# stored as the number; listed in this order (the life of a hole)
HOLE_STATES = {PLANNED: N_("Planned"), IN_WORK: N_("In work"), ORE_HIT: N_("Completed with ore hit"),
               NO_ORE_HIT: N_("Completed without ore hit"), FAILED: N_("Failed")}
LEGACY_STATES = {6: NO_ORE_HIT}      # 6 "Completed" (ore hit not recorded) of an earlier version
RESULTS = {"success": N_("Success"), "failed": N_("Failed"), "in_progress": N_("In progress"),
           "planned": N_("Planned")}
STATE_RESULT = {IN_WORK: "in_progress", ORE_HIT: "success", NO_ORE_HIT: "success", FAILED: "failed",
                PLANNED: "planned"}
RESULT_STATE = {"in_progress": IN_WORK, "success": NO_ORE_HIT, "failed": FAILED, "planned": PLANNED}


def _state_number(v) -> int | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f != int(f):
        return None
    n = LEGACY_STATES.get(int(f), int(f))
    return n if n in HOLE_STATES else None


def result_of_state(state) -> str | None:
    """The drilling result of a stored state (None: no state)."""
    n = _state_number(state)
    return STATE_RESULT.get(n) if n is not None else None


def state_of_result(result, current=None) -> int | None:
    """The state to store for a drilling result. ``success`` keeps a completed state already
    stored (with / without ore hit), else it is NO_ORE_HIT (whether ore was hit is not known)."""
    r = str(result or "").strip().lower().replace(" ", "_")
    r = {"fail": "failed", "failure": "failed", "inprogress": "in_progress", "drilling": "in_progress"}.get(r, r)
    if r not in RESULT_STATE:
        return None
    cur = _state_number(current)
    if r == "success" and cur in (ORE_HIT, NO_ORE_HIT):
        return cur
    return RESULT_STATE[r]


# --- code lists -----------------------------------------------------------------------------------
YES_NO = {"Y": N_("Yes"), "N": N_("No")}
FAIL_CODES = {
    "LC": N_("Lost circulation"), "CV": N_("Cavity encounter / rod drop"), "WI": N_("Water / gas influx"),
    "CL": N_("Hole collapse / caving"), "SP": N_("Stuck rods / stuck pipe"),
    "CB": N_("Core blockage / poor recovery"), "DV": N_("Hole deviation / dogleg"),
    "BG": N_("Bit glazing / polishing"), "BW": N_("Excessive bit wear / bit burn"), "RF": N_("Rod or string failure"),
}
SEVERITY = {"1": N_("1 - minor (hours)"), "2": N_("2 - moderate (shift)"), "3": N_("3 - serious (days)"),
            "4": N_("4 - severe (redrill part)"), "5": N_("5 - hole lost")}
OCC_TYPES = {"cavity": N_("Cavity / karst"), "fault": N_("Fault zone"), "broken": N_("Broken / crushed rock"),
             "clay": N_("Swelling clay"), "water": N_("Water / gas inflow"), "loss": N_("Fluid loss"),
             "other": N_("Other problem")}
BIT_TYPES = {"impregnated": N_("Impregnated diamond"), "surface_set": N_("Surface-set diamond"),
             "pdc": N_("PDC"), "carbide": N_("Tungsten carbide"), "steel": N_("Steel")}
HOLE_SIZES = {"PQ": 122.6, "HQ": 96.0, "NQ": 75.7, "BQ": 60.0, "AQ": 48.0}     # wireline core: hole mm
FLUID_TYPES = {"water": N_("Water"), "polymer": N_("Polymer"), "bentonite": N_("Bentonite mud"),
               "kcl_polymer": N_("KCl-polymer (inhibitive)"), "foam": N_("Foam / air")}
BEDDING_STYLES = {"massive": N_("Massive"), "thick": N_("Thick-bedded"), "thin": N_("Thin-interbedded"),
                  "alternation": N_("Hard-soft alternation")}
CLAY_LEVELS = {"none": N_("None"), "interbeds": N_("Interbeds"), "common": N_("Common")}
KARST_LEVELS = {"none": N_("None"), "vuggy": N_("Vuggy"), "cavernous": N_("Cavernous")}
MEASURE_STAGES = {"P": N_("P - prevention"), "R": N_("R - response")}
# Rock classes of the formation cards (dfpredict keeps their mechanical properties in risk_rock and may add
# classes there; ``rock_classes`` reads them when that table is present).
ROCK_CLASSES = {
    "conglomerate": N_("Conglomerate"), "breccia_sed": N_("Sedimentary breccia"), "sandstone": N_("Sandstone"),
    "quartz_arenite": N_("Quartz arenite (orthoquartzite)"), "greywacke": N_("Greywacke"), "arkose": N_("Arkose"),
    "siltstone": N_("Siltstone"), "mudstone": N_("Mudstone / claystone"), "shale": N_("Shale"),
    "black_shale": N_("Black (carbonaceous) shale"), "limestone": N_("Limestone"), "dolomite": N_("Dolomite"),
    "siliceous_dolomite": N_("Siliceous / cherty dolomite"), "marl": N_("Marl"), "chalk": N_("Chalk"),
    "travertine": N_("Travertine / tufa"), "chert": N_("Chert / flint"),
    "ironstone": N_("Ironstone / banded iron formation"), "gypsum": N_("Gypsum"), "anhydrite": N_("Anhydrite"),
    "halite": N_("Rock salt (halite)"), "coal": N_("Coal"), "quartzite": N_("Quartzite (metamorphic)"),
    "metasandstone": N_("Metasandstone / metasiltstone"), "marble": N_("Marble"), "slate": N_("Slate"),
    "phyllite": N_("Phyllite"), "schist": N_("Mica schist"), "quartz_schist": N_("Quartz-mica schist"),
    "calc_schist": N_("Calc-schist"), "gneiss": N_("Gneiss"), "amphibolite": N_("Amphibolite"),
    "hornfels": N_("Hornfels"), "serpentinite": N_("Serpentinite"), "granite": N_("Granite"),
    "granodiorite": N_("Granodiorite / diorite"), "gabbro": N_("Gabbro / norite"), "basalt": N_("Basalt / dolerite"),
    "andesite": N_("Andesite / rhyolite"), "tuff": N_("Tuff"), "pegmatite": N_("Pegmatite / quartz vein"),
    "sulphide": N_("Massive / disseminated sulphide"), "breccia": N_("Fault breccia / cataclasite"),
    "mylonite": N_("Mylonite"), "clay": N_("Fault gouge / clay / cave fill"),
    "saprolite": N_("Saprolite / weathered rock"),
}

# The measure library (design spec 6.1): predefined, editable in drillhole_measure. Values in braces are
# filled in by the risk system ({from_}, {to}, {depth}, {rpm}, {q}, {wob}, ...).
DEFAULT_MEASURES = (   # code, short title, text, failure codes, stage (P prevention / R response)
    ("M-OP-01", N_("Reduce bit load, feed slowly"),
     N_("Reduce bit load to {wob} kN and feed slowly from {from_} to {to} m"), "CV, DV, CL", "P"),
    ("M-OP-02", N_("Reduce RPM"),
     N_("Reduce RPM to {rpm}; avoid the vibration band {band}"), "RF, DV, CB", "P"),
    ("M-OP-03", N_("Reduce pump flow"),
     N_("Reduce pump flow to {q} L/min, the minimum for bit cooling"), "LC, CL", "P"),
    ("M-OP-04", N_("Short core runs"),
     N_("Shorten core runs to 1.5 m; pull the inner tube at any pressure rise"), "CB, CL", "P"),
    ("M-OP-05", N_("Keep circulating and rotating"),
     N_("Keep circulating and rotating; never leave the string still in {from_}-{to} m"), "SP, CL", "P"),
    ("M-OP-06", N_("Trip slowly"),
     N_("Trip slowly (≤ {v} m/min) through {from_}-{to} m"), "CL, LC", "P"),
    ("M-OP-07", N_("Proven operating window"),
     N_("Use the operating window of past successful runs: RPM {rpm}, flow {q} L/min, ROP {rop} m/h"),
     "LC, CV, WI, CL, SP, CB, DV, BG, BW, RF", "P"),
    ("M-FL-01", N_("Change drilling fluid"),
     N_("Switch to {fluid} mud, density {rho} g/cm³, Marsh viscosity {s} s, before {depth} m"), "CL, SP, LC", "P"),
    ("M-FL-02", N_("Pre-treat with fine LCM"),
     N_("Pre-treat fluid with fine LCM {c} kg/m³"), "LC", "P"),
    ("M-FL-03", N_("Clay inhibitor"),
     N_("Add clay inhibitor (KCl / PHPA) for swelling interbeds"), "CL, SP", "P"),
    ("M-CS-01", N_("Set casing before the zone"),
     N_("Set casing to {depth} m (or reduce hole size {size_from}→{size_to}) before entering the zone"),
     "LC, CL, CV", "P"),
    ("M-CS-02", N_("Casing, cement and LCM on site"),
     N_("Have casing, cement and LCM on site before {depth} m"), "LC, CV, CL", "P"),
    ("M-BH-01", N_("Triple-tube barrel"),
     N_("Use a triple-tube or split-tube barrel through broken rock"), "CB, CL", "P"),
    ("M-BH-02", N_("Stiffen the assembly"),
     N_("Stiffen the assembly: longer barrel, reaming shell and stabilizers"), "DV", "P"),
    ("M-BH-03", N_("Change bit matrix"),
     N_("Change bit: softer matrix for quartzite; harder matrix for abrasive broken sandstone"), "BG, BW", "P"),
    ("M-SV-01", N_("Survey more often"),
     N_("Survey every {x} m through the zone"), "DV", "P"),
    ("M-PL-01", N_("Re-plan the hole"),
     N_("Re-plan: change azimuth or dip so the hole meets bedding at > 45°, or move the collar away from the "
        "risk volume"), "DV, LC, CV", "P"),
    ("M-LC-01", N_("Classify the loss"),
     N_("Cut flow, record loss rate and pressure, classify loss (seepage / partial / total)"), "LC", "R"),
    ("M-LC-02", N_("Fine LCM pill"),
     N_("Pump fine-to-medium LCM pill, {c} kg/m³"), "LC", "R"),
    ("M-LC-03", N_("Coarse LCM pill"),
     N_("Pump coarse fibre or granular LCM pill"), "LC", "R"),
    ("M-LC-04", N_("Sealing gel"),
     N_("Spot a sealing gel or silicate system"), "LC", "R"),
    ("M-LC-05", N_("Drill blind to casing point"),
     N_("Drill ahead without returns using water only, if the hole is stable and the casing point is within {x} m; "
        "watch bit temperature"), "LC", "R"),
    ("M-LC-06", N_("Cement plug and redrill"),
     N_("Set a cement plug, wait on cement, redrill"), "LC, CV", "R"),
    ("M-LC-07", N_("Case through"),
     N_("Run casing through the cavity or loss zone"), "LC, CV", "R"),
    ("M-CV-01", N_("Stop and measure cavity"),
     N_("Stop feed; measure cavity height by rod drop; survey"), "CV", "R"),
    ("M-CV-02", N_("Grout the cavity and redrill"),
     N_("Fill cavity with grout or cement and redrill"), "CV", "R"),
    ("M-CL-01", N_("Raise viscosity"),
     N_("Raise mud viscosity and density; circulate the hole clean"), "CL", "R"),
    ("M-CL-02", N_("Ream and wiper trip"),
     N_("Ream the interval and make a short wiper trip"), "CL, SP", "R"),
    ("M-CL-03", N_("Consolidate with cement"),
     N_("Consolidate with cement and redrill"), "CL", "R"),
    ("M-SP-01", N_("Work the string"),
     N_("Keep circulation; work the string up and down; rotate within torque limit {t}"), "SP", "R"),
    ("M-SP-02", N_("Lubricant pill"),
     N_("Spot a lubricant or freshwater pill"), "SP", "R"),
    ("M-SP-03", N_("Jar / pull"),
     N_("Apply jarring or hydraulic pull within the rod pull limit {f} kN"), "SP", "R"),
    ("M-SP-04", N_("Fish or sidetrack"),
     N_("Back off, fish, or sidetrack"), "SP", "R"),
    ("M-DV-01", N_("Wedge / directional correction"),
     N_("Correct with a wedge or directional tool"), "DV", "R"),
    ("M-BG-01", N_("Dress the bit"),
     N_("Dress the bit (abrasive pass), then raise bit load briefly"), "BG", "R"),
    ("M-WI-01", N_("Control the inflow"),
     N_("Raise fluid density; close the collar valve or set a packer"), "WI", "R"),
    ("M-RF-01", N_("Inspect rods"),
     N_("Inspect threads; replace worn rods; lower RPM"), "RF", "R"),
)
DEFAULT_RIGS = (   # generic wireline core rigs (examples: replace them with your fleet)
    {"rig_model": "Core rig 1000 m", "depth_PQ": 350, "depth_HQ": 700, "depth_NQ": 1000, "depth_BQ": 1300,
     "max_rpm": 1300, "pump_lpm": 120, "pull_kn": 90, "max_torque": 2400},
    {"rig_model": "Core rig 600 m", "depth_PQ": 200, "depth_HQ": 420, "depth_NQ": 620, "depth_BQ": 800,
     "max_rpm": 1000, "pump_lpm": 90, "pull_kn": 60, "max_torque": 1600},
)


def measure_rows() -> list[dict]:
    """The predefined rows of drillhole_measure."""
    return [{"code": c, "title": t, "text": x, "failures": f, "stage": s, "expert_order": i}
            for i, (c, t, x, f, s) in enumerate(DEFAULT_MEASURES)]


def default_reference() -> dict[str, list[dict]]:
    """The rows a new database's reference tables are filled with: table name -> rows."""
    return {MEASURE.name: measure_rows(), RIG.name: [dict(r) for r in DEFAULT_RIGS]}


# --- the tables -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Col:
    name: str
    sql: str                       # portable SQL type
    label: str                     # shown (translated with tr)
    kind: str = "real"             # real / int / text / code / yn / date / note
    choices: dict | None = None    # code -> text, for kind "code" ({}: filled at run time, e.g. the areas)
    unit: str = ""
    tip: str = ""
    lo: float | None = None        # physical limits (entry checks)
    hi: float | None = None
    required: bool = False         # a row without it is not saved
    risk: bool = False             # needed by the risk prediction (a warning when empty)
    width: int = 0                 # preferred column width in tables (0 = automatic)


@dataclass(frozen=True, eq=False)        # one object per table: compared and hashed by identity
class Table:
    name: str                      # in the database, e.g. drillhole_run
    title: str
    key: tuple                     # identifying columns
    cols: tuple
    per_hole: bool = True          # rows belong to a hole (hole_id first)
    interval: bool = False         # depth_from / depth_to rows
    help: str = ""

    @property
    def short(self) -> str:
        """The name without the prefix (also the CSV file name): run, collar, area, ..."""
        return self.name[len(PREFIX):] if self.name.startswith(PREFIX) else self.name

    def col(self, name: str) -> Col:
        return next(c for c in self.cols if c.name == name)

    def has(self, name: str) -> bool:
        return any(c.name == name for c in self.cols)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cols]


HOLE = Col("hole_id", "VARCHAR(64)", N_("Hole ID"), "text", required=True, width=110)
FROM = Col("depth_from", "DOUBLE", N_("From"), "real", unit="m", lo=0, hi=5000, required=True, width=70)
TO = Col("depth_to", "DOUBLE", N_("To"), "real", unit="m", lo=0, hi=5000, required=True, width=70)
NOTES = Col("notes", "TEXT", N_("Notes"), "note")

COLLAR = Table(PREFIX + "collar", N_("Collar"), ("hole_id",), (
    HOLE,
    Col("x", "DOUBLE", N_("X (East)"), required=True),
    Col("y", "DOUBLE", N_("Y (North)"), required=True),
    Col("z", "DOUBLE", N_("Z (Elevation)"), required=True),
    Col("max_depth", "DOUBLE", N_("Length (m)"), lo=0, hi=5000, risk=True,
        tip=N_("Depth reached: the length of a finished hole, the bit depth of a hole being drilled, 0 for a "
               "planned hole")),
    Col("organization", "VARCHAR(100)", N_("Organization"), "text"),
    Col("drilling_manager", "VARCHAR(100)", N_("Drilling manager"), "text"),
    Col("geology_surveyor", "VARCHAR(100)", N_("Geology surveyor"), "text"),
    Col("start_date", "VARCHAR(10)", N_("Start date"), "date"),
    Col("end_date", "VARCHAR(10)", N_("End date"), "date"),
    Col("state", "SMALLINT", N_("State"), "code", {str(k): v for k, v in HOLE_STATES.items()}, risk=True,
        tip=N_("Planned, in work, failed or completed: the failed and completed holes form the drilling history")),
    # --- the drilling fields (risk prediction) ---
    Col("area_id", "VARCHAR(32)", N_("Area"), "code", {}, risk=True, width=70,
        tip=N_("The drilling area (site sheet in Drilling reference > Areas)")),
    Col("rig_model", "VARCHAR(64)", N_("Rig model"), "code", {}),
    Col("plan_azimuth", "DOUBLE", N_("Planned azimuth"), unit="°", lo=0, hi=360,
        tip=N_("Designed azimuth at the collar, degrees clockwise from north")),
    Col("plan_dip", "DOUBLE", N_("Planned dip"), unit="°", lo=0, hi=180,
        tip=N_("Designed dip at the collar, degrees from straight down (0 = down, 90 = horizontal)")),
    Col("planned_depth", "DOUBLE", N_("Planned (target) depth"), unit="m", lo=0, hi=5000, risk=True),
    Col("casing_available", "VARCHAR(64)", N_("Casing on site"), "text",
        tip=N_("Casing sizes available at the rig, e.g. HW;NW (used to filter casing measures)")),
    Col("fluids_available", "VARCHAR(128)", N_("Fluids on site"), "text",
        tip=N_("Fluid systems available at the rig, e.g. water;polymer;bentonite")),
    Col("crew", "VARCHAR(64)", N_("Crew"), "text"),
    NOTES,
), per_hole=False)
COLLAR_ADDED = COLLAR.cols[11:]          # the drilling fields, added to a GeoViewer collar table

SURVEY = Table(PREFIX + "survey", N_("Survey"), ("hole_id", "depth"), (
    HOLE,
    Col("depth", "DOUBLE", N_("Depth"), unit="m", lo=0, hi=5000, required=True),
    Col("azimuth", "DOUBLE", N_("Azimuth"), unit="°", lo=0, hi=360, required=True),
    Col("dip", "DOUBLE", N_("Dip"), unit="°", lo=0, hi=180, required=True,
        tip=N_("Degrees from straight down: 0 = down, 90 = horizontal")),
))

GEOLOGY = Table(PREFIX + "geology", N_("Geology log"), ("hole_id", "depth_from"), (
    HOLE, FROM, TO,
    Col("rock", "VARCHAR(64)", N_("Rock"), "text", tip=N_("Rock code as logged, e.g. DO, LS, SS")),
    Col("note", "TEXT", N_("Note"), "note"),
    Col("formation", "VARCHAR(64)", N_("Formation"), "code", {}, risk=True,
        tip=N_("Links each metre to its property card")),
    Col("weathering", "INTEGER", N_("Weathering"), "int", lo=0, hi=5),
    Col("broken", "VARCHAR(1)", N_("Broken zone"), "yn", YES_NO),
    Col("bedding_angle", "DOUBLE", N_("Bedding to core axis"), unit="°", lo=0, hi=90,
        tip=N_("Angle between bedding and the core axis (90 = perpendicular to the hole)")),
), interval=True)
GEOLOGY_ADDED = GEOLOGY.cols[5:]

HOLE_SIZE = Table(PREFIX + "hole_size", N_("Hole size"), ("hole_id", "depth_from"), (
    HOLE, FROM, TO,
    Col("size", "VARCHAR(8)", N_("Size"), "code", {k: k for k in HOLE_SIZES}, required=True),
    Col("diameter", "DOUBLE", N_("Diameter"), unit="mm", lo=20, hi=400),
), interval=True, help=N_("Each reduction usually marks casing being set."))

BIT = Table(PREFIX + "bit", N_("Bit"), ("hole_id", "depth_from"), (
    HOLE, FROM, TO,
    Col("bit_type", "VARCHAR(20)", N_("Bit type"), "code", BIT_TYPES, required=True),
    Col("matrix", "VARCHAR(20)", N_("Matrix / series"), "text", tip=N_("e.g. series 7 (soft) ... 10 (hard rock)")),
), interval=True)

RUN = Table(PREFIX + "run", N_("Runs (drilling parameters)"), ("hole_id", "depth_from"), (
    HOLE, FROM, TO,
    Col("rpm", "DOUBLE", N_("RPM"), unit="rpm", lo=0, hi=3000),
    Col("rop", "DOUBLE", N_("ROP"), unit="m/h", lo=0, hi=200),
    Col("flow_in", "DOUBLE", N_("Flow in"), unit="L/min", lo=0, hi=1000),
    Col("pump_press", "DOUBLE", N_("Pump pressure"), unit="MPa", lo=0, hi=40),
    Col("return_pct", "DOUBLE", N_("Return"), unit="%", lo=0, hi=120, tip=N_("Estimated return flow (new field)")),
    Col("rod_drop", "DOUBLE", N_("Rod drop"), unit="m", lo=0, hi=50, tip=N_("Direct sign of a cavity (new field)")),
    Col("wob", "DOUBLE", N_("Bit load (WOB)"), unit="kN", lo=0, hi=200, tip=N_("If the rig has gauges")),
    Col("torque", "DOUBLE", N_("Torque"), unit="N·m", lo=0, hi=20000, tip=N_("If the rig has gauges")),
    Col("fluid_type", "VARCHAR(20)", N_("Fluid"), "code", FLUID_TYPES),
    Col("recovery", "DOUBLE", N_("Core recovery"), unit="%", lo=0, hi=110),
    Col("remarks", "VARCHAR(255)", N_("Remarks"), "text"),
), interval=True, help=N_("One row per core run (1.5-6 m). Per-run values support the live warnings."))

OCCURRENCE = Table(PREFIX + "occurrence", N_("Occurrences"), ("hole_id", "depth_from", "type"), (
    HOLE, FROM, TO,
    Col("type", "VARCHAR(16)", N_("Type"), "code", OCC_TYPES, required=True),
    Col("measures", "VARCHAR(255)", N_("Measures"), "text", tip=N_("Measure codes separated by ;")),
    Col("solved", "VARCHAR(1)", N_("Solved"), "yn", YES_NO),
    NOTES,
), interval=True, help=N_("Cavities, fault zones and other problems, solved or not: they train the hazard model "
                          "and give the measured success rates of the measures."))

FAILURE = Table(PREFIX + "failure", N_("Failure"), ("hole_id",), (
    HOLE,
    Col("fail_depth", "DOUBLE", N_("Failure depth"), unit="m", lo=0, hi=5000, required=True),
    Col("fail_code", "VARCHAR(4)", N_("Failure code"), "code", {k: f"{k} - {v}" for k, v in FAIL_CODES.items()},
        required=True),
    Col("severity", "INTEGER", N_("Severity"), "code", SEVERITY, required=True),
    Col("measures", "VARCHAR(255)", N_("Measures tried"), "text", tip=N_("Measure codes separated by ;")),
    NOTES,
), help=N_("One row per failed hole: the failure label."))

AREA = Table(PREFIX + "area", N_("Areas (site sheets)"), ("area_id",), (
    Col("area_id", "VARCHAR(32)", N_("Area ID"), "text", required=True, width=80),
    Col("name", "VARCHAR(100)", N_("Name"), "text", width=150),
    Col("dip_min", "DOUBLE", N_("Bedding dip min"), unit="°", lo=0, hi=90,
        tip=N_("Range of bedding dips in the area (structural style)")),
    Col("dip_max", "DOUBLE", N_("Bedding dip max"), unit="°", lo=0, hi=90),
    Col("fold_tightness", "INTEGER", N_("Fold tightness"), "code",
        {"1": N_("1 - open"), "2": N_("2 - moderate"), "3": N_("3 - tight")}),
    Col("faults_per_km", "DOUBLE", N_("Faults per km"), lo=0, hi=50),
    Col("carbonate_exposed", "VARCHAR(1)", N_("Carbonate exposed"), "yn", YES_NO,
        tip=N_("Carbonate outcrops at surface (Y) or is covered (N)")),
    Col("unconformity", "VARCHAR(1)", N_("Unconformity above carbonate"), "yn", YES_NO),
    Col("sinkholes", "VARCHAR(1)", N_("Sinkholes mapped"), "yn", YES_NO),
    Col("carb_above_valley", "DOUBLE", N_("Carbonate height above valley floor"), unit="m", lo=-1000, hi=2000,
        tip=N_("Height of the top of the carbonate above the local valley floor (karst base level)")),
    Col("water_table", "DOUBLE", N_("Regional water table elevation"), unit="m", lo=-2000, hi=6000,
        tip=N_("Elevation of the regional water table, if known (strong karst indicator)")),
    Col("depth_of_interest", "DOUBLE", N_("Depth of interest"), unit="m", lo=0, hi=5000),
    NOTES,
), per_hole=False, help=N_("One row per drilling location, old or new. Keep it to these few fields: the models "
                           "learn each one from only as many examples as there are areas."))

FORMATION = Table(PREFIX + "formation", N_("Formation cards"), ("formation",), (
    Col("formation", "VARCHAR(64)", N_("Formation"), "text", required=True, width=110,
        tip=N_("One name per formation everywhere, also in the geological model")),
    Col("age", "VARCHAR(40)", N_("Age / system"), "text", tip=N_("e.g. Cambrian, Ordovician")),
    Col("main_rock", "VARCHAR(20)", N_("Main rock"), "code", ROCK_CLASSES, required=True),
    Col("main_pct", "DOUBLE", N_("Main rock share"), unit="%", lo=0, hi=100),
    Col("second_rock", "VARCHAR(20)", N_("Secondary rock"), "code", ROCK_CLASSES),
    Col("second_pct", "DOUBLE", N_("Secondary share"), unit="%", lo=0, hi=100),
    Col("carbonate_pct", "DOUBLE", N_("Carbonate share"), unit="%", lo=0, hi=100, tip=N_("Karst potential")),
    Col("hardness", "INTEGER", N_("Hardness class"), "int", lo=1, hi=5, tip=N_("1 soft - 5 very hard")),
    Col("abrasiveness", "INTEGER", N_("Abrasiveness class"), "int", lo=1, hi=5, tip=N_("1 low - 5 very abrasive")),
    Col("bedding", "VARCHAR(20)", N_("Bedding style"), "code", BEDDING_STYLES),
    Col("swelling_clay", "VARCHAR(20)", N_("Swelling / weak clay"), "code", CLAY_LEVELS),
    Col("karst", "VARCHAR(20)", N_("Known karst / vugs"), "code", KARST_LEVELS),
    Col("ucs_mpa", "DOUBLE", N_("UCS (lab)"), unit="MPa", lo=0.1, hi=500,
        tip=N_("Measured uniaxial compressive strength of the formation; blank = the rock class default")),
    Col("cai_lab", "DOUBLE", N_("CAI (lab)"), lo=0, hi=7,
        tip=N_("Measured Cerchar abrasivity index; blank = the rock class default")),
    Col("thick_min", "DOUBLE", N_("Thickness min"), unit="m", lo=0, hi=3000),
    Col("thick_max", "DOUBLE", N_("Thickness max"), unit="m", lo=0, hi=3000,
        tip=N_("Sets the depth-error band for contacts below this formation")),
    Col("equivalents", "VARCHAR(255)", N_("Equivalent formations"), "text",
        tip=N_("Names of the same formation in other areas, separated by ;")),
    NOTES,
), per_hole=False, help=N_("A geologist fills in one card per formation, once. The models predict a formation never "
                           "drilled before from its card, so learning transfers between areas."))

RIG = Table(PREFIX + "rig", N_("Rig specifications"), ("rig_model",), (
    Col("rig_model", "VARCHAR(64)", N_("Rig model"), "text", required=True, width=120),
    Col("depth_PQ", "DOUBLE", N_("Rated depth PQ"), unit="m", lo=0, hi=5000),
    Col("depth_HQ", "DOUBLE", N_("Rated depth HQ"), unit="m", lo=0, hi=5000),
    Col("depth_NQ", "DOUBLE", N_("Rated depth NQ"), unit="m", lo=0, hi=5000),
    Col("depth_BQ", "DOUBLE", N_("Rated depth BQ"), unit="m", lo=0, hi=5000),
    Col("max_rpm", "DOUBLE", N_("Max RPM"), unit="rpm", lo=0, hi=3000),
    Col("pump_lpm", "DOUBLE", N_("Pump capacity"), unit="L/min", lo=0, hi=1000),
    Col("pull_kn", "DOUBLE", N_("Rod pull limit"), unit="kN", lo=0, hi=1000),
    Col("max_torque", "DOUBLE", N_("Max torque"), unit="N·m", lo=0, hi=20000),
), per_hole=False, help=N_("Turns the rig model into 'how close to its limit'."))

MEASURE = Table(PREFIX + "measure", N_("Measure library"), ("code",), (
    Col("code", "VARCHAR(16)", N_("Code"), "text", required=True, width=85),
    Col("title", "VARCHAR(80)", N_("Short title"), "text", width=200),
    Col("text", "VARCHAR(255)", N_("Measure text"), "text", required=True, width=480,
        tip=N_("Values in braces are filled in by the system: {from_}, {to}, {depth}, {rpm}, {q}, {wob}, ...")),
    Col("failures", "VARCHAR(64)", N_("Failures"), "text", width=110, tip=N_("Failure codes, e.g. LC, CV")),
    Col("stage", "VARCHAR(1)", N_("Stage"), "code", MEASURE_STAGES),
    Col("expert_order", "INTEGER", N_("Expert order"), "int", lo=0, hi=999,
        tip=N_("Priority used until outcomes build up (lower = offered first)")),
), per_hole=False, help=N_("The pre-defined measures (spec 6.1). Edit the texts to your crews' practice; keep the "
                           "codes. The decision table picks them by failure code and alarm level."))

RECORD_TABLES = (RUN, HOLE_SIZE, BIT, OCCURRENCE, FAILURE)      # per-hole drilling records (new tables)
REFERENCE_TABLES = (AREA, FORMATION, RIG, MEASURE)
NEW_TABLES = RECORD_TABLES + REFERENCE_TABLES                   # made by ``prepare``
SHARED_TABLES = (COLLAR, SURVEY, GEOLOGY) + NEW_TABLES          # what both programs read and write
HOLE_RECORDS = (SURVEY, GEOLOGY) + RECORD_TABLES                # a hole's rows, in tab order
BY_NAME = {t.name: t for t in SHARED_TABLES}
BY_SHORT = {t.short: t for t in SHARED_TABLES}
LEGACY_PREFIX = "risk_"                                         # dfpredict before the shared tables


def table(name: str) -> Table:
    """A shared table by its name, with or without the prefix (run, drillhole_run)."""
    low = str(name).lower()
    return BY_NAME.get(low) or BY_SHORT[low]


# --- conventions of the exchange format (CSV, dfpredict's records in memory) ----------------------------------
_EX_RENAME = {COLLAR.name: {"max_depth": "final_depth"}, GEOLOGY.name: {"note": "notes"}}
_DIP_COLS = {COLLAR.name: "plan_dip", SURVEY.name: "dip"}


def exchange_names(t: Table) -> list[str]:
    """The columns of a table in the exchange format (collar: final_depth, ..., state and result)."""
    ren = _EX_RENAME.get(t.name, {})
    out = [ren.get(c, c) for c in t.names]
    if t is COLLAR:
        out.insert(out.index("state") + 1, "result")
    return out


def to_exchange(t: Table, df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a table as stored -> the exchange format: dips negative downward, final_depth,
    result (from state), notes. Other columns are kept as they are."""
    df = df.copy()
    dip = _DIP_COLS.get(t.name)
    if dip in df:
        df[dip] = (pd.to_numeric(df[dip], errors="coerce") - 90.0).round(6)
    if t is COLLAR and "state" in df:
        df["result"] = [result_of_state(v) for v in df["state"]]
        df["state"] = [_state_number(v) for v in df["state"]]
    return df.rename(columns=_EX_RENAME.get(t.name, {}))


def from_exchange(t: Table, df: pd.DataFrame, current_states: dict | None = None) -> pd.DataFrame:
    """Rows in the exchange format -> as stored (the inverse of ``to_exchange``). The collar's state
    is its state column where valid, else it follows from result (``current_states``: hole ->
    state stored now, so that 'success' keeps 'with ore hit' / 'without ore hit')."""
    df = df.rename(columns={v: k for k, v in _EX_RENAME.get(t.name, {}).items()})
    dip = _DIP_COLS.get(t.name)
    if dip in df:
        df[dip] = (pd.to_numeric(df[dip], errors="coerce") + 90.0).round(6)
    if t is COLLAR and ("state" in df or "result" in df):
        cur = current_states or {}
        states = []
        for rec in df.to_dict("records"):
            s = _state_number(rec.get("state"))
            res = rec.get("result")
            if not is_blank(res):
                want = result_of_state(state_of_result(res))     # the result, normalised (None: unknown)
                if want is not None and result_of_state(s) != want:
                    s = state_of_result(res, s if s is not None else cur.get(rec.get("hole_id")))
            states.append(s)
        df["state"] = states
        df = df.drop(columns=["result"], errors="ignore")
    return df


# --- values ---------------------------------------------------------------------------------------
def is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip()) or (
        v is pd.NA) or (isinstance(v, (np.floating,)) and np.isnan(v))


def db_value(c: Col | None, v):
    """A value for the database: blank -> NULL (never 0); numbers as float / int; text stripped."""
    if is_blank(v):
        return None
    if c is not None and (c.kind in ("real", "int") or c.sql in ("SMALLINT", "INTEGER")):
        f = _num(v)                                  # INTEGER codes too: severity "3" -> 3
        if f is None:
            return None if c.kind in ("real", "int") else str(v).strip()
        return int(round(f)) if c.sql in ("SMALLINT", "INTEGER") else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, str):
        return v.strip()
    return v


def _key_text(v) -> str:
    """A depth / key as text for matching rows (1.0 and 1 are the same)."""
    if is_blank(v):
        return ""
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v).strip()


# --- column names as stored ---------------------------------------------------------------------
def stored_columns(db, table_name: str) -> dict[str, str]:
    """Schema column -> the column as stored in ``table_name``, for the columns the table has.
    A GeoViewer table made by an import may name the hole id 'BHID' or the depths 'from' / 'to'
    (recognised as when drillholes are read); other columns are matched ignoring case."""
    from .drillholes import _FROM, _HOLE, _MAXDEPTH, _SDEPTH, _TO, _X, _Y, _Z, _find

    cols = db.columns(table_name)
    low = {c.lower(): c for c in cols}
    t = BY_NAME.get(table_name.lower())
    out: dict[str, str] = {}
    roles = {"hole_id": _HOLE, "depth_from": _FROM, "depth_to": _TO}
    if t is COLLAR:
        roles.update(x=_X, y=_Y, z=_Z, max_depth=_MAXDEPTH)
    elif t is SURVEY:
        roles["depth"] = _SDEPTH
    elif t is GEOLOGY:
        roles.update(rock=("rock", "lith", "lithology", "litho", "code"),
                     note=("note", "notes", "comment", "comments", "remarks"))
    for name, aliases in roles.items():
        c = low.get(name) or _find(cols, aliases, False, name)
        if c is not None and c not in out.values():
            out[name] = c
    for c in (t.cols if t is not None else ()):
        if c.name not in out and c.name.lower() in low:
            out[c.name] = low[c.name.lower()]
    return out


# --- set-up -------------------------------------------------------------------------------------
def ddl(t: Table, mysql: bool) -> str:
    q = (lambda n: f"`{n}`") if mysql else (lambda n: f'"{n}"')
    cols = [f"{q(c.name)} {c.sql}" + (" NOT NULL" if c.required and c.name in t.key else "") for c in t.cols]
    if len(t.key) == 1:
        cols.append(f"PRIMARY KEY ({q(t.key[0])})")
    opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if mysql else ""
    return f"CREATE TABLE {t.name} ({', '.join(cols)}){opts}"


def _add_columns(db, table_name: str, cols) -> list[str]:
    have = {c.lower() for c in db.columns(table_name)}
    notes = []
    for c in cols:
        if c.name.lower() not in have:
            db.execute(f"ALTER TABLE {db.q(table_name)} ADD COLUMN {db.q(c.name)} {c.sql}")
            notes.append(tr("added field {name} to {table}", name=c.name, table=table_name))
    return notes


def prepare(db) -> list[str]:
    """Make what the shared drilling tables lack - never drops a table, a column or a row:
    the drilling fields of drillhole_collar and drillhole_geology (which must exist: GeoViewer's
    drillhole store makes them), the new tables, an index on each hole id; the records of an
    earlier dfpredict database (risk_*) moved in; the measure library and example rigs in empty
    tables. ``db``: an open Database. Returns what was done."""
    notes: list[str] = []
    have = {t.lower(): t for t in db.tables()}
    for t, cols in ((COLLAR, COLLAR_ADDED), (GEOLOGY, GEOLOGY_ADDED)):
        if t.name in have:
            notes += _add_columns(db, have[t.name], cols)
    for t in NEW_TABLES:
        if t.name not in have:
            db.execute(ddl(t, db.is_mysql))
            notes.append(tr("made table {name}", name=t.name))
        else:
            notes += _add_columns(db, have[t.name], t.cols)
        if t.per_hole:
            db.ensure_index(t.name, f"idx_{t.name}_hole", ["hole_id"])
    db.con.commit()
    notes += _convert_legacy_states(db)
    notes += migrate_legacy(db)
    for name, rows in default_reference().items():
        t = BY_NAME[name]
        if not db.scalar(f"SELECT COUNT(*) FROM {db.q(t.name)}", default=0):
            with db.transaction():
                insert_rows(db, t, pd.DataFrame(rows))
            notes.append(tr("filled {table} with {n} predefined rows", table=t.name, n=len(rows)))
    return notes


def _convert_legacy_states(db) -> list[str]:
    """States of an earlier version (6 "Completed", ore hit not recorded) -> the current ones."""
    name = db.table_name(COLLAR.name)
    if name is None or "state" not in {c.lower() for c in db.columns(name)}:
        return []
    sc = stored_columns(db, name)
    notes = []
    with db.transaction():
        for old, new in LEGACY_STATES.items():
            n = db.scalar(f"SELECT COUNT(*) FROM {db.q(name)} WHERE {db.q(sc['state'])} = ?", (old,), default=0)
            if n:
                db.execute(f"UPDATE {db.q(name)} SET {db.q(sc['state'])} = ? WHERE {db.q(sc['state'])} = ?", (new, old))
                notes.append(tr("the state of {n} holes was set to {state}", n=n, state=tr(HOLE_STATES[new])))
    return notes


# --- reading ------------------------------------------------------------------------------------
def read_table(db, t: Table, hole_ids=None) -> pd.DataFrame:
    """The rows of a shared table as stored, with the schema's column names (columns the table
    lacks are blank; columns that are not in the schema are left out). ``hole_ids``: only those."""
    name = db.table_name(t.name)
    if name is None:
        return pd.DataFrame(columns=t.names)
    sc = stored_columns(db, name)
    cols = [c for c in t.names if c in sc]
    sql = f"SELECT {', '.join(db.q(sc[c]) for c in cols)} FROM {db.q(name)}"
    params: list = []
    if hole_ids is not None:
        ids = [str(h) for h in hole_ids]
        if not ids:
            return pd.DataFrame(columns=t.names)
        if len(ids) <= 500:
            sql += f" WHERE {db.q(sc['hole_id'])} IN ({', '.join('?' * len(ids))})"
            params = ids
    df = pd.DataFrame.from_records(db.query(sql, params), columns=cols)
    if hole_ids is not None and not params:
        df = df[df["hole_id"].astype(str).isin(set(map(str, hole_ids)))]
    for c in t.cols:
        if c.name not in df:
            df[c.name] = None
        elif c.kind in ("real", "int"):
            df[c.name] = pd.to_numeric(df[c.name], errors="coerce")
        elif c.name == "state":
            df[c.name] = [_state_number(v) for v in df[c.name]]
        else:
            df[c.name] = [None if is_blank(v) else str(v) if not isinstance(v, str) else v for v in df[c.name]]
    if "hole_id" in df:
        df["hole_id"] = df["hole_id"].astype(str)
    sort = [k for k in ("hole_id", "depth_from", "depth") if k in df and t.per_hole]
    if sort:
        df = df.sort_values(sort, kind="stable")
    return df[t.names].reset_index(drop=True)


def area_ids(db) -> list[str]:
    """The areas: drillhole_area's and those used in drillhole_collar."""
    out: set[str] = set()
    for t, col in ((AREA, "area_id"), (COLLAR, "area_id")):
        name = db.table_name(t.name)
        if name is not None and col in {c.lower() for c in db.columns(name)}:
            out |= {str(r[0]) for r in db.query(f"SELECT DISTINCT {db.q(col)} FROM {db.q(name)}") if not is_blank(r[0])}
    return sorted(out, key=natural_key)


def rock_classes(db) -> dict[str, str]:
    """Rock class -> name for the formation cards: dfpredict's risk_rock where present, else the defaults."""
    name = db.table_name("risk_rock")
    if name is not None:
        rows = db.query(f"SELECT {db.q('rock_class')}, {db.q('name')} FROM {db.q(name)}")
        got = {str(k): str(n or k) for k, n in rows if not is_blank(k)}
        if got:
            return got
    return dict(ROCK_CLASSES)


def natural_key(s):
    """Sort key: WD2 before WD10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


# --- writing that keeps the other program's columns ---------------------------------------------------
def insert_rows(db, t: Table, df: pd.DataFrame, table_name: str | None = None) -> int:
    """Insert rows (schema column names; only the columns ``df`` has) into a table."""
    name = table_name or db.table_name(t.name) or t.name
    sc = stored_columns(db, name)
    cols = [c for c in df.columns if c in sc]
    rows = [[db_value(t.col(c) if t.has(c) else None, v) for c, v in zip(cols, r)]
            for r in df[cols].itertuples(index=False, name=None)]
    if rows:
        db.executemany(f"INSERT INTO {db.q(name)} ({', '.join(db.q(sc[c]) for c in cols)}) "
                       f"VALUES ({', '.join('?' * len(cols))})", rows)
    return len(rows)


def upsert_collar(db, row: dict) -> None:
    """Write one collar row (schema names, as stored: state a number, dip 0 = down): only the given
    columns are set, the others keep their values. A new hole is inserted."""
    name = db.table_name(COLLAR.name) or COLLAR.name
    sc = stored_columns(db, name)
    hid = str(row["hole_id"]).strip()
    cols = [c for c in row if c in sc and c != "hole_id"]
    vals = [db_value(COLLAR.col(c) if COLLAR.has(c) else None, row[c]) for c in cols]
    h = db.q(sc["hole_id"])
    if db.scalar(f"SELECT COUNT(*) FROM {db.q(name)} WHERE {h} = ?", (hid,), default=0):
        if cols:
            db.execute(f"UPDATE {db.q(name)} SET {', '.join(f'{db.q(sc[c])} = ?' for c in cols)} WHERE {h} = ?",
                       vals + [hid])
    else:
        db.execute(f"INSERT INTO {db.q(name)} ({', '.join(db.q(sc[c]) for c in ['hole_id'] + cols)}) "
                   f"VALUES ({', '.join('?' * (len(cols) + 1))})", [hid] + vals)


def write_hole_rows(db, t: Table, hole_id: str, df: pd.DataFrame) -> int:
    """Replace one hole's rows of a per-hole table with ``df`` (schema names, as stored). Columns of
    the table that ``df`` does not have (the other program's, the user's own) keep their values in
    the rows with the same depth; inside the caller's transaction. Returns the rows written."""
    name = db.table_name(t.name) or t.name
    all_cols = db.columns(name)
    sc = stored_columns(db, name)
    h = sc["hole_id"]
    mine = {sc[c] for c in df.columns if c in sc}
    others = [c for c in all_cols if c not in mine and c != h]
    key = next((k for k in ("depth_from", "depth", "fail_depth") if k in sc and k in df.columns), None)
    kept: dict[str, tuple] = {}
    if others and key is not None:
        rows = db.query(f"SELECT {db.q(sc[key])}, {', '.join(db.q(c) for c in others)} FROM {db.q(name)} "
                        f"WHERE {db.q(h)} = ?", (hole_id,))
        for r in rows:
            kept.setdefault(_key_text(r[0]), r[1:])
    db.execute(f"DELETE FROM {db.q(name)} WHERE {db.q(h)} = ?", (hole_id,))
    cols = [c for c in df.columns if c in sc and c != "hole_id"]
    out = []
    for rec in df[cols].itertuples(index=False, name=None):
        vals = [db_value(t.col(c) if t.has(c) else None, v) for c, v in zip(cols, rec)]
        if all(v is None for v in vals):
            continue
        extra = list(kept.get(_key_text(rec[cols.index(key)]), (None,) * len(others))) if key else []
        out.append([hole_id] + vals + (extra if others and key else []))
    if out:
        names = [h] + [sc[c] for c in cols] + (others if others and key else [])
        db.executemany(f"INSERT INTO {db.q(name)} ({', '.join(db.q(c) for c in names)}) "
                       f"VALUES ({', '.join('?' * len(names))})", out)
    return len(out)


def save_reference(db, t: Table, df: pd.DataFrame) -> None:
    """Replace all rows of a reference table (areas, formation cards, rigs, measures)."""
    name = db.table_name(t.name) or t.name
    with db.transaction():
        db.execute(f"DELETE FROM {db.q(name)}")
        insert_rows(db, t, df, name)


# --- entry checks -------------------------------------------------------------------------------
@dataclass
class Issue:
    """A problem found by the entry checks."""

    level: str              # "error" (not saved) or "warning" (saved, the user is told)
    table: str              # the table's title, translated
    row: int | None         # 0-based row in the rows checked
    column: str | None
    message: str

    def text(self) -> str:
        where = f" {tr('row')} {self.row + 1}" if self.row is not None else ""
        return f"{self.table}{where}: {self.message}"


def validate(t: Table, df: pd.DataFrame, context: dict | None = None,
             choices: dict[str, dict] | None = None) -> tuple[set[int], list[Issue]]:
    """Entry checks on the rows of one table (values as stored): required fields, physical ranges,
    codes, From < To, runs without gaps or overlaps (each run's From = the previous To), other
    intervals without overlaps, depths past the end of the hole. ``context``: max_depth (the
    hole's depth reached), max_rpm (its rig). ``choices``: column -> codes used instead of the
    schema's (e.g. the rock classes of risk_rock). Returns the rows with errors and every issue."""
    context = context or {}
    choices = choices or {}
    bad: set[int] = set()
    issues: list[Issue] = []
    title = tr(t.title)

    def err(i, col, msg):
        bad.add(i)
        issues.append(Issue("error", title, i, col, msg))

    def warn(i, col, msg):
        issues.append(Issue("warning", title, i, col, msg))

    recs = df.to_dict("records")
    for i, r in enumerate(recs):
        for c in t.cols:
            if c.name not in r:
                continue
            v = r.get(c.name)
            if is_blank(v):
                if c.required and c.name != "hole_id":
                    err(i, c.name, tr("{field} is required", field=tr(c.label)))
                continue
            if c.kind in ("real", "int"):
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    err(i, c.name, tr("{field}: '{value}' is not a number", field=tr(c.label), value=v))
                    continue
                if (c.lo is not None and x < c.lo) or (c.hi is not None and x > c.hi):
                    err(i, c.name, tr("{field} = {value} is outside {lo} - {hi} {unit}", field=tr(c.label), value=x,
                                      lo=c.lo, hi=c.hi, unit=c.unit))
            elif c.kind in ("code", "yn") and choices.get(c.name, c.choices) and \
                    _code_text(v) not in choices.get(c.name, c.choices):
                err(i, c.name, tr("{field}: unknown code '{value}'", field=tr(c.label), value=v))
        if t.interval:
            f, to = r.get("depth_from"), r.get("depth_to")
            if not is_blank(f) and not is_blank(to) and _num(to) is not None and _num(f) is not None \
                    and _num(to) <= _num(f):
                err(i, "depth_to", tr("To ({to}) must be greater than From ({f})", to=to, f=f))
        if t is RUN:
            rpm, top = _num(r.get("rpm")), _num(context.get("max_rpm"))
            if rpm is not None and top and rpm > top:
                err(i, "rpm", tr("RPM {rpm} is above the rig maximum {max}", rpm=rpm, max=top))
            ret, rem = _num(r.get("return_pct")), str(r.get("remarks") or "").lower()
            if ret is not None and ret >= 100 and "loss" in rem:
                warn(i, "return_pct", tr("A loss is noted but the return is 100%: please check"))
    if t.interval and recs:
        order = sorted((i for i, r in enumerate(recs) if _num(r.get("depth_from")) is not None
                        and _num(r.get("depth_to")) is not None),
                       key=lambda i: (str(recs[i].get("hole_id") or ""), _num(recs[i]["depth_from"])))
        prev_to, prev_hole = None, object()
        for i in order:
            r = recs[i]
            hole = str(r.get("hole_id") or "")
            f, to = _num(r["depth_from"]), _num(r["depth_to"])
            if hole != prev_hole:
                prev_to, prev_hole = None, hole
            if prev_to is not None:
                gap = f - prev_to
                if t is RUN and abs(gap) > 0.005:
                    err(i, "depth_from", tr("Run From {f} m must equal the previous run's To {t} m ({kind})",
                                            f=f, t=prev_to, kind=tr("gap") if gap > 0 else tr("overlap")))
                elif t is not OCCURRENCE and gap < -0.005:
                    err(i, "depth_from", tr("Overlaps the interval above (it ends at {t} m)", t=prev_to))
            prev_to = max(prev_to or 0.0, to) if t is OCCURRENCE else to
    depth = _num(context.get("max_depth"))
    if depth is not None and depth > 0 and recs:
        col = "depth_to" if t.interval else ("fail_depth" if t is FAILURE else ("depth" if t is SURVEY else None))
        for i, r in enumerate(recs):
            v = _num(r.get(col)) if col else None
            if v is not None and v > depth + 0.01:
                warn(i, col, tr("{v} m is below the hole's depth ({d} m)", v=v, d=depth))
    return bad, issues


def _num(v) -> float | None:
    if is_blank(v):
        return None
    try:
        f = float(str(v).replace(",", "")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _code_text(v) -> str:
    """A code as text: 3.0 -> '3' (INTEGER code columns read as numbers)."""
    n = _num(v) if not isinstance(v, str) else None
    return str(int(n)) if n is not None and n == int(n) else str(v).strip()


def check_drilling(collar: dict, tables: dict[str, pd.DataFrame]) -> list[Issue]:
    """Checks across one hole's drilling records (values as stored): a failed hole has its failure
    row, a failure row belongs to a failed hole, and the fields the risk prediction needs when
    the hole has drilling records."""
    issues: list[Issue] = []
    state = _state_number(collar.get("state"))
    fail = tables.get(FAILURE.name)
    has_fail = fail is not None and len(fail.dropna(how="all"))
    if state == FAILED and fail is not None and not has_fail:
        issues.append(Issue("warning", tr(FAILURE.title), None, None,
                            tr("The hole is failed: record its failure (depth, code, severity) in the Failure tab")))
    if has_fail and state is not None and state != FAILED:
        issues.append(Issue("warning", tr(COLLAR.title), None, "state",
                            tr("The hole has a failure row but its state is not Failed")))
    records = any(len(tables.get(t.name, pd.DataFrame()).dropna(how="all")) for t in RECORD_TABLES)
    if records:
        missing = [tr(c.label) for c in COLLAR.cols if c.risk and is_blank(collar.get(c.name))]
        if missing:
            issues.append(Issue("warning", tr(COLLAR.title), None, None,
                                tr("For the drilling history fill in: {fields}", fields=", ".join(missing))))
    return issues


# --- moving the records of an earlier dfpredict database in ---------------------------------------
_LEGACY = (COLLAR, SURVEY, GEOLOGY, HOLE_SIZE, BIT, RUN, OCCURRENCE, FAILURE, AREA, FORMATION, RIG, MEASURE)


def migrate_legacy(db) -> list[str]:
    """Move the records of dfpredict's own tables (risk_collar, risk_survey, ...: exchange format)
    into the shared tables, once; the old tables are then renamed risk_<name>_old.

    * Reference rows (areas, formation cards, rigs, measures) replace rows with the same key.
    * A hole that is not in drillhole_collar is added with all its rows.
    * A hole that is: its empty collar fields are filled; its state follows the recorded result
      when they disagree (and so does its depth reached: e.g. failed at 94.5 m); survey / geology rows are only added when it has none (else the
      geology's drilling fields are filled where the intervals match); its drilling records
      are added for each table where it has none yet.
    """
    have = {t.lower(): t for t in db.tables()}
    if LEGACY_PREFIX + "collar" not in have:
        return []
    frames = {}
    for t in _LEGACY:
        name = have.get(LEGACY_PREFIX + t.short)
        if name is not None:
            raw = db.read_frame(name)
            raw.columns = [str(c).lower() if str(c).lower() in {"hole_id", "depth_from", "depth_to"} else c
                           for c in raw.columns]
            frames[t] = raw
    notes: list[str] = []
    with db.transaction():
        for t in REFERENCE_TABLES:
            df = frames.get(t)
            if df is None or not len(df):
                continue
            k = t.key[0]
            df = df[[c for c in df.columns if t.has(c)]].dropna(subset=[k])
            keys = [str(v) for v in df[k]]
            for part in _chunks(keys, 400):
                db.execute(f"DELETE FROM {db.q(t.name)} WHERE {db.q(k)} IN ({', '.join('?' * len(part))})", part)
            insert_rows(db, t, df)
        collar = frames.get(COLLAR)
        if collar is not None and len(collar):
            cur = read_table(db, COLLAR)
            cur_states = dict(zip(cur["hole_id"], cur["state"]))
            new = from_exchange(COLLAR, collar, cur_states)
            new["hole_id"] = new["hole_id"].astype(str).str.strip()
            existing = set(cur["hole_id"])
            cur_by = cur.set_index("hole_id")
            changed_state = 0
            for rec in new.to_dict("records"):
                row = {k: v for k, v in rec.items() if COLLAR.has(k)}
                if rec["hole_id"] in existing:
                    old = cur_by.loc[rec["hole_id"]]
                    keep = {}
                    for k, v in row.items():
                        if k == "state":
                            if not is_blank(v) and (is_blank(old["state"]) or result_of_state(old["state"])
                                                    != result_of_state(v)):
                                keep[k] = v
                                changed_state += 0 if is_blank(old["state"]) else 1
                        elif k != "hole_id" and is_blank(old.get(k)) and not is_blank(v):
                            keep[k] = v
                    if "state" in keep and not is_blank(old["state"]) and not is_blank(row.get("max_depth")):
                        keep["max_depth"] = row["max_depth"]     # the recorded outcome: e.g. failed at 94.5 m
                    row = {"hole_id": rec["hole_id"], **keep}
                upsert_collar(db, row)
            holes_with = {}
            for t in (SURVEY, GEOLOGY) + RECORD_TABLES:
                name = db.table_name(t.name)
                sc = stored_columns(db, name)
                holes_with[t] = {str(r[0]) for r in db.query(f"SELECT DISTINCT {db.q(sc['hole_id'])} FROM {db.q(name)}")}
            for t in (SURVEY, GEOLOGY) + RECORD_TABLES:
                df = frames.get(t)
                if df is None or not len(df):
                    continue
                df = from_exchange(t, df)
                df = df[[c for c in df.columns if t.has(c)]]
                df["hole_id"] = df["hole_id"].astype(str).str.strip()
                fresh = df[~df["hole_id"].isin(holes_with[t])]
                insert_rows(db, t, fresh)
                if t is GEOLOGY:
                    _fill_geology(db, df[df["hole_id"].isin(holes_with[t])])
            notes.append(tr("moved the drilling records of {n} holes from the risk_ tables into the drillhole_ "
                            "tables ({m} already there were completed)", n=len(new), m=len(existing & set(new["hole_id"]))))
            if changed_state:
                notes.append(tr("the state of {n} holes was set from their recorded drilling result", n=changed_state))
    for t in _LEGACY:
        name = have.get(LEGACY_PREFIX + t.short)
        if name is None:
            continue
        new_name, i = f"{name}_old", 2
        while new_name.lower() in have:
            new_name, i = f"{name}_old{i}", i + 1
        db.execute(f"ALTER TABLE {db.q(name)} RENAME TO {db.q(new_name)}")
        have[new_name.lower()] = new_name
    db.con.commit()
    return notes


def _fill_geology(db, df: pd.DataFrame) -> None:
    """The drilling fields of geology intervals already stored, from legacy rows with the same
    hole, From and To (only where they are empty)."""
    name = db.table_name(GEOLOGY.name)
    sc = stored_columns(db, name)
    cols = [c.name for c in GEOLOGY_ADDED if c.name in sc and c.name in df]
    if not cols or not len(df):
        return
    for rec in df.to_dict("records"):
        for c in cols:
            v = db_value(GEOLOGY.col(c), rec.get(c))
            if v is None:
                continue
            db.execute(f"UPDATE {db.q(name)} SET {db.q(sc[c])} = ? WHERE {db.q(sc['hole_id'])} = ? AND "
                       f"ABS({db.q(sc['depth_from'])} - ?) < 0.01 AND ABS({db.q(sc['depth_to'])} - ?) < 0.01 AND "
                       f"{db.q(sc[c])} IS NULL",
                       (v, rec["hole_id"], float(rec["depth_from"]), float(rec["depth_to"])))


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --- CSV exchange (the drilling history, design spec 3.4: one file per table, CSV UTF-8) ----------------
ALIASES = {   # CSV column names recognised on import (spec 3.1 writes them in capitals: HOLE_ID, FROM, TO ...)
    "hole_id": ("hole_id", "holeid", "hole", "bhid", "dhid"),
    "depth_from": ("depth_from", "from", "from_m", "mfrom", "depthfrom", "run_from"),
    "depth_to": ("depth_to", "to", "to_m", "mto", "depthto", "run_to"),
    "rock": ("rock", "rock_code", "lith", "lithology"),
    "size": ("size", "hole_size", "size_code"),
    "diameter": ("diameter", "diameter_mm", "dia"),
    "bit_type": ("bit_type", "bit", "type"),
    "type": ("type", "occurrence", "occ_type"),
    "fail_code": ("fail_code", "failcode", "code"),
    "pump_press": ("pump_press", "pressure", "pump_pressure"),
    "flow_in": ("flow_in", "flow", "flowin"),
    "return_pct": ("return_pct", "return", "returns"),
    "result": ("result", "status"),
    "area_id": ("area_id", "area"),
    "planned_depth": ("planned_depth", "plan_depth", "target_depth"),
    "final_depth": ("final_depth", "max_depth", "eoh", "depth_reached"),
    "notes": ("notes", "note", "comment", "comments"),
}


def _norm(name: str) -> str:
    return re.sub(r"[\s()\[\]\-.]", "", str(name)).lower()


def map_columns(t: Table, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The columns of a CSV (exchange format) mapped to the table's exchange columns (names matched
    ignoring case and spaces, plus ALIASES), values typed, codes matched by code or text. Returns
    the rows and the exchange columns the file had."""
    normed = {_norm(c): c for c in raw.columns}
    ex = exchange_names(t)
    ren = {v: k for k, v in _EX_RENAME.get(t.name, {}).items()}
    out = pd.DataFrame(index=raw.index)
    present = []
    for name in ex:
        src = next((normed[_norm(a)] for a in (name,) + ALIASES.get(name, ()) if _norm(a) in normed), None)
        if src is None:
            continue
        present.append(name)
        s = raw[src]
        base = ren.get(name, name)
        c = t.col(base) if t.has(base) else None
        if c is not None and c.kind in ("real", "int"):
            out[name] = pd.to_numeric(s.where(s.astype(str).str.strip() != "", None), errors="coerce")
        else:
            s = s.astype(str).str.strip().replace("", None)
            if c is not None and c.name == "state":
                s = s.map(_state_from_text)
            elif c is not None and c.kind in ("code", "yn") and c.choices:
                s = s.map(lambda v, ch=c.choices: _match_code(v, ch))
            out[name] = s
    return out, present


def _state_from_text(v):
    if is_blank(v):
        return None
    n = _state_number(v)
    if n is not None:
        return n
    low = str(v).strip().lower()
    return next((k for k, text in HOLE_STATES.items() if low in (text.lower(), tr(text).lower())), None)


def _match_code(v, choices: dict):
    if is_blank(v):
        return None
    s = _code_text(v)
    for k in choices:
        if s.lower() == str(k).lower():
            return k
    for k, text in choices.items():               # the text instead of the code ("Yes", "Fault zone")
        if s.lower() in (str(text).lower(), tr(text).lower()):
            return k
    if s.lower() in ("yes", "true", "1") and "Y" in choices:
        return "Y"
    if s.lower() in ("no", "false", "0") and "N" in choices:
        return "N"
    return s


def export_csv(target, folder: str | Path, areas: list[str] | None = None) -> dict[str, int]:
    """The drilling records as CSV files (exchange format) in ``folder``: collar.csv, survey.csv,
    geology.csv, hole_size.csv, bit.csv, run.csv, occurrence.csv, failure.csv and the reference
    tables. ``areas``: only the holes of these areas. Returns the rows per file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    counts = {}
    with target.connect(create=False) as db:
        collar = read_table(db, COLLAR)
        if areas:
            collar = collar[collar["area_id"].isin(set(areas))]
        holes = set(collar["hole_id"])
        for t in SHARED_TABLES:
            if t is COLLAR:
                df = collar
            elif t.per_hole:
                df = read_table(db, t)
                df = df[df["hole_id"].isin(holes)]
            else:
                df = read_table(db, t)
            df = to_exchange(t, df).reindex(columns=exchange_names(t))
            df.to_csv(folder / f"{t.short}.csv", index=False, encoding="utf-8")
            counts[t.short] = len(df)
    return counts


def import_csv(target, folder: str | Path, replace_areas: bool = False) -> tuple[dict[str, int], list[Issue]]:
    """Read ``<table>.csv`` files of a folder (exchange format) into the shared tables, in one
    transaction. Reference rows replace rows with the same key. A hole in a file gets the file's
    collar values (only the columns the file has; the others are kept) and, in each file that has
    rows for it, its rows replaced. ``replace_areas``: the holes of the areas in collar.csv that
    are in none of the files are removed. Rows with errors are skipped and reported."""
    folder = Path(folder)
    frames: dict[Table, pd.DataFrame] = {}
    issues: list[Issue] = []
    for t in SHARED_TABLES:
        p = folder / f"{t.short}.csv"
        if not p.exists():
            continue
        raw = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df, present = map_columns(t, raw)
        frames[t] = (df, present)
    counts: dict[str, int] = {}
    if not frames:
        return counts, issues
    with target.connect() as db:
        cur = read_table(db, COLLAR)
        cur_states = dict(zip(cur["hole_id"], cur["state"]))
        stored: dict[Table, pd.DataFrame] = {}
        for t, (df, present) in frames.items():
            db_df = from_exchange(t, df[present], cur_states)
            if "hole_id" in db_df:
                db_df["hole_id"] = db_df["hole_id"].astype(str).str.strip()
            bad, found = validate(t, db_df)
            issues += found
            if t.per_hole or t is COLLAR:
                miss = db_df["hole_id"].isin(["", "None", "nan"]) if "hole_id" in db_df else pd.Series(True, db_df.index)
                bad |= set(np.flatnonzero(miss.to_numpy()))
            stored[t] = db_df.drop(index=db_df.index[sorted(bad)]).reset_index(drop=True) if bad else db_df
            counts[t.short] = len(stored[t])
        with db.transaction():
            for t in REFERENCE_TABLES:
                if t in stored and len(stored[t]):
                    df = stored[t].dropna(subset=[t.key[0]])
                    for part in _chunks([str(v) for v in df[t.key[0]]], 400):
                        db.execute(f"DELETE FROM {db.q(t.name)} WHERE {db.q(t.key[0])} IN "
                                   f"({', '.join('?' * len(part))})", part)
                    insert_rows(db, t, df)
            if COLLAR in stored:
                for rec in stored[COLLAR].to_dict("records"):
                    upsert_collar(db, rec)
            for t in HOLE_RECORDS:
                if t in stored:
                    df = stored[t]
                    for hid, g in df.groupby("hole_id", sort=False):
                        write_hole_rows(db, t, str(hid), g.drop(columns="hole_id"))
            if replace_areas and COLLAR in stored and "area_id" in stored[COLLAR]:
                areas = set(stored[COLLAR]["area_id"].dropna().astype(str))
                keep = set().union(*(set(d["hole_id"]) for t, d in stored.items() if "hole_id" in d))
                gone = [h for h, a in zip(cur["hole_id"], cur["area_id"]) if a in areas and h not in keep]
                if gone:
                    counts["removed"] = len(gone)
                    remove_holes(db, gone)
    return counts, issues


def remove_holes(db, hole_ids: list[str]) -> None:
    """Remove holes from every drillhole_ table with a hole id (inside the caller's transaction)."""
    from .database import DRILLHOLE_PREFIX

    for name in db.tables():
        if not name.lower().startswith(DRILLHOLE_PREFIX):
            continue
        sc = stored_columns(db, name)
        if "hole_id" not in sc:
            continue
        for part in _chunks(hole_ids, 400):
            db.execute(f"DELETE FROM {db.q(name)} WHERE {db.q(sc['hole_id'])} IN ({', '.join('?' * len(part))})", part)
