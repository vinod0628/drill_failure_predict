"""The drilling-record tables (design spec 3.1 - 3.5, 7.2) as the risk system sees them: one schema for the
records in memory, the CSV import / export, the entry checks and the input dialog.

The drillhole records are the project database's drillhole tables, shared with GeoViewer and 3D MPM
Studio and described once in ``io/drilling_schema.py`` (an identical copy in each program):

    drillhole_collar      one row per hole: area, rig, plan, state (-> result)               (table A)
    drillhole_survey      survey stations                                                   (table B)
    drillhole_geology     geology log: rock, formation, broken zone, bedding angle          (table C)
    drillhole_hole_size   hole size by depth (each reduction usually marks casing)          (table D)
    drillhole_bit         bit type by depth                                                 (table E)
    drillhole_run         drilling parameters per run (+ the new fields of spec 3.3)        (table F)
    drillhole_failure     the failure of a failed hole                                      (table G)
    drillhole_occurrence  cavities, fault zones, other problems, with measures and solved   (table H)
    drillhole_area        site sheet, one row per drilling area (old or new)                (spec 5.10)
    drillhole_formation   formation property catalogue: one card per formation              (spec 3.5)
    drillhole_rig         rig specification: rated depth per hole size, max RPM, pump       (table I)
    drillhole_measure     the measure library (editable texts; seeded from spec 6.1)

Only the risk system's own data keeps the prefix ``risk_``:

    risk_warning     warnings issued, with the model version that produced them       (spec 7.4)
    risk_rock        mechanical / drilling properties of each rock class (editable; seeded with defaults)
    risk_rock_code   logged rock code -> rock class (editable; seeded)

The records in memory are in the exchange format of drilling_schema.py: dip negative downward (-90 =
vertical down, as drillers and CSV files write it), final_depth (stored as max_depth), result (stored as the
hole's state, see drilling_schema.result_of_state) and notes (the geology's note). RiskStore converts.
Depths along the hole in metres from the collar; blank = unknown (never 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..i18n import N_
from . import codes

PREFIX = "risk_"                 # the risk system's own tables
SHARED_PREFIX = "drillhole_"     # the drillhole tables shared with GeoViewer (io/drilling_schema.py)
OWN_TABLES = {"warning", "rock", "rock_code"}


@dataclass(frozen=True)
class Col:
    name: str
    sql: str                       # portable SQL type
    label: str                     # shown (translated with tr)
    kind: str = "real"             # real / int / text / code / yn / date / note
    choices: dict | None = None    # code -> text for kind "code"
    unit: str = ""
    tip: str = ""
    lo: float | None = None        # physical limits (entry check, spec 7.3)
    hi: float | None = None
    required: bool = False
    width: int = 0                 # preferred column width in tables (0 = automatic)


@dataclass(frozen=True)
class Table:
    name: str
    title: str
    key: tuple                     # identifying columns
    cols: tuple
    per_hole: bool = True          # rows belong to a hole (hole_id first)
    interval: bool = False         # depth_from / depth_to rows
    help: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def sql_name(self) -> str:
        return (PREFIX if self.name in OWN_TABLES else SHARED_PREFIX) + self.name

    @property
    def shared(self) -> bool:
        """One of the drillhole tables shared with GeoViewer (else the risk system's own)."""
        return self.name not in OWN_TABLES

    def col(self, name: str) -> Col:
        return next(c for c in self.cols if c.name == name)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cols]


_YN = {"Y": N_("Yes"), "N": N_("No")}
HOLE = Col("hole_id", "VARCHAR(64)", N_("Hole ID"), "text", required=True, width=110)
FROM = Col("depth_from", "DOUBLE", N_("From"), "real", unit="m", lo=0, hi=5000, required=True, width=70)
TO = Col("depth_to", "DOUBLE", N_("To"), "real", unit="m", lo=0, hi=5000, required=True, width=70)


def _rock_choices() -> dict:
    return {k: v[0] for k, v in codes.ROCK_CLASSES.items()}


AREA = Table("area", N_("Areas (site sheets)"), ("area_id",), (
    Col("area_id", "VARCHAR(32)", N_("Area ID"), "text", required=True, width=80),
    Col("name", "VARCHAR(100)", N_("Name"), "text", width=150),
    Col("dip_min", "DOUBLE", N_("Bedding dip min"), unit="°", lo=0, hi=90,
        tip=N_("Range of bedding dips in the area (structural style)")),
    Col("dip_max", "DOUBLE", N_("Bedding dip max"), unit="°", lo=0, hi=90),
    Col("fold_tightness", "INTEGER", N_("Fold tightness"), "code",
        {"1": N_("1 - open"), "2": N_("2 - moderate"), "3": N_("3 - tight")}),
    Col("faults_per_km", "DOUBLE", N_("Faults per km"), lo=0, hi=50),
    Col("carbonate_exposed", "VARCHAR(1)", N_("Carbonate exposed"), "yn", _YN,
        tip=N_("Carbonate outcrops at surface (Y) or is covered (N)")),
    Col("unconformity", "VARCHAR(1)", N_("Unconformity above carbonate"), "yn", _YN),
    Col("sinkholes", "VARCHAR(1)", N_("Sinkholes mapped"), "yn", _YN),
    Col("carb_above_valley", "DOUBLE", N_("Carbonate height above valley floor"), unit="m", lo=-1000, hi=2000,
        tip=N_("Height of the top of the carbonate above the local valley floor (karst base level)")),
    Col("water_table", "DOUBLE", N_("Regional water table elevation"), unit="m", lo=-2000, hi=6000,
        tip=N_("Elevation of the regional water table, if known (strong karst indicator)")),
    Col("depth_of_interest", "DOUBLE", N_("Depth of interest"), unit="m", lo=0, hi=5000),
    Col("notes", "TEXT", N_("Notes"), "note"),
), per_hole=False, help=N_("One row per drilling location, old or new. Keep it to these few fields: the models "
                           "learn each one from only as many examples as there are areas."))

FORMATION = Table("formation", N_("Formation cards"), ("formation",), (
    Col("formation", "VARCHAR(64)", N_("Formation"), "text", required=True, width=110,
        tip=N_("One name per formation everywhere, also in the geological model")),
    Col("age", "VARCHAR(40)", N_("Age / system"), "text", tip=N_("e.g. Cambrian, Ordovician")),
    Col("main_rock", "VARCHAR(20)", N_("Main rock"), "code", _rock_choices(), required=True),
    Col("main_pct", "DOUBLE", N_("Main rock share"), unit="%", lo=0, hi=100),
    Col("second_rock", "VARCHAR(20)", N_("Secondary rock"), "code", _rock_choices()),
    Col("second_pct", "DOUBLE", N_("Secondary share"), unit="%", lo=0, hi=100),
    Col("carbonate_pct", "DOUBLE", N_("Carbonate share"), unit="%", lo=0, hi=100, tip=N_("Karst potential")),
    Col("hardness", "INTEGER", N_("Hardness class"), "int", lo=1, hi=5, tip=N_("1 soft - 5 very hard")),
    Col("abrasiveness", "INTEGER", N_("Abrasiveness class"), "int", lo=1, hi=5, tip=N_("1 low - 5 very abrasive")),
    Col("bedding", "VARCHAR(20)", N_("Bedding style"), "code", codes.BEDDING_STYLES),
    Col("swelling_clay", "VARCHAR(20)", N_("Swelling / weak clay"), "code", codes.CLAY_LEVELS),
    Col("karst", "VARCHAR(20)", N_("Known karst / vugs"), "code", codes.KARST_LEVELS),
    Col("ucs_mpa", "DOUBLE", N_("UCS (lab)"), unit="MPa", lo=0.1, hi=500,
        tip=N_("Measured uniaxial compressive strength of the formation; blank = the rock class default")),
    Col("cai_lab", "DOUBLE", N_("CAI (lab)"), lo=0, hi=7,
        tip=N_("Measured Cerchar abrasivity index; blank = the rock class default")),
    Col("thick_min", "DOUBLE", N_("Thickness min"), unit="m", lo=0, hi=3000),
    Col("thick_max", "DOUBLE", N_("Thickness max"), unit="m", lo=0, hi=3000,
        tip=N_("Sets the depth-error band for contacts below this formation")),
    Col("equivalents", "VARCHAR(255)", N_("Equivalent formations"), "text",
        tip=N_("Names of the same formation in other areas, separated by ;")),
    Col("notes", "TEXT", N_("Notes"), "note"),
), per_hole=False, help=N_("A geologist fills in one card per formation, once. The models predict a formation never "
                           "drilled before from its card, so learning transfers between areas."))

RIG = Table("rig", N_("Rig specifications"), ("rig_model",), (
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

COLLAR = Table("collar", N_("Collar"), ("hole_id",), (
    HOLE,
    Col("area_id", "VARCHAR(32)", N_("Area"), "code", {}, required=True, width=70),
    Col("x", "DOUBLE", N_("Easting X"), unit="m", required=True),
    Col("y", "DOUBLE", N_("Northing Y"), unit="m", required=True),
    Col("z", "DOUBLE", N_("Elevation Z"), unit="m", required=True),
    Col("rig_model", "VARCHAR(64)", N_("Rig model"), "code", {}),
    Col("start_date", "VARCHAR(10)", N_("Start date"), "date"),
    Col("end_date", "VARCHAR(10)", N_("End date"), "date"),
    Col("plan_azimuth", "DOUBLE", N_("Planned azimuth"), unit="°", lo=0, hi=360,
        tip=N_("Designed azimuth at the collar, degrees clockwise from north")),
    Col("plan_dip", "DOUBLE", N_("Planned dip"), unit="°", lo=-90, hi=90,
        tip=N_("Designed dip at the collar, negative downward (-90 = vertical down)")),
    Col("planned_depth", "DOUBLE", N_("Planned (target) depth"), unit="m", lo=0, hi=5000, required=True),
    Col("final_depth", "DOUBLE", N_("Final / current depth"), unit="m", lo=0, hi=5000,
        tip=N_("Depth reached: final depth of a finished hole, current bit depth of a hole in progress")),
    Col("result", "VARCHAR(16)", N_("Result"), "code", codes.RESULTS, required=True),
    Col("casing_available", "VARCHAR(64)", N_("Casing on site"), "text",
        tip=N_("Casing sizes available at the rig, e.g. HW;NW (used to filter casing measures)")),
    Col("fluids_available", "VARCHAR(128)", N_("Fluids on site"), "text",
        tip=N_("Fluid systems available at the rig, e.g. water;polymer;bentonite")),
    Col("crew", "VARCHAR(64)", N_("Crew"), "text"),
    Col("notes", "TEXT", N_("Notes"), "note"),
), per_hole=False)

SURVEY = Table("survey", N_("Survey"), ("hole_id", "depth"), (
    HOLE,
    Col("depth", "DOUBLE", N_("Depth"), unit="m", lo=0, hi=5000, required=True),
    Col("azimuth", "DOUBLE", N_("Azimuth"), unit="°", lo=0, hi=360, required=True),
    Col("dip", "DOUBLE", N_("Dip"), unit="°", lo=-90, hi=90, required=True,
        tip=N_("Negative downward: -90 = vertical down")),
), help=N_("A station every 30-50 m; every 10-20 m in trajectory-risk zones."))

GEOLOGY = Table("geology", N_("Geology log"), ("hole_id", "depth_from"), (
    HOLE, FROM, TO,
    Col("rock", "VARCHAR(32)", N_("Rock"), "text", required=True, tip=N_("Rock code as logged, e.g. DO, LS, SS")),
    Col("formation", "VARCHAR(64)", N_("Formation"), "code", {}, tip=N_("Links each metre to its property card")),
    Col("weathering", "INTEGER", N_("Weathering"), "int", lo=0, hi=5),
    Col("broken", "VARCHAR(1)", N_("Broken zone"), "yn", _YN),
    Col("bedding_angle", "DOUBLE", N_("Bedding to core axis"), unit="°", lo=0, hi=90,
        tip=N_("Angle between bedding and the core axis (90 = perpendicular to the hole)")),
    Col("notes", "TEXT", N_("Notes"), "note"),
), interval=True)

HOLE_SIZE = Table("hole_size", N_("Hole size"), ("hole_id", "depth_from"), (
    HOLE, FROM, TO,
    Col("size", "VARCHAR(8)", N_("Size"), "code", {k: k for k in codes.HOLE_SIZES}, required=True),
    Col("diameter", "DOUBLE", N_("Diameter"), unit="mm", lo=20, hi=400),
), interval=True, help=N_("Each reduction usually marks casing being set."))

BIT = Table("bit", N_("Bit"), ("hole_id", "depth_from"), (
    HOLE, FROM, TO,
    Col("bit_type", "VARCHAR(20)", N_("Bit type"), "code", codes.BIT_TYPES, required=True),
    Col("matrix", "VARCHAR(20)", N_("Matrix / series"), "text", tip=N_("e.g. series 7 (soft) ... 10 (hard rock)")),
), interval=True)

RUN = Table("run", N_("Runs (drilling parameters)"), ("hole_id", "depth_from"), (
    HOLE, FROM, TO,
    Col("rpm", "DOUBLE", N_("RPM"), unit="rpm", lo=0, hi=3000),
    Col("rop", "DOUBLE", N_("ROP"), unit="m/h", lo=0, hi=200),
    Col("flow_in", "DOUBLE", N_("Flow in"), unit="L/min", lo=0, hi=1000),
    Col("pump_press", "DOUBLE", N_("Pump pressure"), unit="MPa", lo=0, hi=40),
    Col("return_pct", "DOUBLE", N_("Return"), unit="%", lo=0, hi=120, tip=N_("Estimated return flow (new field)")),
    Col("rod_drop", "DOUBLE", N_("Rod drop"), unit="m", lo=0, hi=50, tip=N_("Direct sign of a cavity (new field)")),
    Col("wob", "DOUBLE", N_("Bit load (WOB)"), unit="kN", lo=0, hi=200, tip=N_("If the rig has gauges")),
    Col("torque", "DOUBLE", N_("Torque"), unit="N·m", lo=0, hi=20000, tip=N_("If the rig has gauges")),
    Col("fluid_type", "VARCHAR(20)", N_("Fluid"), "code", codes.FLUID_TYPES),
    Col("recovery", "DOUBLE", N_("Core recovery"), unit="%", lo=0, hi=110),
    Col("remarks", "VARCHAR(255)", N_("Remarks"), "text"),
), interval=True, help=N_("One row per core run (1.5-6 m). Per-run values support the live warnings."))

FAILURE = Table("failure", N_("Failure"), ("hole_id",), (
    HOLE,
    Col("fail_depth", "DOUBLE", N_("Failure depth"), unit="m", lo=0, hi=5000, required=True),
    Col("fail_code", "VARCHAR(4)", N_("Failure code"), "code",
        {k: f"{k} - {v[0]}" for k, v in codes.FAIL_CODES.items()}, required=True),
    Col("severity", "INTEGER", N_("Severity"), "code", {str(k): v for k, v in codes.SEVERITY.items()},
        required=True),
    Col("measures", "VARCHAR(255)", N_("Measures tried"), "text", tip=N_("Measure codes separated by ;")),
    Col("notes", "TEXT", N_("Notes"), "note"),
), help=N_("One row per failed hole: the failure label."))

OCCURRENCE = Table("occurrence", N_("Occurrences"), ("hole_id", "depth_from", "type"), (
    HOLE, FROM, TO,
    Col("type", "VARCHAR(16)", N_("Type"), "code", codes.OCC_TYPES, required=True),
    Col("measures", "VARCHAR(255)", N_("Measures"), "text", tip=N_("Measure codes separated by ;")),
    Col("solved", "VARCHAR(1)", N_("Solved"), "yn", _YN),
    Col("notes", "TEXT", N_("Notes"), "note"),
), interval=True, help=N_("Cavities, fault zones and other problems, solved or not: they train the hazard model "
                          "and give the measured success rates of the measures."))

WARNING = Table("warning", N_("Warnings issued"), ("hole_id", "created", "depth_from"), (
    HOLE,
    Col("created", "VARCHAR(19)", N_("Created"), "text"),
    FROM, TO,
    Col("fail_codes", "VARCHAR(32)", N_("Failure"), "text"),
    Col("p", "DOUBLE", N_("P"), lo=0, hi=1),
    Col("level", "VARCHAR(8)", N_("Level"), "text"),
    Col("reasons", "VARCHAR(255)", N_("Reasons"), "text"),
    Col("measures", "VARCHAR(255)", N_("Measures"), "text"),
    Col("model_version", "VARCHAR(32)", N_("Model version"), "text"),
), interval=True)

_LVL = {"0": N_("0 - none"), "1": N_("1 - some"), "2": N_("2 - strong")}
_GROUPS = {"clastic": N_("Clastic sedimentary"), "carbonate": N_("Carbonate"), "chemical": N_("Chemical / evaporite"),
           "metamorphic": N_("Metamorphic"), "igneous": N_("Igneous"), "ore": N_("Ore"), "fault": N_("Fault rock"),
           "weathered": N_("Weathered")}
ROCK = Table("rock", N_("Rock properties"), ("rock_class",), (
    Col("rock_class", "VARCHAR(32)", N_("Rock class"), "text", required=True, width=120,
        tip=N_("Key used by the rock codes and the formation cards' main rock")),
    Col("name", "VARCHAR(64)", N_("Name"), "text", width=170),
    Col("group", "VARCHAR(16)", N_("Group"), "code", _GROUPS),
    Col("ucs_min", "DOUBLE", N_("UCS min"), unit="MPa", lo=0.01, hi=600),
    Col("ucs", "DOUBLE", N_("UCS typical"), unit="MPa", lo=0.01, hi=600, required=True,
        tip=N_("Uniaxial compressive strength of fresh intact rock (ISRM classes: <25 weak, 25-50, 50-100, 100-250, "
               ">250 MPa extremely strong)")),
    Col("ucs_max", "DOUBLE", N_("UCS max"), unit="MPa", lo=0.01, hi=600),
    Col("tensile", "DOUBLE", N_("Tensile strength"), unit="MPa", lo=0.01, hi=60),
    Col("e_mod", "DOUBLE", N_("Young's modulus"), unit="GPa", lo=0.01, hi=200),
    Col("poisson", "DOUBLE", N_("Poisson's ratio"), lo=0.0, hi=0.5),
    Col("density", "DOUBLE", N_("Density"), unit="g/cm³", lo=1.0, hi=5.5),
    Col("porosity", "DOUBLE", N_("Porosity"), unit="%", lo=0, hi=60),
    Col("cai_min", "DOUBLE", N_("CAI min"), lo=0, hi=7),
    Col("cai", "DOUBLE", N_("CAI typical"), lo=0, hi=7, required=True,
        tip=N_("Cerchar abrasivity index (ASTM D7625): <1 low, 1-2 moderate, 2-3 high, 3-4 very high, >=4 extreme")),
    Col("cai_max", "DOUBLE", N_("CAI max"), lo=0, hi=7),
    Col("quartz_eq", "DOUBLE", N_("Equivalent quartz"), unit="%", lo=0, hi=100),
    Col("dri", "DOUBLE", N_("Drilling rate index DRI"), lo=0, hi=120, tip=N_("NTNU DRI: higher = easier to drill")),
    Col("mi", "DOUBLE", N_("Hoek-Brown mi"), lo=1, hi=40),
    Col("carbonate", "DOUBLE", N_("Carbonate share"), unit="%", lo=0, hi=100),
    Col("soluble", "INTEGER", N_("Solubility (karst)"), "code", _LVL),
    Col("swelling", "INTEGER", N_("Swelling"), "code", _LVL),
    Col("anisotropy", "INTEGER", N_("Anisotropy (foliation)"), "code", _LVL),
    Col("clastic", "VARCHAR(1)", N_("Clastic"), "yn", _YN),
    Col("fractures", "INTEGER", N_("Fracturing tendency"), "code", _LVL),
    Col("notes", "TEXT", N_("Notes / source"), "note"),
), per_hole=False, help=N_("Mechanical and drilling properties of each rock class - typical values of fresh intact rock "
                           "from the literature (Goodman 1989, Lama & Vutukuri 1978, Hoek 2007, ASTM D7625 / Plinninger "
                           "for CAI, NTNU for DRI). Replace them with your laboratory values; blank cells use the "
                           "default. Retrain the models after changing them."))

ROCK_CODE = Table("rock_code", N_("Rock codes"), ("code",), (
    Col("code", "VARCHAR(16)", N_("Rock code"), "text", required=True, width=90,
        tip=N_("The code as logged in the geology log, e.g. DO, SST, SCH")),
    Col("rock_class", "VARCHAR(32)", N_("Rock class"), "code", {}, required=True, width=170),
    Col("description", "VARCHAR(128)", N_("Description"), "text", width=220),
), per_hole=False, help=N_("Which rock class (and so which mechanical properties) each logged rock code stands for. "
                           "Codes not listed are matched by name (e.g. 'dolomite')."))

MEASURE = Table("measure", N_("Measure library"), ("code",), (
    Col("code", "VARCHAR(16)", N_("Code"), "text", required=True, width=85),
    Col("title", "VARCHAR(80)", N_("Short title"), "text", width=200),
    Col("text", "VARCHAR(255)", N_("Measure text"), "text", required=True, width=480,
        tip=N_("Values in braces are filled in by the system: {from_}, {to}, {depth}, {rpm}, {q}, {wob}, ...")),
    Col("failures", "VARCHAR(64)", N_("Failures"), "text", width=110, tip=N_("Failure codes, e.g. LC, CV")),
    Col("stage", "VARCHAR(1)", N_("Stage"), "code", {"P": N_("P - prevention"), "R": N_("R - response")}),
    Col("expert_order", "INTEGER", N_("Expert order"), "int", lo=0, hi=999,
        tip=N_("Priority used until outcomes build up (lower = offered first)")),
), per_hole=False, help=N_("The pre-defined measures (spec 6.1). Edit the texts to your crews' practice; keep the "
                           "codes. The decision table picks them by failure code and alarm level."))

REFERENCE_TABLES = (AREA, FORMATION, RIG, ROCK, ROCK_CODE, MEASURE)
HOLE_TABLES = (SURVEY, GEOLOGY, HOLE_SIZE, BIT, RUN, OCCURRENCE, FAILURE)
ALL_TABLES = (AREA, FORMATION, RIG, ROCK, ROCK_CODE, MEASURE, COLLAR) + HOLE_TABLES + (WARNING,)
BY_NAME = {t.name: t for t in ALL_TABLES}
