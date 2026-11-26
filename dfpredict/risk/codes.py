"""Controlled code lists and the pre-defined text libraries of the drilling risk system.

Everything shown to the driller is assembled from these libraries (design spec 2, 5.7, 5.8, 6.1, 6.2):
failure codes and their families, occurrence types, rock classes, bit types, hole sizes, fluids,
reason-code messages, measure texts, the decision table and the alarm rules. No Qt here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..i18n import N_

# --- failure codes (spec 2) ---------------------------------------------------------------------
FAMILIES = {
    "karst": N_("Karst and fluid"),
    "instability": N_("Hole instability"),
    "trajectory": N_("Trajectory"),
    "bit": N_("Bit and string"),
}
FAIL_CODES = {   # code -> (name, family)
    "LC": (N_("Lost circulation"), "karst"),
    "CV": (N_("Cavity encounter / rod drop"), "karst"),
    "WI": (N_("Water / gas influx"), "karst"),
    "CL": (N_("Hole collapse / caving"), "instability"),
    "SP": (N_("Stuck rods / stuck pipe"), "instability"),
    "CB": (N_("Core blockage / poor recovery"), "instability"),
    "DV": (N_("Hole deviation / dogleg"), "trajectory"),
    "BG": (N_("Bit glazing / polishing"), "bit"),
    "BW": (N_("Excessive bit wear / bit burn"), "bit"),
    "RF": (N_("Rod or string failure"), "bit"),
}
FAMILY_CODES = {f: [c for c, (_n, fam) in FAIL_CODES.items() if fam == f] for f in FAMILIES}
# Colour of each family in charts / risk volumes (spec 8.1: karst-loss blue, fault-collapse purple,
# deviation brown).
FAMILY_COLORS = {"karst": "#3d8bfd", "instability": "#a55eea", "trajectory": "#a0522d", "bit": "#e8a33d"}
SEVERITY = {1: N_("1 - minor (hours)"), 2: N_("2 - moderate (shift)"), 3: N_("3 - serious (days)"),
            4: N_("4 - severe (redrill part)"), 5: N_("5 - hole lost")}

# --- hole results -------------------------------------------------------------------------------
RESULTS = {"success": N_("Success"), "failed": N_("Failed"), "in_progress": N_("In progress"),
           "planned": N_("Planned")}
RESULT_COLORS = {"success": "#3fb950", "failed": "#f85149", "in_progress": "#d29922", "planned": "#58a6ff"}

# --- occurrences (table H) ----------------------------------------------------------------------
OCC_TYPES = {"cavity": N_("Cavity / karst"), "fault": N_("Fault zone"), "broken": N_("Broken / crushed rock"),
             "clay": N_("Swelling clay"), "water": N_("Water / gas inflow"), "loss": N_("Fluid loss"),
             "other": N_("Other problem")}
HAZARD_TYPES = ("cavity", "fault")       # Model H outputs

# --- rock classes (properties in rockmech.py, editable in the project database: risk_rock) --------
# class -> (name, carbonate %, strength class 1-5 (ISRM, from UCS), abrasivity class 1-5 (from CAI), clastic)
ROCK_CLASSES: dict = {}


def refresh_rock_classes() -> None:
    from .rockmech import ROCKS, abrasivity_class, strength_class

    ROCK_CLASSES.clear()
    for k, p in ROCKS.items():
        ROCK_CLASSES[k] = (p.name, p.carbonate, strength_class(p.ucs_typ), abrasivity_class(p.cai_typ), p.clastic)


refresh_rock_classes()
# Soluble rocks (karst): carbonates and evaporites.
SOLUBLE = {"limestone", "dolomite", "siliceous_dolomite", "marble", "chalk", "travertine", "gypsum", "halite",
           "calc_schist"}
# Rock codes as logged -> class (editable: risk_rock_code). Codes not listed are matched by name.
DEFAULT_ROCK_CODES = {
    "CG": "conglomerate", "CGL": "conglomerate", "SBX": "breccia_sed", "SS": "sandstone", "SST": "sandstone",
    "QA": "quartz_arenite", "GW": "greywacke", "ARK": "arkose", "SLT": "siltstone", "MD": "mudstone",
    "MDST": "mudstone", "CLST": "mudstone", "SH": "shale", "BSH": "black_shale", "LS": "limestone",
    "LST": "limestone", "DO": "dolomite", "DOL": "dolomite", "SDO": "siliceous_dolomite", "ML": "marl",
    "CK": "chalk", "TR": "travertine", "CH": "chert", "IF": "ironstone", "BIF": "ironstone", "GY": "gypsum",
    "AN": "anhydrite", "HL": "halite", "CO": "coal", "QZ": "quartzite", "QTZ": "quartzite", "MSS": "metasandstone",
    "MB": "marble", "SL": "slate", "PH": "phyllite", "SC": "schist", "SCH": "schist", "QMS": "quartz_schist",
    "CSC": "calc_schist", "GN": "gneiss", "AM": "amphibolite", "HF": "hornfels", "SP": "serpentinite",
    "GR": "granite", "GD": "granodiorite", "DI": "granodiorite", "INT": "granodiorite", "GB": "gabbro",
    "BA": "basalt", "DL": "basalt", "AND": "andesite", "RH": "andesite", "TF": "tuff", "PG": "pegmatite",
    "QV": "pegmatite", "ORE": "sulphide", "MS": "sulphide", "SUL": "sulphide", "BX": "breccia", "FBX": "breccia",
    "MY": "mylonite", "GO": "clay", "CL": "clay", "CLAY": "clay", "SAP": "saprolite", "WR": "saprolite",
}
ROCK_CODE_CLASS: dict = dict(DEFAULT_ROCK_CODES)


def apply_rock_codes(rows) -> None:
    """Use the rock-code table of the project database (risk_rock_code rows, as dicts)."""
    ROCK_CODE_CLASS.clear()
    ROCK_CODE_CLASS.update(DEFAULT_ROCK_CODES)
    for r in rows or []:
        code, cls = str(r.get("code") or "").strip().upper(), str(r.get("rock_class") or "").strip()
        if code and cls:
            ROCK_CODE_CLASS[code] = cls


BEDDING_STYLES = {"massive": N_("Massive"), "thick": N_("Thick-bedded"), "thin": N_("Thin-interbedded"),
                  "alternation": N_("Hard-soft alternation")}
BEDDING_CODE = {"massive": 0, "thick": 1, "thin": 2, "alternation": 3}
CLAY_LEVELS = {"none": N_("None"), "interbeds": N_("Interbeds"), "common": N_("Common")}
CLAY_CODE = {"none": 0, "interbeds": 1, "common": 2}
KARST_LEVELS = {"none": N_("None"), "vuggy": N_("Vuggy"), "cavernous": N_("Cavernous")}
KARST_CODE = {"none": 0, "vuggy": 1, "cavernous": 2}

# --- drilling setup -----------------------------------------------------------------------------
BIT_TYPES = {"impregnated": N_("Impregnated diamond"), "surface_set": N_("Surface-set diamond"),
             "pdc": N_("PDC"), "carbide": N_("Tungsten carbide"), "steel": N_("Steel")}
BIT_CODE = {k: i for i, k in enumerate(BIT_TYPES)}
HOLE_SIZES = {"PQ": 122.6, "HQ": 96.0, "NQ": 75.7, "BQ": 60.0, "AQ": 48.0}     # wireline core, hole mm
SIZE_ORDER = list(HOLE_SIZES)
FLUID_TYPES = {"water": N_("Water"), "polymer": N_("Polymer"), "bentonite": N_("Bentonite mud"),
               "kcl_polymer": N_("KCl-polymer (inhibitive)"), "foam": N_("Foam / air")}

# Status of a warning zone (stored in English, shown with tr()).
ZONE_STATUS = {"Drilled": N_("Drilled"), "At bit": N_("At bit"), "Ahead": N_("Ahead")}

# --- alarm levels (spec 5.8) --------------------------------------------------------------------
LEVELS = ("green", "yellow", "orange", "red")
LEVEL_NAMES = {"green": N_("Green"), "yellow": N_("Yellow"), "orange": N_("Orange"), "red": N_("Red")}
LEVEL_COLORS = {"green": "#2ea043", "yellow": "#e3b341", "orange": "#f0883e", "red": "#f85149"}
LEVEL_RGB = {k: tuple(int(v[i:i + 2], 16) / 255.0 for i in (1, 3, 5)) for k, v in LEVEL_COLORS.items()}
LEVEL_ACTION = {"green": N_("Normal drilling"), "yellow": N_("Watch"), "orange": N_("Prepare"),
                "red": N_("Act now / event likely")}
DEFAULT_THRESHOLDS = (0.15, 0.35, 0.60)       # yellow, orange, red starting levels


def level_of(score: float, thresholds=DEFAULT_THRESHOLDS) -> str:
    y, o, r = thresholds
    if score >= r:
        return "red"
    if score >= o:
        return "orange"
    if score >= y:
        return "yellow"
    return "green"


def level_rank(level: str) -> int:
    return LEVELS.index(level) if level in LEVELS else 0


# --- reason codes (spec 5.8, 5.10) --------------------------------------------------------------
REASONS = {
    "R-K01": N_("{from_}-{to} m: {formation}, {d} m below its top. Cavity probability {p}%."),
    "R-K02": N_("In past holes, {x}% met a cavity in {rock_class} within {d} m below a {contact_type} contact."),
    "R-F01": N_("Hole crosses fault {name} at about {depth} m (model ±{u} m)."),
    "R-F02": N_("Fault-zone probability {p}% at {from_}-{to} m (not shown by the model's fault surfaces)."),
    "R-B01": N_("Angle between hole and bedding is {a}° (< 30°): deviation and bit walk likely."),
    "R-L01": N_("Hard {rock_class} expected {from_}-{to} m: bit glazing and wear risk."),
    "R-C01": N_("{formation} has swelling or weak clay ({clay}): caving and sticking risk."),
    "R-H01": N_("Hole at {pct}% of the rig's rated depth for {size}; {n} size reductions so far."),
    "R-D01": N_("Pump pressure fell {x}% at the same flow: possible fluid loss."),
    "R-D02": N_("ROP {x}× normal for {formation}: possible cavity roof."),
    "R-D03": N_("Pump pressure up {x} MPa at the same flow: inner tube may be blocking."),
    "R-D04": N_("ROP fell {x}% over the last 3 runs at the same RPM in {rock_class}: bit glazing."),
    "R-D05": N_("Dogleg {x}°/30 m at {depth} m (limit 3°): trajectory off plan."),
    "R-D06": N_("Return {x}%: fluid is being lost."),
    "R-D07": N_("Rod drop {x} m / no returns: cavity entered."),
    "R-D08": N_("Torque {x}× the median of the last 5 runs: hole tightening."),
    "R-M01": N_("Logged {lith} at {depth} m; model expected {formation}. Warning zones widened to ±{u} m."),
    "R-M02": N_("{contact} met at {d_act} m, model {d_mod} m (offset {off} m). Predictions below shifted."),
    "R-M03": N_("Nearest model control {dist} m away: uncertainty ±{u} m."),
    "R-M04": N_("{feature} = {value} is outside the range of past areas ({min}-{max}); this prediction is less "
                "reliable."),
}
REASON_TITLES = {
    "R-K01": N_("Carbonate below its top"), "R-K02": N_("Past cavity rate at this level"),
    "R-F01": N_("Fault crossing"), "R-F02": N_("Fault-zone probability"), "R-B01": N_("Low incidence angle"),
    "R-L01": N_("Hard / abrasive rock"), "R-C01": N_("Swelling clay"), "R-H01": N_("Hole state"),
    "R-D01": N_("Pressure drop"), "R-D02": N_("ROP jump"), "R-D03": N_("Pressure rise"),
    "R-D04": N_("ROP decline"), "R-D05": N_("Dogleg"), "R-D06": N_("Low returns"), "R-D07": N_("Rod drop"),
    "R-D08": N_("Torque rise"), "R-M01": N_("Model mismatch"), "R-M02": N_("Contact tie"),
    "R-M03": N_("Far from model control"), "R-M04": N_("Outside experience"),
}


# --- measures (spec 6.1) ------------------------------------------------------------------------
@dataclass(frozen=True)
class Measure:
    code: str
    text: str            # template; values in braces are filled by the system
    failures: tuple
    stage: str           # "P" prevention, "R" response
    title: str = ""      # short name for cards


def _m(code, text, failures, stage, title):
    return code, Measure(code, text, tuple(failures.split(", ")) if failures != "all" else tuple(FAIL_CODES),
                         stage, title)


MEASURES = dict([
    _m("M-OP-01", N_("Reduce bit load to {wob} kN and feed slowly from {from_} to {to} m"), "CV, DV, CL", "P",
       N_("Reduce bit load, feed slowly")),
    _m("M-OP-02", N_("Reduce RPM to {rpm}; avoid the vibration band {band}"), "RF, DV, CB", "P",
       N_("Reduce RPM")),
    _m("M-OP-03", N_("Reduce pump flow to {q} L/min, the minimum for bit cooling"), "LC, CL", "P",
       N_("Reduce pump flow")),
    _m("M-OP-04", N_("Shorten core runs to 1.5 m; pull the inner tube at any pressure rise"), "CB, CL", "P",
       N_("Short core runs")),
    _m("M-OP-05", N_("Keep circulating and rotating; never leave the string still in {from_}-{to} m"), "SP, CL",
       "P", N_("Keep circulating and rotating")),
    _m("M-OP-06", N_("Trip slowly (≤ {v} m/min) through {from_}-{to} m"), "CL, LC", "P", N_("Trip slowly")),
    _m("M-OP-07", N_("Use the operating window of past successful runs: RPM {rpm}, flow {q} L/min, ROP {rop} m/h"),
       "all", "P", N_("Proven operating window")),
    _m("M-FL-01", N_("Switch to {fluid} mud, density {rho} g/cm³, Marsh viscosity {s} s, before {depth} m"),
       "CL, SP, LC", "P", N_("Change drilling fluid")),
    _m("M-FL-02", N_("Pre-treat fluid with fine LCM {c} kg/m³"), "LC", "P", N_("Pre-treat with fine LCM")),
    _m("M-FL-03", N_("Add clay inhibitor (KCl / PHPA) for swelling interbeds"), "CL, SP", "P",
       N_("Clay inhibitor")),
    _m("M-CS-01", N_("Set casing to {depth} m (or reduce hole size {size_from}→{size_to}) before entering the zone"),
       "LC, CL, CV", "P", N_("Set casing before the zone")),
    _m("M-CS-02", N_("Have casing, cement and LCM on site before {depth} m"), "LC, CV, CL", "P",
       N_("Casing, cement and LCM on site")),
    _m("M-BH-01", N_("Use a triple-tube or split-tube barrel through broken rock"), "CB, CL", "P",
       N_("Triple-tube barrel")),
    _m("M-BH-02", N_("Stiffen the assembly: longer barrel, reaming shell and stabilizers"), "DV", "P",
       N_("Stiffen the assembly")),
    _m("M-BH-03", N_("Change bit: softer matrix for quartzite; harder matrix for abrasive broken sandstone"),
       "BG, BW", "P", N_("Change bit matrix")),
    _m("M-SV-01", N_("Survey every {x} m through the zone"), "DV", "P", N_("Survey more often")),
    _m("M-PL-01", N_("Re-plan: change azimuth or dip so the hole meets bedding at > 45°, or move the collar away "
                     "from the risk volume"), "DV, LC, CV", "P", N_("Re-plan the hole")),
    _m("M-LC-01", N_("Cut flow, record loss rate and pressure, classify loss (seepage / partial / total)"), "LC",
       "R", N_("Classify the loss")),
    _m("M-LC-02", N_("Pump fine-to-medium LCM pill, {c} kg/m³"), "LC", "R", N_("Fine LCM pill")),
    _m("M-LC-03", N_("Pump coarse fibre or granular LCM pill"), "LC", "R", N_("Coarse LCM pill")),
    _m("M-LC-04", N_("Spot a sealing gel or silicate system"), "LC", "R", N_("Sealing gel")),
    _m("M-LC-05", N_("Drill ahead without returns using water only, if the hole is stable and the casing point is "
                     "within {x} m; watch bit temperature"), "LC", "R", N_("Drill blind to casing point")),
    _m("M-LC-06", N_("Set a cement plug, wait on cement, redrill"), "LC, CV", "R", N_("Cement plug and redrill")),
    _m("M-LC-07", N_("Run casing through the cavity or loss zone"), "LC, CV", "R", N_("Case through")),
    _m("M-CV-01", N_("Stop feed; measure cavity height by rod drop; survey"), "CV", "R", N_("Stop and measure cavity")),
    _m("M-CV-02", N_("Fill cavity with grout or cement and redrill"), "CV", "R", N_("Grout the cavity and redrill")),
    _m("M-CL-01", N_("Raise mud viscosity and density; circulate the hole clean"), "CL", "R", N_("Raise viscosity")),
    _m("M-CL-02", N_("Ream the interval and make a short wiper trip"), "CL, SP", "R", N_("Ream and wiper trip")),
    _m("M-CL-03", N_("Consolidate with cement and redrill"), "CL", "R", N_("Consolidate with cement")),
    _m("M-SP-01", N_("Keep circulation; work the string up and down; rotate within torque limit {t}"), "SP", "R",
       N_("Work the string")),
    _m("M-SP-02", N_("Spot a lubricant or freshwater pill"), "SP", "R", N_("Lubricant pill")),
    _m("M-SP-03", N_("Apply jarring or hydraulic pull within the rod pull limit {f} kN"), "SP", "R", N_("Jar / pull")),
    _m("M-SP-04", N_("Back off, fish, or sidetrack"), "SP", "R", N_("Fish or sidetrack")),
    _m("M-DV-01", N_("Correct with a wedge or directional tool"), "DV", "R", N_("Wedge / directional correction")),
    _m("M-BG-01", N_("Dress the bit (abrasive pass), then raise bit load briefly"), "BG", "R", N_("Dress the bit")),
    _m("M-WI-01", N_("Raise fluid density; close the collar valve or set a packer"), "WI", "R",
       N_("Control the inflow")),
    _m("M-RF-01", N_("Inspect threads; replace worn rods; lower RPM"), "RF", "R", N_("Inspect rods")),
])

# --- decision table (spec 6.2): failure code -> level -> candidate measures ---------------------
DECISION = {
    "LC": {"yellow": ["M-OP-03", "M-OP-07"], "orange": ["M-FL-02", "M-CS-02", "M-OP-06"],
           "red": ["M-LC-01", "M-LC-02", "M-LC-03", "M-LC-04", "M-LC-06", "M-LC-07"]},
    "CV": {"yellow": ["M-OP-01"], "orange": ["M-OP-01", "M-CS-02", "M-PL-01"],
           "red": ["M-CV-01", "M-LC-06", "M-CV-02", "M-LC-07"]},
    "WI": {"yellow": [], "orange": ["M-FL-01"], "red": ["M-WI-01"]},
    "CL": {"yellow": ["M-OP-04", "M-OP-06"], "orange": ["M-FL-01", "M-FL-03", "M-BH-01", "M-CS-01"],
           "red": ["M-CL-01", "M-CL-02", "M-CL-03"]},
    "SP": {"yellow": ["M-OP-05"], "orange": ["M-FL-01", "M-CL-02"],
           "red": ["M-SP-01", "M-SP-02", "M-SP-03", "M-SP-04"]},
    "CB": {"yellow": ["M-OP-04"], "orange": ["M-BH-01"], "red": ["M-CL-01"]},
    "DV": {"yellow": ["M-SV-01"], "orange": ["M-BH-02", "M-OP-01", "M-PL-01"], "red": ["M-DV-01"]},
    "BG": {"yellow": ["M-OP-07"], "orange": ["M-BH-03"], "red": ["M-BG-01"]},
    "BW": {"yellow": ["M-OP-07"], "orange": ["M-BH-03"], "red": ["M-BG-01"]},
    "RF": {"yellow": ["M-OP-02"], "orange": ["M-RF-01"], "red": ["M-SP-04"]},
}
# Text shown in place of a measure where the decision table has an action that is not coded.
DECISION_NOTES = {("WI", "yellow"): N_("Watch flow-out at the collar"),
                  ("WI", "orange"): N_("Collar valve on standby"),
                  ("CB", "red"): N_("Pull the inner tube first"),
                  ("BG", "red"): N_("Bit change"), ("BW", "red"): N_("Bit change")}
# Expert order (used until outcomes build up): lower = offered first.
EXPERT_ORDER = {code: i for i, code in enumerate(MEASURES)}
DEFAULT_MEASURES = dict(MEASURES)
DEFAULT_EXPERT_ORDER = dict(EXPERT_ORDER)


def measure_rows() -> list[dict]:
    """The measure library as rows of risk_measure (defaults)."""
    return [{"code": m.code, "title": m.title, "text": m.text, "failures": ", ".join(m.failures), "stage": m.stage,
             "expert_order": DEFAULT_EXPERT_ORDER.get(m.code, 99)} for m in DEFAULT_MEASURES.values()]


def apply_measures(rows) -> None:
    """Use the measure library of the project database (edited texts, failures, stage, expert order)."""
    MEASURES.clear()
    MEASURES.update(DEFAULT_MEASURES)
    EXPERT_ORDER.clear()
    EXPERT_ORDER.update(DEFAULT_EXPERT_ORDER)
    for r in rows or []:
        code = str(r.get("code") or "").strip()
        if not code or not r.get("text"):
            continue
        fails = tuple(x.strip().upper() for x in str(r.get("failures") or "").replace(";", ",").split(",") if x.strip())
        base = DEFAULT_MEASURES.get(code)
        MEASURES[code] = Measure(code, str(r["text"]), fails or (base.failures if base else tuple(FAIL_CODES)),
                                 str(r.get("stage") or (base.stage if base else "P")).upper()[:1],
                                 str(r.get("title") or (base.title if base else code)))
        try:
            EXPERT_ORDER[code] = int(float(r.get("expert_order")))
        except (TypeError, ValueError):
            EXPERT_ORDER.setdefault(code, 99)
# Which family a measure helps most (for evidence lookups).
EVIDENCE_GROUPS = ("M-CV", "M-LC", "M-CL", "M-CS")     # ranked by solved rate of similar occurrences
PARAM_MEASURES = ("M-OP-01", "M-OP-02", "M-OP-03", "M-OP-07", "M-BH-03")   # filled from successful holes

# --- alarm rules (spec 5.7) ---------------------------------------------------------------------
RULES = {   # rule -> (condition text, family, fail code, level, reason code)
    "R1": (N_("Pump pressure falls > 25% at the same flow within one interval"), "karst", "LC", "orange", "R-D01"),
    "R2": (N_("ROP > 3× the median of the last 5 intervals in the same formation"), "karst", "CV", "red", "R-D02"),
    "R3": (N_("Pump pressure rises > 0.5 MPa at the same flow while ROP falls"), "instability", "CB", "yellow",
           "R-D03"),
    "R4": (N_("ROP falls > 40% over 3 intervals at the same RPM, in quartzite or dolomite"), "bit", "BG", "yellow",
           "R-D04"),
    "R5": (N_("Dogleg > 3° per 30 m, or survey off plan beyond tolerance"), "trajectory", "DV", "orange", "R-D05"),
    "R6": (N_("Return % < 70, or falls by more than 30 points"), "karst", "LC", "orange", "R-D06"),
    "R7": (N_("Return % = 0, or rod drop ≥ 0.2 m"), "karst", "CV", "red", "R-D07"),
    "R8": (N_("Torque > 1.3× the median of the last 5 intervals"), "instability", "SP", "orange", "R-D08"),
}


def family_of(code: str) -> str:
    return FAIL_CODES.get(str(code).upper(), ("", "karst"))[1]


def rock_class_of(rock: str, formation_class: str | None = None) -> str:
    """The rock class of a logged rock code (or name); ``formation_class`` when nothing matches."""
    r = str(rock or "").strip()
    if r.upper() in ROCK_CODE_CLASS:
        return ROCK_CODE_CLASS[r.upper()]
    low = r.lower()
    if low in ROCK_CLASSES:
        return low
    for cls in ROCK_CLASSES:
        if cls in low or low.startswith(cls[:4]):
            return cls
    return formation_class or "sandstone"
