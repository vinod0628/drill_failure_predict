"""Mechanical and drilling properties of rock types - the main control on drilling failure.

Typical intact-rock values compiled from the standard references (ranges vary with grain size,
cementation, weathering and fabric; the *typical* value is a central value for fresh rock):

* strength / stiffness: Goodman, *Introduction to Rock Mechanics* (2nd ed., 1989), Lama & Vutukuri,
  *Handbook on Mechanical Properties of Rocks* (1978), Hoek & Brown (1997) and Hoek, *Practical Rock
  Engineering* (2007) for the Hoek-Brown constant mi, Bieniawski (1989);
* strength classes: ISRM suggested methods (1981): UCS < 25 MPa weak ... > 250 MPa extremely strong;
* abrasivity: Cerchar abrasivity index CAI (ASTM D7625; Plinninger & Restner 2008, Käsling & Thuro 2010)
  and equivalent quartz content (Thuro 1997);
* drillability: NTNU Drilling Rate Index DRI (Bruland 1998; Dahl et al. 2012);
* behaviour: solubility (karst: carbonates, evaporites), swelling (clay minerals, anhydrite -> gypsum),
  anisotropy (foliation / cleavage: slate, phyllite, schist, gneiss, shale bedding).

Used two ways: as default properties of a logged rock (or a formation card without lab values), and in
the demo history as the "truth" of the failure mechanics. A formation card's measured UCS / CAI replace
the defaults. No Qt here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..i18n import N_


@dataclass(frozen=True)
class RockProps:
    name: str
    group: str                 # clastic / carbonate / chemical / metamorphic / igneous / fault / weathered / ore
    ucs: tuple                 # uniaxial compressive strength (MPa): min, typical, max
    tensile: float             # tensile strength (MPa), typical
    e_mod: float               # Young's modulus (GPa), typical
    poisson: float
    density: float             # g/cm3
    porosity: float            # %, typical
    cai: tuple                 # Cerchar abrasivity index: min, typical, max
    quartz_eq: float           # equivalent quartz content %, typical
    dri: float                 # NTNU drilling rate index, typical (higher = easier)
    mi: float                  # Hoek-Brown constant
    carbonate: float           # carbonate share % (karst potential)
    soluble: int = 0           # 0 no, 1 slightly (gypsum in time / anhydrite), 2 soluble (carbonate, halite)
    swelling: int = 0          # 0 none, 1 some, 2 strong (smectite clays, anhydrite)
    anisotropy: int = 0        # 0 isotropic, 1 moderate, 2 strong (foliation / cleavage)
    clastic: bool = False
    fractures: int = 1         # tendency to be broken / jointed in cores: 0 low, 1 normal, 2 high

    @property
    def ucs_typ(self) -> float:
        return float(self.ucs[1])

    @property
    def cai_typ(self) -> float:
        return float(self.cai[1])

    @property
    def brittleness(self) -> float:
        """UCS / tensile strength (B1): high = brittle, chips, may glaze the bit."""
        return self.ucs_typ / max(self.tensile, 0.1)


# Rock class -> properties. (min, typical, max) for UCS (MPa) and CAI.
ROCKS: dict[str, RockProps] = {
    # --- clastic sedimentary ---
    "conglomerate": RockProps(N_("Conglomerate"), "clastic", (30, 90, 230), 8, 30, 0.22, 2.50, 10, (2.0, 3.2, 4.5),
                              70, 50, 21, 10, clastic=True, fractures=1),
    "breccia_sed": RockProps(N_("Sedimentary breccia"), "clastic", (25, 70, 200), 6, 25, 0.24, 2.50, 12,
                             (1.5, 2.8, 4.0), 55, 52, 19, 20, clastic=True, fractures=2),
    "sandstone": RockProps(N_("Sandstone"), "clastic", (20, 80, 170), 8, 20, 0.20, 2.35, 12, (1.5, 3.0, 4.5), 75, 55,
                           17, 5, clastic=True),
    "quartz_arenite": RockProps(N_("Quartz arenite (orthoquartzite)"), "clastic", (80, 160, 280), 12, 40, 0.17, 2.55,
                                5, (3.0, 4.2, 5.2), 92, 38, 18, 0, clastic=True),
    "greywacke": RockProps(N_("Greywacke"), "clastic", (80, 150, 220), 12, 45, 0.22, 2.65, 3, (2.5, 3.3, 4.2), 55,
                           42, 18, 5, clastic=True),
    "arkose": RockProps(N_("Arkose"), "clastic", (40, 90, 160), 8, 25, 0.22, 2.45, 10, (2.0, 3.2, 4.2), 65, 50, 17, 5,
                        clastic=True),
    "siltstone": RockProps(N_("Siltstone"), "clastic", (20, 50, 100), 5, 13, 0.25, 2.45, 10, (0.8, 1.5, 2.5), 45, 60,
                           7, 5, clastic=True, anisotropy=1),
    "mudstone": RockProps(N_("Mudstone / claystone"), "clastic", (5, 20, 50), 2, 5, 0.30, 2.30, 18, (0.3, 0.6, 1.2),
                          20, 70, 4, 5, swelling=1, clastic=True, fractures=2),
    "shale": RockProps(N_("Shale"), "clastic", (10, 40, 100), 4, 12, 0.28, 2.45, 10, (0.5, 1.0, 1.8), 25, 65, 6, 8,
                       swelling=1, anisotropy=2, clastic=True, fractures=2),
    "black_shale": RockProps(N_("Black (carbonaceous) shale"), "clastic", (10, 35, 80), 3, 10, 0.28, 2.40, 10,
                             (0.6, 1.1, 2.0), 30, 65, 6, 5, swelling=1, anisotropy=2, clastic=True, fractures=2),
    # --- carbonate ---
    "limestone": RockProps(N_("Limestone"), "carbonate", (30, 90, 250), 8, 45, 0.27, 2.60, 5, (0.5, 1.0, 2.0), 3, 55,
                           10, 95, soluble=2),
    "dolomite": RockProps(N_("Dolomite"), "carbonate", (60, 120, 300), 10, 55, 0.27, 2.75, 4, (0.8, 1.4, 2.5), 5, 50,
                          9, 95, soluble=2),
    "siliceous_dolomite": RockProps(N_("Siliceous / cherty dolomite"), "carbonate", (120, 200, 350), 14, 70, 0.22,
                                    2.75, 2, (2.0, 3.2, 4.8), 40, 35, 10, 75, soluble=1),
    "marl": RockProps(N_("Marl"), "carbonate", (5, 25, 60), 3, 8, 0.30, 2.40, 18, (0.2, 0.5, 1.0), 12, 70, 7, 50,
                      soluble=1, swelling=1, fractures=2),
    "chalk": RockProps(N_("Chalk"), "carbonate", (2, 10, 30), 1, 4, 0.25, 1.90, 35, (0.1, 0.2, 0.4), 2, 85, 7, 98,
                       soluble=2),
    "travertine": RockProps(N_("Travertine / tufa"), "carbonate", (10, 40, 90), 4, 20, 0.25, 2.30, 15, (0.2, 0.5, 1.0),
                            0, 65, 10, 98, soluble=2, fractures=2),
    # --- chemical / evaporite / organic ---
    "chert": RockProps(N_("Chert / flint"), "chemical", (200, 300, 400), 20, 80, 0.15, 2.60, 1, (4.0, 5.0, 6.0), 97,
                       25, 20, 0),
    "ironstone": RockProps(N_("Ironstone / banded iron formation"), "chemical", (150, 250, 400), 20, 90, 0.20, 3.30, 2,
                           (3.5, 4.5, 5.5), 70, 28, 18, 0, anisotropy=1),
    "gypsum": RockProps(N_("Gypsum"), "chemical", (10, 25, 40), 2, 15, 0.30, 2.30, 5, (0.1, 0.2, 0.5), 0, 78, 10, 0,
                        soluble=2),
    "anhydrite": RockProps(N_("Anhydrite"), "chemical", (40, 80, 120), 6, 60, 0.30, 2.90, 2, (0.3, 0.6, 1.0), 0, 60, 12,
                           0, soluble=1, swelling=2),
    "halite": RockProps(N_("Rock salt (halite)"), "chemical", (15, 25, 35), 1.5, 20, 0.35, 2.20, 1, (0.1, 0.1, 0.3), 0,
                        80, 6, 0, soluble=2),
    "coal": RockProps(N_("Coal"), "chemical", (5, 20, 40), 1.5, 4, 0.30, 1.40, 8, (0.1, 0.3, 0.5), 2, 80, 8, 0,
                      fractures=2, clastic=True),
    # --- metamorphic ---
    "quartzite": RockProps(N_("Quartzite (metamorphic)"), "metamorphic", (150, 220, 350), 15, 70, 0.15, 2.65, 1,
                           (3.5, 4.5, 5.5), 95, 30, 20, 0),
    "metasandstone": RockProps(N_("Metasandstone / metasiltstone"), "metamorphic", (80, 140, 240), 12, 50, 0.20, 2.65,
                               2, (2.5, 3.5, 4.5), 70, 40, 17, 0, anisotropy=1),
    "marble": RockProps(N_("Marble"), "metamorphic", (50, 100, 200), 8, 60, 0.25, 2.70, 1, (0.8, 1.2, 2.0), 2, 50, 9,
                        95, soluble=2),
    "slate": RockProps(N_("Slate"), "metamorphic", (50, 120, 200), 12, 50, 0.20, 2.75, 2, (1.5, 2.3, 3.5), 35, 45, 7, 0,
                       anisotropy=2, fractures=1),
    "phyllite": RockProps(N_("Phyllite"), "metamorphic", (20, 50, 120), 6, 25, 0.25, 2.70, 3, (1.0, 1.8, 3.0), 35, 52,
                          7, 0, anisotropy=2, swelling=1, fractures=2),
    "schist": RockProps(N_("Mica schist"), "metamorphic", (20, 70, 160), 7, 30, 0.20, 2.75, 2, (2.0, 3.0, 4.0), 45, 45,
                        10, 0, anisotropy=2, fractures=1),
    "quartz_schist": RockProps(N_("Quartz-mica schist"), "metamorphic", (60, 120, 200), 10, 45, 0.18, 2.70, 2,
                               (3.0, 3.8, 4.8), 65, 38, 12, 0, anisotropy=2),
    "calc_schist": RockProps(N_("Calc-schist"), "metamorphic", (30, 70, 150), 6, 35, 0.24, 2.75, 2, (1.0, 1.8, 3.0),
                             20, 50, 10, 40, soluble=1, anisotropy=2),
    "gneiss": RockProps(N_("Gneiss"), "metamorphic", (80, 150, 250), 12, 60, 0.22, 2.75, 1, (3.0, 4.0, 5.0), 55, 38, 28,
                        0, anisotropy=1),
    "amphibolite": RockProps(N_("Amphibolite"), "metamorphic", (100, 200, 300), 15, 85, 0.25, 3.00, 1, (2.5, 3.2, 4.0),
                             20, 38, 26, 0, anisotropy=1),
    "hornfels": RockProps(N_("Hornfels"), "metamorphic", (150, 250, 350), 20, 85, 0.20, 2.80, 1, (3.0, 4.0, 5.0), 45,
                          30, 19, 0),
    "serpentinite": RockProps(N_("Serpentinite"), "metamorphic", (30, 80, 150), 6, 30, 0.28, 2.60, 3, (0.5, 1.2, 2.0),
                              3, 55, 8, 0, swelling=1, fractures=2),
    # --- igneous ---
    "granite": RockProps(N_("Granite"), "igneous", (100, 180, 300), 12, 50, 0.22, 2.65, 1, (3.5, 4.5, 5.5), 65, 40, 32,
                         0),
    "granodiorite": RockProps(N_("Granodiorite / diorite"), "igneous", (100, 180, 300), 12, 60, 0.24, 2.75, 1,
                              (3.0, 3.8, 4.5), 45, 40, 28, 0),
    "gabbro": RockProps(N_("Gabbro / norite"), "igneous", (150, 210, 300), 15, 85, 0.26, 2.95, 1, (2.5, 3.2, 4.0), 10,
                        38, 27, 0),
    "basalt": RockProps(N_("Basalt / dolerite"), "igneous", (100, 200, 350), 15, 70, 0.25, 2.85, 2, (2.0, 2.8, 3.5), 10,
                        38, 25, 0, fractures=1),
    "andesite": RockProps(N_("Andesite / rhyolite"), "igneous", (80, 150, 250), 12, 50, 0.23, 2.60, 5, (2.0, 3.0, 4.5),
                          35, 42, 25, 0),
    "tuff": RockProps(N_("Tuff"), "igneous", (5, 30, 80), 3, 8, 0.25, 2.00, 25, (0.5, 1.5, 2.5), 25, 65, 13, 0,
                      swelling=1, fractures=1),
    "pegmatite": RockProps(N_("Pegmatite / quartz vein"), "igneous", (100, 170, 250), 10, 55, 0.20, 2.65, 1,
                           (4.0, 4.8, 6.0), 85, 35, 30, 0, fractures=1),
    # --- ore ---
    "sulphide": RockProps(N_("Massive / disseminated sulphide"), "ore", (80, 150, 250), 10, 80, 0.25, 3.80, 2,
                          (1.5, 2.5, 3.5), 25, 45, 15, 10),
    # --- fault and weathered rock ---
    "breccia": RockProps(N_("Fault breccia / cataclasite"), "fault", (5, 25, 80), 1.5, 8, 0.28, 2.40, 15, (1.0, 2.0, 3.5),
                         40, 65, 12, 20, fractures=2),
    "mylonite": RockProps(N_("Mylonite"), "fault", (50, 100, 200), 8, 40, 0.22, 2.70, 2, (2.0, 3.0, 4.0), 45, 45, 12, 0,
                          anisotropy=2),
    "clay": RockProps(N_("Fault gouge / clay / cave fill"), "fault", (0.1, 1, 5), 0.1, 0.3, 0.40, 1.90, 40,
                      (0.1, 0.3, 0.8), 10, 90, 4, 0, swelling=2, clastic=True, fractures=2),
    "saprolite": RockProps(N_("Saprolite / weathered rock"), "weathered", (0.5, 3, 10), 0.3, 1, 0.35, 1.90, 35,
                           (0.5, 1.0, 2.0), 30, 85, 5, 0, swelling=1, fractures=2),
}

# Older / short class names -> a class above
ALIASES = {"siliceous dolomite": "siliceous_dolomite", "orthoquartzite": "quartz_arenite", "claystone": "mudstone",
           "flint": "chert", "bif": "ironstone", "banded iron": "ironstone", "dolerite": "basalt", "diabase": "basalt",
           "diorite": "granodiorite", "norite": "gabbro", "rhyolite": "andesite", "dacite": "andesite",
           "gouge": "clay", "cataclasite": "breccia", "laterite": "saprolite", "mica schist": "schist"}


def strength_class(ucs: float) -> int:
    """ISRM (1981) strength grade as 1-5: < 25 weak, 25-50 medium, 50-100 strong, 100-250 very strong,
    > 250 MPa extremely strong."""
    if ucs is None or math.isnan(ucs):
        return 3
    return 1 if ucs < 25 else 2 if ucs < 50 else 3 if ucs < 100 else 4 if ucs < 250 else 5


STRENGTH_NAMES = {1: N_("weak (< 25 MPa)"), 2: N_("medium strong (25-50 MPa)"), 3: N_("strong (50-100 MPa)"),
                  4: N_("very strong (100-250 MPa)"), 5: N_("extremely strong (> 250 MPa)")}


def abrasivity_class(cai: float) -> int:
    """CAI as 1-5 (ASTM D7625 / Plinninger): < 1 low, 1-2 moderate, 2-3 high, 3-4 very high, >= 4 extreme."""
    if cai is None or math.isnan(cai):
        return 3
    return 1 if cai < 1 else 2 if cai < 2 else 3 if cai < 3 else 4 if cai < 4 else 5


ABRASIVITY_NAMES = {1: N_("low (CAI < 1)"), 2: N_("moderate (CAI 1-2)"), 3: N_("high (CAI 2-3)"),
                    4: N_("very high (CAI 3-4)"), 5: N_("extremely abrasive (CAI ≥ 4)")}


def props(rock_class: str) -> RockProps:
    k = ALIASES.get(rock_class, rock_class)
    return ROCKS.get(k, ROCKS["sandstone"])


# --- the editable table in the project database (risk_rock) -------------------------------------
DEFAULTS: dict[str, RockProps] = dict(ROCKS)
FIELDS = ("name", "group", "ucs_min", "ucs", "ucs_max", "tensile", "e_mod", "poisson", "density", "porosity",
          "cai_min", "cai", "cai_max", "quartz_eq", "dri", "mi", "carbonate", "soluble", "swelling", "anisotropy",
          "clastic", "fractures")


def to_row(key: str, p: RockProps) -> dict:
    return {"rock_class": key, "name": p.name, "group": p.group, "ucs_min": p.ucs[0], "ucs": p.ucs[1],
            "ucs_max": p.ucs[2], "tensile": p.tensile, "e_mod": p.e_mod, "poisson": p.poisson, "density": p.density,
            "porosity": p.porosity, "cai_min": p.cai[0], "cai": p.cai[1], "cai_max": p.cai[2], "quartz_eq": p.quartz_eq,
            "dri": p.dri, "mi": p.mi, "carbonate": p.carbonate, "soluble": p.soluble, "swelling": p.swelling,
            "anisotropy": p.anisotropy, "clastic": "Y" if p.clastic else "N", "fractures": p.fractures, "notes": None}


def default_rows() -> list[dict]:
    return [to_row(k, p) for k, p in DEFAULTS.items()]


def _f(v, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def from_row(r: dict, base: RockProps | None = None) -> RockProps:
    """Properties from a table row; blank cells keep the default (or ``base``) value."""
    b = base or DEFAULTS.get(str(r.get("rock_class")), DEFAULTS["sandstone"])
    i = lambda k, d: int(round(_f(r.get(k), d)))  # noqa: E731
    clastic = str(r.get("clastic") or ("Y" if b.clastic else "N")).upper() == "Y"
    return RockProps(str(r.get("name") or b.name), str(r.get("group") or b.group),
                     (_f(r.get("ucs_min"), b.ucs[0]), _f(r.get("ucs"), b.ucs[1]), _f(r.get("ucs_max"), b.ucs[2])),
                     _f(r.get("tensile"), b.tensile), _f(r.get("e_mod"), b.e_mod), _f(r.get("poisson"), b.poisson),
                     _f(r.get("density"), b.density), _f(r.get("porosity"), b.porosity),
                     (_f(r.get("cai_min"), b.cai[0]), _f(r.get("cai"), b.cai[1]), _f(r.get("cai_max"), b.cai[2])),
                     _f(r.get("quartz_eq"), b.quartz_eq), _f(r.get("dri"), b.dri), _f(r.get("mi"), b.mi),
                     _f(r.get("carbonate"), b.carbonate), i("soluble", b.soluble), i("swelling", b.swelling),
                     i("anisotropy", b.anisotropy), clastic, i("fractures", b.fractures))


def apply_table(rows) -> int:
    """Use the rock properties of the project database (risk_rock rows, as dicts) from now on: defaults for
    rock classes not in the table, the table's values (blank = default) for the others. Returns the
    number of rock classes taken from the table."""
    ROCKS.clear()
    ROCKS.update(DEFAULTS)
    n = 0
    for r in rows or []:
        key = str(r.get("rock_class") or "").strip()
        if not key:
            continue
        ROCKS[key] = from_row(r, DEFAULTS.get(key))
        n += 1
    try:
        from . import codes

        codes.refresh_rock_classes()
    except ImportError:
        pass
    return n
