"""Demo drilling records: a process-based synthetic history of past holes, and records for the holes of
the demo workspace (finished, in progress, planned).

The history imitates the setting of the design spec (1.): core holes through Proterozoic to Palaeozoic
sedimentary basins (schist, dolomite, quartzite, limestone, shale, ...), from several areas far apart that share rock types but
not formations. Each area has its own column (formations missing, thickness varying laterally), dips,
faults and karst setting. Each hole is drilled metre by metre:

* the trajectory walks toward the bedding normal when the hole meets bedding at a low angle;
* cavities are most likely in cavernous carbonate just below a clastic-over-carbonate contact, above
  the water table, in strongly karstified areas and near faults; fault zones are crossed where the
  hole passes a fault;
* problems follow the causes of spec 2 (losses / cavities, caving and sticking in fault zones and
  swelling clay, deviation at low incidence angles, glazing and wear in hard abrasive rock); a problem
  is solved by measures with their own success rates, or the hole fails;
* the runs carry the precursors (pressure drop before losses, ROP jump at a cavity roof, pressure
  rise before blockage, ROP decline when the bit glazes); return %, rod drop and torque only in the
  newer areas (spec 3.3).

Deterministic from the seed. No Qt here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import codes

# --- formation catalogue of the past areas (spec 3.5) ---------------------------------------------
# name, age, main, main%, second, second%, carb%, hardness, abrasiveness, bedding, clay, karst, tmin, tmax, equiv
CATALOGUE = [
    # Proterozoic basement and lower basin fill
    ("PS1", "Mesoproterozoic", "schist", 85, "quartzite", 15, 0, 3, 3, "thin", "interbeds", "none", 60, 300, "U6"),
    ("PS2", "Neoproterozoic", "schist", 60, "quartzite", 40, 0, 4, 4, "alternation", "none", "none", 30, 200, ""),
    ("PQ1", "Neoproterozoic", "quartzite", 85, "schist", 15, 0, 5, 5, "massive", "none", "none", 20, 120, ""),
    ("PD1", "Neoproterozoic", "dolomite", 80, "schist", 20, 80, 4, 3, "thick", "none", "vuggy", 20, 130, ""),
    ("PS3", "Neoproterozoic", "shale", 70, "schist", 30, 5, 2, 2, "thin", "common", "none", 10, 80, ""),
    ("PSL", "Neoproterozoic", "slate", 70, "phyllite", 30, 0, 4, 3, "thin", "interbeds", "none", 30, 180, ""),
    ("PIF", "Palaeoproterozoic", "ironstone", 75, "chert", 25, 0, 5, 5, "thin", "none", "none", 15, 90, ""),
    ("PMB", "Mesoproterozoic", "marble", 85, "calc_schist", 15, 90, 3, 2, "thick", "none", "cavernous", 20, 120, ""),
    # Palaeozoic cover
    ("QZ1", "Cambrian", "quartzite", 85, "sandstone", 15, 0, 5, 5, "massive", "none", "none", 20, 110, ""),
    ("SH1", "Cambrian", "shale", 75, "siltstone", 25, 10, 1, 1, "thin", "common", "none", 10, 60, "U2"),
    ("DO3", "Cambrian", "dolomite", 85, "shale", 15, 85, 4, 3, "alternation", "interbeds", "vuggy", 30, 150, ""),
    ("LS1", "Ordovician", "limestone", 85, "shale", 15, 90, 3, 2, "thick", "none", "cavernous", 30, 150, ""),
    ("DO1", "Ordovician", "dolomite", 90, "limestone", 10, 95, 4, 3, "massive", "none", "cavernous", 20, 140,
     "U3"),
    ("DO2", "Ordovician", "dolomite", 75, "chert", 25, 75, 5, 4, "thick", "none", "vuggy", 15, 90, ""),
    ("ML1", "Ordovician", "marl", 70, "shale", 30, 50, 2, 1, "thin", "interbeds", "vuggy", 10, 50, ""),
    ("SL1", "Silurian", "siltstone", 70, "shale", 30, 10, 2, 2, "thin", "interbeds", "none", 15, 80, "U4"),
    ("QZ2", "Silurian", "quartzite", 90, "sandstone", 10, 0, 5, 5, "massive", "none", "none", 10, 70, ""),
    ("SS2", "Devonian", "sandstone", 70, "conglomerate", 30, 5, 3, 4, "alternation", "interbeds", "none", 20,
     100, "U1"),
    ("LS2", "Devonian", "limestone", 90, "marl", 10, 95, 3, 2, "thick", "none", "vuggy", 20, 120, ""),
]
# The demo workspace's own formations (never drilled in the past areas: predicted from their cards).
WORKSPACE_CARDS = [
    ("U1", "Devonian", "sandstone", 85, "siltstone", 15, 5, 3, 4, "thick", "none", "none", 90, 160, "SS2"),
    ("U2", "Devonian", "shale", 80, "siltstone", 20, 10, 1, 1, "thin", "common", "none", 25, 50, "SH1"),
    ("U3", "Devonian", "dolomite", 90, "limestone", 10, 95, 4, 3, "massive", "none", "cavernous", 20, 70, "DO1"),
    ("U4", "Devonian", "siltstone", 75, "shale", 25, 10, 2, 2, "thin", "interbeds", "none", 30, 70, "SL1"),
    ("U5", "Silurian", "tuff", 90, "siltstone", 10, 0, 3, 3, "thick", "none", "none", 10, 30, ""),
    ("U6", "Silurian", "schist", 90, "quartzite", 10, 0, 4, 4, "massive", "none", "none", 100, 400, ""),
    ("ORE", "", "sulphide", 80, "dolomite", 20, 20, 4, 3, "massive", "none", "none", 1, 10, ""),
    ("QV", "", "sulphide", 90, "quartzite", 10, 0, 5, 4, "massive", "none", "none", 1, 5, ""),
    ("MS", "", "sulphide", 95, "dolomite", 5, 5, 4, 3, "massive", "none", "none", 1, 20, ""),
    ("INT", "", "granite", 100, None, None, 0, 4, 4, "massive", "none", "none", 10, 300, ""),
]
ROCK_CODE = {"slate": "SL", "phyllite": "PH", "ironstone": "BIF", "marble": "MB", "calc_schist": "CSC",
             "quartzite": "QZ", "sandstone": "SS", "shale": "SH", "siltstone": "SLT", "limestone": "LS",
             "dolomite": "DO", "marl": "ML", "chert": "CH", "conglomerate": "CG", "tuff": "TF", "schist": "SC",
             "granite": "GR", "sulphide": "ORE", "marble": "DO"}
# Columns of the past areas (top to bottom); some formations go missing in some areas.
COLUMNS = {      # Proterozoic - Palaeozoic basins; formations go missing from area to area
    "A01": ["SS2", "SH1", "DO1", "LS1", "QZ1", "PS1"],
    "A02": ["ML1", "LS2", "DO2", "SL1", "PSL", "PQ1", "PS2"],
    "A03": ["SL1", "SH1", "LS1", "DO1", "DO3", "PS1"],
    "A04": ["PS3", "PD1", "PIF", "PQ1", "PS2", "PS1"],
    "A05": ["SS2", "SH1", "LS2", "DO3", "QZ1", "PS1"],
    "A06": ["SL1", "QZ2", "DO2", "LS1", "SH1", "PSL", "PS2"],
    "A07": ["SH1", "DO1", "DO2", "PS3", "PMB", "PS1"],
    "A08": ["PS3", "PD1", "PIF", "PQ1", "PS1"],
}
RIGS = [
    {"rig_model": "CS14", "depth_PQ": 350, "depth_HQ": 700, "depth_NQ": 1000, "depth_BQ": 1300, "max_rpm": 1300,
     "pump_lpm": 120, "pull_kn": 90, "max_torque": 2400},
    {"rig_model": "LF90", "depth_PQ": 300, "depth_HQ": 600, "depth_NQ": 850, "depth_BQ": 1100, "max_rpm": 1200,
     "pump_lpm": 100, "pull_kn": 80, "max_torque": 2000},
    {"rig_model": "XY-44", "depth_PQ": 200, "depth_HQ": 420, "depth_NQ": 620, "depth_BQ": 800, "max_rpm": 1000,
     "pump_lpm": 90, "pull_kn": 60, "max_torque": 1600},
]
# Truth: how often a measure solves a problem (unknown to the system; it learns the rates from the records).
MEASURE_SUCCESS = {
    "M-LC-01": 0.15, "M-LC-02": 0.45, "M-LC-03": 0.55, "M-LC-04": 0.65, "M-LC-06": 0.75, "M-LC-07": 0.92,
    "M-CV-01": 0.20, "M-CV-02": 0.80, "M-CL-01": 0.55, "M-CL-02": 0.50, "M-CL-03": 0.80, "M-SP-01": 0.45,
    "M-SP-02": 0.40, "M-SP-03": 0.55, "M-SP-04": 0.70, "M-DV-01": 0.55, "M-BG-01": 0.60, "M-WI-01": 0.70,
    "M-RF-01": 0.60, "M-CS-01": 0.85, "M-FL-01": 0.55, "M-BH-01": 0.60, "M-BH-03": 0.65,
}
RESPONSE = {"LC": ["M-LC-01", "M-LC-02", "M-LC-03", "M-LC-04", "M-LC-06", "M-LC-07"],
            "CV": ["M-CV-01", "M-CV-02", "M-LC-06", "M-LC-07"],
            "CL": ["M-CL-01", "M-CL-02", "M-CL-03", "M-CS-01"], "SP": ["M-SP-01", "M-SP-02", "M-SP-03", "M-SP-04"],
            "CB": ["M-BH-01", "M-CL-01"], "DV": ["M-DV-01"], "BG": ["M-BG-01", "M-BH-03"],
            "BW": ["M-BH-03"], "RF": ["M-RF-01"], "WI": ["M-WI-01"]}
OCC_OF = {"LC": "loss", "CV": "cavity", "WI": "water", "CL": "broken", "SP": "clay", "CB": "broken", "DV": "other",
          "BG": "other", "BW": "other", "RF": "other"}
NEW_FIELDS_AREAS = {"A07", "A08", "SB3"}      # areas where return %, rod drop and torque are recorded


def cards_frame(rows) -> pd.DataFrame:
    cols = ["formation", "age", "main_rock", "main_pct", "second_rock", "second_pct", "carbonate_pct", "hardness",
            "abrasiveness", "bedding", "swelling_clay", "karst", "thick_min", "thick_max", "equivalents"]
    df = pd.DataFrame(rows, columns=cols)
    df["notes"] = None
    return df


def cards_dict(df: pd.DataFrame) -> dict:
    return {r["formation"]: r for r in df.to_dict("records")}


# --- the geometry of a past area ------------------------------------------------------------------
@dataclass
class FaultPlane:
    name: str
    point: np.ndarray
    normal: np.ndarray        # unit, pointing to the hanging wall
    throw: float
    half_width: float         # damage zone half width (m)

    def signed(self, p: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(p) - self.point) @ self.normal


@dataclass
class AreaWorld:
    area: dict
    column: list[str]
    cards: dict
    rng: np.random.Generator
    karst_factor: float = 1.0
    size: float = 2000.0
    z0: float = 600.0
    faults: list[FaultPlane] = field(default_factory=list)

    def __post_init__(self):
        r = self.rng
        a = self.area
        self.dip = float(r.uniform(a["dip_min"], a["dip_max"]))
        self.dipdir = float(r.uniform(0, 360))
        dd, dp = math.radians(self.dipdir), math.radians(self.dip)
        self.normal = np.array([math.sin(dp) * math.sin(dd), math.sin(dp) * math.cos(dd), math.cos(dp)])
        self.tmean, self.tamp, self.phase = [], [], []
        for f in self.column:
            c = self.cards[f]
            tmin, tmax = float(c["thick_min"]), float(c["thick_max"])
            m = r.uniform(tmin + 0.25 * (tmax - tmin), tmax - 0.25 * (tmax - tmin))
            self.tmean.append(m)
            self.tamp.append(min(m - tmin, tmax - m) * r.uniform(0.8, 1.0))
            self.phase.append(r.uniform(0, 2 * math.pi, 4))
        self.tmean[-1] = 3000.0                     # the lowest unit goes on
        self.tamp[-1] = 0.0
        nf = max(0, int(round(float(a["faults_per_km"]) * self.size / 1000.0 * r.uniform(0.7, 1.3))))
        for i in range(nf):
            strike = r.uniform(0, 360)
            fdip = r.uniform(58, 86)
            dd = math.radians(strike + 90)
            n = np.array([math.sin(math.radians(fdip)) * math.sin(dd), math.sin(math.radians(fdip)) * math.cos(dd),
                          math.cos(math.radians(fdip))])
            p = np.array([r.uniform(0, self.size), r.uniform(0, self.size), self.z0])
            throw = r.uniform(5, 40)
            self.faults.append(FaultPlane(f"{self.area['area_id']}-F{i + 1}", p, n, throw, 3 + 0.35 * throw))

    def ground(self, x, y):
        return self.z0 + 25 * np.sin(np.asarray(x) / 420.0) * np.cos(np.asarray(y) / 530.0)

    def contacts(self, x, y) -> np.ndarray:
        """(U-1, N) base elevations of the units at plan points (before faulting)."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        dd = math.radians(self.dipdir)
        along = (x - self.size / 2) * math.sin(dd) + (y - self.size / 2) * math.cos(dd)
        zref = self.z0 + 40.0 - along * math.tan(math.radians(self.dip))
        z = zref.copy()
        out = []
        for k in range(len(self.column) - 1):
            ph = self.phase[k]
            t = self.tmean[k] + self.tamp[k] * 0.5 * (np.sin(x / 140.0 + ph[0]) * np.cos(y / 120.0 + ph[1])
                                                     + np.sin((x + y) / 260.0 + ph[2]))
            z = z - np.maximum(t, 2.0)
            out.append(z.copy())
        return np.array(out)

    def unit_at(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        zeff = p[:, 2].copy()
        for f in self.faults:                         # hanging wall moved down by the throw
            zeff = zeff + np.where(f.signed(p) > 0, f.throw, 0.0)
        c = self.contacts(p[:, 0], p[:, 1])
        return (c > zeff[None, :]).sum(axis=0)

    def fault_zone(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(inside any damage zone, signed distance to the nearest fault) at points."""
        p = np.atleast_2d(p)
        if not self.faults:
            return np.zeros(len(p), bool), np.full(len(p), 1e9)
        d = np.stack([f.signed(p) for f in self.faults])
        hw = np.array([f.half_width for f in self.faults])[:, None]
        k = np.argmin(np.abs(d), axis=0)
        return (np.abs(d) <= hw).any(axis=0), d[k, np.arange(len(p))]


# --- drilling one hole ----------------------------------------------------------------------------
def _dirvec(az, dip):
    a, d = np.radians(az), np.radians(dip)
    return np.array([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), np.sin(d)])


def _azdip(v):
    e, n, u = v
    return (math.degrees(math.atan2(e, n)) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, u)))))


def simulate_trajectory(rng, collar, az, dip, depth, normal_fn, walk=0.035, noise=0.05):
    """1 m positions and directions: random walk plus a walk toward the bedding normal at low incidence."""
    n = int(math.ceil(depth)) + 1
    xyz = np.empty((n, 3))
    dirs = np.empty((n, 3))
    v = _dirvec(az, dip)
    p = np.asarray(collar, float).copy()
    for i in range(n):
        xyz[i], dirs[i] = p, v
        if i % 5 == 0:
            nrm = normal_fn(p)
            if nrm @ v > 0:
                nrm = -nrm
            inc = math.degrees(math.asin(min(1.0, abs(float(nrm @ v)))))
            if inc < 60:
                t = nrm - (nrm @ v) * v
                tn = np.linalg.norm(t)
                if tn > 1e-9:
                    rate = math.radians(walk * 5 * math.cos(math.radians(inc)))
                    v = v * math.cos(rate) + t / tn * math.sin(rate)
            v = v + rng.normal(0, math.radians(noise) * math.sqrt(5), 3)
            v[2] = min(v[2], -0.05)                    # stays a down hole
            v = v / np.linalg.norm(v)
        p = p + v
    return xyz, dirs


@dataclass
class HoleRecords:
    collar: dict
    survey: list = field(default_factory=list)
    geology: list = field(default_factory=list)
    hole_size: list = field(default_factory=list)
    bit: list = field(default_factory=list)
    run: list = field(default_factory=list)
    occurrence: list = field(default_factory=list)
    failure: list = field(default_factory=list)


def drill_hole(rng: np.random.Generator, hole_id: str, area: dict, rig: dict, collar, az, dip, planned,
               units_md: list[tuple[float, float, str]], incidence: np.ndarray, fault_md: np.ndarray,
               dirs: np.ndarray, cards: dict, karst_factor: float, start_date: str, crew: str,
               stop_at: float | None = None, newer: bool | None = None) -> HoleRecords:
    """Drill one hole along a known path: ``units_md`` (from, to, formation) along the hole, per-metre
    incidence angle and fault-zone flag, directions every metre. Problems, measures, failure, runs.
    ``stop_at``: drilling is still going on at that depth (an ongoing hole, no failure drawn beyond)."""
    n = int(math.ceil(planned))
    md = np.arange(n) + 0.5
    if len(dirs) < n + 1:                              # an ongoing hole: directions only down to the bit
        dirs = np.vstack([dirs, np.repeat(dirs[-1:], n + 1 - len(dirs), axis=0)])
    form = np.empty(n, dtype=object)
    for f, t, name in units_md:
        form[(md >= f) & (md < t)] = name
    form[form == None] = units_md[-1][2]  # noqa: E711
    card = [cards.get(f) or {} for f in form]
    carb = np.array([float(c.get("carbonate_pct") or 0) for c in card])
    karst = np.array([codes.KARST_CODE.get(c.get("karst") or "none", 0) for c in card], float)
    clay = np.array([codes.CLAY_CODE.get(c.get("swelling_clay") or "none", 0) for c in card], float)
    hard = np.array([float(c.get("hardness") or 3) for c in card])
    abr = np.array([float(c.get("abrasiveness") or 3) for c in card])
    alt = np.array([1.0 if c.get("bedding") == "alternation" else 0.0 for c in card])
    clastic = np.array([codes.ROCK_CLASSES.get(c.get("main_rock") or "sandstone")[4] for c in card])
    # rock beds inside each formation (main / secondary rock)
    rock = np.empty(n, dtype=object)
    i = 0
    while i < n:
        c = card[i]
        L = max(1, int(rng.exponential(12)))
        sec = c.get("second_rock")
        use_sec = sec and rng.random() < float(c.get("second_pct") or 0) / 100.0
        rk = sec if use_sec else c.get("main_rock") or "sandstone"
        j = min(n, i + L)
        same = j
        for k in range(i, j):
            if form[k] != form[i]:
                same = k
                break
        rock[i:same] = rk
        i = same
    # mechanical properties of the beds (rockmech.py): strength, abrasivity, foliation, swelling, fracturing
    from . import rockmech

    mp = [rockmech.props(r) for r in rock]
    ucs = np.array([q.ucs_typ for q in mp]) * np.exp(rng.normal(0, 0.12, n))
    cai = np.array([q.cai_typ for q in mp]) * np.exp(rng.normal(0, 0.08, n))
    aniso = np.array([q.anisotropy for q in mp], float)
    swell = np.maximum(clay, np.array([q.swelling for q in mp], float))
    frac = np.array([q.fractures for q in mp], float)
    hard = np.array([rockmech.strength_class(u) for u in ucs], float)
    abr = np.array([rockmech.abrasivity_class(c) for c in cai], float)
    contrast = np.zeros(n)                             # strong / weak bed boundaries (deflect the bit)
    for b in np.nonzero(rock[1:] != rock[:-1])[0] + 1:
        r = abs(math.log10(max(ucs[b], 0.1) / max(ucs[b - 1], 0.1)))
        if r > 0.4:
            contrast[max(0, b - 4):min(n, b + 4)] = np.maximum(contrast[max(0, b - 4):min(n, b + 4)], r)
    # distance below the top of the current unit, unit above clastic, z
    top = np.zeros(n)
    above_clastic = np.zeros(n)
    for f, t, name in units_md:
        m = (md >= f) & (md < t)
        top[m] = md[m] - f
    for k, (f, t, name) in enumerate(units_md):
        if k > 0:
            prev = cards.get(units_md[k - 1][2]) or {}
            m = (md >= f) & (md < t)
            above_clastic[m] = 1.0 if codes.ROCK_CLASSES.get(prev.get("main_rock") or "sandstone")[4] else 0.0
    z = collar[2] + np.cumsum(np.r_[0.0, dirs[:n - 1, 2]])[:n]
    wt = float(area.get("water_table") or -1e9)
    in_fault = fault_md[:n] if len(fault_md) >= n else np.r_[fault_md, np.zeros(n - len(fault_md))]
    inc = incidence[:n] if len(incidence) >= n else np.r_[incidence, np.full(n - len(incidence), 60.0)]
    # --- cavities (truth) ---
    lam = (0.03 * karst_factor * (karst / 2.0) ** 1.3 * (carb / 100.0)
           * (0.35 + 2.4 * above_clastic * np.exp(-top / 25.0) + 0.6 * np.exp(-top / 60.0))
           * np.where(z > wt, 1.0, np.exp(-(wt - z) / 70.0)) * (1.0 + 1.5 * in_fault))
    cav = np.zeros(n, bool)
    cavities = []
    i = 0
    while i < n:
        if rng.random() < lam[i]:
            L = 0.4 + rng.exponential(1.6)
            j = min(n, int(math.ceil(md[i] + L)))
            cav[i:j] = True
            cavities.append((float(md[i] - 0.5 + rng.uniform(0, 0.5)), float(min(planned, md[i] - 0.5 + L))))
            i = j + 2
        else:
            i += 1
    # --- the drilling programme ---
    newer = area["area_id"] in NEW_FIELDS_AREAS if newer is None else newer
    diligent = rng.random() < 0.25                    # crews that case off before the carbonate
    fluid = str(rng.choice(["water", "polymer", "bentonite", "kcl_polymer"], p=[0.35, 0.35, 0.2, 0.1]))
    size_now = "PQ"
    size_changes = [(0.0, "PQ")]
    pq_to = float(rng.uniform(8, 30))
    size_changes.append((pq_to, "HQ"))
    hq_limit = float(rig["depth_HQ"]) * rng.uniform(0.45, 0.8)
    if hq_limit < planned:
        size_changes.append((hq_limit, "NQ"))
    if diligent:
        first_carb = next((f for f, t, nm in units_md if float((cards.get(nm) or {}).get("carbonate_pct") or 0) >= 50
                           and (cards.get(nm) or {}).get("karst") == "cavernous"), None)
        if first_carb is not None and first_carb > pq_to + 5:
            size_changes.append((first_carb - 3.0, "NQ"))
    size_changes = sorted(set(size_changes))
    sizes = []                                         # keep only reductions
    for d, s in size_changes:
        if not sizes or codes.SIZE_ORDER.index(s) > codes.SIZE_ORDER.index(sizes[-1][1]):
            sizes.append((d, s))
    bit_at = np.empty(n, dtype=object)
    for f, t, _nm in units_md:                         # one bit per formation, chosen for its strength
        m = (md >= f) & (md < t)
        if not m.any():
            continue
        u = float(np.median(ucs[m]))
        bit_at[m] = ("impregnated" if u >= 100 else
                     ("impregnated" if rng.random() < 0.6 else "surface_set") if u >= 50 else
                     ("pdc" if rng.random() < 0.3 else "surface_set"))
    bit_at[bit_at == None] = "impregnated"  # noqa: E711
    wrong_bit = rng.random() < 0.18                    # a crew using the wrong bit in strong rock
    if wrong_bit:
        bit_at[ucs >= 150] = str(rng.choice(["surface_set", "pdc"]))
    rpm_base = {"impregnated": 900.0, "surface_set": 620.0, "pdc": 420.0, "carbide": 250.0}
    flow_by = {"PQ": 75.0, "HQ": 52.0, "NQ": 38.0, "BQ": 30.0, "AQ": 24.0}
    hi_flow = rng.random() < 0.3
    # --- per-metre hazards of problems (truth) ---
    size_arr = np.array([next(s for d, s in reversed(sizes) if md[k] >= d) for k in range(n)], dtype=object)
    last_casing = np.array([max(d for d, s in sizes if md[k] >= d) for k in range(n)])
    open_hole = md - last_casing
    stabil = 0.45 if fluid in ("bentonite", "kcl_polymer") else 1.0
    h_inst = (in_fault * 0.012 * (1 + 0.6 * swell) + (swell >= 2) * 0.0009 + (ucs < 25) * 0.0008
              + (frac >= 2) * 0.00012 + alt * 0.00015) * (1 + open_hole / 250.0) * stabil
    dogleg = np.zeros(n)
    if n > 30:
        cosb = np.clip(np.einsum("ij,ij->i", dirs[30:n], dirs[:n - 30]), -1, 1)
        dogleg[30:] = np.degrees(np.arccos(cosb))
    # deviation: low angle to bedding / foliation, foliated rock, strong-weak bed boundaries, faults
    h_dv = (0.000030 * np.where(inc < 30, 6.0, 1.0) * (1 + 0.9 * aniso * (inc < 45)) * (1 + 1.5 * contrast)
            * (1 + 0.8 * in_fault) * (1 + dogleg / 3.0))
    # bit: glazing in strong, dense, less abrasive rock (siliceous dolomite, hornfels, ironstone); wear in
    # abrasive rock (CAI >= 3.5: quartzite, chert, granite, gneiss); string failure in doglegs / foliated hard rock
    mism = ((ucs >= 150) & (bit_at[:n] != "impregnated")).astype(float)
    h_glaze = 0.00020 * (ucs >= 150) * (cai < 4.2) * (1 + 5 * mism)
    h_wear = 0.00009 * (cai >= 3.5) * (1 + 3 * mism) + 0.00004 * (cai >= 3.0) * (frac >= 2)
    h_rf = 0.00004 * (dogleg > 3) + 0.00002 * aniso * (ucs > 100)
    h_bit = h_glaze + h_wear + h_rf
    cav_start = np.zeros(n, bool)
    for f, t in cavities:
        k = int(min(n - 1, max(0, math.floor(f))))
        cav_start[k] = True
    # --- walk down the hole: events in order ---
    rec = HoleRecords({})
    failure = None
    events = []                                       # (depth, code)
    applied_casing = []
    end = float(planned) if stop_at is None else float(stop_at)
    for k in range(min(n, int(math.ceil(end)))):
        d = float(md[k])
        evt = None
        if cav_start[k]:
            p = 0.55 * (1.3 if hi_flow else 1.0) * (0.35 if size_arr[k] != "HQ" and diligent else 1.0)
            if rng.random() < p:
                evt = "LC" if rng.random() < 0.6 else "CV"
        if evt is None and rng.random() < h_inst[k]:
            evt = str(rng.choice(["CL", "SP", "CB"], p=[0.5, 0.3, 0.2]))
        if evt is None and rng.random() < h_dv[k]:
            evt = "DV"
        if evt is None and rng.random() < h_bit[k]:
            w = np.array([h_glaze[k], h_wear[k], h_rf[k]]) + 1e-12
            evt = str(rng.choice(["BG", "BW", "RF"], p=w / w.sum()))
        if evt is None and in_fault[k] and z[k] < wt and rng.random() < 0.002:
            evt = "WI"
        if evt is None:
            continue
        # respond with measures until solved (or give up -> failure)
        tried, solved = [], False
        options = list(RESPONSE[evt])
        rng.shuffle(options)
        n_try = int(rng.integers(1, 4))
        severity_base = {"LC": 3, "CV": 3, "CL": 3, "SP": 4, "CB": 2, "DV": 3, "BG": 2, "BW": 2, "RF": 4, "WI": 3}[evt]
        hard_case = 0.75 if evt in ("CV", "LC") and cav[k] and (cavities and _cav_len(cavities, d) > 2.5) else 1.0
        for mcode in options[:n_try]:
            tried.append(mcode)
            if rng.random() < MEASURE_SUCCESS.get(mcode, 0.5) * hard_case:
                solved = True
                break
        dz = d if not cav[k] else next((f for f, t in cavities if f <= d + 1 and t >= d - 1), d)
        occ_from, occ_to = (dz, dz + 1.0)
        if evt in ("CV", "LC") and cav[k]:
            c = next(((f, t) for f, t in cavities if f <= d + 1 and t >= d - 1), (dz, dz + 1))
            occ_from, occ_to = c
        events.append((d, evt))
        rec.occurrence.append({"depth_from": round(occ_from, 2), "depth_to": round(max(occ_to, occ_from + 0.2), 2),
                               "type": OCC_OF[evt], "measures": ";".join(tried), "solved": "Y" if solved else "N",
                               "notes": f"{evt} event"})
        if solved:
            if evt in ("LC", "CL", "CV") and "M-LC-07" not in tried and rng.random() < 0.5:
                applied_casing.append(d + 2.0)       # casing set after the event (a size reduction below)
            continue
        failure = {"fail_depth": round(d, 1), "fail_code": evt,
                   "severity": int(min(5, severity_base + rng.integers(0, 2))), "measures": ";".join(tried)}
        break
    # cavity / fault occurrences without problems (recorded as met)
    for f, t in cavities:
        if f >= (failure["fail_depth"] if failure else end):
            continue
        if not any(o["type"] == "cavity" and abs(o["depth_from"] - f) < 1.5 for o in rec.occurrence):
            rec.occurrence.append({"depth_from": round(f, 2), "depth_to": round(max(t, f + 0.2), 2), "type": "cavity",
                                   "measures": None, "solved": "Y", "notes": "met while drilling"})
    stop = failure["fail_depth"] if failure else end
    fz = _runs_of(in_fault[:int(math.ceil(stop))] > 0)
    for a0, b0 in fz:
        rec.occurrence.append({"depth_from": float(a0), "depth_to": float(b0), "type": "fault", "measures": None,
                               "solved": "Y", "notes": "fault zone"})
    for d0 in applied_casing:
        if d0 < stop:
            cur = sizes[-1][1]
            nxt = codes.SIZE_ORDER[min(len(codes.SIZE_ORDER) - 1, codes.SIZE_ORDER.index(cur) + 1)]
            later = [s for s in sizes if s[0] > d0]
            if not later and nxt != cur:
                sizes.append((d0, nxt))
    sizes = sorted(sizes)
    # --- geology log (rock beds merged), hole sizes, bits ---
    nstop = max(1, int(math.ceil(stop)))
    kk = 0
    while kk < nstop:
        j = kk
        while j < nstop and rock[j] == rock[kk] and form[j] == form[kk]:
            j += 1
        mid = (kk + j) / 2
        ii = min(nstop - 1, int(mid))
        bed = float(np.clip(inc[ii] + rng.normal(0, 4), 0, 90))
        broken = "Y" if in_fault[kk:j].any() or (alt[ii] and rng.random() < 0.05) else "N"
        rec.geology.append({"depth_from": float(kk), "depth_to": float(min(j, stop)), "rock": ROCK_CODE.get(rock[kk],
                            "SS"), "formation": form[kk], "weathering": int(1 if kk < 15 else 0), "broken": broken,
                            "bedding_angle": round(bed, 1)})
        kk = j
    for idx, (d0, s) in enumerate(sizes):
        if d0 >= stop:
            break
        to = sizes[idx + 1][0] if idx + 1 < len(sizes) and sizes[idx + 1][0] < stop else stop
        rec.hole_size.append({"depth_from": round(d0, 1), "depth_to": round(to, 1), "size": s,
                              "diameter": codes.HOLE_SIZES[s]})
    kk = 0
    while kk < nstop:
        j = kk
        while j < nstop and bit_at[j] == bit_at[kk]:
            j += 1
        rec.bit.append({"depth_from": float(kk), "depth_to": float(min(j, stop)), "bit_type": bit_at[kk]})
        kk = j
    # --- runs with precursors ---
    d0 = 0.0
    fail_d, fail_c = (failure["fail_depth"], failure["fail_code"]) if failure else (None, None)
    ev_d = [(d, c) for d, c in events]
    while d0 < stop - 1e-6:
        L = float(rng.choice([1.5, 3.0, 3.0, 3.0, 6.0])) if d0 > 6 else 3.0
        d1 = min(stop, d0 + L)
        k = min(n - 1, int((d0 + d1) / 2))
        size = next(s for dd, s in reversed(sizes) if (d0 + d1) / 2 >= dd)
        b = bit_at[k]
        rpm = rpm_base.get(b, 700) * float(np.clip(1 - 0.0012 * (ucs[k] - 100), 0.7, 1.15)) * rng.normal(1, 0.07)
        rpm = min(rpm, float(rig["max_rpm"]))
        rop = (6.0 * (100.0 / max(ucs[k], 5.0)) ** 0.5 * (rpm / 750.0) * (1 - 0.04 * cai[k])
               * rng.normal(1, 0.12) * (1.15 if size == "NQ" else 1.0))
        flow = flow_by[size] * (1.25 if hi_flow else 1.0) * rng.normal(1, 0.05)
        press = (0.8 + 0.0035 * d1 + 0.012 * flow) * rng.normal(1, 0.06)
        ret = float(np.clip(rng.normal(95, 4), 60, 110))
        rod = 0.0
        torque = (300 + 1.6 * d1 + 1.5 * ucs[k] + 40 * aniso[k]) * rng.normal(1, 0.06)
        rem = None
        for dd, c in ev_d + ([(fail_d, fail_c)] if fail_d is not None else []):
            if dd is None:
                continue
            ahead = dd - d1                          # metres from the end of this run to the event
            if -L <= ahead <= 6.0:                   # inside the precursor window (spec 5.1: L = 3-10 m)
                w = 1.0 - max(ahead, 0) / 6.0
                if c == "LC":
                    press *= 1 - 0.45 * w
                    ret = max(0.0, ret - 60 * w)
                    rem = "partial loss" if w > 0.6 else rem
                elif c == "CV":
                    if ahead <= 1.5:
                        rop *= 3.5
                        rod = round(float(rng.uniform(0.3, 2.5)), 2)
                        ret = 0.0
                        rem = "rod drop"
                    else:
                        press *= 1 - 0.2 * w
                elif c in ("CB", "CL"):
                    press += 0.7 * w
                    rop *= 1 - 0.35 * w
                elif c == "SP":
                    torque *= 1 + 0.6 * w
                    press += 0.3 * w
                elif c in ("BG", "BW"):
                    rop *= 1 - 0.55 * w
                elif c == "WI":
                    ret = min(120.0, ret + 20 * w)
        if not newer:
            ret, rod, torque = None, None, None
        rec.run.append({"depth_from": round(d0, 2), "depth_to": round(d1, 2), "rpm": round(rpm), "rop": round(rop, 2),
                        "flow_in": round(flow, 1), "pump_press": round(press, 2),
                        "return_pct": None if ret is None else round(ret), "rod_drop": rod,
                        "wob": None, "torque": None if torque is None else round(torque),
                        "fluid_type": fluid, "recovery": round(float(np.clip(rng.normal(96, 3), 50, 100))),
                        "remarks": rem})
        d0 = d1
    # --- survey every 30 m ---
    for d in list(np.arange(0, stop, 30.0)) + [stop]:
        k = min(len(dirs) - 1, int(round(d)))
        a, dp = _azdip(dirs[k])
        rec.survey.append({"depth": round(float(d), 1), "azimuth": round(a, 2), "dip": round(dp, 2)})
    result = "failed" if failure else ("in_progress" if stop_at is not None else "success")
    rec.collar = {"hole_id": hole_id, "area_id": area["area_id"], "x": round(float(collar[0]), 2),
                  "y": round(float(collar[1]), 2), "z": round(float(collar[2]), 2), "rig_model": rig["rig_model"],
                  "start_date": start_date, "end_date": None if stop_at is not None else start_date,
                  "plan_azimuth": round(float(az), 1), "plan_dip": round(float(dip), 1),
                  "planned_depth": float(planned), "final_depth": round(float(stop), 1), "result": result,
                  "casing_available": "HW;NW;BW", "fluids_available": "water;polymer;bentonite;kcl_polymer",
                  "crew": crew, "notes": None}
    if failure:
        rec.failure.append(failure)
    return rec


def _cav_len(cavities, d):
    for f, t in cavities:
        if f <= d + 1 and t >= d - 1:
            return t - f
    return 0.0


def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _units_along(unit_idx: np.ndarray, column: list[str]) -> list[tuple[float, float, str]]:
    out = []
    n = len(unit_idx)
    i = 0
    while i < n:
        j = i
        while j < n and unit_idx[j] == unit_idx[i]:
            j += 1
        out.append((float(i), float(j), column[min(len(column) - 1, max(0, int(unit_idx[i])))]))
        i = j
    return out


# --- the history ----------------------------------------------------------------------------------
def make_areas(rng) -> pd.DataFrame:
    rows = []
    for i, aid in enumerate(COLUMNS):
        cexp = "Y" if rng.random() < 0.5 else "N"
        rows.append({"area_id": aid, "name": f"Past area {i + 1}", "dip_min": float(rng.choice([0, 5, 15, 30])),
                     "dip_max": float(rng.choice([20, 35, 55, 80])), "fold_tightness": str(int(rng.integers(1, 4))),
                     "faults_per_km": round(float(rng.uniform(0.5, 3.0)), 1), "carbonate_exposed": cexp,
                     "unconformity": "Y" if rng.random() < 0.5 else "N", "sinkholes": "Y" if rng.random() < 0.4 else "N",
                     "carb_above_valley": round(float(rng.uniform(-80, 150))), "water_table": None,
                     "depth_of_interest": float(rng.choice([400, 500, 600])), "notes": None})
        rows[-1]["dip_max"] = max(rows[-1]["dip_max"], rows[-1]["dip_min"] + 10)
    return pd.DataFrame(rows)


def area_karst_factor(a: dict, rng) -> float:
    f = 0.35 + 0.6 * (a["carbonate_exposed"] == "Y") + 0.7 * (a["sinkholes"] == "Y") + 0.45 * (a["unconformity"] == "Y")
    f *= 1.0 + 0.004 * max(0.0, float(a["carb_above_valley"]))
    return float(f * rng.lognormal(0, 0.15))


def generate_history(n_per_area: int = 150, seed: int = 7, progress=None) -> dict[str, pd.DataFrame]:
    """The past holes of all areas: every risk_ table as a DataFrame."""
    rng = np.random.default_rng(seed)
    cards_df = cards_frame(CATALOGUE)
    cards = cards_dict(cards_df)
    areas = make_areas(rng)
    out = {k: [] for k in ("collar", "survey", "geology", "hole_size", "bit", "run", "occurrence", "failure")}
    hole_no = 0
    total = len(areas) * n_per_area
    for ai, a in enumerate(areas.to_dict("records")):
        world = AreaWorld(a, COLUMNS[a["area_id"]], cards, np.random.default_rng(seed * 100 + ai))
        kf = area_karst_factor(a, rng)
        # water table: some 40-120 m below the ground (known in about half of the areas)
        wt = world.z0 - float(rng.uniform(40, 120))
        a["water_table"] = round(wt) if rng.random() < 0.6 else None
        areas.loc[ai, "water_table"] = a["water_table"]
        a_truth = dict(a, water_table=wt)
        year = 2012 + ai
        for h in range(n_per_area):
            hole_no += 1
            x, y = rng.uniform(150, world.size - 150, 2)
            zc = float(world.ground(x, y))
            az = float(rng.uniform(0, 360))
            dip = float(rng.choice([-90.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0]))
            planned = float(rng.choice([250, 300, 350, 400, 450, 500, 550, 600, 650]))
            rig = RIGS[int(rng.integers(0, len(RIGS)))]
            if planned > rig["depth_NQ"]:
                planned = float(rig["depth_NQ"]) * 0.9
            normal_fn = lambda p, w=world: w.normal  # noqa: E731
            xyz, dirs = simulate_trajectory(rng, (x, y, zc), az, dip, planned, normal_fn)
            uidx = world.unit_at(xyz)
            units = _units_along(uidx[:int(math.ceil(planned))], world.column)
            nrm = world.normal
            inc = np.degrees(np.arcsin(np.clip(np.abs(dirs @ nrm), 0, 1)))
            infz, _ = world.fault_zone(xyz)
            date = f"{year + h * 3 // n_per_area}-{1 + (h * 7) % 12:02d}-{1 + (h * 13) % 27:02d}"
            rec = drill_hole(rng, f"{a['area_id']}-{h + 1:03d}", a_truth, rig, (x, y, zc), az, dip, planned, units,
                             inc, infz.astype(float), dirs, cards, kf, date, f"crew {1 + h % 5}")
            rec.collar["area_id"] = a["area_id"]
            _collect(out, rec)
            if progress is not None and hole_no % 25 == 0:
                progress(hole_no, total)
    frames = {k: pd.DataFrame(v) for k, v in out.items()}
    frames["area"] = areas
    frames["formation"] = cards_df
    frames["rig"] = pd.DataFrame(RIGS)
    return frames


def _collect(out: dict, rec: HoleRecords) -> None:
    hid = rec.collar["hole_id"]
    out["collar"].append(rec.collar)
    for name in ("survey", "geology", "hole_size", "bit", "run", "occurrence", "failure"):
        for r in getattr(rec, name):
            out[name].append({"hole_id": hid, **r})


# --- the demo workspace (SB3) -----------------------------------------------------------------------
SB3_AREA = {"area_id": "SB3", "name": "Demo workspace (SB3 mixed)", "dip_min": 5.0, "dip_max": 30.0,
            "fold_tightness": "2", "faults_per_km": 4.0, "carbonate_exposed": "N", "unconformity": "Y",
            "sinkholes": "Y", "carb_above_valley": 60.0, "water_table": 175.0, "depth_of_interest": 350.0,
            "notes": "Dolomite U3 under black shale U2 (clastic over carbonate); post-ore fault F4"}


def workspace_records(model, holes: pd.DataFrame, surveys: pd.DataFrame, geology: pd.DataFrame,
                      seed: int = 11) -> dict[str, list]:
    """Records of the finished holes of the demo workspace: their real traces and logs, with the
    problems drawn by the same process as the history (karst factor of a strongly karstified area)."""
    from ..io.drillholes import DrillholeDB

    rng = np.random.default_rng(seed)
    cards = cards_dict(cards_frame(CATALOGUE + WORKSPACE_CARDS))
    out = {k: [] for k in ("collar", "survey", "geology", "hole_size", "bit", "run", "occurrence", "failure")}
    db = DrillholeDB(holes.rename(columns={"max_depth": "max_depth"}), surveys, {})
    area = dict(SB3_AREA)
    kf = area_karst_factor(area, rng) * 1.2
    rig = RIGS[2]
    for i, row in enumerate(holes.itertuples(index=False)):
        tr = db.trace(row.hole_id)
        planned = float(row.max_depth)
        md = np.arange(0, int(math.ceil(planned)) + 1, 1.0)
        xyz = tr.positions_at(md)
        dirs = tr.direction_at(md)
        g = geology[geology["hole_id"] == row.hole_id].sort_values("from")
        units = []
        for f, t, nm in zip(g["from"].astype(float), g["to"].astype(float), g["lith"].astype(str)):
            if units and units[-1][2] == nm:
                units[-1] = (units[-1][0], t, nm)
            else:
                units.append((f, t, nm))
        if not units:
            continue
        idx = model.unit_index(xyz)
        nrm = model.bedding_normal(xyz, idx)
        inc = np.degrees(np.arcsin(np.clip(np.abs(np.einsum("ij,ij->i", dirs, nrm)), 0, 1)))
        hw = float(model.config.get("fault_zone", 12.0))
        fd, _ = model.fault_distances(xyz)
        infz = (fd <= hw).astype(float)
        date = f"2025-{1 + i % 12:02d}-{1 + (i * 7) % 27:02d}"
        rec = drill_hole(rng, row.hole_id, area, rig, xyz[0], *_azdip(dirs[0]), planned, units, inc, infz, dirs,
                         cards, kf, date, f"crew {1 + i % 3}")
        _collect(out, rec)
    return out


def ongoing_hole(model, hole_id: str, collar_xy, az: float, dip: float, planned: float, current: float,
                 seed: int, contact_offset: float = 0.0, pressure_drop: bool = False, start: str = "2026-11-10",
                 walk: float = 0.035) -> HoleRecords:
    """A hole of the demo workspace being drilled now (``current`` = bit depth; 0 = planned, not started).
    The geology met so far is the model's, with contacts ``contact_offset`` m deeper than the model
    (the tie of spec 5.6.4); ``pressure_drop``: the last runs show a pump pressure drop (rule R1)."""
    rng = np.random.default_rng(seed)
    cards = cards_dict(cards_frame(CATALOGUE + WORKSPACE_CARDS))
    x, y = collar_xy
    z = float(model.topography.sample([x], [y])[0]) if model.topography is not None else 316.0
    if model.topography is None:
        z = _ground_guess(model, x, y)
    rig = RIGS[2]
    if current <= 0:
        rec = HoleRecords({"hole_id": hole_id, "area_id": "SB3", "x": x, "y": y, "z": round(z, 2), "rig_model":
                           rig["rig_model"], "start_date": start, "end_date": None, "plan_azimuth": az, "plan_dip": dip,
                           "planned_depth": planned, "final_depth": 0.0, "result": "planned",
                           "casing_available": "HW;NW;BW", "fluids_available": "water;polymer;bentonite",
                           "crew": "crew 2", "notes": "Planned hole of the demo workspace"})
        return rec
    normal_fn = lambda p: model.bedding_normal(np.atleast_2d(p))[0]  # noqa: E731
    xyz, dirs = simulate_trajectory(rng, (x, y, z), az, dip, current + 1, normal_fn, walk=walk, noise=0.06)
    idx = model.unit_index(xyz, dz=-contact_offset)
    units = _units_along(idx, model.all_units)
    nrm = model.bedding_normal(xyz, idx)
    inc = np.degrees(np.arcsin(np.clip(np.abs(np.einsum("ij,ij->i", dirs, nrm)), 0, 1)))
    fd, _ = model.fault_distances(xyz)
    infz = (fd <= float(model.config.get("fault_zone", 12.0))).astype(float)
    area = dict(SB3_AREA)
    rec = drill_hole(rng, hole_id, area, rig, (x, y, z), az, dip, planned, units, inc, infz, dirs, cards,
                     0.0, start, "crew 2", stop_at=current, newer=True)
    rec.occurrence = [o for o in rec.occurrence if o["type"] == "fault"]
    rec.failure = []
    rec.collar.update({"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "plan_azimuth": az, "plan_dip": dip,
                       "result": "in_progress", "notes": "Hole being drilled (demo)"})
    if pressure_drop and len(rec.run) >= 3:
        for k, w in ((-2, 0.85), (-1, 0.62)):
            r = rec.run[k]
            r["pump_press"] = round(r["pump_press"] * w, 2)
            r["return_pct"] = 78 if k == -2 else 64
            r["remarks"] = "returns dropping" if k == -1 else None
    return rec


def _ground_guess(model, x, y) -> float:
    """Ground elevation where the model has no topography: just above the highest contact, from the
    demo's collars (about 316 m)."""
    top = float(model.horizons[0].sample([x], [y])[0]) if model.horizons else 300.0
    return max(top + 120.0, 312.0)
