"""Fixed alarm rules R1 - R8 (design spec 5.7): clear physical signs in the runs and surveys, which
override the model score. R6 - R8 need the new fields of spec 3.3 (return %, rod drop, torque).
No Qt here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import codes

LIMITS = {"press_drop": 0.25, "rop_jump": 3.0, "press_rise": 0.5, "rop_fall": 0.40, "dogleg": 3.0,
          "off_plan": 10.0, "off_plan_pct": 0.03, "return_low": 70.0, "return_fall": 30.0, "rod_drop": 0.2,
          "torque": 1.3, "same": 0.10}


@dataclass
class RuleHit:
    rule: str
    depth_from: float
    depth_to: float
    family: str
    code: str
    level: str
    reason: str                    # reason code (R-D..)
    values: dict = field(default_factory=dict)
    current: bool = False          # on the last run (a live warning)

    @property
    def condition(self) -> str:
        return codes.RULES[self.rule][0]


def _same(a, b, tol=LIMITS["same"]) -> bool:
    return a is not None and b is not None and not (math.isnan(a) or math.isnan(b)) and b > 0 and abs(a - b) <= tol * b


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def evaluate(runs: pd.DataFrame | None, surveys: pd.DataFrame | None = None, plan: dict | None = None,
             rock_class_at=None, formation_at=None) -> list[RuleHit]:
    """All rule hits along the drilled part (the ones on the last run are marked ``current``).
    ``rock_class_at`` / ``formation_at``: depth -> rock class / formation (R2, R4)."""
    hits: list[RuleHit] = []
    if runs is not None and len(runs):
        r = runs.sort_values("depth_from").reset_index(drop=True)
        n = len(r)
        col = lambda c: r[c].map(_f).to_numpy() if c in r else np.full(n, np.nan)  # noqa: E731
        rpm, rop, flow, press = col("rpm"), col("rop"), col("flow_in"), col("pump_press")
        ret, rod, torque = col("return_pct"), col("rod_drop"), col("torque")
        f, t = r["depth_from"].to_numpy(float), r["depth_to"].to_numpy(float)

        def hit(rule, i, **values):
            fam, code, level, reason = codes.RULES[rule][1:]
            hits.append(RuleHit(rule, float(f[i]), float(t[i]), fam, code, level, reason,
                                {k: (round(v, 2) if isinstance(v, float) else v) for k, v in values.items()},
                                current=(i == n - 1)))

        form = [formation_at((a + b) / 2) if formation_at else None for a, b in zip(f, t)]
        cls = [rock_class_at((a + b) / 2) if rock_class_at else "" for a, b in zip(f, t)]
        for i in range(1, n):
            # R1 pressure drop at the same flow
            if _same(flow[i], flow[i - 1]) and press[i - 1] > 0 and press[i] < press[i - 1] * (1 - LIMITS["press_drop"]):
                hit("R1", i, x=round(100 * (1 - press[i] / press[i - 1])))
            # R2 ROP jump vs the median of the last 5 runs in the same formation
            prev = [rop[j] for j in range(max(0, i - 5), i) if form[j] == form[i] and np.isfinite(rop[j])]
            if len(prev) >= 2 and rop[i] > LIMITS["rop_jump"] * float(np.median(prev)):
                hit("R2", i, x=round(rop[i] / float(np.median(prev)), 1), formation=form[i] or cls[i])
            # R3 pressure rise at the same flow while ROP falls
            if _same(flow[i], flow[i - 1]) and press[i] - press[i - 1] > LIMITS["press_rise"] and rop[i] < rop[i - 1]:
                hit("R3", i, x=round(press[i] - press[i - 1], 2))
            # R4 ROP falls > 40 % over 3 runs at the same RPM in quartzite / dolomite
            if i >= 3 and cls[i] in ("quartzite", "dolomite", "chert") and _same(rpm[i], rpm[i - 3]) \
                    and rop[i - 3] > 0 and rop[i] < rop[i - 3] * (1 - LIMITS["rop_fall"]):
                hit("R4", i, x=round(100 * (1 - rop[i] / rop[i - 3])), rock_class=cls[i])
            # R6 / R7 returns and rod drop
            if np.isfinite(ret[i]) and (ret[i] < LIMITS["return_low"] or
                                        (np.isfinite(ret[i - 1]) and ret[i - 1] - ret[i] > LIMITS["return_fall"])):
                if ret[i] > 0:
                    hit("R6", i, x=round(ret[i]))
            if (np.isfinite(ret[i]) and ret[i] == 0) or (np.isfinite(rod[i]) and rod[i] >= LIMITS["rod_drop"]):
                hit("R7", i, x=rod[i] if np.isfinite(rod[i]) and rod[i] > 0 else 0.0)
            # R8 torque
            prev_t = [torque[j] for j in range(max(0, i - 5), i) if np.isfinite(torque[j])]
            if len(prev_t) >= 3 and torque[i] > LIMITS["torque"] * float(np.median(prev_t)):
                hit("R8", i, x=round(torque[i] / float(np.median(prev_t)), 2))
    hits += survey_rules(surveys, plan)
    return hits


def survey_rules(surveys: pd.DataFrame | None, plan: dict | None) -> list[RuleHit]:
    """R5: dogleg > 3° per 30 m between stations, or the last station off the planned line."""
    out = []
    if surveys is None or len(surveys) < 2:
        return out
    s = surveys.sort_values("depth").drop_duplicates("depth").reset_index(drop=True)
    d = s["depth"].to_numpy(float)
    az, dip = np.radians(s["azimuth"].to_numpy(float)), np.radians(s["dip"].to_numpy(float))
    v = np.column_stack([np.cos(dip) * np.sin(az), np.cos(dip) * np.cos(az), np.sin(dip)])
    fam, code, level, reason = codes.RULES["R5"][1:]
    for i in range(1, len(s)):
        dl = d[i] - d[i - 1]
        if dl < 3:
            continue
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v[i] @ v[i - 1])))))
        dls = ang * 30.0 / dl
        if dls > LIMITS["dogleg"]:
            out.append(RuleHit("R5", float(d[i - 1]), float(d[i]), fam, code, level, reason,
                               {"x": round(dls, 1), "depth": round(float(d[i]), 1)}, current=(i == len(s) - 1)))
    if plan and plan.get("plan_azimuth") is not None and plan.get("plan_dip") is not None:
        from .trajectory import drilled_trace

        tr = drilled_trace("", (0, 0, 0), s, float(plan["plan_azimuth"]), float(plan["plan_dip"]), float(d[-1]))
        end = tr.positions_at([d[-1]])[0]
        a, p = math.radians(float(plan["plan_azimuth"])), math.radians(float(plan["plan_dip"]))
        design = np.array([math.cos(p) * math.sin(a), math.cos(p) * math.cos(a), math.sin(p)]) * d[-1]
        off = float(np.linalg.norm(end - design))
        tol = max(LIMITS["off_plan"], LIMITS["off_plan_pct"] * d[-1])
        if off > tol:
            out.append(RuleHit("R5", float(d[-1]), float(d[-1]), fam, code, level, reason,
                               {"x": round(off, 1), "depth": round(float(d[-1]), 1), "off_plan": True}, current=True))
    return out
