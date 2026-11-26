"""Reserve decision support: how long the reserve lasts at the current mining rate, a check of
every mined cut, and the best From / To levels for a new cut. Offline; no Qt.

Forecast
    The mining rate is the median of the recent months' production (robust against one unusual
    month); the slow / fast range is the spread of those months (25 % / 75 % quantiles, or the
    lowest / highest month when there are few). Months left = remaining tonnes / rate.

Cut check
    Each saved cut is compared with the reserve and with the other cuts: ore share of its volume,
    new ore it mined, ore it re-mined (mined by earlier cuts), its grade against the reserve
    average, and its thickness against the usual thickness. Every finding says why in words.

Best levels
    For an outline being entered, every From / To window of the chosen thickness (0.1 m steps) is
    scored on the reserve's sample points not mined yet; the best window for the aim (most metal,
    highest grade, most ore) is suggested, with the next best different windows.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field

import numpy as np

from ..i18n import tr
from . import BAD, INFO, OK, WARN

RECENT_MONTHS = 6            # months used for the mining rate
TREND_MONTHS = 3             # last 3 months against the 3 before
GRADE_GAP = 0.15             # recent grade more than 15 % from the remaining grade: worth saying
SNAP = 0.1                   # metres: From / To steps of the suggestion


def _mi(month: str) -> int:
    y, m = month.split("-")[:2]
    return int(y) * 12 + int(m) - 1


def _month(i: int) -> str:
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def month_text(month: str) -> str:
    from ..scene.production import month_label

    return month_label(month)


def _metal(m: float, unit: str) -> str:
    from ..scene.production import metal_text

    return metal_text(m, unit)


# --- forecast -------------------------------------------------------------------------------------
@dataclass
class Forecast:
    ok: bool
    message: str = ""                                # why there is no forecast (ok = False)
    months: list = field(default_factory=list)       # YYYY-MM, every month from the first cut to "as of"
    tonnes: list = field(default_factory=list)       # mined in each month
    cumulative: list = field(default_factory=list)   # mined up to the end of each month
    total: float = 0.0
    remaining: float = 0.0
    rate: float = 0.0                                # expected t / month
    rate_low: float = 0.0                            # slow months
    rate_high: float = 0.0                           # fast months
    months_left: float = math.inf
    months_left_slow: float = math.inf
    months_left_fast: float = math.inf
    end_month: str = ""                              # YYYY-MM the reserve runs out (expected)
    end_slow: str = ""
    end_fast: str = ""
    used_months: list = field(default_factory=list)  # the months the rate comes from
    trend: float | None = None                       # change of the rate, last 3 vs 3 before (fraction)
    grade_recent: float | None = None
    grade_remaining: float = 0.0
    findings: list = field(default_factory=list)     # [(level, text)]


def forecast(rows: list[dict], total: float, remaining: float, remaining_metal: float, as_of: str,
             unit: str = "g/t") -> Forecast:
    """``rows``: the reserve's monthly table (``scene.reserve.monthly``); ``total`` / ``remaining``
    tonnes and remaining metal as of ``as_of`` (YYYY-MM-DD)."""
    rows = [r for r in rows if r["month"] <= as_of[:7]]
    if not rows or not any(r["tonnes"] > 0 for r in rows):
        return Forecast(False, tr("No production recorded yet - record mined cuts first; the forecast "
                                  "needs at least one month of production."), total=total, remaining=remaining)
    by = {r["month"]: r for r in rows}
    first, last = _mi(rows[0]["month"]), _mi(as_of[:7])
    months = [_month(i) for i in range(first, last + 1)]
    tonnes = [float(by[m]["tonnes"]) if m in by else 0.0 for m in months]
    metal = [float(by[m]["metal"]) if m in by else 0.0 for m in months]
    cum = list(np.cumsum(tonnes))
    # the month of "as of" is not over yet: leave it out of the rate when there are full months
    full = months[:-1] if len(months) > 1 else months
    used = full[-RECENT_MONTHS:]
    t_used = np.array([tonnes[months.index(m)] for m in used])
    rate = float(np.median(t_used))
    if len(t_used) >= 4:
        lo, hi = float(np.quantile(t_used, 0.25)), float(np.quantile(t_used, 0.75))
    else:
        lo, hi = float(t_used.min()), float(t_used.max())
    if rate <= 0:                                   # mostly idle months: use the average instead
        rate = float(t_used.mean())
    if lo <= 0:                                     # idle months: "slow" is half the typical rate
        lo = 0.5 * rate
    hi = max(hi, rate)
    f = Forecast(True, months=months, tonnes=tonnes, cumulative=cum, total=total, remaining=remaining,
                 rate=rate, rate_low=lo, rate_high=hi, used_months=used,
                 grade_remaining=remaining_metal / remaining if remaining > 0 else 0.0)

    def left(r):
        return remaining / r if r > 0 else math.inf

    f.months_left, f.months_left_slow, f.months_left_fast = left(rate), left(lo), left(hi)
    base = _mi(as_of[:7])

    def end(n):
        return _month(base + int(math.ceil(n))) if math.isfinite(n) and n < 12 * 200 else ""

    f.end_month, f.end_slow, f.end_fast = end(f.months_left), end(f.months_left_slow), end(f.months_left_fast)
    if len(full) >= 2 * TREND_MONTHS:
        a = float(np.mean(tonnes[len(full) - TREND_MONTHS:len(full)]))
        b = float(np.mean(tonnes[len(full) - 2 * TREND_MONTHS:len(full) - TREND_MONTHS]))
        f.trend = (a - b) / b if b > 0 else None
    recent = [i for i, m in enumerate(months) if m in full[-TREND_MONTHS:]]
    t_r = sum(tonnes[i] for i in recent)
    f.grade_recent = sum(metal[i] for i in recent) / t_r if t_r > 0 else None

    # --- findings, in plain words
    fd = f.findings
    if remaining <= 0:
        fd.append((OK, tr("The whole reserve is mined.")))
        return f
    if not math.isfinite(f.months_left):
        fd.append((WARN, tr("No ore was mined in the recent months, so the end of the reserve cannot be "
                            "estimated.")))
    else:
        fd.append((INFO, tr("At the recent rate of about {rate:,.0f} t a month, the remaining {rem:,.0f} t last "
                            "about {n} (until {end}).", rate=rate, rem=remaining, n=_duration(f.months_left),
                            end=month_text(f.end_month))))
        if f.end_slow and f.end_fast and f.end_slow != f.end_fast:
            fd.append((INFO, tr("If production stays like the slower months it lasts until {slow}; like the faster "
                                "months, until {fast}.", slow=month_text(f.end_slow), fast=month_text(f.end_fast))))
    if len(used) < 3:
        fd.append((WARN, tr("Only {n} month(s) of production so far - the forecast is rough until there are "
                            "a few more months.", n=len(used))))
    if f.trend is not None:
        if f.trend <= -0.2:
            fd.append((WARN, tr("Production fell {p:.0%} in the last 3 months compared with the 3 months before.",
                                p=-f.trend)))
        elif f.trend >= 0.2:
            fd.append((INFO, tr("Production rose {p:.0%} in the last 3 months compared with the 3 months before.",
                                p=f.trend)))
        else:
            fd.append((OK, tr("Production is steady (within 20 % over the last 6 months).")))
    if f.grade_recent is not None and f.grade_remaining > 0:
        gap = (f.grade_recent - f.grade_remaining) / f.grade_remaining
        g_txt = dict(recent=f"{f.grade_recent:,.2f}", rest=f"{f.grade_remaining:,.2f}", unit=unit)
        if gap > GRADE_GAP:
            fd.append((WARN, tr("Recent cuts are {p:.0%} richer than the ore that is left ({recent} against {rest} "
                                "{unit}): the grade will drop later (high-grading).", p=gap, **g_txt)))
        elif gap < -GRADE_GAP:
            fd.append((WARN, tr("Recent cuts are {p:.0%} poorer than the ore that is left ({recent} against {rest} "
                                "{unit}): check for dilution or mining outside the ore.", p=-gap, **g_txt)))
        else:
            fd.append((OK, tr("The grade mined recently ({recent} {unit}) matches the ore that is left ({rest} {unit}).",
                              **g_txt)))
    return f


def _duration(months: float) -> str:
    if months < 1.5:
        return tr("about a month")
    if months < 24:
        return tr("{n} months", n=int(round(months)))
    return tr("{y:.1f} years ({n} months)", y=months / 12.0, n=int(round(months)))


def projection(f: Forecast, rate: float, horizon: int = 240) -> tuple[list[int], list[float]]:
    """Cumulative tonnes month by month from the last month at ``rate`` until the total is reached
    (month offsets from the first month, values)."""
    if not f.months or rate <= 0:
        return [], []
    start = len(f.months) - 1
    xs, ys = [start], [f.cumulative[-1]]
    cur = f.cumulative[-1]
    for k in range(1, horizon + 1):
        cur = min(f.total, cur + rate)
        xs.append(start + k)
        ys.append(cur)
        if cur >= f.total:
            break
    return xs, ys


# --- check of the mined cuts ----------------------------------------------------------------------------
@dataclass
class CutCheck:
    sid: int
    name: str
    date: str
    level: str                                   # OK / WARN / BAD
    reasons: list                                # [(level, text)]
    tonnes: float = 0.0                          # new ore it mined
    grade: float = 0.0
    metal: float = 0.0
    fill: float = 0.0                            # ore share of the cut's volume (0..1)
    already: float = 0.0                         # tonnes inside it mined by earlier cuts
    thickness: float | None = None
    after_as_of: bool = False


def review_cuts(dep, shapes: list, unit: str = "g/t", as_of_day: int | None = None) -> list[CutCheck]:
    """A plain-language check of every cut (sorted: problems first, then by date)."""
    total_t = float(dep.tonnes.sum())
    avg = float((dep.tonnes * dep.grade).sum()) / total_t if total_t > 0 else 0.0
    prisms = [s.depth for s in shapes if not s.is_solid and s.depth > 0]
    usual = float(np.median(prisms)) if len(prisms) >= 3 else None
    out = []
    for sh in shapes:
        p = dep.preview(sh)
        new, old = p["new"], p["already"]
        reasons = []
        if new.tonnes <= 0.5:
            reasons.append((BAD, tr("It mined no new ore: everything inside it was mined before, or it is outside "
                                    "the reserve.")))
        if p["fill"] < 0.25:
            reasons.append((BAD, tr("Only {p:.0%} of the cut is ore - most of it is outside the reserve (waste).",
                                    p=p["fill"])))
        elif p["fill"] < 0.5:
            reasons.append((WARN, tr("Only {p:.0%} of the cut is ore - a lot of it is outside the reserve.",
                                     p=p["fill"])))
        if old.tonnes > 0.5 and old.tonnes > new.tonnes:
            reasons.append((WARN, tr("Most of it ({t:,.0f} t) was mined by earlier cuts already - is it drawn "
                                     "twice?", t=old.tonnes)))
        if new.tonnes > 0.5 and avg > 0:
            ratio = new.grade / avg
            if ratio < 0.5:
                reasons.append((WARN, tr("Its grade {g:,.2f} {unit} is well below the reserve average "
                                         "{avg:,.2f} {unit} (a low-grade area).", g=new.grade, avg=avg, unit=unit)))
            elif ratio > 2.0:
                reasons.append((INFO, tr("A rich cut: {g:,.2f} {unit}, more than twice the reserve average.",
                                         g=new.grade, unit=unit)))
        if usual and not sh.is_solid and sh.depth > 0 and (sh.depth > 3 * usual or sh.depth < usual / 3):
            reasons.append((WARN, tr("Its thickness {t:,.1f} m is unusual (most cuts are about {u:,.1f} m) - check "
                                     "From / To.", t=sh.depth, u=usual)))
        late = as_of_day is not None and sh.day > as_of_day
        if late:
            reasons.append((INFO, tr("Dated after the 'as of' date - not counted in today's figures.")))
        level = BAD if any(lv == BAD for lv, _ in reasons) else WARN if any(lv == WARN for lv, _ in reasons) else OK
        if not reasons or level == OK and all(lv == INFO for lv, _ in reasons):
            reasons.insert(0, (OK, tr("Looks normal: mostly ore, not mined before, usual grade.")))
        out.append(CutCheck(sh.id, sh.name, sh.date, level, reasons, new.tonnes, new.grade, new.metal, p["fill"],
                            old.tonnes, None if sh.is_solid else sh.depth, late))
    order = {BAD: 0, WARN: 1, OK: 2}
    out.sort(key=lambda c: (order[c.level], c.date, c.sid))
    return out


# --- best From / To for a new cut -------------------------------------------------------------------------
AIMS = ("metal", "grade", "tonnes")


@dataclass
class LevelSuggestion:
    ok: bool
    message: str = ""
    from_: float = 0.0
    to: float = 0.0
    tonnes: float = 0.0
    metal: float = 0.0
    alternatives: list = field(default_factory=list)   # [(from, to, tonnes, metal)]

    @property
    def grade(self) -> float:
        return self.metal / self.tonnes if self.tonnes > 0 else 0.0


def _free_points(dep, sh):
    """Depth coordinate, tonnes and metal of the reserve sample points inside the outline (over the
    whole depth range) that no other cut has mined."""
    iu, iv, ia = sh.axes
    lo, hi = sh.bounds()
    lo[ia], hi[ia] = -1e18, 1e18
    rows = dep.candidates(lo, hi)
    if not len(rows):
        return None
    P = len(dep.offsets)
    pts = dep.points(rows)
    inside = sh.in_outline(pts)
    owner = dep.point_shape[rows].reshape(-1)
    free = inside & ((owner < 0) | (owner == sh.id))
    if not free.any():
        return None
    tp = np.repeat(dep.tonnes[rows] / P, P)[free]
    g = np.repeat(dep.grade[rows], P)[free]
    return pts[free, ia], tp, tp * g


def window_totals(a: np.ndarray, t: np.ndarray, m: np.ndarray, starts: np.ndarray, thickness: float):
    """Tonnes and metal of the points with starts <= a <= starts + thickness, for every start."""
    order = np.argsort(a)
    a, t, m = a[order], t[order], m[order]
    ct, cm = np.concatenate([[0.0], np.cumsum(t)]), np.concatenate([[0.0], np.cumsum(m)])
    i0 = np.searchsorted(a, starts - 1e-9, side="left")
    i1 = np.searchsorted(a, starts + thickness + 1e-9, side="right")
    return ct[i1] - ct[i0], cm[i1] - cm[i0]


def suggest_levels(dep, sh, thickness: float, aim: str = "metal") -> LevelSuggestion:
    """The best From / To window of ``thickness`` for the outline ``sh`` (from below to above)."""
    if thickness <= 0:
        return LevelSuggestion(False, tr("Set the thickness first - the suggestion keeps it."))
    data = _free_points(dep, sh)
    if data is None:
        return LevelSuggestion(False, tr("There is no ore left under this outline (all of it is mined, or it is "
                                         "outside the reserve)."))
    a, t, m = data
    s0 = math.floor((a.min() - thickness) / SNAP) * SNAP
    s1 = math.ceil(a.max() / SNAP) * SNAP
    starts = np.round(np.arange(s0, s1 + SNAP / 2, SNAP), 3)
    wt, wm = window_totals(a, t, m, starts, thickness)
    if aim == "tonnes":
        score = wt
    elif aim == "grade":
        enough = wt >= 0.5 * wt.max()                  # a real amount of ore, not one rich sliver
        score = np.where(enough & (wt > 0), wm / np.maximum(wt, 1e-12), -1.0)
    else:
        score = wm
    if score.max() <= 0:
        return LevelSuggestion(False, tr("There is no ore left under this outline (all of it is mined, or it is "
                                         "outside the reserve)."))
    best = _plateau_middle(score)
    picks = [best]
    masked = score.copy()
    for _ in range(2):                                 # next best windows that do not overlap the earlier ones
        for p in picks:
            near = np.abs(starts - starts[p]) < thickness - 1e-9
            masked[near] = -np.inf
        if not np.isfinite(masked).any() or masked.max() <= 0:
            break
        picks.append(_plateau_middle(masked))
    alts = [(float(starts[p]), float(starts[p] + thickness), float(wt[p]), float(wm[p])) for p in picks[1:]]
    return LevelSuggestion(True, "", float(starts[best]), float(starts[best] + thickness), float(wt[best]),
                           float(wm[best]), alts)


def _plateau_middle(score: np.ndarray) -> int:
    """Index of the best score; when several windows tie, the middle one (centred on the ore)."""
    top = np.max(score)
    idx = np.flatnonzero(score >= top - max(1e-9, abs(top) * 1e-9))
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    return int(run[len(run) // 2])


def today() -> str:
    return _dt.date.today().isoformat()
