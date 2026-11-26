"""When a survey traverse does not close: which single reading mistake would explain the
misclosure. Offline; no Qt.

The usual field-book mistakes are tried one at a time on every setup of the pit, and the whole
network is computed again (the same ``survey_store.compute`` the Survey window uses) for each:

    * a horizontal angle 180° off (a face right reading taken as face left), backsight or foresight
    * the backsight and foresight HA swapped
    * a zenith angle written as a vertical angle (90° - ZA) or from face right (360° - ZA)
    * HI and HT swapped
    * two adjacent digits swapped in a slope distance, a horizontal angle or a compass bearing,
      or the minutes and seconds of an angle swapped
    * a compass bearing taken the wrong way round (180°), or of the wrong kind (magnetic / true / grid)
    * a wrong backsight station (another station of the pit)
    * the sign of the magnetic declination or of the grid convergence (Survey settings)

A correction that brings the misclosure down is a candidate; candidates are ranked by the
misclosure left. The top one is "very likely" when it makes the closure pass and the next
explanation is clearly worse. Nothing is changed: the Survey window shows the result and the
engineer decides.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

from ..i18n import tr
from ..io.survey_store import compute, format_dms

MIN_GAIN = 0.3            # a candidate must remove at least 30 % of the misclosure
CLEAR_LEAD = 3.0          # "very likely": the next explanation leaves 3 x more misclosure


@dataclass
class Fix:
    setup: int                     # index of the setup in the pit (0-based); -1: a survey setting
    kind: str                      # key of the mistake (see KINDS)
    what: str                      # what was wrong, in words
    old: str                       # the value as entered
    new: str                       # the value that closes
    row: object = None             # the corrected SetupRow (None for a setting)
    settings: dict | None = None   # the corrected survey settings (a setting fix)
    misclosure: float = math.inf   # linear misclosure after the correction (m)
    ratio: float = 0.0
    passes: bool = False
    confidence: str = ""           # "very likely" / "likely" / "possible"
    path: list = field(default_factory=list)      # [(station, (x, y, z))] of the pit after the correction
    dz: float = 0.0                # height misclosure after the correction (m)

    @property
    def error(self) -> float:
        """3D misclosure after the correction (plan and height)."""
        return math.hypot(self.misclosure, self.dz)


@dataclass
class Diagnosis:
    status: str                    # "no_closure", "passes", "found", "not_found"
    message: str
    misclosure: float = 0.0
    ratio: float = 0.0
    allowed: float = 0.0           # misclosure that would just pass (length / tolerance)
    length: float = 0.0
    onto: str = ""
    limit: float = 0.0
    fixes: list = field(default_factory=list)
    path: list = field(default_factory=list)      # the pit as computed now
    tried: int = 0
    dz: float = 0.0                # height misclosure now (m)
    onto_xyz: tuple | None = None  # the station the traverse closes on
    miss_xyz: tuple | None = None  # where the traverse lands instead (as surveyed)


def _closure(net, pid):
    res = net.pits.get(pid)
    return None if res is None or res.closure is None else res


def _passes(res, limit: float) -> bool:
    """The closure passes in plan (the tolerance) and in height (no worse than the plan allowance)."""
    _de, _dn, dz, _lin, _ratio, passed, _onto = res.closure
    return bool(passed) and abs(dz) <= max(res.length / limit if limit else 0.0, 0.01)


def _path(net, rows) -> list:
    """The pit's stations in setup order with their coordinates (at, then each foresight)."""
    out, seen = [], set()
    for r in rows:
        for name in (r.at, r.fs):
            if name and name not in seen and name in net.coords:
                seen.add(name)
                out.append((name, net.coords[name]))
    return out


def _swaps(text: str) -> list[str]:
    """Every text with two adjacent digits swapped (only digits; distinct results)."""
    out = []
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a.isdigit() and b.isdigit() and a != b:
            out.append(text[:i] + b + a + text[i + 2:])
    return out


def _number_swaps(v: float) -> list[float]:
    out = []
    for t in _swaps(f"{v:.3f}"):
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _angle_swaps(deg: float) -> list[float]:
    """Angles with adjacent digits of the degrees / minutes / seconds swapped, and minutes <-> seconds."""
    total = round(abs(deg) * 3600)
    d, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    out = []
    if m != s:
        out.append(d + s / 60 + m / 3600)
    for t in _swaps(f"{d:03d}"):
        out.append(int(t) + m / 60 + s / 3600)
    for t in _swaps(f"{m:02d}"):
        if int(t) < 60:
            out.append(d + int(t) / 60 + s / 3600)
    for t in _swaps(f"{s:02d}"):
        if int(t) < 60:
            out.append(d + m / 60 + int(t) / 3600)
    return [a % 360.0 for a in out]


def _ang(v) -> str:
    return format_dms(v) if v is not None else "-"


def _m(v) -> str:
    return f"{v:.3f} m" if v is not None else "-"


def _candidates(rows: list, i: int, other_stations: list):
    """(kind, what, old, new, corrected row) for setup ``i``."""
    r = rows[i]
    rep = dataclasses.replace
    method = "coords" if r.method == "coords" else ("bearing" if r.fs_bearing is not None else "angles")
    if method == "coords":
        return
    if method == "angles":
        if r.fs_ha is not None:
            v = (r.fs_ha + 180.0) % 360.0
            yield ("fs_face", tr("The foresight HA is 180° off (a face right reading taken as face left)"),
                   _ang(r.fs_ha), _ang(v), rep(r, fs_ha=v))
            for v in _angle_swaps(r.fs_ha):
                yield ("fs_digits", tr("Two figures of the foresight HA are swapped"), _ang(r.fs_ha), _ang(v),
                       rep(r, fs_ha=v))
        if r.bs_ha is not None:
            v = (r.bs_ha + 180.0) % 360.0
            yield ("bs_face", tr("The backsight HA is 180° off (a face right reading taken as face left)"),
                   _ang(r.bs_ha), _ang(v), rep(r, bs_ha=v))
            for v in _angle_swaps(r.bs_ha):
                yield ("bs_digits", tr("Two figures of the backsight HA are swapped"), _ang(r.bs_ha), _ang(v),
                       rep(r, bs_ha=v))
            if r.fs_ha is not None and abs(r.fs_ha - r.bs_ha) > 1e-9:
                yield ("ha_swap", tr("The backsight and foresight HA are swapped"),
                       tr("BS {bs}, FS {fs}", bs=_ang(r.bs_ha), fs=_ang(r.fs_ha)),
                       tr("BS {bs}, FS {fs}", bs=_ang(r.fs_ha), fs=_ang(r.bs_ha)), rep(r, bs_ha=r.fs_ha, fs_ha=r.bs_ha))
        if r.bs_station:
            for name in other_stations:
                if name not in (r.bs_station, r.at, r.fs):
                    yield ("bs_station", tr("The backsight was another station"), r.bs_station, name,
                           rep(r, bs_station=name))
    else:                                                # compass
        v = (r.fs_bearing + 180.0) % 360.0
        yield ("brg_reverse", tr("The compass bearing was read the wrong way round (180° off)"),
               _ang(r.fs_bearing), _ang(v), rep(r, fs_bearing=v))
        for v in _angle_swaps(r.fs_bearing):
            yield ("brg_digits", tr("Two figures of the compass bearing are swapped"), _ang(r.fs_bearing), _ang(v),
                   rep(r, fs_bearing=v))
        for kind in ("magnetic", "true", "grid"):
            if kind != (r.fs_bearing_type or "magnetic"):
                yield ("brg_type", tr("The bearing is of another kind (magnetic / true / grid)"),
                       r.fs_bearing_type or "magnetic", kind, rep(r, fs_bearing_type=kind))
    if r.fs_za is not None:
        v = 90.0 - r.fs_za
        if 0 < v < 180:
            yield ("za_vertical", tr("The zenith angle was written as a vertical angle (90° − ZA)"), _ang(r.fs_za),
                   _ang(v), rep(r, fs_za=v))
        v = 360.0 - r.fs_za
        if 0 < v < 180:
            yield ("za_face", tr("The zenith angle is a face right reading (360° − ZA)"), _ang(r.fs_za), _ang(v),
                   rep(r, fs_za=v))
    if r.fs_sd:
        for v in _number_swaps(r.fs_sd):
            if v > 0:
                yield ("sd_digits", tr("Two figures of the slope distance are swapped"), _m(r.fs_sd), _m(v),
                       rep(r, fs_sd=v))
    if r.hi is not None and r.fs_ht is not None and abs(r.hi - r.fs_ht) > 1e-6:
        yield ("hi_ht", tr("The instrument height (HI) and target height (HT) are swapped"),
               tr("HI {hi}, HT {ht}", hi=_m(r.hi), ht=_m(r.fs_ht)), tr("HI {hi}, HT {ht}", hi=_m(r.fs_ht), ht=_m(r.hi)),
               rep(r, hi=r.fs_ht, fs_ht=r.hi))


def diagnose(pits: list, stations: list, setups: dict, settings: dict, pid: int, limit_fixes: int = 6) -> Diagnosis:
    """Look for the single mistake in pit ``pid`` that best explains its misclosure."""
    from ..io.survey_store import DEFAULT_SETTINGS

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    rows = list(setups.get(pid, []))
    net0 = compute(pits, stations, setups, settings)
    res = _closure(net0, pid)
    limit = float(settings["closure_ratio"])
    if res is None:
        return Diagnosis("no_closure", tr("This pit does not close on a station with known coordinates yet, so "
                                          "there is nothing to check. Close the traverse (a foresight onto a known "
                                          "station or back onto the entrance), then try again."),
                         limit=limit, path=_path(net0, rows))
    de, dn, dz, lin, ratio, passed, onto = res.closure
    allowed = res.length / limit if limit else 0.0
    heights_off = abs(dz) > max(allowed, 0.01)
    d = Diagnosis("passes" if passed and not heights_off else "not_found", "", lin, ratio, allowed, res.length, onto,
                  limit, path=_path(net0, rows), dz=dz)
    if onto in net0.coords:
        o = net0.coords[onto]
        d.onto_xyz = tuple(o)
        d.miss_xyz = (o[0] + de, o[1] + dn, o[2] + dz)
    if passed and not heights_off:
        d.message = tr("The traverse closes within the tolerance (1:{ratio:,.0f}, needs 1:{limit:,.0f}): there is "
                       "no mistake to look for.", ratio=min(ratio, 1e9), limit=limit)
        return d
    error0 = math.hypot(lin, dz)
    others = [n for n, _xyz in d.path]
    found = []

    def score(new_setups, new_settings):
        try:
            net = compute(pits, stations, new_setups, new_settings)
        except Exception:  # noqa: BLE001 - an odd variant: not a candidate
            return None, None
        r2 = _closure(net, pid)
        return (r2, net) if r2 is not None else (None, None)

    for i in range(len(rows)):
        for kind, what, old, new, row in _candidates(rows, i, others):
            d.tried += 1
            trial = list(rows)
            trial[i] = row
            r2, net = score({**setups, pid: trial}, settings)
            if r2 is None:
                continue
            err = math.hypot(r2.closure[3], r2.closure[2])
            if err <= error0 * (1 - MIN_GAIN):
                found.append(Fix(i, kind, what, old, new, row, None, r2.closure[3], r2.closure[4],
                                 _passes(r2, limit), path=_path(net, trial), dz=r2.closure[2]))
    has_compass = any(r.fs_bearing is not None and (r.fs_bearing_type or "magnetic") != "grid" for r in rows)
    for key, label in (("declination", tr("The sign of the magnetic declination (Survey settings) is wrong")),
                       ("convergence", tr("The sign of the grid convergence (Survey settings) is wrong"))):
        val = float(settings.get(key) or 0.0)
        if val and (has_compass or key == "convergence"):
            d.tried += 1
            st2 = {**settings, key: -val}
            r2, net = score(setups, st2)
            if r2 is not None and math.hypot(r2.closure[3], r2.closure[2]) <= error0 * (1 - MIN_GAIN):
                found.append(Fix(-1, key, label, _ang(val), _ang(-val), None, st2, r2.closure[3], r2.closure[4],
                                 _passes(r2, limit), path=_path(net, rows), dz=r2.closure[2]))
    found.sort(key=lambda f: f.error)
    unique, seen = [], set()
    for f in found:                                   # the same correction found twice (e.g. digit swaps)
        key = (f.setup, f.new)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    for k, f in enumerate(unique):
        nxt = unique[k + 1].error if k + 1 < len(unique) else math.inf
        if f.passes and k == 0 and nxt >= CLEAR_LEAD * max(f.error, 1e-3):
            f.confidence = "very likely"
        elif f.passes:
            f.confidence = "likely"
        else:
            f.confidence = "possible"
    d.fixes = unique[:limit_fixes]
    if d.fixes:
        d.status = "found"
        best = d.fixes[0]
        where = (tr("setup {n}", n=best.setup + 1) if best.setup >= 0 else tr("the survey settings"))
        d.message = tr("Most likely cause, in {where}: {what}. With {new} instead of {old} the misclosure "
                       "becomes {m:.3f} m ({verdict}).", where=where, what=best.what, new=best.new,
                       old=best.old, m=best.error,
                       verdict=tr("the closure passes") if best.passes else tr("better, but still too large"))
    else:
        d.message = tr("No single reading mistake explains the misclosure. The error is probably spread over "
                       "several readings, or a station was set up or named wrongly: check the instrument and target "
                       "heights, the station names, and re-measure the legs with the largest distances.")
    return d
