"""Predicting the path of a hole ahead of the bit, from its design and the surveys taken so far.

* The drilled part is desurveyed from the surveys with GeoViewer's minimum-curvature ``HoleTrace``.
* Ahead of the bit the hole continues from the last station with the build and turn rates of the
  recent stations (weighted fit), fading with distance, plus the walk toward the bedding normal where
  the hole meets bedding at a low angle (spec 2: "trend toward perpendicular to bedding"), read from
  the geological model.
* A planned hole (no surveys) starts from the designed azimuth / dip at the collar.
* An ensemble of trial paths (random walk of azimuth and dip, with the spread of past holes' surveys)
  gives the uncertainty cone: the risk ahead is averaged over the paths.

No Qt here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..io.drillholes import HoleTrace, _direction, _rf

# Spread of the deviation per 30 m (degrees) when no statistics of past holes are given.
DEFAULT_SPREAD = {"dip": 0.9, "azimuth": 1.4}
FADE = 150.0          # m: the fitted build / turn rates fade with this distance ahead of the bit
WALK = 0.9            # deg / 30 m: walk toward the bedding normal at incidence 0


@dataclass
class PathEnsemble:
    hole_id: str
    md: np.ndarray                 # (M,) depths along the hole (1 m steps from 0)
    mean_xyz: np.ndarray           # (M, 3)
    mean_dirs: np.ndarray          # (M, 3)
    samples: np.ndarray            # (N, M, 3) trial paths (the drilled part is the same in all)
    sample_dirs: np.ndarray        # (N, M, 3)
    drilled_depth: float           # bit depth (0 = not started)
    planned_depth: float
    build_rate: float              # deg / 30 m used ahead (+ = steepening)
    turn_rate: float               # deg / 30 m used ahead (+ = clockwise)
    station_count: int

    @property
    def radius(self) -> np.ndarray:
        """(M,) lateral spread of the trial paths (1 standard deviation, m)."""
        if len(self.samples) < 2:
            return np.zeros(len(self.md))
        dev = self.samples - self.mean_xyz[None]
        return np.sqrt((dev ** 2).sum(axis=2).mean(axis=0))

    def end_offset(self) -> float:
        """Distance (m) at the planned depth between the predicted path and the straight design."""
        return float(self.radius[-1])

    def azimuth_dip(self, i: int) -> tuple[float, float]:
        e, n, u = self.mean_dirs[i]
        return math.degrees(math.atan2(e, n)) % 360.0, math.degrees(math.asin(max(-1, min(1, u))))


def survey_spread(surveys: pd.DataFrame) -> dict:
    """The spread of the azimuth / dip change per 30 m in past holes' surveys (robust)."""
    if surveys is None or len(surveys) < 20:
        return dict(DEFAULT_SPREAD)
    ddip, daz = [], []
    for _hid, g in surveys.sort_values(["hole_id", "depth"]).groupby("hole_id"):
        d = g["depth"].to_numpy(float)
        if len(d) < 3:
            continue
        step = np.diff(d)
        ok = step > 5
        dd = np.diff(g["dip"].to_numpy(float))[ok] / step[ok] * 30.0
        da = (np.diff(g["azimuth"].to_numpy(float))[ok] + 180) % 360 - 180
        da = da / step[ok] * 30.0
        ddip += list(dd - np.median(dd))
        daz += list(da - np.median(da))
    if len(ddip) < 20:
        return dict(DEFAULT_SPREAD)
    mad = lambda v: 1.4826 * float(np.median(np.abs(np.asarray(v) - np.median(v))))  # noqa: E731
    return {"dip": float(np.clip(mad(ddip), 0.2, 4.0)), "azimuth": float(np.clip(mad(daz), 0.3, 6.0))}


def drilled_trace(hole_id: str, collar_xyz, surveys: pd.DataFrame, plan_az: float, plan_dip: float,
                  depth: float) -> HoleTrace:
    """Minimum-curvature trace of the drilled part (dip negative down)."""
    collar = np.asarray(collar_xyz, np.float64)
    if surveys is None or not len(surveys):
        md, az, dip = np.array([0.0]), np.array([plan_az]), np.array([plan_dip])
    else:
        s = surveys.sort_values("depth").drop_duplicates("depth")
        md, az, dip = (s["depth"].to_numpy(float), s["azimuth"].to_numpy(float), s["dip"].to_numpy(float))
        if md[0] > 0:
            md, az, dip = np.r_[0.0, md], np.r_[az[0], az], np.r_[dip[0], dip]
    t = _direction(az, dip)
    xyz = np.empty((len(md), 3))
    xyz[0] = collar
    if len(md) > 1:
        dl = np.diff(md)
        cosb = np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1, 1)
        steps = (dl / 2.0 * _rf(np.arccos(cosb)))[:, None] * (t[:-1] + t[1:])
        xyz[1:] = collar + np.cumsum(steps, axis=0)
    return HoleTrace(hole_id, collar, md, xyz, t, float(depth))


def recent_rates(surveys: pd.DataFrame, n_last: int = 4) -> tuple[float, float]:
    """Build (dip) and turn (azimuth) rates in degrees per 30 m from the last stations (weighted fit,
    the latest counting most). 0 without at least two stations 10 m apart."""
    if surveys is None or len(surveys) < 2:
        return 0.0, 0.0
    s = surveys.sort_values("depth").drop_duplicates("depth").tail(n_last)
    d = s["depth"].to_numpy(float)
    if d[-1] - d[0] < 10:
        return 0.0, 0.0
    w = np.linspace(0.5, 1.0, len(d))
    dip = s["dip"].to_numpy(float)
    az = np.degrees(np.unwrap(np.radians(s["azimuth"].to_numpy(float))))
    b = np.polyfit(d, dip, 1, w=w)[0] * 30.0
    t = np.polyfit(d, az, 1, w=w)[0] * 30.0
    return float(np.clip(b, -6, 6)), float(np.clip(t, -8, 8))


def _rotate(v: np.ndarray, d_az: np.ndarray, d_dip: np.ndarray) -> np.ndarray:
    """Rotate directions (N, 3) by azimuth / dip changes (degrees)."""
    e, n, u = v[:, 0], v[:, 1], v[:, 2]
    az = np.arctan2(e, n) + np.radians(d_az)
    dip = np.arcsin(np.clip(u, -1, 1)) + np.radians(d_dip)
    dip = np.clip(dip, np.radians(-89.9), np.radians(-3.0))           # keeps going down
    c = np.cos(dip)
    return np.column_stack([c * np.sin(az), c * np.cos(az), np.sin(dip)])


def predict_paths(hole_id: str, collar_xyz, plan_az: float, plan_dip: float, planned_depth: float,
                  surveys: pd.DataFrame | None, drilled_depth: float, model=None, n_paths: int = 60,
                  spread: dict | None = None, extend: float = 0.0, seed: int = 0) -> PathEnsemble:
    """The predicted path (mean) and ``n_paths`` trial paths, 1 m steps from the collar to the planned
    depth (+ ``extend`` m). ``model``: the geological model (bedding walk); None: no walk."""
    rng = np.random.default_rng(seed)
    spread = spread or dict(DEFAULT_SPREAD)
    end = float(max(planned_depth, drilled_depth)) + float(extend)
    md = np.arange(0.0, math.floor(end) + 1.0, 1.0)
    if md[-1] < end:
        md = np.r_[md, end]
    drilled_depth = float(max(0.0, min(drilled_depth, end)))
    sv = surveys if surveys is not None and len(surveys) else None
    trace = drilled_trace(hole_id, collar_xyz, sv, plan_az, plan_dip, drilled_depth)
    k_bit = int(np.searchsorted(md, drilled_depth, side="right"))   # md[:k_bit] are drilled
    k_bit = max(k_bit, 1)
    base_xyz = trace.positions_at(md[:k_bit])
    base_dir = trace.direction_at(md[:k_bit])
    build, turn = recent_rates(sv) if sv is not None and drilled_depth > 0 else (0.0, 0.0)
    m_ahead = len(md) - k_bit
    N = max(1, int(n_paths))
    xyz = np.empty((N + 1, len(md), 3))             # path 0 = the mean (no random walk)
    dirs = np.empty((N + 1, len(md), 3))
    xyz[:, :k_bit] = base_xyz
    dirs[:, :k_bit] = base_dir
    if m_ahead > 0:
        p = np.repeat(base_xyz[-1:], N + 1, axis=0)
        v = np.repeat(base_dir[-1:], N + 1, axis=0)
        # trial rates: the fitted ones +- their uncertainty (less sure with fewer stations)
        n_st = 0 if sv is None else len(sv)
        rate_sd = 0.6 if n_st >= 4 else 1.0
        b_i = np.r_[build, build + rng.normal(0, rate_sd, N)]
        t_i = np.r_[turn, turn + rng.normal(0, rate_sd * 1.3, N)]
        s_dip = spread["dip"] / math.sqrt(30.0)
        s_az = spread["azimuth"] / math.sqrt(30.0)
        last = drilled_depth
        for j in range(m_ahead):
            k = k_bit + j
            step = md[k] - md[k - 1] if k > 0 else 1.0
            ahead = md[k] - last
            fade = math.exp(-ahead / FADE)
            d_dip = b_i * fade * step / 30.0
            d_az = t_i * fade * step / 30.0
            if j > 0:
                d_dip = d_dip + np.r_[0.0, rng.normal(0, s_dip * math.sqrt(step), N)]
                d_az = d_az + np.r_[0.0, rng.normal(0, s_az * math.sqrt(step), N)]
            v = _rotate(v, d_az, d_dip)
            if model is not None and model.ok and j % 5 == 0:
                v = _bedding_walk(model, p, v, 5.0)
            p = p + v * step
            xyz[:, k] = p
            dirs[:, k] = v
    return PathEnsemble(hole_id, md, xyz[0], dirs[0], xyz[1:], dirs[1:], drilled_depth, float(planned_depth),
                        build, turn, 0 if sv is None else len(sv))


def _bedding_walk(model, p: np.ndarray, v: np.ndarray, span: float) -> np.ndarray:
    """Turn the directions toward the bedding normal (down-going side) where incidence < 60°."""
    try:
        nrm = model.bedding_normal(p)
    except Exception:  # noqa: BLE001 - outside the model: no walk
        return v
    s = np.sign(np.einsum("ij,ij->i", nrm, v))
    nrm = nrm * np.where(s < 0, -1.0, 1.0)[:, None]               # the normal pointing along the hole
    dot = np.clip(np.einsum("ij,ij->i", nrm, v), -1, 1)
    inc = np.degrees(np.arcsin(np.abs(dot)))
    t = nrm - dot[:, None] * v
    tn = np.linalg.norm(t, axis=1)
    rate = np.radians(WALK * span / 30.0 * np.cos(np.radians(inc))) * (inc < 60) * (tn > 1e-9)
    t = t / np.where(tn > 1e-9, tn, 1.0)[:, None]
    out = v * np.cos(rate)[:, None] + t * np.sin(rate)[:, None]
    out[:, 2] = np.minimum(out[:, 2], -0.05)
    return out / np.linalg.norm(out, axis=1, keepdims=True)
