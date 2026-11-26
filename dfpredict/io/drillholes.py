"""Drillhole database: collar, survey and interval tables, plus desurveying.

Sources:
* CSV files - one collar, one survey and any number of interval tables
  (lithology, assays, ...). See ``read_drillholes_csv``.
* A Surpac Access database (.accdb/.mdb), read with the pure-Python
  ``access-parser`` so no ODBC driver is needed. See ``read_drillholes_access``.
* The project database (geodatabase.sqlite or MySQL): drillhole_collar,
  drillhole_survey and drillhole_<interval table>, filled by the user (e.g. from
  Excel). See ``read_drillholes_sqlite``. A plain SQLite file with collar / survey /
  interval tables can be opened too.

Conventions:
* Azimuth in degrees clockwise from north.
* Dip in the project database (drillhole_survey) and on screen: degrees from vertically
  down - 0 = straight down, 90 = horizontal, 180 = straight up (``dip_zero_down``).
* Dip in Surpac files (.accdb / .mdb) and CSV: negative downward (-90 = straight down,
  0 = horizontal). Pass ``dip_positive_down=True`` (or leave "auto") for data where down
  is positive. ``display_dip`` turns such a dip into the on-screen one (90 + dip).
* Internally (trace, azimuth_dip_at) dip is negative downward; database dips are
  converted on reading (dip - 90).
* Hole paths are computed with the minimum-curvature method.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..i18n import N_, tr

_HOLE = ("holeid", "hole", "bhid", "dhid", "holename", "id")
_X = ("x", "east", "easting", "xcollar", "collarx")
_Y = ("y", "north", "northing", "ycollar", "collary")
_Z = ("z", "rl", "elev", "elevation", "zcollar", "collarz")
_MAXDEPTH = ("maxdepth", "eoh", "totaldepth", "depth", "length")
_SDEPTH = ("depth", "at", "md", "distance", "surveydepth")
_AZI = ("azimuth", "azi", "az", "brg", "bearing")
_DIP = ("dip", "inclination", "incl")
_FROM = ("depthfrom", "from", "fromdepth", "mfrom", "samplefrom")
_TO = ("depthto", "to", "todepth", "mto", "sampleto")


def _norm(name: str) -> str:
    return re.sub(r"[\s_()\[\]\-.]", "", str(name)).lower()


def _find(columns, names: tuple[str, ...], required: bool, what: str) -> str | None:
    """First column matching the highest-priority name in ``names``."""
    normed = [(c, _norm(c)) for c in columns]
    for name in names:
        for c, n in normed:
            if n == name:
                return c
    if required:
        raise ValueError(tr("No {what} column found in {columns}", what=tr(what), columns=list(columns)))
    return None


def _normalise_collar(df: pd.DataFrame) -> pd.DataFrame:
    h = _find(df.columns, _HOLE, True, N_("hole id"))
    x = _find(df.columns, _X, True, N_("collar X"))
    y = _find(df.columns, _Y, True, N_("collar Y"))
    z = _find(df.columns, _Z, True, N_("collar Z"))
    md = _find(df.columns, _MAXDEPTH, False, N_("max depth"))
    out = pd.DataFrame({
        "hole_id": df[h].astype(str).str.strip(),
        "x": pd.to_numeric(df[x], errors="coerce"),
        "y": pd.to_numeric(df[y], errors="coerce"),
        "z": pd.to_numeric(df[z], errors="coerce"),
    })
    out["max_depth"] = pd.to_numeric(df[md], errors="coerce") if md else np.nan
    extra = [c for c in df.columns if c not in {h, x, y, z, md}]
    for c in extra:
        out[c] = df[c].values
    return out.drop_duplicates("hole_id").reset_index(drop=True)


def _normalise_survey(df: pd.DataFrame) -> pd.DataFrame:
    h = _find(df.columns, _HOLE, True, N_("hole id"))
    d = _find(df.columns, _SDEPTH, True, N_("survey depth"))
    a = _find(df.columns, _AZI, True, N_("azimuth"))
    p = _find(df.columns, _DIP, True, N_("dip"))
    out = pd.DataFrame({
        "hole_id": df[h].astype(str).str.strip(),
        "depth": pd.to_numeric(df[d], errors="coerce"),
        "azimuth": pd.to_numeric(df[a], errors="coerce"),
        "dip": pd.to_numeric(df[p], errors="coerce"),
    })
    return out.dropna().sort_values(["hole_id", "depth"]).reset_index(drop=True)


def _normalise_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Return hole_id, depth_from, depth_to + other columns.

    Tables with only a 'to' depth (e.g. Surpac weathering) get 'from' filled
    from the previous interval of the same hole (0 for the first).
    """
    h = _find(df.columns, _HOLE, True, N_("hole id"))
    f = _find(df.columns, _FROM, False, N_("from depth"))
    t = _find(df.columns, _TO, True, N_("to depth"))
    out = pd.DataFrame({"hole_id": df[h].astype(str).str.strip()})
    out["depth_to"] = pd.to_numeric(df[t], errors="coerce")
    if f is not None:
        out["depth_from"] = pd.to_numeric(df[f], errors="coerce")
        out = out[["hole_id", "depth_from", "depth_to"]]
    for c in df.columns:
        if c not in {h, f, t}:
            out[c] = df[c].values
    if f is None:
        out = out.sort_values(["hole_id", "depth_to"], kind="stable")
        prev = out.groupby("hole_id")["depth_to"].shift(1).fillna(0.0)
        out.insert(1, "depth_from", prev.values)
    out = out.dropna(subset=["depth_from", "depth_to"])
    return out.sort_values(["hole_id", "depth_from"], kind="stable").reset_index(drop=True)


def display_dip(dip_internal: float) -> float:
    """The dip shown to the user (0 = straight down, 90 = horizontal) from an internal one
    (negative downward, -90 = straight down)."""
    return 90.0 + float(dip_internal) + 0.0          # + 0.0: never "-0.0"


def _direction(azimuth_deg: np.ndarray, dip_deg: np.ndarray) -> np.ndarray:
    """Unit vectors (E, N, Up) for azimuth (from north, clockwise) and dip (negative down)."""
    az = np.radians(azimuth_deg)
    dp = np.radians(dip_deg)
    c = np.cos(dp)
    return np.stack([c * np.sin(az), c * np.cos(az), np.sin(dp)], axis=-1)


def _rf(beta: np.ndarray) -> np.ndarray:
    """Minimum-curvature ratio factor 2/b * tan(b/2), equal to 1 as b -> 0."""
    out = np.ones_like(beta)
    m = beta > 1e-9
    out[m] = 2.0 / beta[m] * np.tan(beta[m] / 2.0)
    return out


@dataclass
class HoleTrace:
    hole_id: str
    collar: np.ndarray       # (3,)
    md: np.ndarray           # (S,) station depths, starting at 0
    xyz: np.ndarray          # (S, 3) station positions
    dirs: np.ndarray         # (S, 3) unit tangent at each station
    max_depth: float

    def positions_at(self, depths) -> np.ndarray:
        """Positions along the hole at the given downhole depths (min curvature)."""
        q = np.atleast_1d(np.asarray(depths, dtype=np.float64))
        out = np.empty((len(q), 3))
        md, xyz, t = self.md, self.xyz, self.dirs
        if len(md) == 1:
            return xyz[0] + q[:, None] * t[0]
        k = np.clip(np.searchsorted(md, q, side="right") - 1, 0, len(md) - 2)
        beyond = q > md[-1]
        before = q < md[0]
        inside = ~(beyond | before)

        if inside.any():
            ki = k[inside]
            t1, t2 = t[ki], t[ki + 1]
            dl = md[ki + 1] - md[ki]
            f = np.where(dl > 0, (q[inside] - md[ki]) / np.where(dl > 0, dl, 1.0), 0.0)
            cosb = np.clip(np.einsum("ij,ij->i", t1, t2), -1.0, 1.0)
            beta = np.arccos(cosb)
            sinb = np.sin(beta)
            small = sinb < 1e-9
            w1 = np.where(small, 1.0 - f, np.sin((1.0 - f) * beta) / np.where(small, 1.0, sinb))
            w2 = np.where(small, f, np.sin(f * beta) / np.where(small, 1.0, sinb))
            tf = w1[:, None] * t1 + w2[:, None] * t2
            tf /= np.linalg.norm(tf, axis=1, keepdims=True)
            step = f * dl
            out[inside] = xyz[ki] + (step / 2.0 * _rf(f * beta))[:, None] * (t1 + tf)
        if beyond.any():
            out[beyond] = xyz[-1] + (q[beyond] - md[-1])[:, None] * t[-1]
        if before.any():
            out[before] = xyz[0] + (q[before] - md[0])[:, None] * t[0]
        return out

    def direction_at(self, depths) -> np.ndarray:
        """Unit direction (E, N, Up) of the hole at the given depths (minimum curvature:
        the tangent turns evenly along the arc between two survey stations)."""
        q = np.atleast_1d(np.asarray(depths, dtype=np.float64))
        md, t = self.md, self.dirs
        if len(md) == 1:
            return np.repeat(t[:1], len(q), axis=0)
        k = np.clip(np.searchsorted(md, q, side="right") - 1, 0, len(md) - 2)
        dl = md[k + 1] - md[k]
        f = np.clip(np.where(dl > 0, (q - md[k]) / np.where(dl > 0, dl, 1.0), 0.0), 0.0, 1.0)
        t1, t2 = t[k], t[k + 1]
        beta = np.arccos(np.clip(np.einsum("ij,ij->i", t1, t2), -1.0, 1.0))
        sinb = np.sin(beta)
        small = sinb < 1e-9
        w1 = np.where(small, 1.0 - f, np.sin((1.0 - f) * beta) / np.where(small, 1.0, sinb))
        w2 = np.where(small, f, np.sin(f * beta) / np.where(small, 1.0, sinb))
        out = w1[:, None] * t1 + w2[:, None] * t2
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def azimuth_dip_at(self, depth: float) -> tuple[float, float]:
        """Azimuth (degrees clockwise from north) and dip (degrees, negative down) at a depth."""
        e, n, up = self.direction_at([depth])[0]
        az = math.degrees(math.atan2(e, n)) % 360.0 if math.hypot(e, n) > 1e-9 else 0.0
        return az, math.degrees(math.asin(max(-1.0, min(1.0, up))))

    def path(self, step: float | None = None, with_depths: bool = False):
        """Polyline along the hole from the collar to max depth.

        Returns the (P, 3) positions, or (positions, depths) with ``with_depths``.
        """
        end = self.max_depth if np.isfinite(self.max_depth) and self.max_depth > 0 else self.md[-1]
        depths = [self.md[self.md <= end]]
        if step:
            depths.append(np.arange(0.0, end, step))
        depths.append([end])
        d = np.unique(np.concatenate(depths))
        p = self.positions_at(d)
        return (p, d) if with_depths else p


@dataclass
class DrillholeDB:
    collars: pd.DataFrame
    surveys: pd.DataFrame
    intervals: dict[str, pd.DataFrame]
    translations: pd.DataFrame | None = None
    dip_positive_down: bool = False
    dip_zero_down: bool = False      # survey dips are degrees from vertically down (the project database)
    rock_colors: dict = field(default_factory=dict)   # rock code -> RGB, from drillhole_styles
    lazy_fields: dict = field(default_factory=dict)   # interval table -> note columns not loaded
    lazy_source: tuple | None = None                  # (database Target, table prefix) to fetch them

    def interval_notes(self, table: str, hole_id: str, depth_from: float, depth_to: float) -> dict:
        """The note columns (not loaded for the display) of one interval, read from the
        database: column -> text. {} when there are none or the database cannot be read."""
        cols = self.lazy_fields.get(table)
        if not cols or self.lazy_source is None:
            return {}
        target, prefix = self.lazy_source
        try:
            with target.connect(create=False) as db:
                name = db.table_name(prefix + table)
                all_cols = db.columns(name)
                h = _find(all_cols, _HOLE, True, N_("hole id"))
                f = _find(all_cols, _FROM, False, N_("from depth"))
                t = _find(all_cols, _TO, True, N_("to depth"))
                where = f"{db.q(h)} = ? AND {db.q(t)} BETWEEN ? AND ?"
                params = [hole_id, depth_to - 1e-6, depth_to + 1e-6]
                if f is not None:
                    where += f" AND {db.q(f)} BETWEEN ? AND ?"
                    params += [depth_from - 1e-6, depth_from + 1e-6]
                rows = db.query(f"SELECT {', '.join(db.q(c) for c in cols)} FROM {db.q(name)} WHERE {where}",
                                params)
        except Exception:  # noqa: BLE001 - not reachable now: shown without the note
            return {}
        return dict(zip(cols, rows[0])) if rows else {}
    source: str = ""
    warnings: list[str] = field(default_factory=list)
    _traces: dict[str, HoleTrace] = field(default_factory=dict, repr=False)

    @property
    def hole_ids(self) -> list[str]:
        return self.collars["hole_id"].tolist()

    # --- drilling areas (drillhole_collar.area_id, see drilling_schema.py) ------------------
    def _area_column(self) -> str | None:
        return next((c for c in self.collars.columns if str(c).lower() == "area_id"), None)

    def area_counts(self) -> dict[str, int]:
        """Area -> number of holes ({} when the collars have no areas)."""
        col = self._area_column()
        if col is None:
            return {}
        s = self.collars[col].dropna().astype(str).str.strip()
        return {str(k): int(v) for k, v in s[s != ""].value_counts().sort_index().items()}

    def live_areas(self) -> list[str]:
        """The areas of the holes being drilled or planned (state In work / Planned)."""
        col = self._area_column()
        st = next((c for c in self.collars.columns if str(c).lower() == "state"), None)
        if col is None or st is None:
            return []
        from .drilling_schema import IN_WORK, PLANNED, _state_number

        live = [_state_number(v) in (IN_WORK, PLANNED) or str(v).strip().lower() in ("in work", "planned")
                for v in self.collars[st]]
        return sorted({str(a) for a, m in zip(self.collars[col], live) if m and pd.notna(a) and str(a).strip()})

    def only_areas(self, areas) -> "DrillholeDB":
        """The holes of these areas (a new DrillholeDB; surveys and intervals follow)."""
        col = self._area_column()
        if col is None:
            return self
        keep = set(self.collars.loc[self.collars[col].astype(str).isin({str(a) for a in areas}), "hole_id"])
        return self.subset(keep)

    def subset(self, hole_ids) -> "DrillholeDB":
        """Only these holes (a new DrillholeDB; the same translations, colours and settings)."""
        import dataclasses

        keep = {str(h) for h in hole_ids}
        return dataclasses.replace(
            self, collars=self.collars[self.collars["hole_id"].isin(keep)].reset_index(drop=True),
            surveys=self.surveys[self.surveys["hole_id"].isin(keep)].reset_index(drop=True),
            intervals={k: v[v["hole_id"].isin(keep)].reset_index(drop=True) for k, v in self.intervals.items()},
            warnings=list(self.warnings), _traces={})

    def code_descriptions(self, table: str, field_name: str) -> dict[str, str]:
        """Code -> description from a Surpac translation table, if present."""
        tr = self.translations
        if tr is None or tr.empty:
            return {}
        m = (tr["table_name"].astype(str).str.lower() == table.lower()) & (
            tr["field_name"].astype(str).str.lower() == field_name.lower()
        )
        return dict(zip(tr.loc[m, "code"].astype(str), tr.loc[m, "description"].astype(str)))

    def trace(self, hole_id: str) -> HoleTrace:
        if hole_id in self._traces:
            return self._traces[hole_id]
        row = self.collars.loc[self.collars["hole_id"] == hole_id]
        if row.empty:
            raise KeyError(f"No collar for hole {hole_id!r}")
        row = row.iloc[0]
        collar = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
        sv = self.surveys.loc[self.surveys["hole_id"] == hole_id]
        if sv.empty:
            md = np.array([0.0])
            az = np.array([0.0])
            dip = np.array([90.0 if self.dip_positive_down else -90.0])
        else:
            md = sv["depth"].to_numpy(dtype=np.float64)
            az = sv["azimuth"].to_numpy(dtype=np.float64)
            dip = sv["dip"].to_numpy(dtype=np.float64)
            if self.dip_zero_down:
                dip = dip - 90.0                     # 0 = down -> internal -90 = down
            _, first = np.unique(md, return_index=True)
            md, az, dip = md[first], az[first], dip[first]
            if md[0] > 0:
                md = np.concatenate([[0.0], md])
                az = np.concatenate([[az[0]], az])
                dip = np.concatenate([[dip[0]], dip])
        if self.dip_positive_down and not self.dip_zero_down:
            dip = -dip
        t = _direction(az, dip)
        xyz = np.empty((len(md), 3))
        xyz[0] = collar
        if len(md) > 1:
            dl = np.diff(md)
            cosb = np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0)
            steps = (dl / 2.0 * _rf(np.arccos(cosb)))[:, None] * (t[:-1] + t[1:])
            xyz[1:] = collar + np.cumsum(steps, axis=0)
        maxd = float(row["max_depth"]) if pd.notna(row["max_depth"]) else float("nan")
        tr = HoleTrace(hole_id, collar, md, xyz, t, maxd)
        self._traces[hole_id] = tr
        return tr

    def interval_positions(self, table: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Return (intervals, from_xyz (N,3), to_xyz (N,3)) for an interval table.

        Rows whose hole has no collar get NaN positions and a warning.
        """
        df = self.intervals[table]
        p_from = np.full((len(df), 3), np.nan)
        p_to = np.full((len(df), 3), np.nan)
        known = set(self.hole_ids)
        missing = set()
        for hid, idx in df.groupby("hole_id", sort=False).indices.items():
            if hid not in known:
                missing.add(hid)
                continue
            ht = self.trace(hid)
            p_from[idx] = ht.positions_at(df["depth_from"].to_numpy()[idx])
            p_to[idx] = ht.positions_at(df["depth_to"].to_numpy()[idx])
        if missing:
            self.warnings.append(tr("{table}: {n} holes without collar, e.g. {holes}", table=table, n=len(missing),
                                    holes=sorted(missing)[:5]))
        return df, p_from, p_to

    def interval_paths(self, table: str, step: float = 2.0):
        """Each interval as a polyline following the (possibly curved) hole path.

        Returns (rows, points (P, 3), depths (P,), counts (M,)): ``rows`` are the positions
        in the interval table of the M drawable intervals, interval i owning the next
        ``counts[i]`` points. On a straight hole an interval is just its two end points;
        on a curved one it gets a point every ``step`` metres (a straight chord between
        from and to would cut across the bend)."""
        df = self.intervals[table]
        known = set(self.hole_ids)
        rows, pts, deps, counts = [], [], [], []
        frm_all = df["depth_from"].to_numpy(dtype=np.float64)
        to_all = df["depth_to"].to_numpy(dtype=np.float64)
        for hid, idx in df.groupby("hole_id", sort=False).indices.items():
            if hid not in known:
                continue
            try:
                tr = self.trace(hid)
            except KeyError:
                continue
            straight = len(tr.dirs) < 2 or bool(np.allclose(tr.dirs, tr.dirs[0], atol=1e-9))
            for i in idx:
                a, b = frm_all[i], to_all[i]
                if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
                    continue
                n = 1 if straight else max(1, int(np.ceil((b - a) / step)))
                d = np.linspace(a, b, n + 1)
                rows.append(i)
                deps.append(d)
                counts.append(n + 1)
            # positions for this hole in one call
            if deps and len(pts) < len(deps):
                start = len(pts)
                hole_d = np.concatenate(deps[start:])
                hole_p = tr.positions_at(hole_d)
                k = 0
                for d in deps[start:]:
                    pts.append(hole_p[k:k + len(d)])
                    k += len(d)
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)
        return (np.asarray(rows, dtype=np.int64), np.vstack(pts), np.concatenate(deps),
                np.asarray(counts, dtype=np.int64))


def _auto_dip(surveys: pd.DataFrame, dip_positive_down: bool | str) -> bool:
    if dip_positive_down != "auto":
        return bool(dip_positive_down)
    if surveys.empty:
        return False
    return bool((surveys["dip"] > 0).mean() > 0.5)


def read_drillholes_csv(
    collar: str | Path,
    survey: str | Path | None,
    intervals: dict[str, str | Path] | None = None,
    dip_positive_down: bool | str = "auto",
) -> DrillholeDB:
    """Read collar/survey/interval CSV files. ``intervals`` maps a table name to a path."""
    col = _normalise_collar(pd.read_csv(collar))
    if survey is not None:
        sv = _normalise_survey(pd.read_csv(survey))
    else:
        sv = pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"])
    ints = {name: _normalise_intervals(pd.read_csv(p)) for name, p in (intervals or {}).items()}
    return DrillholeDB(col, sv, ints, None, _auto_dip(sv, dip_positive_down), source=str(collar))


def _table(tables: dict[str, pd.DataFrame], name: str | None) -> str | None:
    """The key in ``tables`` matching ``name`` ignoring case."""
    if not name:
        return None
    return next((t for t in tables if t.lower() == name.lower()), None)


def _build_db(tables: dict[str, pd.DataFrame], source: str, msgs: list[str], collar_table: str = "collar",
              survey_table: str = "survey", interval_tables: list[str] | None = None,
              translation_table: str | None = "translation", dip_positive_down: bool | str = "auto",
              dip_zero_down: bool = False) -> DrillholeDB:
    """A DrillholeDB from raw tables (name -> DataFrame), the same for Access and SQLite sources.

    When ``interval_tables`` is None, every table with a hole id and a 'to' depth
    (other than collar / survey / translation / styles) is an interval table."""
    c = _table(tables, collar_table)
    if c is None:
        raise ValueError(tr("No '{table}' table in {file}", table=collar_table, file=Path(source).name))
    col = _normalise_collar(tables[c])
    s = _table(tables, survey_table)
    if s is not None:
        sv = _normalise_survey(tables[s])
    else:
        sv = pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"])
        msgs.append(tr("No '{table}' table: holes drawn straight down", table=survey_table))
    t = _table(tables, translation_table)
    trans = tables[t] if t is not None else None
    skip = {x.lower() for x in (collar_table, survey_table, translation_table or "", "styles")}
    names = interval_tables if interval_tables is not None else [n for n in tables if n.lower() not in skip]
    ints: dict[str, pd.DataFrame] = {}
    for name in names:
        key = _table(tables, name)
        if key is None:
            if interval_tables is not None:
                raise ValueError(tr("No '{table}' table in {file}", table=name, file=Path(source).name))
            continue
        if interval_tables is None and tables[key].empty:
            continue  # e.g. drillhole_assay made by the database set-up, nothing entered yet
        try:
            ints[key] = _normalise_intervals(tables[key])
        except ValueError:
            if interval_tables is not None:
                raise
            continue  # not an interval table
    down = False if dip_zero_down else _auto_dip(sv, dip_positive_down)
    return DrillholeDB(col, sv, ints, trans, down, dip_zero_down=dip_zero_down, source=str(source), warnings=msgs)


# --- Access (.accdb / .mdb) ----------------------------------------------------------------------
def _quiet_access_parser():
    logging.getLogger("access_parser").setLevel(logging.ERROR)
    logging.getLogger("access_parser.access_parser").setLevel(logging.ERROR)


@contextmanager
def _without_index_parsing():
    """access-parser 0.0.6 fails (KeyError) on some tables while listing their primary key
    columns, e.g. Surpac's 'sample' table. The key is not needed to read the rows, so for
    those tables the index list is dropped while the table definition is parsed."""
    import access_parser.access_parser as ap

    orig = ap.parse_table_data

    def no_indexes(*a, **k):
        data = orig(*a, **k)
        data["all_indexes"] = []
        return data

    ap.parse_table_data = no_indexes
    try:
        yield
    finally:
        ap.parse_table_data = orig


def read_access_tables(path: str | Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Every user table of an Access database (name -> DataFrame), without an ODBC driver.

    Returns the tables and messages for tables that could not be read."""
    from access_parser import AccessParser

    _quiet_access_parser()
    db = AccessParser(str(path))
    names = [t for t in db.catalog if not t.startswith("MSys") and not re.match(r"^f_[0-9A-F]{32}", t)]
    tables: dict[str, pd.DataFrame] = {}
    msgs: list[str] = []
    for name in names:
        try:
            try:
                data = db.parse_table(name)
            except KeyError:
                with _without_index_parsing():
                    data = db.parse_table(name)
            tables[name] = pd.DataFrame(data)
        except Exception as e:  # noqa: BLE001 - one bad table must not stop the others
            msgs.append(tr("{name}: could not be read ({error})", name=name, error=repr(e)))
    return tables, msgs


def read_drillholes_access(path: str | Path, collar_table: str = "collar", survey_table: str = "survey",
                           interval_tables: list[str] | None = None, translation_table: str | None = "translation",
                           dip_positive_down: bool | str = "auto") -> DrillholeDB:
    """Read a Surpac Access drillhole database straight from the .accdb / .mdb file.

    Tables that cannot be read are skipped and listed in ``DrillholeDB.warnings``."""
    tables, msgs = read_access_tables(path)
    for m in msgs:
        warnings.warn(m, stacklevel=2)
    return _build_db(tables, str(path), msgs, collar_table, survey_table, interval_tables, translation_table,
                     dip_positive_down)


# --- the project database (SQLite or MySQL) ------------------------------------------------------
# Drillholes kept in the project database (geodatabase.sqlite in the workspace folder, or the
# MySQL database chosen in Settings) as drillhole_<table>: drillhole_collar, drillhole_survey
# and interval tables such as drillhole_geology or drillhole_sample, plus an optional
# drillhole_translation (code descriptions). The tables are filled by the user (e.g. imported
# from Excel); GeoViewer only reads them.
DRILLHOLE_LAYER_NAME = "Drillholes (database)"
# Free-text columns (core descriptions, ...) are not read for the display - they can be long and
# are only needed for one interval at a time (Identify) or one hole (Drillhole Manager).
NOTE_COLUMNS = ("note", "notes", "comment", "comments", "remarks")
_SKIP_TABLES = {"drillhole_meta", "geoviewer_source"}     # bookkeeping of earlier versions, not data


def _drillhole_tables(db) -> list[str]:
    from .database import DRILLHOLE_PREFIX

    return [t for t in db.tables() if t.lower().startswith(DRILLHOLE_PREFIX) and t.lower() not in _SKIP_TABLES]


def read_sqlite_tables(where) -> tuple[dict[str, pd.DataFrame], bool, dict[str, list[str]]]:
    """The drillhole tables of a database (name -> DataFrame, prefix removed), whether it is a
    project database (drillhole_<table> tables, dip 0 = straight down) rather than a plain SQLite
    drillhole database (Surpac dips), and per table the note columns left out (NOTE_COLUMNS)."""
    from .database import DRILLHOLE_PREFIX, PROD_PREFIX, as_target

    tables: dict[str, pd.DataFrame] = {}
    skipped: dict[str, list[str]] = {}
    with as_target(where).connect(create=False) as db:
        prefixed = _drillhole_tables(db)
        names = prefixed or [t for t in db.tables()
                             if t.lower() not in _SKIP_TABLES and not t.lower().startswith(PROD_PREFIX)]
        for t in names:
            key = t[len(DRILLHOLE_PREFIX):] if prefixed else t
            cols = db.columns(t)
            keep = [c for c in cols if c.lower() not in NOTE_COLUMNS]
            if len(keep) < len(cols):
                skipped[key] = [c for c in cols if c not in keep]
            tables[key] = db.read_frame(t, keep)
    return tables, bool(prefixed), skipped


def read_drillholes_sqlite(where, collar_table: str = "collar", survey_table: str = "survey",
                           interval_tables: list[str] | None = None, translation_table: str | None = "translation",
                           dip_positive_down: bool | str = "auto") -> DrillholeDB:
    """Read the drillholes of a database (a Target, or an SQLite file path): the project
    database (drillhole_collar, ..., survey dip 0 = straight down) or plain collar / survey /
    interval tables (Surpac dips, negative downward)."""
    from .database import as_target

    tables, project, skipped = read_sqlite_tables(where)
    db = _build_db(tables, as_target(where).label, [], collar_table, survey_table, interval_tables,
                   translation_table, dip_positive_down, dip_zero_down=project)
    db.lazy_fields = {t: cols for t, cols in skipped.items() if t in db.intervals}
    db.lazy_source = (as_target(where), "drillhole_" if project else "")
    styles = tables.get("styles")
    if styles is not None:
        low = {c.lower(): c for c in styles.columns}
        if "rock" in low and "color" in low:
            for code, color in zip(styles[low["rock"]], styles[low["color"]]):
                rgb = _hex_rgb(color)
                if code is not None and rgb is not None:
                    db.rock_colors[str(code).strip()] = rgb
    return db


def _hex_rgb(text) -> tuple[float, float, float] | None:
    t = str(text or "").strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{6}", t):
        return tuple(int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return None


def project_has_drillhole_tables(target) -> bool:
    """The project database has a drillhole_collar table (holes or not; False when unreachable)."""
    if not target.exists():
        return False
    try:
        with target.connect(create=False) as db:
            return db.has_table("drillhole_collar")
    except Exception:  # noqa: BLE001
        return False


def project_has_drillholes(target) -> bool:
    """The project database has at least one hole in drillhole_collar (False when unreachable)."""
    if not target.exists():
        return False
    try:
        with target.connect(create=False) as db:
            t = db.table_name("drillhole_collar")
            return bool(t) and bool(db.query(f"SELECT 1 FROM {db.q(t)} LIMIT 1"))
    except Exception:  # noqa: BLE001
        return False


def drillhole_signature(target) -> tuple | None:
    """A fingerprint of the drillhole_ tables, so Refresh can tell that they were changed (e.g.
    by a new import). None when the database is unreachable.

    MySQL: from the table metadata (created / last changed, columns) - no rows are counted,
    so it stays quick on big tables. SQLite: names, columns and row counts (Refresh only
    asks when the database file has changed)."""
    from .database import change_signature

    try:
        with target.connect(create=False) as db:
            tracked = change_signature(db, "drillhole_")       # table_updates versions (set-up made)
            if tracked is not None:
                return ("tracked",) + tracked
            if db.is_mysql:
                return _mysql_signature(db)
            return tuple((t.lower(), tuple(db.columns(t)), db.scalar(f"SELECT COUNT(*) FROM {db.q(t)}"))
                         for t in _drillhole_tables(db))
    except Exception:  # noqa: BLE001
        return None


def _mysql_signature(db) -> tuple:
    try:
        db.execute("SET SESSION information_schema_stats_expiry = 0")    # MySQL 8: no cached times
    except Exception:  # noqa: BLE001 - older servers have no such setting (and no cache)
        pass
    like = "drillhole\\_%"
    tabs = db.query("SELECT TABLE_NAME, CREATE_TIME, UPDATE_TIME FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE ?", (like,))
    cols = dict(db.query("SELECT TABLE_NAME, GROUP_CONCAT(COLUMN_NAME ORDER BY ORDINAL_POSITION) "
                         "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE ? "
                         "GROUP BY TABLE_NAME", (like,)))
    return tuple(sorted((str(n).lower(), str(c), str(u), str(cols.get(n, ""))) for n, c, u in tabs
                        if str(n).lower() not in _SKIP_TABLES))
