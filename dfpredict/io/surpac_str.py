"""Reader for Surpac string files (.str).

Layout::

    name,date,purpose,styles_file          <- header line
    0, y1, x1, z1, y2, x2, z2              <- axis record
    sn, Y, X, Z, D1, D2, ...               <- point record (string number sn > 0)
    0, 0.000, 0.000, 0.000,                <- segment separator
    ...
    0, 0.000, 0.000, 0.000, END            <- end of file

Surpac stores coordinates as **Y, X, Z** (northing first). This reader returns
X, Y, Z (east, north, elevation).

Every record after the axis line gets a 1-based record number, separators
included. DTM triangles refer to points by that record number (see
``surpac_dtm``), so ``records_xyz`` keeps one row per record, with NaN rows
for separators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..i18n import tr


class SurpacFormatError(ValueError):
    """Raised when a Surpac file does not match the expected layout."""


@dataclass
class StrHeader:
    name: str = ""
    date: str = ""
    purpose: str = ""
    styles_file: str = ""


@dataclass
class StringFile:
    path: Path
    header: StrHeader
    records_xyz: np.ndarray      # (R, 3) float64, X/Y/Z; NaN for separator records
    records_string: np.ndarray   # (R,) int32, string number; 0 for separators
    records_desc: list[str]      # (R,) raw text after Z (D1, D2, ...) or ""
    seg_string: np.ndarray       # (S,) int32, string number of each segment
    seg_start: np.ndarray        # (S,) int64, first record index (0-based) of segment
    seg_end: np.ndarray          # (S,) int64, one past last record index
    extra: dict = field(default_factory=dict)

    @property
    def n_records(self) -> int:
        return len(self.records_string)

    @property
    def n_segments(self) -> int:
        return len(self.seg_string)

    @property
    def point_mask(self) -> np.ndarray:
        return self.records_string != 0

    @property
    def points(self) -> np.ndarray:
        """(N, 3) XYZ of all real points (separators removed)."""
        return self.records_xyz[self.point_mask]

    @property
    def string_numbers(self) -> np.ndarray:
        return np.unique(self.seg_string)

    def segment_points(self, i: int) -> np.ndarray:
        return self.records_xyz[self.seg_start[i]:self.seg_end[i]]

    def segment_is_closed(self, i: int, tol: float = 1e-6) -> bool:
        """A Surpac closed string repeats its first point as its last point."""
        n = self.seg_end[i] - self.seg_start[i]
        if n < 3:
            return False
        p = self.segment_points(i)
        return bool(np.all(np.abs(p[0] - p[-1]) <= tol))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = self.points
        return pts.min(axis=0), pts.max(axis=0)


def _parse_header(line: str) -> StrHeader:
    parts = [p.strip() for p in line.split(",")]
    parts += [""] * (4 - len(parts))
    return StrHeader(name=parts[0], date=parts[1], purpose=parts[2], styles_file=parts[3])


def parse_point_record(line: str) -> tuple[int, float, float, float, str] | None:
    """Parse ``sn, Y, X, Z[, desc]`` -> (sn, x, y, z, desc). Returns None for blank lines."""
    parts = line.split(",", 4)
    if len(parts) < 4:
        if not line.strip():
            return None
        raise SurpacFormatError(tr("Bad record: {record}", record=repr(line)))
    sn = int(float(parts[0]))
    y = float(parts[1])
    x = float(parts[2])
    z = float(parts[3])
    desc = parts[4].strip() if len(parts) > 4 else ""
    return sn, x, y, z, desc


def read_records(lines: list[str], start: int, stop_on_end: bool = True):
    """Parse point/separator records from ``lines[start:]``.

    Returns (string_numbers, xyz, descs, next_line_index). Stops after a
    separator record whose description is ``END`` when ``stop_on_end``.
    """
    sns: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    descs: list[str] = []
    i = start
    n = len(lines)
    while i < n:
        rec = parse_point_record(lines[i])
        i += 1
        if rec is None:
            continue
        sn, x, y, z, desc = rec
        if sn == 0 and stop_on_end and desc.upper().startswith("END"):
            break
        sns.append(sn)
        xyz.append((x, y, z))
        descs.append(desc)
    return sns, xyz, descs, i


def _read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        return f.read().splitlines()


def read_str(path: str | Path) -> StringFile:
    path = Path(path)
    lines = _read_lines(path)
    if len(lines) < 2:
        raise SurpacFormatError(tr("{path}: too short for a string file", path=path))
    header = _parse_header(lines[0])
    sns, xyz, descs, _ = read_records(lines, 2)

    rec_sn = np.asarray(sns, dtype=np.int32)
    rec_xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rec_xyz[rec_sn == 0] = np.nan

    # A segment is a run of consecutive non-zero records with the same string number.
    seg_string, seg_start, seg_end = [], [], []
    cur_sn, cur_start = 0, 0
    for idx, sn in enumerate(sns):
        if sn != cur_sn:
            if cur_sn != 0:
                seg_string.append(cur_sn)
                seg_start.append(cur_start)
                seg_end.append(idx)
            cur_sn, cur_start = sn, idx
    if cur_sn != 0:
        seg_string.append(cur_sn)
        seg_start.append(cur_start)
        seg_end.append(len(sns))

    return StringFile(
        path=path,
        header=header,
        records_xyz=rec_xyz,
        records_string=rec_sn,
        records_desc=descs,
        seg_string=np.asarray(seg_string, dtype=np.int32),
        seg_start=np.asarray(seg_start, dtype=np.int64),
        seg_end=np.asarray(seg_end, dtype=np.int64),
    )
