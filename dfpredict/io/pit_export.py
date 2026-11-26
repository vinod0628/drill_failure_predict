"""Export the underground survey's pits: centre lines to an ESRI shapefile, drive solids to a
Surpac DTM.

Choose all pits, or one pit with its branches (and their branches) with ``export_pits``.

* **Shapefile** (``.shp`` / ``.shx`` / ``.dbf`` / ``.cpg``): one PolyLineZ record per pit (a pit
  whose setups make several separate lines is one multi-part record), with the pit's name,
  code, parent, main pit, profile and plan length as attributes. M is the 3D chainage. Written
  here without a GIS library. No ``.prj`` is written: the survey has no coordinate system.
* **Surpac DTM** (``.dtm`` + its paired ``.str``): each pit is an OBJECT, each of its lines a
  closed, validated solid TRISOLATION - the same rectangular drive the view draws. Triangles
  wind like Surpac's own solids and carry their neighbours. In the ``.str`` each station's
  profile is a closed ring. The DTM header names only the ``.str`` - no checksum: Surpac
  refuses a wrong one, its algorithm is not published, and its file format says the checksum
  need not be present (and is left out by converters from other systems).
"""

from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..i18n import tr
from ..scene.survey import SurveyData, pit_drives

STR_MAX_STRING = 32000            # Surpac string numbers are 1 ... 32000


@dataclass
class PitExport:
    """What was written."""
    files: list = field(default_factory=list)       # Paths
    pits: int = 0                                    # pits written
    parts: int = 0                                   # lines (shapefile) or solids (DTM)
    skipped: list = field(default_factory=list)      # names of pits without coordinates


# --- which pits ---------------------------------------------------------------------------------
def export_pits(data: SurveyData, pid=None) -> list:
    """The pits to export, in tree order (each main pit followed by its branches, depth first):
    every pit for ``pid`` None, else that pit and all its branches."""
    by_parent: dict = {}
    ids = {p.id for p in data.pits}
    for p in data.pits:
        parent = p.parent_id if p.parent_id in ids else None
        by_parent.setdefault(parent, []).append(p)
    for kids in by_parent.values():
        kids.sort(key=lambda q: q.id)
    if pid is None:
        roots = list(by_parent.get(None, []))
    else:
        root = data.pit(pid)
        roots = [root] if root is not None else []
    out, seen = [], set()

    def walk(p):
        if p.id in seen:                                 # never loop on a bad parent chain
            return
        seen.add(p.id)
        out.append(p)
        for c in by_parent.get(p.id, []):
            walk(c)

    for r in roots:
        walk(r)
    return out


def pit_lines(data: SurveyData, pit) -> list:
    """The pit's centre lines as (K, 3) world X, Y, Z arrays (stations without coordinates are
    left out; a line needs 2 points)."""
    out = []
    for path in data.paths.get(pit.id, []):
        pts = [data.xyz(n) for n in path]
        pts = [x for x in pts if x is not None]
        if len(pts) >= 2:
            out.append(np.array(pts, dtype=np.float64))
    return out


def _plan_length(lines) -> float:
    return float(sum(np.hypot(*np.diff(L[:, :2], axis=0).T).sum() for L in lines))


# --- shapefile ----------------------------------------------------------------------------------
_SHP_POLYLINEZ = 13

# name, type, size, decimals
_DBF_FIELDS = [("PIT_ID", "N", 10, 0), ("NAME", "C", 80, 0), ("CODE", "C", 20, 0), ("PARENT", "C", 20, 0),
               ("MAIN_PIT", "C", 20, 0), ("START_STN", "C", 30, 0), ("WIDTH", "N", 10, 2),
               ("HEIGHT", "N", 10, 2), ("STN_LEVEL", "C", 10, 0), ("LENGTH_2D", "N", 14, 2),
               ("STATIONS", "N", 8, 0)]


def _fit_bytes(text: str, size: int) -> bytes:
    """UTF-8 text cut to ``size`` bytes without splitting a character, space padded."""
    b = (text or "").encode("utf-8")
    while len(b) > size:
        text = text[:-1]
        b = text.encode("utf-8")
    return b.ljust(size, b" ")


def _dbf_value(value, kind, size, dec) -> bytes:
    if kind == "N":
        if value is None:
            return b" " * size
        s = f"{float(value):.{dec}f}" if dec else str(int(round(float(value))))
        return s.encode("ascii")[:size].rjust(size, b" ")
    return _fit_bytes(str(value or ""), size)


def write_dbf(path: Path, rows: list, fields=_DBF_FIELDS) -> None:
    """A dBase III table (the shapefile's attributes); text is UTF-8 (see the .cpg file)."""
    today = dt.date.today()
    rec_len = 1 + sum(f[2] for f in fields)
    head_len = 32 + 32 * len(fields) + 1
    out = bytearray(struct.pack("<BBBBIHH20x", 0x03, today.year - 1900, today.month, today.day,
                                len(rows), head_len, rec_len))
    for name, kind, size, dec in fields:
        out += struct.pack("<11sc4xBB14x", name.encode("ascii"), kind.encode("ascii"), size, dec)
    out += b"\x0d"
    for row in rows:
        out += b" "
        for name, kind, size, dec in fields:
            out += _dbf_value(row.get(name), kind, size, dec)
    out += b"\x1a"
    path.write_bytes(bytes(out))


def _shp_header(length_words: int, bbox, zr, mr) -> bytes:
    return (struct.pack(">7i", 9994, 0, 0, 0, 0, 0, length_words)
            + struct.pack("<2i", 1000, _SHP_POLYLINEZ)
            + struct.pack("<8d", *bbox, *zr, *mr))


def _polylinez(parts: list) -> bytes:
    """The content of one PolyLineZ record (with M = 3D chainage along each part)."""
    P = np.vstack(parts)
    starts = np.cumsum([0] + [len(p) for p in parts[:-1]])
    m = np.concatenate([np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] for p in parts])
    lo, hi = P.min(axis=0), P.max(axis=0)
    return (struct.pack("<i4d2i", _SHP_POLYLINEZ, lo[0], lo[1], hi[0], hi[1], len(parts), len(P))
            + struct.pack(f"<{len(parts)}i", *starts)
            + P[:, :2].astype("<f8").tobytes()
            + struct.pack("<2d", lo[2], hi[2]) + P[:, 2].astype("<f8").tobytes()
            + struct.pack("<2d", m.min(), m.max()) + m.astype("<f8").tobytes())


def write_centerlines_shp(path, data: SurveyData, pid=None) -> PitExport:
    """The pits' centre lines (all pits, or pit ``pid`` and its branches) as a PolyLineZ
    shapefile. Writes ``.shp``, ``.shx``, ``.dbf`` and ``.cpg`` next to each other."""
    path = Path(path).with_suffix(".shp")
    res = PitExport()
    by_id = {p.id: p for p in data.pits}
    records, rows, allp, allm = [], [], [], []
    for p in export_pits(data, pid):
        lines = pit_lines(data, p)
        if not lines:
            res.skipped.append(p.name)
            continue
        content = _polylinez(lines)
        records.append(content)
        allp.append(np.vstack(lines))
        allm += [0.0, max(float(np.linalg.norm(np.diff(L, axis=0), axis=1).sum()) for L in lines)]
        parent = by_id.get(p.parent_id)
        main = by_id.get(data.main_of(p.id))
        rows.append({"PIT_ID": p.id, "NAME": p.name, "CODE": p.code, "PARENT": parent.code if parent else "",
                     "MAIN_PIT": main.code if main else p.code, "START_STN": p.start_station or "",
                     "WIDTH": p.profile_width, "HEIGHT": p.profile_height, "STN_LEVEL": p.station_level,
                     "LENGTH_2D": _plan_length(lines),
                     "STATIONS": sum(1 for s in data.stations.values() if s.pit_id == p.id)})
        res.pits += 1
        res.parts += len(lines)
    if not records:
        raise ValueError(tr("None of these pits has coordinates yet - nothing to export."))
    A = np.vstack(allp)
    lo, hi = A.min(axis=0), A.max(axis=0)
    bbox, zr, mr = (lo[0], lo[1], hi[0], hi[1]), (lo[2], hi[2]), (min(allm), max(allm))
    shp, shx = bytearray(), bytearray()
    offset = 50                                           # in 16-bit words, after the header
    for i, content in enumerate(records, 1):
        words = len(content) // 2
        shp += struct.pack(">2i", i, words) + content
        shx += struct.pack(">2i", offset, words)
        offset += 4 + words
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_shp_header(50 + len(shp) // 2, bbox, zr, mr) + bytes(shp))
    path.with_suffix(".shx").write_bytes(_shp_header(50 + len(shx) // 2, bbox, zr, mr) + bytes(shx))
    write_dbf(path.with_suffix(".dbf"), rows)
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    res.files = [path.with_suffix(e) for e in (".shp", ".shx", ".dbf", ".cpg")]
    return res


# --- Surpac DTM -----------------------------------------------------------------------------------
_EDGES = ((0, 1), (1, 2), (2, 0))      # Surpac: neighbour 1 across v1-v2, 2 across v2-v3, 3 across v3-v1


def drive_triangles(V: np.ndarray, Q: np.ndarray):
    """A drive (``drive_mesh`` vertices and quads) as a closed solid: (vertices (V, 3),
    triangles (M, 3), 0-based), wound like Surpac's solids (clockwise seen from outside in X,
    Y, Z)."""
    T = np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    if np.einsum("ij,ij->i", a - V.mean(axis=0), np.cross(b - a, c - a)).sum() > 0:
        T = T[:, ::-1]
    return V, np.ascontiguousarray(T)


def triangle_neighbours(T: np.ndarray) -> np.ndarray:
    """(M, 3) 1-based neighbour triangle across each edge (Surpac order), 0 where none."""
    m = len(T)
    e = np.stack([T[:, [i, j]] for i, j in _EDGES], axis=1).reshape(-1, 2)      # (3M, 2) directed
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    nb = np.zeros(3 * m, dtype=np.int64)
    first, second = order[:-1][same], order[1:][same]
    nb[first] = second // 3 + 1
    nb[second] = first // 3 + 1
    return nb.reshape(m, 3)


def _desc(text: str) -> str:
    return (text or "").replace(",", " ").strip()


def write_drives_dtm(path, data: SurveyData, pid=None) -> PitExport:
    """The pits' drives (all pits, or pit ``pid`` and its branches) as Surpac solids: writes
    ``<name>.dtm`` and its paired ``<name>.str``. Each pit is one object, numbered 1, 2, ...
    in tree order; the string records carry the pit's code and name (D1, D2)."""
    path = Path(path).with_suffix(".dtm")
    str_path = path.with_suffix(".str")
    res = PitExport()
    recs: list[str] = []                       # string records after the axis line
    tri_blocks: list[str] = []
    obj = 0
    for p in export_pits(data, pid):
        drives = pit_drives(data, p)                 # a branch starts at its parent's wall
        if not drives:
            res.skipped.append(p.name)
            continue
        obj += 1
        sn = min(obj, STR_MAX_STRING)
        tri_blocks.append(f"OBJECT, {obj},")
        d1, d2 = _desc(p.code), _desc(p.name)
        for t_id, (_line, V, Q) in enumerate(drives, 1):
            V, T = drive_triangles(V, Q)
            # each station's profile is a closed ring of the string (its 4 corners, the first
            # repeated to close it, then a separator - 6 records), as Surpac's own drive solids
            # are built from section rings; ``vnum`` = 1-based record number of each vertex
            base = len(recs) + 1
            vnum = base + 6 * (np.arange(len(V)) // 4) + np.arange(len(V)) % 4
            for k in range(0, len(V), 4):
                ring = list(V[k:k + 4]) + [V[k]]
                recs += [f"{sn}, {y:.3f}, {x:.3f}, {z:.3f}, {d1}, {d2}" for x, y, z in ring]
                recs.append("0, 0.000, 0.000, 0.000,")      # separator (counts as a record)
            nb = triangle_neighbours(T)
            R = vnum[T]
            tri_blocks.append(f"TRISOLATION, {t_id}, neighbours=yes,validated=true,closed=yes,direction=solid")
            tri_blocks += [f"{i}, {a}, {b}, {c}, {n1}, {n2}, {n3},"
                           for i, ((a, b, c), (n1, n2, n3)) in enumerate(zip(R.tolist(), nb.tolist()), 1)]
            res.parts += 1
        res.pits += 1
    if not obj:
        raise ValueError(tr("None of these pits has coordinates yet - nothing to export."))
    path.parent.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().strftime("%d-%b-%y")
    str_lines = [f"{str_path.stem},{today},GeoViewer pit drives,",
                 "0,           0.000,           0.000,           0.000,           0.000,           0.000,"
                 "           0.000"] + recs + ["0, 0.000, 0.000, 0.000, END"]
    str_path.write_text("\n".join(str_lines) + "\n", encoding="latin-1", errors="replace")
    # the header names only the string file: NO checksum. Surpac checks a checksum when there is
    # one (and 0 or a blank fails with "String file does not match DTM file"); its file format
    # says the checksum need not be present and is left out by converters from other systems
    dtm_lines = [str_path.name, "0, 0.000, 0.000, 0.000, END"]
    path.write_text("\n".join(dtm_lines + tri_blocks + ["END"]) + "\n", encoding="latin-1", errors="replace")
    res.files = [path, str_path]
    return res
