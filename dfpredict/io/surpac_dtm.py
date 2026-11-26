"""Reader for Surpac DTM / 3DM triangulation files (.dtm).

Layout::

    ore1.str,8078843500;algorithm=standard;fields=x,y     <- paired .str, checksum, options
    [sn, Y, X, Z, ...]                                     <- optional extra points
    0, 0.000, 0.000, 0.000, END                            <- end of point section
    OBJECT, 1,
    TRISOLATION, 1, neighbours=yes,validated=true,closed=yes,direction=solid
    tri_id, v1, v2, v3, n1, n2, n3,                        <- vertices and neighbour triangles
    ...
    END

Vertex numbers are 1-based record numbers in the paired .str file, separator
records included. The DTM's own extra points continue the numbering after the
last .str record. This was checked against the Surpac demo data: no triangle
uses a separator record, and edge lengths are smallest with this numbering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..i18n import tr
from .surpac_str import (
    StringFile,
    SurpacFormatError,
    _read_lines,
    read_records,
    read_str,
)


@dataclass
class Trisolation:
    id: int
    triangles: np.ndarray            # (M, 3) int64, 0-based indices into DtmFile.vertices
    neighbours: np.ndarray | None    # (M, 3) int64 triangle ids (Surpac numbering), or None
    triangle_ids: np.ndarray         # (M,) int64 Surpac triangle ids
    options: dict[str, str] = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return self.options.get("closed", "no").lower() == "yes"

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)


@dataclass
class DtmObject:
    id: int
    trisolations: list[Trisolation] = field(default_factory=list)

    @property
    def n_triangles(self) -> int:
        return sum(t.n_triangles for t in self.trisolations)


@dataclass
class DtmFile:
    path: Path
    str_name: str
    checksum: str
    options: dict[str, str]
    vertices: np.ndarray             # (V, 3) float64 XYZ; NaN rows for separator records
    objects: list[DtmObject]
    string_file: StringFile | None = None

    @property
    def n_triangles(self) -> int:
        return sum(o.n_triangles for o in self.objects)

    def iter_trisolations(self):
        for obj in self.objects:
            for tri in obj.trisolations:
                yield obj, tri

    def compact_mesh(self, obj_id: int | None = None, tri_id: int | None = None):
        """Return (points (P,3), faces (M,3)) keeping only vertices that are used.

        Filters by object id and/or trisolation id when given.
        """
        tris = [
            t.triangles
            for o, t in self.iter_trisolations()
            if (obj_id is None or o.id == obj_id) and (tri_id is None or t.id == tri_id)
        ]
        if not tris:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        faces = np.concatenate(tris)
        used, inverse = np.unique(faces, return_inverse=True)
        return self.vertices[used], inverse.reshape(-1, 3)


def _parse_dtm_header(line: str) -> tuple[str, str, dict[str, str]]:
    str_name, _, rest = line.partition(",")
    parts = rest.split(";")
    checksum = parts[0].strip() if parts else ""
    opts: dict[str, str] = {}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        if k.strip():
            opts[k.strip().lower()] = v.strip()
    return str_name.strip(), checksum, opts


def _parse_options(fields: list[str]) -> dict[str, str]:
    opts: dict[str, str] = {}
    for f in fields:
        k, sep, v = f.partition("=")
        if sep and k.strip():
            opts[k.strip().lower()] = v.strip()
    return opts


def _find_case_insensitive(directory: Path, name: str) -> Path | None:
    candidate = directory / name
    if candidate.exists():
        return candidate
    lname = name.lower()
    for p in directory.iterdir():
        if p.name.lower() == lname:
            return p
    return None


def read_dtm(path: str | Path, string_file: StringFile | str | Path | None = None) -> DtmFile:
    """Read a DTM file and resolve its vertices from the paired string file.

    ``string_file`` may be an already loaded StringFile or a path. When None,
    the .str named on the DTM header line is looked up next to the DTM.
    """
    path = Path(path)
    lines = _read_lines(path)
    if not lines:
        raise SurpacFormatError(tr("{path}: empty DTM file", path=path))
    str_name, checksum, opts = _parse_dtm_header(lines[0])

    # Extra points section (up to the END separator).
    _, spot_xyz, _, i = read_records(lines, 1)
    spot = np.asarray(spot_xyz, dtype=np.float64).reshape(-1, 3)

    # Triangles.
    objects: list[DtmObject] = []
    cur_obj: DtmObject | None = None
    cur_tri: dict | None = None

    def flush_tri():
        nonlocal cur_tri
        if cur_tri is not None and cur_obj is not None:
            rows = np.asarray(cur_tri["rows"], dtype=np.int64).reshape(-1, cur_tri["width"])
            has_nb = rows.shape[1] >= 7
            cur_obj.trisolations.append(
                Trisolation(
                    id=cur_tri["id"],
                    triangle_ids=rows[:, 0],
                    triangles=rows[:, 1:4] - 1,
                    neighbours=rows[:, 4:7] if has_nb else None,
                    options=cur_tri["options"],
                )
            )
        cur_tri = None

    n = len(lines)
    while i < n:
        raw = lines[i]
        i += 1
        s = raw.strip()
        if not s:
            continue
        head = s.split(",", 1)[0].strip().upper()
        if head == "END":
            break
        if head == "OBJECT":
            flush_tri()
            fields = [f.strip() for f in s.split(",")]
            cur_obj = DtmObject(id=int(fields[1]))
            objects.append(cur_obj)
            continue
        if head == "TRISOLATION":
            flush_tri()
            if cur_obj is None:
                cur_obj = DtmObject(id=1)
                objects.append(cur_obj)
            fields = [f.strip() for f in s.split(",")]
            cur_tri = {
                "id": int(fields[1]),
                "options": _parse_options(fields[2:]),
                "rows": [],
                "width": None,
            }
            continue
        if cur_tri is None:
            raise SurpacFormatError(tr("{path}:{line}: triangle outside TRISOLATION", path=path, line=i))
        vals = [int(v) for v in s.rstrip(",").split(",") if v.strip()]
        width = 7 if len(vals) >= 7 else 4
        if cur_tri["width"] is None:
            cur_tri["width"] = width
        elif cur_tri["width"] != width:
            raise SurpacFormatError(tr("{path}:{line}: inconsistent triangle record width", path=path, line=i))
        cur_tri["rows"].extend(vals[:width])
    flush_tri()

    # Resolve the paired string file.
    sf: StringFile | None
    if isinstance(string_file, StringFile):
        sf = string_file
    else:
        if string_file is not None:
            str_path = Path(string_file)
        else:
            found = _find_case_insensitive(path.parent, str_name)
            if found is None:
                raise SurpacFormatError(
                    tr("{path}: paired string file {name} not found next to the DTM", path=path, name=repr(str_name))
                )
            str_path = found
        sf = read_str(str_path)

    vertices = np.vstack([sf.records_xyz, spot]) if len(spot) else sf.records_xyz

    # Validate references.
    nv = len(vertices)
    for obj in objects:
        for t in obj.trisolations:
            if len(t.triangles) == 0:
                continue
            if t.triangles.min() < 0 or t.triangles.max() >= nv:
                raise SurpacFormatError(
                    tr("{path}: object {obj} trisolation {tri} references vertex {vertex} but only {n} are "
                       "available (string file may have been edited after triangulation)",
                       path=path, obj=obj.id, tri=t.id, vertex=t.triangles.max() + 1, n=nv)
                )
            if np.isnan(vertices[t.triangles]).any():
                raise SurpacFormatError(
                    tr("{path}: object {obj} trisolation {tri} references a separator record",
                       path=path, obj=obj.id, tri=t.id)
                )

    return DtmFile(
        path=path,
        str_name=str_name,
        checksum=checksum,
        options=opts,
        vertices=vertices,
        objects=objects,
        string_file=sf,
    )


def object_names(dtm: DtmFile) -> dict[int, str]:
    """Object id -> name for DTMs built from named sources (GOCAD objects, DXF layers).

    The names are kept as JSON in ``dtm.options["object_names"]``; Surpac DTMs have none.
    """
    import json

    raw = dtm.options.get("object_names") if dtm.options else None
    if not raw:
        return {}
    try:
        return {int(k): str(v) for k, v in json.loads(raw).items()}
    except (ValueError, TypeError, AttributeError):
        return {}
