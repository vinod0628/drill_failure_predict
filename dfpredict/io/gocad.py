"""Readers for GOCAD ASCII objects (.ts .pl .vs .so) and GOCAD voxets / SGrids (.vo .sg).

GOCAD / SKUA write these files, and GeoModeller, Leapfrog, Geoscience ANALYST and most
3D modelling packages export them. One file may hold several objects::

    GOCAD TSurf 1                               <- object type and version
    HEADER {
    name:Ore lens 1
    *solid*color:0.8 0.2 0.1 1                  <- RGB(A) 0..1 (or #rrggbb)
    }
    GOCAD_ORIGINAL_COORDINATE_SYSTEM
    NAME Default
    ZPOSITIVE Depth                             <- Depth: z grows downwards (negated here)
    END_ORIGINAL_COORDINATE_SYSTEM
    PROPERTIES grade density                    <- names of the PVRTX extra values
    NO_DATA_VALUES -99999 -99999
    TFACE
    PVRTX 10 1000.0 2000.0 -50.0 1.2 2.7        <- id x y z props...
    VRTX 11 1010.0 2000.0 -50.0
    ATOM 12 10                                  <- a second vertex id for vertex 10
    TRGL 10 11 12
    BSTONE 10                                   <- (ignored, as BORDER)
    END

Supported objects: ``TSurf`` (TRGL), ``PLine`` (ILINE / SEG), ``VSet`` (points) and
``TSolid`` (TETRA: only the boundary triangles of the tetrahedral mesh are kept, i.e.
faces used by exactly one tetrahedron). Other objects (Well, Voxet, SGrid, GShape, ...)
are skipped; voxets and SGrids are read with :func:`read_voxet` / :func:`read_sgrid`.

Vertex ids are arbitrary integers; they are remapped to 0-based indices into
``GocadObject.vertices``. ATOM / PATOM lines reuse the referenced vertex (same index).
Property values equal to the object's ``NO_DATA_VALUES`` become NaN.

Voxets (``GOCAD Voxet 1``, .vo) are an ASCII header plus one raw binary file per property
(``PROP_FILE``). Values sit on grid NODES. Node (i, j, k) is at::

    O + (min_u + i * (max_u - min_u) / (nu - 1)) * U
      + (min_v + j * (max_v - min_v) / (nv - 1)) * V
      + (min_w + k * (max_w - min_w) / (nw - 1)) * W

The binary files are big-endian by default (GOCAD's convention) and stored with u
fastest, then v, then w; :class:`Voxet` keeps each property as an array shaped
``(nw, nv, nu)`` so that ``prop[k, j, i]`` is node (i, j, k) and ``prop.ravel()`` is in
file order.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .block_model import SURPAC_NULL, BlockModel
from .surpac_dtm import DtmFile, DtmObject, Trisolation
from .surpac_str import StrHeader, StringFile

RGB = tuple[float, float, float]

GOCAD_KINDS = ("TSurf", "PLine", "VSet", "TSolid")
_KIND_ALIASES = {k.lower(): k for k in GOCAD_KINDS}
_TOKEN = re.compile(r'"[^"]*"|\S+')


class GocadError(ValueError):
    """Raised when a GOCAD file does not match the expected layout."""


@dataclass
class GocadObject:
    kind: str                                   # 'TSurf', 'PLine', 'VSet' or 'TSolid'
    name: str
    color: RGB | None                           # RGB floats 0..1, or None
    vertices: np.ndarray                        # (N, 3) float64, z up (ZPOSITIVE Depth negated)
    triangles: np.ndarray | None = None         # (M, 3) int64 indices into vertices (TSurf, TSolid)
    segments: list[np.ndarray] | None = None    # PLine: one int64 vertex-index array per polyline
    vertex_props: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N,) float64, NaN = no data
    header: dict[str, str] = field(default_factory=dict)   # raw HEADER keys (name, *solid*color, ...)
    info: dict = field(default_factory=dict)    # coordinate system, n_tetra, geological feature, ...

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return 0 if self.triangles is None else len(self.triangles)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.vertices):
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


# --------------------------------------------------------------------------- helpers


def _tokens(s: str) -> list[str]:
    return [t.strip('"') for t in _TOKEN.findall(s)]


def parse_gocad_color(text: str) -> RGB | None:
    """``"0.8 0.2 0.1 1"`` / ``"#cc3319"`` / ``"204 51 25"`` -> RGB floats 0..1 (None if unreadable)."""
    s = text.strip()
    if s.startswith("#") and len(s) >= 7:
        try:
            return (int(s[1:3], 16) / 255.0, int(s[3:5], 16) / 255.0, int(s[5:7], 16) / 255.0)
        except ValueError:
            return None
    try:
        vals = [float(v) for v in s.replace(",", " ").split()]
    except ValueError:
        return None
    if len(vals) < 3:
        return None
    rgb = vals[:3]
    if max(rgb) > 1.0:
        rgb = [v / 255.0 for v in rgb]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _header_color(header: dict[str, str], kind: str) -> RGB | None:
    prefer = {"TSurf": "*solid*color", "TSolid": "*solid*color", "PLine": "*line*color",
              "VSet": "*atoms*color"}.get(kind, "")
    keys = [prefer] + [k for k in header if k.endswith("color")]
    for k in keys:
        if k in header:
            c = parse_gocad_color(header[k])
            if c is not None:
                return c
    return None


def _remap(ids: np.ndarray, id_index: dict[int, int], path: Path, name: str) -> np.ndarray:
    """Map GOCAD vertex ids (any shape) to 0-based indices; raise on unknown ids."""
    if ids.size == 0:
        return ids.astype(np.int64)
    keys = np.fromiter(id_index.keys(), dtype=np.int64, count=len(id_index))
    vals = np.fromiter(id_index.values(), dtype=np.int64, count=len(id_index))
    lo, hi = int(min(keys.min(), ids.min())), int(max(keys.max(), ids.max()))
    if lo >= 0 and hi < 10 * len(keys) + 1_000_000:
        table = np.full(hi + 1, -1, dtype=np.int64)
        table[keys] = vals
        out = table[ids]
    else:
        out = np.array([id_index.get(int(v), -1) for v in ids.ravel()], dtype=np.int64).reshape(ids.shape)
    if (out < 0).any():
        bad = ids[out < 0].ravel()[0]
        raise GocadError(f"{path}: object {name!r} refers to vertex id {bad} that is not defined")
    return out


def _boundary_faces(tets: np.ndarray) -> np.ndarray:
    """Triangles used by exactly one tetrahedron (oriented outwards for positive tetrahedra)."""
    a, b, c, d = tets.T
    faces = np.concatenate([np.column_stack(f) for f in ((b, c, d), (a, d, c), (a, b, d), (a, c, b))])
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


def _chain(segs: list[tuple[int, int]]) -> list[list[int]]:
    """Join SEG records into polylines when one segment starts where the previous one ends."""
    out: list[list[int]] = []
    for a, b in segs:
        if out and out[-1][-1] == a:
            out[-1].append(b)
        else:
            out.append([a, b])
    return out


def is_closed_mesh(triangles: np.ndarray) -> bool:
    """True when every edge is shared by exactly two triangles (a watertight surface)."""
    if triangles is None or len(triangles) < 4:
        return False
    t = np.asarray(triangles)
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


# --------------------------------------------------------------------------- objects


def _parse_object(lines: list[str], i: int, kind: str, path: Path) -> tuple[GocadObject | None, int]:
    """Parse one object starting after its ``GOCAD <kind>`` line; return (object, next line)."""
    n = len(lines)
    known = kind in GOCAD_KINDS
    header: dict[str, str] = {}
    info: dict = {}
    xyz: list[tuple[float, float, float]] = []
    vprops: list[list[float]] = []
    id_index: dict[int, int] = {}
    tris: list[tuple[int, int, int]] = []
    tets: list[tuple[int, int, int, int]] = []
    lines_segs: list[list[tuple[int, int]]] = []
    prop_names: list[str] = []
    esizes: list[int] = []
    nodata: list[float] = []
    zsign = 1.0
    in_header = in_crs = False
    while i < n:
        s = lines[i].strip()
        i += 1
        if not s or s[0] == "#":
            continue
        if in_header:
            body = s
            if "}" in body:
                body = body.split("}", 1)[0]
                in_header = False
            k, sep, v = body.partition(":")
            if sep:
                header[k.strip()] = v.strip().strip('"')
            continue
        key, _, rest = s.partition(" ")
        if key == "END":
            break
        if not known:
            continue
        if key in ("VRTX", "PVRTX"):
            p = rest.split()
            id_index[int(p[0])] = len(xyz)
            xyz.append((float(p[1]), float(p[2]), float(p[3])))
            if len(p) > 4:
                vprops.append([float(v) for v in p[4:]])
            elif prop_names:
                vprops.append([])
        elif key == "TRGL":
            p = rest.split()
            tris.append((int(p[0]), int(p[1]), int(p[2])))
        elif key in ("ATOM", "PATOM"):
            p = rest.split()
            ref = int(p[1])
            if ref not in id_index:
                raise GocadError(f"{path}: ATOM {p[0]} refers to undefined vertex {ref}")
            id_index[int(p[0])] = id_index[ref]
        elif key == "SEG":
            p = rest.split()
            if not lines_segs:
                lines_segs.append([])
            lines_segs[-1].append((int(p[0]), int(p[1])))
        elif key == "ILINE":
            lines_segs.append([])
        elif key == "TETRA":
            p = rest.split()
            tets.append((int(p[0]), int(p[1]), int(p[2]), int(p[3])))
        elif key == "HEADER":
            body = rest.strip()
            if body.startswith("{"):
                body = body[1:]
                in_header = "}" not in body
                body = body.split("}", 1)[0]
                k, sep, v = body.partition(":")
                if sep:
                    header[k.strip()] = v.strip().strip('"')
        elif key == "HDR":
            k, sep, v = rest.partition(":")
            if sep:
                header[k.strip()] = v.strip().strip('"')
        elif key == "GOCAD_ORIGINAL_COORDINATE_SYSTEM":
            in_crs = True
            info.setdefault("crs", {})
        elif key == "END_ORIGINAL_COORDINATE_SYSTEM":
            in_crs = False
        elif key == "ZPOSITIVE":
            zsign = -1.0 if rest.strip().upper().startswith("DEPTH") else 1.0
            info["zpositive"] = rest.strip()
        elif in_crs:
            info["crs"][key] = " ".join(_tokens(rest))
        elif key == "PROPERTIES":
            prop_names = _tokens(rest)
        elif key == "ESIZES":
            esizes = [int(v) for v in rest.split()]
        elif key == "NO_DATA_VALUES":
            nodata = [float(v) for v in rest.split()]
        elif key in ("GEOLOGICAL_FEATURE", "GEOLOGICAL_TYPE", "STRATIGRAPHIC_POSITION"):
            info[key.lower()] = rest.strip().strip('"')
        # TFACE, TVOLUME, BSTONE, BORDER, PROPERTY_CLASSES, UNITS, ... are ignored.

    if not known:
        return None, i

    name = header.get("name", "")
    verts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    verts[:, 2] *= zsign

    props: dict[str, np.ndarray] = {}
    if prop_names and vprops:
        sizes = esizes if len(esizes) == len(prop_names) else [1] * len(prop_names)
        width = sum(sizes)
        table = np.full((len(vprops), width), np.nan)
        for r, row in enumerate(vprops):
            m = min(len(row), width)
            table[r, :m] = row[:m]
        col = 0
        for pi, (pname, es) in enumerate(zip(prop_names, sizes)):
            nd = nodata[pi] if pi < len(nodata) else None
            for e in range(es):
                a = table[:, col]
                if nd is not None:
                    a = np.where(a == nd, np.nan, a)
                props[pname if es == 1 else f"{pname}_{e}"] = a
                col += 1

    obj = GocadObject(kind=kind, name=name, color=_header_color(header, kind), vertices=verts,
                      vertex_props=props, header=header, info=info)
    if kind == "TSurf":
        obj.triangles = _remap(np.asarray(tris, dtype=np.int64).reshape(-1, 3), id_index, path, name)
    elif kind == "TSolid":
        t4 = _remap(np.asarray(tets, dtype=np.int64).reshape(-1, 4), id_index, path, name)
        info["n_tetra"] = len(t4)
        if len(t4):
            obj.triangles = _boundary_faces(t4)
        elif tris:          # some exporters write the boundary as TRGL
            obj.triangles = _remap(np.asarray(tris, dtype=np.int64).reshape(-1, 3), id_index, path, name)
        else:
            raise GocadError(f"{path}: TSolid {name!r} has no TETRA records")
    elif kind == "PLine":
        polys = []
        for segs in lines_segs:
            if not segs:
                continue
            for chain in _chain(segs):
                polys.append(_remap(np.asarray(chain, dtype=np.int64), id_index, path, name))
        obj.segments = polys
    return obj, i


def read_gocad(path: str | Path) -> list[GocadObject]:
    """Read every TSurf / PLine / VSet / TSolid object of a GOCAD ASCII file.

    Raises GocadError when the file holds no GOCAD object, or only kinds that are not
    supported here (for voxets use :func:`read_voxet`).
    """
    path = Path(path)
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        lines = f.read().splitlines()
    objects: list[GocadObject] = []
    skipped: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        i += 1
        if not s.upper().startswith("GOCAD "):
            continue
        parts = s.split()
        kind = parts[1] if len(parts) > 1 else ""
        canon = _KIND_ALIASES.get(kind.lower(), kind)
        obj, i = _parse_object(lines, i, canon, path)
        if obj is None:
            skipped.append(kind)
        else:
            objects.append(obj)
    if not objects:
        if skipped:
            raise GocadError(f"{path}: no TSurf/PLine/VSet/TSolid object (found {', '.join(skipped)}; "
                             "use read_voxet / read_sgrid for grids)")
        raise GocadError(f"{path}: not a GOCAD ASCII file (no 'GOCAD <type>' line)")
    return objects


def _names_option(names: dict[int, str]) -> str:
    return json.dumps({str(k): v for k, v in names.items()}, ensure_ascii=False)


def gocad_to_dtm(objects: list[GocadObject], path: str | Path | None = None) -> DtmFile | None:
    """Put every TSurf / TSolid into a DtmFile: one DtmObject (id 1..n) with one Trisolation each.

    Object names go to ``DtmFile.options["object_names"]`` (JSON ``{"1": "name", ...}``, read
    it with ``surpac_dtm.object_names``) and to each trisolation's ``options["name"]``; the
    colour (if any) to ``options["color"]`` as ``"r,g,b"``. ``closed`` is "yes" for watertight
    surfaces and TSolids. Returns None when there is no triangle.
    """
    surfs = [o for o in objects if o.kind in ("TSurf", "TSolid") and o.n_triangles]
    if not surfs:
        return None
    verts, objs, names = [], [], {}
    off = 0
    for k, o in enumerate(surfs, start=1):
        m = len(o.triangles)
        opts = {"closed": "yes" if o.kind == "TSolid" or is_closed_mesh(o.triangles) else "no",
                "name": o.name, "kind": o.kind}
        if o.color is not None:
            opts["color"] = ",".join(f"{c:.4f}" for c in o.color)
        objs.append(DtmObject(k, [Trisolation(1, o.triangles.astype(np.int64) + off, None,
                                              np.arange(1, m + 1, dtype=np.int64), opts)]))
        names[k] = o.name
        verts.append(o.vertices)
        off += len(o.vertices)
    return DtmFile(path=Path(path) if path is not None else Path(""), str_name="", checksum="",
                   options={"object_names": _names_option(names)}, vertices=np.vstack(verts), objects=objs)


def gocad_to_strings(objects: list[GocadObject], path: str | Path | None = None,
                     include_points: bool = True) -> StringFile | None:
    """Put PLines (and VSet points, as single-point segments) into a StringFile.

    Each object gets its own string number 1..n in file order; ``extra["names"]`` maps the
    string number to the object name and ``extra["colors"]`` to its RGB (when given).
    Returns None when there is nothing to put in.
    """
    items: list[tuple[GocadObject, list[np.ndarray]]] = []
    for o in objects:
        if o.kind == "PLine" and o.segments:
            items.append((o, [o.vertices[s] for s in o.segments if len(s)]))
        elif o.kind == "VSet" and include_points and len(o.vertices):
            items.append((o, [o.vertices[i:i + 1] for i in range(len(o.vertices))]))
    if not items:
        return None
    sns: list[np.ndarray] = []
    xyz: list[np.ndarray] = []
    seg_sn, starts, ends = [], [], []
    names, colors = {}, {}
    pos = 0
    sep = np.full((1, 3), np.nan)
    for sn, (o, polys) in enumerate(items, start=1):
        names[sn] = o.name
        if o.color is not None:
            colors[sn] = o.color
        for p in polys:
            starts.append(pos)
            pos += len(p)
            ends.append(pos)
            seg_sn.append(sn)
            xyz.extend([p, sep])
            sns.append(np.concatenate([np.full(len(p), sn, dtype=np.int32), np.zeros(1, dtype=np.int32)]))
            pos += 1
    rec_sn = np.concatenate(sns)
    p = Path(path) if path is not None else Path("")
    return StringFile(
        path=p,
        header=StrHeader(name=p.stem),
        records_xyz=np.vstack(xyz).astype(np.float64),
        records_string=rec_sn,
        records_desc=[""] * len(rec_sn),
        seg_string=np.asarray(seg_sn, dtype=np.int32),
        seg_start=np.asarray(starts, dtype=np.int64),
        seg_end=np.asarray(ends, dtype=np.int64),
        extra={"names": names, "colors": colors},
    )


# --------------------------------------------------------------------------- voxets


@dataclass
class Voxet:
    """A GOCAD voxet. Values are at nodes; see the module docstring for node positions."""
    path: Path | None
    name: str
    origin: np.ndarray                     # (3,) AXIS_O
    u: np.ndarray                          # (3,) AXIS_U, as given
    v: np.ndarray                          # (3,) AXIS_V
    w: np.ndarray                          # (3,) AXIS_W
    n: tuple[int, int, int]                # AXIS_N: nodes along u, v, w
    axis_min: np.ndarray                   # (3,) AXIS_MIN (default 0 0 0)
    axis_max: np.ndarray                   # (3,) AXIS_MAX (default 1 1 1)
    properties: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (nw, nv, nu), u fastest
    no_data: dict[str, float | None] = field(default_factory=dict)   # name -> PROP_NO_DATA_VALUE
    units: dict[str, str] = field(default_factory=dict)
    axis_type: tuple[str, ...] = ("even", "even", "even")
    info: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.n))

    def axis_params(self, axis: int) -> np.ndarray:
        """Parameter values (between AXIS_MIN and AXIS_MAX) of the nodes along one axis."""
        nn = self.n[axis]
        lo, hi = float(self.axis_min[axis]), float(self.axis_max[axis])
        return np.full(1, lo) if nn == 1 else lo + (hi - lo) * np.arange(nn) / (nn - 1)

    def nodes(self) -> np.ndarray:
        """(nu*nv*nw, 3) node coordinates in file order (u fastest)."""
        tu, tv, tw = (self.axis_params(a) for a in range(3))
        W, V, U = np.meshgrid(tw, tv, tu, indexing="ij")
        return (self.origin + U.reshape(-1, 1) * self.u + V.reshape(-1, 1) * self.v
                + W.reshape(-1, 1) * self.w)

    def valid_mask(self, name: str) -> np.ndarray:
        a = self.properties[name]
        m = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, dtype=bool)
        nd = self.no_data.get(name)
        if nd is not None:
            m &= a != nd
        return m


def _find_file(base: Path, name: str) -> Path:
    name = name.strip().strip('"')
    cands = [Path(name), base / name, base / Path(name).name]
    for c in cands:
        if c.exists():
            return c
    low = Path(name).name.lower()
    if base.is_dir():
        for q in base.iterdir():
            if q.name.lower() == low:
                return q
    raise GocadError(f"Data file {name!r} not found next to the header in {base}")


def _binary_dtype(etype: str, esize: int, signed: bool | None, byte_order: str) -> np.dtype:
    etype = etype.strip().upper()
    order = ">" if byte_order.lower() in ("big", "msb", ">") else "<"
    if etype in ("IEEE", "FLOAT", "REAL"):
        if esize not in (4, 8):
            raise GocadError(f"Unsupported IEEE element size {esize}")
        return np.dtype(f"{order}f{esize}")
    if etype in ("OCTET", "INTEGER", "INT", "BYTE", "CHAR"):
        if esize not in (1, 2, 4, 8):
            raise GocadError(f"Unsupported integer element size {esize}")
        if signed is None:
            signed = esize > 1
        return np.dtype(f"{order}{'i' if signed else 'u'}{esize}")
    raise GocadError(f"Unsupported PROP_ETYPE {etype!r} (IEEE and Octet are supported)")


def _parse_grid_header(path: Path, expect: tuple[str, ...]):
    """Split a voxet / SGrid header into (kind, header dict, scalar keys, per-property dict)."""
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        lines = f.read().splitlines()
    kind = ""
    header: dict[str, str] = {}
    keys: dict[str, list[str]] = {}
    props: dict[int, dict[str, str]] = {}
    in_header = False
    zsign = 1.0
    in_block = False
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if in_block:
            in_block = "}" not in s
            continue
        if in_header:
            body = s.split("}", 1)[0]
            in_header = "}" not in s
            k, sep, v = body.partition(":")
            if sep:
                header[k.strip()] = v.strip().strip('"')
            continue
        tok = _tokens(s)
        key = tok[0].upper()
        if key == "GOCAD" and len(tok) > 1:
            kind = tok[1]
            continue
        if key == "HEADER":
            body = s.split("{", 1)[1] if "{" in s else ""
            in_header = "}" not in body
            k, sep, v = body.split("}", 1)[0].partition(":")
            if sep:
                header[k.strip()] = v.strip().strip('"')
            continue
        if key == "END":
            break
        if key == "ZPOSITIVE":
            zsign = -1.0 if len(tok) > 1 and tok[1].upper().startswith("DEPTH") else 1.0
            continue
        if key.startswith("PROP") and key not in ("PROPERTY_CLASS_HEADER",) and len(tok) > 2 \
                and tok[1].isdigit():
            props.setdefault(int(tok[1]), {})[key] = " ".join(tok[2:])
            continue
        if "{" in s and "}" not in s:          # PROPERTY_CLASS_HEADER n "name" { ... }: display settings
            in_block = True
            continue
        if key == "PROPERTY_CLASS_HEADER":
            continue
        keys[key] = tok[1:]
    if kind.lower() not in expect:
        raise GocadError(f"{path}: not a GOCAD {'/'.join(expect)} header (found {kind or 'nothing'!r})")
    return kind, header, keys, props, zsign


def _vec(keys: dict[str, list[str]], key: str, default) -> np.ndarray:
    if key in keys and len(keys[key]) >= 3:
        return np.array([float(v) for v in keys[key][:3]], dtype=np.float64)
    return np.asarray(default, dtype=np.float64)


def _read_prop_file(base: Path, p: dict[str, str], count: int, byte_order: str) -> np.ndarray:
    fmt = p.get("PROP_FORMAT", "RAW").upper()
    if fmt != "RAW":
        raise GocadError(f"PROP_FORMAT {fmt} is not supported (only RAW)")
    etype = p.get("PROP_ETYPE", "IEEE")
    esize = int(float(p.get("PROP_ESIZE", "4")))
    signed = None
    if "PROP_SIGNED" in p:
        signed = p["PROP_SIGNED"].strip() not in ("0", "false", "False")
    dt = _binary_dtype(etype, esize, signed, byte_order)
    offset = int(float(p.get("PROP_OFFSET", "0")))
    f = _find_file(base, p["PROP_FILE"])
    a = np.fromfile(f, dtype=dt, count=count, offset=offset)
    if len(a) < count:
        raise GocadError(f"{f}: holds {len(a)} values, the header needs {count}")
    return a.astype(dt.newbyteorder("="))


def read_voxet(path: str | Path, byte_order: str = "big") -> Voxet:
    """Read a GOCAD voxet header (.vo) and its binary property files.

    ``byte_order`` is "big" (GOCAD's default) or "little" (some third-party writers).
    ZPOSITIVE Depth negates the z components of AXIS_O / U / V / W.
    """
    path = Path(path)
    _, header, keys, props, zsign = _parse_grid_header(path, ("voxet",))
    n = tuple(int(float(v)) for v in keys.get("AXIS_N", [])[:3])
    if len(n) != 3:
        raise GocadError(f"{path}: AXIS_N is missing")
    flip = np.array([1.0, 1.0, zsign])
    vox = Voxet(
        path=path,
        name=header.get("name", path.stem),
        origin=_vec(keys, "AXIS_O", (0, 0, 0)) * flip,
        u=_vec(keys, "AXIS_U", (1, 0, 0)) * flip,
        v=_vec(keys, "AXIS_V", (0, 1, 0)) * flip,
        w=_vec(keys, "AXIS_W", (0, 0, 1)) * flip,
        n=n,  # type: ignore[arg-type]
        axis_min=_vec(keys, "AXIS_MIN", (0, 0, 0)),
        axis_max=_vec(keys, "AXIS_MAX", (1, 1, 1)),
        axis_type=tuple(keys.get("AXIS_TYPE", ["even"] * 3)),
        info={"header": header, "axis_name": keys.get("AXIS_NAME"), "axis_unit": keys.get("AXIS_UNIT")},
    )
    if any(t.lower() not in ("even", "regular") for t in vox.axis_type):
        vox.info["warning"] = "uneven axes are read as even"
    count = vox.n_nodes
    for idx in sorted(props):
        p = props[idx]
        name = p.get("PROPERTY", f"property_{idx}")
        if "PROP_FILE" not in p:
            continue
        vox.properties[name] = _read_prop_file(path.parent, p, count, byte_order).reshape(n[2], n[1], n[0])
        nd = p.get("PROP_NO_DATA_VALUE")
        vox.no_data[name] = float(nd) if nd not in (None, "") else None
        if "PROP_UNIT" in p:
            vox.units[name] = p["PROP_UNIT"]
    return vox


def _axis_alignment(vox: Voxet) -> list[int]:
    perm = []
    for name, a in (("U", vox.u), ("V", vox.v), ("W", vox.w)):
        norm = float(np.linalg.norm(a))
        nz = np.abs(a) > 1e-9 * max(norm, 1e-300)
        if norm == 0 or nz.sum() != 1:
            raise ValueError(
                f"Voxet {vox.name!r}: axis {name} = {a.tolist()} is not aligned with X, Y or Z. "
                "Rotated voxets are not supported yet; export the voxet axis-aligned.")
        perm.append(int(np.argmax(nz)))
    if sorted(perm) != [0, 1, 2]:
        raise ValueError(f"Voxet {vox.name!r}: axes U, V, W do not map to distinct X/Y/Z axes")
    return perm


def voxet_to_block_model(vox: Voxet, properties: list[str] | None = None) -> BlockModel:
    """One block per voxet node (centroid = node), dropping nodes where all properties are no data.

    Block size per axis = axis length / (n - 1) (for n == 1: the axis length, or the
    length of the axis vector when AXIS_MIN == AXIS_MAX). No-data values become the
    block model null (-99). Axes must be parallel to X / Y / Z (either direction);
    rotated voxets raise ValueError.
    """
    perm = _axis_alignment(vox)
    names = list(vox.properties) if properties is None else list(properties)
    for nm in names:
        if nm not in vox.properties:
            raise KeyError(f"Voxet {vox.name!r} has no property {nm!r} (has {list(vox.properties)})")
    shape = (vox.n[2], vox.n[1], vox.n[0])
    if names:
        keep = np.zeros(shape, dtype=bool)
        for nm in names:
            keep |= vox.valid_mask(nm)
        flat = np.flatnonzero(keep.ravel())
    else:
        flat = np.arange(vox.n_nodes)
    k, j, i = np.unravel_index(flat, shape)
    axes = (vox.u, vox.v, vox.w)
    cen = np.tile(vox.origin, (len(flat), 1)).astype(np.float64)
    size = np.zeros(3)
    for ax, idx in enumerate((i, j, k)):
        comp = perm[ax]
        a = float(axes[ax][comp])
        t = vox.axis_params(ax)
        cen[:, comp] += t[idx] * a
        nn = vox.n[ax]
        span = abs(float(vox.axis_max[ax] - vox.axis_min[ax])) * abs(a)
        size[comp] = span / (nn - 1) if nn > 1 else (span or abs(a))
    attrs: dict[str, np.ndarray] = {}
    for nm in names:
        a = vox.properties[nm].ravel()[flat].astype(np.float32)
        valid = vox.valid_mask(nm).ravel()[flat]
        a[~valid] = np.float32(SURPAC_NULL)
        attrs[nm] = a
    sizes = np.tile(size.astype(np.float32), (len(flat), 1))
    return BlockModel(path=vox.path, centroids=cen, sizes=sizes, attributes=attrs, null_value=SURPAC_NULL)


# --------------------------------------------------------------------------- SGrids




@dataclass
class SGrid:
    """A GOCAD stratigraphic grid (best effort): geometry plus properties, u fastest.

    ``n`` counts NODES along u, v, w. Node properties are shaped (nw, nv, nu); cell
    properties (PROP_ALIGNMENT CELLS) are shaped (nw-1, nv-1, nu-1). ``points`` holds the
    node coordinates (nw, nv, nu, 3) when the file gives them (POINTS_FILE, or an ASCII
    file of nodes); ``centres`` the cell centres (nw-1, nv-1, nu-1, 3) when the ASCII file
    lists cells. Values equal to ``no_data`` are kept as they are (see ``valid_mask``).
    """
    path: Path | None
    name: str
    n: tuple[int, int, int]
    points: np.ndarray | None = None
    centres: np.ndarray | None = None
    properties: dict[str, np.ndarray] = field(default_factory=dict)
    no_data: dict[str, float | None] = field(default_factory=dict)
    cell_centred: dict[str, bool] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)

    @property
    def cell_shape(self) -> tuple[int, int, int]:
        nu, nv, nw = self.n
        return (max(nw - 1, 1), max(nv - 1, 1), max(nu - 1, 1))

    def valid_mask(self, name: str) -> np.ndarray:
        a = self.properties[name]
        m = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, dtype=bool)
        nd = self.no_data.get(name)
        if nd is not None:
            m &= a != nd
        return m

    def to_points(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(xyz (N, 3), values (N,)) of the valid values of one property, at cell centres or nodes."""
        a = self.properties[name]
        if self.cell_centred.get(name):
            xyz = self.centres
            if xyz is None and self.points is not None:     # centre = mean of the 8 corner nodes
                p = self.points
                xyz = sum(p[a0:a0 + p.shape[0] - 1, b0:b0 + p.shape[1] - 1, c0:c0 + p.shape[2] - 1]
                          for a0 in (0, 1) for b0 in (0, 1) for c0 in (0, 1)) / 8.0
        else:
            xyz = self.points
        if xyz is None:
            raise ValueError(f"SGrid {self.name!r}: no coordinates for property {name!r}")
        m = self.valid_mask(name)
        return xyz[m].reshape(-1, 3), a[m].ravel()


def read_sgrid(path: str | Path, byte_order: str = "big") -> SGrid:
    """Read a GOCAD SGrid header (.sg) with an ASCII_DATA_FILE, or POINTS_FILE + PROP_FILEs.

    ASCII data files list ``x y z prop1 .. propn [i j k]`` per node (with PROP_ALIGNMENT
    CELLS, the value of cell (i, j, k) is on the row of its first node (i, j, k) and the
    last node along each axis carries no data), or one row per cell at its centre. Limits: FLAGS_FILE (dead
    cells), regions and split nodes are ignored; binary point files are float32 xyz.
    """
    path = Path(path)
    _, header, keys, props, zsign = _parse_grid_header(path, ("sgrid",))
    n = tuple(int(float(v)) for v in keys.get("AXIS_N", [])[:3])
    if len(n) != 3:
        raise GocadError(f"{path}: AXIS_N is missing")
    nu, nv, nw = n
    count = nu * nv * nw
    sg = SGrid(path=path, name=header.get("name", path.stem), n=n)  # type: ignore[arg-type]
    cshape = sg.cell_shape
    ncell = int(np.prod(cshape))
    default_cells = " ".join(keys.get("PROP_ALIGNMENT", ["POINTS"])).upper().startswith("CELL")
    order = sorted(props)
    for idx in order:
        p = props[idx]
        nm = p.get("PROPERTY", f"property_{idx}")
        nd = p.get("PROP_NO_DATA_VALUE")
        sg.no_data[nm] = float(nd) if nd not in (None, "") else None
        if "PROP_UNIT" in p:
            sg.units[nm] = p["PROP_UNIT"]
    if "ASCII_DATA_FILE" in keys:
        f = _find_file(path.parent, " ".join(keys["ASCII_DATA_FILE"]))
        names = [props[i].get("PROPERTY", f"property_{i}") for i in order]
        rows = []
        with open(f, "r", encoding="latin-1", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s or s[0] in "*#":
                    continue
                try:
                    rows.append([float(v) for v in s.split()])
                except ValueError:
                    continue
        width = min(len(r) for r in rows) if rows else 0
        if width < 3 + len(names):
            raise GocadError(f"{f}: rows have {width} values, expected x y z and {len(names)} properties")
        data = np.array([r[:width] for r in rows], dtype=np.float64)
        if len(data) == count:                   # one row per node (GOCAD's usual layout)
            row_cells, shape = False, (nw, nv, nu)
        elif len(data) == ncell:                 # one row per cell, at its centre
            row_cells, shape = True, cshape
        else:
            raise GocadError(f"{f}: {len(data)} rows, the header needs {count} nodes or {ncell} cells")
        if width >= 3 + len(names) + 3:          # trailing I J K: put the rows in u-fastest order
            ii, jj, kk = data[:, -3:].astype(np.int64).T
            data = data[np.lexsort((ii, jj, kk))]
        xyz = data[:, :3].reshape(*shape, 3).copy()
        xyz[..., 2] *= zsign
        if row_cells:
            sg.centres = xyz
        else:
            sg.points = xyz
        cells = default_cells or row_cells
        for c, nm in enumerate(names):
            a = data[:, 3 + c].reshape(shape)
            if cells and not row_cells:          # cell values written on each cell's first node
                a = a[:cshape[0], :cshape[1], :cshape[2]]
            sg.properties[nm] = a.copy()
            sg.cell_centred[nm] = cells
    elif "POINTS_FILE" in keys:
        f = _find_file(path.parent, " ".join(keys["POINTS_FILE"]))
        off = int(float(keys.get("POINTS_OFFSET", ["0"])[0]))
        dt = _binary_dtype("IEEE", 4, None, byte_order)
        pts = np.fromfile(f, dtype=dt, count=count * 3, offset=off)
        if len(pts) < count * 3:
            raise GocadError(f"{f}: holds {len(pts) // 3} points, the header needs {count}")
        sg.points = pts.astype(np.float64).reshape(nw, nv, nu, 3)
        sg.points[..., 2] *= zsign
        for idx in order:
            p = props[idx]
            if "PROP_FILE" not in p:
                continue
            nm = p.get("PROPERTY", f"property_{idx}")
            al = p.get("PROP_ALIGNMENT")
            cells = al.upper().startswith("CELL") if al else default_cells
            a = _read_prop_file(path.parent, p, ncell if cells else count, byte_order)
            sg.properties[nm] = a.reshape(cshape if cells else (nw, nv, nu))
            sg.cell_centred[nm] = cells
    else:
        raise GocadError(f"{path}: SGrid without ASCII_DATA_FILE or POINTS_FILE is not supported")
    return sg
