"""Minimal ASCII DXF reader for geometry: LINE/3DLINE, POLYLINE/VERTEX, LWPOLYLINE, 3DFACE,
polyface and polygon meshes (POLYLINE flags 64 / 16) and MESH.

Surpac writes old-style R10/R12 DXF (e.g. ``3DLINE`` entities) that modern
libraries skip, so we read the group codes directly. Line pieces whose ends
meet are chained into polylines. DXF layers become string numbers when they
are numeric, otherwise they are numbered in order of appearance.

Triangles (3DFACE, polyface / polygon meshes, MESH) go into one DtmFile with one
DtmObject per layer; the object id is the layer's number (as for strings) and the
layer names are kept in ``options["object_names"]`` (see ``surpac_dtm.object_names``).
A file whose faces are all on one layer gives a single object, as before.

* Polyface mesh (POLYLINE flag 64): VERTEX records with flag 128|64 are positions;
  VERTEX records with flag 128 alone are faces, whose groups 71..74 are 1-based vertex
  indices (negative = invisible edge, still a vertex; 0 / missing = no vertex).
* Polygon mesh (POLYLINE flag 16): M x N vertices (groups 71 / 72, or 73 / 74 for a
  smoothed surface) in row order; flag 1 closes it in M, flag 32 in N.
* MESH (AutoCAD 2010+): group 92 vertex count then 10/20/30 vertices, group 93 face-list
  size then 90 entries ``n, i1 .. in`` (0-based). Faces with more than three vertices
  are fanned into triangles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .surpac_dtm import DtmFile, DtmObject, Trisolation
from .surpac_str import StrHeader, StringFile


def _pairs(path: Path):
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        try:
            code = int(lines[i].strip())
        except ValueError:
            continue
        yield code, lines[i + 1].strip()


def _entities(path: Path):
    """Yield (type, [(code, value), ...]) for entities in the ENTITIES section."""
    in_entities = False
    cur_type, cur = None, []
    for code, value in _pairs(path):
        if code == 0:
            if cur_type is not None and in_entities:
                yield cur_type, cur
            cur_type, cur = value.upper(), []
            if value.upper() == "ENDSEC":
                in_entities = False
                cur_type = None
            continue
        if code == 2 and cur_type == "SECTION":
            in_entities = value.upper() == "ENTITIES"
            cur_type = None
            continue
        if cur_type is not None:
            cur.append((code, value))


def _get(groups, code, default=0.0):
    for c, v in groups:
        if c == code:
            return float(v)
    return default


def _layer(groups) -> str:
    for c, v in groups:
        if c == 8:
            return v
    return "0"


def _chain(segments: list[tuple[np.ndarray, np.ndarray]], tol=1e-6) -> list[np.ndarray]:
    """Join consecutive line pieces whose end equals the next start."""
    out: list[list[np.ndarray]] = []
    for a, b in segments:
        if out and np.allclose(out[-1][-1], a, atol=tol):
            out[-1].append(b)
        else:
            out.append([a, b])
    return [np.array(p) for p in out]


_EMPTY_MESH = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))


def _fan(face: list[int]) -> list[tuple[int, int, int]]:
    return [(face[0], face[k], face[k + 1]) for k in range(1, len(face) - 1)]


def _polyface(verts: list[tuple[int, tuple, list[int]]]) -> tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of a polyface mesh from its VERTEX records (flag, xyz, [71..74])."""
    pts = [xyz for flag, xyz, _ in verts if flag & 64]
    tris: list[tuple[int, int, int]] = []
    for flag, _, idx in verts:
        if flag & 128 and not flag & 64:
            face = [abs(i) - 1 for i in idx if i != 0]
            face = [f for k, f in enumerate(face) if f not in face[:k]]
            if len(face) >= 3 and max(face) < len(pts):
                tris.extend(_fan(face))
    return (np.asarray(pts, dtype=np.float64).reshape(-1, 3),
            np.asarray(tris, dtype=np.int64).reshape(-1, 3))


def _polygon_mesh(verts: list[tuple[int, tuple, list[int]]], m: int, n: int, mm: int, nn: int,
                  flags: int) -> tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of an M x N polygon mesh (vertices in row order, N per row)."""
    pts = np.asarray([xyz for flag, xyz, _ in verts if not flag & 16], dtype=np.float64).reshape(-1, 3)
    if len(pts) != m * n and mm > 1 and nn > 1 and len(pts) == mm * nn:
        m, n = mm, nn                      # smoothed surface: the fitted vertices
    if m < 2 or n < 2 or len(pts) != m * n:
        return _EMPTY_MESH
    rows = m if flags & 1 else m - 1
    cols = n if flags & 32 else n - 1
    tris = []
    for a in range(rows):
        a1 = (a + 1) % m
        for b in range(cols):
            b1 = (b + 1) % n
            p00, p01, p10, p11 = a * n + b, a * n + b1, a1 * n + b, a1 * n + b1
            tris.append((p00, p01, p11))
            tris.append((p00, p11, p10))
    return pts, np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _mesh_entity(g) -> tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of a MESH entity (group codes in file order)."""
    k, ng = 0, len(g)
    while k < ng and g[k][0] != 92:
        k += 1
    if k == ng:
        return _EMPTY_MESH
    nv = int(float(g[k][1]))
    k += 1
    pts: list[tuple[float, float, float]] = []
    while k < ng and len(pts) < nv:
        if g[k][0] == 10 and k + 2 < ng and g[k + 1][0] == 20 and g[k + 2][0] == 30:
            pts.append((float(g[k][1]), float(g[k + 1][1]), float(g[k + 2][1])))
            k += 3
        else:
            k += 1
    while k < ng and g[k][0] != 93:
        k += 1
    tris: list[tuple[int, int, int]] = []
    if k < ng:
        size = int(float(g[k][1]))
        vals = [int(float(v)) for c, v in g[k + 1:k + 1 + size] if c == 90]
        j = 0
        while j < len(vals):
            cnt = vals[j]
            face = vals[j + 1:j + 1 + cnt]
            j += 1 + cnt
            if len(face) >= 3 and min(face) >= 0 and max(face) < len(pts):
                tris.extend(_fan(face))
    return (np.asarray(pts, dtype=np.float64).reshape(-1, 3),
            np.asarray(tris, dtype=np.int64).reshape(-1, 3))


WELD_TOL = 1e-3          # m: DXF corners closer than this are the same vertex


def _watertight(tris: np.ndarray) -> bool:
    if len(tris) < 4:
        return False
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(c == 2))


def read_dxf(path: str | Path) -> tuple[StringFile | None, DtmFile | None]:
    """Read lines / polylines as a StringFile and triangles as a DtmFile (either may be None)."""
    path = Path(path)
    layer_ids: dict[str, int] = {}

    def layer_no(name: str) -> int:
        if name not in layer_ids:
            try:
                n = int(float(name))
            except ValueError:
                n = 0
            if n <= 0:                            # named layer: next number, skipping ones in use
                n = len(layer_ids) + 1
                used = set(layer_ids.values())
                while n in used:
                    n += 1
            layer_ids[name] = n
        return layer_ids[name]

    polylines: list[tuple[int, np.ndarray]] = []
    line_pieces: dict[int, list] = {}
    faces: dict[int, list[np.ndarray]] = {}                       # layer -> 3DFACE triangles (3, 3)
    meshes: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}   # layer -> [(points, triangles)]

    def add_mesh(layer: int, mesh: tuple[np.ndarray, np.ndarray]) -> None:
        if len(mesh[1]):
            meshes.setdefault(layer, []).append(mesh)

    poly_layer, poly_pts, poly_closed = None, None, False
    poly_flags, poly_mn = 0, (0, 0, 0, 0)
    for etype, g in _entities(path):
        if etype in ("LINE", "3DLINE"):
            a = np.array([_get(g, 10), _get(g, 20), _get(g, 30)])
            b = np.array([_get(g, 11), _get(g, 21), _get(g, 31)])
            line_pieces.setdefault(layer_no(_layer(g)), []).append((a, b))
        elif etype == "POLYLINE":
            poly_layer, poly_pts = layer_no(_layer(g)), []
            poly_flags = int(_get(g, 70, 0))
            poly_closed = poly_flags & 1 == 1
            poly_mn = tuple(int(_get(g, c, 0)) for c in (71, 72, 73, 74))
        elif etype == "VERTEX" and poly_pts is not None:
            if poly_flags & (64 | 16):
                poly_pts.append((int(_get(g, 70, 0)), (_get(g, 10), _get(g, 20), _get(g, 30)),
                                 [int(_get(g, c, 0)) for c in (71, 72, 73, 74)]))
            else:
                poly_pts.append((_get(g, 10), _get(g, 20), _get(g, 30)))
        elif etype == "SEQEND" and poly_pts is not None:
            if poly_flags & 64:
                add_mesh(poly_layer, _polyface(poly_pts))
            elif poly_flags & 16:
                add_mesh(poly_layer, _polygon_mesh(poly_pts, *poly_mn, poly_flags))
            elif len(poly_pts) >= 2:
                p = np.array(poly_pts)
                if poly_closed:
                    p = np.vstack([p, p[:1]])
                polylines.append((poly_layer, p))
            poly_pts = None
        elif etype == "LWPOLYLINE":
            z = _get(g, 38, 0.0)
            xs = [float(v) for c, v in g if c == 10]
            ys = [float(v) for c, v in g if c == 20]
            if len(xs) >= 2:
                p = np.column_stack([xs, ys, np.full(len(xs), z)])
                if int(_get(g, 70, 0)) & 1:
                    p = np.vstack([p, p[:1]])
                polylines.append((layer_no(_layer(g)), p))
        elif etype == "3DFACE":
            d = {}
            for c, v in g:                      # first value of each code, as _get
                d.setdefault(c, v)
            q = np.array([[float(d.get(10 + k, 0.0)), float(d.get(20 + k, 0.0)), float(d.get(30 + k, 0.0))]
                          for k in range(4)])
            if 13 not in d:                     # no fourth corner: a triangle
                q[3] = q[2]
            lay = faces.setdefault(layer_no(d.get(8, "0")), [])
            lay.append(q[:3])
            # a quad unless the fourth corner repeats the third or the first (both are written);
            # exact test: np.allclose's relative tolerance is metres at UTM northings
            if np.abs(q[2] - q[3]).max() > 1e-9 and np.abs(q[0] - q[3]).max() > 1e-9:
                lay.append(q[[0, 2, 3]])
        elif etype == "MESH":
            add_mesh(layer_no(_layer(g)), _mesh_entity(g))

    for lay, pieces in line_pieces.items():
        for p in _chain(pieces):
            polylines.append((lay, p))

    sf = None
    if polylines:
        sns, xyz, starts, ends = [], [], [], []
        for lay, p in polylines:
            starts.append(len(sns))
            sns.extend([lay] * len(p))
            xyz.extend(p.tolist())
            ends.append(len(sns))
            sns.append(0)               # separator record, as in a .str file
            xyz.append([np.nan] * 3)
        sf = StringFile(
            path=path,
            header=StrHeader(name=path.stem),
            records_xyz=np.array(xyz, dtype=np.float64),
            records_string=np.array(sns, dtype=np.int32),
            records_desc=[""] * len(sns),
            seg_string=np.array([lay for lay, _ in polylines], dtype=np.int32),
            seg_start=np.array(starts, dtype=np.int64),
            seg_end=np.array(ends, dtype=np.int64),
        )

    dtm = None
    if faces or meshes:
        names = {no: name for name, no in layer_ids.items()}
        verts: list[np.ndarray] = []
        objects: list[DtmObject] = []
        off = 0
        for lay in sorted(set(faces) | set(meshes)):
            parts = []
            if lay in faces:
                v = np.vstack(faces[lay])
                verts.append(v)
                parts.append(off + np.arange(len(v), dtype=np.int64).reshape(-1, 3))
                off += len(v)
            for pts, tris in meshes.get(lay, []):
                verts.append(pts)
                parts.append(tris + off)
                off += len(pts)
            tris = np.concatenate(parts)
            # weld: 3DFACE corners (and separate meshes) that coincide become one vertex, so a solid
            # exported as faces (3DMine, Surpac, GOCAD DXF export) is a closed mesh again
            first = off - sum(len(x) for x in verts[len(verts) - len(parts):])
            lv = np.vstack(verts[len(verts) - len(parts):])
            key = np.round(lv / WELD_TOL).astype(np.int64)
            _u, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            order = np.argsort(idx)                       # keep the vertices in their file order
            rank = np.empty(len(idx), dtype=np.int64)
            rank[order] = np.arange(len(idx))
            idx = idx[order]
            verts[len(verts) - len(parts):] = [lv[idx]]
            tris = rank[inv.ravel()][tris - first] + first
            off = first + len(idx)
            keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
            tris = tris[keep]
            closed = "yes" if _watertight(tris) else "no"
            objects.append(DtmObject(lay, [Trisolation(1, tris, None, np.arange(1, len(tris) + 1),
                                                       {"closed": closed, "name": names.get(lay, "")})]))
        dtm = DtmFile(
            path=path, str_name="", checksum="",
            options={"object_names": json.dumps({str(o.id): names.get(o.id, "") for o in objects},
                                                ensure_ascii=False)},
            vertices=np.vstack(verts), objects=objects,
        )
    return sf, dtm
