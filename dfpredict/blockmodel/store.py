"""The GeoViewer block model file (.gvbm): compressed and encrypted, so the block values
cannot be read (or copied into another program) without GeoViewer.

Layout::

    header (48 bytes, plain)   magic "GVBM", version, flags, salt, nonce prefix,
                               scrypt parameters, chunk size, total plain length
    chunk*                     u32 length + AES-256-GCM(zlib(plain piece)) with a 16-byte tag

The plain stream is ``u32 meta length + meta JSON + the arrays' raw bytes`` (meta lists
the arrays with their dtype and shape). It is cut into pieces of ``chunk size`` bytes; each
piece is compressed and sealed on its own, with the header, the piece number and a "last
piece" flag as associated data: a changed header, swapped, removed or cut-off pieces and any
changed byte make the file fail to open instead of giving wrong values.

The key is derived (HKDF-SHA256, per-file random salt) from a secret built into GeoViewer:
any GeoViewer opens the file, nothing else can read it. Honest limit: the secret is inside
the program, so a determined attacker who takes GeoViewer apart could recover it.

Blocks are stored on their grid when they align to one (integer cell index and size in
finest cells), else as centroid + size.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

from ..i18n import tr
from ..io.block_model import SURPAC_NULL, BlockModel

MAGIC = b"GVBM"
VERSION = 1
FORMAT = "geoviewer-blockmodel"
CHUNK = 4 << 20                       # plain bytes per sealed piece
_HEADER = struct.Struct("<4sHH16s4s4xIQ4x")      # 48 bytes
_TAG = 16
_INFO = b"geoviewer-blockmodel-v1"

# The built-in secret (split; see the module notes on what it protects against).
_S_A = "MRZZgk0q3JlovQv6GDN4Jg=="
_S_B = "Eu2H1wSOnbTbOkm7TFbklA=="


class BlockFileError(ValueError):
    """Not a GeoViewer block model, or a damaged one."""


# --- keys -----------------------------------------------------------------------------------
def _derive(salt: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    ikm = base64.b64decode(_S_A) + base64.b64decode(_S_B)
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=_INFO).derive(ikm)


def _nonce(prefix: bytes, counter: int) -> bytes:
    return prefix + struct.pack("<Q", counter)


def _aad(header: bytes, counter: int, last: bool) -> bytes:
    return header + struct.pack("<QB", counter, 1 if last else 0)


# --- what is stored -------------------------------------------------------------------------
def _arrays_of(bm: BlockModel) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """(layout meta, [(array name, array), ...]) for a block model."""
    arrays: list[tuple[str, np.ndarray]] = []
    try:
        g = bm.grid()
        layout = {"layout": "grid", "origin": [float(v) for v in g.origin], "cell": [float(v) for v in g.cell]}
        arrays.append(("ijk", np.ascontiguousarray(g.ijk_min, dtype="<i4")))
        if len(g.span) and np.all(g.span == g.span[0]):
            layout["span"] = [int(v) for v in g.span[0]]
        else:
            if g.span.max(initial=1) > 65535:
                raise ValueError("span")
            arrays.append(("span", np.ascontiguousarray(g.span, dtype="<u2")))
    except ValueError:
        layout = {"layout": "free"}
        arrays.append(("centroids", np.ascontiguousarray(bm.centroids, dtype="<f8")))
        arrays.append(("sizes", np.ascontiguousarray(bm.sizes, dtype="<f4")))
    for i, name in enumerate(bm.attributes):
        a = bm.attributes[name]
        dt = "<i4" if bm.is_categorical(name) else "<f4"
        arrays.append((f"attr_{i}", np.ascontiguousarray(a, dtype=dt)))
    return layout, arrays


def _meta(bm: BlockModel, layout: dict, arrays) -> dict:
    info = dict(bm.info or {})
    desc = dict(info.pop("descriptions", {}) or {})
    now = datetime.now().isoformat(timespec="seconds")
    from .. import __version__

    return {
        "format": FORMAT, "version": VERSION, "app_version": __version__,
        "name": str(info.pop("name", "") or (bm.path.stem if bm.path else "")),
        "created": info.pop("created", None) or now, "modified": now,
        "n_blocks": int(bm.n_blocks), "null_value": float(bm.null_value),
        **layout,
        "attributes": [{"name": n, "type": "text" if bm.is_categorical(n) else "float",
                        "description": str(desc.get(n, "")),
                        **({"categories": list(bm.categories[n])} if bm.is_categorical(n) else {})}
                       for n in bm.attributes],
        "history": list(info.pop("history", []) or [])[-200:],
        "info": info,
        "arrays": [{"name": n, "dtype": a.dtype.str, "shape": list(a.shape)} for n, a in arrays],
    }


def save_block_model(bm: BlockModel, path, skip=()) -> Path:
    """Write ``bm`` to ``path`` (.gvbm), compressed and encrypted. Attributes in
    ``skip`` (e.g. the reserve's computed mined_fraction / status) are left out. The file is
    written next to the target first and then swapped in, so a failed save never leaves a
    half-written block model."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    path = Path(path)
    if skip:
        bm = BlockModel(bm.path, bm.centroids, bm.sizes,
                        {k: v for k, v in bm.attributes.items() if k not in skip},
                        {k: v for k, v in bm.categories.items() if k not in skip}, bm.null_value,
                        info=bm.info)
    layout, arrays = _arrays_of(bm)
    meta = json.dumps(_meta(bm, layout, arrays), ensure_ascii=False).encode("utf-8")
    plain_len = 4 + len(meta) + sum(a.nbytes for _n, a in arrays)

    salt, prefix = os.urandom(16), os.urandom(4)
    header = _HEADER.pack(MAGIC, VERSION, 0, salt, prefix, CHUNK, plain_len)
    aes = AESGCM(_derive(salt))

    def pieces():
        buf = bytearray(struct.pack("<I", len(meta)) + meta)
        for _n, a in arrays:
            mv = memoryview(a.reshape(-1).view(np.uint8))
            pos = 0
            while pos < len(mv):
                take = min(CHUNK - len(buf), len(mv) - pos)
                buf += mv[pos:pos + take]
                pos += take
                if len(buf) == CHUNK:
                    yield bytes(buf)
                    buf = bytearray()
        if buf:
            yield bytes(buf)

    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "wb") as f:
            f.write(header)
            counter = 0
            pending = None
            for piece in pieces():
                if pending is not None:
                    ct = aes.encrypt(_nonce(prefix, counter), zlib.compress(pending, 1), _aad(header, counter, False))
                    f.write(struct.pack("<I", len(ct)) + ct)
                    counter += 1
                pending = piece
            ct = aes.encrypt(_nonce(prefix, counter), zlib.compress(pending or b"", 1), _aad(header, counter, True))
            f.write(struct.pack("<I", len(ct)) + ct)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def is_block_file(path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == MAGIC
    except OSError:
        return False


def read_block_model(path, meta_only: bool = False) -> BlockModel | dict:
    """Open a .gvbm file (raises BlockFileError). ``meta_only``: just the meta dict."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        header = f.read(_HEADER.size)
        if len(header) < _HEADER.size or header[:4] != MAGIC:
            raise BlockFileError(tr("{name} is not a GeoViewer block model", name=path.name))
        magic, version, _flags, salt, prefix, _chunk, plain_len = _HEADER.unpack(header)
        if version > VERSION:
            raise BlockFileError(tr("{name} was made by a newer GeoViewer; update GeoViewer to open it",
                                    name=path.name))
        aes = AESGCM(_derive(salt))
        plain = bytearray(plain_len)
        pos = counter = 0
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                raise BlockFileError(tr("{name} is damaged (cut short)", name=path.name))
            (clen,) = struct.unpack("<I", raw)
            ct = f.read(clen)
            if len(ct) < clen:
                raise BlockFileError(tr("{name} is damaged (cut short)", name=path.name))
            last = f.tell() >= size
            try:
                piece = zlib.decompress(aes.decrypt(_nonce(prefix, counter), ct, _aad(header, counter, last)))
            except InvalidTag:
                raise BlockFileError(tr("{name} is damaged or was changed outside GeoViewer",
                                        name=path.name)) from None
            if pos + len(piece) > plain_len:
                raise BlockFileError(tr("{name} is damaged", name=path.name))
            plain[pos:pos + len(piece)] = piece
            pos += len(piece)
            counter += 1
            if counter == 1:
                (mlen,) = struct.unpack_from("<I", plain, 0)
                if meta_only and pos >= 4 + mlen:
                    break
            if last:
                break
        if not meta_only and pos != plain_len:
            raise BlockFileError(tr("{name} is damaged", name=path.name))
    (mlen,) = struct.unpack_from("<I", plain, 0)
    meta = json.loads(bytes(plain[4:4 + mlen]).decode("utf-8"))
    if meta_only:
        return meta
    return _model_from(path, meta, plain, 4 + mlen)


def _model_from(path: Path, meta: dict, plain: bytearray, offset: int) -> BlockModel:
    arrays = {}
    for spec in meta["arrays"]:
        dt = np.dtype(spec["dtype"])
        shape = tuple(spec["shape"])
        n = int(np.prod(shape)) if shape else 1
        a = np.frombuffer(plain, dtype=dt, count=n, offset=offset).reshape(shape)
        arrays[spec["name"]] = a
        offset += n * dt.itemsize
    if meta.get("layout") == "grid":
        origin = np.asarray(meta["origin"], dtype=np.float64)
        cell = np.asarray(meta["cell"], dtype=np.float64)
        ijk = arrays["ijk"].astype(np.float64)
        span = (arrays["span"].astype(np.float64) if "span" in arrays
                else np.broadcast_to(np.asarray(meta.get("span", [1, 1, 1]), dtype=np.float64), ijk.shape))
        centroids = origin + (ijk + span / 2.0) * cell
        sizes = (span * cell).astype(np.float32)
    else:
        centroids = np.array(arrays["centroids"], dtype=np.float64)
        sizes = np.array(arrays["sizes"], dtype=np.float32)
    attrs, cats, desc = {}, {}, {}
    for i, a in enumerate(meta["attributes"]):
        name = a["name"]
        arr = arrays[f"attr_{i}"]
        attrs[name] = arr.astype(np.int32 if a.get("type") == "text" else np.float32, copy=False)
        if not attrs[name].flags.writeable:
            attrs[name] = attrs[name].copy()
        if a.get("type") == "text":
            cats[name] = list(a.get("categories", []))
        if a.get("description"):
            desc[name] = a["description"]
    info = dict(meta.get("info") or {})
    info.update(name=meta.get("name", path.stem), created=meta.get("created"), modified=meta.get("modified"),
                history=list(meta.get("history") or []), descriptions=desc)
    return BlockModel(path, centroids, sizes, attrs, cats, float(meta.get("null_value", SURPAC_NULL)), info=info)


def add_history(bm: BlockModel, what: str) -> None:
    """Note an operation in the block model's history (saved with it)."""
    bm.info.setdefault("history", []).append({"when": datetime.now().isoformat(timespec="seconds"), "what": what})
