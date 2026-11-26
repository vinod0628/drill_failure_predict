"""Binary cache locations for parsed data files.

A cache entry is keyed on the source file's absolute path, size and
modification time, so editing or replacing the source file invalidates it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def default_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".cache")
    return Path(base) / "DrillFailPredict" / "cache"


def cache_path_for(source: Path, kind: str, cache_dir: str | Path | None = None) -> Path:
    source = Path(source).resolve()
    st = source.stat()
    key = f"{kind}|{source}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:20]
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / f"{source.stem}-{digest}.npz"
