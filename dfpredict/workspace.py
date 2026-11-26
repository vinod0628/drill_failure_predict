"""Workspaces: a folder of data files plus a small JSON file describing the display.

``<folder>/geoviewer_workspace.json`` stores, for every layer, the files it is
loaded from (relative to the folder when inside it) and its display settings,
together with the view (mode, camera in world coordinates, map grid) and the
data files already seen in the folder. A folder without that file is opened
by loading the supported files found in it.

Refresh support: every layer remembers the signature (modification time and
size) of each file it depends on. ``changed_layers`` finds layers whose files
changed, ``reload_layers`` loads them again with their display settings, and
``Workspace.find_new_files`` reports data files that appeared in the folder.
Block model CSVs are never loaded as layers: they are collected in ``Workspace.block_csvs``
for the app to import into GeoViewer block models (.gvbm).
All functions here are free of Qt and can run in a worker thread.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import __version__
from .i18n import tr
from .io.drillholes import DRILLHOLE_LAYER_NAME, project_has_drillhole_tables
from .io.surpac_dtm import _parse_dtm_header
from .scene.layers import Layer, file_signature
from .io.database import PROJECT_DB, project_target
from .scene.production import LEGACY_DB
from .scene.loaders import (SUPPORTED_EXTENSIONS, csv_role, database_signature, load_database_layer, load_paths,
                            offline_database_layer)

log = logging.getLogger(__name__)

WORKSPACE_FILE = "geoviewer_workspace.json"
FORMAT = "GeoViewer workspace"
VERSION = 1
SETTLE_SECONDS = 1.5      # a file modified more recently than this may still be being written
MAX_SCAN_DEPTH = 3        # sub-folder levels searched for data files


# --- paths ---------------------------------------------------------------------
def _key(p: Path) -> str:
    """Case-insensitive identity of a path on Windows."""
    return os.path.normcase(os.path.abspath(p))


def _to_text(p: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(p, base)).as_posix() if _key(p).startswith(_key(base) + os.sep) else str(p)
    except ValueError:  # another drive
        return str(p)


def _from_text(s: str, base: Path) -> Path:
    p = Path(s)
    return p if p.is_absolute() else (base / p)


def _recent(p: Path, now: float) -> bool:
    try:
        age = now - os.stat(p).st_mtime
    except OSError:
        return False
    # A just-written file can read a hair in the future; a time stamp far in the future
    # (clock skew on a network share) must not block reloading forever.
    return -5.0 <= age < SETTLE_SECONDS


def _rounded(v):
    if isinstance(v, float):
        return float(f"{v:.9g}")
    if isinstance(v, (list, tuple)):
        return [_rounded(x) for x in v]
    return v


# --- folder scanning -------------------------------------------------------------
def list_data_files(folder: Path, max_depth: int = MAX_SCAN_DEPTH) -> list[Path]:
    """Supported data files in a folder and its sub-folders (hidden/temporary ones skipped)."""
    folder = Path(folder)
    out: list[Path] = []
    base_depth = len(folder.parts)
    for root, dirs, files in os.walk(folder):
        depth = len(Path(root).parts) - base_depth
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__")) and depth < max_depth)
        for name in sorted(files):
            if name.startswith(("~$", ".")):
                continue
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                out.append(Path(root) / name)
    return out


def _dtm_string_file(dtm: Path) -> str | None:
    try:
        with open(dtm, "r", encoding="latin-1", errors="replace") as f:
            return _parse_dtm_header(f.readline())[0].lower() or None
    except OSError:
        return None


def app_made(p: Path) -> bool:
    """Files GeoViewer writes itself: the project database geodatabase.sqlite, and from
    earlier versions production.sqlite and surpac.sqlite next to surpac.accdb. Never "new data"."""
    if p.suffix.lower() not in (".sqlite", ".db"):
        return False
    if p.name.lower() in (PROJECT_DB, LEGACY_DB):
        return True
    return any(p.with_suffix(ext).exists() for ext in (".accdb", ".mdb"))


def loadable_groups(candidates: list[Path], all_files: list[Path]) -> list[list[Path]]:
    """Choose which files become layers and group them for the loader (one group per folder).

    Skipped: the .str paired with a .dtm (the DTM shows it), a .ddb next to its .accdb/.mdb, CSVs that are neither
    block models nor drillhole tables, and drillhole survey/interval CSVs without a collar CSV in the same folder.
    """
    by_dir: dict[Path, list[Path]] = {}
    for p in all_files:
        by_dir.setdefault(p.parent, []).append(p)
    groups: dict[Path, list[Path]] = {}
    for p in candidates:
        siblings = by_dir.get(p.parent, [])
        names = {q.name.lower() for q in siblings}
        ext = p.suffix.lower()
        if ext == ".str":
            paired = {_dtm_string_file(q) for q in siblings if q.suffix.lower() == ".dtm"}
            if p.name.lower() in paired:
                continue
        elif ext == ".ddb":
            if f"{p.stem}.accdb".lower() in names or f"{p.stem}.mdb".lower() in names:
                continue
        groups.setdefault(p.parent, []).append(p)
    out = []
    for files in groups.values():
        roles = {p: csv_role(p) for p in files if p.suffix.lower() == ".csv"}
        has_collar = "collar" in roles.values()
        keep = [p for p in files if p not in roles or roles[p] in ("blocks", "collar")
                or (roles[p] in ("survey", "intervals") and has_collar)]
        if keep:
            out.append(keep)
    return out


def is_block_csv(p: Path) -> bool:
    """A block model CSV export. GeoViewer never shows one as it is: it is imported into a
    GeoViewer block model (.gvbm) in the Block Model Manager first."""
    return p.suffix.lower() == ".csv" and csv_role(p) == "blocks"


def split_block_csvs(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """(other files, block model CSVs to import). A block model CSV with a .gvbm of the same
    name next to it was imported already: the .gvbm shows it, the CSV is left out."""
    others, blocks = [], []
    for p in paths:
        if not is_block_csv(p):
            others.append(p)
        elif not p.with_suffix(".gvbm").exists():
            blocks.append(p)
    return others, blocks


# --- layer specs (what a workspace stores per layer) ----------------------------------
def saved_layers(layers: list[Layer]) -> list[Layer]:
    """Layers written to the workspace file (not the ones made by the app, e.g. mined shapes)."""
    return [lay for lay in layers if getattr(lay, "saved", True)]


def app_layer_states(layers: list[Layer], previous: dict | None = None) -> dict:
    """Name and display settings of the layers the app makes itself (e.g. "Mined shapes", which
    the Reserve tools make again at each start), by kind. ``previous`` (from the file) is kept
    for those not made in this session."""
    out = dict(previous or {})
    for lay in layers:
        if not getattr(lay, "saved", True):
            out[lay.kind] = {"name": lay.name, "settings": lay.settings()}
    return out


def restore_app_layer(ws, layer: Layer) -> None:
    """Give a layer made by the app the name and display settings it had (visible or hidden,
    opacity, colour) when the workspace was last saved."""
    saved = (getattr(ws, "app_layers", None) or {}).get(layer.kind) if ws is not None else None
    if not isinstance(saved, dict):
        return
    layer.apply_settings(saved.get("settings") or {})
    if saved.get("name"):
        layer.name = str(saved["name"])


def layer_spec(layer: Layer, base: Path | None) -> dict:
    if getattr(layer, "offline", None):            # database not reachable: keep what was saved
        return dict(layer.offline_spec)
    files = [(_to_text(p, base) if base is not None else str(p)) for p in layer.inputs]
    spec = {"name": layer.name, "type": type(layer).__name__, "part": layer.part,
            "files": files, "settings": layer.settings()}
    if layer.source:
        spec["source"] = layer.source          # "database": the project database, no files
    return spec


def load_spec(spec: dict, base: Path) -> tuple[Layer | None, list[str]]:
    """Load one layer described by a workspace entry and apply its saved settings."""
    files = spec.get("files", [])
    if (spec.get("type") == "DrillholeLayer" and len(files) == 1
            and Path(files[0]).name.lower() == PROJECT_DB
            and _from_text(files[0], base).parent == Path(base)):
        # geodatabase.sqlite opened as a file (earlier version): it is the project database now
        spec = {**spec, "files": [], "source": "database"}
        if spec.get("name") == Path(PROJECT_DB).stem:
            spec["name"] = DRILLHOLE_LAYER_NAME
    if spec.get("source") == "database":
        layer, errors = load_database_layer(spec.get("type") or "DrillholeLayer", base)
        if layer is not None:
            layer.apply_settings(spec.get("settings") or {})
            if spec.get("name"):
                layer.name = str(spec["name"])
        return layer, errors
    files = [_from_text(f, base) for f in spec.get("files", [])]
    label = spec.get("name") or (files[0].name if files else "?")
    missing = [f.name for f in files if not f.exists()]
    if not files or missing:
        return None, [tr("{name}: file not found ({files})", name=label, files=", ".join(missing) or tr("no files"))]
    layers, errors = load_paths(files)
    match = next((lay for lay in layers if type(lay).__name__ == spec.get("type")
                  and lay.part == spec.get("part", "")), None)
    if match is None:
        return None, errors or [tr("{name}: no {type} found in {file}", name=label, type=spec.get("type"),
                                   file=files[0].name)]
    match.apply_settings(spec.get("settings") or {})
    if spec.get("name"):
        match.name = str(spec["name"])
    return match, errors


# --- change detection / reload ---------------------------------------------------------
@dataclass
class ChangeReport:
    changed: list[Layer] = field(default_factory=list)
    waiting: list[Layer] = field(default_factory=list)   # modified, but maybe still being written
    missing: list[Path] = field(default_factory=list)


def changed_layers(layers: list[Layer]) -> ChangeReport:
    rep = ChangeReport()
    now = time.time()
    for layer in layers:
        if layer.source == "database":
            target = getattr(layer, "db_target", None)
            if target is None:
                continue
            if not target.is_mysql:            # SQLite: nothing to check while the file is unchanged
                fs = file_signature(target.path)
                if fs == getattr(layer, "db_file_sig", None):
                    continue
            cur = database_signature(type(layer).__name__, target)
            if cur is not None and cur == layer.db_signature and not target.is_mysql:
                layer.db_file_sig = fs         # the file changed, the drillholes did not (e.g. a mined shape saved)
            elif cur is not None and cur != layer.db_signature and {"db": cur} != layer.failed_signature:
                rep.changed.append(layer)      # e.g. tables imported again; None: unreachable now
            continue
        modified = False
        settling = False
        current = {}
        for p, sig in layer.signature.items():
            cur = current[p] = file_signature(p)
            if cur is None:
                rep.missing.append(p)
            elif cur != sig:
                modified = True
                settling |= _recent(p, now)
        if modified and current == layer.failed_signature:
            continue  # this version of the files already failed to load; wait for the next change
        if modified:
            (rep.waiting if settling else rep.changed).append(layer)
    return rep


def reload_layers(layers: list[Layer]) -> tuple[list[tuple[Layer, Layer]], list[str]]:
    """Load changed layers again, keeping their display settings. Returns (old, new) pairs."""
    pairs, errors = [], []
    for old in layers:
        snapshot = {p: file_signature(p) for p in old.signature}
        if old.source == "database" and getattr(old, "db_target", None) is not None:
            snapshot = {"db": database_signature(type(old).__name__, old.db_target)}
        base = Path.cwd()
        target = getattr(old, "db_target", None)
        if target is not None and not target.is_mysql:
            base = target.path.parent                      # where its geodatabase.sqlite is
        new, errs = load_spec(layer_spec(old, None), base)
        errors += errs
        if new is None:
            old.failed_signature = snapshot
            continue
        new.name = old.name
        pairs.append((old, new))
    return pairs, errors


# --- the workspace -----------------------------------------------------------------------
class Workspace:
    def __init__(self, folder: str | Path):
        self.folder = Path(os.path.abspath(folder))
        self.known: set[str] = set()        # data files already shown or deliberately skipped
        self.saved_state: str | None = None  # fingerprint of the last saved / loaded state
        # Layers listed in the workspace file that could not be opened (file missing, e.g. a
        # network drive not connected). They are kept in the file and retried by Refresh.
        self.pending_specs: list[dict] = []
        self._unreadable = False             # the workspace file could not be read: back it up on save
        # the drillhole layer of the project database is shown in this session (removed by the
        # user: not added back by Refresh; it is listed again when the workspace is next opened)
        self.database_drillholes_seen = False
        self.database_survey_seen = False       # the same for the survey layer
        self.app_layers: dict = {}              # kind -> {"name", "settings"} of layers the app makes
        self.database_notes: list[str] = []     # what the last init made, e.g. ["made table drillhole_assay"]
        self.database_error = ""                 # why the project database cannot be used ("" = connected)
        self.database_ready = False              # init_database ran (a folder adopted for opened files: not yet)
        # block model CSVs found (folder scan, Refresh, a CSV layer of an older workspace file):
        # shown only once imported as .gvbm; the app takes this list and offers the import
        self.block_csvs: list[Path] = []

    @property
    def file(self) -> Path:
        return self.folder / WORKSPACE_FILE

    @property
    def name(self) -> str:
        return self.folder.name or str(self.folder)

    def read(self) -> dict | None:
        if not self.file.exists():
            return None
        with open(self.file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(tr("{file} is not a GeoViewer workspace file", file=self.file.name))
        return data

    # --- opening ---------------------------------------------------------------
    def load(self) -> tuple[list[Layer], list[str], dict | None]:
        """Load the workspace (or scan the folder). Returns (layers, errors, saved view)."""
        if not self.folder.is_dir():
            return [], [tr("Workspace folder not found: {folder}", folder=self.folder)], None
        errors: list[str] = []
        try:
            data = self.read()
        except (OSError, ValueError) as e:
            data = None
            self._unreadable = True
            errors.append(tr("{file}: {error} - loading the folder contents instead", file=WORKSPACE_FILE, error=e))
        self.init_database()                   # connect: make missing tables / fields (database_error if not)
        all_files = list_data_files(self.folder)
        layers: list[Layer] = []
        view = None
        self.pending_specs = []
        if data is not None:
            seen: set = set()        # (type, part, files [, "database"]) of the layers taken
            for spec in data.get("layers", []):
                if not isinstance(spec, dict):
                    continue
                key = (spec.get("type"), spec.get("part", ""), tuple(spec.get("files", [])), spec.get("source"))
                if key in seen:
                    # a second copy of the same layer (made by "Duplicate view" in earlier
                    # versions, e.g. "Pits (2)"): only the first one is opened
                    log.info("workspace: extra copy of %s (%s) not opened", spec.get("name"), spec.get("type"))
                    continue
                seen.add(key)
                csv = self._block_csv_of(spec)
                if csv is not None:                # a CSV block model layer (earlier versions)
                    self.block_csvs.append(csv)
                    continue
                if spec.get("source") == "database" and self.database_error:
                    layer, errs = offline_database_layer(project_target(self.folder), self.database_error,
                                                         spec), []
                else:
                    layer, errs = load_spec(spec, self.folder)
                errors += errs
                if layer is not None:
                    layers.append(layer)
                else:
                    self.pending_specs.append(spec)
            view = data.get("view")
            app = data.get("app_layers")
            self.app_layers = dict(app) if isinstance(app, dict) else {}
            self.known = {_key(_from_text(f, self.folder)) for f in data.get("known_files", [])}
            for layer in layers:
                self.known |= {_key(p) for p in layer.inputs}
            self.known |= {_key(p) for p in self.block_csvs}
            # Files that appeared while the app was closed are loaded as new layers.
            new = [p for p in all_files if _key(p) not in self.known and not app_made(p)]
        else:
            new = all_files
        new_ready = [p for p in new if not _recent(p, time.time())]
        loadable, blocks = split_block_csvs(new_ready)
        self.block_csvs += blocks
        for group in loadable_groups(loadable, all_files):
            lays, errs = load_paths(group)
            layers += lays
            errors += errs
        self.known |= {_key(p) for p in new_ready}
        self.database_drillholes_seen = any(lay.source == "database" and type(lay).__name__ == "DrillholeLayer"
                                            for lay in layers)
        self.database_survey_seen = any(lay.source == "database" and type(lay).__name__ == "SurveyLayer"
                                        for lay in layers)
        lays, errs = self.new_database_layers()            # always listed (also with nothing in them yet)
        layers += lays
        errors += errs
        return layers, errors, view

    def _block_csv_of(self, spec: dict) -> Path | None:
        """The CSV of a block model layer read straight from a CSV (an older workspace file)."""
        if spec.get("type") != "BlockModelLayer":
            return None
        files = [_from_text(f, self.folder) for f in spec.get("files", [])]
        if len(files) == 1 and files[0].suffix.lower() == ".csv" and files[0].exists():
            return files[0]
        return None

    def init_database(self) -> bool:
        """Connect to the project database and make its missing tables / fields (see
        project_db). True when connected; otherwise ``database_error`` says why (server
        down, wrong password, read-only folder, ...). What was made: ``database_notes``."""
        from .project_db import init_project_database

        target = project_target(self.folder)
        self.database_ready = True
        try:
            self.database_notes = init_project_database(target, self.folder)
            self.database_error = ""
        except Exception as e:  # noqa: BLE001 - unreachable server, read-only folder, no rights
            self.database_notes = []
            self.database_error = str(e) or type(e).__name__
        return not self.database_error

    def new_database_layers(self) -> tuple[list[Layer], list[str]]:
        """The layers of the project database - drillholes and survey - not shown yet in this
        session: always listed (also empty), greyed out while the database cannot be reached."""
        out: list[Layer] = []
        errors: list[str] = []
        target = project_target(self.folder)
        for kind, attr in (("DrillholeLayer", "database_drillholes_seen"), ("SurveyLayer", "database_survey_seen")):
            if getattr(self, attr):
                continue
            if self.database_error:
                setattr(self, attr, True)
                out.append(offline_database_layer(target, self.database_error, kind=kind))
                continue
            if kind == "DrillholeLayer" and not project_has_drillhole_tables(target):
                continue
            layer, errs = load_database_layer(kind, self.folder)
            errors += errs
            if layer is not None:
                setattr(self, attr, True)
                out.append(layer)
        return out, errors

    def mark_existing_files_known(self) -> None:
        """Treat the data files now in the folder as seen (Refresh then adds only newer ones)."""
        if self.folder.is_dir():
            self.known |= {_key(p) for p in list_data_files(self.folder)}

    # --- saving ----------------------------------------------------------------
    def state(self, layers: list[Layer], view: dict | None) -> dict:
        return {
            "format": FORMAT,
            "version": VERSION,
            "app_version": __version__,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "view": view,
            "layers": [layer_spec(lay, self.folder) for lay in saved_layers(layers)]
                      + list(self.pending_specs),
            "app_layers": app_layer_states(layers, self.app_layers),
            "known_files": sorted(_to_text(Path(k), self.folder) for k in self.known),
        }

    @staticmethod
    def fingerprint(layers: list[Layer], view: dict | None, camera: bool = False) -> str:
        """Identity of what a save would write: layers, their settings, view mode and grid,
        and with ``camera`` also the camera (rounded, so float noise is not a change)."""
        specs = [layer_spec(lay, None) for lay in saved_layers(layers)]
        specs.append(app_layer_states(layers))         # e.g. "Mined shapes" hidden: saved too
        keys = ("mode", "grid", "perspective") + (
            ("focal_point", "position", "view_up", "parallel_scale") if camera else ())
        v = {k: _rounded((view or {}).get(k)) for k in keys}
        return json.dumps([specs, v], sort_keys=True, default=str)

    def save(self, layers: list[Layer], view: dict | None) -> Path:
        for layer in layers:
            self.known |= {_key(p) for p in layer.inputs}
        data = self.state(layers, view)
        self.app_layers = data["app_layers"]
        if self._unreadable and self.file.exists():
            try:
                shutil.copyfile(self.file, self.file.with_suffix(".json.bak"))
            except OSError:
                pass
            self._unreadable = False
        tmp = self.file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.file)
        self.saved_state = self.fingerprint(layers, view, camera=True)
        return self.file

    def writable(self) -> bool:
        """True when the workspace file can be written in the folder."""
        if not self.folder.is_dir():
            return False
        if self.file.exists():
            return os.access(self.file, os.W_OK)
        probe = self.folder / (WORKSPACE_FILE + ".probe")
        try:
            with open(probe, "w", encoding="utf-8"):
                pass
            os.remove(probe)
            return True
        except OSError:
            return False

    def load_pending(self) -> list[tuple[dict, Layer]]:
        """Try again the layers that could not be opened with the workspace, for those whose
        files exist now. Returns (spec, layer) pairs; the caller drops loaded specs."""
        out = []
        for spec in list(self.pending_specs):
            if spec.get("source") == "database":           # e.g. the MySQL server was down
                layer, _errs = load_spec(spec, self.folder)
                if layer is not None:
                    out.append((spec, layer))
                continue
            files = [_from_text(f, self.folder) for f in spec.get("files", [])]
            if not files or not all(f.exists() for f in files):
                continue
            if any(_recent(f, time.time()) for f in files):
                continue
            layer, _errs = load_spec(spec, self.folder)
            if layer is not None:
                if spec.get("name"):
                    layer.name = spec["name"]
                out.append((spec, layer))
        return out

    # --- refresh ---------------------------------------------------------------
    def find_new_files(self) -> tuple[list[list[Path]], int]:
        """Data files that appeared since the workspace was opened, grouped for loading,
        and the number still being written (left for the next refresh)."""
        if not self.folder.is_dir():
            return [], 0
        all_files = list_data_files(self.folder)
        new = [p for p in all_files if _key(p) not in self.known and not app_made(p)]
        if not new:
            return [], 0
        now = time.time()
        ready = [p for p in new if not _recent(p, now)]
        self.known |= {_key(p) for p in ready}
        loadable, blocks = split_block_csvs(ready)
        self.block_csvs += blocks
        return loadable_groups(loadable, all_files), len(new) - len(ready)
