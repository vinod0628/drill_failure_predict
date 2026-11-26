"""Turn files into layers (data only; actors are built by the scene in the GUI thread)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io import read_block_csv, read_drillholes_access, read_drillholes_csv, read_dtm, read_ssi, read_str
from ..io.drillholes import DRILLHOLE_LAYER_NAME, drillhole_signature, read_drillholes_sqlite
from ..io.drillholes import _AZI, _DIP, _FROM, _HOLE, _SDEPTH, _TO, _X, _Y, _Z, _norm
from ..i18n import N_, tr
from ..io.surpac_dtm import _find_case_insensitive
from .layers import BlockModelLayer, DrillholeLayer, Layer, StringLayer, SurfaceLayer, file_signature

GOCAD_EXTENSIONS = {".ts", ".pl", ".vs", ".so", ".gp"}     # GOCAD TSurf / PLine / VSet / TSolid / group
SUPPORTED_EXTENSIONS = {".str", ".dtm", ".gvbm", ".csv", ".accdb", ".mdb", ".ddb", ".dxf"} | GOCAD_EXTENSIONS

FILE_FILTER = N_(      # shown as tr(FILE_FILTER) (or file_filter())
    "Supported files (*.str *.dtm *.gvbm *.csv *.accdb *.mdb *.ddb *.dxf *.ts *.pl *.vs *.so *.gp);;"
    "GOCAD surfaces / lines / points (*.ts *.pl *.vs *.so *.gp);;"
    "Surpac strings (*.str);;Surpac DTM (*.dtm);;GeoViewer block model (*.gvbm);;"
    "Block model / drillhole CSV (*.csv);;Drillhole database (*.accdb *.mdb *.ddb);;DXF (*.dxf);;"
    "All files (*.*)"
)


def file_filter() -> str:
    """The open dialog's filter in the current language."""
    return tr(FILE_FILTER)


def _styles_for(path: Path, styles_ref: str):
    """Find the .ssi named in a string file header (e.g. 'SSI_STYLES:styles.ssi')."""
    if not styles_ref:
        return None
    name = styles_ref.split(":")[-1].strip()
    if not name:
        return None
    found = _find_case_insensitive(path.parent, name)
    return read_ssi(found) if found else None


def csv_role(path: Path) -> str:
    """'blocks', 'collar', 'survey', 'intervals' or 'unknown' (also for unreadable files)."""
    try:
        return _csv_role(path)
    except Exception:  # noqa: BLE001
        return "unknown"


def _csv_role(path: Path) -> str:
    cols = {_norm(c) for c in pd.read_csv(path, nrows=0).columns}

    def has(names):
        return any(n in cols for n in names)

    if has({"sizex", "xsize", "dimx", "xinc", "dx", "blocksizex"}):
        return "blocks"
    if has(_HOLE):
        if has(_AZI) and has(_DIP) and has(_SDEPTH):
            return "survey"
        if has(_TO) or has(_FROM):
            return "intervals"
        if has(_X) and has(_Y) and has(_Z):
            return "collar"
    return "unknown"


def load_paths(paths: list[str | Path]) -> tuple[list[Layer], list[str]]:
    """Load files into layers. Returns (layers, error messages).

    Drillhole CSVs selected together (collar + survey + interval tables) are
    combined into one drillhole layer; their roles are detected from columns.
    """
    layers: list[Layer] = []
    errors: list[str] = []
    dh_csv: dict[str, list[Path]] = {"collar": [], "survey": [], "intervals": []}

    def add(layer: Layer, inputs, sources, before, part: str = "") -> None:
        layer.part = part
        layer.track_files(inputs, sources, before)
        layers.append(layer)

    for raw in paths:
        p = Path(raw)
        ext = p.suffix.lower()
        before = {p: file_signature(p)}  # taken before reading: a change while reading is seen next time
        try:
            if ext == ".str":
                sf = read_str(p)
                styles = _styles_for(p, sf.header.styles_file)
                add(StringLayer(name=p.stem, path=p, data=sf, styles=styles),
                    [p], [p, styles.path if styles else None], before)
            elif ext == ".dtm":
                dtm = read_dtm(p)
                styles = _styles_for(p, dtm.string_file.header.styles_file) if dtm.string_file else None
                add(SurfaceLayer(name=p.stem, path=p, data=dtm, styles=styles), [p],
                    [p, dtm.string_file.path if dtm.string_file else None, styles.path if styles else None],
                    before)
            elif ext == ".gvbm":
                from ..blockmodel.store import read_block_model

                add(BlockModelLayer(name=p.stem, path=p, data=read_block_model(p)), [p], [p], before)
            elif ext in (".accdb", ".mdb"):
                add(DrillholeLayer(name=p.stem, path=p, data=read_drillholes_access(p)), [p], [p], before)
            elif ext == ".ddb":
                db = next((q for q in (p.with_suffix(".accdb"), p.with_suffix(".mdb")) if q.exists()), None)
                if db is None:
                    raise FileNotFoundError(tr("No .accdb/.mdb next to {name}", name=p.name))
                before[db] = file_signature(db)
                add(DrillholeLayer(name=p.stem, path=db, data=read_drillholes_access(db)), [p], [p, db], before)
            elif ext == ".csv":
                role = _csv_role(p)
                if role == "blocks":
                    add(BlockModelLayer(name=p.stem, path=p, data=read_block_csv(p)), [p], [p], before)
                elif role in dh_csv:
                    dh_csv[role].append(p)
                else:
                    raise ValueError(tr("CSV columns not recognised as a block model or drillhole table"))
            elif ext in GOCAD_EXTENSIONS:
                from ..io.gocad import gocad_to_dtm, gocad_to_strings, read_gocad

                objs = read_gocad(p)
                dtm, sf = gocad_to_dtm(objs, p), gocad_to_strings(objs, p)
                if dtm is not None:
                    add(SurfaceLayer(name=p.stem, path=p, data=dtm), [p], [p], before, "faces")
                if sf is not None:
                    add(StringLayer(name=f"{p.stem} (lines)" if dtm is not None else p.stem, path=p, data=sf),
                        [p], [p], before, "lines")
                if dtm is None and sf is None:
                    raise ValueError(tr("No surface, line or point object in this GOCAD file"))
            elif ext == ".dxf":
                from ..io.dxf_reader import read_dxf

                sf, dtm = read_dxf(p)
                if sf is not None:
                    add(StringLayer(name=p.stem, path=p, data=sf), [p], [p], before)
                if dtm is not None:
                    add(SurfaceLayer(name=f"{p.stem} (faces)", path=p, data=dtm), [p], [p], before, "faces")
            else:
                raise ValueError(tr("Unsupported file type {ext}", ext=ext))
        except Exception as e:  # noqa: BLE001 - report every failure to the user
            errors.append(f"{p.name}: {e}")

    if dh_csv["collar"]:
        collar = dh_csv["collar"][0]
        files = [collar] + dh_csv["survey"][:1] + dh_csv["intervals"]
        before = {q: file_signature(q) for q in files}
        try:
            db = read_drillholes_csv(
                collar,
                dh_csv["survey"][0] if dh_csv["survey"] else None,
                {q.stem: q for q in dh_csv["intervals"]},
            )
            add(DrillholeLayer(name=f"Drillholes ({collar.stem})", path=collar, data=db), files, files, before)
        except Exception as e:  # noqa: BLE001
            errors.append(tr("Drillholes: {error}", error=e))
    elif dh_csv["survey"] or dh_csv["intervals"]:
        errors.append(tr("Drillhole CSV: select the collar file together with the survey and interval files"))
    return layers, errors


def offline_database_layer(target, reason: str, spec: dict | None = None, kind: str = "DrillholeLayer") -> Layer:
    """A database layer (``kind``: DrillholeLayer / SurveyLayer) of a project database that
    cannot be reached: listed (greyed out) and empty, keeping its saved settings (``spec``)
    until the database is back."""
    import pandas as pd

    kind = (spec or {}).get("type") or kind
    if kind == "SurveyLayer":
        from .survey import SURVEY_LAYER_NAME, SurveyData, SurveyLayer

        name = str((spec or {}).get("name") or SURVEY_LAYER_NAME)
        layer = SurveyLayer(name=name, path=None, data=SurveyData([], {}, {}, target.label), visible=False)
    else:
        from ..io.drillholes import DrillholeDB

        db = DrillholeDB(pd.DataFrame(columns=["hole_id", "x", "y", "z", "max_depth"]),
                         pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"]), {}, source=target.label)
        name = str((spec or {}).get("name") or DRILLHOLE_LAYER_NAME)
        layer = DrillholeLayer(name=name, path=None, data=db, visible=False)
    layer.source = "database"
    layer.db_target = target
    layer.db_signature = None                      # Refresh reloads it as soon as the database answers
    layer.db_file_sig = None
    layer.offline = reason
    layer.offline_spec = dict(spec) if spec else {"name": name, "type": kind, "part": "",
                                                  "files": [], "settings": {}, "source": "database"}
    return layer


def survey_signature(target) -> tuple | None:
    """A fingerprint of the survey tables (table_updates versions, or row counts); None when
    the database cannot be reached."""
    from ..io.database import change_signature

    try:
        with target.connect(create=False) as db:
            tracked = change_signature(db, "survey_")
            if tracked is not None:
                return ("tracked",) + tracked
            return tuple((t, db.scalar(f"SELECT COUNT(*) FROM {db.q(t)}")) for t in db.tables()
                         if t.lower().startswith("survey_"))
    except Exception:  # noqa: BLE001
        return None


def load_database_survey(fallback_folder) -> tuple[Layer | None, list[str]]:
    """The underground survey of the project database (pits, stations, setups) as a layer that
    draws the drives along the pits' centre lines, or (None, errors)."""
    from ..io.database import project_target
    from ..io.survey_store import SurveyStore
    from .survey import SURVEY_LAYER_NAME, SurveyData, SurveyLayer, pit_paths

    target = project_target(fallback_folder)
    try:
        file_sig = None if target.is_mysql else file_signature(target.path)
        sig = survey_signature(target)
        store = SurveyStore(target)
        pits = store.pits()
        data = SurveyData(pits, {s.name: s for s in store.stations()}, pit_paths(store.setups()), target.label,
                          designs=store.designs())
        _add_pit_geology(data, target)
    except Exception as e:  # noqa: BLE001 - unreachable server, no tables
        return None, [f"{SURVEY_LAYER_NAME}: {e}"]
    layer = SurveyLayer(name=SURVEY_LAYER_NAME, path=None, data=data)
    layer.source = "database"
    layer.db_signature = sig
    layer.db_target = target
    layer.db_file_sig = file_sig
    return layer, []


def _add_pit_geology(data, target) -> None:
    """The geology mapped in the pits, the rock colours and the samples' assay fields (none when
    the tables are not there yet)."""
    from ..io.drillhole_store import DrillholeStore
    from ..io.pit_geology_store import PitGeologyStore

    try:
        store = PitGeologyStore(target)
        data.geology = store.geology()
        data.assay_fields = store.assay_fields()
    except Exception as e:  # noqa: BLE001 - e.g. an older database without the tables
        data.warnings.append(f"pit geology: {e}")
    if data.geology:
        try:
            data.rock_colors = DrillholeStore(target).styles()
        except Exception:  # noqa: BLE001 - no rock colours: palette colours
            data.rock_colors = {}


def load_database_layer(kind: str, fallback_folder) -> tuple[Layer | None, list[str]]:
    """A database layer by its type name (DrillholeLayer / SurveyLayer)."""
    if kind == "SurveyLayer":
        return load_database_survey(fallback_folder)
    return load_database_drillholes(fallback_folder)


def database_signature(kind: str, target) -> tuple | None:
    if kind == "SurveyLayer":
        return survey_signature(target)
    return drillhole_signature(target)


def load_database_drillholes(fallback_folder) -> tuple[Layer | None, list[str]]:
    """The drillholes kept in the project database (drillhole_collar, drillhole_survey, ...)
    as a layer, or (None, errors). ``fallback_folder``: where geodatabase.sqlite is when no
    workspace is open."""
    import warnings

    from ..io.database import project_target

    target = project_target(fallback_folder)
    try:
        file_sig = None if target.is_mysql else file_signature(target.path)
        sig = drillhole_signature(target)          # before reading: a change while reading is seen next time
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = read_drillholes_sqlite(target)
    except Exception as e:  # noqa: BLE001 - unreachable server, tables without the needed columns
        return None, [f"{DRILLHOLE_LAYER_NAME}: {e}"]
    layer = DrillholeLayer(name=DRILLHOLE_LAYER_NAME, path=None, data=db)
    layer.source = "database"
    layer.db_signature = sig
    layer.db_target = target
    layer.db_file_sig = file_sig                   # SQLite: the tables are only checked when the file changed
    return layer, []
