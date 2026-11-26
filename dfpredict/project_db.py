"""Setting up the project database: every table GeoViewer uses, made automatically.

``init_project_database`` runs whenever GeoViewer connects (at start, when a workspace is
opened, with Connect in Settings > Database). It only ADDS what is missing - missing tables, and missing
fields of existing tables (as empty columns). It never drops a table, a column or a row,
so a database that is set up and holds data keeps all of it:

* drillhole_collar, drillhole_survey, drillhole_geology, drillhole_assay,
  drillhole_styles, drillhole_translation (+ an index on each hole id column); a
  drillhole_styles from Surpac gets rock / color filled where they are empty;
* table_updates (a version per table) and triggers on every drillhole_ / prod_ table that
  count each insert / update / delete, so Refresh sees changes made anywhere;
* survey_pits, survey_stations, survey_setups, survey_shots, survey_settings (underground
  traverse surveys, see io/survey_store.py), survey_designs + survey_design_points (designed
  pits from the route designer); survey_geo_intervals, survey_geo_structures, survey_geo_samples
  (the geology mapped in the pits, see io/pit_geology_store.py);
* risk_* (drilling records of the risk system, see risk/schema.py), with the predefined reference data
  (rock mechanical properties, rock codes, the measure library, example rigs) filled into empty tables;
* prod_settings, prod_mined_shapes (+ an index on the date); production records of
  earlier versions (production.json, production.sqlite, or geodatabase.sqlite when the
  setting is MySQL) are copied in when there are no production tables yet; outlines
  kept one row per corner are copied into empty outline cells.

With MySQL the database itself must exist (it is never made, so a mistyped name gives an
error instead of a new empty database); with SQLite the file geodatabase.sqlite is made in
the workspace folder.
"""

from __future__ import annotations

from pathlib import Path

from .io.database import DRILLHOLE_PREFIX, PROD_PREFIX, Target, ensure_change_tracking
from .io.drillhole_store import DrillholeStore
from .io.pit_geology_store import PitGeologyStore
from .io.survey_store import SurveyStore
from .scene.production import ProductionLog


def init_project_database(target: Target, folder: str | Path | None) -> list[str]:
    """Make all missing tables in ``target`` (records of earlier versions are taken from
    ``folder``; None: none). Returns what was done, for the user ([] when everything was there).
    Raises DatabaseError / OSError / sqlite3 errors with a message for the user."""
    notes: list[str] = []
    if target.is_mysql:
        target.connect().close()        # a missing database is an error (no database made for a mistyped name)
    else:
        target.path.parent.mkdir(parents=True, exist_ok=True)
    notes += DrillholeStore(target).prepare()
    notes += SurveyStore(target).prepare()
    notes += PitGeologyStore(target).prepare()
    from .risk.store import RiskStore

    notes += RiskStore(target).prepare()          # drilling records + predefined rock properties, codes, measures
    log = ProductionLog.load(Path(folder) if folder is not None else None, target=target)
    if log.imported_from:
        notes.append(f"moved the production records from {Path(log.imported_from).name}")
    notes += log.upgraded + log.ensure_tables()
    with target.connect() as db:                    # table_updates + triggers: Refresh sees every change
        tracked = [t for t in db.tables() if t.lower().startswith((DRILLHOLE_PREFIX, PROD_PREFIX, "survey_"))
                   and t.lower() not in ("drillhole_meta",)]
        try:
            notes += ensure_change_tracking(db, tracked)
        except Exception as e:  # noqa: BLE001 - e.g. no TRIGGER privilege: Refresh falls back to the table layout
            notes.append(f"change tracking not available ({e})")
    return notes
