"""Database access for GeoViewer's own data (drillholes, production records).

All of the app's tables live in one *project database*, kept apart by prefix:

    drillhole_*  drillholes, filled by the user (e.g. imported from Excel): drillhole_collar,
                 drillhole_survey, drillhole_geology, drillhole_sample, ... GeoViewer only reads them.
    prod_*       the production records written by GeoViewer (prod_mined_shapes, prod_settings, ...)

The project database is, by the Database setting (Settings > Database):

* MySQL (the default) - one database on a MySQL server (``DatabaseConfig``), shared by
  everyone who points GeoViewer at it;
* SQLite - the file ``geodatabase.sqlite`` in the workspace folder; nothing to install, the
  folder can be copied or moved as a whole.

Everything goes through :class:`Database`, so the SQL stays portable:
* ``?`` placeholders (converted to ``%s`` for MySQL),
* table / column names quoted with :meth:`Database.q`,
* only integer, double, text and blob columns; dates are ISO text,
* replacing a set of tables uses :meth:`Database.replace_tables` (atomic on both).
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..i18n import tr

SQLITE_SUFFIXES = (".sqlite", ".db")
PROJECT_DB = "geodatabase.sqlite"
DRILLHOLE_PREFIX = "drillhole_"
PROD_PREFIX = "prod_"
BACKENDS = {"sqlite": "SQLite", "mysql": "MySQL"}


# --- settings -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DatabaseConfig:
    """Which database the app uses (stored in the application settings)."""

    backend: str = "mysql"           # "mysql" (the default) or "sqlite"
    host: str = "localhost"
    port: int = 3306
    database: str = "geoviewer"
    user: str = "root"
    password: str = ""

    @property
    def is_mysql(self) -> bool:
        return self.backend == "mysql"

    def describe(self) -> str:
        if self.is_mysql:
            return tr("MySQL database {name} on {host}:{port}", name=self.database, host=self.host,
                      port=self.port)
        return tr("SQLite file {file} in the workspace folder", file=PROJECT_DB)


_config = DatabaseConfig()
_project_folder: Path | None = None


def configure(config: DatabaseConfig) -> None:
    global _config
    _config = config


def current_config() -> DatabaseConfig:
    return _config


def set_project_folder(folder: str | Path | None) -> None:
    """The workspace folder (where geodatabase.sqlite is kept; None: no workspace open)."""
    global _project_folder
    _project_folder = Path(folder) if folder is not None else None


# --- where the project database is ----------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    """A database to connect to: an SQLite file or a MySQL database."""

    backend: str
    path: Path | None = None
    config: DatabaseConfig | None = None

    @property
    def is_mysql(self) -> bool:
        return self.backend == "mysql"

    @property
    def label(self) -> str:
        """For messages: the file path, or 'MySQL geoviewer@host'."""
        if self.is_mysql:
            c = self.config
            return f"MySQL {c.database}@{c.host}"
        return str(self.path)

    @property
    def short(self) -> str:
        return tr("MySQL database {name}", name=self.config.database) if self.is_mysql else self.path.name

    def exists(self) -> bool:
        """The SQLite file is there (a MySQL database is always assumed to be)."""
        return True if self.is_mysql else self.path.exists()

    def connect(self, create: bool = True) -> "Database":
        if self.is_mysql:
            return Database.mysql(self.config)
        return Database.sqlite(self.path, create=create)


def project_target(fallback_folder: str | Path) -> Target:
    """The project database by the current setting. For SQLite: geodatabase.sqlite in the
    open workspace folder, else in ``fallback_folder`` (e.g. the folder of the .accdb)."""
    if _config.is_mysql:
        return Target("mysql", config=_config)
    folder = _project_folder if _project_folder is not None else Path(fallback_folder)
    return Target("sqlite", path=Path(folder) / PROJECT_DB)


def as_target(where) -> Target:
    """A Target from a Target, or from an SQLite file path."""
    return where if isinstance(where, Target) else Target("sqlite", path=Path(where))


# --- dialects ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Dialect:
    name: str
    quote: str                  # identifier quote character
    integer: str
    real: str
    text: str
    blob: str

    def q(self, name: str) -> str:
        return f"{self.quote}{str(name).replace(self.quote, self.quote * 2)}{self.quote}"


SQLITE = Dialect("sqlite", '"', "INTEGER", "REAL", "TEXT", "BLOB")
MYSQL = Dialect("mysql", "`", "BIGINT", "DOUBLE", "LONGTEXT", "LONGBLOB")


class DatabaseError(Exception):
    """A database could not be reached or used; the message is meant for the user."""


class Database:
    """A connection plus the dialect it speaks. Use as a context manager."""

    def __init__(self, con, dialect: Dialect, label: str = ""):
        self.con = con
        self.dialect = dialect
        self.label = label

    # --- connecting ------------------------------------------------------------------------
    @classmethod
    def sqlite(cls, path: str | Path, create: bool = True) -> "Database":
        path = Path(path)
        if not create and not path.exists():
            raise FileNotFoundError(tr("No database file {path}", path=path))
        con = sqlite3.connect(str(path), timeout=15.0)
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA cache_size = -32000")     # 32 MB page cache: big tables read with fewer disk reads
        return cls(con, SQLITE, str(path))

    @classmethod
    def mysql(cls, cfg: DatabaseConfig, database: bool = True) -> "Database":
        """Connect to the MySQL server of ``cfg`` (to its database unless ``database`` is False)."""
        try:
            import pymysql
        except ImportError as e:            # pragma: no cover - bundled with the app
            raise DatabaseError(tr("MySQL support (PyMySQL) is not installed")) from e
        if not cfg.host or not cfg.user:
            raise DatabaseError(tr("MySQL is chosen in Settings > Database, but the server or user is empty"))
        try:
            con = pymysql.connect(host=cfg.host, port=int(cfg.port), user=cfg.user, password=cfg.password,
                                  database=cfg.database if database else None, charset="utf8mb4",
                                  autocommit=False, connect_timeout=8, read_timeout=120, write_timeout=120,
                                  local_infile=False)
        except pymysql.err.OperationalError as e:
            code = e.args[0] if e.args else 0
            text = e.args[1] if len(e.args) > 1 else str(e)
            if code == 1049:
                raise DatabaseError(tr("The MySQL server has no database '{name}'. Check the name in "
                                       "Settings > Database, or create the database first (CREATE DATABASE ...).",
                                       name=cfg.database)) from e
            if code == 1045:
                raise DatabaseError(tr("MySQL refused the user name or password ({error})", error=text)) from e
            if code in (2003, 2005, 2006, 2013):
                raise DatabaseError(tr("Cannot reach the MySQL server {host}:{port}. Check the server name "
                                       "and port, and that the server is running.",
                                       host=cfg.host, port=cfg.port)) from e
            raise DatabaseError(f"MySQL: {text}") from e
        except pymysql.err.MySQLError as e:
            raise DatabaseError(f"MySQL: {e}") from e
        return cls(con, MYSQL, f"MySQL {cfg.database}@{cfg.host}")

    def close(self) -> None:
        if self.con is not None:
            try:
                self.con.close()
            finally:
                self.con = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def is_mysql(self) -> bool:
        return self.dialect.name == "mysql"

    # --- SQL -------------------------------------------------------------------------------
    def q(self, name: str) -> str:
        return self.dialect.q(name)

    def _sql(self, sql: str) -> str:
        # PyMySQL formats every statement with "%": literal % must be doubled
        return sql if not self.is_mysql else sql.replace("%", "%%").replace("?", "%s")

    def execute(self, sql: str, params=()) -> None:
        cur = self.con.cursor()
        try:
            cur.execute(self._sql(sql), tuple(params))
        finally:
            cur.close()

    def executemany(self, sql: str, rows) -> None:
        rows = list(rows)
        if not rows:
            return
        cur = self.con.cursor()
        try:
            if self.is_mysql:
                for i in range(0, len(rows), 2000):       # keep each statement well below max_allowed_packet
                    cur.executemany(self._sql(sql), rows[i:i + 2000])
            else:
                cur.executemany(sql, rows)
        finally:
            cur.close()

    def query(self, sql: str, params=()) -> list[tuple]:
        cur = self.con.cursor()
        try:
            cur.execute(self._sql(sql), tuple(params))
            return [tuple(r) for r in cur.fetchall()]
        finally:
            cur.close()

    def scalar(self, sql: str, params=(), default=None):
        rows = self.query(sql, params)
        return rows[0][0] if rows and rows[0] else default

    @contextmanager
    def transaction(self):
        """All statements inside are written together, or (on an error) not at all.

        On SQLite the transaction is opened explicitly, so CREATE / DROP / ALTER TABLE are
        part of it too. On MySQL those end a transaction on their own; use
        :meth:`replace_tables` for them."""
        if not self.is_mysql and not self.con.in_transaction:
            self.con.execute("BEGIN")
        try:
            yield self
            self.con.commit()
        except BaseException:
            self.con.rollback()
            raise

    # --- schema ----------------------------------------------------------------------------
    def tables(self) -> list[str]:
        if self.is_mysql:
            return sorted(r[0] for r in self.query("SHOW TABLES"))
        rows = self.query("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                          "ORDER BY name")
        return [r[0] for r in rows]

    def has_table(self, name: str) -> bool:
        return name.lower() in {t.lower() for t in self.tables()}

    def table_name(self, name: str) -> str | None:
        """The stored spelling of a table name (names are matched ignoring case)."""
        return next((t for t in self.tables() if t.lower() == name.lower()), None)

    def ensure_index(self, table: str, name: str, columns: list[str]) -> None:
        """Create the index ``name`` on ``table`` (columns) unless it is there already."""
        cols = ", ".join(self.q(c) for c in columns)
        if self.is_mysql:
            if not self.query(f"SHOW INDEX FROM {self.q(table)} WHERE Key_name = ?", (name,)):
                try:
                    self.execute(f"CREATE INDEX {self.q(name)} ON {self.q(table)} ({cols})")
                except Exception:  # noqa: BLE001 - TEXT columns (e.g. imported) need a key length
                    cols = ", ".join(f"{self.q(c)}(64)" for c in columns)
                    self.execute(f"CREATE INDEX {self.q(name)} ON {self.q(table)} ({cols})")
        else:
            self.execute(f"CREATE INDEX IF NOT EXISTS {self.q(name)} ON {self.q(table)} ({cols})")

    def column_types(self, table: str) -> dict[str, str]:
        """Column name -> declared type (upper case), in table order."""
        if self.is_mysql:
            return {r[0]: str(r[1]).upper() for r in self.query(f"SHOW COLUMNS FROM {self.q(table)}")}
        return {r[1]: str(r[2] or "").upper() for r in self.query(f"PRAGMA table_info({self.q(table)})")}

    def add_column(self, table: str, name: str, kind: str = "real") -> None:
        """Add a column; ``kind`` "real" (numbers) or "text"."""
        t = self.dialect.real if kind == "real" else ("VARCHAR(255)" if self.is_mysql else self.dialect.text)
        self.execute(f"ALTER TABLE {self.q(table)} ADD COLUMN {self.q(name)} {t}")

    def rename_column(self, table: str, old: str, new: str) -> None:
        self.execute(f"ALTER TABLE {self.q(table)} RENAME COLUMN {self.q(old)} TO {self.q(new)}")

    def drop_column(self, table: str, name: str) -> None:
        self.execute(f"ALTER TABLE {self.q(table)} DROP COLUMN {self.q(name)}")

    def columns(self, table: str) -> list[str]:
        if self.is_mysql:
            return [r[0] for r in self.query(f"SHOW COLUMNS FROM {self.q(table)}")]
        return [r[1] for r in self.query(f"PRAGMA table_info({self.q(table)})")]

    # --- whole tables ------------------------------------------------------------------------
    def _create_frame(self, table: str, df: pd.DataFrame) -> None:
        types = [self._column_type(df[c]) for c in df.columns]
        cols = ", ".join(f"{self.q(c)} {t}" for c, t in zip(df.columns, types))
        opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if self.is_mysql else ""
        self.execute(f"DROP TABLE IF EXISTS {self.q(table)}")
        self.execute(f"CREATE TABLE {self.q(table)} ({cols}){opts}")
        if len(df) and len(df.columns):
            marks = ", ".join("?" * len(df.columns))
            names = ", ".join(self.q(c) for c in df.columns)
            values = [[_db_value(v) for v in row] for row in df.itertuples(index=False, name=None)]
            self.executemany(f"INSERT INTO {self.q(table)} ({names}) VALUES ({marks})", values)

    def write_frame(self, table: str, df: pd.DataFrame) -> None:
        """Replace one table with the rows of ``df`` (column names and order kept)."""
        self.replace_tables({table: df}, drop_prefix=None)

    def replace_tables(self, frames: dict[str, pd.DataFrame], drop_prefix: str | None) -> None:
        """Swap in a set of tables at once: afterwards the database holds exactly ``frames``
        among the tables starting with ``drop_prefix`` (others are not touched).

        SQLite: one transaction. MySQL (where CREATE / DROP commit on their own): the new
        tables are written under temporary names and then swapped in with a single
        RENAME TABLE statement, which MySQL runs as one step."""
        old = [t for t in self.tables() if drop_prefix and t.lower().startswith(drop_prefix.lower())]
        if not self.is_mysql:
            with self.transaction():
                for t in old:
                    self.execute(f"DROP TABLE IF EXISTS {self.q(t)}")
                for name, df in frames.items():
                    self._create_frame(name, df)
            return
        stamp = _dt.datetime.now().strftime("%H%M%S")
        staged = {name: f"gvnew{stamp}_{i}" for i, name in enumerate(frames)}
        try:
            for name, df in frames.items():
                self._create_frame(staged[name], df)
            self.con.commit()
            existing = {t.lower(): t for t in self.tables()}
            retired = {}
            for i, t in enumerate(sorted(set(old) | {n for n in frames if n.lower() in existing})):
                retired[existing.get(t.lower(), t)] = f"gvold{stamp}_{i}"
            pairs = [f"{self.q(a)} TO {self.q(b)}" for a, b in retired.items()]
            pairs += [f"{self.q(staged[n])} TO {self.q(n)}" for n in frames]
            self.execute("RENAME TABLE " + ", ".join(pairs))
            for t in retired.values():
                self.execute(f"DROP TABLE IF EXISTS {self.q(t)}")
            self.con.commit()
        except BaseException:
            for t in staged.values():
                try:
                    self.execute(f"DROP TABLE IF EXISTS {self.q(t)}")
                except Exception:  # noqa: BLE001
                    pass
            raise

    def read_frame(self, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        """The rows of ``table`` (only ``columns`` when given)."""
        cols = "*" if columns is None else ", ".join(self.q(c) for c in columns)
        cur = self.con.cursor()
        try:
            cur.execute(f"SELECT {cols} FROM {self.q(table)}")
            names = [d[0] for d in cur.description]
            return pd.DataFrame.from_records([tuple(r) for r in cur.fetchall()], columns=names)
        finally:
            cur.close()

    def _column_type(self, s: pd.Series) -> str:
        d = self.dialect
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
            return d.integer
        if pd.api.types.is_float_dtype(s):
            return d.real
        if pd.api.types.is_datetime64_any_dtype(s):
            return d.text
        vals = [v for v in s if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals and all(isinstance(v, (bool, int, np.integer)) for v in vals):
            return d.integer
        if vals and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            return d.real
        if vals and all(isinstance(v, (bytes, bytearray, memoryview)) for v in vals):
            return d.blob
        return d.text


def _db_value(v):
    """A Python value the database driver accepts (NaN / NaT -> NULL, dates -> ISO text)."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
        return None if pd.isna(v) else v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview, int, str)):
        return v
    if v is pd.NaT:
        return None
    return str(v)


# --- helpers -------------------------------------------------------------------------------------
def is_sqlite_file(path: str | Path) -> bool:
    """True for an SQLite database file (checks the 16-byte header, not only the name)."""
    try:
        with open(path, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def backup_sqlite(src: str | Path, dst: str | Path) -> None:
    """Consistent copy of a live SQLite database (safe while another connection writes)."""
    a = sqlite3.connect(str(src), timeout=15.0)
    b = sqlite3.connect(str(dst))
    try:
        a.backup(b)
    finally:
        b.close()
        a.close()


def backup_sqlite_tables(src: str | Path, dst: str | Path, prefix: str) -> None:
    """Copy only the tables starting with ``prefix`` (with their indexes) of an SQLite
    database into a new file ``dst`` (replaced). Unlike a copy of the whole file it stays
    small and quick however big the other tables (e.g. imported drillholes) grow."""
    import os

    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(str(tmp))
    try:
        con.execute("ATTACH DATABASE ? AS src", (str(src),))
        like = prefix.replace("_", "\\_") + "%"
        items = con.execute("SELECT type, name, sql FROM src.sqlite_master WHERE type IN ('table', 'index') "
                            "AND tbl_name LIKE ? ESCAPE '\\' AND sql IS NOT NULL ORDER BY type DESC",
                            (like,)).fetchall()
        with con:
            for kind, name, sql in items:                  # tables first ('table' > 'index'), then indexes
                con.execute(sql)
                if kind == "table":
                    con.execute(f'INSERT INTO "{name}" SELECT * FROM src."{name}"')
        con.execute("DETACH DATABASE src")
    finally:
        con.close()
    os.replace(tmp, dst)


# --- change tracking --------------------------------------------------------------------------------
# table_updates holds a version per table; triggers (made by the set-up) add 1 on every insert,
# update or delete - by GeoViewer, an import, a SQL tool or another computer. Refresh compares
# the versions (one small query) to know which data changed.
UPDATES_TABLE = "table_updates"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_]{1,50}$")


def ensure_change_tracking(db: "Database", tables: list[str]) -> list[str]:
    """Make table_updates and the triggers of ``tables`` where missing (only adds).
    Returns what was done."""
    notes = []
    opts = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if db.is_mysql else ""
    if not db.has_table(UPDATES_TABLE):
        db.execute(f"CREATE TABLE {UPDATES_TABLE} (table_name VARCHAR(64) NOT NULL PRIMARY KEY, "
                   f"version BIGINT NOT NULL DEFAULT 0, updated_at VARCHAR(19)){opts}")
        db.con.commit()
        notes.append(f"made table {UPDATES_TABLE}")
    tables = [t for t in tables if _SAFE_NAME.match(t)]
    have = {str(r[0]).lower() for r in db.query(f"SELECT table_name FROM {UPDATES_TABLE}")}
    with db.transaction():
        for t in tables:
            if t.lower() not in have:
                db.execute(f"INSERT INTO {UPDATES_TABLE} (table_name, version, updated_at) VALUES (?, 0, NULL)", (t,))
    if db.is_mysql:
        triggers = {str(r[0]).lower() for r in db.query(
            "SELECT TRIGGER_NAME FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA = DATABASE()")}
        now = "NOW()"
    else:
        triggers = {str(r[0]).lower() for r in db.query("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
        now = "datetime('now', 'localtime')"
    made = 0
    for t in tables:
        bump = f"UPDATE {UPDATES_TABLE} SET version = version + 1, updated_at = {now} WHERE table_name = '{t}'"
        for event, tag in (("INSERT", "ai"), ("UPDATE", "au"), ("DELETE", "ad")):
            name = f"gv_{t}_{tag}"
            if name.lower() in triggers:
                continue
            if db.is_mysql:
                db.execute(f"CREATE TRIGGER {db.q(name)} AFTER {event} ON {db.q(t)} FOR EACH ROW {bump}")
            else:
                db.execute(f"CREATE TRIGGER IF NOT EXISTS {db.q(name)} AFTER {event} ON {db.q(t)} BEGIN {bump}; END")
            made += 1
    db.con.commit()
    if made:
        notes.append(f"added change tracking ({made} triggers)")
    return notes


def change_signature(db: "Database", prefix: str) -> tuple | None:
    """What changes when the data or the layout of the tables starting with ``prefix``
    changes: their table_updates versions plus the table layout. None without table_updates."""
    if not db.has_table(UPDATES_TABLE):
        return None
    versions = tuple(sorted((str(n).lower(), int(v)) for n, v in db.query(
        f"SELECT table_name, version FROM {UPDATES_TABLE}") if str(n).lower().startswith(prefix)))
    if db.is_mysql:
        try:
            db.execute("SET SESSION information_schema_stats_expiry = 0")
        except Exception:  # noqa: BLE001 - older servers: no such setting
            pass
        like = prefix.replace("_", "\\_") + "%"
        layout = tuple(sorted((str(n).lower(), str(c)) for n, c in db.query(
            "SELECT TABLE_NAME, CREATE_TIME FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() "
            "AND TABLE_NAME LIKE ?", (like,))))
        cols = tuple(sorted((str(n).lower(), str(c)) for n, c in db.query(
            "SELECT TABLE_NAME, GROUP_CONCAT(COLUMN_NAME ORDER BY ORDINAL_POSITION) FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE ? GROUP BY TABLE_NAME", (like,))))
        return versions, layout, cols
    return versions, db.scalar("PRAGMA schema_version")


def check_connection(cfg: DatabaseConfig, brief: bool = False) -> str:
    """Connect with ``cfg`` and describe what is there; raises DatabaseError (user message).
    ``brief``: only the first sentence ("Connected to MySQL 8.0.36.")."""
    if not cfg.is_mysql:
        return tr("SQLite needs no connection: the file is kept in the workspace folder.")
    with Database.mysql(cfg) as db:
        version = db.scalar("SELECT VERSION()")
        tables = db.tables()
    dh = sum(1 for t in tables if t.lower().startswith(DRILLHOLE_PREFIX))
    prod = sum(1 for t in tables if t.lower().startswith(PROD_PREFIX))
    connected = tr("Connected to MySQL {version}.", version=version)
    if brief:
        return connected
    if not dh and not prod:
        what = tr("no GeoViewer tables yet (they are made when a workspace is opened)")
    else:
        what = tr("{dh} drillhole table(s), {prod} production table(s)", dh=dh, prod=prod)
    return connected + " " + tr("Database {name}: {what}.", name=cfg.database, what=what)


def create_mysql_database(cfg: DatabaseConfig) -> None:
    """CREATE DATABASE for ``cfg.database`` (utf8mb4); the user needs the CREATE privilege."""
    name = cfg.database.strip()
    if not name:
        raise DatabaseError(tr("Type a database name first"))
    with Database.mysql(cfg, database=False) as db:
        try:
            db.execute(f"CREATE DATABASE IF NOT EXISTS {db.q(name)} "
                       "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            db.con.commit()
        except Exception as e:  # noqa: BLE001
            raise DatabaseError(tr("Could not create the database {name}: {error}", name=name, error=e)) from e


__all__ = ["BACKENDS", "DRILLHOLE_PREFIX", "PROD_PREFIX", "PROJECT_DB", "Database", "DatabaseConfig",
           "DatabaseError", "Target", "as_target", "backup_sqlite", "configure", "create_mysql_database",
           "current_config", "is_sqlite_file", "project_target", "set_project_folder", "check_connection"]
