from pathlib import Path

import pytest

# GeoViewer's regression tests (inherited) read GeoViewer's demo files, kept with the tests (tests/data/geoviewer_demo).
# The drilling-risk tests use demo_data/sb3_workspace of this project.
DEMO = Path(__file__).resolve().parent / "data" / "geoviewer_demo"
RISK_DEMO = Path(__file__).resolve().parents[1] / "demo_data" / "sb3_workspace"


@pytest.fixture(scope="session")
def demo() -> Path:
    if not DEMO.exists():
        pytest.skip("demo_data folder not present")
    return DEMO


@pytest.fixture(autouse=True)
def _default_database():
    """Every test starts on SQLite with no workspace folder (the app's default is MySQL, which
    tests use only when asked): the setting is app-wide state that a test may change."""
    from dfpredict.io import database

    database.configure(database.DatabaseConfig(backend="sqlite"))
    database.set_project_folder(None)
    yield
    database.configure(database.DatabaseConfig(backend="sqlite"))
    database.set_project_folder(None)


@pytest.fixture(scope="session")
def _test_cache_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("geoviewer_cache")


@pytest.fixture(autouse=True)
def _private_cache(_test_cache_dir, monkeypatch):
    r"""Block model caches made by the tests go to a temporary folder, never into the user's
    %LOCALAPPDATA%\GeoViewer\cache (every test copies the demo files to a new folder, and each
    new path would leave another cache file there)."""
    from dfpredict.io import cache

    monkeypatch.setattr(cache, "default_cache_dir", lambda: _test_cache_dir)
    yield


@pytest.fixture(autouse=True)
def _license_in_tests(monkeypatch):
    """The feature gates and the startup check ask ``licensing.current()`` / ``refresh()``.
    In the tests, pretend this PC has a full, valid license so GUI tests reach the features.
    This injection is in-process only (it is not in the shipped app)."""
    from datetime import date

    from dfpredict import licensing

    lic = licensing.License(name="Test", issued=date(2026, 1, 1), expires=None,
                            edition=2, features=licensing.FEAT_DATABASE | licensing.FEAT_EXPORT)
    ok = licensing.Status(True, lic, "", None)
    monkeypatch.setattr(licensing, "current", lambda: ok)
    monkeypatch.setattr(licensing, "refresh", lambda: ok)
    yield


@pytest.fixture(autouse=True)
def _english():
    """Tests read the English texts: whatever language this PC's GeoViewer is set to."""
    from dfpredict import i18n

    i18n.set_language("en")
    yield
    i18n.set_language("en")
