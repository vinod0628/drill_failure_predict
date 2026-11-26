"""Licensing: signed keys verify, forged / tampered / wrong-PC / expired keys are rejected,
and the machine code round-trips. Uses a throwaway keypair (not the shipped one) by patching
the embedded public key, so the tests do not depend on the vendor key."""

import base64
import datetime as dt
import hashlib

import pytest

from dfpredict import licensing as lic


@pytest.fixture
def vendor(monkeypatch):
    """A fresh keypair; the app verifies with this public key for the duration of the test."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    monkeypatch.setattr(lic, "_PUBKEY", raw)
    monkeypatch.setattr(lic, "_PUBKEY_SHA256", hashlib.sha256(raw).hexdigest())
    return priv


def _issue(priv, **kw):
    base = dict(name="Acme", issued=lic._today(), expires=lic._today() + dt.timedelta(days=30))
    base.update(kw)
    return lic.sign_payload(priv, lic.License(**base))


def test_a_signed_key_round_trips_with_every_field(vendor):
    key = _issue(vendor, name="Acme Mining Ltd", edition=2,
                 features=lic.FEAT_DATABASE | lic.FEAT_EXPORT, expires=dt.date(2030, 6, 1))
    lc = lic.decode(key)
    assert lc.name == "Acme Mining Ltd" and lc.edition_name == "Ultimate"
    assert lc.has(lic.FEAT_DATABASE) and lc.has(lic.FEAT_EXPORT)
    assert lc.expires == dt.date(2030, 6, 1) and lc.machine == b""
    assert key.startswith("GEOV-") and key.count("-") >= 2        # grouped, no base32 '-' collisions


def test_edition_tiers_unlock_the_right_features(vendor):
    """Basic = view only; Professional = edit the database; Ultimate = + export."""
    assert lic.edition_features(0) == 0
    assert lic.edition_features(1) == lic.FEAT_DATABASE
    assert lic.edition_features(2) == lic.FEAT_DATABASE | lic.FEAT_EXPORT
    basic = lic.decode(_issue(vendor, edition=0, features=lic.edition_features(0)))
    assert basic.edition_name == "Basic" and not basic.has(lic.FEAT_DATABASE) and not basic.has(lic.FEAT_EXPORT)
    pro = lic.decode(_issue(vendor, edition=1, features=lic.edition_features(1)))
    assert pro.has(lic.FEAT_DATABASE) and not pro.has(lic.FEAT_EXPORT)


def test_a_forged_key_is_rejected(vendor):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    attacker = Ed25519PrivateKey.generate()                       # a different private key
    key = _issue(attacker)
    with pytest.raises(lic.LicenseError):
        lic.decode(key)
    # garbage and truncated keys too
    for bad in ("", "GEOV-NOTAREALKEY", "hello", "GEOV-" + "A" * 200):
        with pytest.raises(lic.LicenseError):
            lic.decode(bad)


def test_a_tampered_key_is_rejected(vendor):
    key = _issue(vendor, name="Acme")
    body = key.replace("-", "")[len("GEOV"):]
    flipped = body[:30] + ("A" if body[30] != "A" else "B") + body[31:]   # change one char of the payload
    with pytest.raises(lic.LicenseError):
        lic.decode("GEOV-" + flipped)


def test_swapping_the_public_key_is_detected(vendor, monkeypatch):
    """Replacing the embedded key without its checksum is caught (a casual key swap)."""
    monkeypatch.setattr(lic, "_PUBKEY", b"\x01" * 32)             # checksum no longer matches
    with pytest.raises(lic.LicenseError):
        lic.decode(_issue(vendor))


def test_expiry_is_enforced(vendor):
    valid = _issue(vendor, expires=lic._today() + dt.timedelta(days=1))
    assert lic.check(valid).ok
    expired = _issue(vendor, expires=lic._today() - dt.timedelta(days=1))
    st = lic.check(expired)
    assert not st.ok and "expired" in st.reason.lower()
    # a perpetual key must be machine-locked (an unbound perpetual key is refused - see below)
    perpetual = _issue(vendor, expires=None, machine=lic.machine_id())
    assert lic.check(perpetual).ok and lic.decode(perpetual).days_left() is None


def test_machine_binding(vendor, monkeypatch):
    code = lic.machine_code()
    assert len(code) == 23 and code.count("-") == 3              # XXXXX-XXXXX-XXXXX-XXXXX
    bound = _issue(vendor, machine=lic.machine_id_from_code(code))
    assert lic.check(bound).ok                                   # this PC
    # another PC: a different hardware id -> a different machine_id
    monkeypatch.setattr(lic, "_hardware_id", lambda: b"another-machine")
    st = lic.check(bound)
    assert not st.ok and "another computer" in st.reason.lower()
    # an unbound key still runs anywhere
    assert lic.check(_issue(vendor, machine=b"")).ok


def test_machine_code_is_stable_and_hardware_based(vendor, monkeypatch):
    """The same hardware gives the same code every time; different hardware gives a different
    one. It must not use an OS-install value (so it survives an OS reinstall)."""
    monkeypatch.setattr(lic, "_hardware_id", lambda: b"HW-UUID-1234")
    first = lic.machine_code()
    assert lic.machine_code() == first                          # stable
    monkeypatch.setattr(lic, "_hardware_id", lambda: b"HW-UUID-9999")
    assert lic.machine_code() != first                          # per machine
    import inspect
    src = inspect.getsource(lic._hardware_id) + inspect.getsource(lic._machine_raw)
    assert "MachineGuid" not in src                             # not the OS-install guid


def test_keygen_project_is_compatible_with_the_app():
    """The standalone keygen/ project signs keys that THIS app accepts (its license_format and
    embedded public key match the app's). Skipped if the vendor private key is not present."""
    import importlib.util
    import sys
    from pathlib import Path

    from dfpredict import licensing

    kg_dir = Path(__file__).resolve().parents[1] / "keygen"
    pem = kg_dir / "vendor_private_key.pem"
    if not pem.is_file():
        import pytest as _pt
        _pt.skip("keygen vendor_private_key.pem not present (kept securely by the vendor)")

    def _load(name):
        spec = importlib.util.spec_from_file_location(name, kg_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    lf = _load("license_format")
    # the keygen's embedded public key must equal the app's (same vendor keypair)
    assert lf._PUBKEY == licensing._PUBKEY and lf._PUBKEY_SHA256 == licensing._PUBKEY_SHA256
    # a key the keygen signs is accepted by the app, with the right tier / features
    key = lf.sign_payload(lf.load_private_key(pem),
                          lf.License(name="Compat Co", issued=lf._today(), expires=None,
                                     edition=2, features=lf.edition_features(2)))
    lic = licensing.decode(key)
    assert lic.name == "Compat Co" and lic.edition_name == "Ultimate"
    assert lic.has(licensing.FEAT_DATABASE) and lic.has(licensing.FEAT_EXPORT)


def test_unbound_any_computer_license_must_be_short(vendor):
    """An unbound ("any computer") license is accepted only up to ~1 year; a long one - even
    correctly signed, e.g. issued by mistake - is refused so the app would close."""
    import datetime as dt

    one_year = _issue(vendor, machine=b"", expires=lic._today() + dt.timedelta(days=300))
    assert lic.check(one_year).ok
    too_long = _issue(vendor, machine=b"", expires=lic._add_years(lic._today(), 5))
    st = lic.check(too_long)
    assert not st.ok and "too long" in st.reason.lower()
    # an unbound perpetual key is never accepted either
    assert not lic.check(_issue(vendor, machine=b"", expires=None)).ok
    # a MACHINE-LOCKED key may be long (e.g. Basic up to many years) - the limit is only for unbound
    bound_long = _issue(vendor, machine=lic.machine_id(), expires=lic._add_years(lic._today(), 5))
    assert lic.check(bound_long).ok


def test_keygen_caps_unbound_at_one_year():
    """The keygen itself will not issue an unbound key for longer than 1 year, whatever the
    edition (defence in depth with the app check)."""
    import importlib.util
    import sys
    from pathlib import Path

    kg_dir = Path(__file__).resolve().parents[1] / "keygen"
    if not (kg_dir / "vendor_private_key.pem").is_file():
        import pytest as _pt
        _pt.skip("keygen private key not present")
    spec = importlib.util.spec_from_file_location("license_format", kg_dir / "license_format.py")
    lf = importlib.util.module_from_spec(spec)
    sys.modules["license_format"] = lf
    spec.loader.exec_module(lf)
    assert lf.issue_max_years(0, unbound=False) == 30        # bound Basic: 30 years
    assert lf.issue_max_years(2, unbound=False) == 5         # bound Ultimate: 5 years
    assert lf.issue_max_years(0, unbound=True) == 1          # unbound Basic: 1 year
    assert lf.issue_max_years(2, unbound=True) == 1          # unbound Ultimate: 1 year
