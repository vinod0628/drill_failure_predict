"""Licensing: the app verifies a signed license key against an embedded public key.

The vendor holds the private key (see ``tools/keygen.py``) and signs each license. The
app ships only the public key, so a valid key cannot be forged without the private key.
The signature is Ed25519 (short, fast, unforgeable).

A license key carries: licensee name, issue and expiry dates, edition, feature flags and
an optional machine binding (node-lock). The app decodes it, checks the signature, the
expiry and - when bound - the machine code of this PC.

Honest note: this makes keys unforgeable, but no client-side check in a Python program can
be made uncrackable - a determined attacker can still patch the running code. For stronger
protection ship the build through a bytecode obfuscator (PyArmor) or a compiler (Cython /
Nuitka); the signature scheme here is what makes casual key-sharing and key-gen impossible.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from .i18n import tr

# --- the embedded public key (the private half never ships) -----------------------------------
# Split so a search-and-replace swap of the key must match every piece and the checksum below.
_PK_A = "4eze/OE/erKh6oET"
_PK_B = "079YcpfnWSzUd3GloHeLlLMmw4Y="
_PUBKEY = base64.b64decode(_PK_A + _PK_B)
# sha256 of the public key: changing the key without changing this breaks verification.
_PUBKEY_SHA256 = "f654bd527cf477f77bd9653813af7e7c555a48eb6680865cad01eec1984d17c8"

MAGIC = b"GV1"                       # license payload version tag
PREFIX = "GEOV"                      # key string prefix
# editions (tiers): Basic = view only, Professional = + edit the database, Ultimate = + export
EDITIONS = {0: "Basic", 1: "Professional", 2: "Ultimate"}
# feature bits the app gates on (an edition is a tier that turns the right bits on)
FEAT_DATABASE = 1 << 0               # add / edit / remove the project database (drillholes, survey, production)
FEAT_EXPORT = 1 << 1                 # export / print maps and sections
FEATURES = {FEAT_DATABASE: "database", FEAT_EXPORT: "export"}
EDITION_FEATURES = {0: 0, 1: FEAT_DATABASE, 2: FEAT_DATABASE | FEAT_EXPORT}


def edition_features(edition: int) -> int:
    """The features a tier unlocks (Basic none, Professional database, Ultimate database + export)."""
    return EDITION_FEATURES.get(edition, 0)
EPOCH = date(2020, 1, 1)             # license dates are days since here (2 bytes = ~179 years)
# An unbound ("any computer") license is kept short: even if a long one is issued by mistake,
# the app refuses it (a long shareable key is a leak risk). A few days' grace for leap/edge.
UNBOUND_MAX_YEARS = 1
UNBOUND_GRACE_DAYS = 7

# Crockford base32 for the machine code (no I L O U: no reading confusion).
_C32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_C32_MAP = {c: i for i, c in enumerate(_C32)}
_C32_MAP.update({"I": 1, "L": 1, "O": 0, "U": 0})     # forgive the look-alikes on input


class LicenseError(Exception):
    """A key that is missing, malformed, not signed by the vendor, expired or for another PC."""


def _public_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if hashlib.sha256(_PUBKEY).hexdigest() != _PUBKEY_SHA256:        # the key was tampered with
        raise LicenseError(tr("licensing is corrupted"))
    return Ed25519PublicKey.from_public_bytes(_PUBKEY)


def product_salt() -> bytes:
    """Ties the machine code and ids to this product (and its public key)."""
    return hashlib.sha256(b"geoviewer-license/" + _PUBKEY).digest()


# --- base32 (Crockford) -----------------------------------------------------------------------
def b32_encode(data: bytes) -> str:
    bits = int.from_bytes(data, "big")
    width = ((len(data) * 8 + 4) // 5)
    out = []
    for i in range(width - 1, -1, -1):
        out.append(_C32[(bits >> (i * 5)) & 0x1F])
    return "".join(out)


def b32_decode(text: str, nbytes: int) -> bytes:
    bits = 0
    for ch in text:
        if ch in "-_ ":
            continue
        up = ch.upper()
        if up not in _C32_MAP:
            raise LicenseError(f"bad character {ch!r} in the code")
        bits = (bits << 5) | _C32_MAP[up]
    return bits.to_bytes(max(nbytes, (bits.bit_length() + 7) // 8), "big")[-nbytes:]


# --- machine fingerprint ----------------------------------------------------------------------
# The id must survive an OS reinstall, so it comes from the HARDWARE (SMBIOS), not from any
# OS-install value (Windows MachineGuid is regenerated on reinstall - never used here).
def _smbios_system_uuid() -> bytes | None:
    """The SMBIOS System UUID (set by the maker on the board) - stable across OS reinstalls."""
    try:
        import ctypes

        k = ctypes.windll.kernel32
        k.GetSystemFirmwareTable.restype = ctypes.c_uint
        sig = 0x52534D42                                  # 'RSMB'
        size = k.GetSystemFirmwareTable(sig, 0, None, 0)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        k.GetSystemFirmwareTable(sig, 0, buf, size)
        table = buf.raw[8:]                               # skip RawSMBIOSData header
        i, n = 0, len(table)
        while i + 4 < n:
            dtype, length = table[i], table[i + 1]
            if dtype == 1 and length >= 0x19:             # System Information: UUID at +0x08, 16 B
                u = table[i + 8:i + 24]
                if u not in (b"\x00" * 16, b"\xff" * 16):
                    return u
            if dtype == 127:                              # end-of-table
                break
            j = i + length                                # skip the strings (double-null ends them)
            while j + 1 < n and not (table[j] == 0 and table[j + 1] == 0):
                j += 1
            i = j + 2
        return None
    except Exception:  # noqa: BLE001 - not Windows, or no firmware table
        return None


def _firmware_serials() -> bytes | None:
    """Baseboard / BIOS / system serials - also hardware, survive an OS reinstall. One PowerShell
    call; used only when the SMBIOS UUID is unavailable."""
    try:
        import subprocess

        ps = ("$ErrorActionPreference='SilentlyContinue';"
              "(Get-CimInstance Win32_BaseBoard).SerialNumber+'|'+"
              "(Get-CimInstance Win32_BIOS).SerialNumber+'|'+"
              "(Get-CimInstance Win32_ComputerSystemProduct).UUID")
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        cleaned = out.replace("|", "").replace("To be filled by O.E.M.", "").replace("Default string", "").strip()
        return out.encode() if len(cleaned) >= 4 else None
    except Exception:  # noqa: BLE001
        return None


def _hardware_id() -> bytes:
    """A stable per-PC hardware id that survives an OS reinstall."""
    return (_smbios_system_uuid() or _firmware_serials()
            or struct.pack(">Q", __import__("uuid").getnode()))     # last resort: the primary MAC


def _machine_raw() -> bytes:
    return hashlib.sha256(product_salt() + b"/machine/" + _hardware_id()).digest()


def machine_id() -> bytes:
    """The 8 bytes a bound license stores (derivable from the machine code)."""
    return _machine_raw()[:8]


def machine_code() -> str:
    """This PC's machine code, XXXXX-XXXXX-XXXXX-XXXXX - what a customer sends the vendor."""
    code = b32_encode(_machine_raw()[:12])              # 96 bits -> 20 chars
    return "-".join(code[i:i + 5] for i in range(0, 20, 5))


def machine_id_from_code(code: str) -> bytes:
    """The 8-byte binding id a keygen stores, from a typed machine code (dashes/case/look-alikes
    forgiven)."""
    cleaned = "".join(ch for ch in code if ch not in "-_ ")
    if len(cleaned) != 20:
        raise LicenseError("a machine code is 20 characters (XXXXX-XXXXX-XXXXX-XXXXX)")
    return b32_decode(cleaned, 12)[:8]


# --- the license ------------------------------------------------------------------------------
@dataclass
class License:
    name: str
    issued: date
    expires: date | None            # None = perpetual
    edition: int = 1
    features: int = FEAT_DATABASE | FEAT_EXPORT
    machine: bytes = b""            # b"" = runs on any PC; else the 8-byte binding id

    @property
    def edition_name(self) -> str:
        return EDITIONS.get(self.edition, f"edition {self.edition}")

    @property
    def feature_names(self) -> list[str]:
        return [n for bit, n in FEATURES.items() if self.features & bit]

    def has(self, feature: int) -> bool:
        return bool(self.features & feature)

    def days_left(self, today: date | None = None) -> int | None:
        if self.expires is None:
            return None
        return (self.expires - (today or _today())).days

    def is_expired(self, today: date | None = None) -> bool:
        return self.expires is not None and (today or _today()) > self.expires

    def bound_to_this_pc(self) -> bool:
        return not self.machine or self.machine == machine_id()


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _pack(lic: License) -> bytes:
    name = lic.name.encode("utf-8")[:255]
    machine = lic.machine if lic.machine else b"\x00" * 8
    return (MAGIC
            + struct.pack(">HHBB", (lic.issued - EPOCH).days,
                          0 if lic.expires is None else (lic.expires - EPOCH).days,
                          lic.edition & 0xFF, lic.features & 0xFF)
            + machine + struct.pack(">B", len(name)) + name)


def _unpack(payload: bytes) -> License:
    if payload[:3] != MAGIC:
        raise LicenseError(tr("not a GeoViewer license"))
    issued_d, expires_d, edition, features = struct.unpack(">HHBB", payload[3:9])
    machine = payload[9:17]
    (nlen,) = struct.unpack(">B", payload[17:18])
    name = payload[18:18 + nlen].decode("utf-8", "replace")
    if len(payload) != 18 + nlen:
        raise LicenseError(tr("the key is damaged"))
    return License(name=name, issued=EPOCH + _days(issued_d),
                   expires=None if expires_d == 0 else EPOCH + _days(expires_d),
                   edition=edition, features=features,
                   machine=b"" if machine == b"\x00" * 8 else machine)


def _days(n: int):
    from datetime import timedelta

    return timedelta(days=n)


def sign_payload(private_key, lic: License) -> str:
    """Make a key string for ``lic`` (the keygen holds the private key). base32 is used so the
    key has no '-' of its own (the dashes only group it)."""
    payload = _pack(lic)
    sig = private_key.sign(payload)
    blob = base64.b32encode(payload + sig).decode().rstrip("=")
    return PREFIX + "-" + "-".join(blob[i:i + 20] for i in range(0, len(blob), 20))


def decode(key: str) -> License:
    """Verify a key string and return its License, or raise LicenseError."""
    key = (key or "").strip()
    if not key:
        raise LicenseError(tr("no license key"))
    body = key[len(PREFIX) + 1:] if key.upper().startswith(PREFIX + "-") else key
    blob = body.replace("-", "").replace(" ", "").replace("\n", "").upper()
    try:
        data = base64.b32decode(blob + "=" * (-len(blob) % 8))
    except Exception as e:  # noqa: BLE001
        raise LicenseError(tr("the key is not readable")) from e
    if len(data) < 64 + 18:
        raise LicenseError(tr("the key is too short"))
    payload, sig = data[:-64], data[-64:]
    from cryptography.exceptions import InvalidSignature

    try:
        _public_key().verify(sig, payload)
    except InvalidSignature as e:
        raise LicenseError(tr("the key is not valid (wrong signature)")) from e
    return _unpack(payload)


# --- the app's view of the current license ----------------------------------------------------
@dataclass
class Status:
    ok: bool
    license: License | None
    reason: str = ""                # why not ok (for the dialog / log)
    source: Path | None = None

    @property
    def summary(self) -> str:
        if not self.ok:
            return self.reason or "not licensed"
        lic = self.license
        left = lic.days_left()
        when = "perpetual" if left is None else (f"{left} days left" if left >= 0 else "expired")
        return f"{lic.name} - {lic.edition_name} ({when})"


def license_path() -> Path:
    """Where the app keeps the installed license."""
    import os

    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "GeoViewer" / "license.key"


def _search_paths() -> list[Path]:
    import os
    import sys

    out = []
    env = os.environ.get("GEOVIEWER_LICENSE")
    if env:
        out.append(Path(env))
    out.append(license_path())
    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
    out.append(exe_dir / "license.key")
    return out


def read_installed() -> tuple[str, Path | None]:
    for p in _search_paths():
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8").strip(), p
        except OSError:
            continue
    return "", None


def check(key: str | None = None) -> Status:
    """The status of a key (the installed one when ``key`` is None): signed, not expired, and
    for this PC."""
    src = None
    if key is None:
        key, src = read_installed()
        if not key:
            return Status(False, None, tr("No license installed"), None)
    try:
        lic = decode(key)
    except LicenseError as e:
        return Status(False, None, str(e), src)
    if lic.is_expired():
        return Status(False, lic, tr("The license expired on {date:%Y-%m-%d}", date=lic.expires), src)
    if not lic.machine and not _unbound_ok(lic):      # a too-long "any computer" key: refuse it
        return Status(False, lic, tr("This any-computer license is valid for too long "
                                     "(an unlocked license may run at most {n} year); "
                                     "ask the vendor for a machine-locked key", n=UNBOUND_MAX_YEARS), src)
    if not lic.bound_to_this_pc():
        return Status(False, lic, tr("This license is for another computer"), src)
    return Status(True, lic, "", src)


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:                                # 29 Feb
        return d.replace(month=2, day=28, year=d.year + years)


def _unbound_ok(lic: "License") -> bool:
    """An unbound license must have an expiry within ~1 year of its issue date."""
    if lic.expires is None:
        return False                                  # perpetual unbound key: never allowed
    cap = _add_years(lic.issued, UNBOUND_MAX_YEARS) + _days(UNBOUND_GRACE_DAYS)
    return lic.expires <= cap


def install(key: str) -> Status:
    """Verify a key and, if good, save it as the installed license. Raises LicenseError when
    the key itself is bad (so the dialog can show why); an expired / wrong-PC key is saved but
    reported as not ok."""
    lic = decode(key)                                   # raises on a forged / unreadable key
    path = license_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".key.tmp")
    tmp.write_text(key.strip() + "\n", encoding="utf-8")
    tmp.replace(path)
    del lic
    return check(key)


# cached status for the running app (set at startup; the dialog refreshes it)
_status: Status | None = None


def current() -> Status:
    global _status
    if _status is None:
        _status = check()
    return _status


def refresh() -> Status:
    global _status
    _status = check()
    return _status
