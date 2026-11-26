"""Keeping a password in the settings without storing it as plain text.

On Windows the password is encrypted with DPAPI (CryptProtectData) for the current
Windows user: only that user, on that computer, can read it back. Elsewhere (tests on
other systems) it is only base64-encoded, marked with a different prefix.
"""

from __future__ import annotations

import base64
import ctypes
import sys

_DPAPI = "dpapi:"
_PLAIN = "b64:"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _Blob:
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def _dpapi(data: bytes, encrypt: bool) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    src, out = _blob(data), _Blob()
    fn = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(src), None, None, None, None, 0x01, ctypes.byref(out))   # 0x01: no UI
    if not ok:
        raise OSError("Windows could not encrypt / decrypt the password")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def protect(password: str) -> str:
    if not password:
        return ""
    raw = password.encode("utf-8")
    if sys.platform == "win32":
        return _DPAPI + base64.b64encode(_dpapi(raw, True)).decode("ascii")
    return _PLAIN + base64.b64encode(raw).decode("ascii")


def unprotect(text: str) -> str:
    """The password, or "" when it cannot be read (e.g. settings copied from another user)."""
    try:
        if text.startswith(_DPAPI) and sys.platform == "win32":
            return _dpapi(base64.b64decode(text[len(_DPAPI):]), False).decode("utf-8")
        if text.startswith(_PLAIN):
            return base64.b64decode(text[len(_PLAIN):]).decode("utf-8")
    except Exception:  # noqa: BLE001
        return ""
    return ""
