"""Translations of the user interface (i18n).

The English text in the code is the key: ``tr("Open files")`` returns that text in the current
language, or the English itself when no translation exists. Each other language is ONE JSON file
in ``geoviewer/i18n/locales/`` named by its code (``zh_CN.json``)::

    {
      "language": {"code": "zh_CN", "name": "简体中文", "english_name": "Chinese (Simplified)",
                   "qt": "zh_CN", "fonts": ["msyh.ttc", "simhei.ttf"]},
      "messages": {"Open files": "打开文件", "{n:,} holes": "{n:,} 个钻孔", ...}
    }

* ``name`` is shown in the language list (in that language); ``qt`` names Qt's own translation
  of its standard buttons and dialogs (``qtbase_<qt>.qm``); ``fonts`` are font files (searched in
  this folder, then in Windows' Fonts folder) for text drawn by VTK, whose built-in fonts have
  Latin letters only.
* An empty or missing message falls back to English, so a half-translated file works.
* Add a language: ``python tools/i18n_extract.py --new fr --name Français --english-name French``
  writes ``locales/fr.json`` with every message empty; fill it in. ``--check`` lists what is
  missing or wrong; ``--update`` adds new messages to every file after code changes.

Rules for the code:

* ``tr("text")`` where the text is shown; placeholders are named and passed as keywords:
  ``tr("Setup {n} - instrument at {at}", n=i + 1, at=row.at)``. With keywords the text is
  ``str.format``-ed, so literal braces are doubled there. Format specs stay in the key
  (``{n:,}``, ``{v:.3f}``); translations must keep every placeholder.
* ``trn("{n} hole", "{n} holes", n)`` for counts (one form in languages without plurals).
* ``trc("drillhole", "Survey")`` when one English word has two meanings (key "drillhole|Survey").
* Module-level constants are evaluated once at import, possibly before the language is set:
  mark them ``N_("text")`` (no translation, only found by the extraction tool) and translate
  where they are shown: ``tr(STATUS_WORD[st])``.
* Data stays as it is: station / hole / layer names, attribute and field names, file formats'
  column names (CSV, Excel, shapefile), settings keys, SQL and log messages.

No Qt here (the app installs Qt's translator itself, see ``app.py``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

LOCALE_DIR = Path(__file__).resolve().with_name("locales")
DEFAULT = "en"
LANGUAGE_KEY = "general/language"     # the app's QSettings key: "en" or "zh_CN"


@dataclass(frozen=True)
class Language:
    code: str                    # "en", "zh_CN"
    name: str                    # in that language: "English", "简体中文"
    english_name: str = ""
    qt: str = ""                 # Qt's translation of its own texts: qtbase_<qt>.qm ("" = none)
    fonts: tuple = ()            # font files for VTK text


ENGLISH = Language("en", "English", "English", "en", ())

_messages: dict = {}
_language: Language = ENGLISH


def N_(text: str) -> str:
    """Marks a text for translation without translating it (module-level constants)."""
    return text


def tr(text: str, /, **kw) -> str:
    """``text`` in the current language (English if there is no translation), with the named
    placeholders filled in when keywords are given."""
    msg = _messages.get(text) or text
    if not isinstance(msg, str):                     # a plural list used as a plain text
        msg = msg[-1] if msg else text
    if not kw:
        return msg
    try:
        return msg.format(**kw)
    except (KeyError, IndexError, ValueError):       # a broken translation never breaks the app
        return text.format(**kw)


def trc(context: str, text: str, /, **kw) -> str:
    """``text`` where the same English word means different things: ``trc("drillhole", "Survey")``
    (downhole survey) vs ``tr("Survey")`` (underground survey). Catalog key: "context|text"."""
    msg = _messages.get(f"{context}|{text}") or text
    if not isinstance(msg, str):
        msg = msg[-1] if msg else text
    if not kw:
        return msg
    try:
        return msg.format(**kw)
    except (KeyError, IndexError, ValueError):
        return text.format(**kw)


def trn(singular: str, plural: str, n: int, /, **kw) -> str:
    """A text with a count: ``trn("{n} hole", "{n} holes", count)``. The catalog's value for
    ``singular`` is one text (languages without plurals) or a list [one, other]."""
    kw.setdefault("n", n)
    msg = _messages.get(singular)
    if isinstance(msg, list) and msg:
        text = msg[0] if n == 1 or len(msg) == 1 else msg[1]
    elif isinstance(msg, str) and msg:
        text = msg
    else:
        text = singular if n == 1 else plural
    try:
        return text.format(**kw)
    except (KeyError, IndexError, ValueError):
        return (singular if n == 1 else plural).format(**kw)


# --- languages ------------------------------------------------------------------------------------
def _read(path: Path) -> tuple[Language, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("language", {})
    lang = Language(str(meta.get("code") or path.stem), str(meta.get("name") or path.stem),
                    str(meta.get("english_name") or ""), str(meta.get("qt") or ""),
                    tuple(meta.get("fonts") or ()))
    return lang, dict(data.get("messages") or {})


def available_languages() -> list[Language]:
    """English plus every ``locales/*.json`` (sorted by English name)."""
    out = []
    for p in sorted(LOCALE_DIR.glob("*.json")):
        try:
            lang, _m = _read(p)
        except (OSError, ValueError):
            continue
        out.append(lang)
    return [ENGLISH] + sorted(out, key=lambda lang: lang.english_name or lang.name)


def every_translation(text: str) -> set[str]:
    """``text`` and its translation in every language file - for reading back a value a user
    typed in any language (e.g. a hole state in an Excel sheet made while Chinese was on)."""
    out = {text}
    for p in sorted(LOCALE_DIR.glob("*.json")):
        try:
            v = _read(p)[1].get(text)
        except (OSError, ValueError):
            continue
        if isinstance(v, str) and v:
            out.add(v)
    return out


def match_language(code: str | None) -> str:
    """The available language for ``code`` ("zh_CN", "zh-CN", "zh" ...), else English."""
    if not code:
        return DEFAULT
    code = code.replace("-", "_")
    codes = [lang.code for lang in available_languages()]
    if code in codes:
        return code
    lower = {c.lower(): c for c in codes}
    if code.lower() in lower:
        return lower[code.lower()]
    base = code.split("_")[0].lower()
    return next((c for c in codes if c.split("_")[0].lower() == base), DEFAULT)


def set_language(code: str | None) -> Language:
    """Use language ``code`` from now on (unknown codes fall back to English). Texts already
    on screen stay as they are: the app applies a new language at the next start."""
    global _messages, _language
    code = match_language(code)
    if code == DEFAULT:
        _messages, _language = {}, ENGLISH
        return _language
    try:
        lang, messages = _read(LOCALE_DIR / f"{code}.json")
    except (OSError, ValueError):
        _messages, _language = {}, ENGLISH
        return _language
    _messages = {k: v for k, v in messages.items() if v}
    _language = lang
    return lang


def language() -> Language:
    return _language


def font_file() -> str | None:
    """A font file with the current language's letters, for VTK text (None: VTK's own Arial)."""
    windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in _language.fonts:
        for folder in (LOCALE_DIR, windir):
            p = folder / name
            if p.is_file():
                return str(p)
    return None


def apply_vtk_font(text_property, bold: bool | None = None) -> None:
    """Arial for VTK text, or the current language's font file when Arial lacks its letters."""
    path = font_file()
    if path is None:
        text_property.SetFontFamilyToArial()
    else:
        import vtk

        text_property.SetFontFamily(vtk.VTK_FONT_FILE)
        text_property.SetFontFile(path)
    if bold is not None:
        text_property.SetBold(bool(bold))
