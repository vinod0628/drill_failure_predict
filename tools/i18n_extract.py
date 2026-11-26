"""Find every translatable text in the code and check / update the language files.

Texts are the first argument of ``tr(...)``, ``N_(...)``, the first two of ``trn(...)`` and
"context|text" for ``trc(context, text)``, when they are string literals (see geoviewer/i18n/__init__.py for the rules).

    python tools/i18n_extract.py                 list the counts per language
    python tools/i18n_extract.py --check         exit 1 when a language misses texts, has
                                                 texts the code no longer uses, or a
                                                 translation's placeholders differ
    python tools/i18n_extract.py --update        add the code's new texts (empty) to every
                                                 language file, drop unused ones, sort
    python tools/i18n_extract.py --new fr --name Français --english-name French [--qt fr]
                                                 start a new language file, every text empty
    python tools/i18n_extract.py --missing zh_CN print the texts zh_CN has no translation for
"""

from __future__ import annotations

import argparse
import ast
import json
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dfpredict"
LOCALES = SRC / "i18n" / "locales"
FUNCS = {"tr": 1, "N_": 1, "trn": 2}


def extract(src: Path = SRC) -> dict[str, list[str]]:
    """text -> ["file:line", ...] for every translatable literal under ``src``."""
    found: dict[str, list[str]] = {}
    for path in sorted(src.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            raise SystemExit(f"{path}: {e}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name == "trc":
                if len(node.args) >= 2 and all(isinstance(x, ast.Constant) and isinstance(x.value, str)
                                               for x in node.args[:2]):
                    where = f"{path.relative_to(ROOT).as_posix()}:{node.lineno}"
                    found.setdefault(f"{node.args[0].value}|{node.args[1].value}", []).append(where)
                continue
            if name not in FUNCS:
                continue
            for arg in node.args[:FUNCS[name]]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value:
                    where = f"{path.relative_to(ROOT).as_posix()}:{node.lineno}"
                    found.setdefault(arg.value, []).append(where)
    return found


def plural_keys(src: Path = SRC) -> set[str]:
    """The plural forms of trn(): not keys of their own (the singular is the key)."""
    out = set()
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "trn"
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)):
                out.add(node.args[1].value)
    return out


def placeholders(text: str) -> set[str]:
    try:
        return {f for _lit, f, _spec, _conv in string.Formatter().parse(text) if f is not None}
    except ValueError:
        return {"<unbalanced braces>"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict) -> None:
    data["messages"] = dict(sorted(data["messages"].items(), key=lambda kv: kv[0].lower()))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def problems(keys: set[str], data: dict) -> dict[str, list]:
    msgs = data.get("messages", {})
    out = {"missing": sorted(k for k in keys if not msgs.get(k)),
           "unused": sorted(k for k in msgs if k not in keys),
           "placeholders": []}
    for k, v in msgs.items():
        if not v or k not in keys:
            continue
        for form in (v if isinstance(v, list) else [v]):
            if placeholders(form) != placeholders(k):
                out["placeholders"].append((k, form))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--missing", metavar="CODE")
    ap.add_argument("--new", metavar="CODE")
    ap.add_argument("--name", default="")
    ap.add_argument("--english-name", default="")
    ap.add_argument("--qt", default="")
    a = ap.parse_args(argv)

    found = extract()
    keys = set(found) - plural_keys()
    if a.new:
        path = LOCALES / f"{a.new}.json"
        if path.exists():
            print(f"{path} exists already")
            return 1
        LOCALES.mkdir(parents=True, exist_ok=True)
        save(path, {"language": {"code": a.new, "name": a.name or a.new, "english_name": a.english_name,
                                 "qt": a.qt or a.new, "fonts": []},
                    "messages": {k: "" for k in keys}})
        print(f"wrote {path} ({len(keys)} texts to translate)")
        return 0
    if a.missing:
        for k in problems(keys, load(LOCALES / f"{a.missing}.json"))["missing"]:
            print(json.dumps(k, ensure_ascii=False), "  #", found[k][0])
        return 0
    bad = False
    print(f"{len(keys)} texts in the code")
    for path in sorted(LOCALES.glob("*.json")):
        data = load(path)
        if a.update:
            msgs = data.setdefault("messages", {})
            for k in keys:
                msgs.setdefault(k, "")
            for k in [k for k in msgs if k not in keys]:
                del msgs[k]
            save(path, data)
        pr = problems(keys, data)
        print(f"  {path.stem}: {len(keys) - len(pr['missing'])} translated, {len(pr['missing'])} missing, "
              f"{len(pr['unused'])} unused, {len(pr['placeholders'])} placeholder mismatches")
        for k, v in pr["placeholders"]:
            print(f"      placeholders differ: {k!r} -> {v!r}")
        bad |= any(pr.values())
    return 1 if (a.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
