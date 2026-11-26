"""Block model constraints: conditions on DTMs and block values, combined with and / or.

A constraint is a list of conditions, each named by its letter (a, b, c, ...):

* ``{"kind": "dtm", "layer": "ore1", "mode": "inside"}`` - inside / outside a solid, or
  ``"above"`` / ``"below"`` a surface (the block centroid is tested);
* ``{"kind": "value", "name": "sg", "op": ">", "value": 2.0}`` - a block attribute compared
  with a number (``>``, ``>=``, ``<``, ``<=``, ``=``, ``!=``), or a text attribute equal / not
  equal to one of its codes (``"value": "OX"``).

They are combined by ``"and"`` (all conditions), ``"or"`` (any condition) or an expression
of the letters such as ``"a and (b or not c)"``. The expression is parsed here (letters,
and / or / not, parentheses) - it is never evaluated as Python code.
"""

from __future__ import annotations

import re
import string

import numpy as np

from ..i18n import N_, tr

OPS = (">", ">=", "<", "<=", "=", "!=")
OP_TEXT = {">": ">", ">=": "≥", "<": "<", "<=": "≤", "=": "=", "!=": "≠"}
LETTERS = string.ascii_lowercase
MODE_TEXT = {"inside": N_("inside {layer}"), "outside": N_("outside {layer}"), "above": N_("above {layer}"),
             "below": N_("below {layer}")}                 # shown with tr(); the stored mode stays English


def letter(i: int) -> str:
    return LETTERS[i] if i < len(LETTERS) else f"c{i}"


# --- the combination expression -----------------------------------------------------------
_TOKEN = re.compile(r"\s*(\(|\)|[A-Za-z]+)")


def _tokens(text: str) -> list[str]:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(tr("unexpected character '{c}'", c=text[pos:].strip()[:1]))
        out.append(m.group(1).lower())
        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return out


def parse_logic(text: str, n: int):
    """Parse a combination expression over the first ``n`` condition letters into a tree:
    ("var", i) / ("not", t) / ("and", t1, t2) / ("or", t1, t2). Raises ValueError."""
    toks = _tokens(text)
    if not toks:
        raise ValueError(tr("empty expression"))
    names = {letter(i): i for i in range(n)}
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take(expect=None):
        nonlocal pos
        t = peek()
        if t is None:
            raise ValueError(tr("the expression ends too early"))
        if expect is not None and t != expect:
            raise ValueError(tr("'{expect}' expected, found '{found}'", expect=expect, found=t))
        pos += 1
        return t

    def expr():
        node = term()
        while peek() == "or":
            take()
            node = ("or", node, term())
        return node

    def term():
        node = factor()
        while peek() == "and":
            take()
            node = ("and", node, factor())
        return node

    def factor():
        t = peek()
        if t == "not":
            take()
            return ("not", factor())
        if t == "(":
            take()
            node = expr()
            take(")")
            return node
        if t in ("and", "or", ")") or t is None:
            raise ValueError(tr("a condition letter expected") if t is None
                             else tr("a condition letter expected, found {found}", found=repr(t)))
        take()
        if t not in names:
            raise ValueError(tr("there is no condition '{c}'", c=t))
        return ("var", names[t])

    tree = expr()
    if pos != len(toks):
        raise ValueError(tr("unexpected '{found}'", found=toks[pos]))
    return tree


def combine(masks: list[np.ndarray], logic: str) -> np.ndarray:
    """Combine condition masks: "and", "or", or an expression of the letters."""
    if logic == "and":
        out = masks[0].copy()
        for m in masks[1:]:
            out &= m
        return out
    if logic == "or":
        out = masks[0].copy()
        for m in masks[1:]:
            out |= m
        return out

    def ev(node):
        op = node[0]
        if op == "var":
            return masks[node[1]]
        if op == "not":
            return ~ev(node[1])
        a, b = ev(node[1]), ev(node[2])
        return (a & b) if op == "and" else (a | b)

    return ev(parse_logic(logic, len(masks)))


def logic_text(logic: str, n: int) -> str:
    """The combination written out, e.g. "a and b and c"."""
    if logic in ("and", "or"):
        return f" {logic} ".join(letter(i) for i in range(n))
    return logic


# --- one condition -------------------------------------------------------------------------
def value_mask(bm, cond: dict) -> np.ndarray:
    """Blocks meeting a block-value condition (null values never meet it)."""
    name, op, value = cond.get("name"), cond.get("op", ">"), cond.get("value")
    if name not in bm.attributes:
        return np.ones(bm.n_blocks, dtype=bool)          # attribute gone: no restriction
    valid = bm.valid_mask(name)
    v = bm.attributes[name]
    if bm.is_categorical(name):
        cats = list(bm.categories.get(name, []))
        code = cats.index(str(value)) if str(value) in cats else -999
        hit = (v == code) if op != "!=" else (v != code)
        return hit & valid
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.zeros(bm.n_blocks, dtype=bool)
    with np.errstate(invalid="ignore"):
        hit = {">": v > x, ">=": v >= x, "<": v < x, "<=": v <= x,
               "=": np.isclose(v, x), "!=": ~np.isclose(v, x)}[op if op in OPS else ">"]
    return hit & valid


def describe(cond: dict, dtm_kinds: dict | None = None) -> str:
    """Short text of a condition, e.g. "inside ore1", "sg > 2"."""
    if cond.get("kind") == "dtm":
        mode, layer = cond.get("mode", "inside"), cond.get("layer", "?")
        return tr(MODE_TEXT[mode], layer=layer) if mode in MODE_TEXT else f"{mode} {layer}"
    v = cond.get("value")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{cond.get('name', '?')} {OP_TEXT.get(cond.get('op', '>'), '?')} {v}"


def valid_condition(cond) -> bool:
    if not isinstance(cond, dict):
        return False
    if cond.get("kind") == "dtm":
        return isinstance(cond.get("layer"), str) and cond.get("mode") in ("inside", "outside", "above", "below")
    if cond.get("kind") == "value":
        return isinstance(cond.get("name"), str) and cond.get("op") in OPS and cond.get("value") is not None
    return False
