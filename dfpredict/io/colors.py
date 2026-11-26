"""Surpac colour names and colour-spec parsing.

Surpac style files (.ssi) name colours with a Crayola-like palette
("wild strawberry", "raw umber", ...) or give explicit components as
``r=1.0 g=0.6 b=0.2``. Colours are returned as RGB float tuples in 0..1.
"""

from __future__ import annotations

import re
import warnings

RGB = tuple[float, float, float]


def _hex(h: str) -> RGB:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


# Approximate RGB values for colour names seen in Surpac style files.
# Unknown names fall back to FALLBACK_COLOR with a warning; add them here as found.
_NAMED_HEX = {
    "white": "ffffff",
    "black": "000000",
    "red": "ff0000",
    "green": "00ff00",
    "blue": "0000ff",
    "yellow": "ffff00",
    "cyan": "00ffff",
    "tangerine": "f28500",
    "magenta": "ff00ff",
    "orange": "ff8c00",
    "pink": "ff9ecb",
    "grey": "a0a0a0",
    "gray": "a0a0a0",
    "light grey": "d3d3d3",
    "dark grey": "505050",
    "brown": "a0522d",
    "gold": "ffd700",
    "purple": "8000c0",
    "violet": "9f5fdf",
    "apricot": "fdd9b5",
    "aquamarine": "78dbe2",
    "blue green": "0d98ba",
    "brick red": "cb4154",
    "bright blue": "1e6fff",
    "burnt orange": "ff7f49",
    "cerulean": "1dacd6",
    "dark green": "006400",
    "forest green": "228b22",
    "fuchsia": "c364c5",
    "green blue": "1164b4",
    "light green": "90ee90",
    "olive green": "bab86c",
    "raw umber": "714b23",
    "red violet": "c0448f",
    "sky blue": "80daeb",
    "violet blue": "324ab2",
    "wild strawberry": "ff43a4",
    "lime": "bfff00",
    "navy": "000080",
    "maroon": "800000",
    "tan": "d2b48c",
    "salmon": "ff9baa",
    "peach": "ffcba4",
    "lavender": "c8a2e8",
    "turquoise": "40e0d0",
    "sea green": "2e8b57",
    "yellow green": "c5e384",
    "orange red": "ff5349",
    "red orange": "ff5349",
    "yellow orange": "ffb653",
}

NAMED_COLORS: dict[str, RGB] = {k: _hex(v) for k, v in _NAMED_HEX.items()}
FALLBACK_COLOR: RGB = (0.7, 0.7, 0.7)

_COMPONENT_RE = re.compile(r"([rgb])\s*=\s*([-+0-9.eE]+)")


def parse_color(spec: str | None, default: RGB = FALLBACK_COLOR) -> RGB:
    """Parse a Surpac colour spec: a colour name or ``r=.. g=.. b=..``."""
    if spec is None:
        return default
    s = spec.strip().lower()
    if not s:
        return default
    comps = dict(_COMPONENT_RE.findall(s))
    if comps:
        return (
            float(comps.get("r", 0.0)),
            float(comps.get("g", 0.0)),
            float(comps.get("b", 0.0)),
        )
    s = " ".join(s.split())
    if s in NAMED_COLORS:
        return NAMED_COLORS[s]
    warnings.warn(f"Unknown Surpac colour name {spec!r}; using fallback", stacklevel=2)
    return default
