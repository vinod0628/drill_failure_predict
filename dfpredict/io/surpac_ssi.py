"""Reader for Surpac style files (.ssi).

Layout::

    STRINGS:
    default=ssi_label=,HGS_color=lines=text=edges=markers=yellow,HGS_color=faces=(diffuse=yellow,specular=white,gloss=5,transmission=black),...
    1=...                       <- a single string number
    10, 20, 1=...               <- a range: start, end, step
    TRIOBJECTS:
    default=...
    1=...

Lookup order for a number: an exact single-number entry, then the first
matching range in file order, then ``default``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .colors import RGB, parse_color


@dataclass
class Style:
    label: str = ""
    line_color: RGB = (1.0, 1.0, 1.0)
    face_color: RGB = (1.0, 1.0, 1.0)
    specular_color: RGB = (1.0, 1.0, 1.0)
    gloss: float = 5.0
    transmission: RGB = (0.0, 0.0, 0.0)   # black = opaque
    line_pattern: str = ""
    line_weight: float = 1.0
    face_pattern: str = "solid"
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def opacity(self) -> float:
        """Opacity from the transmission colour (black = opaque, white = clear)."""
        return 1.0 - sum(self.transmission) / 3.0


@dataclass
class _Rule:
    start: int
    end: int
    step: int
    style: Style

    def matches(self, n: int) -> bool:
        return self.start <= n <= self.end and (n - self.start) % max(self.step, 1) == 0


@dataclass
class StyleSection:
    default: Style = field(default_factory=Style)
    singles: dict[int, Style] = field(default_factory=dict)
    ranges: list[_Rule] = field(default_factory=list)

    def lookup(self, n: int) -> Style:
        if n in self.singles:
            return self.singles[n]
        for r in self.ranges:
            if r.matches(n):
                return r.style
        return self.default


@dataclass
class StyleSet:
    path: Path | None
    sections: dict[str, StyleSection]

    def string_style(self, string_number: int) -> Style:
        return self.sections.get("STRINGS", StyleSection()).lookup(string_number)

    def object_style(self, object_number: int) -> Style:
        return self.sections.get("TRIOBJECTS", StyleSection()).lookup(object_number)


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside parentheses."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def parse_style(text: str) -> Style:
    st = Style()
    for item in _split_top_level(text):
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            continue
        st.raw[key] = value
        k = key.lower()
        if k == "ssi_label":
            st.label = value.strip()
        elif k == "hgs_color":
            # e.g. "lines=text=edges=markers=blue" or "faces=(diffuse=..,specular=..,gloss=5)"
            if value.strip().lower().startswith("faces="):
                inner = value.split("=", 1)[1].strip().strip("()")
                for part in _split_top_level(inner):
                    pk, _, pv = part.partition("=")
                    pk = pk.strip().lower()
                    if pk == "diffuse":
                        st.face_color = parse_color(pv)
                    elif pk == "specular":
                        st.specular_color = parse_color(pv)
                    elif pk == "transmission":
                        st.transmission = parse_color(pv, default=(0.0, 0.0, 0.0))
                    elif pk == "gloss":
                        try:
                            st.gloss = float(pv)
                        except ValueError:
                            pass
            else:
                # Targets are '='-joined before the colour; the colour itself
                # may contain '=' (r=.. g=.. b=..), so strip known target names.
                rest = value
                targets = {"lines", "text", "edges", "markers"}
                hit = set()
                while True:
                    head, sep, tail = rest.partition("=")
                    if sep and head.strip().lower() in targets:
                        hit.add(head.strip().lower())
                        rest = tail
                    else:
                        break
                color = parse_color(rest)
                if "lines" in hit or not hit:
                    st.line_color = color
        elif k == "hgs_line_pattern":
            st.line_pattern = value.strip()
        elif k == "hgs_line_weight":
            try:
                st.line_weight = float(value)
            except ValueError:
                pass
        elif k == "hgs_face_pattern":
            st.face_pattern = value.strip()
    return st


def _parse_key(key: str) -> tuple[str, int, int, int] | None:
    key = key.strip()
    if key.lower() == "default":
        return ("default", 0, 0, 0)
    parts = [p.strip() for p in key.split(",") if p.strip()]
    try:
        nums = [int(float(p)) for p in parts]
    except ValueError:
        return None
    if len(nums) == 1:
        return ("single", nums[0], nums[0], 1)
    if len(nums) == 2:
        return ("range", nums[0], nums[1], 1)
    if len(nums) >= 3:
        return ("range", nums[0], nums[1], nums[2])
    return None


def read_ssi(path: str | Path) -> StyleSet:
    path = Path(path)
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        lines = f.read().splitlines()
    sections: dict[str, StyleSection] = {}
    cur: StyleSection | None = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.endswith(":") and "=" not in s:
            cur = sections.setdefault(s[:-1].strip().upper(), StyleSection())
            continue
        if cur is None:
            continue
        key, sep, value = s.partition("=")
        if not sep:
            continue
        parsed = _parse_key(key)
        if parsed is None:
            continue
        kind, a, b, step = parsed
        style = parse_style(value)
        if kind == "default":
            cur.default = style
        elif kind == "single":
            cur.singles[a] = style
        else:
            cur.ranges.append(_Rule(a, b, step, style))
    return StyleSet(path=path, sections=sections)
