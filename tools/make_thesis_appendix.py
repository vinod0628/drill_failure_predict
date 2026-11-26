"""Thesis appendices generated from the code (so they always match the program):
rock-mechanics defaults, measure library, feature dictionary, database schema, codes and rules.

    .venv\\Scripts\\python.exe tools\\make_thesis_appendix.py   ->  docs/publications/src/thesis_appendix.html
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfpredict.risk import codes, features, rockmech, schema  # noqa: E402


def e(s) -> str:
    return html.escape(str(s if s is not None else ""), quote=False)


def table(caption: str, head: list[str], rows: list[list]) -> str:
    out = [f'<p class="caption">{caption}</p>', '<table class="t">',
           "<tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def rng(t) -> str:
    return f"{t[1]:g} ({t[0]:g}–{t[2]:g})"


def main() -> Path:
    parts = ['<h1>Appendices</h1>',
             "<p>The appendices are generated from the program's own definitions "
             "(<i>dfpredict/risk/rockmech.py</i>, <i>codes.py</i>, <i>features.py</i> and <i>schema.py</i>), so they "
             "match the software exactly. All values in Appendix A are defaults that users can edit for each workspace "
             "in the Rock mechanics tab of the Drilling Records dialog.</p>"]
    # A. rocks
    parts.append("<h2>Appendix A Default Rock-Mechanical Properties</h2>")
    parts.append("<p>Typical values with ranges in brackets. UCS: uniaxial compressive strength; CAI: Cerchar "
                 "abrasivity index; E: Young's modulus; ρ: density; <i>m</i><sub>i</sub>: Hoek–Brown constant. "
                 "Table A.2 lists the remaining parameters of each class.</p>")
    rows, rows2 = [], []
    for k, p in rockmech.DEFAULTS.items():
        rows.append([e(p.name), e(p.group), rng(p.ucs), f"{p.tensile:g}", rng(p.cai), f"{p.e_mod:g}", f"{p.density:g}"])
        flags = [n for n, v in (("soluble", p.soluble), ("swelling", p.swelling), ("anisotropic", p.anisotropy),
                                ("fractured", p.fractures)) if v]
        rows2.append([e(p.name), f"{p.poisson:g}", f"{p.porosity:g}", f"{p.quartz_eq:g}", f"{p.dri:g}", f"{p.mi:g}",
                      f"{p.carbonate:g}" + (" · " + ", ".join(flags) if flags else "")])
    parts.append(table("Table A.1 Strength, abrasivity, stiffness and density of the 46 rock classes.",
                       ["Rock", "Group", "UCS (MPa)", "Tensile (MPa)", "CAI", "E (GPa)", "ρ (g/cm³)"], rows))
    parts.append(table("Table A.2 Further parameters: Poisson's ratio ν, porosity (%), quartz equivalent (%), "
                       "drilling rate index DRI, Hoek–Brown m<sub>i</sub>, carbonate share (%) and flags.",
                       ["Rock", "ν", "Porosity", "Quartz eq.", "DRI", "m_i", "Carbonate · flags"], rows2))
    # B. codes, rules
    parts.append("<h2>Appendix B Failure Codes, Rules and Reason Codes</h2>")
    parts.append(table("Table B.1 Failure codes and families.", ["Code", "Failure", "Family"],
                       [[c, e(t), e(codes.FAMILIES[f])] for c, (t, f) in codes.FAIL_CODES.items()]))
    parts.append(table("Table B.2 Drilling-response rules R1–R8.", ["Rule", "Condition", "Family", "Code", "Level"],
                       [[r, e(v[0]), e(codes.FAMILIES[v[1]]), v[2], v[3]] for r, v in codes.RULES.items()]))
    parts.append(table("Table B.3 Reason codes and their message templates (placeholders in braces are filled in "
                       "for each zone).", ["Code", "Message template"],
                       [[k, e(v)] for k, v in codes.REASONS.items()]))
    # C. measures
    parts.append("<h2>Appendix C Measure Library</h2>")
    parts.append("<p>Stage P: preventive (before the zone), R: response (after an event). The text template is "
                 "completed by the program with depths, sizes, operating windows and fluid properties.</p>")
    parts.append(table("Table C.1 The 37 measures of the default library.", ["Code", "Measure", "Stage", "For", "Instruction"],
                       [[m["code"], e(m["title"]), m["stage"], e(m["failures"]), e(m["text"])] for m in codes.measure_rows()]))
    rows = []
    for fc, lv in codes.DECISION.items():
        rows.append([fc, ", ".join(lv.get("yellow", [])) or "–", ", ".join(lv.get("orange", [])) or "–",
                     ", ".join(lv.get("red", [])) or "–"])
    parts.append(table("Table C.2 Decision table: candidate measures by failure code and level.",
                       ["Code", "Yellow", "Orange", "Red"], rows))
    # D. features
    parts.append("<h2>Appendix D Feature Dictionary</h2>")
    groups = {"ROCK": features.ROCK, "POSITION": features.POSITION, "CONTACT": features.CONTACT,
              "GEOMETRY": features.GEOMETRY, "FAULTS": features.FAULTS, "HAZARD": features.HAZARD,
              "KARST / SITE": features.KARST, "HOLE": features.HOLE, "SETUP": features.SETUP,
              "RESPONSE": features.RESPONSE}
    rows = []
    for g, names in groups.items():
        for n in names:
            used = "".join(m for m in ("H", "A", "B") if n in features.MODEL_FEATURES[m])
            rows.append([f"<i>{e(n)}</i>", g.title(), e(features.FEATURE_TEXT.get(n, "")), used or "–"])
    parts.append(table("Table D.1 Features, their groups, plain-language meaning (as used in the explanations) and "
                       "the models that use them.", ["Feature", "Group", "Meaning", "Models"], rows))
    # E. schema
    parts.append("<h2>Appendix E Database Schema</h2>")
    parts.append("<p>All tables carry the prefix <i>risk_</i>. Key columns are marked with an asterisk; units in "
                 "brackets.</p>")
    for t in schema.ALL_TABLES:
        cols = []
        for c in t.cols:
            s = ("*" if c.name in t.key else "") + c.name + (f" [{c.unit}]" if c.unit else "")
            cols.append(e(s))
        parts.append(f"<p><b>risk_{e(t.name)}</b> ({e(t.title)}): " + ", ".join(cols) + ".</p>")
    out = ROOT / "docs" / "publications" / "src" / "thesis_appendix.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main())
