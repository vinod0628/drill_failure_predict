"""Suggesting measures (design spec 6): candidates from the decision table by failure code and alarm
level, filtered by site conditions, ranked by the evidence there is:

* cavity / fault-zone / loss / caving measures (M-CV, M-LC, M-CL, M-CS): the solved rate among the
  10 - 20 most similar past occurrences (same rock class and contact setting, similar depth); measures
  tried on failed holes count as not solved;
* operating-window measures (M-OP-01..03, M-OP-07, M-BH-03): the 25th - 75th percentile of RPM, flow and
  ROP of successful runs in the same rock class, hole size and bit type (holes that did not fail within
  the next 20 m) - these values fill the measure text;
* everything else: the expert order (the drilling division's fixed priority).

Every suggestion carries an evidence tag; fewer than 3 past uses = "limited evidence". No Qt here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..i18n import tr
from . import codes
from .store import RiskData

SIMILAR = 20
LIMITED = 3


@dataclass
class Suggestion:
    code: str
    text: str
    evidence: str
    stage: str               # P / R
    uses: int = 0
    solved: int = 0
    kind: str = "expert"     # data / settings / expert

    @property
    def title(self) -> str:
        return tr(codes.MEASURES[self.code].title) if self.code in codes.MEASURES else self.code

    @property
    def rate(self) -> float | None:
        return self.solved / self.uses if self.uses else None


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def fill(template: str, **values) -> str:
    return tr(template).format_map(_SafeDict({k: v for k, v in values.items() if v is not None}))


class MeasureAdvisor:
    """Built once from all drilling records; ``suggest`` is cheap."""

    def __init__(self, data: RiskData, cards: dict):
        self.cards = cards
        self.occ = self._occurrence_table(data)
        self.runs = self._run_table(data)
        self.n_success_holes = int((data.collar["result"] == "success").sum()) if len(data.collar) else 0

    # --- evidence tables ------------------------------------------------------------------------
    def _class_at(self, geo: pd.DataFrame, depth: float) -> tuple[str, str]:
        """(rock class, contact type) at a depth of a hole's log."""
        if not len(geo):
            return "", "other"
        k = int(np.clip(np.searchsorted(geo["depth_from"].to_numpy(float), depth, "right") - 1, 0, len(geo) - 1))
        row = geo.iloc[k]
        cls = codes.rock_class_of(row["rock"], (self.cards.get(row["formation"] or "") or {}).get("main_rock"))
        contact = "other"
        form = row["formation"]
        prev = geo.iloc[:k]
        prev = prev[prev["formation"] != form] if form else prev
        if len(prev):
            p = prev.iloc[-1]
            pc = codes.rock_class_of(p["rock"], (self.cards.get(p["formation"] or "") or {}).get("main_rock"))
            above_clastic = codes.ROCK_CLASSES.get(pc, codes.ROCK_CLASSES["sandstone"])[4]
            carb = codes.ROCK_CLASSES.get(cls, codes.ROCK_CLASSES["sandstone"])[1] >= 50
            if above_clastic and carb:
                contact = "clastic-over-carbonate"
            elif codes.ROCK_CLASSES.get(pc, (0, 0, 3))[2] - codes.ROCK_CLASSES.get(cls, (0, 0, 3))[2] >= 2:
                contact = "hard-over-soft"
        return cls, contact

    def _occurrence_table(self, data: RiskData) -> pd.DataFrame:
        occ = data["occurrence"]
        rows = []
        geo_by = {h: g.sort_values("depth_from") for h, g in data["geology"].groupby("hole_id")}
        for r in occ.itertuples(index=False):
            ms = [m.strip() for m in str(r.measures or "").split(";") if m.strip() and m.strip() != "None"]
            if not ms:
                continue
            cls, contact = self._class_at(geo_by.get(r.hole_id, pd.DataFrame(columns=["depth_from"])),
                                          (float(r.depth_from) + float(r.depth_to)) / 2)
            rows.append({"hole": r.hole_id, "depth": float(r.depth_from), "type": r.type, "cls": cls,
                         "contact": contact, "measures": ms, "solved": str(r.solved).upper() == "Y"})
        # failed holes' measures not already in an unsolved occurrence: not solved
        for r in data["failure"].itertuples(index=False):
            ms = [m.strip() for m in str(r.measures or "").split(";") if m.strip() and m.strip() != "None"]
            if not ms or any(x["hole"] == r.hole_id and not x["solved"] and abs(x["depth"] - float(r.fail_depth)) < 3
                             for x in rows[-200:]):
                continue
            cls, contact = self._class_at(geo_by.get(r.hole_id, pd.DataFrame(columns=["depth_from"])),
                                          float(r.fail_depth))
            rows.append({"hole": r.hole_id, "depth": float(r.fail_depth), "type": codes.family_of(r.fail_code),
                         "cls": cls, "contact": contact, "measures": ms, "solved": False})
        return pd.DataFrame(rows, columns=["hole", "depth", "type", "cls", "contact", "measures", "solved"])

    def _run_table(self, data: RiskData) -> pd.DataFrame:
        runs = data["run"]
        if not len(runs):
            return pd.DataFrame()
        col = data.collar.set_index("hole_id")
        fails = data["failure"].set_index("hole_id")["fail_depth"].to_dict() if len(data["failure"]) else {}
        out = []
        for hid, rr in runs.groupby("hole_id"):
            if hid not in col.index or col.loc[hid, "result"] not in ("success", "failed"):
                continue
            geo = data.hole("geology", hid).sort_values("depth_from")
            sizes = data.hole("hole_size", hid).sort_values("depth_from")
            bits = data.hole("bit", hid).sort_values("depth_from")
            fd = fails.get(hid)
            mids = ((rr["depth_from"] + rr["depth_to"]) / 2).to_numpy(float)
            if fd is not None:
                ok = mids < float(fd) - 20.0          # did not fail within the next 20 m
                rr, mids = rr[ok], mids[ok]
            if not len(rr):
                continue
            pick = lambda df, c: (df[c].to_numpy()[np.clip(np.searchsorted(df["depth_from"].to_numpy(float), mids,  # noqa: E731
                                                                          "right") - 1, 0, len(df) - 1)]
                                  if len(df) else np.full(len(mids), None))
            rocks, forms = pick(geo, "rock"), pick(geo, "formation")
            cls = [codes.rock_class_of(r, (self.cards.get(f or "") or {}).get("main_rock")) for r, f in zip(rocks, forms)]
            out.append(pd.DataFrame({"hole": hid, "cls": cls, "formation": forms, "size": pick(sizes, "size"),
                                     "bit": pick(bits, "bit_type"), "rpm": rr["rpm"].to_numpy(float),
                                     "flow": rr["flow_in"].to_numpy(float), "rop": rr["rop"].to_numpy(float)}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    # --- evidence look-ups ------------------------------------------------------------------------
    def occurrence_evidence(self, code: str, cls: str, contact: str, depth: float) -> tuple[int, int]:
        """(uses, solved) of a measure among the most similar past occurrences where it was tried."""
        o = self.occ
        if not len(o):
            return 0, 0
        m = o["measures"].map(lambda ms: code in ms)
        sub = o[m]
        if not len(sub):
            return 0, 0
        score = (sub["cls"] != cls).astype(float) * 2 + (sub["contact"] != contact).astype(float) \
            + (sub["depth"] - depth).abs() / 200.0
        sub = sub.assign(score=score).nsmallest(SIMILAR, "score")
        solved = sum(1 for ms, s in zip(sub["measures"], sub["solved"]) if s and ms and ms[-1] == code)
        return int(len(sub)), int(solved)

    def operating_window(self, cls: str, size: str | None, bit: str | None, formation: str | None = None) -> dict | None:
        """25th - 75th percentiles of RPM, flow and ROP of successful runs in the same setting."""
        r = self.runs
        if not len(r):
            return None
        for keys in ((("cls", cls), ("formation", formation), ("size", size), ("bit", bit)),
                     (("cls", cls), ("size", size), ("bit", bit)), (("cls", cls), ("size", size)), (("cls", cls),)):
            m = np.ones(len(r), bool)
            for k, v in keys:
                if v:
                    m &= (r[k] == v).to_numpy()
            sub = r[m]
            if len(sub) >= 20 and sub["hole"].nunique() >= 3:
                q = lambda c: (float(np.nanpercentile(sub[c], 25)), float(np.nanpercentile(sub[c], 75)))  # noqa: E731
                return {"rpm": q("rpm"), "flow": q("flow"), "rop": q("rop"), "holes": int(sub["hole"].nunique()),
                        "runs": int(len(sub))}
        return None

    # --- suggestions --------------------------------------------------------------------------------
    def suggest(self, fail_codes: list[str], level: str, ctx: dict, limit: int = 3) -> tuple[list[Suggestion],
                                                                                              list[Suggestion]]:
        """(prevention measures for the zone - top ``limit``, response measures if the event starts).
        ``ctx``: from_, to, band_from, rock_class, contact, depth, size, bit, formation, casing, fluids, rig."""
        lvl = level if level in ("yellow", "orange", "red") else "yellow"
        prevent, respond, seen = [], [], set()
        order = ["yellow", "orange"] if lvl == "red" else (["yellow"] if lvl == "orange" else [])
        for code in fail_codes:
            table = codes.DECISION.get(code, {})
            for m in table.get(lvl if lvl != "red" else "orange", []) + [x for o in order for x in table.get(o, [])]:
                if m not in seen and self._allowed(m, ctx):
                    seen.add(m)
                    prevent.append(self._suggestion(m, code, ctx))
            for m in table.get("red", []):
                if m not in seen and self._allowed(m, ctx):
                    seen.add(m)
                    respond.append(self._suggestion(m, code, ctx))
        rank = lambda s: (0 if s.kind == "data" and s.uses >= LIMITED else (1 if s.kind == "settings" else 2),  # noqa: E731
                          -(s.rate or 0.0), codes.EXPERT_ORDER.get(s.code, 99))
        prevent.sort(key=rank)
        respond.sort(key=rank)
        return prevent[:limit], respond[:limit]

    def _allowed(self, m: str, ctx: dict) -> bool:
        if m in ("M-CS-01", "M-CS-02", "M-LC-07"):
            size = ctx.get("size")
            if size and codes.SIZE_ORDER.index(size) >= len(codes.SIZE_ORDER) - 1:
                return False                                   # no smaller size to reduce to
            if m != "M-CS-02" and ctx.get("casing") is not None and not str(ctx.get("casing")).strip():
                return False
        if m == "M-PL-01" and ctx.get("drilled", False):
            return False                                       # re-plan only before the zone is drilled
        if m == "M-FL-01" and ctx.get("fluids") is not None:
            fl = str(ctx.get("fluids")).lower()
            if not any(f in fl for f in ("bentonite", "polymer", "kcl")):
                return False
        return True

    def _suggestion(self, m: str, code: str, ctx: dict) -> Suggestion:
        meas = codes.MEASURES[m]
        top = ctx.get("band_from", ctx.get("from_", 0.0))
        size = ctx.get("size")
        nxt = codes.SIZE_ORDER[min(len(codes.SIZE_ORDER) - 1, codes.SIZE_ORDER.index(size) + 1)] if size else "NQ"
        fluids = str(ctx.get("fluids") or "bentonite").lower()
        fluid = "bentonite" if "bentonite" in fluids else ("polymer" if "polymer" in fluids else "water-based")
        values = {"from_": f"{max(0.0, top - 2):.0f}", "to": f"{ctx.get('band_to', ctx.get('to', 0)):.0f}",
                  "depth": f"{max(0.0, top - 5):.0f}", "wob": "4-6", "v": "10", "c": "20-40", "rho": "1.05-1.10",
                  "s": "40-50", "x": "5" if m == "M-SV-01" else "10", "fluid": fluid, "size_from": size or "HQ",
                  "size_to": nxt, "band": "resonance ±100 rpm", "t": f"{ctx.get('torque_limit', '80 %')}",
                  "f": f"{ctx.get('pull_limit', '80 % of the rig limit')}", "q": "the minimum", "rpm": "a lower value",
                  "rop": "-"}
        kind, evidence, uses, solved = "expert", tr("Expert rule"), 0, 0
        if m in codes.PARAM_MEASURES:
            w = self.operating_window(ctx.get("rock_class", ""), size, ctx.get("bit"), ctx.get("formation"))
            if w is not None:
                values.update(rpm=f"{w['rpm'][0]:.0f}-{w['rpm'][1]:.0f}", q=f"{w['flow'][0]:.0f}-{w['flow'][1]:.0f}",
                              rop=f"{w['rop'][0]:.1f}-{w['rop'][1]:.1f}")
                if m == "M-OP-03":
                    values["q"] = f"{w['flow'][0] * 0.8:.0f}-{w['flow'][0]:.0f}"
                kind = "settings"
                evidence = tr("Data: settings from {n} successful holes ({runs} runs)", n=w["holes"], runs=w["runs"])
        if m.startswith(codes.EVIDENCE_GROUPS) or kind == "expert":
            uses, solved = self.occurrence_evidence(m, ctx.get("rock_class", ""), ctx.get("contact", "other"),
                                                    float(ctx.get("depth", top)))
            if uses:
                kind = "data"
                evidence = tr("Data: solved {s} of {n} similar cases", s=solved, n=uses)
                if uses < LIMITED:
                    evidence += ", " + tr("limited evidence")
        text = fill(meas.text, **values)
        if m == "M-CS-01" and top - 5 < 1:                    # the zone starts at the collar: case it once drilled
            values["depth"] = f"{float(ctx.get('band_to', ctx.get('to', 0))) + 2:.0f}"
            text = tr("Drill the zone in {size_from} and set casing to {depth} m before continuing in {size_to}",
                      **{k: values[k] for k in ("size_from", "depth", "size_to")})
        return Suggestion(m, text, evidence, meas.stage, uses, solved, kind)
