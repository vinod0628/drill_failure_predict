"""The drilling risk layer of the 3D view (design spec 8.1).

For each predicted hole:

* the drilled part: a tube along the desurveyed path, coloured by the logged formation;
* the part ahead of the bit: a thinner dashed tube along the predicted path, and (optional) the
  uncertainty cone of the trial paths, see-through;
* risk intervals: sleeves around the tube in 1 m segments, green / yellow / orange / red by level, with a
  paler extension for the ±u model band;
* fault crossings (red discs), recorded events (spheres coloured by type / failure code, sized by severity);
* the hole's label with its chance of reaching the target depth.

Made by the Risk prediction window (not a file layer: not saved in the workspace, like the mined shapes).
Identify on a sleeve reports the zone; the prediction window highlights the zone chosen in its table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from ..i18n import N_, tr
from ..risk import codes
from .layers import PALETTE, IdentifyResult, Layer, _xyz_rows

EVENT_COLORS = {"cavity": (0.20, 0.55, 1.0), "loss": (0.35, 0.75, 1.0), "fault": (0.95, 0.25, 0.25),
                "broken": (0.75, 0.45, 0.95), "clay": (0.60, 0.40, 0.25), "water": (0.2, 0.9, 0.9),
                "other": (0.8, 0.8, 0.8)}
FAMILY_RGB = {k: tuple(int(v[i:i + 2], 16) / 255 for i in (1, 3, 5)) for k, v in codes.FAMILY_COLORS.items()}


def _polyline(points: np.ndarray) -> pv.PolyData:
    n = len(points)
    return pv.PolyData(points.astype(np.float64), lines=np.r_[n, np.arange(n)])


def _tube(points: np.ndarray, radius: float, sides: int = 14) -> pv.PolyData | None:
    pts = np.asarray(points, np.float64)
    if len(pts) < 2:
        return None
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]
    pts = pts[keep]
    if len(pts) < 2:
        return None
    return _polyline(pts).tube(radius=radius, n_sides=sides, capping=True)


@dataclass(eq=False)
class RiskLayer(Layer):
    predictions: dict = field(default_factory=dict)       # hole id -> Prediction
    events: dict = field(default_factory=dict)            # hole id -> list of (depth, type, severity, label)
    kind: str = N_("Drilling risk")
    diameter: float = 4.0
    show_cone: bool = True
    show_band: bool = True
    show_events: bool = True
    show_labels: bool = True
    min_level: str = "yellow"
    selected: tuple | None = None                          # (hole id, zone index)
    saved = False

    SETTINGS = Layer.SETTINGS + ("diameter", "show_cone", "show_band", "show_events", "show_labels", "min_level")

    def __post_init__(self):
        self.color = (0.85, 0.85, 0.85)

    # --- geometry ---------------------------------------------------------------------------------
    def _build_actors(self) -> None:
        o = self.scene.origin
        r = self.diameter / 2.0
        unit_colors = self._unit_colors()
        for hid, p in self.predictions.items():
            ens = p.ensemble
            md = ens.md
            xyz = ens.mean_xyz - o
            drilled = p.drilled_depth
            k_bit = int(np.searchsorted(md, drilled, side="right"))
            # drilled part, coloured by the unit logged / modelled along it
            if drilled > 0 and k_bit >= 2:
                units = p.unit
                seg_start = 0
                for k in range(1, k_bit + 1):
                    u_prev = units[min(int(md[seg_start]), len(units) - 1)]
                    u_here = units[min(int(md[min(k, k_bit - 1)]), len(units) - 1)] if k < k_bit else None
                    if k == k_bit or u_here != u_prev:
                        t = _tube(xyz[seg_start:k + 1 if k < k_bit else k_bit], r)
                        if t is not None:
                            t.cell_data["hole"] = np.full(t.n_cells, list(self.predictions).index(hid))
                            t.cell_data["zone"] = np.full(t.n_cells, -1)
                            self._add(t, color=unit_colors.get(u_prev, (0.8, 0.8, 0.8)), opacity=self.opacity,
                                      smooth_shading=True)
                        seg_start = k
            # ahead: dashed thinner tube (4 m on, 2 m off) to the target depth
            ahead = (md >= drilled) & (md <= p.planned_depth)
            idx = np.nonzero(ahead)[0]
            dashes = []
            for a in range(0, len(idx), 6):
                seg = idx[a:a + 5]
                if len(seg) >= 2:
                    t = _tube(xyz[seg], r * 0.55, 10)
                    if t is not None:
                        dashes.append(t)
            if dashes:
                m = dashes[0].merge(dashes[1:]) if len(dashes) > 1 else dashes[0]
                m.cell_data["hole"] = np.full(m.n_cells, list(self.predictions).index(hid))
                m.cell_data["zone"] = np.full(m.n_cells, -1)
                self._add(m, color=(0.92, 0.92, 0.92), opacity=self.opacity, smooth_shading=True)
            # uncertainty cone of the trial paths
            if self.show_cone and len(idx) > 2:
                rad = ens.radius[idx]
                pts = xyz[idx]
                line = _polyline(pts)
                line.point_data["r"] = np.maximum(rad, r * 1.2)
                try:
                    cone = line.tube(scalars="r", absolute=True, n_sides=20, capping=False)
                    self._add(cone, color=(0.75, 0.80, 0.95), opacity=0.13, smooth_shading=True, pickable=False)
                except Exception:  # noqa: BLE001 - older pyvista: no varying radius
                    pass
            # risk sleeves, 1 m segments by level, and the ±u band
            self._sleeves(hid, p, xyz, r)
            # fault crossings ahead of the bit
            for d, name, share in p.crossings:
                if share < 0.2 or d > p.planned_depth:
                    continue
                k = min(len(md) - 1, int(np.searchsorted(md, d)))
                disc = pv.Disc(center=xyz[k], inner=0.0, outer=r * 3.2, normal=ens.mean_dirs[k], c_res=24)
                self._add(disc, color=(0.95, 0.2, 0.2), opacity=0.85, pickable=False)
            # events
            if self.show_events:
                for d, typ, sev, _lab in self.events.get(hid, []):
                    k = min(len(md) - 1, int(np.searchsorted(md, d)))
                    sph = pv.Sphere(radius=r * (1.0 + 0.35 * (sev or 1)), center=xyz[k])
                    self._add(sph, color=EVENT_COLORS.get(typ, (0.8, 0.8, 0.8)), opacity=0.95, pickable=False)
        if self.show_labels and self.predictions:
            pts, labels = [], []
            for hid, p in self.predictions.items():
                pts.append(p.ensemble.mean_xyz[0] - o + np.array([0, 0, self.diameter * 3]))
                labels.append(f"{hid}  {100 * p.p_reach:.0f}% " + tr("to target"))
            act = self.scene.plotter.add_point_labels(np.array(pts, np.float32), labels, font_size=12,
                                                      text_color=self.scene.foreground, shape=None, show_points=False,
                                                      always_visible=True, render=False, name=f"{id(self)}-labels")
            act._geoviewer_layer = self
            self.actors.append(act)

    def _sleeves(self, hid, p, xyz, r) -> None:
        md = p.ensemble.md
        hole_i = list(self.predictions).index(hid)
        min_rank = codes.level_rank(self.min_level)
        # metre centres of the prediction map to path vertices by depth
        for zi, z in enumerate(p.zones):
            if codes.level_rank(z.level) < min_rank:
                continue
            sel = (p.md >= z.from_) & (p.md <= z.to)
            levels = p.level[sel]
            depths = p.md[sel]
            if not len(depths):
                continue
            start = 0
            for k in range(1, len(depths) + 1):
                if k == len(depths) or levels[k] != levels[start]:
                    lv = levels[start]
                    if codes.level_rank(lv) >= 1:
                        a, b = depths[start] - 0.5, depths[k - 1] + 0.5
                        pts = self._path_between(md, xyz, a, b)
                        t = _tube(pts, r * 2.1, 18)
                        if t is not None:
                            t.cell_data["hole"] = np.full(t.n_cells, hole_i)
                            t.cell_data["zone"] = np.full(t.n_cells, zi)
                            hl = self.selected == (hid, zi)
                            self._add(t, color=codes.LEVEL_RGB[lv], opacity=1.0 if hl else 0.9, smooth_shading=True,
                                      show_edges=False)
                    start = k
            if self.show_band and z.u > 0.5:
                for a, b in ((z.band_from, z.from_), (z.to, z.band_to)):
                    if b - a < 0.5:
                        continue
                    t = _tube(self._path_between(md, xyz, a, b), r * 1.8, 18)
                    if t is not None:
                        t.cell_data["hole"] = np.full(t.n_cells, hole_i)
                        t.cell_data["zone"] = np.full(t.n_cells, zi)
                        self._add(t, color=codes.LEVEL_RGB[z.level], opacity=0.28, smooth_shading=True)
            if self.selected == (hid, zi):
                # the selected zone: a faint white shell around the band (the level colour shows through)
                t = _tube(self._path_between(md, xyz, z.band_from, z.band_to), r * 3.0, 24)
                if t is not None:
                    self._add(t, color=(1.0, 1.0, 1.0), opacity=0.16, pickable=False, smooth_shading=True)

    @staticmethod
    def _path_between(md, xyz, a, b) -> np.ndarray:
        q = np.unique(np.r_[a, md[(md > a) & (md < b)], b])
        return np.column_stack([np.interp(q, md, xyz[:, i]) for i in range(3)])

    def _unit_colors(self) -> dict:
        names = []
        for p in self.predictions.values():
            for u in p.unit:
                if u not in names:
                    names.append(u)
        return {u: PALETTE[i % len(PALETTE)] for i, u in enumerate(names)}

    # --- info -----------------------------------------------------------------------------------
    def world_bounds(self):
        pts = [p.ensemble.mean_xyz for p in self.predictions.values()]
        if not pts:
            return None
        a = np.concatenate(pts)
        pad = self.diameter * 2
        return a.min(axis=0) - pad, a.max(axis=0) + pad

    def info(self):
        rows = super().info()
        for hid, p in self.predictions.items():
            rows.append((hid, tr("{p}% chance of reaching {d:.0f} m; {n} warning zones", p=round(100 * p.p_reach),
                                 d=p.planned_depth, n=len(p.zones))))
        return rows

    def zone_at(self, hit) -> tuple[str, int] | None:
        mesh = getattr(hit, "mesh", None)
        cid = getattr(hit, "cell_id", -1)
        if mesh is None or cid is None or cid < 0 or "zone" not in mesh.cell_data:
            return None
        hole_i = int(mesh.cell_data["hole"][cid])
        zi = int(mesh.cell_data["zone"][cid])
        holes = list(self.predictions)
        if not (0 <= hole_i < len(holes)):
            return None
        return holes[hole_i], zi

    def identify(self, hit) -> IdentifyResult:
        found = self.zone_at(hit)
        rows = []
        if found is not None:
            hid, zi = found
            p = self.predictions[hid]
            rows.append((tr("Hole"), hid))
            rows.append((tr("Chance of reaching target"), f"{100 * p.p_reach:.0f}%"))
            if zi >= 0:
                z = p.zones[zi]
                rows += [(tr("Zone"), f"{z.from_:.0f} - {z.to:.0f} m (±{z.u:.0f} m)"),
                         (tr("Level"), tr(codes.LEVEL_NAMES[z.level])), (tr("Failure"), ", ".join(z.codes)),
                         (tr("Risk index P"), f"{z.p:.2f}"), (tr("Status"), tr(z.status))]
                rows += [(c or tr("Reason"), t) for c, t in z.reasons]
                rows += [(s.code, s.text) for s in z.prevent]
        rows += [(tr("Picked location"), None)] + _xyz_rows(self.scene.to_world(hit.local))
        return IdentifyResult(tr(self.kind), rows)
