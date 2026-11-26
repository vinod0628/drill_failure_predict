"""The drilling risk system of one workspace: records, models, geological model and predictor, kept
together and rebuilt only when something they depend on changed. No Qt here (runs in worker threads).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.database import Target
from . import models
from .geomodel import GeoModel, SurfaceInput, build_model, load_config
from .predict import Prediction, Predictor
from .store import RiskData, RiskStore

log = logging.getLogger(__name__)


class RiskEngine:
    def __init__(self, target: Target, folder: str | Path | None):
        self.target = target
        self.folder = Path(folder) if folder is not None else None
        self.store = RiskStore(target)
        self._data: RiskData | None = None
        self._models: models.ModelSet | None = None
        self._model: GeoModel | None = None
        self._model_key = None
        self._predictor: Predictor | None = None
        self.surfaces: list[SurfaceInput] = []
        self.lock = threading.RLock()

    # --- invalidation -------------------------------------------------------------------------
    def records_changed(self) -> None:
        with self.lock:
            self._data = None
            self._predictor = None

    def models_changed(self) -> None:
        with self.lock:
            self._models = None
            self._predictor = None

    def set_surfaces(self, surfaces: list[SurfaceInput]) -> None:
        """The geological model's surfaces (from the layers with a model role)."""
        key = tuple((s.name, s.role, len(s.vertices), len(s.triangles)) for s in surfaces)
        with self.lock:
            self.surfaces = surfaces
            if key != self._model_key:
                self._model = None
                self._predictor = None
                self._model_key = key

    def geomodel_changed(self) -> None:
        with self.lock:
            self._model = None
            self._predictor = None

    # --- parts ----------------------------------------------------------------------------------
    @property
    def data(self) -> RiskData:
        with self.lock:
            if self._data is None:
                self.store.prepare()
                self._data = self.store.load_all()
            return self._data

    @property
    def models(self) -> models.ModelSet | None:
        with self.lock:
            if self._models is None and self.folder is not None:
                self._models = models.load_current(self.folder)
            return self._models

    @property
    def geomodel(self) -> GeoModel:
        with self.lock:
            if self._model is None:
                cfg = load_config(self.folder)
                cards = {r["formation"]: r for r in self.data["formation"].to_dict("records")}
                self._model = build_model(self.surfaces, cfg, cards=cards, control_xy=self.control_xy())
            return self._model

    @property
    def predictor(self) -> Predictor:
        with self.lock:
            if self._predictor is None:
                ms = self.models
                if ms is None:
                    raise RuntimeError("no trained models")
                self._predictor = Predictor(self.data, ms, self.geomodel, self.area_id)
            return self._predictor

    def control_xy(self) -> np.ndarray:
        """Model control for the error band: the collars of the workspace's drillholes - the holes of the
        workspace's area that are drilled (finished or in progress), and the holes without an area (drillholes
        entered in GeoViewer's Drillhole Manager only). The drilling history of other areas is not used: its
        collars are in other coordinates."""
        try:
            c = self.data.collar
            a = self.area_id
            drilled = c["result"].isin(["success", "failed", "in_progress"]) | c["result"].isna()
            here = (c["area_id"] == a) if a else pd.Series(False, index=c.index)
            m = drilled & (here | c["area_id"].isna())
            pts = c.loc[m, ["x", "y"]].dropna().to_numpy(float)
        except Exception:  # noqa: BLE001
            pts = np.zeros((0, 2))
        return np.unique(pts, axis=0) if len(pts) else np.zeros((0, 2))

    @property
    def area_id(self) -> str | None:
        """The workspace's area: set in geomodel.json, else the area of the holes in progress / planned."""
        cfg = load_config(self.folder)
        if cfg.get("area_id"):
            return str(cfg["area_id"])
        c = self._data.collar if self._data is not None else None
        if c is not None and len(c):
            live = c[c["result"].isin(["in_progress", "planned"])]
            if len(live):
                return str(live["area_id"].mode().iloc[0])
        return None

    # --- actions --------------------------------------------------------------------------------
    def live_holes(self) -> pd.DataFrame:
        c = self.data.collar
        prio = {"in_progress": 0, "planned": 1, "failed": 2, "success": 3}
        c = c.assign(_p=c["result"].map(prio).fillna(4)).sort_values(["_p", "hole_id"])
        return c.drop(columns="_p")

    def predict(self, hole_id: str | None = None, design: dict | None = None, n_paths: int = 40,
                progress=None) -> Prediction:
        with self.lock:
            pr = self.predictor
        return pr.predict(hole_id, design=design, n_paths=n_paths, progress=progress)

    def train(self, progress=None) -> models.ModelSet:
        data = self.data
        ms = models.train(data, progress=progress)
        models.save(ms, self.folder)
        with self.lock:
            self._models = ms
            self._predictor = None
        return ms
