"""Loaded model bundle used at inference time (Layers 3, 4, 6)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from configs.settings import ARTIFACT_DIR, DEFAULT_ALERT_THRESHOLD
from preprocessing.features import FEATURES, BEHAVIOUR_FEATURES


class BehaviourModel:
    """Isolation Forest over per-account deviation features, score normalised to [0,1]."""

    def __init__(self, iso, lo: float, hi: float):
        self.iso, self.lo, self.hi = iso, lo, hi

    def score(self, F: pd.DataFrame) -> np.ndarray:
        raw = -self.iso.score_samples(F[BEHAVIOUR_FEATURES].to_numpy())
        return np.clip((raw - self.lo) / (self.hi - self.lo), 0, 1)


class Detectors:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(artifact_dir / "xgb.json")
        self.xgb.set_params(n_jobs=1)
        iso = joblib.load(artifact_dir / "iso_forest.joblib")
        iso.n_jobs = 1                      # joblib fan-out costs more than it saves for 1 row
        norm = json.loads((artifact_dir / "iso_norm.json").read_text())
        self.behaviour = BehaviourModel(iso, norm["lo"], norm["hi"])
        th = json.loads((artifact_dir / "thresholds.json").read_text())
        self.alert_threshold = th.get("alert_threshold", DEFAULT_ALERT_THRESHOLD)
        self._explainer = None

    def classifier_score(self, F: pd.DataFrame) -> np.ndarray:
        return self.xgb.predict_proba(F[FEATURES])[:, 1]

    @property
    def explainer(self):
        if self._explainer is None:
            from explainability.shap_explain import TreeExplainer
            self._explainer = TreeExplainer(self.xgb)
        return self._explainer
