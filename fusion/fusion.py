"""
Deterministic evidence fusion (§27/§31). Runs BEFORE any LLM is called.

The LLM never decides fraud / not-fraud and never resolves model disagreement:
this module does, and the Investigation Agent only narrates its output.

Two modes:
  rule    : escalate if ANY TWO of {classifier, behaviour, graph} exceed their
            individual thresholds. Conservative and fully explainable.
  learned : logistic regression over the three signals, fitted on a held-out
            validation split. Weights are learned, not hand-picked.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from configs.settings import BEHAVIOUR_THRESHOLD, GRAPH_THRESHOLD, ARTIFACT_DIR

SIGNALS = ["classifier_score", "behaviour_score", "graph_score"]


@dataclass
class FusionResult:
    fused_risk: float
    confidence: float
    signals_above_threshold: list[str]
    conflicting: bool
    conflict_reason: str | None
    mode: str

    def dict(self):
        return asdict(self)


class EvidenceFusion:
    def __init__(self, alert_threshold: float, mode: str = "learned"):
        self.alert_threshold = alert_threshold
        self.mode = mode
        self.lr: LogisticRegression | None = None

    # ------------------------------------------------------------- training
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.lr.fit(self._design(X), y)
        return self

    @staticmethod
    def _design(X):
        X = np.asarray(X, float)
        # logit of the classifier probability is the natural scale for stacking
        c = np.clip(X[:, 0], 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(c / (1 - c)), X[:, 1], X[:, 2]])

    def weights(self) -> dict:
        if self.lr is None:
            return {}
        return {"intercept": float(self.lr.intercept_[0]),
                **{f"w_{s}": float(w) for s, w in zip(SIGNALS, self.lr.coef_[0])}}

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        if self.mode == "learned" and self.lr is not None:
            return self.lr.predict_proba(self._design(X))[:, 1]
        return np.array([self._rule(*row)[0] for row in X])

    # ------------------------------------------------------------ rule path
    def _above(self, c, b, g) -> list[str]:
        out = []
        if c >= self.alert_threshold:
            out.append("classifier")
        if b >= BEHAVIOUR_THRESHOLD:
            out.append("behaviour")
        if g >= GRAPH_THRESHOLD:
            out.append("graph")
        return out

    def _rule(self, c, b, g):
        above = self._above(c, b, g)
        # documented mapping from agreement count to confidence
        conf = {0: 0.10, 1: 0.40, 2: 0.75, 3: 0.95}[len(above)]
        risk = max(c, b) if g >= GRAPH_THRESHOLD else c
        return (risk if len(above) >= 2 else min(risk, 0.49)), conf

    # --------------------------------------------------------------- single
    def fuse(self, classifier_score: float, behaviour_score: float, graph_score: float) -> FusionResult:
        c, b, g = classifier_score, behaviour_score, graph_score
        above = self._above(c, b, g)
        if self.mode == "learned" and self.lr is not None:
            risk = float(self.predict_proba([[c, b, g]])[0])
            confidence = risk          # calibrated posterior from the learned fusion
        else:
            risk, confidence = self._rule(c, b, g)

        # disagreement: one strong signal contradicted by every other TRUSTED signal.
        # A signal is trusted if the learned fusion gave it a positive weight, i.e. the
        # data shows it indicates fraud. An uninformative model (weight <= 0) can't veto.
        conflict, reason = False, None
        hi = {"classifier": c >= 0.8, "behaviour": b >= 0.8, "graph": g >= 0.8}
        lo = {"classifier": c < 0.2, "behaviour": b < 0.3, "graph": g < 0.1}
        trusted = set(hi)
        if self.mode == "learned" and self.lr is not None:
            trusted = {n for n, w in zip(["classifier", "behaviour", "graph"], self.lr.coef_[0]) if w > 0}
        for k in hi:
            others = [o for o in trusted if o != k]
            if k in trusted and hi[k] and len(others) >= 2 and all(lo[o] for o in others):
                conflict, reason = True, f"{k} is high but {' and '.join(others)} are low"
        return FusionResult(round(risk, 4), round(float(confidence), 4), above, conflict, reason, self.mode)

    # ---------------------------------------------------------- persistence
    def save(self, path=ARTIFACT_DIR / "fusion.json"):
        path.write_text(json.dumps({"mode": self.mode, "alert_threshold": self.alert_threshold,
                                    "coef": self.lr.coef_.tolist() if self.lr is not None else None,
                                    "intercept": self.lr.intercept_.tolist() if self.lr is not None else None}))

    @classmethod
    def load(cls, path=ARTIFACT_DIR / "fusion.json", mode: str | None = None):
        d = json.loads(path.read_text())
        f = cls(d["alert_threshold"], mode or d["mode"])
        if d["coef"] is not None:
            f.lr = LogisticRegression()
            f.lr.coef_ = np.array(d["coef"])
            f.lr.intercept_ = np.array(d["intercept"])
            f.lr.classes_ = np.array([0, 1])
        return f
