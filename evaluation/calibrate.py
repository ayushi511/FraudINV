"""
"How often is an alert at this risk level real?" — measured on the held-out test period.

    python -m evaluation.calibrate        (stop the server first; takes a few minutes on IBM)

Writes artifacts/calibration.json, which the dashboard uses to say, e.g.,
"In testing, about 7 in 10 alerts at this risk level were real fraud."

The loaders keep every fraud row but only a fraction of legitimate activity, which
inflates the fraud rate. Legitimate alerts are re-weighted by 1/sample_frac so the
reported rates estimate what you'd see on the full dataset.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agents.pipeline import FraudSystem
from configs.settings import ARTIFACT_DIR, DATA_DIR
from preprocessing.features import engineer_features



def _alert_reliability_table(fused, alert, y, weights):
    table = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        in_bin = (fused >= lo) & ((fused < hi) | ((i == 9) & (fused <= hi)))
        selected = alert & in_bin
        fraud_weight = float(weights[selected & y].sum())
        legit_weight = float(weights[selected & ~y].sum())
        table.append({"bin": i, "min": lo, "max": hi, "alerts": int(selected.sum()),
                      "real_rate": (fraud_weight / (fraud_weight + legit_weight))
                      if (fraud_weight + legit_weight) else None})
    return table


def main():
    S = FraudSystem.from_artifacts()
    pool = S.replay_pool
    print(f"Scoring {len(pool):,} held-out transactions...")
    both = pd.concat([S.history.assign(fraud_label=0), pool], ignore_index=True)
    F = engineer_features(both, S.accounts).iloc[len(S.history):].reset_index(drop=True)
    c = S.det.classifier_score(F)
    b = S.det.behaviour.score(F)
    g = S.graph.batch_scores(pool)
    fused = S.fusion.predict_proba(np.column_stack([c, b, g]))
    alert = (c >= S.det.alert_threshold) | (fused >= S.fused_threshold)
    y = pool.fraud_label.to_numpy().astype(bool)

    meta_file = DATA_DIR / "dataset.json"
    frac = (json.loads(meta_file.read_text()).get("sample_frac") if meta_file.exists() else None) or 1.0
    w = np.where(y, 1.0, 1.0 / frac)          # undo the down-sampling of legitimate activity

    thresholds = json.loads((ARTIFACT_DIR / "thresholds.json").read_text())
    low, high = thresholds["LOW"], thresholds["HIGH"]
    bands = []
    band_masks = [
        ("lower", 0.0, low, alert & (fused < low)),
        ("mid", low, high, alert & (fused >= low) & (fused <= high)),
        ("higher", high, 1.0, alert & (fused > high)),
    ]
    for name, lo, hi, m in band_masks:
        fraud, legit = int((m & y).sum()), int((m & ~y).sum())
        wf, wl = float(w[m & y].sum()), float(w[m & ~y].sum())
        bands.append({"band": name, "min": lo, "max": min(hi, 1.0), "alerts": fraud + legit, "fraud": fraud,
                      "legit": legit, "real_rate": (wf / (wf + wl)) if (wf + wl) else None})
    reliability_table = _alert_reliability_table(fused, alert, y, w)
    out = {"source": "held-out test period", "transactions": int(len(pool)), "sample_frac": frac,
           "alert_rate_per_1000": float(1000 * (w * alert).sum() / w.sum()),
           "fraud_caught_rate": float((alert & y).sum() / max(y.sum(), 1)), "bands": bands,
           "alert_calibration": {"reliability_table": reliability_table}}
    (ARTIFACT_DIR / "calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
