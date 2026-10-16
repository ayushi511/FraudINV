"""SHAP explanations for the XGBoost classifier (Layer 6).

TreeSHAP gives exact Shapley values for tree ensembles in polynomial time, so
the explanation is faithful to the model, not an approximation of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from preprocessing.features import FEATURES

DESCRIPTIONS = {
    "log_amount": "transaction amount (log)",
    "amount_zscore": "amount vs sender's own history (z-score)",
    "amount_to_avg_ratio": "amount as a multiple of sender's average",
    "hour": "hour of day",
    "is_night": "night-time transaction (00:00-05:59)",
    "tx_count_1h": "sender transactions in the last hour",
    "tx_count_24h": "sender transactions in the last 24 hours",
    "burst_10min": "sender transactions in the last 10 minutes",
    "amount_sum_1h_ratio": "total sent in last hour vs sender's average",
    "new_beneficiary": "first transfer to this beneficiary",
    "beneficiary_frequency": "prior transfers to this beneficiary",
    "sender_prior_tx_count": "sender's prior transaction count",
    "log_time_since_last": "time since sender's previous transaction (log s)",
    "sender_distinct_receivers": "distinct receivers the sender has paid",
    "receiver_distinct_senders": "distinct senders that paid the receiver",
    "receiver_in_count_24h": "transfers into the receiver in 24 hours",
    "device_account_count": "accounts seen on this device",
    "account_age_days": "sender account age (days)",
    "device_change": "device differs from sender's previous transaction",
    "ip_change": "IP differs from sender's previous transaction",
    "balance_drain_ratio": "fraction of balance moved",
    "balance_zeroed": "sender balance zero after transaction",
}


class TreeExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def shap_values(self, F: pd.DataFrame) -> np.ndarray:
        return self.explainer.shap_values(F[FEATURES])

    def explain_row(self, row: pd.Series, top_k: int = 5) -> list[dict]:
        sv = self.shap_values(row[FEATURES].to_frame().T.astype(float))[0]
        order = np.argsort(-np.abs(sv))[:top_k]
        out = []
        for i in order:
            name = FEATURES[i]
            val = float(row[name])
            out.append({
                "feature": name,
                "description": DESCRIPTIONS.get(name, name),
                "value": round(val, 3),
                "shap_value": round(float(sv[i]), 4),   # log-odds contribution
                "direction": "increases_risk" if sv[i] > 0 else "decreases_risk",
            })
        return out

    def global_importance(self, F: pd.DataFrame) -> pd.Series:
        sv = self.shap_values(F)
        return pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
