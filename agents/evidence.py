"""
Evidence contract (§13). The Investigation Agent sees ONLY this object.

Every fact is an item with a stable evidence_id, so each claim in the report can
cite the exact items it rests on and the validator can check it mechanically.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from configs.settings import BEHAVIOUR_THRESHOLD, GRAPH_THRESHOLD

EVIDENCE_TYPES = ["transaction", "model", "shap", "behaviour", "graph", "historical"]


class _Bundle:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, etype: str, field: str, value: Any, description: str, source: str) -> str:
        eid = f"E{len(self.items) + 1}"
        if isinstance(value, float):
            value = round(value, 3)
        self.items.append({"evidence_id": eid, "type": etype, "field": field, "value": value,
                           "description": description, "source": source})
        return eid


def build_evidence(case_id: str, tx: dict, feats: pd.Series, scores: dict, shap_top: list[dict],
                   graph_ev: dict, fusion: dict, alert_threshold: float) -> dict:
    b = _Bundle()
    ts = pd.to_datetime(tx["timestamp"])
    # --- observed transaction facts
    b.add("transaction", "transaction_id", tx["transaction_id"], "transaction identifier", "ingestion")
    b.add("transaction", "amount", float(tx["amount"]), "transaction amount", "ingestion")
    b.add("transaction", "sender_account", tx["sender_account"], "sending account", "ingestion")
    b.add("transaction", "receiver_account", tx["receiver_account"], "receiving account", "ingestion")
    b.add("transaction", "transaction_type", tx["transaction_type"], "transaction type", "ingestion")
    b.add("transaction", "timestamp", ts.strftime("%Y-%m-%d %H:%M"), "transaction time", "ingestion")
    b.add("transaction", "device_id", tx["device_id"], "device used", "ingestion")

    # --- model predictions (clearly separated from observations)
    b.add("model", "classifier_score", scores["classifier_score"], "XGBoost fraud probability", "xgboost")
    b.add("model", "behaviour_anomaly_score", scores["behaviour_score"],
          "Isolation Forest deviation from this account's own history (0-1)", "isolation_forest")
    b.add("model", "graph_score", scores["graph_score"], "graph suspicion score (0-1)", "graph_engine")
    b.add("model", "fused_risk", fusion["fused_risk"], f"deterministic evidence fusion ({fusion['mode']})", "fusion")

    # --- SHAP
    for s in shap_top:
        b.add("shap", f"shap_{s['feature']}", {"value": s["value"], "shap_value": s["shap_value"], "direction": s["direction"]},
              f"SHAP: {s['description']}", "shap")

    # --- behaviour (per-account deviation)
    b.add("behaviour", "amount_to_avg_ratio", float(feats.amount_to_avg_ratio),
          "amount as a multiple of sender's historical average", "features")
    b.add("behaviour", "amount_zscore", float(feats.amount_zscore), "amount z-score vs sender's history", "features")
    b.add("behaviour", "tx_count_1h", int(feats.tx_count_1h), "sender transactions in the last hour (incl. this)", "features")
    b.add("behaviour", "burst_10min", int(feats.burst_10min), "sender transactions in the last 10 minutes (incl. this)", "features")
    b.add("behaviour", "hour_of_day", int(feats.hour), "hour of day", "features")

    # --- graph
    g = graph_ev
    b.add("graph", "k_hop", g["k_hop"], "graph search radius used", "graph_engine")
    b.add("graph", "receiver_is_flagged", g["receiver_is_flagged"], "receiver was a perpetrator in a past case", "graph_engine")
    b.add("graph", "sender_is_flagged", g["sender_is_flagged"], "sender was a perpetrator in a past case", "graph_engine")
    b.add("graph", "receiver_connected_to_flagged_accounts", g["receiver_connected_to_flagged_accounts"],
          f"flagged accounts within {g['k_hop']} hops of receiver", "graph_engine")
    b.add("graph", "sender_connected_to_flagged_accounts", g["sender_connected_to_flagged_accounts"],
          f"flagged accounts within {g['k_hop']} hops of sender", "graph_engine")
    b.add("graph", "hops_to_nearest_flagged_entity", g["hops_to_nearest_flagged_entity"],
          "shortest path from sender/receiver to a flagged account (null = none within 6)", "graph_engine")
    b.add("graph", "shared_device_with_other_accounts", g["shared_device_with_other_accounts"],
          "other accounts that have used the sender's device", "graph_engine")
    b.add("graph", "fan_in_count_window", g["fan_in_count_window"], "distinct senders into receiver in 72h", "graph_engine")
    b.add("graph", "fan_out_count_window", g["fan_out_count_window"], "distinct receivers from sender in 72h", "graph_engine")
    for p in g["detected_patterns"]:
        b.add("graph", f"pattern_{p['pattern']}", p, f"AML typology detected: {p['pattern']}", "graph_engine")

    # --- historical / account context
    b.add("historical", "first_transaction_to_this_beneficiary", bool(feats.new_beneficiary),
          "first transfer from sender to this receiver", "features")
    b.add("historical", "device_changed", bool(feats.device_change), "device differs from sender's previous transaction", "features")
    b.add("historical", "ip_changed", bool(feats.ip_change), "IP differs from sender's previous transaction", "features")
    b.add("historical", "sender_prior_tx_count", int(feats.sender_prior_tx_count), "sender's prior transaction count", "features")

    corroborating = corroborating_types(scores, feats, alert_threshold)
    b.add("model", "fusion_confidence", fusion["confidence"], "confidence computed by fusion (not by the LLM)", "fusion")
    b.add("model", "corroborating_evidence_count", len(corroborating),
          f"evidence types independently indicating risk: {', '.join(corroborating) or 'none'}", "fusion")
    b.add("model", "signals_conflicting", fusion["conflicting"],
          "models strongly disagree" + (f": {fusion['conflict_reason']}" if fusion["conflict_reason"] else ""), "fusion")
    return {
        "case_id": case_id,
        "items": b.items,
        "summary": {
            "classifier_score": scores["classifier_score"],
            "behaviour_anomaly_score": scores["behaviour_score"],
            "graph_score": scores["graph_score"],
            "fusion": fusion,
            "corroborating_evidence_types": corroborating,
        },
    }


def corroborating_types(scores: dict, feats: pd.Series, alert_threshold: float) -> list[str]:
    out = []
    if scores["classifier_score"] >= alert_threshold:
        out.append("model")
    if scores["behaviour_score"] >= BEHAVIOUR_THRESHOLD:
        out.append("behaviour")
    if scores["graph_score"] >= GRAPH_THRESHOLD:
        out.append("graph")
    if feats.new_beneficiary or feats.device_change or feats.ip_change or feats.is_night:
        out.append("historical")
    return out


def item_index(evidence: dict) -> dict[str, dict]:
    return {i["evidence_id"]: i for i in evidence["items"]}


def find(evidence: dict, field: str) -> dict | None:
    return next((i for i in evidence["items"] if i["field"] == field), None)
