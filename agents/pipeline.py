"""
Agentic investigation layer (§7-8, §15, §25), orchestrated with LangGraph.

    triage ──(no-go)──────────────────────────────────────────► END (stored, no case)
       │
       ▼
    correlate ◄──(widen k-hop, once)──┐
       │                              │
       ▼                              │
    supervisor_gate ──────────────────┘   (medium confidence, first pass)
       │ └──(low confidence / conflicting)──► supervisor_final ─► human review, RAW evidence
       ▼
    investigate (LLM, evidence-only, validated, retried)
       │
       ▼
    supervisor_final ─► human review queue (escalate / needs-more-evidence / raw)

The supervisor's confidence comes from the deterministic fusion, never from the
LLM. Nothing here blocks a transaction or closes a case: it only recommends.
"""
from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, TypedDict

import pandas as pd

from agents.evidence import build_evidence, corroborating_types
from agents.narrator import get_narrator
from agents.validator import validate_report
from configs.settings import (ARTIFACT_DIR, DATA_DIR, DEFAULT_K_HOP, HIGH_CONFIDENCE, LLM_MAX_RETRIES,
                              LOW_CONFIDENCE, MIN_CORROBORATING_TYPES, WIDENED_K_HOP, FUSION_MODE)
from fusion.fusion import EvidenceFusion
from graph.engine import GraphEngine, perpetrators
from models.detectors import Detectors
from preprocessing.features import features_for_new


class CaseState(TypedDict, total=False):
    case_id: str
    tx: dict
    features: Any                   # pd.Series
    scores: dict
    triage: dict
    k_hop: int
    widened: bool
    graph_evidence: dict
    shap_top: list
    fusion: dict
    evidence: dict
    gate: dict
    report: dict | None
    validation: dict
    llm_attempts: int
    decision: dict
    agent_log: Annotated[list, operator.add]


def _log(agent: str, t0: float, status: str = "ok", **summary) -> list:
    return [{"agent": agent, "status": status, "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
             "summary": summary}]


class FraudSystem:
    """Holds the models, the transaction history and the graph; exposes score() and investigate()."""

    def __init__(self, history: pd.DataFrame, accounts: pd.DataFrame, detectors: Detectors,
                 fusion: EvidenceFusion, graph: GraphEngine, narrator=None, fused_threshold: float = 0.5):
        self.history = history.copy()
        self.history["timestamp"] = pd.to_datetime(self.history.timestamp)
        self.accounts = accounts
        self.det = detectors
        self.fusion = fusion
        self.graph = graph
        self.narrator = narrator or get_narrator()
        self.fused_threshold = fused_threshold
        self.ablate: set[str] = set()   # {"graph", "shap", "correlation"} for the ablation study (§34)
        self.app = build_graph(self)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_artifacts(cls, history_frac: float = 0.85, narrator=None, fusion_mode: str | None = None):
        import json
        tx = pd.read_parquet(DATA_DIR / "transactions.parquet")
        acc = pd.read_parquet(DATA_DIR / "accounts.parquet")
        tx["timestamp"] = pd.to_datetime(tx.timestamp)
        cut = tx.timestamp.sort_values().iloc[int(len(tx) * history_frac)]
        hist = tx[tx.timestamp < cut]
        det = Detectors()
        fusion = EvidenceFusion.load(mode=fusion_mode or FUSION_MODE)
        th = json.loads((ARTIFACT_DIR / "thresholds.json").read_text())
        ge = GraphEngine(hist, perpetrators(hist))   # "known perpetrators" = confirmed past cases
        sysm = cls(hist, acc, det, fusion, ge, narrator, th.get("fused_threshold", 0.5))
        sysm.replay_pool = tx[tx.timestamp >= cut].reset_index(drop=True)
        return sysm

    # --------------------------------------------------------------- detection
    def score(self, tx: dict, commit: bool = True) -> dict:
        """Layers 2-5: features -> classifier + behaviour + fast graph score -> fusion. Milliseconds."""
        t0 = time.perf_counter()
        tx = {**tx, "timestamp": pd.to_datetime(tx["timestamp"])}
        feats = features_for_new(tx, self.history, self.accounts)
        F = feats.to_frame().T
        c = float(self.det.classifier_score(F)[0])
        b = float(self.det.behaviour.score(F)[0])
        g_ev = {
            "receiver_is_flagged": tx["receiver_account"] in self.graph.flagged,
            "sender_is_flagged": tx["sender_account"] in self.graph.flagged,
            "receiver_connected_to_flagged_accounts": self.graph.flagged_neighbours(tx["receiver_account"], 1),
            "shared_device_with_other_accounts": len(self.graph.shared_device_accounts(tx["device_id"], tx["sender_account"])),
            "detected_patterns": [],
        }
        g = GraphEngine.score(g_ev)
        fused = self.fusion.fuse(c, b, g)
        alert = c >= self.det.alert_threshold or fused.fused_risk >= self.fused_threshold
        if commit:
            self.commit(tx)
        return {"transaction_id": tx["transaction_id"], "features": feats,
                "scores": {"classifier_score": round(c, 4), "behaviour_score": round(b, 4), "graph_score": round(g, 4)},
                "fusion": fused.dict(), "alert": bool(alert),
                "detection_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def commit(self, tx: dict):
        row = {k: tx.get(k) for k in self.history.columns if k not in ("fraud_label", "pattern", "pattern_id")}
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
        self.graph.add_transaction(row)

    # ----------------------------------------------------------- investigation
    def investigate(self, tx: dict, scored: dict) -> CaseState:
        state: CaseState = {
            "case_id": f"C-{uuid.uuid4().hex[:8].upper()}",
            "tx": {**tx, "timestamp": str(pd.to_datetime(tx["timestamp"]))},
            "features": scored["features"], "scores": scored["scores"], "fusion": scored["fusion"],
            "k_hop": DEFAULT_K_HOP, "widened": False, "llm_attempts": 0, "agent_log": [],
        }
        return self.app.invoke(state)


# =============================================================================
#  Agents (nodes)
# =============================================================================

def make_nodes(sysm: FraudSystem):
    def triage(s: CaseState) -> dict:
        t0 = time.perf_counter()
        sc, fu = s["scores"], s["fusion"]
        corr = corroborating_types(sc, s["features"], sysm.det.alert_threshold)
        if fu["fused_risk"] >= 0.9 or len(corr) >= 3:
            prio = "P1"
        elif fu["fused_risk"] >= sysm.fused_threshold or sc["classifier_score"] >= sysm.det.alert_threshold:
            prio = "P2"
        else:
            prio = "P3"
        go = prio in ("P1", "P2") or (len(corr) >= 2 and "model" not in corr)
        out = {"priority": prio, "investigate": go, "corroborating_types": corr}
        return {"triage": out, "agent_log": _log("triage", t0, **out)}

    def correlate(s: CaseState) -> dict:
        t0 = time.perf_counter()
        tx, k = s["tx"], s["k_hop"]
        try:
            if "graph" in sysm.ablate:
                raise RuntimeError("ablated")
            g_ev = sysm.graph.evidence(tx["sender_account"], tx["receiver_account"], tx["device_id"], tx["timestamp"], k)
            g_score = GraphEngine.score(g_ev)
            status = "ok"
        except Exception as e:          # graph failure: continue with detection + behaviour only (§29)
            g_ev = {"k_hop": k, "sender_is_flagged": False, "receiver_is_flagged": False,
                    "receiver_connected_to_flagged_accounts": 0, "sender_connected_to_flagged_accounts": 0,
                    "hops_to_nearest_flagged_entity": None, "shared_device_with_other_accounts": 0,
                    "shared_device_accounts": [], "fan_in_count_window": 0, "fan_out_count_window": 0,
                    "detected_patterns": [], "error": f"graph evidence unavailable: {e}"}
            g_score, status = 0.0, "degraded"
        scores = {**s["scores"], "graph_score": round(g_score, 4)}
        fused = sysm.fusion.fuse(scores["classifier_score"], scores["behaviour_score"], g_score).dict()
        shap_top = [] if "shap" in sysm.ablate else sysm.det.explainer.explain_row(s["features"], top_k=5)
        ev = build_evidence(s["case_id"], tx, s["features"], scores, shap_top, g_ev, fused, sysm.det.alert_threshold)
        if "correlation" in sysm.ablate:   # LLM sees raw scores only, no correlated evidence
            keep = {"transaction", "model"}
            ev["items"] = [i for i in ev["items"] if i["type"] in keep]
        return {"graph_evidence": g_ev, "scores": scores, "fusion": fused, "shap_top": shap_top, "evidence": ev,
                "agent_log": _log("evidence_correlation", t0, status, k_hop=k, evidence_items=len(ev["items"]),
                                  graph_score=round(g_score, 3), fused_risk=fused["fused_risk"],
                                  patterns=[p["pattern"] for p in g_ev["detected_patterns"]])}

    def supervisor_gate(s: CaseState) -> dict:
        """Pre-LLM routing. Weak or conflicting evidence never gets a confident narrative."""
        t0 = time.perf_counter()
        fu = s["fusion"]
        n_types = len(s["evidence"]["summary"]["corroborating_evidence_types"])
        conf = fu["confidence"]
        if fu["conflicting"] or conf < LOW_CONFIDENCE:
            route, action = "raw", "escalate_for_review"
            why = "conflicting model signals" if fu["conflicting"] else f"low confidence {conf:.2f}"
        elif conf >= HIGH_CONFIDENCE and n_types >= MIN_CORROBORATING_TYPES:
            route, action, why = "investigate", "escalate_for_review", "high confidence, corroborated"
        elif not s["widened"]:
            route, action, why = "widen", None, f"medium confidence {conf:.2f} / {n_types} evidence types: widen search"
        else:
            route, action, why = "investigate", "request_more_evidence", "still medium after widening: partial report"
        gate = {"route": route, "suggested_action": action, "reason": why, "confidence": conf,
                "corroborating_types": n_types, "partial": action == "request_more_evidence"}
        upd: dict = {"gate": gate, "agent_log": _log("supervisor_gate", t0, **gate)}
        if route == "widen":
            upd.update(k_hop=WIDENED_K_HOP, widened=True)
        return upd

    def investigate(s: CaseState) -> dict:
        t0 = time.perf_counter()
        context = {"fused_risk": s["fusion"]["fused_risk"], "confidence": s["fusion"]["confidence"],
                   "suggested_action": s["gate"]["suggested_action"], "partial_report": s["gate"]["partial"],
                   "note": "confidence is computed by deterministic fusion; do not restate your own"}
        feedback, report, val, attempts = None, None, None, 0
        for attempts in range(1, LLM_MAX_RETRIES + 2):
            try:
                draft = sysm.narrator.draft(s["evidence"], context, feedback)
            except Exception as e:     # API / parse failure
                val = {"valid": False, "errors": [f"narrator error: {type(e).__name__}: {e}"]}
                feedback = val["errors"]
                continue
            val = validate_report(draft, s["evidence"])
            if val["valid"]:
                report = draft
                break
            feedback = val["errors"]
        status = "ok" if report else "failed"
        return {"report": report, "validation": val, "llm_attempts": attempts,
                "agent_log": _log("investigation", t0, status, narrator=sysm.narrator.name, attempts=attempts,
                                  valid=bool(report), grounding_rate=(val or {}).get("grounding_rate"))}

    def supervisor_final(s: CaseState) -> dict:
        t0 = time.perf_counter()
        gate, report = s["gate"], s.get("report")
        if gate["route"] == "raw":
            route, reason = "human_review_raw_evidence", gate["reason"]
        elif report is None:
            route, reason = "human_review_raw_evidence", "investigation agent failed validation or errored"
        elif gate["partial"]:
            route, reason = "human_review_needs_more_evidence", gate["reason"]
        else:
            route, reason = "human_review_escalate", gate["reason"]
        action = gate["suggested_action"] or "escalate_for_review"
        if report is not None and report.get("recommended_action") != action:
            report = {**report, "recommended_action": action,
                      "supervisor_note": "recommended action set by supervisor from fused confidence"}
        decision = {"route": route, "reason": reason, "recommended_action": action,
                    "confidence": s["fusion"]["confidence"], "fused_risk": s["fusion"]["fused_risk"],
                    "priority": s["triage"]["priority"], "autonomous_action_taken": False}
        return {"decision": decision, "report": report, "agent_log": _log("supervisor_final", t0, **decision)}

    return triage, correlate, supervisor_gate, investigate, supervisor_final


def route_after_triage(s):
    return "correlate" if s["triage"]["investigate"] else "end"


def route_after_gate(s):
    return {"widen": "correlate", "raw": "supervisor_final", "investigate": "investigate"}[s["gate"]["route"]]


def build_graph(sysm: FraudSystem):
    triage, correlate, gate, investigate, final = make_nodes(sysm)
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:                       # tiny fallback runner, same routing
        return _SequentialApp(triage, correlate, gate, investigate, final)
    g = StateGraph(CaseState)
    g.add_node("triage", triage)
    g.add_node("correlate", correlate)
    g.add_node("supervisor_gate", gate)
    g.add_node("investigate", investigate)
    g.add_node("supervisor_final", final)
    g.set_entry_point("triage")
    g.add_conditional_edges("triage", route_after_triage, {"correlate": "correlate", "end": END})
    g.add_edge("correlate", "supervisor_gate")
    g.add_conditional_edges("supervisor_gate", route_after_gate,
                            {"correlate": "correlate", "supervisor_final": "supervisor_final", "investigate": "investigate"})
    g.add_edge("investigate", "supervisor_final")
    g.add_edge("supervisor_final", END)
    return g.compile()


class _SequentialApp:
    def __init__(self, *nodes):
        self.triage, self.correlate, self.gate, self.investigate, self.final = nodes

    def _apply(self, s, upd):
        log = upd.pop("agent_log", [])
        s.update(upd)
        s["agent_log"] = s.get("agent_log", []) + log

    def invoke(self, s):
        s = dict(s)
        self._apply(s, self.triage(s))
        if route_after_triage(s) == "end":
            return s
        while True:
            self._apply(s, self.correlate(s))
            self._apply(s, self.gate(s))
            if route_after_gate(s) != "correlate":
                break
        if route_after_gate(s) == "investigate":
            self._apply(s, self.investigate(s))
        self._apply(s, self.final(s))
        return s
