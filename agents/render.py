"""Plain-text rendering of a validated report (the §14 layout) for the UI and black book."""
from __future__ import annotations


def render_text(state: dict) -> str | None:
    r = state.get("report")
    if not r:
        return None
    tx, d, fu = state["tx"], state["decision"], state["fusion"]
    risk = "HIGH" if fu["fused_risk"] >= 0.85 else "MEDIUM" if fu["fused_risk"] >= 0.5 else "LOW"

    def cite(c):
        return f"{c['claim']}  [{', '.join(c['evidence_ids'])}]"

    def block(title, items):
        if not items:
            return []
        return [f"{title}:"] + [f"  - {cite(c)}" for c in items]

    lines = [
        f"Case ID: {state['case_id']}        Risk Level: {risk}",
        f"Transaction ID: {tx['transaction_id']}   Fused risk: {fu['fused_risk']:.2f}   Priority: {d['priority']}",
        "",
        "ALERT REASON:", f"  {cite(r['alert_reason'])}",
        "", "TRANSACTION SUMMARY:", f"  {cite(r['transaction_summary'])}", "",
        *block("BEHAVIOURAL FINDINGS (observed)", r.get("behavioural_findings")), "",
        *block("NETWORK FINDINGS (observed)", r.get("network_findings")), "",
        *block("MODEL EXPLANATION (predictions + SHAP)", r.get("model_explanation")), "",
        *block("INFERENCE (reasoning, not fact)", r.get("inference")),
    ]
    if r.get("insufficient_evidence"):
        lines += ["", "INSUFFICIENT EVIDENCE:"] + [f"  - {x}" for x in r["insufficient_evidence"]]
    lines += ["", f"CONFIDENCE: {d['confidence']:.2f} (computed by evidence fusion, not the LLM)",
              f"RECOMMENDED ACTION: {d['recommended_action']}  -  {cite(r['recommendation_rationale'])}",
              "REVIEWER STATUS: Pending   (the system never blocks or closes on its own)"]
    return "\n".join(lines)
