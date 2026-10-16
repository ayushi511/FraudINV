"""
Investigation Agent narrators.

ClaudeNarrator   : LLM over the evidence JSON only, schema-constrained output.
TemplateNarrator : deterministic, no API key needed. Same schema, so the whole
                   pipeline, validator and evaluation run offline. It is also
                   the "no-LLM" arm of the ablation study (Experiment 6).
"""
from __future__ import annotations

import json
import os
import re

from agents.evidence import find
from agents.validator import ALLOWED_ACTIONS, REPORT_SCHEMA
from configs.settings import LLM_MODEL, LLM_TEMPERATURE

SYSTEM_PROMPT = """You are drafting a fraud investigation summary for a human investigator.

Rules (mandatory):
1. Use ONLY the evidence items provided. Do not infer facts that are not present.
2. Every claim must list the evidence_ids it relies on. Every number you write must
   appear in the value of an evidence item you cite. Do not compute new numbers.
3. Keep four things separate: observed evidence (transaction, behaviour, graph,
   historical items), model predictions (model and shap items), your inference
   (the 'inference' list, phrased as reasoning, e.g. "consistent with"), and your
   recommendation.
4. You do NOT decide whether this is fraud. The fused risk and confidence were
   computed deterministically before you were called; you only explain them.
5. If evidence is insufficient for a conclusion, say so in 'insufficient_evidence'.
6. recommended_action must be one of: escalate_for_review, request_more_evidence, monitor.
   Never recommend blocking, reporting or closing - a human makes that decision.
7. Do not put evidence IDs (E1, E2...) inside claim text; put them in evidence_ids.
8. Reply with a single JSON object matching the schema. No prose, no code fences."""


def _user_prompt(evidence: dict, context: dict, feedback: list[str] | None) -> str:
    msg = (f"Routing context (from the supervisor, deterministic): {json.dumps(context)}\n\n"
           f"Evidence bundle:\n{json.dumps(evidence, default=str)}\n\n"
           f"Return JSON with exactly this shape:\n{json.dumps(REPORT_SCHEMA)}")
    if feedback:
        msg += ("\n\nYour previous draft was REJECTED by the evidence validator for these reasons. "
                "Fix every one:\n- " + "\n- ".join(feedback))
    return msg


class ClaudeNarrator:
    name = "claude"

    def __init__(self, model: str = LLM_MODEL):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def draft(self, evidence: dict, context: dict, feedback: list[str] | None = None) -> dict:
        resp = self.client.messages.create(
            model=self.model, max_tokens=2000, temperature=LLM_TEMPERATURE, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(evidence, context, feedback)}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        return json.loads(text)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.0f}" if v.is_integer() and abs(v) >= 100 else f"{v:,.2f}"
    return str(v)


class TemplateNarrator:
    """Rule-written report; every sentence built from evidence items that exist in the bundle.

    Sections whose evidence is absent (graph failure, ablation) are simply omitted
    and listed under insufficient_evidence, mirroring what the LLM is told to do.
    """
    name = "template"

    def draft(self, evidence: dict, context: dict, feedback=None) -> dict:
        E = lambda f: find(evidence, f)  # noqa: E731
        V = lambda f, d=None: (E(f) or {}).get("value", d)  # noqa: E731
        I = lambda *fs: [E(f)["evidence_id"] for f in fs if E(f)]  # noqa: E731
        s = evidence["summary"]
        insufficient: list[str] = []

        reasons, rids = [], []
        for field, test, text in [
            ("classifier_score", lambda v: v >= 0.5, lambda v: f"the fraud classifier scored it {_fmt(v)}"),
            ("amount_to_avg_ratio", lambda v: v >= 3, lambda v: f"the amount is {_fmt(v)}x the sender's average"),
            ("burst_10min", lambda v: v >= 3, lambda v: f"{v} transactions in 10 minutes"),
            ("graph_score", lambda v: v >= 0.5, lambda v: f"graph suspicion score {_fmt(v)}"),
        ]:
            if E(field) and test(V(field)):
                reasons.append(text(V(field))); rids += I(field)
        if not reasons:
            reasons.append(f"fused risk {_fmt(V('fused_risk'))}"); rids += I("fused_risk")

        first = V("first_transaction_to_this_beneficiary")
        summary = (f"{V('transaction_type')} of {_fmt(V('amount'))} from {V('sender_account')} to "
                   f"{V('receiver_account')} at {V('timestamp')}" + ("; first transfer to this beneficiary." if first else "."))

        beh_f = []
        if E("amount_to_avg_ratio"):
            beh_f.append({"claim": f"Amount is {_fmt(V('amount_to_avg_ratio'))}x the sender's historical average "
                                   f"(behaviour anomaly score {_fmt(V('behaviour_anomaly_score'))}).",
                          "evidence_ids": I("amount_to_avg_ratio", "behaviour_anomaly_score")})
        else:
            insufficient.append("No per-account behavioural evidence was provided.")
        if E("tx_count_1h") and V("tx_count_1h") >= 3:
            beh_f.append({"claim": f"{V('tx_count_1h')} transactions from the sender in the last hour, "
                                   f"{V('burst_10min')} of them in the last 10 minutes.",
                          "evidence_ids": I("tx_count_1h", "burst_10min")})
        for f, text in [("device_changed", "The sender used a different device than on the previous transaction."),
                        ("ip_changed", "The sender's IP address changed since the previous transaction.")]:
            if V(f):
                beh_f.append({"claim": text, "evidence_ids": I(f)})

        net = []
        if V("receiver_is_flagged"):
            net.append({"claim": "The receiver was identified as a perpetrator in a previous case.", "evidence_ids": I("receiver_is_flagged")})
        if V("sender_is_flagged"):
            net.append({"claim": "The sender was identified as a perpetrator in a previous case.", "evidence_ids": I("sender_is_flagged")})
        if V("receiver_connected_to_flagged_accounts"):
            net.append({"claim": f"The receiver is within {V('k_hop')} hops of {V('receiver_connected_to_flagged_accounts')} "
                                 "previously flagged accounts.",
                        "evidence_ids": I("receiver_connected_to_flagged_accounts", "k_hop")})
        if (V("shared_device_with_other_accounts") or 0) >= 2:
            net.append({"claim": f"The sending device has also been used by {V('shared_device_with_other_accounts')} other accounts.",
                        "evidence_ids": I("shared_device_with_other_accounts")})
        for it in evidence["items"]:
            if it["field"].startswith("pattern_"):
                p = it["value"]
                detail = {"fan_in": f"{p.get('distinct_senders')} distinct senders into one account",
                          "fan_out": f"{p.get('distinct_receivers')} distinct receivers from one account",
                          "scatter_gather": f"funds split across {p.get('intermediaries')} intermediaries and regathered",
                          "cycle": f"funds moved through a cycle of {p.get('length')} accounts"}[p["pattern"]]
                net.append({"claim": f"AML typology '{p['pattern']}' detected: {detail}.", "evidence_ids": [it["evidence_id"]]})
        if not net:
            insufficient.append("No network evidence links this transaction to known suspicious entities.")

        model_exp = [{"claim": f"Fraud classifier probability {_fmt(V('classifier_score'))}; fused risk "
                               f"{_fmt(V('fused_risk'))} ({s['fusion']['mode']} fusion).",
                      "evidence_ids": I("classifier_score", "fused_risk")}]
        shap_items = [i for i in evidence["items"] if i["type"] == "shap"][:3]
        for it in shap_items:
            sign = "raised" if it["value"]["direction"] == "increases_risk" else "lowered"
            model_exp.append({"claim": f"'{it['description'].removeprefix('SHAP: ')}' {sign} the score "
                                       f"(value {_fmt(it['value']['value'])}).", "evidence_ids": [it["evidence_id"]]})
        if not shap_items:
            insufficient.append("No feature attribution (SHAP) was provided, so the classifier score is unexplained.")

        inference = []
        if (V("graph_score") or 0) >= 0.5 and ((V("shared_device_with_other_accounts") or 0) >= 2 or V("receiver_is_flagged")):
            inference.append({"claim": "The combination of a flagged or shared-device network position with this transfer "
                                       "is consistent with mule-account activity.",
                              "evidence_ids": I("graph_score", *(["receiver_is_flagged"] if V("receiver_is_flagged") else []), *(["shared_device_with_other_accounts"] if (V("shared_device_with_other_accounts") or 0) >= 2 else []))})
        if (V("amount_to_avg_ratio") or 0) >= 3 and (V("device_changed") or (V("burst_10min") or 0) >= 3):
            inference.append({"claim": "A sudden high-value burst after a device change is consistent with account takeover.",
                              "evidence_ids": I("amount_to_avg_ratio", *(["device_changed"] if V("device_changed") else []), *(["burst_10min"] if (V("burst_10min") or 0) >= 3 else []))})
        if s["fusion"]["conflicting"]:
            insufficient.append(f"Model signals disagree: {s['fusion']['conflict_reason']}.")

        action = context.get("suggested_action") or "escalate_for_review"
        if action not in ALLOWED_ACTIONS:
            action = "escalate_for_review"
        return {
            "alert_reason": {"claim": "Flagged because " + ", ".join(reasons) + ".", "evidence_ids": rids},
            "transaction_summary": {"claim": summary, "evidence_ids": I("amount", "sender_account", "receiver_account",
                                                                        "transaction_type", "timestamp",
                                                                        "first_transaction_to_this_beneficiary")},
            "behavioural_findings": beh_f,
            "network_findings": net,
            "model_explanation": model_exp,
            "inference": inference,
            "insufficient_evidence": insufficient,
            "recommended_action": action,
            "recommendation_rationale": {
                "claim": f"Fused risk {_fmt(V('fused_risk'))} with {V('corroborating_evidence_count', 0)} "
                         "corroborating evidence types; a human investigator must make the final decision.",
                "evidence_ids": I("fused_risk", "corroborating_evidence_count")},
        }


def get_narrator(prefer: str | None = None):
    prefer = prefer or os.getenv("NARRATOR", "auto")
    if prefer == "template":
        return TemplateNarrator()
    if prefer in ("auto", "claude") and os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeNarrator()
    if prefer == "claude":
        raise RuntimeError("NARRATOR=claude but ANTHROPIC_API_KEY is not set")
    return TemplateNarrator()
