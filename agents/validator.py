"""
Hallucination control, enforced by construction (§28).

A report is rejected unless:
  1. it matches the fixed schema,
  2. every claim cites at least one evidence_id,
  3. every cited evidence_id exists in the bundle,
  4. every number written in a claim appears in the evidence it cites,
  5. the recommended action is one of the allowed actions (never "block" / "close").
"""
from __future__ import annotations

import re

from agents.evidence import item_index

ALLOWED_ACTIONS = ["escalate_for_review", "request_more_evidence", "monitor"]
SECTIONS = ["alert_reason", "behavioural_findings", "network_findings", "model_explanation", "inference"]

REPORT_SCHEMA = {
    "alert_reason": {"claim": "str", "evidence_ids": ["E1"]},
    "transaction_summary": {"claim": "str", "evidence_ids": ["E1"]},
    "behavioural_findings": [{"claim": "str", "evidence_ids": ["E1"]}],
    "network_findings": [{"claim": "str", "evidence_ids": ["E1"]}],
    "model_explanation": [{"claim": "str", "evidence_ids": ["E1"]}],
    "inference": [{"claim": "str (clearly your reasoning, not an observation)", "evidence_ids": ["E1"]}],
    "insufficient_evidence": ["str: what could not be concluded and why"],
    "recommended_action": "|".join(ALLOWED_ACTIONS),
    "recommendation_rationale": {"claim": "str", "evidence_ids": ["E1"]},
}

_NUM = re.compile(r"(?<![A-Za-z])-?\d[\d,]*\.?\d*")
_EID = re.compile(r"\bE\d+\b")
_ACCT = re.compile(r"\b[A-Z]{1,4}[-_]?\d{3,}\b|\b[A-Z]{2,}-\d+\b")  # account / tx / device IDs


def _numbers(text: str) -> list[float]:
    text = _EID.sub(" ", text)
    text = _ACCT.sub(" ", text)
    out = []
    for m in _NUM.findall(text):
        m = m.replace(",", "").rstrip(".")
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def _grounded_values(items: list[dict]) -> set[float]:
    vals: set[float] = set()

    def walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            for x in (v, round(v), round(v, 1), round(v, 2), round(v * 100), round(v * 100, 1)):
                vals.add(float(x))
        elif isinstance(v, str):
            for n in _numbers(v):
                vals.add(n)
            for m in re.findall(r"\d+", v):   # timestamps: 2026-01-05 02:43 -> 2026, 1, 5, 2, 43
                vals.add(float(m))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for it in items:
        walk(it["value"])
        walk(it.get("description", ""))
    return vals


def _claims(report: dict):
    for key in ["alert_reason", "transaction_summary", "recommendation_rationale"]:
        if key in report:
            yield key, report[key]
    for key in ["behavioural_findings", "network_findings", "model_explanation", "inference"]:
        for c in report.get(key, []) or []:
            yield key, c


def validate_report(report: dict, evidence: dict) -> dict:
    idx = item_index(evidence)
    errors: list[str] = []
    n_claims = n_grounded = 0
    for key in ["alert_reason", "transaction_summary", "recommendation_rationale", "recommended_action"]:
        if key not in report:
            errors.append(f"missing field '{key}'")
    if report.get("recommended_action") not in ALLOWED_ACTIONS:
        errors.append(f"recommended_action must be one of {ALLOWED_ACTIONS}")
    for section, c in _claims(report):
        if not isinstance(c, dict) or "claim" not in c:
            errors.append(f"{section}: claim must be an object with 'claim' and 'evidence_ids'")
            continue
        n_claims += 1
        ids = c.get("evidence_ids") or []
        if not ids:
            errors.append(f"{section}: claim has no evidence_ids: {c['claim'][:60]!r}")
            continue
        bad = [i for i in ids if i not in idx]
        if bad:
            errors.append(f"{section}: cites non-existent evidence {bad}")
            continue
        allowed = _grounded_values([idx[i] for i in ids])
        ungrounded = [n for n in _numbers(c["claim"])
                      if not any(abs(n - a) <= max(0.011, 0.006 * abs(a)) for a in allowed)]
        if ungrounded:
            errors.append(f"{section}: numbers {ungrounded} not found in cited evidence {ids}: {c['claim'][:80]!r}")
            continue
        n_grounded += 1
    return {"valid": not errors, "errors": errors, "claims": n_claims, "grounded_claims": n_grounded,
            "grounding_rate": round(n_grounded / n_claims, 3) if n_claims else 0.0}
