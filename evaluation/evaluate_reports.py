"""
Experiment 6 + ablation (§32, §34).

    python -m evaluation.evaluate_reports --cases 60             # template narrator
    NARRATOR=claude python -m evaluation.evaluate_reports --cases 20

For each variant (full, no_graph, no_shap, no_correlation) the same alerted test
cases go through the agent pipeline. Automatic measures:
  validity        - report passed the evidence validator
  grounding_rate  - claims whose evidence IDs exist and numbers match
  coverage        - of the evidence types that indicated risk, fraction cited
  routing         - precision of the "escalate" route, recall of fraud into it
Also writes a BLIND rubric sheet (reports/rubric_sheet.csv, variant hidden,
order shuffled) for a human rater: correctness, completeness, grounding,
readability, usefulness (1-5). The key is in reports/rubric_key.csv.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from agents.pipeline import FraudSystem
from agents.render import render_text
from configs.settings import REPORT_DIR, RANDOM_SEED
from graph.engine import GraphEngine
from preprocessing.features import engineer_features

VARIANTS = {"full": set(), "no_graph": {"graph"}, "no_shap": {"shap"}, "no_correlation": {"correlation"}}


def alerted_cases(S: FraudSystem, n: int, seed=RANDOM_SEED) -> list[tuple[dict, dict]]:
    pool = S.replay_pool
    both = pd.concat([S.history.assign(fraud_label=0), pool], ignore_index=True)
    # past-only features, so computing them on history + pool in one pass is leakage-free
    F = engineer_features(both, S.accounts).iloc[len(S.history):].reset_index(drop=True)
    c = S.det.classifier_score(F)
    b = S.det.behaviour.score(F)
    g = S.graph.batch_scores(pool)
    fused = S.fusion.predict_proba(np.column_stack([c, b, g]))
    alert = (c >= S.det.alert_threshold) | (fused >= S.fused_threshold)
    idx = np.where(alert)[0]
    rng = np.random.default_rng(seed)
    fraud_idx = [i for i in idx if pool.fraud_label[i] == 1]
    legit_idx = [i for i in idx if pool.fraud_label[i] == 0]
    # every legit alert (false positives matter) + frauds up to n
    take = list(rng.permutation(legit_idx))[: n // 2] + list(rng.permutation(fraud_idx))[: n - min(len(legit_idx), n // 2)]
    out = []
    for i in sorted(take):
        tx = pool.iloc[i].to_dict()
        fu = S.fusion.fuse(float(c[i]), float(b[i]), float(g[i])).dict()
        out.append((tx, {"features": F.iloc[i], "fusion": fu,
                         "scores": {"classifier_score": float(c[i]), "behaviour_score": float(b[i]), "graph_score": float(g[i])}}))
    print(f"alerts in test pool: {len(idx)} ({alert.sum() and pool.fraud_label[idx].mean():.1%} fraud); evaluating {len(out)}")
    return out


def coverage(state) -> float:
    r = state.get("report")
    if not r:
        return 0.0
    types = {i["evidence_id"]: i["type"] for i in state["evidence"]["items"]}
    cited = set()
    for sec in ["behavioural_findings", "network_findings", "model_explanation", "inference"]:
        for c in r.get(sec, []):
            cited |= {types.get(e) for e in c["evidence_ids"]}
    for sec in ["alert_reason", "transaction_summary"]:
        cited |= {types.get(e) for e in r[sec]["evidence_ids"]}
    want = set(state["evidence"]["summary"]["corroborating_evidence_types"]) | (
        {"shap"} if any(t == "shap" for t in types.values()) else set())
    return len(want & cited) / len(want) if want else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    args = ap.parse_args()
    S = FraudSystem.from_artifacts()
    cases = alerted_cases(S, args.cases)
    rows, sheets = [], []
    for name, abl in VARIANTS.items():
        S.ablate = abl
        for tx, scored in cases:
            t0 = time.perf_counter()
            st = S.investigate(tx, scored)
            dt = time.perf_counter() - t0
            if "decision" not in st:
                continue
            v = st.get("validation") or {}
            rows.append({"variant": name, "transaction_id": tx["transaction_id"], "fraud": int(tx["fraud_label"]),
                         "pattern": tx["pattern"], "route": st["decision"]["route"], "valid": bool(st.get("report")),
                         "grounding_rate": v.get("grounding_rate", np.nan), "claims": v.get("claims", 0),
                         "coverage": coverage(st), "attempts": st.get("llm_attempts", 0), "seconds": dt})
            if st.get("report"):
                sheets.append({"variant": name, "case_id": st["case_id"], "report_text": render_text(st)})
    S.ablate = set()
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "report_eval_raw.csv", index=False)

    def summarise(g):
        esc = g.route == "human_review_escalate"
        return pd.Series({
            "cases": len(g),
            "valid_report_rate": g.valid.mean(),
            "mean_grounding_rate": g.grounding_rate.mean(),
            "mean_claims": g.claims.mean(),
            "mean_evidence_coverage": g.coverage.mean(),
            "escalate_precision": g.fraud[esc].mean() if esc.any() else np.nan,
            "fraud_escalated_recall": esc[g.fraud == 1].mean() if (g.fraud == 1).any() else np.nan,
            "raw_route_rate": (g.route == "human_review_raw_evidence").mean(),
            "p50_seconds": g.seconds.median(), "p95_seconds": g.seconds.quantile(0.95),
        })
    summ = df.groupby("variant", sort=False).apply(summarise, include_groups=False)
    md = summ.to_markdown(floatfmt=".3f")
    (REPORT_DIR / "report_eval_summary.md").write_text(md)
    print(md)
    print("\nRoutes (full):", df[df.variant == "full"].route.value_counts().to_dict())

    sh = pd.DataFrame(sheets).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    sh.insert(0, "sheet_id", [f"R{i:03d}" for i in range(len(sh))])
    sh[["sheet_id", "variant", "case_id"]].to_csv(REPORT_DIR / "rubric_key.csv", index=False)
    blind = sh[["sheet_id", "report_text"]].copy()
    for col in ["correctness_1to5", "completeness_1to5", "grounding_1to5", "readability_1to5", "usefulness_1to5"]:
        blind[col] = ""
    blind.to_csv(REPORT_DIR / "rubric_sheet.csv", index=False)
    print(f"\nBlind rubric sheet: {REPORT_DIR / 'rubric_sheet.csv'} ({len(blind)} reports)")


if __name__ == "__main__":
    main()
