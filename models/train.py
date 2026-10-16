"""
Train every detection component and produce the baseline comparison (§33, Experiments 1-4).

    python -m models.train

Split is chronological (70/15/15): fraud models must never train on the future.
  train : XGBoost, Logistic Regression, Isolation Forest
  val   : alert-threshold tuning + learned fusion weights
  test  : every number that goes in the results chapter
Class imbalance: scale_pos_weight + threshold tuning, no SMOTE (§36).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, confusion_matrix)
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from configs.settings import (ARTIFACT_DIR, DATA_DIR, RANDOM_SEED, REPORT_DIR, BEHAVIOUR_THRESHOLD, FUSION_MODE,
                              LOW_CONFIDENCE, HIGH_CONFIDENCE)
from fusion.fusion import EvidenceFusion
from graph.engine import GraphEngine, perpetrators
from models.detectors import BehaviourModel
from preprocessing.features import BEHAVIOUR_FEATURES, FEATURES, RAW_FEATURES, engineer_features


def chrono_split(tx: pd.DataFrame, a=0.70, b=0.85):
    ts = tx.timestamp.sort_values()
    t1, t2 = ts.iloc[int(len(ts) * a)], ts.iloc[int(len(ts) * b)]
    return tx.timestamp < t1, (tx.timestamp >= t1) & (tx.timestamp < t2), tx.timestamp >= t2


def best_f1_threshold(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return float(th[np.nanargmax(f1[:-1])])


def _fused_logits(fusion, X) -> np.ndarray:
    return fusion.lr.decision_function(fusion._design(X))


def _apply_platt_calibration(calibrator, logits) -> np.ndarray:
    calibrated_logits = calibrator.decision_function(np.asarray(logits).reshape(-1, 1))
    return expit(np.clip(calibrated_logits, -20, 20))


def _tied_pair_count(scores) -> int:
    counts = np.unique(scores, return_counts=True)[1]
    return int(np.sum(counts * (counts - 1) // 2))


def _check_calibration_ranking(y, before, after, tolerance=1e-4) -> None:
    before, after = np.asarray(before), np.asarray(after)
    order = np.argsort(before, kind="stable")
    sorted_before, sorted_after = before[order], after[order]

    # Only compare different pre-calibration score groups; ties within a group are allowed.
    prior_max = -np.inf
    start = 0
    while start < len(sorted_before):
        end = start + 1
        while end < len(sorted_before) and sorted_before[end] == sorted_before[start]:
            end += 1
        group = sorted_after[start:end]
        if prior_max > np.min(group):
            raise ValueError("Platt calibration inverted fused-score ranking on the test set")
        prior_max = max(prior_max, float(np.max(group)))
        start = end

    before_roc, after_roc = roc_auc_score(y, before), roc_auc_score(y, after)
    before_pr, after_pr = average_precision_score(y, before), average_precision_score(y, after)
    if abs(before_roc - after_roc) > tolerance or abs(before_pr - after_pr) > tolerance:
        raise ValueError(
            "Platt calibration changed test ranking metrics beyond tolerance "
            f"(ROC-AUC {before_roc:.6f} -> {after_roc:.6f}, "
            f"PR-AUC {before_pr:.6f} -> {after_pr:.6f}; tolerance {tolerance:g})"
        )

    before_ties, after_ties = _tied_pair_count(before), _tied_pair_count(after)
    if after_ties > before_ties:
        print(f"WARNING: Platt calibration increased tied-score pairs by {after_ties - before_ties} "
              f"({before_ties} -> {after_ties}).")


def metrics(name, y, score, threshold) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name, "threshold": round(float(threshold), 4),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, score) if len(set(score)) > 2 else float("nan"),
        "roc_auc": roc_auc_score(y, score) if len(set(score)) > 2 else float("nan"),
        "fpr": fp / max(fp + tn, 1), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def rule_baseline(F: pd.DataFrame) -> np.ndarray:
    """Typical bank rule: big-for-this-customer AND (new payee OR velocity)."""
    return ((F.amount_to_avg_ratio > 3) & ((F.new_beneficiary == 1) | (F.tx_count_1h >= 3))).astype(float).to_numpy()


def xgb_model(pos_weight, n=400):
    return xgb.XGBClassifier(n_estimators=n, max_depth=5, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="aucpr",
                             random_state=RANDOM_SEED, n_jobs=4)


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    tx = pd.read_parquet(DATA_DIR / "transactions.parquet")
    acc = pd.read_parquet(DATA_DIR / "accounts.parquet")
    tx["timestamp"] = pd.to_datetime(tx.timestamp)
    print(f"Loaded {len(tx):,} tx ({tx.fraud_label.mean():.3%} fraud). Engineering features...")
    F = engineer_features(tx, acc)
    y = tx.fraud_label.to_numpy()
    tr, va, te = chrono_split(tx)
    print(f"split  train={tr.sum():,}  val={va.sum():,}  test={te.sum():,}  "
          f"(fraud {y[tr].sum()}/{y[va].sum()}/{y[te].sum()})")
    pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)

    rows, scores = [], {}

    # 1. rule baseline
    scores["rule"] = rule_baseline(F[te])
    rows.append(metrics("Rule-based threshold", y[te], scores["rule"], 0.5))

    # 2. logistic regression
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    lr.fit(F[tr], y[tr])
    th = best_f1_threshold(y[va], lr.predict_proba(F[va])[:, 1])
    scores["lr"] = lr.predict_proba(F[te])[:, 1]
    rows.append(metrics("Logistic Regression", y[te], scores["lr"], th))

    # 3. XGBoost on raw features only (Experiment 2: value of feature engineering)
    xr = xgb_model(pos_w).fit(F.loc[tr, RAW_FEATURES], y[tr])
    th = best_f1_threshold(y[va], xr.predict_proba(F.loc[va, RAW_FEATURES])[:, 1])
    rows.append(metrics("XGBoost (raw features)", y[te], xr.predict_proba(F.loc[te, RAW_FEATURES])[:, 1], th))

    # 4. XGBoost full — the production classifier
    model = xgb_model(pos_w).fit(F[tr], y[tr])
    p_va = model.predict_proba(F[va])[:, 1]
    alert_th = best_f1_threshold(y[va], p_va)
    scores["xgb"] = model.predict_proba(F[te])[:, 1]
    rows.append(metrics("XGBoost (engineered)", y[te], scores["xgb"], alert_th))
    model.save_model(ARTIFACT_DIR / "xgb.json")

    # 5. Isolation Forest (unsupervised: trained without labels)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_SEED, n_jobs=4)
    Xb = F.loc[tr, BEHAVIOUR_FEATURES].to_numpy()
    iso.fit(Xb)
    raw = -iso.score_samples(Xb)
    lo, hi = float(np.percentile(raw, 50)), float(np.percentile(raw, 99.9))
    beh = BehaviourModel(iso, lo, hi)
    joblib.dump(iso, ARTIFACT_DIR / "iso_forest.joblib")
    (ARTIFACT_DIR / "iso_norm.json").write_text(json.dumps({"lo": lo, "hi": hi}))
    b_va, b_te = beh.score(F[va]), beh.score(F[te])
    rows.append(metrics("Isolation Forest (behaviour only)", y[te], b_te, BEHAVIOUR_THRESHOLD))

    # 6. graph score: val uses a graph of train-period history; test uses train+val
    print("Building graphs and scoring graph evidence...")
    ge_va = GraphEngine(tx[tr], perpetrators(tx[tr]))
    g_va = ge_va.batch_scores(tx[va])
    ge_te = GraphEngine(tx[tr | va], perpetrators(tx[tr | va]))
    g_te = ge_te.batch_scores(tx[te])
    rows.append(metrics("Graph score (analytics only)", y[te], g_te, 0.5))

    # 7. fusion (Experiment 6/7): weights fitted on validation, evaluated on test
    fusion = EvidenceFusion(alert_th, mode="learned").fit(np.column_stack([p_va, b_va, g_va]), y[va])
    X_f_va = np.column_stack([p_va, b_va, g_va])
    f_va_logits = _fused_logits(fusion, X_f_va)
    platt = LogisticRegression(max_iter=1000).fit(f_va_logits.reshape(-1, 1), y[va])
    f_va = _apply_platt_calibration(platt, f_va_logits)
    fused_th = best_f1_threshold(y[va], f_va)
    X_f_te = np.column_stack([scores["xgb"], b_te, g_te])
    f_te_before = fusion.predict_proba(X_f_te)
    f_te = _apply_platt_calibration(platt, _fused_logits(fusion, X_f_te))
    _check_calibration_ranking(y[te], f_te_before, f_te)
    rows.append(metrics("XGBoost + Behaviour + Graph (learned fusion)", y[te], f_te, fused_th))

    rule_f = EvidenceFusion(alert_th, mode="rule")
    rf_te = rule_f.predict_proba(np.column_stack([scores["xgb"], b_te, g_te]))
    rows.append(metrics("XGBoost + Behaviour + Graph (rule fusion: 2 of 3)", y[te], rf_te, 0.5))

    fusion.mode = FUSION_MODE
    fusion.save()
    (ARTIFACT_DIR / "thresholds.json").write_text(json.dumps(
        {"alert_threshold": alert_th, "fused_threshold": fused_th, "LOW": LOW_CONFIDENCE, "HIGH": HIGH_CONFIDENCE,
         "fusion_weights": fusion.weights()}, indent=2))

    res = pd.DataFrame(rows)
    res.to_csv(REPORT_DIR / "baseline_comparison.csv", index=False)
    md = res[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]].to_markdown(index=False, floatfmt=".3f")
    (REPORT_DIR / "baseline_comparison.md").write_text(md)
    print("\n" + md)

    # Experiment 3: what does the behaviour model add on the frauds XGBoost misses?
    missed = (y[te] == 1) & (scores["xgb"] < alert_th)
    caught = missed & (b_te >= BEHAVIOUR_THRESHOLD)
    by_pat = pd.DataFrame({"pattern": tx.loc[te, "pattern"].to_numpy(), "y": y[te],
                           "xgb_hit": scores["xgb"] >= alert_th, "fused_hit": f_te >= fused_th})
    per = by_pat[by_pat.y == 1].groupby("pattern")[["xgb_hit", "fused_hit"]].mean().round(3)
    print(f"\nFrauds missed by XGBoost on test: {missed.sum()}, of which behaviour model flags: {caught.sum()}")
    print("\nRecall by typology (test):\n" + per.to_string())
    per.to_csv(REPORT_DIR / "recall_by_pattern.csv")
    print("\nFusion weights:", json.dumps(fusion.weights(), indent=1))

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    imp.to_csv(REPORT_DIR / "xgb_feature_importance.csv")


if __name__ == "__main__":
    main()
