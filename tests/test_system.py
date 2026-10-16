"""pytest -q   (run `python -m datasets.synthetic && python -m models.train` first)"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from agents.narrator import TemplateNarrator
from agents.pipeline import FraudSystem
from agents.validator import validate_report
from evaluation.calibrate import _alert_reliability_table
from fusion.fusion import EvidenceFusion
from graph.engine import GraphEngine
from models.train import _check_calibration_ranking
from preprocessing.features import engineer_features, features_for_new


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session")
def S():
    return FraudSystem.from_artifacts(narrator=TemplateNarrator())


def _tx(i, s, r, amt, ts, dev=None, typ="TRANSFER"):
    return dict(transaction_id=f"T{i}", sender_account=s, receiver_account=r, amount=amt, timestamp=ts,
                transaction_type=typ, device_id=dev or f"D_{s}", ip_address=None, merchant_id=None,
                sender_balance_before=None, sender_balance_after=None, fraud_label=0, pattern="normal", pattern_id=None)


T0 = datetime(2026, 1, 1, 12)


def test_alert_reliability_table_uses_matching_score_bin():
    table = _alert_reliability_table(np.array([0.11, 0.18, 1.0]), np.array([True, True, True]),
                                     np.array([True, False, True]), np.array([1.0, 3.0, 1.0]))

    assert table[1]["min"] == 0.1 and table[1]["max"] == 0.2
    assert table[1]["alerts"] == 2
    assert table[1]["real_rate"] == pytest.approx(0.25)
    assert table[-1]["alerts"] == 1


# ------------------------------------------------------------------ features
def test_online_features_match_batch(S):
    """Train/serve skew check: the online path must equal the batch path exactly."""
    hist = S.history.tail(20000).reset_index(drop=True)
    batch = engineer_features(hist)
    for i in [19990, 19995, 19999]:
        row = hist.iloc[i].to_dict()
        online = features_for_new(row, hist.iloc[:i])
        np.testing.assert_allclose(online.to_numpy(), batch.iloc[i].to_numpy(), rtol=1e-6, atol=1e-6)


def test_features_use_only_the_past():
    rows = [_tx(i, "A", "B", 100, T0 + timedelta(hours=i)) for i in range(5)]
    rows.append(_tx(5, "A", "C", 1000, T0 + timedelta(hours=5)))
    f = engineer_features(pd.DataFrame(rows))
    assert f.iloc[5].amount_to_avg_ratio == pytest.approx(10.0)
    assert f.iloc[5].new_beneficiary == 1 and f.iloc[4].new_beneficiary == 0
    assert f.iloc[0].amount_zscore == 0  # no history -> neutral, not NaN


# --------------------------------------------------------------------- graph
def test_graph_detects_typologies():
    rows = [_tx(0, "X", "Y", 50000, T0), _tx(1, "Y", "Z", 49000, T0 + timedelta(hours=1)),
            _tx(2, "Z", "X", 48000, T0 + timedelta(hours=2))]                     # cycle
    rows += [_tx(10 + i, f"S{i}", "HUB", 9500, T0 + timedelta(minutes=i)) for i in range(8)]   # fan-in
    rows += [_tx(30 + i, "SRC", f"M{i}", 10000, T0 + timedelta(minutes=i)) for i in range(4)]  # scatter
    rows += [_tx(40 + i, f"M{i}", "COL", 9700, T0 + timedelta(hours=3, minutes=i)) for i in range(4)]  # gather
    ge = GraphEngine(pd.DataFrame(rows), flagged={"HUB"})
    cyc = ge.detect_patterns("Z", "X", T0 + timedelta(hours=2))
    assert any(p["pattern"] == "cycle" and p["length"] == 3 for p in cyc["patterns"])
    fin = ge.detect_patterns("S7", "HUB", T0 + timedelta(minutes=8))
    assert any(p["pattern"] == "fan_in" for p in fin["patterns"])
    sg = ge.detect_patterns("SRC", "M0", T0 + timedelta(hours=4))
    assert any(p["pattern"] == "scatter_gather" and p["collector"] == "COL" for p in sg["patterns"])
    ev = ge.evidence("S1", "HUB", "D_S1", T0 + timedelta(minutes=8))
    assert ev["receiver_is_flagged"] and GraphEngine.score(ev) > 0.7


def test_shared_device_ring():
    rows = [_tx(i, f"MULE{i}", "COL", 1000, T0 + timedelta(minutes=i), dev="SHARED") for i in range(5)]
    ge = GraphEngine(pd.DataFrame(rows))
    assert len(ge.shared_device_accounts("SHARED", "MULE0")) == 4


# -------------------------------------------------------------------- fusion
def test_rule_fusion_two_of_three():
    f = EvidenceFusion(alert_threshold=0.5, mode="rule")
    assert f.fuse(0.9, 0.9, 0.0).confidence == 0.75
    assert f.fuse(0.9, 0.1, 0.0).fused_risk < 0.5          # one signal alone doesn't escalate
    assert f.fuse(0.9, 0.9, 0.9).confidence == 0.95


def test_conflict_detection():
    f = EvidenceFusion(alert_threshold=0.5, mode="rule")
    r = f.fuse(0.95, 0.05, 0.0)
    assert r.conflicting and "classifier" in r.conflict_reason


def test_calibration_allows_saturated_ties(capsys):
    y = np.array([0, 0, 1, 1])
    before = np.array([0.1, 0.2, 0.8, 0.9])
    saturated = np.array([0.1, 0.2, 0.999999, 0.999999])

    _check_calibration_ranking(y, before, saturated)

    assert "increased tied-score pairs by 1" in capsys.readouterr().out


def test_calibration_rejects_order_inversions():
    with pytest.raises(ValueError, match="inverted fused-score ranking"):
        _check_calibration_ranking(np.array([0, 1]), np.array([0.1, 0.9]), np.array([0.8, 0.2]))


def test_calibration_rejects_auc_drift_without_inversions():
    y = np.array([0, 1, 0, 1])
    before = np.array([0.1, 0.2, 0.3, 0.4])
    tied = np.array([0.1, 0.1, 0.1, 0.4])

    with pytest.raises(ValueError, match="ranking metrics beyond tolerance"):
        _check_calibration_ranking(y, before, tied)


# ----------------------------------------------------------------- validator
_CASE = {}


def _hi(S):
    """A classifier score safely above THIS dataset's tuned alert threshold."""
    return min(0.9995, S.det.alert_threshold + (1 - S.det.alert_threshold) / 2)


def _case(S):
    """A held-out fraud whose (strong, agreeing) scores reach the escalate route.

    Scores are set explicitly so the tests exercise the pipeline's routing logic
    rather than depending on how easy a particular dataset is.
    """
    if "c" not in _CASE:
        for tx in S.replay_pool[S.replay_pool.fraud_label == 1].head(300).to_dict("records"):
            sc = S.score(tx, commit=False)
            sc = {**sc, "scores": {"classifier_score": _hi(S), "behaviour_score": 0.95,
                                   "graph_score": sc["scores"]["graph_score"]}}
            st = S.investigate(tx, sc)
            if st.get("decision", {}).get("route") == "human_review_escalate":
                _CASE["c"] = (tx, sc)
                break
    if "c" not in _CASE:
        pytest.skip("no held-out fraud case reaches the escalate route on this dataset")
    return _CASE["c"]


def test_validator_rejects_hallucinations(S):
    tx, sc = _case(S)
    st = S.investigate(tx, sc)
    ev, good = st["evidence"], st["report"]
    assert validate_report(good, ev)["valid"]
    bad = {**good, "alert_reason": {"claim": "Amount is 97.3x the average.", "evidence_ids": [good["alert_reason"]["evidence_ids"][0]]}}
    assert not validate_report(bad, ev)["valid"]                                   # invented number
    bad = {**good, "alert_reason": {"claim": "Flagged.", "evidence_ids": ["E999"]}}
    assert not validate_report(bad, ev)["valid"]                                   # non-existent evidence
    bad = {**good, "alert_reason": {"claim": "Flagged.", "evidence_ids": []}}
    assert not validate_report(bad, ev)["valid"]                                   # uncited claim
    bad = {**good, "recommended_action": "block_account"}
    assert not validate_report(bad, ev)["valid"]                                   # autonomous action


# ------------------------------------------------------- supervisor routing
def test_high_confidence_escalates(S):
    tx, sc = _case(S)
    st = S.investigate(tx, sc)
    assert st["decision"]["route"] == "human_review_escalate"
    assert st["decision"]["autonomous_action_taken"] is False
    assert [l["agent"] for l in st["agent_log"]] == ["triage", "evidence_correlation", "supervisor_gate",
                                                     "investigation", "supervisor_final"]


def test_conflicting_signals_skip_the_llm(S):
    tx, sc = _case(S)
    sc = {**sc, "scores": {"classifier_score": _hi(S), "behaviour_score": 0.02, "graph_score": 0.0}}
    old = S.fusion
    S.fusion = EvidenceFusion(S.det.alert_threshold, mode="rule")   # every model trusted
    S.ablate = {"graph"}                            # force graph score 0 in correlation
    try:
        st = S.investigate(tx, sc)
    finally:
        S.fusion, S.ablate = old, set()
    assert st["decision"]["route"] == "human_review_raw_evidence"
    assert "investigation" not in [l["agent"] for l in st["agent_log"]]   # LLM never called
    assert st.get("report") is None


def test_medium_confidence_widens_once_then_partial(S):
    tx, sc = _case(S)
    old = S.fusion
    S.fusion = EvidenceFusion(S.det.alert_threshold, mode="rule")     # 2-of-3 => confidence 0.75
    S.ablate = {"graph"}
    try:
        sc = {**sc, "scores": {"classifier_score": _hi(S), "behaviour_score": 0.95, "graph_score": 0.0}}
        st = S.investigate(tx, sc)
    finally:
        S.fusion, S.ablate = old, set()
    agents = [l["agent"] for l in st["agent_log"]]
    assert agents.count("evidence_correlation") == 2          # widened k-hop exactly once
    assert st["k_hop"] == 3
    assert st["decision"]["route"] == "human_review_needs_more_evidence"
    assert st["report"]["recommended_action"] == "request_more_evidence"


class _Failing:
    name = "failing"

    def draft(self, *a, **k):
        raise TimeoutError("LLM API unavailable")


class _Hallucinating(TemplateNarrator):
    name = "hallucinating"

    def draft(self, evidence, context, feedback=None):
        r = super().draft(evidence, context)
        r["network_findings"] = [{"claim": "Receiver linked to 57 mule accounts.", "evidence_ids": ["E2"]}]
        return r


@pytest.mark.parametrize("narrator", [_Failing(), _Hallucinating()])
def test_llm_failure_routes_to_human_with_raw_evidence(S, narrator):
    tx, sc = _case(S)
    old, S.narrator = S.narrator, narrator
    try:
        st = S.investigate(tx, sc)
    finally:
        S.narrator = old
    assert st["decision"]["route"] == "human_review_raw_evidence"
    assert st["report"] is None and st["llm_attempts"] == 3      # retried, then gave up safely
    assert st["evidence"]["items"]                                # case is never silently dropped


def test_ground_truth_never_reaches_agents(S):
    tx, sc = _case(S)
    st = S.investigate(tx, sc)
    fields = {i["field"] for i in st["evidence"]["items"]}
    assert not {"fraud_label", "pattern", "pattern_id"} & fields
