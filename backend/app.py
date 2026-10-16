"""
FastAPI backend (§17).   uvicorn backend.app:app --reload

Detection runs synchronously on ingest; investigation runs as a background task
(near-real-time, simulated streaming — §19). Nothing is ever auto-blocked.
"""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from agents.pipeline import FraudSystem
from agents.render import render_text
from backend.db import (AgentExecutionLog, Alert, Evidence, InvestigationCase, InvestigationReport, SessionLocal,
                        Transaction, init_db)
from configs.settings import ROOT, GRAPH_THRESHOLD

STATE: dict = {}
SIGNAL_FIELDS = {"classifier_score", "behaviour_anomaly_score", "graph_score", "amount_to_avg_ratio", "burst_10min",
                 "tx_count_1h", "first_transaction_to_this_beneficiary", "receiver_is_flagged", "sender_is_flagged",
                 "receiver_connected_to_flagged_accounts", "shared_device_with_other_accounts", "device_changed",
                 "signals_conflicting", "fan_in_count_window", "fan_out_count_window", "hour_of_day", "k_hop"}
LOCK = threading.Lock()   # FraudSystem holds in-memory history/graph; serialise writers


def jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if isinstance(x, np.generic)
                                 else x.to_dict() if isinstance(x, pd.Series) else str(x)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    STATE["sys"] = FraudSystem.from_artifacts()
    h = STATE["sys"].history
    STATE["types"] = sorted(map(str, h.transaction_type.dropna().unique()))
    STATE["typical_amount"] = float(h.amount.median())
    STATE["scored"] = {}      # tx_id -> detection output (features kept for SHAP / investigation)
    STATE["replay_pos"] = 0
    yield


app = FastAPI(title="Agentic Fraud Investigation", version="1.0", lifespan=lifespan)


class TxIn(BaseModel):
    transaction_id: str
    sender_account: str
    receiver_account: str
    amount: float = Field(gt=0)
    timestamp: datetime
    transaction_type: str = "TRANSFER"
    device_id: str
    ip_address: str | None = None
    merchant_id: str | None = None
    sender_balance_before: float | None = None
    sender_balance_after: float | None = None
    fraud_label: int | None = Field(None, description="dataset ground truth; stored for evaluation only")


class ReviewIn(BaseModel):
    decision: str = Field(pattern="^(approve|escalate|reject)$")
    note: str | None = None


# ----------------------------------------------------------------- persistence
def _store_tx(db, tx: dict):
    if db.get(Transaction, tx["transaction_id"]) is None:
        cols = Transaction.__table__.columns.keys()
        db.add(Transaction(**{k: tx.get(k) for k in cols}))


def _store_case(state: dict, alert_id: int):
    s = state
    with SessionLocal() as db:
        d = s["decision"]
        status = "escalated" if d["route"] == "human_review_escalate" else "open"
        db.add(InvestigationCase(case_id=s["case_id"], alert_id=alert_id, status=status, route=d["route"],
                                 confidence_score=d["confidence"], decision_json=jsonable(d),
                                 graph_json=jsonable(s.get("subgraph"))))
        db.flush()
        for it in s["evidence"]["items"]:
            db.add(Evidence(evidence_id=it["evidence_id"], case_id=s["case_id"], evidence_type=it["type"],
                            content_json=jsonable(it)))
        for i, entry in enumerate(s["agent_log"]):
            db.add(AgentExecutionLog(case_id=s["case_id"], step=i, agent_name=entry["agent"], status=entry["status"],
                                     duration_ms=entry["duration_ms"], output_json=jsonable(entry["summary"])))
        db.add(InvestigationReport(case_id=s["case_id"], report_json=jsonable(s.get("report")),
                                   report_text=render_text(s), narrator=STATE["sys"].narrator.name,
                                   validation_json=jsonable(s.get("validation")),
                                   recommended_action=d["recommended_action"]))
        a = db.get(Alert, alert_id)
        a.priority = d["priority"]
        db.commit()


def behaviour_profile(history: pd.DataFrame, tx: dict) -> dict:
    """The sender's normal behaviour BEFORE this transaction (for the behaviour card)."""
    ts = pd.to_datetime(tx["timestamp"])
    h = history[(history.sender_account == tx["sender_account"]) & (history.timestamp < ts)]
    if h.empty:
        return {"prior_count": 0}
    span_days = max((h.timestamp.max() - h.timestamp.min()).total_seconds() / 86400, 1.0)
    hours = h.timestamp.dt.hour
    return {"prior_count": int(len(h)), "avg_amount": float(h.amount.mean()), "median_amount": float(h.amount.median()),
            "max_amount": float(h.amount.max()), "tx_per_day": float(len(h) / span_days),
            "typical_hours": [int(hours.quantile(0.1)), int(hours.quantile(0.9))],
            "distinct_receivers": int(h.receiver_account.nunique()), "first_seen": str(h.timestamp.min())}


def _run_investigation(tx: dict, scored: dict, alert_id: int) -> dict | None:
    sysm: FraudSystem = STATE["sys"]
    with LOCK:
        state = sysm.investigate(tx, scored)
        if "decision" not in state:   # triage decided no investigation
            with SessionLocal() as db:
                db.get(Alert, alert_id).priority = state["triage"]["priority"]
                db.commit()
            return None
        state["subgraph"] = sysm.graph.k_hop_subgraph([tx["sender_account"], tx["receiver_account"]], k=state["k_hop"])
        state["decision"]["corroborating_types"] = state["evidence"]["summary"]["corroborating_evidence_types"]
        state["decision"]["conflict_reason"] = state["fusion"].get("conflict_reason")
        state["decision"]["behaviour_profile"] = behaviour_profile(sysm.history, tx)
    _store_case(state, alert_id)
    return state


def _ingest(tx: dict, background: BackgroundTasks | None, investigate_sync: bool = False) -> dict:
    sysm: FraudSystem = STATE["sys"]
    with LOCK:
        scored = sysm.score(tx, commit=True)
    STATE["scored"][tx["transaction_id"]] = scored
    out = {"transaction_id": tx["transaction_id"], **scored["scores"], "fused": scored["fusion"],
           "alert": scored["alert"], "detection_ms": scored["detection_ms"]}
    with SessionLocal() as db:
        _store_tx(db, tx)
        if scored["alert"]:
            a = Alert(transaction_id=tx["transaction_id"], risk_score=scored["scores"]["classifier_score"],
                      anomaly_score=scored["scores"]["behaviour_score"], graph_score=scored["scores"]["graph_score"],
                      fused_risk=scored["fusion"]["fused_risk"], graph_flag=scored["scores"]["graph_score"] >= GRAPH_THRESHOLD)
            db.add(a)
            db.commit()
            out["alert_id"] = a.alert_id
        else:
            db.commit()
    if scored["alert"]:
        if investigate_sync:
            st = _run_investigation(tx, scored, out["alert_id"])
            out["case_id"] = st["case_id"] if st else None
            if st:
                out["route"] = st["decision"]["route"]
                out["case_risk"] = st["decision"]["fused_risk"]
                out["signals"] = {i["field"]: i["value"] for i in st["evidence"]["items"]
                                  if i["field"] in SIGNAL_FIELDS or i["field"].startswith("pattern_")}
        elif background is not None:
            background.add_task(_run_investigation, tx, scored, out["alert_id"])
    return out


# ------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health")
def health():
    s = STATE["sys"]
    meta_file = ROOT / "datasets" / "data" / "dataset.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    thresholds_file = ROOT / "artifacts" / "thresholds.json"
    cal_file = ROOT / "artifacts" / "calibration.json"
    with SessionLocal() as db:
        n_tx = db.query(Transaction).count()
        n_alerts = db.query(Alert).count()
    return {"ok": True, "dataset": meta.get("dataset", "unknown"),
            "fused_threshold": s.fused_threshold, "fusion_weights": s.fusion.weights(),
            "thresholds": json.loads(thresholds_file.read_text()),
            "calibration": json.loads(cal_file.read_text()) if cal_file.exists() else None,
            "session": {"transactions_checked": n_tx, "alerts": n_alerts},
            "payment_types": STATE.get("types", []), "typical_amount": STATE.get("typical_amount"), "narrator": s.narrator.name, "fusion_mode": s.fusion.mode,
            "history_rows": len(s.history), "flagged_accounts": len(s.graph.flagged),
            "alert_threshold": s.det.alert_threshold}


@app.get("/evaluation")
def evaluation_metrics():
    """Research evaluation metrics and benchmarks (§37)."""
    res = {}
    reports_dir = ROOT / "reports"
    bc_file = reports_dir / "baseline_comparison.csv"
    if bc_file.exists():
        try:
            df = pd.read_csv(bc_file)
            res["baselines"] = jsonable(df.fillna(0).to_dict(orient="records"))
        except Exception:
            res["baselines"] = []
    pr_file = reports_dir / "recall_by_pattern.csv"
    if pr_file.exists():
        try:
            df = pd.read_csv(pr_file)
            res["pattern_recall"] = jsonable(df.to_dict(orient="records"))
        except Exception:
            res["pattern_recall"] = []
    fi_file = reports_dir / "xgb_feature_importance.csv"
    if fi_file.exists():
        try:
            df = pd.read_csv(fi_file)
            df.columns = ["feature", "importance"]
            res["feature_importance"] = jsonable(df.head(12).to_dict(orient="records"))
        except Exception:
            res["feature_importance"] = []
    lat_file = reports_dir / "latency.csv"
    if lat_file.exists():
        try:
            df = pd.read_csv(lat_file)
            res["latency"] = jsonable(df.fillna(0).to_dict(orient="records"))
        except Exception:
            res["latency"] = []
    res_file = reports_dir / "report_eval_summary.md"
    if res_file.exists():
        try:
            lines = [l.strip() for l in res_file.read_text().splitlines() if l.strip()]
            if len(lines) >= 3:
                cols = [c.strip() for c in lines[0].split("|") if c.strip()]
                rows = []
                for l in lines[2:]:
                    vals = [v.strip() for v in l.split("|") if v.strip()]
                    if len(vals) == len(cols):
                        row_dict = {}
                        for c, v in zip(cols, vals):
                            try:
                                row_dict[c] = float(v)
                            except ValueError:
                                row_dict[c] = v
                        rows.append(row_dict)
                res["ablation"] = rows
        except Exception:
            res["ablation"] = []
    return res


@app.post("/transactions")
def ingest(tx: TxIn, background: BackgroundTasks):
    return _ingest(tx.model_dump(), background)


@app.post("/predict/{tx_id}")
def predict(tx_id: str):
    s = STATE["scored"].get(tx_id)
    if not s:
        raise HTTPException(404, "transaction not scored yet; POST /transactions first")
    return jsonable({k: v for k, v in s.items() if k != "features"} | {"features": s["features"]})


@app.post("/investigate/{tx_id}")
def investigate(tx_id: str):
    s = STATE["scored"].get(tx_id)
    if not s:
        raise HTTPException(404, "transaction not scored yet")
    with SessionLocal() as db:
        t = db.get(Transaction, tx_id)
        alert = db.scalars(select(Alert).where(Alert.transaction_id == tx_id)).first()
        if alert is None:     # analyst forces investigation on a non-alerted transaction
            alert = Alert(transaction_id=tx_id, risk_score=s["scores"]["classifier_score"],
                          anomaly_score=s["scores"]["behaviour_score"], graph_score=s["scores"]["graph_score"],
                          fused_risk=s["fusion"]["fused_risk"], graph_flag=False)
            db.add(alert)
            db.commit()
        tx = {c: getattr(t, c) for c in Transaction.__table__.columns.keys()}
    st = _run_investigation(tx, s, alert.alert_id)
    if st is None:
        return {"investigated": False, "reason": "triage: below investigation threshold"}
    return get_report(st["case_id"])


@app.get("/alerts")
def list_alerts(limit: int = 500):
    with SessionLocal() as db:
        rows = db.execute(select(Alert, InvestigationCase, InvestigationReport, Transaction)
                          .join(Transaction, Transaction.transaction_id == Alert.transaction_id)
                          .outerjoin(InvestigationCase, InvestigationCase.alert_id == Alert.alert_id)
                          .outerjoin(InvestigationReport, InvestigationReport.case_id == InvestigationCase.case_id)
                          .order_by(Alert.fused_risk.desc(), Alert.alert_id.desc()).limit(limit)).all()
        case_ids = [c.case_id for _, c, _, _ in rows if c]
        signals: dict[str, dict] = {}
        if case_ids:
            for e in db.scalars(select(Evidence).where(Evidence.case_id.in_(case_ids))):
                f = e.content_json.get("field", "")
                if f in SIGNAL_FIELDS or f.startswith("pattern_"):
                    signals.setdefault(e.case_id, {})[f] = e.content_json.get("value")
        return [{"alert_id": a.alert_id, "transaction_id": a.transaction_id, "classifier_score": a.risk_score,
                 "signals": signals.get(c.case_id, {}) if c else {},
                 "behaviour_score": a.anomaly_score, "graph_score": a.graph_score, "fused_risk": a.fused_risk,
                 "priority": a.priority, "created_at": a.created_at,
                 "case_id": c.case_id if c else None, "route": c.route if c else None,
                 "reviewer_status": r.reviewer_status if r else None,
                 "amount": t.amount, "sender": t.sender_account, "receiver": t.receiver_account,
                 "timestamp": t.timestamp} for a, c, r, t in rows]


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: int):
    with SessionLocal() as db:
        a = db.get(Alert, alert_id)
        if not a:
            raise HTTPException(404)
        c = db.scalars(select(InvestigationCase).where(InvestigationCase.alert_id == alert_id)).first()
        t = db.get(Transaction, a.transaction_id)
        return {"alert": {k: getattr(a, k) for k in Alert.__table__.columns.keys()},
                "transaction": {k: getattr(t, k) for k in Transaction.__table__.columns.keys() if k != "fraud_label"},
                "case_id": c.case_id if c else None}


@app.get("/graph/{account_id}")
def graph(account_id: str, k: int = 2):
    with LOCK:
        g = STATE["sys"].graph.k_hop_subgraph([account_id], k=min(k, 3))
    if not g["nodes"]:
        raise HTTPException(404, "account not in graph")
    return jsonable(g)


@app.get("/entity/{node_id}")
def entity(node_id: str):
    """What an analyst sees when they click a node: an account's links or a device's accounts."""
    with LOCK:
        g = STATE["sys"].graph
        if node_id not in g.G:
            raise HTTPException(404, "not in the transaction graph")
        kind = g.G.nodes[node_id].get("kind", "account")
        if kind == "device":
            accts = sorted(g.device_accounts.get(node_id, set()))
            return {"id": node_id, "kind": kind, "accounts": [{"id": a, "flagged": a in g.flagged} for a in accts[:25]],
                    "account_count": len(accts)}
        out = [(v, d) for v, d in g.G[node_id].items() if d["rel"] == "SENT_TO"]
        inc = [(u, g.G[u][node_id]) for u in g.G.predecessors(node_id) if g.G[u][node_id]["rel"] == "SENT_TO"]
        devices = [v for v, d in g.G[node_id].items() if d["rel"] == "USED_DEVICE"]
        top = lambda edges: [{"id": v, "amount": round(d["amount"], 2), "count": d["count"], "flagged": v in g.flagged}  # noqa: E731
                             for v, d in sorted(edges, key=lambda e: -e[1]["amount"])[:8]]
        return {"id": node_id, "kind": kind, "flagged": node_id in g.flagged,
                "hops_to_flagged": g.hops_to_flagged(node_id, exclude_self=True),
                "sent_total": round(sum(d["amount"] for _, d in out), 2), "sent_count": sum(d["count"] for _, d in out),
                "received_total": round(sum(d["amount"] for _, d in inc), 2), "received_count": sum(d["count"] for _, d in inc),
                "sends_to": top(out), "receives_from": top(inc),
                "devices": [{"id": d, "shared_with": len(g.device_accounts.get(d, set())) - 1} for d in devices]}


@app.get("/related/{tx_id}")
def related(tx_id: str, limit: int = 8):
    """Recent payments involving this payment's sender or receiver, up to and including it."""
    with SessionLocal() as db:
        t = db.get(Transaction, tx_id)
        if not t:
            raise HTTPException(404)
        accts, ts = [t.sender_account, t.receiver_account], pd.to_datetime(t.timestamp)
    with LOCK:
        h = STATE["sys"].history
        m = (h.sender_account.isin(accts) | h.receiver_account.isin(accts)) & (h.timestamp <= ts)
        rows = h.loc[m, ["transaction_id", "timestamp", "sender_account", "receiver_account", "amount",
                         "transaction_type"]].sort_values("timestamp").tail(limit)
    ids = [str(x) for x in rows.transaction_id]
    with SessionLocal() as db:
        alerted = set(db.scalars(select(Alert.transaction_id).where(Alert.transaction_id.in_(ids)))) if ids else set()
    return [{"transaction_id": r.transaction_id, "timestamp": str(r.timestamp), "sender": r.sender_account,
             "receiver": r.receiver_account, "amount": float(r.amount), "type": r.transaction_type,
             "this": r.transaction_id == tx_id, "alerted": r.transaction_id in alerted}
            for r in rows.iloc[::-1].itertuples(index=False)]


@app.get("/explanation/{tx_id}")
def explanation(tx_id: str):
    s = STATE["scored"].get(tx_id)
    if not s:
        raise HTTPException(404)
    return {"transaction_id": tx_id, "classifier_score": s["scores"]["classifier_score"],
            "shap": STATE["sys"].det.explainer.explain_row(s["features"], top_k=8)}


@app.get("/report/{case_id}")
def get_report(case_id: str):
    with SessionLocal() as db:
        c = db.get(InvestigationCase, case_id)
        if not c:
            raise HTTPException(404)
        r = db.scalars(select(InvestigationReport).where(InvestigationReport.case_id == case_id)).first()
        ev = db.scalars(select(Evidence).where(Evidence.case_id == case_id)).all()
        logs = db.scalars(select(AgentExecutionLog).where(AgentExecutionLog.case_id == case_id)
                          .order_by(AgentExecutionLog.step)).all()
        a = db.get(Alert, c.alert_id)
        return {"case_id": case_id, "transaction_id": a.transaction_id, "status": c.status, "route": c.route,
                "decision": c.decision_json, "report": r.report_json, "report_text": r.report_text,
                "narrator": r.narrator, "validation": r.validation_json,
                "reviewer_status": r.reviewer_status, "reviewer_note": r.reviewer_note,
                "reviewed_at": r.reviewed_at, "created_at": c.created_at,
                "evidence": [e.content_json for e in ev], "graph": c.graph_json,
                "agent_log": [{"agent": l.agent_name, "status": l.status, "duration_ms": l.duration_ms,
                               "output": l.output_json} for l in logs]}


@app.post("/review/{case_id}")
def review(case_id: str, body: ReviewIn):
    status_map = {"approve": ("approved", "closed"), "escalate": ("escalated", "escalated"), "reject": ("rejected", "closed")}
    with SessionLocal() as db:
        c = db.get(InvestigationCase, case_id)
        if not c:
            raise HTTPException(404)
        r = db.scalars(select(InvestigationReport).where(InvestigationReport.case_id == case_id)).first()
        r.reviewer_status, c.status = status_map[body.decision]
        r.reviewer_note, r.reviewed_at = body.note, datetime.utcnow()
        tx_id = db.get(Alert, c.alert_id).transaction_id
        db.commit()
    if body.decision == "approve":
        # confirmed fraud: the receiver becomes a known perpetrator for future graph evidence
        with SessionLocal() as db:
            t = db.get(Transaction, tx_id)
        with LOCK:
            STATE["sys"].graph.flag([t.receiver_account])
    return {"case_id": case_id, "reviewer_status": r.reviewer_status, "case_status": c.status,
            "reviewed_at": r.reviewed_at}


# ------------------------------------------------------------- CSV upload
ALIASES = {
    "transaction_id": ["transaction_id", "id", "txn_id", "reference", "ref"],
    "sender_account": ["sender_account", "sender", "from", "from_account", "payer", "source"],
    "receiver_account": ["receiver_account", "receiver", "to", "to_account", "payee", "beneficiary", "destination"],
    "amount": ["amount", "amt", "value"],
    "timestamp": ["timestamp", "time", "date", "datetime", "date_time"],
    "transaction_type": ["transaction_type", "type", "payment_type", "mode", "channel", "payment_format"],
    "device_id": ["device_id", "device"],
    "ip_address": ["ip_address", "ip"],
    "expected": ["expected", "expected_result", "label"],
    "note": ["what_this_tests", "note", "description", "comment"],
}
MAX_UPLOAD_ROWS = 300


class UploadIn(BaseModel):
    filename: str = "upload.csv"
    csv_text: str


def _norm(c: str) -> str:
    return "".join(ch for ch in str(c).lower() if ch.isalnum())


@app.post("/upload")
def upload(body: UploadIn):
    """Check every row of an uploaded CSV through the full pipeline (scoring, alert, investigation)."""
    import io
    import uuid as _uuid
    try:
        raw = pd.read_csv(io.StringIO(body.csv_text), dtype=str).fillna("")
    except Exception as e:
        raise HTTPException(400, f"Couldn't read the file as CSV: {e}")
    cols = {_norm(c): c for c in raw.columns}
    mapping = {k: next((cols[_norm(a)] for a in al if _norm(a) in cols), None) for k, al in ALIASES.items()}
    missing = [k for k in ["sender_account", "receiver_account", "amount", "timestamp"] if not mapping[k]]
    if missing:
        raise HTTPException(400, "Missing required column(s): " + ", ".join(missing) +
                            ". Download the sample file to see the expected columns.")
    sysm: FraudSystem = STATE["sys"]
    known_types = set(sysm.history.transaction_type.dropna().unique())
    default_type = sysm.history.transaction_type.mode().iat[0]
    batch = _uuid.uuid4().hex[:4].upper()
    rows, errors, warnings = [], [], []
    for i, r in raw.head(MAX_UPLOAD_ROWS).iterrows():
        get = lambda k: str(r[mapping[k]]).strip() if mapping[k] else ""  # noqa: E731
        n = i + 2   # row number as seen in a spreadsheet (header is row 1)
        snd, rcv = get("sender_account"), get("receiver_account")
        amt_s = "".join(ch for ch in get("amount") if ch.isdigit() or ch in ".-")
        ts = pd.to_datetime(get("timestamp"), errors="coerce", format="mixed")
        problems = [m for ok, m in [(snd, "no sender"), (rcv, "no receiver"), (amt_s, "no amount"),
                                    (not pd.isna(ts), "date/time not recognised")] if not ok]
        amt = float(amt_s) if amt_s and amt_s not in "-." else 0.0
        if amt_s and amt <= 0:
            problems.append("amount must be more than 0")
        if problems:
            errors.append({"row": n, "problem": ", ".join(problems)})
            continue
        ttype = get("transaction_type") or default_type
        if ttype not in known_types:
            warnings.append(f"Row {n}: payment type '{ttype}' never appeared in the training data, so the model can't use it.")
        rows.append({"row": n, "expected": get("expected") or None, "note": get("note") or None, "tx": {
            "transaction_id": f"UP{batch}-{get('transaction_id') or n}", "sender_account": snd, "receiver_account": rcv,
            "amount": amt, "timestamp": ts.to_pydatetime(), "transaction_type": ttype,
            "device_id": get("device_id") or f"DEV_{snd}", "ip_address": get("ip_address") or None,
            "merchant_id": None, "sender_balance_before": None, "sender_balance_after": None, "fraud_label": None}})
    if len(raw) > MAX_UPLOAD_ROWS:
        warnings.append(f"Only the first {MAX_UPLOAD_ROWS} rows were checked.")
    results = []
    for item in sorted(rows, key=lambda x: x["tx"]["timestamp"]):     # in time order, so patterns build up
        out = _ingest(item["tx"], None, investigate_sync=True)
        t = item["tx"]
        results.append({"row": item["row"], "expected": item["expected"], "note": item["note"], "transaction_id": t["transaction_id"],
                        "sender": t["sender_account"], "receiver": t["receiver_account"], "amount": t["amount"],
                        "timestamp": str(t["timestamp"]), "type": t["transaction_type"],
                        "alert": out["alert"], "fused_risk": out.get("case_risk", out["fused"]["fused_risk"]),
                        "case_id": out.get("case_id"), "route": out.get("route"), "signals": out.get("signals", {})})
    results.sort(key=lambda x: x["row"])
    return jsonable({"filename": body.filename, "checked": len(results), "flagged": sum(r["alert"] for r in results),
                     "skipped": len(errors), "errors": errors, "warnings": sorted(set(warnings)), "results": results})


@app.get("/upload/sample")
def upload_sample():
    """A demo CSV built from the LIVE system, so it uses real accounts and the dataset's own scale."""
    from datetime import timedelta
    from fastapi.responses import Response
    sysm: FraudSystem = STATE["sys"]
    h, g = sysm.history, sysm.graph
    t0 = (h.timestamp.max() + timedelta(days=1)).normalize() + timedelta(hours=9)
    typ = h.transaction_type.mode().iat[0]
    base = float(h.amount.median())
    counts = h.sender_account.value_counts()
    regular = [a for a in counts.index[:400] if a not in g.flagged][:3]          # accounts with lots of history
    flagged = [a for a in g.flagged if a in g.A and g.A.degree(a) > 1][:2] or ["DEMO_KNOWN_FRAUD"]
    lines = ["transaction_id,timestamp,sender_account,receiver_account,amount,transaction_type,device_id,expected,what_this_tests"]
    add = lambda i, dt, s, r, a, d, e, why: lines.append(  # noqa: E731
        f"{i},{(t0 + dt).strftime('%Y-%m-%d %H:%M')},{s},{r},{a:.2f},{typ},{d},{e},{why}")
    for j, a in enumerate(regular[:2]):
        mine = h[h.sender_account == a]
        usual_rcv = mine.receiver_account.mode().iat[0]
        add(f"N{j+1}", timedelta(minutes=30 * j), a, usual_rcv, float(mine.amount.median()), f"DEV_{a}", "normal",
            "Usual amount to a usual payee")
    add("K1", timedelta(hours=15), regular[-1] if regular else "DEMO_PAYER", flagged[0], base * 6, f"DEV_{regular[-1] if regular else 'DEMO_PAYER'}",
        "suspicious", "Large first payment to an account from a past fraud case")
    for j in range(10):
        add(f"F{j+1}", timedelta(hours=1, minutes=40 * j), f"DEMO_SENDER_{j+1:02d}", "DEMO_MULE_01", base * (1.8 + 0.05 * j),
            f"DEV_DEMO_SENDER_{j+1:02d}", "normal" if j < 4 else "suspicious", f"Fan-in build-up: sender {j+1} of 10 into one account")
    add("F11", timedelta(hours=9), "DEMO_MULE_01", flagged[-1], base * 18, "DEV_DEMO_MULE_01", "suspicious",
        "Mule forwards the collected money to a known fraud account")
    for j in range(6):
        add(f"B{j+1}", timedelta(hours=17, minutes=2 * j), "DEMO_VICTIM_01", f"DEMO_NEW_PAYEE_{j+1}", base * 5,
            "DEMO_NEW_PHONE", "normal" if j < 2 else "suspicious", f"Night-time burst: payment {j+1} of 6 to new payees")
    for j in range(4):
        add(f"D{j+1}", timedelta(hours=11, minutes=15 * j), f"DEMO_RING_{j+1}", "DEMO_COLLECTOR", base * 3,
            "DEMO_SHARED_DEVICE", "normal" if j < 2 else "suspicious", f"Ring of accounts on one device: account {j+1} of 4")
    return Response("\n".join(lines) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="demo_transactions.csv"'})


@app.post("/demo/replay")
def demo_replay(n: int = 20, fraud_boost: bool = True):
    """Replay the next n held-out transactions (demo helper). fraud_boost skips most legit rows
    (they are still added to history) so a short demo produces alerts."""
    pool = STATE["sys"].replay_pool
    results, sent = [], 0
    while sent < n and STATE["replay_pos"] < len(pool):
        row = pool.iloc[STATE["replay_pos"]].to_dict()
        STATE["replay_pos"] += 1
        row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        row.pop("pattern", None), row.pop("pattern_id", None)
        if fraud_boost and not row["fraud_label"] and np.random.random() > 0.01:
            with LOCK:
                STATE["sys"].commit(row)
            continue
        results.append(_ingest(row, None, investigate_sync=True))
        sent += 1
    return {"replayed": sent, "alerts": sum(r["alert"] for r in results), "results": jsonable(results)}
