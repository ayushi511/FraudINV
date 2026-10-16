"""Relational store for the §10 schema. SQLite by default; set DATABASE_URL for PostgreSQL:
    DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/fraud
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from configs.settings import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"
    account_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_age_days: Mapped[int | None] = mapped_column(Integer)
    location: Mapped[str | None] = mapped_column(String)


class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id: Mapped[str] = mapped_column(String, primary_key=True)
    sender_account: Mapped[str] = mapped_column(String, index=True)
    receiver_account: Mapped[str] = mapped_column(String, index=True)
    amount: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    transaction_type: Mapped[str] = mapped_column(String)
    device_id: Mapped[str] = mapped_column(String, index=True)
    ip_address: Mapped[str | None] = mapped_column(String)
    merchant_id: Mapped[str | None] = mapped_column(String)
    sender_balance_before: Mapped[float | None] = mapped_column(Float)
    sender_balance_after: Mapped[float | None] = mapped_column(Float)
    # ground truth from the dataset: stored for evaluation, NEVER passed to agents
    fraud_label: Mapped[int | None] = mapped_column(Integer)


class Alert(Base):
    __tablename__ = "alerts"
    alert_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.transaction_id"), index=True)
    risk_score: Mapped[float] = mapped_column(Float)          # classifier
    anomaly_score: Mapped[float] = mapped_column(Float)       # behaviour
    graph_score: Mapped[float] = mapped_column(Float)
    fused_risk: Mapped[float] = mapped_column(Float)
    graph_flag: Mapped[bool] = mapped_column(Boolean)
    priority: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class InvestigationCase(Base):
    __tablename__ = "investigation_cases"
    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.alert_id"))
    status: Mapped[str] = mapped_column(String, default="open")   # open / escalated / closed
    route: Mapped[str] = mapped_column(String)
    confidence_score: Mapped[float] = mapped_column(Float)
    decision_json: Mapped[dict] = mapped_column(JSON)
    graph_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Evidence(Base):
    __tablename__ = "evidence"
    evidence_pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evidence_id: Mapped[str] = mapped_column(String)      # E1..En, unique within a case
    case_id: Mapped[str] = mapped_column(ForeignKey("investigation_cases.case_id"), index=True)
    evidence_type: Mapped[str] = mapped_column(String)
    content_json: Mapped[dict] = mapped_column(JSON)


class AgentExecutionLog(Base):
    __tablename__ = "agent_execution_log"
    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("investigation_cases.case_id"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    agent_name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    duration_ms: Mapped[float] = mapped_column(Float)
    output_json: Mapped[dict] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class InvestigationReport(Base):
    __tablename__ = "investigation_reports"
    report_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("investigation_cases.case_id"), index=True)
    report_json: Mapped[dict | None] = mapped_column(JSON)     # None => raw-evidence route, no narrative
    report_text: Mapped[str | None] = mapped_column(Text)
    narrator: Mapped[str] = mapped_column(String)
    validation_json: Mapped[dict | None] = mapped_column(JSON)
    recommended_action: Mapped[str] = mapped_column(String)
    reviewer_status: Mapped[str] = mapped_column(String, default="pending")  # pending/approved/escalated/rejected
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)


engine = create_engine(DATABASE_URL, future=True,
                       connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db(drop: bool = False):
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
