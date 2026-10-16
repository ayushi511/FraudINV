"""Central configuration. Every threshold the examiner might ask about lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

RANDOM_SEED = 42

# --- Detection -------------------------------------------------------------
# Alert threshold on the XGBoost probability. Tuned on validation PR curve by
# models/train.py and written to artifacts/thresholds.json; this is the fallback.
DEFAULT_ALERT_THRESHOLD = 0.5
# Keep PaySim-style post-transaction balance columns OUT of features (leakage, §35).
USE_BALANCE_FEATURES = False

# --- Behaviour / graph signal thresholds (used by rule-based fusion, §31) ----
BEHAVIOUR_THRESHOLD = 0.60     # normalised isolation-forest score
GRAPH_THRESHOLD = 0.50         # graph suspicion score
DEFAULT_K_HOP = 2
WIDENED_K_HOP = 3

# --- Supervisor / human-in-the-loop (§15) ------------------------------------
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.50
MIN_CORROBORATING_TYPES = 3

# --- Fusion mode: "rule" (any 2 of 3) or "learned" (logistic regression) ----
FUSION_MODE = os.getenv("FUSION_MODE", "learned")

# --- LLM ---------------------------------------------------------------------
# If ANTHROPIC_API_KEY is unset, the Investigation Agent uses a deterministic
# template narrator that produces the same JSON schema, so the pipeline and the
# validator still run offline.
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_TEMPERATURE = 0.1
LLM_MAX_RETRIES = 2

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'fraud.db'}")
