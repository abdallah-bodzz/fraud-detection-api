"""
config.py
---------
Central configuration loaded from environment variables (or .env).
One place to change paths and settings — nothing hardcoded elsewhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
# BASE_DIR anchors on this file's own location rather than the current
# working directory, so `uvicorn src.main:app` resolves identically whether
# launched from the project root, a CI runner, or a container WORKDIR.
#
# config.py lives at <project_root>/src/config.py, so two `.parent` calls
# reach the project root. This is a fixed, known layout — not a package
# meant to be nested or installed — so an explicit relative anchor is used
# instead of a marker-file directory walk, which would add complexity
# without a corresponding real risk here.
BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(env_var: str, default_relative: str) -> Path:
    """
    Resolve a configurable path against BASE_DIR.

    An absolute path supplied via the environment is honoured as-is; a
    relative path (env-supplied or default) is joined onto BASE_DIR. This
    keeps .env overrides working whether the deployer supplies an absolute
    volume-mount path (common in containerized deployments) or a simple
    relative path for local development.
    """
    raw = os.getenv(env_var, default_relative)
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else BASE_DIR / candidate


MODEL_PATH = _resolve_path("MODEL_PATH", "models/fraud_model.joblib")
SCALER_PATH = _resolve_path("SCALER_PATH", "models/scaler.joblib")

# ── API ────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Model behaviour ────────────────────────────────────────────────────────
# 0.4 chosen over default 0.5 — see notebooks/03_evaluation.ipynb
# At this threshold, recall on fraud is ~0.88 while precision stays ~0.87
# Business reasoning: missing a fraud (~$88 avg loss) is costlier
# than a false positive (customer friction, ~$2 review cost)
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", 0.4))

# ── Business cost assumptions (used in monitoring logs) ───────────────────
AVG_FRAUD_AMOUNT_USD = 122.21   # mean Amount where Class=1 in training data
FALSE_POSITIVE_COST_USD = 2.00  # estimated manual review cost per flagged tx