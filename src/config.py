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
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/fraud_model.joblib")
SCALER_PATH = BASE_DIR / os.getenv("SCALER_PATH", "models/scaler.joblib")

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
