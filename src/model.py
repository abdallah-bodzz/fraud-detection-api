"""
model.py
--------
All ML logic: loading artifacts and running predictions.
The API layer (main.py) never touches numpy or joblib directly.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import MODEL_PATH, SCALER_PATH, PREDICTION_THRESHOLD, AVG_FRAUD_AMOUNT_USD, FALSE_POSITIVE_COST_USD
from src.schemas import TransactionInput, PredictionResponse
from src.utils import logger


# ── Feature order must match training exactly ──────────────────────────────
FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


class FraudDetectionModel:
    """
    Wraps the trained XGBoost model and scaler.
    Loaded once at startup, reused for every request (no cold load per call).
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self._loaded = False

    def load(self) -> None:
        """Load model and scaler from disk. Called once at API startup."""
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python train_model.py` first to train and save the model."
            )
        if not Path(SCALER_PATH).exists():
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. "
                "Run `python train_model.py` first."
            )

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self._loaded = True
        logger.info(f"Model loaded from {MODEL_PATH}")
        logger.info(f"Scaler loaded from {SCALER_PATH}")
        logger.info(f"Decision threshold: {PREDICTION_THRESHOLD}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, transaction: TransactionInput) -> PredictionResponse:
        """
        Run a single transaction through the pipeline:
        1. Dict → DataFrame (preserves column order)
        2. Scale Time and Amount (V1-V28 are already PCA-transformed)
        3. Get fraud probability from XGBoost
        4. Apply threshold and build business-framed response
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # ── 1. Build feature DataFrame ─────────────────────────────────────
        data = transaction.model_dump()
        df = pd.DataFrame([data])[FEATURE_COLUMNS]

        # ── 2. Scale Time and Amount (same scaler fitted during training) ──
        df[["Time", "Amount"]] = self.scaler.transform(df[["Time", "Amount"]])

        # ── 3. Get fraud probability ───────────────────────────────────────
        fraud_prob = float(self.model.predict_proba(df)[0][1])

        # ── 4. Apply business threshold ────────────────────────────────────
        is_fraud = fraud_prob >= PREDICTION_THRESHOLD

        # ── 5. Risk band ───────────────────────────────────────────────────
        if fraud_prob < 0.2:
            risk_level = "LOW"
        elif fraud_prob < PREDICTION_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # ── 6. Business-framed note ────────────────────────────────────────
        business_note = _build_business_note(
            fraud_prob, is_fraud, transaction.Amount
        )

        logger.info(
            f"Prediction | amount=${transaction.Amount:.2f} | "
            f"prob={fraud_prob:.4f} | fraud={is_fraud} | risk={risk_level}"
        )

        return PredictionResponse(
            fraud_probability=round(fraud_prob, 4),
            is_fraud=is_fraud,
            threshold_used=PREDICTION_THRESHOLD,
            risk_level=risk_level,
            business_note=business_note,
            transaction_amount=transaction.Amount,
        )


def _build_business_note(prob: float, is_fraud: bool, amount: float) -> str:
    """
    Translate model output into a sentence a business stakeholder understands.
    Avoids ML jargon. Frames the decision in cost terms.
    """
    if is_fraud:
        return (
            f"Transaction flagged for review. Model is {prob*100:.1f}% confident "
            f"this is fraudulent. Blocking protects an estimated ${amount:.2f} "
            f"against a review cost of ~${FALSE_POSITIVE_COST_USD:.2f} — "
            f"economically justified above any probability > "
            f"{FALSE_POSITIVE_COST_USD / AVG_FRAUD_AMOUNT_USD:.2%}."
        )
    else:
        return (
            f"Transaction approved. Fraud probability is {prob*100:.1f}% — "
            f"below the {PREDICTION_THRESHOLD*100:.0f}% decision threshold. "
            f"At this confidence level, blocking would cause unnecessary "
            f"customer friction for a legitimate transaction."
        )


# ── Module-level singleton (loaded once at app startup) ────────────────────
fraud_model = FraudDetectionModel()
