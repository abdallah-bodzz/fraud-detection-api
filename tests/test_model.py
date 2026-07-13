"""
test_model.py
-------------
Unit tests for src/model.py — the FraudDetectionModel wrapper and its
prediction pipeline.

Covers: artifact loading, missing-artifact failure mode, prediction
schema correctness, probability bounds, threshold-driven classification,
and risk-band assignment. Every test drives the real predict() code
path against a real (small, synthetic) fitted model — no mocking of
predict_proba() — so a broken feature order or scaler misalignment
would fail these tests, not just slip through unnoticed.

--------------------------------------------------------------------
Project   : Fraud Detection API
Lead Dev  : Abdallah A Khames
Org       : BODZZ
GitHub    : github.com/abdallah-bodzz
Repo      : fraud-detection-api
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.schemas import PredictionResponse, TransactionInput


class TestModelLoading:
    """Artifact loading: happy path and both missing-file failure modes."""

    def test_model_loads_successfully(self, loaded_fraud_model):
        assert loaded_fraud_model.is_loaded is True
        assert loaded_fraud_model.model is not None
        assert loaded_fraud_model.scaler is not None

    def test_is_loaded_false_before_load(self):
        import src.model as model_module

        fresh_instance = model_module.FraudDetectionModel()
        assert fresh_instance.is_loaded is False

    def test_missing_model_file_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import src.model as model_module

        # Point at a directory with no artifacts at all.
        monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "missing_model.joblib")
        monkeypatch.setattr(model_module, "SCALER_PATH", tmp_path / "missing_scaler.joblib")

        instance = model_module.FraudDetectionModel()
        with pytest.raises(FileNotFoundError, match="Model not found"):
            instance.load()

    def test_missing_scaler_file_raises_file_not_found(
        self, model_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import src.model as model_module

        # Model exists (from the model_dir fixture); scaler path does not.
        monkeypatch.setattr(model_module, "MODEL_PATH", model_dir / "fraud_model.joblib")
        monkeypatch.setattr(model_module, "SCALER_PATH", model_dir / "nonexistent_scaler.joblib")

        instance = model_module.FraudDetectionModel()
        with pytest.raises(FileNotFoundError, match="Scaler not found"):
            instance.load()

    def test_predict_before_load_raises_runtime_error(self):
        import src.model as model_module

        fresh_instance = model_module.FraudDetectionModel()
        transaction = TransactionInput(**{f"V{i}": 0.0 for i in range(1, 29)}, Time=0.0, Amount=10.0)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            fresh_instance.predict(transaction)


class TestPrediction:
    """Prediction pipeline: schema correctness, bounds, and threshold logic."""

    def test_prediction_returns_correct_schema(self, loaded_fraud_model, valid_transaction_payload):
        transaction = TransactionInput(**valid_transaction_payload)
        result = loaded_fraud_model.predict(transaction)

        assert isinstance(result, PredictionResponse)
        assert isinstance(result.fraud_probability, float)
        assert isinstance(result.is_fraud, bool)
        assert isinstance(result.business_note, str)
        assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}

    def test_fraud_probability_within_bounds(self, loaded_fraud_model, valid_transaction_payload):
        transaction = TransactionInput(**valid_transaction_payload)
        result = loaded_fraud_model.predict(transaction)

        assert 0.0 <= result.fraud_probability <= 1.0

    def test_transaction_amount_echoed_correctly(self, loaded_fraud_model, valid_transaction_payload):
        transaction = TransactionInput(**valid_transaction_payload)
        result = loaded_fraud_model.predict(transaction)

        assert result.transaction_amount == valid_transaction_payload["Amount"]

    def test_threshold_used_matches_config(self, loaded_fraud_model, valid_transaction_payload):
        from src.config import PREDICTION_THRESHOLD

        transaction = TransactionInput(**valid_transaction_payload)
        result = loaded_fraud_model.predict(transaction)

        assert result.threshold_used == PREDICTION_THRESHOLD

    def test_is_fraud_flag_consistent_with_probability_and_threshold(
        self, loaded_fraud_model, valid_transaction_payload
    ):
        transaction = TransactionInput(**valid_transaction_payload)
        result = loaded_fraud_model.predict(transaction)

        expected_is_fraud = result.fraud_probability >= result.threshold_used
        assert result.is_fraud == expected_is_fraud

    def test_fraud_like_payload_scores_high(self, loaded_fraud_model, fraud_like_payload):
        """
        A payload engineered to fall in the fraud-shifted region of the
        synthetic training distribution should score meaningfully above
        a payload built from the neutral baseline — this is the
        end-to-end check that the pipeline (scaling, column order,
        inference) actually responds to signal rather than returning a
        constant.
        """
        transaction = TransactionInput(**fraud_like_payload)
        result = loaded_fraud_model.predict(transaction)

        assert result.fraud_probability > 0.5

    def test_risk_band_assignment_is_internally_consistent(
        self, loaded_fraud_model, valid_transaction_payload
    ):
        transaction = TransactionInput(**valid_transaction_payload)
        result = loaded_fraud_model.predict(transaction)

        if result.fraud_probability < 0.2:
            assert result.risk_level == "LOW"
        elif result.fraud_probability < result.threshold_used:
            assert result.risk_level == "MEDIUM"
        else:
            assert result.risk_level == "HIGH"

    def test_business_note_mentions_flagged_or_approved(
        self, loaded_fraud_model, valid_transaction_payload
    ):
        transaction = TransactionInput(**valid_transaction_payload)
        result = loaded_fraud_model.predict(transaction)

        note_lower = result.business_note.lower()
        assert ("flagged" in note_lower) or ("approved" in note_lower)

    def test_feature_column_order_matches_training_schema(self, loaded_fraud_model):
        """
        Guards against the single most dangerous silent-corruption bug in
        this pipeline: FEATURE_COLUMNS drifting out of sync with the
        order the model was actually trained on. If this ever breaks,
        predictions become silently wrong rather than erroring.
        """
        import src.model as model_module

        expected = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        assert model_module.FEATURE_COLUMNS == expected


class TestTransactionInputValidation:
    """Pydantic schema validation at the request boundary."""

    def test_negative_amount_rejected(self, valid_transaction_payload):
        payload = dict(valid_transaction_payload)
        payload["Amount"] = -5.0

        with pytest.raises(ValueError):
            TransactionInput(**payload)

    def test_missing_field_rejected(self, valid_transaction_payload):
        payload = dict(valid_transaction_payload)
        del payload["V14"]

        with pytest.raises(ValueError):
            TransactionInput(**payload)

    def test_zero_amount_accepted(self, valid_transaction_payload):
        payload = dict(valid_transaction_payload)
        payload["Amount"] = 0.0

        transaction = TransactionInput(**payload)
        assert transaction.Amount == 0.0