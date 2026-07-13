"""
conftest.py
-----------
Shared pytest fixtures for the fraud-detection API test suite.

Design choice: fixtures train a small, *real* XGBoost model on synthetic
data rather than mocking predict_proba(). A mocked model can't catch
schema mismatches, scaler-column misalignment, or a broken feature
order — the exact class of bug most likely to slip through review. The
synthetic model trains in well under a second, so this costs nothing in
suite runtime.

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
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SYNTHETIC_SEED = 13
SYNTHETIC_ROWS = 400


def _make_synthetic_frame() -> pd.DataFrame:
    """
    Build a small, deterministic synthetic dataset shaped like the real
    creditcard.csv schema. Fraud rows (Class=1) are drawn from a shifted
    distribution on a couple of PCA columns so the model has an actual
    signal to learn — a model trained on pure noise would produce
    unstable probabilities that make threshold-based assertions flaky.
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    n_fraud = 40
    n_legit = SYNTHETIC_ROWS - n_fraud

    legit = {col: rng.normal(0, 1, n_legit) for col in FEATURE_COLUMNS[1:-1]}
    fraud = {col: rng.normal(0, 1, n_fraud) for col in FEATURE_COLUMNS[1:-1]}
    # Inject separable signal on V14 / V17, mirroring the real dataset's
    # most-separating features per 01_eda.ipynb.
    fraud["V14"] = rng.normal(-4, 1, n_fraud)
    fraud["V17"] = rng.normal(-4, 1, n_fraud)

    legit_df = pd.DataFrame(legit)
    legit_df["Time"] = rng.uniform(0, 172_000, n_legit)
    legit_df["Amount"] = np.abs(rng.normal(60, 40, n_legit))
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud)
    fraud_df["Time"] = rng.uniform(0, 172_000, n_fraud)
    fraud_df["Amount"] = np.abs(rng.normal(120, 90, n_fraud))
    fraud_df["Class"] = 1

    combined = pd.concat([legit_df, fraud_df], ignore_index=True)
    return combined.sample(frac=1, random_state=SYNTHETIC_SEED).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Session-scoped: the same synthetic dataset is reused across tests."""
    return _make_synthetic_frame()


@pytest.fixture(scope="session")
def trained_artifacts(synthetic_df: pd.DataFrame) -> dict[str, Any]:
    """
    Fit a scaler and a small XGBoost model on the synthetic dataset,
    mirroring train_model.py's preprocessing exactly (scale Time and
    Amount only; leave V1-V28 untouched).
    """
    X = synthetic_df[FEATURE_COLUMNS].copy()
    y = synthetic_df["Class"].to_numpy()

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="aucpr",
        random_state=SYNTHETIC_SEED,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X, y)

    return {"model": model, "scaler": scaler, "X": X, "y": y}


@pytest.fixture()
def model_dir(tmp_path: Path, trained_artifacts: dict[str, Any]) -> Path:
    """
    Write the trained model + scaler to a temp directory as .joblib files,
    mirroring the real models/ layout. Function-scoped so each test gets
    a clean directory even though the underlying artifacts are shared.
    """
    joblib.dump(trained_artifacts["model"], tmp_path / "fraud_model.joblib")
    joblib.dump(trained_artifacts["scaler"], tmp_path / "scaler.joblib")
    return tmp_path


@pytest.fixture()
def loaded_fraud_model(model_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Return a FraudDetectionModel instance loaded against the temp
    artifacts above.

    src.model imports MODEL_PATH / SCALER_PATH by value at module import
    time, so patching src.config after the fact has no effect — the
    names bound inside src.model must be patched directly.
    """
    import src.model as model_module

    monkeypatch.setattr(model_module, "MODEL_PATH", model_dir / "fraud_model.joblib")
    monkeypatch.setattr(model_module, "SCALER_PATH", model_dir / "scaler.joblib")

    instance = model_module.FraudDetectionModel()
    instance.load()
    return instance


@pytest.fixture()
def valid_transaction_payload() -> dict[str, float]:
    """
    A well-formed request body matching TransactionInput's schema exactly.
    Values are plausible PCA-scale floats plus a realistic Time/Amount.
    """
    payload = {f"V{i}": 0.05 * i for i in range(1, 29)}
    payload["Time"] = 12_345.0
    payload["Amount"] = 149.62
    return payload


@pytest.fixture()
def fraud_like_payload() -> dict[str, float]:
    """
    A payload engineered to score high on the synthetic model — V14 and
    V17 pushed into the fraud-shifted range the fixture model was
    trained on. Used to exercise the is_fraud=True branch deterministically.
    """
    payload = {f"V{i}": 0.05 * i for i in range(1, 29)}
    payload["V14"] = -4.5
    payload["V17"] = -4.5
    payload["Time"] = 40_000.0
    payload["Amount"] = 480.0
    return payload