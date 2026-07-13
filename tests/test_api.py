"""
test_api.py
-----------
Integration tests for src/main.py — the FastAPI application layer.

Covers: liveness endpoint, end-to-end prediction through HTTP, input
validation error responses, rate limiting, and the 503 path when the
model isn't loaded. Uses FastAPI's TestClient against the real app
object with the model swapped for the synthetic fixture model, so
these tests exercise real routing, real Pydantic validation, and the
real middleware stack — not a hand-rolled stand-in.

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
from fastapi.testclient import TestClient


@pytest.fixture()
def api_client(model_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Build a TestClient against the real FastAPI app, with the module-level
    fraud_model singleton pointed at the synthetic test artifacts and
    loaded before any request is made.

    The app's lifespan handler normally calls fraud_model.load() on
    startup; TestClient's context manager triggers that lifespan, so we
    patch the path constants *before* entering it rather than loading
    the model ourselves and skipping lifespan entirely — this keeps the
    test honest about exercising the real startup path.
    """
    import src.model as model_module

    monkeypatch.setattr(model_module, "MODEL_PATH", model_dir / "fraud_model.joblib")
    monkeypatch.setattr(model_module, "SCALER_PATH", model_dir / "scaler.joblib")

    # Reset the singleton's state so each test starts from a clean, unloaded
    # model — otherwise state leaks across tests via the shared singleton.
    model_module.fraud_model._loaded = False
    model_module.fraud_model.model = None
    model_module.fraud_model.scaler = None

    from src.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def reset_rate_limit_store():
    """
    Clear the in-memory rate-limit store before and after every test.
    Without this, request counts accumulate across tests in the same
    session and the rate-limit test would be affected by unrelated
    tests that ran earlier, and vice versa.
    """
    import src.main as main_module

    main_module._rate_limit_store.clear()
    yield
    main_module._rate_limit_store.clear()


class TestHealthEndpoint:
    def test_health_returns_200_when_model_loaded(self, api_client):
        response = api_client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_health_reports_configured_threshold(self, api_client):
        from src.config import PREDICTION_THRESHOLD

        response = api_client.get("/health")

        assert response.json()["threshold"] == PREDICTION_THRESHOLD


class TestDashboardRoute:
    def test_root_serves_dashboard_html(self, api_client):
        response = api_client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Fraud Detection API" in response.text

    def test_dashboard_excluded_from_openapi_schema(self, api_client):
        """
        The dashboard is a UI route, not an API contract — it shouldn't
        pollute the OpenAPI schema that /docs renders and that API
        consumers generate clients from.
        """
        response = api_client.get("/openapi.json")

        assert response.status_code == 200
        assert "/" not in response.json()["paths"]

    def test_docs_still_serves_swagger_ui(self, api_client):
        """
        The dashboard supplements /docs, it does not replace it —
        developers integrating against the API still need the
        schema-driven interactive reference.
        """
        response = api_client.get("/docs")

        assert response.status_code == 200
        assert "swagger" in response.text.lower()


class TestPredictEndpoint:
    def test_predict_returns_200_for_valid_payload(self, api_client, valid_transaction_payload):
        response = api_client.post("/predict_transaction", json=valid_transaction_payload)

        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert isinstance(body["is_fraud"], bool)

    def test_predict_response_matches_request_amount(self, api_client, valid_transaction_payload):
        response = api_client.post("/predict_transaction", json=valid_transaction_payload)

        assert response.json()["transaction_amount"] == valid_transaction_payload["Amount"]

    def test_predict_rejects_missing_field(self, api_client, valid_transaction_payload):
        payload = dict(valid_transaction_payload)
        del payload["V1"]

        response = api_client.post("/predict_transaction", json=payload)

        assert response.status_code == 422

    def test_predict_rejects_negative_amount(self, api_client, valid_transaction_payload):
        payload = dict(valid_transaction_payload)
        payload["Amount"] = -100.0

        response = api_client.post("/predict_transaction", json=payload)

        assert response.status_code == 422

    def test_predict_rejects_non_numeric_field(self, api_client, valid_transaction_payload):
        payload = dict(valid_transaction_payload)
        payload["V1"] = "not_a_number"

        response = api_client.post("/predict_transaction", json=payload)

        assert response.status_code == 422

    def test_predict_fraud_like_payload_flags_high_risk(self, api_client, fraud_like_payload):
        response = api_client.post("/predict_transaction", json=fraud_like_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["fraud_probability"] > 0.5
        assert body["risk_level"] == "HIGH"
        assert body["is_fraud"] is True


class TestModelNotReady:
    def test_predict_returns_503_when_model_not_loaded(
        self, model_dir: Path, monkeypatch: pytest.MonkeyPatch, valid_transaction_payload
    ):
        """
        Simulates a request arriving while the model failed to load or
        hasn't finished loading — main.py should surface this as 503,
        not a raw 500 or an unhandled exception.
        """
        import src.model as model_module

        monkeypatch.setattr(model_module, "MODEL_PATH", model_dir / "fraud_model.joblib")
        monkeypatch.setattr(model_module, "SCALER_PATH", model_dir / "scaler.joblib")

        from src.main import app

        with TestClient(app) as client:
            # Force the singleton back to an unloaded state after the
            # lifespan handler has already loaded it, to simulate a
            # mid-lifecycle failure without needing a second app instance.
            import src.model as loaded_module

            loaded_module.fraud_model._loaded = False

            response = client.post("/predict_transaction", json=valid_transaction_payload)

        assert response.status_code == 503


class TestRateLimiting:
    def test_requests_within_limit_succeed(
        self, api_client, valid_transaction_payload, monkeypatch
    ):
        import src.main as main_module

        monkeypatch.setattr(main_module, "RATE_LIMIT_REQUESTS", 5)

        for _ in range(5):
            response = api_client.post("/predict_transaction", json=valid_transaction_payload)
            assert response.status_code == 200

    def test_requests_beyond_limit_return_429(
        self, api_client, valid_transaction_payload, monkeypatch
    ):
        import src.main as main_module

        monkeypatch.setattr(main_module, "RATE_LIMIT_REQUESTS", 3)

        for _ in range(3):
            response = api_client.post("/predict_transaction", json=valid_transaction_payload)
            assert response.status_code == 200

        blocked_response = api_client.post("/predict_transaction", json=valid_transaction_payload)
        assert blocked_response.status_code == 429

    def test_rate_limit_is_per_client_ip(self, api_client, valid_transaction_payload, monkeypatch):
        """
        TestClient sends every request from the same simulated client, so
        this test asserts the limiter's key structure (per-IP dict) rather
        than simulating two distinct IPs end-to-end — a full multi-IP
        integration test would need a live server, which is out of scope
        for this suite.
        """
        import src.main as main_module

        monkeypatch.setattr(main_module, "RATE_LIMIT_REQUESTS", 2)

        api_client.post("/predict_transaction", json=valid_transaction_payload)
        api_client.post("/predict_transaction", json=valid_transaction_payload)

        assert len(main_module._rate_limit_store) == 1
        (_, timestamps), = main_module._rate_limit_store.items()
        assert len(timestamps) == 2