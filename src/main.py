"""
main.py
-------
FastAPI application entry point.
Run with: uvicorn src.main:app --reload
"""

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import PREDICTION_THRESHOLD
from src.model import fraud_model
from src.schemas import HealthResponse, PredictionResponse, TransactionInput
from src.utils import logger

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DASHBOARD_PATH = STATIC_DIR / "dashboard.html"

# ── Rate limiting (in-memory, single-process) ──────────────────────────────
# Simple sliding-window counter. Good enough for a demo/CV project.
# In production: replace with Redis + a proper middleware.
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 60))   # requests
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))        # seconds


def _check_rate_limit(client_ip: str) -> None:
    """Raise 429 if client exceeds RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW seconds."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    requests = _rate_limit_store[client_ip]

    # Drop timestamps outside the window
    _rate_limit_store[client_ip] = [t for t in requests if t > window_start]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        detail = (
            f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} "
            f"requests per {RATE_LIMIT_WINDOW}s."
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
        )

    _rate_limit_store[client_ip].append(now)


# ── App lifecycle ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, release at shutdown."""
    logger.info("Starting Fraud Detection API...")
    try:
        fraud_model.load()
        logger.info("Model ready. API is live.")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise  # crash early — don't serve if model is missing
    yield
    logger.info("Shutting down API.")


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Transaction Fraud Detection API",
    description=(
        "Detects fraudulent credit card transactions using an XGBoost model "
        "trained on the ULB Credit Card Fraud dataset.\n\n"
        "**Threshold**: Predictions use 0.4 (not 0.5) because missing a fraud "
        "costs ~$122 on average vs ~$2 for a false positive review. "
        "The threshold was tuned to maximize business value, not raw accuracy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware: request timing log ─────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → "
        f"{response.status_code} ({duration_ms:.1f}ms)"
    )
    return response


# ── Routes ─────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def dashboard():
    """
    Serves the branded dashboard as the landing page. The interactive API
    reference (Swagger UI) remains available at /docs — this route exists
    alongside it, not instead of it, since developers integrating against
    the API still need the schema-driven reference.
    """
    return FileResponse(DASHBOARD_PATH)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness check. Returns whether the model is loaded and ready."""
    return HealthResponse(
        status="ok" if fraud_model.is_loaded else "model_not_loaded",
        model_loaded=fraud_model.is_loaded,
        threshold=PREDICTION_THRESHOLD,
    )


@app.post(
    "/predict_transaction",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict if a transaction is fraudulent",
)
async def predict_transaction(transaction: TransactionInput, request: Request):
    """
    Accepts a single transaction's features and returns:
    - fraud probability (0–1)
    - binary classification (is_fraud)
    - risk level band (LOW / MEDIUM / HIGH)
    - business-framed interpretation

    All 30 features (Time, V1–V28, Amount) are required.
    V1–V28 must already be PCA-transformed (as in the raw dataset).
    """
    client_ip = request.client.host
    _check_rate_limit(client_ip)

    try:
        result = fraud_model.predict(transaction)
    except RuntimeError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Try again shortly.",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Check logs.",
        ) from e

    return result


# ── Run directly (for dev) ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    from src.config import API_HOST, API_PORT

    uvicorn.run("src.main:app", host=API_HOST, port=API_PORT, reload=True)