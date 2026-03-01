"""
ScriptGuard Inference API

Start:
    uvicorn scriptguard.api.main:app --host 0.0.0.0 --port 8000
    SCRIPTGUARD_MODEL_PATH=models/checkpoint-51676 \
    SCRIPTGUARD_SCALER_PATH=models/feature_scaler.joblib \
    uvicorn scriptguard.api.main:app ...

Endpoints:
    POST /classify          JSON body {"code": "..."}
    POST /classify/file     multipart/form-data file upload
    GET  /health            always 200
    GET  /ready             200 if model loaded, 503 otherwise
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from scriptguard.utils.logger import logger

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_classifier = None


def _load_classifier() -> None:
    global _classifier
    from scriptguard.inference.classifier import ScriptGuardClassifier

    model_path = os.environ.get("SCRIPTGUARD_MODEL_PATH", "models/checkpoint-51676")
    scaler_path = os.environ.get("SCRIPTGUARD_SCALER_PATH", "models/feature_scaler.joblib")

    logger.info(f"Loading ScriptGuard model from {model_path} (scaler: {scaler_path})")
    _classifier = ScriptGuardClassifier(model_path=model_path, scaler_path=scaler_path)
    logger.info("Model ready")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_classifier()
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
        # Don't crash the server — /ready will return 503
    yield


app = FastAPI(
    title="ScriptGuard API",
    description="Malicious Python script detector powered by GraphCodeBERT + feature fusion",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    code: str
    threshold: Optional[float] = None


class ClassifyResponse(BaseModel):
    label: str           # "malicious" | "benign"
    confidence: float    # probability for the predicted label
    malicious_prob: float  # raw P(malicious) regardless of label


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
def ready():
    if _classifier is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "model not loaded"},
        )
    return {"status": "ready"}


def _run_classify(code: str, threshold: Optional[float]) -> ClassifyResponse:
    if _classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    if not code or not code.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="code must not be empty",
        )

    original_threshold = _classifier._decision_threshold
    try:
        if threshold is not None:
            _classifier._decision_threshold = threshold
        label, confidence = _classifier.classify(code)
    finally:
        _classifier._decision_threshold = original_threshold

    # classify() returns confidence for the predicted label —
    # reconstruct raw malicious_prob for callers who want it
    if label == "malicious":
        malicious_prob = confidence
    else:
        malicious_prob = 1.0 - confidence

    return ClassifyResponse(label=label, confidence=confidence, malicious_prob=malicious_prob)


@app.post("/classify", response_model=ClassifyResponse, tags=["classify"])
def classify_json(request: ClassifyRequest):
    """Classify Python source code passed as JSON string."""
    return _run_classify(request.code, request.threshold)


@app.post("/classify/file", response_model=ClassifyResponse, tags=["classify"])
async def classify_file(
    file: UploadFile = File(...),
    threshold: Optional[float] = None,
):
    """Classify a Python script uploaded as a file."""
    raw = await file.read()
    try:
        code = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode file: {e}")
    return _run_classify(code, threshold)
