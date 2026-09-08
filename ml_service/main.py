"""
ml_service/main.py
──────────────────
FastAPI application entry point for the ML / XAI / RAG service.

Phase 0: stub only — exposes GET /health to confirm the service is reachable.
Later phases will register routers for:
  - /predict        (Phase 4+)
  - /explain        (Phase 6+)
  - /correlate      (Phase 8+)
  - /risk-score     (Phase 9+)
  - /rag/query      (Phase 10+)
  - /recommend      (Phase 11+)

Run locally:
    uvicorn main:app --reload --port 8000
"""

import sys
import time
from pathlib import Path

# Ensure project root directory is on sys.path
_ML_DIR = Path(__file__).resolve().parent
_ROOT = _ML_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from ml_service.src.config import settings

app = FastAPI(
    title="Hybrid Threat Analysis — ML Service",
    description=(
        "Python/FastAPI service providing ML inference, SHAP explainability, "
        "event correlation, risk scoring, RAG retrieval, and Gemini integration."
    ),
    version="0.1.0",
)

# Allow the Node/Express backend and the Next.js dev server to call this service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4000",   # Express backend
        "http://localhost:3000",   # Next.js frontend (dev)
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()


@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe. Returns service status and uptime.

    The Express backend pings this endpoint on startup to verify
    the ML service is reachable before accepting dashboard requests.
    """
    return {
        "status": "ok",
        "service": "ml_service",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "dataset_dir": str(settings.DATASET_DIR),
        "models_dir": str(settings.MODELS_DIR),
    }
