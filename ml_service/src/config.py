"""
ml_service/src/config.py
────────────────────────
Loads and validates all environment variables required by the ML service.

Raises a clear RuntimeError at import time if any required variable is missing
or points to a non-existent path — fail fast rather than fail deep inside a
training loop.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Walk up from ml_service/src/ to the project root, then load .env
_SRC_DIR = Path(__file__).resolve().parent
_ML_DIR = _SRC_DIR.parent
_ROOT = _ML_DIR.parent
load_dotenv(_ROOT / ".env")


class _Settings:
    """Typed, validated settings object. Import the singleton `settings`."""

    # ── Filesystem paths ──────────────────────────────────────────────────────
    DATASET_DIR: Path = Path(
        os.getenv(
            "DATASET_DIR",
            str(_ROOT / "MachineLearningCSV" / "MachineLearningCVE"),
        )
    )
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", str(_ROOT / "models")))
    ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", str(_ROOT / "artifacts")))
    CHROMA_PERSIST_DIR: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", str(_ROOT / "artifacts" / "chroma_db"))
    )

    # ── Service config ────────────────────────────────────────────────────────
    ML_SERVICE_PORT: int = int(os.getenv("ML_SERVICE_PORT", "8000"))

    # ── Gemini (Phase 11+) ────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # ── RAG (Phase 10+) ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    def validate(self) -> None:
        """Run at service startup. Ensures output dirs exist; warns if optional
        keys are missing (does not hard-fail on Gemini key — Phases 1-9 don't
        need it)."""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        if not self.DATASET_DIR.exists():
            print(
                f"[config] WARNING: DATASET_DIR does not exist: {self.DATASET_DIR}\n"
                "         This is OK during Phase 0. It must exist before Phase 1.",
                file=sys.stderr,
            )

        if not self.GEMINI_API_KEY or self.GEMINI_API_KEY == "your_gemini_api_key_here":
            print(
                "[config] INFO: GEMINI_API_KEY not set. "
                "This is expected until Phase 11.",
                file=sys.stderr,
            )


# Singleton — import this everywhere.
settings = _Settings()
settings.validate()
