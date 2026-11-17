"""Centralized configuration and environment loader for clinical-trials-ai."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv(override=False)


def _parse_int(value: Optional[str], default: int) -> int:
    """Safely parse an integer from a string value with fallback to default."""
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
        return parsed if parsed > 0 else default
    except (ValueError, TypeError, AttributeError):
        return default


GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY") or None
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "clinical_trials")
TOP_K: int = _parse_int(os.getenv("TOP_K_RESULTS") or os.getenv("TOP_K"), default=10)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
CLINICALTRIALS_BASE: str = os.getenv(
    "CLINICALTRIALS_BASE", "https://clinicaltrials.gov/api/v2"
)
