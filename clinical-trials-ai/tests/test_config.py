"""Tests for config module."""

import importlib
import pytest
import config


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch):
    """Clean config-related environment variables before each test."""
    monkeypatch.setattr("dotenv.main.find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr("dotenv.find_dotenv", lambda *args, **kwargs: "")
    env_vars = [
        "GROQ_API_KEY",
        "CHROMA_DB_PATH",
        "COLLECTION_NAME",
        "TOP_K_RESULTS",
        "TOP_K",
        "EMBEDDING_MODEL",
        "EMBEDDING_DEVICE",
        "GROQ_MODEL",
        "CLINICALTRIALS_BASE",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    # Reload config to initial clean state
    importlib.reload(config)
    yield
    # Restore clean state
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    importlib.reload(config)


def test_default_config():
    """Verify default configuration attributes and types when no env vars are set."""
    assert config.GROQ_API_KEY is None
    assert config.CHROMA_DB_PATH == "./data/chroma_db"
    assert config.COLLECTION_NAME == "clinical_trials"
    assert config.TOP_K == 10
    assert isinstance(config.TOP_K, int)
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.EMBEDDING_DEVICE == "cpu"
    assert config.GROQ_MODEL == "qwen/qwen3.8-27b"
    assert config.CLINICALTRIALS_BASE == "https://clinicaltrials.gov/api/v2"


def test_env_overrides(monkeypatch):
    """Verify that all environment variables properly override defaults."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_api_key_12345")
    monkeypatch.setenv("CHROMA_DB_PATH", "/custom/path/chroma_db")
    monkeypatch.setenv("COLLECTION_NAME", "custom_trials_collection")
    monkeypatch.setenv("TOP_K_RESULTS", "25")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    monkeypatch.setenv("EMBEDDING_DEVICE", "mps")
    monkeypatch.setenv("GROQ_MODEL", "mixtral-8x7b-32768")
    monkeypatch.setenv("CLINICALTRIALS_BASE", "https://custom.clinicaltrials.gov/api/v2")

    importlib.reload(config)

    assert config.GROQ_API_KEY == "gsk_test_api_key_12345"
    assert config.CHROMA_DB_PATH == "/custom/path/chroma_db"
    assert config.COLLECTION_NAME == "custom_trials_collection"
    assert config.TOP_K == 25
    assert isinstance(config.TOP_K, int)
    assert config.EMBEDDING_MODEL == "sentence-transformers/all-mpnet-base-v2"
    assert config.EMBEDDING_DEVICE == "mps"
    assert config.GROQ_MODEL == "mixtral-8x7b-32768"
    assert config.CLINICALTRIALS_BASE == "https://custom.clinicaltrials.gov/api/v2"


def test_top_k_fallback_from_top_k_env(monkeypatch):
    """Verify TOP_K env variable is picked up when TOP_K_RESULTS is unset."""
    monkeypatch.delenv("TOP_K_RESULTS", raising=False)
    monkeypatch.setenv("TOP_K", "15")

    importlib.reload(config)

    assert config.TOP_K == 15
    assert isinstance(config.TOP_K, int)


@pytest.mark.parametrize(
    "invalid_val",
    [
        "not_a_number",
        "invalid123",
        "-5",
        "0",
        "",
        "   ",
        "12.34",
    ],
)
def test_top_k_invalid_fallback(monkeypatch, invalid_val):
    """Verify safe fallback to default 10 for invalid, non-positive, or non-integer values."""
    monkeypatch.setenv("TOP_K_RESULTS", invalid_val)

    importlib.reload(config)

    assert config.TOP_K == 10
    assert isinstance(config.TOP_K, int)


def test_missing_groq_api_key(monkeypatch):
    """Verify missing GROQ_API_KEY does not throw fatal errors and evaluates to None."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    importlib.reload(config)

    assert config.GROQ_API_KEY is None
