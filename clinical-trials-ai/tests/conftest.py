"""Central pytest fixtures and hermetic test doubles for clinical-trials-ai test suite."""

import hashlib
import importlib
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import chromadb
import httpx
import numpy as np
import pytest

import config


# ============================================================================
# 1. Configuration & Environment Fixtures
# ============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """Ensure test runs in a clean, predictable environment without leaking real keys."""
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
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    importlib.reload(config)
    yield monkeypatch
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    importlib.reload(config)


# ============================================================================
# 2. ClinicalTrials.gov API Mock Data & Fixtures
# ============================================================================

_DEFAULT = object()


def make_mock_study(
    nct_id: str = "NCT02102165",
    brief_title: str = "Trial of Pembrolizumab in Lung Cancer",
    official_title: str = "Phase 3 Study of Pembrolizumab",
    conditions: Any = _DEFAULT,
    brief_summary: str = "Evaluating efficacy and safety in patients with NSCLC.",
    eligibility_criteria: str = "Inclusion:\n* Age >= 18\n* Histologically confirmed NSCLC\nExclusion:\n* Prior immunotherapy",
    phases: Any = _DEFAULT,
    overall_status: str = "RECRUITING",
    study_type: str = "INTERVENTIONAL",
    locations: Any = _DEFAULT,
) -> Dict[str, Any]:
    """Generate a realistic ClinicalTrials.gov API v2 study dictionary."""
    if conditions is _DEFAULT:
        conditions = ["Non-Small Cell Lung Cancer"]
    if phases is _DEFAULT:
        phases = ["PHASE3"]
    if locations is _DEFAULT:
        locations = [{"city": "Boston", "state": "MA", "country": "United States"}]

    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": brief_title,
                "officialTitle": official_title,
            },
            "statusModule": {
                "overallStatus": overall_status,
            },
            "descriptionModule": {
                "briefSummary": brief_summary,
            },
            "conditionsModule": {
                "conditions": conditions,
            },
            "designModule": {
                "phases": phases,
                "studyType": study_type,
            },
            "eligibilityModule": {
                "eligibilityCriteria": eligibility_criteria,
            },
            "contactsLocationsModule": {
                "locations": locations,
            },
        }
    }


@pytest.fixture
def sample_studies() -> List[Dict[str, Any]]:
    """Return a curated set of 5 diverse clinical studies."""
    return [
        make_mock_study(
            nct_id="NCT00000001",
            brief_title="Immunotherapy Trial for Melanoma",
            conditions=["Melanoma", "Skin Cancer"],
            phases=["PHASE2"],
            overall_status="RECRUITING",
        ),
        make_mock_study(
            nct_id="NCT00000002",
            brief_title="Novel Insulin Regimen for Type 2 Diabetes",
            conditions=["Type 2 Diabetes"],
            phases=["PHASE3"],
            overall_status="RECRUITING",
        ),
        make_mock_study(
            nct_id="NCT00000003",
            brief_title="Beta-Blocker Study in Heart Failure",
            conditions=["Congestive Heart Failure"],
            phases=["PHASE4"],
            overall_status="ACTIVE_NOT_RECRUITING",
        ),
        make_mock_study(
            nct_id="NCT00000004",
            brief_title="Cognitive Training in Alzheimer's Disease",
            conditions=["Alzheimer Disease", "Dementia"],
            phases=None,  # Observational
            overall_status="RECRUITING",
            study_type="OBSERVATIONAL",
        ),
        make_mock_study(
            nct_id="NCT00000005",
            brief_title="Safety of Dual Kinase Inhibitor",
            conditions=["Breast Cancer"],
            phases=["PHASE1", "PHASE2"],  # Multi-phase
            overall_status="RECRUITING",
        ),
    ]


@pytest.fixture
def sample_study_payload(sample_studies) -> Dict[str, Any]:
    """Single realistic study object matching ClinicalTrials.gov REST API v2 schema."""
    return sample_studies[0]


@pytest.fixture
def mock_httpx_get(sample_studies):
    """Mock httpx.Client.get returning ClinicalTrials.gov study payloads."""
    with patch("httpx.Client.get") as mock_get:
        def _get_handler(url, params=None, **kwargs):
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.is_success = True
            mock_resp.json.return_value = {"studies": sample_studies}
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        mock_get.side_effect = _get_handler
        yield mock_get


# ============================================================================
# 3. Deterministic Embedding Fixtures
# ============================================================================

def make_deterministic_vector(text: Optional[str] = "", dim: int = 384) -> List[float]:
    """Generate a deterministic 384-dimensional unit vector from input text."""
    clean_text = text or ""
    seed = int(hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


@pytest.fixture
def mock_embedder(monkeypatch):
    """Patch core.embedder functions with fast deterministic mock vectors."""
    def _embed_text(texts: List[str]) -> List[List[float]]:
        return [make_deterministic_vector(t) for t in texts]

    def _embed_query(query: str) -> List[float]:
        return make_deterministic_vector(query)

    try:
        import core.embedder as emb
        monkeypatch.setattr(emb, "embed_text", _embed_text, raising=False)
        monkeypatch.setattr(emb, "embed_query", _embed_query, raising=False)
    except (ImportError, ModuleNotFoundError):
        pass

    return {"embed_text": _embed_text, "embed_query": _embed_query}


# ============================================================================
# 4. Isolated ChromaDB Vector Store Fixtures
# ============================================================================

@pytest.fixture
def temp_chroma_path(tmp_path, monkeypatch) -> str:
    """Provide an isolated ChromaDB storage directory for each test."""
    db_path = str(tmp_path / "chroma_test_db")
    monkeypatch.setattr(config, "CHROMA_DB_PATH", db_path)
    monkeypatch.setenv("CHROMA_DB_PATH", db_path)
    return db_path


@pytest.fixture
def ephemeral_chroma(monkeypatch):
    """Provide an in-memory ChromaDB collection completely isolated from disk."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="test_clinical_trials",
        metadata={"hnsw:space": "cosine"}
    )
    try:
        import core.vector_store as vs
        monkeypatch.setattr(vs, "get_chroma_client", lambda: client, raising=False)
        monkeypatch.setattr(vs, "get_collection", lambda: collection, raising=False)
    except (ImportError, ModuleNotFoundError):
        pass
    return collection


@pytest.fixture
def isolated_persistent_chroma(tmp_path, monkeypatch):
    """Provide an isolated persistent ChromaDB instance in a temp directory."""
    db_path = str(tmp_path / "test_chroma_db")
    monkeypatch.setattr(config, "CHROMA_DB_PATH", db_path)
    monkeypatch.setenv("CHROMA_DB_PATH", db_path)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="clinical_trials",
        metadata={"hnsw:space": "cosine"}
    )
    try:
        import core.vector_store as vs
        monkeypatch.setattr(vs, "get_chroma_client", lambda: client, raising=False)
        monkeypatch.setattr(vs, "get_collection", lambda: collection, raising=False)
    except (ImportError, ModuleNotFoundError):
        pass
    return {"client": client, "collection": collection, "path": db_path}


# ============================================================================
# 5. Groq LLM Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_groq_llm(monkeypatch):
    """Mock Groq client returning realistic 3-bullet summaries and Q&A answers."""
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "• Who can join: Adults 18+ with confirmed NSCLC.\n"
        "• What it tests: Efficacy and safety of novel Compound X.\n"
        "• Key criteria: No prior immunotherapy treatment."
    )
    mock_completion = MagicMock(choices=[mock_choice])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    try:
        import core.groq_client as gc
        monkeypatch.setattr(gc, "get_groq_client", lambda: mock_client, raising=False)
    except (ImportError, ModuleNotFoundError):
        pass

    return mock_client


@pytest.fixture
def mock_groq_completion(mock_groq_llm):
    """Alias for mock_groq_llm returning patient summary."""
    return mock_groq_llm


@pytest.fixture
def mock_groq_offline(monkeypatch):
    """Simulate missing GROQ_API_KEY environment."""
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    try:
        import core.groq_client as gc
        monkeypatch.setattr(gc, "get_groq_client", lambda: None, raising=False)
    except (ImportError, ModuleNotFoundError):
        pass
    yield


# ============================================================================
# 6. Streamlit Cache Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def teardown_streamlit_cache():
    """Clear Streamlit cache resources after test runs to avoid state leakage."""
    yield
    try:
        import streamlit as st
        st.cache_resource.clear()
        st.cache_data.clear()
    except Exception:
        pass
