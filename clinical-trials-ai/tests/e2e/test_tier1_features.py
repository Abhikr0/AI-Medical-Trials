"""Tier 1 Feature Coverage Tests (F1 through F15) for clinical-trials-ai.

Each feature in the inventory is tested with at least 5 distinct test cases
validating primary behaviors, contracts, and core operational modes.
Hermetic, offline execution is enforced using pytest fixtures and test doubles.
"""

import importlib
import os
import py_compile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
from tests.conftest import make_deterministic_vector, make_mock_study


# ============================================================================
# Helper Import Functions for Progressive Milestone Execution
# ============================================================================

def _get_trial_fetcher():
    try:
        import core.trial_fetcher as tf
        return tf
    except (ImportError, ModuleNotFoundError):
        pytest.skip("core.trial_fetcher not yet implemented on disk (M2)")


def _get_embedder():
    try:
        import core.embedder as emb
        return emb
    except (ImportError, ModuleNotFoundError):
        pytest.skip("core.embedder not yet implemented on disk (M3)")


def _get_vector_store():
    try:
        import core.vector_store as vs
        return vs
    except (ImportError, ModuleNotFoundError):
        pytest.skip("core.vector_store not yet implemented on disk (M3)")


def _get_groq_client():
    try:
        import core.groq_client as gc
        return gc
    except (ImportError, ModuleNotFoundError):
        pytest.skip("core.groq_client not yet implemented on disk (M4)")


def _get_sidebar():
    try:
        import ui.sidebar as sb
        return sb
    except (ImportError, ModuleNotFoundError):
        pytest.skip("ui.sidebar not yet implemented on disk (M5)")


def _get_components():
    try:
        import ui.components as comp
        return comp
    except (ImportError, ModuleNotFoundError):
        pytest.skip("ui.components not yet implemented on disk (M5)")


def _get_chat():
    try:
        import ui.chat as ch
        return ch
    except (ImportError, ModuleNotFoundError):
        pytest.skip("ui.chat not yet implemented on disk (M5)")


# ============================================================================
# Feature 1: Configuration & Environment (config.py)
# ============================================================================

def test_tier1_f01_config_default_attributes_exist(clean_env):
    """F1: Verify all required configuration attributes and defaults exist."""
    for attr in [
        "GROQ_API_KEY",
        "CHROMA_DB_PATH",
        "COLLECTION_NAME",
        "TOP_K",
        "EMBEDDING_MODEL",
        "EMBEDDING_DEVICE",
        "GROQ_MODEL",
        "CLINICALTRIALS_BASE",
    ]:
        assert hasattr(config, attr), f"Missing config attribute {attr}"
    assert config.TOP_K == 10
    assert config.CHROMA_DB_PATH == "./data/chroma_db"
    assert config.COLLECTION_NAME == "clinical_trials"
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.GROQ_MODEL == "qwen/qwen3.8-27b"


def test_tier1_f01_config_env_overrides_respected(clean_env, monkeypatch):
    """F1: Verify environment variables cleanly override default config values."""
    monkeypatch.setenv("CHROMA_DB_PATH", "/tmp/custom_chroma")
    monkeypatch.setenv("TOP_K_RESULTS", "15")
    monkeypatch.setenv("COLLECTION_NAME", "custom_trials")
    importlib.reload(config)
    assert config.CHROMA_DB_PATH == "/tmp/custom_chroma"
    assert config.TOP_K == 15
    assert config.COLLECTION_NAME == "custom_trials"


def test_tier1_f01_config_top_k_coerced_to_int(clean_env, monkeypatch):
    """F1: Verify TOP_K is coerced to a valid integer."""
    monkeypatch.setenv("TOP_K_RESULTS", "20")
    importlib.reload(config)
    assert isinstance(config.TOP_K, int)
    assert config.TOP_K == 20


def test_tier1_f01_dotenv_file_loading(clean_env, tmp_path):
    """F1: Verify .env file loading via python-dotenv."""
    from dotenv import load_dotenv

    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("COLLECTION_NAME=test_collection\n")
    load_dotenv(dotenv_path=str(dotenv_file), override=True)
    importlib.reload(config)
    assert config.COLLECTION_NAME == "test_collection"


def test_tier1_f01_env_example_matches_keys():
    """F1: Verify .env.example contains all essential configuration keys."""
    env_example_path = Path(".env.example")
    assert env_example_path.exists(), ".env.example file does not exist"
    content = env_example_path.read_text()
    for key in [
        "GROQ_API_KEY",
        "CHROMA_DB_PATH",
        "COLLECTION_NAME",
        "TOP_K_RESULTS",
        "EMBEDDING_MODEL",
        "GROQ_MODEL",
    ]:
        assert key in content, f"Key {key} not found in .env.example"


# ============================================================================
# Feature 2: Python Environment & Scaffolding
# ============================================================================

def test_tier1_f02_core_package_importable():
    """F2: Verify core package can be imported without errors."""
    import core
    assert hasattr(core, "__file__") or hasattr(core, "__path__")


def test_tier1_f02_ui_package_importable():
    """F2: Verify ui package can be imported without errors."""
    import ui
    assert hasattr(ui, "__file__") or hasattr(ui, "__path__")


def test_tier1_f02_requirements_contains_pinned_deps():
    """F2: Verify requirements.txt includes all required core dependencies."""
    req_file = Path("requirements.txt")
    assert req_file.exists(), "requirements.txt does not exist"
    content = req_file.read_text().lower()
    for dep in [
        "streamlit",
        "sentence-transformers",
        "chromadb",
        "groq",
        "httpx",
        "python-dotenv",
        "torch",
    ]:
        assert dep in content, f"Dependency {dep} missing from requirements.txt"


def test_tier1_f02_data_directory_structure_exists():
    """F2: Verify data directory exists on filesystem."""
    data_dir = Path("data")
    assert data_dir.exists(), "data directory does not exist"
    assert data_dir.is_dir(), "data path is not a directory"


def test_tier1_f02_app_entrypoint_valid_python():
    """F2: Verify app.py compiles cleanly without syntax errors if present."""
    app_path = Path("app.py")
    if not app_path.exists():
        pytest.skip("app.py not yet created by M5")
    py_compile.compile(str(app_path), doraise=True)


# ============================================================================
# Feature 3: ClinicalTrials API Client (core/trial_fetcher.py)
# ============================================================================

def test_tier1_f03_fetch_trials_query_params_constructed():
    """F3: Verify API query parameters are constructed from conditions list."""
    tf = _get_trial_fetcher()
    captured_requests = []

    def mock_get(url, params=None, **kwargs):
        captured_requests.append({"url": url, "params": params or {}})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"studies": []}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("httpx.Client.get", side_effect=mock_get):
        tf.fetch_trials(["asthma", "cough"], page_size=50)

    assert len(captured_requests) > 0, "No HTTP request was made"
    params = captured_requests[0]["params"]
    cond_val = str(params.get("query.cond") or params.get("query") or "")
    assert "asthma" in cond_val.lower()
    assert "cough" in cond_val.lower()
    assert int(params.get("pageSize", 0)) == 50


def test_tier1_f03_fetch_trials_recruiting_status_filter():
    """F3: Verify default filter.overallStatus parameter is RECRUITING."""
    tf = _get_trial_fetcher()
    captured_requests = []

    def mock_get(url, params=None, **kwargs):
        captured_requests.append({"url": url, "params": params or {}})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"studies": []}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("httpx.Client.get", side_effect=mock_get):
        tf.fetch_trials(["diabetes"])

    assert len(captured_requests) > 0
    params = captured_requests[0]["params"]
    status_param = params.get("filter.overallStatus") or params.get("status")
    assert status_param == "RECRUITING"


def test_tier1_f03_fetch_trials_returns_study_list(sample_studies):
    """F3: Verify fetch_trials returns study list extracted from response."""
    tf = _get_trial_fetcher()
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"studies": sample_studies[:2]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        results = tf.fetch_trials(["cancer"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert "protocolSection" in results[0]


def test_tier1_f03_fetch_trials_field_projections():
    """F3: Verify request requests projection of required trial fields."""
    tf = _get_trial_fetcher()
    captured_requests = []

    def mock_get(url, params=None, **kwargs):
        captured_requests.append({"url": url, "params": params or {}})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"studies": []}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("httpx.Client.get", side_effect=mock_get):
        tf.fetch_trials(["cancer"])

    assert len(captured_requests) > 0
    params = captured_requests[0]["params"]
    fields_str = str(params.get("fields", ""))
    for required_field in ["NCTId", "BriefTitle", "BriefSummary", "OverallStatus"]:
        assert required_field.lower() in fields_str.lower()


def test_tier1_f03_fetch_trials_custom_page_size():
    """F3: Verify custom page size parameter is passed cleanly."""
    tf = _get_trial_fetcher()
    captured_requests = []

    def mock_get(url, params=None, **kwargs):
        captured_requests.append({"url": url, "params": params or {}})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"studies": []}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("httpx.Client.get", side_effect=mock_get):
        tf.fetch_trials(["pain"], page_size=25)

    assert len(captured_requests) > 0
    params = captured_requests[0]["params"]
    assert int(params.get("pageSize", 0)) == 25


# ============================================================================
# Feature 4: Study Extraction & Text Flattening (core/trial_fetcher.py)
# ============================================================================

def test_tier1_f04_extract_trial_returns_text_and_meta(sample_study_payload):
    """F4: Verify extract_trial returns embeddable text string and sanitized metadata."""
    tf = _get_trial_fetcher()
    text, metadata = tf.extract_trial(sample_study_payload)
    assert isinstance(text, str)
    assert len(text) > 0
    assert isinstance(metadata, dict)
    assert metadata["nct_id"] == "NCT00000001"
    assert metadata["status"] == "RECRUITING"
    assert "phase" in metadata
    assert "title" in metadata


def test_tier1_f04_parse_phase_single_phase():
    """F4: Verify parse_phase normalizes single phase list to readable string."""
    tf = _get_trial_fetcher()
    parsed = tf.parse_phase(["PHASE1"])
    assert parsed in ["Phase 1", "Phase I", "PHASE1"]


def test_tier1_f04_parse_phase_multi_phase():
    """F4: Verify parse_phase joins multiple phases into readable format."""
    tf = _get_trial_fetcher()
    parsed = tf.parse_phase(["PHASE2", "PHASE3"])
    assert "2" in parsed and "3" in parsed


def test_tier1_f04_extract_trial_eligibility_truncation():
    """F4: Verify eligibility criteria is truncated safely without unbounded growth."""
    tf = _get_trial_fetcher()
    huge_study = make_mock_study(
        nct_id="NCT99999999",
        eligibility_criteria="Inclusion: " + "A" * 4000
    )
    text, metadata = tf.extract_trial(huge_study)
    assert len(text) < 3000


def test_tier1_f04_extract_trial_nct_id_fallback():
    """F4: Verify fallback NCT ID is provided when identificationModule is absent."""
    tf = _get_trial_fetcher()
    study_no_nct = {
        "protocolSection": {
            "identificationModule": {},
            "statusModule": {"overallStatus": "RECRUITING"},
            "descriptionModule": {"briefSummary": "Summary"},
        }
    }
    text, metadata = tf.extract_trial(study_no_nct)
    assert "nct_id" in metadata
    assert len(str(metadata["nct_id"])) > 0


# ============================================================================
# Feature 5: Pre-loading Broad Categories (core/trial_fetcher.py)
# ============================================================================

def test_tier1_f05_preload_standard_categories_invoked():
    """F5: Verify preload queries all 4 standard broad categories."""
    tf = _get_trial_fetcher()
    called_conditions = []

    def mock_fetch(conditions, page_size=200, status="RECRUITING"):
        called_conditions.extend(conditions)
        return []

    with patch.object(tf, "fetch_trials", side_effect=mock_fetch):
        tf.preload_standard_trials()

    all_conds_str = " ".join(called_conditions).lower()
    for cat in ["cancer", "diabetes", "heart disease", "neurological"]:
        assert cat in all_conds_str, f"Category '{cat}' was not requested in preload"


def test_tier1_f05_preload_deduplication(sample_studies):
    """F5: Verify duplicate study IDs across categories are deduplicated."""
    tf = _get_trial_fetcher()
    dup_study = sample_studies[0]  # NCT00000001

    def mock_fetch(conditions, page_size=200, status="RECRUITING"):
        return [dup_study]

    with patch.object(tf, "fetch_trials", side_effect=mock_fetch):
        results = tf.preload_standard_trials()

    ncts = [s["protocolSection"]["identificationModule"]["nctId"] for s in results]
    assert ncts.count("NCT00000001") == 1


def test_tier1_f05_preload_custom_categories():
    """F5: Verify custom category list can be passed and queried."""
    tf = _get_trial_fetcher()
    called_conditions = []

    def mock_fetch(conditions, page_size=200, status="RECRUITING"):
        called_conditions.extend(conditions)
        return []

    with patch.object(tf, "fetch_trials", side_effect=mock_fetch):
        tf.preload_standard_trials(categories=["pediatric", "rare"])

    assert called_conditions == ["pediatric", "rare"] or set(called_conditions) == {"pediatric", "rare"}


def test_tier1_f05_preload_aggregates_total_count():
    """F5: Verify preload aggregates distinct studies from all categories."""
    tf = _get_trial_fetcher()
    studies_batch_gen = (
        [make_mock_study(nct_id=f"NCT{cat}_{i:04d}") for i in range(50)]
        for cat in range(4)
    )

    with patch.object(tf, "fetch_trials", side_effect=lambda *args, **kwargs: next(studies_batch_gen)):
        results = tf.preload_standard_trials()

    assert len(results) == 200


def test_tier1_f05_preload_returns_list_of_dicts(sample_studies):
    """F5: Verify output of preload_standard_trials is a list of study dicts."""
    tf = _get_trial_fetcher()
    with patch.object(tf, "fetch_trials", return_value=sample_studies):
        results = tf.preload_standard_trials()
    assert isinstance(results, list)
    assert all(isinstance(s, dict) for s in results)


# ============================================================================
# Feature 6: Sentence Transformer Embeddings (core/embedder.py)
# ============================================================================

def test_tier1_f06_load_model_returns_sentence_transformer():
    """F6: Verify load_model returns an embedder instance with encode capability."""
    emb = _get_embedder()
    model = emb.load_model()
    assert hasattr(model, "encode"), "Loaded model does not have encode method"


def test_tier1_f06_embed_text_single_string_batch():
    """F6: Verify embed_text returns 384-dimensional vector for single text."""
    emb = _get_embedder()
    res = emb.embed_text(["Clinical trial for lung cancer treatment."])
    assert isinstance(res, list)
    assert len(res) == 1
    assert len(res[0]) == 384


def test_tier1_f06_embed_text_multi_string_batch():
    """F6: Verify embed_text returns batch of 384-dimensional vectors."""
    emb = _get_embedder()
    res = emb.embed_text(["Trial A", "Trial B", "Trial C"])
    assert isinstance(res, list)
    assert len(res) == 3
    assert all(len(v) == 384 for v in res)


def test_tier1_f06_embed_query_returns_vector():
    """F6: Verify embed_query returns 1D list of 384 floats."""
    emb = _get_embedder()
    res = emb.embed_query("fatigue and joint pain")
    assert isinstance(res, list)
    assert len(res) == 384


def test_tier1_f06_embed_text_cpu_execution():
    """F6: Verify embedding model executes on CPU device."""
    emb = _get_embedder()
    model = emb.load_model()
    device_str = str(getattr(model, "device", "cpu"))
    assert "cpu" in device_str.lower()


# ============================================================================
# Feature 7: ChromaDB Vector Store Persistence (core/vector_store.py)
# ============================================================================

def test_tier1_f07_get_chroma_client_persistent(temp_chroma_path):
    """F7: Verify get_chroma_client returns client connected to storage path."""
    vs = _get_vector_store()
    client = vs.get_chroma_client()
    assert client is not None
    assert hasattr(client, "get_or_create_collection") or hasattr(client, "create_collection")


def test_tier1_f07_get_collection_cosine_space(temp_chroma_path):
    """F7: Verify get_collection returns clinical_trials collection."""
    vs = _get_vector_store()
    coll = vs.get_collection()
    assert "clinical_trials" in coll.name
    meta = coll.metadata or {}
    if "hnsw:space" in meta:
        assert meta["hnsw:space"] == "cosine"


def test_tier1_f07_upsert_trials_populates_collection(temp_chroma_path):
    """F7: Verify upsert_trials adds new trials and increases collection count."""
    vs = _get_vector_store()
    initial_count = vs.get_trial_count()
    vs.upsert_trials(
        ids=["NCT1", "NCT2"],
        embeddings=[[0.05] * 384, [0.1] * 384],
        documents=["Trial Doc 1", "Trial Doc 2"],
        metadatas=[
            {"phase": "Phase 1", "status": "RECRUITING"},
            {"phase": "Phase 2", "status": "RECRUITING"},
        ],
    )
    assert vs.get_trial_count() == initial_count + 2


def test_tier1_f07_upsert_trials_idempotency(temp_chroma_path):
    """F7: Verify re-upserting same IDs updates records without inflating count."""
    vs = _get_vector_store()
    vs.upsert_trials(
        ids=["NCT1", "NCT2"],
        embeddings=[[0.05] * 384, [0.1] * 384],
        documents=["Trial Doc 1", "Trial Doc 2"],
        metadatas=[
            {"phase": "Phase 1", "status": "RECRUITING"},
            {"phase": "Phase 2", "status": "RECRUITING"},
        ],
    )
    count_after_first = vs.get_trial_count()
    vs.upsert_trials(
        ids=["NCT1", "NCT2"],
        embeddings=[[0.05] * 384, [0.1] * 384],
        documents=["Trial Doc 1 Updated", "Trial Doc 2 Updated"],
        metadatas=[
            {"phase": "Phase 1", "status": "COMPLETED"},
            {"phase": "Phase 2", "status": "COMPLETED"},
        ],
    )
    assert vs.get_trial_count() == count_after_first


def test_tier1_f07_get_trial_count_matches_actual(temp_chroma_path):
    """F7: Verify get_trial_count matches number of upserted trials."""
    vs = _get_vector_store()
    vs.upsert_trials(
        ids=[f"NCT_COUNT_{i}" for i in range(5)],
        embeddings=[[float(i) / 10.0] * 384 for i in range(5)],
        documents=[f"Doc {i}" for i in range(5)],
        metadatas=[{"phase": "Phase 3", "status": "RECRUITING"} for i in range(5)],
    )
    assert vs.get_trial_count() >= 5


# ============================================================================
# Feature 8: Semantic Similarity Search (core/vector_store.py)
# ============================================================================

def test_tier1_f08_search_trials_returns_top_k(temp_chroma_path):
    """F8: Verify search_trials returns requested number of top results."""
    vs = _get_vector_store()
    vs.upsert_trials(
        ids=[f"NCT_SEARCH_{i}" for i in range(10)],
        embeddings=[make_deterministic_vector(f"Doc {i}") for i in range(10)],
        documents=[f"Study document text {i}" for i in range(10)],
        metadatas=[{"phase": "Phase 3", "status": "RECRUITING"} for i in range(10)],
    )
    query_vec = make_deterministic_vector("Search query")
    results = vs.search_trials(query_vec, n_results=5)
    assert "documents" in results
    assert "metadatas" in results
    assert "distances" in results
    assert len(results["documents"][0]) == 5
    assert len(results["metadatas"][0]) == 5
    assert len(results["distances"][0]) == 5


def test_tier1_f08_search_trials_cosine_distance_ranking(temp_chroma_path):
    """F8: Verify returned matches are ordered by ascending cosine distance."""
    vs = _get_vector_store()
    vs.upsert_trials(
        ids=[f"NCT_RANK_{i}" for i in range(6)],
        embeddings=[make_deterministic_vector(f"Rank doc {i}") for i in range(6)],
        documents=[f"Rank doc {i}" for i in range(6)],
        metadatas=[{"phase": "Phase 1", "status": "RECRUITING"} for i in range(6)],
    )
    query_vec = make_deterministic_vector("Rank doc 0")
    results = vs.search_trials(query_vec, n_results=6)
    distances = results["distances"][0]
    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i + 1] + 1e-5


def test_tier1_f08_search_trials_with_phase_filter(temp_chroma_path):
    """F8: Verify phase filter filters out trials from other phases."""
    vs = _get_vector_store()
    vs.upsert_trials(
        ids=["NCT_P1", "NCT_P3"],
        embeddings=[make_deterministic_vector("P1"), make_deterministic_vector("P3")],
        documents=["Doc P1", "Doc P3"],
        metadatas=[
            {"phase": "Phase 1", "status": "RECRUITING"},
            {"phase": "Phase 3", "status": "RECRUITING"},
        ],
    )
    query_vec = make_deterministic_vector("P3")
    results = vs.search_trials(query_vec, n_results=10, phase="Phase 3")
    for meta in results["metadatas"][0]:
        assert meta["phase"] == "Phase 3"


def test_tier1_f08_search_trials_with_status_filter(temp_chroma_path):
    """F8: Verify status filter filters out trials with other statuses."""
    vs = _get_vector_store()
    vs.upsert_trials(
        ids=["NCT_S1", "NCT_S2"],
        embeddings=[make_deterministic_vector("S1"), make_deterministic_vector("S2")],
        documents=["Doc S1", "Doc S2"],
        metadatas=[
            {"phase": "Phase 2", "status": "RECRUITING"},
            {"phase": "Phase 2", "status": "COMPLETED"},
        ],
    )
    query_vec = make_deterministic_vector("S1")
    results = vs.search_trials(query_vec, n_results=10, status="RECRUITING")
    for meta in results["metadatas"][0]:
        assert meta["status"] == "RECRUITING"


def test_tier1_f08_search_trials_score_calculation():
    """F8: Verify cosine distance to match percentage calculation."""
    try:
        from ui.components import calculate_match_percentage
    except (ImportError, ModuleNotFoundError):
        pytest.skip("ui.components.calculate_match_percentage not yet implemented")

    computed_score = calculate_match_percentage(0.15)
    assert computed_score == 85.0


# ============================================================================
# Feature 9: Groq Plain-English Summarization (core/groq_client.py)
# ============================================================================

def test_tier1_f09_get_groq_client_configured(monkeypatch):
    """F9: Verify get_groq_client returns client when GROQ_API_KEY is configured."""
    gc = _get_groq_client()
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test123")
    client = gc.get_groq_client()
    assert client is not None


def test_tier1_f09_summarize_trial_calls_chat_completions(mock_groq_llm):
    """F9: Verify summarize_trial calls chat completions with qwen/qwen3.8-27b."""
    gc = _get_groq_client()
    gc.summarize_trial("Trial detail text")
    assert mock_groq_llm.chat.completions.create.called
    kwargs = mock_groq_llm.chat.completions.create.call_args[1]
    assert kwargs.get("model") == "qwen/qwen3.8-27b"


def test_tier1_f09_summarize_trial_returns_bullet_points(mock_groq_llm):
    """F9: Verify summary returns structured bullet points."""
    gc = _get_groq_client()
    summary = gc.summarize_trial("Trial detail text")
    assert any(bullet in summary for bullet in ["•", "-", "*"])


def test_tier1_f09_summarize_trial_prompt_structure(mock_groq_llm):
    """F9: Verify prompt instructs model to summarize in patient-friendly bullets."""
    gc = _get_groq_client()
    gc.summarize_trial("Trial detail text")
    call_kwargs = mock_groq_llm.chat.completions.create.call_args[1]
    messages = call_kwargs.get("messages", [])
    prompt_text = " ".join(m.get("content", "") for m in messages).lower()
    assert "bullet" in prompt_text or "summar" in prompt_text


def test_tier1_f09_summarize_trial_input_truncation(mock_groq_llm):
    """F9: Verify large input document is truncated to safe context limit."""
    gc = _get_groq_client()
    huge_text = "Trial detail " * 1000  # >10,000 chars
    gc.summarize_trial(huge_text)
    call_kwargs = mock_groq_llm.chat.completions.create.call_args[1]
    messages = call_kwargs.get("messages", [])
    prompt_text = " ".join(m.get("content", "") for m in messages)
    assert len(prompt_text) < 5000


# ============================================================================
# Feature 10: Grounded Interactive Q&A (core/groq_client.py)
# ============================================================================

def test_tier1_f10_ask_about_trials_includes_system_prompt(mock_groq_llm):
    """F10: Verify ask_about_trials injects grounding system prompt."""
    gc = _get_groq_client()
    gc.ask_about_trials("Is this trial for adults?", ["Context text"], [])
    assert mock_groq_llm.chat.completions.create.called
    messages = mock_groq_llm.chat.completions.create.call_args[1]["messages"]
    assert messages[0]["role"] == "system"


def test_tier1_f10_ask_about_trials_concatenates_contexts(mock_groq_llm):
    """F10: Verify trial contexts are incorporated into system prompt."""
    gc = _get_groq_client()
    contexts = ["Context 1 for trial A", "Context 2 for trial B", "Context 3 for trial C"]
    gc.ask_about_trials("Question?", contexts, [])
    messages = mock_groq_llm.chat.completions.create.call_args[1]["messages"]
    sys_content = messages[0]["content"]
    for c in contexts:
        assert c in sys_content


def test_tier1_f10_ask_about_trials_preserves_chat_history(mock_groq_llm):
    """F10: Verify conversational chat history is preserved across turns."""
    gc = _get_groq_client()
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    gc.ask_about_trials("Q2", ["Context text"], history)
    messages = mock_groq_llm.chat.completions.create.call_args[1]["messages"]
    roles_and_contents = [(m["role"], m["content"]) for m in messages]
    assert ("user", "Q1") in roles_and_contents
    assert ("assistant", "A1") in roles_and_contents
    assert ("user", "Q2") in roles_and_contents


def test_tier1_f10_ask_about_trials_returns_answer(mock_groq_llm):
    """F10: Verify ask_about_trials returns string response."""
    gc = _get_groq_client()
    ans = gc.ask_about_trials("Question?", ["Context"], [])
    assert isinstance(ans, str)
    assert len(ans) > 0


def test_tier1_f10_ask_about_trials_parameters(mock_groq_llm):
    """F10: Verify Groq API model and token parameters for Q&A."""
    gc = _get_groq_client()
    gc.ask_about_trials("Question?", ["Context"], [])
    kwargs = mock_groq_llm.chat.completions.create.call_args[1]
    assert kwargs.get("model") == "qwen/qwen3.8-27b"
    assert kwargs.get("max_tokens") in [300, 500]


# ============================================================================
# Feature 11: Multi-Symptom Input UI (ui/sidebar.py, app.py)
# ============================================================================

def test_tier1_f11_symptom_options_predefined():
    """F11: Verify predefined symptom list has at least 10 clinical symptoms."""
    sb = _get_sidebar()
    symptoms = getattr(sb, "DEFAULT_SYMPTOMS", None) or getattr(sb, "COMMON_SYMPTOMS", None)
    if symptoms is None:
        try:
            from ui.components import DEFAULT_SYMPTOMS as symptoms
        except (ImportError, AttributeError):
            symptoms = None
    if symptoms is None:
        pytest.skip("Predefined symptoms list not found in ui")
    assert len(symptoms) >= 10
    lower_symptoms = [s.lower() for s in symptoms]
    for required in ["fatigue", "joint pain", "headache"]:
        assert any(required in s for s in lower_symptoms)


def test_tier1_f11_custom_symptom_merged():
    """F11: Verify custom entered symptom is merged into symptom list."""
    sb = _get_sidebar()
    merge_func = getattr(sb, "merge_symptoms", None)
    if not merge_func:
        pytest.skip("ui.sidebar.merge_symptoms not yet implemented")
    merged = merge_func(["fatigue"], "tinnitus")
    assert "fatigue" in merged and "tinnitus" in merged


def test_tier1_f11_query_concatenation():
    """F11: Verify multi-symptom list formats into concatenated query string."""
    sb = _get_sidebar()
    format_func = getattr(sb, "format_symptoms_query", None)
    if not format_func:
        pytest.skip("ui.sidebar.format_symptoms_query not yet implemented")
    q = format_func(["fatigue", "joint pain"])
    assert q == "fatigue joint pain"


def test_tier1_f11_search_disabled_when_empty():
    """F11: Verify search query validation handles empty symptoms list."""
    sb = _get_sidebar()
    is_valid_func = getattr(sb, "is_valid_query", None)
    if not is_valid_func:
        pytest.skip("ui.sidebar.is_valid_query not yet implemented")
    assert is_valid_func([]) is False


def test_tier1_f11_whitespace_symptoms_filtered():
    """F11: Verify blank and whitespace-only symptom entries are stripped."""
    sb = _get_sidebar()
    clean_func = getattr(sb, "clean_symptoms", None)
    if not clean_func:
        pytest.skip("ui.sidebar.clean_symptoms not yet implemented")
    cleaned = clean_func(["fatigue", "   ", ""])
    assert cleaned == ["fatigue"]


# ============================================================================
# Feature 12: Sidebar Filtering & Trial Count Metric (ui/sidebar.py, app.py)
# ============================================================================

def test_tier1_f12_phase_filter_options_complete():
    """F12: Verify Phase filter options include All and Phases 1-4."""
    sb = _get_sidebar()
    options = getattr(sb, "PHASE_OPTIONS", None)
    if options is None:
        pytest.skip("ui.sidebar.PHASE_OPTIONS not yet implemented")
    for phase in ["All", "Phase 1", "Phase 2", "Phase 3", "Phase 4"]:
        assert phase in options


def test_tier1_f12_status_filter_options_complete():
    """F12: Verify Recruitment Status filter options include All and RECRUITING."""
    sb = _get_sidebar()
    options = getattr(sb, "STATUS_OPTIONS", None)
    if options is None:
        pytest.skip("ui.sidebar.STATUS_OPTIONS not yet implemented")
    assert "All" in options
    assert any("RECRUITING" in opt for opt in options)


def test_tier1_f12_top_k_slider_bounds():
    """F12: Verify Top-K slider bounds are within 5 to 20 with default 10."""
    sb = _get_sidebar()
    if not (hasattr(sb, "TOP_K_MIN") and hasattr(sb, "TOP_K_MAX") and hasattr(sb, "TOP_K_DEFAULT")):
        pytest.skip("ui.sidebar top_k constants not yet implemented")
    min_val = getattr(sb, "TOP_K_MIN")
    max_val = getattr(sb, "TOP_K_MAX")
    default_val = getattr(sb, "TOP_K_DEFAULT")
    assert min_val == 5
    assert max_val == 20
    assert default_val == 10


def test_tier1_f12_metric_displays_chroma_count():
    """F12: Verify trial count formatting for sidebar metric display."""
    sb = _get_sidebar()
    format_metric = getattr(sb, "format_trial_count_metric", None)
    if not format_metric:
        pytest.skip("ui.sidebar.format_trial_count_metric not yet implemented")
    label, val = format_metric(42)
    assert val == 42 or str(val) == "42"


def test_tier1_f12_all_filter_passes_none_to_store():
    """F12: Verify selecting 'All' filters converts to None for ChromaDB query."""
    sb = _get_sidebar()
    normalize_filters = getattr(sb, "normalize_filters", None)
    if not normalize_filters:
        pytest.skip("ui.sidebar.normalize_filters not yet implemented")
    phase, status = normalize_filters("All", "All")
    assert phase is None
    assert status is None


# ============================================================================
# Feature 13: Manual DB Refresh Button (ui/sidebar.py, app.py)
# ============================================================================

def test_tier1_f13_refresh_button_triggers_ingestion():
    """F13: Verify DB refresh pipeline connects to trial fetching."""
    sb = _get_sidebar()
    refresh_func = getattr(sb, "refresh_database", None) or getattr(sb, "trigger_refresh", None)
    if not refresh_func:
        pytest.skip("ui.sidebar.refresh_database not yet implemented")

    sample = make_mock_study(nct_id="NCT_TIER1_REF_0")
    with patch("core.trial_fetcher.preload_standard_trials", return_value=[sample]) as mock_preload:
        with patch("core.vector_store.upsert_trials"):
            refresh_func()
            assert mock_preload.called


def test_tier1_f13_refresh_pipeline_embeds_studies(mock_embedder):
    """F13: Verify DB refresh pipeline passes documents to sentence embedder."""
    sb = _get_sidebar()
    refresh_func = getattr(sb, "refresh_database", None) or getattr(sb, "trigger_refresh", None)
    if not refresh_func:
        pytest.skip("ui.sidebar.refresh_database not yet implemented")

    sample = make_mock_study(nct_id="NCT_TIER1_REF_1", brief_title="Refresh Study 1")
    with patch("core.trial_fetcher.preload_standard_trials", return_value=[sample]):
        with patch("core.embedder.embed_text", wraps=mock_embedder["embed_text"]) as mock_embed:
            with patch("core.vector_store.upsert_trials"):
                refresh_func()
                assert mock_embed.called
                args, _ = mock_embed.call_args
                assert len(args[0]) >= 1


def test_tier1_f13_refresh_pipeline_upserts_chroma(mock_embedder):
    """F13: Verify DB refresh pipeline connects to vector store upsert."""
    sb = _get_sidebar()
    refresh_func = getattr(sb, "refresh_database", None) or getattr(sb, "trigger_refresh", None)
    if not refresh_func:
        pytest.skip("ui.sidebar.refresh_database not yet implemented")

    sample = make_mock_study(nct_id="NCT_TIER1_REF_2", brief_title="Refresh Study 2")
    with patch("core.trial_fetcher.preload_standard_trials", return_value=[sample]):
        with patch("core.vector_store.upsert_trials") as mock_upsert:
            refresh_func()
            assert mock_upsert.called
            call_args = mock_upsert.call_args
            call_ids = call_args[1].get("ids") if call_args[1] and "ids" in call_args[1] else call_args[0][0]
            assert "NCT_TIER1_REF_2" in call_ids


def test_tier1_f13_refresh_success_message():
    """F13: Verify format of refresh completion message."""
    sb = _get_sidebar()
    format_success = getattr(sb, "format_refresh_success", None)
    if not format_success:
        pytest.skip("ui.sidebar.format_refresh_success not yet implemented")
    msg = format_success(200)
    assert "200" in msg


def test_tier1_f13_refresh_updates_trial_count(temp_chroma_path):
    """F13: Verify upserting trials increases total trial count."""
    try:
        import core.vector_store as vs
        initial = vs.get_trial_count()
        vs.upsert_trials(
            ids=["NCT_REFRESH_1"],
            embeddings=[make_deterministic_vector("test")],
            documents=["Document test"],
            metadatas=[{"phase": "Phase 1", "status": "RECRUITING"}],
        )
        assert vs.get_trial_count() == initial + 1
    except ImportError:
        pytest.skip("core.vector_store not yet implemented")


# ============================================================================
# Feature 14: Ranked Trial Result Cards (ui/components.py, app.py)
# ============================================================================

def test_tier1_f14_result_card_computes_match_score():
    """F14: Verify match score percentage computation for result cards."""
    comp = _get_components()
    calc = getattr(comp, "calculate_match_percentage", None)
    if not calc:
        pytest.skip("ui.components.calculate_match_percentage not yet implemented")
    score = calc(0.12)
    assert score == 88.0 or score == "88.0%" or score == 88


def test_tier1_f14_result_card_nct_link_generated():
    """F14: Verify ClinicalTrials.gov external study link formatting."""
    comp = _get_components()
    link_fn = getattr(comp, "get_study_url", None)
    if not link_fn:
        pytest.skip("ui.components.get_study_url not yet implemented")
    url = link_fn("NCT04567890")
    assert url == "https://clinicaltrials.gov/study/NCT04567890"


def test_tier1_f14_result_card_status_badge():
    """F14: Verify status badge text formatting."""
    comp = _get_components()
    badge_fn = getattr(comp, "get_status_badge", None)
    if not badge_fn:
        pytest.skip("ui.components.get_status_badge not yet implemented")
    badge = badge_fn("RECRUITING")
    assert "RECRUITING" in str(badge)


def test_tier1_f14_result_card_phase_caption():
    """F14: Verify phase caption formatting."""
    comp = _get_components()
    caption_fn = getattr(comp, "get_phase_caption", None)
    if not caption_fn:
        pytest.skip("ui.components.get_phase_caption not yet implemented")
    cap = caption_fn("Phase 2")
    assert "Phase 2" in cap


def test_tier1_f14_result_card_first_item_expanded():
    """F14: Verify only top-ranked card is auto-expanded by default."""
    comp = _get_components()
    is_exp = getattr(comp, "should_expand_card", None)
    if not is_exp:
        pytest.skip("ui.components.should_expand_card not yet implemented")
    assert is_exp(0) is True
    assert is_exp(1) is False


# ============================================================================
# Feature 15: Interactive Chat Panel UI (ui/chat.py, app.py)
# ============================================================================

def test_tier1_f15_chat_history_initialized_in_session():
    """F15: Verify chat_history session state is initialized to empty list."""
    ch = _get_chat()
    init_fn = getattr(ch, "init_chat_history", None)
    if not init_fn:
        pytest.skip("ui.chat.init_chat_history not yet implemented")
    session = {}
    init_fn(session)
    assert session.get("chat_history") == []


def test_tier1_f15_chat_renders_existing_messages():
    """F15: Verify rendering of user and assistant chat history messages."""
    ch = _get_chat()
    render_fn = getattr(ch, "render_chat_messages", None)
    if not render_fn:
        pytest.skip("ui.chat.render_chat_messages not yet implemented")
    history = [
        {"role": "user", "content": "What are side effects?"},
        {"role": "assistant", "content": "Common side effects include fatigue."},
    ]
    with patch("streamlit.chat_message") as mock_cm:
        render_fn(history)
        assert mock_cm.call_count == 2


def test_tier1_f15_chat_input_appends_user_message():
    """F15: Verify submitting a question appends user message to chat history."""
    ch = _get_chat()
    append_fn = getattr(ch, "append_user_message", None)
    if not append_fn:
        pytest.skip("ui.chat.append_user_message not yet implemented")
    history = []
    append_fn(history, "What are side effects?")
    assert len(history) == 1
    assert history[0] == {"role": "user", "content": "What are side effects?"}


def test_tier1_f15_chat_appends_assistant_response():
    """F15: Verify LLM response is appended as assistant message to chat history."""
    ch = _get_chat()
    append_fn = getattr(ch, "append_assistant_message", None)
    if not append_fn:
        pytest.skip("ui.chat.append_assistant_message not yet implemented")
    history = []
    append_fn(history, "Fatigue is common.")
    assert len(history) == 1
    assert history[0] == {"role": "assistant", "content": "Fatigue is common."}


def test_tier1_f15_chat_history_cleared_on_new_search():
    """F15: Verify initiating a new search resets chat history to empty."""
    ch = _get_chat()
    clear_fn = getattr(ch, "clear_chat_history", None)
    if not clear_fn:
        pytest.skip("ui.chat.clear_chat_history not yet implemented")
    session = {"chat_history": [{"role": "user", "content": "Hi"}]}
    clear_fn(session)
    assert session["chat_history"] == []
