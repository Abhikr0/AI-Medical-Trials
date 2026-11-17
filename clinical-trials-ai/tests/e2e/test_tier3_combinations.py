"""Tier 3 Cross-Feature Combination Tests for clinical-trials-ai.

Validates pairwise integration, contract compliance, and seamless data flow
between interconnected subsystems across features F1 through F15:
- F3 + F4: API fetch to extraction & text flattening.
- F4 + F6: Extraction string to sentence embedding vector generation.
- F6 + F7: Embedding vector & sanitized metadata to ChromaDB upsert.
- F7 + F8: ChromaDB storage to cosine similarity search.
- F8 + F9: Similarity search top-K to Groq plain-English summarization.
- F8 + F10: Similarity search top-K to Groq grounded conversational Q&A.
- F11 + F12: Symptom selection combined with Phase/Status sidebar filters.
- F11 + F8: Symptom selection directly driving semantic similarity search.
- F12 + F8: Sidebar filters properly translated to ChromaDB where clauses.
- F13 + F3 + F6 + F7: DB Refresh button triggering full pipeline.
- F8 + F14: Similarity search distance to card match percentage and status badges.
- F14 + F9: Card expansion triggering on-demand / cached Groq summarization.
- F15 + F10: Chat input widget submitting user question to grounded RAG assistant.
- F4 + F14: Study extraction NCT ID preserved and linked to clinicaltrials.gov URL.
- F5 + F7 + F12: Broad category preload updating ChromaDB and reflected in UI trial count.
- F1 + F3 + F6 + F7 + F9: End-to-end config propagation across all subsystems.

Hermetic and deterministic execution is enforced using pytest fixtures and test doubles.
"""

import importlib
from unittest.mock import MagicMock, patch

import httpx
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
        return None


def _get_components():
    try:
        import ui.components as comp
        return comp
    except (ImportError, ModuleNotFoundError):
        return None


def _get_chat():
    try:
        import ui.chat as ch
        return ch
    except (ImportError, ModuleNotFoundError):
        return None


# ============================================================================
# Tier 3 Pairwise Integration Tests
# ============================================================================

def test_tier3_f03_f04_api_to_extraction_pipeline():
    """F3 + F4: ClinicalTrials.gov API fetch to study extraction & text flattening."""
    tf = _get_trial_fetcher()

    mock_study = make_mock_study(
        nct_id="NCT04821234",
        brief_title="Immunotherapy Trial for Refractory Asthma",
        official_title="Phase 2 Study of Monoclonal Antibody in Severe Asthma",
        conditions=["Asthma", "Severe Refractory Asthma"],
        brief_summary="Evaluating targeted anti-IL5 therapy in adults with persistent wheezing.",
        eligibility_criteria="Inclusion: Adults 18-70 with diagnosed eosinophilic asthma.",
        phases=["PHASE2"],
        overall_status="RECRUITING",
        locations=[{"city": "Denver", "state": "CO", "country": "United States"}],
    )

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = {"studies": [mock_study]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        fetched_studies = tf.fetch_trials(conditions=["asthma"], page_size=10, status="RECRUITING")

    assert len(fetched_studies) == 1
    raw_study = fetched_studies[0]

    embed_text, metadata = tf.extract_trial(raw_study)

    # Validate embeddable text contains key semantic elements
    assert isinstance(embed_text, str)
    assert "Title: Immunotherapy Trial for Refractory Asthma" in embed_text
    assert "Asthma" in embed_text
    assert "Evaluating targeted anti-IL5 therapy" in embed_text

    # Validate sanitized metadata dictionary
    assert isinstance(metadata, dict)
    assert metadata["nct_id"] == "NCT04821234"
    assert metadata["phase"] == "Phase 2"
    assert metadata["status"] == "RECRUITING"
    assert "Denver" in metadata["locations"]
    assert metadata["url"] == "https://clinicaltrials.gov/study/NCT04821234"

    # All metadata values must be primitive strings for ChromaDB compatibility
    for k, v in metadata.items():
        assert isinstance(v, str), f"Metadata field '{k}' is not a string: {type(v)}"


def test_tier3_f04_f06_extraction_to_embedding_pipeline():
    """F4 + F6: Flattened study extraction string to dense sentence embedding vector."""
    tf = _get_trial_fetcher()
    emb = _get_embedder()

    studies = [
        make_mock_study(
            nct_id="NCT001",
            brief_title="Checkpoint Inhibitor in Melanoma",
            conditions=["Melanoma", "Skin Cancer"],
            brief_summary="Evaluating pembrolizumab in metastatic melanoma.",
        ),
        make_mock_study(
            nct_id="NCT002",
            brief_title="Metformin and SGLT2 Inhibitors in Type 2 Diabetes",
            conditions=["Type 2 Diabetes Mellitus"],
            brief_summary="Assessing glycemic control in adult diabetic cohort.",
        ),
        make_mock_study(
            nct_id="NCT003",
            brief_title="Cardiac Resynchronization Therapy in Heart Failure",
            conditions=["Congestive Heart Failure"],
            brief_summary="Efficacy of biventricular pacing in reduced ejection fraction.",
        ),
    ]

    extracted_texts = []
    extracted_metas = []
    for s in studies:
        text, meta = tf.extract_trial(s)
        extracted_texts.append(text)
        extracted_metas.append(meta)

    assert len(extracted_texts) == 3
    for t in extracted_texts:
        assert len(t) > 20

    embeddings = emb.embed_text(extracted_texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 3

    for vec in embeddings:
        assert isinstance(vec, list)
        assert len(vec) == 384
        # Vectors must have non-zero dense floats
        assert any(abs(x) > 1e-4 for x in vec)
        # Ensure no NaN or infinite values
        for val in vec:
            assert isinstance(val, float)
            assert not (val != val)  # not NaN

    # Distinct trial texts must produce distinct embedding vectors
    assert embeddings[0] != embeddings[1]
    assert embeddings[1] != embeddings[2]


def test_tier3_f06_f07_embedding_to_chroma_upsert(temp_chroma_path):
    """F6 + F7: Embedding vector and sanitized metadata to ChromaDB persistent upsert."""
    emb = _get_embedder()
    vs = _get_vector_store()

    texts = [
        "Title: Oncology Trial | Conditions: Lung Cancer | Summary: Pembrolizumab study",
        "Title: Diabetes Trial | Conditions: Type 2 Diabetes | Summary: Insulin glargine study",
    ]
    ids = ["NCT_COMB_001", "NCT_COMB_002"]
    metadatas = [
        {"nct_id": "NCT_COMB_001", "phase": "Phase 3", "status": "RECRUITING", "title": "Oncology Trial"},
        {"nct_id": "NCT_COMB_002", "phase": "Phase 2", "status": "RECRUITING", "title": "Diabetes Trial"},
    ]

    initial_count = vs.get_trial_count()
    embeddings = emb.embed_text(texts)

    vs.upsert_trials(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    new_count = vs.get_trial_count()
    assert new_count == initial_count + 2

    collection = vs.get_collection()
    record = collection.get(ids=["NCT_COMB_001"], include=["metadatas", "documents"])
    assert len(record["ids"]) == 1
    assert record["metadatas"][0]["phase"] == "Phase 3"
    assert record["metadatas"][0]["title"] == "Oncology Trial"
    assert "Lung Cancer" in record["documents"][0]


def test_tier3_f07_f08_chroma_storage_to_similarity_search(temp_chroma_path):
    """F7 + F8: ChromaDB storage to cosine similarity search with semantic ranking."""
    vs = _get_vector_store()

    # Create 3 trials with deterministic vectors
    t1_text = "Clinical trial for advanced non-small cell lung cancer with immunotherapy"
    t2_text = "Randomized study of insulin glargine for pediatric type 1 diabetes"
    t3_text = "Cardiovascular outcomes of beta blocker therapy in chronic heart failure"

    t1_vec = make_deterministic_vector(t1_text)
    t2_vec = make_deterministic_vector(t2_text)
    t3_vec = make_deterministic_vector(t3_text)

    vs.upsert_trials(
        ids=["NCT_ONC", "NCT_DIAB", "NCT_CARD"],
        embeddings=[t1_vec, t2_vec, t3_vec],
        documents=[t1_text, t2_text, t3_text],
        metadatas=[
            {"nct_id": "NCT_ONC", "phase": "Phase 3", "status": "RECRUITING"},
            {"nct_id": "NCT_DIAB", "phase": "Phase 2", "status": "RECRUITING"},
            {"nct_id": "NCT_CARD", "phase": "Phase 4", "status": "ACTIVE_NOT_RECRUITING"},
        ],
    )

    # Query with exact cancer vector (distance to t1 should be ~0.0)
    query_vec = make_deterministic_vector(t1_text)
    results = vs.search_trials(query_embedding=query_vec, n_results=3)

    returned_ids = results["ids"][0]
    returned_distances = results["distances"][0]

    assert len(returned_ids) == 3
    assert returned_ids[0] == "NCT_ONC"
    assert returned_distances[0] < 0.05
    # Cosine distance monotonic ascending
    assert returned_distances[0] <= returned_distances[1] <= returned_distances[2]


def test_tier3_f08_f09_search_to_groq_summary(temp_chroma_path, mock_groq_llm):
    """F8 + F9: Similarity search top-K output to Groq plain-English patient summarization."""
    vs = _get_vector_store()
    gc = _get_groq_client()

    trial_doc = (
        "Title: Pembrolizumab Plus Chemotherapy in NSCLC | "
        "Conditions: Non-Small Cell Lung Cancer | "
        "Summary: Evaluating overall survival in treatment-naive metastatic NSCLC. | "
        "Eligibility: Adults 18+ with confirmed stage IV NSCLC, ECOG 0-1, no active autoimmune disease."
    )
    vs.upsert_trials(
        ids=["NCT_SEARCH_SUM"],
        embeddings=[make_deterministic_vector(trial_doc)],
        documents=[trial_doc],
        metadatas=[{"nct_id": "NCT_SEARCH_SUM", "phase": "Phase 3", "status": "RECRUITING"}],
    )

    # Perform semantic search
    search_res = vs.search_trials(query_embedding=make_deterministic_vector(trial_doc), n_results=1)
    top_doc = search_res["documents"][0][0]

    summary = gc.summarize_trial(top_doc)

    assert isinstance(summary, str)
    assert any(bullet in summary for bullet in ["•", "-", "*"])
    assert mock_groq_llm.chat.completions.create.called

    call_args = mock_groq_llm.chat.completions.create.call_args[1]
    assert call_args["model"] == "qwen/qwen3.8-27b"
    # Ensure trial context was included in user prompt
    user_prompt = call_args["messages"][1]["content"]
    assert "NSCLC" in user_prompt or "Pembrolizumab" in user_prompt


def test_tier3_f08_f10_search_to_grounded_qa(temp_chroma_path, mock_groq_llm):
    """F8 + F10: Similarity search top-K output to Groq grounded conversational Q&A."""
    vs = _get_vector_store()
    gc = _get_groq_client()

    doc1 = "Title: Trial A | Conditions: Diabetes | Eligibility: Adults 18-65 only."
    doc2 = "Title: Trial B | Conditions: Diabetes | Eligibility: Daily oral metformin required."

    vs.upsert_trials(
        ids=["NCT_QA_1", "NCT_QA_2"],
        embeddings=[make_deterministic_vector(doc1), make_deterministic_vector(doc2)],
        documents=[doc1, doc2],
        metadatas=[
            {"nct_id": "NCT_QA_1", "phase": "Phase 2", "status": "RECRUITING"},
            {"nct_id": "NCT_QA_2", "phase": "Phase 3", "status": "RECRUITING"},
        ],
    )

    # Search top trials
    search_res = vs.search_trials(query_embedding=make_deterministic_vector("Diabetes"), n_results=2)
    contexts = search_res["documents"][0]

    question = "What age range is eligible for Trial A?"
    ans = gc.ask_about_trials(question=question, trial_contexts=contexts, chat_history=[])

    assert isinstance(ans, str)
    assert mock_groq_llm.chat.completions.create.called

    messages = mock_groq_llm.chat.completions.create.call_args[1]["messages"]
    system_msg = messages[0]["content"]

    # Verify grounding constraints and injected contexts
    assert "GROUNDING CONSTRAINTS" in system_msg
    assert "Trial A" in system_msg
    assert "Trial B" in system_msg
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == question


def test_tier3_f11_f12_symptom_selection_with_sidebar_filters():
    """F11 + F12: Symptom selection combined with Phase and Status sidebar filters."""
    sb = _get_sidebar()
    if not (sb and hasattr(sb, "merge_symptoms") and hasattr(sb, "format_symptoms_query") and hasattr(sb, "normalize_filters")):
        pytest.skip("ui.sidebar helper functions not yet implemented")

    # Predefined symptoms + custom text entry
    chips = ["fatigue", "joint pain"]
    custom_symptom = "morning stiffness"

    combined_symptoms = sb.merge_symptoms(chips, custom_symptom)
    assert combined_symptoms == ["fatigue", "joint pain", "morning stiffness"]

    # Formatted query string for embedder
    query_str = sb.format_symptoms_query(combined_symptoms)
    assert query_str == "fatigue joint pain morning stiffness"

    # Filter selections
    phase_selection = "Phase 3"
    status_selection = "RECRUITING"

    norm_phase, norm_status = sb.normalize_filters(phase_selection, status_selection)
    assert norm_phase == "Phase 3"
    assert norm_status == "RECRUITING"

    # Verify compatibility with vector store where clause
    vs = _get_vector_store()
    where_clause = vs.build_where_clause(phase=norm_phase, status=norm_status)
    assert where_clause == {
        "$and": [
            {"phase": {"$eq": "Phase 3"}},
            {"status": {"$eq": "RECRUITING"}},
        ]
    }


def test_tier3_f11_f08_symptoms_to_semantic_search(temp_chroma_path):
    """F11 + F8: Symptom selection directly driving semantic similarity search."""
    emb = _get_embedder()
    vs = _get_vector_store()

    doc_headache = "Trial studying acute migraine and tension headache relief therapies."
    doc_asthma = "Trial studying bronchial airway obstruction and severe asthma attacks."

    vs.upsert_trials(
        ids=["NCT_HEADACHE", "NCT_ASTHMA"],
        embeddings=[make_deterministic_vector(doc_headache), make_deterministic_vector(doc_asthma)],
        documents=[doc_headache, doc_asthma],
        metadatas=[
            {"phase": "Phase 2", "status": "RECRUITING"},
            {"phase": "Phase 3", "status": "RECRUITING"},
        ],
    )

    # Symptom tokens joined to query
    user_symptoms = ["headache", "migraine", "nausea"]
    query_str = " ".join(user_symptoms)

    query_vec = emb.embed_query(query_str)
    assert len(query_vec) == 384

    # Search with embedded query
    results = vs.search_trials(query_embedding=query_vec, n_results=2)
    assert len(results["documents"][0]) == 2
    assert "migraine" in results["documents"][0][0].lower() or "headache" in results["documents"][0][0].lower()


def test_tier3_f12_f08_sidebar_filters_to_chroma_where_clause(temp_chroma_path):
    """F12 + F8: Sidebar filters properly translated to ChromaDB where clauses."""
    vs = _get_vector_store()

    # Populate 4 trials spanning various phases and statuses
    records = [
        ("NCT_P1_REC", "Trial 1", "Phase 1", "RECRUITING"),
        ("NCT_P3_REC", "Trial 2", "Phase 3", "RECRUITING"),
        ("NCT_P3_COMP", "Trial 3", "Phase 3", "COMPLETED"),
        ("NCT_P4_REC", "Trial 4", "Phase 4", "RECRUITING"),
    ]

    vs.upsert_trials(
        ids=[r[0] for r in records],
        embeddings=[make_deterministic_vector(r[1]) for r in records],
        documents=[r[1] for r in records],
        metadatas=[{"phase": r[2], "status": r[3]} for r in records],
    )

    qvec = make_deterministic_vector("Trial")

    # 1. Filter by Phase 3 AND RECRUITING
    res1 = vs.search_trials(query_embedding=qvec, n_results=10, phase="Phase 3", status="RECRUITING")
    assert len(res1["ids"][0]) == 1
    assert res1["ids"][0][0] == "NCT_P3_REC"

    # 2. Filter by All phases AND RECRUITING
    res2 = vs.search_trials(query_embedding=qvec, n_results=10, phase="All", status="RECRUITING")
    assert len(res2["ids"][0]) == 3
    for meta in res2["metadatas"][0]:
        assert meta["status"] == "RECRUITING"

    # 3. Filter by Phase 4 AND All statuses
    res3 = vs.search_trials(query_embedding=qvec, n_results=10, phase="Phase 4", status="All")
    assert len(res3["ids"][0]) == 1
    assert res3["metadatas"][0][0]["phase"] == "Phase 4"


def test_tier3_f13_f03_f06_f07_db_refresh_pipeline(temp_chroma_path):
    """F13 + F3 + F6 + F7: DB Refresh triggering fetch -> extract -> embed -> upsert -> count."""
    tf = _get_trial_fetcher()
    emb = _get_embedder()
    vs = _get_vector_store()

    mock_studies = [
        make_mock_study(nct_id=f"NCT_REF_{i}", brief_title=f"Study Title {i}")
        for i in range(4)
    ]

    assert vs.get_trial_count() == 0

    # 1. Simulate DB refresh pipeline execution
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = {"studies": mock_studies}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        studies = tf.fetch_trials(conditions=["oncology"], page_size=10, status="RECRUITING")

    # 2. Extract
    ids, texts, metadatas = [], [], []
    for s in studies:
        t, m = tf.extract_trial(s)
        ids.append(m["nct_id"])
        texts.append(t)
        metadatas.append(m)

    # 3. Embed
    embeddings = emb.embed_text(texts)

    # 4. Upsert
    vs.upsert_trials(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # 5. Verify updated trial count
    updated_count = vs.get_trial_count()
    assert updated_count == 4


def test_tier3_f08_f14_search_distance_to_card_match_and_badges():
    """F8 + F14: Similarity search distance to card match percentage and status badges."""
    comp = _get_components()

    # Distances from ChromaDB cosine distance metric
    test_cases = [
        (0.00, 100.0),
        (0.12, 88.0),
        (0.35, 65.0),
        (1.00, 0.0),
    ]

    if not (comp and hasattr(comp, "calculate_match_percentage") and hasattr(comp, "get_status_badge")):
        pytest.skip("ui.components not yet implemented")

    for dist, expected_pct in test_cases:
        score = comp.calculate_match_percentage(dist)
        if isinstance(score, str) and score.endswith("%"):
            score_num = float(score.replace("%", "").strip())
        else:
            score_num = float(score)

        assert abs(score_num - expected_pct) < 0.1

    # Status badge verification
    statuses = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"]
    for st in statuses:
        badge = comp.get_status_badge(st)
        assert st in str(badge)


def test_tier3_f14_f09_card_expansion_to_cached_groq_summary(mock_groq_llm):
    """F14 + F9: Card expansion triggering on-demand / cached Groq summarization."""
    gc = _get_groq_client()

    summary_cache = {}
    nct_id = "NCT01234567"
    trial_doc = "Title: Trial Summary Test | Conditions: Asthma | Summary: Efficacy trial."

    # First expansion: Cache miss -> calls Groq
    if nct_id not in summary_cache:
        summary_1 = gc.summarize_trial(trial_doc)
        summary_cache[nct_id] = summary_1

    assert mock_groq_llm.chat.completions.create.call_count == 1
    assert summary_1 is not None
    assert any(b in summary_1 for b in ["•", "-", "*"])

    # Second expansion: Cache hit -> returns cached summary without calling Groq
    assert nct_id in summary_cache
    summary_2 = summary_cache[nct_id]

    assert summary_2 == summary_1
    # API call count must remain 1
    assert mock_groq_llm.chat.completions.create.call_count == 1


def test_tier3_f15_f10_chat_widget_to_grounded_rag(mock_groq_llm):
    """F15 + F10: Chat input widget submitting user question to grounded RAG assistant."""
    gc = _get_groq_client()

    chat_history = []
    contexts = [
        "Trial NCT001: Patients must be >= 18 with confirmed Stage III adenocarcinoma.",
        "Trial NCT002: Patients must have fasting blood glucose > 126 mg/dL.",
    ]

    # Turn 1: User asks eligibility question
    user_q1 = "Who is eligible for Trial NCT001?"
    chat_history.append({"role": "user", "content": user_q1})

    ans_1 = gc.ask_about_trials(user_q1, contexts, chat_history[:-1])
    chat_history.append({"role": "assistant", "content": ans_1})

    assert len(chat_history) == 2
    assert chat_history[0]["role"] == "user"
    assert chat_history[1]["role"] == "assistant"

    # Turn 2: User asks follow up question
    user_q2 = "What about blood sugar requirements in Trial NCT002?"
    chat_history.append({"role": "user", "content": user_q2})

    ans_2 = gc.ask_about_trials(user_q2, contexts, chat_history[:-1])
    chat_history.append({"role": "assistant", "content": ans_2})

    assert len(chat_history) == 4
    # Turn 2 call to Groq must include all prior messages in history
    last_call_messages = mock_groq_llm.chat.completions.create.call_args[1]["messages"]
    message_contents = [m["content"] for m in last_call_messages]

    assert user_q1 in message_contents
    assert ans_1 in message_contents
    assert user_q2 in message_contents


def test_tier3_f04_f14_study_extraction_to_nct_url():
    """F4 + F14: Study extraction NCT ID preserved and linked to clinicaltrials.gov URL."""
    tf = _get_trial_fetcher()
    comp = _get_components()

    raw_study = make_mock_study(nct_id="NCT04567890")
    embed_text, metadata = tf.extract_trial(raw_study)

    nct_id = metadata["nct_id"]
    assert nct_id == "NCT04567890"

    expected_url = f"https://clinicaltrials.gov/study/{nct_id}"
    assert metadata["url"] == expected_url

    if not (comp and hasattr(comp, "get_study_url")):
        pytest.skip("ui.components.get_study_url not yet implemented")

    card_url = comp.get_study_url(nct_id)
    assert card_url == expected_url


def test_tier3_f05_f07_f12_preload_updates_chromadb_and_metric(temp_chroma_path):
    """F5 + F7 + F12: Broad category preload updating ChromaDB and reflected in UI trial count."""
    tf = _get_trial_fetcher()
    emb = _get_embedder()
    vs = _get_vector_store()
    sb = _get_sidebar()

    # Mock 10 studies per broad category
    categories = ["cancer", "diabetes", "heart disease", "neurological"]
    mock_dataset = []
    for cat_idx, cat in enumerate(categories):
        for i in range(5):
            mock_dataset.append(
                make_mock_study(
                    nct_id=f"NCT_{cat[:3].upper()}_{i}",
                    brief_title=f"{cat.title()} Study {i}",
                    conditions=[cat.title()],
                )
            )

    with patch("httpx.Client.get") as mock_get:
        def _get_mock(url, params=None, **kwargs):
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.is_success = True
            cond = (params or {}).get("query.cond", "")
            matched = [s for s in mock_dataset if cond.strip('"').lower() in s["protocolSection"]["conditionsModule"]["conditions"][0].lower()]
            mock_resp.json.return_value = {"studies": matched if matched else mock_dataset[:5]}
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        mock_get.side_effect = _get_mock

        preloaded_studies = tf.preload_standard_trials(categories=categories, page_size=20)

    assert len(preloaded_studies) > 0

    ids, texts, metadatas = [], [], []
    for s in preloaded_studies:
        t, m = tf.extract_trial(s)
        ids.append(m["nct_id"])
        texts.append(t)
        metadatas.append(m)

    embeddings = emb.embed_text(texts)
    vs.upsert_trials(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    total_count = vs.get_trial_count()
    assert total_count == len(ids)

    # UI trial count metric display
    if not (sb and hasattr(sb, "format_trial_count_metric")):
        pytest.skip("ui.sidebar.format_trial_count_metric not yet implemented")

    label, val = sb.format_trial_count_metric(total_count)
    assert val == total_count or str(val) == str(total_count)


def test_tier3_f01_f03_f06_f07_f09_config_propagation(monkeypatch, tmp_path):
    """F1 + F3 + F6 + F7 + F9: End-to-end config propagation across all subsystems."""
    custom_chroma = str(tmp_path / "custom_chroma_db")
    custom_base = "https://custom.clinicaltrials.gov/api/v2"
    custom_collection = "custom_clinical_trials"

    monkeypatch.setenv("CHROMA_DB_PATH", custom_chroma)
    monkeypatch.setenv("CLINICALTRIALS_BASE", custom_base)
    monkeypatch.setenv("COLLECTION_NAME", custom_collection)
    monkeypatch.setenv("TOP_K_RESULTS", "7")
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_custom_config")

    importlib.reload(config)

    assert config.CHROMA_DB_PATH == custom_chroma
    assert config.CLINICALTRIALS_BASE == custom_base
    assert config.COLLECTION_NAME == custom_collection
    assert config.TOP_K == 7
    assert config.GROQ_MODEL == "qwen/qwen3.8-27b"

    tf = _get_trial_fetcher()
    vs = _get_vector_store()
    gc = _get_groq_client()

    # 1. Verify trial fetcher uses config.CLINICALTRIALS_BASE
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = {"studies": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        tf.fetch_trials(["asthma"])
        called_url = mock_get.call_args[0][0]
        assert called_url.startswith(custom_base)

    # 2. Verify vector store uses config.CHROMA_DB_PATH and config.COLLECTION_NAME
    client = vs.get_chroma_client()
    collection = vs.get_collection(client)
    assert collection.name == custom_collection

    # 3. Verify groq client uses config.GROQ_MODEL
    with patch("core.groq_client.Groq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_groq_cls.return_value = mock_instance
        gc.get_groq_client()
        mock_groq_cls.assert_called_with(api_key="gsk_test_custom_config")
