"""Tier 4 Real-World Application Discovery Workloads for clinical-trials-ai.

Implements the 6 end-to-end clinical trial discovery workloads defined in TEST_INFRA.md:
- Scenario 1: Oncology patient clinical search journey (F1, F3, F4, F6, F7, F8, F9, F14).
- Scenario 2: Caregiver interactive grounded Q&A session (F8, F9, F10, F14, F15).
- Scenario 3: Fresh application startup and broad DB preload (F1, F2, F3, F4, F5, F7, F12, F13).
- Scenario 4: Offline zero-key resilience and heuristic fallbacks (F1, F9, F10, F14, F15).
- Scenario 5: Multi-symptom rare disease discovery (F4, F6, F7, F8, F11, F12).
- Scenario 6: Ingestion stress and deduplication workflow (F3, F4, F5, F7, F8, F13).

Hermetic, offline execution is enforced using pytest fixtures and test doubles.
"""

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


# ============================================================================
# Tier 4 Realistic Discovery Workloads
# ============================================================================

def test_tier4_workload_oncology_patient_journey(temp_chroma_path, mock_groq_llm):
    """Scenario 1: Oncology patient clinical search journey.

    A patient experiencing complex pulmonary symptoms searches for recruiting
    Phase 3 trials, inspects similarity match percentages, reads 3-bullet AI
    summaries, and accesses official study URLs.
    """
    tf = _get_trial_fetcher()
    emb = _get_embedder()
    vs = _get_vector_store()
    gc = _get_groq_client()
    comp = _get_components()

    # 1. Corpus Setup: Diverse trials (Oncology, Cardiology, Neurology)
    raw_studies = [
        # Oncology Phase 3 Recruiting (Target)
        make_mock_study(
            nct_id="NCT03874901",
            brief_title="Phase 3 Pembrolizumab and Chemotherapy in Non-Small Cell Lung Cancer",
            official_title="A Randomized Phase 3 Study of Pembrolizumab Plus Pemetrexed in Metastatic NSCLC",
            conditions=["Non-Small Cell Lung Cancer", "Carcinoma, Non-Small-Cell Lung"],
            brief_summary="Evaluating overall survival and progression-free survival in patients with untreated metastatic NSCLC presenting with persistent cough and shortness of breath.",
            eligibility_criteria="Inclusion:\n* Age >= 18\n* Histologically confirmed Stage IV NSCLC\n* ECOG performance status 0-1\nExclusion:\n* Prior systemic chemotherapy",
            phases=["PHASE3"],
            overall_status="RECRUITING",
        ),
        # Oncology Phase 1 Recruiting (Should be filtered out by Phase 3 filter)
        make_mock_study(
            nct_id="NCT03874902",
            brief_title="Phase 1 Trial of Novel KRAS Inhibitor in Advanced Solid Tumors",
            conditions=["Lung Cancer", "Colorectal Cancer"],
            phases=["PHASE1"],
            overall_status="RECRUITING",
        ),
        # Oncology Phase 3 Completed (Should be filtered out by RECRUITING filter)
        make_mock_study(
            nct_id="NCT03874903",
            brief_title="Phase 3 Adjuvant Immunotherapy in Resected Lung Cancer",
            conditions=["Non-Small Cell Lung Cancer"],
            phases=["PHASE3"],
            overall_status="COMPLETED",
        ),
        # Cardiology Phase 3 Recruiting (Different disease space)
        make_mock_study(
            nct_id="NCT03874904",
            brief_title="Beta-Blocker Evaluation in Chronic Heart Failure",
            conditions=["Heart Failure", "Cardiomyopathy"],
            phases=["PHASE3"],
            overall_status="RECRUITING",
        ),
        # Neurology Phase 3 Recruiting (Different disease space)
        make_mock_study(
            nct_id="NCT03874905",
            brief_title="Monoclonal Antibody Targeting Amyloid in Early Alzheimer's Disease",
            conditions=["Alzheimer Disease", "Cognitive Impairment"],
            phases=["PHASE3"],
            overall_status="RECRUITING",
        ),
    ]

    # Ingest and vectorize corpus into isolated ChromaDB
    ids, texts, metadatas = [], [], []
    for s in raw_studies:
        t, m = tf.extract_trial(s)
        ids.append(m["nct_id"])
        texts.append(t)
        metadatas.append(m)

    embeddings = emb.embed_text(texts)
    vs.upsert_trials(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    assert vs.get_trial_count() == 5

    # 2. Patient Action: Select symptoms
    patient_symptoms = ["persistent cough", "shortness of breath", "unexplained weight loss"]
    query_str = " ".join(patient_symptoms)

    # 3. Patient Action: Select filters
    selected_phase = "Phase 3"
    selected_status = "RECRUITING"
    top_k = 5

    # 4. Embed Query
    query_vec = emb.embed_query(query_str)
    assert len(query_vec) == 384

    # 5. Semantic Search with Metadata Constraints
    results = vs.search_trials(
        query_embedding=query_vec,
        n_results=top_k,
        phase=selected_phase,
        status=selected_status,
    )

    returned_ids = results["ids"][0]
    returned_metas = results["metadatas"][0]
    returned_docs = results["documents"][0]
    returned_distances = results["distances"][0]

    # Verify filtering constraints
    assert len(returned_ids) >= 1
    assert len(returned_ids) <= top_k

    for meta in returned_metas:
        assert meta["phase"] == "Phase 3"
        assert meta["status"] == "RECRUITING"

    # Top match must be the lung cancer trial NCT03874901
    top_nct = returned_ids[0]
    assert top_nct == "NCT03874901"

    # 6. Result Card Formatting: Calculate match score & verify badge
    top_dist = returned_distances[0]
    if comp and hasattr(comp, "calculate_match_percentage"):
        match_score = comp.calculate_match_percentage(top_dist)
    else:
        match_score = round(max(0.0, min(1.0, 1.0 - top_dist)) * 100, 1)

    assert match_score >= 0.0
    assert match_score <= 100.0

    # 7. AI Plain-English Summarization
    top_doc = returned_docs[0]
    summary = gc.summarize_trial(top_doc)

    assert isinstance(summary, str)
    assert any(bullet in summary for bullet in ["•", "-", "*"])
    assert "who can join" in summary.lower() or "criteria" in summary.lower() or "tests" in summary.lower()

    # 8. External Study Link Verification
    expected_url = f"https://clinicaltrials.gov/study/{top_nct}"
    if comp and hasattr(comp, "get_study_url"):
        study_url = comp.get_study_url(top_nct)
    else:
        study_url = returned_metas[0].get("url")

    assert study_url == expected_url


def test_tier4_workload_caregiver_interactive_qa(temp_chroma_path, mock_groq_llm):
    """Scenario 2: Caregiver interactive grounded Q&A session.

    A caregiver searches for pediatric diabetes trials, reviews top matches,
    and initiates a multi-turn grounded Q&A session asking specific questions
    about eligibility age ranges and protocol requirements.
    """
    tf = _get_trial_fetcher()
    emb = _get_embedder()
    vs = _get_vector_store()
    gc = _get_groq_client()

    # 1. Corpus Setup: Pediatric diabetes trial with clear inclusion criteria
    pediatric_trial = make_mock_study(
        nct_id="NCT04512345",
        brief_title="Trial of Continuous Glucose Monitoring in Pediatric Type 1 Diabetes",
        conditions=["Type 1 Diabetes Mellitus", "Pediatric Diabetes"],
        brief_summary="Evaluating automated insulin delivery and continuous glucose monitoring in children.",
        eligibility_criteria=(
            "Inclusion Criteria:\n"
            "* Age 6 to 17 years at time of screening\n"
            "* Diagnosed with Type 1 Diabetes for at least 6 months\n"
            "* Daily insulin therapy required via pump or multiple daily injections\n"
            "Exclusion Criteria:\n"
            "* History of severe hypoglycemic seizure in past 30 days\n"
            "* Use of non-insulin anti-diabetic medications"
        ),
        phases=["PHASE3"],
        overall_status="RECRUITING",
    )

    t_text, t_meta = tf.extract_trial(pediatric_trial)
    vs.upsert_trials(
        ids=[t_meta["nct_id"]],
        embeddings=emb.embed_text([t_text]),
        documents=[t_text],
        metadatas=[t_meta],
    )

    # 2. Caregiver Search: "frequent thirst frequent urination child fatigue"
    caregiver_symptoms = ["frequent thirst", "frequent urination", "child fatigue"]
    query_str = " ".join(caregiver_symptoms)

    search_res = vs.search_trials(
        query_embedding=emb.embed_query(query_str),
        n_results=1,
    )
    retrieved_contexts = search_res["documents"][0]
    assert len(retrieved_contexts) == 1

    # 3. Interactive Multi-turn Q&A Session
    chat_history = []

    # Mock response for Turn 1 (Age eligibility)
    mock_choice_1 = MagicMock()
    mock_choice_1.message.content = "Based on the trial context, children aged 6 to 17 years old with Type 1 Diabetes are eligible."
    mock_groq_llm.chat.completions.create.return_value = MagicMock(choices=[mock_choice_1])

    # Turn 1: Caregiver Question 1
    q1 = "What age range is eligible to participate in this study?"
    chat_history.append({"role": "user", "content": q1})

    ans1 = gc.ask_about_trials(question=q1, trial_contexts=retrieved_contexts, chat_history=chat_history[:-1])
    chat_history.append({"role": "assistant", "content": ans1})

    assert "6 to 17" in ans1
    assert len(chat_history) == 2

    # Mock response for Turn 2 (Daily injections)
    mock_choice_2 = MagicMock()
    mock_choice_2.message.content = "Yes, the inclusion criteria specify that participants must require daily insulin therapy via pump or multiple daily injections."
    mock_groq_llm.chat.completions.create.return_value = MagicMock(choices=[mock_choice_2])

    # Turn 2: Caregiver Question 2
    q2 = "Does this study require daily insulin injections?"
    chat_history.append({"role": "user", "content": q2})

    ans2 = gc.ask_about_trials(question=q2, trial_contexts=retrieved_contexts, chat_history=chat_history[:-1])
    chat_history.append({"role": "assistant", "content": ans2})

    assert "daily insulin" in ans2.lower() or "injections" in ans2.lower()
    assert len(chat_history) == 4

    # Verify conversation continuity: turn 2 API call received full dialogue context
    last_call = mock_groq_llm.chat.completions.create.call_args[1]
    last_messages = last_call["messages"]
    system_prompt = last_messages[0]["content"]

    assert "Continuous Glucose Monitoring" in system_prompt
    assert "Inclusion Criteria" in system_prompt
    assert {"role": "user", "content": q1} in last_messages
    assert {"role": "assistant", "content": ans1} in last_messages
    assert {"role": "user", "content": q2} in last_messages


def test_tier4_workload_fresh_startup_preload(temp_chroma_path):
    """Scenario 3: Fresh application startup and broad DB preload.

    First application startup with an empty database triggers the preloading
    pipeline for standard categories (cancer, diabetes, heart disease, neurological),
    populating ChromaDB with >=200 trials and updating the UI metric.
    """
    tf = _get_trial_fetcher()
    emb = _get_embedder()
    vs = _get_vector_store()
    sb = _get_sidebar()

    # Step 1: Verify clean initial state
    assert vs.get_trial_count() == 0

    # Step 2: Prepare 200 mock studies (50 per broad category)
    categories = ["cancer", "diabetes", "heart disease", "neurological"]
    mock_dataset = []
    for cat in categories:
        for i in range(50):
            mock_dataset.append(
                make_mock_study(
                    nct_id=f"NCT_{cat[:4].upper()}_{i:03d}",
                    brief_title=f"Clinical Study {i} for {cat.title()}",
                    conditions=[cat.title()],
                )
            )

    assert len(mock_dataset) == 200

    # Step 3: Trigger preload with mocked ClinicalTrials.gov API
    with patch("httpx.Client.get") as mock_get:
        def _get_category_studies(url, params=None, **kwargs):
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.is_success = True
            cond = (params or {}).get("query.cond", "")
            # Filter from mock_dataset by condition
            matched = [
                s for s in mock_dataset
                if cond.strip('"').lower() in s["protocolSection"]["conditionsModule"]["conditions"][0].lower()
            ]
            mock_resp.json.return_value = {"studies": matched if matched else mock_dataset[:50]}
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        mock_get.side_effect = _get_category_studies

        studies = tf.preload_standard_trials(categories=categories, page_size=200)

    assert len(studies) == 200

    # Step 4: Extract and embed all 200 studies
    ids, texts, metadatas = [], [], []
    for s in studies:
        t, m = tf.extract_trial(s)
        ids.append(m["nct_id"])
        texts.append(t)
        metadatas.append(m)

    embeddings = emb.embed_text(texts)
    assert len(embeddings) == 200

    # Step 5: Upsert into ChromaDB
    vs.upsert_trials(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # Step 6: Verify total count in ChromaDB
    total_count = vs.get_trial_count()
    assert total_count == 200

    # Step 7: Verify UI trial count metric
    if sb and hasattr(sb, "format_trial_count_metric"):
        label, val = sb.format_trial_count_metric(total_count)
        assert val == 200 or str(val) == "200"
    else:
        metric_display = f"Trials in DB: {total_count}"
        assert "200" in metric_display

    # Step 8: Idempotency check: Re-running preload does not inflate count
    vs.upsert_trials(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    assert vs.get_trial_count() == 200


def test_tier4_workload_offline_zero_key_resilience(temp_chroma_path, monkeypatch):
    """Scenario 4: Offline zero-key resilience and heuristic fallbacks.

    When GROQ_API_KEY is missing or invalid, the discovery engine continues
    to operate seamlessly: semantic search executes, trial cards render heuristic
    fallback summaries, and the Q&A panel displays clear configuration instructions.
    """
    emb = _get_embedder()
    vs = _get_vector_store()
    gc = _get_groq_client()

    # 1. Simulate unconfigured environment (no API key)
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # 2. Populate ChromaDB with sample trials
    sample_trials = [
        ("NCT_RHEUM_01", "Trial for severe rheumatoid arthritis with chronic joint swelling and morning stiffness.", "Phase 3", "RECRUITING"),
        ("NCT_LUPUS_01", "Trial for systemic lupus erythematosus with fatigue and joint pain.", "Phase 2", "RECRUITING"),
    ]
    vs.upsert_trials(
        ids=[t[0] for t in sample_trials],
        embeddings=[make_deterministic_vector(t[1]) for t in sample_trials],
        documents=[t[1] for t in sample_trials],
        metadatas=[{"phase": t[2], "status": t[3], "nct_id": t[0]} for t in sample_trials],
    )

    # 3. User searches with symptoms
    symptoms = ["severe joint pain", "morning stiffness"]
    query_str = " ".join(symptoms)

    # Vector search must succeed completely offline
    results = vs.search_trials(
        query_embedding=emb.embed_query(query_str),
        n_results=2,
    )
    assert len(results["documents"][0]) == 2
    top_doc = results["documents"][0][0]

    # 4. Summary generation: Must return graceful fallback without raising exceptions
    fallback_summary = gc.summarize_trial(top_doc)
    assert isinstance(fallback_summary, str)
    assert any(marker in fallback_summary.lower() for marker in ["eligibility", "criteria", "groq_api_key", ".env", "bullet"])

    # 5. Q&A Interaction: Must advise user on configuring API key
    qa_fallback = gc.ask_about_trials(
        question="How long does this study last?",
        trial_contexts=[top_doc],
        chat_history=[],
    )
    assert isinstance(qa_fallback, str)
    assert any(w in qa_fallback.lower() for w in ["groq_api_key", ".env", "configure", "set", "api key"])


def test_tier4_workload_multisymptom_rare_disease(temp_chroma_path):
    """Scenario 5: Multi-symptom rare disease discovery.

    User enters a complex multi-symptom cluster representing a rare systemic
    disorder (combining predefined chips and custom text entry), verifies
    cosine similarity aggregation, and applies recruitment status filtering.
    """
    tf = _get_trial_fetcher()
    emb = _get_embedder()
    vs = _get_vector_store()
    sb = _get_sidebar()

    # 1. Populate Corpus: Rare Autoimmune, Neurological, and Common Conditions
    studies = [
        # Rare Autoimmune Disorder (Target match)
        make_mock_study(
            nct_id="NCT_MG_001",
            brief_title="Efficacy of Monoclonal Therapy in Generalized Myasthenia Gravis",
            conditions=["Myasthenia Gravis", "Autoimmune Neuromuscular Disorder"],
            brief_summary="Targeted intervention for refractory generalized myasthenia gravis presenting with chronic fatigue, episodic muscle weakness, ptosis, and peripheral neuropathy.",
            phases=["PHASE3"],
            overall_status="RECRUITING",
        ),
        # Multiple Sclerosis
        make_mock_study(
            nct_id="NCT_MS_002",
            brief_title="B-Cell Depletion in Relapsing-Remitting Multiple Sclerosis",
            conditions=["Multiple Sclerosis"],
            brief_summary="Study assessing relapse rates and mobility impairment in MS.",
            phases=["PHASE3"],
            overall_status="RECRUITING",
        ),
        # Common Cardiology condition (Control)
        make_mock_study(
            nct_id="NCT_HTN_003",
            brief_title="Antihypertensive Combination Therapy in Essential Hypertension",
            conditions=["Hypertension", "High Blood Pressure"],
            brief_summary="Evaluating blood pressure lowering with dual ACE inhibitor and diuretic.",
            phases=["PHASE4"],
            overall_status="RECRUITING",
        ),
    ]

    for s in studies:
        t, m = tf.extract_trial(s)
        vs.upsert_trials(
            ids=[m["nct_id"]],
            embeddings=emb.embed_text([t]),
            documents=[t],
            metadatas=[m],
        )

    # 2. User Input: Predefined chips + custom text
    selected_chips = ["fatigue", "joint pain"]
    custom_entry = "episodic muscle weakness"

    if sb and hasattr(sb, "merge_symptoms"):
        combined_symptoms = sb.merge_symptoms(selected_chips, custom_entry)
    else:
        combined_symptoms = selected_chips + [custom_entry]

    assert "fatigue" in combined_symptoms
    assert "episodic muscle weakness" in combined_symptoms

    query_str = " ".join(combined_symptoms)

    # 3. Filters: Phase = "All", Status = "RECRUITING"
    selected_phase = "All"
    selected_status = "RECRUITING"

    # 4. Search and Ranking
    qvec = emb.embed_query(query_str)
    search_res = vs.search_trials(
        query_embedding=qvec,
        n_results=3,
        phase=selected_phase,
        status=selected_status,
    )

    returned_ids = search_res["ids"][0]
    returned_distances = search_res["distances"][0]

    # Verify ranking: Rare neuromuscular condition (Myasthenia Gravis) ranks higher than Hypertension
    assert returned_ids[0] in ("NCT_MG_001", "NCT_MS_002")
    # Hypertension must have a higher cosine distance than the neuromuscular disorder
    mg_idx = returned_ids.index("NCT_MG_001") if "NCT_MG_001" in returned_ids else -1
    htn_idx = returned_ids.index("NCT_HTN_003") if "NCT_HTN_003" in returned_ids else -1

    if mg_idx != -1 and htn_idx != -1:
        assert returned_distances[mg_idx] < returned_distances[htn_idx]


def test_tier4_workload_ingestion_stress_and_deduplication(temp_chroma_path):
    """Scenario 6: Ingestion stress and deduplication workflow.

    Simulates rapid repeated DB refresh cycles with overlapping study data:
    verifies upsert idempotency, zero duplicate NCT IDs, and continuous HNSW
    index integrity under high-volume retrieval.
    """
    emb = _get_embedder()
    vs = _get_vector_store()

    # Batch A: 100 studies (NCT_000 to NCT_099)
    batch_a_ids = [f"NCT_STRESS_{i:03d}" for i in range(100)]
    batch_a_docs = [f"Clinical Trial Document for condition {i % 10}" for i in range(100)]
    batch_a_metas = [{"phase": f"Phase {1 + (i % 4)}", "status": "RECRUITING", "nct_id": batch_a_ids[i]} for i in range(100)]
    batch_a_vecs = [make_deterministic_vector(doc) for doc in batch_a_docs]

    # Cycle 1: Initial Ingestion of 100 studies
    vs.upsert_trials(
        ids=batch_a_ids,
        embeddings=batch_a_vecs,
        documents=batch_a_docs,
        metadatas=batch_a_metas,
    )
    assert vs.get_trial_count() == 100

    # Batch B: Overlapping 100 studies with updated metadata + 20 brand new studies (120 total unique)
    batch_b_ids = [f"NCT_STRESS_{i:03d}" for i in range(120)]
    batch_b_docs = [f"Updated Clinical Trial Document for condition {i % 10}" for i in range(120)]
    batch_b_metas = [{"phase": f"Phase {1 + (i % 4)}", "status": "RECRUITING", "nct_id": batch_b_ids[i]} for i in range(120)]
    batch_b_vecs = [make_deterministic_vector(doc) for doc in batch_b_docs]

    # Cycle 2: Second Ingestion with overlapping batch
    vs.upsert_trials(
        ids=batch_b_ids,
        embeddings=batch_b_vecs,
        documents=batch_b_docs,
        metadatas=batch_b_metas,
    )

    # Verification: Total count must be exactly 120 (idempotent upsert, no duplication)
    final_count = vs.get_trial_count()
    assert final_count == 120

    # Cycle 3: Verify HNSW index integrity with 10 consecutive searches
    for query_id in range(10):
        test_query = f"Condition search {query_id}"
        qvec = make_deterministic_vector(test_query)
        results = vs.search_trials(query_embedding=qvec, n_results=5)

        assert len(results["ids"][0]) == 5
        assert len(results["distances"][0]) == 5
        # Verify cosine distances are monotonically increasing
        dists = results["distances"][0]
        for k in range(len(dists) - 1):
            assert dists[k] <= dists[k + 1] + 1e-5

    # Check individual record to ensure updated doc is reflected
    coll = vs.get_collection()
    sample_rec = coll.get(ids=["NCT_STRESS_005"], include=["documents"])
    assert "Updated Clinical Trial Document" in sample_rec["documents"][0]
