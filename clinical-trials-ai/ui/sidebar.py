"""Sidebar controls, symptom management, and database refresh for clinical-trials-ai."""

import html
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from core.vector_store import get_trial_count


COMMON_SYMPTOMS: List[str] = [
    "fatigue",
    "joint pain",
    "chest pain",
    "shortness of breath",
    "headache",
    "nausea",
    "fever",
    "depression",
    "anxiety",
    "memory loss",
    "weight loss",
    "vision problems",
    "neuropathy",
    "back pain",
    "cough",
    "insomnia",
]
DEFAULT_SYMPTOMS: List[str] = COMMON_SYMPTOMS

PHASE_OPTIONS: List[str] = [
    "All",
    "Early Phase 1",
    "Phase 1",
    "Phase 2",
    "Phase 3",
    "Phase 4",
    "Not Applicable",
]

STATUS_OPTIONS: List[str] = [
    "All",
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "COMPLETED",
    "SUSPENDED",
    "TERMINATED",
    "WITHDRAWN",
]

TOP_K_MIN: int = 5
TOP_K_MAX: int = 20
TOP_K_DEFAULT: int = 10

MAX_SYMPTOM_LENGTH: int = 200


def parse_custom_symptoms(raw: Optional[str]) -> List[str]:
    """Parse comma-separated custom symptom input string.

    Args:
        raw: Comma-separated string of custom symptoms.

    Returns:
        List of non-empty stripped symptom strings.
    """
    if not raw or not isinstance(raw, str):
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def sanitize_custom_symptom(raw: Any) -> Any:
    """Sanitize custom symptom string or list of symptoms.

    Caps length to MAX_SYMPTOM_LENGTH (200 characters) and strips whitespace.

    Args:
        raw: Single string or list of strings.

    Returns:
        Sanitized string or list of sanitized strings.
    """
    if isinstance(raw, list):
        return [str(s)[:MAX_SYMPTOM_LENGTH].strip() for s in raw if s]
    if isinstance(raw, str):
        return raw[:MAX_SYMPTOM_LENGTH].strip()
    return ""


def clean_symptoms(raw: List[str]) -> List[str]:
    """Clean and sanitize a list of symptom strings.

    Strips whitespace, removes blank items, and caps length to 200 characters.

    Args:
        raw: Raw list of symptom strings.

    Returns:
        Cleaned list of strings.
    """
    if not raw or not isinstance(raw, list):
        return []
    cleaned = []
    for item in raw:
        if item is not None:
            s = str(item).strip()
            if s:
                cleaned.append(s[:MAX_SYMPTOM_LENGTH])
    return cleaned


def merge_symptoms(
    selected: Optional[List[str]],
    custom: Optional[str],
) -> List[str]:
    """Combine selected symptoms from multi-select and custom text input.

    Splits custom input on commas, strips whitespace, caps each item to
    MAX_SYMPTOM_LENGTH, and deduplicates while preserving insertion order.

    Args:
        selected: List of symptoms chosen from predefined multi-select.
        custom: Free-text custom symptoms (optionally comma-separated).

    Returns:
        Deduplicated list of merged symptoms.
    """
    merged: List[str] = []
    seen: set = set()

    for item in (selected or []):
        if item is not None:
            clean_item = str(item).strip()[:MAX_SYMPTOM_LENGTH]
            if clean_item and clean_item.lower() not in seen:
                seen.add(clean_item.lower())
                merged.append(clean_item)

    if custom and isinstance(custom, str):
        for part in parse_custom_symptoms(custom):
            clean_part = part[:MAX_SYMPTOM_LENGTH].strip()
            if clean_part and clean_part.lower() not in seen:
                seen.add(clean_part.lower())
                merged.append(clean_part)

    return merged


def deduplicate_symptoms(
    selected: Optional[List[str]],
    custom: Optional[str],
) -> List[str]:
    """Deduplicate selected and custom symptoms (alias to merge_symptoms)."""
    return merge_symptoms(selected, custom)


def format_symptoms_query(symptoms: List[str]) -> str:
    """Format a list of symptoms into a single space-separated query string.

    Args:
        symptoms: List of symptom strings.

    Returns:
        Concatenated query string.
    """
    if not symptoms:
        return ""
    return " ".join([str(s).strip() for s in symptoms if s and str(s).strip()])


def is_valid_query(symptoms: List[str]) -> bool:
    """Validate whether the symptom list contains at least one non-empty symptom.

    Args:
        symptoms: List of symptom strings.

    Returns:
        True if at least one valid symptom exists, False otherwise.
    """
    return bool(symptoms and any(str(s).strip() for s in symptoms))


def clamp_top_k(val: int) -> int:
    """Clamp top_k value strictly to [TOP_K_MIN, TOP_K_MAX].

    Args:
        val: Requested top_k integer.

    Returns:
        Clamped integer within bounds [5, 20].
    """
    try:
        int_val = int(val)
        return max(TOP_K_MIN, min(TOP_K_MAX, int_val))
    except (ValueError, TypeError):
        return TOP_K_DEFAULT


def normalize_phase_filter(phase: Optional[str]) -> Optional[str]:
    """Normalize phase filter selection.

    Unrecognized phases or 'All' are converted to None.

    Args:
        phase: Selected phase string.

    Returns:
        Valid phase string or None.
    """
    valid_phases = {
        "Phase 1",
        "Phase 2",
        "Phase 3",
        "Phase 4",
        "Early Phase 1",
        "Not Applicable",
    }
    if phase and phase != "All" and phase in valid_phases:
        return phase
    return None


def normalize_status_filter(status: Optional[str]) -> Optional[str]:
    """Normalize recruitment status filter selection.

    Unrecognized statuses or 'All' are converted to None.

    Args:
        status: Selected recruitment status string.

    Returns:
        Valid status string or None.
    """
    valid_statuses = {
        "RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "ENROLLING_BY_INVITATION",
        "COMPLETED",
        "SUSPENDED",
        "TERMINATED",
        "WITHDRAWN",
    }
    if status and status != "All" and status in valid_statuses:
        return status
    return None


def normalize_filters(
    phase: Optional[str],
    status: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize phase and status filter selections for ChromaDB queries.

    Args:
        phase: Selected phase filter string (or 'All').
        status: Selected recruitment status filter string (or 'All').

    Returns:
        Tuple of (normalized_phase, normalized_status) where 'All' maps to None.
    """
    return normalize_phase_filter(phase), normalize_status_filter(status)


def format_count(count: int) -> str:
    """Format integer count with thousands comma separator.

    Args:
        count: Integer count.

    Returns:
        Comma-separated string representation (e.g. '1,000').
    """
    try:
        return f"{int(count):,}"
    except (ValueError, TypeError):
        return "0"


def format_trial_count_metric(count: int) -> Tuple[str, str]:
    """Format trial count for Streamlit sidebar metric.

    Args:
        count: Current indexed trial count.

    Returns:
        Tuple of (label, formatted_value).
    """
    return "Indexed Trials in DB", format_count(count)


def format_refresh_success(count: int) -> str:
    """Format database refresh success notification message.

    Args:
        count: Total number of trials indexed after refresh.

    Returns:
        Formatted success message string.
    """
    return f"Successfully refreshed database! Total trials: {format_count(count)}"


def refresh_database() -> int:
    """Fetch, embed, and index standard trials into local ChromaDB.

    Queries ClinicalTrials.gov API for standard condition categories,
    extracts structured embeddable text and metadata, computes dense
    SentenceTransformer embeddings, and upserts them into ChromaDB.

    Returns:
        Updated total count of indexed trials.
    """
    from core.trial_fetcher import preload_standard_trials, extract_trial
    from core.embedder import embed_text
    from core.vector_store import upsert_trials, get_trial_count

    studies = preload_standard_trials(page_size=200)
    if studies:
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for study in studies:
            doc_text, meta = extract_trial(study)
            nct_id = meta.get("nct_id")
            if nct_id:
                ids.append(nct_id)
                documents.append(doc_text)
                metadatas.append(meta)

        if ids and documents:
            embeddings = embed_text(documents)
            upsert_trials(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    return get_trial_count()


def trigger_refresh() -> int:
    """Trigger DB refresh pipeline (alias to refresh_database)."""
    return refresh_database()


def render_sidebar() -> Dict[str, Any]:
    """Render Streamlit sidebar controls for symptom selection and filtering.

    Renders:
    - App title & branding
    - Multi-symptom selector + custom symptom input
    - Phase and Recruitment Status filter dropdowns
    - Top-K result slider
    - 'Find Matching Trials' submit button
    - Database trial count metric and 'Refresh Trials DB' action

    Returns:
        Dictionary containing:
        - 'symptoms': List of merged, cleaned symptom strings
        - 'phase': Normalized phase filter (or None for 'All')
        - 'status': Normalized status filter (or None for 'All')
        - 'top_k': Selected number of top results (clamped 5-20)
        - 'find_clicked': Boolean indicating if 'Find Matching Trials' was clicked
    """
    with st.sidebar:
        st.markdown("## 🧬 Clinical Trials AI")
        st.caption("Intelligent Trial Discovery & Symptom Matcher")
        st.divider()

        # Multi-symptom selector
        selected_chips = st.multiselect(
            "Select Symptoms",
            options=COMMON_SYMPTOMS,
            default=[],
            help="Choose symptoms commonly associated with clinical trials.",
        )

        # Custom symptom input
        custom_input = st.text_input(
            "Or enter custom symptoms (comma-separated):",
            placeholder="e.g. morning stiffness, neuropathy",
            help="Type custom symptoms separated by commas.",
        )

        merged_symptoms = merge_symptoms(selected_chips, custom_input)

        st.divider()

        # Filter: Trial Phase
        selected_phase = st.selectbox(
            "Filter by Phase",
            options=PHASE_OPTIONS,
            index=0,
            help="Filter studies by development phase.",
        )

        # Filter: Recruitment Status (Default: RECRUITING)
        default_status_idx = (
            STATUS_OPTIONS.index("RECRUITING")
            if "RECRUITING" in STATUS_OPTIONS
            else 0
        )
        selected_status = st.selectbox(
            "Filter by Status",
            options=STATUS_OPTIONS,
            index=default_status_idx,
            help="Filter studies by current clinical recruitment status.",
        )

        # Top-K slider
        top_k = st.slider(
            "Max Results",
            min_value=TOP_K_MIN,
            max_value=TOP_K_MAX,
            value=TOP_K_DEFAULT,
            step=1,
            help="Maximum number of top-matching trials to display.",
        )

        # Find Trials button
        find_clicked = st.button(
            "🔍 Find Matching Trials",
            type="primary",
            use_container_width=True,
        )

        st.divider()

        # Database Management Section
        st.markdown("### 🗄️ Database Status")
        current_count = get_trial_count()
        label, formatted_val = format_trial_count_metric(current_count)
        st.metric(label, formatted_val)

        if st.button("🔄 Refresh Trials DB", use_container_width=True):
            with st.spinner("Fetching, embedding, and indexing standard trials..."):
                try:
                    new_count = refresh_database()
                    st.success(format_refresh_success(new_count))
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to refresh trials database: {e}")

        norm_phase, norm_status = normalize_filters(selected_phase, selected_status)

        return {
            "symptoms": merged_symptoms,
            "phase": norm_phase,
            "status": norm_status,
            "top_k": clamp_top_k(top_k),
            "find_clicked": find_clicked,
        }
