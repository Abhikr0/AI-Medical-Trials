"""UI reusable components and result card renderers for clinical-trials-ai."""

import html
from typing import Any, Dict, Optional
import streamlit as st


COMMON_SYMPTOMS = [
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
DEFAULT_SYMPTOMS = COMMON_SYMPTOMS


def calculate_match_percentage(distance: float) -> float:
    """Calculate percentage similarity from cosine distance.

    Formula: round(max(0.0, min(100.0, (1.0 - distance) * 100.0)), 1).
    Clamps strictly between 0.0 and 100.0.

    Args:
        distance: Cosine distance from vector search.

    Returns:
        Similarity percentage as a float between 0.0 and 100.0.
    """
    try:
        dist_float = float(distance)
        pct = (1.0 - dist_float) * 100.0
        return round(max(0.0, min(100.0, pct)), 1)
    except (ValueError, TypeError):
        return 0.0


def get_study_url(nct_id: Optional[str]) -> Optional[str]:
    """Generate external ClinicalTrials.gov study URL.

    Args:
        nct_id: NCT Identifier (e.g. 'NCT04567890').

    Returns:
        External study URL or None if missing or empty.
    """
    if not nct_id or not str(nct_id).strip():
        return None
    return f"https://clinicaltrials.gov/study/{str(nct_id).strip()}"


def get_status_badge(status: Optional[str]) -> str:
    """Format status badge text.

    Args:
        status: Trial recruitment status string.

    Returns:
        Formatted status badge string.
    """
    st_val = str(status).strip() if status else "UNKNOWN"
    return st_val


def get_phase_caption(phase: Optional[str]) -> str:
    """Format phase caption text.

    Args:
        phase: Trial phase string.

    Returns:
        Formatted phase caption.
    """
    phase_val = str(phase).strip() if phase else "Not Specified"
    return f"Phase: {phase_val}"


def should_expand_card(index: int) -> bool:
    """Determine whether a result card should be auto-expanded by default.

    Args:
        index: Zero-based position index of the result card.

    Returns:
        True if first item (index 0), False otherwise.
    """
    return index == 0


def format_card_title(meta: Optional[Dict[str, Any]]) -> str:
    """Format study title with fallback if title is missing.

    Args:
        meta: Study metadata dictionary.

    Returns:
        Clean human-readable title string.
    """
    if not meta or not isinstance(meta, dict):
        return "Clinical Trial"
    title = meta.get("title") or meta.get("brief_title")
    if title and str(title).strip():
        return str(title).strip()
    nct_id = meta.get("nct_id")
    if nct_id and str(nct_id).strip():
        return f"Clinical Trial ({str(nct_id).strip()})"
    return "Clinical Trial"


def render_empty_results_message() -> None:
    """Display an informative message when search returns no matching trials."""
    st.info("ℹ️ No matching clinical trials found. Try adjusting your symptoms or filters.")


def render_trial_card(
    doc: str,
    meta: Dict[str, Any],
    distance: float,
    index: int = 0,
) -> None:
    """Render a rich Streamlit trial result card using st.container().

    Displays:
    - Match score badge with color coding (Green >=75%, Orange 50-74.9%, Blue <50%)
    - Recruitment Status badge and Phase tag
    - NCT ID with external link to clinicaltrials.gov
    - Brief title and study conditions
    - Expandable AI Plain-English Summary backed by session cache

    Args:
        doc: Searchable document text.
        meta: Sanitized trial metadata dictionary.
        distance: Cosine distance from query embedding.
        index: Result rank index.
    """
    match_pct = calculate_match_percentage(distance)
    nct_id = str(meta.get("nct_id", "")).strip() if meta else ""
    title = format_card_title(meta)
    status = meta.get("status", "UNKNOWN") if meta else "UNKNOWN"
    phase = meta.get("phase", "Not Specified") if meta else "Not Specified"
    conditions = meta.get("conditions", "") if meta else ""
    locations = meta.get("locations", "") if meta else ""
    study_url = get_study_url(nct_id)

    # Color coding for match percentage
    if match_pct >= 75.0:
        match_badge = f":green[**{match_pct}% Match**]"
    elif match_pct >= 50.0:
        match_badge = f":orange[**{match_pct}% Match**]"
    else:
        match_badge = f":blue[**{match_pct}% Match**]"

    with st.container(border=True):
        col_title, col_match = st.columns([4, 1])
        with col_title:
            if study_url:
                st.markdown(f"#### [{title}]({study_url})")
            else:
                st.markdown(f"#### {title}")
        with col_match:
            st.markdown(match_badge)

        # Meta tags bar
        tag_items = []
        if nct_id:
            if study_url:
                tag_items.append(f"🆔 [{nct_id}]({study_url})")
            else:
                tag_items.append(f"🆔 {nct_id}")
        if status:
            tag_items.append(f"🏷️ **Status:** `{get_status_badge(status)}`")
        if phase:
            tag_items.append(f"🔬 **{get_phase_caption(phase)}**")

        st.caption(" • ".join(tag_items))

        if conditions:
            st.markdown(f"**Conditions:** {conditions}")

        if locations:
            st.caption(f"📍 **Locations:** {locations}")

        # Expandable AI Plain-English Summary
        is_expanded = should_expand_card(index)
        with st.expander("🤖 AI Plain-English Summary", expanded=is_expanded):
            if "summaries_cache" not in st.session_state:
                st.session_state["summaries_cache"] = {}

            if nct_id:
                if nct_id not in st.session_state["summaries_cache"]:
                    with st.spinner("Generating AI plain-English summary..."):
                        try:
                            from core.groq_client import summarize_trial
                            summary_text = summarize_trial(doc)
                        except Exception as e:
                            summary_text = f"Summary temporarily unavailable: {e}"
                        st.session_state["summaries_cache"][nct_id] = summary_text

                st.markdown(st.session_state["summaries_cache"][nct_id])
            else:
                # If no NCT ID, summarize doc directly without caching by NCT
                with st.spinner("Generating AI summary..."):
                    try:
                        from core.groq_client import summarize_trial
                        st.markdown(summarize_trial(doc))
                    except Exception as e:
                        st.warning(f"Summary temporarily unavailable: {e}")
