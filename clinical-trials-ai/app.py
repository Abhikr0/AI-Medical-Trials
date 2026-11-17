"""Streamlit main entry point for AI Clinical Trials Discovery App."""

from typing import Any, Dict, List, Optional
import streamlit as st

from core.embedder import embed_query
from core.vector_store import get_trial_count, search_trials
from ui.chat import render_chat_panel
from ui.components import render_empty_results_message, render_trial_card
from ui.sidebar import refresh_database, render_sidebar


def refresh_trials_pipeline() -> int:
    """Trigger trial ingestion, embedding, and vector storage pipeline.

    Returns:
        Updated total number of indexed trials.
    """
    return refresh_database()


def init_app_state() -> None:
    """Initialize persistent Streamlit session state keys."""
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "summaries_cache" not in st.session_state:
        st.session_state["summaries_cache"] = {}
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = None


def main() -> None:
    """Main application execution routine."""
    st.set_page_config(
        page_title="AI Clinical Trials Discovery",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_app_state()

    # Render sidebar controls
    sidebar_state = render_sidebar()

    # Application Header Banner
    st.title("🧬 AI Clinical Trials Discovery")
    st.caption(
        "Match symptoms semantically to clinical trials with AI summaries and grounded Q&A."
    )

    # Check trial database population status
    trial_count = get_trial_count()
    if trial_count == 0:
        st.warning(
            "⚠️ The local trial database is currently empty. "
            "Click **'🔄 Refresh Trials DB'** in the sidebar to populate trials."
        )

    # Handle Search Submission
    if sidebar_state["find_clicked"]:
        symptoms = sidebar_state["symptoms"]
        if not symptoms:
            st.info("ℹ️ Please select or enter at least one symptom to search.")
        else:
            query_text = " ".join(symptoms)
            with st.spinner("Searching matching clinical trials..."):
                try:
                    query_embedding = embed_query(query_text)
                    results = search_trials(
                        query_embedding=query_embedding,
                        n_results=sidebar_state["top_k"],
                        phase=sidebar_state["phase"],
                        status=sidebar_state["status"],
                    )
                    st.session_state["search_results"] = results

                    # Clear chat history when new search is executed
                    if st.session_state.get("last_query") != query_text:
                        st.session_state["chat_history"] = []
                        st.session_state["last_query"] = query_text
                except Exception as e:
                    st.error(f"Search failed: {e}")

    # Two-Column Layout: Left for Result Cards, Right for Interactive Chat
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        search_results = st.session_state.get("search_results")
        if search_results is not None:
            raw_docs = search_results.get("documents", [[]])
            raw_metas = search_results.get("metadatas", [[]])
            raw_dists = search_results.get("distances", [[]])

            docs = raw_docs[0] if raw_docs else []
            metas = raw_metas[0] if raw_metas else []
            distances = raw_dists[0] if raw_dists else []

            if not docs:
                render_empty_results_message()
            else:
                st.subheader(f"🎯 Matching Clinical Trials ({len(docs)} found)")
                for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
                    render_trial_card(doc=doc, meta=meta, distance=dist, index=i)
        elif trial_count > 0:
            st.markdown("### 🔍 How to Discover Clinical Trials")
            st.markdown(
                """
                1. **Select or enter symptoms** in the sidebar on the left.
                2. Optionally refine results by **Phase** or **Recruitment Status**.
                3. Click **🔍 Find Matching Trials** to semantically match your symptoms to trial criteria.
                4. Inspect matched trials, expand **🤖 AI Plain-English Summaries**, or ask questions in the chat panel on the right.
                """
            )

    with col_right:
        search_results = st.session_state.get("search_results")
        trial_docs: List[str] = []
        if search_results:
            raw_docs = search_results.get("documents", [[]])
            if raw_docs and raw_docs[0]:
                trial_docs = raw_docs[0]

        render_chat_panel(trial_docs)


if __name__ == "__main__":
    main()
