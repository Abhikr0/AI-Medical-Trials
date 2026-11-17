"""Unit tests for Streamlit UI helper functions, state management, and component logic."""

import html
import py_compile
from unittest.mock import MagicMock, patch
import pytest

from ui.components import (
    COMMON_SYMPTOMS,
    DEFAULT_SYMPTOMS,
    calculate_match_percentage,
    format_card_title,
    get_phase_caption,
    get_status_badge,
    get_study_url,
    render_empty_results_message,
    render_trial_card,
    should_expand_card,
)
from ui.sidebar import (
    PHASE_OPTIONS,
    STATUS_OPTIONS,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    TOP_K_MIN,
    clamp_top_k,
    clean_symptoms,
    deduplicate_symptoms,
    format_count,
    format_refresh_success,
    format_symptoms_query,
    format_trial_count_metric,
    is_valid_query,
    merge_symptoms,
    normalize_filters,
    normalize_phase_filter,
    normalize_status_filter,
    parse_custom_symptoms,
    refresh_database,
    render_sidebar,
    sanitize_custom_symptom,
    trigger_refresh,
)
from ui.chat import (
    append_assistant_message,
    append_user_message,
    can_submit_chat,
    clear_chat_history,
    ensure_chat_state,
    format_chat_error,
    init_chat_history,
    render_chat_messages,
    render_chat_panel,
    sanitize_chat_input,
)
import app


# ============================================================================
# 1. Match Percentage Calculation & Result Card Helpers
# ============================================================================

class TestMatchPercentageCalculation:
    """Tests for cosine distance to similarity percentage math."""

    def test_distance_zero_is_100(self):
        assert calculate_match_percentage(0.0) == 100.0

    def test_distance_one_is_zero(self):
        assert calculate_match_percentage(1.0) == 0.0

    def test_fractional_distance(self):
        assert calculate_match_percentage(0.12) == 88.0
        assert calculate_match_percentage(0.25) == 75.0
        assert calculate_match_percentage(0.5) == 50.0

    def test_clamping_above_one(self):
        assert calculate_match_percentage(1.5) == 0.0
        assert calculate_match_percentage(2.0) == 0.0

    def test_clamping_below_zero(self):
        assert calculate_match_percentage(-0.5) == 100.0
        assert calculate_match_percentage(-1.0) == 100.0

    def test_invalid_type_fallback(self):
        assert calculate_match_percentage("invalid") == 0.0
        assert calculate_match_percentage(None) == 0.0


class TestCardDisplayHelpers:
    """Tests for result card formatting, titles, badges, and study URLs."""

    def test_get_study_url_valid(self):
        assert (
            get_study_url("NCT01234567")
            == "https://clinicaltrials.gov/study/NCT01234567"
        )
        assert (
            get_study_url("  NCT04567890  ")
            == "https://clinicaltrials.gov/study/NCT04567890"
        )

    def test_get_study_url_missing(self):
        assert get_study_url("") is None
        assert get_study_url("   ") is None
        assert get_study_url(None) is None

    def test_get_status_badge(self):
        assert get_status_badge("RECRUITING") == "RECRUITING"
        assert get_status_badge("COMPLETED") == "COMPLETED"
        assert get_status_badge("") == "UNKNOWN"
        assert get_status_badge(None) == "UNKNOWN"

    def test_get_phase_caption(self):
        assert get_phase_caption("Phase 2") == "Phase: Phase 2"
        assert get_phase_caption("Phase 1, Phase 2") == "Phase: Phase 1, Phase 2"
        assert get_phase_caption("") == "Phase: Not Specified"
        assert get_phase_caption(None) == "Phase: Not Specified"

    def test_should_expand_card(self):
        assert should_expand_card(0) is True
        assert should_expand_card(1) is False
        assert should_expand_card(2) is False

    def test_format_card_title_with_title(self):
        meta = {"title": "Immunotherapy for Solid Tumors", "nct_id": "NCT123"}
        assert format_card_title(meta) == "Immunotherapy for Solid Tumors"

    def test_format_card_title_with_brief_title(self):
        meta = {"brief_title": "Diabetes Prevention Trial", "nct_id": "NCT456"}
        assert format_card_title(meta) == "Diabetes Prevention Trial"

    def test_format_card_title_fallback_to_nct_id(self):
        meta = {"nct_id": "NCT789"}
        assert format_card_title(meta) == "Clinical Trial (NCT789)"

    def test_format_card_title_empty_fallback(self):
        assert format_card_title({}) == "Clinical Trial"
        assert format_card_title(None) == "Clinical Trial"


# ============================================================================
# 2. Symptom Parsing, Sanitization, and Merging
# ============================================================================

class TestSymptomProcessing:
    """Tests for multi-symptom merging, whitespace stripping, and deduplication."""

    def test_parse_custom_symptoms(self):
        assert parse_custom_symptoms("fever, chills, body ache") == [
            "fever",
            "chills",
            "body ache",
        ]
        assert parse_custom_symptoms("  headache ,  , fatigue ") == [
            "headache",
            "fatigue",
        ]
        assert parse_custom_symptoms("") == []
        assert parse_custom_symptoms(None) == []

    def test_sanitize_custom_symptom(self):
        huge_str = "a" * 300
        sanitized = sanitize_custom_symptom(huge_str)
        assert len(sanitized) == 200

        huge_list = ["b" * 250, "c" * 50]
        sanitized_list = sanitize_custom_symptom(huge_list)
        assert len(sanitized_list[0]) == 200
        assert len(sanitized_list[1]) == 50

    def test_clean_symptoms(self):
        raw = ["fatigue", "   ", "", "headache"]
        assert clean_symptoms(raw) == ["fatigue", "headache"]
        assert clean_symptoms([]) == []
        assert clean_symptoms(None) == []

    def test_merge_symptoms_both_present(self):
        selected = ["fatigue", "joint pain"]
        custom = "morning stiffness, fever"
        merged = merge_symptoms(selected, custom)
        assert merged == ["fatigue", "joint pain", "morning stiffness", "fever"]

    def test_merge_symptoms_selected_only(self):
        selected = ["fatigue", "nausea"]
        assert merge_symptoms(selected, None) == ["fatigue", "nausea"]
        assert merge_symptoms(selected, "") == ["fatigue", "nausea"]

    def test_merge_symptoms_custom_only(self):
        custom = "back pain, insomnia"
        assert merge_symptoms([], custom) == ["back pain", "insomnia"]
        assert merge_symptoms(None, custom) == ["back pain", "insomnia"]

    def test_merge_symptoms_deduplication(self):
        selected = ["fatigue", "headache"]
        custom = "fatigue, HEADACHE, nausea"
        merged = merge_symptoms(selected, custom)
        assert merged == ["fatigue", "headache", "nausea"]
        assert deduplicate_symptoms(selected, custom) == merged

    def test_format_symptoms_query(self):
        assert (
            format_symptoms_query(["fatigue", "joint pain"])
            == "fatigue joint pain"
        )
        assert format_symptoms_query(["fatigue", "  ", "cough"]) == "fatigue cough"
        assert format_symptoms_query([]) == ""

    def test_is_valid_query(self):
        assert is_valid_query(["fatigue"]) is True
        assert is_valid_query([]) is False
        assert is_valid_query(["   ", ""]) is False


# ============================================================================
# 3. Sidebar Filtering, Top-K, and Metrics
# ============================================================================

class TestSidebarFiltersAndMetrics:
    """Tests for filter normalization, metric formatting, and slider bounds."""

    def test_clamp_top_k(self):
        assert clamp_top_k(10) == 10
        assert clamp_top_k(5) == 5
        assert clamp_top_k(20) == 20
        assert clamp_top_k(0) == 5
        assert clamp_top_k(-5) == 5
        assert clamp_top_k(100) == 20
        assert clamp_top_k("invalid") == TOP_K_DEFAULT

    def test_normalize_phase_filter(self):
        assert normalize_phase_filter("Phase 1") == "Phase 1"
        assert normalize_phase_filter("Phase 3") == "Phase 3"
        assert normalize_phase_filter("All") is None
        assert normalize_phase_filter("Phase 99") is None
        assert normalize_phase_filter(None) is None

    def test_normalize_status_filter(self):
        assert normalize_status_filter("RECRUITING") == "RECRUITING"
        assert normalize_status_filter("COMPLETED") == "COMPLETED"
        assert normalize_status_filter("All") is None
        assert normalize_status_filter("UNKNOWN_STATUS") is None
        assert normalize_status_filter(None) is None

    def test_normalize_filters(self):
        assert normalize_filters("Phase 2", "RECRUITING") == ("Phase 2", "RECRUITING")
        assert normalize_filters("All", "All") == (None, None)
        assert normalize_filters("InvalidPhase", "InvalidStatus") == (None, None)

    def test_format_count(self):
        assert format_count(0) == "0"
        assert format_count(42) == "42"
        assert format_count(1000) == "1,000"
        assert format_count(1000000) == "1,000,000"
        assert format_count("invalid") == "0"

    def test_format_trial_count_metric(self):
        label, val = format_trial_count_metric(42)
        assert label == "Indexed Trials in DB"
        assert val == "42"

    def test_format_refresh_success(self):
        msg = format_refresh_success(200)
        assert "200" in msg
        assert "Successfully refreshed database" in msg


# ============================================================================
# 4. Interactive Chat Helpers & Session Management
# ============================================================================

class TestChatPanelHelpers:
    """Tests for chat history state, message appending, sanitization, and errors."""

    def test_sanitize_chat_input(self):
        raw = "<script>alert('xss')</script> Hello"
        sanitized = sanitize_chat_input(raw)
        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized
        assert sanitize_chat_input("") == ""
        assert sanitize_chat_input(None) == ""

    def test_init_chat_history(self):
        session = {}
        history = init_chat_history(session)
        assert history == []
        assert session["chat_history"] == []
        assert ensure_chat_state(session) == []

    def test_append_user_and_assistant_messages(self):
        history = []
        append_user_message(history, "What are the trial phases?")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What are the trial phases?"

        append_assistant_message(history, "These trials span Phase 1 through 3.")
        assert len(history) == 2
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "These trials span Phase 1 through 3."

    def test_clear_chat_history(self):
        session = {
            "chat_history": [{"role": "user", "content": "Question"}]
        }
        clear_chat_history(session)
        assert session["chat_history"] == []

    def test_can_submit_chat(self):
        assert can_submit_chat(["Doc 1", "Doc 2"]) is True
        assert can_submit_chat([]) is False
        assert can_submit_chat(None) is False

    def test_format_chat_error(self):
        err = Exception("Rate limit exceeded")
        msg = format_chat_error(err)
        assert "Rate limit exceeded" in msg
        assert "AI Assistant temporarily unavailable" in msg


# ============================================================================
# 5. UI Rendering & Integration Verification (Mocked Streamlit)
# ============================================================================

class TestUIRenderingWithMocks:
    """Tests UI components rendering with mocked streamlit calls."""

    @patch("streamlit.container")
    @patch("streamlit.columns")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    @patch("streamlit.expander")
    def test_render_trial_card_with_cache(
        self,
        mock_expander,
        mock_caption,
        mock_markdown,
        mock_columns,
        mock_container,
    ):
        mock_columns.return_value = [MagicMock(), MagicMock()]
        session_cache = {"summaries_cache": {"NCT001": "• Summary point 1\n• Summary point 2"}}

        with patch("streamlit.session_state", session_cache):
            meta = {
                "nct_id": "NCT001",
                "title": "Oncology Phase 2 Study",
                "status": "RECRUITING",
                "phase": "Phase 2",
                "conditions": "Melanoma",
            }
            render_trial_card("Study Document Text", meta=meta, distance=0.15, index=0)

            assert mock_container.called
            assert mock_expander.called
            # Summary should be loaded from cache without calling LLM
            mock_markdown.assert_any_call("• Summary point 1\n• Summary point 2")

    @patch("streamlit.container")
    @patch("streamlit.columns")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    @patch("streamlit.expander")
    @patch("streamlit.spinner")
    def test_render_trial_card_cache_miss_calls_groq(
        self,
        mock_spinner,
        mock_expander,
        mock_caption,
        mock_markdown,
        mock_columns,
        mock_container,
    ):
        mock_columns.return_value = [MagicMock(), MagicMock()]
        session_cache = {"summaries_cache": {}}

        with patch("streamlit.session_state", session_cache):
            with patch("core.groq_client.summarize_trial") as mock_summarize:
                mock_summarize.return_value = "• Newly generated summary"
                meta = {
                    "nct_id": "NCT002",
                    "title": "Cardiology Trial",
                    "status": "RECRUITING",
                    "phase": "Phase 3",
                }
                render_trial_card("Cardiology Doc", meta=meta, distance=0.20, index=1)

                assert mock_summarize.called
                assert session_cache["summaries_cache"]["NCT002"] == "• Newly generated summary"

    @patch("streamlit.sidebar")
    @patch("streamlit.multiselect")
    @patch("streamlit.text_input")
    @patch("streamlit.selectbox")
    @patch("streamlit.slider")
    @patch("streamlit.button")
    @patch("streamlit.metric")
    @patch("core.vector_store.get_trial_count", return_value=50)
    def test_render_sidebar_returns_expected_structure(
        self,
        mock_count,
        mock_metric,
        mock_btn,
        mock_slider,
        mock_sel,
        mock_txt,
        mock_multi,
        mock_sb,
    ):
        mock_multi.return_value = ["fatigue"]
        mock_txt.return_value = "fever"
        mock_sel.side_effect = ["Phase 2", "RECRUITING"]
        mock_slider.return_value = 10
        mock_btn.side_effect = [True, False]

        state = render_sidebar()

        assert "fatigue" in state["symptoms"]
        assert "fever" in state["symptoms"]
        assert state["phase"] == "Phase 2"
        assert state["status"] == "RECRUITING"
        assert state["top_k"] == 10
        assert state["find_clicked"] is True

    @patch("streamlit.chat_message")
    @patch("streamlit.markdown")
    def test_render_chat_messages(self, mock_markdown, mock_chat_message):
        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
        ]
        render_chat_messages(history)
        assert mock_chat_message.call_count == 2
        mock_markdown.assert_any_call("Question 1")
        mock_markdown.assert_any_call("Answer 1")

    @patch("core.trial_fetcher.preload_standard_trials")
    @patch("core.trial_fetcher.extract_trial")
    @patch("core.embedder.embed_text")
    @patch("core.vector_store.upsert_trials")
    @patch("core.vector_store.get_trial_count", return_value=100)
    def test_refresh_database_pipeline(
        self,
        mock_count,
        mock_upsert,
        mock_embed,
        mock_extract,
        mock_preload,
    ):
        mock_preload.return_value = [{"raw": "study_1"}]
        mock_extract.return_value = (
            "Doc text",
            {"nct_id": "NCT001", "phase": "Phase 1", "status": "RECRUITING"},
        )
        mock_embed.return_value = [[0.1] * 384]

        count = refresh_database()

        assert mock_preload.called
        assert mock_extract.called
        assert mock_embed.called
        assert mock_upsert.called
        assert count == 100
        assert trigger_refresh() == 100
        assert app.refresh_trials_pipeline() == 100


# ============================================================================
# 6. Syntax Compilation Verification
# ============================================================================

def test_syntax_compilation_all_ui_files():
    """Verify that all UI modules and test files compile without syntax errors."""
    files_to_compile = [
        "app.py",
        "ui/components.py",
        "ui/sidebar.py",
        "ui/chat.py",
        "tests/test_ui_helpers.py",
    ]
    for file_path in files_to_compile:
        py_compile.compile(file_path, doraise=True)
