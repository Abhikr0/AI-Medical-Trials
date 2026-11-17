"""Unit tests for core.groq_client module.

100% hermetic and offline using unittest.mock.
Verifies lazy client initialization, summarization with 3-bullet plain-English output,
grounded Q&A against retrieved contexts, and graceful fallbacks for missing keys and API errors.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import AuthenticationError, RateLimitError

import config
from core.groq_client import (
    DEFAULT_MODEL,
    FALLBACK_QA_MESSAGE,
    FALLBACK_SUMMARY,
    MAX_SUMMARY_INPUT_CHARS,
    ask_about_trials,
    get_groq_client,
    summarize_trial,
)


def _make_auth_error(message: str = "Invalid API key") -> AuthenticationError:
    """Create a mock groq.AuthenticationError with dummy HTTP response."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return AuthenticationError(message, response=response, body=None)


def _make_rate_limit_error(message: str = "Rate limit reached") -> RateLimitError:
    """Create a mock groq.RateLimitError with dummy HTTP response."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(message, response=response, body=None)


# ============================================================================
# 1. get_groq_client Tests
# ============================================================================


def test_get_groq_client_valid_key(monkeypatch):
    """Verify get_groq_client instantiates Groq when GROQ_API_KEY is valid."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_valid_key_12345")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with patch("core.groq_client.Groq") as mock_groq_cls:
        client = get_groq_client()
        assert client is not None
        mock_groq_cls.assert_called_once_with(api_key="gsk_valid_key_12345")


def test_get_groq_client_none_key(monkeypatch):
    """Verify get_groq_client returns None without error when key is None."""
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    client = get_groq_client()
    assert client is None


def test_get_groq_client_empty_key(monkeypatch):
    """Verify get_groq_client returns None when key is empty string."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    client = get_groq_client()
    assert client is None


def test_get_groq_client_whitespace_key(monkeypatch):
    """Verify get_groq_client returns None when key is pure whitespace."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "   \t\n  ")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    client = get_groq_client()
    assert client is None


def test_get_groq_client_placeholder_key(monkeypatch):
    """Verify get_groq_client returns None when key starts with placeholder 'your_'."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "your_groq_api_key_here")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    client = get_groq_client()
    assert client is None


def test_get_groq_client_env_fallback(monkeypatch):
    """Verify get_groq_client falls back to os.environ when config.GROQ_API_KEY is unset."""
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env_fallback_key")

    with patch("core.groq_client.Groq") as mock_groq_cls:
        client = get_groq_client()
        assert client is not None
        mock_groq_cls.assert_called_once_with(api_key="gsk_env_fallback_key")


def test_get_groq_client_instantiation_exception(monkeypatch):
    """Verify get_groq_client safely handles instantiation exceptions."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test_key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with patch("core.groq_client.Groq", side_effect=Exception("Initialization failed")):
        client = get_groq_client()
        assert client is None


# ============================================================================
# 2. summarize_trial Tests
# ============================================================================


def test_summarize_trial_success(monkeypatch):
    """Verify summarize_trial returns 3-bullet plain-English patient summary."""
    expected_summary = (
        "• Eligibility: Adults 18+ with confirmed NSCLC.\n"
        "• What it tests: Safety and efficacy of Pembrolizumab combination.\n"
        "• Key criteria: Measurable disease, no prior immunotherapy."
    )
    mock_choice = MagicMock()
    mock_choice.message.content = expected_summary
    mock_response = MagicMock(choices=[mock_choice])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = summarize_trial("Trial detail: Phase 3 study of Pembrolizumab in NSCLC patients.")

    assert result == expected_summary
    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "qwen/qwen3.8-27b"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 300
    assert len(kwargs["messages"]) == 2
    assert kwargs["messages"][0]["role"] == "system"
    assert "bullet" in kwargs["messages"][0]["content"].lower() or "summar" in kwargs["messages"][0]["content"].lower()
    assert kwargs["messages"][1]["role"] == "user"


def test_summarize_trial_missing_key_fallback(monkeypatch):
    """Verify summarize_trial returns structured fallback summary when client is None."""
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: None)

    summary = summarize_trial("Trial detail text")
    assert summary == FALLBACK_SUMMARY
    assert "Eligibility: See full criteria on study page." in summary
    assert "Configure GROQ_API_KEY in .env" in summary


def test_summarize_trial_empty_input(monkeypatch):
    """Verify summarize_trial returns fallback when trial_text is empty or whitespace."""
    mock_client = MagicMock()
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    assert summarize_trial("") == FALLBACK_SUMMARY
    assert summarize_trial("   \n\t  ") == FALLBACK_SUMMARY
    assert not mock_client.chat.completions.create.called


def test_summarize_trial_non_string_input(monkeypatch):
    """Verify summarize_trial returns fallback summary when non-string inputs are provided."""
    mock_client = MagicMock()
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    for bad_input in (123, 45.67, ["trial", "text"], {"text": "trial"}, None, True, object()):
        assert summarize_trial(bad_input) == FALLBACK_SUMMARY

    assert not mock_client.chat.completions.create.called


def test_summarize_trial_auth_error(monkeypatch):
    """Verify summarize_trial handles AuthenticationError with user-friendly message."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_auth_error()
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = summarize_trial("Trial detail text")
    assert "⚠️ Groq API key is invalid" in result
    assert "GROQ_API_KEY in .env" in result


def test_summarize_trial_rate_limit_error(monkeypatch):
    """Verify summarize_trial handles RateLimitError gracefully."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_rate_limit_error()
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = summarize_trial("Trial detail text")
    assert "⚠️ Groq rate limit reached" in result


def test_summarize_trial_generic_error(monkeypatch):
    """Verify summarize_trial handles general exceptions gracefully."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Groq service unreachable")
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = summarize_trial("Trial detail text")
    assert "⚠️ Summary temporarily unavailable" in result
    assert "Groq service unreachable" in result


def test_summarize_trial_input_truncation(monkeypatch):
    """Verify large trial documents are truncated to prevent prompt context blowout."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "• Summary point 1\n• Summary point 2\n• Summary point 3"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    huge_text = "Clinical trial detailed protocol description. " * 300  # ~14,000 chars
    summarize_trial(huge_text)

    kwargs = mock_client.chat.completions.create.call_args[1]
    user_prompt = kwargs["messages"][1]["content"]
    assert len(user_prompt) <= MAX_SUMMARY_INPUT_CHARS + 200
    assert len(user_prompt) < len(huge_text)


def test_summarize_trial_empty_choices_fallback(monkeypatch):
    """Verify fallback is returned if response has empty choices or empty content."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(choices=[])
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = summarize_trial("Trial text")
    assert result == FALLBACK_SUMMARY


# ============================================================================
# 3. ask_about_trials Tests
# ============================================================================


def test_ask_about_trials_success(monkeypatch):
    """Verify ask_about_trials returns model answer grounded in trial contexts."""
    expected_answer = "Based on the trial documents, Phase 3 trials are recruiting adults 18+."
    mock_choice = MagicMock()
    mock_choice.message.content = expected_answer
    mock_response = MagicMock(choices=[mock_choice])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    trial_contexts = [
        "NCT001: Phase 3 trial for diabetes. Eligibility: age 18-70.",
        "NCT002: Phase 2 trial for hypertension. Eligibility: age 21-65.",
    ]
    chat_history = [
        {"role": "user", "content": "Hello, can you help me find trials?"},
        {"role": "assistant", "content": "Yes, I can assist with clinical trials."},
    ]

    answer = ask_about_trials(
        question="What is the age requirement for the diabetes trial?",
        trial_contexts=trial_contexts,
        chat_history=chat_history,
    )

    assert answer == expected_answer
    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "qwen/qwen3.8-27b"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 500

    messages = kwargs["messages"]
    assert len(messages) == 4
    # System prompt with contexts
    assert messages[0]["role"] == "system"
    assert "NCT001: Phase 3 trial for diabetes" in messages[0]["content"]
    assert "NCT002: Phase 2 trial for hypertension" in messages[0]["content"]
    assert "GROUNDING CONSTRAINTS" in messages[0]["content"]
    # Chat history preserved
    assert messages[1] == {"role": "user", "content": "Hello, can you help me find trials?"}
    assert messages[2] == {"role": "assistant", "content": "Yes, I can assist with clinical trials."}
    # Current question
    assert messages[3] == {
        "role": "user",
        "content": "What is the age requirement for the diabetes trial?",
    }


def test_ask_about_trials_missing_key_fallback(monkeypatch):
    """Verify ask_about_trials returns informative guidance when client is None."""
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: None)

    answer = ask_about_trials("Is this trial safe?", ["Context text"], [])
    assert answer == FALLBACK_QA_MESSAGE
    assert "GROQ_API_KEY in your .env" in answer


def test_ask_about_trials_empty_question(monkeypatch):
    """Verify ask_about_trials handles empty or whitespace question gracefully."""
    mock_client = MagicMock()
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    assert "⚠️ Please ask a question" in ask_about_trials("", ["Context"])
    assert "⚠️ Please ask a question" in ask_about_trials("   \n\t  ", ["Context"])
    assert not mock_client.chat.completions.create.called


def test_ask_about_trials_non_string_question(monkeypatch):
    """Verify ask_about_trials returns prompt message when question is not a string."""
    mock_client = MagicMock()
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    for bad_question in (123, 45.67, ["question"], {"q": "text"}, None, True, object()):
        result = ask_about_trials(bad_question, ["Context"])
        assert "⚠️ Please ask a question" in result

    assert not mock_client.chat.completions.create.called


def test_ask_about_trials_malformed_contexts_with_none(monkeypatch):
    """Verify ask_about_trials handles malformed trial_contexts containing None, ints, or blanks."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Answer based on valid trial."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = ask_about_trials(
        question="What is this trial about?",
        trial_contexts=[None, 123, "   ", "Valid trial context: cancer study."],
        chat_history=[],
    )

    assert result == "Answer based on valid trial."
    mock_client.chat.completions.create.assert_called_once()
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    system_content = messages[0]["content"]
    assert "Valid trial context: cancer study." in system_content
    assert "123" in system_content
    assert "None" not in system_content


def test_ask_about_trials_malformed_chat_history_with_non_dicts(monkeypatch):
    """Verify ask_about_trials filters out non-dict items in chat_history without raising AttributeError."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Answer to follow-up."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    malformed_history = [
        "bad_str",
        123,
        None,
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"invalid_key": "no role"},
        {"role": "system", "content": "ignored role"},
        object(),
    ]

    result = ask_about_trials(
        question="Followup question?",
        trial_contexts=["Context"],
        chat_history=malformed_history,
    )

    assert result == "Answer to follow-up."
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    # System prompt (index 0) + valid user (hi) + valid assistant (hello there) + current question (index 3)
    assert len(messages) == 4
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello there"}
    assert messages[3]["content"] == "Followup question?"


def test_ask_about_trials_context_grounding_prompt(mock_groq_llm):
    """Verify system prompt includes grounding rules and all provided contexts."""
    contexts = [
        "Study NCT111: Pembrolizumab immunotherapy for lung cancer.",
        "Study NCT222: Nivolumab therapy for melanoma patients.",
    ]
    ask_about_trials("What treatments are being tested?", contexts, [])

    messages = mock_groq_llm.chat.completions.create.call_args[1]["messages"]
    system_content = messages[0]["content"]

    for ctx in contexts:
        assert ctx in system_content

    assert "GROUNDING CONSTRAINTS" in system_content
    assert "Answer strictly and only based on the retrieved trial contexts" in system_content
    assert "consult a medical professional" in system_content.lower()


def test_ask_about_trials_chat_history_preservation(mock_groq_llm):
    """Verify conversational chat history is strictly preserved in sequential order."""
    history = [
        {"role": "user", "content": "Turn 1 user question"},
        {"role": "assistant", "content": "Turn 1 assistant answer"},
        {"role": "user", "content": "Turn 2 user followup"},
        {"role": "assistant", "content": "Turn 2 assistant followup answer"},
    ]
    ask_about_trials("Turn 3 user query", ["Context"], history)

    messages = mock_groq_llm.chat.completions.create.call_args[1]["messages"]
    roles_and_contents = [(m["role"], m["content"]) for m in messages]

    assert ("user", "Turn 1 user question") in roles_and_contents
    assert ("assistant", "Turn 1 assistant answer") in roles_and_contents
    assert ("user", "Turn 2 user followup") in roles_and_contents
    assert ("assistant", "Turn 2 assistant followup answer") in roles_and_contents
    assert ("user", "Turn 3 user query") == roles_and_contents[-1]


def test_ask_about_trials_auth_error(monkeypatch):
    """Verify ask_about_trials catches AuthenticationError and returns clear guidance."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_auth_error()
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = ask_about_trials("Can I join?", ["Context"])
    assert "⚠️ Groq API key is invalid" in result


def test_ask_about_trials_rate_limit_error(monkeypatch):
    """Verify ask_about_trials catches RateLimitError and returns clear guidance."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_rate_limit_error()
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = ask_about_trials("Can I join?", ["Context"])
    assert "⚠️ Groq rate limit reached" in result


def test_ask_about_trials_error_handling(monkeypatch):
    """Verify ask_about_trials catches general exceptions without crashing."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API connection timeout")
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = ask_about_trials("Can I join?", ["Context"])
    assert "⚠️ Unable to answer question" in result
    assert "API connection timeout" in result


def test_ask_about_trials_empty_response(monkeypatch):
    """Verify ask_about_trials handles empty response content gracefully."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    monkeypatch.setattr("core.groq_client.get_groq_client", lambda: mock_client)

    result = ask_about_trials("Can I join?", ["Context"])
    assert "⚠️ No response generated from AI." in result
