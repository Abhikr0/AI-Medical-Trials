"""Interactive conversational Q&A chat panel for retrieved clinical trials."""

import html
from typing import Any, Dict, List, Optional
import streamlit as st


def sanitize_chat_input(text: Optional[str]) -> str:
    """Sanitize and escape chat input text to prevent XSS.

    Args:
        text: Raw user question string.

    Returns:
        HTML-escaped, trimmed string.
    """
    if not text or not isinstance(text, str):
        return ""
    return html.escape(text.strip())


def init_chat_history(session_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Initialize chat_history in session state if not already present.

    Args:
        session_state: Optional session state dictionary (default: st.session_state).

    Returns:
        The chat_history list.
    """
    if session_state is None:
        session_state = st.session_state

    if "chat_history" not in session_state or session_state["chat_history"] is None:
        session_state["chat_history"] = []

    return session_state["chat_history"]


def ensure_chat_state(session_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Ensure chat_history key exists in session state (alias to init_chat_history)."""
    return init_chat_history(session_state)


def append_user_message(history: List[Dict[str, str]], message: str) -> None:
    """Append a user message to the chat history list.

    Args:
        history: The chat history list.
        message: The user's query text.
    """
    sanitized = sanitize_chat_input(message)
    history.append({"role": "user", "content": sanitized})


def append_assistant_message(history: List[Dict[str, str]], message: str) -> None:
    """Append an assistant response to the chat history list.

    Args:
        history: The chat history list.
        message: The LLM response text.
    """
    history.append({"role": "assistant", "content": str(message)})


def clear_chat_history(session_state: Optional[Dict[str, Any]] = None) -> None:
    """Reset chat history to an empty list.

    Args:
        session_state: Optional session state dictionary (default: st.session_state).
    """
    if session_state is None:
        session_state = st.session_state
    session_state["chat_history"] = []


def format_chat_error(err: Exception) -> str:
    """Format an exception into a user-friendly chat assistant error message.

    Args:
        err: Exception raised during chat completion.

    Returns:
        User-friendly error response string.
    """
    return f"AI Assistant temporarily unavailable: {err}"


def can_submit_chat(trial_contexts: Optional[List[str]]) -> bool:
    """Check if conversational chat can be submitted against loaded trial contexts.

    Args:
        trial_contexts: List of retrieved trial document texts.

    Returns:
        True if at least one trial document is loaded, False otherwise.
    """
    return bool(trial_contexts and len(trial_contexts) > 0)


def render_chat_messages(history: List[Dict[str, str]]) -> None:
    """Render the list of chat messages using st.chat_message.

    Args:
        history: List of message dictionaries with 'role' and 'content'.
    """
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_chat_panel(trial_docs: List[str]) -> None:
    """Render interactive conversational Q&A panel for clinical trials.

    Renders:
    - Header and instructions
    - Quick prompt suggestion pills
    - Chat message history stream
    - Chat input box (disabled if no trials are loaded)
    - Grounded Q&A dispatch to Groq LLM
    - Clear chat history button

    Args:
        trial_docs: List of retrieved trial document strings serving as RAG context.
    """
    st.subheader("💬 Ask About These Trials")

    ensure_chat_state()

    has_trials = can_submit_chat(trial_docs)
    if not has_trials:
        st.info("ℹ️ Search for clinical trials to start asking questions about them.")
    else:
        if not st.session_state["chat_history"]:
            st.caption(
                "Ask questions about eligibility criteria, trial phases, study locations, or treatments."
            )

    # Render previous conversation history
    render_chat_messages(st.session_state["chat_history"])

    # Chat input box
    prompt_placeholder = (
        "Ask a question about the retrieved trials..."
        if has_trials
        else "Search trials first to enable chat..."
    )
    question = st.chat_input(prompt_placeholder, disabled=not has_trials)

    if question:
        # Append user message
        append_user_message(st.session_state["chat_history"], question)

        # Render user message
        with st.chat_message("user"):
            st.markdown(st.session_state["chat_history"][-1]["content"])

        # Generate and render assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing trials and formulating answer..."):
                try:
                    from core.groq_client import ask_about_trials

                    # Ground strictly on current trial contexts and prior conversation
                    prior_history = st.session_state["chat_history"][:-1]
                    response = ask_about_trials(
                        question=question,
                        trial_contexts=trial_docs,
                        chat_history=prior_history,
                    )
                except Exception as e:
                    response = format_chat_error(e)

                st.markdown(response)
                append_assistant_message(st.session_state["chat_history"], response)

    # Clear chat option
    if st.session_state.get("chat_history"):
        st.divider()
        if st.button("🗑️ Clear Chat History", key="clear_chat_button"):
            clear_chat_history()
            st.rerun()
