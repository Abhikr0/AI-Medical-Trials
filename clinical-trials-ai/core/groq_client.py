"""Groq AI client integration for clinical trial summarization and grounded Q&A."""

import logging
import os
from typing import Any, Dict, List, Optional

from groq import AuthenticationError, Groq, GroqError, RateLimitError

import config

logger = logging.getLogger(__name__)

# Fallback text returned when GROQ_API_KEY is not configured
FALLBACK_SUMMARY = (
    "• Eligibility: See full criteria on study page.\n"
    "• Focus: Clinical evaluation under active trial protocol.\n"
    "• Note: Configure GROQ_API_KEY in .env for AI-generated summaries."
)

FALLBACK_QA_MESSAGE = (
    "⚠️ Groq API key is not configured. Please set GROQ_API_KEY in your .env file "
    "to enable AI trial Q&A."
)

DEFAULT_MODEL = "qwen/qwen3.8-27b"
MAX_SUMMARY_INPUT_CHARS = 3000


def get_groq_client() -> Optional[Groq]:
    """Return Groq client if GROQ_API_KEY is configured and valid, else None."""
    api_key = getattr(config, "GROQ_API_KEY", None)
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    api_key_str = str(api_key).strip()
    if not api_key_str or api_key_str.startswith("your_"):
        return None

    try:
        return Groq(api_key=api_key_str)
    except Exception as e:
        logger.warning("Failed to initialize Groq client: %s", e)
        return None


def summarize_trial(trial_text: str) -> str:
    """Generate 3-bullet plain-English patient summary focusing on eligibility, study test, and key criteria (with graceful offline fallback)."""
    try:
        if not isinstance(trial_text, str) or not trial_text.strip():
            return FALLBACK_SUMMARY

        client = get_groq_client()
        if client is None:
            return FALLBACK_SUMMARY

        truncated_text = trial_text[:MAX_SUMMARY_INPUT_CHARS]
        model = getattr(config, "GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL

        system_prompt = (
            "You are an expert clinical research assistant. Summarize clinical trials into concise, "
            "plain-English bullet points for patients. Focus on: eligibility, what the study tests, and key criteria."
        )
        user_prompt = (
            f"Please summarize the following clinical trial in exactly 3 concise bullet points:\n\n{truncated_text}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
        return FALLBACK_SUMMARY
    except AuthenticationError:
        return "⚠️ Groq API key is invalid. Please verify GROQ_API_KEY in .env."
    except RateLimitError:
        return "⚠️ Groq rate limit reached. Please wait a moment and try again."
    except Exception as e:
        return f"⚠️ Summary temporarily unavailable: {e}"


def ask_about_trials(
    question: str,
    trial_contexts: List[str],
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Perform grounded conversational Q&A strictly against retrieved trial documents (with graceful offline fallback)."""
    try:
        if not isinstance(question, str) or not question.strip():
            return "⚠️ Please ask a question about the clinical trials."

        client = get_groq_client()
        if client is None:
            return FALLBACK_QA_MESSAGE

        model = getattr(config, "GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL

        sanitized_contexts: List[str] = []
        if trial_contexts:
            if isinstance(trial_contexts, (str, bytes)):
                trial_contexts = [trial_contexts]
            try:
                sanitized_contexts = [
                    str(c).strip()
                    for c in trial_contexts
                    if c is not None and str(c).strip()
                ]
            except Exception:
                sanitized_contexts = []

        context_str = (
            "\n\n---\n\n".join(sanitized_contexts)
            if sanitized_contexts
            else "No specific trial context provided."
        )

        system_prompt = (
            "You are a helpful and knowledgeable clinical trials assistant. Your role is to answer questions "
            "about clinical trials based strictly on the provided trial contexts.\n\n"
            "GROUNDING CONSTRAINTS:\n"
            "- Answer strictly and only based on the retrieved trial contexts below.\n"
            "- Do NOT extrapolate, hallucinate, or assume facts not explicitly stated in the context.\n"
            "- If the answer cannot be determined from the provided trials, clearly state that the information "
            "is not available in the retrieved trials and advise the user to consult a medical professional or "
            "the full study protocol on ClinicalTrials.gov.\n\n"
            f"RETRIEVED TRIAL CONTEXTS:\n{context_str}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if chat_history:
            if isinstance(chat_history, (list, tuple)):
                for msg in chat_history:
                    if isinstance(msg, dict):
                        role = msg.get("role")
                        content = msg.get("content")
                        if role in ("user", "assistant") and content:
                            messages.append({"role": role, "content": str(content)})

        messages.append({"role": "user", "content": question.strip()})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
        return "⚠️ No response generated from AI."
    except AuthenticationError:
        return "⚠️ Groq API key is invalid. Please verify GROQ_API_KEY in .env."
    except RateLimitError:
        return "⚠️ Groq rate limit reached. Please wait a moment and try again."
    except Exception as e:
        return f"⚠️ Unable to answer question: {e}"
