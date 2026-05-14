"""Gemini (Google GenAI) judge configurator for RAGAS."""

from __future__ import annotations

import os


def _leer_env_int(nombre: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""
    try:
        return int(os.getenv(nombre, str(default)))
    except (TypeError, ValueError):
        return default


def configurar_llm_evaluacion_google(
    google_timeout: int | None = None,
    google_retries: int | None = None,
):
    """Configure Gemini 2.5 Flash + gemini-embedding-001 as the RAGAS judge."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
        raise SystemExit(1)
    google_timeout = google_timeout or _leer_env_int("EVAL_GOOGLE_TIMEOUT", 45)
    google_retries = google_retries if google_retries is not None else _leer_env_int("EVAL_GOOGLE_RETRIES", 2)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        eval_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=0,
            request_timeout=google_timeout,
            retries=google_retries,
        )
        eval_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_key,
            request_options={"timeout": google_timeout},
        )
        print("Evaluation LLM: Gemini 2.5 Flash (langchain-google-genai)")
        print("Evaluation embeddings: Google gemini-embedding-001 (langchain-google-genai)")
        print(f"Google timeout/retries: {google_timeout}s / {google_retries}")
        return eval_llm, eval_embeddings
    except ImportError as err:
        print(f"Error: {err}")
        print("  Install with: pip install langchain-google-genai")
        raise SystemExit(1)
