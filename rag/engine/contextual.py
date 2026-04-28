"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration remains owned by rag.chat_pdfs and is synchronized before each
function call so web/API toggles and test monkeypatches keep working.
"""

import base64
import contextlib
import io
import json
import logging
import os
import re
import requests
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import ollama
from pypdf import PdfReader

from rag.cli.display import ui
from rag.engine.runtime import sync_runtime_globals


def _sync_runtime_globals() -> None:
    sync_runtime_globals(globals())


_sync_runtime_globals()
# SECTION 11: INDEXING AND COLLECTION MANAGEMENT
# ─────────────────────────────────────────────


# --- 11.1 Contextual retrieval ---


def _detectar_idioma(texto: str) -> str:
    """Heuristic language detector from a document text sample.

    Counts distinctive function-word occurrences to distinguish Spanish,
    Catalan, and English. Designed for documents where the first ~4000
    characters are available; accuracy degrades on very short samples.

    Args:
        texto: Representative text sample from the document (ideally ≥500 chars).

    Returns:
        Full language name suitable for prompt injection: 'Spanish',
        'Catalan', or 'English'.
    """
    t = texto.lower()
    # Catalan-distinctive markers (absent or rare in Spanish)
    ca = (t.count("però ") + t.count("també ") + t.count("molt ")
          + t.count(" amb ") + t.count(" va ") + t.count("els ")
          + t.count("l'") + t.count("d'") + t.count("s'")
          + t.count("n'") + t.count("m'"))
    # Spanish-distinctive markers (absent or rare in Catalan)
    es = (t.count("también ") + t.count("además ") + t.count("pero ")
          + t.count("muy ") + t.count(" con ") + t.count("los ")
          + t.count("las ") + t.count("así ") + t.count("sin ")
          + t.count("después"))
    # English-distinctive markers
    en = (t.count(" the ") + t.count(" is ") + t.count(" are ")
          + t.count(" was ") + t.count(" were ") + t.count(" have ")
          + t.count(" this ") + t.count(" that ") + t.count(" from ")
          + t.count(" with "))
    scores = {"Spanish": es, "Catalan": ca, "English": en}
    return max(scores, key=scores.get)


def generar_contexto_situacional(
    chunk_text: str,
    texto_base: str,
    idioma_doc: str = "",
) -> str:
    """Generate 2-3 sentences of situational context for a chunk using an LLM.

    Produces a brief document summary plus a note on how the chunk fits
    within the larger document, to improve retrieval accuracy. The output
    is always written in the document's own language.

    Args:
        chunk_text: The text of the chunk to contextualize.
        texto_base: A representative excerpt of the full document.
        idioma_doc: Document language ('Spanish', 'Catalan', 'English').
            When empty, falls back to heuristic detection from ``texto_base``.

    Returns:
        Situational context string (with trailing ``\\n\\n``), or empty
        string if disabled or on failure.
    """
    if not USAR_CONTEXTUAL_RETRIEVAL:
        return ""

    idioma = idioma_doc or _detectar_idioma(texto_base)

    system_prompt = (
        f"You are an expert at analyzing academic documents. "
        f"MANDATORY: Write your entire response in {idioma} — the same language as the document. "
        f"Do NOT translate. Do NOT switch to any other language, including English. "
        f"When given a full document and an excerpt from it, produce exactly 2-3 sentences: "
        f"first a brief summary of what the document is about, then how the excerpt fits within it. "
        f"No introductions, no labels, no meta-commentary. "
        f"Do NOT include bibliographic citation markers such as [1], [38], or similar."
    )

    user_prompt = (
        f"<document>\\n{texto_base}\\n</document>\\n\\n"
        f"<excerpt>\\n{chunk_text}\\n</excerpt>\\n\\n"
        f"Write the 2-3 sentence situational context in {idioma}."
    )

    try:
        response = ollama.chat(
            model=MODELO_CONTEXTUAL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            think=False,
            options={"temperature": 0.1, "num_predict": 250},
        )
        contexto = response['message']['content'].strip()
        if contexto:
            return f"{contexto}\\n\\n"
    except Exception as e:
        logging.warning(f"Error generating situational context: {e}")
    return ""



def _with_runtime_sync(func):
    def wrapper(*args, **kwargs):
        _sync_runtime_globals()
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    return wrapper


for _name, _obj in list(globals().items()):
    if callable(_obj) and getattr(_obj, "__module__", None) == __name__ and _name not in {
        "_sync_runtime_globals", "_with_runtime_sync"
    }:
        globals()[_name] = _with_runtime_sync(_obj)

_sync_runtime_globals()

