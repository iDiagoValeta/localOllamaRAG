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
# SECTION 8: SEMANTIC RERANKING
# ─────────────────────────────────────────────


_reranker_model = None


def _detectar_dispositivo_reranker() -> str:
    """Detect the best available device for the reranker.

    Returns:
        ``"cuda"`` if a CUDA GPU is available, otherwise ``"cpu"``.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def obtener_modelo_reranker():
    """Return the Cross-Encoder singleton, loading it lazily on first call.

    Uses FP16 on CUDA when a GPU is available. The model variant is
    controlled by ``RERANKER_MODEL_QUALITY`` (``"quality"`` or fast).

    Returns:
        The loaded ``CrossEncoder`` instance, or ``None`` on failure.
    """
    global _reranker_model

    if not USAR_RERANKER:
        return None

    if _reranker_model is None:
        try:
            if RERANKER_MODEL_QUALITY == "quality":
                modelo_nombre = "BAAI/bge-reranker-v2-m3"
            else:
                modelo_nombre = "cross-encoder/ms-marco-MiniLM-L-6-v2"

            device = _detectar_dispositivo_reranker()

            ui.debug(f"Loading reranker: {modelo_nombre}")
            ui.debug(f"device: {device.upper()}" + (" (FP16)" if device == "cuda" else ""))

            model_kwargs = {"torch_dtype": "float16"} if device == "cuda" else {}
            import io, contextlib
            with contextlib.redirect_stderr(io.StringIO()):
                _reranker_model = CrossEncoder(
                    modelo_nombre,
                    device=device,
                    model_kwargs=model_kwargs,
                )

            ui.debug(f"reranker loaded on {device.upper()}")
        except Exception as e:
            logging.error(f"Error loading reranker model: {e}")
            return None

    return _reranker_model


def rerank_resultados(
    pregunta: str,
    documentos_recuperados: List[Dict[str, Any]],
    top_k: int = TOP_K_AFTER_RERANK
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reorder candidates using a Cross-Encoder for finer relevance scoring.

    Args:
        pregunta: The user query.
        documentos_recuperados: Candidate chunks from prior retrieval stages.
        top_k: Number of top results to keep after reranking.

    Returns:
        Tuple of (top-k documents with ``score_reranker``, metrics dict).
    """
    metricas = {
        'candidatos_entrada': len(documentos_recuperados),
        'resultados_salida': 0,
        'tiempo_reranking': 0,
        'modelo_usado': RERANKER_MODEL_QUALITY,
        'dispositivo': _detectar_dispositivo_reranker()
    }

    if not USAR_RERANKER or not documentos_recuperados:
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas

    reranker = obtener_modelo_reranker()
    if reranker is None:
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas

    try:
        import time
        inicio = time.time()

        textos_documentos = []
        for doc in documentos_recuperados:
            texto = doc['doc']
            if '\\n\\n' in texto:
                partes = texto.split('\\n\\n', 1)
                texto = partes[-1]
            textos_documentos.append(texto)

        import io, contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            ranks = reranker.rank(
                pregunta,
                textos_documentos,
                top_k=min(top_k, len(textos_documentos)),
                return_documents=False
            )

        documentos_reordenados = []
        for rank_info in ranks:
            idx = rank_info['corpus_id']
            score = rank_info['score']

            doc_original = documentos_recuperados[idx].copy()
            doc_original['score_reranker'] = float(score)
            doc_original['score_final'] = float(score)
            documentos_reordenados.append(doc_original)

        metricas['tiempo_reranking'] = time.time() - inicio
        metricas['resultados_salida'] = len(documentos_reordenados)

        if LOGGING_METRICAS:
            dev = metricas['dispositivo'].upper()
            logging.info(f"Reranking ({dev}): {metricas['candidatos_entrada']} -> {metricas['resultados_salida']} in {metricas['tiempo_reranking']:.2f}s")

        return documentos_reordenados, metricas

    except Exception as e:
        logging.error(f"Error in reranking: {e}")
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas


def generar_queries_con_llm(pregunta: str) -> List[str]:
    """Generate 3 search sub-queries via an auxiliary LLM.

    Each sub-query targets a different semantic aspect of the original
    question and is written in the same language.

    Ollama ``think=False`` disables the reasoning trace for thinking-capable
    models (Gemma 4, Qwen3, etc.); see ``scripts/tests/test_gemma4_aux_nothink.py``.
    With ``think=True``, ``num_predict`` can be consumed by the trace alone, leaving
    an empty answer — production therefore keeps ``think=False`` and ``num_predict`` 400.
    To print raw model output in the terminal, run ``scripts/tests/debug_aux_subqueries.py``.

    Args:
        pregunta: The original user question.

    Returns:
        Up to 3 generated queries, or empty list on failure.
    """
    try:
        prompt = (
            "Generate exactly 3 search queries to retrieve relevant content "
            "from academic documents about the question below.\n\n"
            "Requirements:\n"
            "- Each query must target a DIFFERENT semantic aspect of the question\n"
            "- Write every query in the EXACT SAME LANGUAGE as the question\n"
            "- Output ONLY the 3 queries, one per line\n"
            "- No numbering, no bullets, no labels, no explanations\n\n"
            f"Question: {pregunta}\n\n"
            "Queries:\n"
        )

        response = ollama.generate(
            model=MODELO_CHAT,
            prompt=prompt,
            think=False,
            options={
                "temperature": 0.5,
                "num_predict": 400,
                "stop": ["\n\n\n"],
            },
        )

        queries = [
            q.strip().lstrip("0123456789.-) ")
            for q in response["response"].strip().split("\n")
            if q.strip() and len(q.strip()) > 20
        ]

        return queries[:3]

    except Exception as e:
        logging.warning(f"Error generating queries with LLM ({MODELO_CHAT}): {e}")
        return []


def _validar_coherencia_query(query: str) -> bool:
    """Detect whether a query is an incoherent bag-of-words.

    Checks unique-word ratio, word repetition, and presence of
    connectors to ensure the query reads as a natural sentence.

    Args:
        query: The candidate query string.

    Returns:
        ``True`` if the query appears coherent, ``False`` otherwise.
    """
    words = query.lower().split()
    if len(words) < 2:
        return True

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.7:
        return False

    word_freq = Counter(words)
    if word_freq.most_common(1)[0][1] >= 3:
        return False

    connectors = {
    # English
    "the", "a", "an", "is", "are", "how", "what", "why",
    "when", "where", "which", "does", "do", "to", "in", "of",
    "that", "for", "and", "with", "by", "on", "as",
    # Spanish
    "cómo", "qué", "cuál", "cuáles", "cuándo", "dónde", "por",
    "para", "que", "son", "está", "entre", "con", "los", "las",
    # Valencian
    "com", "quins", "quines", "quan", "quin", "quina", "per", "que",
    }

    has_connectors = any(w in connectors for w in words)
    if len(words) > 8 and not has_connectors:
        return False

    return True


def _filtrar_terminos_criticos(terminos: List[str]) -> List[str]:
    """Keep only high-discrimination domain-specific terms for exhaustive search.

    Filters out generic blacklisted single words and retains multi-word
    terms, capitalized terms, and acronyms.

    Args:
        terminos: Candidate critical terms.

    Returns:
        Filtered list of domain-specific terms.
    """
    filtered = []
    for term in terminos:
        words = term.lower().split()

        if len(words) == 1 and term.lower() in GENERIC_TERMS_BLACKLIST:
            continue

        if len(words) >= 2:
            filtered.append(term)
            continue

        if term[0].isupper() or term.isupper():
            filtered.append(term)

    return filtered


# ─────────────────────────────────────────────



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

