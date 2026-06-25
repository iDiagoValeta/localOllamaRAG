"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

import contextlib
import io
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import ollama

from rag.cli.display import ui
from rag.engine.runtime import get_runtime

cfg = get_runtime()
# SECTION 8: SEMANTIC RERANKING
# ─────────────────────────────────────────────


_reranker_model = None
_reranker_device = None  # device the singleton actually loaded on ("cuda"/"cpu")


def _detectar_dispositivo_reranker() -> str:
    """Detect the preferred device for the reranker.

    Honors the ``RERANKER_DEVICE`` env override (``"cpu"`` / ``"cuda"``). Without
    the override, returns ``"cuda"`` when a CUDA GPU is available, otherwise
    ``"cpu"``. The desktop app no longer pins this to CPU: every stage runs on
    the GPU by default and ``obtener_modelo_reranker`` falls back to CPU if the
    GPU load fails (limited VRAM, driver issues).

    Returns:
        ``"cuda"`` or ``"cpu"``.
    """
    override = os.getenv("RERANKER_DEVICE", "").strip().lower()
    if override in ("cpu", "cuda"):
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _cargar_crossencoder(modelo_nombre: str, device: str):
    """Load a ``CrossEncoder`` on ``device`` (FP16 on CUDA). Raises on failure."""
    model_kwargs = {"torch_dtype": "float16"} if device == "cuda" else {}
    with contextlib.redirect_stderr(io.StringIO()):
        return cfg.CrossEncoder(modelo_nombre, device=device, model_kwargs=model_kwargs)


def obtener_modelo_reranker():
    """Return the Cross-Encoder singleton, loading it lazily on first call.

    Loads on the GPU by default (FP16 on CUDA). If the GPU load fails -- common
    on limited VRAM or with a missing/incompatible driver -- it falls back to
    CPU instead of disabling the reranker. The model variant is controlled by
    ``RERANKER_MODEL_QUALITY`` (``"quality"`` or fast).

    Returns:
        The loaded ``CrossEncoder`` instance, or ``None`` if both devices fail.
    """
    global _reranker_model, _reranker_device

    if not cfg.USAR_RERANKER:
        return None

    if _reranker_model is None:
        if cfg.RERANKER_MODEL_QUALITY == "quality":
            modelo_nombre = "BAAI/bge-reranker-v2-m3"
        else:
            modelo_nombre = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        device = _detectar_dispositivo_reranker()
        ui.debug(f"Loading reranker: {modelo_nombre}")
        ui.debug(f"device: {device.upper()}" + (" (FP16)" if device == "cuda" else ""))

        try:
            _reranker_model = _cargar_crossencoder(modelo_nombre, device)
            _reranker_device = device
            ui.debug(f"reranker loaded on {device.upper()}")
        except Exception as e:
            if device == "cuda":
                logging.warning(f"Reranker GPU load failed ({e}); falling back to CPU")
                ui.debug("reranker GPU load failed; falling back to CPU")
                try:
                    _reranker_model = _cargar_crossencoder(modelo_nombre, "cpu")
                    _reranker_device = "cpu"
                    ui.debug("reranker loaded on CPU (fallback)")
                except Exception as e2:
                    logging.error(f"Error loading reranker model on CPU: {e2}")
                    return None
            else:
                logging.error(f"Error loading reranker model: {e}")
                return None

    return _reranker_model


def rerank_resultados(
    pregunta: str,
    documentos_recuperados: List[Dict[str, Any]],
    top_k: int = cfg.TOP_K_FINAL
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
        'modelo_usado': cfg.RERANKER_MODEL_QUALITY,
        'dispositivo': _detectar_dispositivo_reranker()
    }

    if not cfg.USAR_RERANKER or not documentos_recuperados:
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas

    reranker = cfg.obtener_modelo_reranker()
    if reranker is None:
        metricas['resultados_salida'] = len(documentos_recuperados)
        return documentos_recuperados, metricas

    # Reflect the device the model actually loaded on (CPU if the GPU load fell back).
    if _reranker_device:
        metricas['dispositivo'] = _reranker_device

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

        if cfg.LOGGING_METRICAS:
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
    models (Gemma 4, Qwen3, etc.); see ``research/tests/core/test_gemma4_aux_nothink.py``.
    With ``think=True``, ``num_predict`` can be consumed by the trace alone, leaving
    an empty answer — production therefore keeps ``think=False`` and ``num_predict`` 400.
    To print raw model output in the terminal, run ``research/tests/core/debug_aux_subqueries.py``.

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
            model=cfg.MODELO_CHAT,
            prompt=prompt,
            think=False,
            keep_alive=0,
            options={
                "temperature": 0.5,
                "num_predict": 400,
                "num_ctx": cfg.OLLAMA_QUERY_NUM_CTX,
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
        logging.warning(f"Error generating queries with LLM ({cfg.MODELO_CHAT}): {e}")
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




