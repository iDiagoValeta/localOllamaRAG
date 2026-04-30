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
def guardar_debug_rag(
    pregunta: str,
    mensaje_usuario: str = "",
    respuesta: str = "",
    fragmentos: Optional[List[Dict[str, Any]]] = None,
    motivo_interrupcion: Optional[str] = None,
    metricas: Optional[Dict[str, Any]] = None
) -> None:
    """Dump a full RAG interaction to ``debug_rag/`` for inspection.

    The output file (timestamped + slug) includes sub-queries, keywords,
    pipeline configuration, the full prompt, the model response, and all
    retrieved fragments with their scores.

    Args:
        pregunta: Original user question.
        mensaje_usuario: Complete user message (with ``<context>``).
        respuesta: Model response text.
        fragmentos: Retrieved fragments used for context.
        motivo_interrupcion: Reason for early interruption, if any.
        metricas: Full pipeline metrics dict.
    """
    fragmentos = fragmentos or []

    if not GUARDAR_DEBUG_RAG:
        return

    try:
        os.makedirs(CARPETA_DEBUG_RAG, exist_ok=True)

        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r'[^\w\s]', '', pregunta)[:40].strip().replace(' ', '_')
        nombre_archivo = f"{timestamp}_{slug}.txt"
        ruta = os.path.join(CARPETA_DEBUG_RAG, nombre_archivo)

        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"  DEBUG RAG - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            if metricas and (metricas.get('sub_queries') or metricas.get('queries_semanticas') or metricas.get('keywords') or metricas.get('terminos_criticos') or metricas.get('fase_semantica') or metricas.get('fase_keywords') or metricas.get('fase_exhaustiva') or metricas.get('fase_reranking')):
                f.write("─" * 80 + "\n")
                f.write("  RETRIEVAL PIPELINE (sub-queries, keywords, terms, metrics)\n")
                f.write("─" * 80 + "\n")
                sub_q = metricas.get('sub_queries', [])
                if sub_q:
                    f.write("\nSub-queries (Query Decomposition):\n")
                    for i, sq in enumerate(sub_q, 1):
                        f.write(f"  {i}. {sq}\n")
                queries_sem = metricas.get('queries_semanticas', [])
                if queries_sem:
                    f.write("\nQueries used in semantic search:\n")
                    for i, q in enumerate(queries_sem, 1):
                        f.write(f"  {i}. {q}\n")
                keywords = metricas.get('keywords', [])
                if keywords:
                    f.write(f"\nExtracted keywords ({len(keywords)}):\n  {', '.join(keywords[:30])}\n")
                    if len(keywords) > 30:
                        f.write(f"  ... and {len(keywords) - 30} more\n")
                terminos = metricas.get('terminos_criticos', [])
                if terminos:
                    f.write(f"\nCritical terms (exhaustive search):\n  {', '.join(terminos)}\n")
                fase_kw = metricas.get('fase_keywords', {})
                if fase_kw:
                    f.write(f"\nKeyword metrics: {fase_kw.get('keywords_encontradas', 0)}/{fase_kw.get('keywords_totales', 0)} found, {fase_kw.get('resultados_totales', 0)} results\n")
                f.write(f"\nFull metrics:\n{json.dumps(metricas, indent=2, ensure_ascii=False, default=str)}\n\n")

            if motivo_interrupcion:
                f.write("─" * 80 + "\n")
                f.write("  EARLY INTERRUPTION\n")
                f.write("─" * 80 + "\n")
                f.write(f"{motivo_interrupcion}\n\n")
                if metricas:
                    f.write("Search metrics:\n")
                    f.write(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n\n")

            f.write("─" * 80 + "\n")
            f.write("  PIPELINE CONFIGURATION\n")
            f.write("─" * 80 + "\n")
            f.write(f"RAG Model: {_inferir_descripcion_modelo(MODELO_RAG)}\n")
            f.write(f"Contextual Retrieval (Indexing): {'YES' if USAR_CONTEXTUAL_RETRIEVAL else 'NO'}\n")
            f.write(f"Query Decomposition: {'YES' if USAR_LLM_QUERY_DECOMPOSITION else 'NO'}\n")
            f.write(f"Hybrid Search (keywords): {'YES' if USAR_BUSQUEDA_HIBRIDA else 'NO'}\n")
            f.write(f"Exhaustive Search: {'YES' if USAR_BUSQUEDA_EXHAUSTIVA else 'NO'}\n")
            f.write(f"Reranker: {'YES' if USAR_RERANKER else 'NO'}\n")
            f.write(f"Expand Context: {'YES' if EXPANDIR_CONTEXTO else 'NO'}\n")
            f.write(f"Optimize Context: {'YES' if USAR_OPTIMIZACION_CONTEXTO else 'NO'}\n")
            f.write(f"RECOMP Synthesis: {'YES' if USAR_RECOMP_SYNTHESIS else 'NO'}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  ORIGINAL QUESTION\n")
            f.write("─" * 80 + "\n")
            f.write(f"{pregunta}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  SYSTEM PROMPT\n")
            f.write("─" * 80 + "\n")
            if _modelo_necesita_system_prompt(MODELO_RAG):
                f.write(f"{SYSTEM_PROMPT_RAG}\n\n")
            else:
                f.write("(baked into Modelfile — not sent via API)\n\n")

            context_match = re.search(r'<context>(.*?)</context>', mensaje_usuario, re.DOTALL)
            contexto_enviado = context_match.group(1).strip() if context_match else "(empty)"

            f.write("─" * 80 + "\n")
            if USAR_RECOMP_SYNTHESIS:
                f.write("  RECOMP SYNTHESIS SENT TO FINAL MODEL (instead of raw chunks)\n")
            else:
                f.write("  RAW CONTEXT SENT TO FINAL MODEL\n")
            f.write("─" * 80 + "\n")
            f.write(f"{contexto_enviado}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  USER MESSAGE (full actual prompt)\n")
            f.write("─" * 80 + "\n")
            f.write(f"{mensaje_usuario or '(not sent to model)'}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  MODEL RESPONSE\n")
            f.write("─" * 80 + "\n")
            f.write(f"{respuesta or '(not generated)'}\n\n")

            f.write("─" * 80 + "\n")
            f.write(f"  RETRIEVED FRAGMENTS ({len(fragmentos)})\n")
            f.write("─" * 80 + "\n")
            for i, frag in enumerate(fragmentos, 1):
                meta = frag.get('metadata', {})
                score = frag.get('score_final', 'N/A')
                score_rr = frag.get('score_reranker', 'N/A')
                f.write(f"\n--- Fragment {i} ---\n")
                pag = meta.get('page', 0)
                f.write(f"Source: {meta.get('source', '?')}, page {pag + 1 if isinstance(pag, int) else pag}\n")
                f.write(f"Final score: {score}  |  Reranker score: {score_rr}\n")
                matches = frag.get('matches', [])
                if matches:
                    f.write(f"Matched keywords: {', '.join(matches)}\n")
                query_matches = frag.get('query_matches', [])
                if query_matches:
                    f.write(f"Matched query(s): {query_matches}\n")
                f.write(f"Section: {meta.get('section_header', '(no header)')}\n")
                doc_text = frag.get('doc', '')
                if '\\n\\n' in doc_text:
                    ctx_part, orig_part = doc_text.split('\\n\\n', 1)
                    f.write(f"[Contextual Retrieval]:\n{ctx_part}\n\n")
                    f.write(f"[Document text]:\n{orig_part}\n")
                else:
                    f.write(f"[Document text]:\n{doc_text}\n")

        logging.info(f"Debug RAG saved: {ruta}")

    except Exception as e:
        logging.warning(f"Error saving debug RAG: {e}")



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

