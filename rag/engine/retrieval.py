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
# SECTION 9: HYBRID RETRIEVAL PIPELINE
# ─────────────────────────────────────────────


def realizar_busqueda_hibrida(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Orchestrate the full hybrid retrieval pipeline.

    Combines multi-query semantic search, keyword search, exhaustive
    deep scan, RRF fusion, and optional Cross-Encoder reranking into
    a single ranked result set.

    Args:
        pregunta: User query.
        collection: ChromaDB collection to search.

    Returns:
        Tuple of (ranked fragments, best score, full metrics dict).
    """
    ui.debug("Starting hybrid search...")

    metricas_totales = {
        'fase_semantica': {},
        'fase_keywords': {},
        'fase_exhaustiva': {},
        'fase_reranking': {},
        'candidatos_fusion': 0,
        'resultados_finales': 0
    }

    llm_queries = []
    if USAR_LLM_QUERY_DECOMPOSITION and len(pregunta) > 60:
        ui.debug("decomposing query...")
        llm_queries = generar_queries_con_llm(pregunta)
        if llm_queries:
            ui.debug(f"{len(llm_queries)} sub-queries generated")

    ui.debug("semantic search...")

    queries = [pregunta]

    _KEEP_SHORT = {"to", "of", "in", "on", "by", "for", "as", "is", "are", "was",
                    "and", "or", "the", "its", "how", "not", "no", "what", "does"}
    palabras_clave_pregunta = [
        p for p in pregunta.split()
        if len(p) > 4 or p.lower().strip('"?,()') in _KEEP_SHORT
    ]
    query_corta = ' '.join(palabras_clave_pregunta[:20]).strip()
    if query_corta and query_corta != pregunta:
        queries.append(query_corta)

    keywords_expandidas = extraer_keywords(pregunta)

    if llm_queries:
        fallback_q = llm_queries[0]
        if _validar_coherencia_query(fallback_q) and fallback_q not in queries:
            queries.append(fallback_q)
    elif keywords_expandidas:
        query_kw = ' '.join(keywords_expandidas[:6]).strip()
        if query_kw and _validar_coherencia_query(query_kw) and query_kw not in queries:
            queries.append(query_kw)

    for lq in llm_queries:
        if lq not in queries:
            queries.append(lq)

    ui.debug(f"{len(queries)} query variant(s)")

    all_semantic_results = {}

    for q_idx, query in enumerate(queries):
        query_con_prefijo = f"{EMBED_PREFIX_QUERY}{query}"
        response_emb = ollama.embeddings(model=MODELO_EMBEDDING, prompt=query_con_prefijo)

        results_semantic = collection.query(
            query_embeddings=[response_emb["embedding"]],
            n_results=N_RESULTADOS_SEMANTICOS,
            include=['documents', 'distances', 'metadatas']
        )

        for idx, (doc, distancia, metadata) in enumerate(zip(
            results_semantic['documents'][0],
            results_semantic['distances'][0],
            results_semantic['metadatas'][0]
        ), 1):
            chunk_id = f"{metadata['source']}_pag{metadata['page']}_chunk{metadata.get('chunk', 0)}"

            if chunk_id not in all_semantic_results:
                all_semantic_results[chunk_id] = {
                    'doc': doc,
                    'metadata': metadata,
                    'distancia': distancia,
                    'id': chunk_id,
                    'score_semantic': 0.0,
                    'score_keyword': 0.0,
                    'matches': [],
                    'query_matches': []
                }

            all_semantic_results[chunk_id]['score_semantic'] += 1.0 / (idx + RRF_K)
            all_semantic_results[chunk_id]['query_matches'].append(q_idx + 1)
            if distancia < all_semantic_results[chunk_id]['distancia']:
                all_semantic_results[chunk_id]['distancia'] = distancia

    metricas_totales['fase_semantica'] = {
        'queries_generadas': len(queries),
        'fragmentos_unicos': len(all_semantic_results)
    }

    ui.debug(f"{len(all_semantic_results)} unique fragments")

    results_keyword = []
    metricas_keywords = {}
    if USAR_BUSQUEDA_HIBRIDA:
        ui.debug("keyword search...")
        keywords = extraer_keywords(pregunta)
        if keywords:
            ui.debug(f"detected: {', '.join(keywords[:8])}")
        results_keyword, metricas_keywords = busqueda_por_keywords(pregunta, collection)
        metricas_totales['fase_keywords'] = metricas_keywords

    ui.debug("fusing results...")

    fragmentos_data = all_semantic_results.copy()

    for idx, result in enumerate(results_keyword, 1):
        chunk_id = result['id']

        if chunk_id in fragmentos_data:
            fragmentos_data[chunk_id]['score_keyword'] += 1.0 / (idx + RRF_K)
            if result['keyword_match'] not in fragmentos_data[chunk_id]['matches']:
                fragmentos_data[chunk_id]['matches'].append(result['keyword_match'])
        else:
            fragmentos_data[chunk_id] = {
                'doc': result['doc'],
                'metadata': result['metadata'],
                'distancia': result['distancia'],
                'id': chunk_id,
                'score_semantic': 0.0,
                'score_keyword': 1.0 / (idx + RRF_K),
                'matches': [result['keyword_match']],
                'query_matches': []
            }

    for frag in fragmentos_data.values():
        frag['score_final'] = (frag['score_semantic'] * 0.55 + frag['score_keyword'] * 0.45)

    terminos_criticos = _filtrar_terminos_criticos(
        [k for k in keywords_expandidas[:12] if len(k) > 3]
    )

    metricas_exhaustiva = {}
    if USAR_BUSQUEDA_EXHAUSTIVA and terminos_criticos:
        ui.debug(f"deep search: {', '.join(terminos_criticos[:6])}")
        resultados_exhaustivos, metricas_exhaustiva = busqueda_exhaustiva_texto(
            terminos_criticos, collection, max_results=30
        )
        metricas_totales['fase_exhaustiva'] = metricas_exhaustiva

        for idx, result in enumerate(resultados_exhaustivos):
            chunk_id = result['id']

            if chunk_id in fragmentos_data:
                fragmentos_data[chunk_id]['score_keyword'] += 0.3 * result['num_matches']
                fragmentos_data[chunk_id]['matches'].extend(
                    m for m in result['matches'] if m not in fragmentos_data[chunk_id]['matches']
                )
            else:
                fragmentos_data[chunk_id] = {
                    'doc': result['doc'],
                    'metadata': result['metadata'],
                    'distancia': float('inf'),
                    'id': chunk_id,
                    'score_semantic': 0.0,
                    'score_keyword': 0.3 * result['num_matches'],
                    'matches': result['matches'],
                    'query_matches': []
                }

        for frag in fragmentos_data.values():
            frag['score_final'] = (frag['score_semantic'] * 0.55 + frag['score_keyword'] * 0.45)

    fragmentos_ranked = sorted(
        fragmentos_data.values(),
        key=lambda x: x['score_final'],
        reverse=True
    )

    metricas_totales['candidatos_fusion'] = len(fragmentos_ranked)

    if USAR_RERANKER and fragmentos_ranked:
        n_candidatos = min(TOP_K_RERANK_CANDIDATES, len(fragmentos_ranked))
        ui.debug(f"reranking top {n_candidatos} candidates...")

        candidatos_rerank = fragmentos_ranked[:TOP_K_RERANK_CANDIDATES]
        fragmentos_ranked, metricas_rerank = rerank_resultados(
            pregunta,
            candidatos_rerank,
            top_k=TOP_K_AFTER_RERANK
        )
        metricas_totales['fase_reranking'] = metricas_rerank
        ui.debug(f"top {len(fragmentos_ranked)} after reranking")

    mejor_score = fragmentos_ranked[0]['score_final'] if fragmentos_ranked else 0
    metricas_totales['resultados_finales'] = len(fragmentos_ranked)

    metricas_totales['sub_queries'] = llm_queries
    metricas_totales['queries_semanticas'] = queries
    metricas_totales['keywords'] = list(keywords_expandidas)
    metricas_totales['terminos_criticos'] = terminos_criticos

    if LOGGING_METRICAS:
        sem_unicos = metricas_totales['fase_semantica'].get('fragmentos_unicos', 0)
        kw_total = metricas_keywords.get('resultados_totales', 0)
        logging.info(
            f"Full pipeline: Semantic({sem_unicos}) + "
            f"Keywords({kw_total}) -> "
            f"Fusion({metricas_totales['candidatos_fusion']}) -> "
            f"Reranking({metricas_totales['resultados_finales']})"
        )

    return fragmentos_ranked, mejor_score, metricas_totales


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

