"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

import logging
from typing import Any, Dict, List, Tuple

import chromadb
import ollama

from rag.cli.display import ui
from rag.engine.runtime import get_runtime

cfg = get_runtime()
# SECTION 9: HYBRID RETRIEVAL PIPELINE
# ─────────────────────────────────────────────


def realizar_busqueda_hibrida(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Orchestrate the full hybrid retrieval pipeline.

    Combines multi-query semantic search, BM25 lexical search, RRF fusion,
    and optional Cross-Encoder reranking into a single ranked result set.

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
        'fase_reranking': {},
        'candidatos_fusion': 0,
        'resultados_finales': 0
    }

    llm_queries = []
    if cfg.USAR_LLM_QUERY_DECOMPOSITION and len(pregunta) > 60:
        ui.debug("decomposing query...")
        llm_queries = cfg.generar_queries_con_llm(pregunta)
        if llm_queries:
            ui.debug(f"{len(llm_queries)} sub-queries generated")

    ui.debug("semantic search...")

    queries = [pregunta]

    keywords_expandidas = cfg.extraer_keywords(pregunta)

    if llm_queries:
        fallback_q = llm_queries[0]
        if cfg._validar_coherencia_query(fallback_q) and fallback_q not in queries:
            queries.append(fallback_q)
    elif keywords_expandidas:
        query_kw = ' '.join(keywords_expandidas[:6]).strip()
        if query_kw and cfg._validar_coherencia_query(query_kw) and query_kw not in queries:
            queries.append(query_kw)

    for lq in llm_queries:
        if lq not in queries:
            queries.append(lq)

    ui.debug(f"{len(queries)} query variant(s)")

    all_semantic_results = {}

    for q_idx, query in enumerate(queries):
        query_con_prefijo = f"{cfg.EMBED_PREFIX_QUERY}{query}"
        response_emb = ollama.embeddings(model=cfg.MODELO_EMBEDDING, prompt=query_con_prefijo)

        results_semantic = collection.query(
            query_embeddings=[response_emb["embedding"]],
            n_results=cfg.N_RESULTADOS_SEMANTICOS,
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

            all_semantic_results[chunk_id]['score_semantic'] += 1.0 / (idx + cfg.RRF_K)
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
    if cfg.USAR_BUSQUEDA_HIBRIDA:
        ui.debug("BM25 lexical search...")
        results_keyword, metricas_keywords = cfg.busqueda_lexica_bm25(pregunta, collection)
        metricas_totales['fase_keywords'] = metricas_keywords

    ui.debug("fusing results...")

    fragmentos_data = all_semantic_results.copy()

    for idx, result in enumerate(results_keyword, 1):
        chunk_id = result['id']

        if chunk_id in fragmentos_data:
            fragmentos_data[chunk_id]['score_keyword'] += 1.0 / (idx + cfg.RRF_K)
            if 'BM25' not in fragmentos_data[chunk_id]['matches']:
                fragmentos_data[chunk_id]['matches'].append('BM25')
        else:
            fragmentos_data[chunk_id] = {
                'doc': result['doc'],
                'metadata': result['metadata'],
                'distancia': result['distancia'],
                'id': chunk_id,
                'score_semantic': 0.0,
                'score_keyword': 1.0 / (idx + cfg.RRF_K),
                'matches': ['BM25'],
                'query_matches': []
            }

    for frag in fragmentos_data.values():
        frag['score_final'] = (
            frag['score_semantic'] * cfg.PESO_SEMANTICO_RRF
            + frag['score_keyword'] * cfg.PESO_BM25_RRF
        )

    fragmentos_ranked = sorted(
        fragmentos_data.values(),
        key=lambda x: x['score_final'],
        reverse=True
    )

    metricas_totales['candidatos_fusion'] = len(fragmentos_ranked)

    if cfg.USAR_RERANKER and fragmentos_ranked:
        n_candidatos = min(cfg.TOP_K_RERANK_CANDIDATES, len(fragmentos_ranked))
        ui.debug(f"reranking top {n_candidatos} candidates...")

        candidatos_rerank = fragmentos_ranked[:cfg.TOP_K_RERANK_CANDIDATES]
        fragmentos_ranked, metricas_rerank = cfg.rerank_resultados(
            pregunta,
            candidatos_rerank,
            top_k=cfg.TOP_K_FINAL
        )
        metricas_totales['fase_reranking'] = metricas_rerank
        ui.debug(f"top {len(fragmentos_ranked)} after reranking")

    mejor_score = fragmentos_ranked[0]['score_final'] if fragmentos_ranked else 0
    metricas_totales['resultados_finales'] = len(fragmentos_ranked)

    metricas_totales['sub_queries'] = llm_queries
    metricas_totales['queries_semanticas'] = queries
    metricas_totales['keywords'] = list(keywords_expandidas)

    if cfg.LOGGING_METRICAS:
        sem_unicos = metricas_totales['fase_semantica'].get('fragmentos_unicos', 0)
        kw_total = metricas_keywords.get('resultados_totales', 0)
        logging.info(
            f"Full pipeline: Semantic({sem_unicos}) + "
            f"BM25({kw_total}) -> "
            f"Fusion({metricas_totales['candidatos_fusion']}) -> "
            f"Reranking({metricas_totales['resultados_finales']})"
        )

    return fragmentos_ranked, mejor_score, metricas_totales




