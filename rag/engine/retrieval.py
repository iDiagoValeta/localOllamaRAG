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

from monkeygrab.application.rrf_fusion import fuse_semantic_and_keyword
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment
from rag.cli.display import ui
from rag.engine.runtime import get_runtime

def _embedding_keep_alive(q_idx: int, n_queries: int):
    # ponytail: unload embed model after the last query so the RAG generator fits in VRAM
    return 0 if q_idx >= n_queries - 1 else None


if __name__ == "__main__":
    assert _embedding_keep_alive(0, 3) is None
    assert _embedding_keep_alive(2, 3) == 0
    raise SystemExit(0)

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
    embedding_dim = 0
    resultados_por_query: List[int] = []
    # Raw per-query-variant hits (not merged), fed to the RRF fusion delegate
    # below (monkeygrab.application.rrf_fusion.fuse_semantic_and_keyword) --
    # kept alongside all_semantic_results rather than derived from it, so
    # fase_semantica's dedup-by-min-distance metrics stay computed exactly as
    # before.
    semantic_hits_por_query: List[List[Fragment]] = []

    for q_idx, query in enumerate(queries):
        query_con_prefijo = f"{cfg.EMBED_PREFIX_QUERY}{query}"
        response_emb = ollama.embeddings(
            model=cfg.MODELO_EMBEDDING,
            prompt=query_con_prefijo,
            keep_alive=_embedding_keep_alive(q_idx, len(queries)),
        )
        if not embedding_dim:
            embedding_dim = len(response_emb["embedding"])

        results_semantic = collection.query(
            query_embeddings=[response_emb["embedding"]],
            n_results=cfg.N_RESULTADOS_SEMANTICOS,
            include=['documents', 'distances', 'metadatas']
        )
        resultados_por_query.append(len(results_semantic['documents'][0]))

        hits_query: List[Fragment] = []
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

            hits_query.append(Fragment(
                doc=doc,
                metadata=ChunkMetadata(
                    source=metadata['source'], page=metadata['page'], chunk=metadata.get('chunk', 0)
                ),
                distancia=distancia,
            ))
        semantic_hits_por_query.append(hits_query)

    distancias = [v['distancia'] for v in all_semantic_results.values()]
    metricas_totales['fase_semantica'] = {
        'queries_generadas': len(queries),
        'fragmentos_unicos': len(all_semantic_results),
        'modelo_embedding': cfg.MODELO_EMBEDDING,
        'dimension_embedding': embedding_dim,
        'n_resultados_por_query': cfg.N_RESULTADOS_SEMANTICOS,
        'resultados_por_query': resultados_por_query,
        'distancia_l2_min': min(distancias) if distancias else 0.0,
        'distancia_l2_max': max(distancias) if distancias else 0.0,
        'distancia_l2_media': sum(distancias) / len(distancias) if distancias else 0.0,
    }

    ui.debug(f"{len(all_semantic_results)} unique fragments")

    results_keyword = []
    metricas_keywords = {}
    if cfg.USAR_BUSQUEDA_HIBRIDA:
        ui.debug("BM25 lexical search...")
        results_keyword, metricas_keywords = cfg.busqueda_lexica_bm25(pregunta, collection)
        metricas_totales['fase_keywords'] = metricas_keywords

    ui.debug("fusing results...")

    # RRF fusion (semantic branch across query variants + BM25 branch) is
    # delegated to monkeygrab.application.rrf_fusion.fuse_semantic_and_keyword
    # -- equivalence-tested against this exact call in
    # tests/unit/application/test_rrf_fusion_equivalence.py and
    # test_rrf_fusion_multi_query_equivalence.py. Fragment/ChunkMetadata only
    # carry source/page/chunk (enough for the RRF math and the chunk_id);
    # the full original 'doc'/'metadata' dicts (every key ChromaDB/BM25
    # attached) are looked up by id afterward instead of being reconstructed
    # from the domain entities, so no key is lost or invented at the
    # boundary.
    doc_y_metadata_por_id: Dict[str, Dict[str, Any]] = {
        chunk_id: {'doc': data['doc'], 'metadata': data['metadata']}
        for chunk_id, data in all_semantic_results.items()
    }

    keyword_hits: List[Fragment] = []
    for result in results_keyword:
        cid = result['id']
        if cid not in doc_y_metadata_por_id:
            doc_y_metadata_por_id[cid] = {'doc': result['doc'], 'metadata': result['metadata']}
        keyword_hits.append(Fragment(
            doc=result['doc'],
            metadata=ChunkMetadata(
                source=result['metadata']['source'],
                page=result['metadata']['page'],
                chunk=result['metadata'].get('chunk', 0),
            ),
            distancia=result['distancia'],
        ))

    fused = fuse_semantic_and_keyword(
        semantic_hits_per_query=semantic_hits_por_query,
        keyword_hits=keyword_hits,
        rrf_k=cfg.RRF_K,
        weight_semantic=cfg.PESO_SEMANTICO_RRF,
        weight_keyword=cfg.PESO_BM25_RRF,
    )

    fragmentos_ranked = [
        {
            'doc': doc_y_metadata_por_id[f.id]['doc'],
            'metadata': doc_y_metadata_por_id[f.id]['metadata'],
            'distancia': f.distancia,
            'id': f.id,
            'score_semantic': f.score_semantic,
            'score_keyword': f.score_keyword,
            'matches': list(f.matches),
            'query_matches': list(f.query_matches),
            'score_final': f.score_final,
        }
        for f in fused
    ]

    metricas_totales['candidatos_fusion'] = len(fragmentos_ranked)
    solo_semantica = sum(1 for f in fragmentos_ranked if f['score_semantic'] > 0 and f['score_keyword'] == 0)
    solo_lexica = sum(1 for f in fragmentos_ranked if f['score_keyword'] > 0 and f['score_semantic'] == 0)
    ambas_ramas = sum(1 for f in fragmentos_ranked if f['score_semantic'] > 0 and f['score_keyword'] > 0)
    metricas_totales['fase_fusion'] = {
        'peso_semantico': cfg.PESO_SEMANTICO_RRF,
        'peso_lexico': cfg.PESO_BM25_RRF,
        'rrf_k': cfg.RRF_K,
        'candidatos_totales': len(fragmentos_ranked),
        'solo_semantica': solo_semantica,
        'solo_lexica': solo_lexica,
        'ambas_ramas': ambas_ramas,
    }

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

