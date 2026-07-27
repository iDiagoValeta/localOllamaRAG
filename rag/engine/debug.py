"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from rag.engine.runtime import get_runtime

cfg = get_runtime()


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

    if not cfg.GUARDAR_DEBUG_RAG:
        return

    try:
        os.makedirs(cfg.CARPETA_DEBUG_RAG, exist_ok=True)

        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r'[^\w\s]', '', pregunta)[:40].strip().replace(' ', '_')
        nombre_archivo = f"{timestamp}_{slug}.txt"
        ruta = os.path.join(cfg.CARPETA_DEBUG_RAG, nombre_archivo)

        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"  DEBUG RAG - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            if metricas and (metricas.get('sub_queries') or metricas.get('queries_semanticas') or metricas.get('keywords') or metricas.get('fase_semantica') or metricas.get('fase_keywords') or metricas.get('fase_reranking')):
                f.write("─" * 80 + "\n")
                f.write("  RETRIEVAL PIPELINE (sub-queries, keywords, BM25, metrics)\n")
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
                fase_kw = metricas.get('fase_keywords', {})
                if fase_kw:
                    f.write(f"\nBM25 metrics: {fase_kw.get('documentos_indexados', 0)} docs indexed, {fase_kw.get('terminos_query', 0)} query terms, {fase_kw.get('resultados_totales', 0)} results (top score {fase_kw.get('mejor_score', 0.0):.2f})\n")
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
            f.write(f"RAG Model: {cfg._inferir_descripcion_modelo(cfg.MODELO_RAG)}\n")
            f.write(f"Contextual Retrieval (Indexing): {'YES' if cfg.USAR_CONTEXTUAL_RETRIEVAL else 'NO'}\n")
            f.write(f"Query Decomposition: {'YES' if cfg.USAR_LLM_QUERY_DECOMPOSITION else 'NO'}\n")
            f.write(f"Hybrid Search (BM25): {'YES' if cfg.USAR_BUSQUEDA_HIBRIDA else 'NO'}\n")
            f.write(f"Reranker: {'YES' if cfg.USAR_RERANKER else 'NO'}\n")
            f.write(f"Expand Context: {'YES' if cfg.EXPANDIR_CONTEXTO else 'NO'}\n")
            f.write(f"Optimize Context: {'YES' if cfg.USAR_OPTIMIZACION_CONTEXTO else 'NO'}\n")
            f.write(f"RECOMP Synthesis: {'YES' if cfg.USAR_RECOMP_SYNTHESIS else 'NO'}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  ORIGINAL QUESTION\n")
            f.write("─" * 80 + "\n")
            f.write(f"{pregunta}\n\n")

            f.write("─" * 80 + "\n")
            f.write("  SYSTEM PROMPT\n")
            f.write("─" * 80 + "\n")
            f.write(f"{cfg.SYSTEM_PROMPT_RAG}\n\n")

            context_match = re.search(r'<context>(.*?)</context>', mensaje_usuario, re.DOTALL)
            contexto_enviado = context_match.group(1).strip() if context_match else "(empty)"

            f.write("─" * 80 + "\n")
            if cfg.USAR_RECOMP_SYNTHESIS:
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
            fragmentos_estructurados = []
            for i, frag in enumerate(fragmentos, 1):
                meta = frag.get('metadata', {})
                score = frag.get('score_final', 'N/A')
                score_rr = frag.get('score_reranker', 'N/A')
                dist = frag.get('distancia', 'N/A')
                doc_text = frag.get('doc', '')
                f.write(f"\n--- Fragment {i} ---\n")
                pag = meta.get('page', 0)
                pag_humana = pag + 1 if isinstance(pag, int) else pag
                f.write(f"Source: {meta.get('source', '?')}, page {pag_humana}\n")
                f.write(f"Final score: {score}  |  Reranker score: {score_rr}\n")
                f.write(f"L2 distance: {dist}  |  Chars: {len(doc_text)}\n")
                matches = frag.get('matches', [])
                if matches:
                    f.write(f"Lexical match: {', '.join(matches)}\n")
                query_matches = frag.get('query_matches', [])
                if query_matches:
                    f.write(f"Matched query(s): {query_matches}\n")
                f.write(f"Section: {meta.get('section_header', '(no header)')}\n")
                if '\\n\\n' in doc_text:
                    ctx_part, orig_part = doc_text.split('\\n\\n', 1)
                    f.write(f"[Contextual Retrieval]:\n{ctx_part}\n\n")
                    f.write(f"[Document text]:\n{orig_part}\n")
                else:
                    f.write(f"[Document text]:\n{doc_text}\n")
                fragmentos_estructurados.append({
                    'orden': i,
                    'source': meta.get('source', '?'),
                    'page': pag_humana,
                    'section': meta.get('section_header', ''),
                    'format': meta.get('format', 'text'),
                    'score_final': score if score != 'N/A' else None,
                    'score_reranker': score_rr if score_rr != 'N/A' else None,
                    'distancia_l2': dist if dist != 'N/A' else None,
                    'matches': matches,
                    'query_matches': query_matches,
                    'chars': len(doc_text),
                })

        # Structured sidecar: same payload in machine-readable form so the trace
        # can be inspected or rendered without parsing the prose dump.
        try:
            sidecar = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'question': pregunta,
                'config': {
                    'modelo_rag': cfg._inferir_descripcion_modelo(cfg.MODELO_RAG),
                    'modelo_rag_tag': cfg.MODELO_RAG,
                    'modelo_embedding': cfg.MODELO_EMBEDDING,
                    'modelo_recomp': cfg.MODELO_RECOMP,
                    'modelo_chat': cfg.MODELO_CHAT,
                    'contextual_retrieval': cfg.USAR_CONTEXTUAL_RETRIEVAL,
                    'query_decomposition': cfg.USAR_LLM_QUERY_DECOMPOSITION,
                    'busqueda_hibrida': cfg.USAR_BUSQUEDA_HIBRIDA,
                    'reranker': cfg.USAR_RERANKER,
                    'expandir_contexto': cfg.EXPANDIR_CONTEXTO,
                    'optimizar_contexto': cfg.USAR_OPTIMIZACION_CONTEXTO,
                    'recomp_synthesis': cfg.USAR_RECOMP_SYNTHESIS,
                    'n_resultados_semanticos': cfg.N_RESULTADOS_SEMANTICOS,
                    'n_resultados_keyword': cfg.N_RESULTADOS_KEYWORD,
                    'top_k_rerank_candidates': cfg.TOP_K_RERANK_CANDIDATES,
                    'top_k_final': cfg.TOP_K_FINAL,
                    'n_top_para_expansion': cfg.N_TOP_PARA_EXPANSION,
                    'rrf_k': cfg.RRF_K,
                    'peso_semantico_rrf': cfg.PESO_SEMANTICO_RRF,
                    'peso_bm25_rrf': cfg.PESO_BM25_RRF,
                    'umbral_score_reranker': cfg.UMBRAL_SCORE_RERANKER,
                    'max_contexto_chars': cfg.MAX_CONTEXTO_CHARS,
                    'reranker_model_quality': cfg.RERANKER_MODEL_QUALITY,
                    'bm25_k1': cfg.BM25_K1,
                    'bm25_b': cfg.BM25_B,
                    'num_ctx_rag': cfg.OLLAMA_RAG_NUM_CTX,
                },
                'metrics': metricas or {},
                'context_sent': contexto_enviado,
                'answer': respuesta,
                'answer_chars': len(respuesta or ''),
                'fragments': fragmentos_estructurados,
            }
            ruta_json = os.path.splitext(ruta)[0] + '.json'
            with open(ruta_json, 'w', encoding='utf-8') as fj:
                json.dump(sidecar, fj, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logging.warning(f"Error saving debug RAG JSON sidecar: {e}")

        logging.info(f"Debug RAG saved: {ruta}")

    except Exception as e:
        logging.warning(f"Error saving debug RAG: {e}")


