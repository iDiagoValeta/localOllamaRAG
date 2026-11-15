"""Hybrid retrieval entry point for the web app.

Wires the port adapters and runs ``monkeygrab.application.retrieve.Retrieve``,
then converts its output into the dicts the interfaces have always consumed.
No retrieval logic lives here: the evaluation gate constructs the same use
case directly, so what is measured there is what ships.
"""

import logging
from typing import Any, Dict, List, Tuple

from monkeygrab.application.retrieve import Retrieve
from monkeygrab.ports.vector_store import VectorStore
from rag.engine import wiring


def realizar_busqueda_hibrida(
    pregunta: str,
    collection: VectorStore
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Retrieve and rank the evidence for a question.

    Runs query decomposition, semantic and BM25 search, RRF fusion, reranking
    and the relevance-threshold filter, subject to the pipeline flags in
    ``rag.chat_pdfs``. A failing query decomposition degrades to searching
    with the original question alone; every other stage raises on failure,
    per the project-wide hard-fail policy, rather than silently returning
    worse results.

    Args:
        pregunta: User query.
        collection: FAISS store to search.

    Returns:
        Tuple of (ranked fragments, best score, metrics dict). The fragment
        list is already threshold-filtered and truncated to ``TOP_K_FINAL``.
    """
    logging.debug("Starting hybrid search...")

    config = wiring.app_config_from_runtime()
    store = collection

    use_case = Retrieve(
        wiring.embedder(config),
        store,
        config,
        lexical_index=wiring.lexical_index(store, config) if config.flags.usar_busqueda_hibrida else None,
        reranker=wiring.room_making_reranker(config) if config.flags.usar_reranker else None,
        query_decomposer=wiring.query_decomposer(config) if config.flags.usar_llm_query_decomposition else None,
    )

    # The reranker and the generator share the card when the measured free
    # memory lets them (issue #630): the reranker unloads Ollama's models
    # first only if it would not fit, and stays loaded after retrieval only
    # if the generator is already resident on the GPU beside it. Otherwise
    # they take the card in turn (issue #584): kept resident before the
    # generator loaded, the reranker left it on the CPU.
    finished = False
    try:
        result = use_case.run(pregunta)
        finished = True
    finally:
        # Also when the run raises (a CUDA OOM in the reranker itself): left
        # resident, the reranker kept ~2 GB and every later query failed too.
        if config.flags.usar_reranker and not (
            finished and wiring.generator_resident_on_gpu(config)
        ):
            wiring.release_reranker()
            # A generator split across GPU and CPU stays split until it is
            # reloaded, so it goes too and the answer loads it onto the
            # card the reranker just freed.
            if finished:
                wiring.release_ollama_models(config)

    fragmentos_ranked = [wiring.fragment_to_dict(f) for f in result.fragments]
    metricas = wiring.retrieval_metrics_to_legacy(result.metrics, config)
    mejor_score = fragmentos_ranked[0]["score_final"] if fragmentos_ranked else 0

    logging.debug(f"{len(fragmentos_ranked)} fragment(s) after ranking")

    if config.flags.logging_metricas:
        logging.info(
            "Retrieval: %d query variant(s) -> fusion(%d) -> final(%d)",
            len(result.metrics["query_variants"]),
            result.metrics["fused_candidates"],
            result.metrics["final_count"],
        )

    return fragmentos_ranked, mejor_score, metricas
