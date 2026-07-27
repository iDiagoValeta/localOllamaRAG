"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

import contextlib
import io
import json
import logging
import time
import requests
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from monkeygrab.application.answer import _limit_by_char_budget
from monkeygrab.application.retrieve import _filter_by_reranker_threshold
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment
from rag.engine.runtime import get_runtime

cfg = get_runtime()
OLLAMA_BASE_URL = "http://localhost:11434"

# Placeholder metadata for the two threshold/budget filters below: both only
# ever read score fields or `doc` length, never metadata, but Fragment
# requires a ChunkMetadata to construct. Shared frozen instance, safe to
# reuse across calls.
_METADATA_MARCADOR = ChunkMetadata(source="", page=0)


def _modelos_ollama_configurados() -> set[str]:
    return {
        m for m in (
            cfg.MODELO_RAG,
            cfg.MODELO_CHAT,
            cfg.MODELO_EMBEDDING,
            cfg.MODELO_CONTEXTUAL,
            cfg.MODELO_RECOMP,
            cfg.MODELO_OCR,
        ) if m
    }


def liberar_modelos_ollama(excepto: str | None = None) -> None:
    """Ask Ollama to drop loaded weights (keep_alive=0) so the next model can fit."""
    for nombre in _modelos_ollama_configurados():
        if excepto and nombre == excepto:
            continue
        try:
            requests.post(
                f"{cfg.OLLAMA_BASE_URL}/api/generate",
                json={"model": nombre, "keep_alive": 0},
                timeout=30,
            )
        except Exception as exc:
            logging.debug("Ollama unload %s: %s", nombre, exc)


def _ollama_generate_stream(model: str, prompt: str, options: dict, system: Optional[str] = None):
    """Stream tokens from the Ollama ``/api/generate`` endpoint.

    ``think=False`` avoids spending ``num_predict`` on a reasoning trace when
    ``MODELO_RAG`` (or any substitute) is a thinking model.

    Args:
        model: Ollama model name.
        prompt: User prompt.
        options: Generation options (temperature, top_p, etc.).
        system: Optional system prompt. Pass ``None`` for fine-tuned models
            that already have the system prompt baked into their Modelfile.

    Yields:
        Parsed JSON objects from each streamed line.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": options,
        "keep_alive": cfg.OLLAMA_KEEP_ALIVE,
    }
    if system:
        payload["system"] = system

    liberar_modelos_ollama(excepto=model)

    url = f"{cfg.OLLAMA_BASE_URL}/api/generate"
    attempts = max(1, cfg.OLLAMA_GENERATE_RETRIES)
    for attempt in range(attempts):
        try:
            with requests.post(
                url,
                json=payload,
                stream=True,
                timeout=cfg.OLLAMA_REQUEST_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("done") and data.get("done_reason") not in (None, "stop"):
                            logging.warning(
                                "RAG generator stopped early: done_reason=%s",
                                data.get("done_reason"),
                            )
                        yield data
                return
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status >= 500 and attempt + 1 < attempts:
                logging.warning(
                    "Ollama generate %s HTTP %s (attempt %d/%d); unloading and retrying",
                    model,
                    status,
                    attempt + 1,
                    attempts,
                )
                liberar_modelos_ollama()
                time.sleep(cfg.OLLAMA_GENERATE_RETRY_DELAY)
                continue
            raise


def _preparar_mensaje_usuario_rag(pregunta: str, fragmentos: List[Dict[str, Any]]) -> str:
    """Build the user message sent to the generator: question + <context>."""
    if cfg.USAR_RECOMP_SYNTHESIS:
        contexto_str = cfg.sintetizar_contexto_recomp(fragmentos, query_usuario=pregunta)
    else:
        contexto_str = cfg.construir_contexto_para_modelo(fragmentos)
    return f"{pregunta}\n\n<context>{contexto_str}</context>"


_GEN_STATS_KEYS = (
    "model", "done_reason", "total_duration", "load_duration",
    "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
)


def generar_tokens_respuesta(mensaje_usuario: str, stats: Optional[Dict[str, Any]] = None):
    """Yield RAG generator tokens using the canonical generation settings.

    When ``stats`` is provided, the timing and token-count fields of the final
    Ollama ``done`` chunk are copied into it so the debug dump can report
    prompt/eval token counts and decoding speed.
    """
    for chunk in cfg._ollama_generate_stream(
        model=cfg.MODELO_RAG,
        prompt=mensaje_usuario,
        options={
            "temperature": 0.15,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "num_predict": -1,
            "num_ctx": cfg.OLLAMA_RAG_NUM_CTX,
        },
        system=cfg.SYSTEM_PROMPT_RAG,
    ):
        content = chunk.get("response", "")
        if content:
            yield content
        if stats is not None and chunk.get("done"):
            for k in _GEN_STATS_KEYS:
                if k in chunk:
                    stats[k] = chunk[k]


def _generar_respuesta_stream(mensaje_usuario: str, on_token=None, stats: Optional[Dict[str, Any]] = None) -> str:
    """Stream the generator response, optionally forwarding tokens to a callback."""
    respuesta_completa = ""
    for content in cfg.generar_tokens_respuesta(mensaje_usuario, stats=stats):
        respuesta_completa += content
        if on_token is not None:
            on_token(content)
    return respuesta_completa


def _filtrar_por_umbral_reranker(
    fragmentos_ranked: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply the single reranker relevance threshold.

    Implementation lives in monkeygrab.application.retrieve
    (_filter_by_reranker_threshold / _relevance_score); equivalence-tested
    against this function in tests/unit/application/
    test_retrieve_threshold_equivalence.py. The dict-based signature is kept
    because rag/chat_pdfs.py's public API is frozen (CLAUDE.md rule 7): each
    dict is converted to a throwaway Fragment carrying only the two score
    fields the filter reads, and the ORIGINAL dicts -- not reconstructions
    -- are returned, matched back to the survivors by object identity so
    every other key (doc, metadata, matches, ...) passes through untouched.
    """
    fragmentos_dominio = [
        Fragment(
            doc=f.get("doc", ""),
            metadata=_METADATA_MARCADOR,
            score_reranker=f.get("score_reranker"),
            score_final=f.get("score_final", 0.0),
        )
        for f in fragmentos_ranked
    ]
    conservados = _filter_by_reranker_threshold(
        fragmentos_dominio, cfg.USAR_RERANKER, cfg.UMBRAL_SCORE_RERANKER
    )
    conservados_ids = {id(f) for f in conservados}
    return [
        original
        for original, dominio in zip(fragmentos_ranked, fragmentos_dominio)
        if id(dominio) in conservados_ids
    ]


def _fragmento_expandible(fragmento: Dict[str, Any]) -> bool:
    """Return True when neighbor expansion makes sense for this fragment."""
    metadata = fragmento.get("metadata", {})
    if metadata.get("format") == "image":
        return False
    return "chunk" in metadata and metadata.get("total_chunks_in_page") is not None


def _expandir_fragmentos_contexto(
    fragmentos: List[Dict[str, Any]],
    collection: chromadb.Collection,
) -> Tuple[List[Dict[str, Any]], int]:
    """Append adjacent chunks for the top selected text fragments."""
    if not cfg.EXPANDIR_CONTEXTO or not fragmentos:
        return fragmentos, 0

    fragmentos_expandidos = list(fragmentos)
    ids_usados = {f.get("id") for f in fragmentos_expandidos}
    chunks_adicionales = []

    for frag in fragmentos_expandidos[:cfg.N_TOP_PARA_EXPANSION]:
        if not _fragmento_expandible(frag):
            continue
        ids_vecinos = cfg.expandir_con_chunks_adyacentes(
            frag["id"], frag["metadata"], n_vecinos=1
        )
        if not ids_vecinos:
            continue
        try:
            vecinos = collection.get(ids=ids_vecinos, include=["documents", "metadatas"])
            for v_doc, v_meta in zip(vecinos["documents"], vecinos["metadatas"]):
                v_id = (
                    f"{v_meta['source']}_pag{v_meta['page']}"
                    f"_chunk{v_meta.get('chunk', 0)}"
                )
                if v_id not in ids_usados:
                    chunks_adicionales.append({
                        "doc": v_doc,
                        "metadata": v_meta,
                        "distancia": float("inf"),
                        "score_final": 0.0,
                        "id": v_id,
                    })
                    ids_usados.add(v_id)
        except Exception:
            pass

    if chunks_adicionales:
        fragmentos_expandidos.extend(chunks_adicionales)
    return fragmentos_expandidos, len(chunks_adicionales)


def _limitar_fragmentos_por_chars(
    fragmentos: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Apply the retrieved-evidence character budget before context synthesis.

    Implementation lives in monkeygrab.application.answer
    (_limit_by_char_budget); equivalence-tested against this function in
    tests/unit/application/test_answer_char_budget_equivalence.py. The
    budget only ever keeps a PREFIX of the input (it walks fragments in
    order and stops at the first one that would overflow), so the kept
    original dicts are recovered by taking the same-length prefix of
    `fragmentos` rather than reconstructing them from the throwaway
    Fragments built just to carry `doc` length.
    """
    fragmentos_dominio = [
        Fragment(doc=f["doc"], metadata=_METADATA_MARCADOR) for f in fragmentos
    ]
    conservados, n_descartados = _limit_by_char_budget(fragmentos_dominio, cfg.MAX_CONTEXTO_CHARS)
    return fragmentos[:len(conservados)], n_descartados


def preparar_fragmentos_para_generacion(
    fragmentos_ranked: List[Dict[str, Any]],
    collection: chromadb.Collection,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert ranked retrieval candidates into the final generator evidence."""
    candidatos = _filtrar_por_umbral_reranker(fragmentos_ranked)
    fragmentos_base = list(candidatos[:cfg.TOP_K_FINAL])
    fragmentos_expandidos, n_expandidos = _expandir_fragmentos_contexto(
        fragmentos_base, collection
    )
    fragmentos_finales, n_descartados_chars = _limitar_fragmentos_por_chars(
        fragmentos_expandidos
    )

    metricas = {
        "candidatos_entrada": len(fragmentos_ranked),
        "candidatos_relevantes": len(candidatos),
        "fragmentos_base": len(fragmentos_base),
        "fragmentos_expandidos": n_expandidos,
        "fragmentos_descartados_por_chars": n_descartados_chars,
        "fragmentos_finales": len(fragmentos_finales),
        "max_contexto_chars": cfg.MAX_CONTEXTO_CHARS,
    }
    return fragmentos_finales, metricas


def generar_respuesta(
    pregunta: str,
    fragmentos: List[Dict[str, Any]],
    metricas: Optional[Dict[str, Any]] = None,
    on_token=None,
) -> str:
    """Generate a RAG response and optionally stream tokens through a callback.

    Builds context from fragments, streams the response via Ollama,
    optionally emits visible tokens through ``on_token``, and saves a
    debug dump.

    Args:
        pregunta: User question.
        fragmentos: Retrieved context fragments.
        metricas: Pipeline metrics for debug logging.
        on_token: Optional callable receiving each streamed token.

    Returns:
        Complete response text.
    """
    import time
    mensaje_usuario = cfg._preparar_mensaje_usuario_rag(pregunta, fragmentos)
    gen_stats: Dict[str, Any] = {}
    inicio = time.time()
    respuesta_completa = cfg._generar_respuesta_stream(
        mensaje_usuario, on_token=on_token, stats=gen_stats
    )
    if metricas is not None:
        fase_gen = dict(gen_stats)
        fase_gen["wall_time_s"] = time.time() - inicio
        fase_gen["respuesta_chars"] = len(respuesta_completa)
        eval_count = gen_stats.get("eval_count")
        eval_duration = gen_stats.get("eval_duration")
        if eval_count and eval_duration:
            fase_gen["tokens_por_segundo"] = eval_count / (eval_duration / 1e9)
        metricas = {**metricas, "fase_generacion": fase_gen}
    cfg.guardar_debug_rag(pregunta, mensaje_usuario, respuesta_completa, fragmentos, metricas=metricas)
    return respuesta_completa


def generar_respuesta_silenciosa(pregunta: str, fragmentos: List[Dict[str, Any]], metricas: Optional[Dict[str, Any]] = None) -> str:
    """Generate a RAG response silently (no terminal output).

    Same as ``generar_respuesta`` but without printing or debug dump.

    Args:
        pregunta: User question.
        fragmentos: Retrieved context fragments.
        metricas: Pipeline metrics (unused here, kept for API compat).

    Returns:
        Complete response text.
    """
    mensaje_usuario = cfg._preparar_mensaje_usuario_rag(pregunta, fragmentos)
    return cfg._generar_respuesta_stream(mensaje_usuario)


def evaluar_pregunta_rag(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[str, List[str]]:
    """Run the full RAG pipeline silently for evaluation (RAGAS).

    Performs search, filtering, neighbor expansion, and generation
    without any terminal output.

    Args:
        pregunta: The evaluation question.
        collection: ChromaDB collection to search.

    Returns:
        Tuple of (response text, list of context strings).
    """
    if len(pregunta.strip()) < cfg.MIN_LONGITUD_PREGUNTA_RAG:
        return ("", [])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        fragmentos_ranked, _, _ = cfg.realizar_busqueda_hibrida(pregunta, collection)
        if not fragmentos_ranked:
            return ("", [])
        fragmentos_finales, _ = cfg.preparar_fragmentos_para_generacion(
            fragmentos_ranked,
            collection,
        )
        if not fragmentos_finales:
            return ("", [])
        respuesta = cfg.generar_respuesta_silenciosa(pregunta, fragmentos_finales)
        contexts = [f['doc'] for f in fragmentos_finales]
        return (respuesta, contexts)


if __name__ == "__main__":
    assert _modelos_ollama_configurados()
    liberar_modelos_ollama()

