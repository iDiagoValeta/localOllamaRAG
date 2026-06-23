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
import requests
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from rag.engine.runtime import get_runtime

cfg = get_runtime()
OLLAMA_BASE_URL = "http://localhost:11434"


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
    }
    if system:
        payload["system"] = system
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=cfg.OLLAMA_REQUEST_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("done") and data.get("done_reason") not in (None, "stop"):
                    logging.warning("RAG generator stopped early: done_reason=%s", data.get("done_reason"))
                yield data


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
    system = cfg.SYSTEM_PROMPT_RAG if cfg._modelo_necesita_system_prompt(cfg.MODELO_RAG) else None
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
        system=system,
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


def _score_relevancia_fragmento(fragmento: Dict[str, Any]) -> float:
    """Return the active relevance score for post-reranker filtering."""
    try:
        return float(fragmento.get("score_reranker", fragmento.get("score_final", 0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _filtrar_por_umbral_reranker(
    fragmentos_ranked: List[Dict[str, Any]],
    permitir_fallback_bajo_score: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Apply the single reranker relevance threshold."""
    if not cfg.USAR_RERANKER:
        return list(fragmentos_ranked), False

    fragmentos_filtrados = [
        f for f in fragmentos_ranked
        if _score_relevancia_fragmento(f) >= cfg.UMBRAL_SCORE_RERANKER
    ]
    if fragmentos_filtrados or not permitir_fallback_bajo_score:
        return fragmentos_filtrados, False
    return list(fragmentos_ranked), True


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
    """Apply the retrieved-evidence character budget before context synthesis."""
    contexto_total = sum(len(f["doc"]) for f in fragmentos)
    if contexto_total <= cfg.MAX_CONTEXTO_CHARS:
        return fragmentos, 0

    fragmentos_truncados = []
    chars_acum = 0
    for f in fragmentos:
        if chars_acum + len(f["doc"]) > cfg.MAX_CONTEXTO_CHARS:
            break
        fragmentos_truncados.append(f)
        chars_acum += len(f["doc"])
    return fragmentos_truncados, len(fragmentos) - len(fragmentos_truncados)


def preparar_fragmentos_para_generacion(
    fragmentos_ranked: List[Dict[str, Any]],
    collection: chromadb.Collection,
    permitir_fallback_bajo_score: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert ranked retrieval candidates into the final generator evidence."""
    candidatos, fallback_bajo_score = _filtrar_por_umbral_reranker(
        fragmentos_ranked,
        permitir_fallback_bajo_score=permitir_fallback_bajo_score,
    )
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
        "fallback_bajo_score": fallback_bajo_score,
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
            permitir_fallback_bajo_score=cfg.EVAL_RAGBENCH_RERANKER_LOW_SCORE_FALLBACK,
        )
        if not fragmentos_finales:
            return ("", [])
        respuesta = cfg.generar_respuesta_silenciosa(pregunta, fragmentos_finales)
        contexts = [f['doc'] for f in fragmentos_finales]
        return (respuesta, contexts)




