"""Answer generation entry points for the CLI and the web app.

Wires the port adapters and runs ``monkeygrab.application.answer.Answer``, then
converts between the domain fragments the core uses and the dicts the
interfaces have always consumed.

The three steps stay three separate functions rather than one call because a
caller streaming to a web client has work to do between them: it sends the
cited sources before the first token arrives, which it cannot do if preparing
the evidence and generating from it are the same call.
"""

from typing import Any, Dict, List, Optional, Tuple

from rag.engine import wiring
from monkeygrab.config.env import read_env_ollama_base_url
from monkeygrab.ports.vector_store import VectorStore
from rag.engine.runtime import get_runtime

cfg = get_runtime()

# The generator's HTTP endpoint, re-exported by rag/chat_pdfs.py because the
# CLI health check and the web control panel both report Ollama's reachability
# from it. Resolved through the same reader AppConfig uses for
# models.ollama.base_url, so what those two report is the server the adapters
# actually send generation to -- it used to be a literal, which made setting
# OLLAMA_BASE_URL move the diagnostic without moving the traffic.
OLLAMA_BASE_URL = read_env_ollama_base_url()


class _NoCollection:
    """Stand-in for the collection when the step about to run has no store access.

    Building the user message and streaming never touch the vector store --
    neighbour expansion already happened, in an earlier call that did have the
    collection. The use case still takes the port for the whole of its work, and
    the public API of these two functions has no collection to give it.

    Raising on any attribute access keeps that honest: if either step ever
    starts reading the store, it fails loudly here instead of silently querying
    whatever collection happened to be at hand.
    """

    def __getattr__(self, name):
        raise RuntimeError(
            f"this step must not reach the vector store (accessed {name!r})"
        )


_NO_COLLECTION = _NoCollection()


def preparar_fragmentos_para_generacion(
    fragmentos_ranked: List[Dict[str, Any]],
    collection: VectorStore,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Turn ranked retrieval candidates into the evidence the generator sees.

    Expands the top fragments with their neighbouring chunks and trims the
    result to the character budget.

    Args:
        fragmentos_ranked: Ranked fragments from ``realizar_busqueda_hibrida``.
        collection: FAISS store used to fetch neighbour chunks.

    Returns:
        Tuple of (evidence fragments, metrics).
    """
    config = wiring.app_config_from_runtime()
    use_case = wiring.answer(collection, config)

    fragments = [wiring.fragment_from_dict(f) for f in fragmentos_ranked]
    seleccionados, metrics = use_case.select_evidence(fragments)
    fragmentos_finales = [wiring.fragment_to_dict(f) for f in seleccionados]

    metricas = {
        "candidatos_entrada": len(fragmentos_ranked),
        "fragmentos_expandidos": metrics["fragments_expanded"],
        "fragmentos_descartados_por_chars": metrics["fragments_discarded_by_chars"],
        "fragmentos_finales": metrics["fragments_final"],
        "max_contexto_chars": config.context.max_context_chars,
    }
    return fragmentos_finales, metricas


def _preparar_mensaje_usuario_rag(
    pregunta: str,
    fragmentos: List[Dict[str, Any]],
    *,
    metricas_out: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the user message sent to the generator: question plus context.

    Args:
        pregunta: User question.
        fragmentos: Evidence from ``preparar_fragmentos_para_generacion``.
        metricas_out: When given, receives whether RECOMP synthesis was used
            (``recomp_used``) and whether it failed and fell back to the raw
            context (``recomp_error``). The use case already reports both;
            without this out-parameter they were dropped between it and the
            evaluation gate, which is the only place that can count fallbacks
            (issue #497).

    Returns:
        The complete user message.
    """
    config = wiring.app_config_from_runtime()
    use_case = wiring.answer(_NO_COLLECTION, config)

    fragments = [wiring.fragment_from_dict(f) for f in fragmentos]
    mensaje, metrics = use_case.build_user_message(pregunta, fragments)
    if metricas_out is not None:
        # Both keys always land: "not used" and "not an error" are real
        # observations, and an absent key would be indistinguishable from a
        # caller that never ran the context stage.
        metricas_out["recomp_used"] = metrics.get("recomp_used", False)
        metricas_out["recomp_error"] = metrics.get("recomp_error", False)
    return mensaje


def generar_tokens_respuesta(mensaje_usuario: str, stats: Optional[Dict[str, Any]] = None):
    """Yield the generator's answer token by token.

    Args:
        mensaje_usuario: Output of ``_preparar_mensaje_usuario_rag``.
        stats: When given, the timing and token counts of the final streamed
            chunk are copied into it, so the debug dump can report decoding
            speed.

    Yields:
        Answer text, one streamed chunk at a time.
    """
    config = wiring.app_config_from_runtime()
    use_case = wiring.answer(_NO_COLLECTION, config)

    for chunk in use_case.stream(mensaje_usuario):
        if chunk.text:
            yield chunk.text
        if stats is not None and chunk.done:
            stats.update(_generation_stats(chunk))


_GEN_STATS_FIELDS = (
    "model", "done_reason", "total_duration", "load_duration",
    "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
)


def _generation_stats(chunk) -> Dict[str, Any]:
    """Copy the fields Ollama reported on the final chunk, skipping the absent ones."""
    return {
        field: value
        for field, value in ((f, getattr(chunk, f)) for f in _GEN_STATS_FIELDS)
        if value is not None
    }


def _generar_respuesta_stream(mensaje_usuario: str, on_token=None, stats: Optional[Dict[str, Any]] = None) -> str:
    """Stream the generator response, optionally forwarding tokens to a callback."""
    respuesta_completa = ""
    for content in cfg.generar_tokens_respuesta(mensaje_usuario, stats=stats):
        respuesta_completa += content
        if on_token is not None:
            on_token(content)
    return respuesta_completa


def generar_respuesta_silenciosa(
    pregunta: str,
    fragmentos: List[Dict[str, Any]],
    metricas: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a RAG response without terminal output or a debug dump.

    Args:
        pregunta: User question.
        fragmentos: Retrieved context fragments.
        metricas: Pipeline metrics (unused here, kept for API compatibility).
        stats: Optional dict filled in place with what Ollama reported on the
            final chunk -- ``eval_count``, ``eval_duration``, ``load_duration``
            and the rest of ``_GEN_STATS_FIELDS``. Appended as a keyword rather
            than folded into ``metricas`` because this function's signature is
            part of the public API (``AGENTS.md`` rule 7) and every existing
            caller passes ``metricas`` positionally.

            The streaming path has always collected these; only the silent one
            dropped them, which is why the evaluation gate -- the one caller
            that most needs to compare generators -- could measure seconds per
            case but not tokens per second.

            Also carries ``stage`` while the call is in flight: ``"context"``
            during message preparation (RECOMP synthesis on the auxiliary
            model, when that flag is on) and ``"generator"`` once the model
            under test is streaming. The gate bounds this whole call with one
            wall-clock budget, and without the marker a cut could not say
            which of the two it cut (issue #234).

            Message preparation also writes ``recomp_used`` and
            ``recomp_error`` here (issue #497), so the artifact can count the
            cases where RECOMP overflowed and the raw context was used
            instead. ``recomp_error`` is always present, ``False`` when the
            synthesis did not fail.

    Returns:
        Complete response text.
    """
    if stats is not None:
        stats["stage"] = "context"
        # Keyword, and only when there is a dict to fill: existing callers
        # and test doubles of this function take two positional arguments.
        mensaje_usuario = cfg._preparar_mensaje_usuario_rag(
            pregunta, fragmentos, metricas_out=stats
        )
    else:
        mensaje_usuario = cfg._preparar_mensaje_usuario_rag(pregunta, fragmentos)
    if stats is not None:
        stats["stage"] = "generator"
    return cfg._generar_respuesta_stream(mensaje_usuario, stats=stats)
