"""Answer -- neighbor expansion (opt.) -> char budget -> context synthesis
(opt.) -> generation.

Takes the fragments ``Retrieve`` already ranked and threshold-filtered,
expands them with neighboring chunks, trims to the character budget,
optionally synthesizes a facts briefing via RECOMP instead of feeding raw
chunks, and streams the final generation.

``run()`` consumes ``ChatModel.stream()``'s ``GenerationChunk`` items directly
rather than plain text: it appends ``chunk.text`` to the answer (and forwards
it to ``on_token``, for a caller streaming to a web client) while it goes,
and once the final chunk arrives (``chunk.done``) it copies the generation
metadata into ``AnswerResult.metrics["generation"]`` -- the same fields and
``tokens_per_second`` derivation ``generar_respuesta``'s debug dump computes
today, now reachable from a plain ``ChatModel`` caller instead of requiring
the ``stats`` out-parameter ``generar_tokens_respuesta`` used to thread
through by hand. Persisting the debug dump itself stays a caller concern
(``guardar_debug_rag``), same as before.

RECOMP synthesis (``rag.engine.context.sintetizar_contexto_recomp``) is not
one of the four modules this migration is scoped to move, but it is
reproduced here because ``Answer``'s use-case description explicitly
includes "sintetizar contexto (opcional)" and no equivalent exists anywhere
else -- skipping it would leave the use case unable to reproduce today's
default pipeline (``USAR_RECOMP_SYNTHESIS`` defaults ``True``). Its silent
fallback-to-raw-context on failure is kept as an explicit, visible decision
in ``_synthesize_recomp`` (three separate conditions -- LLM call failure,
empty synthesis, missing header -- each falls back to
``build_context_for_model``), matching the original's behavior; per the
``ChatModel`` port's failure policy this is an allowed "explicit fallback
built by the caller out of two ports", not a silently degrading adapter.
"""

import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from monkeygrab.application.context_assembly import (
    RECOMP_FACTS_HEADER,
    _marcar_fragmento_incompleto,
    _texto_fuente_fragmento,
    build_context_for_model,
    normalize_recomp_output,
    optimize_context_text,
    strip_ollama_think_blocks,
)
from monkeygrab.application.text_chunking import adjacent_chunk_ids
from monkeygrab.config.app_config import AppConfig
from monkeygrab.config.flags import PipelineFlagsConfig
from monkeygrab.domain.fragment import Fragment
from monkeygrab.domain.generation_chunk import GenerationChunk
from monkeygrab.ports.chat_model import ChatModel
from monkeygrab.ports.vector_store import VectorStore


@dataclasses.dataclass
class AnswerResult:
    """Output of ``Answer.run``.

    Attributes:
        text: Complete generated answer.
        metrics: Observability data (fragments expanded/discarded, whether
            RECOMP synthesis ran, context char counts).
    """

    text: str
    metrics: Dict[str, Any]


def _is_expandable(fragment: Fragment) -> bool:
    """Return True when neighbor expansion makes sense for this fragment.

    Moved from ``rag.engine.generation._fragmento_expandible``. The
    original also checks ``"chunk" in metadata`` (legacy dict metadata could
    omit the key); ``ChunkMetadata.chunk`` always has a default of 0, so
    that check is trivially true here and is dropped.
    """
    if fragment.metadata.format == "image":
        return False
    return fragment.metadata.total_chunks_in_page is not None


def _expand_with_neighbors(
    fragments: Sequence[Fragment],
    vector_store: VectorStore,
    n_top_for_expansion: int,
) -> "tuple[List[Fragment], int]":
    """Append adjacent chunks for the top selected text fragments.

    Moved from ``rag.engine.generation._expandir_fragmentos_contexto``.
    Unlike the original's ``try/except Exception: pass`` around
    ``collection.get``, a ``VectorStore`` failure here propagates -- the
    port's failure policy is hard-fail, and unlike ``ChatModel``/``Reranker``
    its docstring does not sanction a caller-level silent-skip fallback for
    this call.

    Args:
        fragments: Fragments to consider for expansion (assumed already
            ranked best-first).
        vector_store: Used to fetch neighbor chunks by id.
        n_top_for_expansion: Only the first this-many fragments are
            considered as expansion anchors.

    Returns:
        Tuple of (fragments with neighbor additions appended, number added).
    """
    if not fragments:
        return list(fragments), 0

    expanded = list(fragments)
    used_ids = {f.id for f in expanded}
    additions: List[Fragment] = []

    for frag in expanded[:n_top_for_expansion]:
        if not _is_expandable(frag):
            continue
        neighbor_ids = adjacent_chunk_ids(frag.metadata, n_neighbors=1)
        if not neighbor_ids:
            continue
        for neighbor in vector_store.get_by_ids(neighbor_ids):
            if neighbor.id not in used_ids:
                additions.append(neighbor)
                used_ids.add(neighbor.id)

    if additions:
        expanded.extend(additions)
    return expanded, len(additions)


def _limit_by_char_budget(
    fragments: Sequence[Fragment], max_context_chars: int
) -> "tuple[List[Fragment], int]":
    """Apply the retrieved-evidence character budget before context synthesis.

    Moved from ``rag.engine.generation._limitar_fragmentos_por_chars``.
    """
    contexto_total = sum(len(f.doc) for f in fragments)
    if contexto_total <= max_context_chars:
        return list(fragments), 0

    fragmentos_truncados = []
    chars_acum = 0
    for f in fragments:
        if chars_acum + len(f.doc) > max_context_chars:
            break
        fragmentos_truncados.append(f)
        chars_acum += len(f.doc)
    return fragmentos_truncados, len(fragments) - len(fragmentos_truncados)


# Field names to copy from a done GenerationChunk into AnswerResult's
# "generation" metrics -- literal match of _GEN_STATS_KEYS in
# rag/engine/generation.py, applied to the typed chunk instead of a raw dict.
_GEN_STATS_FIELDS = (
    "model", "done_reason", "total_duration", "load_duration",
    "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
)


def _generation_stats(chunk: GenerationChunk) -> Dict[str, Any]:
    """Build the "generation" metrics dict from the final streamed chunk.

    Moved from the stats-copying half of ``generar_tokens_respuesta`` (only
    keys actually present in the Ollama ``done`` line are copied -- here,
    fields the adapter left ``None`` because Ollama's line did not report
    them) plus the ``tokens_por_segundo`` derivation from ``generar_respuesta``
    (``rag/engine/generation.py``): both used to be unreachable from any
    ``ChatModel``-only caller once ``stream()`` returned plain text, which is
    exactly the gap this port change closes.
    """
    stats: Dict[str, Any] = {
        field: value
        for field, value in ((f, getattr(chunk, f)) for f in _GEN_STATS_FIELDS)
        if value is not None
    }
    if chunk.eval_count and chunk.eval_duration:
        stats["tokens_per_second"] = chunk.eval_count / (chunk.eval_duration / 1e9)
    return stats


class Answer:
    """Neighbor expansion (opt.) -> char budget -> context synthesis (opt.) -> generation.

    ``recomp_chat_model`` is ``Optional``: when not wired in, RECOMP
    synthesis is skipped regardless of ``flags.usar_recomp_synthesis`` and
    the raw formatted context is used instead -- there is nothing to invoke
    otherwise, same pattern as ``Retrieve``'s optional ports.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        rag_chat_model: ChatModel,
        config: AppConfig,
        recomp_chat_model: Optional[ChatModel] = None,
        system_prompt: Optional[str] = None,
    ):
        """Args:
            vector_store: Used for neighbor-chunk expansion.
            rag_chat_model: ``ChatModel`` wired to the "rag" role; its
                ``stream`` produces the final answer.
            config: Root config; ``flags``, ``context`` and
                ``retrieval.n_top_for_expansion`` are read fresh on every
                ``run()``.
            recomp_chat_model: ``ChatModel`` wired to the "recomp" role, or
                ``None`` to disable RECOMP synthesis.
            system_prompt: System prompt for the final generation
                (``SYSTEM_PROMPT_RAG`` in ``rag/chat_pdfs.py``). Not sourced
                from ``AppConfig`` -- system prompts are not part of it (see
                ``AppConfig``'s module docstring); the caller wiring this use
                case supplies it explicitly.
        """
        self._vector_store = vector_store
        self._rag_chat_model = rag_chat_model
        self._recomp_chat_model = recomp_chat_model
        self._system_prompt = system_prompt
        self._config = config

    def run(
        self,
        question: str,
        fragments: Sequence[Fragment],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> AnswerResult:
        """Run neighbor expansion, char budgeting, context synthesis and generation.

        Args:
            question: User question (also used as the RECOMP synthesis focus).
            fragments: Ranked, threshold-filtered fragments from ``Retrieve``.
            on_token: Optional callback invoked with each streamed token.

        Returns:
            ``AnswerResult`` with the complete answer text and metrics.
        """
        config = self._config
        flags = config.flags

        expanded, n_expanded = _expand_with_neighbors(
            fragments, self._vector_store, config.retrieval.n_top_for_expansion
        ) if flags.expandir_contexto else (list(fragments), 0)

        limited, n_discarded = _limit_by_char_budget(expanded, config.context.max_context_chars)

        context_str, context_metrics = self._build_context(limited, question, flags)

        user_message = f"{question}\n\n<context>{context_str}</context>"

        text = ""
        generation_stats: Dict[str, Any] = {}
        for chunk in self._rag_chat_model.stream(user_message, system=self._system_prompt):
            if chunk.text:
                text += chunk.text
                if on_token is not None:
                    on_token(chunk.text)
            if chunk.done:
                generation_stats = _generation_stats(chunk)

        metrics: Dict[str, Any] = {
            "fragments_expanded": n_expanded,
            "fragments_discarded_by_chars": n_discarded,
            "fragments_final": len(limited),
            "generation": generation_stats,
            **context_metrics,
        }
        return AnswerResult(text=text, metrics=metrics)


    def _build_context(
        self, fragments: Sequence[Fragment], question: str, flags: PipelineFlagsConfig
    ) -> "tuple[str, Dict[str, Any]]":
        if not flags.usar_recomp_synthesis or not fragments or self._recomp_chat_model is None:
            context_str, metrics = build_context_for_model(fragments, flags.usar_optimizacion_contexto)
            return context_str, {**metrics, "recomp_used": False}
        return self._synthesize_recomp(fragments, question, flags)

    def _synthesize_recomp(
        self, fragments: Sequence[Fragment], question: str, flags: PipelineFlagsConfig
    ) -> "tuple[str, Dict[str, Any]]":
        """Synthesize context using the RECOMP model instead of raw chunks.

        Moved from ``rag.engine.context.sintetizar_contexto_recomp``. Falls
        back to ``build_context_for_model`` in the same three conditions as
        the original: no usable evidence text, the LLM call fails, or the
        synthesized output is too short / missing the expected header.
        """
        textos_preparados = []
        for f in fragments:
            cuerpo = _texto_fuente_fragmento(f.doc or "")
            if flags.usar_optimizacion_contexto:
                cuerpo = optimize_context_text(cuerpo)
                cuerpo = _marcar_fragmento_incompleto(cuerpo)
            content = cuerpo.replace("\n", " ").strip()
            content = re.sub(r'\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', content)  # strip citation markers
            if content:
                n = len(textos_preparados) + 1
                textos_preparados.append(f"Fragment {n}:\n{content}")

        contexto_raw = "\n\n".join(textos_preparados)
        if not contexto_raw.strip():
            context_str, metrics = build_context_for_model(fragments, flags.usar_optimizacion_contexto)
            return context_str, {**metrics, "recomp_used": False}

        q = (question or "").strip()
        bloque_pregunta = (
            f"## User question\n{q}\n"
            if q
            else "## User question\n(No question provided; extract the main technical facts from the excerpts.)\n"
        )

        system_prompt = (
            "You compress retrieved document excerpts into a brief briefing for a downstream "
            "answer model.\n"
            "GROUNDING:\n"
            "- Use ONLY information stated in the evidence excerpts. No outside knowledge.\n"
            "- Preserve technical terms, notation, formulas, and numbers exactly as written.\n"
            "ENUMERATION (critical):\n"
            "- If the question asks for a list or a count (e.g. 'three types', 'two ways'), "
            "search ALL fragments and enumerate EVERY item you find, even if items are spread "
            "across different fragments. Never say an item 'is not mentioned' if it appears "
            "anywhere in the excerpts.\n"
            "- When an excerpt ends with [excerpt ends mid-sentence], the list may continue in "
            "another fragment — collect items from ALL fragments before writing your bullets.\n"
            "STYLE:\n"
            "- Write ONLY facts that help answer the user question. Do NOT describe the documents, "
            "the paper, or the excerpts (forbidden openers: \"This paper\", \"The excerpt\", "
            "\"This section\", \"The document\", \"The text\", \"The fragment\").\n"
            "- Do NOT restate meta-summaries; every bullet must be substantive content from the excerpts.\n"
            "- Do NOT cite fragment numbers, sources, or page numbers.\n"
            "OUTPUT FORMAT (exactly this structure, markdown):\n"
            "## Facts relevant to the question\n"
            "- (first grounded fact)\n"
            "- (second grounded fact)\n"
            "Use one bullet per distinct fact; merge duplicates. If nothing in the excerpts bears "
            "on the question, output exactly one bullet: "
            "\"Insufficient evidence in the excerpts to answer the question.\"\n"
            "Language: same as the evidence excerpts (or the user question if excerpts mix languages)."
        )

        user_prompt = (
            f"{bloque_pregunta}"
            "## Evidence excerpts (verbatim from retrieval; may be partial)\n"
            f"{contexto_raw}\n\n"
            "Produce the briefing using the required OUTPUT FORMAT. No text before "
            "## Facts relevant to the question."
        )

        try:
            raw = self._recomp_chat_model.generate(user_prompt, system=system_prompt)
        except Exception:
            # Explicit use-case-level fallback -- see the module docstring.
            context_str, metrics = build_context_for_model(fragments, flags.usar_optimizacion_contexto)
            return context_str, {**metrics, "recomp_used": False, "recomp_error": True}

        sintesis = normalize_recomp_output(strip_ollama_think_blocks(raw.strip()))

        if len(sintesis) < 20 or RECOMP_FACTS_HEADER.lower() not in sintesis.lower():
            context_str, metrics = build_context_for_model(fragments, flags.usar_optimizacion_contexto)
            return context_str, {**metrics, "recomp_used": False}

        return sintesis, {"recomp_used": True, "chars_original": len(contexto_raw), "chars_optimized": len(sintesis)}
