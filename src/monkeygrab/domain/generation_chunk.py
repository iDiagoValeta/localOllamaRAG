"""GenerationChunk -- one item streamed from ChatModel.stream().

# ─────────────────────────────────────────────
# SECTION 1: ENTITY
# ─────────────────────────────────────────────
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GenerationChunk:
    """One line of a streamed Ollama ``/api/generate`` response.

    Mirrors, field for field, what ``rag.engine.generation._ollama_generate_stream``
    yields per parsed JSON line -- ``ChatModel.stream()`` replaces that
    function's raw-dict shape with this typed one. Every line carries
    ``text`` (empty on the final line, which reports an empty ``"response"``)
    and ``done``; only the final line (``done=True``) also carries the
    generation-metadata fields ``generar_tokens_respuesta`` copies into its
    debug-dump stats dict (``_GEN_STATS_KEYS`` in
    ``rag/engine/generation.py``) and the ``eval_count``/``eval_duration``
    pair ``generar_respuesta`` divides to derive tokens-per-second -- on any
    earlier line they are ``None``, exactly as they are simply absent from
    Ollama's own intermediate JSON lines.

    Attributes:
        text: Token text for this line (``""`` on the final line).
        done: Whether this is the final line of the stream.
        model: Model name that produced the response (final line only).
        done_reason: Why generation stopped, e.g. ``"stop"``, ``"length"``
            (final line only). A value other than ``"stop"``/``None`` means
            the response was cut short -- ``ChatModel.stream()``'s failure
            policy requires the adapter to log this, not just carry it.
        total_duration: Total generation time in nanoseconds (final line only).
        load_duration: Model load time in nanoseconds (final line only).
        prompt_eval_count: Number of prompt tokens evaluated (final line only).
        prompt_eval_duration: Prompt evaluation time in nanoseconds (final line only).
        eval_count: Number of tokens generated (final line only).
        eval_duration: Generation time in nanoseconds, for tokens-per-second
            (``eval_count / (eval_duration / 1e9)``) (final line only).
    """

    text: str = ""
    done: bool = False
    model: Optional[str] = None
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
