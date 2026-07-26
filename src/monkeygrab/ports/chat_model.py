"""ChatModel -- Ollama-style text/vision generation, single-shot or streamed.

# ─────────────────────────────────────────────
# SECTION 1: PORT
# ─────────────────────────────────────────────
"""

from typing import Iterator, Optional, Protocol, Sequence


class ChatModel(Protocol):
    """One Ollama-backed model role: generate text, optionally from images.

    Every pipeline role that talks to an LLM (``MODELO_CHAT``,
    ``MODELO_CONTEXTUAL``, ``MODELO_RECOMP``, ``MODELO_OCR``, ``MODELO_RAG``)
    is a separate model *name*, dependency-injected as its own
    ``ChatModel`` instance -- this port models the single capability
    "talk to one model", not model selection or role wiring.

    Two methods cover every real call site:

    - ``generate``: single-shot, full-text-back calls --
      ``generar_queries_con_llm`` (``rag/engine/reranking.py``,
      ``ollama.generate``), ``sintetizar_contexto_recomp``
      (``rag/engine/context.py``, ``POST /api/chat``),
      ``generar_contexto_situacional`` (``rag/engine/contextual.py``,
      ``ollama.chat``), and ``describir_imagen_con_llm``
      (``rag/engine/images.py``, ``ollama.chat`` with ``images=[b64]``) --
      hence the optional ``images`` parameter, unused by every caller
      except the OCR role.
    - ``stream``: token-streaming calls -- ``_ollama_generate_stream`` /
      ``generar_tokens_respuesta`` (``rag/engine/generation.py``), used
      only for the final RAG/chat answer shown to the user.

    Sampling parameters (``temperature``, ``num_predict``, ``num_ctx``,
    ``think``, ``keep_alive``, ...) differ per call site today but are
    runtime tuning, not part of what makes this port "a chat model" --
    they stay adapter/config concerns, not Protocol parameters.

    Failure policy: hard-fail. Raise on any generation failure. Several
    callers today catch broadly and fall back to empty output or raw
    unsynthesized context (``generar_queries_con_llm`` returns ``[]``,
    ``sintetizar_contexto_recomp`` falls back to raw chunks in five
    different failure conditions, ``describir_imagen_con_llm`` returns
    ``""``) -- none of that silent degradation is this port's job; a
    caller that wants a fallback makes it an explicit decision using two
    ports, not something one adapter does invisibly.
    """

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Sequence[bytes] = (),
    ) -> str:
        """Generate a complete response in one call.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.
            images: Optional raw image bytes to condition generation on
                (vision models only; empty for every non-OCR role).

        Returns:
            The complete generated text.

        Raises:
            Exception: On any generation failure.
        """
        ...

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[str]:
        """Generate a response, yielding text as it is produced.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.

        Yields:
            Successive text chunks; concatenating all of them is the
            complete response.

        Raises:
            Exception: On any generation failure.
        """
        ...
