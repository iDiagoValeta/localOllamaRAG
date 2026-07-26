"""OllamaEmbedder -- Embedder adapter over ``ollama.embeddings``.

# ─────────────────────────────────────────────
# SECTION 1: ADAPTER
# ─────────────────────────────────────────────
"""

from typing import List

import ollama

from monkeygrab.config.models import ModelsConfig

# Not subclassed from monkeygrab.ports.embedder.Embedder: Protocol conformance
# here is structural (duck typing), the same contract every other adapter in
# this package satisfies without inheriting its port.


class OllamaEmbedder:
    """Embeds one piece of text via ``ollama.embeddings``.

    Wraps the exact call both real call sites make today --
    ``_indexar_chunk`` (``rag/engine/indexing.py``) and the per-query-variant
    loop in ``realizar_busqueda_hibrida`` (``rag/engine/retrieval.py``) --
    with the model name read from ``ModelsConfig.embedding`` instead of the
    ``rag.chat_pdfs`` global ``MODELO_EMBEDDING``.

    Per the ``Embedder`` port contract, task-prefix handling
    (``search_query: `` / ``search_document: ``) is the caller's job:
    ``text`` arrives here already prefixed, so this adapter never branches
    on query-vs-document.

    Failure policy: hard-fail. The one exception is the truncate-and-retry
    the port docstring explicitly allows as an "adapter-level convenience"
    (some models reject an overlong single embedding request) --
    ``_indexar_chunk`` today catches that specific failure, truncates to
    1000 chars and retries once. That one retry is kept; every other
    failure, including a second failure after the retry, raises.
    """

    def __init__(self, models: ModelsConfig):
        """Args:
            models: Model-role config; only ``models.embedding`` is read.
        """
        self._model = models.embedding

    def embed(self, text: str) -> List[float]:
        """Embed ``text`` via ``ollama.embeddings``.

        Args:
            text: Text to embed, already prefixed by the caller if needed.

        Returns:
            The embedding vector.

        Raises:
            RuntimeError: On any embedding failure (after the one
                truncate-and-retry for overlong input).
        """
        try:
            response = ollama.embeddings(model=self._model, prompt=text)
            return response["embedding"]
        except Exception as exc:
            if not _looks_like_overlong_input(exc):
                raise RuntimeError(
                    f"Ollama embedding failed for model {self._model!r}: {exc}"
                ) from exc
            try:
                response = ollama.embeddings(model=self._model, prompt=text[:1000])
                return response["embedding"]
            except Exception as exc2:
                raise RuntimeError(
                    f"Ollama embedding failed for model {self._model!r}, even after "
                    f"truncating the input to 1000 chars: {exc2}"
                ) from exc2


def _looks_like_overlong_input(exc: Exception) -> bool:
    """Match the same heuristic ``_indexar_chunk`` uses to decide the input was too long."""
    message = str(exc).lower()
    return "context length" in message or "500" in message
