"""Embedder -- text to embedding vector.

# ─────────────────────────────────────────────
# SECTION 1: PORT
# ─────────────────────────────────────────────
"""

from typing import List, Protocol


class Embedder(Protocol):
    """Turns one piece of text into one embedding vector.

    Matches the two real call sites, both single-text, single-vector calls
    to ``ollama.embeddings(model=..., prompt=...)``:
    ``_indexar_chunk`` in ``rag/engine/indexing.py`` (embeds one chunk's
    text at a time before ``collection.add``) and the per-query-variant
    loop in ``realizar_busqueda_hibrida`` in ``rag/engine/retrieval.py``
    (embeds each query variant before ``collection.query``). Batch
    embedding is not something the current pipeline does, so it is not
    part of this port.

    Task-prefix handling (``search_query: `` / ``search_document: `` for
    Nomic-family models, see ``_derivar_prefijos_embedding`` in
    ``rag/chat_pdfs.py``) stays the caller's job: it prepends the right
    prefix to ``text`` before calling ``embed``, so this port never needs
    to know whether the text is a query or a document.

    Failure policy: hard-fail. Raise on any embedding failure. The retry
    used in ``_indexar_chunk`` today (catch, truncate to 1000 chars,
    retry once) is an adapter-level convenience for degenerate input, not
    a silent-degradation path -- if it is kept in a future adapter, a
    second failure after the retry still raises.
    """

    def embed(self, text: str) -> List[float]:
        """Embed ``text`` into a single dense vector.

        Args:
            text: Text to embed, already prefixed by the caller if needed.

        Returns:
            The embedding vector.

        Raises:
            Exception: On any embedding failure.
        """
        ...
