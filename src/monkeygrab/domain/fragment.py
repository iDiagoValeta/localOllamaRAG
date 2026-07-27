"""Fragment -- a retrieved chunk, scored across the retrieval pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from monkeygrab.domain.chunk_metadata import ChunkMetadata


@dataclass(frozen=True)
class Fragment:
    """A chunk as it flows through retrieval, ranking and generation.

    Field-for-field match of the dicts built and passed around in
    ``rag/engine/retrieval.py`` (``realizar_busqueda_hibrida``'s
    ``all_semantic_results`` / ``fragmentos_data`` / ``fragmentos_ranked``),
    ``rag/engine/lexical.py`` (``busqueda_lexica_bm25``'s result dicts, minus
    ``bm25_score`` -- that field never survives RRF fusion: fused entries in
    ``realizar_busqueda_hibrida`` are rebuilt from ``score_semantic`` /
    ``score_keyword`` / ``score_final`` only), and
    ``rag/engine/generation.py`` (``_expandir_fragmentos_contexto``'s
    neighbor-chunk dicts, and ``_score_relevancia_fragmento``'s
    ``score_reranker`` / ``score_final`` fallback).

    Attributes:
        doc: Chunk text as stored/retrieved (called ``doc`` in every
            dict that flows through the pipeline today, kept as-is
            rather than renamed to avoid a false equivalence with
            ``Chunk.text``: a ``Fragment`` may carry a truncated or
            re-fetched copy of that same text).
        metadata: Position and format of the underlying chunk. ``id`` is
            derived from this (see below), never stored redundantly --
            every id construction site in the pipeline
            (``realizar_busqueda_hibrida``, ``busqueda_lexica_bm25``,
            ``_expandir_fragmentos_contexto``) already uses the exact same
            ``f"{source}_pag{page}_chunk{chunk}"`` formula, so a stored
            field could only ever go stale relative to it.
        distancia: Semantic (L2) distance from the query embedding.
            ``inf`` for fragments pulled in without a semantic score
            (BM25-only hits, neighbor-expansion additions).
        score_semantic: Reciprocal-Rank-Fusion contribution from semantic
            search, summed across query variants.
        score_keyword: Reciprocal-Rank-Fusion contribution from BM25.
        score_final: Fused ranking score (``score_semantic * weight +
            score_keyword * weight``), overwritten with the Cross-Encoder
            score when reranking runs.
        score_reranker: Cross-Encoder relevance score, ``None`` until/unless
            reranking has scored this fragment.
        matches: Which retrieval branches hit this fragment, e.g.
            ``("BM25",)``.
        query_matches: 1-based indices of the query variants (original +
            sub-queries) that retrieved this fragment semantically.
    """

    doc: str
    metadata: ChunkMetadata
    distancia: float = float("inf")
    score_semantic: float = 0.0
    score_keyword: float = 0.0
    score_final: float = 0.0
    score_reranker: Optional[float] = None
    matches: Tuple[str, ...] = field(default_factory=tuple)
    query_matches: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        """Fragment id, e.g. ``"paper.pdf_pag2_chunk3"`` -- same formula as ``Chunk.id``."""
        return f"{self.metadata.source}_pag{self.metadata.page}_chunk{self.metadata.chunk}"
