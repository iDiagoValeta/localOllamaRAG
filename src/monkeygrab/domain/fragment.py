"""Fragment -- a retrieved chunk, scored across the retrieval pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from monkeygrab.domain.chunk_metadata import ChunkMetadata


@dataclass(frozen=True)
class Fragment:
    """A chunk as it flows through retrieval, ranking and generation.

    One entity carries a chunk from the moment a retrieval branch finds it to
    the moment its text reaches the generator, accumulating scores on the
    way. Each branch writes only its own score field, so a fragment records
    where its evidence came from and not just how highly it ranked.

    Attributes:
        doc: Chunk text as stored or retrieved. Not named ``text``, to avoid
            implying equivalence with ``Chunk.text``: a fragment may carry a
            truncated or re-fetched copy of it.
        metadata: Position and format of the underlying chunk. ``id`` is
            derived from it rather than stored, so the two can never
            disagree.
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
