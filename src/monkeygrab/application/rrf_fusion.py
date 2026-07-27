"""rrf_fusion -- Reciprocal-Rank-Fusion of semantic and lexical retrieval.

Moved from the fusion block inside
``rag.engine.retrieval.realizar_busqueda_hibrida`` (the part between "fusing
results..." and the reranker call): given already-ranked semantic hits (one
ranked list per query variant) and an already-ranked lexical hit list, fuse
them into one ranked ``Fragment`` list. Everything upstream of fusion
(embedding a query variant, calling the vector store, calling the lexical
index) is orchestration that belongs to the ``Retrieve`` use case, not to
this pure function -- it only does the RRF math.

Equivalence with ``realizar_busqueda_hibrida``'s fusion block is asserted in
``tests/unit/application/test_rrf_fusion_equivalence.py`` by replaying the
exact fixture from
``tests/characterization/test_retrieval_rrf.py::test_rrf_fusion_favors_cross_branch_consensus_over_single_branch_top_rank``.
"""

import dataclasses
from typing import Dict, List, Sequence

from monkeygrab.domain.fragment import Fragment


def fuse_semantic_and_keyword(
    semantic_hits_per_query: Sequence[Sequence[Fragment]],
    keyword_hits: Sequence[Fragment],
    rrf_k: int,
    weight_semantic: float,
    weight_keyword: float,
) -> List[Fragment]:
    """Fuse ranked semantic and lexical hit lists via weighted RRF.

    For each branch, a fragment's RRF contribution is
    ``sum(1 / (rank + rrf_k))`` over every ranked list it appears in (ranks
    are 1-based, taken from each list's existing order -- neither branch is
    re-sorted by score before fusion, exactly as
    ``realizar_busqueda_hibrida`` consumes ``collection.query``'s and
    ``LexicalIndex.search``'s output as-is). ``score_final`` is the weighted
    sum of both branch totals.

    Fragment identity for accumulation is ``Fragment.id`` (``f"{source}_pag
    {page}_chunk{chunk}"``), matching the ``chunk_id`` keys
    ``realizar_busqueda_hibrida`` builds its ``all_semantic_results``/
    ``fragmentos_data`` dicts with.

    Args:
        semantic_hits_per_query: One ranked ``Fragment`` list per query
            variant (original question plus any decomposed sub-queries),
            nearest-first, as returned by ``VectorStore.query``.
        keyword_hits: Ranked ``Fragment`` list from ``LexicalIndex.search``,
            best-first.
        rrf_k: Reciprocal-Rank-Fusion rank-damping constant.
        weight_semantic: Fusion weight applied to the semantic RRF total.
        weight_keyword: Fusion weight applied to the keyword RRF total.

    Returns:
        Fused fragments, ``score_final``-descending. Ties keep first-seen
        order (semantic branch, in query-variant order, then keyword-only
        additions) -- the same order a Python dict-then-``sorted`` pipeline
        produces, which is what the original does.
    """
    # Accumulator state per fragment id, built in first-seen order so a
    # stable sort on score_final reproduces the original's tie-break order
    # (it also accumulates into a dict before sorting its .values()).
    by_id: Dict[str, Fragment] = {}

    for q_idx, hits in enumerate(semantic_hits_per_query):
        for rank, frag in enumerate(hits, start=1):
            fid = frag.id
            rrf_contribution = 1.0 / (rank + rrf_k)
            if fid not in by_id:
                by_id[fid] = dataclasses.replace(
                    frag,
                    score_semantic=rrf_contribution,
                    score_keyword=0.0,
                    matches=(),
                    query_matches=(q_idx + 1,),
                )
            else:
                existing = by_id[fid]
                by_id[fid] = dataclasses.replace(
                    existing,
                    score_semantic=existing.score_semantic + rrf_contribution,
                    distancia=min(existing.distancia, frag.distancia),
                    query_matches=existing.query_matches + (q_idx + 1,),
                )

    for rank, frag in enumerate(keyword_hits, start=1):
        fid = frag.id
        rrf_contribution = 1.0 / (rank + rrf_k)
        if fid in by_id:
            existing = by_id[fid]
            matches = existing.matches if "BM25" in existing.matches else existing.matches + ("BM25",)
            by_id[fid] = dataclasses.replace(
                existing,
                score_keyword=existing.score_keyword + rrf_contribution,
                matches=matches,
            )
        else:
            by_id[fid] = dataclasses.replace(
                frag,
                score_semantic=0.0,
                score_keyword=rrf_contribution,
                matches=("BM25",),
                query_matches=(),
            )

    fused = [
        dataclasses.replace(
            frag,
            score_final=frag.score_semantic * weight_semantic + frag.score_keyword * weight_keyword,
        )
        for frag in by_id.values()
    ]

    return sorted(fused, key=lambda f: f.score_final, reverse=True)
