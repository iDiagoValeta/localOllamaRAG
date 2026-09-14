"""Source references for an answer: documents and pages, from ranked fragments.

Lives in the engine rather than the web layer because two entry points read
it -- the web's ``/api/rag`` and the headless service's ``/stores/<id>/rag`` --
and importing ``rag.web.app`` from the headless would run that module's
import-time settings load onto the shared globals.
"""
from typing import Any, Dict, List


def format_sources(fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format source references for the JSON response.

    Args:
        fragments: List of fragment dicts with metadata (source, page).
            Fragments must be ordered by descending relevance score so that
            the first occurrence of each document is its highest-scoring fragment.

    Returns:
        List of dicts with 'document', 'pages', and 'best_page' keys, sorted
        by document name. 'best_page' is the page of the highest-scoring fragment
        for that document, intended as the default scroll target in the viewer.
    """
    sources_map = {}
    for frag in fragments:
        meta = frag.get("metadata", {})
        doc = meta.get("source", "?")
        page = meta.get("page", 0)
        page_num = page + 1 if isinstance(page, int) else page
        score = frag.get("score_reranker", frag.get("score_final", 0.0))
        if doc not in sources_map:
            sources_map[doc] = {"pages": set(), "best_page": page_num, "best_score": score}
        else:
            if score > sources_map[doc]["best_score"]:
                sources_map[doc]["best_score"] = score
                sources_map[doc]["best_page"] = page_num
        sources_map[doc]["pages"].add(page_num)

    return [
        {"document": doc, "pages": sorted(info["pages"]), "best_page": info["best_page"]}
        for doc, info in sorted(sources_map.items())
    ]
