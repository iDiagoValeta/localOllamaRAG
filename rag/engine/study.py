"""Summary and outline entry points for the CLI and the web app.

Issue #140. Wires the port adapters and runs
``monkeygrab.application.study.Study``, converting between the domain
fragments the core uses and the dicts the interfaces have always consumed --
the same shape ``rag/engine/generation.py`` has for ``Answer``.

It takes the fragments a caller has already retrieved rather than retrieving
them itself. That is not indirection for its own sake: the CLI shows
the sources before it starts generating, and the web app streams the retrieval
phase separately, so a single "retrieve and summarise" call would take the
choice of when to show what away from both of them.
"""

from typing import Any, Dict, List, Optional

from monkeygrab.application.study import (
    MalformedSummaryError,
    Study,
    summary_to_dict,
)
from rag.engine import wiring

__all__ = [
    "MalformedSummaryError",
    "fragmentos_de_documento",
    "resumir_fragmentos",
]


def _study() -> Study:
    """Build ``Study`` on the RAG generator role.

    The summariser is the RAG model rather than the auxiliary chat one: a
    summary is the same job as answering -- read this evidence, write prose
    about it -- and pointing it at the smaller helper model would quietly make
    summaries worse than answers over identical material.
    """
    config = wiring.app_config_from_runtime()
    return Study(wiring.rag_chat_model(config))


def resumir_fragmentos(
    fragmentos: List[Dict[str, Any]],
    idioma: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarise already-retrieved fragments into sections with their pages.

    Args:
        fragmentos: Fragment dicts as the interfaces hold them.
        idioma: Language for the summary, e.g. "Castellano". ``None`` lets the
            model follow the material.

    Returns:
        ``{"source_document": str, "sections": [{"heading", "body",
        "source_pages"}]}`` -- plain JSON, so the web app can return it
        unchanged and the CLI can render it without a second shape.

    Raises:
        MalformedSummaryError: The generator's reply could not be read as a
            summary. Not caught here: an interface decides how to tell its
            user, and swallowing it would leave them with an empty panel and
            no reason for it.
    """
    config = wiring.app_config_from_runtime()
    fragments = [wiring.fragment_from_dict(f) for f in fragmentos]
    return summary_to_dict(_study().summarize(fragments, config, language=idioma))


# How many chunks of one document to feed a summary. The context budget cuts
# the text long before this does; the cap exists so a 300-page PDF does not
# turn a summary into a full-corpus scan first and a truncation second.
_MAX_DOCUMENT_CHUNKS = 400


def fragmentos_de_documento(nombre: str) -> List[Dict[str, Any]]:
    """Every stored chunk of one document, in storage order.

    Retrieval answers "what is relevant to this question". Summarising a whole
    document is a different question -- "what does this say" -- and running a
    similarity search for it would silently drop the parts no query happened
    to match, which is exactly the content a summary is supposed to cover.

    Args:
        nombre: The document's ``source`` as ``obtener_documentos_indexados``
            reports it.

    Returns:
        Fragment dicts for that document, capped at ``_MAX_DOCUMENT_CHUNKS``.
        Empty when nothing in the store carries that source -- the caller
        decides whether that is a typo or an empty document.
    """
    config = wiring.app_config_from_runtime()
    store = wiring.vector_store(config)
    fragments = [
        wiring.fragment_to_dict(fragment)
        for fragment in store.get_page(limit=None, offset=0)
        if fragment.metadata.source == nombre
    ]
    return fragments[:_MAX_DOCUMENT_CHUNKS]
