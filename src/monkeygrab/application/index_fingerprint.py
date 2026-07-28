"""Index fingerprint -- what must match for a stored index to still be valid.

An index is only comparable with the pipeline that built it. Reusing one that
was built under different chunking, a different embedding model or a different
index-time flag makes a run measure a mixture of two pipelines while reporting
a single number, and an optimisation loop would attribute that difference to
whatever it happened to change last. This module names exactly which
configuration decides an index's content, so a mismatch can force a rebuild
instead of being discovered as an unexplained score.

Pure: no I/O and no infrastructure imports. Persisting the value is the vector
store's job; deciding what it means is this module's.
"""

import hashlib
import json
from typing import Any, Dict

from monkeygrab.config.app_config import AppConfig

# The fixed multimodal stack. Not read from config because it is not
# configurable (docs/design/2026-07-26-monkeygrab-v2.md). Bump by hand when the
# extractor or the embedding model is replaced -- exactly the kind of change
# that has to invalidate every index built before it.
_EXTRACTOR_ID = "mineru"
_EMBEDDING_ID = "jinaai/jina-clip-v2@512"

_FINGERPRINT_CHARS = 16


def index_recipe(config: AppConfig) -> Dict[str, Any]:
    """Return the configuration that decides an index's stored content.

    Only what changes stored chunks belongs here. Retrieval fan-out, fusion
    weights, the reranker threshold and the answer models are all read at query
    time and leave the index untouched; including them would make the loop
    reindex the whole corpus on every candidate it evaluates.

    Args:
        config: The configuration an index was, or would be, built under.

    Returns:
        A JSON-serializable dict describing that recipe.
    """
    recipe: Dict[str, Any] = {
        "extractor": _EXTRACTOR_ID,
        "embedding": _EMBEDDING_ID,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "min_chunk_length": config.chunking.min_chunk_length,
        "contextual_retrieval": config.flags.usar_contextual_retrieval,
        "image_embeddings": config.flags.usar_embeddings_imagen,
    }
    # The contextual model and its document sample only reach stored text when
    # that stage actually runs. Including them unconditionally would make an
    # index built with the stage OFF depend on a model that never got called.
    if config.flags.usar_contextual_retrieval:
        recipe["contextual_model"] = config.models.contextual
        recipe["contextual_doc_chars"] = config.chunking.contextual_doc_chars
    return recipe


def compute_index_fingerprint(config: AppConfig) -> str:
    """Return a short stable digest of ``index_recipe(config)``.

    Args:
        config: The configuration an index was, or would be, built under.

    Returns:
        16 hex characters -- short enough to read in a log line, wide enough
        that a collision between two real recipes is not a practical concern.
    """
    canonical = json.dumps(index_recipe(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]
