"""Index fingerprint -- what must match for a stored index to still be valid.

An index is only comparable with the pipeline that built it. Reusing one that
was built under different chunking, a different embedding model or a different
index-time flag makes a run measure a mixture of two pipelines while reporting
a single number, and an optimisation loop would attribute that difference to
whatever it happened to change last. This module names exactly which
configuration decides an index's content, so a mismatch can force a rebuild
instead of being discovered as an unexplained score.

Configuration is not the only thing that decides stored content, though --
the CODE that turns a config into stored text can change too, and nothing
above catches that: editing the chunking algorithm or the contextual-
enrichment prompt changes what gets stored without moving a single config
value. ``_RECIPE_VERSION`` closes that gap (see its docstring for what counts
as a version bump and what does not).

Pure: no I/O and no infrastructure imports. Persisting the value is the vector
store's job; deciding what it means is this module's.
"""

import hashlib
import json
from typing import Any, Dict, Optional

from monkeygrab.config.app_config import AppConfig

# The fixed multimodal stack. Not read from config because it is not
# configurable (docs/design/2026-07-26-monkeygrab-v2.md). Bump by hand when the
# extractor or the embedding model is replaced -- exactly the kind of change
# that has to invalidate every index built before it.
_EXTRACTOR_ID = "mineru"
_EMBEDDING_ID = "jinaai/jina-clip-v2@512"

# Bumped BY HAND when the code that turns a recipe into stored content
# changes -- never for a configuration change, since every config value this
# module cares about already has its own recipe key and invalidates on its
# own. Counts as a bump: the chunking algorithm (how text is split into
# chunks), the contextual-enrichment prompt template, the image-extraction
# logic -- anything that can make the SAME configuration produce different
# stored text or vectors than it used to. Does not count: a comment, a log
# line, a refactor that leaves stored output byte-identical, or adding a new
# *configuration* field (that already invalidates through its own recipe key,
# e.g. contextual_num_ctx below).
#
# Serialized by omission at version 1: version 1 is defined as "the recipe
# exactly as it existed before recipe_version was introduced", so the key is
# left out of the dict entirely while this constant is 1 -- adding it costs
# zero reindexing on its own. The key starts appearing only once a human
# bumps this past 1, which is also the point every existing index legitimately
# needs rebuilding. See
# tests/unit/application/test_index_fingerprint.py::test_recipe_version_is_omitted_at_v1_and_matches_pre_change_main
# for the test that makes this safe rather than clever, and
# ::test_recipe_version_bump_actually_changes_the_recipe_and_the_fingerprint
# for proof the mechanism itself is live, not just its state at v1.
#
# When you bump this: test_recipe_version_is_omitted_at_v1_and_matches_pre_change_main
# will fail on its hardcoded hash -- DELETE it, do not "fix" it by pasting in
# the new hash. Its only job was to pin v1's zero-cost guarantee, which no
# longer applies once this is 2+. ::test_default_app_config_fingerprint_is_pinned
# is a different kind of pin (the default config's current fingerprint, not
# v1's zero-cost property) and DOES need its hash updated, not deleted.
_RECIPE_VERSION = 1

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
    if _RECIPE_VERSION != 1:
        recipe["recipe_version"] = _RECIPE_VERSION
    # The contextual model, its context window and its document sample only
    # reach stored text when that stage actually runs. Including them
    # unconditionally would make an index built with the stage OFF depend on
    # a model call that never happened.
    if config.flags.usar_contextual_retrieval:
        recipe["contextual_model"] = config.models.contextual
        recipe["contextual_doc_chars"] = config.chunking.contextual_doc_chars
        recipe["contextual_num_ctx"] = config.models.ollama.contextual_num_ctx
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


def fingerprint_is_stale(stored: Optional[str], expected: str) -> bool:
    """Whether a stored index provably no longer matches the active recipe.

    Distinct from ``run_eval.should_rebuild`` (tests/eval/run_eval.py), which
    treats "unknown" the same as "mismatch" because an eval run must never
    measure a mixture of two pipelines. The product instead has to tell the
    two apart: a missing fingerprint means every index built before this
    feature existed, which is not something the user did, and warning about
    it at every launch would be noise, not information. Only a fingerprint
    that actively disagrees is something the user changed a setting since.

    Args:
        stored: The fingerprint the store recorded, or ``None`` if it never
            recorded one.
        expected: The fingerprint of the configuration currently in force.

    Returns:
        True only when ``stored`` is present and differs from ``expected``.
    """
    return stored is not None and stored != expected
