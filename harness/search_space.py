"""search_space -- the declared configuration space block C searches over.

Written by hand as data, never inferred by introspecting ``AppConfig``:
adding a parameter is a visible decision here because each one multiplies the
space against a budget of three or four full-search-set candidates per night
(docs/design/2026-07-28-loop-automejorable.md section 4 -- 2.8 h/candidate).

Validated at import time against the real ``AppConfig``: every declared value
is applied via ``AppConfig().with_overrides(**{key: value})``, which already
raises ``ValueError`` on an unknown section or field (see
``AppConfig.with_overrides``'s docstring), so this module cannot silently
drift away from the config it searches. That validation call checks field
*existence*, not value legality -- ``with_overrides`` does not type-check.

This space is NOT "every field ``AppConfig`` has" -- it is "every field the
measurement can actually move", and those two sets differ. Two examples
found auditing this PR: ``flags.usar_llm_query_decomposition`` exists on
``AppConfig`` but is absent from ``SEARCH_SPACE`` because the gate hardcodes
its collaborator to ``None`` regardless of the flag (issue #64, see the
comment where it would otherwise sit); ``retrieval.min_question_length``
exists on ``AppConfig`` but is never declared here either -- nothing under
``src/monkeygrab/`` reads it, the real check lives in the legacy
``rag.chat_pdfs`` globals, so it is decorative on this config object and
tuning it would move nothing.

Four things live here besides the declared tunables:

- ``INDEX_TIME_KEYS``: excluded from stage 1 (issue #31 spec section 2.1).
  Any of these changes what is stored, which moves the index fingerprint and
  forces a full MinerU + jina-clip reindex -- a full evaluation already costs
  ~2.8 h, and a reindex on top of that turns a night's three or four
  candidates into one. They get their own slower tier once block B (#30)
  lands and the timings are remeasured. ``_validate_declared_space`` raises
  if any of them appears in ``SEARCH_SPACE``, so a future contributor who
  adds one gets a red test instead of a silent overnight reindex.
- ``expand_overrides`` / ``is_feasible``: ``weight_semantic_rrf`` and
  ``weight_bm25_rrf`` are coupled (see the docstring on ``expand_overrides``),
  and three declared parameters interact (section 2.3). Neither coupling is
  expressible as an independent row in ``SEARCH_SPACE``, so both live in code
  next to the table they constrain.
- ``PENDING_REACHABILITY_KEYS``: keys declared here but not yet proven to
  reach the stage that consumes them through ``evaluate()``'s
  ``config_overrides``. Currently empty -- the four keys it once named were
  resolved by the sibling PR (#56), see its own docstring for the full
  history -- but the set and the mechanism it drives
  (``evaluator.verify_reachable``) stay, because this audit caught two
  *different* ways a declared key can turn out inert (a config not reaching
  a stage; a gate hardcoding a collaborator regardless of the flag, issue
  #64) and a future addition gets the same check for free.
- ``proposal_order``: the order ``GridProposer`` walks the space in, which is
  NOT ``SEARCH_SPACE``'s declaration order -- known-reachable keys are walked
  first so a run that starts before a future reachability gap closes still
  spends its early iterations on knobs proven to do something.
"""

import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from monkeygrab.config.app_config import AppConfig  # noqa: E402

# INDEX-TIME EXCLUSION

# Changing any of these invalidates the stored index (chunk boundaries,
# contextual-retrieval summaries or embedded images all move), so they are
# out of stage 1's action space entirely -- not just "not currently tuned".
INDEX_TIME_KEYS: frozenset = frozenset({
    "chunking.chunk_size",
    "chunking.chunk_overlap",
    "chunking.min_chunk_length",
    "chunking.contextual_doc_chars",
    "flags.usar_contextual_retrieval",
    "flags.usar_embeddings_imagen",
})

# DECLARED TUNABLES

# (dotted_key, allowed_values, why, stage) -- retrieval/generation-time only,
# per issue #31 spec section 2.2. ``stage`` is documentation, naming which
# pipeline phase reads the field ("retrieval", "reranking" or "generation");
# it is NOT what drives the reachability gate below -- PENDING_REACHABILITY_KEYS
# is the authoritative set for that, because "stage" alone does not predict
# whether config_overrides actually reaches it (see that constant's docstring).
#
# ``usar_busqueda_hibrida`` and ``usar_reranker`` are deliberately absent: the
# design doc names turning the reranker off as the *sabotage* used to prove
# criterion 2 (measured 2026-07-29, see
# docs/design/2026-07-28-loop-automejorable.md section 2). A knob whose "off"
# position is the known-bad control used to validate the gate does not belong
# in the same space the optimiser searches -- the loop could otherwise
# "discover" the sabotage's opposite and get credit for reinventing the
# default.
SEARCH_SPACE: Tuple[Tuple[str, Tuple[Any, ...], str, str], ...] = (
    (
        "retrieval.top_k_final",
        (4, 6, 8, 12, 16),
        "Criterion 2 (2026-07-29): forcing this to 1 dropped retrieval pass "
        "rate from 0.73 to 0.27 -- the single most load-bearing knob measured.",
        "retrieval",
    ),
    (
        "retrieval.top_k_rerank_candidates",
        (100, 200, 300),
        "How much the reranker gets to see.",
        "reranking",
    ),
    (
        "retrieval.n_semantic_results",
        (40, 80, 120),
        "Semantic branch fan-out.",
        "retrieval",
    ),
    (
        "retrieval.n_keyword_results",
        (20, 40, 80),
        "BM25 branch fan-out.",
        "retrieval",
    ),
    (
        "retrieval.rrf_k",
        (30, 60, 100),
        "Rank damping; low k trusts top ranks harder.",
        "retrieval",
    ),
    (
        "retrieval.weight_semantic_rrf",
        (0.3, 0.45, 0.55, 0.7),
        "Fusion weight; weight_bm25_rrf is derived, see expand_overrides().",
        "retrieval",
    ),
    (
        "retrieval.bm25_k1",
        (1.2, 1.5, 2.0),
        "Term-frequency saturation.",
        "retrieval",
    ),
    (
        "retrieval.bm25_b",
        (0.5, 0.75, 0.9),
        "Length normalisation.",
        "retrieval",
    ),
    (
        "retrieval.n_top_for_expansion",
        (0, 3, 6),
        "Neighbour-chunk expansion breadth; gated by flags.expandir_contexto, "
        "see is_feasible().",
        "retrieval",
    ),
    (
        "reranking.score_threshold",
        (0.4, 0.55, 0.65, 0.8),
        "What the reranker is allowed to drop.",
        "reranking",
    ),
    (
        "context.max_context_chars",
        (16000, 24000, 32000),
        "Evidence budget handed to the generator.",
        "generation",
    ),
    (
        "flags.usar_recomp_synthesis",
        (True, False),
        "Synthesised briefing vs raw chunks.",
        "generation",
    ),
    (
        "flags.expandir_contexto",
        (True, False),
        "Neighbour-chunk expansion on/off.",
        "generation",
    ),
    # flags.usar_llm_query_decomposition is deliberately ABSENT (issue #64,
    # filed while auditing search-space reachability with the sibling PR,
    # #56): tests/eval/run_eval.py:358 builds Retrieve(..., query_decomposer=None)
    # unconditionally, while the product wires it whenever the flag is on
    # (rag/engine/retrieval.py:50, default on). The gate therefore measures a
    # retrieval pipeline no user runs, and flipping this flag inside an
    # evaluation cannot change anything -- it would not raise (evaluate()
    # accepts the override) and would not move a single case either. Fixing
    # the gate needs sign-off because it will move the pass rate; until then,
    # this key stays out of the declared space rather than spend iterations
    # on a knob proven to do nothing. Put it back once #64 closes.
    (
        "flags.usar_optimizacion_contexto",
        (True, False),
        "Strip PDF extraction noise from fragment text before generation.",
        "generation",
    ),
)

DECLARED_KEYS: frozenset = frozenset(key for key, _values, _why, _stage in SEARCH_SPACE)
ALLOWED_VALUES: dict = {key: values for key, values, _why, _stage in SEARCH_SPACE}
_STAGE_OF: dict = {key: stage for key, _values, _why, stage in SEARCH_SPACE}

# REACHABILITY GAP

# History, not a live warning: found 2026-08-12 while building the sibling
# PR (#56, the run_eval.evaluate() library API this harness's evaluator.py
# wraps) that evaluate() threads its AppConfig into retrieval and indexing,
# but generation goes through rag/engine/generation.py's
# generar_respuesta_silenciosa, which never received that config directly.
# #56 then audited all 48 dotted AppConfig fields end to end and closed the
# gap for all four keys this set used to name:
#   - context.max_context_chars / flags.expandir_contexto were ALREADY
#     reaching generation -- both are read by Answer.select_evidence, which
#     runs under the config evaluate() threads through; the original finding
#     was overcautious about these two.
#   - flags.usar_recomp_synthesis / flags.usar_optimizacion_contexto were
#     genuinely dropped: they are read in build_user_message, which phase 2
#     re-executes under a *fresh* config from wiring.app_config_from_runtime().
#     #56 now applies them via the sanctioned rag.set_pipeline_flags(...)
#     runtime setter for the duration of the evaluation, restored in a
#     finally (with a test that an exception mid-run still restores the
#     previous globals).
#
# Kept as a set (now empty) rather than deleted, because the mechanism it
# drives -- evaluator.verify_reachable, proposal_order()'s reachable-first
# walk -- caught two DIFFERENT failure classes during this same audit: this
# one (a config silently not reaching a stage) and issue #64
# (flags.usar_llm_query_decomposition, where the gate hardcodes
# query_decomposer=None regardless of the flag -- see that key's removal
# comment above). A future key added to SEARCH_SPACE gets the same startup
# check for free; nobody has to remember to re-verify it by hand.
PENDING_REACHABILITY_KEYS: frozenset = frozenset()


def _validate_declared_space(entries: Sequence[Tuple[str, Tuple[Any, ...], str, str]]) -> None:
    """Raise if ``entries`` names an index-time key, or a field ``AppConfig`` lacks.

    Shared by the module-level self-check below and by
    ``harness/tests/test_search_space.py``, which calls it directly with a
    synthetic entry to prove an index-time addition is actually caught (spec
    test 2) rather than trusting the module-level call never regresses.

    Args:
        entries: Same shape as ``SEARCH_SPACE``.

    Raises:
        ValueError: An entry's key is in ``INDEX_TIME_KEYS``, or
            ``AppConfig.with_overrides`` rejects one of its declared values.
    """
    base = AppConfig()
    for key, values, _why, _stage in entries:
        if key in INDEX_TIME_KEYS:
            raise ValueError(
                f"{key!r} is an index-time parameter (see INDEX_TIME_KEYS) -- "
                "it cannot be a stage-1 tunable without a reindex per candidate."
            )
        for value in values:
            base.with_overrides(**{key: value})  # raises ValueError if key is unknown


_validate_declared_space(SEARCH_SPACE)


# COUPLED PARAMETERS


def expand_overrides(overrides: Mapping[str, Any]) -> dict:
    """Derive ``retrieval.weight_bm25_rrf`` from ``retrieval.weight_semantic_rrf``.

    ``src/monkeygrab/application/rrf_fusion.py::fuse_semantic_and_keyword``
    computes ``score_final = score_semantic * weight_semantic + score_keyword
    * weight_keyword`` and sorts by it -- a linear combination used only for
    ordering. Scaling both weights by any positive constant scales every
    fragment's ``score_final`` by that same constant, which cannot change a
    sort order; only the ratio between the two weights matters. Searching
    both independently would spend budget on points that cannot differ, so
    only ``weight_semantic_rrf`` is declared in ``SEARCH_SPACE`` and
    ``weight_bm25_rrf`` is always derived as ``1 - weight_semantic_rrf``
    (verified against the fusion code above, not assumed -- see the PR
    report for confirmation).

    Every override dict this module hands to ``AppConfig.with_overrides``
    (directly, or via a caller applying overrides to build the config an
    evaluation actually runs) must go through this function first, so a
    ledger entry's raw ``config_overrides`` (just the one declared knob) and
    its ``effective_config`` (both weights, summing to 1) stay reconstructible
    from each other.

    Args:
        overrides: Dotted-key overrides, as a proposer emits them.

    Returns:
        A new dict; ``overrides`` is not mutated. Unchanged unless
        ``"retrieval.weight_semantic_rrf"`` is present and
        ``"retrieval.weight_bm25_rrf"`` is not already given explicitly.
    """
    expanded = dict(overrides)
    if "retrieval.weight_semantic_rrf" in expanded and "retrieval.weight_bm25_rrf" not in expanded:
        expanded["retrieval.weight_bm25_rrf"] = round(1.0 - expanded["retrieval.weight_semantic_rrf"], 10)
    return expanded


def is_feasible(overrides: Mapping[str, Any], reference: AppConfig) -> bool:
    """Whether ``overrides`` applied to ``reference`` yields a runnable config.

    Feasibility predicate from issue #31 spec section 2.3, checked against
    the *effective* config (``reference`` with ``overrides`` applied), not
    against the raw override dict -- a candidate that only touches one field
    is feasible or not depending on what the other fields already are.

    Three rules:

    - ``top_k_final <= top_k_rerank_candidates``: the reranker cannot keep
      more fragments than it was given.
    - ``n_top_for_expansion <= top_k_final``: expansion cannot reach past the
      fragments that survived reranking.
    - ``n_top_for_expansion > 0`` only if ``flags.expandir_contexto`` is
      True. Verified against ``src/monkeygrab/application/answer.py``'s
      ``select_evidence``: ``_expand_with_neighbors(..., n_top_for_expansion)``
      only runs ``if config.flags.expandir_contexto`` (see the PR report) --
      the flag really does gate the parameter, so the spec's constraint holds
      and was not dropped. (This coupling is unaffected by expandir_contexto's
      reachability gap above: the gap is about whether an *override* to it
      reaches generation, not about what the field means inside AppConfig.)

    Args:
        overrides: Dotted-key overrides to test.
        reference: Base config the overrides would be applied to.

    Returns:
        False if ``overrides`` names an unknown field/section, or if the
        resulting config would violate any of the three rules above.
    """
    try:
        effective = reference.with_overrides(**expand_overrides(overrides))
    except ValueError:
        return False

    r = effective.retrieval
    if r.top_k_final > r.top_k_rerank_candidates:
        return False
    if r.n_top_for_expansion > r.top_k_final:
        return False
    if r.n_top_for_expansion > 0 and not effective.flags.expandir_contexto:
        return False
    return True


def allowed_values(key: str) -> Optional[Tuple[Any, ...]]:
    """Declared values for ``key``, or ``None`` if it is not in ``SEARCH_SPACE``."""
    return ALLOWED_VALUES.get(key)


def proposal_order() -> Tuple[str, ...]:
    """Declared keys, known-reachable ones first, each group in declared order.

    ``GridProposer`` walks the space in this order rather than
    ``SEARCH_SPACE``'s declaration order, so a run started before the
    reachability gap (``PENDING_REACHABILITY_KEYS``) is closed still spends
    its early iterations on knobs proven to reach the pipeline, instead of
    interleaving them with knobs that might currently be silent no-ops.

    Returns:
        Every key in ``DECLARED_KEYS``, reachable keys before pending ones,
        stable within each group.
    """
    declared_order = [key for key, _values, _why, _stage in SEARCH_SPACE]
    reachable = [k for k in declared_order if k not in PENDING_REACHABILITY_KEYS]
    pending = [k for k in declared_order if k in PENDING_REACHABILITY_KEYS]
    return tuple(reachable + pending)
