"""slow_tier -- the index-time tier, declared once so it is not designed twice.

Issue #102. ``search_space.INDEX_TIME_KEYS`` is excluded from stage 1 because
changing any of them invalidates the stored index: a flag-only recipe change
(``usar_descripcion_imagen`` off to on) forced discarding and rebuilding dev and
blind even with MinerU's parse cache hot, 215 s per store, and the gate
measurement around it took 36 minutes (measured 2026-08-23, RTX 4060 laptop).
Fine twice. Unaffordable inside a search loop that touches index knobs per
candidate.

The exclusion is right and this module does not lift it. What it adds is the
accounting that would let it be lifted deliberately, on numbers, rather than by
someone deleting a line from ``INDEX_TIME_KEYS`` and discovering overnight what
it costs.

## The three things the issue asked for

**Declared.** ``SLOW_TIER_SPACE`` names the index-time knobs and their
candidate values in one place, separate from ``SEARCH_SPACE``. Declaring a knob
here is not the same as searching it: stage 1 still refuses to, and
``search_space._validate_declared_space`` still goes red if one leaks into the
fast space.

**Batched.** ``batch_by_recipe`` groups candidates by the index they need. Two
candidates that differ only in a retrieval knob share one index, so they share
one reindex — and the size of that group is the whole economics of the tier. A
tier that reindexed per candidate would be the thing the issue says nobody can
afford; one that reindexes per *recipe* is a fixed cost amortised over however
many retrieval variants ride on it.

**Costed in fast-tier units.** ``estimate_cost`` reports a batch's price as a
multiple of an ordinary evaluation, because that is the unit the loop's budget
is already spent in. "This batch costs 41 fast-tier evaluations" is a sentence
an operator can weigh against a patience of 3; "this batch costs 1927 seconds"
is not.

## What this module refuses to do

It does not measure. The seconds it multiplies come from the caller, which got
them from a real run; nothing here calls a clock, and no default reindex cost
is baked in. A cost model with a plausible-looking constant nobody measured is
exactly the "loop sobre una medida que miente" the design doc opens with,
arriving as a helpful default.

It also has no opinion about whether a slow-tier candidate is worth running.
That decision needs block B's case expansion (#30) to say whether index-time
knobs can pay off at all, and it belongs to an operator reading a budget, not
to a module that can only multiply.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from harness.search_space import INDEX_TIME_KEYS

# The index-time knobs and the values a slow-tier campaign would try. Declared
# apart from SEARCH_SPACE on purpose: that table is what stage 1 walks, and a
# key in both would be a key stage 1 tries to search.
#
# Values are deliberately few. Each one is a full corpus rebuild, so a
# three-value knob is three reindexes before a single retrieval variant is
# measured, and the grid over two knobs multiplies rather than adds.
SLOW_TIER_SPACE: Dict[str, Tuple[Any, ...]] = {
    "chunking.chunk_size": (800, 1200),
    "chunking.chunk_overlap": (100, 200),
    "flags.usar_contextual_retrieval": (False, True),
    "flags.usar_embeddings_imagen": (False, True),
}


class SlowTierError(RuntimeError):
    """A slow-tier request that cannot be costed or is not slow tier at all."""


def requires_reindex(overrides: Mapping[str, Any]) -> bool:
    """Whether these overrides change what is stored, not just how it is read.

    The one question that decides which tier a candidate belongs to. Asked of
    the override keys rather than of a config diff: a candidate that sets an
    index-time key back to the reference's own value still moves the index
    fingerprint through the gate's plumbing, and pretending otherwise would put
    a reindex in the fast tier's budget.
    """
    return any(key in INDEX_TIME_KEYS for key in overrides)


def recipe_of(overrides: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """The index-time part of ``overrides``, normalised for grouping.

    Sorted so two candidates that set the same knobs in a different order share
    a recipe; everything else is dropped, because retrieval knobs read the same
    index rather than building another one.
    """
    return tuple(sorted((k, v) for k, v in overrides.items() if k in INDEX_TIME_KEYS))


def batch_by_recipe(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Tuple[Tuple[Tuple[str, Any], ...], List[Mapping[str, Any]]]]:
    """Group candidates by the index they need, in first-seen order.

    Returns:
        ``[(recipe, candidates)]``. The empty recipe -- candidates needing no
        reindex -- is included and its group costs no rebuild, so a caller can
        pass a mixed list and get a truthful bill rather than having to
        pre-sort by tier.

    First-seen order rather than sorted: the proposer's order is a decision it
    made, and re-ordering batches here would quietly override it.
    """
    grouped: Dict[Tuple[Tuple[str, Any], ...], List[Mapping[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(recipe_of(candidate), []).append(candidate)
    return list(grouped.items())


@dataclasses.dataclass(frozen=True)
class TierCost:
    """What a batched slow-tier campaign would cost, in two units.

    Attributes:
        reindexes: Distinct indexes that must be built. This is the number the
            tier lives or dies on, and it is the count of recipes, not of
            candidates.
        evaluations: Candidate evaluations across every batch.
        seconds: Total wall clock, from the caller's measured costs.
        fast_tier_equivalents: ``seconds`` divided by the cost of one ordinary
            evaluation. The unit the loop's budget is already spent in, so a
            batch can be weighed against a patience or an iteration cap without
            converting anything by hand.
    """

    reindexes: int
    evaluations: int
    seconds: float
    fast_tier_equivalents: float


def estimate_cost(
    batches: Iterable[Tuple[Tuple[Tuple[str, Any], ...], Sequence[Mapping[str, Any]]]],
    *,
    reindex_seconds: float,
    evaluation_seconds: float,
) -> TierCost:
    """Price a batched campaign from measured costs.

    Args:
        batches: The output of ``batch_by_recipe``.
        reindex_seconds: Measured cost of one full rebuild, per the machine
            this would run on. No default: see the module docstring.
        evaluation_seconds: Measured cost of one ordinary evaluation, the unit
            ``fast_tier_equivalents`` is expressed in.

    Raises:
        SlowTierError: Either cost is not positive. A zero or negative
            evaluation cost would make every batch look free or negatively
            priced, which is worse than refusing to answer.
    """
    if reindex_seconds <= 0 or evaluation_seconds <= 0:
        raise SlowTierError(
            "reindex_seconds and evaluation_seconds must both be positive measured "
            f"costs, got {reindex_seconds!r} and {evaluation_seconds!r}"
        )

    reindexes = 0
    evaluations = 0
    for recipe, group in batches:
        # The empty recipe reads the index that is already there.
        if recipe:
            reindexes += 1
        evaluations += len(group)

    seconds = reindexes * reindex_seconds + evaluations * evaluation_seconds
    return TierCost(
        reindexes=reindexes,
        evaluations=evaluations,
        seconds=seconds,
        fast_tier_equivalents=round(seconds / evaluation_seconds, 1),
    )


def describe_cost(cost: TierCost) -> str:
    """One line an operator can weigh against a patience budget."""
    return (
        f"slow tier: {cost.reindexes} reindex(es) + {cost.evaluations} evaluation(s) "
        f"= {cost.seconds / 60:.0f} min, {cost.fast_tier_equivalents} fast-tier "
        "evaluations' worth"
    )
