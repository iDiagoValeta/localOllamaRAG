"""loop -- the ratchet, the latency constraint and termination for block C.

Implements issue #31 spec section 5.5, with corrections applied after
measuring against the real reference gate runs (local, gitignored --
tests/eval/runs/{20260729T020233Z,20260729T040824Z,20260729T081129Z,
20260812T194812Z}_mineru-jina_clip-faiss.json; see harness/README.md for the
full numbers and why they cannot be re-derived from a fresh clone):

1. **Per-bucket latency, not one blended median.** Answered cases (call the
   generator) and retrieval-only cases (figure_retrieval, table_retrieval)
   differ enough in cost -- ~28.5s vs. ~4.1s as of the latest measurement
   (2026-08-12, tests/eval/runs/20260812T194812Z_mineru-jina_clip-faiss.json,
   local/gitignored; this gap was ~50x before issue #27's keep-alive fix and
   is ~7x now, but the fix only ever changed the *magnitude*, not whether it
   exists) -- that a single median over both is dominated by which bucket
   happens to be larger and would let a candidate trade retrieval quality
   for fewer/cheaper generator calls without the constraint noticing.
   ``_latency_breach`` checks each bucket against its own reference median
   independently.
2. **``resolution_warning`` in the final report.** The search set (32
   ``source: corpus`` cases) can win at most 5 cases today, and moved only 3
   net flips under the most destructive single-field change anyone has
   measured (``RAG_TOP_K_FINAL=1``) -- both below the ~6-flip threshold
   design doc section 3 sets for a paired difference not attributable to
   chance. Every report says so, so an accepted improvement reads as a
   candidate for confirmation, not a demonstrated result.
3. **A ``rejected_regression`` verdict also fires at the search-set stage**
   when the retrieval-only bucket loses cases net against the reference
   (paired by id), even if the blended ``objective_adjusted`` still went up.
   Blended acceptance alone is exactly the trade design doc section 2's
   "Consecuencia para el arnés" warns about: a candidate that swaps
   retrieval quality for factual-answer luck would otherwise read as a plain
   improvement. The fast tier's own regression check only weakly guards
   this (of its 5 retrieval-only ids, 3 already fail today, leaving 2 to
   catch a regression) -- this check runs over all ~9 retrieval-only ids in
   the full search set instead.
4. **A ``inconclusive`` verdict when any record carries
    ``infrastructure_error``.** ``tests/eval/run_eval.py`` marks a case this
    way when it could not be evaluated at all (dead/overloaded Ollama, a
    retrieval exception) and refuses to compare such a run against the
    baseline ("this run measured nothing" -- see ``run_eval.main``). Reading
    those records as ordinary failures would let a dead Ollama read as a
    quality collapse (fast tier: a good candidate wrongly
    ``rejected_regression``) or an artificially low reference silently
    inflate every later candidate into a false ``accepted`` -- exactly the
    "loop sobre una medida que miente produce basura reproducible" failure
    the design doc opens with. The reference measurement itself gets no
    verdict to fall back on: if it carries an infrastructure error,
    ``run_loop`` raises before entering the loop at all, since no ratchet
    baseline can be trusted.
5. **Recovery mode (issue #92): regressions pair against earned passes, not
    lucky ones.** The regression filters compare a candidate against the
    in-run reference -- correct while that reference is healthy, but against
    a deliberately worsened one (criterion 5's setup) it blocks recovery:
    measured on the real pipeline 2026-08-22, the ``RAG_TOP_K_FINAL=1``
    sabotage lucked into passing ``planck-sigma8-es``, a case every healthy
    configuration fails, so every recovery candidate was
    ``rejected_regression`` at the fast tier for re-failing exactly that
    case. When the ledger's prior history contains a *comparable* state
    (same models, chunking and index-time flags -- see
    ``_comparable_config_view``) with a strictly higher objective than the
    in-run reference, the loop starts in recovery mode and pairs both
    regression checks against that high-water state's per-case pass vector
    instead: losing a case only a degraded reference passed is not losing
    ground anyone ever earned. The ratchet itself still starts at the
    degraded reference's objective, so acceptance still requires beating
    it; latency stays paired against the in-run reference (a hard budget,
    not an earned quality). With no comparable history the loop behaves
    exactly as before -- recovery mode is unavailable then, not guessed.

**Scope, stated plainly (not fixed by this PR): stage 1 is a single-field
sweep from a fixed reference, not a compounding hill climb.**
``GridProposer`` always perturbs ``reference`` (per its own docstring,
"coordinate descent from the reference configuration" -- issue #31 spec
section 5.4's own wording), and ``run_loop`` never rebinds ``reference`` or
the reference latency medians to an accepted candidate. Two accepted
single-field changes cannot compound into one two-field configuration
within a single run, and after the first acceptance the ratchet has already
risen, so a further single-field change measured from the *original*
reference is unlikely to clear it -- in practice this makes `patience`
likely to fire not long after the first acceptance. This matches criterion
5 (search recovers from one deliberately worsened point) and the design
doc's own framing of block C as proving the search mechanism works, not as
delivering a multi-step optimizer; widening it to rebind the reference (a
real hill climb) or to search combinations is future work, not silently
implied by "ratchet".

The evaluator is always a callable injected by the caller (``cli.py`` for a
real or demo run, a test double in ``harness/tests/``) -- this module never
imports ``tests.eval.run_eval`` or anything that would let it, which is the
decision that makes the whole harness testable with no GPU and no Ollama
(issue #31 spec section 5.1).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from harness import evaluator as evaluator_mod
from harness import ledger as ledger_mod
from harness import proposers as proposers_mod
from harness import search_space

# NOISE FLOOR

# Measured 2026-07-29 (docs/design/2026-07-28-loop-automejorable.md,
# criterion 1 note): two runs of the same configuration and code, same index
# (both recorded a cache hit), zero flips across all 51 gold cases --
# tests/eval/runs/20260729T020233Z_mineru-jina_clip-faiss.json and
# tests/eval/runs/20260729T040824Z_mineru-jina_clip-faiss.json (both local,
# gitignored, so not reproducible from a fresh clone -- see harness/README.md).
# Independently re-verified in this PR restricted to just the 32-case search
# set this harness actually uses: still 0 flips. A delta of zero cases is not
# an improvement. If tests/eval/grade.py's scoring rules ever change, this
# floor must be remeasured -- it is specific to that grader, not a law of the
# pipeline.
NOISE_FLOOR_CASES = 0

# The latency constraint's multiplier (design doc section 5).
LATENCY_CEILING_MULTIPLIER = 1.20

# RESOLUTION LIMIT (see module docstring point 2)

SEARCH_SET_AVAILABLE_FAILURES = 5
DEMONSTRABILITY_FLIP_THRESHOLD = 6
SEARCH_SET_NET_FLIPS_UNDER_KNOWN_SABOTAGE = 3
SEARCH_SET_SABOTAGE_DESCRIPTION = (
    "RAG_TOP_K_FINAL=1 (single fragment retrieved; criterion 2, 2026-07-29)"
)

RESOLUTION_WARNING: Dict[str, Any] = {
    "available_search_set_failures": SEARCH_SET_AVAILABLE_FAILURES,
    "demonstrability_flip_threshold": DEMONSTRABILITY_FLIP_THRESHOLD,
    "net_flips_under_known_sabotage": SEARCH_SET_NET_FLIPS_UNDER_KNOWN_SABOTAGE,
    "sabotage_description": SEARCH_SET_SABOTAGE_DESCRIPTION,
    "message": (
        "The search set can win at most 5 cases and moved only 3 net flips under "
        "a known-catastrophic single-field sabotage, below the ~6-flip threshold "
        "design doc section 3 sets for a demonstrable paired difference. An "
        "accepted improvement on today's corpus is a candidate for confirmation, "
        "not a demonstrated result -- see "
        "docs/design/2026-07-28-loop-automejorable.md section 3 and issue #30 "
        "(block B, the corpus expansion this harness is sized against)."
    ),
}


def _bucket_stats(records: Sequence[evaluator_mod.CaseRecord], key_fn) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, int]] = {}
    for r in records:
        key = key_fn(r)
        b = buckets.setdefault(key, {"total": 0, "passed": 0})
        b["total"] += 1
        b["passed"] += int(r.passed)
    return {
        key: {**b, "pass_rate": round(b["passed"] / b["total"], 4) if b["total"] else 0.0}
        for key, b in buckets.items()
    }


def _summary(records: Sequence[evaluator_mod.CaseRecord]) -> Dict[str, Any]:
    """Overall pass/total/rate, plus a per-case-type breakdown.

    The per-case-type split (MEDIUM 3 in the #65 review) is what makes a
    retrieval-quality-for-factual-luck trade visible in the ledger even when
    the blended ``objective_adjusted`` does not show it -- mirrors
    ``run_eval.py``'s own ``build_summary``/``_bucket_stats`` shape.
    """
    total = len(records)
    passed = sum(1 for r in records if r.passed)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_case_type": _bucket_stats(records, lambda r: r.case_type),
    }


def _has_infrastructure_error(records: Sequence[evaluator_mod.CaseRecord]) -> bool:
    return any(r.infrastructure_error for r in records)


def _regressed_ids(
    reference_passed_by_id: Dict[str, bool],
    candidate_records: Sequence[evaluator_mod.CaseRecord],
) -> List[str]:
    """Case ids that passed for the pairing baseline and failed for the candidate (paired).

    ``reference_passed_by_id`` is normally the in-run reference's fast-tier
    pass vector; in recovery mode it is the high-water entry's instead
    (module docstring point 5, issue #92).
    """
    return [r.id for r in candidate_records if reference_passed_by_id.get(r.id) and not r.passed]


def _retrieval_only_net_change(
    reference_passed_by_id: Dict[str, bool],
    candidate_records: Sequence[evaluator_mod.CaseRecord],
) -> int:
    """Net change (candidate minus baseline) in retrieval-only passing count, paired by id.

    Positive: more retrieval-only cases pass than in the baseline. Negative:
    fewer do -- what ``run_loop`` refuses to accept regardless of the
    blended objective (see module docstring point 3). The baseline map comes
    from the in-run reference records, or from the high-water entry while in
    recovery mode (point 5).
    """
    net = 0
    for r in candidate_records:
        if r.case_type not in evaluator_mod.RETRIEVAL_ONLY_CASE_TYPES:
            continue
        was_passing = bool(reference_passed_by_id.get(r.id, False))
        if r.passed and not was_passing:
            net += 1
        elif not r.passed and was_passing:
            net -= 1
    return net


def _passed_by_id(records: Sequence[evaluator_mod.CaseRecord]) -> Dict[str, bool]:
    return {r.id: r.passed for r in records}


def _passed_by_id_from_dicts(case_records: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    return {r["id"]: bool(r["passed"]) for r in case_records}


# RECOVERY MODE (issue #92): what makes two ledger entries comparable.

# Index-time flags decide stored chunk text; retrieval/generation flags are
# what candidates vary by design and must never affect comparability. Model
# roles matter (a different generator answers differently); Ollama runtime
# knobs (num_ctx, timeouts, keep-alive) do not decide case outcomes.
_INDEX_TIME_FLAG_KEYS = (
    "usar_contextual_retrieval",
    "usar_embeddings_imagen",
    "usar_descripcion_imagen",
)
_MODEL_ROLE_KEYS = ("rag", "chat", "contextual", "recomp")


def _comparable_config_view(effective_config: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce an entry's effective config to what decides per-case outcomes.

    Two entries are comparable when this view matches: same models, same
    chunking, same index-time flags. Everything a proposer searches
    (retrieval fan-out, fusion weights, reranker threshold) is deliberately
    excluded -- those differences between historical entries are exactly the
    evidence recovery mode needs. Test doubles may carry partial configs;
    missing sections read as ``None`` and compare like-for-like.
    """
    models = effective_config.get("models") or {}
    flags = effective_config.get("flags") or {}
    return {
        "models": {key: models.get(key) for key in _MODEL_ROLE_KEYS},
        "chunking": effective_config.get("chunking"),
        # bool(): entries written before an index-time flag existed lack the
        # key entirely; every such flag ships default-off, so absent == off
        # and an old ledger stays comparable across the schema's growth
        # (issue #92 validation campaign hit exactly this against the
        # 2026-08-19 healthy-campaign entries).
        "index_time_flags": {key: bool(flags.get(key)) for key in _INDEX_TIME_FLAG_KEYS},
    }


def is_comparable_search_set_entry(
    entry: ledger_mod.LedgerEntry, comparable_view: Dict[str, Any]
) -> bool:
    """Whether ``entry`` can serve as a recovery-mode baseline for this view.

    Only complete, non-``inconclusive`` search-set evaluations carry the full
    pass vector recovery pairing needs; fast-tier-rejected entries do not.
    """
    return (
        entry.evaluated_case_set == "search_set"
        and entry.verdict != "inconclusive"
        and _comparable_config_view(entry.effective_config) == comparable_view
    )


def _diff_comparable_views(
    launch_view: Dict[str, Any], entry_view: Dict[str, Any]
) -> List[str]:
    """Human-readable field-level differences between two comparable views."""
    diffs: List[str] = []
    for role in _MODEL_ROLE_KEYS:
        launch_value = launch_view["models"].get(role)
        entry_value = entry_view["models"].get(role)
        if launch_value != entry_value:
            diffs.append(f"models.{role}: this launch {launch_value!r}, ledger {entry_value!r}")
    if launch_view["chunking"] != entry_view["chunking"]:
        diffs.append(
            f"chunking: this launch {launch_view['chunking']!r}, ledger {entry_view['chunking']!r}"
        )
    for flag in _INDEX_TIME_FLAG_KEYS:
        launch_flag = launch_view["index_time_flags"].get(flag)
        entry_flag = entry_view["index_time_flags"].get(flag)
        if launch_flag != entry_flag:
            diffs.append(
                f"index_time_flags.{flag}: this launch {launch_flag!r}, ledger {entry_flag!r}"
            )
    return diffs


def describe_ledger_comparability(reference, entries) -> Dict[str, Any]:
    """What an operator should know about prior history BEFORE paying evaluations.

    Recovery mode's actual arming additionally depends on the reference
    objective, which only a full search-set evaluation reveals -- so this
    reports comparability only and never promises arming (issue #100: a
    silently incomparable ledger is how a campaign spends hours fabricating
    evidence against its own fix).

    Returns:
        A dict with ``history_entries``, ``comparable_search_set_states``,
        ``high_water_objective_adjusted`` (max among comparable states, or
        ``None``), and ``incomparable_reasons`` -- empty unless entries exist
        and none of them is comparable, in which case it names the fields
        that differ against the latest entry.
    """
    comparable_view = _comparable_config_view(dataclasses.asdict(reference))
    comparable = [e for e in entries if is_comparable_search_set_entry(e, comparable_view)]
    info: Dict[str, Any] = {
        "history_entries": len(entries),
        "comparable_search_set_states": len(comparable),
        "high_water_objective_adjusted": max(
            (e.objective_adjusted for e in comparable), default=None
        ),
        "incomparable_reasons": [],
    }
    if entries and not comparable:
        latest = entries[-1]
        diffs = _diff_comparable_views(
            comparable_view, _comparable_config_view(latest.effective_config)
        )
        info["incomparable_reasons"] = diffs or [
            "config matches the latest entry but it carries no complete search-set pass vector"
        ]
    return info


def _historical_high_water(
    entries: Sequence[ledger_mod.LedgerEntry],
    comparable_view: Dict[str, Any],
) -> Optional[ledger_mod.LedgerEntry]:
    """The best comparable search-set state in prior ledger history, or ``None``.

    Only complete, non-``inconclusive`` search-set evaluations count: a
    fast-tier-rejected entry has no full pass vector, and an inconclusive one
    may be measuring a dead Ollama rather than quality. Ties resolve to the
    latest iteration so repeated measurements of one configuration converge.
    Computed once at loop start from PRIOR history only -- entries written by
    the current run never move the pairing baseline mid-campaign.
    """
    best: Optional[ledger_mod.LedgerEntry] = None
    for entry in entries:
        if is_comparable_search_set_entry(entry, comparable_view):
            if best is None or entry.objective_adjusted >= best.objective_adjusted:
                best = entry
    return best


def _latency_breach(
    candidate: Dict[str, Optional[float]], reference: Dict[str, Optional[float]]
) -> Optional[str]:
    """``None`` if both buckets are within ``LATENCY_CEILING_MULTIPLIER`` of reference, else why not."""
    for bucket in ("answered", "retrieval_only"):
        ref_value = reference.get(bucket)
        cand_value = candidate.get(bucket)
        if ref_value is None or cand_value is None:
            continue
        ceiling = ref_value * LATENCY_CEILING_MULTIPLIER
        if cand_value > ceiling:
            return (
                f"{bucket} median {cand_value:.1f}s exceeds ceiling {ceiling:.1f}s "
                f"({LATENCY_CEILING_MULTIPLIER}x reference {ref_value:.1f}s)"
            )
    return None


def _build_entry(
    *,
    iteration: int,
    parent_iteration: Optional[int],
    git_commit: Optional[str],
    overrides: Dict[str, Any],
    meta: Dict[str, Any],
    evaluated_case_set: str,
    result: evaluator_mod.EvaluationResult,
    unreachable_ids: Sequence[str],
    reference_latency: Dict[str, Optional[float]],
    verdict: str,
    reason: str,
    candidate_latency: Optional[Dict[str, Optional[float]]] = None,
    regression_baseline_iteration: Optional[int] = None,
) -> ledger_mod.LedgerEntry:
    objective_raw, objective_adjusted = evaluator_mod.compute_objective(
        result.records, unreachable_ids
    )
    if candidate_latency is None:
        candidate_latency = evaluator_mod.median_latency_by_bucket(result.records)
    return ledger_mod.LedgerEntry(
        schema_version=ledger_mod.SCHEMA_VERSION,
        iteration=iteration,
        parent_iteration=parent_iteration,
        git_commit=git_commit,
        config_overrides=dict(overrides),
        effective_config=dict(result.effective_config),
        proposer=meta.get("proposer", "unknown"),
        proposer_rationale=meta.get("rationale"),
        proposer_model=meta.get("model"),
        proposer_fallback=bool(meta.get("fallback", False)),
        proposer_fallback_reason=meta.get("fallback_reason"),
        evaluated_case_set=evaluated_case_set,
        case_records=[dataclasses.asdict(r) for r in result.records],
        summary=_summary(result.records),
        objective_raw=objective_raw,
        objective_adjusted=objective_adjusted,
        median_latency_answered_s=candidate_latency.get("answered"),
        median_latency_retrieval_only_s=candidate_latency.get("retrieval_only"),
        reference_median_latency_answered_s=reference_latency.get("answered"),
        reference_median_latency_retrieval_only_s=reference_latency.get("retrieval_only"),
        latency_ceiling_multiplier=LATENCY_CEILING_MULTIPLIER,
        verdict=verdict,
        reason=reason,
        regression_baseline_iteration=regression_baseline_iteration,
    )
    if candidate_latency is None:
        candidate_latency = evaluator_mod.median_latency_by_bucket(result.records)
    return ledger_mod.LedgerEntry(
        schema_version=ledger_mod.SCHEMA_VERSION,
        iteration=iteration,
        parent_iteration=parent_iteration,
        git_commit=git_commit,
        config_overrides=dict(overrides),
        effective_config=dict(result.effective_config),
        proposer=meta.get("proposer", "unknown"),
        proposer_rationale=meta.get("rationale"),
        proposer_model=meta.get("model"),
        proposer_fallback=bool(meta.get("fallback", False)),
        proposer_fallback_reason=meta.get("fallback_reason"),
        evaluated_case_set=evaluated_case_set,
        case_records=[dataclasses.asdict(r) for r in result.records],
        summary=_summary(result.records),
        objective_raw=objective_raw,
        objective_adjusted=objective_adjusted,
        median_latency_answered_s=candidate_latency.get("answered"),
        median_latency_retrieval_only_s=candidate_latency.get("retrieval_only"),
        reference_median_latency_answered_s=reference_latency.get("answered"),
        reference_median_latency_retrieval_only_s=reference_latency.get("retrieval_only"),
        latency_ceiling_multiplier=LATENCY_CEILING_MULTIPLIER,
        verdict=verdict,
        reason=reason,
    )


def run_loop(
    *,
    reference: Any,
    evaluate: evaluator_mod.EvaluatorFn,
    proposer: Any,
    search_set_ids: Sequence[str],
    fast_tier_ids: Sequence[str],
    unreachable_ids: Sequence[str] = (),
    max_iterations: Optional[int] = None,
    patience: Optional[int] = 3,
    ledger_dir=ledger_mod.LEDGER_DIR,
    verify_reachability: bool = True,
    reachability_probe_case_ids: Sequence[str] = (),
    reference_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the search until termination, writing one ledger entry per iteration.

    Args:
        reference: The ``AppConfig`` the loop's ratchet, feasibility checks
            and latency ceiling are all measured against (typically
            ``AppConfig.from_env()``). Fixed for the whole run -- see the
            module docstring's "Scope" note.
        evaluate: Injected ``EvaluatorFn`` -- never imported here (spec
            section 5.1).
        proposer: A ``GridProposer`` or ``LlmProposer`` instance.
        search_set_ids: Case ids the objective is computed over.
        fast_tier_ids: Case ids the cheap regression filter runs over.
        unreachable_ids: Case ids excluded from the adjusted objective.
        max_iterations: Hard iteration budget, or ``None`` for no cap.
        patience: Consecutive non-accepted iterations before stopping, or
            ``None`` for no cap. At least one of ``max_iterations``/
            ``patience`` must be set, so the loop is guaranteed to terminate.
            An ``inconclusive`` verdict counts toward patience like any other
            non-accepted outcome.
        ledger_dir: Where to write ledger entries (default: ``harness/ledger/``).
            Meant to be the SAME directory across many invocations over time
            (design doc section 4: "append-only, versioned" spans the whole
            history, not one run) -- ``GridProposer``'s "already tried"
            check and ``next_iteration_number`` both depend on reading it
            back via ``ledger.read_history``.
        verify_reachability: Run ``evaluator.verify_reachable`` before
            measuring the reference. Disabling this is only for tests that
            intentionally use an evaluator too narrow to answer the probe
            (e.g. one that only ever sees a fixed, tiny case set).
        reachability_probe_case_ids: Forwarded to ``evaluator.verify_reachable``.
        reference_overrides: Dotted-key pins (e.g. from ``cli --set``) applied
            to EVERY evaluation this run makes, the reference's included, and
            merged under each candidate's own overrides. Without them the
            reference measurement would silently re-derive its config from
            environment and settings.json while the harness's ``reference``
            object claims otherwise -- a pin that never reaches the measured
            pipeline is how a campaign fabricates a baseline (issue #101,
            caught live 2026-08-23: the sabotaged reference scored healthy-27
            because only the bookkeeping object saw the sabotage).

    Returns:
        The final report: reference stats, ratchet, best iteration, iteration
        count, termination reason, ``resolution_warning``, and every entry
        written this run. Always returned, on every termination path
        (criterion 8) -- including when the search space is exhausted.

    Raises:
        ValueError: Neither ``max_iterations`` nor ``patience`` is set.
        evaluator.ReachabilityError: ``evaluate`` rejects a declared key at
            startup (only when ``verify_reachability`` is True).
        evaluator.InconclusiveEvaluationError: The reference measurement
            itself carries an ``infrastructure_error`` record -- no ratchet
            baseline can be trusted, so the loop never starts.
    """
    if max_iterations is None and patience is None:
        raise ValueError("run_loop needs max_iterations and/or patience to guarantee termination")

    reference_overrides = dict(reference_overrides or {})

    if verify_reachability:
        evaluator_mod.verify_reachable(evaluate, reachability_probe_case_ids)

    reference_full = evaluate(dict(reference_overrides or {}), tuple(search_set_ids))
    reference_fast = evaluate(dict(reference_overrides or {}), tuple(fast_tier_ids))
    if _has_infrastructure_error(reference_full.records) or _has_infrastructure_error(
        reference_fast.records
    ):
        raise evaluator_mod.InconclusiveEvaluationError(
            "reference evaluation carries at least one infrastructure_error record -- "
            "no ratchet baseline can be trusted; fix Ollama/the index and retry"
        )
    reference_objective_raw, reference_objective_adjusted = evaluator_mod.compute_objective(
        reference_full.records, unreachable_ids
    )
    reference_latency = evaluator_mod.median_latency_by_bucket(reference_full.records)

    entries_so_far: List[ledger_mod.LedgerEntry] = list(ledger_mod.read_history(ledger_dir))
    iteration = ledger_mod.next_iteration_number(ledger_dir)
    parent_iteration: Optional[int] = entries_so_far[-1].iteration if entries_so_far else None

    # Recovery mode (issue #92): when prior comparable history measured a
    # strictly better state than the in-run reference, the reference is
    # degraded, so regression pairing switches to the high-water pass
    # vector. Frozen here from PRIOR history; this run's own entries never
    # move it mid-campaign.
    comparable_view = _comparable_config_view(dataclasses.asdict(reference))
    high_water = _historical_high_water(entries_so_far, comparable_view)
    if high_water is not None and high_water.objective_adjusted > reference_objective_adjusted:
        recovery_mode = True
        # The high-water entry carries the full search-set pass vector, which
        # covers every fast-tier id (the fast tier is a search-set subset).
        recovery_pass_vector = _passed_by_id_from_dicts(high_water.case_records)
        fast_regression_baseline = recovery_pass_vector
        search_regression_baseline = recovery_pass_vector
        recovery_baseline_iteration: Optional[int] = high_water.iteration
        eligible_history_entries = sum(
            1
            for e in entries_so_far
            if is_comparable_search_set_entry(e, comparable_view)
        )
    else:
        recovery_mode = False
        fast_regression_baseline = _passed_by_id(reference_fast.records)
        search_regression_baseline = _passed_by_id(reference_full.records)
        recovery_baseline_iteration = None
        eligible_history_entries = 0

    ratchet = reference_objective_adjusted
    best_iteration: Optional[int] = None

    consecutive_non_accepted = 0
    iterations_run = 0
    last_result = reference_full
    new_entries: List[ledger_mod.LedgerEntry] = []
    termination_reason = "unknown"

    while True:
        if max_iterations is not None and iterations_run >= max_iterations:
            termination_reason = "max_iterations"
            break
        if patience is not None and consecutive_non_accepted >= patience:
            termination_reason = "patience"
            break

        try:
            overrides = proposer.propose(search_space, entries_so_far, last_result)
        except proposers_mod.SearchSpaceExhausted:
            termination_reason = "search_space_exhausted"
            break

        meta = dict(getattr(proposer, "last_meta", {}))
        git_commit = ledger_mod.read_git_commit()

        fast_result = evaluate(
            {**reference_overrides, **overrides}, tuple(fast_tier_ids)
        )
        if _has_infrastructure_error(fast_result.records):
            entry = _build_entry(
                iteration=iteration,
                parent_iteration=parent_iteration,
                git_commit=git_commit,
                overrides=overrides,
                meta=meta,
                evaluated_case_set="fast_tier",
                result=fast_result,
                unreachable_ids=unreachable_ids,
                reference_latency=reference_latency,
                verdict="inconclusive",
                reason="fast-tier evaluation carries infrastructure_error record(s) -- not scored",
            )
        else:
            regressed = _regressed_ids(fast_regression_baseline, fast_result.records)

            if regressed:
                baseline_note = (
                    f" (paired against high-water iteration {recovery_baseline_iteration})"
                    if recovery_mode
                    else ""
                )
                entry = _build_entry(
                    iteration=iteration,
                    parent_iteration=parent_iteration,
                    git_commit=git_commit,
                    overrides=overrides,
                    meta=meta,
                    evaluated_case_set="fast_tier",
                    result=fast_result,
                    unreachable_ids=unreachable_ids,
                    reference_latency=reference_latency,
                    verdict="rejected_regression",
                    reason=f"fast-tier regression on {regressed}{baseline_note}",
                    regression_baseline_iteration=recovery_baseline_iteration,
                )
            else:
                full_result = evaluate(
                    {**reference_overrides, **overrides}, tuple(search_set_ids)
                )
                last_result = full_result

                if _has_infrastructure_error(full_result.records):
                    entry = _build_entry(
                        iteration=iteration,
                        parent_iteration=parent_iteration,
                        git_commit=git_commit,
                        overrides=overrides,
                        meta=meta,
                        evaluated_case_set="search_set",
                        result=full_result,
                        unreachable_ids=unreachable_ids,
                        reference_latency=reference_latency,
                        verdict="inconclusive",
                        reason="search-set evaluation carries infrastructure_error record(s) -- not scored",
                    )
                else:
                    candidate_latency = evaluator_mod.median_latency_by_bucket(full_result.records)
                    breach = _latency_breach(candidate_latency, reference_latency)
                    retrieval_net = _retrieval_only_net_change(
                        search_regression_baseline, full_result.records
                    )
                    _objective_raw, objective_adjusted = evaluator_mod.compute_objective(
                        full_result.records, unreachable_ids
                    )

                    if breach:
                        verdict, reason = "rejected_latency", breach
                    elif retrieval_net < 0:
                        verdict = "rejected_regression"
                        baseline_note = (
                            f" vs high-water iteration {recovery_baseline_iteration}"
                            if recovery_mode
                            else " vs reference"
                        )
                        reason = f"retrieval-only bucket lost cases net{baseline_note} ({retrieval_net:+d})"
                    elif objective_adjusted - ratchet <= NOISE_FLOOR_CASES:
                        verdict = "rejected_no_gain"
                        reason = (
                            f"objective_adjusted {objective_adjusted} did not exceed ratchet "
                            f"{ratchet} beyond the noise floor ({NOISE_FLOOR_CASES})"
                        )
                    else:
                        verdict = "accepted"
                        reason = (
                            f"objective_adjusted {objective_adjusted} exceeds ratchet {ratchet}"
                        )
                        ratchet = objective_adjusted
                        best_iteration = iteration

                    entry = _build_entry(
                        iteration=iteration,
                        parent_iteration=parent_iteration,
                        git_commit=git_commit,
                        overrides=overrides,
                        meta=meta,
                        evaluated_case_set="search_set",
                        result=full_result,
                        unreachable_ids=unreachable_ids,
                        reference_latency=reference_latency,
                        verdict=verdict,
                        reason=reason,
                        candidate_latency=candidate_latency,
                        # Provenance on every SCORED entry, accepted ones
                        # included: which pass vector the regression checks
                        # paired against is part of the evidence.
                        regression_baseline_iteration=recovery_baseline_iteration,
                    )

        ledger_mod.write_entry(entry, ledger_dir)
        entries_so_far.append(entry)
        new_entries.append(entry)

        consecutive_non_accepted = (
            0 if entry.verdict == "accepted" else consecutive_non_accepted + 1
        )
        parent_iteration = entry.iteration
        iteration += 1
        iterations_run += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "objective_raw": reference_objective_raw,
            "objective_adjusted": reference_objective_adjusted,
            "median_latency_answered_s": reference_latency.get("answered"),
            "median_latency_retrieval_only_s": reference_latency.get("retrieval_only"),
            # Provenance: what every measurement this run actually applied,
            # so a pinned campaign is auditable from the report alone.
            "overrides_applied": dict(reference_overrides),
        },
        "ratchet": ratchet,
        "best_iteration": best_iteration,
        "iterations_run": iterations_run,
        "termination_reason": termination_reason,
        "recovery_mode": {
            # Issue #92: whether regression pairing ran against the ledger's
            # historical high-water state instead of the in-run reference.
            "active": recovery_mode,
            "baseline_iteration": recovery_baseline_iteration,
            "baseline_objective_adjusted": (
                high_water.objective_adjusted if recovery_mode and high_water is not None else None
            ),
            "eligible_history_entries": eligible_history_entries,
        },
        "resolution_warning": RESOLUTION_WARNING,
        "iterations": [e.to_dict() for e in new_entries],
    }
