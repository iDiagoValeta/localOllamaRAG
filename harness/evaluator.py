"""evaluator -- adapts tests/eval/run_eval's evaluation to the harness loop.

Owns everything about "what gets asked of the evaluation": which case ids
belong to the search set, the blind set and the fast tier; the objective
(raw and unreachable-adjusted pass count) and the per-bucket latency medians
computed from an evaluation's records; the reachability gate that fails
startup instead of the loop silently running on top of an inert knob; and
the ``EvaluatorFn`` adapter to the real ``tests/eval/run_eval.evaluate()``.

The harness may read and run the evaluation, never redefine how it scores
(issue #31 spec section 0). Nothing here imports ``grade`` -- run_eval.py
does, for its own purposes, and this module imports *run_eval*, not grade,
exactly as harness/tests/test_harness_boundaries.py checks. ``grade.py``,
``gold_cases.jsonl`` and ``baseline_min_pass_rate.txt`` are only ever read.

Two run-record facts drove this module's shape, both confirmed against a
real gate run (``tests/eval/runs/20260729T020233Z_mineru-jina_clip-faiss.json``,
a local, gitignored artifact -- see harness/README.md):

- A record has no ``source`` field (``case_type``, ``elapsed_seconds``,
  ``id``, ``lang``, ``model``, ``paper``, ``passed``, ``reason`` only). Search
  vs. blind membership is never read off a record -- it is decided entirely
  by which case ids the caller asks ``evaluate()`` for, sourced from
  ``search_set_case_ids()``/``load_fast_tier()`` (both filtered from
  ``gold_cases.jsonl`` by ``source``) and never from anything a proposer or
  an evaluation result controls. That is what makes the blind set
  structurally unreachable (spec section 4, test 5).
- Per-case-type latency differs by roughly a factor of fifty (answered
  ~200 s vs. retrieval-only ~4 s), so a single blended median is dominated by
  which bucket happened to be larger and would mask a retrieval-quality
  collapse a candidate paid for with fewer generator calls. The latency
  constraint (``loop.py``) is therefore per-bucket, not blended --
  ``median_latency_by_bucket`` is what makes that possible.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from harness import search_space

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "tests" / "eval"
GOLD_FILE = EVAL_DIR / "gold_cases.jsonl"
FAST_TIER_FILE = Path(__file__).resolve().parent / "fast_tier.txt"
UNREACHABLE_FILE = Path(__file__).resolve().parent / "unreachable_cases.txt"

# Mirrors tests/eval/run_eval.py's private _RETRIEVAL_ONLY_CASE_TYPES: these
# case types skip generation entirely (a pass means retrieval surfaced the
# right content kind, not that any answer used it), which is exactly why
# their latency belongs in its own bucket. Duplicated rather than imported
# because it is a leading-underscore name in a module this one only imports
# lazily (see _run_eval_module) -- harness/tests/test_evaluator.py asserts
# this tuple stays equal to run_eval's, so the two cannot silently drift.
RETRIEVAL_ONLY_CASE_TYPES: Tuple[str, ...] = ("figure_retrieval", "table_retrieval")


@dataclass(frozen=True)
class CaseRecord:
    """One graded case, the fields every ledger entry and latency bucket needs.

    Attributes:
        id: Gold-case id (``gold_cases.jsonl``'s ``"id"``).
        case_type: One of ``factual_number``, ``factual_concept``,
            ``figure_retrieval``, ``table_retrieval``.
        passed: Grading verdict.
        elapsed_seconds: Wall time for this case (retrieval, and generation
            when the case type calls the generator).
    """

    id: str
    case_type: str
    passed: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class EvaluationResult:
    """What one ``EvaluatorFn`` call returns.

    Attributes:
        records: One ``CaseRecord`` per requested case id.
        effective_config: ``dataclasses.asdict`` of the ``AppConfig`` this
            evaluation actually ran with (reference config plus the expanded
            overrides) -- kept alongside the raw ``config_overrides`` deltas
            in the ledger so a default changing later cannot silently change
            what an old entry meant (issue #31 spec section 5.3).
    """

    records: Tuple[CaseRecord, ...]
    effective_config: Dict[str, Any]


EvaluatorFn = Callable[[Mapping[str, Any], Sequence[str]], EvaluationResult]


def median_latency_by_bucket(records: Sequence[CaseRecord]) -> Dict[str, Optional[float]]:
    """Median wall time, split into the answered and retrieval-only buckets.

    Per-bucket, not blended -- see this module's docstring for why a single
    median over case types that differ by ~50x in cost is not a meaningful
    latency signal.

    Args:
        records: Case records from one evaluation.

    Returns:
        ``{"answered": median_or_None, "retrieval_only": median_or_None}``.
        ``None`` for a bucket with zero records in ``records`` (e.g. a
        fast-tier probe evaluated with only retrieval-only ids).
    """
    answered = [r.elapsed_seconds for r in records if r.case_type not in RETRIEVAL_ONLY_CASE_TYPES]
    retrieval_only = [r.elapsed_seconds for r in records if r.case_type in RETRIEVAL_ONLY_CASE_TYPES]
    return {
        "answered": statistics.median(answered) if answered else None,
        "retrieval_only": statistics.median(retrieval_only) if retrieval_only else None,
    }


def compute_objective(records: Sequence[CaseRecord], unreachable_ids: Iterable[str] = ()) -> Tuple[int, int]:
    """Raw and unreachable-adjusted passing-case counts (issue #31 spec section 3).

    A case in ``unreachable_ids`` is dropped from consideration entirely for
    the adjusted count -- neither a pass nor a fail counts against it -- so
    the loop never spends iterations chasing a case no stage-1 configuration
    can fix (design doc section 3, "Margen inalcanzable"). Today
    ``unreachable_cases.txt`` is empty (evidence-backed, see that file's
    header), so ``raw == adjusted`` for every real evaluation; the two are
    kept distinct here so that stays true by construction rather than by
    coincidence once a case earns a place there.

    Args:
        records: Case records to score.
        unreachable_ids: Ids to exclude from the adjusted count.

    Returns:
        ``(raw, adjusted)``, both plain counts, not rates.
    """
    unreachable = set(unreachable_ids)
    raw = sum(1 for r in records if r.passed)
    adjusted = sum(1 for r in records if r.passed and r.id not in unreachable)
    return raw, adjusted


# GOLD CASES / CASE SETS

_gold_cases_cache: Optional[List[Dict[str, Any]]] = None


def _gold_cases() -> List[Dict[str, Any]]:
    """Parse gold_cases.jsonl once and cache it (read-only, never written)."""
    global _gold_cases_cache
    if _gold_cases_cache is None:
        cases = []
        for line in GOLD_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cases.append(json.loads(line))
        _gold_cases_cache = cases
    return _gold_cases_cache


def search_set_case_ids() -> Tuple[str, ...]:
    """Case ids with ``source == "corpus"`` -- the loop's objective (spec section 4)."""
    return tuple(c["id"] for c in _gold_cases() if c["source"] == "corpus")


def blind_set_case_ids() -> Tuple[str, ...]:
    """Case ids with ``source == "arxiv"`` -- never requested by the harness."""
    return tuple(c["id"] for c in _gold_cases() if c["source"] == "arxiv")


def _read_id_list(path: Path) -> List[str]:
    """Parse a harness case-id file: one id per non-comment, non-blank line.

    A trailing justification/comment after the id (space- or ``#``-separated)
    is accepted and ignored -- ``unreachable_cases.txt`` uses it, ``fast_tier.txt``
    does not need to.
    """
    ids = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split()[0])
    return ids


def load_fast_tier(path: Path = FAST_TIER_FILE) -> Tuple[str, ...]:
    """Fixed regression-filter case ids (spec section 4); validated as a search-set subset.

    Raises:
        ValueError: ``path`` names an id outside ``search_set_case_ids()``.
    """
    ids = tuple(_read_id_list(path))
    search_ids = set(search_set_case_ids())
    unknown = [i for i in ids if i not in search_ids]
    if unknown:
        raise ValueError(f"{path}: case id(s) outside the search set: {unknown}")
    return ids


def load_unreachable_ids(path: Path = UNREACHABLE_FILE) -> Tuple[str, ...]:
    """Case ids excluded from the maximised scalar (spec section 3); validated as a search-set subset.

    Raises:
        ValueError: ``path`` names an id outside ``search_set_case_ids()``.
    """
    ids = tuple(_read_id_list(path))
    search_ids = set(search_set_case_ids())
    unknown = [i for i in ids if i not in search_ids]
    if unknown:
        raise ValueError(f"{path}: case id(s) outside the search set: {unknown}")
    return ids


# REACHABILITY GATE


class ReachabilityError(RuntimeError):
    """Raised when ``evaluate()`` rejects a declared search-space key at startup.

    Not a bug in the harness: it means ``evaluate()`` is doing exactly what
    the hard-fail policy (rule 8) asks of it -- refusing to silently ignore
    an override it cannot honour end to end. The harness's job is to surface
    that loudly, before the loop's first iteration, rather than let a
    candidate "improve" on a knob that never reached the pipeline.
    """


def verify_reachable(evaluate: EvaluatorFn, probe_case_ids: Sequence[str] = ()) -> None:
    """Fail loudly if ``evaluate`` cannot honour a declared search-space key.

    Calls ``evaluate`` exactly once, with every declared key set to its
    first allowed value simultaneously, and lets whatever ``evaluate`` raises
    propagate wrapped in ``ReachabilityError``. One call, not one per key: a
    per-key probe against the real evaluator would cost up to
    ``len(search_space.DECLARED_KEYS)`` full evaluations before the loop even
    starts, which defeats the point of a startup gate. ``probe_case_ids``
    defaults to empty, on the assumption that a hard-fail check belongs at
    config-construction time (this codebase's own pattern -- see
    ``AppConfig.with_overrides``) rather than requiring a case to actually
    run; this is documented as an assumption, not a guarantee, because this
    harness does not control ``evaluate()``'s implementation (issue #31,
    sibling PR #56). If ``evaluate()`` only detects an unreachable override
    while processing a case, a caller must pass at least one id here for the
    gate to be meaningful.

    ``search_space.PENDING_REACHABILITY_KEYS`` is empty today -- #56 audited
    and closed the reachability gap it used to name -- but this gate stays:
    the same audit also found ``flags.usar_llm_query_decomposition`` silently
    inert for a different reason (the gate hardcodes its collaborator to
    ``None``, issue #64) and simply removed that key from
    ``search_space.SEARCH_SPACE`` instead, since ``evaluate()`` would not
    raise for it either. This gate is what turns a *future* key like either
    of those into a loud startup failure the moment ``evaluate()`` starts
    enforcing it, instead of a silent no-op discovered by reading a
    suspiciously flat ledger.

    Args:
        evaluate: The evaluator the loop is about to run with.
        probe_case_ids: Case ids to pass on the probe call (default: none).

    Raises:
        ReachabilityError: ``evaluate`` raised for the combined probe
            overrides; the original exception is chained and its message
            (expected to name the offending key(s), per #56) is included.
    """
    probe_overrides = {key: search_space.ALLOWED_VALUES[key][0] for key in search_space.DECLARED_KEYS}
    try:
        evaluate(probe_overrides, tuple(probe_case_ids))
    except Exception as exc:  # noqa: BLE001 -- any failure here must name the key, never be swallowed
        raise ReachabilityError(
            "evaluate() rejected one or more declared search-space keys at startup "
            f"(see search_space.PENDING_REACHABILITY_KEYS): {exc}"
        ) from exc


# REAL EVALUATOR (depends on sibling PR #56)


def _run_eval_module():
    """Import tests/eval/run_eval.py as a top-level ``run_eval`` module.

    Mirrors tests/eval/test_index_reuse.py's own ``sys.path.insert`` +
    ``import run_eval`` convention, so this is the one place harness/ reaches
    into tests/eval/ from, and every caller resolves to the same module
    object.
    """
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    import run_eval  # noqa: E402

    return run_eval


def real_evaluate(config_overrides: Mapping[str, Any], case_ids: Sequence[str]) -> EvaluationResult:
    """Adapt ``tests/eval/run_eval.evaluate()`` (issue #31 spec section 5.2) to ``EvaluatorFn``.

    TODO: depends on the sibling PR (#56) landing ``run_eval.evaluate()`` on
    main. Guarded rather than imported at module level so importing
    ``harness.evaluator`` (which ``harness/tests/`` does, in CI, with no
    engine dependencies installed) never fails on this function's account --
    only *calling* it before #56 lands does, with a message that says why.

    Args:
        config_overrides: Dotted-key overrides, as a proposer emits them
            (``retrieval.weight_bm25_rrf`` is derived here via
            ``search_space.expand_overrides``, not required from the caller).
        case_ids: Case ids to evaluate.

    Returns:
        The adapted ``EvaluationResult``.

    Raises:
        NotImplementedError: ``run_eval.py`` has no ``evaluate()`` yet.
    """
    run_eval = _run_eval_module()
    if not hasattr(run_eval, "evaluate"):
        raise NotImplementedError(
            "tests/eval/run_eval.py has no evaluate() yet -- the real evaluator "
            "depends on the sibling PR (issue #31 spec section 5.2, #56). Use "
            "evaluator.build_demo_evaluator() to exercise the harness without it."
        )
    raw = run_eval.evaluate(
        models=run_eval.DEFAULT_MODELS,
        case_ids=tuple(case_ids),
        config_overrides=search_space.expand_overrides(config_overrides),
        update_baseline=False,
        write_report=True,
    )
    records = tuple(
        CaseRecord(
            id=r["id"],
            case_type=r["case_type"],
            passed=bool(r["passed"]),
            elapsed_seconds=float(r["elapsed_seconds"]),
        )
        for r in raw["results"]
    )
    return EvaluationResult(records=records, effective_config=raw.get("effective_config", {}))


# DEMO EVALUATOR (no GPU, no Ollama -- harness/cli.py --dry-run)


def build_demo_evaluator(seed_overrides: Optional[Mapping[str, Any]] = None) -> EvaluatorFn:
    """A tiny deterministic in-process evaluator, for smoke-testing the CLI/loop wiring.

    Not a substitute for the real pipeline and never claims to be: it scores
    a candidate purely by how many of a handful of declared keys match a
    fixed synthetic optimum, with fixed per-case-type latencies. It exists so
    ``harness/cli.py --dry-run`` can exercise the whole loop -- proposer,
    feasibility, ledger, the latency constraint, termination -- with no GPU,
    no Ollama and no PDFs. See ``harness/tests/`` for the same idea used with
    landscapes shaped to test specific loop behaviors (regression, latency
    rejection, recovery); this one is only for the CLI demo.

    Args:
        seed_overrides: Dotted-key values the synthetic landscape treats as
            optimal. Defaults to a fixed point so repeated ``--dry-run``
            invocations are comparable.

    Returns:
        An ``EvaluatorFn`` closed over the synthetic scoring function.
    """
    optimum = dict(seed_overrides or {"retrieval.top_k_final": 8, "retrieval.weight_semantic_rrf": 0.55})
    all_ids = search_set_case_ids()
    case_types = {c["id"]: c["case_type"] for c in _gold_cases()}

    def _evaluate(config_overrides: Mapping[str, Any], case_ids: Sequence[str]) -> EvaluationResult:
        effective = search_space.expand_overrides(config_overrides)
        distance = sum(0 if effective.get(key) == value else 1 for key, value in optimum.items())
        ids = tuple(case_ids) or all_ids
        records = []
        for index, case_id in enumerate(ids):
            case_type = case_types.get(case_id, "factual_number")
            is_retrieval_only = case_type in RETRIEVAL_ONLY_CASE_TYPES
            # Deterministic, not random: farther from the synthetic optimum
            # fails a larger fraction of cases (smaller modulus -> more
            # failures), and index 0 always fails once distance > 0 so a
            # perturbation is guaranteed to show up in a 1-case probe too.
            passed = distance == 0 or (index % (distance + 1)) != 0
            records.append(CaseRecord(
                id=case_id, case_type=case_type, passed=passed,
                elapsed_seconds=4.0 if is_retrieval_only else 200.0,
            ))
        return EvaluationResult(records=tuple(records), effective_config=effective)

    return _evaluate
