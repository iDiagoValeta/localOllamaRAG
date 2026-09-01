"""ledger -- append-only evidence ledger for block-C iterations.

One JSON file per iteration under ``harness/ledger/`` plus a summary line in
``harness/ledger/index.jsonl``, carrying enough to satisfy criterion 7
(docs/design/2026-07-28-loop-automejorable.md section 2): reconstruct the
configuration an iteration ran with, and get the same result within the
noise floor.

Writing is the whole extent of this module's git awareness: it never adds,
commits or pushes anything. Design section 5 -- "el loop propone, el humano
integra" -- and issue #31 spec section 1 make that a hard prohibition, not a
default; ``harness/tests/test_harness_boundaries.py`` asserts no module under
``harness/`` imports ``subprocess`` (or any git-shelling mechanism) at all,
which is why ``_read_git_commit`` below reads ``.git`` metadata directly
instead of shelling out to ``git rev-parse HEAD``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 4

LEDGER_DIR = Path(__file__).resolve().parent / "ledger"
INDEX_FILE_NAME = "index.jsonl"

# Matches exactly what write_entry names an entry file (f"{iteration:04d}_{timestamp}.json"),
# so next_iteration_number/read_history never pick up another *.json file a
# caller happens to drop in the same directory -- cli.py's own report.json
# lives right next to these (issue #65 PR review, HIGH 3): a naive `*.json`
# glob fed report.json into LedgerEntry.from_dict and crashed with
# "unexpected keyword argument 'generated_at'" on the very next invocation
# against a non-empty ledger_dir, which the default harness/ledger/ becomes
# after one real run.
_ENTRY_FILENAME_RE = re.compile(r"^\d{4}_.*\.json$")

# Which evaluator produced an entry (issue #115). `None` is a fourth state and
# the default: an entry written before schema v4 CANNOT be known to be real,
# because a --dry-run pointed at that ledger is exactly the defect this field
# closes. Asserting "real" for them would repeat the mistake #107 avoided --
# treating an absent record as a positive claim.
EVALUATOR_DEMO = "demo"
EVALUATOR_REAL = "real"

VERDICTS = (
    "accepted",
    "rejected_regression",
    "rejected_latency",
    "rejected_no_gain",
    "inconclusive",
)


@dataclasses.dataclass(frozen=True)
class LedgerEntry:
    """One iteration's full evidence record (issue #31 spec section 5.3).

    Attributes:
        schema_version: Bumped whenever a field is added, renamed or removed,
            so an old entry can be told apart from a new one by a reader that
            has not been updated yet.
        iteration: 1-based sequence number.
        parent_iteration: Which entry this candidate was proposed from, or
            ``None`` for the first iteration (proposed from the reference).
        git_commit: SHA the evaluation ran at, or ``None`` if it could not be
            determined (see ``read_git_commit``).
        config_overrides: The raw proposer deltas -- NOT expanded (e.g.
            ``retrieval.weight_bm25_rrf`` stays implicit if the proposer only
            set ``weight_semantic_rrf``). Reconstructing the same effective
            config from this requires ``search_space.expand_overrides``.
        effective_config: ``dataclasses.asdict`` of the full ``AppConfig``
            this iteration actually ran with -- kept alongside the deltas so
            a later default change cannot silently change what this entry
            meant.
        proposer: ``"grid"`` or ``"llm"``.
        proposer_rationale: The LLM's verbatim rationale, or ``None`` for grid
            (or for an LLM call that fell back to grid).
        proposer_model: Ollama model id that produced the rationale, or
            ``None``.
        proposer_fallback: Whether an ``LlmProposer`` fell back to
            ``GridProposer`` this iteration.
        proposer_fallback_reason: Why, if ``proposer_fallback`` is True.
        evaluated_case_set: ``"fast_tier"`` if this iteration was rejected at
            the regression-filter stage (no full search-set evaluation ran),
            else ``"search_set"``.
        case_records: Per-case ``{id, case_type, passed, elapsed_seconds}``
            from whichever set ``evaluated_case_set`` names.
        summary: ``{total, passed, pass_rate}`` over ``case_records``.
        objective_raw: Passing-case count, unreachable cases included.
        objective_adjusted: Passing-case count, unreachable cases excluded
            (``evaluator.compute_objective``). What the ratchet tracks.
        median_latency_answered_s: This iteration's answered-case median, or
            ``None`` if ``case_records`` has no answered case.
        median_latency_retrieval_only_s: This iteration's retrieval-only
            median, or ``None``.
        reference_median_latency_answered_s: The reference configuration's
            answered-case median, frozen at loop start -- what
            ``median_latency_answered_s`` is checked against.
        reference_median_latency_retrieval_only_s: Same, retrieval-only.
        latency_ceiling_multiplier: The constraint's multiplier (1.20, design
            doc section 5) -- recorded per entry rather than assumed constant,
            so a later change to the ceiling does not retroactively change
            what an old entry's ``rejected_latency`` verdict meant.
        verdict: One of ``VERDICTS``.
        reason: Human-readable explanation (which cases regressed, which
            bucket breached the ceiling and by how much, or the gain over the
            ratchet).
        regression_baseline_iteration: The ledger iteration whose per-case
            pass vector regressions were paired against, or ``None`` when
            that was the in-run reference itself (schema v2; absent in v1
            files, where it is always ``None``). Set to the high-water
            entry's iteration when the loop ran in recovery mode -- issue
            #92: against a degraded reference, pairing against the reference
            would reject every candidate that restores healthy behaviour,
            because the sabotage can luck into passing a case the healthy
            pipeline fails. See ``loop._historical_high_water``.
        environment_fingerprint: The installed stack this iteration was
            measured on (``harness.environment.environment_fingerprint``), or
            ``None`` when it could not be read -- and always ``None`` in v1/v2
            files, which predate the field (schema v3, issue #107).
            ``effective_config`` records what was *configured*; this records
            what was *installed*, and the two drift independently: a MinerU
            upgrade or an upstream fix changes what a case can pass without
            moving a single config value, which is how a healthy August
            baseline came to hold passes the current stack cannot reach.
            Unknown is deliberately not the same as different --
            ``environment.compare`` returns three answers, and
            ``loop.is_comparable_search_set_entry`` rejects only a KNOWN
            mismatch, so pre-v3 history stays usable and is reported as
            unverified rather than silently trusted.
        evaluator: ``EVALUATOR_REAL``, ``EVALUATOR_DEMO``, or ``None`` for an
            entry written before schema v4 (issue #115). ``None`` means
            unknown, not real: a ``--dry-run --ledger-dir <a real ledger>``
            used to append its synthetic iterations here with nothing to tell
            them apart, and a demo entry is not inert once written --
            ``GridProposer`` reads history back to skip points already tried
            and ``loop._historical_high_water`` picks a pairing baseline from
            it, so a synthetic iteration that scored well became a baseline a
            real campaign was judged against.
    """

    schema_version: int
    iteration: int
    parent_iteration: Optional[int]
    git_commit: Optional[str]
    config_overrides: Dict[str, Any]
    effective_config: Dict[str, Any]
    proposer: str
    proposer_rationale: Optional[str]
    proposer_model: Optional[str]
    proposer_fallback: bool
    proposer_fallback_reason: Optional[str]
    evaluated_case_set: str
    case_records: List[Dict[str, Any]]
    summary: Dict[str, Any]
    objective_raw: int
    objective_adjusted: int
    median_latency_answered_s: Optional[float]
    median_latency_retrieval_only_s: Optional[float]
    reference_median_latency_answered_s: Optional[float]
    reference_median_latency_retrieval_only_s: Optional[float]
    latency_ceiling_multiplier: float
    verdict: str
    reason: str
    regression_baseline_iteration: Optional[int] = None
    environment_fingerprint: Optional[Dict[str, Any]] = None
    evaluator: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        return cls(**data)


def _find_git_dir(start: Path) -> Optional[Path]:
    """Walk upward from ``start`` looking for a ``.git`` file or directory."""
    current = start
    for _ in range(20):
        candidate = current / ".git"
        if candidate.exists():
            return candidate
        if current.parent == current:
            return None
        current = current.parent
    return None


def _resolve_ref(common_dir: Path, ref_path: str) -> Optional[str]:
    """Resolve ``ref_path`` (e.g. "refs/heads/main") to a commit sha.

    Tries the loose ref file first, then ``packed-refs`` -- a repository with
    many branches (this one included) periodically packs old refs there.
    """
    direct = common_dir / ref_path
    if direct.exists():
        return direct.read_text(encoding="utf-8").strip() or None
    packed = common_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[1] == ref_path:
                return parts[0]
    return None


def read_git_commit(start: Optional[Path] = None) -> Optional[str]:
    """Best-effort current commit sha, read from ``.git`` metadata directly.

    Deliberately avoids ``subprocess``/GitPython: this is what lets
    ``harness/tests/test_harness_boundaries.py`` assert no module under
    ``harness/`` imports ``subprocess`` at all, closing off any git-write
    code path structurally (issue #31 spec section 1, item 5) rather than by
    convention.

    Handles both a plain ``.git`` directory and a worktree's ``.git`` file
    (``gitdir: <path>``, as ``git worktree add`` produces, and what this
    harness is developed under), including the worktree's separate
    per-worktree ``HEAD`` and its ``commondir`` pointer back to the shared
    refs. Never raises: a missing or unexpected ``.git`` layout returns
    ``None`` rather than crashing the loop over a nice-to-have field.

    Args:
        start: Where to start walking upward for ``.git``. Defaults to this
            file's own directory.

    Returns:
        The commit sha HEAD points at, or ``None``.
    """
    try:
        git_path = _find_git_dir(start or Path(__file__).resolve().parent)
        if git_path is None:
            return None

        if git_path.is_dir():
            git_dir = git_path
            common_dir = git_dir
        else:
            content = git_path.read_text(encoding="utf-8").strip()
            if not content.startswith("gitdir:"):
                return None
            gitdir_value = content.split(":", 1)[1].strip()
            git_dir = Path(gitdir_value)
            if not git_dir.is_absolute():
                git_dir = (git_path.parent / git_dir).resolve()
            commondir_file = git_dir / "commondir"
            if commondir_file.exists():
                common_rel = commondir_file.read_text(encoding="utf-8").strip()
                common_dir = (git_dir / common_rel).resolve()
            else:
                common_dir = git_dir

        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return None
        head = head_file.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            return _resolve_ref(common_dir, head.split(":", 1)[1].strip())
        return head or None
    except OSError:
        return None


def _entry_files(ledger_dir: Path) -> List[Path]:
    """Every ``NNNN_<timestamp>.json`` file directly under ``ledger_dir``.

    Deliberately NOT a bare ``*.json`` glob: ``harness/cli.py`` writes
    ``report.json`` into the same directory as a normal part of a run, and a
    naive glob would feed it (and anything else dropped in later) to
    ``LedgerEntry.from_dict`` -- see ``_ENTRY_FILENAME_RE``'s comment.
    """
    if not ledger_dir.exists():
        return []
    return [p for p in ledger_dir.glob("*.json") if _ENTRY_FILENAME_RE.match(p.name)]


def next_iteration_number(ledger_dir: Path = LEDGER_DIR) -> int:
    """1 for an empty/missing ledger dir, else one past the highest recorded iteration."""
    highest = 0
    for path in _entry_files(ledger_dir):
        highest = max(highest, int(path.name.split("_", 1)[0]))
    return highest + 1


def write_entry(entry: LedgerEntry, ledger_dir: Path = LEDGER_DIR) -> Path:
    """Write ``entry``'s full JSON and append its summary line to the index.

    Append-only: never overwrites a previous iteration's file, and the index
    is opened in append mode. Writing is local-filesystem only -- see this
    module's docstring for why nothing here ever touches git.

    Args:
        entry: The iteration to record.
        ledger_dir: Defaults to ``harness/ledger/``; overridden by tests and
            by ``harness/cli.py --dry-run`` (see that module) so a demo run
            never writes into the real ledger.

    Returns:
        Path to the entry's own JSON file.
    """
    ledger_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entry_path = ledger_dir / f"{entry.iteration:04d}_{timestamp}.json"
    entry_path.write_text(
        json.dumps(entry.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    index_line = json.dumps(
        {
            "iteration": entry.iteration,
            "parent_iteration": entry.parent_iteration,
            "proposer": entry.proposer,
            "verdict": entry.verdict,
            "objective_adjusted": entry.objective_adjusted,
            "entry_file": entry_path.name,
        },
        ensure_ascii=False,
    )
    with (ledger_dir / INDEX_FILE_NAME).open("a", encoding="utf-8") as fh:
        fh.write(index_line + "\n")

    return entry_path


def read_entry(path: Path) -> LedgerEntry:
    """Load one ledger entry JSON file back into a ``LedgerEntry``."""
    return LedgerEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))


def read_history(ledger_dir: Path = LEDGER_DIR) -> List[LedgerEntry]:
    """Every entry under ``ledger_dir``, oldest iteration first.

    Reads the full per-entry JSON files (not just the index summary), since
    proposers need each entry's ``config_overrides`` to avoid repeating a
    point already tried (criterion-adjacent to spec test 7). Only files
    matching the entry naming pattern are read -- see ``_entry_files``.
    """
    entries = [read_entry(p) for p in _entry_files(ledger_dir)]
    return sorted(entries, key=lambda e: e.iteration)


def read_entry_by_iteration(ledger_dir: Path, iteration: int) -> LedgerEntry:
    """The entry whose ``iteration`` matches, or ``FileNotFoundError``.

    Criterion 7's reconstruct-and-rerun path (``harness.cli --replay``)
    looks an entry up this way rather than by filename: the filename's
    timestamp is incidental, the iteration number is what the index and
    ``parent_iteration`` links use.
    """
    for entry in read_history(ledger_dir):
        if entry.iteration == iteration:
            return entry
    raise FileNotFoundError(f"no ledger entry for iteration {iteration} under {ledger_dir}")
