"""Tests for harness/ledger.py -- issue #31 spec section 6, test 10, plus
the append-only read/write mechanics loop.py depends on.
"""

import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from monkeygrab.config.app_config import AppConfig
from harness import ledger, search_space


def _entry(iteration=1, overrides=None, effective_config=None, verdict="accepted"):
    overrides = overrides or {}
    return ledger.LedgerEntry(
        schema_version=ledger.SCHEMA_VERSION,
        iteration=iteration,
        parent_iteration=None,
        git_commit="deadbeef",
        config_overrides=overrides,
        effective_config=effective_config or {},
        proposer="grid",
        proposer_rationale=None,
        proposer_model=None,
        proposer_fallback=False,
        proposer_fallback_reason=None,
        evaluated_case_set="search_set",
        case_records=[{"id": "a", "case_type": "factual_number", "passed": True, "elapsed_seconds": 1.0}],
        summary={"total": 1, "passed": 1, "pass_rate": 1.0},
        objective_raw=1,
        objective_adjusted=1,
        median_latency_answered_s=200.0,
        median_latency_retrieval_only_s=4.0,
        reference_median_latency_answered_s=200.0,
        reference_median_latency_retrieval_only_s=4.0,
        latency_ceiling_multiplier=1.20,
        verdict=verdict,
        reason="test entry",
    )


# TEST 10 -- a ledger entry round-trips: reconstruct the config from an
# entry and get the same overrides (criterion 7).


def test_ledger_entry_round_trips_through_json(tmp_path):
    original = _entry(overrides={"retrieval.top_k_final": 12})
    path = ledger.write_entry(original, ledger_dir=tmp_path)
    loaded = ledger.read_entry(path)
    assert loaded == original


def test_ledger_entry_reconstructs_the_same_effective_config(tmp_path):
    base = AppConfig()
    overrides = {"retrieval.top_k_final": 12, "retrieval.weight_semantic_rrf": 0.7}
    effective = base.with_overrides(**search_space.expand_overrides(overrides))
    original = _entry(overrides=overrides, effective_config=dataclasses.asdict(effective))

    path = ledger.write_entry(original, ledger_dir=tmp_path)
    loaded = ledger.read_entry(path)

    assert loaded.config_overrides == overrides
    reconstructed = base.with_overrides(**search_space.expand_overrides(loaded.config_overrides))
    assert dataclasses.asdict(reconstructed) == loaded.effective_config


# APPEND-ONLY MECHANICS.


def test_write_entry_appends_to_index_without_touching_earlier_entries(tmp_path):
    first = _entry(iteration=1)
    second = _entry(iteration=2, overrides={"retrieval.top_k_final": 4})
    path1 = ledger.write_entry(first, ledger_dir=tmp_path)
    path2 = ledger.write_entry(second, ledger_dir=tmp_path)

    assert path1 != path2
    assert ledger.read_entry(path1) == first
    assert ledger.read_entry(path2) == second

    index_lines = (tmp_path / ledger.INDEX_FILE_NAME).read_text(encoding="utf-8").splitlines()
    assert len(index_lines) == 2


def test_next_iteration_number_is_one_for_an_empty_ledger(tmp_path):
    assert ledger.next_iteration_number(tmp_path) == 1


def test_next_iteration_number_continues_after_existing_entries(tmp_path):
    ledger.write_entry(_entry(iteration=1), ledger_dir=tmp_path)
    ledger.write_entry(_entry(iteration=5), ledger_dir=tmp_path)
    assert ledger.next_iteration_number(tmp_path) == 6


def test_read_history_is_sorted_by_iteration(tmp_path):
    ledger.write_entry(_entry(iteration=3), ledger_dir=tmp_path)
    ledger.write_entry(_entry(iteration=1), ledger_dir=tmp_path)
    ledger.write_entry(_entry(iteration=2), ledger_dir=tmp_path)
    history = ledger.read_history(tmp_path)
    assert [e.iteration for e in history] == [1, 2, 3]


def test_read_history_on_a_missing_directory_is_empty(tmp_path):
    assert ledger.read_history(tmp_path / "does_not_exist") == []


# GIT COMMIT: no subprocess, best-effort.


def test_read_git_commit_returns_none_outside_any_git_repo(tmp_path):
    assert ledger.read_git_commit(tmp_path) is None


def test_read_git_commit_finds_this_repos_head():
    # This repo IS a git repo (see conftest/CI); a real sha should come back,
    # not None -- proves the worktree .git-file path (see the module
    # docstring) actually resolves, not just the plain-.git-directory path.
    sha = ledger.read_git_commit(Path(__file__).resolve())
    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)
