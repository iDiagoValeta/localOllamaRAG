"""Tests for the eval runner's index-reuse rule."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import should_rebuild  # noqa: E402


def test_matching_fingerprint_reuses():
    assert should_rebuild("abc123", "abc123") is False


def test_different_fingerprint_rebuilds():
    assert should_rebuild("abc123", "def456") is True


def test_unknown_fingerprint_rebuilds():
    # A store from before fingerprinting existed. Its recipe cannot be
    # established, so reusing it would silently mix two pipelines in one score.
    assert should_rebuild(None, "abc123") is True
