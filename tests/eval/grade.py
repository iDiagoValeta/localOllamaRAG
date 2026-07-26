"""grade -- deterministic scoring for gold_cases.jsonl (no LLM judge).

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- 1. NORMALIZATION      markdown/LaTeX/thousands-separator cleanup
#  +-- 2. TOKEN MATCHING     word-boundary substring match
#  +-- 3. GRADERS            grade_answer, grade_retrieval
#
# ─────────────────────────────────────────────

Two independent graders, matching the two case shapes in gold_cases.jsonl:

- ``grade_answer``: a generated answer against ``case["accepted_answers"]``
  (``factual_number`` / ``factual_concept`` cases).
- ``grade_retrieval``: the content kinds of the retrieved top-k against
  ``case["expect_kind_any"]`` (``figure_retrieval`` / ``table_retrieval``
  cases).

Both are pure functions over plain data (strings/lists) so they run without
network, GPU or even the rest of the pipeline -- see ``test_grade.py``.

Normalization rationale (the substring-match bug this module fixes):
a naive ``needle in haystack`` check, as used by the ``explore/mineru-
faiss-multimodal`` spike this module supersedes, makes ``"512"`` match
inside ``"1512"`` or inside a page number -- any accepted literal that
happens to be a substring of a longer unrelated token in the answer counts
as a match. This module instead requires the accepted literal to appear as
a whole token: bounded by a non-alphanumeric character (or the start/end of
the string) on both sides, the same idea as regex ``\\b``. See
``test_grade.py::test_number_does_not_match_inside_a_longer_number`` for the
case this was written to catch.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence

# ─────────────────────────────────────────────
# SECTION 1: NORMALIZATION
# ─────────────────────────────────────────────

# Markdown emphasis a model's answer or the retrieved context may wrap
# numbers/words in (**bold**, _italic_, `code`) -- stripped so it never
# affects matching.
_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]+")

# LaTeX sizing commands (\left(, \right]) carry no literal content.
_LATEX_SIZING_RE = re.compile(r"\\(left|right)")

# Thousands separators: "1,024" -> "1024", "12,345,678" -> "12345678".
# Only a comma sitting between digits with exactly a 3-digit group on its
# right (immediately followed by a non-digit or end of string) is treated
# as a separator, so an enumeration like "1, 2, 3" is left alone -- the
# groups there are single digits, not thousands.
_THOUSANDS_SEPARATOR_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Canonicalize text for comparison.

    Lowercases, strips Markdown emphasis and LaTeX math delimiters/braces
    and sizing commands, collapses thousands separators, and collapses
    whitespace. Applied identically to the haystack (generated answer) and
    to every accepted literal, so both sides of a comparison go through the
    same transform.

    Args:
        text: Raw text (an answer, or one accepted literal).

    Returns:
        Normalized text, safe to compare or search with word-boundary
        matching.
    """
    s = text or ""
    s = _MARKDOWN_EMPHASIS_RE.sub("", s)
    s = s.replace("$", "")  # LaTeX math delimiters
    s = _LATEX_SIZING_RE.sub("", s)
    s = s.replace("{", "").replace("}", "")  # LaTeX grouping braces, e.g. 10^{-4}
    s = s.lower()
    s = _THOUSANDS_SEPARATOR_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


# ─────────────────────────────────────────────
# SECTION 2: TOKEN MATCHING
# ─────────────────────────────────────────────


def _contains_token(haystack: str, needle: str) -> bool:
    """Whether normalized ``needle`` occurs in normalized ``haystack`` as a whole token.

    "Whole token" means the match is not immediately preceded or followed
    by another alphanumeric/underscore character -- Python's ``\\b`` regex
    boundary. This is what stops ``"512"`` from matching inside ``"1512"``
    (no boundary between the two digits) while still matching ``"512"``
    inside ``"page 512."`` (bounded by a space and a period).

    Caveat: if ``needle`` itself starts or ends with a non-alphanumeric
    character (e.g. a bare ``"%"`` or a leading ``"-"``), ``\\b`` cannot
    anchor on that end and the match may be looser than intended -- accepted
    literals in gold_cases.jsonl are written to start and end with a letter
    or digit for this reason (see the module docstring of gold_cases.jsonl's
    companion README).

    Args:
        haystack: Already-normalized text to search in.
        needle: Raw accepted literal (normalized internally).

    Returns:
        True if the token is found.
    """
    needle_norm = _normalize(needle)
    if not needle_norm:
        return False
    pattern = r"\b" + re.escape(needle_norm) + r"\b"
    return re.search(pattern, haystack) is not None


# ─────────────────────────────────────────────
# SECTION 3: GRADERS
# ─────────────────────────────────────────────


def grade_answer(answer: str, case: Dict[str, Any]) -> Dict[str, Any]:
    """Score a generated answer against a case's accepted literals.

    Args:
        answer: The pipeline's generated answer text.
        case: A parsed gold_cases.jsonl record with an ``accepted_answers``
            list (``factual_number`` / ``factual_concept`` cases).

    Returns:
        ``{"pass": bool, "reason": str}``. ``reason`` names the literal that
        matched, or explains why none did -- kept human-readable since this
        is what a CI failure log shows.
    """
    accepted = case.get("accepted_answers") or []
    if not accepted:
        return {"pass": False, "reason": "case has no accepted_answers to grade against"}

    haystack = _normalize(answer)
    for literal in accepted:
        if _contains_token(haystack, literal):
            return {"pass": True, "reason": f"matched {literal!r}"}
    return {"pass": False, "reason": f"none of {accepted} found in answer"}


def grade_retrieval(hit_kinds: Sequence[str], case: Dict[str, Any]) -> Dict[str, Any]:
    """Score retrieval by whether an expected content kind was surfaced.

    Args:
        hit_kinds: The ``kind`` (e.g. ``"text"``, ``"table"``, ``"image"``)
            of every fragment in the retrieved top-k, in any order.
        case: A parsed gold_cases.jsonl record with an ``expect_kind_any``
            list (``figure_retrieval`` / ``table_retrieval`` cases).

    Returns:
        ``{"pass": bool, "reason": str}``.
    """
    expected = case.get("expect_kind_any") or []
    if not expected:
        return {"pass": False, "reason": "case has no expect_kind_any to grade against"}

    hit_set = set(hit_kinds)
    for kind in expected:
        if kind in hit_set:
            return {"pass": True, "reason": f"kind {kind!r} present in retrieved hits"}
    return {"pass": False, "reason": f"wanted one of {expected}, got {sorted(hit_set)}"}
