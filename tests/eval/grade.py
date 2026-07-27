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

A later audit of gold_cases.jsonl found three narrower pitfalls in that same
word-boundary idea, each fixed with its own guard and covered by its own
tests: a bare number glued to a unit suffix with no separating space (e.g.
``"110M"``) not matching (``_NUMERIC_UNIT_SUFFIXES`` in ``_contains_token``);
a bare number matching as a fragment of an unrelated longer decimal (e.g.
``"28"`` inside ``"28.4"`` or inside ``"0.28"``) via the ``(?<!\\.)`` /
``(?!\\.\\d)`` guards in ``_contains_token``; and a Spanish-notation decimal
comma (``"3,57"``) never being recognized, fixed by trying a second
normalization (``_normalize_decimal_comma``) alongside the default one.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence

# NORMALIZATION

# Markdown emphasis a model's answer or the retrieved context may wrap
# numbers/words in (**bold**, _italic_, `code`) -- stripped so it never
# affects matching.
_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]+")

# LaTeX sizing commands (\left(, \right]) carry no literal content.
_LATEX_SIZING_RE = re.compile(r"\\(left|right)")

# Digit, optional spaces, "x", optional spaces, digit -- the spacing around a
# multiplication sign is typography, not content: "16 x 16" is "16x16".
_MULTIPLICATION_SPACING_RE = re.compile(r"(\d)\s*[xX]\s*(\d)")

# Thousands separators: "1,024" -> "1024", "12,345,678" -> "12345678".
# Only a comma sitting between digits with exactly a 3-digit group on its
# right (immediately followed by a non-digit or end of string) is treated
# as a separator, so an enumeration like "1, 2, 3" is left alone -- the
# groups there are single digits, not thousands.
_THOUSANDS_SEPARATOR_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

# Spanish decimal comma: "3,57" -> "3.57". Unlike _THOUSANDS_SEPARATOR_RE,
# this fires on a comma between digits regardless of how many digits follow
# -- used by _normalize_decimal_comma, a second candidate normalization of
# the answer tried alongside _normalize (see that function's docstring for
# why grading needs both instead of picking one interpretation).
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

_WHITESPACE_RE = re.compile(r"\s+")


def _strip_markup(text: str) -> str:
    """Lowercase and strip the Markdown/LaTeX markup shared by both normalizations.

    Args:
        text: Raw text (an answer, or one accepted literal).

    Returns:
        Lowercased text with Markdown emphasis and LaTeX math delimiters,
        grouping braces and sizing commands removed. Digit-grouping
        (thousands separator vs. decimal comma) is left to the caller --
        see ``_normalize`` and ``_normalize_decimal_comma``.
    """
    s = text or ""
    s = _MARKDOWN_EMPHASIS_RE.sub("", s)
    s = s.replace("$", "")  # LaTeX math delimiters
    s = _LATEX_SIZING_RE.sub("", s)
    s = s.replace("{", "").replace("}", "")  # LaTeX grouping braces, e.g. 10^{-4}
    # Multiplication reaches us three ways for the same value: the Unicode sign
    # from a PDF, the LaTeX command from a model writing math, and a plain "x"
    # from a model writing prose. A patch size of 16x16 is the same answer in
    # all three, so they collapse to one form -- including the spacing, since
    # "16 x 16" and "16x16" differ only in typography.
    s = s.replace("\\times", "x").replace("×", "x")
    s = _MULTIPLICATION_SPACING_RE.sub(r"\1x\2", s)
    return s.lower()


def _normalize(text: str) -> str:
    """Canonicalize text for comparison, reading a comma as a thousands separator.

    Strips Markdown/LaTeX markup, collapses thousands separators ("1,024"
    -> "1024"), and collapses whitespace. Applied identically to the
    haystack (generated answer) and to every accepted literal, so both
    sides of a comparison go through the same transform.

    Args:
        text: Raw text (an answer, or one accepted literal).

    Returns:
        Normalized text, safe to compare or search with word-boundary
        matching.
    """
    s = _strip_markup(text)
    s = _THOUSANDS_SEPARATOR_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _normalize_decimal_comma(text: str) -> str:
    """Canonicalize text for comparison, reading a comma as a Spanish decimal point.

    A Spanish-language answer may write a decimal as "3,57" instead of
    "3.57". A comma between digits is inherently ambiguous without locale
    context -- "1,024" could be "mil veinticuatro" (a thousands-grouped
    integer, ``_normalize``'s reading) or a number with three decimal
    digits. Rather than guess a single interpretation from the string
    alone, ``grade_answer`` runs the answer through both normalizations and
    accepts a match against either; this function supplies the
    decimal-comma reading. Accepted literals never need this: they are
    always written in "." notation in gold_cases.jsonl.

    Args:
        text: Raw text (a generated answer).

    Returns:
        Normalized text with every digit-comma-digit run turned into
        digit-dot-digit, safe to compare or search with word-boundary
        matching.
    """
    s = _strip_markup(text)
    s = _DECIMAL_COMMA_RE.sub(".", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


# TOKEN MATCHING

# A bare numeric literal ("110", "28.4") -- eligible for the two guards
# below. Word/phrase literals ("6 identical layers", "l=24", "no") go
# through the plain \b...\b path unchanged.
_NUMERIC_LITERAL_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Unit suffixes a bare number may be glued to with no separating space in a
# paper's own notation ("110M parameters", "a 28% relative improvement").
# Plain \b...\b cannot see this: the suffix's leading character can be a
# word character too, so there is no boundary between digit and letter.
# Kept narrow -- every entry only ever widens acceptance, so it is included
# only where this corpus's own papers actually glue it to a number:
#   - m / b / k: order-of-magnitude abbreviations, e.g. BERT's
#     "Total Parameters=110M".
#   - "%": percentage, e.g. ResNet's "a 28% relative improvement".
#   - gev / tev: these papers currently write them with a space
#     ("126.0 GeV"), but PDF text extraction is not guaranteed to preserve
#     that, so the glued form is accepted defensively.
#   - x: multiplier notation; PDF text extraction commonly flattens these
#     papers' Unicode "x" (multiplication sign) to plain ASCII "x", e.g.
#     ResNet's "8x deeper than VGG nets".
_NUMERIC_UNIT_SUFFIXES = ("gev", "tev", "%", "m", "b", "k", "x")
_NUMERIC_SUFFIX_ALTERNATION = "|".join(_NUMERIC_UNIT_SUFFIXES)


def _contains_token(haystack: str, needle: str) -> bool:
    """Whether normalized ``needle`` occurs in normalized ``haystack`` as a whole token.

    "Whole token" means the match is not immediately preceded or followed
    by another alphanumeric/underscore character -- Python's ``\\b`` regex
    boundary. This is what stops ``"512"`` from matching inside ``"1512"``
    (no boundary between the two digits) while still matching ``"512"``
    inside ``"page 512."`` (bounded by a space and a period).

    For a bare numeric literal, ``\\b`` alone is not enough and gets two
    extra guards:

    - A trailing alternation between (a) one of ``_NUMERIC_UNIT_SUFFIXES``
      followed by "not a word character", so ``"110"`` matches inside
      ``"110M"`` even though digit and letter share no ``\\b``; or (b) the
      plain ``\\b``, additionally guarded by ``(?!\\.\\d)`` so the literal
      is not accepted as merely the integer part of an unrelated longer
      decimal (``"28"`` must not match inside ``"28.4"``). Branch (a) never
      needs that guard itself: a decimal point never starts a unit suffix.
    - A leading ``(?<!\\.)``, so the literal is also not accepted as the
      fractional part of an unrelated longer decimal (``"24"`` must not
      match inside ``"0.24"``) -- plain ``\\b`` treats the decimal point as
      a valid boundary, which is wrong when the digits on both sides of it
      are really one number.

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
    escaped = re.escape(needle_norm)
    if _NUMERIC_LITERAL_RE.match(needle_norm):
        pattern = (
            r"(?<!\.)\b"
            + escaped
            + r"(?:(?:"
            + _NUMERIC_SUFFIX_ALTERNATION
            + r")(?!\w)|\b(?!\.\d))"
        )
    else:
        pattern = r"\b" + escaped + r"\b"
    return re.search(pattern, haystack) is not None


# GRADERS


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
    required_all = case.get("required_all") or []
    if not accepted and not required_all:
        return {"pass": False, "reason": "case has no accepted_answers to grade against"}

    # Two candidate readings of the answer, differing only in how a
    # digit-comma-digit run is interpreted (thousands separator vs. Spanish
    # decimal point -- see _normalize_decimal_comma). A literal counts as
    # matched if it is found in either.
    haystack_thousands = _normalize(answer)
    haystack_decimal = _normalize_decimal_comma(answer)

    def _present(literal: str) -> bool:
        return (
            _contains_token(haystack_thousands, literal)
            or _contains_token(haystack_decimal, literal)
        )

    # required_all demands every literal be present, which buys precision
    # without dictating phrasing. Requiring one exact sentence rejected correct
    # answers over a preposition -- "ratio was 24" failed a literal spelling
    # "ratio of 24" -- while a bare "24" would match any stray digit. Asking for
    # "signal-to-noise" AND "24" separately rejects neither.
    if required_all:
        missing = [literal for literal in required_all if not _present(literal)]
        if missing:
            return {"pass": False, "reason": f"missing required {missing}"}
        return {"pass": True, "reason": f"all of {required_all} present"}

    for literal in accepted:
        if _present(literal):
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
