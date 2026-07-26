"""Tests for tests/eval/grade.py -- deterministic, no network, no LLM.

Covers the two graders directly (grade_answer / grade_retrieval), the
normalization decisions documented in grade.py's module docstring (word
boundaries, thousands separators, LaTeX/Markdown cleanup), and a schema
sanity check over gold_cases.jsonl so a malformed case is caught here
rather than at eval-run time.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grade import grade_answer, grade_retrieval  # noqa: E402

GOLD_FILE = ROOT / "gold_cases.jsonl"

_VALID_CASE_TYPES = {"factual_number", "factual_concept", "figure_retrieval", "table_retrieval"}
_VALID_LANGS = {"en", "es"}
_VALID_KINDS = {"text", "table", "image"}
_RETRIEVAL_CASE_TYPES = {"figure_retrieval", "table_retrieval"}


# ─────────────────────────────────────────────
# grade_answer -- positive / negative
# ─────────────────────────────────────────────


def test_exact_number_matches():
    case = {"accepted_answers": ["28.4"]}
    result = grade_answer("The BLEU score is 28.4 points.", case)
    assert result["pass"]


def test_wrong_number_does_not_match():
    case = {"accepted_answers": ["28.4"]}
    result = grade_answer("The BLEU score is 24.6 points.", case)
    assert not result["pass"]


def test_any_of_multiple_accepted_answers_is_enough():
    case = {"accepted_answers": ["41.8", "41.0"]}
    assert grade_answer("The model reaches 41.0 BLEU.", case)["pass"]
    assert grade_answer("The model reaches 41.8 BLEU.", case)["pass"]
    assert not grade_answer("The model reaches 39.9 BLEU.", case)["pass"]


def test_matching_is_case_insensitive():
    case = {"accepted_answers": ["Large Hadron Collider"]}
    assert grade_answer("It was found at the large hadron collider.", case)["pass"]


def test_missing_accepted_answers_fails_closed():
    result = grade_answer("anything", {"accepted_answers": []})
    assert not result["pass"]
    result = grade_answer("anything", {})
    assert not result["pass"]


# ─────────────────────────────────────────────
# grade_answer -- the boundary bug this module fixes
# ─────────────────────────────────────────────


def test_number_does_not_match_inside_a_longer_number():
    """The bug this grader was written to fix: a naive substring check
    (as used by the explore/mineru-faiss-multimodal spike's grade.py) makes
    "512" match inside "1512" -- an unrelated, larger number that merely
    contains the same digits. Word-boundary matching must reject this."""
    case = {"accepted_answers": ["512"]}
    assert not grade_answer("arXiv:1512.03385 describes the model.", case)["pass"]
    assert not grade_answer("See page 1512 of the appendix.", case)["pass"]


def test_number_matches_at_a_real_word_boundary():
    """The flip side of the above: "512" must still match when it is
    genuinely its own token, not embedded in a longer number."""
    case = {"accepted_answers": ["512"]}
    assert grade_answer("d_model = 512 for the base model.", case)["pass"]
    assert grade_answer("See page 512.", case)["pass"]
    assert grade_answer("The answer is 512.", case)["pass"]


def test_short_word_does_not_match_inside_a_longer_word():
    """Same boundary logic guards word literals, not just numbers: "no"
    must not match inside "know" or "normal"."""
    case = {"accepted_answers": ["no"]}
    assert not grade_answer("As is well known, the model converges.", case)["pass"]
    assert not grade_answer("This is normal behavior.", case)["pass"]
    assert grade_answer("No, DPO does not require sampling.", case)["pass"]


# ─────────────────────────────────────────────
# grade_answer -- normalization: thousands separators, LaTeX, Markdown
# ─────────────────────────────────────────────


def test_thousands_separator_is_tolerated():
    case = {"accepted_answers": ["1024"]}
    assert grade_answer("The hidden size is 1,024.", case)["pass"]


def test_thousands_separator_stripping_requires_a_full_3digit_group():
    """A comma only counts as a thousands separator when exactly a 3-digit
    group follows it. Without that guard, stripping every digit-comma-digit
    would turn the unrelated "12,3" into "123" and falsely match."""
    case = {"accepted_answers": ["123"]}
    assert not grade_answer("Ran for 12,3 epochs.", case)["pass"]


def test_small_enumerations_are_not_treated_as_thousands_groups():
    """"1,2,3" is a list of single digits, not a thousands-separated
    number -- stripping its commas must not glue "2" and "3" into "23"."""
    case = {"accepted_answers": ["23"]}
    assert not grade_answer("We tried settings 1,2,3 and 4.", case)["pass"]


def test_latex_braces_and_delimiters_are_stripped():
    case = {"accepted_answers": ["10^-4"]}
    assert grade_answer("The learning rate was $10^{-4}$.", case)["pass"]


def test_markdown_emphasis_is_stripped():
    case = {"accepted_answers": ["rl-free"]}
    assert grade_answer("DPO is a simple **RL-free** algorithm.", case)["pass"]


# ─────────────────────────────────────────────
# grade_retrieval
# ─────────────────────────────────────────────


def test_retrieval_passes_when_expected_kind_present():
    case = {"expect_kind_any": ["image"]}
    assert grade_retrieval(["text", "image"], case)["pass"]


def test_retrieval_fails_when_expected_kind_absent():
    case = {"expect_kind_any": ["image"]}
    assert not grade_retrieval(["text", "text"], case)["pass"]


def test_retrieval_any_of_multiple_expected_kinds_is_enough():
    case = {"expect_kind_any": ["table", "image"]}
    assert grade_retrieval(["text", "table"], case)["pass"]


def test_retrieval_missing_expectation_fails_closed():
    assert not grade_retrieval(["text"], {"expect_kind_any": []})["pass"]
    assert not grade_retrieval(["text"], {})["pass"]


# ─────────────────────────────────────────────
# gold_cases.jsonl -- schema sanity (parses, every case is well-formed)
# ─────────────────────────────────────────────


def _load_gold_cases():
    cases = []
    for line in GOLD_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def test_gold_file_parses_and_has_a_minimum_number_of_cases():
    cases = _load_gold_cases()
    assert len(cases) >= 30, f"expected at least 30 gold cases, got {len(cases)}"


def test_gold_case_ids_are_unique():
    cases = _load_gold_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case id in gold_cases.jsonl"


def test_gold_cases_have_required_fields_for_their_type():
    cases = _load_gold_cases()
    for case in cases:
        cid = case.get("id", "<missing id>")
        assert case.get("paper"), f"{cid}: missing 'paper'"
        assert case.get("source") in {"corpus", "arxiv"}, f"{cid}: bad 'source'"
        if case["source"] == "arxiv":
            assert case.get("arxiv_id"), f"{cid}: source=arxiv requires 'arxiv_id'"
        assert case.get("case_type") in _VALID_CASE_TYPES, f"{cid}: bad 'case_type'"
        assert case.get("lang") in _VALID_LANGS, f"{cid}: bad 'lang'"
        assert case.get("question"), f"{cid}: missing 'question'"
        assert case.get("verified_pages"), f"{cid}: missing 'verified_pages'"

        if case["case_type"] in _RETRIEVAL_CASE_TYPES:
            kinds = case.get("expect_kind_any")
            assert kinds, f"{cid}: retrieval case missing 'expect_kind_any'"
            assert set(kinds) <= _VALID_KINDS, f"{cid}: bad kind in {kinds}"
        else:
            assert case.get("accepted_answers"), f"{cid}: factual case missing 'accepted_answers'"


def test_gold_cases_cover_both_languages_and_all_case_types():
    """Guards the corpus shape the design doc asks for (section 7.1):
    numeric facts, concepts, figure retrieval, table retrieval, and
    Spanish-language queries over English documents -- not just English
    factual questions."""
    cases = _load_gold_cases()
    case_types_seen = {c["case_type"] for c in cases}
    langs_seen = {c["lang"] for c in cases}
    assert case_types_seen == _VALID_CASE_TYPES, f"missing case types: {_VALID_CASE_TYPES - case_types_seen}"
    assert "es" in langs_seen, "no Spanish-language case found"


def test_gold_cases_include_papers_outside_the_dev_set():
    """Section 7.1: part of the corpus must be papers the retrieval/
    reranking heuristics were never tuned against, to catch overfitting to
    the development documents rather than genuine retrieval quality."""
    cases = _load_gold_cases()
    arxiv_papers = {c["paper"] for c in cases if c["source"] == "arxiv"}
    assert len(arxiv_papers) >= 3, f"expected >=3 blind-set papers, got {arxiv_papers}"
