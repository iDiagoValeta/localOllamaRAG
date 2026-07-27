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
# grade_answer -- a bare integer must not match inside a longer decimal
# ─────────────────────────────────────────────


def test_integer_literal_does_not_match_the_integer_part_of_a_longer_decimal():
    """Plain ``\\b`` treats a decimal point as a valid boundary (it's a
    non-word character), so a naive ``\\b...\\b`` match let an integer
    literal match the whole-number part of an unrelated decimal: "28"
    inside "28.4", "24" inside "24.5". Both verified passing (wrongly)
    before the fix, with exactly these audit examples."""
    assert not grade_answer("The score improved to 28.4 points.", {"accepted_answers": ["28"]})["pass"]
    assert not grade_answer("The result was 24.5 on the benchmark.", {"accepted_answers": ["24"]})["pass"]


def test_integer_literal_does_not_match_the_fractional_part_of_a_longer_decimal():
    """Same bug, the other direction: ``\\b`` also let an integer literal
    match the fractional digits of an unrelated decimal, e.g. "24" inside
    "0.24"."""
    assert not grade_answer("The measured value was 0.24 in the trial.", {"accepted_answers": ["24"]})["pass"]


def test_integer_literal_still_matches_when_genuinely_standalone():
    """The new guards must not overreach: a real standalone integer,
    including one immediately followed by a sentence-ending period (not a
    decimal point), must still match."""
    assert grade_answer("The result was 24 points on the benchmark.", {"accepted_answers": ["24"]})["pass"]
    assert grade_answer("The score was 28.", {"accepted_answers": ["28"]})["pass"]


def test_decimal_literal_is_unaffected_by_the_new_guards():
    """A literal that is itself a decimal keeps working exactly as before:
    it matches its own occurrence and still rejects being a fragment of an
    unrelated longer number (pre-existing ``\\b`` behavior, e.g. the
    "1512"/"512" case)."""
    assert grade_answer("The BLEU score is 28.4 points.", {"accepted_answers": ["28.4"]})["pass"]
    assert not grade_answer("The BLEU score is 128.45 points.", {"accepted_answers": ["28.4"]})["pass"]


# ─────────────────────────────────────────────
# grade_answer -- numeric literal followed by a unit suffix
# ─────────────────────────────────────────────


def test_numeric_literal_matches_when_followed_by_a_recognized_unit_suffix():
    """``\\b`` cannot see a digit/letter junction ("110" then "M" are both
    word characters), so a model answering in the paper's own notation --
    "110M parameters" for BERT_BASE -- was scored as wrong. This is the
    exact audit example (gold_cases.jsonl's bert-base-params)."""
    case = {"accepted_answers": ["110"]}
    assert grade_answer("BERT_BASE has a total of 110M parameters.", case)["pass"]


def test_numeric_literal_matches_with_percent_suffix():
    """Same bug, "%" suffix: ResNet's abstract glues it directly to the
    number ("a 28% relative improvement")."""
    case = {"accepted_answers": ["28"]}
    assert grade_answer("They obtain a 28% relative improvement on COCO.", case)["pass"]


def test_numeric_literal_does_not_match_with_an_unrecognized_suffix():
    """The suffix set is deliberately narrow. An attached word that merely
    starts with a recognized suffix letter must still fail -- otherwise the
    guard would readmit false positives like "110kg" for a literal "110"."""
    case = {"accepted_answers": ["110"]}
    assert not grade_answer("The device drew 110kg of load.", case)["pass"]
    assert not grade_answer("Room 1104 has the equipment.", case)["pass"]


# ─────────────────────────────────────────────
# grade_answer -- Spanish decimal comma
# ─────────────────────────────────────────────


def test_spanish_decimal_comma_is_recognized():
    """``_normalize`` alone only strips thousands-separator commas; it
    never turns a Spanish decimal comma into a point, so no
    Spanish-language numeric answer could pass. Four cases verified failing
    in the audit, matching the corresponding *-es gold cases."""
    assert grade_answer("El ensemble alcanzó un 3,57% de error top-5.", {"accepted_answers": ["3.57"]})["pass"]
    assert grade_answer("BERT obtiene una puntuación GLUE de 80,5.", {"accepted_answers": ["80.5"]})["pass"]
    assert grade_answer("Según Planck 2018, sigma8 vale 0,811.", {"accepted_answers": ["0.811"]})["pass"]
    assert grade_answer("Se recogió una luminosidad de 5,8 fb-1.", {"accepted_answers": ["5.8"]})["pass"]


def test_ambiguous_three_digit_comma_group_accepts_both_thousands_and_decimal_reading():
    """"1,024" is genuinely ambiguous without locale context: a
    thousands-grouped integer (1024) or a 3-decimal-digit fraction (1.024).
    Chosen rule: try both normalizations of the answer and accept a match
    against either, rather than guess a single reading. The thousands side
    of this exact ambiguous shape is already covered by
    test_thousands_separator_is_tolerated; this covers the decimal side."""
    assert grade_answer("El valor obtenido fue 1,024 en el experimento.", {"accepted_answers": ["1.024"]})["pass"]
    assert grade_answer("The hidden size is 1,024.", {"accepted_answers": ["1024"]})["pass"]


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
            # A factual case needs something to grade against, and either
            # mechanism qualifies: accepted_answers (any one literal matches) or
            # required_all (every literal must be present, for cases where
            # precision needs two separate anchors rather than one exact phrase).
            assert case.get("accepted_answers") or case.get("required_all"), (
                f"{cid}: factual case needs 'accepted_answers' or 'required_all'"
            )


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


# ─────────────────────────────────────────────
# gold_cases.jsonl -- case-level fixes for near-unfalsifiable bare-digit literals
# ─────────────────────────────────────────────
#
# A bare single/double-digit accepted literal (e.g. ["6"]) can be matched by
# a wrong answer that merely happens to mention the same digit in an
# unrelated context (a figure number, a section number...). Normalization
# cannot fix this -- it's a property of the case's accepted_answers, not of
# the matching logic. The cases below were rewritten to require the exact
# phrasing verified in the source PDF instead of the bare digit; each test
# checks both directions: the old false-positive wrong answer is now
# rejected, and a realistically-phrased correct answer is still accepted.


def _find_case(cases, case_id):
    for c in cases:
        if c["id"] == case_id:
            return c
    raise AssertionError(f"case {case_id!r} not found in gold_cases.jsonl")


def test_att_n_layers_rejects_coincidental_figure_reference():
    """Verified page 3 of attention-transformers.pdf: "The encoder is
    composed of a stack of N = 6 identical layers." """
    case = _find_case(_load_gold_cases(), "att-n-layers")
    wrong = "The encoder has 12 layers, and Figure 6 shows the architecture."
    correct = "The Transformer base model uses N = 6 identical layers in the encoder."
    assert not grade_answer(wrong, case)["pass"]
    assert grade_answer(correct, case)["pass"]


def test_resnet_coco_improvement_accepts_the_bare_number_it_asks_for():
    """Verified page 1 (abstract) of arXiv 1512.03385: "we obtain a 28%
    relative improvement on the COCO object detection dataset."

    The question instructs the model to answer with the number only, so
    demanding the full phrase contradicted it — a model that complied was
    graded wrong. The bare literal is safe because the grader refuses to match
    an integer inside a longer number or inside a decimal.
    """
    case = _find_case(_load_gold_cases(), "resnet-coco-improvement")
    assert grade_answer("28", case)["pass"]
    assert grade_answer("A 28% relative improvement on COCO.", case)["pass"]
    assert not grade_answer("The improvement was 28.4 points.", case)["pass"]
    assert not grade_answer("Baseline scored 128 overall.", case)["pass"]


def test_multiplication_notation_is_one_answer_in_every_spelling():
    """A patch size is the same fact whether it arrives as LaTeX, as the
    Unicode sign, or as prose, and whether or not it carries spaces.

    This was a real false negative: the model answered "$16 \\times 16$" and was
    graded wrong against a literal spelling "16x16".
    """
    case = _find_case(_load_gold_cases(), "vit-patch-size-es")
    for spelling in ("$16 \\times 16$", "16×16", "16 x 16", "16x16"):
        assert grade_answer(f"Usa parches de {spelling} pixeles.", case)["pass"], spelling


def test_required_all_needs_every_anchor_present():
    """gw-snr is graded by two anchors instead of one exact sentence.

    The paper writes "matched-filter signal-to-noise ratio of 24"; a model
    writing "ratio was 24" is equally correct, and a bare "24" would match any
    stray digit. Requiring both anchors rejects neither.
    """
    case = _find_case(_load_gold_cases(), "gw-snr")
    assert grade_answer("The signal-to-noise ratio was 24.", case)["pass"]
    assert grade_answer("Observed with a signal-to-noise ratio of 24.", case)["pass"]
    assert not grade_answer("The signal-to-noise ratio was high.", case)["pass"]
    assert not grade_answer("Figure 24 shows the strain data.", case)["pass"]


def test_bert_large_layers_rejects_a_coincidental_number():
    """Verified page 3 of arXiv 1810.04805: "BERTLARGE (L=24, H=1024,
    A=16, Total Parameters=340M)." """
    case = _find_case(_load_gold_cases(), "bert-large-layers")
    wrong = "Section 24 of the appendix covers ablations."
    correct = "BERT_LARGE uses L=24 Transformer layers."
    assert not grade_answer(wrong, case)["pass"]
    assert grade_answer(correct, case)["pass"]


def test_gw_snr_rejects_a_coincidental_number():
    """Verified page 1 (abstract) of gravitational-waves.pdf: "observed
    with a matched-filter signal-to-noise ratio of 24." """
    case = _find_case(_load_gold_cases(), "gw-snr")
    wrong = "The event was the 24th candidate reviewed by the collaboration."
    correct = "The signal was observed with a matched-filter signal-to-noise ratio of 24."
    assert not grade_answer(wrong, case)["pass"]
    assert grade_answer(correct, case)["pass"]
