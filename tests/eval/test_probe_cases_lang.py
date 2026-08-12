"""Schema check for tests/eval/probe_cases_lang.jsonl -- pure, no network.

probe_cases_lang.jsonl is the language-axis diagnostic batch from
docs/design/2026-07-28-loop-automejorable.md section 3 ("Sonda previa"): a
proposal pending human audit, deliberately kept separate from
gold_cases.jsonl (see that file and tests/eval/README.md's "Sonda de
idioma" section for why). This test only pins that the file parses and has
the fields run_probe_lang.py and grade.py need -- it does not run the
pipeline and does not touch gold_cases.jsonl.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_probe_lang import _SOURCE_PDFS, PROBE_CASES_FILE  # noqa: E402

_VALID_CASE_TYPES = {"factual_number", "factual_concept", "figure_retrieval", "table_retrieval"}
_VALID_LANGS = {"es", "ca"}
_VALID_KINDS = {"text", "table", "image"}
_RETRIEVAL_CASE_TYPES = {"figure_retrieval", "table_retrieval"}


def _load_probe_cases():
    cases = []
    for line in PROBE_CASES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def test_probe_file_parses_and_is_sized_like_a_diagnostic_batch():
    """Section 3: "Seis documentos, dos o tres casos cada uno" -- roughly 12-18 cases."""
    cases = _load_probe_cases()
    assert 12 <= len(cases) <= 18, f"expected 12-18 diagnostic cases, got {len(cases)}"


def test_probe_case_ids_are_unique():
    cases = _load_probe_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case id in probe_cases_lang.jsonl"


def test_probe_covers_exactly_six_documents_across_both_languages():
    cases = _load_probe_cases()
    papers = {c["paper"] for c in cases}
    assert len(papers) == 6, f"expected 6 documents, got {papers}"
    langs = {c["lang"] for c in cases}
    assert langs == _VALID_LANGS, f"expected both es and ca, got {langs}"


def test_every_paper_slug_resolves_to_a_shipped_pdf():
    """Cross-check against run_probe_lang._SOURCE_PDFS, which is what actually
    stages each PDF for indexing -- a case referencing an unmapped slug would
    fail at run time instead of at review time."""
    cases = _load_probe_cases()
    for case in cases:
        slug = case["paper"]
        assert slug in _SOURCE_PDFS, f"{case['id']}: paper {slug!r} has no entry in _SOURCE_PDFS"
        assert _SOURCE_PDFS[slug].exists(), f"{case['id']}: {_SOURCE_PDFS[slug]} does not exist"


def test_probe_cases_have_required_fields_for_their_type():
    cases = _load_probe_cases()
    for case in cases:
        cid = case.get("id", "<missing id>")
        assert case.get("paper"), f"{cid}: missing 'paper'"
        assert case.get("case_type") in _VALID_CASE_TYPES, f"{cid}: bad 'case_type'"
        assert case.get("lang") in _VALID_LANGS, f"{cid}: bad 'lang'"
        assert case.get("question"), f"{cid}: missing 'question'"
        assert case.get("verified_pages"), f"{cid}: missing 'verified_pages'"
        assert all(isinstance(p, int) for p in case["verified_pages"]), f"{cid}: 'verified_pages' must be ints"

        if case["case_type"] in _RETRIEVAL_CASE_TYPES:
            kinds = case.get("expect_kind_any")
            assert kinds, f"{cid}: retrieval case missing 'expect_kind_any'"
            assert set(kinds) <= _VALID_KINDS, f"{cid}: bad kind in {kinds}"
        else:
            assert case.get("accepted_answers") or case.get("required_all"), (
                f"{cid}: factual case needs 'accepted_answers' or 'required_all'"
            )


def test_probe_is_mostly_factual_with_a_couple_of_retrieval_cases():
    """Guards the shape the task asked for: mostly factual_number/concept,
    a couple of figure_retrieval/table_retrieval -- not a retrieval-heavy
    batch that would tell us nothing about answer quality."""
    cases = _load_probe_cases()
    retrieval = [c for c in cases if c["case_type"] in _RETRIEVAL_CASE_TYPES]
    factual = [c for c in cases if c["case_type"] not in _RETRIEVAL_CASE_TYPES]
    assert len(factual) > len(retrieval), "expected factual cases to dominate the batch"
    assert 1 <= len(retrieval) <= 6, f"expected a handful of retrieval-only cases, got {len(retrieval)}"
