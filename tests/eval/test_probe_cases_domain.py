"""Schema check for tests/eval/probe_cases_domain.jsonl -- pure, no network.

probe_cases_domain.jsonl is the domain-axis diagnostic batch from
docs/design/2026-07-28-loop-automejorable.md section 3 ("Sonda previa"): a
proposal pending human audit, deliberately kept separate from
gold_cases.jsonl. This test only pins that the file parses and has the
fields run_probe_domain.py and grade.py need -- it does not run the
pipeline and does not touch gold_cases.jsonl.

Mirrors tests/eval/test_probe_cases_lang.py; the axis-specific bounds
differ (three arXiv documents, English only) because the design doc's
"seis documentos, dos o tres casos cada uno" bound describes the language
axis's shipped stores, not this one.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_probe_domain import _SOURCE_PDFS, PROBE_CASES_FILE  # noqa: E402

_VALID_CASE_TYPES = {"factual_number", "factual_concept", "figure_retrieval", "table_retrieval"}
_VALID_LANGS = {"en"}
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
    """Three documents, two to four cases each -- section 3's diagnostic scale."""
    cases = _load_probe_cases()
    assert 6 <= len(cases) <= 12, f"expected 6-12 diagnostic cases, got {len(cases)}"


def test_probe_case_ids_are_unique():
    cases = _load_probe_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case id in probe_cases_domain.jsonl"


def test_probe_covers_exactly_three_documents_in_english():
    cases = _load_probe_cases()
    papers = {c["paper"] for c in cases}
    assert len(papers) == 3, f"expected 3 documents, got {papers}"
    for case in cases:
        assert case.get("lang") in _VALID_LANGS, f"{case['id']}: bad 'lang'"


def test_every_paper_slug_resolves_to_a_cached_arxiv_pdf():
    """Cross-check against run_probe_domain._SOURCE_PDFS, which is what
    actually stages each PDF for indexing. The cache is gitignored (the ids
    in the mapping are the reproducible part), so existence here means the
    developer running the probe has fetched them via fetch_papers.py."""
    cases = _load_probe_cases()
    for case in cases:
        slug = case["paper"]
        assert slug in _SOURCE_PDFS, f"{case['id']}: paper {slug!r} has no entry in _SOURCE_PDFS"
        assert _SOURCE_PDFS[slug].exists(), (
            f"{case['id']}: {_SOURCE_PDFS[slug]} missing -- fetch it with "
            "python tests/eval/fetch_papers.py --dest tests/eval/probe_cache_domain "
            "2608.18973v1 2608.18375v1 2608.17955v1"
        )


def test_probe_cases_have_required_fields_for_their_type():
    cases = _load_probe_cases()
    for case in cases:
        cid = case.get("id", "<missing id>")
        assert case.get("paper"), f"{cid}: missing 'paper'"
        assert case.get("case_type") in _VALID_CASE_TYPES, f"{cid}: bad 'case_type'"
        assert case.get("question"), f"{cid}: missing 'question'"
        assert case.get("verified_pages"), f"{cid}: missing 'verified_pages'"
        assert all(isinstance(p, int) for p in case["verified_pages"]), (
            f"{cid}: 'verified_pages' must be ints"
        )

        if case["case_type"] in _RETRIEVAL_CASE_TYPES:
            kinds = case.get("expect_kind_any")
            assert kinds, f"{cid}: retrieval case missing 'expect_kind_any'"
            assert set(kinds) <= _VALID_KINDS, f"{cid}: bad kind in {kinds}"
        else:
            assert case.get("accepted_answers") or case.get("required_all"), (
                f"{cid}: factual case needs 'accepted_answers' or 'required_all'"
            )


def test_probe_is_mostly_factual_with_a_couple_of_retrieval_cases():
    cases = _load_probe_cases()
    retrieval = [c for c in cases if c["case_type"] in _RETRIEVAL_CASE_TYPES]
    factual = [c for c in cases if c["case_type"] not in _RETRIEVAL_CASE_TYPES]
    assert len(factual) > len(retrieval), "expected factual cases to dominate the batch"
    assert 1 <= len(retrieval) <= 6, (
        f"expected a handful of retrieval-only cases, got {len(retrieval)}"
    )


def test_run_probe_domain_pins_the_run_eval_privates_it_reuses():
    import run_eval

    for name in (
        "_RETRIEVAL_ONLY_CASE_TYPES",
        "_fragment_to_dict",
        "_scoped_model_roles",
        "_release_gpu_models",
        "_generation_keep_alive",
    ):
        assert hasattr(run_eval, name), (
            f"run_eval.{name} no longer exists -- run_probe_domain.py depends on it"
        )
