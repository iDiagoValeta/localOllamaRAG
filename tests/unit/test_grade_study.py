"""grade_study: what a Study artifact has to be, not what it should say.

Issue #140's gold cases. A summary, an outline and a quiz have no single
correct answer, so grading them by literal match would either accept anything
or reject correct work over phrasing. They do have structure, and structure is
checkable with no judge model — which is what keeps this gate deterministic.

The line these pin: **usable, not ideal**. An outline that finds some of a
paper's real headings has demonstrably read it; demanding a particular tree
would grade the model's editorial choices instead. The quiz is the exception
and deliberately stricter, because its key is not a matter of taste: it points
at a real option or it grades a reader against nothing, and that is the one
failure the reader cannot catch for themselves.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "tests" / "eval")):
    if path not in sys.path:
        sys.path.insert(0, path)

from tests.eval.grade import grade_study  # noqa: E402


def _section(heading="H", body="Body."):
    return {"heading": heading, "body": body, "source_pages": [1]}


def _question(prompt="Q?", options=("a", "b", "c"), key=0):
    return {"prompt": prompt, "options": list(options), "correct_index": key,
            "source_pages": [1]}


class TestSummary:
    def test_enough_sourced_sections_pass(self):
        artifact = {"sections": [_section(), _section(), _section()]}
        assert grade_study(artifact, {"case_type": "study_summary"})["pass"] is True

    def test_too_few_sections_fail(self):
        artifact = {"sections": [_section()]}
        result = grade_study(artifact, {"case_type": "study_summary", "min_sections": 3})
        assert result["pass"] is False

    def test_a_heading_with_no_body_fails(self):
        # A promise the summary does not keep.
        artifact = {"sections": [_section(), _section(), _section(body="  ")]}
        result = grade_study(artifact, {"case_type": "study_summary"})
        assert result["pass"] is False
        assert "no body" in result["reason"]

    def test_a_required_term_missing_fails(self):
        artifact = {"sections": [_section(body="It uses recurrence.")] * 3}
        case = {"case_type": "study_summary", "required_all": ["attention"]}
        assert grade_study(artifact, case)["pass"] is False

    def test_a_required_term_matches_case_insensitively(self):
        artifact = {"sections": [_section(body="It relies on Attention.")] * 3}
        case = {"case_type": "study_summary", "required_all": ["attention"]}
        assert grade_study(artifact, case)["pass"] is True


class TestOutline:
    def test_nested_titles_count_toward_the_minimum(self):
        # The tree is the artifact; counting only top-level nodes would fail an
        # outline that is deep rather than wide.
        artifact = {"nodes": [
            {"title": "A", "children": [
                {"title": "A.1", "children": [{"title": "A.1.1", "children": []}]}]},
            {"title": "B", "children": [{"title": "B.1", "children": []}]},
        ]}
        assert grade_study(artifact, {"case_type": "study_outline", "min_nodes": 5})["pass"] is True

    def test_too_few_titles_fail(self):
        artifact = {"nodes": [{"title": "A", "children": []}]}
        assert grade_study(artifact, {"case_type": "study_outline"})["pass"] is False

    def test_finding_one_expected_heading_is_enough(self):
        # Any-of on purpose: which sections a model names is editorial.
        artifact = {"nodes": [{"title": t, "children": []} for t in
                              ("Intro", "Model Architecture", "Results", "X", "Y")]}
        case = {"case_type": "study_outline", "min_nodes": 5,
                "expect_titles_any": ["Model Architecture", "Positional Encoding"]}
        assert grade_study(artifact, case)["pass"] is True

    def test_finding_none_of_them_fails(self):
        artifact = {"nodes": [{"title": t, "children": []} for t in "abcde"]}
        case = {"case_type": "study_outline", "min_nodes": 5,
                "expect_titles_any": ["Model Architecture"]}
        assert grade_study(artifact, case)["pass"] is False


class TestQuizIsStricterAndWhy:
    def test_valid_questions_pass(self):
        artifact = {"questions": [_question(), _question()]}
        assert grade_study(artifact, {"case_type": "study_quiz"})["pass"] is True

    def test_a_key_out_of_range_fails(self):
        # The failure a reader cannot catch: it looks answerable and grades
        # them against an option nobody chose.
        artifact = {"questions": [_question(key=9), _question()]}
        result = grade_study(artifact, {"case_type": "study_quiz"})
        assert result["pass"] is False
        assert "out of range" in result["reason"]

    def test_a_negative_key_fails_rather_than_indexing_from_the_end(self):
        artifact = {"questions": [_question(key=-1), _question()]}
        assert grade_study(artifact, {"case_type": "study_quiz"})["pass"] is False

    def test_a_boolean_key_fails(self):
        # True is an int in Python and would silently select option 1.
        artifact = {"questions": [_question(key=True), _question()]}
        assert grade_study(artifact, {"case_type": "study_quiz"})["pass"] is False

    def test_repeated_options_fail(self):
        # Ambiguous key: a reader picking the other identical option was right
        # and grades wrong.
        artifact = {"questions": [_question(options=("a", "a", "b")), _question()]}
        assert grade_study(artifact, {"case_type": "study_quiz"})["pass"] is False

    def test_a_single_option_is_not_a_choice(self):
        artifact = {"questions": [_question(options=("only",)), _question()]}
        assert grade_study(artifact, {"case_type": "study_quiz"})["pass"] is False

    def test_a_question_with_no_prompt_fails(self):
        artifact = {"questions": [_question(prompt="  "), _question()]}
        assert grade_study(artifact, {"case_type": "study_quiz"})["pass"] is False

    def test_too_few_questions_fail(self):
        artifact = {"questions": [_question()]}
        case = {"case_type": "study_quiz", "min_questions": 3}
        assert grade_study(artifact, case)["pass"] is False


class TestTheEdges:
    @pytest.mark.parametrize("kind", ["study_summary", "study_outline", "study_quiz"])
    def test_an_empty_artifact_never_passes(self, kind):
        assert grade_study({}, {"case_type": kind})["pass"] is False

    def test_an_unknown_case_type_says_so_rather_than_passing(self):
        result = grade_study({"sections": [_section()] * 5}, {"case_type": "study_haiku"})
        assert result["pass"] is False
        assert "does not handle" in result["reason"]
