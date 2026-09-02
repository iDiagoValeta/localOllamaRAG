"""Study.quiz: what a question has to carry before it is worth asking.

Issue #140. The summary parse asks "can a reader use this". A quiz needs a
stricter question, because its failure mode is different in kind: a summary
section that is thin is merely thin, while a question whose key points at the
wrong option *teaches the reader something false*, and does it with the
authority of an answer key. The reader has no way to tell it apart from a
correct one -- that is the whole reason they are taking the quiz.

So the line here is drawn at "is this question safe to grade", and everything
below is about where exactly that falls.

Against a doubled ``ChatModel``: no GPU, no Ollama, no network, so this runs
in the fast gate's dependency-free job like the rest of ``application/``.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from monkeygrab.application.study import (  # noqa: E402
    MalformedQuizError,
    Study,
    quiz_to_dict,
)
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402


class _FakeChatModel:
    """Returns a canned reply and records what it was asked."""

    def __init__(self, reply="[]"):
        self.reply = reply
        self.calls = []

    def generate(self, prompt, *, system=None, images=(), response_format=None):
        self.calls.append(
            {"prompt": prompt, "system": system, "response_format": response_format}
        )
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def stream(self, prompt, *, system=None):  # pragma: no cover - unused here
        raise NotImplementedError


class _ExplodingChatModel:
    def generate(self, *args, **kwargs):
        raise AssertionError("the generator must not be called")

    def stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _fragment(source="paper.pdf", page=1, text="Some retrieved text."):
    return Fragment(doc=text, metadata=ChunkMetadata(source=source, page=page))


_TWO_QUESTIONS = (
    '[{"prompt": "What does BM25 rank?", '
    '"options": ["Documents", "Images", "Users"], "correct_index": 0},'
    ' {"prompt": "What fuses the two rankings?", '
    '"options": ["RRF", "FAISS"], "correct_index": 0}]'
)


class TestTheHappyPath:
    def test_questions_keep_the_generators_order(self):
        study = Study(_FakeChatModel(_TWO_QUESTIONS))
        quiz = study.quiz([_fragment()], AppConfig.from_env())
        assert [q.prompt for q in quiz.questions] == [
            "What does BM25 rank?",
            "What fuses the two rankings?",
        ]

    def test_the_key_points_at_the_option_it_named(self):
        study = Study(_FakeChatModel(_TWO_QUESTIONS))
        quiz = study.quiz([_fragment()], AppConfig.from_env())
        assert quiz.questions[0].correct_option == "Documents"

    def test_each_question_carries_the_pages_the_evidence_came_from(self):
        study = Study(_FakeChatModel(_TWO_QUESTIONS))
        quiz = study.quiz(
            [_fragment(page=7), _fragment(page=3), _fragment(page=7)],
            AppConfig.from_env(),
        )
        assert all(q.source_pages == (3, 7) for q in quiz.questions)

    def test_a_requested_language_reaches_the_prompt(self):
        model = _FakeChatModel(_TWO_QUESTIONS)
        Study(model).quiz([_fragment()], AppConfig.from_env(), language="Castellano")
        assert "Castellano" in model.calls[0]["prompt"]

    def test_the_requested_count_reaches_the_prompt(self):
        model = _FakeChatModel(_TWO_QUESTIONS)
        Study(model).quiz([_fragment()], AppConfig.from_env(), question_count=3)
        assert "3" in model.calls[0]["prompt"]

    def test_the_format_instruction_goes_in_the_system_prompt(self):
        # Same split as summarize: format is a standing instruction about the
        # reply, not part of the request, so it does not compete with the
        # material for the model's attention in the user turn.
        model = _FakeChatModel(_TWO_QUESTIONS)
        Study(model).quiz([_fragment()], AppConfig.from_env())
        assert "JSON" in (model.calls[0]["system"] or "")

    def test_a_fenced_reply_is_unwrapped(self):
        study = Study(_FakeChatModel(f"```json\n{_TWO_QUESTIONS}\n```"))
        quiz = study.quiz([_fragment()], AppConfig.from_env())
        assert len(quiz.questions) == 2


class TestAnAnswerKeyItRefusesToTrust:
    """The cases that separate this parse from the summary's.

    Each of these would parse fine as data. They are dropped because grading a
    reader against them would be wrong, not because the JSON is bad.
    """

    def test_an_out_of_range_key_drops_the_question_rather_than_repairing_it(self):
        # Falling back to 0 would produce a question that looks answerable and
        # grades the reader against an option the model never chose. There is
        # no evidence for which option was meant, so there is nothing to fix.
        study = Study(
            _FakeChatModel(
                '[{"prompt": "Bad key", "options": ["A", "B"], "correct_index": 5},'
                ' {"prompt": "Good one", "options": ["A", "B"], "correct_index": 1}]'
            )
        )
        quiz = study.quiz([_fragment()], AppConfig.from_env())
        assert [q.prompt for q in quiz.questions] == ["Good one"]

    def test_a_negative_key_is_not_read_as_a_python_index(self):
        # -1 is a legal Python index and would silently select the last
        # option. From the model it means nothing of the sort.
        study = Study(
            _FakeChatModel('[{"prompt": "Q", "options": ["A", "B"], "correct_index": -1}]')
        )
        with pytest.raises(MalformedQuizError):
            study.quiz([_fragment()], AppConfig.from_env())

    def test_a_non_integer_key_drops_the_question(self):
        study = Study(
            _FakeChatModel('[{"prompt": "Q", "options": ["A", "B"], "correct_index": "A"}]')
        )
        with pytest.raises(MalformedQuizError):
            study.quiz([_fragment()], AppConfig.from_env())

    def test_a_single_option_is_not_a_choice(self):
        study = Study(_FakeChatModel('[{"prompt": "Q", "options": ["Only"], "correct_index": 0}]'))
        with pytest.raises(MalformedQuizError):
            study.quiz([_fragment()], AppConfig.from_env())

    def test_duplicate_options_drop_the_question(self):
        # Two identical choices make the key ambiguous: a reader picking the
        # other "A" answered correctly and would be graded wrong.
        study = Study(
            _FakeChatModel('[{"prompt": "Q", "options": ["A", "A"], "correct_index": 0}]')
        )
        with pytest.raises(MalformedQuizError):
            study.quiz([_fragment()], AppConfig.from_env())

    def test_a_question_with_no_prompt_drops(self):
        study = Study(_FakeChatModel('[{"prompt": "", "options": ["A", "B"], "correct_index": 0}]'))
        with pytest.raises(MalformedQuizError):
            study.quiz([_fragment()], AppConfig.from_env())


class TestWhatItRefusesToGuess:
    def test_malformed_json_raises_rather_than_returning_part_of_a_quiz(self):
        study = Study(_FakeChatModel('[{"prompt": "Cut off", "options": ["A"'))
        with pytest.raises(MalformedQuizError):
            study.quiz([_fragment()], AppConfig.from_env())

    def test_a_json_object_instead_of_an_array_raises(self):
        study = Study(_FakeChatModel('{"prompt": "Just one", "options": ["A", "B"]}'))
        with pytest.raises(MalformedQuizError):
            study.quiz([_fragment()], AppConfig.from_env())

    def test_a_generator_failure_propagates_untouched(self):
        study = Study(_FakeChatModel(RuntimeError("ollama is down")))
        with pytest.raises(RuntimeError, match="ollama is down"):
            study.quiz([_fragment()], AppConfig.from_env())


class TestBounds:
    def test_more_questions_than_asked_for_are_truncated(self):
        reply = "[" + ",".join(
            f'{{"prompt": "Q{i}", "options": ["A", "B"], "correct_index": 0}}' for i in range(40)
        ) + "]"
        study = Study(_FakeChatModel(reply))
        quiz = study.quiz([_fragment()], AppConfig.from_env(), question_count=5)
        assert len(quiz.questions) == 5

    def test_fewer_questions_than_asked_for_is_not_an_error(self):
        # The material may not support five questions. Padding would mean
        # inventing them, which is the one thing the prompt forbids.
        study = Study(_FakeChatModel(_TWO_QUESTIONS))
        quiz = study.quiz([_fragment()], AppConfig.from_env(), question_count=5)
        assert len(quiz.questions) == 2

    def test_a_nonsensical_count_is_rejected_before_the_model_is_called(self):
        study = Study(_ExplodingChatModel())
        with pytest.raises(ValueError):
            study.quiz([_fragment()], AppConfig.from_env(), question_count=0)


class TestSourceAttribution:
    def test_one_document_is_named(self):
        study = Study(_FakeChatModel(_TWO_QUESTIONS))
        quiz = study.quiz(
            [_fragment(source="planck.pdf"), _fragment(source="planck.pdf", page=2)],
            AppConfig.from_env(),
        )
        assert quiz.source_document == "planck.pdf"

    def test_several_documents_name_none_rather_than_the_first(self):
        study = Study(_FakeChatModel(_TWO_QUESTIONS))
        quiz = study.quiz(
            [_fragment(source="a.pdf"), _fragment(source="b.pdf")], AppConfig.from_env()
        )
        assert quiz.source_document == ""


class TestEmptyInput:
    def test_no_fragments_gives_an_empty_quiz_without_calling_the_model(self):
        quiz = Study(_ExplodingChatModel()).quiz([], AppConfig.from_env())
        assert quiz.is_empty
        assert quiz.source_document == ""


class TestTheInterfaceView:
    def test_the_dict_view_is_plain_json(self):
        import json

        study = Study(_FakeChatModel(_TWO_QUESTIONS))
        quiz = study.quiz([_fragment(page=4)], AppConfig.from_env())
        payload = quiz_to_dict(quiz)
        assert json.loads(json.dumps(payload)) == payload
        assert payload["questions"][0]["options"] == ["Documents", "Images", "Users"]
        assert payload["questions"][0]["correct_index"] == 0
        assert payload["questions"][0]["source_pages"] == [4]


class TestTheSchemaIsAskedForInTheChannelThatEnforcesIt:
    """Issue #153. A prompt can only ask; the schema constrains decoding.

    Measured 2026-09-02 on `higgs-boson.pdf`, whose quizzes failed every time:
    unconstrained 2/5, `format:"json"` 0/5 (valid JSON every time, and a dict
    rather than the array every time -- plain JSON mode fixes syntax, not
    shape), schema 5/5.
    """

    def test_the_quiz_asks_for_its_schema(self):
        model = _FakeChatModel(_TWO_QUESTIONS)
        Study(model).quiz([_fragment()], AppConfig.from_env())
        schema = model.calls[0]["response_format"]
        assert schema["type"] == "array"
        assert set(schema["items"]["required"]) == {"prompt", "options", "correct_index"}

    def test_the_schema_bounds_the_key_it_asks_for(self):
        # 3-4 options and a key in 0..3: the model cannot emit 7 for a
        # four-option question. The parse still checks the key against the
        # options actually returned, because a schema cannot know how many
        # there were.
        model = _FakeChatModel(_TWO_QUESTIONS)
        Study(model).quiz([_fragment()], AppConfig.from_env())
        item = model.calls[0]["response_format"]["items"]["properties"]
        assert item["options"]["minItems"] == 3
        assert item["correct_index"]["maximum"] == 3

    def test_a_schema_does_not_replace_the_parse(self):
        # Every field the schema demands, and a key pointing at nothing. A
        # schema constrains structure, never truth.
        model = _FakeChatModel(
            '[{"prompt": "Q", "options": ["a", "b", "c"], "correct_index": 9}]'
        )
        with pytest.raises(MalformedQuizError):
            Study(model).quiz([_fragment()], AppConfig.from_env())
