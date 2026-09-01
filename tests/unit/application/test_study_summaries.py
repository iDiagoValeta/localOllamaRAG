"""Study.summarize: the parse boundary, and what it refuses to guess.

Issue #140. The generator returns text and a summary has structure, so the
only genuinely new failure mode here is a model that answers with almost-JSON.
These tests are mostly about that line -- what counts as usable, what raises,
and what the use case declines to infer.

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
    MalformedOutlineError,
    MalformedSummaryError,
    Study,
    summary_to_dict,
)
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402


class _FakeChatModel:
    """Returns a canned reply and records what it was asked."""

    def __init__(self, reply="[]"):
        self.reply = reply
        self.calls = []

    def generate(self, prompt, *, system=None, images=()):
        self.calls.append({"prompt": prompt, "system": system})
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
    return Fragment(
        doc=text,
        metadata=ChunkMetadata(source=source, page=page),
    )


_TWO_SECTIONS = (
    '[{"heading": "What it is", "body": "First body."},'
    ' {"heading": "How it works", "body": "Second body."}]'
)


class TestTheHappyPath:
    def test_sections_keep_the_generators_order(self):
        study = Study(_FakeChatModel(_TWO_SECTIONS))
        summary = study.summarize([_fragment()], AppConfig())
        assert [s.heading for s in summary.sections] == ["What it is", "How it works"]

    def test_each_section_carries_the_pages_the_evidence_came_from(self):
        study = Study(_FakeChatModel(_TWO_SECTIONS))
        summary = study.summarize(
            [_fragment(page=4), _fragment(page=2), _fragment(page=4)], AppConfig()
        )
        # Ascending and deduplicated: a reader opens pages, and 2 twice is not
        # more evidence than 2 once.
        assert all(s.source_pages == (2, 4) for s in summary.sections)

    def test_a_requested_language_reaches_the_prompt(self):
        model = _FakeChatModel(_TWO_SECTIONS)
        Study(model).summarize([_fragment()], AppConfig(), language="Castellano")
        assert "Castellano" in model.calls[0]["prompt"]

    def test_the_format_instruction_goes_in_the_system_prompt(self):
        """Format is an instruction to the model; the material is the prompt.
        Mixing them lets a document that happens to contain the word 'JSON'
        argue with the instruction."""
        model = _FakeChatModel(_TWO_SECTIONS)
        Study(model).summarize([_fragment()], AppConfig())
        assert "JSON array" in model.calls[0]["system"]
        assert "JSON array" not in model.calls[0]["prompt"]

    def test_a_fenced_reply_is_unwrapped(self):
        """The most common deviation from 'JSON and nothing else' across every
        model tried, and one regex against an outage."""
        study = Study(_FakeChatModel(f"```json\n{_TWO_SECTIONS}\n```"))
        assert len(study.summarize([_fragment()], AppConfig()).sections) == 2


class TestWhatItRefusesToGuess:
    def test_malformed_json_raises_rather_than_returning_part_of_a_summary(self):
        """A summary missing a section looks complete. A reader cannot tell a
        section the model chose not to write from one lost to truncation."""
        study = Study(_FakeChatModel('[{"heading": "Cut off", "body": "It en'))
        with pytest.raises(MalformedSummaryError) as exc:
            study.summarize([_fragment()], AppConfig())
        # The message carries the start of the reply: diagnosing this means
        # seeing what the model actually said.
        assert "Cut off" in str(exc.value)

    def test_a_json_object_instead_of_an_array_raises(self):
        study = Study(_FakeChatModel('{"heading": "Just one", "body": "Alone."}'))
        with pytest.raises(MalformedSummaryError) as exc:
            study.summarize([_fragment()], AppConfig())
        assert "dict" in str(exc.value)

    def test_an_array_of_empty_sections_raises(self):
        study = Study(_FakeChatModel('[{"heading": "", "body": ""}, {}]'))
        with pytest.raises(MalformedSummaryError):
            study.summarize([_fragment()], AppConfig())

    def test_a_generator_failure_propagates_untouched(self):
        """Hard-fail policy: no empty summary standing in for a dead model."""
        study = Study(_FakeChatModel(RuntimeError("ollama is down")))
        with pytest.raises(RuntimeError, match="ollama is down"):
            study.summarize([_fragment()], AppConfig())


class TestWhatItAcceptsOnPurpose:
    def test_one_unusable_section_among_usable_ones_is_dropped_not_fatal(self):
        """The line is 'can a reader use this', not 'does it match the schema'.
        Failing the whole summary over one empty element would turn a cosmetic
        difference between models into an outage."""
        study = Study(
            _FakeChatModel('[{"heading": "Real", "body": "Body."}, {}, {"body": "Also real."}]')
        )
        sections = study.summarize([_fragment()], AppConfig()).sections
        assert len(sections) == 2
        assert sections[1].heading == ""

    def test_a_section_with_only_a_body_is_kept(self):
        study = Study(_FakeChatModel('[{"body": "No heading, still readable."}]'))
        assert study.summarize([_fragment()], AppConfig()).sections[0].body


class TestSourceAttribution:
    def test_one_document_is_named(self):
        study = Study(_FakeChatModel(_TWO_SECTIONS))
        summary = study.summarize(
            [_fragment(source="planck.pdf", page=1), _fragment(source="planck.pdf", page=3)],
            AppConfig(),
        )
        assert summary.source_document == "planck.pdf"

    def test_several_documents_name_none_rather_than_the_first(self):
        """Naming one document for a summary written from several would be a
        claim the summary does not support."""
        study = Study(_FakeChatModel(_TWO_SECTIONS))
        summary = study.summarize(
            [_fragment(source="a.pdf"), _fragment(source="b.pdf")], AppConfig()
        )
        assert summary.source_document == ""


class TestEmptyInput:
    def test_no_fragments_gives_an_empty_summary_without_calling_the_model(self):
        """Nothing to summarise is not a failure, and paying a generation call
        to be told so is waste."""
        summary = Study(_ExplodingChatModel()).summarize([], AppConfig())
        assert summary.is_empty
        assert summary.sections == ()


class TestTheInterfaceView:
    def test_the_dict_view_is_plain_json(self):
        """One shape for the web app and the CLI, so they cannot drift."""
        import json

        study = Study(_FakeChatModel(_TWO_SECTIONS))
        summary = study.summarize([_fragment(page=7)], AppConfig())
        payload = summary_to_dict(summary)
        assert json.loads(json.dumps(payload)) == payload
        assert payload["sections"][0]["source_pages"] == [7]


class TestOutline:
    """The tree, and the two bounds that keep a bad reply from being unbounded."""

    _NESTED = (
        '[{"title": "Introduction"},'
        ' {"title": "Method", "children": ['
        '   {"title": "Data"}, {"title": "Model"}]}]'
    )

    def test_nesting_is_preserved(self):
        outline = Study(_FakeChatModel(self._NESTED)).outline([_fragment()], AppConfig())
        assert [n.title for n in outline.nodes] == ["Introduction", "Method"]
        assert [c.title for c in outline.nodes[1].children] == ["Data", "Model"]

    def test_depth_is_reported_from_the_tree(self):
        outline = Study(_FakeChatModel(self._NESTED)).outline([_fragment()], AppConfig())
        assert outline.nodes[0].depth == 1
        assert outline.nodes[1].depth == 2

    def test_nesting_deeper_than_the_limit_is_truncated_not_fatal(self):
        """An outline is a navigational aid: one cut off at three levels is
        still usable, unlike a summary missing a section. Truncating is the
        deliberate difference from _parse_sections."""
        deep = '[{"title": "L1", "children": [{"title": "L2", "children": ['
        deep += '{"title": "L3", "children": [{"title": "L4", "children": ['
        deep += '{"title": "L5"}]}]}]}]}]'
        outline = Study(_FakeChatModel(deep)).outline([_fragment()], AppConfig())
        assert outline.nodes[0].depth == 4

    def test_a_flood_of_siblings_is_capped(self):
        """Parses fine, produces an outline nobody can read."""
        flood = "[" + ",".join(f'{{"title": "H{i}"}}' for i in range(200)) + "]"
        outline = Study(_FakeChatModel(flood)).outline([_fragment()], AppConfig())
        assert 0 < len(outline.nodes) <= 60

    def test_an_untitled_node_is_dropped_with_its_children(self):
        """Re-parenting them would invent a structure the model did not
        describe."""
        reply = '[{"title": "Kept"}, {"children": [{"title": "Orphan"}]}]'
        outline = Study(_FakeChatModel(reply)).outline([_fragment()], AppConfig())
        titles = [n.title for n in outline.nodes]
        assert titles == ["Kept"]
        assert not any(c.title == "Orphan" for n in outline.nodes for c in n.children)

    def test_malformed_json_raises_its_own_error_type(self):
        """Distinct from MalformedSummaryError so a caller offering both can
        tell which artifact failed without parsing a message."""
        with pytest.raises(MalformedOutlineError):
            Study(_FakeChatModel("not json at all")).outline([_fragment()], AppConfig())

    def test_an_array_with_no_titled_node_raises(self):
        with pytest.raises(MalformedOutlineError):
            Study(_FakeChatModel('[{"children": []}, {}]')).outline([_fragment()], AppConfig())

    def test_empty_input_does_not_call_the_generator(self):
        outline = Study(_ExplodingChatModel()).outline([], AppConfig())
        assert outline.is_empty

    def test_the_language_request_reaches_the_prompt(self):
        model = _FakeChatModel(self._NESTED)
        Study(model).outline([_fragment()], AppConfig(), language="Valencià")
        assert "Valencià" in model.calls[0]["prompt"]

    def test_a_generator_failure_propagates(self):
        with pytest.raises(RuntimeError, match="down"):
            Study(_FakeChatModel(RuntimeError("down"))).outline([_fragment()], AppConfig())
