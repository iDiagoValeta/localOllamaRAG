"""Regressions for context assembly (issue #397).

Four of the five defects in the issue are covered here with hand-built
fragments: a lone numeric fragment is data, not a page number (1); a
truncated fragment keeps its marker when optimization is off (3); a closing
colon or an undecorated trailing digit does not prove the excerpt is complete
(4); and the ``Solución: La <n>`` cleanup only fires on a line of its own (5).

Point (2) -- which chunks neighbor expansion requests from the previous page
-- is not changed here: ``tests/characterization/test_chunking.py`` pins the
current ids and the characterization contract forbids editing it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.application.answer import Answer  # noqa: E402
from monkeygrab.application.context_assembly import (  # noqa: E402
    build_context_for_model,
    optimize_context_text,
)
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402
from monkeygrab.domain.generation_chunk import GenerationChunk  # noqa: E402

MARKER = "[excerpt ends mid-sentence]"


def _fragment(doc, chunk=0):
    return Fragment(
        doc=doc,
        metadata=ChunkMetadata(source="paper.pdf", page=0, chunk=chunk),
    )


class _FakeChatModel:
    """Minimal ChatModel double that records prompts and returns canned text."""

    def __init__(self, response="## Facts relevant to the question\n- A grounded fact."):
        self.response = response
        self.generate_calls = []
        self.stream_calls = []

    def generate(self, prompt, *, system=None, images=(), response_format=None):
        self.generate_calls.append(prompt)
        return self.response

    def stream(self, prompt, *, system=None):
        self.stream_calls.append(prompt)
        yield GenerationChunk(text="answer", done=True)


@pytest.mark.parametrize("texto", ["42", "7", "123", "  42  "])
def test_lone_numeric_fragment_is_preserved(texto):
    """A fragment that is only a number carries table/figure data; without a
    block break there is no evidence it is a page number (issue #397 point 1)."""
    assert optimize_context_text(texto) == texto.strip()


def test_trailing_page_number_after_a_block_break_is_still_removed():
    """The safe page-number case keeps working: isolated by a blank line at
    the end of the fragment."""
    assert optimize_context_text("Body text ends here.\n\n42") == "Body text ends here."


def test_solucion_answer_key_is_removed_only_from_its_own_line():
    """The cleanup is anchored to a full line, so the same words inside a
    sentence survive (issue #397 point 5)."""
    assert optimize_context_text("Solución: La 3\nBody after the answer key.") == (
        "Body after the answer key."
    )

    sentence = "La respuesta es Solución: La 3 del ejercicio."
    assert optimize_context_text(sentence) == sentence


@pytest.mark.parametrize(
    ("doc", "marcado"),
    [
        ("The following items:", True),
        ("Total count 42", True),
        ("The constant is 3.14", False),
        ("The result is 42.", False),
        ("Ends cleanly.", False),
    ],
)
def test_incomplete_marker_needs_real_closing_punctuation(doc, marcado):
    """A colon or a bare trailing digit does not close a fragment; only a
    decimal number or actual terminal punctuation does (issue #397 point 4)."""
    contexto, _metrics = build_context_for_model([_fragment(doc)], optimize_text=False)

    assert (MARKER in contexto) is marcado


def test_recomp_marks_truncated_fragment_with_optimization_disabled():
    """The incomplete marker is independent of the optimization flag (issue
    #397 point 3)."""
    rag_model = _FakeChatModel()
    recomp_model = _FakeChatModel()
    config = AppConfig().with_overrides(
        **{
            "flags.usar_recomp_synthesis": True,
            "flags.usar_optimizacion_contexto": False,
            "flags.expandir_contexto": False,
        }
    )

    Answer(None, rag_model, config, recomp_chat_model=recomp_model).run(
        "question?", [_fragment("The list continues with a cut")]
    )

    assert MARKER in recomp_model.generate_calls[0]
