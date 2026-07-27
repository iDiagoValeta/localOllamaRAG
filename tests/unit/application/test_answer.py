"""Unit tests for monkeygrab.application.answer.Answer.

Char budgeting already has a dedicated equivalence test
(``test_answer_char_budget_equivalence.py``); this file covers neighbor
expansion, RECOMP synthesis (and its fallback conditions), and the final
generation streaming, all with hand-written port fakes -- there is no single
original function to diff against for the orchestration itself (RECOMP
synthesis prompts an LLM; the original conflates it with the char-budget and
expansion steps inside ``preparar_fragmentos_para_generacion``/
``generar_respuesta``, which this migration deliberately splits across
``Retrieve``/``Answer`` per the task's use-case boundaries).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.application.answer import Answer, _expand_with_neighbors, _is_expandable  # noqa: E402
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from monkeygrab.domain.chunk_metadata import ChunkMetadata  # noqa: E402
from monkeygrab.domain.fragment import Fragment  # noqa: E402
from monkeygrab.domain.generation_chunk import GenerationChunk  # noqa: E402


def _fragment(id_, doc="body text", fmt=None, total_in_page=None, score_final=0.0):
    source, rest = id_.split("_pag", 1)
    page_str, chunk_str = rest.split("_chunk", 1)
    return Fragment(
        doc=doc,
        metadata=ChunkMetadata(
            source=source, page=int(page_str), chunk=int(chunk_str),
            format=fmt, total_chunks_in_page=total_in_page,
        ),
        score_final=score_final,
    )


class FakeVectorStore:
    def __init__(self, by_id=None, raises=False):
        self._by_id = by_id or {}
        self.raises = raises
        self.get_by_ids_calls = []

    def get_by_ids(self, ids):
        self.get_by_ids_calls.append(list(ids))
        if self.raises:
            raise RuntimeError("vector store unavailable")
        return [self._by_id[i] for i in ids if i in self._by_id]


class FakeChatModel:
    def __init__(self, tokens=("Hello", " world"), raises=False, response="response", final_chunk=None):
        self._tokens = tokens
        self._raises = raises
        self._response = response
        # Defaults to a done chunk with no metadata (Ollama sometimes reports
        # none of these on a line), same as the "no generation_stats" case.
        self._final_chunk = final_chunk if final_chunk is not None else GenerationChunk(done=True)
        self.stream_calls = []
        self.generate_calls = []

    def stream(self, prompt, *, system=None):
        self.stream_calls.append({"prompt": prompt, "system": system})
        for t in self._tokens:
            yield GenerationChunk(text=t, done=False)
        yield self._final_chunk

    def generate(self, prompt, *, system=None, images=()):
        self.generate_calls.append({"prompt": prompt, "system": system})
        if self._raises:
            raise RuntimeError("recomp model unavailable")
        return self._response


def _config(**overrides):
    cfg = AppConfig()
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    return cfg


def test_image_fragments_are_never_expandable():
    frag = _fragment("a.pdf_pag0_chunk0", fmt="image", total_in_page=3)
    assert _is_expandable(frag) is False


def test_fragment_without_total_chunks_in_page_is_not_expandable():
    frag = _fragment("a.pdf_pag0_chunk0", total_in_page=None)
    assert _is_expandable(frag) is False


def test_expand_fetches_neighbors_and_dedupes_already_present_ids():
    anchor = _fragment("a.pdf_pag0_chunk1", total_in_page=3)
    neighbor_prev = _fragment("a.pdf_pag0_chunk0")
    neighbor_next = _fragment("a.pdf_pag0_chunk2")
    store = FakeVectorStore(by_id={neighbor_prev.id: neighbor_prev, neighbor_next.id: neighbor_next})

    expanded, n_added = _expand_with_neighbors([anchor], store, n_top_for_expansion=3)

    assert n_added == 2
    assert {f.id for f in expanded} == {anchor.id, neighbor_prev.id, neighbor_next.id}


def test_expand_only_considers_the_top_n_anchors():
    anchors = [_fragment(f"a.pdf_pag0_chunk{i}", total_in_page=10) for i in range(5)]
    store = FakeVectorStore(by_id={})

    _expand_with_neighbors(anchors, store, n_top_for_expansion=2)

    assert len(store.get_by_ids_calls) == 2  # only the first 2 anchors triggered a lookup


def test_expand_hard_fails_when_vector_store_fails():
    anchor = _fragment("a.pdf_pag0_chunk1", total_in_page=3)
    store = FakeVectorStore(raises=True)

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        _expand_with_neighbors([anchor], store, n_top_for_expansion=3)


def test_raw_context_used_when_recomp_disabled():
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Some retrieved evidence text.")]
    rag_model = FakeChatModel()
    config = _config(**{"flags.usar_recomp_synthesis": False, "flags.expandir_contexto": False})

    result = Answer(FakeVectorStore(), rag_model, config).run("question?", fragments)

    assert result.text == "Hello world"
    assert result.metrics["recomp_used"] is False
    assert "Some retrieved evidence text." in rag_model.stream_calls[0]["prompt"]


def test_raw_context_used_when_no_recomp_model_wired_in():
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence.")]
    rag_model = FakeChatModel()
    config = _config(**{"flags.expandir_contexto": False})  # usar_recomp_synthesis stays True (default)

    result = Answer(FakeVectorStore(), rag_model, config, recomp_chat_model=None).run("q?", fragments)

    assert result.metrics["recomp_used"] is False


def test_recomp_synthesis_used_when_output_is_well_formed():
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence about the topic.")]
    rag_model = FakeChatModel()
    recomp_model = FakeChatModel(response="## Facts relevant to the question\n- A grounded fact.")
    config = _config(**{"flags.expandir_contexto": False})

    result = Answer(FakeVectorStore(), rag_model, config, recomp_chat_model=recomp_model).run(
        "question?", fragments
    )

    assert result.metrics["recomp_used"] is True
    assert "Facts relevant" in rag_model.stream_calls[0]["prompt"]


def test_recomp_falls_back_to_raw_context_when_output_missing_header():
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence about the topic.")]
    rag_model = FakeChatModel()
    recomp_model = FakeChatModel(response="Some prose without the expected markdown header at all.")
    config = _config(**{"flags.expandir_contexto": False})

    result = Answer(FakeVectorStore(), rag_model, config, recomp_chat_model=recomp_model).run(
        "question?", fragments
    )

    assert result.metrics["recomp_used"] is False


def test_recomp_falls_back_to_raw_context_when_output_too_short():
    """A response under 20 chars can never contain the 34-char header text
    either, so this and the missing-header test exercise the same fallback
    branch pair simultaneously -- that overlap is inherent to the original's
    two independent `len(sintesis) < 20` / `header not in sintesis` checks,
    not a gap in this test."""
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence about the topic.")]
    rag_model = FakeChatModel()
    recomp_model = FakeChatModel(response="short")
    config = _config(**{"flags.expandir_contexto": False})

    result = Answer(FakeVectorStore(), rag_model, config, recomp_chat_model=recomp_model).run(
        "question?", fragments
    )

    assert result.metrics["recomp_used"] is False


def test_recomp_falls_back_to_raw_context_on_model_failure():
    """Explicit use-case-level fallback (see module docstring): a failing
    RECOMP ChatModel must not abort the answer -- it degrades to raw
    formatted context, matching the original's broad except-and-fallback."""
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence about the topic.")]
    rag_model = FakeChatModel()
    recomp_model = FakeChatModel(raises=True)
    config = _config(**{"flags.expandir_contexto": False})

    result = Answer(FakeVectorStore(), rag_model, config, recomp_chat_model=recomp_model).run(
        "question?", fragments
    )

    assert result.metrics["recomp_used"] is False
    assert result.metrics["recomp_error"] is True
    assert result.text == "Hello world"  # generation still happened


def test_on_token_callback_receives_each_streamed_chunk():
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence.")]
    rag_model = FakeChatModel(tokens=("a", "b", "c"))
    config = _config(**{"flags.usar_recomp_synthesis": False, "flags.expandir_contexto": False})
    seen = []

    result = Answer(FakeVectorStore(), rag_model, config).run("q?", fragments, on_token=seen.append)

    assert seen == ["a", "b", "c"]
    assert result.text == "abc"


def test_system_prompt_is_forwarded_to_the_rag_chat_model():
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence.")]
    rag_model = FakeChatModel()
    config = _config(**{"flags.usar_recomp_synthesis": False, "flags.expandir_contexto": False})

    Answer(FakeVectorStore(), rag_model, config, system_prompt="SYSTEM PROMPT TEXT").run("q?", fragments)

    assert rag_model.stream_calls[0]["system"] == "SYSTEM PROMPT TEXT"


def test_generation_metrics_are_populated_from_the_final_streamed_chunk():
    """The debug dump / tokens-per-second derivation generar_respuesta computes
    today needs exactly these fields off the final chunk -- this is the gap
    ChatModel.stream() returning GenerationChunk (instead of plain text)
    closes: they must reach AnswerResult.metrics, not get silently dropped."""
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence.")]
    final_chunk = GenerationChunk(
        done=True, model="phi4-finetuned:latest", done_reason="stop",
        total_duration=5_000_000_000, load_duration=100,
        prompt_eval_count=120, prompt_eval_duration=200,
        eval_count=80, eval_duration=2_000_000_000,  # 2s -> 40 tokens/s
    )
    rag_model = FakeChatModel(final_chunk=final_chunk)
    config = _config(**{"flags.usar_recomp_synthesis": False, "flags.expandir_contexto": False})

    result = Answer(FakeVectorStore(), rag_model, config).run("q?", fragments)

    assert result.metrics["generation"]["model"] == "phi4-finetuned:latest"
    assert result.metrics["generation"]["done_reason"] == "stop"
    assert result.metrics["generation"]["eval_count"] == 80
    assert result.metrics["generation"]["eval_duration"] == 2_000_000_000
    assert result.metrics["generation"]["tokens_per_second"] == pytest.approx(40.0)


def test_generation_metrics_omit_fields_the_final_chunk_did_not_report():
    """Only fields actually present on the done chunk are copied -- matches
    generar_tokens_respuesta's `if k in chunk: stats[k] = chunk[k]`, applied
    to GenerationChunk's `None` defaults instead of a missing dict key."""
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence.")]
    rag_model = FakeChatModel(final_chunk=GenerationChunk(done=True, model="m"))
    config = _config(**{"flags.usar_recomp_synthesis": False, "flags.expandir_contexto": False})

    result = Answer(FakeVectorStore(), rag_model, config).run("q?", fragments)

    assert result.metrics["generation"] == {"model": "m"}


def test_generation_metrics_omit_tokens_per_second_without_eval_count_and_duration():
    fragments = [_fragment("a.pdf_pag0_chunk0", doc="Evidence.")]
    rag_model = FakeChatModel(final_chunk=GenerationChunk(done=True, model="m"))
    config = _config(**{"flags.usar_recomp_synthesis": False, "flags.expandir_contexto": False})

    result = Answer(FakeVectorStore(), rag_model, config).run("q?", fragments)

    assert "tokens_per_second" not in result.metrics["generation"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
