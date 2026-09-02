"""Unit tests for monkeygrab.application.index_corpus.IndexCorpus.

``detect_document_language`` is a pure function and is pinned directly on its
expected output per language. ``IndexCorpus`` itself -- extraction, chunking,
contextual enrichment, embedding and storage orchestration -- is exercised
against hand-written port fakes, so no Ollama server, PDF or vector store is
touched.
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.application.index_corpus import IndexCorpus, detect_document_language  # noqa: E402
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from monkeygrab.domain.extracted_image import ExtractedImage  # noqa: E402
from monkeygrab.domain.extracted_page import ExtractedPage  # noqa: E402


@pytest.mark.parametrize(
    "texto, esperado",
    [
        (
            "Este documento también describe el sistema pero con más detalle así como los resultados.",
            "Spanish",
        ),
        (
            "Aquest document també descriu el sistema però amb més detall i els resultats obtinguts.",
            "Catalan",
        ),
        (
            "This document also describes the system but with more detail and the results that were obtained.",
            "English",
        ),
        # No distinctive marker in either sample: the tie resolves to Spanish, the
        # first key of the score dict. Pinned because the caller injects the result
        # into a prompt, so the tie-break must stay a fixed language, not vary.
        ("", "Spanish"),
        ("Palabras sueltas sin marcadores claros", "Spanish"),
    ],
)
def test_detect_document_language(texto, esperado):
    assert detect_document_language(texto) == esperado


class FakeExtractor:
    def __init__(self, pages):
        self._pages = pages

    def extract(self, pdf_path):
        return self._pages


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class FakeMultimodalEmbedder(FakeEmbedder):
    """Text embedder that also satisfies ImageEmbedder (jina-style)."""

    def __init__(self):
        super().__init__()
        self.image_calls = []

    def embed_image(self, image_path, *, caption=""):
        self.image_calls.append({"path": image_path, "caption": caption})
        return [0.9, 0.8, 0.7]


class FakeVectorStore:
    def __init__(self):
        self.added = []

    def add(self, chunk, embedding):
        self.added.append((chunk, embedding))


class FakeContextualModel:
    def __init__(self, response="Situational summary.", raises=False):
        self._response = response
        self._raises = raises
        self.calls = []

    def generate(self, prompt, *, system=None, images=(), response_format=None):
        self.calls.append({"prompt": prompt, "system": system})
        if self._raises:
            raise RuntimeError("contextual model unavailable")
        return self._response


class FakeImageExtractor:
    def __init__(self, images_by_page):
        self._images_by_page = images_by_page
        self.calls = []

    def extract(self, pdf_path):
        self.calls.append(pdf_path)
        return self._images_by_page


class FakeImageDescriber:
    def __init__(self, response="The figure shows a line chart.", raises=False):
        self._response = response
        self._raises = raises
        self.calls = []

    def generate(self, prompt, *, system=None, images=(), response_format=None):
        self.calls.append({"prompt": prompt, "system": system, "images": images})
        if self._raises:
            raise RuntimeError("vision model unavailable")
        return self._response


def _small_config(**overrides):
    cfg = AppConfig().with_overrides(
        **{
            "chunking.chunk_size": 200,
            "chunking.min_chunk_length": 10,
            "chunking.chunk_overlap": 0,
        }
    )
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    return cfg


def test_indexes_one_chunk_per_page_with_correct_metadata_and_no_contextual_model():
    pages = [
        ExtractedPage(
            page=0, text="A page with enough content to survive the minimum chunk length."
        )
    ]
    extractor = FakeExtractor(pages)
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    use_case = IndexCorpus(extractor, embedder, store, _small_config(), contextual_model=None)
    result = use_case.run("ignored/path.pdf", filename="paper.pdf")

    assert result.chunks_indexed == 1
    assert len(store.added) == 1
    chunk, embedding = store.added[0]
    assert chunk.metadata.source == "paper.pdf"
    assert chunk.metadata.page == 0
    assert chunk.metadata.chunk == 0
    assert chunk.metadata.format == "markdown"
    assert chunk.id == "paper.pdf_pag0_chunk0"
    assert embedding == [0.1, 0.2, 0.3]
    # No contextual model wired in: chunk text is untouched (no situational prefix).
    assert chunk.text == "A page with enough content to survive the minimum chunk length."


def test_pages_below_min_chunk_length_are_skipped():
    pages = [ExtractedPage(page=0, text="short"), ExtractedPage(page=1, text="")]
    use_case = IndexCorpus(FakeExtractor(pages), FakeEmbedder(), FakeVectorStore(), _small_config())

    result = use_case.run("x.pdf", filename="x.pdf")

    assert result.chunks_indexed == 0
    assert result.metrics["pages_indexed"] == 0


def test_embedder_receives_document_text_without_legacy_prefix():
    pages = [
        ExtractedPage(page=0, text="Enough content here to clear the small test threshold easily.")
    ]
    embedder = FakeEmbedder()
    config = _small_config()

    IndexCorpus(FakeExtractor(pages), embedder, FakeVectorStore(), config).run("x.pdf", "x.pdf")

    assert embedder.calls[0] == pages[0].text


def test_contextual_enrichment_prepends_situational_summary_with_literal_separator():
    pages = [
        ExtractedPage(page=0, text="Enough content here to clear the small test threshold easily.")
    ]
    contextual = FakeContextualModel(response="This document is about testing.")
    store = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, _small_config(), contextual_model=contextual
    ).run("x.pdf", "x.pdf")

    chunk, _embedding = store.added[0]
    # Literal 4-char "\n\n" separator (backslash-n-backslash-n), per
    # the situational-summary separator contract.
    assert chunk.text.startswith("This document is about testing.\\n\\n")
    assert contextual.calls  # the ChatModel port was actually invoked


def test_contextual_flag_off_skips_enrichment_even_with_a_model_wired_in():
    pages = [
        ExtractedPage(page=0, text="Enough content here to clear the small test threshold easily.")
    ]
    contextual = FakeContextualModel(response="Should not appear.")
    store = FakeVectorStore()
    config = _small_config(**{"flags.usar_contextual_retrieval": False})

    IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, config, contextual_model=contextual
    ).run("x.pdf", "x.pdf")

    chunk, _embedding = store.added[0]
    assert not contextual.calls
    assert "Should not appear" not in chunk.text


def test_contextual_model_failure_falls_back_to_no_enrichment():
    """Explicit use-case-level fallback: a failing ChatModel degrades to
    "store the chunk without contextual enrichment", matching the original's
    `except Exception: ...; return ""` -- it must not propagate and abort
    indexing of an otherwise-good chunk."""
    pages = [
        ExtractedPage(page=0, text="Enough content here to clear the small test threshold easily.")
    ]
    contextual = FakeContextualModel(raises=True)
    store = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, _small_config(), contextual_model=contextual
    ).run("x.pdf", "x.pdf")

    assert len(store.added) == 1
    chunk, _embedding = store.added[0]
    assert chunk.text == "Enough content here to clear the small test threshold easily."


def test_contextual_prompt_template_is_pinned(monkeypatch):
    """Pins the situational-context prompt's fixed wording via a hash, so an
    edit to it fails a test instead of shipping silently.

    monkeygrab.application.index_fingerprint._RECIPE_VERSION exists
    precisely because this prompt template is one of the things that can
    change stored chunk text without moving a single AppConfig value (see
    that module's docstring). This test can't stop someone from editing the
    template, but it can force the edit through a test failure instead of
    past it.

    The hash covers only the fixed wording, not the per-call values
    (chunk_text/texto_base/idioma_doc are held fixed here) that get
    interpolated into it on a real call.

    If this fails because you intentionally changed the prompt: update the
    hash below AND bump index_fingerprint._RECIPE_VERSION in the same
    change -- do not update just this hash.
    """
    contextual = FakeContextualModel()
    use_case = IndexCorpus(
        FakeExtractor([]),
        FakeEmbedder(),
        FakeVectorStore(),
        AppConfig(),
        contextual_model=contextual,
    )

    use_case._generate_situational_context(
        chunk_text="a fixed chunk of text",
        texto_base="a fixed document sample",
        idioma_doc="English",
    )

    system_prompt, user_prompt = contextual.calls[0]["system"], contextual.calls[0]["prompt"]
    digest = hashlib.sha256((system_prompt + "\x00" + user_prompt).encode("utf-8")).hexdigest()[:16]
    assert digest == "9c4a5743523304fe"


def _no_text_config(**overrides):
    """A config whose pages never survive min_chunk_length, isolating image
    indexing from the text-chunking path in the assertions below."""
    return _small_config(**{"chunking.min_chunk_length": 10_000, **overrides})


def test_no_image_extractor_wired_in_skips_image_indexing_even_with_flag_on():
    pages = [ExtractedPage(page=0, text="short")]

    use_case = IndexCorpus(
        FakeExtractor(pages),
        FakeMultimodalEmbedder(),
        FakeVectorStore(),
        _no_text_config(),
        image_extractor=None,
    )
    result = use_case.run("x.pdf", "x.pdf")

    assert result.chunks_indexed == 0
    assert result.metrics["image_chunks_indexed"] == 0


def test_indexes_image_directly_in_the_shared_multimodal_space():
    pages = [ExtractedPage(page=0, text="short")]
    image = ExtractedImage(
        image_bytes=b"raw-png-bytes", width=220, height=180, ext="png", caption="Fig. 1"
    )
    images = FakeImageExtractor({1: [image]})
    embedder = FakeMultimodalEmbedder()
    store = FakeVectorStore()
    config = _no_text_config()

    result = IndexCorpus(
        FakeExtractor(pages),
        embedder,
        store,
        config,
        image_extractor=images,
    ).run("doc.pdf", "doc.pdf")

    assert result.metrics["image_chunks_indexed"] == 1
    assert len(store.added) == 1
    chunk, embedding = store.added[0]
    assert chunk.metadata.format == "image"
    assert chunk.metadata.chunk == 10_000
    assert chunk.text == "Fig. 1"
    assert embedding == [0.9, 0.8, 0.7]
    assert len(embedder.image_calls) == 1
    assert embedder.image_calls[0]["caption"] == "Fig. 1"
    assert images.calls == ["doc.pdf"]


def test_image_flag_off_skips_image_indexing_even_with_both_ports_wired_in():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor(
        {0: [ExtractedImage(image_bytes=b"x", width=200, height=200, ext="png")]}
    )
    embedder = FakeMultimodalEmbedder()
    config = _no_text_config(**{"flags.usar_embeddings_imagen": False})

    use_case = IndexCorpus(
        FakeExtractor(pages),
        embedder,
        FakeVectorStore(),
        config,
        image_extractor=images,
    )
    result = use_case.run("x.pdf", "x.pdf")

    assert result.chunks_indexed == 0
    assert not images.calls
    assert not embedder.image_calls


def test_multiple_images_on_a_page_get_sequential_offset_chunk_indices():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor(
        {
            0: [
                ExtractedImage(image_bytes=b"a", width=200, height=200, ext="png"),
                ExtractedImage(image_bytes=b"b", width=200, height=200, ext="png"),
            ]
        }
    )
    embedder = FakeMultimodalEmbedder()
    store = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages),
        embedder,
        store,
        _no_text_config(),
        image_extractor=images,
    ).run("x.pdf", "x.pdf")

    chunk_indices = sorted(chunk.metadata.chunk for chunk, _emb in store.added)
    assert chunk_indices == [10_000, 10_001]


def _one_image_config(**overrides):
    """No-text config plus the description flag on, isolating the describer
    path from text chunking in the assertions below."""
    return _no_text_config(**{"flags.usar_descripcion_imagen": True, **overrides})


def test_description_leads_stored_text_when_describer_wired_and_flag_on():
    # Enough English prose for language detection, but still below the
    # no-text config's min_chunk_length so only image chunks are stored.
    filler = (
        "This paper analyses cosmological parameters with detailed "
        "comparisons across several independent observational surveys. "
    )
    pages = [ExtractedPage(page=0, text=filler * 4)]
    images = FakeImageExtractor(
        {1: [ExtractedImage(image_bytes=b"px", width=220, height=180, ext="png", caption="Fig. 1")]}
    )
    describer = FakeImageDescriber(response="A line chart of lensing spectra.")
    store = FakeVectorStore()

    result = IndexCorpus(
        FakeExtractor(pages),
        FakeMultimodalEmbedder(),
        store,
        _one_image_config(),
        image_extractor=images,
        image_describer=describer,
    ).run("doc.pdf", "doc.pdf")

    assert result.metrics["image_description_failures"] == 0
    chunk, _embedding = store.added[0]
    assert chunk.text == "A line chart of lensing spectra.\n\nFig. 1"
    # The vision model received the raw bytes and a language instruction.
    call = describer.calls[0]
    assert call["images"] == [b"px"]
    assert "English" in call["system"]


def test_description_without_caption_replaces_placeholder_text():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor(
        {0: [ExtractedImage(image_bytes=b"px", width=10, height=10, ext="png", caption="")]}
    )
    describer = FakeImageDescriber(response="A dense statistical table.")
    store = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages),
        FakeMultimodalEmbedder(),
        store,
        _one_image_config(),
        image_extractor=images,
        image_describer=describer,
    ).run("x.pdf", "x.pdf")

    chunk, _embedding = store.added[0]
    assert chunk.text == "A dense statistical table."
    assert "[figure]" not in chunk.text


def test_description_flag_off_or_port_absent_keeps_caption_text():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor(
        {0: [ExtractedImage(image_bytes=b"px", width=10, height=10, ext="png", caption="Fig. 2")]}
    )
    describer = FakeImageDescriber(response="Should never be stored.")
    store_with_port = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages),
        FakeMultimodalEmbedder(),
        store_with_port,
        _no_text_config(),
        image_extractor=images,
        image_describer=describer,
    ).run("flag-off.pdf", "flag-off.pdf")
    IndexCorpus(
        FakeExtractor(pages),
        FakeMultimodalEmbedder(),
        FakeVectorStore(),
        _one_image_config(),
        image_extractor=images,
        image_describer=None,
    ).run("no-port.pdf", "no-port.pdf")

    chunk, _embedding = store_with_port.added[0]
    assert chunk.text == "Fig. 2"
    assert not describer.calls


def test_failing_describer_degrades_to_caption_and_counts_the_failure():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor(
        {
            0: [
                ExtractedImage(image_bytes=b"px", width=10, height=10, ext="png", caption="Fig. 3"),
                ExtractedImage(image_bytes=b"py", width=10, height=10, ext="png", caption=""),
            ]
        }
    )
    describer = FakeImageDescriber(raises=True)
    store = FakeVectorStore()

    result = IndexCorpus(
        FakeExtractor(pages),
        FakeMultimodalEmbedder(),
        store,
        _one_image_config(),
        image_extractor=images,
        image_describer=describer,
    ).run("x.pdf", "x.pdf")

    texts = [chunk.text for chunk, _emb in store.added]
    assert texts == ["Fig. 3", "[figure] page=0 idx=1"]
    assert result.metrics["image_description_failures"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
