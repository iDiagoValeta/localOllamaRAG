"""Unit + equivalence tests for monkeygrab.application.index_corpus.IndexCorpus.

``detect_document_language`` is equivalence-tested against
``rag.engine.contextual._detectar_idioma`` (both pure functions). The rest
of ``IndexCorpus`` -- extraction/chunking/contextual-enrichment/embedding/
storage orchestration -- has no single original function to diff against
(``indexar_documentos`` mixes this with folder iteration, pypdf fallback and
image indexing that are explicitly out of scope, see
``monkeygrab.application.index_corpus``'s module docstring), so it is
covered here with hand-written port fakes instead.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import rag.chat_pdfs as rag  # noqa: E402

from monkeygrab.application.index_corpus import IndexCorpus, detect_document_language  # noqa: E402
from monkeygrab.config.app_config import AppConfig  # noqa: E402
from monkeygrab.domain.extracted_image import ExtractedImage  # noqa: E402
from monkeygrab.domain.extracted_page import ExtractedPage  # noqa: E402


# ─────────────────────────────────────────────
# detect_document_language vs _detectar_idioma
# ─────────────────────────────────────────────


@pytest.mark.parametrize("texto", [
    "Este documento también describe el sistema pero con más detalle así como los resultados.",
    "Aquest document també descriu el sistema però amb més detall i els resultats obtinguts.",
    "This document also describes the system but with more detail and the results that were obtained.",
    "",
    "Palabras sueltas sin marcadores claros",
])
def test_detect_document_language_matches_original(texto):
    assert detect_document_language(texto) == rag._detectar_idioma(texto)


# ─────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────


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

    def generate(self, prompt, *, system=None, images=()):
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


class FakeOcrChatModel:
    """ChatModel double for the "ocr" (vision) role."""

    def __init__(self, response="A diagram showing three connected blocks.", raises=False):
        self._response = response
        self._raises = raises
        self.calls = []

    def generate(self, prompt, *, system=None, images=()):
        self.calls.append({"prompt": prompt, "images": images})
        if self._raises:
            raise RuntimeError("ocr model unavailable")
        return self._response


def _small_config(**overrides):
    cfg = AppConfig().with_overrides(**{
        "chunking.chunk_size": 200,
        "chunking.min_chunk_length": 10,
        "chunking.chunk_overlap": 0,
    })
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    return cfg


# ─────────────────────────────────────────────
# IndexCorpus orchestration
# ─────────────────────────────────────────────


def test_indexes_one_chunk_per_page_with_correct_metadata_and_no_contextual_model():
    pages = [ExtractedPage(page=0, text="A page with enough content to survive the minimum chunk length.")]
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


def test_embedder_receives_the_doc_prefix_from_config():
    pages = [ExtractedPage(page=0, text="Enough content here to clear the small test threshold easily.")]
    embedder = FakeEmbedder()
    config = _small_config(**{"models.embedding": "nomic-embed-text:latest"})
    # nomic models get a "search_document: " prefix (see derive_embedding_prefixes)
    assert config.models.embed_prefix_doc == "search_document: "

    IndexCorpus(FakeExtractor(pages), embedder, FakeVectorStore(), config).run("x.pdf", "x.pdf")

    assert embedder.calls[0].startswith("search_document: ")


def test_contextual_enrichment_prepends_situational_summary_with_literal_separator():
    pages = [ExtractedPage(page=0, text="Enough content here to clear the small test threshold easily.")]
    contextual = FakeContextualModel(response="This document is about testing.")
    store = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, _small_config(), contextual_model=contextual
    ).run("x.pdf", "x.pdf")

    chunk, _embedding = store.added[0]
    # Literal 4-char "\n\n" separator (backslash-n-backslash-n), per
    # generar_contexto_situacional / _texto_fuente_fragmento's contract.
    assert chunk.text.startswith("This document is about testing.\\n\\n")
    assert contextual.calls  # the ChatModel port was actually invoked


def test_contextual_flag_off_skips_enrichment_even_with_a_model_wired_in():
    pages = [ExtractedPage(page=0, text="Enough content here to clear the small test threshold easily.")]
    contextual = FakeContextualModel(response="Should not appear.")
    store = FakeVectorStore()
    config = _small_config(**{"flags.usar_contextual_retrieval": False})

    IndexCorpus(FakeExtractor(pages), FakeEmbedder(), store, config, contextual_model=contextual).run("x.pdf", "x.pdf")

    chunk, _embedding = store.added[0]
    assert not contextual.calls
    assert "Should not appear" not in chunk.text


def test_contextual_model_failure_falls_back_to_no_enrichment():
    """Explicit use-case-level fallback: a failing ChatModel degrades to
    "store the chunk without contextual enrichment", matching the original's
    `except Exception: ...; return ""` -- it must not propagate and abort
    indexing of an otherwise-good chunk."""
    pages = [ExtractedPage(page=0, text="Enough content here to clear the small test threshold easily.")]
    contextual = FakeContextualModel(raises=True)
    store = FakeVectorStore()

    IndexCorpus(FakeExtractor(pages), FakeEmbedder(), store, _small_config(), contextual_model=contextual).run(
        "x.pdf", "x.pdf"
    )

    assert len(store.added) == 1
    chunk, _embedding = store.added[0]
    assert chunk.text == "Enough content here to clear the small test threshold easily."


# ─────────────────────────────────────────────
# Image indexing
# ─────────────────────────────────────────────


def _no_text_config(**overrides):
    """A config whose pages never survive min_chunk_length, isolating image
    indexing from the text-chunking path in the assertions below."""
    return _small_config(**{"chunking.min_chunk_length": 10_000, **overrides})


def test_no_image_extractor_wired_in_skips_image_indexing_even_with_flag_on():
    pages = [ExtractedPage(page=0, text="short")]
    ocr = FakeOcrChatModel()

    use_case = IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), FakeVectorStore(), _no_text_config(),
        image_extractor=None, ocr_chat_model=ocr,
    )
    result = use_case.run("x.pdf", "x.pdf")

    assert result.chunks_indexed == 0
    assert result.metrics["image_chunks_indexed"] == 0
    assert not ocr.calls


def test_image_extractor_without_an_ocr_model_skips_image_indexing():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor({0: [ExtractedImage(image_bytes=b"x", width=200, height=200, ext="png")]})

    use_case = IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), FakeVectorStore(), _no_text_config(),
        image_extractor=images, ocr_chat_model=None,
    )
    result = use_case.run("x.pdf", "x.pdf")

    assert result.chunks_indexed == 0
    assert not images.calls  # extraction itself never runs without a way to describe the result


def test_image_flag_off_skips_image_indexing_even_with_both_ports_wired_in():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor({0: [ExtractedImage(image_bytes=b"x", width=200, height=200, ext="png")]})
    ocr = FakeOcrChatModel()
    config = _no_text_config(**{"flags.usar_embeddings_imagen": False})

    use_case = IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), FakeVectorStore(), config,
        image_extractor=images, ocr_chat_model=ocr,
    )
    result = use_case.run("x.pdf", "x.pdf")

    assert result.chunks_indexed == 0
    assert not ocr.calls


def test_indexes_one_image_chunk_with_offset_index_and_correct_metadata():
    pages = [ExtractedPage(page=0, text="short")]
    image = ExtractedImage(image_bytes=b"raw-bytes", width=300, height=150, ext="png", caption="Figure 1")
    images = FakeImageExtractor({2: [image]})
    ocr = FakeOcrChatModel(response="A bar chart comparing three configurations across two metrics.")
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    use_case = IndexCorpus(
        FakeExtractor(pages), embedder, store, _no_text_config(),
        image_extractor=images, ocr_chat_model=ocr,
    )
    result = use_case.run("some/path.pdf", filename="paper.pdf")

    assert result.chunks_indexed == 1
    assert result.metrics["image_chunks_indexed"] == 1
    assert len(store.added) == 1
    chunk, embedding = store.added[0]
    assert chunk.metadata.source == "paper.pdf"
    assert chunk.metadata.page == 2
    assert chunk.metadata.chunk == 10_000  # _IMAGE_CHUNK_OFFSET + img_idx(0)
    assert chunk.metadata.format == "image"
    assert chunk.metadata.image_width == 300
    assert chunk.metadata.image_height == 150
    assert chunk.metadata.total_chunks_in_page is None  # image chunks never set this
    assert chunk.id == "paper.pdf_pag2_chunk10000"
    assert chunk.text == "A bar chart comparing three configurations across two metrics."
    assert embedding == [0.1, 0.2, 0.3]
    # extract() is called with the real pdf_path, same as the text-extraction path.
    assert images.calls == ["some/path.pdf"]
    # The vision ChatModel is called with the image bytes, not base64 text --
    # base64 encoding is the OllamaChatModel adapter's job, not this use case's.
    assert ocr.calls[0]["images"] == [b"raw-bytes"]


def test_multiple_images_on_a_page_get_sequential_offset_chunk_indices():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor({0: [
        ExtractedImage(image_bytes=b"a", width=200, height=200, ext="png"),
        ExtractedImage(image_bytes=b"b", width=200, height=200, ext="png"),
    ]})
    ocr = FakeOcrChatModel(response="A sufficiently long and coherent image description here.")
    store = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, _no_text_config(),
        image_extractor=images, ocr_chat_model=ocr,
    ).run("x.pdf", "x.pdf")

    chunk_indices = sorted(chunk.metadata.chunk for chunk, _emb in store.added)
    assert chunk_indices == [10_000, 10_001]


def test_degenerate_image_description_is_not_indexed():
    """Spam-filtered OCR output (low lexical diversity) must not produce a chunk,
    matching describir_imagen_con_llm's own `_es_descripcion_spam` gate."""
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor({0: [ExtractedImage(image_bytes=b"x", width=200, height=200, ext="png")]})
    spam = "no text no text no text no text no text no text no text no text no text no text"
    ocr = FakeOcrChatModel(response=spam)
    store = FakeVectorStore()

    result = IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, _no_text_config(),
        image_extractor=images, ocr_chat_model=ocr,
    ).run("x.pdf", "x.pdf")

    assert result.chunks_indexed == 0
    assert result.metrics["image_chunks_indexed"] == 0
    assert store.added == []


def test_image_description_that_only_echoes_the_caption_is_not_indexed():
    pages = [ExtractedPage(page=0, text="short")]
    caption = "Figure 3 shows the overall system architecture diagram"
    image = ExtractedImage(image_bytes=b"x", width=200, height=200, ext="png", caption=caption)
    images = FakeImageExtractor({0: [image]})
    # Near-identical to the caption -- token overlap > 85%, not much longer.
    ocr = FakeOcrChatModel(response=caption)
    store = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, _no_text_config(),
        image_extractor=images, ocr_chat_model=ocr,
    ).run("x.pdf", "x.pdf")

    assert store.added == []


def test_ocr_model_failure_skips_the_image_without_aborting_indexing():
    """Explicit use-case-level fallback, matching the original's
    `except Exception: ...; return ""`: a failing vision model must not
    propagate and abort indexing of the rest of the document."""
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor({0: [ExtractedImage(image_bytes=b"x", width=200, height=200, ext="png")]})
    ocr = FakeOcrChatModel(raises=True)
    store = FakeVectorStore()

    result = IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, _no_text_config(),
        image_extractor=images, ocr_chat_model=ocr,
    ).run("x.pdf", "x.pdf")

    assert result.chunks_indexed == 0
    assert store.added == []


def test_contextual_enrichment_applies_to_image_chunks_when_flag_and_model_are_both_present():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor({0: [ExtractedImage(image_bytes=b"x", width=200, height=200, ext="png")]})
    ocr = FakeOcrChatModel(response="A sufficiently long and coherent image description here.")
    contextual = FakeContextualModel(response="This document covers testing infrastructure.")
    store = FakeVectorStore()

    IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, _no_text_config(),
        contextual_model=contextual, image_extractor=images, ocr_chat_model=ocr,
    ).run("x.pdf", "x.pdf")

    chunk, _embedding = store.added[0]
    assert chunk.text.startswith("This document covers testing infrastructure.\\n\\n")
    assert contextual.calls


def test_contextual_flag_off_skips_enrichment_for_image_chunks_too():
    pages = [ExtractedPage(page=0, text="short")]
    images = FakeImageExtractor({0: [ExtractedImage(image_bytes=b"x", width=200, height=200, ext="png")]})
    description = "A sufficiently long and coherent image description here."
    ocr = FakeOcrChatModel(response=description)
    contextual = FakeContextualModel(response="Should not appear.")
    store = FakeVectorStore()
    config = _no_text_config(**{"flags.usar_contextual_retrieval": False})

    IndexCorpus(
        FakeExtractor(pages), FakeEmbedder(), store, config,
        contextual_model=contextual, image_extractor=images, ocr_chat_model=ocr,
    ).run("x.pdf", "x.pdf")

    chunk, _embedding = store.added[0]
    assert not contextual.calls
    assert chunk.text == description


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
