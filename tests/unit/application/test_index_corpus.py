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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
