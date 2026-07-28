"""IndexCorpus -- structured PDF extraction, multimodal embedding and storage.

Orchestrates one document through ``rag.engine.indexing.indexar_documentos``'s
per-file pipeline: extract pages (``PdfExtractor``), split each page into
chunks (``monkeygrab.application.text_chunking``), optionally enrich each
chunk with an LLM-generated situational summary (``ChatModel``, "contextual"
role), embed text and tables (``Embedder``), and store them (``VectorStore``).
When image indexing is enabled, figures extracted by MinerU are embedded
directly through the same multimodal model and stored in the shared vector
space.

Scope, deliberately narrower than ``indexar_documentos``:

- **One document per call**, not a whole folder. ``os.listdir``-driven
  folder iteration is interface/orchestration concern, not something a
  ``PdfExtractor`` port (single-file ``extract(pdf_path)``) or this use case
  needs to own; a caller loops over files and calls ``IndexCorpus.run`` once
  per PDF.
- **No extraction fallback branch.** The ``PdfExtractor`` port is hard-fail
  by design. Every text chunk this use case produces is Markdown.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

from monkeygrab.application.text_chunking import split_markdown_into_chunks
from monkeygrab.config.app_config import AppConfig
from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.extracted_page import ExtractedPage
from monkeygrab.ports.chat_model import ChatModel
from monkeygrab.ports.embedder import Embedder
from monkeygrab.ports.image_extractor import ImageExtractor
from monkeygrab.ports.pdf_extractor import PdfExtractor
from monkeygrab.ports.vector_store import VectorStore

# Image chunks share the same "{source}_pag{page}_chunk{n}" id scheme as text
# chunks, so they need an index range that never collides with a page's real
# text-chunk count. Literal copy of `_IMAGEN_CHUNK_OFFSET` in
# `rag/chat_pdfs.py`: no page realistically produces 10,000 text chunks.
_IMAGE_CHUNK_OFFSET = 10_000


@dataclasses.dataclass
class IndexCorpusResult:
    """Output of ``IndexCorpus.run``.

    Attributes:
        chunks_indexed: Total number of chunks successfully embedded and stored.
        metrics: Observability data (page/chunk counts, detected language).
    """

    chunks_indexed: int
    metrics: Dict[str, Any]


def detect_document_language(texto: str) -> str:
    """Heuristic language detector from a document text sample.

    Counts
    distinctive function-word occurrences to distinguish Spanish, Catalan,
    and English.

    Args:
        texto: Representative text sample from the document (ideally >= 500 chars).

    Returns:
        Full language name suitable for prompt injection: ``'Spanish'``,
        ``'Catalan'``, or ``'English'``.
    """
    t = texto.lower()
    ca = (t.count("però ") + t.count("també ") + t.count("molt ")
          + t.count(" amb ") + t.count(" va ") + t.count("els ")
          + t.count("l'") + t.count("d'") + t.count("s'")
          + t.count("n'") + t.count("m'"))
    es = (t.count("también ") + t.count("además ") + t.count("pero ")
          + t.count("muy ") + t.count(" con ") + t.count("los ")
          + t.count("las ") + t.count("así ") + t.count("sin ")
          + t.count("después"))
    en = (t.count(" the ") + t.count(" is ") + t.count(" are ")
          + t.count(" was ") + t.count(" were ") + t.count(" have ")
          + t.count(" this ") + t.count(" that ") + t.count(" from ")
          + t.count(" with "))
    scores = {"Spanish": es, "Catalan": ca, "English": en}
    return max(scores, key=scores.get)


def _build_document_sample(pages: Sequence[ExtractedPage], contextual_doc_chars: int) -> str:
    """Build the document-level text sample used by contextual retrieval.

    Moved from ``indexar_documentos``'s ``_preparar_texto_base_doc`` closure:
    concatenates up to the first 10 pages' text, capped at
    ``contextual_doc_chars`` total.
    """
    partes: List[str] = []
    caracteres = 0
    for page in pages[:10]:
        if not page.text:
            continue
        restante = contextual_doc_chars - caracteres
        if restante <= 0:
            break
        parte = page.text[:restante]
        partes.append(parte)
        caracteres += len(parte)
    return "\n\n".join(partes)[:contextual_doc_chars]


def _text_chunk_format(chunk_text: str) -> str:
    """Classify a text chunk as a table or as prose.

    An extractor that preserves table structure emits HTML, and a chunk carrying
    a table is a different kind of content from a paragraph: it answers "what
    value is in this cell" rather than "what does this section say". Marking it
    lets retrieval surface tables for questions that need one.

    Without this, a preserved table is indexed as ordinary prose and becomes
    invisible as a table — which is why table retrieval measured 0/5 on the
    text-flattening path even though the numbers were present in the text.

    The test is the HTML tag rather than the extractor's identity, so any backend
    that keeps tables intact benefits and none has to announce itself.

    Args:
        chunk_text: Raw chunk text, before contextual enrichment.

    Returns:
        ``"table"`` when the chunk contains an HTML table, ``"markdown"``
        otherwise.
    """
    return "table" if "<table" in chunk_text.lower() else "markdown"


class IndexCorpus:
    """Extract -> chunk -> (contextualize, opt.) -> embed -> store, one PDF at a time,
    plus direct image embedding when image indexing is enabled.

    ``contextual_model`` is ``Optional``: when not wired in, contextual
    enrichment is skipped regardless of ``flags.usar_contextual_retrieval``
    -- there is nothing to invoke otherwise, the same pattern used by
    ``Retrieve``'s and ``Answer``'s optional ports.

    ``image_extractor`` is required for image indexing. The configured Jina CLIP
    adapter satisfies both the text and image embedding ports, so figures never
    pass through a separate vision-caption model.
    """

    def __init__(
        self,
        extractor: PdfExtractor,
        embedder: Embedder,
        vector_store: VectorStore,
        config: AppConfig,
        contextual_model: Optional[ChatModel] = None,
        image_extractor: Optional[ImageExtractor] = None,
    ):
        """Args:
            extractor: Reads a PDF into per-page raw text.
            embedder: Embeds each chunk's final text before storage.
            vector_store: Stores each embedded chunk.
            config: Root config; ``chunking``, ``flags`` and
                the configured embedder is read fresh on every ``run()``.
            contextual_model: ``ChatModel`` wired to the "contextual" role,
                or ``None`` to disable contextual enrichment (text AND image
                chunks).
            image_extractor: Reads a PDF's raster images, or ``None`` to
                disable image indexing regardless of
                ``flags.usar_embeddings_imagen``.
        """
        self._extractor = extractor
        self._embedder = embedder
        self._vector_store = vector_store
        self._contextual_model = contextual_model
        self._image_extractor = image_extractor
        self._config = config

    def run(self, pdf_path: str, filename: str) -> IndexCorpusResult:
        """Index one PDF file.

        Args:
            pdf_path: Path to the PDF file to read.
            filename: Name recorded as ``ChunkMetadata.source`` (and used to
                build chunk ids) -- kept distinct from ``pdf_path`` so
                callers can index from any location while storing a stable,
                display-friendly source name (matches ``indexar_documentos``,
                which indexes by the bare filename it listed from the docs
                folder, not the full path).

        Returns:
            ``IndexCorpusResult`` with the total chunks indexed.
        """
        config = self._config
        chunking = config.chunking
        flags = config.flags

        pages = self._extractor.extract(pdf_path)

        texto_base_doc = _build_document_sample(pages, chunking.contextual_doc_chars)
        idioma_doc = detect_document_language(texto_base_doc)

        total_chunks = 0
        total_pages_used = 0

        for page in pages:
            if not page.text or len(page.text) < chunking.min_chunk_length:
                continue
            total_pages_used += 1

            text_chunks = split_markdown_into_chunks(
                page.text, chunking.chunk_size, chunking.chunk_overlap, chunking.min_chunk_length,
            )

            for chunk_idx, text_chunk in enumerate(text_chunks):
                chunk_text = text_chunk.text

                if flags.usar_contextual_retrieval and self._contextual_model is not None:
                    situational = self._generate_situational_context(
                        chunk_text, texto_base_doc, idioma_doc
                    )
                    final_text = (situational + chunk_text).strip()
                else:
                    final_text = chunk_text

                metadata = ChunkMetadata(
                    source=filename,
                    page=page.page,
                    chunk=chunk_idx,
                    total_chunks_in_page=len(text_chunks),
                    format=_text_chunk_format(chunk_text),
                    section_header=text_chunk.header,
                )
                chunk = Chunk(text=final_text, metadata=metadata)

                embedding = self._embedder.embed(final_text)
                self._vector_store.add(chunk, embedding)
                total_chunks += 1

        total_image_chunks = 0
        if flags.usar_embeddings_imagen and self._image_extractor is not None:
            total_image_chunks = self._index_images(pdf_path, filename)
            total_chunks += total_image_chunks

        metrics = {
            "pages_total": len(pages),
            "pages_indexed": total_pages_used,
            "detected_language": idioma_doc,
            "image_chunks_indexed": total_image_chunks,
        }
        return IndexCorpusResult(chunks_indexed=total_chunks, metrics=metrics)

    def _index_images(self, pdf_path: str, filename: str) -> int:
        """Extract figures and embed them via ``ImageEmbedder.embed_image``.

        Writes each ``ExtractedImage``'s bytes to a temp file because the
        ``ImageEmbedder`` port takes a path (the jina worker loads from disk).
        Chunk ``text`` keeps the caption when present so lexical/debug views
        still have something readable; the vector itself is from pixels.
        """
        import os
        import tempfile

        images_by_page = self._image_extractor.extract(pdf_path)
        n_indexed = 0
        embed_image = self._embedder.embed_image

        for page_num, page_images in images_by_page.items():
            for img_idx, image in enumerate(page_images):
                suffix = f".{image.ext}" if image.ext else ".png"
                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(image.image_bytes)
                    embedding = embed_image(tmp_path, caption=image.caption or "")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                caption = (image.caption or "").strip()
                text = caption if caption else f"[figure] page={page_num} idx={img_idx}"
                metadata = ChunkMetadata(
                    source=filename,
                    page=page_num,
                    chunk=_IMAGE_CHUNK_OFFSET + img_idx,
                    format="image",
                    section_header="",
                    image_width=image.width,
                    image_height=image.height,
                )
                self._vector_store.add(Chunk(text=text, metadata=metadata), embedding)
                n_indexed += 1

        return n_indexed

    def _generate_situational_context(self, chunk_text: str, texto_base: str, idioma_doc: str) -> str:
        """Generate 2-3 sentences of situational context for a chunk via an LLM.

        Written in the document's own language,
        minus its own ``USAR_CONTEXTUAL_RETRIEVAL`` guard (the caller already
        gates on the flag before invoking this) and the direct ``ollama.chat``
        call (delegated to the injected ``ChatModel`` port).

        Args:
            chunk_text: The text of the chunk to contextualize.
            texto_base: A representative excerpt of the full document.
            idioma_doc: Document language ('Spanish', 'Catalan', 'English').

        Returns:
            Situational context string (with trailing ``\\n\\n``), or an
            empty string on failure.
        """
        idioma = idioma_doc or detect_document_language(texto_base)

        system_prompt = (
            f"You are an expert at analyzing academic documents. "
            f"MANDATORY: Write your entire response in {idioma} — the same language as the document. "
            f"Do NOT translate. Do NOT switch to any other language, including English. "
            f"When given a full document and an excerpt from it, produce exactly 2-3 sentences: "
            f"first a brief summary of what the document is about, then how the excerpt fits within it. "
            f"No introductions, no labels, no meta-commentary. "
            f"Do NOT include bibliographic citation markers such as [1], [38], or similar."
        )

        user_prompt = (
            f"<document>\\n{texto_base}\\n</document>\\n\\n"
            f"<excerpt>\\n{chunk_text}\\n</excerpt>\\n\\n"
            f"Write the 2-3 sentence situational context in {idioma}."
        )

        try:
            contexto = self._contextual_model.generate(user_prompt, system=system_prompt).strip()
        except Exception:
            # Explicit use-case-level fallback (see the ChatModel port's
            # failure policy): contextual enrichment is optional -- on
            # failure the chunk is stored without it, matching the
            # original's `except Exception: ...; return ""`.
            return ""
        return f"{contexto}\\n\\n" if contexto else ""
