"""IndexCorpus -- extract -> chunk -> (contextualize, opt.) -> embed -> store,
plus image extraction -> description -> (contextualize, opt.) -> embed -> store.

Orchestrates one document through ``rag.engine.indexing.indexar_documentos``'s
per-file pipeline: extract pages (``PdfExtractor``), split each page into
chunks (``monkeygrab.application.text_chunking``), optionally enrich each
chunk with an LLM-generated situational summary (``ChatModel``, "contextual"
role), embed (``Embedder``) and store (``VectorStore``) -- then, when image
indexing is wired in and enabled, extract raster images (``ImageExtractor``),
describe each with a vision ``ChatModel`` ("ocr" role), filter degenerate
descriptions, optionally contextualize them the same way as text chunks, and
store them as their own chunks using the same offset-index scheme
``indexar_documentos`` uses (see ``_IMAGE_CHUNK_OFFSET`` below).

Scope, deliberately narrower than ``indexar_documentos``:

- **One document per call**, not a whole folder. ``os.listdir``-driven
  folder iteration is interface/orchestration concern, not something a
  ``PdfExtractor`` port (single-file ``extract(pdf_path)``) or this use case
  needs to own; a caller loops over files and calls ``IndexCorpus.run`` once
  per PDF.
- **No pypdf fallback branch.** The ``PdfExtractor`` port is hard-fail by
  design (see the port docstring): there is no second extractor to silently
  degrade to inside this use case, so the "format": "plain_text" branch
  ``indexar_documentos`` takes when pymupdf4llm raises does not have an
  equivalent here -- every text chunk this use case produces is "format":
  "markdown". This is an intentional consequence of the F1 hard-fail policy
  (docs/design/2026-07-26-monkeygrab-v2.md, section 3), not an oversight.
"""

import dataclasses
import logging
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

    Moved from ``rag.engine.contextual._detectar_idioma``. Counts
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


_PROMPT_ECHO_MARKERS = (
    # Prompt template fragments -- update if the prompt wording changes.
    "Using this caption as context, describe the visual content",
    "describe the visual content of this image in detail",
    "Focus on what is depicted, not on any text overlaid",
    "describe the arrows between blocks",
    "name the inputs and outputs",
    "lists every labeled block",
    "The figure caption reads",  # model echoing the injected caption prefix
)


def _is_description_spam(texto: str) -> bool:
    """Detect degenerate OCR output: repeated phrases or 'no text' spam.

    Moved from ``rag.engine.images._es_descripcion_spam``. Two complementary
    checks: low lexical diversity (unique words / total words < 0.35, catches
    any repetitive loop) and a 'no'/'text' token ratio over 20% (catches the
    specific "no text, no text, ..." pattern).
    """
    palabras = texto.lower().split()
    if len(palabras) < 10:
        return False
    diversidad = len(set(palabras)) / len(palabras)
    if diversidad < 0.35:
        return True
    ratio_no_text = (palabras.count("no") + palabras.count("text")) / len(palabras)
    return ratio_no_text > 0.20


def _is_prompt_echo(descripcion: str) -> bool:
    """Detect when the vision model echoes the prompt instead of describing the image.

    Moved from ``rag.engine.images._es_prompt_echo``.
    """
    desc_lower = descripcion.lower()
    return any(marker.lower() in desc_lower for marker in _PROMPT_ECHO_MARKERS)


def _is_caption_only(descripcion: str, caption: str) -> bool:
    """Check if the description merely echoes the caption with no new visual content.

    Moved from ``rag.engine.images._es_solo_caption``: true when >85% of
    caption tokens appear in the description and the description is not
    substantially longer than the caption.
    """
    if not caption:
        return False
    tokens_desc = set(descripcion.lower().split())
    tokens_cap = set(caption.lower().split())
    if not tokens_cap:
        return False
    overlap = len(tokens_desc & tokens_cap) / len(tokens_cap)
    return overlap > 0.85 and len(tokens_desc) < len(tokens_cap) * 1.3


class IndexCorpus:
    """Extract -> chunk -> (contextualize, opt.) -> embed -> store, one PDF at a time,
    plus the same pipeline for images when image indexing is wired in and enabled.

    ``contextual_model`` is ``Optional``: when not wired in, contextual
    enrichment is skipped regardless of ``flags.usar_contextual_retrieval``
    -- there is nothing to invoke otherwise, the same pattern used by
    ``Retrieve``'s and ``Answer``'s optional ports.

    ``image_extractor`` is required for any image path. How those images become
    vectors depends on the stack: a multimodal embedder (``stack.is_multimodal``
    plus ``embed_image``) indexes figures directly; a text-only embedder still
    needs ``ocr_chat_model`` to caption them first. Without a way to produce a
    vector, image extraction is skipped rather than run for nothing.
    """

    def __init__(
        self,
        extractor: PdfExtractor,
        embedder: Embedder,
        vector_store: VectorStore,
        config: AppConfig,
        contextual_model: Optional[ChatModel] = None,
        image_extractor: Optional[ImageExtractor] = None,
        ocr_chat_model: Optional[ChatModel] = None,
    ):
        """Args:
            extractor: Reads a PDF into per-page raw text.
            embedder: Embeds each chunk's final text before storage.
            vector_store: Stores each embedded chunk.
            config: Root config; ``chunking``, ``flags`` and
                ``models.embed_prefix_doc`` are read fresh on every ``run()``.
            contextual_model: ``ChatModel`` wired to the "contextual" role,
                or ``None`` to disable contextual enrichment (text AND image
                chunks).
            image_extractor: Reads a PDF's raster images, or ``None`` to
                disable image indexing regardless of
                ``flags.usar_embeddings_imagen``.
            ocr_chat_model: ``ChatModel`` wired to the "ocr" (vision) role,
                used to describe each extracted image, or ``None`` to
                disable image indexing regardless of
                ``flags.usar_embeddings_imagen``.
        """
        self._extractor = extractor
        self._embedder = embedder
        self._vector_store = vector_store
        self._contextual_model = contextual_model
        self._image_extractor = image_extractor
        self._ocr_chat_model = ocr_chat_model
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

                embedding = self._embedder.embed(f"{config.models.embed_prefix_doc}{final_text}")
                self._vector_store.add(chunk, embedding)
                total_chunks += 1

        total_image_chunks = 0
        if flags.usar_embeddings_imagen and self._image_extractor is not None:
            if self._uses_native_image_embed():
                total_image_chunks = self._index_images_native(pdf_path, filename)
            elif self._ocr_chat_model is not None:
                total_image_chunks = self._index_images(
                    pdf_path, filename, texto_base_doc, idioma_doc
                )
            total_chunks += total_image_chunks

        metrics = {
            "pages_total": len(pages),
            "pages_indexed": total_pages_used,
            "detected_language": idioma_doc,
            "image_chunks_indexed": total_image_chunks,
        }
        return IndexCorpusResult(chunks_indexed=total_chunks, metrics=metrics)

    def _uses_native_image_embed(self) -> bool:
        """True when figures can go straight into the embedder (no OCR caption)."""
        return self._config.stack.is_multimodal and callable(
            getattr(self._embedder, "embed_image", None)
        )


    def _index_images_native(self, pdf_path: str, filename: str) -> int:
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

    def _index_images(
        self, pdf_path: str, filename: str, texto_base_doc: str, idioma_doc: str
    ) -> int:
        """Extract, describe and store every qualifying image of the PDF.

        Moved from the ``if imagenes_pdf:`` block of ``indexar_documentos``
        (``rag/engine/indexing.py``, ~line 223): same offset-index chunk-id
        scheme, same "describe, then contextualize like a text chunk"
        pipeline per image, same "no description -> skip this image" rule.

        Args:
            pdf_path: Path to the PDF file to read images from.
            filename: Recorded as ``ChunkMetadata.source``, same as text chunks.
            texto_base_doc: Document-level sample, reused for both the vision
                prompt's language instruction and situational-context generation.
            idioma_doc: Document language, same as text chunks.

        Returns:
            Number of image chunks stored.
        """
        images_by_page = self._image_extractor.extract(pdf_path)
        n_indexed = 0

        for page_num, page_images in images_by_page.items():
            for img_idx, image in enumerate(page_images):
                descripcion = self._describe_image(image.image_bytes, image.caption, idioma_doc)
                if not descripcion:
                    continue

                if self._config.flags.usar_contextual_retrieval and self._contextual_model is not None:
                    situational = self._generate_situational_context(
                        descripcion, texto_base_doc, idioma_doc
                    )
                    final_description = (situational + descripcion).strip()
                else:
                    final_description = descripcion

                metadata = ChunkMetadata(
                    source=filename,
                    page=page_num,
                    chunk=_IMAGE_CHUNK_OFFSET + img_idx,
                    format="image",
                    section_header="",
                    image_width=image.width,
                    image_height=image.height,
                )
                chunk = Chunk(text=final_description, metadata=metadata)

                embedding = self._embedder.embed(
                    f"{self._config.models.embed_prefix_doc}{final_description}"
                )
                self._vector_store.add(chunk, embedding)
                n_indexed += 1

        return n_indexed

    def _describe_image(self, image_bytes: bytes, caption: str, idioma_doc: str) -> str:
        """Generate a vision-model description of one image, filtered for degenerate output.

        Moved from ``rag.engine.images.describir_imagen_con_llm``, minus its
        own ``USAR_EMBEDDINGS_IMAGEN`` guard (the caller already gates on the
        flag and on ``ocr_chat_model`` being wired in) and the direct
        ``ollama.chat`` call (delegated to the injected "ocr" ``ChatModel``
        port, via ``images=[image_bytes]`` the same way the original calls
        ``ollama.chat(..., images=[image_b64])`` -- base64 encoding is the
        ``OllamaChatModel`` adapter's job, per the ``ChatModel`` port
        contract, not this use case's).

        Args:
            image_bytes: Raw image bytes.
            caption: Caption text found near the image in the PDF, if any.
            idioma_doc: Document language, used to instruct the vision model.

        Returns:
            Non-empty description on success, or ``""`` on failure or
            degenerate output (spam, prompt echo, caption-only).
        """
        idioma_instruccion = (
            f"Write your entire description in {idioma_doc}. "
            if idioma_doc and idioma_doc != "English"
            else ""
        )
        prompt = (
            idioma_instruccion
            + (f"The figure caption reads: '{caption}'. " if caption else "")
            + "Describe the visual content of this image. "
            "If it is a flowchart or architecture diagram: list every labeled block "
            "(e.g. MatMul, SoftMax, Encoder, Linear), name the inputs and outputs "
            "(e.g. Q, K, V), describe the arrows between blocks, and state the "
            "architecture family (e.g. Transformer, CNN). "
            "If it is a table: state the number of rows and columns, list the column "
            "headers, and describe the type of data in each column. "
            "Only describe what you can actually see. Do not invent numbers, metrics, "
            "or data that are not visible in the image. "
            "End with one sentence summarizing what this figure illustrates."
        )

        try:
            descripcion = self._ocr_chat_model.generate(prompt, images=[image_bytes]).strip()
        except Exception as exc:
            # Explicit use-case-level fallback (see the ChatModel port's
            # failure policy): image description is optional enrichment --
            # on failure the image is simply not indexed, matching the
            # original's `except Exception: ...; return ""`.
            logging.warning("Error describing image: %s", exc)
            return ""

        if _is_description_spam(descripcion):
            logging.warning("Discarding degenerate OCR output (spam)")
            return ""
        if _is_prompt_echo(descripcion):
            logging.warning("Discarding prompt echo")
            return ""
        if caption and _is_caption_only(descripcion, caption):
            logging.warning("Discarding description that merely echoes the caption")
            return ""
        return descripcion if len(descripcion) > 10 else ""

    def _generate_situational_context(self, chunk_text: str, texto_base: str, idioma_doc: str) -> str:
        """Generate 2-3 sentences of situational context for a chunk via an LLM.

        Moved from ``rag.engine.contextual.generar_contexto_situacional``,
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
