"""PymupdfExtractor -- PdfExtractor adapter over pymupdf4llm."""

from typing import List

import pymupdf4llm

from monkeygrab.domain.extracted_page import ExtractedPage

# Not subclassed from monkeygrab.ports.pdf_extractor.PdfExtractor: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port -- that is
# the whole point of defining ports as Protocol instead of ABC.


class PymupdfExtractor:
    """Extracts PDF pages as Markdown text via ``pymupdf4llm.to_markdown``.

    Thin translation of ``pymupdf4llm.to_markdown(pdf_path, page_chunks=True)``
    -- the same call ``rag.engine.indexing.indexar_documentos`` makes on its
    primary extraction path -- into ``ExtractedPage`` entities. No config is
    needed: extraction has no tunable parameter today, only downstream
    chunking does (owned by the caller, per the ``PdfExtractor`` port
    docstring).

    Hard-fail, deliberately without the pypdf fallback. Today's
    ``indexar_documentos`` (``rag/engine/indexing.py:180-234``) catches any
    ``pymupdf4llm`` exception and silently re-extracts the same file with
    ``pypdf`` -- a lower-fidelity extractor (no Markdown structure, no
    section headers) that produces a different, uncomparable chunk set
    without the caller ever finding out extraction quality changed. The
    project's hard-fail policy (docs/design/2026-07-26-monkeygrab-v2.md,
    section 3, "Politica de fallos") retires that fallback: a caller that
    still wants a fallback chain builds it explicitly out of two
    ``PdfExtractor`` adapters, not by having this one swallow the error.
    """

    def extract(self, pdf_path: str) -> List[ExtractedPage]:
        """Extract every page of ``pdf_path`` via pymupdf4llm.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            One ``ExtractedPage`` per page, in page order, 0-based
            (pymupdf4llm reports 1-based page numbers; normalized here to
            match the convention every other layer of the pipeline uses).

        Raises:
            RuntimeError: If pymupdf4llm fails to extract the file for any
                reason (corrupt PDF, unsupported format, missing file, ...).
        """
        try:
            page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        except Exception as exc:
            raise RuntimeError(
                f"pymupdf4llm failed to extract {pdf_path!r}: {exc}. "
                "This adapter does not fall back to pypdf (hard-fail policy, "
                "see docs/design/2026-07-26-monkeygrab-v2.md section 3) -- "
                "confirm the PDF is not corrupt or encrypted, or use a "
                "different PdfExtractor adapter."
            ) from exc

        return [
            ExtractedPage(
                page=page_info["metadata"]["page"] - 1,
                text=page_info.get("text", ""),
            )
            for page_info in page_chunks
        ]
