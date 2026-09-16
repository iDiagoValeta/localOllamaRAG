"""TextFileExtractor -- a plain-text or Markdown file as one extracted page."""
from pathlib import Path
from typing import List

from monkeygrab.domain.extracted_page import ExtractedPage

TEXT_SUFFIXES = (".txt", ".md", ".markdown")


class TextFileExtractor:
    """Implements the ``PdfExtractor`` port for text files.

    The port's contract is "one file to per-page raw text"; a text file has no
    pages, so it becomes a single zero-based page and the caller's Markdown
    chunker splits it by heading as it does with MinerU's output.

    Failure policy: hard-fail. A missing file, undecodable bytes or a suffix
    this adapter does not own raise; nothing is skipped.
    """

    def extract(self, pdf_path: str) -> List[ExtractedPage]:
        """Read ``pdf_path`` as UTF-8 (BOM tolerated) into one page.

        Args:
            pdf_path: Path to a ``.txt``, ``.md`` or ``.markdown`` file.

        Returns:
            One ``ExtractedPage`` with ``page=0``.

        Raises:
            ValueError: The suffix is not a text suffix.
            FileNotFoundError, UnicodeDecodeError: From the read itself.
        """
        path = Path(pdf_path)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError(
                f"TextFileExtractor handles {', '.join(TEXT_SUFFIXES)}, not {path.suffix!r}"
            )
        return [ExtractedPage(page=0, text=path.read_text(encoding="utf-8-sig"))]
