"""BySuffixExtractor and PdfOnlyImageExtractor: routing by file suffix, nothing else."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.extraction.by_suffix_extractor import BySuffixExtractor, PdfOnlyImageExtractor
from monkeygrab.domain.extracted_page import ExtractedPage


class _Recording:
    def __init__(self, label):
        self.label = label
        self.calls = []

    def extract(self, path):
        self.calls.append(path)
        return [ExtractedPage(page=0, text=self.label)]


def test_pdf_goes_to_the_pdf_extractor():
    pdf, text = _Recording("pdf"), _Recording("text")
    pages = BySuffixExtractor(pdf, text).extract("/x/paper.PDF")
    assert pages[0].text == "pdf" and pdf.calls == ["/x/paper.PDF"] and text.calls == []


@pytest.mark.parametrize("name", ["a.txt", "b.md", "c.markdown"])
def test_text_suffixes_go_to_the_text_extractor(name):
    pdf, text = _Recording("pdf"), _Recording("text")
    assert BySuffixExtractor(pdf, text).extract(name)[0].text == "text"
    assert pdf.calls == []


def test_unknown_suffix_raises_instead_of_skipping():
    with pytest.raises(ValueError, match=".docx"):
        BySuffixExtractor(_Recording("pdf"), _Recording("text")).extract("/x/doc.docx")


def test_images_are_extracted_only_from_pdf():
    class _Images:
        def __init__(self):
            self.calls = []

        def extract(self, path):
            self.calls.append(path)
            return {0: ["img"]}

    inner = _Images()
    wrapper = PdfOnlyImageExtractor(inner)
    assert wrapper.extract("/x/notes.md") == {}
    assert inner.calls == []
    assert wrapper.extract("/x/paper.pdf") == {0: ["img"]}
    assert inner.calls == ["/x/paper.pdf"]
