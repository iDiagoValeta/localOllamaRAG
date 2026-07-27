"""Unit tests for monkeygrab.adapters.extraction.pymupdf_image_extractor.PymupdfImageExtractor.

Stubs the ``fitz`` module the adapter imports -- no real PDF, no filesystem
access -- so these run in milliseconds and never touch PyMuPDF for real.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab.adapters.extraction import pymupdf_image_extractor as module  # noqa: E402
from monkeygrab.adapters.extraction.pymupdf_image_extractor import PymupdfImageExtractor  # noqa: E402
from monkeygrab.config.chunking import ChunkingConfig  # noqa: E402
from monkeygrab.domain.extracted_image import ExtractedImage  # noqa: E402


class _FakeRect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


class _FakePage:
    def __init__(self, images, caption_text=""):
        # images: list of (xref, width, height, ext, image_bytes) tuples.
        self._images = images
        self._caption_text = caption_text

    def get_images(self, full=True):
        return [(xref,) for xref, *_ in self._images]

    def get_image_rects(self, xref):
        return [_FakeRect(0, 0, 10, 10)]

    def get_text(self, mode, clip=None):
        return self._caption_text


class _FakeDoc:
    def __init__(self, pages, image_data_by_xref, closed_flag):
        self._pages = pages
        self._image_data = image_data_by_xref
        self._closed_flag = closed_flag

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, idx):
        return self._pages[idx]

    def extract_image(self, xref):
        return self._image_data[xref]

    def close(self):
        self._closed_flag.append(True)


def _make_fake_fitz(pages, image_data_by_xref, closed_flag=None):
    closed_flag = closed_flag if closed_flag is not None else []

    class _FakeFitzModule:
        Rect = _FakeRect

        @staticmethod
        def open(pdf_path):
            return _FakeDoc(pages, image_data_by_xref, closed_flag)

    return _FakeFitzModule()


def _config(**overrides):
    return ChunkingConfig(**overrides) if overrides else ChunkingConfig()


def test_extracts_one_qualifying_image_with_its_caption(monkeypatch):
    page = _FakePage(images=[(1, 200, 200, "png", b"bytes")], caption_text="Figure 1: a diagram")
    image_data = {1: {"width": 200, "height": 200, "ext": "png", "image": b"raw-bytes"}}
    monkeypatch.setattr(module, "fitz", _make_fake_fitz([page], image_data))

    result = PymupdfImageExtractor(_config(min_image_size_px=100)).extract("doc.pdf")

    assert result == {0: [ExtractedImage(
        image_bytes=b"raw-bytes", width=200, height=200, ext="png", caption="Figure 1: a diagram",
    )]}


def test_images_below_the_minimum_size_are_skipped(monkeypatch):
    page = _FakePage(images=[(1, 50, 50, "png", b"tiny")])
    image_data = {1: {"width": 50, "height": 50, "ext": "png", "image": b"tiny-bytes"}}
    monkeypatch.setattr(module, "fitz", _make_fake_fitz([page], image_data))

    result = PymupdfImageExtractor(_config(min_image_size_px=100)).extract("doc.pdf")

    assert result == {}


def test_at_most_max_images_per_page_are_kept(monkeypatch):
    page = _FakePage(images=[(i, 200, 200, "png", b"x") for i in range(5)])
    image_data = {i: {"width": 200, "height": 200, "ext": "png", "image": f"img{i}".encode()} for i in range(5)}
    monkeypatch.setattr(module, "fitz", _make_fake_fitz([page], image_data))

    result = PymupdfImageExtractor(_config(min_image_size_px=100, max_images_per_page=2)).extract("doc.pdf")

    assert len(result[0]) == 2


def test_pages_with_no_qualifying_images_are_omitted_from_the_mapping(monkeypatch):
    page_with = _FakePage(images=[(1, 200, 200, "png", b"x")])
    page_without = _FakePage(images=[])
    image_data = {1: {"width": 200, "height": 200, "ext": "png", "image": b"raw"}}
    monkeypatch.setattr(module, "fitz", _make_fake_fitz([page_without, page_with], image_data))

    result = PymupdfImageExtractor(_config(min_image_size_px=100)).extract("doc.pdf")

    assert list(result.keys()) == [1]  # page 0 (no images) is absent, not an empty list


def test_document_is_closed_after_extraction(monkeypatch):
    page = _FakePage(images=[])
    closed_flag = []
    monkeypatch.setattr(module, "fitz", _make_fake_fitz([page], {}, closed_flag))

    PymupdfImageExtractor(_config()).extract("doc.pdf")

    assert closed_flag == [True]


def test_a_single_image_extraction_failure_is_skipped_not_fatal(monkeypatch):
    """One corrupt xref must not abort extraction of the rest of the page/PDF --
    the port's documented carve-out from the project's hard-fail default."""
    page = _FakePage(images=[(1, 200, 200, "png", b"bad"), (2, 200, 200, "png", b"good")])

    class _PartiallyBrokenImageData(dict):
        def __getitem__(self, xref):
            if xref == 1:
                raise ValueError("corrupt image stream")
            return super().__getitem__(xref)

    image_data = _PartiallyBrokenImageData({2: {"width": 200, "height": 200, "ext": "png", "image": b"good-bytes"}})
    monkeypatch.setattr(module, "fitz", _make_fake_fitz([page], image_data))

    result = PymupdfImageExtractor(_config(min_image_size_px=100)).extract("doc.pdf")

    assert result == {0: [ExtractedImage(image_bytes=b"good-bytes", width=200, height=200, ext="png", caption="")]}


def test_whole_pdf_open_failure_returns_an_empty_mapping_instead_of_raising(monkeypatch):
    class _FailingFitzModule:
        @staticmethod
        def open(pdf_path):
            raise RuntimeError("cannot open corrupt.pdf")

    monkeypatch.setattr(module, "fitz", _FailingFitzModule())

    result = PymupdfImageExtractor(_config()).extract("corrupt.pdf")

    assert result == {}


def test_caption_longer_than_300_chars_is_dropped(monkeypatch):
    long_caption = "x" * 301
    page = _FakePage(images=[(1, 200, 200, "png", b"x")], caption_text=long_caption)
    image_data = {1: {"width": 200, "height": 200, "ext": "png", "image": b"raw"}}
    monkeypatch.setattr(module, "fitz", _make_fake_fitz([page], image_data))

    result = PymupdfImageExtractor(_config(min_image_size_px=100)).extract("doc.pdf")

    assert result[0][0].caption == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
