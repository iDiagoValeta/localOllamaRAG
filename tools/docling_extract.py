"""Docling CLI shim: one PDF in, its pages as JSON out.

Runs inside the isolated ``.venv-docling`` interpreter, never in the product's
venv (same isolation as MinerU, see ``mineru_extractor``): Docling pulls torch
and its layout/table models, and those pins must not collide with the product's.

    .venv-docling/bin/python tools/docling_extract.py <pdf> <out.json>

Output: ``[{"page": <1-based>, "text": "..."}]`` with one entry per PDF page.
Tables are rendered as HTML (``<table>...</table>``), the same convention the
MinerU adapter follows, so a chunk that is a table stays grep-able as one.

OCR is off unless ``DOCLING_OCR=1``: the PDFs this RAG indexes carry their text,
and OCR on every page is what makes extraction slow; scanned PDFs opt in.
"""

import json
import os
import sys

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem


def convert(pdf_path: str) -> list[dict]:
    options = PdfPipelineOptions()
    options.do_ocr = os.environ.get("DOCLING_OCR") == "1"
    options.do_table_structure = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    doc = converter.convert(pdf_path).document

    pages: dict[int, list[str]] = {number: [] for number in doc.pages}
    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        page = item.prov[0].page_no
        if isinstance(item, TableItem):
            text = item.export_to_html(doc=doc)
        else:
            text = getattr(item, "text", "") or ""
        if text.strip():
            pages.setdefault(page, []).append(text.strip())
    return [{"page": number, "text": "\n\n".join(parts)} for number, parts in sorted(pages.items())]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: docling_extract.py <pdf> <out.json>", file=sys.stderr)
        return 2
    pages = convert(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump(pages, handle, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
