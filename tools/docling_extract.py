"""Docling CLI shim: one PDF in, its pages and figures out.

Runs inside the isolated ``.venv-docling`` interpreter, never in the product's
venv (same isolation as MinerU, see ``mineru_extractor``): Docling pulls torch
and its layout/table models, and those pins must not collide with the product's.

    .venv-docling/bin/python tools/docling_extract.py <pdf> <out_dir>

Output, in ``<out_dir>``: ``result.json`` holding
``{"pages": [{"page": <1-based>, "text": "..."}], "figures": [{"page": <1-based>,
"file": "figures/<n>.png", "caption": "..."}]}`` with one page entry per PDF
page, and the cropped figures under ``figures/``. One conversion serves both the
text and the image port, as MinerU's single run does: converting twice would
double the cost for nothing.

Tables are rendered as HTML (``<table>...</table>``) with their caption inside,
the same convention the MinerU adapter follows, so a chunk that is a table stays
grep-able as one. A paragraph that runs across a page break is split at the
break, so each part is cited on the page it is printed on.

OCR is off unless ``DOCLING_OCR=1``: the PDFs this RAG indexes carry their text,
and OCR on every page is what makes extraction slow; scanned PDFs opt in.
"""

import json
import os
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem

# Figures are cropped at 2x the PDF's 72 dpi; 1x leaves small figures too
# coarse for the image embedder to read.
_IMAGES_SCALE = 2.0


def convert(pdf_path: str, out_dir: Path) -> dict:
    options = PdfPipelineOptions()
    options.do_ocr = os.environ.get("DOCLING_OCR") == "1"
    options.do_table_structure = True
    options.generate_picture_images = True
    options.images_scale = _IMAGES_SCALE
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    doc = converter.convert(pdf_path).document

    # A table's caption is already inside its HTML; emitting it again as a
    # paragraph would index it twice.
    table_captions = {
        ref.cref
        for item, _ in doc.iterate_items()
        if isinstance(item, TableItem)
        for ref in item.captions
    }

    pages: dict[int, list[str]] = {number: [] for number in doc.pages}
    figures: list[dict] = []
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None) or item.self_ref in table_captions:
            continue
        if isinstance(item, PictureItem):
            image = item.get_image(doc)
            if image is not None:
                name = f"figures/{len(figures)}.png"
                image.save(out_dir / name)
                figures.append({"page": item.prov[0].page_no, "file": name, "caption": item.caption_text(doc)})
            continue
        if isinstance(item, TableItem):
            parts = [(item.prov[0].page_no, item.export_to_html(doc=doc))]
        else:
            text = getattr(item, "text", "") or ""
            if len(item.prov) > 1:
                parts = [(prov.page_no, text[prov.charspan[0]:prov.charspan[1]]) for prov in item.prov]
            else:
                parts = [(item.prov[0].page_no, text)]
        for page, part in parts:
            if part.strip():
                pages.setdefault(page, []).append(part.strip())
    return {
        "pages": [{"page": number, "text": "\n\n".join(parts)} for number, parts in sorted(pages.items())],
        "figures": figures,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: docling_extract.py <pdf> <out_dir>", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[2])
    result = convert(sys.argv[1], out_dir)
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
