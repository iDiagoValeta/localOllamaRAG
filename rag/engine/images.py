"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration remains owned by rag.chat_pdfs and is synchronized before each
function call so web/API toggles and test monkeypatches keep working.
"""

import base64
import contextlib
import io
import json
import logging
import os
import re
import requests
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import ollama
from pypdf import PdfReader

from rag.cli.display import ui
from rag.engine.runtime import sync_runtime_globals


def _sync_runtime_globals() -> None:
    sync_runtime_globals(globals())


_sync_runtime_globals()
# --- 11.2 Image extraction ---


_PROMPT_ECHO_MARKERS = (
    # Prompt template fragments — update if the prompt wording changes
    "Using this caption as context, describe the visual content",
    "describe the visual content of this image in detail",
    "Focus on what is depicted, not on any text overlaid",
    "describe the arrows between blocks",
    "name the inputs and outputs",
    "lists every labeled block",
    "The figure caption reads",     # model echoing the injected caption prefix
)


def _es_descripcion_spam(texto: str) -> bool:
    """Detect degenerate OCR output: repeated phrases or 'no text' spam.

    Two complementary checks:
    - Low lexical diversity: if unique words / total words < 0.35, the text
      is almost certainly a repetitive loop (any pattern, not only 'no text').
    - 'no text' specific: ratio of 'no' + 'text' tokens > 20%.

    Args:
        texto: Raw description returned by the vision model.

    Returns:
        True if the description is considered degenerate spam.
    """
    palabras = texto.lower().split()
    if len(palabras) < 10:
        return False
    # Low lexical diversity catches "the image is a row, and the image is a row..."
    diversidad = len(set(palabras)) / len(palabras)
    if diversidad < 0.35:
        return True
    # Specific check for "no text, no text, no text..." pattern
    ratio_no_text = (palabras.count("no") + palabras.count("text")) / len(palabras)
    return ratio_no_text > 0.20


def _es_prompt_echo(descripcion: str) -> bool:
    """Detect when the vision model echoes back the prompt instead of describing the image.

    Some vision models (e.g. glm-ocr) occasionally treat the text content of the
    prompt as image text to transcribe, returning the prompt verbatim.

    Args:
        descripcion: Raw description returned by the vision model.

    Returns:
        True if the description contains a literal fragment of the system prompt.
    """
    desc_lower = descripcion.lower()
    return any(marker.lower() in desc_lower for marker in _PROMPT_ECHO_MARKERS)


def _es_solo_caption(descripcion: str, caption: str) -> bool:
    """Check if the description merely echoes the caption with no new visual content.

    Args:
        descripcion: Description returned by the vision model.
        caption: Caption text used as context in the prompt.

    Returns:
        True if >85% of caption tokens appear in the description and the
        description is not substantially longer than the caption.
    """
    if not caption:
        return False
    tokens_desc = set(descripcion.lower().split())
    tokens_cap  = set(caption.lower().split())
    if not tokens_cap:
        return False
    overlap = len(tokens_desc & tokens_cap) / len(tokens_cap)
    return overlap > 0.85 and len(tokens_desc) < len(tokens_cap) * 1.3


def extraer_imagenes_pdf(
    ruta_pdf: str,
    max_por_pagina: int = MAX_IMAGENES_POR_PAGINA,
    min_size_px: int = MIN_IMAGEN_SIZE_PX,
) -> Dict[int, List[Dict[str, Any]]]:
    """Extract all raster images from a PDF, grouped by zero-based page number.

    Opens the PDF once with PyMuPDF (fitz) and iterates over every page.
    Images below ``min_size_px`` on either side (icons, decorations) are
    discarded.  At most ``max_por_pagina`` images are kept per page,
    in document order.

    Args:
        ruta_pdf: Absolute path to the PDF file.
        max_por_pagina: Maximum number of images to keep per page.
        min_size_px: Minimum width **and** height in pixels; smaller images
            are skipped.

    Returns:
        Dict mapping zero-based page number to a list of image dicts.
        Each dict has keys ``"image_bytes"`` (raw bytes), ``"width"``,
        ``"height"``, ``"ext"`` (format string, e.g. ``"png"``), and
        ``"caption"`` (text found immediately below the image bbox on the
        same page; empty string if not found).
        Pages with no qualifying images are omitted from the dict.
    """
    if not FITZ_DISPONIBLE:
        return {}

    imagenes_por_pagina: Dict[int, List[Dict[str, Any]]] = {}

    try:
        doc = fitz.open(ruta_pdf)

        for num_pag in range(len(doc)):
            page = doc[num_pag]
            image_list = page.get_images(full=True)
            imagenes_pagina: List[Dict[str, Any]] = []

            for img_info in image_list:
                if len(imagenes_pagina) >= max_por_pagina:
                    break

                xref = img_info[0]
                try:
                    img_data = doc.extract_image(xref)
                    width = img_data.get("width", 0)
                    height = img_data.get("height", 0)

                    if width < min_size_px or height < min_size_px:
                        continue

                    caption_text = ""
                    try:
                        img_rects = page.get_image_rects(xref)
                        if img_rects:
                            img_rect = img_rects[0]
                            below_rect = fitz.Rect(
                                img_rect.x0,
                                img_rect.y1,
                                img_rect.x1,
                                img_rect.y1 + CAPTION_MARGIN_PX,
                            )
                            candidate = page.get_text("text", clip=below_rect).strip()
                            # Normalize: collapse newlines and extra spaces
                            candidate = " ".join(candidate.split())
                            if candidate and len(candidate) <= 300:
                                caption_text = candidate
                    except Exception:
                        pass  # caption is optional; never interrupt image extraction

                    imagenes_pagina.append({
                        "image_bytes": img_data["image"],
                        "width": width,
                        "height": height,
                        "ext": img_data.get("ext", "png"),
                        "caption": caption_text,
                    })
                except Exception as e:
                    logging.warning(
                        f"Error extracting image xref={xref} from {ruta_pdf} page {num_pag}: {e}"
                    )

            if imagenes_pagina:
                imagenes_por_pagina[num_pag] = imagenes_pagina

        doc.close()

    except Exception as e:
        logging.warning(f"Error reading {ruta_pdf} with fitz for image extraction: {e}")

    return imagenes_por_pagina


def describir_imagen_con_llm(
    image_bytes: bytes,
    caption: str = "",
    idioma_doc: str = "English",
) -> str:
    """Generate a textual description of an academic image using ``MODELO_OCR``.

    Sends the raw image bytes to ``gemma4:e4b`` (or the model set in
    ``OLLAMA_OCR_MODEL``) via Ollama and returns a description focused on
    the visual content: architecture components and data flow for diagrams,
    column structure for tables, axis labels and trends for charts.

    If a ``caption`` is provided, it is injected as context at the start of
    the prompt so the model can anchor its description to the figure's topic
    without simply echoing the caption text.

    Degenerate outputs are filtered before returning:
    - OCR spam (>20% of words are 'no'/'text') → returns ``""``
    - Description that merely echoes the caption → returns ``""``

    The returned string is plain text embedded with ``MODELO_EMBEDDING``
    and stored as a regular chunk in ChromaDB — no extra infrastructure
    is required for cross-modal retrieval.

    The function is a no-op (returns ``""``) when ``USAR_EMBEDDINGS_IMAGEN``
    is ``False`` or when the Ollama call fails.

    Args:
        image_bytes: Raw image bytes in any format supported by Ollama
            (PNG, JPEG, WebP, …).
        caption: Optional caption text extracted from the PDF near the image.
            Used as context to orient the description; not transcribed literally.
        idioma_doc: Language of the surrounding document ('Spanish', 'Catalan',
            'English'). The description will be written in this language so it
            is semantically consistent with the document's text chunks.

    Returns:
        Non-empty description string on success, or ``""`` on failure,
        degenerate output, or when image embeddings are disabled.
    """
    if not USAR_EMBEDDINGS_IMAGEN:
        return ""

    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        idioma_instruccion = (
            f"Write your entire description in {idioma_doc}. "
            if idioma_doc and idioma_doc != "English"
            else ""
        )

        response = ollama.chat(
            model=MODELO_OCR,
            messages=[{
                "role": "user",
                "content": (
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
                ),
                "images": [image_b64],
            }],
            think=False,
            options={"temperature": 0.1, "num_predict": 2000},
        )
        descripcion = response["message"]["content"].strip()

        if _es_descripcion_spam(descripcion):
            logging.warning(f"Discarding degenerate OCR output (spam) from {MODELO_OCR}")
            return ""
        if _es_prompt_echo(descripcion):
            logging.warning(f"Discarding prompt echo from {MODELO_OCR}")
            return ""
        if caption and _es_solo_caption(descripcion, caption):
            logging.warning(f"Discarding description that merely echoes the caption from {MODELO_OCR}")
            return ""
        return descripcion if len(descripcion) > 10 else ""

    except Exception as e:
        logging.warning(f"Error describing image with {MODELO_OCR}: {e}")
        return ""



def _with_runtime_sync(func):
    def wrapper(*args, **kwargs):
        _sync_runtime_globals()
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    return wrapper


for _name, _obj in list(globals().items()):
    if callable(_obj) and getattr(_obj, "__module__", None) == __name__ and _name not in {
        "_sync_runtime_globals", "_with_runtime_sync"
    }:
        globals()[_name] = _with_runtime_sync(_obj)

_sync_runtime_globals()

