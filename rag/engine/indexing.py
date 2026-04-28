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
# --- 11.3 Indexing ---


def indexar_documentos(
    carpeta: str,
    collection: chromadb.Collection,
    solo_archivos: Optional[List[str]] = None,
    silent: bool = False,
    progress_callback=None,
) -> int:
    """Index PDFs from a folder into ChromaDB.

    Uses pymupdf4llm for Markdown extraction (preferred) with pypdf as
    fallback. Each page is split into chunks, optionally enriched with
    contextual retrieval, embedded via Ollama, and stored in ChromaDB.

    Args:
        carpeta: Path to the folder containing PDF files.
        collection: ChromaDB collection to index into.
        solo_archivos: If set, only index these specific filenames
            (for incremental adds without full re-index).
        silent: Suppress all terminal output (for background/web use).
        progress_callback: Called with ``{"file", "file_index",
            "total_files"}`` at the start of each file.

    Returns:
        Total number of chunks successfully indexed.
    """
    global PYMUPDF_AVAILABLE

    os.makedirs(carpeta, exist_ok=True)
    archivos_pdf = [f for f in os.listdir(carpeta) if f.endswith('.pdf')]
    if solo_archivos is not None:
        archivos_pdf = [f for f in archivos_pdf if f in solo_archivos]

    if not archivos_pdf:
        if not silent:
            ui.warning("No PDF files found in folder")
        return 0

    if not silent:
        ui.pipeline_start("Indexing documents...")

    total_chunks = 0

    def _indexar_chunk(id_doc: str, chunk_text: str, chunk_doc_text: str,
                       metadata: Dict, collection_ref: chromadb.Collection) -> bool:
        """Embed a chunk and add it to ChromaDB. Retries with truncation on length errors."""
        text_to_embed = f"{EMBED_PREFIX_DOC}{chunk_text[:MAX_CHARS_EMBED]}"

        try:
            response = ollama.embeddings(model=MODELO_EMBEDDING, prompt=text_to_embed)
            embedding = response["embedding"]

            collection_ref.add(
                ids=[id_doc],
                embeddings=[embedding],
                documents=[chunk_doc_text],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            if "context length" in str(e).lower() or "500" in str(e):
                logging.warning(f"Long chunk at {id_doc}, truncating to 1000 chars")
                text_to_embed = f"{EMBED_PREFIX_DOC}{chunk_text[:1000]}"
                try:
                    response = ollama.embeddings(model=MODELO_EMBEDDING, prompt=text_to_embed)
                    collection_ref.add(
                        ids=[id_doc],
                        embeddings=[response["embedding"]],
                        documents=[chunk_doc_text],
                        metadatas=[metadata]
                    )
                    return True
                except Exception as e2:
                    logging.error(f"Persistent embedding error for {id_doc}: {e2}")
            else:
                logging.error(f"Error embedding {id_doc}: {e}")
            return False

    for idx, archivo in enumerate(archivos_pdf):
        if progress_callback:
            try:
                progress_callback({"file": archivo, "file_index": idx + 1, "total_files": len(archivos_pdf)})
            except Exception:
                pass
        if not silent:
            ui.pipeline_update(f"Processing: {archivo}")
        usar_pypdf_fallback = False

        try:
            ruta_pdf = os.path.join(carpeta, archivo)

            imagenes_pdf: Dict[int, List[Dict[str, Any]]] = {}
            if USAR_EMBEDDINGS_IMAGEN and FITZ_DISPONIBLE:
                imagenes_pdf = extraer_imagenes_pdf(ruta_pdf)
                n_imgs_total = sum(len(v) for v in imagenes_pdf.values())
                if n_imgs_total > 0 and not silent:
                    ui.debug(f"  {n_imgs_total} images found across {len(imagenes_pdf)} page(s)")

            if PYMUPDF_AVAILABLE:
                try:
                    page_chunks = pymupdf4llm.to_markdown(ruta_pdf, page_chunks=True)

                    _textos_paginas = [p['text'][:500] for p in page_chunks[:10]]
                    texto_base_doc = "\n\n".join(_textos_paginas)[:4000]
                    idioma_doc = _detectar_idioma(texto_base_doc)

                    for page_info in page_chunks:
                        i = page_info['metadata']['page']
                        texto = page_info['text']

                        if not texto or len(texto) < MIN_CHUNK_LENGTH:
                            continue

                        chunks = dividir_en_chunks(texto)

                        for chunk_idx, chunk_info in enumerate(chunks):
                            chunk_text = chunk_info['text'] if isinstance(chunk_info, dict) else chunk_info
                            chunk_header = chunk_info.get('header', '') if isinstance(chunk_info, dict) else ''

                            id_doc = f"{archivo}_pag{i}_chunk{chunk_idx}"

                            metadata = {
                                "source": archivo,
                                "page": i,
                                "chunk": chunk_idx,
                                "total_chunks_in_page": len(chunks),
                                "format": "markdown",
                                "section_header": chunk_header
                            }

                            if USAR_CONTEXTUAL_RETRIEVAL:
                                contexto_sit = generar_contexto_situacional(chunk_text, texto_base_doc, idioma_doc)
                                chunk_text_con_contexto = (contexto_sit + chunk_text).strip()
                            else:
                                chunk_text_con_contexto = chunk_text

                            if _indexar_chunk(id_doc, chunk_text_con_contexto, chunk_text_con_contexto, metadata, collection):
                                total_chunks += 1

                except Exception as e:
                    logging.error(f"Error with pymupdf4llm on {archivo}: {e}, using pypdf fallback")
                    usar_pypdf_fallback = True

            if not PYMUPDF_AVAILABLE or usar_pypdf_fallback:
                reader = PdfReader(ruta_pdf)

                _textos_paginas = [p.extract_text()[:500] for p in reader.pages[:10]]
                texto_base_doc = "\n\n".join(_textos_paginas)[:4000]
                idioma_doc = _detectar_idioma(texto_base_doc)

                for i, page in enumerate(reader.pages):
                    texto = page.extract_text()

                    if not texto or len(texto) < MIN_CHUNK_LENGTH:
                        continue

                    chunks = dividir_en_chunks(texto)

                    for chunk_idx, chunk_info in enumerate(chunks):
                        chunk_text = chunk_info['text'] if isinstance(chunk_info, dict) else chunk_info
                        chunk_header = chunk_info.get('header', '') if isinstance(chunk_info, dict) else ''

                        id_doc = f"{archivo}_pag{i}_chunk{chunk_idx}"

                        metadata = {
                            "source": archivo,
                            "page": i,
                            "chunk": chunk_idx,
                            "total_chunks_in_page": len(chunks),
                            "format": "plain_text",
                            "section_header": chunk_header
                        }

                        if USAR_CONTEXTUAL_RETRIEVAL:
                            contexto_sit = generar_contexto_situacional(chunk_text, texto_base_doc, idioma_doc)
                            chunk_text_con_contexto = (contexto_sit + chunk_text).strip()
                        else:
                            chunk_text_con_contexto = chunk_text

                        if _indexar_chunk(id_doc, chunk_text_con_contexto, chunk_text_con_contexto, metadata, collection):
                            total_chunks += 1

            if imagenes_pdf:
                if not silent:
                    ui.debug("  describing and indexing images...")
                for num_pag, pagina_imagenes in imagenes_pdf.items():
                    for img_idx, img_info in enumerate(pagina_imagenes):
                        caption = img_info.get("caption", "")
                        descripcion = describir_imagen_con_llm(img_info["image_bytes"], caption=caption, idioma_doc=idioma_doc)
                        if not descripcion:
                            continue

                        contexto_img = generar_contexto_situacional(descripcion, texto_base_doc, idioma_doc)
                        descripcion_enriquecida = (contexto_img + descripcion).strip()

                        img_chunk_idx = _IMAGEN_CHUNK_OFFSET + img_idx
                        id_img = f"{archivo}_pag{num_pag}_chunk{img_chunk_idx}"

                        metadata_img: Dict[str, Any] = {
                            "source": archivo,
                            "page": num_pag,
                            "chunk": img_chunk_idx,
                            "format": "image",
                            "section_header": "",
                            "image_width": img_info["width"],
                            "image_height": img_info["height"],
                        }

                        if _indexar_chunk(id_img, descripcion_enriquecida, descripcion_enriquecida, metadata_img, collection):
                            total_chunks += 1

        except Exception as e:
            logging.error(f"Error processing {archivo}: {e}")
            if not silent:
                ui.error(f"error in {archivo}: {e}")

    if not silent:
        ui.pipeline_stop()
    return total_chunks


def obtener_documentos_indexados(collection: chromadb.Collection) -> List[str]:
    """List unique document names (``source``) in the collection.

    Args:
        collection: ChromaDB collection to inspect.

    Returns:
        Sorted list of document filenames.
    """
    try:
        all_metadata = collection.get(include=['metadatas'])
        documentos = set()
        for meta in all_metadata['metadatas']:
            if 'source' in meta:
                documentos.add(meta['source'])
        return sorted(list(documentos))
    except Exception:
        return []


# ─────────────────────────────────────────────



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

