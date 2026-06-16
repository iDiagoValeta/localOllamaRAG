"""Auxiliary implementation module for rag.chat_pdfs.

This module keeps business logic split out of the public facade. Runtime
configuration stays owned by rag.chat_pdfs and is read lazily through ``cfg``
(a live reference to that module), so web/API toggles and test monkeypatches
are observed without any per-call synchronization.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
import ollama
from pypdf import PdfReader

from rag.cli.display import ui
from rag.engine.runtime import get_runtime

cfg = get_runtime()
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
        text_to_embed = f"{cfg.EMBED_PREFIX_DOC}{chunk_text}"

        try:
            response = ollama.embeddings(model=cfg.MODELO_EMBEDDING, prompt=text_to_embed)
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
                text_to_embed = f"{cfg.EMBED_PREFIX_DOC}{chunk_text[:1000]}"
                try:
                    response = ollama.embeddings(model=cfg.MODELO_EMBEDDING, prompt=text_to_embed)
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

    def _preparar_texto_base_doc(textos_paginas: List[str]) -> str:
        """Build the document-level sample used by contextual retrieval."""
        partes: List[str] = []
        caracteres = 0

        for texto in textos_paginas:
            if not texto:
                continue
            restante = cfg.CONTEXTUAL_DOC_CHARS - caracteres
            if restante <= 0:
                break
            parte = texto[:restante]
            partes.append(parte)
            caracteres += len(parte)

        return "\n\n".join(partes)[:cfg.CONTEXTUAL_DOC_CHARS]

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
            if cfg.USAR_EMBEDDINGS_IMAGEN:
                imagenes_pdf = cfg.extraer_imagenes_pdf(ruta_pdf)
                n_imgs_total = sum(len(v) for v in imagenes_pdf.values())
                if n_imgs_total > 0 and not silent:
                    ui.debug(f"  {n_imgs_total} images found across {len(imagenes_pdf)} page(s)")

            try:
                page_chunks = cfg.pymupdf4llm.to_markdown(ruta_pdf, page_chunks=True)

                _textos_paginas = [p.get('text', '') for p in page_chunks[:10]]
                texto_base_doc = _preparar_texto_base_doc(_textos_paginas)
                idioma_doc = cfg._detectar_idioma(texto_base_doc)

                for page_info in page_chunks:
                    # pymupdf4llm reports page numbers 1-based; normalize to the
                    # 0-based convention used by the pypdf fallback and image
                    # extraction. Downstream (citations, debug, viewer) adds +1 to
                    # recover the physical sheet number.
                    i = page_info['metadata']['page'] - 1
                    texto = page_info['text']

                    if not texto or len(texto) < cfg.MIN_CHUNK_LENGTH:
                        continue

                    chunks = cfg.dividir_en_chunks(texto)

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

                        if cfg.USAR_CONTEXTUAL_RETRIEVAL:
                            contexto_sit = cfg.generar_contexto_situacional(chunk_text, texto_base_doc, idioma_doc)
                            chunk_text_con_contexto = (contexto_sit + chunk_text).strip()
                        else:
                            chunk_text_con_contexto = chunk_text

                        if _indexar_chunk(id_doc, chunk_text_con_contexto, chunk_text_con_contexto, metadata, collection):
                            total_chunks += 1

            except Exception as e:
                logging.error(f"Error with pymupdf4llm on {archivo}: {e}, using pypdf fallback")
                usar_pypdf_fallback = True

            if usar_pypdf_fallback:
                reader = PdfReader(ruta_pdf)

                _textos_paginas = [(p.extract_text() or "") for p in reader.pages[:10]]
                texto_base_doc = _preparar_texto_base_doc(_textos_paginas)
                idioma_doc = cfg._detectar_idioma(texto_base_doc)

                for i, page in enumerate(reader.pages):
                    texto = page.extract_text()

                    if not texto or len(texto) < cfg.MIN_CHUNK_LENGTH:
                        continue

                    chunks = cfg.dividir_en_chunks(texto)

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

                        if cfg.USAR_CONTEXTUAL_RETRIEVAL:
                            contexto_sit = cfg.generar_contexto_situacional(chunk_text, texto_base_doc, idioma_doc)
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
                        descripcion = cfg.describir_imagen_con_llm(img_info["image_bytes"], caption=caption, idioma_doc=idioma_doc)
                        if not descripcion:
                            continue

                        contexto_img = cfg.generar_contexto_situacional(descripcion, texto_base_doc, idioma_doc)
                        descripcion_enriquecida = (contexto_img + descripcion).strip()

                        img_chunk_idx = cfg._IMAGEN_CHUNK_OFFSET + img_idx
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




