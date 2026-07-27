"""Indexing entry point for the CLI and the web app.

Wires the extraction, embedding and storage adapters selected by the stack
configuration and runs ``monkeygrab.application.index_corpus.IndexCorpus``
over a folder of PDFs. Chunking, contextual enrichment and image captioning
all live in the use case; what remains here is folder iteration and progress
reporting.
"""

import logging
import os
from typing import List, Optional

import chromadb

from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
from monkeygrab.adapters.extraction.pymupdf_image_extractor import PymupdfImageExtractor
from monkeygrab.application.index_corpus import IndexCorpus
from monkeygrab.composition import build_stack
from monkeygrab.config.app_config import AppConfig
from monkeygrab.config.stack import EXTRACTOR_MINERU, VECTOR_STORE_CHROMA
from rag.cli.display import ui
from rag.engine import wiring


def _build_image_extractor(config: AppConfig):
    """Return the image extractor matching the configured PDF extractor."""
    if config.stack.extractor == EXTRACTOR_MINERU:
        from monkeygrab.adapters.extraction.mineru_extractor import MineruImageExtractor

        return MineruImageExtractor()
    return PymupdfImageExtractor(config.chunking)


def indexar_documentos(
    carpeta: str,
    collection: chromadb.Collection,
    solo_archivos: Optional[List[str]] = None,
    silent: bool = False,
    progress_callback=None,
) -> int:
    """Index PDFs from a folder via ``IndexCorpus`` and the configured stack.

    Backends come from ``build_stack(AppConfig)`` (``PDF_EXTRACTOR``,
    ``VECTOR_STORE``, ``EMBEDDER``). With no env overrides that is the
    historic pymupdf + Ollama + Chroma path. The ``collection`` argument is
    still required for API compatibility with CLI/web/eval: when the stack
    uses Chroma, writes go through that exact collection object.

    Args:
        carpeta: Path to the folder containing PDF files.
        collection: ChromaDB collection to index into (Chroma stack only).
        solo_archivos: If set, only index these specific filenames
            (for incremental adds without full re-index).
        silent: Suppress all terminal output (for background/web use).
        progress_callback: Called with ``{"file", "file_index",
            "total_files"}`` at the start of each file.

    Returns:
        Total number of chunks successfully indexed.
    """
    os.makedirs(carpeta, exist_ok=True)
    archivos_pdf = [f for f in os.listdir(carpeta) if f.endswith(".pdf")]
    if solo_archivos is not None:
        archivos_pdf = [f for f in archivos_pdf if f in solo_archivos]

    if not archivos_pdf:
        if not silent:
            ui.warning("No PDF files found in folder")
        return 0

    if not silent:
        ui.pipeline_start("Indexing documents...")

    config = wiring.app_config_from_runtime()
    stack = build_stack(config)

    # Callers open this collection before indexing; reuse it so their handle
    # sees the writes. FAISS ignores it (paths already carry the stack slug).
    vector_store = stack.vector_store
    if config.stack.vector_store == VECTOR_STORE_CHROMA:
        vector_store = wiring.vector_store(collection)

    ollama = config.models.ollama
    contextual_model = None
    if config.flags.usar_contextual_retrieval:
        contextual_model = OllamaChatModel(
            config.models.contextual,
            num_ctx=ollama.contextual_num_ctx,
            keep_alive=ollama.keep_alive,
            request_timeout=ollama.request_timeout,
            generate_retries=ollama.generate_retries,
            generate_retry_delay=ollama.generate_retry_delay,
            options={"temperature": 0.1, "num_predict": 250},
        )

    image_extractor = None
    ocr_chat_model = None
    if config.flags.usar_embeddings_imagen:
        image_extractor = _build_image_extractor(config)
        # Multimodal embedders satisfy ImageEmbedder; OCR would only fight them
        # for the same 8 GB VRAM. Text-only stacks still need the vision role.
        if not config.stack.is_multimodal:
            ocr_chat_model = OllamaChatModel(
                config.models.ocr,
                num_ctx=ollama.ocr_num_ctx,
                keep_alive=ollama.keep_alive,
                request_timeout=ollama.request_timeout,
                generate_retries=ollama.generate_retries,
                generate_retry_delay=ollama.generate_retry_delay,
                options={"temperature": 0.1, "num_predict": 2000},
            )

    use_case = IndexCorpus(
        stack.extractor,
        stack.embedder,
        vector_store,
        config,
        contextual_model=contextual_model,
        image_extractor=image_extractor,
        ocr_chat_model=ocr_chat_model,
    )

    total_chunks = 0
    for idx, archivo in enumerate(archivos_pdf):
        if progress_callback:
            try:
                progress_callback(
                    {"file": archivo, "file_index": idx + 1, "total_files": len(archivos_pdf)}
                )
            except Exception:
                pass
        if not silent:
            ui.pipeline_update(f"Processing: {archivo}")

        ruta_pdf = os.path.join(carpeta, archivo)
        try:
            result = use_case.run(ruta_pdf, archivo)
            total_chunks += result.chunks_indexed
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
        all_metadata = collection.get(include=["metadatas"])
        documentos = set()
        for meta in all_metadata["metadatas"]:
            if "source" in meta:
                documentos.add(meta["source"])
        return sorted(list(documentos))
    except Exception:
        return []
