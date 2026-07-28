"""Indexing entry point for the CLI and the web app.

Wires the fixed extraction, embedding and storage adapters and runs
``monkeygrab.application.index_corpus.IndexCorpus`` over a folder of PDFs.
Chunking, contextual enrichment and direct image embedding live in the use
case; what remains here is folder iteration and progress reporting.
"""

import logging
import os
from typing import List, Optional

from monkeygrab.adapters.chat.ollama_chat import OllamaChatModel
from monkeygrab.adapters.extraction.mineru_extractor import MineruImageExtractor
from monkeygrab.application.index_corpus import IndexCorpus
from monkeygrab.composition import build_extractor
from monkeygrab.config.app_config import AppConfig
from monkeygrab.ports.vector_store import VectorStore
from rag.cli.display import ui
from rag.engine import wiring


def _build_image_extractor(config: AppConfig):
    del config
    return MineruImageExtractor()


def indexar_documentos(
    carpeta: str,
    collection: VectorStore,
    solo_archivos: Optional[List[str]] = None,
    silent: bool = False,
    progress_callback=None,
) -> int:
    """Index PDFs from a folder via ``IndexCorpus`` and the configured stack.

    MinerU, jina-clip-v2 and FAISS are the only supported backends.

    Args:
        carpeta: Path to the folder containing PDF files.
        collection: Active FAISS vector store.
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
    if config.flags.usar_embeddings_imagen:
        image_extractor = _build_image_extractor(config)

    use_case = IndexCorpus(
        build_extractor(config),
        wiring.embedder(config),
        collection,
        config,
        contextual_model=contextual_model,
        image_extractor=image_extractor,
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


def obtener_documentos_indexados(collection: VectorStore) -> List[str]:
    """List unique document names (``source``) in the collection.

    Args:
        collection: FAISS store to inspect.

    Returns:
        Sorted list of document filenames.
    """
    try:
        return sorted({
            fragment.metadata.source
            for fragment in collection.get_page(None, 0)
            if fragment.metadata.source
        })
    except Exception:
        return []
