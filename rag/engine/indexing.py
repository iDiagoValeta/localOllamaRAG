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
from monkeygrab.application.index_fingerprint import compute_index_fingerprint, fingerprint_is_stale
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
            base_url=ollama.base_url,
        )

    image_extractor = None
    if config.flags.usar_embeddings_imagen:
        image_extractor = _build_image_extractor(config)

    image_describer = None
    if config.flags.usar_descripcion_imagen:
        image_describer = OllamaChatModel(
            config.models.chat,
            num_ctx=ollama.query_num_ctx,
            keep_alive=ollama.keep_alive,
            request_timeout=ollama.request_timeout,
            generate_retries=ollama.generate_retries,
            generate_retry_delay=ollama.generate_retry_delay,
            options={"temperature": 0.1, "num_predict": 400},
            base_url=ollama.base_url,
        )

    use_case = IndexCorpus(
        build_extractor(config),
        wiring.embedder(config),
        collection,
        config,
        contextual_model=contextual_model,
        image_extractor=image_extractor,
        image_describer=image_describer,
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

    # Only a full-folder run (solo_archivos=None) can vouch for the *entire*
    # store having been produced under `config`'s recipe -- every call site
    # only makes that call against an already-empty collection (fresh start,
    # or cleared first by /reindex), which is what FaissVectorStore.add's
    # duplicate-id guard would otherwise reject. An incremental add
    # (solo_archivos given) leaves whatever the rest of the store already
    # recorded untouched: overwriting it here would let a partial add
    # silently launder away a real mismatch from an earlier config change,
    # exactly what index_fingerprint_mismatch exists to catch.
    #
    # Written even when some individual files failed above (per-file
    # exceptions are caught and logged, not re-raised): the fingerprint
    # asserts the *recipe* whatever ended up stored was built under, not that
    # every requested file is present -- unlike run_eval.ensure_indexed, whose
    # write-after-verify exists to guarantee a specific set of gold-case
    # papers landed, a different property. Every chunk actually added this
    # pass did go through `config`, so the recipe claim holds regardless of
    # which files errored; a fully failed run leaves an empty store, which the
    # `collection.count() == 0` branch in the CLI/web re-attempts on next
    # launch without ever consulting this fingerprint.
    if solo_archivos is None:
        collection.write_fingerprint(compute_index_fingerprint(config))

    if not silent:
        ui.pipeline_stop()
    return total_chunks


def index_fingerprint_mismatch(collection: VectorStore) -> bool:
    """Whether the store's recorded fingerprint disagrees with the config in force.

    Detection only -- callers decide what to do with the answer (the product
    warns and leaves reindexing to the user; see rag/cli/app.py and
    rag/web/app.py). A store that has never recorded a fingerprint (every
    index built before this feature existed) reads as *unknown*, not stale --
    see ``fingerprint_is_stale``'s docstring.

    This is a diagnostic, not a pipeline step: unlike an adapter (see the
    hard-fail policy in .claude/CLAUDE.md section 1 rule 8), it must never be
    able to take startup down with it. A locked or unreadable sidecar file
    (e.g. an antivirus holding it open on Windows, the platform the packaged
    .exe ships on) reads as "cannot tell" rather than aborting the CLI before
    the prompt or the web app's /api/init -- same fallback shape as
    ``obtener_documentos_indexados`` below.

    Args:
        collection: FAISS store to check.

    Returns:
        True only when the store recorded a fingerprint that actively
        disagrees with the current configuration. False if the check itself
        could not be completed.
    """
    try:
        config = wiring.app_config_from_runtime()
        expected = compute_index_fingerprint(config)
        return fingerprint_is_stale(collection.read_fingerprint(), expected)
    except Exception:
        return False


def obtener_documentos_indexados(collection: VectorStore) -> List[str]:
    """List unique document names (``source``) in the collection.

    Args:
        collection: FAISS store to inspect.

    Returns:
        Sorted list of document filenames.
    """
    try:
        return sorted(
            {
                fragment.metadata.source
                for fragment in collection.get_page(None, 0)
                if fragment.metadata.source
            }
        )
    except Exception:
        return []
