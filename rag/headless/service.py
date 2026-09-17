"""What the headless routes do, without Flask: index a folder, report a store, answer.

The per-request ``AppConfig`` only selects the store: ``wiring.vector_store(config)``
is keyed by the request's paths. Models, flags and every other section come from
the process configuration (``wiring.app_config_from_runtime()``: environment plus
``rag.chat_pdfs`` defaults), because the engine entry points read it themselves.
The headless never loads ``settings.json`` nor mutates those globals, so the
environment is the whole configuration of a headless process.
"""
import contextlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

from monkeygrab.config.app_config import AppConfig

import rag.chat_pdfs  # noqa: F401 -- import before rag.engine.wiring, see its module docstring
from rag.engine import wiring
from rag.engine.generation import (
    _preparar_mensaje_usuario_rag,
    generar_tokens_respuesta,
    preparar_fragmentos_para_generacion,
)
from rag.engine.indexing import (
    index_fingerprint_mismatch,
    indexar_documentos,
    listar_documentos,
    obtener_documentos_indexados,
)
from rag.engine.retrieval import realizar_busqueda_hibrida
from rag.engine.sources import format_sources


class StoreConflict(ValueError):
    """The same store id arrived with different paths within this process."""


class StoreBusy(RuntimeError):
    """An index run is already in progress for this store id."""


class QuestionTooShort(ValueError):
    """Below ``retrieval.min_question_length``; nothing is retrieved or generated."""


@dataclass(frozen=True)
class StorePaths:
    """Where a store's documents and its index live. Both absolute."""

    docs_folder: str
    data_dir: str


class StoreRegistry:
    """Binds each store id to one pair of paths for the life of the process.

    The id is also ``wiring``'s cache key by way of the paths: letting the
    same id point at two folders would let one caller's query read another's
    index, which is the #57 class of bug in a new coat.
    """

    def __init__(self) -> None:
        self._paths: Dict[str, StorePaths] = {}
        self._lock = threading.Lock()
        self._index_locks: Dict[str, threading.Lock] = {}

    def resolve(self, store_id: str, docs_folder: str, data_dir: str) -> StorePaths:
        paths = StorePaths(os.path.abspath(docs_folder), os.path.abspath(data_dir))
        with self._lock:
            known = self._paths.get(store_id)
            if known is None:
                self._paths[store_id] = paths
                return paths
        if known != paths:
            raise StoreConflict(
                f"store {store_id!r} is bound to {known.docs_folder} / {known.data_dir}, "
                f"not {paths.docs_folder} / {paths.data_dir}"
            )
        return known

    @contextlib.contextmanager
    def indexing(self, store_id: str) -> Iterator[None]:
        """Serialize index runs per store id; a second one fails instead of queuing.

        A blocking lock would queue the second request behind the first one's
        full index run (minutes, for a cold store); the caller wants to know
        now that one is already in flight, not wait to find out.

        Raises:
            StoreBusy: An index run for this store id is already in progress.
        """
        with self._lock:
            lock = self._index_locks.setdefault(store_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise StoreBusy(f"store {store_id!r} is already being indexed")
        try:
            yield
        finally:
            lock.release()


def config_for(paths: StorePaths) -> AppConfig:
    """The process's runtime config with this store's paths substituted.

    Starts from ``wiring.app_config_from_runtime()`` (environment plus the
    live ``rag.chat_pdfs`` globals) rather than a bare ``AppConfig.from_env()``,
    so the store opens under the same models, flags and chunking the engine
    entry points themselves read (issue #262). Only the two path fields differ.
    """
    return wiring.app_config_from_runtime().with_overrides(
        **{"paths.docs_folder": paths.docs_folder, "paths.data_dir": paths.data_dir}
    )


def index_store(paths: StorePaths) -> Dict[str, Any]:
    """Index every document in ``docs_folder`` that the store does not hold yet.

    A store with nothing in it gets a full run (which also writes the index
    fingerprint); a store with documents gets only the pending file names,
    matching what the web's upload path does.

    Returns:
        Counts, the fingerprint on disk afterwards and wall seconds.

    Raises:
        RuntimeError: From ``indexar_documentos`` when every file failed, or
            raised here when some (but not all) pending files did not land in
            the store -- ``indexar_documentos`` only raises on total failure,
            but a corpus version with a missing file is not a valid version.
    """
    started = time.perf_counter()
    config = config_for(paths)
    store = wiring.vector_store(config)
    present = set(obtener_documentos_indexados(store))
    available = listar_documentos(paths.docs_folder)
    pending = [name for name in available if name not in present]
    chunks = 0
    failed: list = []
    if pending:
        chunks = indexar_documentos(
            paths.docs_folder, store, solo_archivos=None if not present else pending, silent=True
        )
        after = set(obtener_documentos_indexados(store))
        failed = [name for name in pending if name not in after]
        if failed:
            raise RuntimeError(
                f"{len(failed)} document(s) failed to index or produced no chunks: "
                f"{', '.join(failed)}"
            )
    return {
        "documents_indexed": len(pending) - len(failed),
        "documents_skipped": len(available) - len(pending),
        "chunks_indexed": chunks,
        "fingerprint": store.read_fingerprint(),
        "seconds": round(time.perf_counter() - started, 3),
    }


def status_store(paths: StorePaths) -> Dict[str, Any]:
    """Describe a store without building it: a never-indexed store reads as absent."""
    config = config_for(paths)
    if not os.path.isdir(config.paths.path_db):
        return {"exists": False, "documents": [], "chunks_total": 0, "fingerprint": None, "stale": False}
    store = wiring.vector_store(config)
    return {
        "exists": True,
        "documents": obtener_documentos_indexados(store),
        "chunks_total": store.count(),
        "fingerprint": store.read_fingerprint(),
        "stale": index_fingerprint_mismatch(store),
    }


def answer_stream(paths: StorePaths, question: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Retrieve, prepare evidence, then stream the generated answer.

    Yields:
        ``("no_results", {})`` when nothing relevant was found; otherwise
        ``("token", {"token": str})`` per streamed piece and one final
        ``("done", {"done": True, "sources": [...], "metrics": {...}})``.

    Raises:
        QuestionTooShort: Before touching any model.
        Exception: Anything retrieval or generation raises, unchanged.
    """
    question = question.strip()
    config = config_for(paths)
    if len(question) < config.retrieval.min_question_length:
        raise QuestionTooShort(
            f"question shorter than {config.retrieval.min_question_length} characters"
        )
    store = wiring.vector_store(config)
    ranked, _best, metrics = realizar_busqueda_hibrida(question, store)
    if not ranked:
        yield "no_results", {}
        return
    finales, context_metrics = preparar_fragmentos_para_generacion(ranked, store)
    if not finales:
        yield "no_results", {}
        return
    sources = format_sources(finales)
    message = _preparar_mensaje_usuario_rag(question, finales)
    stats: Dict[str, Any] = {}
    for token in generar_tokens_respuesta(message, stats):
        yield "token", {"token": token}
    yield "done", {
        "done": True,
        "sources": sources,
        "metrics": {**metrics, "fase_contexto": context_metrics, "generation": stats},
    }
