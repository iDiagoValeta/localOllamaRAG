"""Composition root for the fixed multimodal stack: MinerU, embeddinggemma-2 and FAISS.

Every builder takes the immutable ``AppConfig`` and returns the port the use
cases consume. ``rag.engine.wiring`` derives that config from the mutable
runtime and builds the rest of what the use cases need: a chat model per role,
and the reranker and BM25 index it caches between queries.
"""

from pathlib import Path
from typing import NamedTuple

from monkeygrab.config.app_config import AppConfig
from monkeygrab.ports.embedder import Embedder
from monkeygrab.ports.image_extractor import ImageExtractor
from monkeygrab.ports.pdf_extractor import PdfExtractor
from monkeygrab.ports.vector_store import VectorStore


class Stack(NamedTuple):
    extractor: PdfExtractor
    vector_store: VectorStore
    embedder: Embedder


def isolated_python() -> str:
    """Return the interpreter of the isolated MinerU environment.

    Returns:
        Path to ``.venv-mineru``'s Python, Windows layout first.

    Raises:
        FileNotFoundError: Neither conventional layout exists.
    """
    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / ".venv-mineru" / "Scripts" / "python.exe",
        root / ".venv-mineru" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "The multimodal pipeline needs the isolated MinerU environment at "
        f"{candidates[0]} or {candidates[1]}"
    )


def build_extractor(config: AppConfig) -> PdfExtractor:
    del config
    from monkeygrab.adapters.extraction.by_suffix_extractor import BySuffixExtractor
    from monkeygrab.adapters.extraction.mineru_extractor import MineruExtractor
    from monkeygrab.adapters.extraction.text_extractor import TextFileExtractor

    return BySuffixExtractor(MineruExtractor(), TextFileExtractor())


def build_image_extractor(config: AppConfig) -> ImageExtractor:
    """Build the image extractor used when image embeddings are enabled.

    Args:
        config: Current config; the current extractor has nothing to read from
            it yet, but it takes the same argument as the other builders so
            callers do not special-case it.

    Returns:
        MinerU's image extractor, restricted to PDFs.
    """
    del config
    from monkeygrab.adapters.extraction.by_suffix_extractor import PdfOnlyImageExtractor
    from monkeygrab.adapters.extraction.mineru_extractor import MineruImageExtractor

    return PdfOnlyImageExtractor(MineruImageExtractor())


def build_vector_store(config: AppConfig) -> VectorStore:
    from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore

    return FaissVectorStore(config.paths)


def build_embedder(config: AppConfig) -> Embedder:
    """Build the embeddinggemma-2 client for the configured llama-server.

    Args:
        config: Runtime configuration naming the embedder endpoint.

    Returns:
        An HTTP embeddinggemma-2 client.
    """
    from monkeygrab.adapters.embedding.llama_server_embedder import LlamaServerEmbedder

    embedding = config.models.embedding
    return LlamaServerEmbedder(
        embedding.base_url,
        dimensions=embedding.dimensions,
        timeout_s=embedding.timeout,
    )


def build_stack(config: AppConfig) -> Stack:
    return Stack(
        extractor=build_extractor(config),
        vector_store=build_vector_store(config),
        embedder=build_embedder(config),
    )
