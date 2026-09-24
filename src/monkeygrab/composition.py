"""Single composition root for the multimodal pipeline."""

from pathlib import Path
from typing import NamedTuple

from monkeygrab.config.app_config import AppConfig
from monkeygrab.ports.embedder import Embedder
from monkeygrab.ports.pdf_extractor import PdfExtractor
from monkeygrab.ports.vector_store import VectorStore


class Stack(NamedTuple):
    extractor: PdfExtractor
    vector_store: VectorStore
    embedder: Embedder


def _isolated_python() -> str:
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


def build_vector_store(config: AppConfig) -> VectorStore:
    from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore

    return FaissVectorStore(config.paths)


def build_embedder(config: AppConfig) -> Embedder:
    del config
    from monkeygrab.adapters.embedding.jina_clip_embedder import JinaClipEmbedder

    return JinaClipEmbedder(_isolated_python())


def build_stack(config: AppConfig) -> Stack:
    return Stack(
        extractor=build_extractor(config),
        vector_store=build_vector_store(config),
        embedder=build_embedder(config),
    )
