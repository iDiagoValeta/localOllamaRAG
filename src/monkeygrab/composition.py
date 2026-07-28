"""Single composition root for the multimodal pipeline."""

from pathlib import Path
from typing import Any, NamedTuple

from monkeygrab.config.app_config import AppConfig


class Stack(NamedTuple):
    extractor: Any
    vector_store: Any
    embedder: Any


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
        f"{candidates[0]}"
    )


def build_extractor(config: AppConfig) -> Any:
    del config
    from monkeygrab.adapters.extraction.mineru_extractor import MineruExtractor

    return MineruExtractor()


def build_vector_store(config: AppConfig) -> Any:
    from monkeygrab.adapters.vectorstore.faiss_store import FaissVectorStore

    return FaissVectorStore(config.paths)


def build_embedder(config: AppConfig) -> Any:
    del config
    from monkeygrab.adapters.embedding.jina_clip_embedder import JinaClipEmbedder

    return JinaClipEmbedder(_isolated_python())


def build_stack(config: AppConfig) -> Stack:
    return Stack(
        extractor=build_extractor(config),
        vector_store=build_vector_store(config),
        embedder=build_embedder(config),
    )
