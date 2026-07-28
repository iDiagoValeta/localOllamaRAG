"""Ports -- Protocols the application layer will depend on, adapters implement.

Every Protocol here has exactly one job and takes/returns only domain types
(``monkeygrab.domain``) and primitives -- never ``faiss.Index``,
``ollama.Client``, or any other infrastructure type. This is what lets an
adapter be replaced by rewiring dependency injection, not by touching a use case.

Hard-fail policy (project-wide, see docs/design/2026-07-26-monkeygrab-v2.md
section 3 "Politica de fallos"): every adapter implementing a port here
raises on failure instead of degrading -- no silent reranker CUDA-to-CPU-to
-disabled, no silent extraction fallback, no silent RECOMP-to-raw
-context fallback. A caller that wants a fallback chain builds it explicitly
out of two ports and a decision, not by one adapter secretly trying a second
strategy. Each Protocol's docstring restates this for its own failure modes.
"""

from monkeygrab.ports.chat_model import ChatModel
from monkeygrab.ports.embedder import Embedder
from monkeygrab.ports.image_embedder import ImageEmbedder
from monkeygrab.ports.image_extractor import ImageExtractor
from monkeygrab.ports.lexical_index import LexicalIndex
from monkeygrab.ports.model_unloader import ModelUnloader
from monkeygrab.ports.pdf_extractor import PdfExtractor
from monkeygrab.ports.reranker import Reranker
from monkeygrab.ports.vector_store import VectorStore

__all__ = [
    "ChatModel",
    "Embedder",
    "ImageEmbedder",
    "ImageExtractor",
    "LexicalIndex",
    "ModelUnloader",
    "PdfExtractor",
    "Reranker",
    "VectorStore",
]
