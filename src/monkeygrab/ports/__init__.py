"""Ports -- Protocols the application layer will depend on, adapters implement.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- pdf_extractor.py   PdfExtractor   -- PDF -> per-page raw text
#  +-- embedder.py        Embedder       -- text -> embedding vector
#  +-- vector_store.py    VectorStore    -- add/query/get/count over chunks
#  +-- lexical_index.py   LexicalIndex   -- BM25-style text search
#  +-- reranker.py        Reranker       -- Cross-Encoder-style re-scoring
#  +-- chat_model.py      ChatModel      -- Ollama-style generate/stream
#
# ─────────────────────────────────────────────

Every Protocol here has exactly one job and takes/returns only domain types
(``monkeygrab.domain``) and primitives -- never ``chromadb.Collection``,
``ollama.Client``, or any other infrastructure type. This is what lets an
adapter be swapped (Chroma <-> FAISS, pymupdf <-> MinerU, Ollama <-> anything
else) by rewiring dependency injection, not by touching a use case.

Hard-fail policy (project-wide, see docs/design/2026-07-26-monkeygrab-v2.md
section 3 "Politica de fallos"): every adapter implementing a port here
raises on failure instead of degrading -- no silent reranker CUDA-to-CPU-to
-disabled, no silent pymupdf4llm-to-pypdf fallback, no silent RECOMP-to-raw
-context fallback. A caller that wants a fallback chain builds it explicitly
out of two ports and a decision, not by one adapter secretly trying a second
strategy. Each Protocol's docstring restates this for its own failure modes.
"""

from monkeygrab.ports.chat_model import ChatModel
from monkeygrab.ports.embedder import Embedder
from monkeygrab.ports.lexical_index import LexicalIndex
from monkeygrab.ports.pdf_extractor import PdfExtractor
from monkeygrab.ports.reranker import Reranker
from monkeygrab.ports.vector_store import VectorStore

__all__ = [
    "ChatModel",
    "Embedder",
    "LexicalIndex",
    "PdfExtractor",
    "Reranker",
    "VectorStore",
]
