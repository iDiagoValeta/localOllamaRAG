"""Domain entities -- pure data, zero infrastructure imports.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- chunk_metadata.py   ChunkMetadata   -- position/format of a stored chunk
#  +-- chunk.py            Chunk           -- text unit ready to embed and store
#  +-- extracted_page.py   ExtractedPage   -- one page of raw PDF-extracted text
#  +-- fragment.py         Fragment        -- a retrieved/scored chunk
#
# ─────────────────────────────────────────────

No module here may import chromadb, ollama, fitz, torch, requests, or any
other infrastructure library -- enforced by
``tests/unit/test_architecture_boundaries.py``. This is what makes these
entities safe to reuse from any adapter, use case, or interface layer without
pulling in a specific vector store, model runtime, or PDF library.
"""

from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.extracted_page import ExtractedPage
from monkeygrab.domain.fragment import Fragment

__all__ = ["Chunk", "ChunkMetadata", "ExtractedPage", "Fragment"]
