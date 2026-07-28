"""Domain entities -- pure data, zero infrastructure imports.

No module here may import faiss, ollama, PIL, torch, requests, or any
other infrastructure library -- enforced by
``tests/unit/test_architecture_boundaries.py``. This is what makes these
entities safe to reuse from any adapter, use case, or interface layer without
pulling in a specific vector store, model runtime, or PDF library.
"""

from monkeygrab.domain.chunk import Chunk
from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.extracted_image import ExtractedImage
from monkeygrab.domain.extracted_page import ExtractedPage
from monkeygrab.domain.fragment import Fragment
from monkeygrab.domain.generation_chunk import GenerationChunk

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ExtractedImage",
    "ExtractedPage",
    "Fragment",
    "GenerationChunk",
]
