"""Domain entities -- pure data, zero infrastructure imports.

No module here may import faiss, ollama, PIL, torch, requests, or any
other infrastructure library -- enforced by
``tests/unit/test_architecture_boundaries.py``. This is what makes these
entities safe to reuse from any adapter, use case, or interface layer without
pulling in a specific vector store, model runtime, or PDF library.

No names are re-exported here: import each entity from its own module
(``monkeygrab.domain.chunk`` and so on), so the package surface matches the
files a reader can see.
"""
