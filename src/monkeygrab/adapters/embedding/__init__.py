"""Embedding adapters -- Embedder implementations.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- ollama_embedder.py    OllamaEmbedder    -- ollama.embeddings
#  +-- jina_clip_embedder.py JinaClipEmbedder  -- text+image, via out-of-process worker
#  +-- jina_clip_worker.py   (not an adapter)  -- runs only under .venv-mineru; see its docstring
#
# ─────────────────────────────────────────────
"""
