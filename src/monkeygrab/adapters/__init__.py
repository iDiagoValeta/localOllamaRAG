"""Adapters -- infrastructure implementations of the Protocols in monkeygrab.ports.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- extraction/    pymupdf_extractor.py         PymupdfExtractor      -- PdfExtractor
#  +-- extraction/    pymupdf_image_extractor.py   PymupdfImageExtractor -- ImageExtractor
#  +-- embedding/     ollama_embedder.py           OllamaEmbedder        -- Embedder
#  +-- vectorstore/   chroma_store.py              ChromaVectorStore     -- VectorStore
#  +-- lexical/       bm25_index.py                Bm25LexicalIndex      -- LexicalIndex
#  +-- reranking/     cross_encoder_reranker.py    CrossEncoderReranker  -- Reranker
#  +-- chat/          ollama_chat.py               OllamaChatModel       -- ChatModel
#  +-- chat/          ollama_model_unloader.py     OllamaModelUnloader   -- ModelUnloader
#
# ─────────────────────────────────────────────

Each adapter takes its configuration by constructor -- ``AppConfig`` or the
specific sub-config it needs -- and never reads ``rag.chat_pdfs`` globals or
calls ``get_runtime()`` (enforced by
``tests/unit/adapters/test_adapters_do_not_import_rag_chat_pdfs.py``). This is
what makes an adapter swappable by rewiring dependency injection instead of
by changing import order, the exact problem
docs/design/2026-07-26-monkeygrab-v2.md section 1 names in ``rag/chat_pdfs.py``
today.

Hard-fail policy (project-wide, see ``monkeygrab.ports`` and
docs/design/2026-07-26-monkeygrab-v2.md section 3): every adapter here raises
on failure instead of silently degrading. No adapter is imported at package
level -- each submodule imports its own third-party dependency (chromadb,
ollama, rank_bm25, sentence_transformers, pymupdf4llm) directly, so importing
``monkeygrab.adapters`` itself stays cheap; import the specific adapter module
you need.
"""
