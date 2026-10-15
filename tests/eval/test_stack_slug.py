"""The report's stack name follows the active embedder.

Two gate runs over different embedders used to write artifacts under the same
``mineru-jina_clip-faiss`` name, so the file name alone could not tell them
apart.
"""

import rag.chat_pdfs as rag
from tests.eval import run_eval


def test_stack_slug_names_jina_by_default(monkeypatch):
    monkeypatch.setattr(rag, "EMBEDDER", "jina")

    assert run_eval._stack_slug() == "mineru-jina_clip-faiss"


def test_stack_slug_names_embeddinggemma_when_selected(monkeypatch):
    monkeypatch.setattr(rag, "EMBEDDER", "gemma2")

    assert run_eval._stack_slug() == "mineru-eg2-faiss"
