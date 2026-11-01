"""Indexing frees Ollama's models before each file's extraction (issue #582).

The observed shape: two PDFs into a fresh store, contextual retrieval on. The
first file's enrichment left ``gemma4:e4b`` resident for its 120 s keep-alive,
and MinerU, starting on the second file next to it and the embedder, failed
with a CUDA out-of-memory. The test records the order of the unload and the
per-file run, since an unload after extraction would be as useless as none.

Lives with the loose tests because it imports the engine package.
"""

from types import SimpleNamespace

import rag.chat_pdfs  # noqa: F401  (loads the engine package in its supported order)
from monkeygrab.config.app_config import AppConfig
from rag.engine import indexing, wiring


class _Store:
    def read_fingerprint(self):
        return None

    def write_fingerprint(self, value):
        pass


def test_every_file_is_extracted_with_ollama_models_released(tmp_path, monkeypatch):
    events = []

    class _RecordingIndexCorpus:
        def __init__(self, *_a, **_kw):
            pass

        def run(self, _path, filename):
            events.append(("run", filename))
            return SimpleNamespace(chunks_indexed=1)

    config = AppConfig().with_overrides(**{"flags.usar_embeddings_imagen": False})
    monkeypatch.setattr(wiring, "app_config_from_runtime", lambda: config)
    monkeypatch.setattr(wiring, "embedder", lambda _config: None)
    monkeypatch.setattr(wiring, "release_ollama_models", lambda _config: events.append(("unload", None)))
    monkeypatch.setattr(indexing, "IndexCorpus", _RecordingIndexCorpus)
    monkeypatch.setattr(indexing, "build_extractor", lambda _config: None)
    monkeypatch.setattr(indexing, "_extractor_version", lambda: None)
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(b"%PDF-1.4")

    indexing.indexar_documentos(str(tmp_path), _Store(), silent=True)

    assert events == [("unload", None), ("run", "a.pdf"), ("unload", None), ("run", "b.pdf")]
