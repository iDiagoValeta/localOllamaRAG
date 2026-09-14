"""Headless service functions, with the engine entry points doubled.

The functions under test are the glue between a per-request AppConfig and the
same ``rag.engine`` entry points the web uses; the doubles record what they
were handed so the test can assert the store and folder were the request's,
not the process defaults.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest

from rag.headless import service
from rag.headless.service import QuestionTooShort, StoreConflict, StorePaths, StoreRegistry


class _Store:
    def __init__(self, docs=(), fingerprint="fp1"):
        self.docs = list(docs)
        self.fingerprint = fingerprint

    def count(self):
        return 3 * len(self.docs)

    def read_fingerprint(self):
        return self.fingerprint


def test_registry_binds_a_store_id_to_its_paths_once(tmp_path):
    registry = StoreRegistry()
    first = registry.resolve("cv1", str(tmp_path / "docs"), str(tmp_path / "data"))
    assert first == StorePaths(str(tmp_path / "docs"), str(tmp_path / "data"))
    assert registry.resolve("cv1", str(tmp_path / "docs"), str(tmp_path / "data")) == first
    with pytest.raises(StoreConflict):
        registry.resolve("cv1", str(tmp_path / "other"), str(tmp_path / "data"))


def test_config_for_derives_the_faiss_path_inside_data_dir(tmp_path):
    config = service.config_for(StorePaths(str(tmp_path / "docs"), str(tmp_path / "data")))
    assert config.paths.docs_folder == str(tmp_path / "docs")
    assert config.paths.data_dir == str(tmp_path / "data")
    assert config.paths.path_db == str(tmp_path / "data" / "vector_db" / "docs_jina_clip")


def test_index_store_indexes_only_pending_documents(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.pdf", "b.md"):
        (docs / name).write_bytes(b"x")
    store = _Store(docs=["a.pdf"])
    calls = []
    monkeypatch.setattr(service.wiring, "vector_store", lambda config: store)
    monkeypatch.setattr(service, "obtener_documentos_indexados", lambda s: list(s.docs))

    def fake_indexar(carpeta, collection, solo_archivos=None, silent=False, progress_callback=None):
        calls.append({"carpeta": carpeta, "collection": collection, "solo": solo_archivos, "silent": silent})
        collection.docs.extend(solo_archivos or [])
        return 5

    monkeypatch.setattr(service, "indexar_documentos", fake_indexar)
    result = service.index_store(StorePaths(str(docs), str(tmp_path / "data")))
    assert calls == [{"carpeta": str(docs), "collection": store, "solo": ["b.md"], "silent": True}]
    assert result["documents_indexed"] == 1
    assert result["documents_skipped"] == 1
    assert result["chunks_indexed"] == 5
    assert result["fingerprint"] == "fp1"
    assert result["seconds"] >= 0


def test_index_store_on_an_empty_store_runs_a_full_index(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(b"x")
    store = _Store()
    calls = []
    monkeypatch.setattr(service.wiring, "vector_store", lambda config: store)
    monkeypatch.setattr(service, "obtener_documentos_indexados", lambda s: list(s.docs))
    monkeypatch.setattr(service, "indexar_documentos",
                        lambda carpeta, collection, solo_archivos=None, silent=False, progress_callback=None:
                        calls.append(solo_archivos) or 2)
    service.index_store(StorePaths(str(docs), str(tmp_path / "data")))
    assert calls == [None]


def test_index_store_with_nothing_pending_does_not_call_the_indexer(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(b"x")
    monkeypatch.setattr(service.wiring, "vector_store", lambda config: _Store(docs=["a.pdf"]))
    monkeypatch.setattr(service, "obtener_documentos_indexados", lambda s: list(s.docs))
    monkeypatch.setattr(service, "indexar_documentos", lambda *a, **k: pytest.fail("must not index"))
    result = service.index_store(StorePaths(str(docs), str(tmp_path / "data")))
    assert result["documents_indexed"] == 0 and result["chunks_indexed"] == 0


def test_status_store_reports_documents_chunks_and_staleness(monkeypatch, tmp_path):
    store = _Store(docs=["a.pdf", "b.md"])
    monkeypatch.setattr(service.wiring, "vector_store", lambda config: store)
    monkeypatch.setattr(service, "obtener_documentos_indexados", lambda s: list(s.docs))
    monkeypatch.setattr(service, "index_fingerprint_mismatch", lambda s: True)
    (tmp_path / "data" / "vector_db" / "docs_jina_clip").mkdir(parents=True)
    status = service.status_store(StorePaths(str(tmp_path / "docs"), str(tmp_path / "data")))
    assert status == {
        "exists": True, "documents": ["a.pdf", "b.md"], "chunks_total": 6,
        "fingerprint": "fp1", "stale": True,
    }


def test_status_store_of_a_store_that_was_never_built(monkeypatch, tmp_path):
    monkeypatch.setattr(service.wiring, "vector_store", lambda config: pytest.fail("must not open"))
    status = service.status_store(StorePaths(str(tmp_path / "docs"), str(tmp_path / "data")))
    assert status == {"exists": False, "documents": [], "chunks_total": 0, "fingerprint": None, "stale": False}


def _stub_pipeline(monkeypatch, *, ranked, finales, tokens):
    monkeypatch.setattr(service.wiring, "vector_store", lambda config: _Store())
    monkeypatch.setattr(service, "realizar_busqueda_hibrida", lambda q, s: (ranked, 0.9, {"m": 1}))
    monkeypatch.setattr(service, "preparar_fragmentos_para_generacion", lambda r, s: (finales, {"c": 2}))
    monkeypatch.setattr(service, "_preparar_mensaje_usuario_rag", lambda q, f: f"MSG:{q}")
    seen = []

    def fake_tokens(mensaje, stats=None):
        seen.append(mensaje)
        for t in tokens:
            yield t
        if stats is not None:
            stats.update({"model": "m", "eval_count": 2})

    monkeypatch.setattr(service, "generar_tokens_respuesta", fake_tokens)
    return seen


def test_answer_stream_yields_tokens_then_done_with_sources(monkeypatch, tmp_path):
    frag = {"metadata": {"source": "a.pdf", "page": 3}, "score_reranker": 0.9}
    seen = _stub_pipeline(monkeypatch, ranked=[frag], finales=[frag], tokens=["Ho", "la"])
    events = list(service.answer_stream(StorePaths("d", "x"), "¿Qué dice el documento?"))
    assert events[0] == ("token", {"token": "Ho"})
    assert events[1] == ("token", {"token": "la"})
    kind, done = events[2]
    assert kind == "done" and done["done"] is True
    assert done["sources"] == [{"document": "a.pdf", "pages": [4], "best_page": 4}]
    assert done["metrics"]["fase_contexto"] == {"c": 2}
    assert done["metrics"]["generation"] == {"model": "m", "eval_count": 2}
    assert seen == ["MSG:¿Qué dice el documento?"]


def test_answer_stream_reports_no_results_without_generating(monkeypatch):
    seen = _stub_pipeline(monkeypatch, ranked=[], finales=[], tokens=["x"])
    assert list(service.answer_stream(StorePaths("d", "x"), "¿Hay algo aquí?")) == [("no_results", {})]
    assert seen == []


def test_answer_stream_rejects_a_short_question(monkeypatch):
    _stub_pipeline(monkeypatch, ranked=[], finales=[], tokens=[])
    with pytest.raises(QuestionTooShort):
        list(service.answer_stream(StorePaths("d", "x"), "hola"))
