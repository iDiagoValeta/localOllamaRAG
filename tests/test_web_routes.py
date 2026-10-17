"""Tests for the web layer's contract with the pipeline.

Covers the routes the frontend depends on, and the shape of the fragment dicts
the source panel is handed.
"""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.domain.chunk_metadata import ChunkMetadata
from monkeygrab.domain.fragment import Fragment
from rag.engine.settings import PERSISTED_FLAGS, STORE_IDS
from rag.engine.wiring import fragment_to_dict
from rag.web import app as web_app
from rag.web.app import _BUILTIN_STORES, _SETTINGS_MAP, _format_sources, app


def test_upload_route_is_preserved_for_api_compatibility():
    routes = {str(rule) for rule in app.url_map.iter_rules()}

    assert "/api/upload" in routes
    assert "/api/reindex" in routes


def _fragment(page, score_reranker=None, score_final=0.0):
    return Fragment(
        doc="text",
        metadata=ChunkMetadata(source="paper.pdf", page=page, chunk=0),
        score_reranker=score_reranker,
        score_final=score_final,
    )


def test_sources_panel_handles_fragments_that_were_never_reranked():
    """With reranking off no fragment carries a reranker score, and the panel
    falls back to score_final to pick each document's best page.

    That fallback is `frag.get("score_reranker", frag["score_final"])`, which a
    present-but-None key silently defeats: the panel then compares None against
    None and raises. Two fragments of one document are the minimum needed to
    reach that comparison.
    """
    fragments = [
        fragment_to_dict(_fragment(page=0, score_final=0.1)),
        fragment_to_dict(_fragment(page=1, score_final=0.2)),
    ]

    sources = _format_sources(fragments)

    assert sources == [{"document": "paper.pdf", "pages": [1, 2], "best_page": 2}]


def test_sources_panel_prefers_the_reranker_score_when_there_is_one():
    fragments = [
        fragment_to_dict(_fragment(page=0, score_reranker=0.9, score_final=0.1)),
        fragment_to_dict(_fragment(page=1, score_reranker=0.2, score_final=0.99)),
    ]

    sources = _format_sources(fragments)

    assert sources == [{"document": "paper.pdf", "pages": [1, 2], "best_page": 1}]


def test_every_toggle_the_panel_exposes_is_persisted():
    """The control panel's toggles and the saved flag list must be the same set.

    They are written in two files on purpose -- the frontend keys belong to the
    API, the engine variables to rag/engine/settings.py, which the CLI also
    reads. A toggle missing from the saved list would apply to the running
    session and silently vanish on restart; a saved flag with no toggle would
    outlive the UI that set it.
    """
    assert set(_SETTINGS_MAP.values()) == set(PERSISTED_FLAGS)


def test_the_labelled_stores_are_the_ones_the_engine_resolves():
    """A label for a store the engine cannot resolve would list a corpus nothing can select."""
    assert set(_BUILTIN_STORES) == set(STORE_IDS)


def _wire_settings_endpoint(monkeypatch):
    """Double the collection lookup and persistence; keep the real flag mutation.

    ``api_settings_post`` setattr's straight onto ``rag.chat_pdfs``, so every
    flag a test touches is pinned with monkeypatch to be restored afterwards.
    """
    monkeypatch.setattr(web_app, "_get_collection", lambda: None)
    monkeypatch.setattr(web_app.rag_engine, "guardar_ajustes_persistidos", lambda: None)
    monkeypatch.setattr(web_app.rag_engine, "index_fingerprint_mismatch", lambda coll: False)


@pytest.mark.parametrize("valor", ["false", "0", 0, 1, None])
def test_settings_rejects_a_non_bool_flag_value_naming_the_key(monkeypatch, valor):
    """Issue #357: bool("false") is True, so the endpoint must not coerce.

    A client that stringifies its payload would otherwise turn the flag on
    while reporting success. The 400 names the offending frontend key so the
    caller can fix the request instead of guessing which toggle failed.
    """
    _wire_settings_endpoint(monkeypatch)
    monkeypatch.setattr(web_app.rag_engine, "USAR_RERANKER", True)

    resp = web_app.app.test_client().post("/api/settings", json={"reranker": valor})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert body["key"] == "reranker"
    assert web_app.rag_engine.USAR_RERANKER is True


def test_a_rejected_settings_request_applies_none_of_its_flags(monkeypatch):
    """Validate the whole payload before mutating: a 400 must not half-apply.

    ``_SETTINGS_MAP`` order puts hybridSearch before reranker, so a per-key
    check that has already set the first flag would leave the running session
    on a configuration the failed request never completed.
    """
    _wire_settings_endpoint(monkeypatch)
    monkeypatch.setattr(web_app.rag_engine, "USAR_BUSQUEDA_HIBRIDA", True)
    monkeypatch.setattr(web_app.rag_engine, "USAR_RERANKER", True)

    resp = web_app.app.test_client().post(
        "/api/settings", json={"hybridSearch": False, "reranker": "false"}
    )

    assert resp.status_code == 400
    assert web_app.rag_engine.USAR_BUSQUEDA_HIBRIDA is True
