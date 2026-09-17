"""HTTP contract of the headless service, with the service functions doubled."""
import json
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag.headless.app import create_app
from rag.headless.health import HealthReport
from rag.headless.service import QuestionTooShort, StoreRegistry


def _health(ok=True, reason=None):
    return lambda config: HealthReport(ok, reason, "abc", {"available": ok}, {"present": True, "python": "/p"},
                                       {"rag": {"backend": "openai", "model": "m"}})


def _service(*, events=None, index=None, status=None, index_error=None, status_error=None,
             index_gate=None, index_entered=None):
    def answer_stream(paths, question):
        if len(question) < 10:
            raise QuestionTooShort("short")
        for event in events or []:
            if event == ("error", None):
                raise RuntimeError("model down")
            yield event

    def index_store(paths):
        if index_gate is not None:
            if index_entered is not None:
                index_entered.set()
            index_gate.wait()
        if index_error:
            raise index_error
        return index or {"documents_indexed": 1, "documents_skipped": 0, "chunks_indexed": 4,
                         "fingerprint": "fp", "seconds": 1.5}

    def status_store(paths):
        if status_error:
            raise status_error
        return status or {"exists": True, "documents": ["a.pdf"], "chunks_total": 4,
                          "fingerprint": "fp", "stale": False}

    return types.SimpleNamespace(
        answer_stream=answer_stream,
        index_store=index_store,
        status_store=status_store,
    )


def _client(**kwargs):
    kwargs.setdefault("service", _service())
    kwargs.setdefault("health", _health())
    kwargs.setdefault("registry", StoreRegistry())
    app = create_app(**kwargs)
    app.testing = True
    return app.test_client()


def _events(body: bytes):
    out = []
    for block in body.decode("utf-8").strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


PATHS = {"docs_folder": "/d", "data_dir": "/x"}


def test_health_ok():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True and resp.get_json()["commit"] == "abc"


def test_health_503_with_reason_when_the_stack_cannot_run():
    resp = _client(health=_health(False, "CUDA is not available")).get("/health")
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["ok"] is False and body["reason"] == "CUDA is not available"


def test_index_returns_the_service_result_with_the_store_id():
    resp = _client().post("/stores/cv1/index", json=PATHS)
    assert resp.status_code == 200
    assert resp.get_json()["store_id"] == "cv1" and resp.get_json()["chunks_indexed"] == 4


def test_index_requires_both_paths():
    resp = _client().post("/stores/cv1/index", json={"docs_folder": "/d"})
    assert resp.status_code == 400
    assert "data_dir" in resp.get_json()["message"]


def test_same_store_id_with_other_paths_is_409():
    client = _client()
    client.post("/stores/cv1/index", json=PATHS)
    resp = client.post("/stores/cv1/index", json={"docs_folder": "/other", "data_dir": "/x"})
    assert resp.status_code == 409


def test_index_failure_is_502_with_the_message():
    resp = _client(service=_service(index_error=RuntimeError("Indexing failed on all 2 file(s)"))).post(
        "/stores/cv1/index", json=PATHS
    )
    assert resp.status_code == 502
    assert "Indexing failed" in resp.get_json()["message"]


def test_status_reads_paths_from_the_query_string():
    resp = _client().get("/stores/cv1/status", query_string=PATHS)
    assert resp.status_code == 200 and resp.get_json()["documents"] == ["a.pdf"]


def test_status_failure_is_502_with_the_message():
    resp = _client(service=_service(status_error=RuntimeError("index corrupt"))).get(
        "/stores/cv1/status", query_string=PATHS
    )
    assert resp.status_code == 502
    assert "index corrupt" in resp.get_json()["message"]


def test_rag_streams_tokens_then_done():
    events = [("token", {"token": "Ho"}), ("token", {"token": "la"}),
              ("done", {"done": True, "sources": [{"document": "a.pdf", "pages": [1], "best_page": 1}], "metrics": {}})]
    resp = _client(service=_service(events=events)).post(
        "/stores/cv1/rag", json={**PATHS, "message": "¿Qué dice el documento?", "stream": True}
    )
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    assert _events(resp.data) == events


def test_rag_without_stream_returns_json():
    events = [("token", {"token": "Ho"}), ("token", {"token": "la"}),
              ("done", {"done": True, "sources": [], "metrics": {"fase_contexto": {}}})]
    resp = _client(service=_service(events=events)).post(
        "/stores/cv1/rag", json={**PATHS, "message": "¿Qué dice el documento?", "stream": False}
    )
    assert resp.get_json() == {"ok": True, "response": "Hola", "sources": [],
                               "metrics": {"fase_contexto": {}}}


def test_rag_no_results_is_a_200_with_ok_false():
    resp = _client(service=_service(events=[("no_results", {})])).post(
        "/stores/cv1/rag", json={**PATHS, "message": "¿Qué dice el documento?"}
    )
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is False and resp.get_json()["error"] == "no_results"


def test_rag_short_question_is_400():
    resp = _client().post("/stores/cv1/rag", json={**PATHS, "message": "hola"})
    assert resp.status_code == 400 and resp.get_json()["error"] == "question_too_short"


def test_rag_rejects_a_non_boolean_stream_flag():
    resp = _client().post(
        "/stores/cv1/rag", json={**PATHS, "message": "¿Qué dice el documento?", "stream": "false"}
    )
    assert resp.status_code == 400
    assert "stream" in resp.get_json()["message"]


def test_rag_mid_stream_failure_becomes_an_error_event():
    events = [("token", {"token": "Ho"}), ("error", None)]
    resp = _client(service=_service(events=events)).post(
        "/stores/cv1/rag", json={**PATHS, "message": "¿Qué dice el documento?", "stream": True}
    )
    parsed = _events(resp.data)
    assert parsed[0] == ("token", {"token": "Ho"})
    assert parsed[1][0] == "error" and "model down" in parsed[1][1]["error"]


def test_token_is_required_when_configured():
    client = _client(token="s3cret")
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_health_is_probed_with_the_process_config():
    seen = []

    def health(config):
        seen.append(config)
        return _health()(config)

    _client(health=health).get("/health")
    assert len(seen) == 1 and seen[0].models is not None


def test_concurrent_index_on_the_same_store_is_409():
    gate = threading.Event()
    entered = threading.Event()
    registry = StoreRegistry()
    app = create_app(service=_service(index_gate=gate, index_entered=entered), health=_health(),
                      registry=registry)
    app.testing = True
    client_a = app.test_client()
    client_b = app.test_client()
    results = {}

    def run_first():
        results["first"] = client_a.post("/stores/cv1/index", json=PATHS)

    thread = threading.Thread(target=run_first)
    thread.start()
    try:
        # A failing assert below must still release the gate and join the
        # thread -- otherwise a non-daemon thread is left blocked on it and
        # pytest hangs instead of reporting the failure.
        assert entered.wait(timeout=2), "the first request never reached index_store"
        resp = client_b.post("/stores/cv1/index", json=PATHS)
        assert resp.status_code == 409
        assert resp.get_json()["error"] == "index_in_progress"
    finally:
        gate.set()
        thread.join(timeout=10)
    assert results["first"].status_code == 200


def test_non_ascii_bearer_is_a_json_401():
    client = _client(token="s3cret")
    resp = client.get("/health", headers={"Authorization": "Bearer é"})
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"
