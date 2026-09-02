"""/api/study: one route, three artifacts, and the status codes that separate them.

Issue #140. The core builds the artifacts and the CLI already calls them; this
is the web layer's half. What matters here is not that a summary comes back —
that is the core's job and is tested there — but that the four ways this can
fail come back as four different answers.

The one worth stating: a generator that ignores the format is **422**, not 500.
It is not the server failing, it is the model replying in a shape the parse
refuses, and a UI should tell the user "try again, or another model" rather
than "something broke". For the quiz that distinction carries more: the core
raises rather than hand back questions whose answer key it could not verify, so
422 there means "not safe to grade you against", never "nothing found".
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.application.study import (  # noqa: E402
    MalformedOutlineError,
    MalformedQuizError,
    MalformedSummaryError,
)
from rag.web import app as web  # noqa: E402


@pytest.fixture
def client():
    web.app.config["TESTING"] = True
    with web.app.test_client() as c:
        yield c


@pytest.fixture
def stub(monkeypatch):
    """Point the route at doubles: no FAISS, no Ollama, no GPU."""
    state = {"fragments": [{"doc": "text"}], "raises": {}, "calls": []}

    monkeypatch.setattr(web, "_get_collection", lambda: object())
    monkeypatch.setattr(
        web.rag_engine, "obtener_documentos_indexados", lambda coll: ["paper.pdf"]
    )
    monkeypatch.setattr(
        web.rag_engine, "fragmentos_de_documento", lambda name: state["fragments"]
    )

    def _artifact(kind, marker):
        def call(fragmentos, **kwargs):
            state["calls"].append((kind, kwargs))
            if kind in state["raises"]:
                raise state["raises"][kind]
            return {"marker": marker}

        return call

    monkeypatch.setattr(web.rag_engine, "resumir_fragmentos", _artifact("summary", "s"))
    monkeypatch.setattr(web.rag_engine, "esquema_de_fragmentos", _artifact("outline", "o"))
    monkeypatch.setattr(web.rag_engine, "cuestionario_de_fragmentos", _artifact("quiz", "q"))
    return state


def _post(client, **body):
    body.setdefault("document", "paper.pdf")
    return client.post("/api/study", json=body)


class TestTheThreeArtifacts:
    @pytest.mark.parametrize("kind, marker", [("summary", "s"), ("outline", "o"), ("quiz", "q")])
    def test_each_kind_reaches_its_own_engine_call(self, client, stub, kind, marker):
        response = _post(client, kind=kind)

        assert response.status_code == 200
        assert response.json["artifact"] == {"marker": marker}
        assert response.json["kind"] == kind
        assert [c[0] for c in stub["calls"]] == [kind]

    def test_summary_is_the_default_kind(self, client, stub):
        assert _post(client).json["kind"] == "summary"

    def test_the_language_reaches_the_engine(self, client, stub):
        _post(client, kind="outline", language="Valencià")
        assert stub["calls"][0][1]["idioma"] == "Valencià"

    def test_the_question_count_reaches_the_quiz_only_when_asked(self, client, stub):
        _post(client, kind="quiz", question_count=7)
        assert stub["calls"][0][1]["num_preguntas"] == 7

    def test_an_omitted_count_leaves_the_core_default_alone(self, client, stub):
        # Passing None through would override the use case's own default with
        # a value the caller never chose.
        _post(client, kind="quiz")
        assert "num_preguntas" not in stub["calls"][0][1]


class TestTheFourWaysThisFails:
    def test_an_unknown_kind_is_rejected_before_any_work(self, client, stub):
        response = _post(client, kind="haiku")
        assert response.status_code == 400
        assert stub["calls"] == []

    def test_a_missing_document_is_rejected_before_any_work(self, client, stub):
        response = client.post("/api/study", json={"kind": "summary"})
        assert response.status_code == 400
        assert stub["calls"] == []

    def test_an_unindexed_document_is_404_not_500(self, client, stub):
        response = _post(client, document="no-existe.pdf")
        assert response.status_code == 404
        assert stub["calls"] == []

    def test_a_document_with_no_fragments_is_409(self, client, stub):
        stub["fragments"] = []
        response = _post(client)
        assert response.status_code == 409
        assert stub["calls"] == []

    @pytest.mark.parametrize(
        "kind, error",
        [
            ("summary", MalformedSummaryError("not json")),
            ("outline", MalformedOutlineError("not json")),
            ("quiz", MalformedQuizError("no usable question")),
        ],
    )
    def test_a_malformed_reply_is_422_and_says_so(self, client, stub, kind, error):
        stub["raises"][kind] = error
        response = _post(client, kind=kind)

        # 422 rather than 500: the model ignored the format, the server did not
        # break, and the UI should offer a retry rather than an apology.
        assert response.status_code == 422
        assert response.json["kind"] == "malformed"

    def test_a_bad_question_count_is_400_not_500(self, client, stub):
        stub["raises"]["quiz"] = ValueError("question_count must be between 1 and 20, got 0")
        response = _post(client, kind="quiz", question_count=0)
        assert response.status_code == 400

    def test_an_unexpected_failure_is_500(self, client, stub):
        stub["raises"]["summary"] = RuntimeError("ollama is down")
        response = _post(client, kind="summary")
        assert response.status_code == 500
        assert response.json["ok"] is False


def test_the_route_is_registered():
    assert "/api/study" in {str(rule) for rule in web.app.url_map.iter_rules()}
