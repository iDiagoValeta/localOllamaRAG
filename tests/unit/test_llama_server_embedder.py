"""Unit tests for monkeygrab.adapters.embedding.llama_server_embedder.LlamaServerEmbedder.

Doubles ``urllib.request.urlopen`` in the adapter module -- no HTTP, no
llama-server, no model -- so these run in the fast/architecture CI gate. The
fake response is a tiny context-manager object carrying a status and a body;
a per-test responder decides what the fake server answers or whether it raises
a transport/HTTP error. Every expected vector is derived here with the
test-local ``_unit`` helper, never with the module's own normalization, so the
assertion cannot mirror the code under test.
"""

import base64
import http.client
import io
import json
import math
import sys
import urllib.error
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from monkeygrab.adapters.embedding import llama_server_embedder as module
from monkeygrab.adapters.embedding.llama_server_embedder import LlamaServerEmbedder

_BASE_URL = "http://localhost:8080"
_NATIVE_DIM = module._NATIVE_DIM


class _FakeResponse:
    """Stands in for the object ``urllib.request.urlopen`` returns.

    A context manager exposing a ``status`` attribute and ``read()``, exactly
    the two members the adapter's HTTP helper touches.
    """

    def __init__(self, payload, status=200):
        if isinstance(payload, bytes):
            self._body = payload
        else:
            self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _install_urlopen(monkeypatch, responder):
    """Replace ``urlopen`` with ``responder`` and return the recorded requests."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        return responder(request)

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    return calls


def _json_body(request):
    return json.loads(request.data.decode("utf-8"))


def _path(request):
    return urllib.parse.urlparse(request.full_url).path


def _unit(vector):
    """Independently re-derived L2 normalization, so expectations don't mirror the module."""
    norm = math.sqrt(sum(component * component for component in vector))
    return [component / norm for component in vector]


def _native(fill):
    return [fill] * _NATIVE_DIM


def _text_response(vector):
    return _FakeResponse({"data": [{"embedding": vector}]})


def test_embed_returns_a_normalized_truncated_vector(monkeypatch):
    native = [float(index + 1) for index in range(_NATIVE_DIM)]
    _install_urlopen(monkeypatch, lambda request: _text_response(native))
    embedder = LlamaServerEmbedder(_BASE_URL, dimensions=512)

    result = embedder.embed("hola")

    assert len(result) == 512
    assert result == pytest.approx(_unit(native[:512]))
    assert math.sqrt(sum(component * component for component in result)) == pytest.approx(1.0)


def test_embed_posts_the_text_as_a_document_to_the_v1_endpoint(monkeypatch):
    calls = _install_urlopen(monkeypatch, lambda request: _text_response(_native(0.5)))
    embedder = LlamaServerEmbedder(_BASE_URL)

    embedder.embed("hola")

    request = calls[0]
    assert request.method == "POST"
    assert _path(request) == "/v1/embeddings"
    assert _json_body(request) == {"input": "title: none | text: hola"}


def test_embed_query_posts_the_text_as_a_search_query(monkeypatch):
    calls = _install_urlopen(monkeypatch, lambda request: _text_response(_native(0.5)))
    embedder = LlamaServerEmbedder(_BASE_URL, dimensions=256)

    vector = embedder.embed_query("hola")

    assert _json_body(calls[0]) == {"input": "task: search result | query: hola"}
    assert len(vector) == 256


def test_embed_ignores_keep_alive(monkeypatch):
    calls = _install_urlopen(monkeypatch, lambda request: _text_response(_native(0.5)))
    embedder = LlamaServerEmbedder(_BASE_URL)

    embedder.embed("hola", keep_alive=0)

    assert _json_body(calls[0]) == {"input": "title: none | text: hola"}


def test_format_query_prefixes_a_search_result_task():
    assert module.format_query("hola") == "task: search result | query: hola"


def test_format_document_defaults_the_title_to_none():
    assert module.format_document("cuerpo") == "title: none | text: cuerpo"


def test_format_document_uses_the_given_title():
    assert module.format_document("cuerpo", title="Mi Doc") == "title: Mi Doc | text: cuerpo"


def test_embed_image_builds_the_props_and_multimodal_requests(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    raw = b"\x89PNG fake image bytes"
    image.write_bytes(raw)
    image_vector = [0.25] * _NATIVE_DIM

    def responder(request):
        if _path(request) == "/props":
            return _FakeResponse({"media_marker": "<__media_ABC123__>"})
        return _FakeResponse([{"embedding": image_vector}])

    calls = _install_urlopen(monkeypatch, responder)
    embedder = LlamaServerEmbedder(_BASE_URL, dimensions=512)

    result = embedder.embed_image(str(image))

    assert calls[0].method == "GET"
    assert _path(calls[0]) == "/props"

    request = calls[1]
    assert request.method == "POST"
    assert _path(request) == "/embeddings"
    assert _json_body(request) == {
        "content": [
            {
                "prompt_string": "<__media_ABC123__>",
                "multimodal_data": [base64.b64encode(raw).decode("ascii")],
            }
        ]
    }
    assert result == pytest.approx(_unit(image_vector[:512]))


def test_embed_image_accepts_a_nested_embedding_list(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"image")
    image_vector = [0.4] * _NATIVE_DIM

    def responder(request):
        if _path(request) == "/props":
            return _FakeResponse({"media_marker": "<__media_X__>"})
        # llama-server may nest the vector one level deep: [[floats]].
        return _FakeResponse([{"embedding": [image_vector]}])

    _install_urlopen(monkeypatch, responder)
    embedder = LlamaServerEmbedder(_BASE_URL, dimensions=512)

    result = embedder.embed_image(str(image))

    assert result == pytest.approx(_unit(image_vector[:512]))


def test_embed_image_caches_the_media_marker(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"image")
    props_paths = []

    def responder(request):
        if _path(request) == "/props":
            props_paths.append(request)
            return _FakeResponse({"media_marker": "<__media_X__>"})
        return _FakeResponse([{"embedding": _native(0.3)}])

    _install_urlopen(monkeypatch, responder)
    embedder = LlamaServerEmbedder(_BASE_URL)

    embedder.embed_image(str(image))
    embedder.embed_image(str(image))

    assert len(props_paths) == 1


def test_embed_image_combines_the_caption_as_a_normalized_sum(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"image")
    image_raw = [1.0] + [0.0] * (_NATIVE_DIM - 1)
    caption_raw = [0.0, 1.0] + [0.0] * (_NATIVE_DIM - 2)

    def responder(request):
        path = _path(request)
        if path == "/props":
            return _FakeResponse({"media_marker": "<__media_X__>"})
        if path == "/v1/embeddings":
            assert _json_body(request) == {"input": "title: none | text: una foto"}
            return _text_response(caption_raw)
        return _FakeResponse([{"embedding": image_raw}])

    calls = _install_urlopen(monkeypatch, responder)
    embedder = LlamaServerEmbedder(_BASE_URL, dimensions=512)

    result = embedder.embed_image(str(image), caption="una foto")

    assert [_path(request) for request in calls] == ["/props", "/embeddings", "/v1/embeddings"]
    expected = _unit([a + b for a, b in zip(_unit(image_raw[:512]), _unit(caption_raw[:512]))])
    assert result == pytest.approx(expected)
    assert sum(component * component for component in result) == pytest.approx(1.0)


@pytest.mark.parametrize("caption", ["", "   ", "\t\n"])
def test_embed_image_treats_a_blank_caption_as_no_caption(monkeypatch, tmp_path, caption):
    image = tmp_path / "figure.png"
    image.write_bytes(b"image")

    def responder(request):
        if _path(request) == "/props":
            return _FakeResponse({"media_marker": "<__media_X__>"})
        return _FakeResponse([{"embedding": _native(0.3)}])

    calls = _install_urlopen(monkeypatch, responder)
    embedder = LlamaServerEmbedder(_BASE_URL)

    embedder.embed_image(str(image), caption=caption)

    assert [_path(request) for request in calls] == ["/props", "/embeddings"]


def test_embed_raises_when_the_server_cannot_be_reached(monkeypatch):
    def responder(request):
        raise urllib.error.URLError("connection refused")

    _install_urlopen(monkeypatch, responder)
    embedder = LlamaServerEmbedder(_BASE_URL)

    with pytest.raises(RuntimeError) as excinfo:
        embedder.embed("hola")

    assert _BASE_URL in str(excinfo.value)
    assert "connection refused" in str(excinfo.value)


def test_embed_raises_a_runtime_error_when_the_server_times_out(monkeypatch):
    def responder(request):
        # A socket timeout surfaces as a bare TimeoutError, not as URLError.
        raise TimeoutError("timed out")

    _install_urlopen(monkeypatch, responder)
    embedder = LlamaServerEmbedder(_BASE_URL)

    with pytest.raises(RuntimeError) as excinfo:
        embedder.embed("hola")

    assert _BASE_URL in str(excinfo.value)
    assert "timed out" in str(excinfo.value)


def test_embed_raises_a_runtime_error_when_the_response_body_is_incomplete(monkeypatch):
    class _IncompleteBodyResponse(_FakeResponse):
        def read(self):
            raise http.client.IncompleteRead(b"partial")

    _install_urlopen(monkeypatch, lambda request: _IncompleteBodyResponse({}))
    embedder = LlamaServerEmbedder(_BASE_URL)

    with pytest.raises(RuntimeError) as excinfo:
        embedder.embed("hola")

    assert _BASE_URL in str(excinfo.value)
    assert "IncompleteRead" in str(excinfo.value)


def test_embed_raises_on_a_non_200_response_naming_status_and_body(monkeypatch):
    def responder(request):
        raise urllib.error.HTTPError(
            request.full_url, 500, "Internal Server Error", {}, io.BytesIO(b"boom from server")
        )

    _install_urlopen(monkeypatch, responder)
    embedder = LlamaServerEmbedder(_BASE_URL)

    with pytest.raises(RuntimeError) as excinfo:
        embedder.embed("hola")

    assert _BASE_URL in str(excinfo.value)
    assert "500" in str(excinfo.value)
    assert "boom from server" in str(excinfo.value)


def test_embed_raises_on_malformed_json(monkeypatch):
    _install_urlopen(monkeypatch, lambda request: _FakeResponse(b"{not-json"))
    embedder = LlamaServerEmbedder(_BASE_URL)

    with pytest.raises(RuntimeError, match="malformed JSON"):
        embedder.embed("hola")


def test_embed_raises_on_an_unexpected_vector_length(monkeypatch):
    _install_urlopen(monkeypatch, lambda request: _text_response([0.1, 0.2]))
    embedder = LlamaServerEmbedder(_BASE_URL)

    with pytest.raises(RuntimeError, match="unexpected"):
        embedder.embed("hola")


def test_embed_image_raises_when_props_lacks_a_media_marker(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"image")

    _install_urlopen(monkeypatch, lambda request: _FakeResponse({"something": "else"}))
    embedder = LlamaServerEmbedder(_BASE_URL)

    with pytest.raises(RuntimeError, match="media_marker"):
        embedder.embed_image(str(image))


def test_embed_image_raises_when_the_file_is_missing(monkeypatch, tmp_path):
    calls = _install_urlopen(monkeypatch, lambda request: _FakeResponse({}))
    embedder = LlamaServerEmbedder(_BASE_URL)

    with pytest.raises(FileNotFoundError):
        embedder.embed_image(str(tmp_path / "missing.png"))

    assert calls == []


def test_invalid_dimensions_raise_value_error():
    with pytest.raises(ValueError, match="dimensions"):
        LlamaServerEmbedder(_BASE_URL, dimensions=100)


def test_close_is_a_no_op_and_the_instance_is_never_unusable(monkeypatch):
    _install_urlopen(monkeypatch, lambda request: _text_response(_native(0.5)))
    embedder = LlamaServerEmbedder(_BASE_URL)

    assert embedder.is_unusable is False

    embedder.close()
    embedder.close()

    assert embedder.is_unusable is False
    assert embedder.embed("hola") == pytest.approx(_unit(_native(0.5)[:512]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host", "localhost:8091"])
def test_rejects_a_base_url_that_is_not_http(url):
    with pytest.raises(ValueError, match="http"):
        module.LlamaServerEmbedder(url)
