"""LlamaServerEmbedder -- Embedder/ImageEmbedder adapter over a llama.cpp ``llama-server``.

Talks HTTP to an already-running ``llama-server`` started with ``--embedding``
and serving the GGUF model ``google/embeddinggemma-2`` (768 dimensions,
Matryoshka: 768/512/256/128). This adapter does NOT start, configure or stop
that server; its residency and model are the server process's concern. The
whole file imports nothing but the standard library (``urllib``), so it and its
unit tests run in the product environment without a GPU, torch or the model.

Two wire endpoints, both verified against the real server:

- Text: ``POST {base_url}/v1/embeddings`` body ``{"input": "<text>"}`` ->
  ``{"data": [{"embedding": [...768 floats...]}]}``.
- Image: first ``GET {base_url}/props`` for the ``media_marker`` string (a
  random value like ``"<__media_XXXX__>"``, cached after the first read), then
  ``POST {base_url}/embeddings`` body
  ``{"content": [{"prompt_string": <marker>, "multimodal_data": [<base64>]}]}``
  -> a JSON list whose ``[0]["embedding"]`` is a flat list of floats or a list
  holding one such list. Images take no task prefix.

Task prefixes (EmbeddingGemma 2 model card) are applied here rather than by
the caller, unlike what the ``Embedder`` port describes: the indexing use case
calls ``embed`` with raw chunk text for every embedder, and jina-clip needs no
prefix at all. ``embed`` therefore formats its input as a document and
``embed_query`` -- which ``Retrieve`` prefers when an embedder offers it --
formats it as a search query. Images are embedded without a prefix.

Failure policy: hard-fail (project-wide). Any connection error, non-200
response, malformed JSON or wrong vector length raises ``RuntimeError`` naming
``base_url``; a missing image file raises ``FileNotFoundError``. There are no
retries and no fallbacks.
"""

import base64
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, List, Optional

# The server's native width before any Matryoshka truncation.
_NATIVE_DIM = 768
# Matryoshka widths embeddinggemma-2 supports; anything else is a config error.
_ALLOWED_DIMENSIONS = (768, 512, 256, 128)

# Not subclassed from monkeygrab.ports.embedder.Embedder /
# monkeygrab.ports.image_embedder.ImageEmbedder: Protocol conformance here is
# structural (duck typing), the same contract every other adapter in this
# package satisfies without inheriting its port.


def format_query(text: str) -> str:
    """Return ``text`` prefixed as an EmbeddingGemma 2 search query.

    Args:
        text: The raw query text.

    Returns:
        The text with the model's query task prefix applied. The ``Embedder``
        port leaves prefixing to the caller, so the caller uses this helper.
    """
    return f"task: search result | query: {text}"


def format_document(text: str, title: Optional[str] = None) -> str:
    """Return ``text`` prefixed as an EmbeddingGemma 2 document.

    Args:
        text: The raw document text.
        title: The document's title, or ``None`` (the default) to send the
            literal ``"none"`` the model card specifies for untitled text.

    Returns:
        The text with the model's document task prefix applied.
    """
    resolved_title = "none" if title is None else title
    return f"title: {resolved_title} | text: {text}"


def _l2_normalize(vector: List[float]) -> List[float]:
    """Return ``vector`` scaled to unit L2 norm. Pure function, no I/O."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise RuntimeError("llama-server embedder: cannot L2-normalize a zero vector")
    return [component / norm for component in vector]


class LlamaServerEmbedder:
    """Embeds text and images by HTTP against a running ``llama-server``.

    Text goes out with its document or query prefix (``embed`` /
    ``embed_query``) and the returned 768-dimensional vector is truncated to ``dimensions``
    (Matryoshka) and L2-normalized. Images are sent through llama-server's
    multimodal ``/embeddings`` endpoint (the marker from ``/props`` plus the
    file's base64 bytes), truncated and normalized identically, so text and
    images share one aligned vector space.

    Image+caption combination rule: when ``caption`` is non-empty after
    stripping whitespace, the caption is embedded as a document
    (``format_document``) and the result is the L2-normalized element-wise SUM
    of the normalized image vector and the normalized caption vector -- never a
    raw, sub-unit-norm average (the same rule the Jina CLIP adapter documents).
    A blank caption embeds the image alone.

    Failure policy: hard-fail, no retries, no fallbacks. See the module
    docstring for what each failure raises.
    """

    def __init__(
        self,
        base_url: str,
        *,
        dimensions: int = 512,
        timeout_s: float = 120.0,
    ):
        """Args:
            base_url: Root URL of the running ``llama-server`` (e.g.
                ``http://localhost:8080``). This adapter never starts it.
            dimensions: Matryoshka output width; one of ``768``, ``512``,
                ``256``, ``128``. The server always returns 768 floats and this
                adapter truncates them.
            timeout_s: Per-request socket timeout in seconds.

        Raises:
            ValueError: If ``dimensions`` is not one of the supported widths.
        """
        if dimensions not in _ALLOWED_DIMENSIONS:
            raise ValueError(
                f"dimensions must be one of {_ALLOWED_DIMENSIONS} (embeddinggemma-2 "
                f"Matryoshka widths), got {dimensions!r}"
            )
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self._timeout_s = timeout_s
        self._media_marker_cache: Optional[str] = None

    # lifecycle

    @property
    def is_unusable(self) -> bool:
        """Whether this instance has retired itself for good.

        Always ``False``: unlike the in-process Jina CLIP adapter, this one
        owns no worker process, so it has no liveness of its own to lose. It
        exists to satisfy the structural contract ``rag/engine/wiring.py``
        reads (see ``_is_unusable``) without that caller having to special-case
        this adapter.
        """
        return False

    def close(self) -> None:
        """Release nothing. Present for the embedder lifecycle contract.

        ``llama-server`` is an external process whose residency this adapter
        does not own, so there is nothing to shut down. Safe to call any number
        of times.
        """
        return None

    # HTTP

    def _build_request(
        self, url: str, *, method: str, payload: Optional[dict] = None
    ) -> urllib.request.Request:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        return request

    def _read_json(self, request: urllib.request.Request) -> Any:
        """Perform ``request`` and return the parsed JSON body.

        Raises:
            RuntimeError: On any transport error, non-200 status, or malformed
                JSON, naming ``base_url`` and the server's own message.
        """
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                status = getattr(response, "status", 200)
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"{self._base_url}: llama-server returned HTTP {exc.code}: {detail!r}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"{self._base_url}: could not reach llama-server: {exc.reason}"
            ) from exc

        if status != 200:
            raise RuntimeError(
                f"{self._base_url}: llama-server returned HTTP {status}: {raw[:500]!r}"
            )
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"{self._base_url}: llama-server returned malformed JSON: {raw[:500]!r}"
            ) from exc

    # ADAPTER

    def embed(self, text: str, *, keep_alive: Optional[int] = None) -> List[float]:
        """Embed ``text`` as a document via llama-server's ``/v1/embeddings``.

        Satisfies the ``Embedder`` port. ``text`` is wrapped with
        ``format_document`` first. The 768-float server vector is truncated to
        ``dimensions`` and L2-normalized.

        Args:
            text: Raw document text.
            keep_alive: Accepted for signature compatibility with the
                ``Embedder`` port and ignored: ``llama-server`` manages its own
                model residency in the server process, not per request, so
                Ollama's per-call unload has no counterpart here.

        Returns:
            The ``dimensions``-dimensional L2-normalized embedding vector.

        Raises:
            RuntimeError: On any connection, HTTP, JSON or vector-shape failure.
        """
        return self._embed_text(format_document(text))

    def embed_query(self, text: str) -> List[float]:
        """Embed ``text`` as a search query (``format_query``), same output as ``embed``.

        Raises:
            RuntimeError: On any connection, HTTP, JSON or vector-shape failure.
        """
        return self._embed_text(format_query(text))

    def _embed_text(self, text: str) -> List[float]:
        request = self._build_request(
            f"{self._base_url}/v1/embeddings", method="POST", payload={"input": text}
        )
        parsed = self._read_json(request)
        try:
            vector = parsed["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"{self._base_url}: unexpected /v1/embeddings response shape: {parsed!r}"
            ) from exc
        return self._finalize(vector)

    def embed_image(self, image_path: str, *, caption: str = "") -> List[float]:
        """Embed the image at ``image_path``, combined with ``caption`` if real.

        Satisfies the ``ImageEmbedder`` port. The image is embedded through
        llama-server's multimodal endpoint with no prefix; a non-empty caption
        (after stripping) is embedded as a document and combined as the
        L2-normalized sum of the two normalized vectors (see the class
        docstring).

        Args:
            image_path: Path to the image file on disk.
            caption: Caption text found near the image, if any. Blank means
                "no caption": the image is embedded alone.

        Returns:
            The ``dimensions``-dimensional L2-normalized combined vector.

        Raises:
            FileNotFoundError: If ``image_path`` does not exist.
            RuntimeError: On any connection, HTTP, JSON or vector-shape failure,
                including a ``/props`` response without a usable ``media_marker``.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"llama-server embedder: image file not found: {image_path!r}"
            )

        marker = self._media_marker()
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        request = self._build_request(
            f"{self._base_url}/embeddings",
            method="POST",
            payload={
                "content": [
                    {"prompt_string": marker, "multimodal_data": [encoded]}
                ]
            },
        )
        parsed = self._read_json(request)
        image_vector = self._finalize(self._extract_image_embedding(parsed))

        caption = caption.strip()
        if not caption:
            return image_vector

        caption_vector = self.embed(caption)
        combined = [a + b for a, b in zip(image_vector, caption_vector)]
        return _l2_normalize(combined)

    # helpers

    def _media_marker(self) -> str:
        """Return the cached image marker from ``/props``, fetching it once."""
        if self._media_marker_cache is not None:
            return self._media_marker_cache

        request = self._build_request(f"{self._base_url}/props", method="GET")
        parsed = self._read_json(request)
        marker = parsed.get("media_marker") if isinstance(parsed, dict) else None
        if not isinstance(marker, str) or not marker:
            raise RuntimeError(
                f"{self._base_url}: /props response has no usable media_marker: {parsed!r}"
            )
        self._media_marker_cache = marker
        return marker

    def _extract_image_embedding(self, parsed: Any) -> Any:
        """Unwrap the image ``embedding`` whether flat or nested one level."""
        try:
            embedding = parsed[0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"{self._base_url}: unexpected /embeddings response shape: {parsed!r}"
            ) from exc
        while (
            isinstance(embedding, list)
            and len(embedding) == 1
            and isinstance(embedding[0], list)
        ):
            embedding = embedding[0]
        return embedding

    def _finalize(self, vector: Any) -> List[float]:
        """Validate the server's 768-float vector, then truncate and normalize."""
        if not isinstance(vector, list) or len(vector) != _NATIVE_DIM:
            got = len(vector) if isinstance(vector, list) else type(vector).__name__
            raise RuntimeError(
                f"{self._base_url}: llama-server returned a vector of unexpected shape: "
                f"expected {_NATIVE_DIM} floats, got {got}"
            )
        if not all(isinstance(component, (int, float)) for component in vector):
            raise RuntimeError(
                f"{self._base_url}: llama-server returned a non-numeric vector component"
            )
        return _l2_normalize([float(component) for component in vector[: self._dimensions]])
