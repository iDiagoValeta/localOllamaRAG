"""OllamaChatModel -- ChatModel adapter over Ollama generate/chat."""

import base64
import json
import logging
import time
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Sequence

import ollama
import requests

from monkeygrab.config.env import DEFAULT_OLLAMA_BASE_URL
from monkeygrab.domain.generation_chunk import GenerationChunk
from monkeygrab.ports.model_unloader import ModelUnloader

# Not subclassed from monkeygrab.ports.chat_model.ChatModel: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.

# Only the fallback for a caller that names no endpoint; the pipeline always
# passes config.models.ollama.base_url, which is where OLLAMA_BASE_URL lands.
_DEFAULT_BASE_URL = DEFAULT_OLLAMA_BASE_URL


@lru_cache(maxsize=None)
def ollama_client_for(base_url: str) -> ollama.Client:
    """Return the shared ``ollama.Client`` for one endpoint.

    Cached per URL because ``wiring`` builds a fresh adapter for every query:
    a client per instance would open a new HTTP connection pool per question
    and never close it. The module-level ``ollama.chat`` this replaced had the
    same one-client-per-process behaviour, minus the configurable host.

    Public because the CLI and web ``/chat`` modes call ``ollama.chat``
    directly instead of going through this adapter, and they must reach the
    same server the RAG pipeline does.
    """
    return ollama.Client(host=base_url)


class OllamaChatModel:
    """One Ollama-backed model role: single-shot generation or token streaming.

    One instance is one role, constructed with that role's context window and
    default sampling options. Options stay a free-form dict rather than named
    parameters, mirroring the shape Ollama's own API takes, because which
    knobs matter varies by role and by model.

    ``think`` is hardcoded off. Thinking-capable models otherwise spend their
    prediction budget on a reasoning trace and return an empty answer, which
    is indistinguishable from a failure at every call site.

    ``generate`` goes through the ``ollama`` client, which covers both real
    shapes: a one-turn prompt, and a system plus user message with optional
    image bytes. ``stream`` instead talks to ``/api/generate`` over raw HTTP,
    because it needs line-by-line JSON and a retry policy limited to 5xx --
    neither of which the client exposes. Both are bound to ``base_url``, so
    the two paths cannot end up talking to different servers.

    ``model_unloader`` frees VRAM before streaming starts, and again before
    retrying a 5xx. It is injected rather than computed here because
    unloading "every *other* role's model" requires knowing all the roles at
    once, which a single-role adapter deliberately does not. ``None`` means
    no VRAM management.

    Failure policy: hard-fail. Any Ollama failure raises.
    """

    def __init__(
        self,
        model: str,
        *,
        num_ctx: int,
        keep_alive: int = 0,
        request_timeout: int = 900,
        generate_retries: int = 1,
        generate_retry_delay: int = 3,
        options: Optional[Dict[str, Any]] = None,
        base_url: str = _DEFAULT_BASE_URL,
        model_unloader: Optional[ModelUnloader] = None,
    ):
        """Args:
            model: Ollama model name for this role.
            num_ctx: Context window for this role (merged into every call's
                ``options``, overriding any ``num_ctx`` already in ``options``).
            keep_alive: Seconds to keep the model loaded after the call
                (``0`` unloads immediately, matching ``OLLAMA_KEEP_ALIVE``).
            request_timeout: HTTP timeout in seconds for ``stream``.
            generate_retries: Total attempts for ``stream`` on repeated 5xx
                responses (``1`` = no retry).
            generate_retry_delay: Seconds to wait between ``stream`` retries.
            options: Role-specific sampling defaults (temperature,
                num_predict, top_p, repeat_penalty, stop, ...), merged with
                ``num_ctx`` on every call.
            base_url: Ollama HTTP server base URL, used by both ``generate``
                and ``stream``. Passing it explicitly to the client is what
                makes it win over the ambient ``OLLAMA_HOST`` the client would
                otherwise read on its own.
            model_unloader: ``ModelUnloader`` used by ``stream`` to free VRAM
                before generating (and before retrying a 5xx), or ``None`` to
                skip that entirely -- see the class docstring.
        """
        self._model = model
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._request_timeout = request_timeout
        self._generate_retries = max(1, generate_retries)
        self._generate_retry_delay = generate_retry_delay
        self._base_options = dict(options or {})
        self._base_url = base_url
        self._model_unloader = model_unloader

    def _options(self) -> Dict[str, Any]:
        merged = dict(self._base_options)
        merged["num_ctx"] = self._num_ctx
        return merged

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Sequence[bytes] = (),
    ) -> str:
        """Generate a complete response in one call via ``ollama.chat``.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.
            images: Optional raw image bytes (vision models only).

        Returns:
            The complete generated text.

        Raises:
            RuntimeError: On any generation failure.
        """
        message: Dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = [base64.b64encode(img).decode("utf-8") for img in images]

        messages = [message]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        try:
            response = ollama_client_for(self._base_url).chat(
                model=self._model,
                messages=messages,
                think=False,
                keep_alive=self._keep_alive,
                options=self._options(),
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama generate failed for model {self._model!r}: {exc}") from exc

        return response["message"]["content"]

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[GenerationChunk]:
        """Stream a response from ``/api/generate`` over raw HTTP.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.

        Yields:
            One ``GenerationChunk`` per parsed response line, exactly as
            ``_ollama_generate_stream`` yields one raw dict per line --
            including the final ``done=True`` line (metadata only, usually
            empty ``text``), which the original's own token-forwarding loop
            (``generar_tokens_respuesta``) filters out of its *token*
            output but still reads for stats; here that split is the
            caller's job, not this adapter's (see ``GenerationChunk``).

        Raises:
            RuntimeError: On any generation failure (after exhausting the
                5xx retry budget).
        """
        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": self._options(),
            "keep_alive": self._keep_alive,
        }
        if system:
            payload["system"] = system

        if self._model_unloader is not None:
            self._model_unloader.unload_all_except(self._model)

        url = f"{self._base_url}/api/generate"

        for attempt in range(self._generate_retries):
            try:
                with requests.post(
                    url=url, json=payload, stream=True, timeout=self._request_timeout
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        done = bool(data.get("done"))
                        if done and data.get("done_reason") not in (None, "stop"):
                            logging.warning(
                                "%s generation stopped early: done_reason=%s",
                                self._model, data.get("done_reason"),
                            )
                        yield GenerationChunk(
                            text=data.get("response", ""),
                            done=done,
                            model=data.get("model"),
                            done_reason=data.get("done_reason"),
                            total_duration=data.get("total_duration"),
                            load_duration=data.get("load_duration"),
                            prompt_eval_count=data.get("prompt_eval_count"),
                            prompt_eval_duration=data.get("prompt_eval_duration"),
                            eval_count=data.get("eval_count"),
                            eval_duration=data.get("eval_duration"),
                        )
                return
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status >= 500 and attempt + 1 < self._generate_retries:
                    if self._model_unloader is not None:
                        self._model_unloader.unload_all_except(None)
                    time.sleep(self._generate_retry_delay)
                    continue
                raise RuntimeError(
                    f"Ollama stream generation failed for model {self._model!r}: {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Ollama stream generation failed for model {self._model!r}: {exc}"
                ) from exc
