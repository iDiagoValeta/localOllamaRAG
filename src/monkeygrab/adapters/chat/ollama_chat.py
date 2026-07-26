"""OllamaChatModel -- ChatModel adapter over Ollama generate/chat.

# ─────────────────────────────────────────────
# SECTION 1: ADAPTER
# ─────────────────────────────────────────────
"""

import base64
import json
import time
from typing import Any, Dict, Iterator, Optional, Sequence

import ollama
import requests

# Not subclassed from monkeygrab.ports.chat_model.ChatModel: Protocol
# conformance here is structural (duck typing), the same contract every other
# adapter in this package satisfies without inheriting its port.

# Not derived from AppConfig: rag/engine/generation.py hardcodes this same
# value (`OLLAMA_BASE_URL = "http://localhost:11434"`) rather than reading it
# from an env var, so there is no config field to source it from without
# inventing one outside this task's scope.
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaChatModel:
    """One Ollama-backed model role: single-shot generation or token streaming.

    Per the ``ChatModel`` port docstring, every pipeline role that talks to
    an LLM (``MODELO_CHAT``, ``MODELO_CONTEXTUAL``, ``MODELO_RECOMP``,
    ``MODELO_OCR``, ``MODELO_RAG``) is a separate model *name*, wired as its
    own ``ChatModel`` instance -- so one ``OllamaChatModel`` instance is one
    role, constructed with that role's ``num_ctx`` and default sampling
    ``options`` (temperature, num_predict, top_p, repeat_penalty, stop, ...).
    Those defaults vary per role today (see ``generar_queries_con_llm``,
    ``generar_contexto_situacional``, ``sintetizar_contexto_recomp``,
    ``describir_imagen_con_llm``, ``generar_tokens_respuesta``) and are kept
    as a free-form ``options`` dict rather than exploded into named
    parameters, mirroring the shape Ollama's own API already takes.

    ``think`` is hardcoded ``False`` (not a constructor parameter): every
    real call site sets it, unconditionally, to keep thinking-capable models
    (Gemma 4, Qwen3, ...) from spending their ``num_predict`` budget on a
    reasoning trace instead of the answer.

    ``generate`` uses ``ollama.chat`` for both of today's real shapes: a
    plain single-user-message prompt (``generar_queries_con_llm``, which
    calls ``ollama.generate`` directly today but is functionally a one-turn
    chat) and a system+user message with optional image bytes
    (``generar_contexto_situacional``, ``describir_imagen_con_llm``).
    ``stream`` talks to ``/api/generate`` over raw HTTP directly, like
    ``_ollama_generate_stream`` in ``rag/engine/generation.py`` does today,
    because it needs line-by-line JSON streaming and 5xx-only retry that the
    ``ollama`` client does not expose.

    Not replicated here: ``liberar_modelos_ollama``, the VRAM-freeing call
    that unloads every *other* configured model role before streaming a
    response. That requires knowing every role's model name at once, which
    is orchestration across multiple ``ChatModel`` instances, not something
    a single-role adapter can do -- it has no home yet in this architecture
    and belongs in the future application layer that wires all the roles
    together.

    Failure policy: hard-fail. Every real call site today catches broadly and
    substitutes empty output or raw unsynthesized context on failure; none of
    that survives here -- any Ollama failure raises.
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
            base_url: Ollama HTTP server base URL, used by ``stream`` only
                (``generate`` goes through the ``ollama`` client, which reads
                its own ``OLLAMA_HOST``).
        """
        self._model = model
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._request_timeout = request_timeout
        self._generate_retries = max(1, generate_retries)
        self._generate_retry_delay = generate_retry_delay
        self._base_options = dict(options or {})
        self._base_url = base_url

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
            response = ollama.chat(
                model=self._model,
                messages=messages,
                think=False,
                keep_alive=self._keep_alive,
                options=self._options(),
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama generate failed for model {self._model!r}: {exc}") from exc

        return response["message"]["content"]

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[str]:
        """Stream a response from ``/api/generate`` over raw HTTP.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.

        Yields:
            Successive text chunks.

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
                        content = data.get("response", "")
                        if content:
                            yield content
                return
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status >= 500 and attempt + 1 < self._generate_retries:
                    time.sleep(self._generate_retry_delay)
                    continue
                raise RuntimeError(
                    f"Ollama stream generation failed for model {self._model!r}: {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Ollama stream generation failed for model {self._model!r}: {exc}"
                ) from exc
