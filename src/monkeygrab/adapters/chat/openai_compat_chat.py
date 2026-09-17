"""OpenAICompatChatModel -- ChatModel adapter over an OpenAI-style chat API.

One instance is one model role on one server that speaks
``POST {base_url}/chat/completions`` the way OpenAI does: llama-server,
Daimon's model gateway, LM Studio, vLLM. It exists so the roles that generate
text can run on the same server the host application already serves, instead
of requiring an Ollama next to it.

What the OpenAI API does not report is left ``None`` on the final
``GenerationChunk``: there are no load or evaluation durations, so a caller
deriving tokens-per-second gets ``None`` rather than a number invented here.

Failure policy: hard-fail. Any non-2xx status, malformed body or broken
stream raises ``RuntimeError``. No retries: a caller that wants them composes
them explicitly.
"""
import base64
import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import requests

from monkeygrab.domain.generation_chunk import GenerationChunk


def _image_data_uri(image: bytes) -> str:
    """Wrap raw image bytes as a data URI, sniffing PNG vs JPEG from the header."""
    mime = "image/jpeg" if image[:3] == b"\xff\xd8\xff" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(image).decode("ascii")


class OpenAICompatChatModel:
    """One OpenAI-compatible model role: single-shot generation or token streaming.

    Not subclassed from ``monkeygrab.ports.chat_model.ChatModel``: conformance
    is structural, like every other adapter in this package.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str = "",
        options: Optional[Dict[str, Any]] = None,
        timeout: int = 900,
    ):
        """Args:
            model: Model name the server should serve for this role.
            base_url: Server base URL; ``/chat/completions`` is appended.
            api_key: Bearer token, or empty to send no Authorization header.
            options: Sampling parameters already in OpenAI names
                (``temperature``, ``top_p``, ``max_tokens``, ``stop``),
                copied into every request payload.
            timeout: HTTP timeout in seconds for both ``generate`` and
                ``stream``.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._options = dict(options or {})
        self._timeout = timeout

    def url(self) -> str:
        """The chat completions endpoint this adapter posts to."""
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _messages(
        self, prompt: str, system: Optional[str], images: Sequence[bytes]
    ) -> List[Dict[str, Any]]:
        content: Any = prompt
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": _image_data_uri(image)}}
                for image in images
            ]
        messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return messages

    def _payload(self, messages: List[Dict[str, Any]], *, stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self._model, "messages": messages, "stream": stream}
        payload.update(self._options)
        return payload

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Sequence[bytes] = (),
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Generate a complete response in one call.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.
            images: Raw image bytes, sent as data-URI ``image_url`` parts.
            response_format: JSON Schema the reply must conform to, sent as
                OpenAI's ``json_schema`` response format.

        Returns:
            The complete generated text.

        Raises:
            RuntimeError: On any HTTP failure, a body without
                ``choices[0].message.content``, or content that is not a
                string (tool calls and refusals arrive as ``null``).
        """
        payload = self._payload(self._messages(prompt, system, images), stream=False)
        if response_format is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": dict(response_format)},
            }
        try:
            response = requests.post(
                self.url(), headers=self._headers(), json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise RuntimeError(
                    f"OpenAI-compatible generate returned non-string content "
                    f"for model {self._model!r}: {type(content).__name__}"
                )
            return content
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI-compatible generate failed for model {self._model!r}: {exc}"
            ) from exc

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[GenerationChunk]:
        """Stream a response as ``GenerationChunk`` items.

        Args:
            prompt: User/task prompt.
            system: Optional system prompt.

        Yields:
            One chunk per content delta, then exactly one ``done=True`` chunk
            carrying the served model, the finish reason and the token counts
            from the server's ``usage`` (requested via ``stream_options``).

        Raises:
            RuntimeError: On any HTTP failure or a connection that breaks
                mid-stream.
        """
        payload = self._payload(self._messages(prompt, system, ()), stream=True)
        payload["stream_options"] = {"include_usage": True}
        served: Optional[str] = None
        done_reason: Optional[str] = None
        usage: Dict[str, Any] = {}
        try:
            with requests.post(
                self.url(), headers=self._headers(), json=payload, stream=True,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                for raw in response.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    served = chunk.get("model") or served
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        if choice.get("finish_reason"):
                            done_reason = choice["finish_reason"]
                        text = (choice.get("delta") or {}).get("content") or ""
                        if text:
                            yield GenerationChunk(text=text)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI-compatible stream failed for model {self._model!r}: {exc}"
            ) from exc
        if done_reason not in (None, "stop"):
            logging.warning(
                "%s generation stopped early: finish_reason=%s", self._model, done_reason
            )
        yield GenerationChunk(
            text="",
            done=True,
            model=served or self._model,
            done_reason=done_reason,
            prompt_eval_count=usage.get("prompt_tokens"),
            eval_count=usage.get("completion_tokens"),
        )
