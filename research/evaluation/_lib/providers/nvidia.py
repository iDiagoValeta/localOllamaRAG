"""NVIDIA NIM (OpenAI-compatible) judge configurator for RAGAS.

Required environment:
    NVIDIA_API_KEY
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any, Callable

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    Embeddings = object  # type: ignore[misc,assignment]

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CHAT_MODEL = "mistralai/mistral-medium-3.5-128b"
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_REASONING_EFFORT = "auto"
MISTRAL_SMALL_MODEL_ID = "mistralai/mistral-small-4-119b-2603"


class NvidiaEmbeddings(Embeddings):
    """NVIDIA embedding wrapper with required input_type for asymmetric models."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int,
        max_retries: int,
        rate_limiter: Any,
        query_input_type: str,
        document_input_type: str,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.rate_limiter = rate_limiter
        self.query_input_type = query_input_type
        self.document_input_type = document_input_type

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        self.rate_limiter.acquire(blocking=True)
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            extra_body={"input_type": input_type},
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.document_input_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.query_input_type)[0]


def _is_mistral_small_model(model: str) -> bool:
    return model.lower().strip() == MISTRAL_SMALL_MODEL_ID


def resolve_reasoning_effort(args: argparse.Namespace) -> str | None:
    if args.nvidia_reasoning_effort != "auto":
        return args.nvidia_reasoning_effort
    if _is_mistral_small_model(args.nvidia_model):
        return "none"
    return None


def _build_extra_body(args: argparse.Namespace) -> dict[str, Any] | None:
    extra_body: dict[str, Any] = {}
    reasoning_effort = resolve_reasoning_effort(args)
    if reasoning_effort is not None:
        extra_body["reasoning_effort"] = reasoning_effort
    return extra_body or None


def build_nvidia_configurator(args: argparse.Namespace) -> Callable:
    """Return a ``(google_timeout, google_retries) -> (llm, embeddings)`` callable."""
    def configurar_llm_nvidia(google_timeout=None, google_retries=None):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            print("NVIDIA_API_KEY not found in environment.")
            raise SystemExit(1)

        try:
            from langchain_core.rate_limiters import InMemoryRateLimiter
            from langchain_openai import ChatOpenAI
            from ragas.llms.base import LangchainLLMWrapper
        except ImportError as err:
            print(f"Error: {err}")
            print("Install with: pip install langchain-openai ragas")
            raise SystemExit(1) from err

        class NvidiaChatOpenAI(ChatOpenAI):
            """ChatOpenAI variant that emits NVIDIA NIM-compatible payload keys."""

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                if "max_completion_tokens" in payload:
                    payload["max_tokens"] = payload.pop("max_completion_tokens")
                payload.pop("n", None)
                return payload

        requests_per_second = max(args.nvidia_rate_limit_per_minute, 1) / 60.0
        limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=args.nvidia_rate_check_seconds,
            max_bucket_size=1,
        )

        extra_body = _build_extra_body(args)
        chat_kwargs: dict[str, Any] = {
            "model": args.nvidia_model,
            "api_key": api_key,
            "base_url": args.nvidia_base_url,
            "temperature": args.nvidia_temperature,
            "top_p": args.nvidia_top_p,
            "timeout": args.nvidia_timeout,
            "max_retries": args.nvidia_max_retries,
            "max_tokens": args.nvidia_max_tokens,
            "rate_limiter": limiter,
        }
        if extra_body is not None:
            chat_kwargs["extra_body"] = extra_body

        raw_eval_llm = NvidiaChatOpenAI(**chat_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LangchainLLMWrapper is deprecated.*",
                category=DeprecationWarning,
            )
            eval_llm = LangchainLLMWrapper(raw_eval_llm, bypass_n=True)

        eval_embeddings = None
        if args.nvidia_embedding_model.lower() != "none":
            eval_embeddings = NvidiaEmbeddings(
                api_key=api_key,
                base_url=args.nvidia_base_url,
                model=args.nvidia_embedding_model,
                timeout=args.nvidia_timeout,
                max_retries=args.nvidia_max_retries,
                rate_limiter=limiter,
                query_input_type=args.nvidia_embedding_query_input_type,
                document_input_type=args.nvidia_embedding_document_input_type,
            )

        print(f"Evaluation LLM: NVIDIA {args.nvidia_model}")
        print(
            "Evaluation embeddings: "
            + ("disabled" if eval_embeddings is None else f"NVIDIA {args.nvidia_embedding_model}")
        )
        print(f"NVIDIA base_url: {args.nvidia_base_url}")
        print(f"Shared API rate limit: {args.nvidia_rate_limit_per_minute} calls/minute")
        print(
            "NVIDIA request config: "
            f"max_tokens={args.nvidia_max_tokens}, "
            f"reasoning_effort={resolve_reasoning_effort(args) or 'default'}, "
            f"bypass_n=True"
        )
        print(
            "RAGAS throughput config: "
            f"workers={args.ragas_max_workers}, "
            f"batch_size={args.ragas_batch_size or 'auto'}"
        )
        return eval_llm, eval_embeddings

    return configurar_llm_nvidia


def add_nvidia_args(parser: argparse.ArgumentParser) -> None:
    """Attach NVIDIA-specific CLI flags to an argparse parser."""
    parser.add_argument("--nvidia-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--nvidia-embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--nvidia-embedding-query-input-type", default="query")
    parser.add_argument("--nvidia-embedding-document-input-type", default="passage")
    parser.add_argument("--nvidia-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--nvidia-temperature", type=float, default=0.0)
    parser.add_argument("--nvidia-top-p", type=float, default=1.0)
    parser.add_argument("--nvidia-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--nvidia-reasoning-effort",
        choices=("auto", "none", "high"),
        default=DEFAULT_REASONING_EFFORT,
        help="NVIDIA reasoning_effort. auto disables reasoning for Mistral Small.",
    )
    parser.add_argument("--nvidia-timeout", type=int, default=120)
    parser.add_argument("--nvidia-max-retries", type=int, default=3)
    parser.add_argument("--nvidia-rate-limit-per-minute", type=int, default=40)
    parser.add_argument("--nvidia-rate-check-seconds", type=float, default=0.25)
