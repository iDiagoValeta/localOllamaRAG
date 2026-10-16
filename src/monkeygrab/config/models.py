"""Model roles used by the multimodal pipeline."""

from dataclasses import dataclass

from monkeygrab.config.env import DEFAULT_OLLAMA_BASE_URL


@dataclass(frozen=True)
class OllamaRuntimeConfig:
    rag_num_ctx: int = 16384
    query_num_ctx: int = 2048
    recomp_num_ctx: int = 8192
    contextual_num_ctx: int = 32768
    request_timeout: int = 900
    # Wall-clock cap on one generation call, in seconds; 0 leaves only the
    # read timeout. A streamed generation that never stops keeps every byte
    # of its read timeout alive, so nothing else bounds it (issue #249).
    generation_deadline: int = 0
    keep_alive: int = 120
    generate_retries: int = 2
    generate_retry_delay: int = 3
    base_url: str = DEFAULT_OLLAMA_BASE_URL


# The two chat backends a role can run on. "ollama" is the product's own
# server; "openai" is any server speaking the OpenAI chat API (llama-server,
# Daimon's model gateway, LM Studio, vLLM). Validated by read_env_choice so a
# typo fails at startup instead of silently keeping Ollama.
CHAT_BACKENDS = ("ollama", "openai")


# Selectable embedders. "gemma2" is embeddinggemma-2 behind an OpenAI-compatible
# llama-server and the default (better recall and MRR than jina-clip-v2 on the
# gold set; CPU-friendly). "jina" is jina-clip-v2, run in the isolated interpreter.
EMBEDDERS = ("jina", "gemma2")

# Matryoshka widths embeddinggemma-2 can be truncated to. An index built at one
# width is a different vector space from another, so the width is part of both
# the store path and the index fingerprint; an unsupported value must abort
# rather than silently change the geometry.
EMBEDDING_DIMENSIONS = (768, 512, 256, 128)


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    """Where and how the HTTP embedder (embeddinggemma-2) is reached."""

    base_url: str = "http://127.0.0.1:8091"
    dimensions: int = 512
    timeout: int = 120


@dataclass(frozen=True)
class OpenAICompatRuntimeConfig:
    """Connection settings shared by every role that runs on the OpenAI backend."""

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    timeout: int = 900


@dataclass(frozen=True)
class ModelsConfig:
    """Ollama generation roles plus the selectable embedder."""

    rag: str = "gemma4:e4b"
    chat: str = "gemma4:e4b"
    contextual: str = "gemma4:e4b"
    recomp: str = "gemma4:e4b"
    desc: str = "gemma4"
    ollama: OllamaRuntimeConfig = OllamaRuntimeConfig()
    rag_backend: str = "ollama"
    chat_backend: str = "ollama"
    contextual_backend: str = "ollama"
    recomp_backend: str = "ollama"
    openai: OpenAICompatRuntimeConfig = OpenAICompatRuntimeConfig()
    embedder: str = "gemma2"
    embedding: EmbeddingRuntimeConfig = EmbeddingRuntimeConfig()


def infer_model_description(model_name: str) -> str:
    return model_name.split(":")[0]


def embedding_display_name(embedder: str) -> str:
    """Return the embedder's model name for metrics and the UI.

    Args:
        embedder: Embedder key from ``EMBEDDERS``.

    Returns:
        The version-less model name shown to users; unknown keys fall back to
        the jina name, which is the product default.
    """
    if embedder == "gemma2":
        return "google/embeddinggemma-2"
    return "jinaai/jina-clip-v2"
