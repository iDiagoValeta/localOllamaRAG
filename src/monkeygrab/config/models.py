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


@dataclass(frozen=True)
class OpenAICompatRuntimeConfig:
    """Connection settings shared by every role that runs on the OpenAI backend."""

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    timeout: int = 900


@dataclass(frozen=True)
class ModelsConfig:
    """Ollama generation roles; jina-clip-v2 is the fixed embedder."""

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


def infer_model_description(model_name: str) -> str:
    return model_name.split(":")[0]
