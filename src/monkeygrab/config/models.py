"""Model roles used by the multimodal pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaRuntimeConfig:
    num_ctx: int = 8192
    rag_num_ctx: int = 16384
    aux_num_ctx: int = 8192
    query_num_ctx: int = 2048
    recomp_num_ctx: int = 8192
    contextual_num_ctx: int = 32768
    request_timeout: int = 900
    keep_alive: int = 0
    generate_retries: int = 2
    generate_retry_delay: int = 3


@dataclass(frozen=True)
class ModelsConfig:
    """Ollama generation roles; jina-clip-v2 is the fixed embedder."""

    rag: str = "gemma4:e4b"
    chat: str = "gemma4:e4b"
    contextual: str = "gemma4:e4b"
    recomp: str = "gemma4:e4b"
    desc: str = "gemma4"
    ollama: OllamaRuntimeConfig = OllamaRuntimeConfig()


def infer_model_description(model_name: str) -> str:
    return model_name.split(":")[0]
