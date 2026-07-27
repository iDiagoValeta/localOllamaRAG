"""Immutable configuration -- AppConfig and its 7 sub-configs.

``AppConfig.from_env()`` reproduces ``rag/chat_pdfs.py`` section 3's
defaults exactly (see ``tests/unit/test_app_config_defaults.py``).
``AppConfig.with_overrides(**changes)`` is the immutable replacement for
``set_pipeline_flags``/``set_model_roles_runtime``/``set_docs_folder_runtime``.
"""

from monkeygrab.config.app_config import AppConfig
from monkeygrab.config.chunking import ChunkingConfig
from monkeygrab.config.context import ContextConfig
from monkeygrab.config.flags import PipelineFlagsConfig
from monkeygrab.config.models import ModelsConfig, OllamaRuntimeConfig
from monkeygrab.config.paths import PathsConfig
from monkeygrab.config.reranking import RerankingConfig
from monkeygrab.config.retrieval import RetrievalConfig

__all__ = [
    "AppConfig",
    "ChunkingConfig",
    "ContextConfig",
    "ModelsConfig",
    "OllamaRuntimeConfig",
    "PathsConfig",
    "PipelineFlagsConfig",
    "RerankingConfig",
    "RetrievalConfig",
]
