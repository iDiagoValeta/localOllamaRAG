"""Immutable configuration -- AppConfig and its 7 sub-configs.

``AppConfig.from_env()`` reproduces ``rag/chat_pdfs.py`` section 3's
defaults exactly (see ``tests/unit/test_app_config_defaults.py``).
``AppConfig.with_overrides(**changes)`` is the immutable replacement for
``set_pipeline_flags``/``set_model_roles_runtime``/``set_docs_folder_runtime``.

Only the names consumers actually import through the package are re-exported
here; the individual sub-configs live in their own modules.
"""

from monkeygrab.config import env
from monkeygrab.config.app_config import AppConfig

__all__ = ["AppConfig", "env"]
