"""Persistence of the user's runtime choices: model roles, store, pipeline flags.

The web control panel writes these to ``settings.json`` in the writable data dir
so the desktop app reopens where the user left it. Both interfaces read it: a
choice made in the web UI is the configuration the CLI must run under too.
Otherwise the two sessions index and answer under different recipes over the
same store, and the index-fingerprint warning (issue #36) fires in a CLI session
that contains nothing to explain it and cannot fix it -- reindexing there would
rebuild under the CLI's own unsynchronised config.

Precedence is environment > settings.json > module defaults. A variable exported
for this run (or read from ``.env``) is an explicit statement about this run and
outranks a choice saved during an earlier one; everything the environment leaves
unset falls back to the persisted choice, and only then to the defaults in
rag/chat_pdfs.py. This is the precedence the README already documents for the
environment, now applied to persisted state as well.

Best-effort by design: a missing, unreadable or corrupt file leaves the defaults
standing instead of blocking startup, the same contract rag/engine/history.py
has. Persisted state is a convenience, never a source of truth for the pipeline.
"""

import json
import logging
import os
from typing import Any, Optional

from rag.engine.runtime import get_runtime

cfg = get_runtime()

# The three fixed language stores, each bound to rag/docs/<id>/. Lives here
# rather than in the web app because the CLI resolves a persisted store name too.
STORE_IDS = ("en", "es", "ca")

# Pipeline flags the control panel persists. The first two are index-time flags:
# they decide what an index contains and are part of the index fingerprint,
# which is why an interface observing a different value than the one the index
# was built under is a defect rather than a preference.
PERSISTED_FLAGS = (
    "USAR_CONTEXTUAL_RETRIEVAL",
    "USAR_EMBEDDINGS_IMAGEN",
    "USAR_LLM_QUERY_DECOMPOSITION",
    "USAR_BUSQUEDA_HIBRIDA",
    "USAR_RERANKER",
    "EXPANDIR_CONTEXTO",
    "USAR_OPTIMIZACION_CONTEXTO",
    "USAR_RECOMP_SYNTHESIS",
)

# Env var that pins the corpus. Set, it wins over any saved store.
_DOCS_FOLDER_ENV = "DOCS_FOLDER"


def ruta_ajustes() -> str:
    """Return the absolute path of the settings file in the writable data dir."""
    return os.path.join(cfg.DATA_DIR, "settings.json")


def resolver_carpeta_store(nombre: str) -> Optional[str]:
    """Resolve a store identifier to its docs folder.

    Args:
        nombre: Store identifier (``en``, ``es`` or ``ca``).

    Returns:
        Absolute path to ``rag/docs/<nombre>``, or ``None`` when the identifier
        is not one of the fixed stores or its folder is missing.
    """
    if nombre not in STORE_IDS:
        return None
    carpeta = os.path.join(cfg.BASE_DIR, "docs", nombre)
    return carpeta if os.path.isdir(carpeta) else None


def guardar_ajustes_persistidos() -> None:
    """Write the active model roles, store and pipeline flags to the settings file."""
    try:
        data = {
            "roles": cfg.get_model_roles(),
            "active_store": _nombre_store_activo(),
            "flags": {var: bool(getattr(cfg, var, False)) for var in PERSISTED_FLAGS},
        }
        ruta = ruta_ajustes()
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Error saving settings: {e}")


def cargar_ajustes_persistidos() -> None:
    """Apply the saved choices onto the runtime configuration.

    Roles are applied before the store, which re-derives the FAISS paths, and
    flags last. Entries the environment pins, and entries naming something this
    build does not have, are skipped.
    """
    try:
        # utf-8-sig tolerates a BOM if the file was hand-edited with a Windows tool.
        with open(ruta_ajustes(), "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return

    _aplicar_roles(data.get("roles"))
    _aplicar_store(data.get("active_store"))
    _aplicar_flags(data.get("flags"))


def _nombre_store_activo() -> str:
    """Return the store identifier of the active docs folder (its basename)."""
    return os.path.basename(os.path.abspath(cfg.CARPETA_DOCS))


def _fijado_por_entorno(rol: str) -> bool:
    """Whether the environment pins the model for ``rol``."""
    variable = cfg.MODEL_ROLE_ENV_VARS.get(rol, "")
    return bool(variable and os.getenv(variable))


def _aplicar_roles(roles: Any) -> None:
    """Apply saved model roles, skipping every role the environment pins."""
    if not isinstance(roles, dict):
        return
    overrides = {
        rol: modelo
        for rol, modelo in roles.items()
        if rol in cfg.MODEL_ROLE_VARS
        and isinstance(modelo, str)
        and modelo.strip()
        and not _fijado_por_entorno(rol)
    }
    if not overrides:
        return
    try:
        cfg.set_model_roles_runtime(overrides)
    except Exception as e:
        logging.warning(f"Error applying saved model roles: {e}")


def _aplicar_store(nombre: Any) -> None:
    """Apply the saved store unless DOCS_FOLDER pins the corpus for this run."""
    if not isinstance(nombre, str) or not nombre or os.getenv(_DOCS_FOLDER_ENV):
        return
    if nombre == _nombre_store_activo():
        return
    carpeta = resolver_carpeta_store(nombre)
    if carpeta is None:
        return
    try:
        cfg.set_docs_folder_runtime(carpeta)
    except Exception as e:
        logging.warning(f"Error applying saved store: {e}")


def _aplicar_flags(flags: Any) -> None:
    """Apply saved pipeline flags, ignoring any name outside PERSISTED_FLAGS."""
    if not isinstance(flags, dict):
        return
    for var, valor in flags.items():
        if var in PERSISTED_FLAGS:
            setattr(cfg, var, bool(valor))
