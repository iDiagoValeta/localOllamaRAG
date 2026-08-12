"""Shared settings.json load/apply logic for the CLI and the web app.

The web control panel persists model roles and pipeline flags to
``settings.json`` under the writable data dir (see ``rag/web/app.py``'s
``_save_persisted_settings``) so the desktop app remembers a user's choices
across restarts. Before this module existed, the CLI never read that file: it
always started from ``rag.chat_pdfs``'s hardcoded defaults, so a contextual
model or an index-time flag changed in the web UI silently diverged from what
the CLI ran with (issue #36's follow-up from PR #59's review). Both
interfaces now go through the functions here so there is exactly one
implementation of "what does a persisted setting mean" instead of two that can
drift.

Model-role precedence is NOT the same for both callers, and that asymmetry is
deliberate, not an oversight -- see ``apply_model_roles``'s ``prefer_env_var``
parameter. For the CLI (``prefer_env_var=True``, the default), strongest
first:

1. An explicit process environment variable (``OLLAMA_RAG_MODEL`` and the
   other roles in ``MODEL_ROLE_ENV_VARS``) -- ".claude/CLAUDE.md" section 3
   says the process environment always wins over the defaults, and from the
   CLI's point of view a persisted settings.json choice is just as implicit a
   source as the hardcoded default: it is a file some OTHER process wrote,
   not a decision this invocation made.
2. The persisted ``settings.json`` value, when the corresponding env var was
   not set.
3. The module's hardcoded default (the second argument of each ``os.getenv``
   call in ``rag/chat_pdfs.py``), when neither of the above applies.

For the web app (``prefer_env_var=False``), settings.json always wins over
the env var. It is not restoring a remembered default there -- it is
resuming the app's OWN prior state: the user picked that role through this
same control panel, which never consulted the environment either (``POST
/api/models`` calls ``set_model_roles_runtime`` unconditionally). Applying
env-var precedence there would silently revert an explicit UI choice on
every restart, and the next time anything else got saved,
``_save_persisted_settings`` would overwrite the file with that reverted
value -- permanently erasing the user's choice with no warning. Caught in
review, not a hypothetical: any repo config with `.env` supplying an
``OLLAMA_*_MODEL`` reaches this path every time (``rag/chat_pdfs.py``'s
``load_dotenv(override=False)`` puts a `.env` value in ``os.environ`` exactly
like an exported shell variable, so this module cannot tell the two apart,
nor should it need to).

Pipeline flags have no environment-variable layer at all today (they are
plain hardcoded booleans in ``rag/chat_pdfs.py``), so their precedence is only
steps 2 and 3, identically for both callers.

The active vector store (``active_store`` in settings.json) is deliberately
NOT handled here. Adopting it would change which corpus the CLI opens purely
because the web UI was last pointed at a different language store -- a much
bigger behavioural change than adopting a model role or a flag, and one the
CLI has no picker for and no way to signal. The CLI keeps following
``DOCS_FOLDER`` / the default corpus, exactly as before; only the web app
still reads and writes ``active_store``, inline in ``rag/web/app.py``.

Neither interface writes back through this module: the CLI has no runtime
control to change a role or a flag, so there is nothing for it to persist,
and reading the file once at startup and never again means a web app running
concurrently cannot race the CLI over it.
"""

import json
import os
from typing import Any, Dict

SETTINGS_FILENAME = "settings.json"


def settings_path(rag_engine) -> str:
    """Absolute path to the persisted-settings file for the active data dir.

    Both interfaces resolve this the same way because both read
    ``rag_engine.DATA_DIR`` -- ``rag.chat_pdfs``'s own resolution of
    ``MONKEYGRAB_DATA_DIR`` (the packaged desktop app's per-user data root) or
    the package directory in a source/dev run. Whichever process wrote the
    file, this is where the other looks for it.

    Args:
        rag_engine: The ``rag.chat_pdfs`` module (or a stand-in exposing the
            same ``DATA_DIR`` attribute).

    Returns:
        The absolute path to ``settings.json``.
    """
    return os.path.join(rag_engine.DATA_DIR, SETTINGS_FILENAME)


def load_settings(rag_engine) -> Dict[str, Any]:
    """Read and parse settings.json, tolerating a missing or corrupt file.

    This is a startup diagnostic input, not a pipeline step: neither the CLI
    prompt nor the web app's ``/api/init`` may be blocked by a file that does
    not exist yet, was hand-edited into invalid JSON, or was only partially
    written by a crashed process -- same precedent as
    ``rag.engine.indexing.index_fingerprint_mismatch``, which never lets a
    diagnostic take down startup with it.

    Args:
        rag_engine: The ``rag.chat_pdfs`` module (or a stand-in).

    Returns:
        The parsed settings dict, or ``{}`` if the file is absent, unreadable,
        not valid JSON, or valid JSON that is not a JSON object.
    """
    try:
        # utf-8-sig tolerates a BOM if the file was hand-edited with a Windows tool.
        with open(settings_path(rag_engine), "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def apply_model_roles(rag_engine, data: Dict[str, Any], *, prefer_env_var: bool = True) -> None:
    """Apply persisted model roles.

    Args:
        rag_engine: The ``rag.chat_pdfs`` module (or a stand-in exposing
            ``MODEL_ROLE_VARS``, ``MODEL_ROLE_ENV_VARS`` and
            ``set_model_roles_runtime``).
        data: The dict returned by ``load_settings``.
        prefer_env_var: When ``True`` (the CLI's case, and the default), a
            role is skipped if its backing environment variable
            (``MODEL_ROLE_ENV_VARS``) is set. When ``False`` (the web app's
            case), the persisted value always applies -- see this module's
            docstring for why the two callers need different answers here.

    Uses the sanctioned ``set_model_roles_runtime`` switch rather than
    ``setattr``, matching the web control panel.
    """
    roles = data.get("roles")
    if not isinstance(roles, dict):
        return

    def _blocked_by_env(role: str) -> bool:
        if not prefer_env_var:
            return False
        return bool(os.getenv(rag_engine.MODEL_ROLE_ENV_VARS.get(role, "")))

    valid = {
        role: model for role, model in roles.items()
        if role in rag_engine.MODEL_ROLE_VARS
        and isinstance(model, str) and model.strip()
        and not _blocked_by_env(role)
    }
    if not valid:
        return
    try:
        rag_engine.set_model_roles_runtime(valid)
    except Exception:
        return

    # set_model_roles_runtime recomputes MODELO_DESC from MODELO_RAG whenever
    # "rag" is among the applied roles. Only the CLI's env-preferring path
    # restores an explicit MODELO_DESC override afterwards: the web app's
    # restore is the app's own prior state (see prefer_env_var above), and
    # MODELO_DESC has no persisted counterpart to conflict with in the first
    # place.
    if prefer_env_var and "rag" in valid:
        explicit_desc = os.getenv("MODELO_DESC")
        if explicit_desc:
            rag_engine.MODELO_DESC = explicit_desc


def apply_pipeline_flags(rag_engine, data: Dict[str, Any]) -> None:
    """Apply persisted pipeline flags, index-time and inference-time alike.

    No environment variable backs these flags today, so precedence is just
    the persisted choice over the module's hardcoded default -- identically
    for the CLI and the web app. Only names listed in
    ``rag_engine.PERSISTABLE_FLAG_VARS`` are applied, so an unrelated or
    renamed key in a stale settings.json is ignored rather than creating a
    new global.

    Args:
        rag_engine: The ``rag.chat_pdfs`` module (or a stand-in exposing
            ``PERSISTABLE_FLAG_VARS``).
        data: The dict returned by ``load_settings``.
    """
    flags = data.get("flags")
    if not isinstance(flags, dict):
        return

    for var, val in flags.items():
        if var in rag_engine.PERSISTABLE_FLAG_VARS:
            setattr(rag_engine, var, bool(val))
