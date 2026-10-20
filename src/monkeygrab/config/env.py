"""Environment-variable readers for AppConfig.from_env().

One hard-fail contract for both configuration paths: a variable that IS SET
to an unparseable value aborts startup instead of silently becoming a
default nobody asked for (docs/design/2026-07-26-monkeygrab-v2.md, "Politica
de fallos"). ``rag/chat_pdfs.py`` aliases the int/float readers from here, so
the web startup path and the core reject the same input identically. An
unset variable still yields the caller's default.
"""

import os

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def read_env_ollama_base_url() -> str:
    """Resolve the Ollama HTTP endpoint every component must talk to.

    ``OLLAMA_BASE_URL`` is this project's own name for it and the one
    ``.env.example`` documents. ``OLLAMA_HOST`` is Ollama's own name, honoured
    by the ``ollama`` client and by the web control panel's probes before this
    was wired, so it stays as a fallback rather than silently ceasing to work.

    Ollama documents ``OLLAMA_HOST`` without a scheme (``127.0.0.1:11434``),
    which the raw ``requests`` calls in the adapters would reject as an invalid
    URL. A missing scheme is filled in instead, and a trailing slash dropped,
    so that ``f"{base_url}/api/generate"`` is well-formed for every accepted
    spelling.

    Returns:
        The base URL, without a trailing slash.
    """
    raw = (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "").strip()
    if not raw:
        return DEFAULT_OLLAMA_BASE_URL
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def read_env_int(name: str, default: int) -> int:
    """Read an integer environment variable, or ``default`` if unset.

    Args:
        name: Environment variable name.
        default: Value to use when ``name`` is unset.

    Returns:
        The parsed integer.

    Raises:
        ValueError: If ``name`` is set but not a valid integer.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw!r}") from exc


def read_env_float(name: str, default: float) -> float:
    """Read a float environment variable, or ``default`` if unset.

    Args:
        name: Environment variable name.
        default: Value to use when ``name`` is unset.

    Returns:
        The parsed float.

    Raises:
        ValueError: If ``name`` is set but not a valid float.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {raw!r}") from exc


def read_env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable, or ``default`` if unset.

    Accepts ``true``/``false``, ``1``/``0``, ``yes``/``no``, ``y``/``n`` and
    ``on``/``off`` (case-insensitive, surrounding whitespace ignored), the
    spellings the rest of the codebase already uses for ad-hoc flag parsing.

    Args:
        name: Environment variable name.
        default: Value to use when ``name`` is unset.

    Returns:
        The parsed boolean.

    Raises:
        ValueError: If ``name`` is set but not a recognized boolean spelling.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


def read_env_str(name: str, default: str) -> str:
    """Read a string environment variable, or ``default`` if unset.

    Plain passthrough (no parsing can fail), kept alongside the int/float
    readers so every ``AppConfig.from_env`` field reads through one
    consistent ``env.read_env_*`` family instead of raw ``os.getenv``.

    A variable set to an empty or whitespace-only value counts as unset: that
    is how shell exports and ``.env`` files spell "not configured", and
    returning it would turn ``MONKEYGRAB_DATA_DIR=`` into the process cwd once
    ``os.path.abspath`` sees it (issue #356).

    Args:
        name: Environment variable name.
        default: Value to use when ``name`` is unset.

    Returns:
        The environment value, or ``default``.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw


def read_env_docs_root(default: str) -> str:
    """Resolve the root that holds the three language stores.

    The packaged desktop app points this at a per-user folder, because the
    bundle's own ``rag/docs`` sits inside the read-only install: upload,
    reindex and document deletion all write there (issue #344).

    An empty or whitespace-only value counts as unset, the same contract
    ``read_env_str`` documents -- otherwise ``MONKEYGRAB_DOCS_ROOT=`` would
    resolve to the process cwd.

    Args:
        default: Root to use when the variable is unset, normally
            ``<rag dir>/docs``.

    Returns:
        The absolute docs root.
    """
    return os.path.abspath(read_env_str("MONKEYGRAB_DOCS_ROOT", default))


def read_env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    """Read a string environment variable constrained to a fixed set of values.

    Args:
        name: Environment variable name.
        default: Value to use when ``name`` is unset. Must be one of ``choices``.
        choices: Every value accepted for this variable.

    Returns:
        The environment value, or ``default``.

    Raises:
        ValueError: If ``name`` is set to a value outside ``choices``.
    """
    raw = os.getenv(name, default)
    if raw not in choices:
        valid = ", ".join(choices)
        raise ValueError(f"Invalid value for {name}: {raw!r}. Valid: {valid}")
    return raw
