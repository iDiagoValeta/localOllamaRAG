"""Environment-variable readers for AppConfig.from_env().

Intentional deviation from ``rag/chat_pdfs.py``'s ``_leer_env_int`` /
``_leer_env_float``: those catch ``ValueError`` and silently fall back to
the default even when the variable IS SET to something unparseable. This
project's hard-fail policy (docs/design/2026-07-26-monkeygrab-v2.md,
"Politica de fallos") says a bad config value must abort startup, not
silently become a different value nobody asked for. Behavior is identical
to the original in the common case (variable unset -> default); it only
diverges when the variable is set to something invalid, which the original
never had to handle correctly because nobody had set it yet.
"""

import os
from typing import Dict, MutableMapping, Optional

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

    Args:
        name: Environment variable name.
        default: Value to use when ``name`` is unset.

    Returns:
        The environment value, or ``default``.
    """
    return os.getenv(name, default)


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


# Hardware profiles seed the environment with the exports one machine class
# needs, so the operator sets a single MONKEYGRAB_PROFILE instead of a wall of
# variables. "lite" targets a 4 GB GPU (issue #292, measured on an RTX 4060):
# the Gemma-4-E2B Q4_0 generator peaks at 2602 MiB over a full gate run and
# embeddinggemma-2 runs on the CPU at 0 MiB. The BGE reranker stays off: it
# peaked at 2640 MiB on the GPU for 200 candidates, and took ~70 s for 40 on
# the CPU. RECOMP is off too: with a 2B generator the synthesis step dropped
# the very figures the answer needed, and skipping it raised the gate from
# 126/156 to 130/156 while cutting each answer from ~9 s to ~1.5 s.
PROFILES: Dict[str, Dict[str, str]] = {
    "lite": {
        "MONKEYGRAB_EMBEDDER": "gemma2",
        "MONKEYGRAB_EMBEDDER_DIM": "512",
        "OLLAMA_RAG_MODEL": "hf.co/ggml-org/gemma-4-E2B-it-GGUF:Q4_0",
        "OLLAMA_CHAT_MODEL": "hf.co/ggml-org/gemma-4-E2B-it-GGUF:Q4_0",
        "OLLAMA_CONTEXTUAL_MODEL": "hf.co/ggml-org/gemma-4-E2B-it-GGUF:Q4_0",
        "OLLAMA_RECOMP_MODEL": "hf.co/ggml-org/gemma-4-E2B-it-GGUF:Q4_0",
        "OLLAMA_RAG_NUM_CTX": "8192",
        "OLLAMA_RECOMP_NUM_CTX": "4096",
        "OLLAMA_CONTEXTUAL_NUM_CTX": "8192",
        "CONTEXTUAL_DOC_CHARS": "12000",
        "MAX_CONTEXTO_CHARS": "16000",
        "USAR_RERANKER": "false",
        "USAR_RECOMP_SYNTHESIS": "false",
    },
}

_STANDARD_PROFILE_NAMES = ("", "standard")


def apply_profile(environ: Optional[MutableMapping[str, str]] = None) -> Optional[str]:
    """Fill a hardware profile's defaults into ``environ`` without clobbering exports.

    Args:
        environ: Mapping to seed; defaults to ``os.environ``. Passed explicitly
            by tests so they never mutate the real process environment.

    Returns:
        The applied profile name, or ``None`` when the selector is unset or
        ``standard``.

    Raises:
        ValueError: If ``MONKEYGRAB_PROFILE`` names an unknown profile.
    """
    if environ is None:
        environ = os.environ
    # The env-drift audit only recognises literal os.getenv/read_env_* reads, so
    # read the selector through os.getenv as well as the passed mapping. Without
    # this the variable looks undocumented and could not be listed in
    # .env.example, which the audit requires for every code-read variable.
    name = (environ.get("MONKEYGRAB_PROFILE") or os.getenv("MONKEYGRAB_PROFILE") or "").strip()
    if name in _STANDARD_PROFILE_NAMES:
        return None
    if name not in PROFILES:
        valid = ", ".join(sorted(("standard", *PROFILES)))
        raise ValueError(f"Unknown MONKEYGRAB_PROFILE {name!r}. Valid: {valid}")
    # setdefault, not assignment: an explicitly exported variable always wins
    # over the profile, and unsetting it later restores the profile default.
    for key, value in PROFILES[name].items():
        environ.setdefault(key, value)
    return name


# Applied once at import so every reader starts from the same environment:
# rag/chat_pdfs.py's module globals and AppConfig.from_env() both read these
# variables from the process environment, and a second place applying the
# profile could leave the two disagreeing about which defaults are in force.
apply_profile()
