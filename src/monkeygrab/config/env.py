"""Environment-variable readers for AppConfig.from_env().

# ─────────────────────────────────────────────
# SECTION 1: READERS
# ─────────────────────────────────────────────

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
from typing import Optional


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
