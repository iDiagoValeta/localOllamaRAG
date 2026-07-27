"""Tests for selecting a backend by configuration.

This is what turns the ports from an interface exercise into something usable:
if a technology cannot be swapped by configuration, comparing two of them means
editing wiring code, which is expensive enough that the comparison never happens.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.config.stack import (  # noqa: E402
    EMBEDDER_JINA_CLIP,
    EMBEDDER_OLLAMA,
    EXTRACTOR_MINERU,
    EXTRACTOR_PYMUPDF,
    VECTOR_STORE_CHROMA,
    VECTOR_STORE_FAISS,
    StackConfig,
    stack_from_env,
)


def test_unset_environment_selects_the_current_production_stack(monkeypatch):
    """An unset environment must not change what the app does today."""
    for var in ("PDF_EXTRACTOR", "VECTOR_STORE", "EMBEDDER"):
        monkeypatch.delenv(var, raising=False)

    stack = stack_from_env()

    assert stack.extractor == EXTRACTOR_PYMUPDF
    assert stack.vector_store == VECTOR_STORE_CHROMA
    assert stack.embedder == EMBEDDER_OLLAMA
    assert not stack.is_multimodal


def test_each_selector_is_read_from_its_own_variable(monkeypatch):
    monkeypatch.setenv("PDF_EXTRACTOR", EXTRACTOR_MINERU)
    monkeypatch.setenv("VECTOR_STORE", VECTOR_STORE_FAISS)
    monkeypatch.setenv("EMBEDDER", EMBEDDER_JINA_CLIP)

    stack = stack_from_env()

    assert stack.extractor == EXTRACTOR_MINERU
    assert stack.vector_store == VECTOR_STORE_FAISS
    assert stack.embedder == EMBEDDER_JINA_CLIP
    assert stack.is_multimodal


def test_stacks_can_be_mixed(monkeypatch):
    """Backends are independent: MinerU with Chroma is a legitimate combination.

    The comparison this exists for needs one variable changed at a time, so the
    selectors must not be secretly coupled to each other.
    """
    monkeypatch.setenv("PDF_EXTRACTOR", EXTRACTOR_MINERU)
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    monkeypatch.delenv("EMBEDDER", raising=False)

    stack = stack_from_env()

    assert stack.extractor == EXTRACTOR_MINERU
    assert stack.vector_store == VECTOR_STORE_CHROMA
    assert not stack.is_multimodal


@pytest.mark.parametrize(
    "variable",
    ["PDF_EXTRACTOR", "VECTOR_STORE", "EMBEDDER"],
)
def test_an_unimplemented_backend_raises_instead_of_falling_back(monkeypatch, variable):
    """Coercing an unknown value to the default is how the previous
    configuration hid the fact that its MinerU path did not exist at all: it
    documented PDF_EXTRACTOR=mineru while no such code existed, and asking for
    it silently produced pymupdf output.
    """
    monkeypatch.setenv(variable, "something-that-does-not-exist")

    with pytest.raises(ValueError, match=variable):
        stack_from_env()


def test_different_stacks_have_different_slugs():
    """Two stacks produce vectors of different dimension and meaning, so their
    indexes must never resolve to the same directory.
    """
    old = StackConfig()
    new = StackConfig(
        extractor=EXTRACTOR_MINERU,
        vector_store=VECTOR_STORE_FAISS,
        embedder=EMBEDDER_JINA_CLIP,
    )

    assert old.slug != new.slug
    assert EXTRACTOR_MINERU in new.slug and EMBEDDER_JINA_CLIP in new.slug


def test_stack_config_is_immutable():
    stack = StackConfig()
    with pytest.raises(Exception):
        stack.extractor = EXTRACTOR_MINERU  # type: ignore[misc]
