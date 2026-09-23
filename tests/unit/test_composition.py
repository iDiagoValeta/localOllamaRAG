"""Tests for the composition root assembling the multimodal pipeline.

``monkeygrab.composition`` is the single place that wires concrete adapters
into the ``Stack`` the use cases run on. The builders themselves must stay
cheap to call in the fast gate: no GPU, no model download, no Ollama server.
Every test here therefore runs against doubles -- stub adapter modules in
``sys.modules`` or a patched ``_isolated_python`` -- and asserts wiring, not
behavior of the adapters themselves.

Covered:

- ``build_stack`` delegates to the three builders and packs their results
  into a ``Stack`` in order.
- Each builder forwards the right config value to the right adapter
  (``config.paths`` to the FAISS store, the isolated interpreter to the
  jina-clip embedder, both inner extractors to the suffix router).
- ``_isolated_python`` raises ``FileNotFoundError`` with an actionable
  message naming both conventional ``.venv-mineru`` layouts when neither
  exists, and returns the existing candidate otherwise.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from monkeygrab import composition  # noqa: E402
from monkeygrab.config.app_config import AppConfig  # noqa: E402


def _install_stub_module(monkeypatch, name, **attrs):
    """Install a stub module under ``name`` in ``sys.modules``.

    Args:
        monkeypatch: The pytest fixture used to undo the install.
        name: Dotted module name to stub.
        **attrs: Attributes to set on the stub module.

    Returns:
        The created module object.
    """
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_build_stack_packs_each_builder_result_in_order(monkeypatch):
    """``build_stack`` returns a ``Stack`` of exactly what the builders built."""
    extractor, store, embedder = object(), object(), object()
    seen = {}

    def _extractor(config):
        seen["extractor"] = config
        return extractor

    def _store(config):
        seen["store"] = config
        return store

    def _embedder(config):
        seen["embedder"] = config
        return embedder

    monkeypatch.setattr(composition, "build_extractor", _extractor)
    monkeypatch.setattr(composition, "build_vector_store", _store)
    monkeypatch.setattr(composition, "build_embedder", _embedder)

    config = AppConfig()
    stack = composition.build_stack(config)

    assert isinstance(stack, composition.Stack)
    assert stack.extractor is extractor
    assert stack.vector_store is store
    assert stack.embedder is embedder
    assert seen == {"extractor": config, "store": config, "embedder": config}


def test_build_extractor_routes_pdfs_and_text_files(monkeypatch):
    """The suffix router owns both a MinerU and a text inner extractor."""
    mineru_instance = object()
    text_instance = object()
    captured = {}

    class _FakeMineru:
        def __init__(self):
            captured["mineru"] = self

    class _FakeText:
        def __init__(self):
            captured["text"] = self

    class _FakeBySuffix:
        def __init__(self, pdf, text):
            captured["pdf"] = pdf
            captured["text"] = text
            self._pdf = pdf
            self._text = text

    # Instances are identified by identity, not by the stub constructors
    # returning them, so replace __new__ to hand out the sentinels while
    # still recording that construction happened.
    _FakeMineru.__new__ = lambda cls: mineru_instance
    _FakeText.__new__ = lambda cls: text_instance

    _install_stub_module(
        monkeypatch,
        "monkeygrab.adapters.extraction.by_suffix_extractor",
        BySuffixExtractor=_FakeBySuffix,
    )
    _install_stub_module(
        monkeypatch,
        "monkeygrab.adapters.extraction.mineru_extractor",
        MineruExtractor=_FakeMineru,
    )
    _install_stub_module(
        monkeypatch,
        "monkeygrab.adapters.extraction.text_extractor",
        TextFileExtractor=_FakeText,
    )

    result = composition.build_extractor(AppConfig())

    assert isinstance(result, _FakeBySuffix)
    assert captured["pdf"] is mineru_instance
    assert captured["text"] is text_instance


def test_build_vector_store_receives_config_paths(monkeypatch):
    """The FAISS store is built from ``config.paths``, nothing else."""
    captured = {}

    class _FakeStore:
        def __init__(self, paths):
            captured["paths"] = paths

    _install_stub_module(
        monkeypatch,
        "monkeygrab.adapters.vectorstore.faiss_store",
        FaissVectorStore=_FakeStore,
    )

    config = AppConfig()
    result = composition.build_vector_store(config)

    assert isinstance(result, _FakeStore)
    assert captured["paths"] is config.paths


def test_build_embedder_forwards_the_isolated_interpreter(monkeypatch):
    """The embedder runs under ``.venv-mineru``, never the current interpreter."""
    captured = {}

    class _FakeEmbedder:
        def __init__(self, python_executable):
            captured["python"] = python_executable

    _install_stub_module(
        monkeypatch,
        "monkeygrab.adapters.embedding.jina_clip_embedder",
        JinaClipEmbedder=_FakeEmbedder,
    )
    monkeypatch.setattr(composition, "_isolated_python", lambda: "/fake/.venv-mineru/bin/python")

    result = composition.build_embedder(AppConfig())

    assert isinstance(result, _FakeEmbedder)
    assert captured["python"] == "/fake/.venv-mineru/bin/python"


def test_isolated_python_raises_actionable_error_when_venv_is_missing(monkeypatch):
    """A missing ``.venv-mineru`` fails loudly naming both expected layouts."""
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    with pytest.raises(FileNotFoundError, match=r"\.venv-mineru"):
        composition._isolated_python()

    try:
        composition._isolated_python()
    except FileNotFoundError as exc:
        message = str(exc)
    else:  # pragma: no cover - the assert above already failed
        message = ""
    assert "Scripts" in message
    assert "bin" in message


def test_isolated_python_prefers_the_first_existing_candidate(monkeypatch):
    """Whichever conventional layout exists wins, Windows layout first."""
    root = Path(composition.__file__).resolve().parents[2]
    windows = str(root / ".venv-mineru" / "Scripts" / "python.exe")
    posix = str(root / ".venv-mineru" / "bin" / "python")

    monkeypatch.setattr(Path, "is_file", lambda self: True)
    assert composition._isolated_python() == windows

    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == posix)
    assert composition._isolated_python() == posix
