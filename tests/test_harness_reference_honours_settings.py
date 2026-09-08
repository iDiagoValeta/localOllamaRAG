"""The harness's reference config must describe the product as configured
(issue: the loop measured a configuration nobody runs).

``harness.cli._build_reference`` built its reference with
``AppConfig.from_env()``, which reads the environment and otherwise falls back
to ``rag/chat_pdfs.py``'s module defaults. It never read ``settings.json`` --
the file the web UI writes and the app applies at startup. So on a machine
whose saved choice is one model, the loop would search for improvements around
a different one, and around whatever pipeline flags the defaults happen to
carry.

Lives here rather than in ``harness/tests/`` because it imports the engine
stack (``rag.engine.wiring``), which the dependency-free CI job cannot load.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from harness import cli  # noqa: E402


@pytest.fixture(autouse=True)
def restore_runtime_globals():
    """Undo what cargar_ajustes_persistidos writes.

    It applies onto ``rag.chat_pdfs``'s module-level globals, which outlive a
    test. Without this, a test that saves a role leaks it into the next one --
    and the leak is invisible, because the next test then reads a plausible
    value that simply came from the wrong place.
    """
    import rag.chat_pdfs as cfg

    names = [n for n in dir(cfg) if n.startswith(("MODELO_", "USAR_", "CARPETA_DOCS"))]
    saved = {n: getattr(cfg, n) for n in names}
    yield
    for name, value in saved.items():
        setattr(cfg, name, value)


@pytest.fixture
def saved_settings(tmp_path, monkeypatch):
    """Point the settings module at a throwaway file and return a writer."""
    import rag.chat_pdfs  # noqa: F401
    from rag.engine import settings as settings_mod

    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_mod, "ruta_ajustes", lambda: str(path))

    def write(payload):
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return write


def test_reference_uses_the_saved_model_roles(saved_settings, monkeypatch):
    monkeypatch.delenv("OLLAMA_RAG_MODEL", raising=False)
    saved_settings({"roles": {"rag": "saved-model:latest"}})

    reference = cli._build_reference()

    assert reference.models.rag == "saved-model:latest"


def test_the_environment_still_outranks_the_saved_role(saved_settings, monkeypatch):
    # Precedence is environment > settings.json > module defaults (AGENTS.md
    # section 1, rule 7). A campaign launched with an exported model must
    # measure that model, not the one the UI last saved.
    #
    # The global is set here as well as the variable, because that is the real
    # sequence: a launch exports OLLAMA_RAG_MODEL *before* the process starts,
    # so rag.chat_pdfs binds it at import. Setting only the variable would test
    # a state that cannot occur -- the environment changing after import -- and
    # would fail for that reason rather than for a defect.
    import rag.chat_pdfs as cfg

    monkeypatch.setenv("OLLAMA_RAG_MODEL", "pinned-by-env:latest")
    monkeypatch.setattr(cfg, "MODELO_RAG", "pinned-by-env:latest")
    saved_settings({"roles": {"rag": "saved-model:latest"}})

    reference = cli._build_reference()

    assert reference.models.rag == "pinned-by-env:latest"


def test_reference_uses_the_saved_pipeline_flags(saved_settings, monkeypatch):
    # Not only models: the saved flags change what is indexed and what runs
    # per query, so a reference that ignores them measures a different
    # pipeline from the one the user has.
    saved_settings({"flags": {"USAR_CONTEXTUAL_RETRIEVAL": False}})

    reference = cli._build_reference()

    assert reference.flags.usar_contextual_retrieval is False


def test_reference_still_builds_with_no_settings_file(tmp_path, monkeypatch):
    # A clean clone has no settings.json; the reference must fall back to the
    # module defaults rather than raising.
    import rag.chat_pdfs  # noqa: F401
    from rag.engine import settings as settings_mod

    monkeypatch.setattr(settings_mod, "ruta_ajustes", lambda: str(tmp_path / "absent.json"))
    monkeypatch.delenv("OLLAMA_RAG_MODEL", raising=False)

    reference = cli._build_reference()

    assert reference.models.rag
