"""Tests for the persisted settings both interfaces read (issue #36 follow-up).

The web control panel saves model roles, the active store and the pipeline
flags; the CLI used to ignore that file and start from rag/chat_pdfs.py's
defaults. Two of those flags and one of those roles are part of the index
fingerprint, so the divergence had a visible symptom: a CLI session warning that
the stored index no longer matches the active configuration, with nothing in
that session to explain it and no way to fix it -- reindexing there would have
rebuilt the store under the CLI's own unsynchronised configuration.

What these tests pin is the resulting contract: the file is applied, the
environment still outranks it, and a startup path never dies on it.

Imports rag.chat_pdfs (and rag.cli through the entry-point test), so the
dependency-free fast CI gate skips this file -- see tests/conftest.py.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rag.chat_pdfs as rag_engine
from rag.engine import settings

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "rag" / "docs"
CHAT_PDFS = ROOT / "rag" / "chat_pdfs.py"

# Reproduces `python rag/chat_pdfs.py` -- the command the README documents --
# inside a subprocess: the file is executed under the name __main__, which is
# what makes the module-identity question below real rather than theoretical.
# The trailing __main__ guard is stripped so the CLI is launched explicitly,
# with a double in place of the real one.
_DIRECT_RUN_DRIVER = r'''
import io, json, os, sys, types

sys.path.insert(0, ROOT_PLACEHOLDER)
sys.path.insert(0, SRC_PLACEHOLDER)

source = io.open(CHAT_PDFS_PLACEHOLDER, encoding="utf-8").read()
source = source.replace('if __name__ == "__main__":\n    main()', "")
main_module = types.ModuleType("__main__")
main_module.__file__ = CHAT_PDFS_PLACEHOLDER
sys.modules["__main__"] = main_module
exec(compile(source, CHAT_PDFS_PLACEHOLDER, "exec"), main_module.__dict__)

class _CLIDouble:
    def __init__(self, engine):
        self.engine = engine

    def run(self):
        from rag.engine import settings
        print("RESULT " + json.dumps({
            "shares_engine_config": self.engine is settings.cfg,
            "contextual": self.engine.MODELO_CONTEXTUAL,
            "store": os.path.basename(self.engine.CARPETA_DOCS),
        }))

import rag.cli
rag.cli.MonkeyGrabCLI = _CLIDouble
main_module.main()
'''


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """Pin every global the loader mutates and point it at an empty data dir.

    ``set_model_roles_runtime`` and ``set_docs_folder_runtime`` really do
    reassign rag.chat_pdfs's globals -- they are the mechanism under test, not
    something to double -- so each one is pinned through monkeypatch to be
    restored afterwards. The ambient environment is cleared for the same reason:
    these tests assert the precedence rule, not the developer's shell.
    """
    monkeypatch.setattr(rag_engine, "DATA_DIR", str(tmp_path))
    for var in rag_engine.MODEL_ROLE_VARS.values():
        monkeypatch.setattr(rag_engine, var, "default-model")
    monkeypatch.setattr(rag_engine, "MODELO_DESC", "default-model")
    for flag in settings.PERSISTED_FLAGS:
        monkeypatch.setattr(rag_engine, flag, True)
    monkeypatch.setattr(rag_engine, "CARPETA_DOCS", str(DOCS_ROOT / "en"))
    monkeypatch.setattr(rag_engine, "PATH_DB", str(tmp_path / "vector_db" / "en_jina_clip"))
    monkeypatch.setattr(rag_engine, "COLLECTION_NAME", "docs_en")
    for variable in rag_engine.MODEL_ROLE_ENV_VARS.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("DOCS_FOLDER", raising=False)
    return tmp_path


def _save(data_dir, **content):
    """Write a settings file with exactly the keys a test cares about."""
    (Path(data_dir) / "settings.json").write_text(
        json.dumps(content, ensure_ascii=False), encoding="utf-8"
    )


def test_saved_roles_store_and_flags_are_applied(data_dir):
    _save(
        data_dir,
        roles={"contextual": "saved-contextual"},
        active_store="es",
        flags={"USAR_CONTEXTUAL_RETRIEVAL": False},
    )

    settings.cargar_ajustes_persistidos()

    assert rag_engine.MODELO_CONTEXTUAL == "saved-contextual"
    assert os.path.basename(rag_engine.CARPETA_DOCS) == "es"
    assert rag_engine.COLLECTION_NAME == "docs_es"
    assert rag_engine.USAR_CONTEXTUAL_RETRIEVAL is False


def test_the_environment_outranks_a_saved_role(data_dir, monkeypatch):
    """An exported OLLAMA_*_MODEL survives; roles it does not name still follow the file.

    The contextual model is part of the index recipe: silently replacing an
    explicitly declared one would index the corpus under a configuration the run
    never asked for, which is the failure the fingerprint exists to report.
    """
    monkeypatch.setenv("OLLAMA_CONTEXTUAL_MODEL", "env-contextual")
    monkeypatch.setattr(rag_engine, "MODELO_CONTEXTUAL", "env-contextual")
    _save(data_dir, roles={"contextual": "saved-contextual", "rag": "saved-rag"})

    settings.cargar_ajustes_persistidos()

    assert rag_engine.MODELO_CONTEXTUAL == "env-contextual"
    assert rag_engine.MODELO_RAG == "saved-rag"


def test_docs_folder_pins_the_corpus_over_the_saved_store(data_dir, monkeypatch):
    monkeypatch.setenv("DOCS_FOLDER", str(DOCS_ROOT / "en"))
    _save(data_dir, active_store="ca")

    settings.cargar_ajustes_persistidos()

    assert os.path.basename(rag_engine.CARPETA_DOCS) == "en"
    assert rag_engine.COLLECTION_NAME == "docs_en"


def test_no_settings_file_leaves_the_defaults_standing(data_dir):
    settings.cargar_ajustes_persistidos()

    assert rag_engine.MODELO_CONTEXTUAL == "default-model"
    assert rag_engine.USAR_CONTEXTUAL_RETRIEVAL is True
    assert os.path.basename(rag_engine.CARPETA_DOCS) == "en"


@pytest.mark.parametrize("payload", ["{not json", '["a", "list"]'])
def test_an_unusable_settings_file_never_blocks_startup(data_dir, payload):
    """Both interfaces read this file before serving a single request."""
    (Path(data_dir) / "settings.json").write_text(payload, encoding="utf-8")

    settings.cargar_ajustes_persistidos()

    assert rag_engine.MODELO_CONTEXTUAL == "default-model"
    assert rag_engine.USAR_CONTEXTUAL_RETRIEVAL is True


def test_names_outside_the_persisted_set_are_ignored(data_dir, monkeypatch):
    """A hand-edited file cannot invent a role, a flag or a store.

    LOGGING_METRICAS is the interesting one: it is a real engine flag that the
    control panel deliberately does not expose, so accepting any name found in
    the file would let the file reach further than the UI that writes it.
    """
    monkeypatch.setattr(rag_engine, "LOGGING_METRICAS", True)
    _save(
        data_dir,
        roles={"unknown_role": "x"},
        active_store="zz",
        flags={"USAR_INVENTADO": True, "LOGGING_METRICAS": False},
    )

    settings.cargar_ajustes_persistidos()

    assert not hasattr(rag_engine, "USAR_INVENTADO")
    assert rag_engine.LOGGING_METRICAS is True
    assert os.path.basename(rag_engine.CARPETA_DOCS) == "en"


def test_saving_then_loading_reproduces_the_active_configuration(data_dir):
    rag_engine.set_model_roles_runtime({"contextual": "chosen-contextual"})
    rag_engine.set_docs_folder_runtime(str(DOCS_ROOT / "ca"))
    rag_engine.USAR_RECOMP_SYNTHESIS = False

    settings.guardar_ajustes_persistidos()

    rag_engine.set_model_roles_runtime({"contextual": "something-else"})
    rag_engine.set_docs_folder_runtime(str(DOCS_ROOT / "en"))
    rag_engine.USAR_RECOMP_SYNTHESIS = True

    settings.cargar_ajustes_persistidos()

    assert rag_engine.MODELO_CONTEXTUAL == "chosen-contextual"
    assert os.path.basename(rag_engine.CARPETA_DOCS) == "ca"
    assert rag_engine.USAR_RECOMP_SYNTHESIS is False


def test_cli_startup_runs_under_the_saved_configuration(data_dir, monkeypatch):
    """The CLI entry point applies the file before the CLI reads any of it.

    This is the divergence itself: without the call in ``main()`` the CLI opens
    the index the web app built and answers over it under different models and
    flags.
    """
    _save(
        data_dir,
        roles={"contextual": "saved-contextual"},
        flags={"USAR_CONTEXTUAL_RETRIEVAL": False},
    )
    seen = {}

    class _CLIDouble:
        def __init__(self, engine):
            seen["engine"] = engine

        def run(self):
            seen["contextual"] = rag_engine.MODELO_CONTEXTUAL
            seen["contextual_flag"] = rag_engine.USAR_CONTEXTUAL_RETRIEVAL

    import rag.cli

    monkeypatch.setattr(rag.cli, "MonkeyGrabCLI", _CLIDouble)

    rag_engine.main()

    assert seen["engine"] is rag_engine
    assert seen["contextual"] == "saved-contextual"
    assert seen["contextual_flag"] is False


def test_running_the_script_directly_gives_the_cli_the_engine_s_own_config(tmp_path):
    """`python rag/chat_pdfs.py` must hand the CLI the module the engine reads.

    Run as a script this file is ``__main__``; importing it by name would build
    a second module object with its own globals. The engine modules bind the
    ``__main__`` one, so a CLI holding the other copy would list PDFs, print its
    startup panel and pass a docs folder to indexing from a configuration that
    retrieval and generation never saw -- with the saved store applied to one
    side only, that indexes one corpus into another's store.
    """
    (tmp_path / "settings.json").write_text(
        json.dumps({"roles": {"contextual": "saved-contextual"}, "active_store": "ca"}),
        encoding="utf-8",
    )
    driver = (
        _DIRECT_RUN_DRIVER
        .replace("ROOT_PLACEHOLDER", repr(str(ROOT)))
        .replace("SRC_PLACEHOLDER", repr(str(ROOT / "src")))
        .replace("CHAT_PDFS_PLACEHOLDER", repr(str(CHAT_PDFS)))
    )
    env = {k: v for k, v in os.environ.items() if k not in rag_engine.MODEL_ROLE_ENV_VARS.values()}
    env.pop("DOCS_FOLDER", None)
    env["MONKEYGRAB_DATA_DIR"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300, env=env,
    )

    if result.returncode != 0 and "ModuleNotFoundError" in result.stderr:
        pytest.skip(
            "rag.chat_pdfs cannot be imported here (missing engine dependency). "
            f"Comparison skipped, not passed:\n{result.stderr.strip()[-300:]}"
        )
    assert result.returncode == 0, result.stderr[-2000:]
    reported = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert reported, f"driver printed no result:\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}"
    observed = json.loads(reported[-1][len("RESULT "):])

    assert observed == {
        "shares_engine_config": True,
        "contextual": "saved-contextual",
        "store": "ca",
    }


def test_every_role_declares_the_variable_that_pins_it():
    """A role missing from MODEL_ROLE_ENV_VARS would silently lose its env override."""
    assert set(rag_engine.MODEL_ROLE_ENV_VARS) == set(rag_engine.MODEL_ROLE_VARS)


def test_role_env_vars_name_the_getenv_calls_that_read_them():
    """MODEL_ROLE_ENV_VARS must match the variables rag/chat_pdfs.py actually reads.

    A stale name here fails invisibly: the loader would test a variable nobody
    sets, conclude the role is unpinned, and overwrite an explicitly exported
    OLLAMA_*_MODEL with the saved one. Parsed rather than executed so the check
    reads the source of the defaults, not the mutable module state a running
    session may already have changed.
    """
    read_from = {}
    for node in ast.parse(CHAT_PDFS.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Attribute) and func.attr == "getenv") or not node.value.args:
            continue
        first = node.value.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                read_from[target.id] = first.value

    expected = {
        rag_engine.MODEL_ROLE_VARS[role]: variable
        for role, variable in rag_engine.MODEL_ROLE_ENV_VARS.items()
    }
    actual = {module_var: read_from.get(module_var) for module_var in expected}
    assert actual == expected
