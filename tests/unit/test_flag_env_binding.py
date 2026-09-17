"""Unit tests for the USAR_* environment binding (round2 fix brief).

Pipeline flags default exactly as before; an explicit export wins for that
process only. What these tests pin:

- With the flag variables unset, ``AppConfig.from_env()`` reproduces the
  historical literals (same defaults the drift guard checks, asserted here
  directly so the binding itself is covered even where the engine is absent).
- With ``USAR_RERANKER=False`` (and every other flag variable), the binding
  lands on ``AppConfig.flags`` and on ``rag.chat_pdfs``'s own globals, so
  ``wiring.app_config_from_runtime()`` -- which overlays the latter onto the
  former -- cannot silently diverge between the two.
- Truthy/falsey spellings match the rest of the codebase; a set-but-invalid
  value hard-fails instead of silently becoming the default.

``rag.chat_pdfs`` is only ever imported in an isolated subprocess: importing
it here would bind this pytest process's own globals to whatever the ambient
shell exports, and ``rag.web.app``'s import-time settings load mutates them
further (see ``tests/unit/test_app_config_defaults.py``). The subprocess also
keeps this file importable in the dependency-free fast CI gate -- a missing
engine dependency skips the agreement checks instead of failing them.

Dependencies: stdlib only plus ``pytest`` and the pure ``monkeygrab.config``
layer -- this file runs in the fast gate's ``architecture`` job.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from monkeygrab.config import AppConfig, PipelineFlagsConfig  # noqa: E402
from monkeygrab.config import env  # noqa: E402

# (Flag env var, PipelineFlagsConfig field, historical default). One line per
# bound flag, so an added or dropped binding is a visible diff.
_FLAG_MAP = [
    ("USAR_CONTEXTUAL_RETRIEVAL", "usar_contextual_retrieval", True),
    ("USAR_LLM_QUERY_DECOMPOSITION", "usar_llm_query_decomposition", True),
    ("USAR_BUSQUEDA_HIBRIDA", "usar_busqueda_hibrida", True),
    ("USAR_RERANKER", "usar_reranker", True),
    ("EXPANDIR_CONTEXTO", "expandir_contexto", True),
    ("USAR_OPTIMIZACION_CONTEXTO", "usar_optimizacion_contexto", True),
    ("USAR_RECOMP_SYNTHESIS", "usar_recomp_synthesis", True),
    ("USAR_EMBEDDINGS_IMAGEN", "usar_embeddings_imagen", True),
    ("USAR_DESCRIPCION_IMAGEN", "usar_descripcion_imagen", False),
    ("LOGGING_METRICAS", "logging_metricas", True),
    ("GUARDAR_DEBUG_RAG", "guardar_debug_rag", True),
]

_FLAG_VARS = [var for var, _, _ in _FLAG_MAP]


@pytest.fixture
def clean_flag_env(monkeypatch):
    """Scrub every flag variable: these tests assert defaults and explicit
    exports, not whatever the developer's shell happens to have set."""
    for var in _FLAG_VARS:
        monkeypatch.delenv(var, raising=False)


def test_unset_env_keeps_historical_defaults(clean_flag_env):
    """No export anywhere: from_env() reproduces the historical literals."""
    assert AppConfig.from_env().flags == PipelineFlagsConfig()


@pytest.mark.parametrize("var,field,default", _FLAG_MAP)
def test_each_flag_binds_its_variable(monkeypatch, clean_flag_env, var, field, default):
    """Exporting the opposite of the default flips only that flag."""
    monkeypatch.setenv(var, "False" if default else "True")
    flags = AppConfig.from_env().flags
    assert getattr(flags, field) is (not default)
    for other_var, other_field, other_default in _FLAG_MAP:
        if other_field != field:
            assert getattr(flags, other_field) is other_default


def test_usar_reranker_false_leaves_everything_else_standing(monkeypatch, clean_flag_env):
    """The brief's concrete case: disabling BGE changes nothing else."""
    monkeypatch.setenv("USAR_RERANKER", "False")
    assert AppConfig.from_env().flags == PipelineFlagsConfig(usar_reranker=False)


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("yes", True),
        ("y", True),
        ("on", True),
        ("  no  ", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("no", False),
        ("n", False),
        ("off", False),
    ],
)
def test_bool_spellings(monkeypatch, clean_flag_env, spelling, expected):
    monkeypatch.setenv("USAR_RERANKER", spelling)
    assert AppConfig.from_env().flags.usar_reranker is expected


@pytest.mark.parametrize("raw", ["", "2", "maybe", "verdadero", "true!"])
def test_invalid_bool_hard_fails(monkeypatch, clean_flag_env, raw):
    """A set-but-unparseable flag aborts instead of silently becoming the default."""
    monkeypatch.setenv("USAR_RERANKER", raw)
    with pytest.raises(ValueError, match="USAR_RERANKER"):
        env.read_env_bool("USAR_RERANKER", True)
    with pytest.raises(ValueError, match="USAR_RERANKER"):
        AppConfig.from_env()


def _run_subprocess(script: str, extra_env: dict) -> dict:
    """Run ``script`` with only ``extra_env`` added to a flag-scrubbed environ."""
    environ = {k: v for k, v in os.environ.items() if k not in _FLAG_VARS}
    environ.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=environ,
    )
    if result.returncode != 0:
        if "ModuleNotFoundError" in result.stderr:
            pytest.skip(
                "rag.chat_pdfs cannot be imported here (missing engine "
                "dependency). Agreement check skipped, not passed."
            )
        raise AssertionError(f"subprocess failed:\n{result.stderr}")
    return json.loads(result.stdout)


_CHAT_PDFS_PREAMBLE = (
    "import json, sys\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
    "import rag.chat_pdfs as rag\n"
)


def test_chat_pdfs_globals_agree_with_from_env_on_explicit_export():
    """USAR_RERANKER=False in the environment reaches both configuration paths."""
    names = _FLAG_VARS
    script = (
        _CHAT_PDFS_PREAMBLE
        + f"print(json.dumps({{name: getattr(rag, name) for name in {names!r}}}))\n"
    )
    values = _run_subprocess(script, {"USAR_RERANKER": "False"})
    assert values["USAR_RERANKER"] is False
    for var in names:
        if var != "USAR_RERANKER":
            assert values[var] is (var != "USAR_DESCRIPCION_IMAGEN")


def test_chat_pdfs_globals_keep_defaults_when_env_unset():
    values = _run_subprocess(
        _CHAT_PDFS_PREAMBLE
        + f"print(json.dumps({{name: getattr(rag, name) for name in {_FLAG_VARS!r}}}))\n",
        {},
    )
    for var, _, default in _FLAG_MAP:
        assert values[var] is default


def test_app_config_from_runtime_agrees_with_from_env():
    """The wiring path headless actually reads sees the exported flag too."""
    script = (
        _CHAT_PDFS_PREAMBLE
        + "from monkeygrab.config import AppConfig\n"
        + "from rag.engine import wiring\n"
        + "print(json.dumps({\n"
        + '  "from_env": AppConfig.from_env().flags.usar_reranker,\n'
        + '  "runtime": wiring.app_config_from_runtime().flags.usar_reranker,\n'
        + "}))\n"
    )
    values = _run_subprocess(script, {"USAR_RERANKER": "False"})
    assert values == {"from_env": False, "runtime": False}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
