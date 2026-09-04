"""The web entry point applies settings.json before it serves anything.

Until the terminal interface was removed, this contract was pinned on the CLI
side: ``chat_pdfs.main()`` called ``cargar_ajustes_persistidos()`` before
handing the engine to the CLI, and a test drove that entry point to prove it.
The web app carries the same call at import time (``rag/web/app.py``), but
nothing checked it -- every test in ``tests/unit/test_persisted_settings.py``
calls the loader itself, so deleting the call from ``app.py`` left the whole
suite green.

The divergence that call prevents is not hypothetical: the control panel writes
the file, and two of the flags plus one of the roles it holds are part of the
index fingerprint. An interface that starts from the module defaults answers
over an index built under someone else's recipe, and reports a fingerprint
warning it gives the user no way to explain.

Runs in a subprocess on purpose. Importing ``rag.web.app`` applies the real
settings.json to ``rag.chat_pdfs``'s module globals, which are shared across
the whole session -- doing that in-process is how this suite acquired
cross-test flakiness before (issue #96).

Lives in ``tests/`` rather than ``tests/unit/`` because it imports the web
stack, which the dependency-free fast CI gate does not install.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "rag" / "docs"

# Importing the Flask module is the whole subject: the call under test runs at
# import time, so the probe never touches the loader itself.
_PROBE = """
import json
import os

import rag.web.app  # noqa: F401
import rag.chat_pdfs as engine

print("RESULT " + json.dumps({
    "contextual": engine.MODELO_CONTEXTUAL,
    "recomp": engine.USAR_RECOMP_SYNTHESIS,
    "store": os.path.basename(engine.CARPETA_DOCS),
}))
"""


def _run_probe(data_dir):
    """Import the web app in a clean interpreter and report the config it got."""
    env = dict(os.environ)
    env["MONKEYGRAB_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    # The environment outranks the file by design, so an exported role or
    # folder here would mask exactly the failure this test looks for.
    for variable in list(env):
        if variable.startswith("OLLAMA_") or variable == "DOCS_FOLDER":
            del env[variable]

    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("RESULT ")),
        None,
    )
    assert line is not None, f"probe printed no result:\n{result.stdout}\n{result.stderr}"
    return json.loads(line[len("RESULT "):])


def test_importing_the_web_app_applies_the_saved_configuration(tmp_path):
    """Roles, active store and flags come from the file, not the defaults."""
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "roles": {"contextual": "saved-contextual"},
                "active_store": "ca",
                "flags": {"USAR_RECOMP_SYNTHESIS": False},
            }
        ),
        encoding="utf-8",
    )

    got = _run_probe(tmp_path)

    assert got["contextual"] == "saved-contextual"
    assert got["store"] == "ca"
    assert got["recomp"] is False


def test_a_missing_settings_file_still_starts(tmp_path):
    """No file is the first-run case, not an error: defaults, no exception."""
    got = _run_probe(tmp_path)

    assert got["contextual"] != "saved-contextual"
