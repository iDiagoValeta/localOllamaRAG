"""The setup script's reporting, which is the part that can be wrong quietly.

Issue #120. Building the environments is `pip install` lines; what needs a test
is what the script says afterwards, because a check that reports the wrong
state is worse than no check -- it sends someone to rebuild an environment that
is fine, or tells them a broken one is ready.

One distinction carries the design and is asserted here: **missing is not the
same as not-built-here.** Ollama is the one component this script does not
install; a missing model is a `pull` away and is reported with that command,
not as a setup failure.

Standard library only, like the script itself, so this runs in the fast gate's
dependency-free `architecture` job.
"""

import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import setup_environments as setup


class TestInterpreterLayout:
    def test_prefers_the_windows_layout_when_it_exists(self, tmp_path):
        scripts = tmp_path / "Scripts"
        scripts.mkdir()
        (scripts / "python.exe").write_text("", encoding="utf-8")
        assert setup._venv_python(tmp_path).name == "python.exe"

    def test_falls_back_to_the_posix_layout(self, tmp_path):
        assert setup._venv_python(tmp_path) == tmp_path / "bin" / "python"


def _installed_models(monkeypatch, names):
    """Double Ollama's ``/api/tags`` with the models it would report."""
    payload = json.dumps({"models": [{"name": name} for name in names]}).encode("utf-8")

    class _Response:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: _Response())


def _no_saved_settings(monkeypatch, tmp_path):
    """Point the data dir at an empty directory, so no settings.json is found.

    Without this the tests below read the repo's own ``rag/settings.json`` and
    assert against whatever this machine happens to have saved.
    """
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", str(tmp_path))


def _saved_settings(monkeypatch, tmp_path, roles):
    """Write a settings.json holding ``roles`` and point the data dir at it."""
    monkeypatch.setenv("MONKEYGRAB_DATA_DIR", str(tmp_path))
    (tmp_path / "settings.json").write_text(
        json.dumps({"roles": roles, "active_store": "ca"}), encoding="utf-8"
    )


class TestConfiguredModels:
    def test_defaults_to_one_model_for_all_four_roles(self, monkeypatch, tmp_path):
        _no_saved_settings(monkeypatch, tmp_path)
        for var in setup.OLLAMA_ROLE_VARS.values():
            monkeypatch.delenv(var, raising=False)
        assert setup._configured_models() == [setup.DEFAULT_OLLAMA_MODEL]

    def test_the_environment_decides_and_duplicates_collapse(self, monkeypatch, tmp_path):
        _no_saved_settings(monkeypatch, tmp_path)
        for var in setup.OLLAMA_ROLE_VARS.values():
            monkeypatch.setenv(var, "small:model")
        monkeypatch.setenv("OLLAMA_RAG_MODEL", "big:model")
        assert setup._configured_models() == ["big:model", "small:model"]


class TestSavedRolesArePartOfThePrecedence:
    """Issue #215. The check must resolve roles the way the product does.

    Reading environment-then-module-default skipped the middle term and was
    wrong in both directions: it named a model this machine would never load,
    and it stayed silent about a saved role that was not pulled.
    """

    def test_the_saved_roles_are_what_gets_checked(self, monkeypatch, tmp_path):
        _saved_settings(monkeypatch, tmp_path, {role: "saved:model" for role in setup.OLLAMA_ROLE_VARS})
        for var in setup.OLLAMA_ROLE_VARS.values():
            monkeypatch.delenv(var, raising=False)
        assert setup._configured_models() == ["saved:model"]

    def test_the_environment_still_outranks_a_saved_role(self, monkeypatch, tmp_path):
        _saved_settings(monkeypatch, tmp_path, {role: "saved:model" for role in setup.OLLAMA_ROLE_VARS})
        monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_CONTEXTUAL_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_RECOMP_MODEL", raising=False)
        monkeypatch.setenv("OLLAMA_RAG_MODEL", "pinned:model")
        assert setup._configured_models() == ["pinned:model", "saved:model"]

    def test_a_role_saved_without_a_value_falls_through_to_the_default(
        self, monkeypatch, tmp_path
    ):
        _saved_settings(monkeypatch, tmp_path, {"rag": "  ", "chat": "saved:model"})
        for var in setup.OLLAMA_ROLE_VARS.values():
            monkeypatch.delenv(var, raising=False)
        assert setup._configured_models() == [setup.DEFAULT_OLLAMA_MODEL, "saved:model"]

    def test_a_corrupt_settings_file_leaves_the_defaults_standing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MONKEYGRAB_DATA_DIR", str(tmp_path))
        (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
        for var in setup.OLLAMA_ROLE_VARS.values():
            monkeypatch.delenv(var, raising=False)
        assert setup._configured_models() == [setup.DEFAULT_OLLAMA_MODEL]

    def test_a_saved_role_that_is_not_pulled_is_reported_missing(self, monkeypatch, tmp_path):
        """The false negative: green while the run is going to fail at first use."""
        _saved_settings(monkeypatch, tmp_path, {role: "saved:model" for role in setup.OLLAMA_ROLE_VARS})
        for var in setup.OLLAMA_ROLE_VARS.values():
            monkeypatch.delenv(var, raising=False)
        _installed_models(monkeypatch, ["something:else"])

        passed, message = setup._check_ollama()
        assert not passed
        assert "ollama pull saved:model" in message

    def test_a_saved_role_that_is_pulled_is_reported_present(self, monkeypatch, tmp_path):
        """The false positive: told to pull gigabytes it already has."""
        _saved_settings(monkeypatch, tmp_path, {role: "saved:model" for role in setup.OLLAMA_ROLE_VARS})
        for var in setup.OLLAMA_ROLE_VARS.values():
            monkeypatch.delenv(var, raising=False)
        _installed_models(monkeypatch, ["saved:model"])

        passed, message = setup._check_ollama()
        assert passed
        assert setup.DEFAULT_OLLAMA_MODEL not in message


class TestSettingsLocationDrift:
    """The settings path is duplicated here; this is what stops it drifting.

    ``_settings_path`` cannot import the product to ask where the file lives --
    the fast gate's architecture job runs on the standard library alone. So the
    rule is copied, and copied rules go stale silently. Reading the product's
    own line back is the cheapest thing that fails loudly when it moves.
    """

    def test_the_product_still_derives_its_data_dir_the_way_this_script_assumes(self):
        source = (REPO_ROOT / "rag" / "chat_pdfs.py").read_text(encoding="utf-8")
        assert 'DATA_DIR = os.path.abspath(os.getenv("MONKEYGRAB_DATA_DIR") or BASE_DIR)' in source
        assert "BASE_DIR = os.path.dirname(os.path.abspath(__file__))" in source

    def test_the_product_still_defaults_every_role_to_the_model_this_script_names(self):
        source = (REPO_ROOT / "rag" / "chat_pdfs.py").read_text(encoding="utf-8")
        for var in setup.OLLAMA_ROLE_VARS.values():
            assert f'os.getenv("{var}", "{setup.DEFAULT_OLLAMA_MODEL}")' in source

    def test_the_path_lands_on_the_repo_settings_file_by_default(self, monkeypatch):
        monkeypatch.delenv("MONKEYGRAB_DATA_DIR", raising=False)
        assert setup._settings_path() == REPO_ROOT / "rag" / "settings.json"


class TestReporting:
    """``run_checks`` decides FAIL vs warn, and that decision is the point."""

    def _with_checks(self, monkeypatch, results):
        monkeypatch.setattr(
            setup,
            "_check_interpreter",
            lambda venv, label: results.get(label, (True, f"{label}: fine")),
        )
        for name in ("_check_cuda", "_check_mineru_binary", "_check_embedder", "_check_ollama"):
            monkeypatch.setattr(setup, name, lambda name=name: results[name])

    def test_everything_present_exits_zero(self, monkeypatch, capsys):
        self._with_checks(monkeypatch, {
            "_check_cuda": (True, "CUDA: a card"),
            "_check_mineru_binary": (True, "MinerU: 3.4.5"),
            "_check_embedder": (True, "Embedder: reachable at http://embedder.example/health"),
            "_check_ollama": (True, "Ollama: reachable"),
        })
        assert setup.run_checks() == 0
        assert "All components present" in capsys.readouterr().out

    def test_a_missing_ollama_model_is_a_warning_carrying_the_pull_command(
        self, monkeypatch, capsys
    ):
        self._with_checks(monkeypatch, {
            "_check_cuda": (True, "CUDA: a card"),
            "_check_mineru_binary": (True, "MinerU: 3.4.5"),
            "_check_embedder": (True, "Embedder: reachable at http://embedder.example/health"),
            "_check_ollama": (False, "Ollama: reachable, missing model(s) ... ollama pull x:y"),
        })
        assert setup.run_checks() == 0
        assert "ollama pull x:y" in capsys.readouterr().out

    def test_a_stopped_embedder_is_a_warning_not_a_claim_that_all_is_present(
        self, monkeypatch, capsys
    ):
        """Issue #385: the check said 'All components present' without asking."""
        self._with_checks(monkeypatch, {
            "_check_cuda": (True, "CUDA: a card"),
            "_check_mineru_binary": (True, "MinerU: 3.4.5"),
            "_check_embedder": (
                False,
                "Embedder: llama-server is not reachable at http://embedder.example/health",
            ),
            "_check_ollama": (True, "Ollama: reachable"),
        })
        assert setup.run_checks() == 0
        out = capsys.readouterr().out
        assert "Embedder" in out
        assert "All components present" not in out
        assert "warning(s) above" in out

    def test_a_genuinely_missing_component_fails_and_exits_one(self, monkeypatch, capsys):
        self._with_checks(monkeypatch, {
            "_check_cuda": (False, "CUDA: not available to .venv-mineru"),
            "_check_mineru_binary": (True, "MinerU: 3.4.5"),
            "_check_embedder": (True, "Embedder: reachable at http://embedder.example/health"),
            "_check_ollama": (True, "Ollama: reachable"),
        })
        assert setup.run_checks() == 1
        assert "1 component(s) missing" in capsys.readouterr().out


class TestEmbedderCheck:
    """Issue #385: a prerequisite this script does not build still gets checked."""

    def _config(self, monkeypatch, base_url, images=False):
        config = SimpleNamespace(
            models=SimpleNamespace(embedding=SimpleNamespace(base_url=base_url)),
            flags=SimpleNamespace(usar_embeddings_imagen=images),
        )
        monkeypatch.setattr(setup, "AppConfig", SimpleNamespace(from_env=lambda: config))

    def _record_urlopen(self, monkeypatch, response=None, error=None):
        calls = []

        def fake_urlopen(url, timeout):
            calls.append((url, timeout))
            if error is not None:
                raise error
            return response

        monkeypatch.setattr(setup.urllib.request, "urlopen", fake_urlopen)
        return calls

    def test_the_health_url_comes_from_appconfig_with_a_short_timeout(self, monkeypatch):
        monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://from-env.example:1111")
        self._config(monkeypatch, "http://from-config.example:2222/")

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        calls = self._record_urlopen(monkeypatch, _Response())

        passed, message = setup._check_embedder()

        assert passed
        assert calls[0][0] == "http://from-config.example:2222/health"
        assert calls[0][1] == setup._EMBEDDER_HEALTH_TIMEOUT_SECONDS
        assert calls[0][1] <= 5
        assert "http://from-config.example:2222/health" in message

    def test_a_non_200_answer_is_a_failure_naming_the_endpoint(self, monkeypatch):
        self._config(monkeypatch, "http://embedder.example:9000")
        error = urllib.error.HTTPError(
            "http://embedder.example:9000/health", 503, "Service Unavailable", {}, None
        )
        self._record_urlopen(monkeypatch, error=error)

        passed, message = setup._check_embedder()

        assert not passed
        assert "http://embedder.example:9000/health" in message
        assert "503" in message

    def test_an_unreachable_embedder_says_how_to_start_llama_server(self, monkeypatch):
        self._config(monkeypatch, "http://embedder.example:9000")
        self._record_urlopen(monkeypatch, error=urllib.error.URLError("Connection refused"))

        passed, message = setup._check_embedder()

        assert not passed
        assert "http://embedder.example:9000/health" in message
        assert "llama-server" in message
        assert "-b 2048 -ub 2048" in message
        assert "--mmproj mmproj-embeddinggemma-2-Q8_0.gguf" in message

    def _serve(self, monkeypatch, props):
        """Double ``/health`` (200) and ``/props`` (``props`` as JSON)."""
        calls = []

        class _Reply:
            status = 200

            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(url, timeout):
            calls.append(url)
            return _Reply(json.dumps(props).encode("utf-8"))

        monkeypatch.setattr(setup.urllib.request, "urlopen", fake_urlopen)
        return calls

    def test_a_server_with_vision_passes_when_image_embeddings_are_on(self, monkeypatch):
        self._config(monkeypatch, "http://embedder.example:9000", images=True)
        calls = self._serve(monkeypatch, {"modalities": {"vision": True}})

        passed, _message = setup._check_embedder()

        assert passed
        assert calls == [
            "http://embedder.example:9000/health",
            "http://embedder.example:9000/props",
        ]

    def test_a_server_without_vision_fails_when_image_embeddings_are_on(self, monkeypatch):
        self._config(monkeypatch, "http://embedder.example:9000", images=True)
        self._serve(monkeypatch, {"modalities": {"vision": False}})

        passed, message = setup._check_embedder()

        assert not passed
        assert "--mmproj mmproj-embeddinggemma-2-Q8_0.gguf" in message
        assert "USAR_EMBEDDINGS_IMAGEN" in message

    def test_a_server_without_vision_passes_when_image_embeddings_are_off(self, monkeypatch):
        self._config(monkeypatch, "http://embedder.example:9000", images=False)
        calls = self._serve(monkeypatch, {"modalities": {"vision": False}})

        passed, _message = setup._check_embedder()

        assert passed
        assert calls == ["http://embedder.example:9000/health"]


class TestSurvivingWhatItDiagnoses:
    """A diagnostic must outlive the broken install it is describing.

    Found by re-reading the script rather than by a failure: every check
    shells out to something that may be half-installed, and only
    ``TimeoutExpired`` was caught. Anything else -- an ``OSError`` from a
    binary that is not executable, a ``JSONDecodeError`` from a truncated
    reply -- aborted the whole run and left the remaining components
    unreported, which is exactly the state the script exists to describe.
    """

    def test_one_check_raising_does_not_abort_the_others(self, monkeypatch, capsys):
        monkeypatch.setattr(setup, "_check_interpreter", lambda venv, label: (True, f"{label}: ok"))
        monkeypatch.setattr(setup, "_check_cuda", lambda: (_ for _ in ()).throw(OSError("boom")))
        monkeypatch.setattr(setup, "_check_mineru_binary", lambda: (True, "MinerU: 3.4.5"))
        monkeypatch.setattr(setup, "_check_ollama", lambda: (True, "Ollama: reachable"))

        assert setup.run_checks() == 1
        out = capsys.readouterr().out
        assert "CUDA: raised OSError: boom" in out
        # The checks after the raising one still ran and still reported.
        assert "MinerU: 3.4.5" in out
        assert "Ollama: reachable" in out

    def test_a_binary_that_prints_no_version_is_still_reported_as_present(
        self, monkeypatch, tmp_path
    ):
        """It used to raise IndexError off an empty splitlines()."""
        binary = tmp_path / "bin" / "mineru"
        binary.parent.mkdir(parents=True)
        binary.write_text("", encoding="utf-8")
        monkeypatch.setattr(setup, "ISOLATED_VENV", tmp_path)

        class _Silent:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: _Silent())
        passed, message = setup._check_mineru_binary()
        assert passed
        assert "printed nothing" in message

    def test_a_binary_that_exits_nonzero_is_not_present(self, monkeypatch, tmp_path):
        """Issue #385: an existing file that cannot run was reported as MinerU."""
        binary = tmp_path / "bin" / "mineru"
        binary.parent.mkdir(parents=True)
        binary.write_text("", encoding="utf-8")
        monkeypatch.setattr(setup, "ISOLATED_VENV", tmp_path)

        class _Failed:
            returncode = 2
            stdout = ""
            stderr = "ImportError: libcudart.so.12: cannot open shared object file"

        monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: _Failed())
        passed, message = setup._check_mineru_binary()
        assert not passed
        assert "MinerU" in message
        assert "2" in message
        assert "ImportError" in message
