"""The run artifact's conditions block (issue #222) -- what decided a pass
rate, kept apart from what it decided.

No ``rag`` or ``monkeygrab.adapters`` import here: every helper under test
takes what it needs as an argument or reaches it through
``importlib.metadata``/``subprocess``, never the engine. That keeps this file
collected (not silently skipped by ``tests/conftest.py``) in the
dependency-free fast gate, the same way test_summary_split.py and
test_preflight_model_tags.py are.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import run_eval  # noqa: E402
from monkeygrab.config.app_config import AppConfig  # noqa: E402


# _package_version


def test_an_installed_package_reports_a_version_string():
    # pytest itself is guaranteed installed -- this is the test runner.
    version = run_eval._package_version("pytest")
    assert isinstance(version, str) and version


def test_a_missing_package_reports_none():
    assert run_eval._package_version("definitely-not-a-real-package-222") is None


# _gpu_info


def test_gpu_info_degrades_to_none_when_torch_absent_and_nvidia_smi_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def _raise(*_a, **_kw):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("subprocess.run", _raise)

    assert run_eval._gpu_info() == {"name": None, "vram_total_mib": None}


def test_gpu_info_parses_nvidia_smi_when_torch_is_not_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    class _Result:
        stdout = "NVIDIA GeForce RTX 4090, 24564\n"

    monkeypatch.setattr("subprocess.run", lambda *_a, **_kw: _Result())

    assert run_eval._gpu_info() == {"name": "NVIDIA GeForce RTX 4090", "vram_total_mib": 24564}


def test_gpu_info_prefers_an_already_imported_torch_over_nvidia_smi(monkeypatch):
    class _Props:
        name = "NVIDIA A100"
        total_memory = 42 * 2**20  # bytes -> 42 MiB, chosen to be checkable exactly

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_properties(_index):
            return _Props()

    class _FakeTorch:
        cuda = _Cuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    def _fail_if_called(*_a, **_kw):
        raise AssertionError("nvidia-smi must not run when torch already answered")

    monkeypatch.setattr("subprocess.run", _fail_if_called)

    assert run_eval._gpu_info() == {"name": "NVIDIA A100", "vram_total_mib": 42}


def test_gpu_info_is_none_when_torch_is_loaded_but_reports_no_cuda(monkeypatch):
    class _Cuda:
        @staticmethod
        def is_available():
            return False

    class _FakeTorch:
        cuda = _Cuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    assert run_eval._gpu_info() == {"name": None, "vram_total_mib": None}


# _git_info


def test_git_info_reports_a_clean_tree(monkeypatch):
    def _fake_run(args, **_kw):
        class _Result:
            stdout = "abc1234\n" if "rev-parse" in args else ""

        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)

    assert run_eval._git_info() == {"hash": "abc1234", "dirty": False}


def test_git_info_reports_a_dirty_tree(monkeypatch):
    def _fake_run(args, **_kw):
        class _Result:
            stdout = "abc1234\n" if "rev-parse" in args else " M rag/chat_pdfs.py\n"

        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)

    assert run_eval._git_info() == {"hash": "abc1234", "dirty": True}


def test_git_info_degrades_to_none_when_git_is_unavailable(monkeypatch):
    def _raise(*_a, **_kw):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr("subprocess.run", _raise)

    assert run_eval._git_info() == {"hash": None, "dirty": None}


def test_git_info_keeps_the_commit_when_only_the_dirty_check_fails(monkeypatch):
    """A shallow clone or a git version missing --porcelain must not lose the
    commit hash it already had -- two separate calls, two separate failures."""
    calls = {"n": 0}

    def _fake_run(args, **_kw):
        calls["n"] += 1
        if "rev-parse" in args:
            class _Result:
                stdout = "abc1234\n"
            return _Result()
        raise RuntimeError("status failed")

    monkeypatch.setattr("subprocess.run", _fake_run)

    assert run_eval._git_info() == {"hash": "abc1234", "dirty": None}


# _gold_cases_sha256


def test_gold_cases_sha256_matches_a_direct_hash():
    import hashlib

    expected = hashlib.sha256(run_eval.GOLD_FILE.read_bytes()).hexdigest()
    assert run_eval._gold_cases_sha256() == expected


def test_gold_cases_sha256_degrades_to_none_when_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "GOLD_FILE", tmp_path / "missing.jsonl")
    assert run_eval._gold_cases_sha256() is None


# _run_conditions


@pytest.fixture
def _quiet_conditions(monkeypatch):
    """Stub every host-dependent lookup so _run_conditions never touches the
    network or a GPU -- only the pure pieces (versions, gold hash) run for
    real."""
    monkeypatch.setattr(run_eval, "_ollama_server_version", lambda: None)
    monkeypatch.setattr(run_eval, "_gpu_info", lambda: {"name": None, "vram_total_mib": None})
    monkeypatch.setattr(run_eval, "_git_info", lambda: {"hash": None, "dirty": None})


def test_run_conditions_has_every_field_the_issue_asks_for(_quiet_conditions):
    sampling = {"rag": {"temperature": 0.15}, "chat": {}, "recomp": {}}
    conditions = run_eval._run_conditions({"dev": None, "blind": None}, sampling)

    assert set(conditions) == {
        "config", "versions", "hardware", "git_commit", "gold_sha256",
        "sampling", "seed", "keep_alive_seconds",
    }
    assert set(conditions["versions"]) == {
        "mineru", "sentence_transformers", "torch", "transformers", "faiss", "ollama_server",
    }
    assert conditions["seed"] is None
    assert conditions["keep_alive_seconds"] == int(run_eval._EVAL_GENERATION_KEEP_ALIVE_SECONDS)
    assert len(conditions["gold_sha256"]) == 64
    assert conditions["sampling"] == sampling


def test_run_conditions_reuses_the_config_it_is_given_not_a_fresh_default(_quiet_conditions):
    # A marker that no real AppConfig.from_env() would ever produce -- if
    # _run_conditions started rebuilding its own config instead of embedding
    # the one evaluate() already built, this value would not survive.
    marker_config = {"dev": {"marker": "distinctive-test-value"}, "blind": None}

    conditions = run_eval._run_conditions(marker_config, sampling={})

    assert conditions["config"] is marker_config
    assert conditions["config"] != {"dev": None, "blind": None}


def test_run_conditions_config_is_not_bare_appconfig_defaults(_quiet_conditions):
    """Guard against a regression where the block is built from a fresh
    ``AppConfig()`` instead of the run's actual, possibly-overridden one."""
    import dataclasses

    real_config = {"dev": dataclasses.asdict(AppConfig().with_overrides(**{"retrieval.top_k_final": 3}))}
    bare_defaults = {"dev": dataclasses.asdict(AppConfig())}

    conditions = run_eval._run_conditions(real_config, sampling={})

    assert conditions["config"] != bare_defaults
    assert conditions["config"]["dev"]["retrieval"]["top_k_final"] == 3


def test_run_conditions_survives_every_sub_lookup_failing(monkeypatch):
    """Package lookups, the Ollama version, the GPU and git can all fail
    independently -- none of it may stop the artifact from being written."""
    monkeypatch.setattr(run_eval, "_package_version", lambda _dist: None)
    monkeypatch.setattr(run_eval, "_ollama_server_version", lambda: None)
    monkeypatch.setattr(run_eval, "_gpu_info", lambda: {"name": None, "vram_total_mib": None})
    monkeypatch.setattr(run_eval, "_git_info", lambda: {"hash": None, "dirty": None})
    monkeypatch.setattr(run_eval, "_gold_cases_sha256", lambda: None)

    conditions = run_eval._run_conditions({"dev": None, "blind": None}, sampling={})

    assert conditions["versions"] == {
        "mineru": None, "sentence_transformers": None, "torch": None,
        "transformers": None, "faiss": None, "ollama_server": None,
    }
    assert conditions["hardware"] == {"name": None, "vram_total_mib": None}
    assert conditions["git_commit"] == {"hash": None, "dirty": None}
    assert conditions["gold_sha256"] is None
