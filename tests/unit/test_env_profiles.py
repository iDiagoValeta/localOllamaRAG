"""Unit tests for MONKEYGRAB_PROFILE seeding and the embedder endpoint.

A profile is a named bundle of default exports for one hardware target. It is
applied with ``setdefault``, so a variable the user exported always wins; these
tests pin that contract, the ``standard``/unknown handling, and the fact that
``AppConfig.from_env`` reads every variable the profile seeds.
"""

import os

import pytest

from monkeygrab.config import env as env_mod
from monkeygrab.config.app_config import AppConfig
from monkeygrab.config.env import PROFILES, apply_profile, is_env_pinned


def test_lite_profile_turns_off_reranker_and_recomp():
    assert "lite" in PROFILES
    assert PROFILES["lite"]["USAR_RERANKER"] == "false"
    assert PROFILES["lite"]["USAR_RECOMP_SYNTHESIS"] == "false"


def test_apply_profile_seeds_every_unset_entry(monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_PROFILE", raising=False)
    environ = {"MONKEYGRAB_PROFILE": "lite"}

    applied = apply_profile(environ)

    assert applied == "lite"
    for key, value in PROFILES["lite"].items():
        assert environ[key] == value


def test_apply_profile_never_overrides_an_explicit_value(monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_PROFILE", raising=False)
    environ = {"MONKEYGRAB_PROFILE": "lite", "OLLAMA_RAG_MODEL": "mine:latest"}

    apply_profile(environ)

    assert environ["OLLAMA_RAG_MODEL"] == "mine:latest"
    # The other entries are still filled in.
    assert environ["OLLAMA_RAG_NUM_CTX"] == "8192"


@pytest.mark.parametrize("selector", ["standard", "", "   "])
def test_apply_profile_standard_is_a_noop(monkeypatch, selector):
    monkeypatch.delenv("MONKEYGRAB_PROFILE", raising=False)
    environ = {"MONKEYGRAB_PROFILE": selector}

    assert apply_profile(environ) is None
    assert environ == {"MONKEYGRAB_PROFILE": selector}


def test_apply_profile_unset_is_a_noop(monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_PROFILE", raising=False)
    environ = {}

    assert apply_profile(environ) is None
    assert environ == {}


def test_apply_profile_unknown_profile_raises(monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_PROFILE", raising=False)

    with pytest.raises(ValueError, match="lite"):
        apply_profile({"MONKEYGRAB_PROFILE": "turbo"})


def test_from_env_embedder_defaults(monkeypatch):
    for name in ("MONKEYGRAB_EMBEDDER_URL", "MONKEYGRAB_EMBEDDER_DIM"):
        monkeypatch.delenv(name, raising=False)

    cfg = AppConfig.from_env()

    assert cfg.models.embedding.base_url == "http://127.0.0.1:8091"
    assert cfg.models.embedding.dimensions == 512


def test_from_env_reads_the_embedder_endpoint(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_URL", "http://embed.example:8091")
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_DIM", "256")

    cfg = AppConfig.from_env()

    assert cfg.models.embedding.base_url == "http://embed.example:8091"
    assert cfg.models.embedding.dimensions == 256


def test_from_env_rejects_unsupported_embedder_dimension(monkeypatch):
    monkeypatch.setenv("MONKEYGRAB_EMBEDDER_DIM", "384")

    with pytest.raises(ValueError, match="768"):
        AppConfig.from_env()


def test_lite_profile_values_drive_from_env(monkeypatch):
    monkeypatch.delenv("MONKEYGRAB_PROFILE", raising=False)
    environ = {"MONKEYGRAB_PROFILE": "lite"}
    apply_profile(environ)
    for key, value in environ.items():
        if key != "MONKEYGRAB_PROFILE":
            monkeypatch.setenv(key, value)

    cfg = AppConfig.from_env()

    assert cfg.models.embedding.dimensions == 512
    assert cfg.flags.usar_reranker is False
    assert cfg.flags.usar_recomp_synthesis is False
    assert cfg.models.ollama.rag_num_ctx == 8192
    assert cfg.models.ollama.recomp_num_ctx == 4096
    assert cfg.chunking.contextual_doc_chars == 12000
    assert cfg.context.max_context_chars == 16000


def test_is_env_pinned_distinguishes_an_export_from_a_profile_seed(monkeypatch):
    """A profile seed lives in os.environ but is a default, not an export.

    settings.py must not treat a variable ``apply_profile`` filled in as the
    operator's explicit choice for this run (issue #301), so ``is_env_pinned``
    is False for a seeded key, True for a genuine export, and False for a
    variable that is not set at all.
    """
    monkeypatch.setenv("OLLAMA_RAG_MODEL", "exported:latest")
    monkeypatch.setattr(env_mod, "_PROFILE_SEEDED_KEYS", set())
    assert is_env_pinned("OLLAMA_RAG_MODEL") is True

    monkeypatch.setattr(env_mod, "_PROFILE_SEEDED_KEYS", {"OLLAMA_RAG_MODEL"})
    assert is_env_pinned("OLLAMA_RAG_MODEL") is False

    monkeypatch.delenv("OLLAMA_RAG_MODEL", raising=False)
    assert is_env_pinned("OLLAMA_RAG_MODEL") is False


def test_apply_profile_records_only_the_keys_it_seeds(monkeypatch):
    """apply_profile records its seeds, and only its seeds, for is_env_pinned.

    A key already exported before the profile is applied must stay pinned, not
    get absorbed into the seed set; every other key the profile fills in must
    land in the set so a saved choice can still override it.
    """
    monkeypatch.setenv("MONKEYGRAB_PROFILE", "lite")
    monkeypatch.setattr(env_mod, "_PROFILE_SEEDED_KEYS", set())
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "mine:latest")
    for key in PROFILES["lite"]:
        if key != "OLLAMA_CHAT_MODEL":
            monkeypatch.delenv(key, raising=False)

    # apply_profile seeds os.environ directly, and delenv(raising=False) records
    # no restore for a key that was absent, so clear those keys ourselves.
    try:
        assert apply_profile() == "lite"

        assert is_env_pinned("OLLAMA_CHAT_MODEL") is True
        assert is_env_pinned("OLLAMA_RAG_MODEL") is False
        assert "OLLAMA_CHAT_MODEL" not in env_mod._PROFILE_SEEDED_KEYS
        assert env_mod._PROFILE_SEEDED_KEYS == set(PROFILES["lite"]) - {"OLLAMA_CHAT_MODEL"}
    finally:
        for key in PROFILES["lite"]:
            if key != "OLLAMA_CHAT_MODEL":
                os.environ.pop(key, None)


def test_lite_profile_assigns_the_aux_model_to_the_rag_model():
    """Issue #302: the lite AUX model is the row's own generator, not Ling-3.0-tiny.

    ``tests/eval/run_eval.py`` reads ``OLLAMA_AUX_MODEL`` with a default of
    Ling-3.0-tiny, which a 4 GB GPU cannot hold. The lite profile must pin it to
    the same small generator as the other roles, matching docs/model-history.md.
    """
    assert PROFILES["lite"]["OLLAMA_AUX_MODEL"] == PROFILES["lite"]["OLLAMA_RAG_MODEL"]
