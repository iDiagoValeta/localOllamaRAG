"""Unit tests for monkeygrab.config.AppConfig.from_env() defaults.

Compares every field of a freshly built ``AppConfig`` against the live
``rag.chat_pdfs`` module constants it must reproduce (see
docs/design/2026-07-26-monkeygrab-v2.md, section 4: "AppConfig ...
replicando EXACTAMENTE los defaults actuales"). This is the drift guard: if
``rag/chat_pdfs.py`` ever changes a default without this test being updated
in the same change, the test fails instead of the two configs silently
diverging.

Run with the ambient environment clean of the RAG_*/OLLAMA_*/DOCS_FOLDER/
MONKEYGRAB_DATA_DIR variables (the project's normal dev setup) -- these
tests assert *default* values, not "whatever the current shell happens to
have exported".
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monkeygrab.config import AppConfig  # noqa: E402


# (AppConfig dotted path, matching rag.chat_pdfs attribute name). Listed
# flat and explicit -- one line per constant in rag/chat_pdfs.py section 3 --
# so a missing/extra field is a visible diff, not a silent gap.
_FIELD_MAP = [
    ("models.rag", "MODELO_RAG"),
    ("models.chat", "MODELO_CHAT"),
    ("models.contextual", "MODELO_CONTEXTUAL"),
    ("models.recomp", "MODELO_RECOMP"),
    ("models.desc", "MODELO_DESC"),
    ("models.ollama.rag_num_ctx", "OLLAMA_RAG_NUM_CTX"),
    ("models.ollama.query_num_ctx", "OLLAMA_QUERY_NUM_CTX"),
    ("models.ollama.recomp_num_ctx", "OLLAMA_RECOMP_NUM_CTX"),
    ("models.ollama.contextual_num_ctx", "OLLAMA_CONTEXTUAL_NUM_CTX"),
    ("models.ollama.request_timeout", "OLLAMA_REQUEST_TIMEOUT"),
    ("models.ollama.keep_alive", "OLLAMA_KEEP_ALIVE"),
    ("models.ollama.generate_retries", "OLLAMA_GENERATE_RETRIES"),
    ("models.ollama.generate_retry_delay", "OLLAMA_GENERATE_RETRY_DELAY"),
    ("chunking.chunk_size", "CHUNK_SIZE"),
    ("chunking.chunk_overlap", "CHUNK_OVERLAP"),
    ("chunking.min_chunk_length", "MIN_CHUNK_LENGTH"),
    ("chunking.contextual_doc_chars", "CONTEXTUAL_DOC_CHARS"),
    ("retrieval.n_semantic_results", "N_RESULTADOS_SEMANTICOS"),
    ("retrieval.n_keyword_results", "N_RESULTADOS_KEYWORD"),
    ("retrieval.top_k_rerank_candidates", "TOP_K_RERANK_CANDIDATES"),
    ("retrieval.top_k_final", "TOP_K_FINAL"),
    ("retrieval.n_top_for_expansion", "N_TOP_PARA_EXPANSION"),
    ("retrieval.rrf_k", "RRF_K"),
    ("retrieval.bm25_k1", "BM25_K1"),
    ("retrieval.bm25_b", "BM25_B"),
    ("retrieval.weight_semantic_rrf", "PESO_SEMANTICO_RRF"),
    ("retrieval.weight_bm25_rrf", "PESO_BM25_RRF"),
    ("retrieval.min_question_length", "MIN_LONGITUD_PREGUNTA_RAG"),
    ("reranking.score_threshold", "UMBRAL_SCORE_RERANKER"),
    ("context.max_context_chars", "MAX_CONTEXTO_CHARS"),
    ("flags.usar_contextual_retrieval", "USAR_CONTEXTUAL_RETRIEVAL"),
    ("flags.usar_llm_query_decomposition", "USAR_LLM_QUERY_DECOMPOSITION"),
    ("flags.usar_busqueda_hibrida", "USAR_BUSQUEDA_HIBRIDA"),
    ("flags.usar_reranker", "USAR_RERANKER"),
    ("flags.expandir_contexto", "EXPANDIR_CONTEXTO"),
    ("flags.usar_optimizacion_contexto", "USAR_OPTIMIZACION_CONTEXTO"),
    ("flags.usar_recomp_synthesis", "USAR_RECOMP_SYNTHESIS"),
    ("flags.usar_embeddings_imagen", "USAR_EMBEDDINGS_IMAGEN"),
    ("flags.logging_metricas", "LOGGING_METRICAS"),
    ("flags.guardar_debug_rag", "GUARDAR_DEBUG_RAG"),
    ("paths.base_dir", "BASE_DIR"),
    ("paths.data_dir", "DATA_DIR"),
    ("paths.docs_folder", "CARPETA_DOCS"),
    ("paths.path_db", "PATH_DB"),
    ("paths.collection_name", "COLLECTION_NAME"),
    ("paths.historial_path", "HISTORIAL_PATH"),
    ("paths.max_historial_mensajes", "MAX_HISTORIAL_MENSAJES"),
    ("paths.carpeta_debug_rag", "CARPETA_DEBUG_RAG"),
]


# Fields that exist only in the new configuration and therefore have nothing in
# rag.chat_pdfs to be compared against. Listed explicitly rather than skipped by
# a prefix rule, so that adding a field still forces a decision here instead of
# quietly escaping the drift guard.
#
# The backend selectors are new capability: the old engine had no way to choose an
# implementation, which is why swapping one meant editing wiring code. Their
# defaults are covered by tests/unit/test_stack_selection.py, which asserts that
# an unset environment reproduces the current production stack.
_FIELDS_WITHOUT_ENGINE_COUNTERPART = set()


def _resolve(obj, dotted_path):
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _fresh_rag_chat_pdfs_snapshot():
    """Import rag.chat_pdfs in an isolated subprocess and return the
    _FIELD_MAP attributes as a dict.

    Deliberately NOT a plain ``import rag.chat_pdfs`` in this process:
    tests/test_web_routes.py imports rag.web.app, whose module-level setup
    loads the user's (gitignored, per-machine) settings.json and applies it
    via rag.chat_pdfs.set_model_roles_runtime(...) -- a real in-place
    mutation of rag.chat_pdfs's globals for the rest of the pytest process,
    independent of test order. Comparing against that shared, possibly
    already-mutated module would make this test's outcome depend on
    whatever some other test file happened to do first and on the local
    machine's settings.json contents. A subprocess only ever imports
    rag.chat_pdfs itself, so it reflects nothing but the environment.
    """
    names = sorted({rag_attr for _, rag_attr in _FIELD_MAP})
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        # rag.engine.* now delegates parts of its logic to monkeygrab.application
        # (see the clean-architecture migration), so rag.chat_pdfs transitively
        # needs the "src" layout root too -- mirrors pytest.ini's own
        # `pythonpath = . src`, which the main pytest process gets for free but
        # this subprocess does not.
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import rag.chat_pdfs as rag\n"
        f"print(json.dumps({{name: getattr(rag, name) for name in {names!r}}}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        # A missing engine dependency means this environment cannot host the
        # comparison at all, which is a different fact from "the defaults
        # diverged" and must not be reported as the latter. The fast CI gate
        # installs no infrastructure on purpose, so it lands here; the engine
        # job has the full stack and does run the comparison.
        if "ModuleNotFoundError" in result.stderr:
            pytest.skip(
                "rag.chat_pdfs cannot be imported here (missing engine "
                f"dependency). Comparison skipped, not passed:\n{result.stderr.strip()[-300:]}"
            )
        raise AssertionError(
            f"failed to import rag.chat_pdfs in a subprocess:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def test_every_mapped_field_matches_chat_pdfs_default():
    """AppConfig.from_env() reproduces every constant in the map above.

    A single assert over the whole map (rather than one test per field)
    keeps the failure message a complete diff instead of one field at a
    time -- deliberate for a 59-constant equality check.
    """
    cfg = AppConfig.from_env()
    fresh_rag = _fresh_rag_chat_pdfs_snapshot()
    mismatches = []
    for dotted_path, rag_attr in _FIELD_MAP:
        ours = _resolve(cfg, dotted_path)
        theirs = fresh_rag[rag_attr]
        if ours != theirs:
            mismatches.append(f"{dotted_path}={ours!r} != rag.{rag_attr}={theirs!r}")
    assert not mismatches, "\n".join(mismatches)


def test_field_map_is_exhaustive_over_appconfig_leaf_fields():
    """Every leaf field actually present on AppConfig is covered by _FIELD_MAP.

    Guards the guard: without this, a newly added AppConfig field with no
    counterpart in ``rag.chat_pdfs`` (or simply forgotten from the map)
    would silently never be compared.
    """
    import dataclasses

    cfg = AppConfig()
    mapped = {path for path, _ in _FIELD_MAP} | _FIELDS_WITHOUT_ENGINE_COUNTERPART

    def _leaf_paths(obj, prefix=""):
        paths = set()
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            path = f"{prefix}{f.name}"
            if dataclasses.is_dataclass(value):
                paths |= _leaf_paths(value, prefix=f"{path}.")
            else:
                paths.add(path)
        return paths

    actual_leaves = _leaf_paths(cfg)
    assert actual_leaves == mapped, (
        f"AppConfig fields not covered by _FIELD_MAP: {actual_leaves - mapped}\n"
        f"_FIELD_MAP entries with no matching AppConfig field: {mapped - actual_leaves}"
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
