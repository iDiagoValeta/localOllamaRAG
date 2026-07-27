"""AppConfig -- immutable configuration root for the whole pipeline.

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- 1. AppConfig            frozen dataclass grouping the 7 sub-configs
#  +-- 2. from_env()           builds AppConfig from process environment
#  +-- 3. with_overrides()     returns a NEW AppConfig, never mutates
#
# ─────────────────────────────────────────────

Replaces the mutable module-globals of ``rag/chat_pdfs.py`` (section 3) and
its three runtime mutators (``set_pipeline_flags``, ``set_model_roles_runtime``,
``set_docs_folder_runtime``) with one object, injected into use cases instead
of read through a service locator. This is what makes the "stale default
argument" bug (see ``tests/characterization/test_stale_default_config_bug.py``)
structurally impossible: a function that wants config takes an ``AppConfig``
parameter with no default, so there is nothing for a value to go stale
against -- every call sees exactly the config instance its caller passed it.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict

from monkeygrab.config import env
from monkeygrab.config.chunking import ChunkingConfig
from monkeygrab.config.context import ContextConfig
from monkeygrab.config.flags import PipelineFlagsConfig
from monkeygrab.config.models import (
    ModelsConfig,
    OllamaRuntimeConfig,
    derive_embedding_prefixes,
    infer_model_description,
)
from monkeygrab.config.paths import RAG_BASE_DIR, PathsConfig, derive_db_paths
from monkeygrab.config.reranking import RerankingConfig
from monkeygrab.config.retrieval import RetrievalConfig
from monkeygrab.config.stack import StackConfig, stack_from_env

# The 7 sections with runtime-override support via with_overrides(); anything
# nested deeper (e.g. models.ollama.*) is set at construction time only, same
# as today -- OLLAMA_*_NUM_CTX was never part of set_pipeline_flags or
# set_model_roles_runtime's runtime-mutable surface either.
_SECTION_NAMES = ("models", "chunking", "retrieval", "reranking", "context", "flags", "paths")


# ─────────────────────────────────────────────
# SECTION 1: APPCONFIG
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class AppConfig:
    """Immutable root of all pipeline configuration.

    Groups every constant read from ``rag/chat_pdfs.py`` section 3 into 7
    coherent sub-configs (see each module's docstring for the field-by-field
    mapping back to the original ``os.getenv`` calls / literals):
    ``models`` (roles + Ollama runtime), ``chunking``, ``retrieval``
    (semantic/BM25 fan-out + RRF), ``reranking``, ``context`` (character
    budget), ``flags`` (pipeline toggles), ``paths``.

    Attributes:
        models: Model roles and the Ollama runtime they execute under.
        chunking: Indexing-time text/image splitting parameters.
        retrieval: Semantic/BM25 fan-out and Reciprocal-Rank-Fusion.
        reranking: Cross-Encoder relevance threshold.
        context: Character budget for generation evidence.
        flags: Pipeline stage toggles.
        paths: Filesystem locations and derived vector-store paths.
    """

    models: ModelsConfig = field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    flags: PipelineFlagsConfig = field(default_factory=PipelineFlagsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Which implementation stands behind each swappable port. Its default is the
    # current production stack, so an unset environment behaves exactly as before.
    stack: StackConfig = field(default_factory=StackConfig)

    # ─────────────────────────────────────────────
    # SECTION 2: FROM_ENV
    # ─────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build an ``AppConfig`` from the process environment.

        Reproduces the exact defaults ``rag/chat_pdfs.py`` section 3 binds
        at import time -- every ``os.getenv(NAME, default)`` call there has
        a one-to-one counterpart here reading the same ``NAME`` with the
        same ``default``, and every plain Python literal (the 10 pipeline
        flags, ``MAX_HISTORIAL_MENSAJES``) is reproduced as the same
        literal, not newly wired to an environment variable. See
        ``tests/unit/test_app_config_defaults.py`` for the field-by-field
        equality check against a live ``rag.chat_pdfs`` import.

        Unlike ``rag/chat_pdfs.py``'s ``_leer_env_int``/``_leer_env_float``,
        an environment variable that IS SET but not parseable raises
        instead of silently falling back to the default (hard-fail policy;
        see ``monkeygrab.config.env``). ``RERANKER_QUALITY`` is similarly
        validated against its two known values instead of silently
        treating anything other than ``"quality"`` as ``"fast"``.

        Returns:
            A fully-populated ``AppConfig``.

        Raises:
            ValueError: If any environment variable is set to an invalid value.
        """
        rag_model = env.read_env_str("OLLAMA_RAG_MODEL", "gemma4:e4b")
        embedding_model = env.read_env_str("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
        embed_prefix_query, embed_prefix_doc = derive_embedding_prefixes(embedding_model)

        models = ModelsConfig(
            rag=rag_model,
            chat=env.read_env_str("OLLAMA_CHAT_MODEL", "gemma4:e4b"),
            embedding=embedding_model,
            contextual=env.read_env_str("OLLAMA_CONTEXTUAL_MODEL", "gemma4:e4b"),
            recomp=env.read_env_str("OLLAMA_RECOMP_MODEL", "gemma4:e4b"),
            ocr=env.read_env_str("OLLAMA_OCR_MODEL", "gemma4:e4b"),
            desc=env.read_env_str("MODELO_DESC", infer_model_description(rag_model)),
            embed_prefix_query=embed_prefix_query,
            embed_prefix_doc=embed_prefix_doc,
            reranker_quality=env.read_env_choice("RERANKER_QUALITY", "quality", ("quality", "fast")),
            ollama=OllamaRuntimeConfig(
                num_ctx=env.read_env_int("OLLAMA_NUM_CTX", 8192),
                rag_num_ctx=env.read_env_int("OLLAMA_RAG_NUM_CTX", 16384),
                aux_num_ctx=env.read_env_int("OLLAMA_AUX_NUM_CTX", 8192),
                query_num_ctx=env.read_env_int("OLLAMA_QUERY_NUM_CTX", 2048),
                recomp_num_ctx=env.read_env_int("OLLAMA_RECOMP_NUM_CTX", 8192),
                contextual_num_ctx=env.read_env_int("OLLAMA_CONTEXTUAL_NUM_CTX", 32768),
                ocr_num_ctx=env.read_env_int("OLLAMA_OCR_NUM_CTX", 8192),
                request_timeout=env.read_env_int("OLLAMA_REQUEST_TIMEOUT", 900),
                keep_alive=env.read_env_int("OLLAMA_KEEP_ALIVE", 0),
                generate_retries=env.read_env_int("OLLAMA_GENERATE_RETRIES", 2),
                generate_retry_delay=env.read_env_int("OLLAMA_GENERATE_RETRY_DELAY", 3),
            ),
        )

        chunking = ChunkingConfig(
            chunk_size=env.read_env_int("RAG_CHUNK_SIZE", 2000),
            chunk_overlap=env.read_env_int("RAG_CHUNK_OVERLAP", 400),
            min_chunk_length=env.read_env_int("RAG_MIN_CHUNK_LENGTH", 150),
            contextual_doc_chars=env.read_env_int("CONTEXTUAL_DOC_CHARS", 24000),
            max_images_per_page=env.read_env_int("RAG_MAX_IMAGES_PER_PAGE", 5),
            min_image_size_px=env.read_env_int("RAG_MIN_IMAGE_SIZE_PX", 100),
            caption_margin_px=env.read_env_int("RAG_CAPTION_MARGIN_PX", 80),
        )

        retrieval = RetrievalConfig(
            n_semantic_results=env.read_env_int("RAG_N_RESULTADOS_SEMANTICOS", 80),
            n_keyword_results=env.read_env_int("RAG_N_RESULTADOS_KEYWORD", 40),
            top_k_rerank_candidates=env.read_env_int("RAG_TOP_K_RERANK_CANDIDATES", 200),
            top_k_final=env.read_env_int("RAG_TOP_K_FINAL", 8),
            n_top_for_expansion=env.read_env_int("RAG_N_TOP_PARA_EXPANSION", 3),
            rrf_k=env.read_env_int("RAG_RRF_K", 60),
            bm25_k1=env.read_env_float("RAG_BM25_K1", 1.5),
            bm25_b=env.read_env_float("RAG_BM25_B", 0.75),
            weight_semantic_rrf=env.read_env_float("RAG_PESO_SEMANTICO_RRF", 0.55),
            weight_bm25_rrf=env.read_env_float("RAG_PESO_BM25_RRF", 0.45),
            min_question_length=env.read_env_int("RAG_MIN_LONGITUD_PREGUNTA", 10),
        )

        reranking = RerankingConfig(
            score_threshold=env.read_env_float("RAG_UMBRAL_SCORE_RERANKER", 0.65),
        )

        context = ContextConfig(
            max_context_chars=env.read_env_int("MAX_CONTEXTO_CHARS", 24000),
        )

        # All 10 pipeline flags are hardcoded True in rag/chat_pdfs.py section
        # 3.4 -- none of them are read from the environment there, so none
        # are read from the environment here either.
        flags = PipelineFlagsConfig()

        base_dir = RAG_BASE_DIR
        data_dir = os.path.abspath(env.read_env_str("MONKEYGRAB_DATA_DIR", base_dir))
        docs_folder = env.read_env_str("DOCS_FOLDER", os.path.join(base_dir, "docs", "en"))
        path_db, collection_name = derive_db_paths(docs_folder, embedding_model, data_dir)

        paths = PathsConfig(
            base_dir=base_dir,
            data_dir=data_dir,
            docs_folder=docs_folder,
            path_db=path_db,
            collection_name=collection_name,
            historial_path=os.path.join(data_dir, "historial_chat.json"),
            max_historial_mensajes=40,
            carpeta_debug_rag=os.path.join(data_dir, "debug_rag"),
        )

        return cls(
            models=models, chunking=chunking, retrieval=retrieval,
            reranking=reranking, context=context, flags=flags, paths=paths,
            stack=stack_from_env(),
        )

    # ─────────────────────────────────────────────
    # SECTION 3: WITH_OVERRIDES
    # ─────────────────────────────────────────────

    def with_overrides(self, **cambios: Any) -> "AppConfig":
        """Return a NEW ``AppConfig`` with the given fields replaced.

        Replaces ``set_pipeline_flags``, ``set_model_roles_runtime`` and
        ``set_docs_folder_runtime`` from ``rag/chat_pdfs.py``: since
        ``AppConfig`` is immutable, "changing config" can only ever mean
        "here is a new config" -- there is no shared mutable state left for
        one caller's change to leak into another caller's in-flight
        request, and no default-argument snapshot that can go stale.

        Keys are dotted ``"<section>.<field>"`` strings, e.g.
        ``config.with_overrides(**{"flags.usar_reranker": False,
        "models.embedding": "nomic-embed-text:latest"})``. ``<section>`` is
        one of ``models``, ``chunking``, ``retrieval``, ``reranking``,
        ``context``, ``flags``, ``paths``; ``<field>`` must be a real field
        of that section's dataclass.

        Side effects, mirroring today's runtime mutators exactly:

        - ``"models.rag"``: recomputes ``models.desc`` (was: the ``if
          overrides.get("rag")`` branch of ``set_model_roles_runtime``).
        - ``"models.embedding"``: recomputes ``models.embed_prefix_query``/
          ``embed_prefix_doc`` and ``paths.path_db``/``collection_name``
          (was: the ``embedding_changed`` branch of
          ``set_model_roles_runtime``).
        - ``"paths.docs_folder"``: recomputes ``paths.path_db``/
          ``collection_name`` (was: ``set_docs_folder_runtime``).

        There is no ``carpeta=None``-style "restore defaults" override:
        immutability makes it redundant -- the caller already holds a
        reference to whatever ``AppConfig`` it wants to go back to.

        Args:
            **cambios: Dotted ``"section.field"`` keys mapped to new values.

        Returns:
            A new ``AppConfig``; ``self`` is never mutated.

        Raises:
            ValueError: If a key is not ``"section.field"``-shaped, names
                an unknown section, or names a field that section does not
                have.
        """
        sections: Dict[str, Any] = {name: getattr(self, name) for name in _SECTION_NAMES}
        changes_by_section: Dict[str, Dict[str, Any]] = {name: {} for name in _SECTION_NAMES}

        for dotted_key, value in cambios.items():
            if "." not in dotted_key:
                raise ValueError(
                    f"Override key must be '<section>.<field>', got {dotted_key!r}. "
                    f"Valid sections: {', '.join(_SECTION_NAMES)}"
                )
            section_name, field_name = dotted_key.split(".", 1)
            if section_name not in sections:
                raise ValueError(
                    f"Unknown config section {section_name!r}. Valid: {', '.join(_SECTION_NAMES)}"
                )
            valid_fields = {f.name for f in dataclasses.fields(sections[section_name])}
            if field_name not in valid_fields:
                raise ValueError(
                    f"Unknown field {field_name!r} in section {section_name!r}. "
                    f"Valid: {', '.join(sorted(valid_fields))}"
                )
            changes_by_section[section_name][field_name] = value

        if "docs_folder" in changes_by_section["paths"]:
            # set_docs_folder_runtime (rag/chat_pdfs.py) always normalized via
            # os.path.abspath before assigning CARPETA_DOCS; the web PDF
            # viewer resolves paths.docs_folder directly against disk, so a
            # relative override left as-is here would resolve against
            # whatever the process's cwd happens to be at request time
            # instead of staying stable. Normalize before it feeds
            # derive_db_paths below, same as the original.
            changes_by_section["paths"]["docs_folder"] = os.path.abspath(
                changes_by_section["paths"]["docs_folder"]
            )

        new_models = sections["models"]
        if changes_by_section["models"]:
            new_models = dataclasses.replace(new_models, **changes_by_section["models"])
        if "rag" in changes_by_section["models"]:
            new_models = dataclasses.replace(new_models, desc=infer_model_description(new_models.rag))
        if "embedding" in changes_by_section["models"]:
            query_prefix, doc_prefix = derive_embedding_prefixes(new_models.embedding)
            new_models = dataclasses.replace(
                new_models, embed_prefix_query=query_prefix, embed_prefix_doc=doc_prefix
            )

        new_paths = sections["paths"]
        if changes_by_section["paths"]:
            new_paths = dataclasses.replace(new_paths, **changes_by_section["paths"])
        if "embedding" in changes_by_section["models"] or "docs_folder" in changes_by_section["paths"]:
            path_db, collection_name = derive_db_paths(
                new_paths.docs_folder, new_models.embedding, new_paths.data_dir
            )
            new_paths = dataclasses.replace(new_paths, path_db=path_db, collection_name=collection_name)

        new_sections = {"models": new_models, "paths": new_paths}
        for name in _SECTION_NAMES:
            if name in ("models", "paths"):
                continue
            new_sections[name] = (
                dataclasses.replace(sections[name], **changes_by_section[name])
                if changes_by_section[name]
                else sections[name]
            )

        return dataclasses.replace(self, **new_sections)
